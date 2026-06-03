from __future__ import annotations

import subprocess
import threading
import time
import uuid
import os
from pathlib import Path
from typing import Any

from app.operation_log import log_operation
from app.profile_store import create_profile, profile_path
from app.trainers import get_trainer, list_trainers
from app.training_store import (
    append_job_log,
    delete_job_files,
    job_log_path,
    list_jobs as store_list_jobs,
    load_job,
    load_recipe,
    normalize_env_vars,
    now_text,
    public_job,
    save_job,
)


INFERENCE_PARAM_KEYS = {
    "policy_type",
    "temporal_agg",
    "n_obs_steps",
    "n_action_steps",
    "chunk_size",
    "horizon",
    "num_inference_steps",
}


class TrainingScheduler:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._current_process: subprocess.Popen[str] | None = None
        self._current_job_id: str | None = None

    def ensure_started(self) -> None:
        with self._lock:
            self._mark_stale_running_jobs()
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._worker_loop, name="training-scheduler", daemon=True)
            self._thread.start()

    def submit(self, recipe_id: str) -> dict[str, Any]:
        recipe = load_recipe(recipe_id)
        trainer_cls = get_trainer(recipe["framework"])
        inspection = trainer_cls.inspect_recipe(recipe)
        if inspection.get("errors"):
            raise ValueError("; ".join(inspection["errors"]))
        job_id = unique_job_id()
        queue_index = next_queue_index()
        job = {
            "job_id": job_id,
            "recipe_id": recipe_id,
            "recipe_snapshot": recipe,
            "status": "queued",
            "pid": None,
            "queue_index": queue_index,
            "created_at": now_text(),
            "started_at": None,
            "finished_at": None,
            "progress": {},
            "log_file": str(job_log_path(job_id)),
            "error": None,
            "returncode": None,
            "auto_generated_profile_id": None,
        }
        save_job(job)
        log_operation("training_job_submit", "success", target=job_id, details={"recipe_id": recipe_id})
        self.ensure_started()
        return public_job(job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = load_job(job_id)
            if job.get("status") == "queued":
                job["status"] = "cancelled"
                job["finished_at"] = now_text()
                save_job(job)
                log_operation("training_job_cancel", "success", target=job_id)
                return public_job(job)
            if job.get("status") == "running":
                job["cancel_requested"] = True
                save_job(job)
                if self._current_job_id == job_id and self._current_process:
                    self._current_process.terminate()
                log_operation("training_job_cancel", "success", target=job_id)
                return public_job(job)
            raise ValueError(f"cannot cancel job in status {job.get('status')}")

    def requeue(self, job_id: str) -> dict[str, Any]:
        job = load_job(job_id)
        if job.get("status") == "running":
            raise ValueError("cannot requeue a running job")
        job["status"] = "queued"
        job["pid"] = None
        job["queue_index"] = next_queue_index()
        job["started_at"] = None
        job["finished_at"] = None
        job["progress"] = {}
        job["error"] = None
        job["returncode"] = None
        job["auto_generated_profile_id"] = None
        save_job(job)
        log_operation("training_job_requeue", "success", target=job_id)
        self.ensure_started()
        return public_job(job)

    def delete(self, job_id: str) -> dict[str, Any]:
        job = load_job(job_id)
        if job.get("status") == "running":
            raise ValueError("cannot delete a running job; cancel it first")
        result = delete_job_files(job_id)
        log_operation("training_job_delete", "success", target=job_id)
        return result

    def reorder(self, job_ids: list[str]) -> list[dict[str, Any]]:
        queued = {job["job_id"]: job for job in store_list_jobs() if job.get("status") == "queued"}
        for index, job_id in enumerate(job_ids):
            if job_id not in queued:
                continue
            job = queued[job_id]
            job["queue_index"] = index
            save_job(job)
        log_operation("training_queue_reorder", "success", details={"job_ids": job_ids})
        self.ensure_started()
        return list_training_jobs()

    def status(self) -> dict[str, Any]:
        jobs = store_list_jobs()
        return {
            "running": next((job["job_id"] for job in jobs if job.get("status") == "running"), None),
            "queued": sum(1 for job in jobs if job.get("status") == "queued"),
            "worker_alive": bool(self._thread and self._thread.is_alive()),
        }

    def _worker_loop(self) -> None:
        while True:
            job = pick_next_queued_job()
            if not job:
                return
            try:
                self._run_job(job)
            except Exception as exc:
                job["status"] = "failed"
                job["finished_at"] = now_text()
                job["error"] = str(exc)
                save_job(job)
                log_operation("training_job_done", "failed", target=job.get("job_id"), error=str(exc))

    def _run_job(self, job: dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        recipe = job["recipe_snapshot"]
        trainer = get_trainer(recipe["framework"])()
        command = trainer.build_command(recipe)
        process_env = build_training_env(recipe)
        with self._lock:
            job["status"] = "running"
            job["started_at"] = now_text()
            job["pid"] = None
            job["command"] = command
            job["env_vars"] = public_training_env(recipe)
            save_job(job)
        append_job_log(job_id, "$ " + " ".join(command))
        env_summary = format_training_env_for_log(job.get("env_vars") or {})
        if env_summary:
            append_job_log(job_id, "# env " + env_summary)
        process = subprocess.Popen(
            command,
            cwd=str(Path(recipe.get("dataset_path") or ".").parent if recipe.get("dataset_path") else Path.cwd()),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        with self._lock:
            self._current_process = process
            self._current_job_id = job_id
            job["pid"] = process.pid
            save_job(job)
        if process.stdout:
            for line in process.stdout:
                append_job_log(job_id, line)
                progress = trainer.parse_progress(line.strip())
                if progress:
                    latest = load_job(job_id)
                    latest["progress"] = {
                        **(latest.get("progress") or {}),
                        **progress,
                        "last_update": now_text(),
                        "elapsed_seconds": elapsed_seconds(latest.get("started_at")),
                    }
                    save_job(latest)
        returncode = process.wait()
        latest = load_job(job_id)
        cancelled = bool(latest.get("cancel_requested"))
        latest["returncode"] = returncode
        latest["finished_at"] = now_text()
        latest["pid"] = None
        if cancelled:
            latest["status"] = "cancelled"
            latest["error"] = "cancelled by user"
        elif returncode == 0:
            latest["status"] = "done"
            latest["auto_generated_profile_id"] = auto_create_profile(latest)
        else:
            latest["status"] = "failed"
            latest["error"] = f"training command exited with code {returncode}"
        save_job(latest)
        with self._lock:
            self._current_process = None
            self._current_job_id = None
        log_operation(
            "training_job_done",
            "success" if latest["status"] == "done" else latest["status"],
            target=job_id,
            details={"returncode": returncode, "profile_id": latest.get("auto_generated_profile_id")},
            error=latest.get("error"),
        )

    def _mark_stale_running_jobs(self) -> None:
        for job in store_list_jobs():
            if job.get("status") != "running":
                continue
            if self._current_job_id == job.get("job_id") and self._current_process:
                continue
            job["status"] = "failed"
            job["finished_at"] = now_text()
            job["error"] = "service restarted or worker lost while job was running"
            job["pid"] = None
            save_job(job)


SCHEDULER = TrainingScheduler()


def build_training_env(recipe: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    recipe_env = normalize_env_vars(recipe.get("env_vars") or {})
    env.update(recipe_env)
    gpu_devices = str(recipe.get("gpu_devices") or "").strip()
    if gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices
    return env


def public_training_env(recipe: dict[str, Any]) -> dict[str, str]:
    env = normalize_env_vars(recipe.get("env_vars") or {})
    gpu_devices = str(recipe.get("gpu_devices") or "").strip()
    if gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices
    return {key: env[key] for key in sorted(env)}


def format_training_env_for_log(env_vars: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted((env_vars or {}).items()))


def unique_job_id() -> str:
    return f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def next_queue_index() -> int:
    queued = [int(job.get("queue_index", 0)) for job in store_list_jobs() if job.get("status") == "queued"]
    return max(queued, default=-1) + 1


def pick_next_queued_job() -> dict[str, Any] | None:
    queued = [job for job in store_list_jobs() if job.get("status") == "queued"]
    if not queued:
        return None
    return sorted(queued, key=lambda item: (item.get("queue_index", 0), item.get("created_at") or ""))[0]


def elapsed_seconds(started_at: str | None) -> float:
    if not started_at:
        return 0.0
    try:
        start = time.mktime(time.strptime(started_at, "%Y-%m-%d %H:%M:%S"))
        return round(max(0.0, time.time() - start), 3)
    except ValueError:
        return 0.0


def auto_create_profile(job: dict[str, Any]) -> str | None:
    recipe = job.get("recipe_snapshot") or {}
    if not recipe.get("auto_profile_on_complete", True):
        return None
    profile_id = unique_profile_id(str(recipe.get("id") or job["job_id"]))
    hp = recipe.get("hyperparams") or {}
    runtime_params = {key: value for key, value in hp.items() if key in INFERENCE_PARAM_KEYS}
    extra_params = {
        **(recipe.get("extra_params") or {}),
        "_auto_generated_from_recipe": recipe.get("id"),
        "_auto_generated_from_job": job.get("job_id"),
    }
    profile = create_profile(
        {
            "id": profile_id,
            "name": recipe.get("profile_name") or recipe.get("name") or profile_id,
            "description": f"自动生成自训练配方 {recipe.get('id')}（作业 {job.get('job_id')}）",
            "checkpoint_path": recipe.get("output_dir") or "",
            "adapter": recipe.get("profile_adapter") or "lerobot_official",
            "device": recipe.get("device") or "cuda",
            "runtime_params": runtime_params,
            "extra_params": extra_params,
        }
    )
    return str(profile["id"])


def unique_profile_id(base_id: str) -> str:
    candidate = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in base_id.lower()).strip("_") or "trained_model"
    if not profile_path(candidate).exists():
        return candidate
    suffix = uuid.uuid4().hex[:6]
    return f"{candidate}_{suffix}"


def inspect_training_recipe(recipe_id: str) -> dict[str, Any]:
    from app.training_store import save_recipe

    recipe = load_recipe(recipe_id)
    trainer_cls = get_trainer(recipe["framework"])
    inspection = trainer_cls.inspect_recipe(recipe)
    recipe["inspection"] = inspection
    recipe["status"] = "invalid" if inspection.get("errors") else "inspected"
    save_recipe(recipe)
    return recipe


def submit_training_job(recipe_id: str) -> dict[str, Any]:
    return SCHEDULER.submit(recipe_id)


def cancel_training_job(job_id: str) -> dict[str, Any]:
    return SCHEDULER.cancel(job_id)


def requeue_training_job(job_id: str) -> dict[str, Any]:
    return SCHEDULER.requeue(job_id)


def delete_training_job(job_id: str) -> dict[str, Any]:
    return SCHEDULER.delete(job_id)


def reorder_training_queue(job_ids: list[str]) -> list[dict[str, Any]]:
    return SCHEDULER.reorder(job_ids)


def list_training_jobs() -> list[dict[str, Any]]:
    SCHEDULER.ensure_started()
    return [public_job(job) for job in store_list_jobs()]


def get_training_job(job_id: str) -> dict[str, Any]:
    SCHEDULER.ensure_started()
    return public_job(load_job(job_id))


def training_runtime_status() -> dict[str, Any]:
    SCHEDULER.ensure_started()
    return {"frameworks": list_trainers(), "worker": SCHEDULER.status()}
