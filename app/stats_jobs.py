from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from app.lerobot_cli import RecomputeStatsRequest, build_recompute_stats_command, display_command, preview_recompute_stats_command
from app.operation_log import log_operation
from app.stats_job_store import (
    append_job_log,
    job_log_path,
    list_jobs as store_list_jobs,
    load_job,
    now_text,
    public_job,
    read_job_log,
    save_job,
)


class StatsScheduler:
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
            self._thread = threading.Thread(target=self._worker_loop, name="stats-scheduler", daemon=True)
            self._thread.start()

    def submit(self, request: RecomputeStatsRequest) -> dict[str, Any]:
        preview = preview_recompute_stats_command(request)
        if not preview.get("valid"):
            raise ValueError("; ".join(preview.get("errors", [])))
        args = preview["args"]
        job_id = unique_job_id()
        job = {
            "job_id": job_id,
            "status": "queued",
            "pid": None,
            "queue_index": next_queue_index(),
            "created_at": now_text(),
            "started_at": None,
            "finished_at": None,
            "repo_id": request.repo_id,
            "root": request.root,
            "new_repo_id": request.new_repo_id,
            "new_root": request.new_root,
            "request": request.model_dump(),
            "command": args,
            "display_command": preview["display_command"],
            "log_file": str(job_log_path(job_id)),
            "error": None,
            "returncode": None,
        }
        save_job(job)
        log_operation("stats_job_create", "success", target=job_id, details={"repo_id": request.repo_id, "root": request.root})
        self.ensure_started()
        return public_job(job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = load_job(job_id)
            if job.get("status") == "queued":
                job["status"] = "cancelled"
                job["finished_at"] = now_text()
                save_job(job)
                log_operation("stats_job_cancel", "success", target=job_id)
                return public_job(job)
            if job.get("status") == "running":
                job["cancel_requested"] = True
                save_job(job)
                if self._current_job_id == job_id and self._current_process:
                    self._current_process.terminate()
                log_operation("stats_job_cancel", "success", target=job_id)
                return public_job(job)
            raise ValueError(f"cannot cancel job in status {job.get('status')}")

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
                job["pid"] = None
                save_job(job)
                log_operation("stats_job_done", "failed", target=job.get("job_id"), error=str(exc))

    def _run_job(self, job: dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        args = list(job["command"])
        with self._lock:
            job["status"] = "running"
            job["started_at"] = now_text()
            job["pid"] = None
            save_job(job)
        append_job_log(job_id, "$ " + display_command(args))
        cwd = str(Path(job.get("root") or ".").expanduser().resolve().parent)
        process = subprocess.Popen(
            args,
            cwd=cwd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        with self._lock:
            self._current_process = process
            self._current_job_id = job_id
            latest = load_job(job_id)
            latest["pid"] = process.pid
            save_job(latest)
        if process.stdout:
            for line in process.stdout:
                append_job_log(job_id, line)
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
        else:
            latest["status"] = "failed"
            latest["error"] = f"stats command exited with code {returncode}"
        save_job(latest)
        with self._lock:
            self._current_process = None
            self._current_job_id = None
        log_operation(
            "stats_job_done",
            "success" if latest["status"] == "done" else latest["status"],
            target=job_id,
            details={"returncode": returncode},
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


SCHEDULER = StatsScheduler()


def submit_stats_job(request: RecomputeStatsRequest) -> dict[str, Any]:
    return SCHEDULER.submit(request)


def cancel_stats_job(job_id: str) -> dict[str, Any]:
    return SCHEDULER.cancel(job_id)


def list_stats_jobs() -> list[dict[str, Any]]:
    SCHEDULER.ensure_started()
    return [public_job(job) for job in store_list_jobs()]


def get_stats_job(job_id: str) -> dict[str, Any]:
    SCHEDULER.ensure_started()
    return public_job(load_job(job_id))


def stats_job_log(job_id: str, tail: int = 200) -> dict[str, Any]:
    return read_job_log(job_id, tail=tail)


def stats_runtime_status() -> dict[str, Any]:
    SCHEDULER.ensure_started()
    return {"worker": SCHEDULER.status()}


def unique_job_id() -> str:
    return f"stats_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def next_queue_index() -> int:
    queued = [int(job.get("queue_index", 0)) for job in store_list_jobs() if job.get("status") == "queued"]
    return max(queued, default=-1) + 1


def pick_next_queued_job() -> dict[str, Any] | None:
    queued = [job for job in store_list_jobs() if job.get("status") == "queued"]
    if not queued:
        return None
    return sorted(queued, key=lambda item: (item.get("queue_index", 0), item.get("created_at") or ""))[0]
