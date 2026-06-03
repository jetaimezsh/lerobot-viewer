from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from app.training_templates import training_template_by_id


APP_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = APP_ROOT / "state"
RECIPE_DIR = STATE_DIR / "training_recipes"
JOB_DIR = STATE_DIR / "training_jobs"
PIPELINE_DIR = STATE_DIR / "training_pipelines"
ID_RE = re.compile(r"^[a-z0-9_-]+$")
ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TRAINING_LAUNCHERS = {"direct", "accelerate"}


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def validate_id(raw_id: str, label: str = "id") -> str:
    value = str(raw_id or "").strip()
    if not value or not ID_RE.fullmatch(value):
        raise ValueError(f"{label} must match ^[a-z0-9_-]+$")
    return value


def recipe_path(recipe_id: str) -> Path:
    return RECIPE_DIR / f"{validate_id(recipe_id, 'recipe id')}.json"


def job_path(job_id: str) -> Path:
    return JOB_DIR / f"{validate_id(job_id, 'job id')}.json"


def job_log_path(job_id: str) -> Path:
    return JOB_DIR / f"{validate_id(job_id, 'job id')}.log"


def pipeline_path(pipeline_id: str) -> Path:
    return PIPELINE_DIR / f"{validate_id(pipeline_id, 'pipeline id')}.json"


def list_recipes() -> list[dict[str, Any]]:
    if not RECIPE_DIR.exists():
        return []
    recipes = []
    for path in sorted(RECIPE_DIR.glob("*.json")):
        try:
            recipes.append(public_recipe(load_recipe(path.stem)))
        except Exception:
            continue
    return recipes


def load_recipe(recipe_id: str) -> dict[str, Any]:
    path = recipe_path(recipe_id)
    if not path.exists():
        raise KeyError(f"training recipe not found: {recipe_id}")
    return read_json_retry(path)


def save_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    recipe_id = validate_id(str(recipe.get("id", "")), "recipe id")
    recipe = normalize_recipe({**recipe, "id": recipe_id})
    RECIPE_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(recipe_path(recipe_id), recipe)
    return recipe


def create_recipe(payload: dict[str, Any]) -> dict[str, Any]:
    recipe_id = validate_id(str(payload.get("id", "")), "recipe id")
    if recipe_path(recipe_id).exists():
        raise ValueError(f"training recipe already exists: {recipe_id}")
    template = training_template_by_id(payload.get("template_id"))
    created_at = now_text()
    recipe = {
        "id": recipe_id,
        "name": payload.get("name") or recipe_id,
        "description": payload.get("description") or template.get("description", ""),
        "framework": payload.get("framework") or template.get("framework") or "lerobot_train",
        "dataset_path": payload.get("dataset_path") or "",
        "episode_filter": payload.get("episode_filter"),
        "output_dir": payload.get("output_dir") or "",
        "hyperparams": {**template.get("hyperparams", {}), **(payload.get("hyperparams") or {})},
        "device": payload.get("device") or "cuda",
        "launcher": payload.get("launcher") or "direct",
        "num_processes": payload.get("num_processes") or 1,
        "gpu_devices": payload.get("gpu_devices") or "",
        "env_vars": payload.get("env_vars") or {},
        "extra_params": payload.get("extra_params") or {},
        "auto_profile_on_complete": payload.get("auto_profile_on_complete", True),
        "profile_name": payload.get("profile_name") or payload.get("name") or recipe_id,
        "profile_adapter": payload.get("profile_adapter") or "lerobot_official",
        "created_at": created_at,
        "updated_at": created_at,
        "inspection": {},
        "status": "created",
    }
    return save_recipe(recipe)


def update_recipe(recipe_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    recipe = load_recipe(recipe_id)
    for key, value in payload.items():
        if key in {"id", "created_at"} or value is None:
            continue
        recipe[key] = value
    recipe["updated_at"] = now_text()
    return save_recipe(recipe)


def delete_recipe(recipe_id: str) -> dict[str, Any]:
    path = recipe_path(recipe_id)
    if not path.exists():
        raise KeyError(f"training recipe not found: {recipe_id}")
    path.unlink()
    return {"ok": True, "deleted": recipe_id}


def normalize_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    recipe.setdefault("description", "")
    recipe.setdefault("framework", "lerobot_train")
    recipe.setdefault("dataset_path", "")
    recipe.setdefault("episode_filter", None)
    recipe.setdefault("output_dir", "")
    recipe.setdefault("hyperparams", {})
    recipe.setdefault("device", "cuda")
    recipe["launcher"] = normalize_launcher(recipe.get("launcher"))
    recipe["num_processes"] = normalize_num_processes(recipe.get("num_processes"))
    recipe["gpu_devices"] = str(recipe.get("gpu_devices") or "").strip()
    recipe["env_vars"] = normalize_env_vars(recipe.get("env_vars") or {})
    recipe.setdefault("extra_params", {})
    recipe.setdefault("auto_profile_on_complete", True)
    recipe.setdefault("profile_name", recipe.get("name") or recipe.get("id"))
    recipe.setdefault("profile_adapter", "lerobot_official")
    recipe.setdefault("inspection", {})
    recipe.setdefault("status", "created")
    recipe.setdefault("created_at", now_text())
    recipe.setdefault("updated_at", recipe["created_at"])
    return recipe


def normalize_launcher(value: Any) -> str:
    launcher = str(value or "direct").strip() or "direct"
    if launcher not in TRAINING_LAUNCHERS:
        raise ValueError(f"training launcher must be one of: {', '.join(sorted(TRAINING_LAUNCHERS))}")
    return launcher


def normalize_num_processes(value: Any) -> int:
    try:
        result = int(value or 1)
    except Exception as exc:
        raise ValueError("num_processes must be an integer") from exc
    if result < 1:
        raise ValueError("num_processes must be >= 1")
    return result


def normalize_env_vars(raw: dict[str, Any] | None) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for key, value in (raw or {}).items():
        name = str(key).strip()
        if not name:
            continue
        if not ENV_VAR_RE.fullmatch(name):
            raise ValueError(f"invalid environment variable name: {name}")
        if value is None:
            text = ""
        elif isinstance(value, (str, int, float, bool)):
            text = str(value)
        else:
            raise ValueError(f"environment variable value must be scalar: {name}")
        env_vars[name] = text
    return env_vars


def public_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": recipe.get("id"),
        "name": recipe.get("name"),
        "description": recipe.get("description", ""),
        "framework": recipe.get("framework", ""),
        "dataset_path": recipe.get("dataset_path", ""),
        "episode_filter": recipe.get("episode_filter"),
        "output_dir": recipe.get("output_dir", ""),
        "hyperparams": recipe.get("hyperparams", {}),
        "device": recipe.get("device", ""),
        "launcher": recipe.get("launcher", "direct"),
        "num_processes": recipe.get("num_processes", 1),
        "gpu_devices": recipe.get("gpu_devices", ""),
        "env_vars": recipe.get("env_vars", {}),
        "extra_params": recipe.get("extra_params", {}),
        "auto_profile_on_complete": bool(recipe.get("auto_profile_on_complete", True)),
        "profile_name": recipe.get("profile_name", ""),
        "profile_adapter": recipe.get("profile_adapter", "lerobot_official"),
        "status": recipe.get("status", "created"),
        "inspection": recipe.get("inspection", {}),
        "created_at": recipe.get("created_at"),
        "updated_at": recipe.get("updated_at"),
    }


def list_jobs() -> list[dict[str, Any]]:
    if not JOB_DIR.exists():
        return []
    jobs = []
    for path in sorted(JOB_DIR.glob("*.json")):
        try:
            jobs.append(load_job(path.stem))
        except Exception:
            continue
    return sorted(jobs, key=lambda item: (job_sort_bucket(item), item.get("queue_index", 0), item.get("created_at") or ""))


def load_job(job_id: str) -> dict[str, Any]:
    path = job_path(job_id)
    if not path.exists():
        raise KeyError(f"training job not found: {job_id}")
    return read_json_retry(path)


def save_job(job: dict[str, Any]) -> dict[str, Any]:
    job_id = validate_id(str(job.get("job_id", "")), "job id")
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(job_path(job_id), job)
    return job


def delete_job_files(job_id: str) -> dict[str, Any]:
    removed = []
    for path in [job_path(job_id), job_log_path(job_id)]:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    if not removed:
        raise KeyError(f"training job not found: {job_id}")
    return {"ok": True, "deleted": job_id, "removed": removed}


def append_job_log(job_id: str, line: str) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    with job_log_path(job_id).open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def read_job_log(job_id: str, tail: int = 200) -> dict[str, Any]:
    path = job_log_path(job_id)
    if not path.exists():
        return {"job_id": job_id, "lines": [], "text": ""}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = lines[-max(1, min(tail, 5000)) :]
    return {"job_id": job_id, "lines": selected, "text": "\n".join(selected)}


def job_sort_bucket(job: dict[str, Any]) -> int:
    status = job.get("status")
    if status == "running":
        return 0
    if status == "queued":
        return 1
    return 2


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job.get("job_id"),
        "recipe_id": job.get("recipe_id"),
        "recipe_name": (job.get("recipe_snapshot") or {}).get("name"),
        "status": job.get("status"),
        "pid": job.get("pid"),
        "queue_index": job.get("queue_index", 0),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "progress": job.get("progress", {}),
        "log_file": job.get("log_file"),
        "error": job.get("error"),
        "returncode": job.get("returncode"),
        "auto_generated_profile_id": job.get("auto_generated_profile_id"),
        "recipe_snapshot": job.get("recipe_snapshot", {}),
    }


def list_pipelines() -> list[dict[str, Any]]:
    if not PIPELINE_DIR.exists():
        return []
    items = []
    for path in sorted(PIPELINE_DIR.glob("*.json")):
        try:
            items.append(load_pipeline(path.stem))
        except Exception:
            continue
    return sorted(items, key=lambda item: item.get("created_at") or "", reverse=True)


def load_pipeline(pipeline_id: str) -> dict[str, Any]:
    path = pipeline_path(pipeline_id)
    if not path.exists():
        raise KeyError(f"training pipeline not found: {pipeline_id}")
    return read_json_retry(path)


def save_pipeline(pipeline: dict[str, Any]) -> dict[str, Any]:
    pipeline_id = validate_id(str(pipeline.get("pipeline_id", "")), "pipeline id")
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(pipeline_path(pipeline_id), pipeline)
    return pipeline


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json_retry(path: Path) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, PermissionError) as exc:
            last_error = exc
            time.sleep(0.02)
    if last_error:
        raise last_error
    raise RuntimeError(f"failed to read JSON: {path}")
