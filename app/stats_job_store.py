from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = APP_ROOT / "state"
JOB_DIR = STATE_DIR / "stats_jobs"
ID_RE = re.compile(r"^[a-z0-9_-]+$")


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def validate_id(raw_id: str, label: str = "job id") -> str:
    value = str(raw_id or "").strip()
    if not value or not ID_RE.fullmatch(value):
        raise ValueError(f"{label} must match ^[a-z0-9_-]+$")
    return value


def job_path(job_id: str) -> Path:
    return JOB_DIR / f"{validate_id(job_id)}.json"


def job_log_path(job_id: str) -> Path:
    return JOB_DIR / f"{validate_id(job_id)}.log"


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
        raise KeyError(f"stats job not found: {job_id}")
    return read_json_retry(path)


def save_job(job: dict[str, Any]) -> dict[str, Any]:
    job_id = validate_id(str(job.get("job_id", "")))
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    job_path(job_id).write_text(json.dumps(job, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return job


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


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "pid": job.get("pid"),
        "queue_index": job.get("queue_index", 0),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "repo_id": job.get("repo_id"),
        "root": job.get("root"),
        "new_repo_id": job.get("new_repo_id"),
        "new_root": job.get("new_root"),
        "command": job.get("command"),
        "display_command": job.get("display_command"),
        "log_file": job.get("log_file"),
        "error": job.get("error"),
        "returncode": job.get("returncode"),
    }


def job_sort_bucket(job: dict[str, Any]) -> int:
    status = job.get("status")
    if status == "running":
        return 0
    if status == "queued":
        return 1
    return 2


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
