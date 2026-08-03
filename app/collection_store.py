from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = APP_ROOT / "state"
TASK_DIR = STATE_DIR / "collection_tasks"
ID_RE = re.compile(r"^[a-z0-9_-]+$")


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def validate_id(raw_id: str, label: str = "task id") -> str:
    value = str(raw_id or "").strip()
    if not value or not ID_RE.fullmatch(value):
        raise ValueError(f"{label} must match ^[a-z0-9_-]+$")
    return value


def task_path(task_id: str) -> Path:
    return TASK_DIR / f"{validate_id(task_id)}.json"


def task_log_path(task_id: str) -> Path:
    return TASK_DIR / f"{validate_id(task_id)}.log"


def list_tasks() -> list[dict[str, Any]]:
    if not TASK_DIR.exists():
        return []
    tasks = []
    for path in sorted(TASK_DIR.glob("*.json")):
        try:
            tasks.append(load_task(path.stem))
        except Exception:
            continue
    return sorted(tasks, key=lambda item: item.get("created_at") or "", reverse=True)


def load_task(task_id: str) -> dict[str, Any]:
    path = task_path(task_id)
    if not path.exists():
        raise KeyError(f"collection task not found: {task_id}")
    return read_json_retry(path)


def save_task(task: dict[str, Any]) -> dict[str, Any]:
    task_id = validate_id(str(task.get("task_id", "")))
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    task_path(task_id).write_text(json.dumps(task, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return task


def delete_task(task_id: str) -> dict[str, Any]:
    task = load_task(task_id)
    task_path(task_id).unlink(missing_ok=True)
    task_log_path(task_id).unlink(missing_ok=True)
    return task


def append_task_log(task_id: str, line: str) -> None:
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    with task_log_path(task_id).open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def read_task_log(task_id: str, tail: int = 200) -> dict[str, Any]:
    path = task_log_path(task_id)
    if not path.exists():
        return {"task_id": task_id, "lines": [], "text": ""}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = lines[-max(1, min(tail, 5000)) :]
    return {"task_id": task_id, "lines": selected, "text": "\n".join(selected)}


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
