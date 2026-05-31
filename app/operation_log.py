from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = APP_ROOT / "logs"
LOG_PATH = LOG_DIR / "operations.jsonl"


def log_operation(
    action: str,
    status: str,
    target: str | None = None,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "status": status,
        "target": target,
        "details": details or {},
    }
    if error:
        record["error"] = error
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        # Operation logging must never break the primary user action.
        pass
    return record


def read_operation_logs(limit: int = 200) -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    try:
        lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines[-max(1, min(limit, 1000)):]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"timestamp": None, "action": "log_parse_error", "status": "failed", "raw": line})
    return records
