from __future__ import annotations

import time
import uuid
from typing import Any

from app.operation_log import log_operation
from app.training_executor import submit_training_job
from app.training_store import list_pipelines, load_pipeline, now_text, save_pipeline


def create_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    recipe_id = str(payload.get("recipe_id") or "").strip()
    if not recipe_id:
        raise ValueError("recipe_id is required")
    pipeline_id = f"pipe_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    job = submit_training_job(recipe_id)
    pipeline = {
        "pipeline_id": pipeline_id,
        "recipe_id": recipe_id,
        "training_job_id": job["job_id"],
        "comparison_profile_ids": payload.get("comparison_profile_ids") or [],
        "episodes": payload.get("episodes") or [],
        "status": "training",
        "created_at": now_text(),
        "updated_at": now_text(),
        "steps": [
            {"name": "submit_training", "status": "done", "job_id": job["job_id"]},
            {"name": "training", "status": "running"},
            {"name": "backtest", "status": "waiting"},
            {"name": "report", "status": "waiting"},
        ],
    }
    save_pipeline(pipeline)
    log_operation("training_pipeline_create", "success", target=pipeline_id, details={"recipe_id": recipe_id})
    return pipeline


def list_training_pipelines() -> list[dict[str, Any]]:
    return list_pipelines()


def get_training_pipeline(pipeline_id: str) -> dict[str, Any]:
    return load_pipeline(pipeline_id)
