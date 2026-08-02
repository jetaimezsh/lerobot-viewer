from __future__ import annotations

from pathlib import Path

from app import stats_job_store
from app.stats_jobs import StatsScheduler


def test_scheduler_marks_stale_running_jobs_failed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stats_job_store, "JOB_DIR", tmp_path)
    stats_job_store.save_job(
        {
            "job_id": "stats_20260802_deadbeef",
            "status": "running",
            "pid": 123,
            "queue_index": 0,
            "created_at": "2026-08-02 10:00:00",
            "started_at": "2026-08-02 10:00:01",
            "finished_at": None,
            "repo_id": "source",
            "root": "C:/datasets/source",
            "new_repo_id": None,
            "new_root": None,
            "command": ["lerobot-edit-dataset", "--help"],
            "display_command": "lerobot-edit-dataset --help",
            "log_file": str(tmp_path / "stats_20260802_deadbeef.log"),
            "error": None,
            "returncode": None,
        }
    )

    scheduler = StatsScheduler()
    scheduler.ensure_started()

    job = stats_job_store.load_job("stats_20260802_deadbeef")
    assert job["status"] == "failed"
    assert job["pid"] is None
    assert "service restarted" in job["error"]
