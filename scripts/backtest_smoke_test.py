from __future__ import annotations

import sys
import tempfile
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.backtest_store import export_backtest_run, load_backtest_run
from app.backtesting import (
    BacktestEpisodeRef,
    BacktestRunRequest,
    ProfileTestRequest,
    close_profile_adapter,
    get_backtest_job,
    inspect_profile_record,
    list_backtest_jobs,
    load_profile_adapter,
    model_runtime_status,
    run_backtest,
    run_episode_backtest,
    submit_backtest_job,
    test_profile_on_frame,
)
from app.profile_store import create_profile, delete_profile, load_profile
from app.main import DatasetCache
from scripts.smoke_test import assert_equal, create_no_video_dataset


def new_profile_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def cleanup_profile(profile_id: str) -> None:
    close_profile_adapter(profile_id)
    try:
        delete_profile(profile_id)
    except Exception:
        pass


def create_mock_profile(profile_id: str, action: list[float], runtime_params: dict | None = None) -> dict:
    params = {"action": action, **(runtime_params or {})}
    return create_profile(
        {
            "id": profile_id,
            "name": profile_id,
            "adapter": "mock",
            "device": "cpu",
            "runtime_params": params,
            "extra_params": {"test": True},
        }
    )


def check_profile_crud_and_inspection() -> None:
    profile_id = new_profile_id("official")
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        checkpoint = Path(directory) / "pretrained_model"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"type": "act", "n_obs_steps": 2}', encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"test")
        try:
            create_profile(
                {
                    "id": profile_id,
                    "name": "official smoke",
                    "template_id": "act",
                    "checkpoint_path": str(checkpoint),
                    "adapter": "lerobot_official",
                    "device": "cpu",
                }
            )
            inspected = inspect_profile_record(profile_id)
            assert_equal(inspected["inspection"]["valid"], True, "official profile inspection valid")
            assert_equal(inspected["inspection"]["policy_type"], "act", "official profile policy type")
            assert_equal(inspected["checkpoint_config"]["n_obs_steps"], 2, "checkpoint config captured")
        finally:
            cleanup_profile(profile_id)


def check_mock_backtest_metrics() -> None:
    profile_id = new_profile_id("mock_metrics")
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        dataset = Path(directory) / "dataset"
        create_no_video_dataset(dataset, [3])
        cache = DatasetCache(dataset)
        try:
            profile = create_mock_profile(profile_id, [0.0, 0.0])
            load_profile_adapter(profile_id)
            from app.backtesting import LOADED_ADAPTERS

            result = run_episode_backtest(cache, LOADED_ADAPTERS[profile_id], profile, 0)
            assert_equal(result["status"], "done", "mock backtest status")
            assert_equal(result["profile_id"], profile_id, "mock backtest profile id")
            assert_equal(result["frames"], 3, "mock backtest frames")
            assert_equal(result["action_dim"], 2, "mock action dimension")
            assert_equal(result["metrics"]["worst_frame"], 2, "mock worst frame")
            assert_equal(len(result["series"]), 2, "mock action series count")
            quick = test_profile_on_frame(
                profile_id,
                ProfileTestRequest(dataset_path=str(dataset), episode_index=0, frame_index=1),
                lambda path: cache,
            )
            assert_equal(quick["action_shape"], [2], "profile real-frame test action shape")
        finally:
            cleanup_profile(profile_id)


def check_multi_dataset_backtest_request() -> None:
    profile_id = new_profile_id("mock_multi")
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        work = Path(directory)
        dataset_a = work / "dataset_a"
        dataset_b = work / "dataset_b"
        create_no_video_dataset(dataset_a, [3], task="task a")
        create_no_video_dataset(dataset_b, [4], task="task b")
        caches = {
            str(dataset_a.resolve()): DatasetCache(dataset_a),
            str(dataset_b.resolve()): DatasetCache(dataset_b),
        }
        try:
            create_mock_profile(profile_id, [0.0, 0.0])
            result = run_backtest(
                BacktestRunRequest(
                    profile_ids=[profile_id],
                    episodes=[
                        BacktestEpisodeRef(dataset_path=str(dataset_a), episode_index=0),
                        BacktestEpisodeRef(dataset_path=str(dataset_b), episode_index=0),
                    ],
                ),
                lambda path: caches[str(Path(path).resolve())],
            )
            assert_equal(result["summary"]["done"], 2, "multi-dataset backtest done count")
            assert_equal(result["profile_ids"], [profile_id], "multi-dataset profile ids")
            assert_equal(len(result["dataset_paths"]), 2, "multi-dataset path count")
            assert_equal(len({item["episode_key"] for item in result["results"]}), 2, "multi-dataset episode keys")
            assert_equal(result["results"][0]["dataset_name"], "dataset_a", "first result dataset name")
            assert_equal(result["results"][1]["dataset_name"], "dataset_b", "second result dataset name")
            assert_equal(result["results"][0]["profile_snapshot"]["id"], profile_id, "profile snapshot captured")
            persisted = load_backtest_run(result["run_id"])
            assert_equal(persisted["summary"]["done"], 2, "persisted backtest done count")
            csv_text, csv_media, csv_name = export_backtest_run(result, "csv")
            assert_equal("profile_id" in csv_text, True, "csv export includes profile_id")
            assert_equal(csv_media.startswith("text/csv"), True, "csv export media type")
            assert_equal(csv_name.endswith(".csv"), True, "csv export filename")
            html_text, html_media, html_name = export_backtest_run(result, "html")
            assert_equal("<table>" in html_text, True, "html export includes table")
            assert_equal(html_media.startswith("text/html"), True, "html export media type")
            assert_equal(html_name.endswith(".html"), True, "html export filename")
        finally:
            cleanup_profile(profile_id)


def check_backtest_worker_job() -> None:
    profile_id = new_profile_id("mock_worker")
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        work = Path(directory)
        dataset = work / "dataset"
        create_no_video_dataset(dataset, [3], task="worker task")
        caches = {str(dataset.resolve()): DatasetCache(dataset)}
        try:
            create_mock_profile(profile_id, [0.0, 0.0], {"sleep_ms": 30})
            first_job = submit_backtest_job(
                BacktestRunRequest(
                    profile_ids=[profile_id],
                    episodes=[BacktestEpisodeRef(dataset_path=str(dataset), episode_index=0)],
                    max_frames=2,
                ),
                lambda path: caches[str(Path(path).resolve())],
            )
            second_job = submit_backtest_job(
                BacktestRunRequest(
                    profile_ids=[profile_id],
                    episodes=[BacktestEpisodeRef(dataset_path=str(dataset), episode_index=0)],
                    max_frames=2,
                ),
                lambda path: caches[str(Path(path).resolve())],
            )
            seen_jobs = {item["job_id"] for item in list_backtest_jobs()}
            assert_equal(first_job["job_id"] in seen_jobs, True, "first queued job listed")
            assert_equal(second_job["job_id"] in seen_jobs, True, "second queued job listed")
            latest = first_job
            second_latest = second_job
            for _ in range(40):
                latest = get_backtest_job(first_job["job_id"])
                second_latest = get_backtest_job(second_job["job_id"])
                if latest["status"] in {"done", "failed"} and second_latest["status"] in {"done", "failed"}:
                    break
                time.sleep(0.05)
            assert_equal(latest["status"], "done", "worker job status")
            assert_equal(second_latest["status"], "done", "second worker job status")
            assert_equal(latest["summary"]["done"], 1, "worker job done count")
            assert_equal(second_latest["summary"]["done"], 1, "second worker job done count")
            assert_equal(bool(latest["run_id"]), True, "worker job persisted run id")
            assert_equal(bool(second_latest["run_id"]), True, "second worker job persisted run id")
            assert_equal(latest["run_id"] != second_latest["run_id"], True, "queued jobs have separate runs")
            persisted = load_backtest_run(latest["run_id"])
            assert_equal(persisted["summary"]["done"], 1, "worker persisted done count")
        finally:
            cleanup_profile(profile_id)


def main() -> None:
    status = model_runtime_status()
    assert_equal(status["linux_only"], True, "model runtime linux-only flag")
    assert_equal("adapters" in status, True, "adapter list present")
    check_profile_crud_and_inspection()
    print("ok: profile crud and inspection passed")
    check_mock_backtest_metrics()
    print("ok: mock backtest metrics passed")
    check_multi_dataset_backtest_request()
    print("ok: multi-dataset profile backtest request passed")
    check_backtest_worker_job()
    print("ok: backtest worker job passed")


if __name__ == "__main__":
    main()
