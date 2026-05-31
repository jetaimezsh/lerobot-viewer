from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.backtesting import (
    BacktestEpisodeRef,
    BacktestRunRequest,
    LOADED_ADAPTERS,
    MODEL_REGISTRY,
    MockBacktestAdapter,
    get_backtest_job,
    model_runtime_status,
    register_model,
    run_backtest,
    run_episode_backtest,
    submit_backtest_job,
)
from app.backtest_store import export_backtest_run, load_backtest_run
from app.main import DatasetCache
from scripts.smoke_test import assert_equal, create_no_video_dataset
from app.backtesting import ModelRegisterRequest


def check_mock_backtest_metrics() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        dataset = Path(directory) / "dataset"
        create_no_video_dataset(dataset, [3])
        cache = DatasetCache(dataset)
        adapter = MockBacktestAdapter([0.0, 0.0])
        result = run_episode_backtest(cache, adapter, "mock", 0)
        assert_equal(result["status"], "done", "mock backtest status")
        assert_equal(result["frames"], 3, "mock backtest frames")
        assert_equal(result["action_dim"], 2, "mock action dimension")
        assert_equal(result["metrics"]["worst_frame"], 2, "mock worst frame")
        assert_equal(len(result["series"]), 2, "mock action series count")


def check_model_registration_inspection() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        checkpoint = Path(directory) / "pretrained_model"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"type": "act"}', encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"test")
        model = register_model(
            ModelRegisterRequest(
                name="test act",
                checkpoint_path=str(checkpoint),
                adapter_type="lerobot_official",
                device="cpu",
            )
        )
        assert_equal(model["inspection"]["valid"], True, "checkpoint inspection valid")
        assert_equal(model["inspection"]["policy_type"], "act", "checkpoint policy type")
        assert_equal(model["inspection"]["file_count"], 1, "checkpoint model file count")


def check_multi_dataset_backtest_request() -> None:
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
        model_id = "mock-multi-dataset"
        MODEL_REGISTRY[model_id] = {
            "id": model_id,
            "name": "mock multi dataset",
            "checkpoint_path": str(work / "mock"),
            "adapter_type": "lerobot_official",
            "device": "cpu",
            "script_path": None,
            "status": "loaded",
            "loaded": True,
            "created_at": "test",
            "inspection": {"valid": True},
        }
        LOADED_ADAPTERS[model_id] = MockBacktestAdapter([0.0, 0.0])
        try:
            result = run_backtest(
                BacktestRunRequest(
                    model_ids=[model_id],
                    episodes=[
                        BacktestEpisodeRef(dataset_path=str(dataset_a), episode_index=0),
                        BacktestEpisodeRef(dataset_path=str(dataset_b), episode_index=0),
                    ],
                ),
                lambda path: caches[str(Path(path).resolve())],
            )
        finally:
            LOADED_ADAPTERS.pop(model_id, None)
            MODEL_REGISTRY.pop(model_id, None)
        assert_equal(result["summary"]["done"], 2, "multi-dataset backtest done count")
        assert_equal(len(result["dataset_paths"]), 2, "multi-dataset path count")
        assert_equal(len({item["episode_key"] for item in result["results"]}), 2, "multi-dataset episode keys")
        assert_equal(result["results"][0]["dataset_name"], "dataset_a", "first result dataset name")
        assert_equal(result["results"][1]["dataset_name"], "dataset_b", "second result dataset name")
        persisted = load_backtest_run(result["run_id"])
        assert_equal(persisted["summary"]["done"], 2, "persisted backtest done count")
        csv_text, csv_media, csv_name = export_backtest_run(result, "csv")
        assert_equal("model_id" in csv_text, True, "csv export includes model_id")
        assert_equal(csv_media.startswith("text/csv"), True, "csv export media type")
        assert_equal(csv_name.endswith(".csv"), True, "csv export filename")
        html_text, html_media, html_name = export_backtest_run(result, "html")
        assert_equal("<table>" in html_text, True, "html export includes table")
        assert_equal(html_media.startswith("text/html"), True, "html export media type")
        assert_equal(html_name.endswith(".html"), True, "html export filename")


def check_backtest_worker_job() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        work = Path(directory)
        dataset = work / "dataset"
        create_no_video_dataset(dataset, [3], task="worker task")
        caches = {str(dataset.resolve()): DatasetCache(dataset)}
        model_id = "mock-worker"
        MODEL_REGISTRY[model_id] = {
            "id": model_id,
            "name": "mock worker",
            "checkpoint_path": str(work / "mock"),
            "adapter_type": "lerobot_official",
            "device": "cpu",
            "script_path": None,
            "status": "loaded",
            "loaded": True,
            "created_at": "test",
            "inspection": {"valid": True},
        }
        LOADED_ADAPTERS[model_id] = MockBacktestAdapter([0.0, 0.0])
        try:
            job = submit_backtest_job(
                BacktestRunRequest(
                    model_ids=[model_id],
                    episodes=[BacktestEpisodeRef(dataset_path=str(dataset), episode_index=0)],
                    max_frames=2,
                ),
                lambda path: caches[str(Path(path).resolve())],
            )
            latest = job
            for _ in range(40):
                latest = get_backtest_job(job["job_id"])
                if latest["status"] in {"done", "failed"}:
                    break
                time.sleep(0.05)
            assert_equal(latest["status"], "done", "worker job status")
            assert_equal(latest["summary"]["done"], 1, "worker job done count")
            assert_equal(bool(latest["run_id"]), True, "worker job persisted run id")
            persisted = load_backtest_run(latest["run_id"])
            assert_equal(persisted["summary"]["done"], 1, "worker persisted done count")
        finally:
            LOADED_ADAPTERS.pop(model_id, None)
            MODEL_REGISTRY.pop(model_id, None)


def main() -> None:
    status = model_runtime_status()
    assert_equal(status["linux_only"], True, "model runtime linux-only flag")
    check_mock_backtest_metrics()
    print("ok: mock backtest metrics passed")
    check_model_registration_inspection()
    print("ok: model registration inspection passed")
    check_multi_dataset_backtest_request()
    print("ok: multi-dataset backtest request passed")
    check_backtest_worker_job()
    print("ok: backtest worker job passed")


if __name__ == "__main__":
    main()
