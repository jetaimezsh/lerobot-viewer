from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.backtesting import MockBacktestAdapter, model_runtime_status, register_model, run_episode_backtest
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


def main() -> None:
    status = model_runtime_status()
    assert_equal(status["linux_only"], True, "model runtime linux-only flag")
    check_mock_backtest_metrics()
    print("ok: mock backtest metrics passed")
    check_model_registration_inspection()
    print("ok: model registration inspection passed")


if __name__ == "__main__":
    main()
