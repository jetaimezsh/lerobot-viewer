from __future__ import annotations

import os
import sys
import tempfile
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import app.backtesting as backtesting_module
from app.adapters import BacktestAdapter
from app.backtest_store import (
    export_action_csv,
    export_action_zip,
    export_backtest_run,
    load_backtest_job_record,
    load_backtest_run,
)
from app.backtesting import (
    BacktestEpisodeRef,
    BacktestRunRequest,
    ProfileTestRequest,
    build_observation,
    cancel_backtest_job,
    close_profile_adapter,
    decode_video_observations,
    get_backtest_job,
    inspect_profile_record,
    list_backtest_jobs,
    load_profile_adapter,
    model_runtime_status,
    run_backtest,
    run_episode_backtest,
    submit_backtest_job,
    test_profile_on_frame as run_profile_frame_check,
)
from app.profile_store import create_profile, delete_profile, load_profile, update_profile
from app.adapters.lerobot_official import LeRobotOfficialAdapter, make_processors_compat, policy_action_to_array
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


def check_multiple_profile_persistence() -> None:
    first_id = new_profile_id("mock_first")
    second_id = new_profile_id("mock_second")
    try:
        create_profile(
            {
                "id": first_id,
                "name": "first profile",
                "adapter": "mock",
                "device": "cpu",
                "checkpoint_path": "/tmp/first-checkpoint",
                "runtime_params": {"action": [1.0, 2.0]},
            }
        )
        create_profile(
            {
                "id": second_id,
                "name": "second profile",
                "adapter": "mock",
                "device": "cpu",
                "checkpoint_path": "/tmp/second-checkpoint",
                "runtime_params": {"action": [3.0, 4.0]},
            }
        )
        update_profile(first_id, {"name": "first profile updated", "checkpoint_path": "/tmp/first-updated"})
        first = load_profile(first_id)
        second = load_profile(second_id)
        assert_equal(first["name"], "first profile updated", "first profile updated independently")
        assert_equal(first["checkpoint_path"], "/tmp/first-updated", "first checkpoint updated independently")
        assert_equal(second["name"], "second profile", "second profile name preserved")
        assert_equal(second["checkpoint_path"], "/tmp/second-checkpoint", "second checkpoint preserved")
    finally:
        cleanup_profile(first_id)
        cleanup_profile(second_id)


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
            quick = run_profile_frame_check(
                profile_id,
                ProfileTestRequest(dataset_path=str(dataset), episode_index=0, frame_index=1),
                lambda path: cache,
            )
            assert_equal(quick["action_shape"], [2], "profile real-frame test action shape")
        finally:
            cleanup_profile(profile_id)


def check_multi_dataset_backtest_request() -> None:
    profile_id = new_profile_id("mock_multi")
    second_profile_id = new_profile_id("mock_multi_b")
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
            create_mock_profile(second_profile_id, [1.0, 1.0])
            result = run_backtest(
                BacktestRunRequest(
                    profile_ids=[profile_id, second_profile_id],
                    episodes=[
                        BacktestEpisodeRef(dataset_path=str(dataset_a), episode_index=0),
                        BacktestEpisodeRef(dataset_path=str(dataset_b), episode_index=0),
                    ],
                ),
                lambda path: caches[str(Path(path).resolve())],
            )
            assert_equal(result["summary"]["done"], 4, "multi-dataset multi-profile backtest done count")
            assert_equal(result["profile_ids"], [profile_id, second_profile_id], "multi-dataset profile ids")
            assert_equal(len(result["dataset_paths"]), 2, "multi-dataset path count")
            assert_equal(len({item["episode_key"] for item in result["results"]}), 2, "multi-dataset episode keys")
            assert_equal(result["results"][0]["dataset_name"], "dataset_a", "first result dataset name")
            assert_equal(result["results"][1]["dataset_name"], "dataset_b", "second result dataset name")
            assert_equal(result["results"][0]["profile_snapshot"]["id"], profile_id, "profile snapshot captured")
            persisted = load_backtest_run(result["run_id"])
            assert_equal(persisted["summary"]["done"], 4, "persisted backtest done count")
            csv_text, csv_media, csv_name = export_backtest_run(result, "csv")
            assert_equal("profile_id" in csv_text, True, "csv export includes profile_id")
            assert_equal(csv_media.startswith("text/csv"), True, "csv export media type")
            assert_equal(csv_name.endswith(".csv"), True, "csv export filename")
            html_text, html_media, html_name = export_backtest_run(result, "html")
            assert_equal("<table>" in html_text, True, "html export includes table")
            assert_equal(html_media.startswith("text/html"), True, "html export media type")
            assert_equal(html_name.endswith(".html"), True, "html export filename")
            action_csv, action_media, action_name = export_action_csv(result, 0)
            assert_equal("ground_truth_action_0" in action_csv, True, "action csv includes ground truth")
            assert_equal("predicted_action_0" in action_csv, True, "action csv includes prediction")
            assert_equal("error_action_0" in action_csv, True, "action csv includes error")
            assert_equal(action_media.startswith("text/csv"), True, "action csv media type")
            assert_equal(action_name.endswith("_actions.csv"), True, "action csv filename")
            zip_bytes, zip_media, zip_name = export_action_zip(result)
            assert_equal(zip_media, "application/zip", "action zip media type")
            assert_equal(zip_name.endswith("_actions.zip"), True, "action zip filename")
            with zipfile.ZipFile(BytesIO(zip_bytes), "r") as archive:
                names = archive.namelist()
                assert_equal(len(names), 4, "action zip one csv per model-episode")
                first_csv = archive.read(names[0]).decode("utf-8-sig")
                assert_equal("predicted_action_1" in first_csv, True, "action zip csv includes action dimensions")
            assert_equal(profile_id in backtesting_module.LOADED_ADAPTERS, False, "first run-loaded profile auto unloaded")
            assert_equal(second_profile_id in backtesting_module.LOADED_ADAPTERS, False, "second run-loaded profile auto unloaded")
        finally:
            cleanup_profile(profile_id)
            cleanup_profile(second_profile_id)


def check_backtest_environment_variables() -> None:
    profile_id = new_profile_id("mock_env")
    env_key = "LEROBOT_VIEWER_BACKTEST_ACTION"
    original = os.environ.get(env_key)
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        work = Path(directory)
        dataset = work / "dataset"
        create_no_video_dataset(dataset, [2], task="env task")
        cache = DatasetCache(dataset)
        try:
            create_mock_profile(profile_id, [0.0, 0.0], {"action_from_env": env_key})
            result = run_backtest(
                BacktestRunRequest(
                    profile_ids=[profile_id],
                    episodes=[BacktestEpisodeRef(dataset_path=str(dataset), episode_index=0)],
                    env_vars={env_key: "5.5,6.5", "CUDA_VISIBLE_DEVICES": "0,1", "SECRET_TOKEN": "hidden"},
                ),
                lambda path: cache,
            )
            assert_equal(result["summary"]["done"], 1, "env backtest done count")
            assert_equal(result["env_vars"][env_key], "5.5,6.5", "env var recorded")
            assert_equal(result["env_vars"]["CUDA_VISIBLE_DEVICES"], "0,1", "cuda env var recorded")
            assert_equal(result["env_vars"]["SECRET_TOKEN"], "***", "sensitive env var masked")
            assert_equal(result["results"][0]["series"][0]["predicted"][0], 5.5, "env action dim 0 applied")
            assert_equal(result["results"][0]["series"][1]["predicted"][0], 6.5, "env action dim 1 applied")
            assert_equal(os.environ.get(env_key), original, "env var restored after backtest")
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
            persisted_job = load_backtest_job_record(first_job["job_id"])
            assert_equal(persisted_job["status"], "done", "worker job status persisted")
            assert_equal(persisted_job["run_id"], latest["run_id"], "worker job run id persisted")
            persisted = load_backtest_run(latest["run_id"])
            assert_equal(persisted["summary"]["done"], 1, "worker persisted done count")
            assert_equal("result" in latest, False, "worker job does not keep full result in memory")
            persisted_job = load_backtest_job_record(first_job["job_id"])
            assert_equal("result" in persisted_job, False, "worker job does not persist full result")
        finally:
            cleanup_profile(profile_id)


def check_backtest_worker_cancel_queued_job() -> None:
    profile_id = new_profile_id("mock_cancel")
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        work = Path(directory)
        dataset = work / "dataset"
        create_no_video_dataset(dataset, [3], task="cancel task")
        caches = {str(dataset.resolve()): DatasetCache(dataset)}
        try:
            create_mock_profile(profile_id, [0.0, 0.0], {"sleep_ms": 80})
            first_job = submit_backtest_job(
                BacktestRunRequest(
                    profile_ids=[profile_id],
                    episodes=[BacktestEpisodeRef(dataset_path=str(dataset), episode_index=0)],
                    max_frames=3,
                ),
                lambda path: caches[str(Path(path).resolve())],
            )
            second_job = submit_backtest_job(
                BacktestRunRequest(
                    profile_ids=[profile_id],
                    episodes=[BacktestEpisodeRef(dataset_path=str(dataset), episode_index=0)],
                    max_frames=3,
                ),
                lambda path: caches[str(Path(path).resolve())],
            )
            cancelled = cancel_backtest_job(second_job["job_id"])
            assert_equal(cancelled["status"], "cancelled", "queued backtest job cancelled")
            assert_equal(cancelled["run_id"], None, "cancelled job has no run id")
            latest = first_job
            for _ in range(40):
                latest = get_backtest_job(first_job["job_id"])
                if latest["status"] in {"done", "failed"}:
                    break
                time.sleep(0.05)
            assert_equal(latest["status"], "done", "first job still completes after queued cancel")
            second_latest = get_backtest_job(second_job["job_id"])
            assert_equal(second_latest["status"], "cancelled", "cancelled job remains cancelled")
            assert_equal(second_latest["run_id"], None, "cancelled job never creates a run")
        finally:
            cleanup_profile(profile_id)


class VideoBatchAdapter(BacktestAdapter):
    adapter_name = "video_batch_test"
    label = "Video batch test"
    test_only = False

    def __init__(self) -> None:
        super().__init__()
        self.predict_calls = 0
        self.predict_batch_calls: list[int] = []

    def load(self, profile: dict) -> None:
        self.profile = profile

    def predict(self, observation: dict) -> np.ndarray:
        self.predict_calls += 1
        assert_equal("observation.images.front" in observation, True, "front image passed to video adapter")
        assert_equal("observation.images.wrist" in observation, True, "wrist image passed to video adapter")
        return np.asarray([0.0, 0.0], dtype=np.float64)

    def predict_batch(self, observations: list[dict]) -> list[np.ndarray]:
        self.predict_batch_calls.append(len(observations))
        for observation in observations:
            assert_equal("observation.images.front" in observation, True, "front image passed to video adapter")
            assert_equal("observation.images.wrist" in observation, True, "wrist image passed to video adapter")
        return [np.asarray([0.0, 0.0], dtype=np.float64) for _ in observations]

    def reset_episode(self) -> None:
        return None

    def unload(self) -> None:
        self.profile = None


def check_chunked_video_backtest_decode() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        dataset = Path(directory) / "dataset"
        create_no_video_dataset(dataset, [45])
        cache = DatasetCache(dataset)
        cache.features = {
            **cache.features,
            "observation.images.front": {"dtype": "video", "shape": [3, 8, 8]},
            "observation.images.wrist": {"dtype": "video", "shape": [3, 8, 8]},
        }
        cache.video_keys = ["observation.images.front", "observation.images.wrist"]
        profile = {
            "id": "video_batch_profile",
            "name": "video batch profile",
            "checkpoint_path": "",
            "adapter": "video_batch_test",
            "device": "cpu",
            "runtime_params": {},
            "extra_params": {},
            "status": "created",
            "inspection": {},
        }
        adapter = VideoBatchAdapter()
        adapter.load(profile)
        calls: list[tuple[int, int]] = []
        original_decode = backtesting_module.decode_video_observations
        original_chunk = backtesting_module.BACKTEST_VIDEO_DECODE_CHUNK_FRAMES

        def fake_decode(_cache, _episode, start_frame: int, frame_count: int) -> dict[str, np.ndarray]:
            calls.append((start_frame, frame_count))
            assert_equal(frame_count <= 7, True, "video decode chunk does not exceed configured batch size")
            return {
                "observation.images.front": np.zeros((frame_count, 3, 8, 8), dtype=np.float32),
                "observation.images.wrist": np.ones((frame_count, 3, 8, 8), dtype=np.float32),
            }

        try:
            backtesting_module.BACKTEST_VIDEO_DECODE_CHUNK_FRAMES = 7
            backtesting_module.decode_video_observations = fake_decode
            result = run_episode_backtest(cache, adapter, profile, 0)
            assert_equal(result["status"], "done", "chunked video backtest status")
            assert_equal(result["frames"], 45, "chunked video backtest frames")
            assert_equal(calls, [(0, 7), (7, 7), (14, 7), (21, 7), (28, 7), (35, 7), (42, 3)], "video decode chunk plan")
            assert_equal(adapter.predict_batch_calls, [7, 7, 7, 7, 7, 7, 3], "video predictions batched by chunk")
            assert_equal(adapter.predict_calls, 0, "video backtest does not fall back to per-frame predict")
        finally:
            backtesting_module.decode_video_observations = original_decode
            backtesting_module.BACKTEST_VIDEO_DECODE_CHUNK_FRAMES = original_chunk


def check_video_observation_builder() -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "observation.images.front": {"dtype": "video", "shape": [3, 8, 8]},
        "observation.images.wrist": {"dtype": "video", "shape": [3, 8, 8]},
        "action": {"dtype": "float32", "shape": [2]},
    }
    frame = pd.Series(
        {
            "observation.state": np.array([1, 2], dtype=np.float32),
            "action": np.array([0, 0], dtype=np.float32),
            "task_index": 0,
            "timestamp": 0.0,
        }
    )
    video_observations = {
        "observation.images.front": np.zeros((2, 3, 8, 8), dtype=np.float32),
        "observation.images.wrist": np.ones((2, 3, 8, 8), dtype=np.float32),
    }
    observation = build_observation(frame, features, video_observations, 1)
    assert_equal("observation.images.front" in observation, True, "front image observation present")
    assert_equal("observation.images.wrist" in observation, True, "wrist image observation present")
    assert_equal(list(observation["observation.images.wrist"].shape), [3, 8, 8], "wrist image chw shape")
    assert_equal(float(observation["observation.images.wrist"].max()), 1.0, "wrist image normalized value")


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def shape(self):
        return self.value.shape

    def reshape(self, *shape):
        return FakeTensor(self.value.reshape(*shape))

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.value, axis))

    def to(self, _device):
        return self

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    @staticmethod
    def as_tensor(value):
        return FakeTensor(value)

    @staticmethod
    def inference_mode():
        return FakeInferenceMode()


class FakePolicy:
    config = {"type": "fake"}

    def __init__(self):
        self.seen_batch = None

    def select_action(self, batch):
        self.seen_batch = batch
        assert_equal(batch.get("normalized"), True, "official adapter preprocessor applied")
        sample = batch.get("sample") or {}
        state = sample.get("observation.state")
        if getattr(state, "shape", ()) and state.shape[0] == 2:
            return np.asarray([[0.25, 0.5], [0.75, 1.0]], dtype=np.float32)
        return np.asarray([[0.25, 0.5]], dtype=np.float32)


def check_official_adapter_processor_flow() -> None:
    adapter = LeRobotOfficialAdapter()
    policy = FakePolicy()
    adapter.profile = {"id": "fake", "device": "cpu", "checkpoint_path": "/tmp/fake"}
    adapter.torch = FakeTorch()
    adapter.policy = policy
    adapter.preprocessor = lambda sample: {"normalized": True, "sample": sample}
    adapter.postprocessor = lambda action: np.asarray(action) * 4.0
    action = adapter.predict({"observation.state": np.asarray([1.0, 2.0]), "timestamp": 0.0})
    assert_equal(action.tolist(), [1.0, 2.0], "official adapter postprocessor unnormalizes action")
    assert_equal("sample" in policy.seen_batch, True, "official adapter forwards preprocessed sample")
    actions = adapter.predict_batch([
        {"observation.state": np.asarray([1.0, 2.0]), "timestamp": 0.0},
        {"observation.state": np.asarray([3.0, 4.0]), "timestamp": 0.1},
    ])
    assert_equal([action.tolist() for action in actions], [[1.0, 2.0], [3.0, 4.0]], "official adapter batched actions split per observation")
    assert_equal(policy_action_to_array({"action": [[3.0, 4.0]]}).tolist(), [3.0, 4.0], "policy action dict flattened")

    def keyword_only_factory(*, policy_cfg, pretrained_path):
        assert_equal(policy_cfg["type"], "fake", "processor compat forwards policy config")
        assert_equal(pretrained_path, "/tmp/fake", "processor compat forwards checkpoint path")
        return "pre", "post"

    preprocessor, postprocessor = make_processors_compat(
        keyword_only_factory,
        {"type": "fake"},
        "/tmp/fake",
        {"action": {"mean": [0.0], "std": [1.0]}},
    )
    assert_equal([preprocessor, postprocessor], ["pre", "post"], "processor factory compatibility")


def check_real_video_decode_if_available() -> None:
    from app.editing import ffmpeg_executable

    dataset = PROJECT_ROOT / "sample_datasets" / "pusht"
    if not dataset.exists() or not ffmpeg_executable():
        print("skip: video observation decode check skipped")
        return
    cache = DatasetCache(dataset)
    episode = cache.episode_record(0)
    decoded = decode_video_observations(cache, episode, 0, 2)
    key = cache.video_keys[0]
    assert_equal(key in decoded, True, "decoded video key present")
    assert_equal(list(decoded[key].shape[:2]), [2, 3], "decoded video nchw prefix")
    assert_equal(float(decoded[key].min()) >= 0.0, True, "decoded video min range")
    assert_equal(float(decoded[key].max()) <= 1.0, True, "decoded video max range")


def main() -> None:
    status = model_runtime_status()
    assert_equal(status["linux_only"], True, "model runtime linux-only flag")
    assert_equal("adapters" in status, True, "adapter list present")
    check_profile_crud_and_inspection()
    print("ok: profile crud and inspection passed")
    check_multiple_profile_persistence()
    print("ok: multiple profile persistence passed")
    check_mock_backtest_metrics()
    print("ok: mock backtest metrics passed")
    check_multi_dataset_backtest_request()
    print("ok: multi-dataset profile backtest request passed")
    check_backtest_environment_variables()
    print("ok: backtest environment variables passed")
    check_backtest_worker_job()
    print("ok: backtest worker job passed")
    check_backtest_worker_cancel_queued_job()
    print("ok: backtest queued job cancellation passed")
    check_chunked_video_backtest_decode()
    print("ok: chunked video backtest decode passed")
    check_video_observation_builder()
    print("ok: video observation builder passed")
    check_official_adapter_processor_flow()
    print("ok: official adapter processor flow passed")
    check_real_video_decode_if_available()


if __name__ == "__main__":
    main()
