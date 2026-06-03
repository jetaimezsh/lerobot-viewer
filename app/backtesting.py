from __future__ import annotations

import importlib
import platform
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.adapters import BacktestAdapter, get_adapter, list_adapters
from app.backtest_store import save_backtest_run
from app.editing import ffmpeg_executable, ffprobe_video_stream, read_episode_frames, resolve_dataset_path
from app.operation_log import log_operation
from app.profile_store import (
    load_profile,
    list_profiles as store_list_profiles,
    profile_snapshot,
    public_profile,
    save_profile,
)


class BacktestEpisodeRef(BaseModel):
    dataset_path: str
    episode_index: int


class BacktestRunRequest(BaseModel):
    profile_ids: list[str] = Field(default_factory=list)
    episodes: list[BacktestEpisodeRef] = Field(default_factory=list)
    max_frames: int | None = None


class ProfileTestRequest(BaseModel):
    dataset_path: str
    episode_index: int
    frame_index: int = 0


LOADED_ADAPTERS: dict[str, BacktestAdapter] = {}
BACKTEST_RUNS: dict[str, dict[str, Any]] = {}
BACKTEST_JOBS: dict[str, dict[str, Any]] = {}
BACKTEST_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="backtest-worker")
BACKTEST_LOCK = Lock()


def model_runtime_status() -> dict[str, Any]:
    checks = [
        python_package_status("torch"),
        python_package_status("lerobot"),
        python_package_status("safetensors"),
        python_package_status("numpy", np.__version__),
        python_package_status("pandas", pd.__version__),
    ]
    torch_check = next((item for item in checks if item["id"] == "torch"), None)
    cuda = {"available": False, "device_count": 0, "devices": []}
    if torch_check and torch_check["ok"]:
        try:
            torch = importlib.import_module("torch")
            cuda_available = bool(torch.cuda.is_available())
            cuda = {
                "available": cuda_available,
                "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
                "devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(int(torch.cuda.device_count()))
                ] if cuda_available else [],
            }
        except Exception as exc:
            cuda = {"available": False, "device_count": 0, "devices": [], "error": str(exc)}

    is_linux = platform.system().lower() == "linux"
    missing = [item["id"] for item in checks if not item["ok"] and item["required_for_lerobot"]]
    if not is_linux:
        missing.insert(0, "linux")
    return {
        "os": platform.platform(),
        "is_linux": is_linux,
        "linux_only": True,
        "ready_for_lerobot_backtest": is_linux and not missing,
        "checks": checks,
        "cuda": cuda,
        "missing": missing,
        "profiles_loaded": sorted(LOADED_ADAPTERS.keys()),
        "adapters": list_adapters(),
        "worker": backtest_worker_status(),
    }


def python_package_status(name: str, version: str | None = None) -> dict[str, Any]:
    if version is None:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
        except Exception as exc:
            return {
                "id": name,
                "label": name,
                "ok": False,
                "required_for_lerobot": name in {"torch", "lerobot"},
                "detail": str(exc),
            }
    return {
        "id": name,
        "label": name,
        "ok": True,
        "required_for_lerobot": name in {"torch", "lerobot"},
        "detail": version,
    }


def list_profile_records() -> list[dict[str, Any]]:
    return [public_profile_runtime(profile) for profile in store_list_profiles()]


def public_profile_runtime(profile: dict[str, Any]) -> dict[str, Any]:
    result = public_profile(profile)
    result["loaded"] = profile["id"] in LOADED_ADAPTERS
    if result["loaded"]:
        result["status"] = "loaded"
    return result


def inspect_profile_record(profile_id: str) -> dict[str, Any]:
    profile = load_profile(profile_id)
    adapter_cls = get_adapter(profile["adapter"])
    inspection = adapter_cls.inspect_profile(profile)
    profile["inspection"] = inspection
    if inspection.get("checkpoint_config"):
        profile["checkpoint_config"] = inspection["checkpoint_config"]
    profile["status"] = "invalid" if inspection.get("errors") else "inspected"
    save_profile(profile)
    return public_profile_runtime(profile)


def load_profile_adapter(profile_id: str) -> dict[str, Any]:
    profile = load_profile(profile_id)
    if profile_id in LOADED_ADAPTERS:
        return public_profile_runtime(profile)
    adapter_cls = get_adapter(profile["adapter"])
    inspection = adapter_cls.inspect_profile(profile)
    if inspection.get("errors"):
        profile["inspection"] = inspection
        profile["status"] = "invalid"
        save_profile(profile)
        raise RuntimeError("; ".join(inspection["errors"]))
    adapter = adapter_cls()
    adapter.load(profile)
    profile["inspection"] = {**inspection, **adapter.runtime_info()}
    profile["checkpoint_config"] = inspection.get("checkpoint_config", profile.get("checkpoint_config", {}))
    profile["status"] = "loaded"
    save_profile(profile)
    LOADED_ADAPTERS[profile_id] = adapter
    return public_profile_runtime(profile)


def unload_profile_adapter(profile_id: str) -> dict[str, Any]:
    adapter = LOADED_ADAPTERS.pop(profile_id, None)
    if adapter:
        adapter.unload()
    profile = load_profile(profile_id)
    profile["status"] = "inspected" if profile.get("inspection", {}).get("valid") else "created"
    save_profile(profile)
    return public_profile_runtime(profile)


def close_profile_adapter(profile_id: str) -> None:
    adapter = LOADED_ADAPTERS.pop(profile_id, None)
    if adapter:
        adapter.unload()


def test_profile_on_frame(profile_id: str, request: ProfileTestRequest, cache_loader: Any) -> dict[str, Any]:
    profile = load_profile(profile_id)
    adapter = LOADED_ADAPTERS.get(profile_id)
    if adapter is None:
        load_profile_adapter(profile_id)
        adapter = LOADED_ADAPTERS[profile_id]
    cache = cache_loader(request.dataset_path)
    episode = cache.episode_record(int(request.episode_index))
    frames = read_episode_frames(cache, episode)
    if frames.empty:
        raise ValueError("episode has no frames")
    frame_index = min(max(int(request.frame_index), 0), len(frames) - 1)
    frame = frames.iloc[frame_index]
    video_observations = (
        decode_video_observations(cache, episode, frame_index, 1)
        if adapter_needs_video_observations(adapter, cache.features)
        else {}
    )
    adapter.prepare_backtest_context(cache)
    observation = build_observation(frame, cache.features, video_observations, 0)
    started = time.perf_counter()
    action = adapter.predict(observation)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "profile": profile_snapshot(profile),
        "dataset_path": str(cache.root),
        "episode_index": int(request.episode_index),
        "frame_index": frame_index,
        "observation_keys": sorted(observation.keys()),
        "action_shape": list(np.asarray(action).shape),
        "action_preview": [round(float(item), 8) for item in np.asarray(action).reshape(-1)[:20]],
        "elapsed_ms": elapsed_ms,
    }


def run_backtest(request: BacktestRunRequest, cache_loader: Any) -> dict[str, Any]:
    if not request.profile_ids:
        raise ValueError("at least one profile is required")
    if not request.episodes:
        raise ValueError("at least one episode is required")

    caches: dict[str, Any] = {}
    for ref in request.episodes:
        dataset_path = str(resolve_dataset_path(ref.dataset_path))
        if dataset_path not in caches:
            caches[dataset_path] = cache_loader(dataset_path)

    run_id = uuid.uuid4().hex[:12]
    results: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for profile_id in request.profile_ids:
        try:
            profile = load_profile(profile_id)
            snapshots.append(profile_snapshot(profile))
            adapter = LOADED_ADAPTERS.get(profile_id)
            if adapter is None:
                try:
                    load_profile_adapter(profile_id)
                    adapter = LOADED_ADAPTERS[profile_id]
                except Exception as exc:
                    results.extend(failed_results(profile, request.episodes, str(exc)))
                    continue
            for ref in request.episodes:
                dataset_path = str(resolve_dataset_path(ref.dataset_path))
                results.append(
                    run_episode_backtest(
                        caches[dataset_path],
                        adapter,
                        profile,
                        int(ref.episode_index),
                        request.max_frames,
                        dataset_path=dataset_path,
                    )
                )
        except Exception as exc:
            fallback = {"id": profile_id, "name": profile_id, "adapter": "unknown", "device": "", "runtime_params": {}, "extra_params": {}}
            snapshots.append(fallback)
            results.extend(failed_results(fallback, request.episodes, str(exc)))

    run = {
        "run_id": run_id,
        "dataset_paths": sorted(caches.keys()),
        "dataset_path": sorted(caches.keys())[0] if len(caches) == 1 else None,
        "profile_ids": request.profile_ids,
        "model_ids": request.profile_ids,
        "profiles": snapshots,
        "episodes": [public_episode_ref(caches[str(resolve_dataset_path(ref.dataset_path))], ref) for ref in request.episodes],
        "episode_indexes": [int(ref.episode_index) for ref in request.episodes],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summarize_results(results),
        "results": results,
    }
    BACKTEST_RUNS[run_id] = run
    save_backtest_run(run)
    return run


def submit_backtest_job(request: BacktestRunRequest, cache_loader: Any) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": None,
        "finished_at": None,
        "request": request.model_dump(),
        "summary": None,
        "run_id": None,
        "error": None,
    }
    with BACKTEST_LOCK:
        BACKTEST_JOBS[job_id] = job
    future = BACKTEST_EXECUTOR.submit(_run_backtest_job, job_id, request, cache_loader)
    with BACKTEST_LOCK:
        BACKTEST_JOBS[job_id]["future"] = future
    return public_backtest_job(job)


def list_backtest_jobs() -> list[dict[str, Any]]:
    with BACKTEST_LOCK:
        jobs = [public_backtest_job(job, include_result=False) for job in BACKTEST_JOBS.values()]
    return sorted(jobs, key=lambda item: item.get("created_at") or "", reverse=True)


def get_backtest_job(job_id: str) -> dict[str, Any]:
    with BACKTEST_LOCK:
        job = BACKTEST_JOBS.get(job_id)
        if not job:
            raise KeyError(f"backtest job not found: {job_id}")
        return public_backtest_job(job)


def backtest_worker_status() -> dict[str, Any]:
    with BACKTEST_LOCK:
        queued = sum(1 for job in BACKTEST_JOBS.values() if job.get("status") == "queued")
        running = sum(1 for job in BACKTEST_JOBS.values() if job.get("status") == "running")
    return {
        "enabled": True,
        "max_workers": 1,
        "queued": queued,
        "running": running,
        "linux_inference_only": True,
    }


def _run_backtest_job(job_id: str, request: BacktestRunRequest, cache_loader: Any) -> None:
    with BACKTEST_LOCK:
        job = BACKTEST_JOBS[job_id]
        job["status"] = "running"
        job["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        run = run_backtest(request, cache_loader)
        with BACKTEST_LOCK:
            job = BACKTEST_JOBS[job_id]
            job["status"] = "done"
            job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            job["run_id"] = run.get("run_id")
            job["summary"] = run.get("summary")
            job["result"] = run
        log_operation(
            "backtest_job_done",
            "success",
            target=job_id,
            details={"run_id": run.get("run_id"), "summary": run.get("summary", {})},
        )
    except Exception as exc:
        with BACKTEST_LOCK:
            job = BACKTEST_JOBS[job_id]
            job["status"] = "failed"
            job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            job["error"] = str(exc)
        log_operation("backtest_job_done", "failed", target=job_id, error=str(exc))


def public_backtest_job(job: dict[str, Any], include_result: bool = True) -> dict[str, Any]:
    return {
        key: value
        for key, value in job.items()
        if key != "future" and not isinstance(value, Future) and (include_result or key != "result")
    }


def public_episode_ref(cache: Any, ref: BacktestEpisodeRef) -> dict[str, Any]:
    episode = cache.episode_record(int(ref.episode_index))
    fps = float(cache.info.get("fps") or 0)
    length = int(episode.get("length", 0))
    return {
        "dataset_path": str(cache.root),
        "dataset_name": cache.root.name,
        "dataset_id": dataset_ref_id(cache.root),
        "episode_index": int(ref.episode_index),
        "length": length,
        "duration": round(length / fps, 6) if fps > 0 else None,
        "fps": fps if fps > 0 else None,
        "tasks": clean_sequence(episode.get("tasks", [])),
        "video_keys": cache.video_keys,
    }


def failed_results(profile: dict[str, Any], episode_refs: list[BacktestEpisodeRef], error: str) -> list[dict[str, Any]]:
    snapshot = profile_snapshot(profile) if "checkpoint_path" in profile else profile
    profile_id = str(profile.get("id"))
    return [
        {
            "profile_id": profile_id,
            "model_id": profile_id,
            "profile_name": profile.get("name", profile_id),
            "profile_snapshot": snapshot,
            "dataset_path": str(resolve_dataset_path(ref.dataset_path)),
            "dataset_name": Path(ref.dataset_path).name,
            "dataset_id": dataset_ref_id(resolve_dataset_path(ref.dataset_path)),
            "episode_index": int(ref.episode_index),
            "episode_key": episode_result_key(resolve_dataset_path(ref.dataset_path), int(ref.episode_index)),
            "status": "failed",
            "error": error,
        }
        for ref in episode_refs
    ]


def run_episode_backtest(
    cache: Any,
    adapter: BacktestAdapter,
    profile: dict[str, Any],
    episode_index: int,
    max_frames: int | None = None,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    episode = cache.episode_record(episode_index)
    source = episode_result_source(cache, episode, dataset_path)
    profile_id = str(profile["id"])
    profile_info = {
        "profile_id": profile_id,
        "model_id": profile_id,
        "profile_name": profile.get("name", profile_id),
        "profile_snapshot": profile_snapshot(profile),
    }
    frames = read_episode_frames(cache, episode)
    if max_frames is not None and max_frames > 0:
        frames = frames.head(max_frames)
    if "action" not in frames.columns:
        return {
            **source,
            **profile_info,
            "episode_index": episode_index,
            "status": "failed",
            "error": "episode has no action column",
        }

    ground_truth = np.array([flatten_action(value) for value in frames["action"]], dtype=np.float64)
    video_observations = (
        decode_video_observations(cache, episode, 0, len(frames))
        if adapter_needs_video_observations(adapter, cache.features)
        else {}
    )
    predictions = []
    adapter.prepare_backtest_context(cache)
    adapter.reset_episode()
    try:
        for frame_offset, (_, frame) in enumerate(frames.iterrows()):
            observation = build_observation(frame, cache.features, video_observations, frame_offset)
            predictions.append(flatten_action(adapter.predict(observation)))
    except Exception as exc:
        return {**source, **profile_info, "episode_index": episode_index, "status": "failed", "error": str(exc)}

    predicted = np.array(predictions, dtype=np.float64)
    if predicted.shape != ground_truth.shape:
        return {
            **source,
            **profile_info,
            "episode_index": episode_index,
            "status": "failed",
            "error": f"predicted action shape {predicted.shape} != ground truth {ground_truth.shape}",
        }
    metrics = action_metrics(ground_truth, predicted)
    return {
        **source,
        **profile_info,
        "episode_index": episode_index,
        "status": "done",
        "frames": int(len(frames)),
        "action_dim": int(ground_truth.shape[1]) if ground_truth.ndim == 2 else 1,
        "metrics": metrics,
        "series": action_series(ground_truth, predicted),
    }


def episode_result_source(cache: Any, episode: Any, dataset_path: str | None = None) -> dict[str, Any]:
    root = Path(dataset_path or cache.root).resolve()
    fps = float(cache.info.get("fps") or 0)
    length = int(episode.get("length", 0))
    episode_index = int(episode.get("episode_index", 0))
    return {
        "dataset_path": str(root),
        "dataset_name": root.name,
        "dataset_id": dataset_ref_id(root),
        "episode_key": episode_result_key(root, episode_index),
        "length": length,
        "duration": round(length / fps, 6) if fps > 0 else None,
        "fps": fps if fps > 0 else None,
        "tasks": clean_sequence(episode.get("tasks", [])),
        "video_keys": cache.video_keys,
    }


def clean_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def episode_result_key(dataset_path: Path, episode_index: int) -> str:
    return f"{dataset_ref_id(dataset_path)}:{int(episode_index)}"


def dataset_ref_id(path: Path) -> str:
    import hashlib

    return hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:12]


def build_observation(
    frame: pd.Series,
    features: dict[str, dict[str, Any]],
    video_observations: dict[str, np.ndarray] | None = None,
    frame_offset: int = 0,
) -> dict[str, Any]:
    observation = {}
    for key, feature in features.items():
        if key == "action":
            continue
        if feature.get("dtype") == "video":
            if video_observations and key in video_observations:
                frames = video_observations[key]
                if frame_offset >= len(frames):
                    raise ValueError(f"decoded video frames for {key} shorter than episode frames")
                observation[key] = frames[frame_offset]
            continue
        if key in frame:
            observation[key] = frame[key]
    if "task_index" in frame:
        observation["task_index"] = int(frame["task_index"])
    if "timestamp" in frame:
        observation["timestamp"] = float(frame["timestamp"])
    return observation


def decode_video_observations(cache: Any, episode: pd.Series, start_frame: int, frame_count: int) -> dict[str, np.ndarray]:
    if not getattr(cache, "video_keys", None) or frame_count <= 0:
        return {}
    executable = ffmpeg_executable()
    if not executable:
        raise RuntimeError("ffmpeg is required to build image observations for video datasets")
    decoded: dict[str, np.ndarray] = {}
    fps = float(cache.info.get("fps") or 0)
    if fps <= 0:
        raise RuntimeError("dataset fps must be positive to decode image observations")
    for video_key in cache.video_keys:
        video_path = cache.video_file_for_episode(episode, video_key)
        prefix = f"videos/{video_key}"
        from_timestamp = float(episode.get(f"{prefix}/from_timestamp", 0.0) or 0.0)
        seek_time = from_timestamp + (int(start_frame) / fps)
        stream = ffprobe_video_stream(video_path)
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"failed to probe video dimensions for {video_key}: {video_path}")
        raw = run_ffmpeg_rawvideo(
            executable=executable,
            video_path=video_path,
            seek_time=seek_time,
            frame_count=frame_count,
            width=width,
            height=height,
            video_key=video_key,
        )
        expected_bytes = frame_count * height * width * 3
        if len(raw) < expected_bytes:
            actual_frames = len(raw) // max(height * width * 3, 1)
            raise RuntimeError(
                f"decoded video frames for {video_key} shorter than requested: {actual_frames} < {frame_count}"
            )
        array = np.frombuffer(raw[:expected_bytes], dtype=np.uint8).reshape(frame_count, height, width, 3)
        decoded[video_key] = np.transpose(array.astype(np.float32) / 255.0, (0, 3, 1, 2))
    return decoded


def adapter_needs_video_observations(adapter: BacktestAdapter, features: dict[str, dict[str, Any]]) -> bool:
    has_video = any(feature.get("dtype") == "video" for feature in features.values())
    if not has_video:
        return False
    if getattr(adapter, "test_only", False):
        return False
    return True


def run_ffmpeg_rawvideo(
    executable: str,
    video_path: Path,
    seek_time: float,
    frame_count: int,
    width: int,
    height: int,
    video_key: str,
) -> bytes:
    import subprocess

    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, seek_time):.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        str(int(frame_count)),
        "-vf",
        f"scale={width}:{height}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, timeout=max(30, int(frame_count) * 2), check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"ffmpeg failed to decode image observations for {video_key}: {stderr}")
    return result.stdout


def flatten_action(value: Any) -> list[float]:
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in np.array(value, dtype=object).reshape(-1)]
    return [float(value)]


def action_metrics(ground_truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    error = predicted - ground_truth
    abs_error = np.abs(error)
    squared = error ** 2
    per_dim_mae = np.mean(abs_error, axis=0)
    per_dim_rmse = np.sqrt(np.mean(squared, axis=0))
    per_frame_max = np.max(abs_error, axis=1)
    worst_frame = int(np.argmax(per_frame_max)) if len(per_frame_max) else 0
    return {
        "mae": round(float(np.mean(abs_error)), 8),
        "rmse": round(float(np.sqrt(np.mean(squared))), 8),
        "max_error": round(float(np.max(abs_error)), 8) if abs_error.size else 0,
        "worst_frame": worst_frame,
        "per_dim_mae": [round(float(item), 8) for item in per_dim_mae],
        "per_dim_rmse": [round(float(item), 8) for item in per_dim_rmse],
    }


def action_series(ground_truth: np.ndarray, predicted: np.ndarray) -> list[dict[str, Any]]:
    error = predicted - ground_truth
    dimensions = ground_truth.shape[1] if ground_truth.ndim == 2 else 1
    return [
        {
            "dimension": index,
            "ground_truth": [round(float(item), 8) for item in ground_truth[:, index]],
            "predicted": [round(float(item), 8) for item in predicted[:, index]],
            "error": [round(float(item), 8) for item in error[:, index]],
        }
        for index in range(dimensions)
    ]


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    done = [item for item in results if item.get("status") == "done"]
    failed = [item for item in results if item.get("status") != "done"]
    if not done:
        return {"total": len(results), "done": 0, "failed": len(failed)}
    return {
        "total": len(results),
        "done": len(done),
        "failed": len(failed),
        "mean_mae": round(float(np.mean([item["metrics"]["mae"] for item in done])), 8),
        "mean_rmse": round(float(np.mean([item["metrics"]["rmse"] for item in done])), 8),
        "max_error": round(float(max(item["metrics"]["max_error"] for item in done)), 8),
    }
