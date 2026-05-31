from __future__ import annotations

import importlib
import json
import math
import platform
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.backtest_store import save_backtest_run
from app.editing import read_episode_frames, resolve_dataset_path
from app.operation_log import log_operation


class ModelRegisterRequest(BaseModel):
    name: str | None = None
    checkpoint_path: str
    adapter_type: str = "lerobot_official"
    device: str = "cuda"
    script_path: str | None = None


class ModelLoadRequest(BaseModel):
    model_id: str


class ModelDeleteRequest(BaseModel):
    model_id: str


class BacktestEpisodeRef(BaseModel):
    dataset_path: str
    episode_index: int


class BacktestRunRequest(BaseModel):
    dataset_path: str | None = None
    model_ids: list[str] = Field(default_factory=list)
    episode_indexes: list[int] = Field(default_factory=list)
    episodes: list[BacktestEpisodeRef] = Field(default_factory=list)
    max_frames: int | None = None


class BacktestAdapter(Protocol):
    def inspect(self) -> dict[str, Any]:
        ...

    def load(self) -> None:
        ...

    def reset_episode(self) -> None:
        ...

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


MODEL_REGISTRY: dict[str, dict[str, Any]] = {}
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
        "registered_models": list_models(),
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


def register_model(request: ModelRegisterRequest) -> dict[str, Any]:
    model_id = uuid.uuid4().hex[:12]
    record = {
        "id": model_id,
        "name": request.name or default_model_name(request),
        "checkpoint_path": str(resolve_dataset_path(request.checkpoint_path)),
        "adapter_type": request.adapter_type,
        "device": request.device,
        "script_path": str(resolve_dataset_path(request.script_path)) if request.script_path else None,
        "status": "registered",
        "loaded": False,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    record["inspection"] = inspect_model_record(record)
    if not record["inspection"]["valid"]:
        record["status"] = "invalid"
    MODEL_REGISTRY[model_id] = record
    return public_model_record(record)


def default_model_name(request: ModelRegisterRequest) -> str:
    path = Path(request.checkpoint_path)
    return path.name or path.parent.name or "model"


def list_models() -> list[dict[str, Any]]:
    return [public_model_record(record) for record in MODEL_REGISTRY.values()]


def delete_model(model_id: str) -> dict[str, Any]:
    adapter = LOADED_ADAPTERS.pop(model_id, None)
    if adapter:
        adapter.close()
    record = MODEL_REGISTRY.pop(model_id, None)
    if not record:
        raise ValueError(f"model not found: {model_id}")
    return {"ok": True, "deleted": model_id}


def inspect_model(model_id: str) -> dict[str, Any]:
    record = require_model(model_id)
    record["inspection"] = inspect_model_record(record)
    if not record["inspection"]["valid"]:
        record["status"] = "invalid"
    elif record["status"] == "invalid":
        record["status"] = "registered"
    return public_model_record(record)


def inspect_model_record(record: dict[str, Any]) -> dict[str, Any]:
    if record["adapter_type"] == "custom_script":
        return inspect_custom_script_record(record)
    return inspect_lerobot_record(record)


def inspect_lerobot_record(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["checkpoint_path"])
    errors: list[str] = []
    warnings: list[str] = []
    files: list[dict[str, Any]] = []
    config: dict[str, Any] | None = None
    if not path.exists():
        errors.append(f"checkpoint path does not exist: {path}")
    elif path.is_dir():
        config_path = path / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"failed to read config.json: {exc}")
        else:
            warnings.append("checkpoint directory has no config.json")
        for pattern in ["*.safetensors", "*.bin", "*.pt", "*.pth"]:
            for item in path.glob(pattern):
                files.append({"name": item.name, "size_bytes": item.stat().st_size})
    elif path.is_file():
        files.append({"name": path.name, "size_bytes": path.stat().st_size})
        warnings.append("single checkpoint file found; official LeRobot checkpoints are usually directories")

    total_size = sum(int(item["size_bytes"]) for item in files)
    policy_type = infer_policy_type(config, path)
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "path_exists": path.exists(),
        "path_type": "dir" if path.is_dir() else "file" if path.is_file() else "missing",
        "policy_type": policy_type,
        "config_keys": sorted(config.keys()) if isinstance(config, dict) else [],
        "files": files[:20],
        "file_count": len(files),
        "size_bytes": total_size,
        "size_mb": round(total_size / (1024 * 1024), 3),
        "loader": "lerobot_official",
    }


def inspect_custom_script_record(record: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(record["checkpoint_path"])
    script = Path(record["script_path"] or "")
    errors = []
    if not checkpoint.exists():
        errors.append(f"checkpoint path does not exist: {checkpoint}")
    if not script.exists() or not script.is_file():
        errors.append(f"custom script does not exist: {script}")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": ["custom script execution is reserved for a later v3 iteration"],
        "path_exists": checkpoint.exists(),
        "path_type": "dir" if checkpoint.is_dir() else "file" if checkpoint.is_file() else "missing",
        "policy_type": "custom_script",
        "script_path": str(script) if record["script_path"] else None,
        "files": [],
        "file_count": 0,
        "size_bytes": directory_size(checkpoint) if checkpoint.exists() else 0,
        "size_mb": round(directory_size(checkpoint) / (1024 * 1024), 3) if checkpoint.exists() else 0,
        "loader": "custom_script",
    }


def infer_policy_type(config: dict[str, Any] | None, path: Path) -> str | None:
    if isinstance(config, dict):
        for key in ["type", "policy_type", "name", "architecture"]:
            if config.get(key):
                return str(config[key])
        policy = config.get("policy")
        if isinstance(policy, dict):
            for key in ["type", "policy_type", "name"]:
                if policy.get(key):
                    return str(policy[key])
    return path.name if path.exists() else None


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def load_model(model_id: str) -> dict[str, Any]:
    record = require_model(model_id)
    if record["adapter_type"] == "custom_script":
        raise RuntimeError("custom script adapter is reserved for a later v3 iteration")
    adapter = LeRobotOfficialAdapter(record)
    adapter.load()
    LOADED_ADAPTERS[model_id] = adapter
    record["status"] = "loaded"
    record["loaded"] = True
    record["inspection"] = {**record.get("inspection", {}), **adapter.inspect()}
    return public_model_record(record)


def unload_model(model_id: str) -> dict[str, Any]:
    record = require_model(model_id)
    adapter = LOADED_ADAPTERS.pop(model_id, None)
    if adapter:
        adapter.close()
    record["status"] = "registered"
    record["loaded"] = False
    return public_model_record(record)


def run_backtest(request: BacktestRunRequest, cache_loader: Any) -> dict[str, Any]:
    if not request.model_ids:
        raise ValueError("at least one model is required")
    episode_refs = normalize_backtest_episode_refs(request)
    if not episode_refs:
        raise ValueError("at least one episode is required")

    caches: dict[str, Any] = {}
    for ref in episode_refs:
        dataset_path = str(resolve_dataset_path(ref.dataset_path))
        if dataset_path not in caches:
            caches[dataset_path] = cache_loader(dataset_path)

    run_id = uuid.uuid4().hex[:12]
    results = []
    for model_id in request.model_ids:
        record = require_model(model_id)
        adapter = LOADED_ADAPTERS.get(model_id)
        if adapter is None:
            if record["adapter_type"] == "custom_script":
                results.extend(failed_results(model_id, episode_refs, "custom script adapter is not enabled yet"))
                continue
            try:
                adapter = LeRobotOfficialAdapter(record)
                adapter.load()
                LOADED_ADAPTERS[model_id] = adapter
                record["status"] = "loaded"
                record["loaded"] = True
            except Exception as exc:
                results.extend(failed_results(model_id, episode_refs, str(exc)))
                continue
        for ref in episode_refs:
            dataset_path = str(resolve_dataset_path(ref.dataset_path))
            results.append(
                run_episode_backtest(
                    caches[dataset_path],
                    adapter,
                    model_id,
                    int(ref.episode_index),
                    request.max_frames,
                    dataset_path=dataset_path,
                )
            )

    run = {
        "run_id": run_id,
        "dataset_paths": sorted(caches.keys()),
        "dataset_path": sorted(caches.keys())[0] if len(caches) == 1 else None,
        "model_ids": request.model_ids,
        "episodes": [public_episode_ref(caches[str(resolve_dataset_path(ref.dataset_path))], ref) for ref in episode_refs],
        "episode_indexes": [int(ref.episode_index) for ref in episode_refs],
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


def normalize_backtest_episode_refs(request: BacktestRunRequest) -> list[BacktestEpisodeRef]:
    if request.episodes:
        return request.episodes
    if request.dataset_path and request.episode_indexes:
        return [
            BacktestEpisodeRef(dataset_path=request.dataset_path, episode_index=int(index))
            for index in request.episode_indexes
        ]
    return []


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


def failed_results(model_id: str, episode_refs: list[BacktestEpisodeRef], error: str) -> list[dict[str, Any]]:
    return [
        {
            "model_id": model_id,
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
    model_id: str,
    episode_index: int,
    max_frames: int | None = None,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    episode = cache.episode_record(episode_index)
    source = episode_result_source(cache, episode, dataset_path)
    frames = read_episode_frames(cache, episode)
    if max_frames is not None and max_frames > 0:
        frames = frames.head(max_frames)
    if "action" not in frames.columns:
        return {
            **source,
            "model_id": model_id,
            "episode_index": episode_index,
            "status": "failed",
            "error": "episode has no action column",
        }

    ground_truth = np.array([flatten_action(value) for value in frames["action"]], dtype=np.float64)
    predictions = []
    adapter.reset_episode()
    try:
        for _, frame in frames.iterrows():
            observation = build_observation(frame, cache.features)
            predictions.append(flatten_action(adapter.predict(observation)))
    except Exception as exc:
        return {**source, "model_id": model_id, "episode_index": episode_index, "status": "failed", "error": str(exc)}

    predicted = np.array(predictions, dtype=np.float64)
    if predicted.shape != ground_truth.shape:
        return {
            **source,
            "model_id": model_id,
            "episode_index": episode_index,
            "status": "failed",
            "error": f"predicted action shape {predicted.shape} != ground truth {ground_truth.shape}",
        }
    metrics = action_metrics(ground_truth, predicted)
    return {
        **source,
        "model_id": model_id,
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


def build_observation(frame: pd.Series, features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    observation = {}
    for key, feature in features.items():
        if key == "action" or feature.get("dtype") == "video" or key not in frame:
            continue
        observation[key] = frame[key]
    if "task_index" in frame:
        observation["task_index"] = int(frame["task_index"])
    if "timestamp" in frame:
        observation["timestamp"] = float(frame["timestamp"])
    return observation


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


def require_model(model_id: str) -> dict[str, Any]:
    record = MODEL_REGISTRY.get(model_id)
    if not record:
        raise ValueError(f"model not found: {model_id}")
    return record


def public_model_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "name": record["name"],
        "checkpoint_path": record["checkpoint_path"],
        "adapter_type": record["adapter_type"],
        "device": record["device"],
        "script_path": record.get("script_path"),
        "status": record["status"],
        "loaded": bool(record.get("loaded")),
        "created_at": record.get("created_at"),
        "inspection": record.get("inspection", {}),
    }


class LeRobotOfficialAdapter:
    def __init__(self, record: dict[str, Any]):
        self.record = record
        self.policy = None
        self.torch = None

    def inspect(self) -> dict[str, Any]:
        parameter_count = None
        if self.policy is not None and hasattr(self.policy, "parameters"):
            try:
                parameter_count = int(sum(parameter.numel() for parameter in self.policy.parameters()))
            except Exception:
                parameter_count = None
        return {"parameter_count": parameter_count}

    def load(self) -> None:
        if platform.system().lower() != "linux":
            raise RuntimeError("model inference backtesting is Linux-only in v3")
        self.torch = importlib.import_module("torch")
        path = self.record["checkpoint_path"]
        policy = self._load_policy(path)
        if hasattr(policy, "to"):
            policy = policy.to(self.record["device"])
        if hasattr(policy, "eval"):
            policy.eval()
        self.policy = policy

    def _load_policy(self, path: str) -> Any:
        errors = []
        try:
            policies = importlib.import_module("lerobot.policies")
            pretrained_cls = getattr(policies, "PreTrainedPolicy", None)
            if pretrained_cls is not None and hasattr(pretrained_cls, "from_pretrained"):
                return pretrained_cls.from_pretrained(path)
        except Exception as exc:
            errors.append(f"PreTrainedPolicy.from_pretrained: {exc}")

        try:
            factory = importlib.import_module("lerobot.policies.factory")
            policy_type = self.record.get("inspection", {}).get("policy_type")
            if policy_type and hasattr(factory, "get_policy_class"):
                policy_cls = factory.get_policy_class(policy_type)
                if hasattr(policy_cls, "from_pretrained"):
                    return policy_cls.from_pretrained(path)
        except Exception as exc:
            errors.append(f"factory.get_policy_class: {exc}")

        raise RuntimeError("failed to load LeRobot policy: " + " | ".join(errors))

    def reset_episode(self) -> None:
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        if self.policy is None or self.torch is None:
            raise RuntimeError("model is not loaded")
        batch = {key: self._to_tensor(value) for key, value in observation.items()}
        with self.torch.inference_mode():
            action = self.policy.select_action(batch)
        if hasattr(action, "detach"):
            action = action.detach().to("cpu").numpy()
        array = np.asarray(action, dtype=np.float64)
        if array.ndim == 2:
            array = array[0]
        return array.reshape(-1)

    def _to_tensor(self, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        array = np.asarray(value)
        if array.dtype.kind in {"U", "S", "O"}:
            return value
        tensor = self.torch.as_tensor(array)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        tensor = tensor.unsqueeze(0).to(self.record["device"])
        return tensor

    def close(self) -> None:
        self.policy = None


class MockBacktestAdapter:
    def __init__(self, action: list[float]):
        self.action = np.asarray(action, dtype=np.float64)

    def inspect(self) -> dict[str, Any]:
        return {"parameter_count": 0}

    def load(self) -> None:
        return None

    def reset_episode(self) -> None:
        return None

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        return self.action

    def close(self) -> None:
        return None
