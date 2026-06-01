from __future__ import annotations

import importlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from app.adapters import BacktestAdapter, register_adapter


class LeRobotOfficialAdapter(BacktestAdapter):
    adapter_name = "lerobot_official"
    label = "LeRobot official checkpoint"

    def __init__(self) -> None:
        super().__init__()
        self.policy = None
        self.torch = None

    @classmethod
    def default_runtime_params(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> list[str]:
        errors = []
        if not isinstance(params, dict):
            errors.append("runtime_params must be an object")
        return errors

    @classmethod
    def inspect_profile(cls, profile: dict[str, Any]) -> dict[str, Any]:
        path = Path(profile.get("checkpoint_path") or "")
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

        params_errors = cls.validate_params(profile.get("runtime_params") or {})
        errors.extend(params_errors)
        total_size = sum(int(item["size_bytes"]) for item in files)
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "path_exists": path.exists(),
            "path_type": "dir" if path.is_dir() else "file" if path.is_file() else "missing",
            "policy_type": infer_policy_type(config, path),
            "checkpoint_config": extract_checkpoint_config(config),
            "config_keys": sorted(config.keys()) if isinstance(config, dict) else [],
            "files": files[:20],
            "file_count": len(files),
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 3),
            "loader": cls.adapter_name,
        }

    def load(self, profile: dict[str, Any]) -> None:
        if platform.system().lower() != "linux":
            raise RuntimeError("LeRobot official model inference is Linux-only")
        self.torch = importlib.import_module("torch")
        self.profile = profile
        policy = self._load_policy(profile["checkpoint_path"])
        if hasattr(policy, "to"):
            policy = policy.to(profile.get("device") or "cuda")
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
            policy_type = (self.profile or {}).get("runtime_params", {}).get("policy_type")
            if not policy_type:
                policy_type = (self.profile or {}).get("inspection", {}).get("policy_type")
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
        if self.policy is None or self.torch is None or self.profile is None:
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
        tensor = tensor.unsqueeze(0).to((self.profile or {}).get("device") or "cuda")
        return tensor

    def unload(self) -> None:
        self.policy = None
        self.torch = None
        self.profile = None

    def runtime_info(self) -> dict[str, Any]:
        parameter_count = None
        if self.policy is not None and hasattr(self.policy, "parameters"):
            try:
                parameter_count = int(sum(parameter.numel() for parameter in self.policy.parameters()))
            except Exception:
                parameter_count = None
        return {"parameter_count": parameter_count}


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


def extract_checkpoint_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    readonly = {}
    for key in ["type", "policy_type", "name", "architecture", "n_obs_steps", "n_action_steps", "chunk_size", "horizon"]:
        if key in config:
            readonly[key] = config[key]
    policy = config.get("policy")
    if isinstance(policy, dict):
        for key in ["type", "policy_type", "name", "n_obs_steps", "n_action_steps", "chunk_size", "horizon"]:
            if key in policy:
                readonly[f"policy.{key}"] = policy[key]
    return readonly


register_adapter(LeRobotOfficialAdapter.adapter_name, LeRobotOfficialAdapter)
