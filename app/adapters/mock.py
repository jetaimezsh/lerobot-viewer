from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from app.adapters import BacktestAdapter, register_adapter


class MockAdapter(BacktestAdapter):
    adapter_name = "mock"
    label = "Mock adapter"
    visible_in_ui = False
    test_only = True

    @classmethod
    def default_runtime_params(cls) -> dict[str, Any]:
        return {"action": [0.0, 0.0]}

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> list[str]:
        action = params.get("action", [0.0, 0.0])
        if not isinstance(action, list) or not action:
            return ["mock action must be a non-empty numeric list"]
        try:
            [float(item) for item in action]
        except Exception:
            return ["mock action must contain only numbers"]
        return []

    @classmethod
    def inspect_profile(cls, profile: dict[str, Any]) -> dict[str, Any]:
        errors = cls.validate_params(profile.get("runtime_params") or {})
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": ["mock adapter is intended for tests and offline UI verification"],
            "policy_type": "mock",
            "file_count": 0,
            "size_mb": 0,
            "loader": cls.adapter_name,
            "checkpoint_config": {},
        }

    def load(self, profile: dict[str, Any]) -> None:
        self.profile = profile

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        params = (self.profile or {}).get("runtime_params") or {}
        sleep_ms = float(params.get("sleep_ms", 0) or 0)
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000)
        env_key = params.get("action_from_env")
        if env_key:
            raw = os.environ.get(str(env_key), "")
            if raw:
                return np.asarray([float(item.strip()) for item in raw.split(",") if item.strip()], dtype=np.float64)
        return np.asarray(params.get("action", [0.0, 0.0]), dtype=np.float64)

    def reset_episode(self) -> None:
        return None

    def unload(self) -> None:
        self.profile = None

    def runtime_info(self) -> dict[str, Any]:
        return {"parameter_count": 0}


register_adapter(MockAdapter.adapter_name, MockAdapter)
