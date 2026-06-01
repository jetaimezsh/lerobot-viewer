from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BacktestAdapter(ABC):
    adapter_name = "base"
    label = "Base"
    visible_in_ui = True
    test_only = False

    @classmethod
    def default_runtime_params(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> list[str]:
        return []

    @classmethod
    def inspect_profile(cls, profile: dict[str, Any]) -> dict[str, Any]:
        return {"valid": True, "errors": [], "warnings": []}

    def __init__(self) -> None:
        self.profile: dict[str, Any] | None = None

    @abstractmethod
    def load(self, profile: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        ...

    @abstractmethod
    def reset_episode(self) -> None:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    def runtime_info(self) -> dict[str, Any]:
        return {}


ADAPTER_REGISTRY: dict[str, type[BacktestAdapter]] = {}


def register_adapter(name: str, adapter_cls: type[BacktestAdapter]) -> None:
    if name in ADAPTER_REGISTRY:
        raise ValueError(f"adapter already registered: {name}")
    ADAPTER_REGISTRY[name] = adapter_cls


def get_adapter(name: str) -> type[BacktestAdapter]:
    adapter_cls = ADAPTER_REGISTRY.get(name)
    if adapter_cls is None:
        raise ValueError(f"unknown adapter: {name}")
    return adapter_cls


def list_adapters(include_test: bool = False) -> list[dict[str, Any]]:
    adapters = []
    for name, adapter_cls in sorted(ADAPTER_REGISTRY.items()):
        if adapter_cls.test_only and not include_test:
            continue
        adapters.append(
            {
                "id": name,
                "label": adapter_cls.label,
                "visible_in_ui": adapter_cls.visible_in_ui,
                "test_only": adapter_cls.test_only,
                "default_runtime_params": adapter_cls.default_runtime_params(),
            }
        )
    return adapters


def load_builtin_adapters() -> None:
    from app.adapters import lerobot_official  # noqa: F401
    from app.adapters import mock  # noqa: F401


load_builtin_adapters()
