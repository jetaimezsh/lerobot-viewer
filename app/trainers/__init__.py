from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TrainingFramework(ABC):
    framework_name = "base"
    label = "Base Trainer"
    visible_in_ui = True
    test_only = False

    @classmethod
    def default_hyperparams(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def validate_recipe(cls, recipe: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        if not recipe.get("dataset_path"):
            errors.append("dataset_path is required")
        if not recipe.get("output_dir"):
            errors.append("output_dir is required")
        if not isinstance(recipe.get("hyperparams", {}), dict):
            errors.append("hyperparams must be an object")
        return errors

    @classmethod
    def inspect_recipe(cls, recipe: dict[str, Any]) -> dict[str, Any]:
        errors = cls.validate_recipe(recipe)
        return {"valid": not errors, "errors": errors, "warnings": []}

    @abstractmethod
    def build_command(self, recipe: dict[str, Any]) -> list[str]:
        ...

    def parse_progress(self, line: str) -> dict[str, Any] | None:
        return None

    def estimate_output_dir(self, recipe: dict[str, Any]) -> str:
        return str(recipe.get("output_dir") or "")


TRAINER_REGISTRY: dict[str, type[TrainingFramework]] = {}


def register_trainer(name: str, trainer_cls: type[TrainingFramework]) -> None:
    if name in TRAINER_REGISTRY:
        raise ValueError(f"trainer already registered: {name}")
    TRAINER_REGISTRY[name] = trainer_cls


def get_trainer(name: str) -> type[TrainingFramework]:
    trainer_cls = TRAINER_REGISTRY.get(name)
    if trainer_cls is None:
        raise ValueError(f"unknown trainer: {name}")
    return trainer_cls


def list_trainers(include_test: bool = False) -> list[dict[str, Any]]:
    result = []
    for name, trainer_cls in sorted(TRAINER_REGISTRY.items()):
        if trainer_cls.test_only and not include_test:
            continue
        result.append(
            {
                "id": name,
                "label": trainer_cls.label,
                "visible_in_ui": trainer_cls.visible_in_ui,
                "test_only": trainer_cls.test_only,
                "default_hyperparams": trainer_cls.default_hyperparams(),
            }
        )
    return result


def load_builtin_trainers() -> None:
    from app.trainers import lerobot_train  # noqa: F401
    from app.trainers import mock  # noqa: F401


load_builtin_trainers()
