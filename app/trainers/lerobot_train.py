from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from app.trainers import TrainingFramework, register_trainer


class LeRobotTrainFramework(TrainingFramework):
    framework_name = "lerobot_train"
    label = "LeRobot Train CLI"

    @classmethod
    def default_hyperparams(cls) -> dict[str, Any]:
        return {
            "policy_type": "act",
            "batch_size": 64,
            "epochs": 1000,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "lr_scheduler": "cosine",
            "num_workers": 4,
        }

    @classmethod
    def inspect_recipe(cls, recipe: dict[str, Any]) -> dict[str, Any]:
        errors = cls.validate_recipe(recipe)
        warnings: list[str] = []
        dataset_path = Path(str(recipe.get("dataset_path") or ""))
        output_dir = Path(str(recipe.get("output_dir") or ""))
        command = shutil.which("lerobot-train")
        if dataset_path and not dataset_path.exists():
            errors.append(f"dataset_path does not exist: {dataset_path}")
        if output_dir:
            parent = output_dir.parent
            if parent and not parent.exists():
                warnings.append(f"output parent does not exist yet: {parent}")
        if command is None:
            errors.append("lerobot-train command not found in PATH")
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "command": command,
        }

    def build_command(self, recipe: dict[str, Any]) -> list[str]:
        hp = recipe.get("hyperparams") or {}
        command = ["lerobot-train"]
        command.append(f"policy.type={hp.get('policy_type', 'act')}")
        command.append(f"dataset.root={recipe['dataset_path']}")
        command.append(f"output_dir={recipe['output_dir']}")
        command.append(f"device={recipe.get('device', 'cuda')}")
        for key, value in hp.items():
            if key == "policy_type":
                continue
            command.append(f"{key}={format_cli_value(value)}")
        episode_filter = recipe.get("episode_filter")
        if episode_filter:
            command.append(f"episode_filter={format_cli_value(episode_filter)}")
        return command

    def parse_progress(self, line: str) -> dict[str, Any] | None:
        patterns = [
            r"Epoch\s+(\d+)/(\d+).*?loss[:=]\s*([\deE.+-]+).*?lr[:=]\s*([\deE.+-]+)",
            r"epoch[:=]\s*(\d+).*?total_epochs[:=]\s*(\d+).*?loss[:=]\s*([\deE.+-]+).*?lr[:=]\s*([\deE.+-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if not match:
                continue
            return {
                "epoch": int(match.group(1)),
                "total_epochs": int(match.group(2)),
                "loss": float(match.group(3)),
                "lr": float(match.group(4)),
            }
        return None


def format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_cli_value(item) for item in value) + "]"
    return str(value)


register_trainer("lerobot_train", LeRobotTrainFramework)
