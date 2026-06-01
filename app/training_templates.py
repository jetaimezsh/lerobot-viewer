from __future__ import annotations

from copy import deepcopy
from typing import Any


BUILTIN_TRAINING_TEMPLATES: dict[str, dict[str, Any]] = {
    "act_train": {
        "id": "act_train",
        "label": "ACT 训练",
        "framework": "lerobot_train",
        "hyperparams": {
            "policy_type": "act",
            "batch_size": 64,
            "epochs": 1000,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "lr_scheduler": "cosine",
            "num_workers": 4,
            "temporal_agg": True,
        },
        "description": "LeRobot 官方 ACT 训练模板。",
    },
    "diffusion_train": {
        "id": "diffusion_train",
        "label": "Diffusion Policy 训练",
        "framework": "lerobot_train",
        "hyperparams": {
            "policy_type": "diffusion",
            "batch_size": 64,
            "epochs": 1000,
            "learning_rate": 0.0001,
            "num_inference_steps": 100,
            "num_workers": 4,
        },
        "description": "LeRobot 官方 Diffusion Policy 训练模板。",
    },
    "vq_bet_train": {
        "id": "vq_bet_train",
        "label": "VQ-BeT 训练",
        "framework": "lerobot_train",
        "hyperparams": {
            "policy_type": "vq_bet",
            "batch_size": 64,
            "epochs": 500,
            "learning_rate": 0.0001,
            "num_workers": 4,
        },
        "description": "LeRobot 官方 VQ-BeT 训练模板。",
    },
    "blank_train": {
        "id": "blank_train",
        "label": "空白训练模板",
        "framework": "lerobot_train",
        "hyperparams": {},
        "description": "从空白训练参数开始。",
    },
}


def builtin_training_templates() -> list[dict[str, Any]]:
    return [deepcopy(template) for template in BUILTIN_TRAINING_TEMPLATES.values()]


def training_template_by_id(template_id: str | None) -> dict[str, Any]:
    if not template_id:
        template_id = "blank_train"
    template = BUILTIN_TRAINING_TEMPLATES.get(template_id)
    if not template:
        raise ValueError(f"unknown training template: {template_id}")
    return deepcopy(template)
