from __future__ import annotations

from copy import deepcopy
from typing import Any


BUILTIN_TEMPLATES: dict[str, dict[str, Any]] = {
    "act": {
        "id": "act",
        "label": "ACT (Action Chunking Transformer)",
        "adapter": "lerobot_official",
        "runtime_params": {
            "policy_type": "act",
            "temporal_agg": True,
        },
        "description": "LeRobot official ACT checkpoint. Structural checkpoint parameters are read-only.",
    },
    "diffusion": {
        "id": "diffusion",
        "label": "Diffusion Policy",
        "adapter": "lerobot_official",
        "runtime_params": {
            "policy_type": "diffusion",
            "num_inference_steps": 100,
        },
        "description": "LeRobot official diffusion policy checkpoint.",
    },
    "vq_bet": {
        "id": "vq_bet",
        "label": "VQ-BeT",
        "adapter": "lerobot_official",
        "runtime_params": {
            "policy_type": "vq_bet",
        },
        "description": "LeRobot official VQ-BeT checkpoint.",
    },
    "blank": {
        "id": "blank",
        "label": "Blank profile",
        "adapter": "lerobot_official",
        "runtime_params": {},
        "description": "Start from an empty runtime parameter set.",
    },
}


def builtin_templates() -> list[dict[str, Any]]:
    return [deepcopy(template) for template in BUILTIN_TEMPLATES.values()]


def template_by_id(template_id: str | None) -> dict[str, Any]:
    if not template_id:
        template_id = "blank"
    template = BUILTIN_TEMPLATES.get(template_id)
    if not template:
        raise ValueError(f"unknown template: {template_id}")
    return deepcopy(template)
