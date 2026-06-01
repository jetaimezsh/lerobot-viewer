from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from app.trainers import TrainingFramework, register_trainer


class MockTrainingFramework(TrainingFramework):
    framework_name = "mock"
    label = "Mock Trainer"
    visible_in_ui = False
    test_only = True

    @classmethod
    def default_hyperparams(cls) -> dict[str, Any]:
        return {
            "policy_type": "act",
            "epochs": 3,
            "sleep_ms": 10,
            "fail": False,
        }

    @classmethod
    def inspect_recipe(cls, recipe: dict[str, Any]) -> dict[str, Any]:
        errors = cls.validate_recipe(recipe)
        return {"valid": not errors, "errors": errors, "warnings": [], "command": sys.executable}

    def build_command(self, recipe: dict[str, Any]) -> list[str]:
        hp = recipe.get("hyperparams") or {}
        code = MOCK_TRAIN_CODE
        return [
            sys.executable,
            "-c",
            code,
            str(recipe.get("output_dir") or ""),
            str(int(hp.get("epochs", 3))),
            str(int(hp.get("sleep_ms", 10))),
            "1" if hp.get("fail") else "0",
            str(hp.get("policy_type", "act")),
        ]

    def parse_progress(self, line: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None
        if payload.get("type") != "progress":
            return None
        return {
            "epoch": payload.get("epoch"),
            "total_epochs": payload.get("total_epochs"),
            "loss": payload.get("loss"),
            "lr": payload.get("lr"),
        }


MOCK_TRAIN_CODE = r'''
import json
import sys
import time
from pathlib import Path

output_dir = Path(sys.argv[1])
epochs = int(sys.argv[2])
sleep_ms = int(sys.argv[3])
should_fail = sys.argv[4] == "1"
policy_type = sys.argv[5]
output_dir.mkdir(parents=True, exist_ok=True)
for epoch in range(1, epochs + 1):
    time.sleep(max(sleep_ms, 0) / 1000)
    loss = round(1.0 / epoch, 6)
    print(json.dumps({"type": "progress", "epoch": epoch, "total_epochs": epochs, "loss": loss, "lr": 0.0001}), flush=True)
if should_fail:
    print("mock training failed intentionally", flush=True)
    sys.exit(7)
(output_dir / "config.json").write_text(json.dumps({"policy_type": policy_type, "mock": True}), encoding="utf-8")
(output_dir / "model.safetensors").write_text("mock weights", encoding="utf-8")
print(json.dumps({"type": "done", "output_dir": str(output_dir)}), flush=True)
'''


register_trainer("mock", MockTrainingFramework)
