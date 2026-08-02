from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import DATASETS, app
from test_schema_editing import write_minimal_dataset


def test_schema_dry_run_api_returns_normalized_plan(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_minimal_dataset(source)
    client = TestClient(app)

    response = client.post(
        "/api/schema-edit/dry-run",
        json={
            "path": str(source),
            "operations": [
                {"type": "drop_feature_keys", "feature": "observation.state", "keys": ["gripper"]},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True
    assert payload["normalized_plan"]["drop_feature_keys"]["observation.state"]["keep_indices"] == [0, 1, 2]
    assert payload["affected"]["data_shards"] == 2


def test_schema_apply_api_writes_output(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_minimal_dataset(source)
    monkeypatch.setattr("app.schema_editing.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})
    client = TestClient(app)

    response = client.post(
        "/api/schema-edit/apply",
        json={
            "path": str(source),
            "output_path": str(output),
            "operations": [
                {"type": "rename_feature", "source": "observation.action_joints", "target": "action"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert "action" in info["features"]
    assert "observation.action_joints" not in info["features"]


def test_dataset_schema_endpoint_uses_loaded_dataset_key(tmp_path: Path) -> None:
    DATASETS.clear()
    source = tmp_path / "source"
    write_minimal_dataset(source)
    client = TestClient(app)

    opened = client.post("/api/datasets/open", json={"path": str(source)}).json()
    response = client.get(f"/api/datasets/{opened['id']}/schema")

    assert response.status_code == 200
    names = [feature["name"] for feature in response.json()["features"]]
    assert "observation.state" in names


def test_suggest_output_directory_creates_child_name_and_avoids_collision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    parent = tmp_path / "outputs"
    write_minimal_dataset(source)
    parent.mkdir()
    (parent / "source_schema_edit").mkdir()
    client = TestClient(app)

    response = client.post(
        "/api/path/suggest-output-directory",
        json={"parent": str(parent), "source_path": str(source), "suffix": "schema_edit"},
    )

    assert response.status_code == 200
    assert response.json()["path"] == str(parent / "source_schema_edit_1")
