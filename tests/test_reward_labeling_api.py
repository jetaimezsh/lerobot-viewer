from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from app.main import app
from test_schema_editing import write_minimal_dataset


def test_reward_label_apply_api_writes_output(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "reward_output"
    write_minimal_dataset(source)
    monkeypatch.setattr("app.reward_labeling.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})
    client = TestClient(app)

    response = client.post(
        "/api/reward-label/apply",
        json={
            "path": str(source),
            "output_path": str(output),
            "feature": "reward",
            "default_value": 0.0,
            "labels": [{"episode_index": 0, "start_frame": 0, "end_frame": 1, "value": 1.0}],
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert "reward" in info["features"]
    reward = pq.read_table(output / "data/chunk-000/file-000.parquet").column("reward").combine_chunks().to_pylist()
    assert reward == [1.0, 0.0]
