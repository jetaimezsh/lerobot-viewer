from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from app.reward_labeling import RewardLabelApplyRequest, apply_reward_label_plan, validate_reward_label_plan
from test_schema_editing import Cache, write_minimal_dataset


class RewardCache(Cache):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.episodes = pd.concat(
            [pd.read_parquet(path) for path in sorted((root / "meta/episodes").glob("**/*.parquet"))],
            ignore_index=True,
        )


def column_values(path: Path, column: str) -> list[float]:
    return [float(value) for value in pq.read_table(path).column(column).combine_chunks().to_pylist()]


def test_reward_labeling_adds_scalar_feature_and_range_values(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "reward_output"
    write_minimal_dataset(source)
    monkeypatch.setattr("app.reward_labeling.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})

    request = RewardLabelApplyRequest.model_validate(
        {
            "path": str(source),
            "output_path": str(output),
            "feature": "reward",
            "default_value": 0.0,
            "labels": [
                {"episode_index": 0, "start_frame": 1, "end_frame": 2, "value": 1.0},
                {"episode_index": 1, "start_frame": 0, "end_frame": 2, "value": 0.5},
            ],
        }
    )

    result = apply_reward_label_plan(RewardCache(source), request, output)

    assert result["ok"] is True
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["features"]["reward"] == {"dtype": "float32", "shape": [1], "names": None}
    assert column_values(output / "data/chunk-000/file-000.parquet", "reward") == [0.0, 1.0]
    assert column_values(output / "data/chunk-000/file-001.parquet", "reward") == [0.5, 0.5]
    stats = json.loads((output / "meta/stats.json").read_text(encoding="utf-8"))
    assert stats["reward"]["count"] == [4]
    assert stats["reward"]["max"] == [1.0]
    episode_table = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet")
    assert "stats/reward/mean" in episode_table.column_names


def test_reward_labeling_validation_rejects_out_of_range_labels(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_minimal_dataset(source)
    request = RewardLabelApplyRequest.model_validate(
        {
            "path": str(source),
            "output_path": str(tmp_path / "output"),
            "feature": "reward",
            "labels": [{"episode_index": 0, "start_frame": 0, "end_frame": 20, "value": 1.0}],
        }
    )

    result = validate_reward_label_plan(RewardCache(source), request)

    assert result["valid"] is False
    assert any("exceeds length" in error for error in result["errors"])


def test_reward_labeling_supports_int_dtype_across_all_episodes(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "int_output"
    write_minimal_dataset(source)
    monkeypatch.setattr("app.reward_labeling.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})
    request = RewardLabelApplyRequest.model_validate(
        {
            "path": str(source),
            "output_path": str(output),
            "feature": "success",
            "dtype": "int64",
            "default_value": 0,
            "labels": [{"episode_index": 1, "start_frame": 1, "end_frame": 2, "value": 1}],
        }
    )

    result = apply_reward_label_plan(RewardCache(source), request, output)

    assert result["ok"] is True
    assert column_values(output / "data/chunk-000/file-000.parquet", "success") == [0.0, 0.0]
    assert column_values(output / "data/chunk-000/file-001.parquet", "success") == [0.0, 1.0]
    data_type = pq.read_table(output / "data/chunk-000/file-001.parquet").schema.field("success").type
    assert str(data_type) == "int64"
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["features"]["success"]["dtype"] == "int64"
