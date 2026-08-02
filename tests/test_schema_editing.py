from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from app.schema_editing import (
    DropFeatureKeysOperation,
    DropFeatureOperation,
    RenameFeatureKeysOperation,
    RenameFeatureOperation,
    ReorderFeatureKeysOperation,
    apply_schema_edit_plan,
    validate_schema_edit_plan,
)


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        self.features = self.info["features"]
        self.stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))


def fixed_float(rows: list[list[float]]) -> pa.FixedSizeListArray:
    width = len(rows[0])
    flat = [value for row in rows for value in row]
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), list_size=width)


def fixed_int(rows: list[list[int]]) -> pa.FixedSizeListArray:
    width = len(rows[0])
    flat = [value for row in rows for value in row]
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.int64()), list_size=width)


def write_minimal_dataset(root: Path, include_action: bool = False, bad_state_width: bool = False) -> None:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos/observation.images.front/chunk-000").mkdir(parents=True)
    (root / "videos/observation.images.front/chunk-000/file-000.mp4").write_bytes(b"video-bytes")

    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [4],
            "names": ["joint_1", "joint_2", "tcp_x", "gripper"],
        },
        "observation.action_joints": {
            "dtype": "float32",
            "shape": [3],
            "names": ["joint_1", "joint_2", "gripper"],
        },
        "observation.force": {
            "dtype": "float32",
            "shape": [6],
            "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.images.front": {"dtype": "video", "shape": [64, 64, 3], "names": None},
    }
    if include_action:
        features["action"] = {"dtype": "float32", "shape": [3], "names": ["a", "b", "c"]}

    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "robot_type": "test",
        "total_episodes": 2,
        "total_frames": 4,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100.0,
        "video_files_size_in_mb": 500.0,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    stats = {
        "observation.state": stat_entry(4),
        "observation.action_joints": stat_entry(3),
        "observation.force": stat_entry(6),
        "timestamp": stat_entry(1),
        "frame_index": stat_entry(1),
        "episode_index": stat_entry(1),
        "index": stat_entry(1),
        "task_index": stat_entry(1),
        "observation.images.front": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "count": [1]},
    }
    if include_action:
        stats["action"] = stat_entry(3)
    (root / "meta/stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    pq.write_table(pa.table({"task_index": pa.array([0], pa.int64()), "task": pa.array(["pick"])}), root / "meta/tasks.parquet")
    episode_table = pa.table(
        {
            "episode_index": pa.array([0, 1], pa.int64()),
            "length": pa.array([2, 2], pa.int64()),
            "dataset_from_index": pa.array([0, 2], pa.int64()),
            "dataset_to_index": pa.array([2, 4], pa.int64()),
            "data/chunk_index": pa.array([0, 0], pa.int64()),
            "data/file_index": pa.array([0, 1], pa.int64()),
            "task_index": pa.array([0, 0], pa.int64()),
            "stats/observation.state/mean": fixed_float([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "stats/observation.state/std": fixed_float([[1, 1, 1, 1], [2, 2, 2, 2]]),
            "stats/observation.state/count": fixed_int([[2], [2]]),
            "stats/observation.action_joints/mean": fixed_float([[9, 8, 7], [6, 5, 4]]),
            "stats/observation.action_joints/std": fixed_float([[1, 1, 1], [1, 1, 1]]),
            "stats/observation.force/mean": fixed_float([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]),
        }
    )
    pq.write_table(episode_table, root / "meta/episodes/chunk-000/file-000.parquet")

    state_rows_0 = [[10, 20, 30, 40], [11, 21, 31, 41]]
    if bad_state_width:
        state_rows_0 = [[10, 20, 30], [11, 21, 31]]
    write_data_shard(root / "data/chunk-000/file-000.parquet", 0, 0, state_rows_0, include_action=include_action)
    write_data_shard(root / "data/chunk-000/file-001.parquet", 1, 2, [[12, 22, 32, 42], [13, 23, 33, 43]], include_action=include_action)


def stat_entry(width: int) -> dict[str, Any]:
    return {
        "min": list(range(width)),
        "max": list(range(10, 10 + width)),
        "mean": list(range(20, 20 + width)),
        "std": list(range(30, 30 + width)),
        "q01": list(range(40, 40 + width)),
        "count": [4],
    }


def write_data_shard(path: Path, episode_index: int, start_index: int, state_rows: list[list[float]], include_action: bool = False) -> None:
    columns = {
        "observation.state": fixed_float(state_rows),
        "observation.action_joints": fixed_float([[1, 2, 3], [4, 5, 6]]),
        "observation.force": fixed_float([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]),
        "timestamp": pa.array([0.0, 0.1], pa.float32()),
        "frame_index": pa.array([0, 1], pa.int64()),
        "episode_index": pa.array([episode_index, episode_index], pa.int64()),
        "index": pa.array([start_index, start_index + 1], pa.int64()),
        "task_index": pa.array([0, 0], pa.int64()),
    }
    if include_action:
        columns["action"] = fixed_float([[100, 200, 300], [400, 500, 600]])
    table = pa.table(columns)
    pq.write_table(table, path)


def read_vectors(path: Path, column: str) -> list[list[float]]:
    return pq.read_table(path).column(column).combine_chunks().to_pylist()


def test_drop_feature_keys_updates_data_info_and_stats(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_minimal_dataset(source)
    monkeypatch.setattr("app.schema_editing.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})

    result = apply_schema_edit_plan(
        Cache(source),
        [DropFeatureKeysOperation(type="drop_feature_keys", feature="observation.state", keys=["gripper"])],
        output,
    )

    assert result["ok"] is True
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.state"]["shape"] == [3]
    assert info["features"]["observation.state"]["names"] == ["joint_1", "joint_2", "tcp_x"]
    stats = json.loads((output / "meta/stats.json").read_text(encoding="utf-8"))
    assert stats["observation.state"]["mean"] == [20, 21, 22]
    assert stats["observation.state"]["count"] == [4]
    assert read_vectors(output / "data/chunk-000/file-000.parquet", "observation.state") == [[10, 20, 30], [11, 21, 31]]
    episode_stats = read_vectors(output / "meta/episodes/chunk-000/file-000.parquet", "stats/observation.state/mean")
    assert episode_stats == [[1, 2, 3], [5, 6, 7]]


def test_combined_drop_feature_and_rename_preserves_video(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_minimal_dataset(source)
    monkeypatch.setattr("app.schema_editing.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})

    result = apply_schema_edit_plan(
        Cache(source),
        [
            DropFeatureOperation(type="drop_feature", feature="observation.force"),
            RenameFeatureOperation(type="rename_feature", source="observation.action_joints", target="action"),
        ],
        output,
    )

    assert result["ok"] is True
    data_table = pq.read_table(output / "data/chunk-000/file-000.parquet")
    assert "observation.force" not in data_table.column_names
    assert "observation.action_joints" not in data_table.column_names
    assert "action" in data_table.column_names
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert "observation.force" not in info["features"]
    assert "observation.action_joints" not in info["features"]
    assert "action" in info["features"]
    episode_table = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet")
    assert "stats/observation.force/mean" not in episode_table.column_names
    assert "stats/observation.action_joints/mean" not in episode_table.column_names
    assert "stats/action/mean" in episode_table.column_names
    assert (output / "videos/observation.images.front/chunk-000/file-000.mp4").read_bytes() == b"video-bytes"


def test_drop_existing_target_and_rename_replacement_in_one_plan(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_minimal_dataset(source, include_action=True)
    monkeypatch.setattr("app.schema_editing.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})

    result = apply_schema_edit_plan(
        Cache(source),
        [
            DropFeatureOperation(type="drop_feature", feature="action"),
            RenameFeatureOperation(type="rename_feature", source="observation.action_joints", target="action"),
        ],
        output,
    )

    assert result["ok"] is True
    data_table = pq.read_table(output / "data/chunk-000/file-000.parquet")
    assert "observation.action_joints" not in data_table.column_names
    assert "action" in data_table.column_names
    assert read_vectors(output / "data/chunk-000/file-000.parquet", "action") == [[1, 2, 3], [4, 5, 6]]
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert "observation.action_joints" not in info["features"]
    assert info["features"]["action"]["names"] == ["joint_1", "joint_2", "gripper"]
    stats = json.loads((output / "meta/stats.json").read_text(encoding="utf-8"))
    assert stats["action"]["mean"] == [20, 21, 22]


def test_apply_allows_existing_empty_output_directory(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    output = tmp_path / "empty_output"
    write_minimal_dataset(source)
    output.mkdir()
    monkeypatch.setattr("app.schema_editing.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})

    result = apply_schema_edit_plan(
        Cache(source),
        [ReorderFeatureKeysOperation(type="reorder_feature_keys", feature="observation.state", order=["joint_2", "joint_1", "tcp_x", "gripper"])],
        output,
    )

    assert result["ok"] is True
    assert (output / "meta/info.json").exists()


def test_apply_rejects_existing_non_empty_output_without_overwrite(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    output = tmp_path / "non_empty_output"
    write_minimal_dataset(source)
    output.mkdir()
    (output / "keep.txt").write_text("existing", encoding="utf-8")
    monkeypatch.setattr("app.schema_editing.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})

    result = apply_schema_edit_plan(
        Cache(source),
        [ReorderFeatureKeysOperation(type="reorder_feature_keys", feature="observation.state", order=["joint_2", "joint_1", "tcp_x", "gripper"])],
        output,
    )

    assert result["ok"] is False
    assert any("not empty" in error for error in result["errors"])
    assert (output / "keep.txt").read_text(encoding="utf-8") == "existing"


def test_rename_feature_keys_updates_info_only(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_minimal_dataset(source)
    monkeypatch.setattr("app.schema_editing.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})

    result = apply_schema_edit_plan(
        Cache(source),
        [
            RenameFeatureKeysOperation(
                type="rename_feature_keys",
                feature="observation.state",
                mapping={"gripper": "claw"},
            )
        ],
        output,
    )

    assert result["ok"] is True
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.state"]["shape"] == [4]
    assert info["features"]["observation.state"]["names"] == ["joint_1", "joint_2", "tcp_x", "claw"]
    stats = json.loads((output / "meta/stats.json").read_text(encoding="utf-8"))
    assert stats["observation.state"]["mean"] == [20, 21, 22, 23]
    assert read_vectors(output / "data/chunk-000/file-000.parquet", "observation.state") == [[10, 20, 30, 40], [11, 21, 31, 41]]


def test_reorder_feature_keys_reorders_data_and_stats(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_minimal_dataset(source)
    monkeypatch.setattr("app.schema_editing.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})

    result = apply_schema_edit_plan(
        Cache(source),
        [
            ReorderFeatureKeysOperation(
                type="reorder_feature_keys",
                feature="observation.state",
                order=["gripper", "joint_1", "joint_2", "tcp_x"],
            )
        ],
        output,
    )

    assert result["ok"] is True
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.state"]["shape"] == [4]
    assert info["features"]["observation.state"]["names"] == ["gripper", "joint_1", "joint_2", "tcp_x"]
    stats = json.loads((output / "meta/stats.json").read_text(encoding="utf-8"))
    assert stats["observation.state"]["mean"] == [23, 20, 21, 22]
    assert stats["observation.state"]["count"] == [4]
    assert read_vectors(output / "data/chunk-000/file-000.parquet", "observation.state") == [[40, 10, 20, 30], [41, 11, 21, 31]]
    episode_stats = read_vectors(output / "meta/episodes/chunk-000/file-000.parquet", "stats/observation.state/mean")
    assert episode_stats == [[4, 1, 2, 3], [8, 5, 6, 7]]


def test_validation_rejects_collision_protected_and_missing_names(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_minimal_dataset(source, include_action=True)
    cache = Cache(source)
    result = validate_schema_edit_plan(
        cache,
        [
            RenameFeatureOperation(type="rename_feature", source="observation.action_joints", target="action"),
            DropFeatureOperation(type="drop_feature", feature="timestamp"),
        ],
    )
    assert result["valid"] is False
    assert any("already exists" in error for error in result["errors"])
    assert any("protected" in error for error in result["errors"])

    cache.info["features"]["observation.state"]["names"] = None
    cache.features = cache.info["features"]
    result = validate_schema_edit_plan(
        cache,
        [DropFeatureKeysOperation(type="drop_feature_keys", feature="observation.state", keys=["gripper"])],
    )
    assert result["valid"] is False
    assert any("names are required" in error for error in result["errors"])


def test_apply_failure_removes_temp_and_keeps_source(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_minimal_dataset(source, bad_state_width=True)
    monkeypatch.setattr("app.schema_editing.validate_lerobot_v3_dataset", lambda *args, **kwargs: {"valid": True, "errors": [], "warnings": []})

    try:
        apply_schema_edit_plan(
            Cache(source),
            [DropFeatureKeysOperation(type="drop_feature_keys", feature="observation.state", keys=["gripper"])],
            output,
        )
    except ValueError as exc:
        assert "list_size" in str(exc) or "length" in str(exc)
    else:
        raise AssertionError("expected apply to fail")

    assert not output.exists()
    assert not list(tmp_path.glob(".output.schema-edit-*.tmp"))
    assert read_vectors(source / "data/chunk-000/file-000.parquet", "observation.state") == [[10, 20, 30], [11, 21, 31]]
