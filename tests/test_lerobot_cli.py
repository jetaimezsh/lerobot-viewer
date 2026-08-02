from __future__ import annotations

from pathlib import Path

from app.lerobot_cli import RecomputeStatsRequest, build_recompute_stats_command, preview_recompute_stats_command


def test_recompute_command_default_output_is_stable(tmp_path: Path) -> None:
    root = tmp_path / "source_dataset"
    root.mkdir()
    request = RecomputeStatsRequest(repo_id="source_dataset", root=str(root), executable="lerobot-edit-dataset")

    args = build_recompute_stats_command(request)

    assert args[:2] == ["lerobot-edit-dataset", "--repo_id"]
    assert "--operation.type" in args
    assert args[args.index("--operation.type") + 1] == "recompute_stats"
    assert args[args.index("--new_repo_id") + 1] == "source_dataset_recomputed_stats"
    assert args[args.index("--new_root") + 1].endswith("source_dataset_recomputed_stats")
    assert args[args.index("--operation.skip_image_video") + 1] == "true"
    assert args[args.index("--operation.relative_action") + 1] == "false"
    assert args[args.index("--operation.overwrite") + 1] == "false"


def test_recompute_command_relative_action_serializes_excluded_joints(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    request = RecomputeStatsRequest(
        repo_id="source",
        root=str(root),
        new_repo_id="source_relative_stats",
        new_root=str(tmp_path / "relative"),
        relative_action=True,
        relative_exclude_joints=["gripper"],
        chunk_size=30,
        num_workers=4,
        executable="lerobot-edit-dataset",
    )

    args = build_recompute_stats_command(request)

    assert args[args.index("--operation.relative_action") + 1] == "true"
    assert args[args.index("--operation.relative_exclude_joints") + 1] == '["gripper"]'
    assert args[args.index("--operation.chunk_size") + 1] == "30"
    assert args[args.index("--operation.num_workers") + 1] == "4"


def test_preview_reports_missing_cli_as_structured_error(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    request = RecomputeStatsRequest(repo_id="source", root=str(root), executable=str(tmp_path / "missing.exe"))

    result = preview_recompute_stats_command(request)

    assert result["valid"] is False
    assert result["errors"]
    assert result["args"][0].endswith("missing.exe")
