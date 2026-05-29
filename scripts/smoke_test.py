from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.editing import EditOperation, apply_edit_plan, apply_merge_plan, ffmpeg_executable
from app.main import DatasetCache


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_at_least(actual: int, minimum: int, label: str) -> None:
    if actual < minimum:
        raise AssertionError(f"{label}: expected at least {minimum}, got {actual}")


def check_pusht(path: Path) -> None:
    cache = DatasetCache(path)
    summary = cache.summary()
    assert_equal(summary["codebase_version"], "v3.0", "pusht version")
    assert_equal(summary["total_episodes"], 206, "pusht episode count")
    assert_equal(summary["total_frames"], 25650, "pusht frame count")
    assert_equal(summary["video_keys"], ["observation.image"], "pusht video keys")

    episodes = cache.list_episodes()
    assert_equal(len(episodes), 206, "pusht listed episodes")
    assert_equal(len(episodes[0]["videos"]), 1, "pusht episode 0 video count")

    detail = cache.load_episode_detail(0)
    assert_equal(len(detail["timeline"]["elapsed"]), 161, "pusht episode 0 frame count")
    assert_at_least(len(detail["series"]), 10, "pusht numeric series count")
    assert_equal(len(detail["videos"]), 1, "pusht episode detail video count")


def check_multiview(path: Path) -> None:
    cache = DatasetCache(path)
    summary = cache.summary()
    expected_keys = ["observation.images.front", "observation.images.wrist"]
    assert_equal(summary["codebase_version"], "v3.0", "multiview version")
    assert_equal(summary["total_episodes"], 1, "multiview episode count")
    assert_equal(summary["video_keys"], expected_keys, "multiview video keys")

    episodes = cache.list_episodes()
    assert_equal(len(episodes), 1, "multiview listed episodes")
    assert_equal(len(episodes[0]["videos"]), 2, "multiview episode video count")

    detail = cache.load_episode_detail(0)
    assert_equal(len(detail["timeline"]["elapsed"]), 3, "multiview frame count")
    assert_equal(len(detail["videos"]), 2, "multiview detail video count")
    assert_at_least(len(detail["series"]), 4, "multiview numeric series count")


def create_no_video_dataset(root: Path, lengths: list[int], task: str = "test task") -> None:
    (root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    frames = []
    episodes = []
    global_index = 0
    for episode_index, length in enumerate(lengths):
        start = global_index
        for frame_index in range(length):
            frames.append(
                {
                    "observation.state": np.array(
                        [episode_index + frame_index / 10, episode_index + 1 + frame_index / 10],
                        dtype=np.float32,
                    ),
                    "action": np.array([frame_index, frame_index + 1], dtype=np.float32),
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": np.float32(frame_index / 10),
                    "index": global_index,
                    "task_index": 0,
                    "next.done": frame_index == length - 1,
                }
            )
            global_index += 1
        episodes.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": start,
                "dataset_to_index": global_index,
                "tasks": [task],
                "length": length,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )

    pd.DataFrame(frames).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    pd.DataFrame(episodes).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task")).to_parquet(root / "meta/tasks.parquet")
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "fps": 10,
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "splits": {"train": f"0:{len(lengths)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "meta/stats.json").write_text("{}", encoding="utf-8")


def check_no_video_edit_and_merge() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        work = Path(directory)
        src = work / "src"
        create_no_video_dataset(src, [5, 6, 4])

        cache = DatasetCache(src)
        edited_path = work / "edited"
        edited = apply_edit_plan(
            cache,
            [
                EditOperation(type="delete_episode", episode_index=0),
                EditOperation(type="trim_episode", episode_index=1, start_time=0.1, end_time=0.4),
            ],
            edited_path,
            overwrite=True,
        )
        assert_equal(edited["ok"], True, "no-video edit apply ok")
        edited_cache = DatasetCache(edited_path)
        assert_equal(edited_cache.summary()["total_episodes"], 2, "edited episode count")
        assert_equal(edited_cache.summary()["total_frames"], 7, "edited frame count")

        merged_path = work / "merged"
        merged = apply_merge_plan([DatasetCache(src), DatasetCache(edited_path)], merged_path, overwrite=True)
        assert_equal(merged["ok"], True, "no-video merge apply ok")
        merged_cache = DatasetCache(merged_path)
        assert_equal(merged_cache.summary()["total_episodes"], 5, "merged episode count")
        assert_equal(merged_cache.summary()["total_frames"], 22, "merged frame count")
        assert_equal(merged_cache.summary()["total_tasks"], 1, "merged task count")


def check_video_edit(path: Path) -> None:
    if not ffmpeg_executable():
        print("skip: ffmpeg not found, video edit apply check skipped")
        return
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        output_path = Path(directory) / "video_edited"
        source_cache = DatasetCache(path)
        operations = [EditOperation(type="trim_episode", episode_index=0, start_time=0.1, end_time=0.3)]
        operations.extend(
            EditOperation(type="delete_episode", episode_index=int(index))
            for index in source_cache.episodes["episode_index"].tolist()
            if int(index) != 0
        )
        edited = apply_edit_plan(source_cache, operations, output_path, overwrite=True)
        assert_equal(edited["ok"], True, "video edit apply ok")
        edited_cache = DatasetCache(output_path)
        assert_equal(edited_cache.summary()["total_episodes"], 1, "video edited episode count")
        assert_equal(edited_cache.summary()["total_frames"], 2, "video edited frame count")
        assert_equal(edited_cache.summary()["video_keys"], source_cache.summary()["video_keys"], "video edited keys")
        print(f"ok: video edit apply checks passed ({path})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local LeRobot viewer smoke checks.")
    parser.add_argument(
        "--pusht",
        type=Path,
        default=Path("sample_datasets/pusht"),
        help="Path to the local lerobot/pusht sample dataset.",
    )
    parser.add_argument(
        "--multiview",
        type=Path,
        default=Path("tmp_multiview_dataset"),
        help="Path to the local synthetic multi-view dataset.",
    )
    args = parser.parse_args()

    if args.pusht.exists():
        check_pusht(args.pusht.resolve())
        print(f"ok: pusht dataset checks passed ({args.pusht})")
        check_video_edit(args.pusht.resolve())
    else:
        print(f"skip: pusht dataset not found ({args.pusht})")

    if args.multiview.exists():
        check_multiview(args.multiview.resolve())
        print(f"ok: multi-view dataset checks passed ({args.multiview})")
    else:
        print(f"skip: multi-view dataset not found ({args.multiview})")

    check_no_video_edit_and_merge()
    print("ok: no-video edit and merge checks passed")


if __name__ == "__main__":
    main()
