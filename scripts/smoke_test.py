from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    else:
        print(f"skip: pusht dataset not found ({args.pusht})")

    if args.multiview.exists():
        check_multiview(args.multiview.resolve())
        print(f"ok: multi-view dataset checks passed ({args.multiview})")
    else:
        print(f"skip: multi-view dataset not found ({args.multiview})")


if __name__ == "__main__":
    main()
