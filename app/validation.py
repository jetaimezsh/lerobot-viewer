from __future__ import annotations

import importlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def validate_lerobot_v3_dataset(root: Path, run_official: bool = True) -> dict[str, Any]:
    root = root.resolve()
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "root": str(root),
        "summary": {},
        "official": official_lerobot_validation(root) if run_official else {"status": "skipped"},
    }

    try:
        info = read_json(root / "meta/info.json")
        tasks = read_tasks(root)
        episodes = read_episodes(root)
        result["summary"] = {
            "total_episodes": int(info.get("total_episodes", len(episodes))),
            "total_frames": int(info.get("total_frames", 0)),
            "total_tasks": int(info.get("total_tasks", len(tasks))),
            "features": len(info.get("features", {})),
        }
        validate_info(root, info, result)
        validate_tasks(tasks, info, result)
        validate_episode_metadata(root, info, episodes, result)
        validate_video_files(root, info, episodes, result)
        validate_frame_data(root, info, episodes, tasks, result)
        validate_stats(root, info, result)
    except Exception as exc:
        result["valid"] = False
        result["errors"].append(str(exc))

    official = result["official"]
    if official.get("status") == "failed":
        result["valid"] = False
        result["errors"].append(f"lerobot 官方校验失败: {official.get('error')}")

    if result["errors"]:
        result["valid"] = False
    return result


def official_lerobot_validation(root: Path) -> dict[str, Any]:
    try:
        module = importlib.import_module("lerobot.datasets.lerobot_dataset")
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": "未安装 lerobot，跳过官方 LeRobotDataset 校验",
            "error": str(exc),
        }

    dataset_cls = getattr(module, "LeRobotDataset", None)
    if dataset_cls is None:
        return {"status": "failed", "error": "lerobot.datasets.lerobot_dataset 中未找到 LeRobotDataset"}

    attempts = [
        {"repo_id": root.name, "root": root},
        {"repo_id": root.name, "root": str(root)},
        {"root": root},
        {"root": str(root)},
    ]
    errors = []
    for kwargs in attempts:
        try:
            dataset = dataset_cls(**kwargs)
            length = len(dataset)
            if length:
                _ = dataset[0]
            return {
                "status": "passed",
                "repo_id": kwargs.get("repo_id"),
                "root": str(kwargs.get("root", root)),
                "length": int(length),
            }
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")
    return {"status": "failed", "error": " | ".join(errors)}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"缺少文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_tasks(root: Path) -> pd.DataFrame:
    path = root / "meta/tasks.parquet"
    if not path.exists():
        raise ValueError(f"缺少文件: {path}")
    df = pd.read_parquet(path)
    if df.index.name == "task":
        df = df.reset_index()
    if "task_index" not in df.columns:
        raise ValueError("meta/tasks.parquet 缺少 task_index")
    return df


def read_episodes(root: Path) -> pd.DataFrame:
    episode_dir = root / "meta/episodes"
    if not episode_dir.exists():
        raise ValueError(f"缺少目录: {episode_dir}")
    files = sorted(episode_dir.glob("**/*.parquet"))
    if not files:
        raise ValueError(f"未找到 episode parquet: {episode_dir}")
    df = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    if "episode_index" not in df.columns:
        raise ValueError("episode metadata 缺少 episode_index")
    return df.sort_values("episode_index").reset_index(drop=True)


def validate_info(root: Path, info: dict[str, Any], result: dict[str, Any]) -> None:
    if info.get("codebase_version") != "v3.0":
        result["errors"].append(f"codebase_version 不是 v3.0: {info.get('codebase_version')}")
    for key in ["features", "fps", "data_path", "total_episodes", "total_frames", "total_tasks"]:
        if key not in info:
            result["errors"].append(f"info.json 缺少字段: {key}")
    for key in ["chunks_size", "data_files_size_in_mb", "video_files_size_in_mb"]:
        if key not in info:
            result["warnings"].append(f"info.json 缺少字段: {key}（v3.0 建议包含）")
    if any(feature.get("dtype") == "video" for feature in info.get("features", {}).values()) and not info.get("video_path"):
        result["errors"].append("info.json 有 video feature 但缺少 video_path")
    if not (root / "meta/stats.json").exists():
        result["errors"].append("缺少 meta/stats.json")
    # Validate video feature video_info completeness.
    for video_key, feature in info.get("features", {}).items():
        if feature.get("dtype") != "video":
            continue
        video_info = feature.get("video_info")
        if not isinstance(video_info, dict):
            result["errors"].append(f"video feature {video_key} 缺少 video_info")
            continue
        for video_field in ["video.fps", "video.codec", "video.pix_fmt"]:
            if video_field not in video_info:
                result["errors"].append(f"video feature {video_key} video_info 缺少 {video_field}")
        if "has_audio" not in video_info:
            result["warnings"].append(f"video feature {video_key} video_info 缺少 has_audio")
        if "is_depth_map" not in video_info:
            result["warnings"].append(f"video feature {video_key} video_info 缺少 is_depth_map")


def validate_tasks(tasks: pd.DataFrame, info: dict[str, Any], result: dict[str, Any]) -> None:
    indexes = sorted(int(value) for value in tasks["task_index"].tolist())
    expected = list(range(len(indexes)))
    if indexes != expected:
        result["errors"].append(f"task_index 必须连续: {indexes} != {expected}")
    if int(info.get("total_tasks", len(tasks))) != len(tasks):
        result["errors"].append(f"total_tasks 与 tasks 行数不一致: {info.get('total_tasks')} != {len(tasks)}")


def validate_episode_metadata(root: Path, info: dict[str, Any], episodes: pd.DataFrame, result: dict[str, Any]) -> None:
    required = ["length", "dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index"]
    for column in required:
        if column not in episodes.columns:
            result["errors"].append(f"episode metadata 缺少字段: {column}")
            return

    indexes = [int(value) for value in episodes["episode_index"].tolist()]
    expected_indexes = list(range(len(episodes)))
    if indexes != expected_indexes:
        result["errors"].append(f"episode_index 必须连续: {indexes[:10]} != {expected_indexes[:10]}")

    expected_start = 0
    for _, episode in episodes.iterrows():
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        length = int(episode["length"])
        if start != expected_start:
            result["errors"].append(f"episode {episode['episode_index']} dataset_from_index 不连续: {start} != {expected_start}")
        if end - start != length:
            result["errors"].append(f"episode {episode['episode_index']} length 与 dataset 范围不一致")
        expected_start = end
        data_path = info["data_path"].format(
            chunk_index=int(episode["data/chunk_index"]),
            file_index=int(episode["data/file_index"]),
        )
        if not (root / data_path).exists():
            result["errors"].append(f"数据文件不存在: {root / data_path}")

    if int(info.get("total_episodes", len(episodes))) != len(episodes):
        result["errors"].append(f"total_episodes 与 episodes 行数不一致: {info.get('total_episodes')} != {len(episodes)}")
    if int(info.get("total_frames", expected_start)) != expected_start:
        result["errors"].append(f"total_frames 与 episode 范围不一致: {info.get('total_frames')} != {expected_start}")

    for video_key, feature in info.get("features", {}).items():
        if feature.get("dtype") != "video":
            continue
        for suffix in ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]:
            column = f"videos/{video_key}/{suffix}"
            if column not in episodes.columns:
                result["errors"].append(f"video episode metadata 缺少字段: {column}")


def validate_video_files(root: Path, info: dict[str, Any], episodes: pd.DataFrame, result: dict[str, Any]) -> None:
    video_path_template = info.get("video_path")
    if not video_path_template:
        return
    fps = float(info.get("fps", 0) or 0)
    probed_durations: dict[Path, float | None] = {}
    warned_probe = False
    for video_key, feature in info.get("features", {}).items():
        if feature.get("dtype") != "video":
            continue
        prefix = f"videos/{video_key}"
        required = [f"{prefix}/{suffix}" for suffix in ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]]
        if any(column not in episodes.columns for column in required):
            continue
        for _, episode in episodes.iterrows():
            video_path = root / video_path_template.format(
                video_key=video_key,
                chunk_index=int(episode[f"{prefix}/chunk_index"]),
                file_index=int(episode[f"{prefix}/file_index"]),
            )
            if not video_path.exists():
                result["errors"].append(f"视频文件不存在: {video_path}")
                continue
            from_timestamp = float(episode[f"{prefix}/from_timestamp"])
            to_timestamp = float(episode[f"{prefix}/to_timestamp"])
            if not math.isfinite(from_timestamp) or not math.isfinite(to_timestamp) or to_timestamp <= from_timestamp:
                result["errors"].append(f"episode {episode['episode_index']} 视频时间边界非法: {video_key}")
                continue
            if fps > 0:
                expected_duration = int(episode["length"]) / fps
                actual_duration = to_timestamp - from_timestamp
                if abs(actual_duration - expected_duration) > max(0.1, 1.5 / fps):
                    result["warnings"].append(
                        f"episode {episode['episode_index']} 视频时间长度与 episode length/fps 不一致: {video_key}"
                    )
            if video_path not in probed_durations:
                probed_durations[video_path] = ffprobe_duration(video_path)
            duration = probed_durations[video_path]
            if duration is None:
                if not warned_probe:
                    result["warnings"].append("未找到 ffprobe 或无法读取视频时长，跳过视频文件时长边界校验")
                    warned_probe = True
                continue
            if to_timestamp > duration + max(0.2, 2.0 / fps if fps > 0 else 0.2):
                result["errors"].append(
                    f"episode {episode['episode_index']} 视频结束时间超过文件时长: {to_timestamp:.3f}s > {duration:.3f}s"
                )


def validate_frame_data(root: Path, info: dict[str, Any], episodes: pd.DataFrame, tasks: pd.DataFrame, result: dict[str, Any]) -> None:
    fps = float(info.get("fps", 0))
    task_indexes = {int(value) for value in tasks["task_index"].tolist()}
    feature_columns = {
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") != "video"
    }

    for _, episode in episodes.iterrows():
        df = pd.read_parquet(
            root / info["data_path"].format(
                chunk_index=int(episode["data/chunk_index"]),
                file_index=int(episode["data/file_index"]),
            )
        )
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        sliced = df[(df["index"] >= start) & (df["index"] < end)].copy() if "index" in df.columns else df.head(int(episode["length"])).copy()
        length = int(episode["length"])
        if len(sliced) != length:
            result["errors"].append(f"episode {episode['episode_index']} 数据行数不等于 length: {len(sliced)} != {length}")
            continue
        missing_columns = sorted(feature_columns - set(sliced.columns))
        if missing_columns:
            result["errors"].append(f"数据文件缺少 feature 列: {missing_columns}")
        validate_frame_sequence(sliced, episode, fps, task_indexes, result)
        validate_feature_shapes(sliced, info, result)


def validate_frame_sequence(df: pd.DataFrame, episode: pd.Series, fps: float, task_indexes: set[int], result: dict[str, Any]) -> None:
    episode_index = int(episode["episode_index"])
    length = len(df)
    if "episode_index" in df.columns and set(int(v) for v in df["episode_index"].unique()) != {episode_index}:
        result["errors"].append(f"episode {episode_index} 数据中的 episode_index 不一致")
    if "frame_index" in df.columns and df["frame_index"].tolist() != list(range(length)):
        result["errors"].append(f"episode {episode_index} frame_index 不连续")
    if "index" in df.columns:
        expected = list(range(int(episode["dataset_from_index"]), int(episode["dataset_to_index"])))
        if df["index"].tolist() != expected:
            result["errors"].append(f"episode {episode_index} 全局 index 不连续")
    if "timestamp" in df.columns and fps > 0:
        expected_ts = np.arange(length, dtype=np.float64) / fps
        actual = df["timestamp"].astype(float).to_numpy()
        if not np.allclose(actual, expected_ts, atol=1e-5):
            result["warnings"].append(f"episode {episode_index} timestamp 不是严格 frame_index/fps")
    if "task_index" in df.columns:
        unknown = sorted(set(int(v) for v in df["task_index"].unique()) - task_indexes)
        if unknown:
            result["errors"].append(f"episode {episode_index} 存在未知 task_index: {unknown}")


def validate_feature_shapes(df: pd.DataFrame, info: dict[str, Any], result: dict[str, Any]) -> None:
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") == "video" or key not in df.columns:
            continue
        expected_shape = feature.get("shape")
        if not expected_shape:
            continue
        first = df[key].iloc[0]
        actual_shape = value_shape(first)
        if len(expected_shape) == 1 and expected_shape[0] == 1 and actual_shape in [(), (1,)]:
            continue
        if tuple(expected_shape) != actual_shape:
            result["warnings"].append(f"feature {key} shape 可能不一致: info={expected_shape}, data={actual_shape}")


def value_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return tuple(np.array(value, dtype=object).shape)
    return ()


def ffprobe_duration(path: Path) -> float | None:
    executable = shutil.which("ffprobe")
    if not executable:
        for candidate in [
            Path("C:/ProgramData/chocolatey/bin/ffprobe.exe"),
            Path("C:/ffmpeg/bin/ffprobe.exe"),
            Path.cwd() / "tools/ffmpeg/bin/ffprobe.exe",
        ]:
            if candidate.exists():
                executable = str(candidate)
                break
    if not executable:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            for candidate in [
                Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe"),
                Path("C:/ffmpeg/bin/ffmpeg.exe"),
                Path.cwd() / "tools/ffmpeg/bin/ffmpeg.exe",
            ]:
                if candidate.exists():
                    ffmpeg = str(candidate)
                    break
        if ffmpeg:
            return ffmpeg_duration(Path(ffmpeg), path)
    if not executable:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def ffmpeg_duration(executable: Path, path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                str(executable),
                "-hide_banner",
                "-i",
                str(path),
            ],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return None
    output = result.stderr or result.stdout
    marker = "Duration:"
    if marker not in output:
        return None
    try:
        duration_text = output.split(marker, 1)[1].split(",", 1)[0].strip()
        hours, minutes, seconds = duration_text.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except Exception:
        return None


def validate_stats(root: Path, info: dict[str, Any], result: dict[str, Any]) -> None:
    stats = read_json(root / "meta/stats.json")
    total_frames = int(info.get("total_frames", 0))
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") == "video":
            continue
        if key not in stats:
            result["warnings"].append(f"meta/stats.json 缺少 feature stats: {key}")
            continue
        for stat_name in ["min", "max", "mean", "std", "count"]:
            if stat_name not in stats[key]:
                result["warnings"].append(f"meta/stats.json 缺少 {key}/{stat_name}")
        count = stats[key].get("count")
        if count and isinstance(count, list) and int(count[0]) != total_frames:
            result["warnings"].append(f"stats {key}/count 与 total_frames 不一致: {count[0]} != {total_frames}")
