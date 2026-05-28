from __future__ import annotations

import math
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class EditOperation(BaseModel):
    type: str
    episode_index: int
    start_time: float | None = None
    end_time: float | None = None


class EditDryRunRequest(BaseModel):
    path: str
    operations: list[EditOperation] = Field(default_factory=list)


class EditApplyRequest(EditDryRunRequest):
    output_path: str
    overwrite: bool = False


class MergeValidationRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)


def validate_edit_plan(cache: Any, operations: list[EditOperation]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    normalized: list[dict[str, Any]] = []
    deleted: set[int] = set()
    trimmed: dict[int, int] = {}
    original_lengths = {
        int(row["episode_index"]): int(row.get("length", 0))
        for row in cache.episodes.to_dict(orient="records")
    }
    fps = float(cache.info["fps"])

    seen: dict[int, str] = {}
    for op in operations:
        episode_index = int(op.episode_index)
        if episode_index not in original_lengths:
            errors.append(f"episode 不存在: {episode_index}")
            continue
        if episode_index in seen:
            errors.append(f"episode {episode_index} 有多个编辑操作，当前版本每个 episode 只允许一个操作")
            continue
        seen[episode_index] = op.type

        if op.type == "delete_episode":
            deleted.add(episode_index)
            normalized.append({"type": op.type, "episode_index": episode_index})
            continue

        if op.type == "trim_episode":
            length = original_lengths[episode_index]
            start = op.start_time
            end = op.end_time
            if start is None or end is None:
                errors.append(f"episode {episode_index} 裁剪操作缺少 start_time 或 end_time")
                continue
            if not math.isfinite(start) or not math.isfinite(end):
                errors.append(f"episode {episode_index} 裁剪时间不是有效数字")
                continue
            if start < 0 or end <= start:
                errors.append(f"episode {episode_index} 裁剪区间非法: {start} - {end}")
                continue
            duration = length / fps
            if end > duration + 1e-6:
                errors.append(f"episode {episode_index} 裁剪终点超过 episode 时长: {end:.3f}s > {duration:.3f}s")
                continue
            start_frame = max(0, min(length - 1, int(math.floor(start * fps))))
            end_frame = max(start_frame + 1, min(length, int(math.ceil(end * fps))))
            new_length = end_frame - start_frame
            trimmed[episode_index] = new_length
            normalized.append(
                {
                    "type": op.type,
                    "episode_index": episode_index,
                    "start_time": round(start_frame / fps, 6),
                    "end_time": round(end_frame / fps, 6),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "new_length": new_length,
                    "old_length": length,
                }
            )
            continue

        errors.append(f"未知编辑操作: {op.type}")

    predicted_frames = 0
    for episode_index, length in original_lengths.items():
        if episode_index in deleted:
            continue
        predicted_frames += trimmed.get(episode_index, length)

    if cache.video_keys and normalized:
        warnings.append("数据集包含视频，应用编辑时需要同步裁剪/重写视频文件")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "operations": normalized,
        "original": {
            "episodes": len(original_lengths),
            "frames": int(cache.info.get("total_frames", sum(original_lengths.values()))),
            "video_keys": cache.video_keys,
        },
        "predicted": {
            "episodes": len(original_lengths) - len(deleted),
            "frames": predicted_frames,
            "deleted_episodes": len(deleted),
            "trimmed_episodes": len(trimmed),
        },
        "requires_video_processing": bool(cache.video_keys and normalized),
    }


def dataset_validation_summary(cache: Any) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if cache.info.get("codebase_version") != "v3.0":
        errors.append("codebase_version 不是 v3.0")
    if not cache.video_keys:
        warnings.append("数据集没有视频字段，编辑时只会处理 Parquet 和 metadata")

    episode_count = len(cache.episodes)
    total_frames = int(cache.info.get("total_frames", 0))
    length_sum = int(cache.episodes["length"].sum()) if "length" in cache.episodes.columns else 0
    if total_frames and length_sum and total_frames != length_sum:
        errors.append(f"total_frames 与 episode length 总和不一致: {total_frames} != {length_sum}")

    missing_data_files = []
    for _, episode in cache.episodes.iterrows():
        try:
            cache.data_file_for_episode(episode)
        except Exception as exc:  # FastAPI HTTPException in current cache implementation.
            missing_data_files.append(str(exc))
            if len(missing_data_files) >= 3:
                break
    if missing_data_files:
        errors.extend(missing_data_files)

    return {
        "path": str(cache.root),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": cache.summary(),
    }


def validate_merge_compatibility(caches: list[Any]) -> dict[str, Any]:
    datasets = [dataset_validation_summary(cache) for cache in caches]
    errors: list[str] = []
    warnings: list[str] = []

    if len(caches) < 2:
        errors.append("至少需要选择 2 个数据集进行合并")
    for item in datasets:
        if not item["valid"]:
            errors.append(f"数据集校验失败: {item['path']}")

    if caches:
        first = caches[0]
        first_features = canonical_features(first.features)
        first_video_keys = first.video_keys
        first_fps = float(first.info["fps"])
        first_robot_type = first.info.get("robot_type")
        for cache in caches[1:]:
            if float(cache.info["fps"]) != first_fps:
                errors.append(f"fps 不一致: {first.root}={first_fps}, {cache.root}={cache.info['fps']}")
            if canonical_features(cache.features) != first_features:
                errors.append(f"features schema 不一致: {cache.root}")
            if cache.video_keys != first_video_keys:
                errors.append(f"video keys 不一致: {cache.root}")
            if cache.info.get("robot_type") != first_robot_type:
                warnings.append(f"robot_type 不一致: {first.root}={first_robot_type}, {cache.root}={cache.info.get('robot_type')}")

    total_episodes = sum(int(cache.info.get("total_episodes", len(cache.episodes))) for cache in caches)
    total_frames = sum(int(cache.info.get("total_frames", 0)) for cache in caches)
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "datasets": datasets,
        "predicted": {
            "episodes": total_episodes,
            "frames": total_frames,
            "dataset_count": len(caches),
        },
    }


def apply_edit_plan(cache: Any, operations: list[EditOperation], output_path: Path, overwrite: bool = False) -> dict[str, Any]:
    plan = validate_edit_plan(cache, operations)
    if not plan["valid"]:
        return {"ok": False, "errors": plan["errors"], "warnings": plan["warnings"], "dry_run": plan}
    if not plan["operations"]:
        return {"ok": False, "errors": ["没有待应用的编辑操作"], "warnings": plan["warnings"], "dry_run": plan}
    if cache.video_keys:
        return {
            "ok": False,
            "errors": ["当前执行版本暂不支持含视频数据集的落盘编辑；需要先接入 ffmpeg 视频裁剪/重写"],
            "warnings": plan["warnings"],
            "dry_run": plan,
        }

    output_path = output_path.resolve()
    if output_path == cache.root.resolve():
        return {"ok": False, "errors": ["输出目录不能等于源数据集目录"], "warnings": [], "dry_run": plan}
    if output_path.exists():
        if not overwrite:
            return {"ok": False, "errors": [f"输出目录已存在: {output_path}"], "warnings": [], "dry_run": plan}
        if not output_path.is_dir():
            return {"ok": False, "errors": [f"输出路径不是目录: {output_path}"], "warnings": [], "dry_run": plan}
        shutil.rmtree(output_path)

    edited = build_edited_dataset(cache, plan["operations"])
    write_dataset(cache, edited["frames"], edited["episodes"], output_path)

    return {
        "ok": True,
        "output_path": str(output_path),
        "dry_run": plan,
        "summary": {
            "episodes": int(len(edited["episodes"])),
            "frames": int(len(edited["frames"])),
            "data_file": str(output_path / "data/chunk-000/file-000.parquet"),
            "episodes_file": str(output_path / "meta/episodes/chunk-000/file-000.parquet"),
        },
    }


def build_edited_dataset(cache: Any, normalized_operations: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    operations_by_episode = {int(op["episode_index"]): op for op in normalized_operations}
    fps = float(cache.info["fps"])
    new_frames: list[pd.DataFrame] = []
    new_episode_rows: list[dict[str, Any]] = []
    global_index = 0
    new_episode_index = 0

    for _, episode in cache.episodes.sort_values("episode_index").iterrows():
        source_episode_index = int(episode["episode_index"])
        operation = operations_by_episode.get(source_episode_index)
        if operation and operation["type"] == "delete_episode":
            continue

        frame_df = read_episode_frames(cache, episode)
        old_length = len(frame_df)
        if operation and operation["type"] == "trim_episode":
            frame_df = frame_df.iloc[int(operation["start_frame"]): int(operation["end_frame"])].copy()
        else:
            frame_df = frame_df.copy()

        new_length = len(frame_df)
        if new_length <= 0:
            continue

        frame_df = normalize_frame_columns(frame_df, new_episode_index, global_index, fps)
        row = base_episode_row(episode)
        row.update(
            {
                "episode_index": new_episode_index,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + new_length,
                "length": new_length,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )
        row.update(flatten_stats_for_episode(frame_df, cache.features))

        new_frames.append(frame_df)
        new_episode_rows.append(row)
        global_index += new_length
        new_episode_index += 1

    if not new_frames:
        raise ValueError("编辑后没有剩余 episode")

    return {
        "frames": pd.concat(new_frames, ignore_index=True),
        "episodes": pd.DataFrame(new_episode_rows),
    }


def read_episode_frames(cache: Any, episode: pd.Series) -> pd.DataFrame:
    df = pd.read_parquet(cache.data_file_for_episode(episode))
    if "dataset_from_index" in episode and "dataset_to_index" in episode and "index" in df.columns:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        sliced = df[(df["index"] >= start) & (df["index"] < end)].copy()
        if not sliced.empty:
            return sliced.reset_index(drop=True)
    if "episode_index" in df.columns:
        sliced = df[df["episode_index"] == int(episode["episode_index"])].copy()
        if not sliced.empty:
            return sliced.reset_index(drop=True)
    length = int(episode.get("length", len(df)))
    return df.head(length).copy().reset_index(drop=True)


def normalize_frame_columns(df: pd.DataFrame, episode_index: int, global_start: int, fps: float) -> pd.DataFrame:
    length = len(df)
    if "episode_index" in df.columns:
        df["episode_index"] = episode_index
    if "frame_index" in df.columns:
        df["frame_index"] = np.arange(length, dtype=np.int64)
    if "index" in df.columns:
        df["index"] = np.arange(global_start, global_start + length, dtype=np.int64)
    if "timestamp" in df.columns:
        df["timestamp"] = np.arange(length, dtype=np.float32) / np.float32(fps)
    if "next.done" in df.columns:
        df["next.done"] = False
        df.loc[df.index[-1], "next.done"] = True
    return df


def base_episode_row(episode: pd.Series) -> dict[str, Any]:
    return {
        key: value
        for key, value in episode.to_dict().items()
        if not key.startswith("stats/") and not key.startswith("videos/")
    }


def write_dataset(cache: Any, frames: pd.DataFrame, episodes: pd.DataFrame, output_path: Path) -> None:
    (output_path / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)

    frames.to_parquet(output_path / "data/chunk-000/file-000.parquet", index=False)
    episodes.to_parquet(output_path / "meta/episodes/chunk-000/file-000.parquet", index=False)
    shutil.copy2(cache.root / "meta/tasks.parquet", output_path / "meta/tasks.parquet")

    info = dict(cache.info)
    info["total_episodes"] = int(len(episodes))
    info["total_frames"] = int(len(frames))
    info["splits"] = {"train": f"0:{len(episodes)}"}
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    if "video_path" in info:
        info.pop("video_path")
    with (output_path / "meta/info.json").open("w", encoding="utf-8") as f:
        json.dump(clean_json_value(info), f, ensure_ascii=False, indent=2)

    stats = global_stats(frames, cache.features)
    with (output_path / "meta/stats.json").open("w", encoding="utf-8") as f:
        json.dump(clean_json_value(stats), f, ensure_ascii=False, indent=2)


def flatten_stats_for_episode(df: pd.DataFrame, features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, stats in global_stats(df, features).items():
        for stat_name, value in stats.items():
            result[f"stats/{key}/{stat_name}"] = value
    return result


def global_stats(df: pd.DataFrame, features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats = {}
    for key, feature in features.items():
        if key not in df.columns or feature.get("dtype") == "video":
            continue
        values = column_values(df[key])
        if values.size == 0:
            continue
        stats[key] = {
            "min": clean_json_value(np.nanmin(values, axis=0).tolist()),
            "max": clean_json_value(np.nanmax(values, axis=0).tolist()),
            "mean": clean_json_value(np.nanmean(values, axis=0).tolist()),
            "std": clean_json_value(np.nanstd(values, axis=0).tolist()),
            "count": [int(values.shape[0])],
        }
    return stats


def column_values(series: pd.Series) -> np.ndarray:
    rows = []
    for value in series:
        if isinstance(value, np.ndarray):
            rows.append(value.reshape(-1))
        elif isinstance(value, (list, tuple)):
            rows.append(np.array(value, dtype=object).reshape(-1))
        else:
            rows.append(np.array([value], dtype=object))
    if not rows:
        return np.array([])
    return np.array(rows, dtype=np.float64)


def clean_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return clean_json_value(value.tolist())
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_features(features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, feature in sorted(features.items()):
        result[key] = {
            "dtype": feature.get("dtype"),
            "shape": feature.get("shape"),
            "names": feature.get("names"),
        }
    return result


def resolve_dataset_path(raw_path: str) -> Path:
    return Path(raw_path).expanduser().resolve()
