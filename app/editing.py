from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EditOperation(BaseModel):
    type: str
    episode_index: int
    start_time: float | None = None
    end_time: float | None = None


class EditDryRunRequest(BaseModel):
    path: str
    operations: list[EditOperation] = Field(default_factory=list)


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
