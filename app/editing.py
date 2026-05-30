from __future__ import annotations

import copy
import math
import json
import importlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.validation import validate_lerobot_v3_dataset


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


class EditToolStatusRequest(BaseModel):
    path: str | None = None


class MergeValidationRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)


class MergeApplyRequest(MergeValidationRequest):
    output_path: str
    overwrite: bool = False


def editing_tool_status(cache: Any | None = None) -> dict[str, Any]:
    checks = [
        python_package_check("pandas", pd.__version__),
        python_package_check("pyarrow"),
        python_package_check("numpy", np.__version__),
        lerobot_package_check(),
        filesystem_write_check(),
        ffmpeg_check(),
    ]
    missing = build_missing_items(checks)
    capabilities = build_edit_capabilities(checks)
    dataset = None
    if cache is not None:
        has_video = bool(cache.video_keys)
        dataset_can_apply = (
            capabilities["delete_episode_video"]["available"] and capabilities["trim_episode_video"]["available"]
            if has_video
            else capabilities["delete_episode_no_video"]["available"] and capabilities["trim_episode_no_video"]["available"]
        )
        dataset = {
            "path": str(cache.root),
            "has_video": has_video,
            "video_keys": cache.video_keys,
            "can_apply_now": dataset_can_apply,
            "reason": (
                "当前数据集不含视频，可执行删除/裁剪并生成新目录"
                if not has_video
                else "当前数据集包含视频，需安装并接入 ffmpeg 后才能执行落盘编辑"
            ),
        }

    required_ok = all(check["ok"] for check in checks if check["required_for_no_video"])
    ffmpeg_ok = next((check["ok"] for check in checks if check["id"] == "ffmpeg"), False)
    recommendations = build_recommendations(missing)
    return {
        "ready_for_no_video_edits": required_ok,
        "ready_for_video_edits": required_ok and ffmpeg_ok,
        "checks": checks,
        "missing": missing,
        "capabilities": list(capabilities.values()),
        "recommendations": recommendations,
        "dataset": dataset,
    }


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


def apply_merge_plan(caches: list[Any], output_path: Path, overwrite: bool = False) -> dict[str, Any]:
    validation = validate_merge_compatibility(caches)
    if not validation["valid"]:
        return {"ok": False, "errors": validation["errors"], "warnings": validation["warnings"], "validation": validation}
    if any(cache.video_keys for cache in caches):
        return {
            "ok": False,
            "errors": ["当前执行版本暂不支持含视频数据集的合并落盘；需要先安装并接入 ffmpeg 视频拼接/重写"],
            "warnings": validation["warnings"],
            "validation": validation,
        }

    output_path = output_path.resolve()
    source_roots = {cache.root.resolve() for cache in caches}
    if output_path in source_roots:
        return {"ok": False, "errors": ["输出目录不能等于任一源数据集目录"], "warnings": [], "validation": validation}
    if output_path.exists():
        if not overwrite:
            return {"ok": False, "errors": [f"输出目录已存在: {output_path}"], "warnings": [], "validation": validation}
        if not output_path.is_dir():
            return {"ok": False, "errors": [f"输出路径不是目录: {output_path}"], "warnings": [], "validation": validation}
        shutil.rmtree(output_path)

    merged = build_merged_dataset(caches)
    write_dataset(
        cache=caches[0],
        frames=merged["frames"],
        episodes=merged["episodes"],
        output_path=output_path,
        tasks_df=merged["tasks"],
    )
    validation = validate_lerobot_v3_dataset(output_path)
    if not validation["valid"]:
        return {
            "ok": False,
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "validation": validation,
        }
    return {
        "ok": True,
        "output_path": str(output_path),
        "validation": validation,
        "summary": {
            "datasets": len(caches),
            "episodes": int(len(merged["episodes"])),
            "frames": int(len(merged["frames"])),
            "tasks": int(len(merged["tasks"])),
            "data_file": str(output_path / "data/chunk-000/file-000.parquet"),
            "episodes_file": str(output_path / "meta/episodes/chunk-000/file-000.parquet"),
        },
    }


def apply_edit_plan(cache: Any, operations: list[EditOperation], output_path: Path, overwrite: bool = False) -> dict[str, Any]:
    plan = validate_edit_plan(cache, operations)
    if not plan["valid"]:
        return {"ok": False, "errors": plan["errors"], "warnings": plan["warnings"], "dry_run": plan}
    if not plan["operations"]:
        return {"ok": False, "errors": ["没有待应用的编辑操作"], "warnings": plan["warnings"], "dry_run": plan}
    if cache.video_keys:
        executable = ffmpeg_executable()
        if not executable:
            return {
                "ok": False,
                "errors": ["当前数据集包含视频，执行删除/裁剪落盘需要 ffmpeg；当前环境未找到可执行的 ffmpeg"],
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

    try:
        edited = build_edited_dataset(cache, plan["operations"])
        tasks_df = rebuild_tasks_for_frames(cache, edited["frames"], edited["episodes"])
        video_info_overrides = {}
        if cache.video_keys:
            video_info_overrides = write_edited_videos(cache, edited["video_jobs"], edited["episodes"], output_path)
        write_dataset(
            cache,
            edited["frames"],
            edited["episodes"],
            output_path,
            tasks_df=tasks_df,
            video_info_overrides=video_info_overrides,
        )
    except Exception as exc:
        return {
            "ok": False,
            "errors": [str(exc)],
            "warnings": plan["warnings"],
            "dry_run": plan,
        }
    validation = validate_lerobot_v3_dataset(output_path)
    if not validation["valid"]:
        return {
            "ok": False,
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "dry_run": plan,
            "validation": validation,
        }

    return {
        "ok": True,
        "output_path": str(output_path),
        "dry_run": plan,
        "validation": validation,
        "summary": {
            "episodes": int(len(edited["episodes"])),
            "frames": int(len(edited["frames"])),
            "videos": list(cache.video_keys),
            "data_file": str(output_path / "data/chunk-000/file-000.parquet"),
            "episodes_file": str(output_path / "meta/episodes/chunk-000/file-000.parquet"),
        },
    }


def build_merged_dataset(caches: list[Any]) -> dict[str, pd.DataFrame]:
    tasks_df, task_maps = build_merged_tasks(caches)
    first = caches[0]
    fps = float(first.info["fps"])
    merged_frames: list[pd.DataFrame] = []
    merged_episodes: list[dict[str, Any]] = []
    global_index = 0
    next_episode_index = 0

    for cache_index, cache in enumerate(caches):
        task_map = task_maps[cache_index]
        for _, episode in cache.episodes.sort_values("episode_index").iterrows():
            frame_df = read_episode_frames(cache, episode).copy()
            if "task_index" in frame_df.columns:
                frame_df["task_index"] = frame_df["task_index"].map(lambda value: task_map.get(int(value), int(value)))
            frame_df = normalize_frame_columns(frame_df, next_episode_index, global_index, fps)
            length = len(frame_df)

            row = base_episode_row(episode)
            if "task_index" in row and row["task_index"] is not None:
                row["task_index"] = task_map.get(int(row["task_index"]), int(row["task_index"]))
            row.update(
                {
                    "episode_index": next_episode_index,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": global_index,
                    "dataset_to_index": global_index + length,
                    "length": length,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )
            row.update(flatten_stats_for_episode(frame_df, first.features))
            merged_frames.append(frame_df)
            merged_episodes.append(row)
            global_index += length
            next_episode_index += 1

    if not merged_frames:
        raise ValueError("合并后没有 episode")
    return {
        "frames": pd.concat(merged_frames, ignore_index=True),
        "episodes": pd.DataFrame(merged_episodes),
        "tasks": tasks_df,
    }


def build_merged_tasks(caches: list[Any]) -> tuple[pd.DataFrame, list[dict[int, int]]]:
    task_to_index: dict[str, int] = {}
    task_maps: list[dict[int, int]] = []
    for cache in caches:
        tasks = read_tasks_table(cache.root)
        cache_map: dict[int, int] = {}
        for row in tasks.to_dict(orient="records"):
            task_text = task_text_from_record(row)
            old_index = int(row.get("task_index", len(cache_map)))
            if task_text not in task_to_index:
                task_to_index[task_text] = len(task_to_index)
            cache_map[old_index] = task_to_index[task_text]
        task_maps.append(cache_map)

    ordered_tasks = sorted(task_to_index.items(), key=lambda item: item[1])
    return (
        pd.DataFrame(
            {"task_index": [index for _, index in ordered_tasks]},
            index=pd.Index([task for task, _ in ordered_tasks], name="task"),
        ),
        task_maps,
    )


def rebuild_tasks_for_frames(cache: Any, frames: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    source_tasks = read_tasks_table(cache.root)
    task_lookup = {}
    for row in source_tasks.to_dict(orient="records"):
        task_lookup[int(row.get("task_index", len(task_lookup)))] = task_text_from_record(row)

    used_old_indexes = sorted(int(value) for value in frames["task_index"].dropna().unique()) if "task_index" in frames.columns else [0]
    old_to_new = {old_index: new_index for new_index, old_index in enumerate(used_old_indexes)}
    if "task_index" in frames.columns:
        frames["task_index"] = frames["task_index"].map(lambda value: old_to_new[int(value)])
    if "task_index" in episodes.columns:
        episodes["task_index"] = episodes["task_index"].map(lambda value: old_to_new.get(int(value), int(value)))

    task_texts = [task_lookup.get(old_index, str(old_index)) for old_index in used_old_indexes]
    return pd.DataFrame(
        {"task_index": list(range(len(task_texts)))},
        index=pd.Index(task_texts, name="task"),
    )


def read_tasks_table(root: Path) -> pd.DataFrame:
    df = pd.read_parquet(root / "meta/tasks.parquet")
    if df.index.name == "task":
        df = df.reset_index()
    return df


def task_text_from_record(record: dict[str, Any]) -> str:
    for key in ["task", "name", "text"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return json.dumps(clean_json_value(record), ensure_ascii=False, sort_keys=True)


def build_edited_dataset(cache: Any, normalized_operations: list[dict[str, Any]]) -> dict[str, Any]:
    operations_by_episode = {int(op["episode_index"]): op for op in normalized_operations}
    fps = float(cache.info["fps"])
    new_frames: list[pd.DataFrame] = []
    new_episode_rows: list[dict[str, Any]] = []
    video_jobs: list[dict[str, Any]] = []
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
            source_start_frame = int(operation["start_frame"])
            source_end_frame = int(operation["end_frame"])
            frame_df = frame_df.iloc[source_start_frame:source_end_frame].copy()
        else:
            source_start_frame = 0
            source_end_frame = old_length
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
        if cache.video_keys:
            video_jobs.append(
                {
                    "source_episode_index": source_episode_index,
                    "source_episode": episode.copy(),
                    "new_episode_index": new_episode_index,
                    "start_frame": source_start_frame,
                    "end_frame": source_end_frame,
                    "length": new_length,
                }
            )
        global_index += new_length
        new_episode_index += 1

    if not new_frames:
        raise ValueError("编辑后没有剩余 episode")

    return {
        "frames": pd.concat(new_frames, ignore_index=True),
        "episodes": pd.DataFrame(new_episode_rows),
        "video_jobs": video_jobs,
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


def write_edited_videos(cache: Any, video_jobs: list[dict[str, Any]], episodes: pd.DataFrame, output_path: Path) -> dict[str, dict[str, Any]]:
    executable = ffmpeg_executable()
    if not executable:
        raise RuntimeError("ffmpeg is required for video dataset edits")
    fps = float(cache.info["fps"])
    if fps <= 0:
        raise RuntimeError("dataset fps must be positive to rewrite videos")
    if not video_jobs:
        raise RuntimeError("video edit produced no output episodes")

    output_path.mkdir(parents=True, exist_ok=True)
    video_info_overrides: dict[str, dict[str, Any]] = {}
    for video_key in cache.video_keys:
        output_video = output_video_path(cache, output_path, video_key)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        first_source_video = cache.video_file_for_episode(video_jobs[0]["source_episode"], video_key)
        encoding = video_encoding_for_key(cache, video_key, first_source_video, fps)
        video_info_overrides[video_key] = {
            "video.fps": encoding["fps"],
            "video.codec": encoding["codec"],
            "video.pix_fmt": encoding["pix_fmt"],
            "has_audio": False,
        }
        cumulative = 0.0
        with tempfile.TemporaryDirectory(dir=output_path) as directory:
            temp_dir = Path(directory)
            segment_paths = []
            for job_index, job in enumerate(video_jobs):
                source_episode = job["source_episode"]
                source_video = cache.video_file_for_episode(source_episode, video_key)
                prefix = f"videos/{video_key}"
                from_col = f"{prefix}/from_timestamp"
                source_from = float(source_episode.get(from_col, 0.0))
                start_time = source_from + float(job["start_frame"]) / fps
                duration = float(job["length"]) / fps
                segment_path = temp_dir / f"segment-{job_index:06d}.mp4"
                run_ffmpeg(
                    [
                        executable,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{start_time:.6f}",
                        "-t",
                        f"{duration:.6f}",
                        "-i",
                        str(source_video),
                        "-an",
                        "-vf",
                        f"fps={format_fps(encoding['fps'])}",
                        "-frames:v",
                        str(int(job["length"])),
                        *encoding["encoder_options"],
                        "-pix_fmt",
                        encoding["pix_fmt"],
                        "-movflags",
                        "+faststart",
                        str(segment_path),
                    ]
                )
                if not segment_path.exists() or segment_path.stat().st_size == 0:
                    raise RuntimeError(f"ffmpeg did not create a valid segment: {segment_path}")
                segment_paths.append(segment_path)

                new_episode_index = int(job["new_episode_index"])
                row_mask = episodes["episode_index"].astype(int) == new_episode_index
                episode_start = cumulative
                cumulative += duration
                episodes.loc[row_mask, f"{prefix}/chunk_index"] = 0
                episodes.loc[row_mask, f"{prefix}/file_index"] = 0
                episodes.loc[row_mask, f"{prefix}/from_timestamp"] = round(episode_start, 6)
                episodes.loc[row_mask, f"{prefix}/to_timestamp"] = round(cumulative, 6)

            if len(segment_paths) == 1:
                shutil.copy2(segment_paths[0], output_video)
            else:
                concat_list = temp_dir / "concat.txt"
                concat_list.write_text(
                    "\n".join(f"file '{ffmpeg_concat_path(path)}'" for path in segment_paths),
                    encoding="utf-8",
                )
                run_ffmpeg(
                    [
                        executable,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_list),
                        "-c",
                        "copy",
                        str(output_video),
                    ]
                )
            if not output_video.exists() or output_video.stat().st_size == 0:
                raise RuntimeError(f"ffmpeg did not create a valid output video: {output_video}")
    return video_info_overrides


def video_encoding_for_key(cache: Any, video_key: str, source_video: Path, dataset_fps: float) -> dict[str, Any]:
    feature = cache.features.get(video_key, {})
    video_info = feature.get("video_info") or {}
    stream_info = ffprobe_video_stream(source_video)
    codec = normalize_video_codec(stream_info.get("codec_name") or video_info.get("video.codec"))
    pix_fmt = str(stream_info.get("pix_fmt") or video_info.get("video.pix_fmt") or "yuv420p")
    fps = parse_frame_rate(stream_info.get("avg_frame_rate") or stream_info.get("r_frame_rate")) or float(video_info.get("video.fps") or dataset_fps)
    keyframe_interval = source_keyframe_interval(source_video)
    return {
        "codec": codec,
        "pix_fmt": pix_fmt,
        "fps": fps,
        "keyframe_interval": keyframe_interval,
        "encoder_options": ffmpeg_encoder_options(codec, keyframe_interval),
    }


def normalize_video_codec(codec: Any) -> str:
    normalized = str(codec or "h264").lower()
    aliases = {
        "avc1": "h264",
        "h265": "hevc",
        "vp09": "vp9",
    }
    return aliases.get(normalized, normalized)


def ffmpeg_encoder_options(codec: str, keyframe_interval: int | None = None) -> list[str]:
    gop_options = ["-g", str(keyframe_interval)] if keyframe_interval and keyframe_interval > 0 else []
    if codec == "av1":
        return [
            "-c:v",
            "libaom-av1",
            "-usage",
            "realtime",
            "-cpu-used",
            "8",
            "-row-mt",
            "1",
            "-lag-in-frames",
            "0",
            "-auto-alt-ref",
            "0",
            "-denoise-noise-level",
            "0",
            "-crf",
            "30",
            "-b:v",
            "0",
            *gop_options,
        ]
    if codec == "h264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            *gop_options,
            *(["-keyint_min", str(keyframe_interval), "-sc_threshold", "0"] if keyframe_interval and keyframe_interval > 0 else []),
        ]
    if codec == "hevc":
        return [
            "-c:v",
            "libx265",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            *gop_options,
            *(["-keyint_min", str(keyframe_interval), "-sc_threshold", "0"] if keyframe_interval and keyframe_interval > 0 else []),
        ]
    if codec == "vp9":
        return ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "6", "-crf", "32", "-b:v", "0", *gop_options]
    raise RuntimeError(f"unsupported source video codec for frame-accurate trim: {codec}")


def ffprobe_video_stream(path: Path) -> dict[str, Any]:
    executable = ffprobe_executable()
    if not executable:
        return {}
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,r_frame_rate,avg_frame_rate,width,height",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return {}
    try:
        streams = json.loads(result.stdout).get("streams") or []
    except json.JSONDecodeError:
        return {}
    return streams[0] if streams else {}


def source_keyframe_interval(path: Path) -> int | None:
    executable = ffprobe_executable()
    if not executable:
        return None
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            "%+#300",
            "-show_frames",
            "-show_entries",
            "frame=key_frame",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return None
    keyframe_indexes: list[int] = []
    frame_index = 0
    for line in result.stdout.splitlines():
        value = line.split(",", 1)[0].strip()
        if value not in {"0", "1"}:
            continue
        if value == "1":
            keyframe_indexes.append(frame_index)
            if len(keyframe_indexes) >= 6:
                break
        frame_index += 1
    intervals = [
        current - previous
        for previous, current in zip(keyframe_indexes, keyframe_indexes[1:])
        if current > previous
    ]
    if not intervals:
        return None
    return int(round(float(np.median(intervals))))


def ffprobe_executable() -> str | None:
    executable = shutil.which("ffprobe")
    if executable:
        return executable
    for candidate in [
        Path("C:/ProgramData/chocolatey/bin/ffprobe.exe"),
        Path("C:/ffmpeg/bin/ffprobe.exe"),
        Path.cwd() / "tools/ffmpeg/bin/ffprobe.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def parse_frame_rate(value: Any) -> float | None:
    if not value:
        return None
    text = str(value)
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_float = float(denominator)
            if denominator_float == 0:
                return None
            return float(numerator) / denominator_float
        return float(text)
    except ValueError:
        return None


def output_video_path(cache: Any, output_path: Path, video_key: str) -> Path:
    template = cache.info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    return output_path / template.format(video_key=video_key, chunk_index=0, file_index=0)


def run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "ffmpeg failed").strip()
        raise RuntimeError(message)


def ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        imageio_executable = imageio_ffmpeg.get_ffmpeg_exe()
        if imageio_executable and Path(imageio_executable).exists():
            return str(imageio_executable)
    except Exception:
        pass
    for candidate in [
        Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe"),
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path.cwd() / "tools/ffmpeg/bin/ffmpeg.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def format_fps(fps: float) -> str:
    if float(fps).is_integer():
        return str(int(fps))
    return f"{fps:.6f}".rstrip("0").rstrip(".")


def ffmpeg_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def write_dataset(
    cache: Any,
    frames: pd.DataFrame,
    episodes: pd.DataFrame,
    output_path: Path,
    tasks_df: pd.DataFrame | None = None,
    video_info_overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    (output_path / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)

    frames.to_parquet(output_path / "data/chunk-000/file-000.parquet", index=False)
    episodes.to_parquet(output_path / "meta/episodes/chunk-000/file-000.parquet", index=False)
    if tasks_df is None:
        shutil.copy2(cache.root / "meta/tasks.parquet", output_path / "meta/tasks.parquet")
        total_tasks = int(cache.info.get("total_tasks", len(cache.tasks)))
    else:
        tasks_df.to_parquet(output_path / "meta/tasks.parquet")
        total_tasks = int(len(tasks_df))

    info = copy.deepcopy(cache.info)
    info["total_episodes"] = int(len(episodes))
    info["total_frames"] = int(len(frames))
    info["total_tasks"] = total_tasks
    info["splits"] = {"train": f"0:{len(episodes)}"}
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    has_video_features = any(feature.get("dtype") == "video" for feature in info.get("features", {}).values())
    if has_video_features:
        info["video_path"] = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        for video_key, override in (video_info_overrides or {}).items():
            feature = info.get("features", {}).get(video_key)
            if not feature or feature.get("dtype") != "video":
                continue
            video_info = feature.setdefault("video_info", {})
            video_info.update(override)
    elif "video_path" in info:
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


def python_package_check(name: str, version: str | None = None) -> dict[str, Any]:
    if version is None:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "installed")
        except Exception as exc:
            return {
                "id": name,
                "label": f"Python package: {name}",
                "ok": False,
                "required_for_no_video": True,
                "required_for_video": True,
                "detail": str(exc),
            }
    return {
        "id": name,
        "label": f"Python package: {name}",
        "ok": True,
        "required_for_no_video": True,
        "required_for_video": True,
        "detail": version,
    }


def lerobot_package_check() -> dict[str, Any]:
    try:
        module = importlib.import_module("lerobot")
        version = getattr(module, "__version__", "installed")
        importlib.import_module("lerobot.datasets.lerobot_dataset")
        return {
            "id": "lerobot",
            "label": "Official package: lerobot",
            "ok": True,
            "required_for_no_video": False,
            "required_for_video": False,
            "required_for_official_validation": True,
            "detail": version,
        }
    except Exception as exc:
        return {
            "id": "lerobot",
            "label": "Official package: lerobot",
            "ok": False,
            "required_for_no_video": False,
            "required_for_video": False,
            "required_for_official_validation": True,
            "detail": str(exc),
        }


def filesystem_write_check() -> dict[str, Any]:
    try:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "write_test.txt"
            path.write_text("ok", encoding="utf-8")
            ok = path.read_text(encoding="utf-8") == "ok"
    except Exception as exc:
        return {
            "id": "filesystem_write",
            "label": "输出目录写入能力",
            "ok": False,
            "required_for_no_video": True,
            "required_for_video": True,
            "detail": str(exc),
        }
    return {
        "id": "filesystem_write",
        "label": "输出目录写入能力",
        "ok": ok,
        "required_for_no_video": True,
        "required_for_video": True,
        "detail": f"cwd={Path.cwd()}",
    }


def ffmpeg_check() -> dict[str, Any]:
    executable = ffmpeg_executable()
    if not executable:
        return {
            "id": "ffmpeg",
            "label": "ffmpeg 视频处理工具",
            "ok": False,
            "required_for_no_video": False,
            "required_for_video": True,
            "detail": "未在 PATH、常见安装目录或 imageio-ffmpeg 中找到 ffmpeg",
        }
    try:
        result = subprocess.run(
            [executable, "-version"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        first_line = (result.stdout or result.stderr).splitlines()[0] if (result.stdout or result.stderr) else executable
        ok = result.returncode == 0
    except Exception as exc:
        return {
            "id": "ffmpeg",
            "label": "ffmpeg 视频处理工具",
            "ok": False,
            "required_for_no_video": False,
            "required_for_video": True,
            "detail": str(exc),
        }
    return {
        "id": "ffmpeg",
        "label": "ffmpeg 视频处理工具",
        "ok": ok,
        "required_for_no_video": False,
        "required_for_video": True,
        "detail": first_line,
    }


def build_missing_items(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = []
    for check in checks:
        if check["ok"]:
            continue
        if check["id"] == "ffmpeg":
            missing.append(
                {
                    "id": "ffmpeg",
                    "name": "ffmpeg",
                    "impact": "无法对包含视频的数据集执行删除 episode、裁剪 episode 后生成合法 v3.0 输出目录",
                    "reason": "视频文件必须和 Parquet frame、episode metadata 同步裁剪/重写，否则视频时间轴会与数据不一致",
                    "fix": "安装 ffmpeg，并确保 ffmpeg.exe 可以在 PowerShell 中直接运行",
                    "raw_detail": check["detail"],
                }
            )
        elif check["id"] == "filesystem_write":
            missing.append(
                {
                    "id": "filesystem_write",
                    "name": "输出目录写入能力",
                    "impact": "无法生成新的数据集目录",
                    "reason": "编辑操作会写入新的 data、meta 和 stats 文件",
                    "fix": "换一个有写入权限的项目目录或输出目录",
                    "raw_detail": check["detail"],
                }
            )
        elif check["id"] in {"pandas", "pyarrow", "numpy"}:
            missing.append(
                {
                    "id": check["id"],
                    "name": check["id"],
                    "impact": "无法读取、修改或写入 LeRobot v3.0 的 Parquet/数组数据",
                    "reason": f"{check['id']} 是数据修改流程的必需 Python 包",
                    "fix": "运行 requirements.txt 安装，或在 System环境 页面点击安装 requirements.txt",
                    "raw_detail": check["detail"],
                }
            )
        elif check["id"] == "lerobot":
            missing.append(
                {
                    "id": "lerobot",
                    "name": "lerobot 官方包",
                    "impact": "无法执行 Hugging Face LeRobotDataset 官方打开校验",
                    "reason": "官方包不参与当前无视频数据写入，但它能作为训练前兼容性校验的最后一道检查",
                    "fix": "安装 lerobot 包后重新运行 System环境 检测和输出数据集校验",
                    "raw_detail": check["detail"],
                }
            )
    return missing


def build_edit_capabilities(checks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ok_by_id = {check["id"]: bool(check["ok"]) for check in checks}
    base_missing = [item for item in ["pandas", "pyarrow", "numpy", "filesystem_write"] if not ok_by_id.get(item)]
    video_missing = base_missing + ([] if ok_by_id.get("ffmpeg") else ["ffmpeg"])
    return {
        "delete_episode_no_video": {
            "id": "delete_episode_no_video",
            "name": "删除 episode（无视频数据集）",
            "available": not base_missing,
            "blocked_by": base_missing,
        },
        "trim_episode_no_video": {
            "id": "trim_episode_no_video",
            "name": "裁剪 episode（无视频数据集）",
            "available": not base_missing,
            "blocked_by": base_missing,
        },
        "delete_episode_video": {
            "id": "delete_episode_video",
            "name": "删除 episode（含视频数据集）",
            "available": not video_missing,
            "blocked_by": video_missing,
        },
        "trim_episode_video": {
            "id": "trim_episode_video",
            "name": "裁剪 episode（含视频数据集）",
            "available": not video_missing,
            "blocked_by": video_missing,
        },
        "merge_no_video": {
            "id": "merge_no_video",
            "name": "合并多个数据集（无视频）",
            "available": not base_missing,
            "blocked_by": base_missing,
        },
        "merge_video": {
            "id": "merge_video",
            "name": "合并多个数据集（含视频）",
            "available": False,
            "blocked_by": (video_missing + ["视频合并功能待实现"]) if video_missing else ["视频合并功能待实现"],
        },
    }


def build_recommendations(missing: list[dict[str, Any]]) -> list[str]:
    if not missing:
        return ["当前环境已具备无视频和含视频数据编辑所需工具。"]
    recommendations = []
    for item in missing:
        recommendations.append(f"{item['name']}: {item['fix']}")
    return recommendations


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
