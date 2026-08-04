from __future__ import annotations

import copy
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from app.editing import clean_json_value
from app.validation import validate_lerobot_v3_dataset


ALLOWED_LABEL_DTYPES = {"float32", "float64", "int32", "int64", "bool"}
PROTECTED_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
STANDARD_STATS = ("min", "max", "mean", "std", "count")


class RewardLabelOperation(BaseModel):
    episode_index: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    value: float

    @model_validator(mode="after")
    def validate_range(self) -> "RewardLabelOperation":
        if self.end_frame <= self.start_frame:
            raise ValueError("end_frame must be greater than start_frame")
        return self


class RewardLabelRequest(BaseModel):
    path: str
    feature: str
    dtype: Literal["float32", "float64", "int32", "int64", "bool"] = "float32"
    default_value: float | int | bool = 0.0
    reset_existing: bool = False
    labels: list[RewardLabelOperation] = Field(default_factory=list)


class RewardLabelApplyRequest(RewardLabelRequest):
    output_path: str
    overwrite: bool = False


@dataclass(frozen=True)
class RewardFeatureSpec:
    name: str
    dtype: str
    default_value: float | int | bool
    reset_existing: bool
    existed: bool


def validate_reward_label_plan(cache: Any, request: RewardLabelRequest) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    feature_name = clean_feature_name(request.feature)
    if not feature_name:
        errors.append("feature must not be empty")
    if feature_name in PROTECTED_FEATURES:
        errors.append(f"feature is protected: {feature_name}")
    if request.dtype not in ALLOWED_LABEL_DTYPES:
        errors.append(f"unsupported label dtype: {request.dtype}")

    existing_feature = cache.features.get(feature_name)
    if existing_feature:
        dtype = str(existing_feature.get("dtype", "")).lower()
        shape = existing_feature.get("shape")
        if dtype not in ALLOWED_LABEL_DTYPES:
            errors.append(f"{feature_name}: existing feature dtype is not supported for feature labeling: {dtype}")
        if shape not in (None, [], [1]):
            errors.append(f"{feature_name}: existing feature must be scalar or shape [1], got {shape}")

    episode_lengths = {
        int(row["episode_index"]): int(row.get("length", 0))
        for row in cache.episodes.to_dict(orient="records")
    }
    touched_episodes: set[int] = set()
    for label in request.labels:
        length = episode_lengths.get(int(label.episode_index))
        if length is None:
            errors.append(f"episode does not exist: {label.episode_index}")
            continue
        if label.end_frame > length:
            errors.append(
                f"episode {label.episode_index}: label range {label.start_frame}-{label.end_frame} exceeds length {length}"
            )
        touched_episodes.add(int(label.episode_index))

    data_shards = list_data_shards(cache.root)
    episode_shards = list_episode_shards(cache.root)
    if not data_shards:
        errors.append("no data parquet shards found under data/")

    return {
        "valid": not errors,
        "errors": dedupe_keep_order(errors),
        "warnings": dedupe_keep_order(warnings),
        "feature": {
            "name": feature_name,
            "dtype": request.dtype,
            "shape": [1],
            "default_value": cast_label_value(request.default_value, request.dtype),
            "reset_existing": request.reset_existing,
            "existed": bool(existing_feature),
        },
        "affected": {
            "data_shards": len(data_shards),
            "episode_metadata_shards": len(episode_shards),
            "labels": len(request.labels),
            "episodes": len(touched_episodes),
            "frames": sum(label.end_frame - label.start_frame for label in request.labels),
        },
    }


def apply_reward_label_plan(cache: Any, request: RewardLabelApplyRequest, output_path: Path) -> dict[str, Any]:
    dry_run = validate_reward_label_plan(cache, request)
    if not dry_run["valid"]:
        return {"ok": False, "errors": dry_run["errors"], "warnings": dry_run["warnings"], "dry_run": dry_run}

    source_root = cache.root.resolve()
    output_path = output_path.expanduser().resolve()
    if output_path == source_root:
        return {"ok": False, "errors": ["output_path must not equal source path"], "warnings": [], "dry_run": dry_run}
    if output_path.exists() and not output_path.is_dir():
        return {"ok": False, "errors": [f"output path is not a directory: {output_path}"], "warnings": [], "dry_run": dry_run}
    if output_path.exists() and not request.overwrite and any(output_path.iterdir()):
        return {
            "ok": False,
            "errors": [f"output path already exists and is not empty: {output_path}"],
            "warnings": [],
            "dry_run": dry_run,
        }

    spec = RewardFeatureSpec(
        name=dry_run["feature"]["name"],
        dtype=dry_run["feature"]["dtype"],
        default_value=cast_label_value(request.default_value, dry_run["feature"]["dtype"]),
        reset_existing=bool(request.reset_existing),
        existed=bool(dry_run["feature"]["existed"]),
    )
    labels_by_episode = labels_by_episode_index(request.labels)
    temp_path = output_path.parent / f".{output_path.name}.reward-label-{uuid.uuid4().hex}.tmp"
    backup_path = None
    try:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        shutil.copytree(source_root, temp_path)

        write_info(temp_path, cache.info, spec)
        processed_data_shards = rewrite_data_shards(temp_path, cache, spec, labels_by_episode)
        stats, episode_stats = compute_reward_stats(temp_path, cache, spec.name)
        write_stats(temp_path, cache.stats, spec.name, stats)
        processed_episode_metadata_shards = rewrite_episode_stats(temp_path, spec.name, episode_stats)

        validation = validate_lerobot_v3_dataset(temp_path, run_official=True, full_sweep=False)
        if not validation.get("valid"):
            return {
                "ok": False,
                "errors": validation.get("errors", []),
                "warnings": validation.get("warnings", []),
                "dry_run": dry_run,
                "validation": validation,
            }

        if output_path.exists():
            backup_path = output_path.parent / f".{output_path.name}.reward-label-backup-{uuid.uuid4().hex}.tmp"
            output_path.rename(backup_path)
        temp_path.rename(output_path)
        if backup_path and backup_path.exists():
            shutil.rmtree(backup_path)

        return {
            "ok": True,
            "output_path": str(output_path),
            "feature": dry_run["feature"],
            "processed_data_shards": processed_data_shards,
            "processed_episode_metadata_shards": processed_episode_metadata_shards,
            "dry_run": dry_run,
            "validation": validation,
        }
    except Exception:
        if backup_path and backup_path.exists() and not output_path.exists():
            backup_path.rename(output_path)
        raise
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


def clean_feature_name(value: str) -> str:
    return str(value or "").strip()


def list_data_shards(root: Path) -> list[Path]:
    return sorted((root / "data").glob("**/*.parquet")) if (root / "data").exists() else []


def list_episode_shards(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").glob("**/*.parquet")) if (root / "meta" / "episodes").exists() else []


def labels_by_episode_index(labels: list[RewardLabelOperation]) -> dict[int, list[RewardLabelOperation]]:
    result: dict[int, list[RewardLabelOperation]] = {}
    for label in labels:
        result.setdefault(int(label.episode_index), []).append(label)
    return result


def write_info(root: Path, source_info: dict[str, Any], spec: RewardFeatureSpec) -> None:
    info = copy.deepcopy(source_info)
    features = info.setdefault("features", {})
    features[spec.name] = {
        "dtype": spec.dtype,
        "shape": [1],
        "names": None,
    }
    path = root / "meta" / "info.json"
    path.write_text(json.dumps(clean_json_value(info), ensure_ascii=False, indent=2), encoding="utf-8")


def rewrite_data_shards(
    root: Path,
    cache: Any,
    spec: RewardFeatureSpec,
    labels_by_episode: dict[int, list[RewardLabelOperation]],
) -> int:
    processed = 0
    np_dtype = numpy_dtype(spec.dtype)
    episodes = cache.episodes.set_index("episode_index", drop=False)
    for shard in list_data_shards(root):
        df = pd.read_parquet(shard)
        if spec.name not in df.columns or spec.reset_existing:
            df[spec.name] = np.full(len(df), spec.default_value, dtype=np_dtype)
        else:
            df[spec.name] = df[spec.name].astype(np_dtype)

        for episode_index, labels in labels_by_episode.items():
            episode = episodes.loc[episode_index] if episode_index in episodes.index else None
            for label in labels:
                mask = frame_range_mask(df, episode, label)
                if mask.any():
                    df.loc[mask, spec.name] = cast_label_value(label.value, spec.dtype)
        df.to_parquet(shard, index=False)
        processed += 1
    return processed


def frame_range_mask(df: pd.DataFrame, episode: Any, label: RewardLabelOperation) -> pd.Series:
    if "episode_index" in df.columns and "frame_index" in df.columns:
        return (
            (df["episode_index"].astype(int) == int(label.episode_index))
            & (df["frame_index"].astype(int) >= int(label.start_frame))
            & (df["frame_index"].astype(int) < int(label.end_frame))
        )
    if episode is not None and "index" in df.columns and "dataset_from_index" in episode:
        start = int(episode["dataset_from_index"]) + int(label.start_frame)
        end = int(episode["dataset_from_index"]) + int(label.end_frame)
        return (df["index"].astype(int) >= start) & (df["index"].astype(int) < end)
    return pd.Series(False, index=df.index)


def compute_reward_stats(root: Path, cache: Any, feature_name: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    frames = pd.concat((pd.read_parquet(path) for path in list_data_shards(root)), ignore_index=True)
    global_stats = stats_for_values(frames[feature_name])
    episode_stats: dict[int, dict[str, Any]] = {}
    for _, episode in cache.episodes.sort_values("episode_index").iterrows():
        episode_index = int(episode["episode_index"])
        if "episode_index" in frames.columns and "frame_index" in frames.columns:
            sliced = frames[frames["episode_index"].astype(int) == episode_index]
        elif "index" in frames.columns and "dataset_from_index" in episode and "dataset_to_index" in episode:
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            sliced = frames[(frames["index"].astype(int) >= start) & (frames["index"].astype(int) < end)]
        else:
            sliced = pd.DataFrame()
        episode_stats[episode_index] = stats_for_values(sliced[feature_name]) if not sliced.empty else empty_stats()
    return global_stats, episode_stats


def stats_for_values(values: pd.Series) -> dict[str, Any]:
    arr = values.astype(float).to_numpy()
    if arr.size == 0:
        return empty_stats()
    return {
        "min": [float(np.nanmin(arr))],
        "max": [float(np.nanmax(arr))],
        "mean": [float(np.nanmean(arr))],
        "std": [float(np.nanstd(arr))],
        "count": [int(arr.size)],
    }


def numpy_dtype(dtype: str) -> Any:
    if dtype == "float32":
        return np.float32
    if dtype == "float64":
        return np.float64
    if dtype == "int32":
        return np.int32
    if dtype == "int64":
        return np.int64
    if dtype == "bool":
        return np.bool_
    raise ValueError(f"unsupported label dtype: {dtype}")


def cast_label_value(value: Any, dtype: str) -> float | int | bool:
    if dtype == "bool":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
    if dtype.startswith("int"):
        return int(value)
    return float(value)


def empty_stats() -> dict[str, Any]:
    return {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [0]}


def write_stats(root: Path, source_stats: dict[str, Any], feature_name: str, feature_stats: dict[str, Any]) -> None:
    stats = copy.deepcopy(source_stats or {})
    stats[feature_name] = feature_stats
    path = root / "meta" / "stats.json"
    path.write_text(json.dumps(clean_json_value(stats), ensure_ascii=False, indent=2), encoding="utf-8")


def rewrite_episode_stats(root: Path, feature_name: str, episode_stats: dict[int, dict[str, Any]]) -> int:
    processed = 0
    for shard in list_episode_shards(root):
        df = pd.read_parquet(shard)
        if "episode_index" not in df.columns:
            continue
        for stat_name in STANDARD_STATS:
            column = f"stats/{feature_name}/{stat_name}"
            df[column] = [
                copy.deepcopy(episode_stats.get(int(episode_index), empty_stats())[stat_name])
                for episode_index in df["episode_index"]
            ]
        df.to_parquet(shard, index=False)
        processed += 1
    return processed


def dedupe_keep_order(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
