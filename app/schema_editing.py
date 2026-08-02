from __future__ import annotations

import copy
import json
import math
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from app.validation import validate_lerobot_v3_dataset


PROTECTED_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
IMAGE_VIDEO_DTYPES = {"image", "video"}
NUMERIC_DTYPE_PREFIXES = ("float", "int", "uint", "bool")
STATS_COUNT_NAME = "count"


class DropFeatureOperation(BaseModel):
    type: Literal["drop_feature"]
    feature: str


class DropFeatureKeysOperation(BaseModel):
    type: Literal["drop_feature_keys"]
    feature: str
    keys: list[str]


class RenameFeatureOperation(BaseModel):
    type: Literal["rename_feature"]
    source: str
    target: str


class RenameFeatureKeysOperation(BaseModel):
    type: Literal["rename_feature_keys"]
    feature: str
    mapping: dict[str, str]


class ReorderFeatureKeysOperation(BaseModel):
    type: Literal["reorder_feature_keys"]
    feature: str
    order: list[str]


SchemaOperation = Annotated[
    DropFeatureOperation
    | DropFeatureKeysOperation
    | RenameFeatureOperation
    | RenameFeatureKeysOperation
    | ReorderFeatureKeysOperation,
    Field(discriminator="type"),
]


class SchemaEditDryRunRequest(BaseModel):
    path: str
    operations: list[SchemaOperation] = Field(default_factory=list)


class SchemaEditApplyRequest(SchemaEditDryRunRequest):
    output_path: str
    overwrite: bool = False


@dataclass(frozen=True)
class NormalizedNames:
    flat_names: list[str]
    original_kind: Literal["list", "dict"]
    groups: list[tuple[str, list[str]]] | None = None


def inspect_editable_schema(cache: Any) -> dict[str, Any]:
    features = []
    for name, feature in cache.features.items():
        reason = schema_edit_reason(name, feature)
        names = feature.get("names")
        can_drop_keys = can_drop_feature_keys(feature) and not reason
        features.append(
            {
                "name": name,
                "dtype": feature.get("dtype"),
                "shape": feature.get("shape"),
                "names": names,
                "editable": reason is None,
                "can_drop_keys": can_drop_keys,
                "protected": name in PROTECTED_FEATURES,
                "reason": reason,
            }
        )
    return {"features": features}


def validate_schema_edit_plan(cache: Any, operations: list[SchemaOperation]) -> dict[str, Any]:
    return _build_validation(cache, operations, scan_episode_stats=True)


def apply_schema_edit_plan(
    cache: Any,
    operations: list[SchemaOperation],
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    dry_run = validate_schema_edit_plan(cache, operations)
    if not dry_run["valid"]:
        return {"ok": False, "errors": dry_run["errors"], "warnings": dry_run["warnings"], "dry_run": dry_run}

    source_root = cache.root.resolve()
    output_path = output_path.expanduser().resolve()
    if output_path == source_root:
        return {"ok": False, "errors": ["output_path must not equal source path"], "warnings": [], "dry_run": dry_run}
    if output_path.exists() and not output_path.is_dir():
        return {"ok": False, "errors": [f"output path is not a directory: {output_path}"], "warnings": [], "dry_run": dry_run}
    if output_path.exists() and not overwrite and any(output_path.iterdir()):
        return {
            "ok": False,
            "errors": [f"output path already exists and is not empty: {output_path}"],
            "warnings": [],
            "dry_run": dry_run,
        }

    plan = dry_run["normalized_plan"]
    temp_path = output_path.parent / f".{output_path.name}.schema-edit-{uuid.uuid4().hex}.tmp"
    backup_path = None
    processed_data_shards = 0
    processed_episode_metadata_shards = 0
    copied_files = 0
    try:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True)

        new_info = transform_info(cache.info, plan)
        new_stats = transform_stats(cache.stats, plan)

        for src in iter_source_files(source_root):
            rel = src.relative_to(source_root)
            dst = temp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            rel_posix = rel.as_posix()
            if is_data_parquet(rel_posix):
                transform_data_parquet(src, dst, plan)
                processed_data_shards += 1
            elif is_episode_parquet(rel_posix):
                transform_episode_metadata_parquet(src, dst, plan)
                processed_episode_metadata_shards += 1
            elif rel_posix == "meta/info.json":
                write_json(dst, new_info)
            elif rel_posix == "meta/stats.json":
                write_json(dst, new_stats)
            else:
                shutil.copy2(src, dst)
                copied_files += 1

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
            backup_path = output_path.parent / f".{output_path.name}.schema-edit-backup-{uuid.uuid4().hex}.tmp"
            output_path.rename(backup_path)
        temp_path.rename(output_path)
        if backup_path and backup_path.exists():
            shutil.rmtree(backup_path)

        return {
            "ok": True,
            "output_path": str(output_path),
            "processed_data_shards": processed_data_shards,
            "processed_episode_metadata_shards": processed_episode_metadata_shards,
            "copied_files": copied_files,
            "copied_video_files": count_video_files(output_path),
            "schema_before": dry_run["before_features"],
            "schema_after": dry_run["after_features"],
            "validation": validation,
            "dry_run": dry_run,
        }
    except Exception:
        if backup_path and backup_path.exists() and not output_path.exists():
            backup_path.rename(output_path)
        raise
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


def _build_validation(cache: Any, operations: list[SchemaOperation], scan_episode_stats: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    features = cache.features
    stats = cache.stats or {}
    normalized = empty_normalized_plan()

    if not operations:
        errors.append("at least one schema operation is required")

    requested_drop_features = {
        clean_feature_name(op.feature)
        for op in operations
        if op.type == "drop_feature" and clean_feature_name(op.feature)
    }
    seen_sources: dict[str, str] = {}
    rename_sources: set[str] = set()
    rename_targets: set[str] = set()
    drop_features: set[str] = set()
    key_edit_features: set[str] = set()

    for op in operations:
        if op.type == "drop_feature":
            feature_name = clean_feature_name(op.feature)
            validate_source_feature(feature_name, features, errors)
            validate_feature_editable(feature_name, features.get(feature_name), errors)
            remember_feature_use(feature_name, "drop_feature", seen_sources, errors)
            drop_features.add(feature_name)
            if feature_name:
                normalized["drop_features"].append(feature_name)

        elif op.type == "drop_feature_keys":
            feature_name = clean_feature_name(op.feature)
            validate_source_feature(feature_name, features, errors)
            remember_feature_use(feature_name, "drop_feature_keys", seen_sources, errors)
            key_edit_features.add(feature_name)
            feature = features.get(feature_name)
            key_result = normalize_drop_keys(feature_name, feature, op.keys)
            errors.extend(key_result["errors"])
            if key_result["valid"]:
                normalized["drop_feature_keys"][feature_name] = key_result["entry"]
                validate_stats_for_key_edit(stats, feature_name, key_result["entry"]["old_shape"][0], errors)

        elif op.type == "rename_feature_keys":
            feature_name = clean_feature_name(op.feature)
            validate_source_feature(feature_name, features, errors)
            remember_feature_use(feature_name, "rename_feature_keys", seen_sources, errors)
            key_edit_features.add(feature_name)
            feature = features.get(feature_name)
            key_result = normalize_rename_keys(feature_name, feature, op.mapping)
            errors.extend(key_result["errors"])
            if key_result["valid"]:
                normalized["rename_feature_keys"][feature_name] = key_result["entry"]

        elif op.type == "reorder_feature_keys":
            feature_name = clean_feature_name(op.feature)
            validate_source_feature(feature_name, features, errors)
            remember_feature_use(feature_name, "reorder_feature_keys", seen_sources, errors)
            key_edit_features.add(feature_name)
            feature = features.get(feature_name)
            key_result = normalize_reorder_keys(feature_name, feature, op.order)
            errors.extend(key_result["errors"])
            if key_result["valid"]:
                normalized["reorder_feature_keys"][feature_name] = key_result["entry"]
                validate_stats_for_key_edit(stats, feature_name, key_result["entry"]["old_shape"][0], errors)

        elif op.type == "rename_feature":
            source = clean_feature_name(op.source)
            target = clean_feature_name(op.target)
            validate_source_feature(source, features, errors)
            validate_feature_editable(source, features.get(source), errors)
            remember_feature_use(source, "rename_feature", seen_sources, errors)
            rename_sources.add(source)
            if not target:
                errors.append("rename target must not be empty")
            if target in features and target not in requested_drop_features:
                errors.append(f"rename target already exists: {target}")
            if target in PROTECTED_FEATURES:
                errors.append(f"rename target is protected: {target}")
            if target in rename_targets:
                errors.append(f"multiple rename operations target the same feature: {target}")
            if target:
                rename_targets.add(target)
            if source and target:
                normalized["rename_features"][source] = target

    for source, target in normalized["rename_features"].items():
        if target in rename_sources:
            errors.append(f"rename chain is not supported: {source} -> {target}")

    for feature_name in sorted(drop_features & key_edit_features):
        errors.append(f"feature cannot be both dropped and key-edited: {feature_name}")
    for feature_name in sorted(drop_features & rename_sources):
        errors.append(f"feature cannot be both dropped and renamed: {feature_name}")
    for feature_name in sorted(key_edit_features & rename_sources):
        errors.append(f"feature cannot be both key-edited and renamed: {feature_name}")

    data_shards = list_data_shards(cache.root)
    episode_shards = list_episode_shards(cache.root)
    if not data_shards:
        errors.append("no data parquet shards found under data/")

    if scan_episode_stats and not errors:
        validate_episode_stats_for_vector_key_edits(episode_shards, normalized, errors)

    before_features = inspect_editable_schema(cache)["features"]
    after_info = transform_info(cache.info, normalized) if not errors else copy.deepcopy(cache.info)
    after_features = inspect_features_from_info(after_info)
    affected = {
        "data_shards": len(data_shards),
        "episode_metadata_shards": len(episode_shards),
        "video_files_reencoded": 0,
        "estimated_read_bytes": estimate_files_size(data_shards + episode_shards),
        "estimated_write_bytes": estimate_files_size(data_shards + episode_shards),
    }

    return {
        "valid": not errors,
        "errors": dedupe_keep_order(errors),
        "warnings": dedupe_keep_order(warnings),
        "normalized_plan": normalized,
        "affected": affected,
        "before_features": before_features,
        "after_features": after_features,
    }


def empty_normalized_plan() -> dict[str, Any]:
    return {
        "drop_feature_keys": {},
        "rename_feature_keys": {},
        "reorder_feature_keys": {},
        "drop_features": [],
        "rename_features": {},
    }


def clean_feature_name(value: str) -> str:
    return str(value or "").strip()


def validate_source_feature(name: str, features: dict[str, Any], errors: list[str]) -> None:
    if not name:
        errors.append("feature name must not be empty")
    elif name not in features:
        errors.append(f"feature does not exist: {name}")


def validate_feature_editable(name: str, feature: dict[str, Any] | None, errors: list[str]) -> None:
    reason = schema_edit_reason(name, feature)
    if reason:
        errors.append(f"{name}: {reason}")


def remember_feature_use(name: str, use: str, seen: dict[str, str], errors: list[str]) -> None:
    if not name:
        return
    previous = seen.get(name)
    if previous:
        errors.append(f"feature has multiple schema operations: {name} ({previous}, {use})")
    seen[name] = use


def normalize_drop_keys(feature_name: str, feature: dict[str, Any] | None, keys: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    if feature is None:
        return {"valid": False, "errors": errors, "entry": None}
    reason = schema_edit_reason(feature_name, feature)
    if reason:
        errors.append(f"{feature_name}: {reason}")
    if not can_drop_feature_keys(feature):
        errors.append(f"{feature_name}: key deletion is only supported for numeric one-dimensional vector features")

    shape = feature.get("shape") or []
    old_dim = int(shape[0]) if isinstance(shape, list) and len(shape) == 1 else 0
    normalized_names = normalize_names(feature.get("names"), old_dim, errors, feature_name)
    requested = [str(key).strip() for key in keys]
    if not requested:
        errors.append(f"{feature_name}: at least one key is required")
    if any(not key for key in requested):
        errors.append(f"{feature_name}: key name must not be empty")
    duplicates = sorted({key for key in requested if requested.count(key) > 1})
    if duplicates:
        errors.append(f"{feature_name}: duplicate keys requested: {duplicates}")

    if normalized_names:
        missing = [key for key in requested if key and key not in normalized_names.flat_names]
        if missing:
            errors.append(f"{feature_name}: keys not found: {missing}")
        drop_indices = [normalized_names.flat_names.index(key) for key in requested if key in normalized_names.flat_names]
        keep_indices = [idx for idx in range(old_dim) if idx not in set(drop_indices)]
        if not keep_indices:
            errors.append(f"{feature_name}: deleting all keys is not allowed; drop the whole feature instead")
    else:
        drop_indices = []
        keep_indices = []

    if errors:
        return {"valid": False, "errors": errors, "entry": None}

    assert normalized_names is not None
    entry = {
        "drop_keys": requested,
        "drop_indices": drop_indices,
        "keep_indices": keep_indices,
        "old_shape": [old_dim],
        "new_shape": [len(keep_indices)],
        "new_names": rebuild_names(normalized_names, keep_indices),
    }
    return {"valid": True, "errors": [], "entry": entry}


def normalize_rename_keys(feature_name: str, feature: dict[str, Any] | None, mapping: dict[str, str]) -> dict[str, Any]:
    errors: list[str] = []
    if feature is None:
        return {"valid": False, "errors": errors, "entry": None}
    reason = schema_edit_reason(feature_name, feature)
    if reason:
        errors.append(f"{feature_name}: {reason}")
    if not can_drop_feature_keys(feature):
        errors.append(f"{feature_name}: key rename is only supported for numeric one-dimensional vector features")

    shape = feature.get("shape") or []
    old_dim = int(shape[0]) if isinstance(shape, list) and len(shape) == 1 else 0
    normalized_names = normalize_names(feature.get("names"), old_dim, errors, feature_name)
    cleaned = {str(source).strip(): str(target).strip() for source, target in (mapping or {}).items()}
    if not cleaned:
        errors.append(f"{feature_name}: at least one key rename is required")
    if any(not source for source in cleaned):
        errors.append(f"{feature_name}: source key name must not be empty")
    if any(not target for target in cleaned.values()):
        errors.append(f"{feature_name}: target key name must not be empty")

    if normalized_names:
        missing = [source for source in cleaned if source and source not in normalized_names.flat_names]
        if missing:
            errors.append(f"{feature_name}: keys not found: {missing}")
        renamed_flat = [cleaned.get(name, name) for name in normalized_names.flat_names]
        duplicates = sorted({name for name in renamed_flat if renamed_flat.count(name) > 1})
        if duplicates:
            errors.append(f"{feature_name}: duplicate key names after rename: {duplicates}")
    else:
        renamed_flat = []

    if errors:
        return {"valid": False, "errors": errors, "entry": None}

    assert normalized_names is not None
    entry = {
        "mapping": cleaned,
        "old_shape": [old_dim],
        "new_shape": [old_dim],
        "new_names": rebuild_names_from_flat(normalized_names, renamed_flat),
    }
    return {"valid": True, "errors": [], "entry": entry}


def normalize_reorder_keys(feature_name: str, feature: dict[str, Any] | None, order: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    if feature is None:
        return {"valid": False, "errors": errors, "entry": None}
    reason = schema_edit_reason(feature_name, feature)
    if reason:
        errors.append(f"{feature_name}: {reason}")
    if not can_drop_feature_keys(feature):
        errors.append(f"{feature_name}: key reorder is only supported for numeric one-dimensional vector features")

    shape = feature.get("shape") or []
    old_dim = int(shape[0]) if isinstance(shape, list) and len(shape) == 1 else 0
    normalized_names = normalize_names(feature.get("names"), old_dim, errors, feature_name)
    requested = [str(key).strip() for key in order]
    if not requested:
        errors.append(f"{feature_name}: key order is required")
    if any(not key for key in requested):
        errors.append(f"{feature_name}: key name must not be empty")
    duplicates = sorted({key for key in requested if requested.count(key) > 1})
    if duplicates:
        errors.append(f"{feature_name}: duplicate keys requested: {duplicates}")

    if normalized_names:
        missing = [key for key in normalized_names.flat_names if key not in requested]
        extras = [key for key in requested if key and key not in normalized_names.flat_names]
        if missing:
            errors.append(f"{feature_name}: reordered keys missing existing names: {missing}")
        if extras:
            errors.append(f"{feature_name}: reordered keys do not exist: {extras}")
        order_indices = [normalized_names.flat_names.index(key) for key in requested if key in normalized_names.flat_names]
    else:
        order_indices = []

    if errors:
        return {"valid": False, "errors": errors, "entry": None}

    assert normalized_names is not None
    entry = {
        "order": requested,
        "order_indices": order_indices,
        "old_shape": [old_dim],
        "new_shape": [old_dim],
        "new_names": [normalized_names.flat_names[idx] for idx in order_indices],
    }
    return {"valid": True, "errors": [], "entry": entry}


def normalize_names(raw: Any, expected_dim: int, errors: list[str], feature_name: str) -> NormalizedNames | None:
    if raw is None:
        errors.append(f"{feature_name}: names are required for key editing")
        return None
    if isinstance(raw, list):
        flat = [str(item) for item in raw]
        kind: Literal["list", "dict"] = "list"
        groups = None
    elif isinstance(raw, dict):
        groups = []
        flat = []
        for key, value in raw.items():
            if not isinstance(value, list):
                errors.append(f"{feature_name}: names group {key} must be a list")
                return None
            names = [str(item) for item in value]
            groups.append((str(key), names))
            flat.extend(names)
        kind = "dict"
    else:
        errors.append(f"{feature_name}: names must be a list or dict")
        return None
    if len(flat) != expected_dim:
        errors.append(f"{feature_name}: names length {len(flat)} does not match shape {expected_dim}")
    duplicates = sorted({name for name in flat if flat.count(name) > 1})
    if duplicates:
        errors.append(f"{feature_name}: duplicate names are not supported for key editing: {duplicates}")
    return NormalizedNames(flat_names=flat, original_kind=kind, groups=groups)


def rebuild_names(names: NormalizedNames, keep_indices: list[int]) -> list[str] | dict[str, list[str]]:
    keep = set(keep_indices)
    if names.original_kind == "list":
        return [name for idx, name in enumerate(names.flat_names) if idx in keep]
    result: dict[str, list[str]] = {}
    offset = 0
    for group_name, group_values in names.groups or []:
        kept = []
        for local_idx, value in enumerate(group_values):
            if offset + local_idx in keep:
                kept.append(value)
        result[group_name] = kept
        offset += len(group_values)
    return result


def rebuild_names_from_flat(names: NormalizedNames, flat_values: list[str]) -> list[str] | dict[str, list[str]]:
    if names.original_kind == "list":
        return list(flat_values)
    result: dict[str, list[str]] = {}
    offset = 0
    for group_name, group_values in names.groups or []:
        group_len = len(group_values)
        result[group_name] = list(flat_values[offset : offset + group_len])
        offset += group_len
    return result


def schema_edit_reason(name: str, feature: dict[str, Any] | None) -> str | None:
    if name in PROTECTED_FEATURES:
        return "protected LeRobot frame feature"
    if not feature:
        return "feature metadata is missing"
    dtype = str(feature.get("dtype", "")).lower()
    if dtype in IMAGE_VIDEO_DTYPES:
        return "image/video schema editing is not supported in v1"
    return None


def can_drop_feature_keys(feature: dict[str, Any]) -> bool:
    dtype = str(feature.get("dtype", "")).lower()
    shape = feature.get("shape")
    return dtype.startswith(NUMERIC_DTYPE_PREFIXES) and isinstance(shape, list) and len(shape) == 1 and int(shape[0]) > 1


def is_data_parquet(rel_posix: str) -> bool:
    return rel_posix.startswith("data/") and rel_posix.endswith(".parquet")


def is_episode_parquet(rel_posix: str) -> bool:
    return rel_posix.startswith("meta/episodes/") and rel_posix.endswith(".parquet")


def list_data_shards(root: Path) -> list[Path]:
    return sorted((root / "data").glob("**/*.parquet")) if (root / "data").exists() else []


def list_episode_shards(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").glob("**/*.parquet")) if (root / "meta" / "episodes").exists() else []


def iter_source_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def estimate_files_size(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def inspect_features_from_info(info: dict[str, Any]) -> list[dict[str, Any]]:
    features = []
    for name, feature in (info.get("features") or {}).items():
        reason = schema_edit_reason(name, feature)
        features.append(
            {
                "name": name,
                "dtype": feature.get("dtype"),
                "shape": feature.get("shape"),
                "names": feature.get("names"),
                "editable": reason is None,
                "can_drop_keys": can_drop_feature_keys(feature) and reason is None,
                "protected": name in PROTECTED_FEATURES,
                "reason": reason,
            }
        )
    return features


def transform_info(info: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    new_info = copy.deepcopy(info)
    features = new_info.get("features") or {}
    for feature_name, entry in plan["drop_feature_keys"].items():
        feature = features[feature_name]
        feature["shape"] = entry["new_shape"]
        feature["names"] = copy.deepcopy(entry["new_names"])
    for feature_name, entry in plan["rename_feature_keys"].items():
        features[feature_name]["names"] = copy.deepcopy(entry["new_names"])
    for feature_name, entry in plan["reorder_feature_keys"].items():
        features[feature_name]["shape"] = entry["new_shape"]
        features[feature_name]["names"] = copy.deepcopy(entry["new_names"])
    for feature_name in plan["drop_features"]:
        features.pop(feature_name, None)
    rename_map = plan["rename_features"]
    if rename_map:
        renamed = {}
        for key, value in features.items():
            renamed[rename_map.get(key, key)] = value
        new_info["features"] = renamed
    return new_info


def transform_stats(stats: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    new_stats = copy.deepcopy(stats or {})
    for feature_name, entry in plan["drop_feature_keys"].items():
        if feature_name in new_stats:
            new_stats[feature_name] = reindex_stats_entry(new_stats[feature_name], entry["keep_indices"])
    for feature_name, entry in plan["reorder_feature_keys"].items():
        if feature_name in new_stats:
            new_stats[feature_name] = reindex_stats_entry(new_stats[feature_name], entry["order_indices"])
    for feature_name in plan["drop_features"]:
        new_stats.pop(feature_name, None)
    rename_map = plan["rename_features"]
    if rename_map:
        renamed = {}
        for key, value in new_stats.items():
            renamed[rename_map.get(key, key)] = value
        new_stats = renamed
    return new_stats


def validate_stats_for_key_edit(stats: dict[str, Any], feature_name: str, old_dim: int, errors: list[str]) -> None:
    entry = (stats or {}).get(feature_name)
    if entry is None:
        errors.append(f"stats.json is missing stats for key-edited feature: {feature_name}")
        return
    for stat_name, value in entry.items():
        if stat_name == STATS_COUNT_NAME:
            continue
        if isinstance(value, list) and len(value) != old_dim:
            errors.append(f"stats.json {feature_name}/{stat_name} length {len(value)} != {old_dim}")


def reindex_stats_entry(entry: dict[str, Any], keep_indices: list[int]) -> dict[str, Any]:
    result = copy.deepcopy(entry)
    for stat_name, value in list(result.items()):
        if stat_name == STATS_COUNT_NAME:
            continue
        if isinstance(value, list):
            result[stat_name] = [copy.deepcopy(value[idx]) for idx in keep_indices]
    return result


def vector_key_index_edits(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for feature_name, entry in plan["drop_feature_keys"].items():
        result[feature_name] = {"indices": entry["keep_indices"], "old_shape": entry["old_shape"], "new_shape": entry["new_shape"]}
    for feature_name, entry in plan["reorder_feature_keys"].items():
        result[feature_name] = {"indices": entry["order_indices"], "old_shape": entry["old_shape"], "new_shape": entry["new_shape"]}
    return result


def validate_episode_stats_for_vector_key_edits(shards: list[Path], plan: dict[str, Any], errors: list[str]) -> None:
    vector_edits = vector_key_index_edits(plan)
    for shard in shards:
        table = pq.read_table(shard)
        for feature_name, entry in vector_edits.items():
            prefix = f"stats/{feature_name}/"
            for column_name in table.column_names:
                if not column_name.startswith(prefix):
                    continue
                stat_name = column_name[len(prefix) :]
                if stat_name == STATS_COUNT_NAME:
                    continue
                validate_vector_column(table.column(column_name), entry["old_shape"][0], column_name, shard, errors)


def validate_vector_column(
    column: pa.ChunkedArray,
    expected_dim: int,
    feature_name: str,
    source_path: Path,
    errors: list[str],
) -> None:
    try:
        _ = slice_vector_column(column, expected_dim, list(range(expected_dim)), feature_name, source_path)
    except Exception as exc:
        errors.append(str(exc))


def transform_data_parquet(source_path: Path, destination_path: Path, plan: dict[str, Any]) -> None:
    table = pq.read_table(source_path)
    for feature_name, entry in vector_key_index_edits(plan).items():
        ensure_column_exists(table, feature_name, source_path)
        sliced = slice_vector_column(table.column(feature_name), entry["old_shape"][0], entry["indices"], feature_name, source_path)
        table = table.set_column(table.column_names.index(feature_name), feature_name, sliced)
    drop_features = [name for name in plan["drop_features"] if name in table.column_names]
    if drop_features:
        table = table.drop(drop_features)
    rename_map = plan["rename_features"]
    if rename_map:
        for source in rename_map:
            ensure_column_exists(table, source, source_path)
        table = table.rename_columns([rename_map.get(name, name) for name in table.column_names])
    pq.write_table(table, destination_path, compression="snappy", use_dictionary=True)
    validate_written_data_shard(source_path, destination_path, plan)


def transform_episode_metadata_parquet(source_path: Path, destination_path: Path, plan: dict[str, Any]) -> None:
    table = pq.read_table(source_path)
    for feature_name, entry in vector_key_index_edits(plan).items():
        prefix = f"stats/{feature_name}/"
        for column_name in list(table.column_names):
            if not column_name.startswith(prefix):
                continue
            stat_name = column_name[len(prefix) :]
            if stat_name == STATS_COUNT_NAME:
                continue
            sliced = slice_vector_column(table.column(column_name), entry["old_shape"][0], entry["indices"], column_name, source_path)
            table = table.set_column(table.column_names.index(column_name), column_name, sliced)
    drop_columns: list[str] = []
    for feature_name in plan["drop_features"]:
        prefix = f"stats/{feature_name}/"
        drop_columns.extend([name for name in table.column_names if name.startswith(prefix)])
    if drop_columns:
        table = table.drop(drop_columns)
    rename_map = plan["rename_features"]
    if rename_map:
        names = []
        for name in table.column_names:
            renamed = name
            for source, target in rename_map.items():
                prefix = f"stats/{source}/"
                if name.startswith(prefix):
                    renamed = f"stats/{target}/{name[len(prefix):]}"
                    break
            names.append(renamed)
        table = table.rename_columns(names)
    pq.write_table(table, destination_path, compression="snappy", use_dictionary=True)


def ensure_column_exists(table: pa.Table, column_name: str, source_path: Path) -> None:
    if column_name not in table.column_names:
        raise ValueError(f"{source_path}: missing column {column_name}")


def slice_vector_column(
    column: pa.ChunkedArray,
    expected_dim: int,
    keep_indices: list[int],
    feature_name: str,
    source_path: Path,
) -> pa.Array:
    array = column.combine_chunks()
    if array.null_count:
        raise ValueError(f"{source_path}: {feature_name} contains null rows")
    if pa.types.is_fixed_size_list(array.type):
        if array.type.list_size != expected_dim:
            raise ValueError(f"{source_path}: {feature_name} list_size {array.type.list_size} != {expected_dim}")
        value_type = array.type.value_type
        values = array.values.to_numpy(zero_copy_only=False)
        try:
            matrix = values.reshape(len(array), expected_dim)
        except ValueError as exc:
            raise ValueError(f"{source_path}: {feature_name} cannot reshape to {expected_dim}D") from exc
        sliced = matrix[:, keep_indices]
        flat = pa.array(sliced.reshape(-1), type=value_type)
        return pa.FixedSizeListArray.from_arrays(flat, list_size=len(keep_indices))
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        value_type = array.type.value_type
        rows = []
        for row_index, item in enumerate(array.to_pylist()):
            if item is None:
                raise ValueError(f"{source_path}: {feature_name} row {row_index} is null")
            if len(item) != expected_dim:
                raise ValueError(f"{source_path}: {feature_name} row {row_index} length {len(item)} != {expected_dim}")
            rows.append([item[idx] for idx in keep_indices])
        return pa.array(rows, type=pa.list_(value_type, list_size=len(keep_indices)))
    raise ValueError(f"{source_path}: {feature_name} is not a list vector column: {array.type}")


def validate_written_data_shard(source_path: Path, destination_path: Path, plan: dict[str, Any]) -> None:
    source = pq.read_table(source_path)
    dest = pq.read_table(destination_path)
    if source.num_rows != dest.num_rows:
        raise ValueError(f"{destination_path}: row count changed from {source.num_rows} to {dest.num_rows}")
    replacement_targets = set(plan["rename_features"].values())
    for feature_name in plan["drop_features"]:
        if feature_name in replacement_targets:
            continue
        if feature_name in dest.column_names:
            raise ValueError(f"{destination_path}: dropped feature still exists: {feature_name}")
    for source_name, target_name in plan["rename_features"].items():
        if source_name in dest.column_names:
            raise ValueError(f"{destination_path}: renamed source still exists: {source_name}")
        if target_name not in dest.column_names:
            raise ValueError(f"{destination_path}: renamed target missing: {target_name}")
    for feature_name, entry in vector_key_index_edits(plan).items():
        errors: list[str] = []
        validate_vector_column(dest.column(feature_name), entry["new_shape"][0], feature_name, destination_path, errors)
        if errors:
            raise ValueError("; ".join(errors))


def count_video_files(root: Path) -> int:
    videos = root / "videos"
    return len([path for path in videos.glob("**/*") if path.is_file()]) if videos.exists() else 0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json_value(payload), ensure_ascii=False, indent=2), encoding="utf-8")


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


def dedupe_keep_order(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
