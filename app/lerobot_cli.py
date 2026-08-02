from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


RECOMPUTE_FIELDS = {
    "skip_image_video": "--operation.skip_image_video",
    "relative_action": "--operation.relative_action",
    "relative_exclude_joints": "--operation.relative_exclude_joints",
    "chunk_size": "--operation.chunk_size",
    "num_workers": "--operation.num_workers",
    "overwrite": "--operation.overwrite",
}


class RecomputeStatsRequest(BaseModel):
    repo_id: str
    root: str
    new_repo_id: str | None = None
    new_root: str | None = None
    skip_image_video: bool = True
    relative_action: bool = False
    relative_exclude_joints: list[str] = Field(default_factory=list)
    chunk_size: int = Field(default=50, ge=1)
    num_workers: int = Field(default=0, ge=0)
    overwrite: bool = False
    executable: str | None = None


def lerobot_cli_status(executable: str | None = None) -> dict[str, Any]:
    exe = resolve_lerobot_edit_dataset(executable)
    if not exe:
        return {
            "available": False,
            "executable": executable,
            "lerobot_version": lerobot_version(),
            "supports_recompute_stats": False,
            "supported_fields": [],
            "help_error": "lerobot-edit-dataset was not found",
        }
    try:
        result = subprocess.run(
            [exe, "--help"],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return {
            "available": False,
            "executable": exe,
            "lerobot_version": lerobot_version(),
            "supports_recompute_stats": False,
            "supported_fields": [],
            "help_error": str(exc),
        }
    help_text = (result.stdout or "") + "\n" + (result.stderr or "")
    supported_fields = [field for field, flag in RECOMPUTE_FIELDS.items() if flag in help_text]
    return {
        "available": result.returncode == 0,
        "executable": exe,
        "lerobot_version": lerobot_version(),
        "supports_recompute_stats": "recompute_stats" in help_text,
        "supported_fields": supported_fields,
        "help_error": None if result.returncode == 0 else help_text[-4000:],
    }


def build_recompute_stats_command(request: RecomputeStatsRequest) -> list[str]:
    exe = resolve_lerobot_edit_dataset(request.executable) or (request.executable or "lerobot-edit-dataset")
    repo_id = request.repo_id.strip()
    root = str(Path(request.root).expanduser())
    new_repo_id = (request.new_repo_id or f"{repo_id}_recomputed_stats").strip()
    new_root = str(Path(request.new_root).expanduser()) if request.new_root else str(Path(root).parent / f"{Path(root).name}_recomputed_stats")
    args = [
        exe,
        "--repo_id",
        repo_id,
        "--root",
        root,
        "--new_repo_id",
        new_repo_id,
        "--new_root",
        new_root,
        "--push_to_hub",
        "false",
        "--operation.type",
        "recompute_stats",
        "--operation.skip_image_video",
        bool_arg(request.skip_image_video),
        "--operation.relative_action",
        bool_arg(request.relative_action),
    ]
    if request.relative_action and request.relative_exclude_joints:
        args.extend(["--operation.relative_exclude_joints", json.dumps(request.relative_exclude_joints, ensure_ascii=False)])
    args.extend(
        [
            "--operation.chunk_size",
            str(request.chunk_size),
            "--operation.num_workers",
            str(request.num_workers),
            "--operation.overwrite",
            bool_arg(request.overwrite),
        ]
    )
    return args


def preview_recompute_stats_command(request: RecomputeStatsRequest) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    status = lerobot_cli_status(request.executable)
    if not status.get("available"):
        errors.append(status.get("help_error") or "lerobot-edit-dataset is not available")
    if not status.get("supports_recompute_stats"):
        errors.append("lerobot-edit-dataset does not support recompute_stats")
    missing_fields = [field for field in RECOMPUTE_FIELDS if field not in status.get("supported_fields", [])]
    if missing_fields:
        errors.append(f"lerobot-edit-dataset help does not expose fields: {missing_fields}")
    if not request.repo_id.strip():
        errors.append("repo_id is required")
    root = Path(request.root).expanduser()
    if not request.root.strip():
        errors.append("root is required")
    elif not root.exists() or not root.is_dir():
        errors.append(f"root does not exist or is not a directory: {root}")
    if request.relative_action and root.exists():
        validate_relative_action_request(root, request, errors)
    args = build_recompute_stats_command(request)
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "args": args,
        "display_command": display_command(args),
        "env": status,
    }


def validate_relative_action_request(root: Path, request: RecomputeStatsRequest, errors: list[str]) -> None:
    info_path = root / "meta/info.json"
    episodes_dir = root / "meta/episodes"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"failed to read info.json for relative-action validation: {exc}")
        return
    features = info.get("features") or {}
    for name in ["action", "observation.state"]:
        feature = features.get(name)
        if not isinstance(feature, dict):
            errors.append(f"relative_action requires feature: {name}")
            continue
        if not is_numeric_vector_feature(feature):
            errors.append(f"relative_action requires numeric one-dimensional vector feature: {name}")
    action = features.get("action") or {}
    action_names = action.get("names")
    if request.relative_exclude_joints:
        if not isinstance(action_names, list):
            errors.append("relative_exclude_joints requires action names")
        else:
            missing = [name for name in request.relative_exclude_joints if name not in action_names]
            if missing:
                errors.append(f"relative_exclude_joints not found in action names: {missing}")
    if episodes_dir.exists():
        try:
            import pandas as pd

            frames = [pd.read_parquet(path) for path in episodes_dir.glob("**/*.parquet")]
            max_length = max((int(df["length"].max()) for df in frames if "length" in df.columns), default=0)
            if max_length and max_length < request.chunk_size:
                errors.append(f"longest episode length {max_length} is smaller than chunk_size {request.chunk_size}")
        except Exception as exc:
            errors.append(f"failed to validate episode lengths for relative_action: {exc}")


def resolve_lerobot_edit_dataset(executable: str | None = None) -> str | None:
    if executable:
        path = Path(executable).expanduser()
        if path.exists():
            return str(path)
        found = shutil.which(executable)
        if found:
            return found
        return executable
    candidates = [
        Path(sys.executable).with_name("lerobot-edit-dataset.exe"),
        Path(sys.executable).with_name("lerobot-edit-dataset"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which("lerobot-edit-dataset")


def display_command(args: list[str]) -> str:
    return subprocess.list2cmdline(args)


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def lerobot_version() -> str | None:
    try:
        import lerobot

        return getattr(lerobot, "__version__", "installed")
    except Exception:
        return None


def is_numeric_vector_feature(feature: dict[str, Any]) -> bool:
    dtype = str(feature.get("dtype", "")).lower()
    shape = feature.get("shape")
    return dtype.startswith(("float", "int", "uint", "bool")) and isinstance(shape, list) and len(shape) == 1
