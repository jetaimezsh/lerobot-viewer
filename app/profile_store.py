from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from app.model_templates import template_by_id


APP_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = APP_ROOT / "state" / "model_profiles"
PROFILE_ID_RE = re.compile(r"^[a-z0-9_-]+$")


def validate_profile_id(profile_id: str) -> str:
    normalized = profile_id.strip()
    if not normalized or not PROFILE_ID_RE.fullmatch(normalized):
        raise ValueError("profile id must match ^[a-z0-9_-]+$")
    return normalized


def profile_path(profile_id: str) -> Path:
    return PROFILE_DIR / f"{validate_profile_id(profile_id)}.json"


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def list_profiles() -> list[dict[str, Any]]:
    if not PROFILE_DIR.exists():
        return []
    profiles = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            profiles.append(public_profile(load_profile(path.stem)))
        except Exception:
            continue
    return profiles


def load_profile(profile_id: str) -> dict[str, Any]:
    path = profile_path(profile_id)
    if not path.exists():
        raise KeyError(f"profile not found: {profile_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_profile(profile: dict[str, Any]) -> dict[str, Any]:
    profile_id = validate_profile_id(str(profile.get("id", "")))
    profile = normalize_profile({**profile, "id": profile_id})
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile_path(profile_id).write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def create_profile(payload: dict[str, Any]) -> dict[str, Any]:
    profile_id = validate_profile_id(str(payload.get("id", "")))
    if profile_path(profile_id).exists():
        raise ValueError(f"profile already exists: {profile_id}")
    template = template_by_id(payload.get("template_id"))
    created_at = now_text()
    profile = {
        "id": profile_id,
        "name": payload.get("name") or profile_id,
        "description": payload.get("description") or template.get("description", ""),
        "checkpoint_path": payload.get("checkpoint_path") or "",
        "adapter": payload.get("adapter") or template.get("adapter") or "lerobot_official",
        "device": payload.get("device") or "cuda",
        "checkpoint_config": {},
        "runtime_params": {**template.get("runtime_params", {}), **(payload.get("runtime_params") or {})},
        "extra_params": payload.get("extra_params") or {},
        "created_at": created_at,
        "updated_at": created_at,
        "inspection": {},
        "status": "created",
    }
    return save_profile(profile)


def update_profile(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = load_profile(profile_id)
    immutable = {"id", "created_at"}
    for key, value in payload.items():
        if key in immutable or value is None:
            continue
        current[key] = value
    current["updated_at"] = now_text()
    return save_profile(current)


def delete_profile(profile_id: str) -> dict[str, Any]:
    path = profile_path(profile_id)
    if not path.exists():
        raise KeyError(f"profile not found: {profile_id}")
    path.unlink()
    return {"ok": True, "deleted": profile_id}


def normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    profile.setdefault("description", "")
    profile.setdefault("checkpoint_path", "")
    profile.setdefault("adapter", "lerobot_official")
    profile.setdefault("device", "cuda")
    profile.setdefault("checkpoint_config", {})
    profile.setdefault("runtime_params", {})
    profile.setdefault("extra_params", {})
    profile.setdefault("inspection", {})
    profile.setdefault("status", "created")
    profile.setdefault("created_at", now_text())
    profile.setdefault("updated_at", profile["created_at"])
    return profile


def public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": profile.get("id"),
        "name": profile.get("name"),
        "description": profile.get("description", ""),
        "checkpoint_path": profile.get("checkpoint_path", ""),
        "adapter": profile.get("adapter", ""),
        "device": profile.get("device", ""),
        "status": profile.get("status", "created"),
        "loaded": bool(profile.get("loaded")),
        "updated_at": profile.get("updated_at"),
        "created_at": profile.get("created_at"),
        "runtime_params": profile.get("runtime_params", {}),
        "extra_params": profile.get("extra_params", {}),
        "checkpoint_config": profile.get("checkpoint_config", {}),
        "inspection": profile.get("inspection", {}),
    }


def profile_snapshot(profile: dict[str, Any]) -> dict[str, Any]:
    snapshot = public_profile(profile)
    snapshot.pop("loaded", None)
    return snapshot
