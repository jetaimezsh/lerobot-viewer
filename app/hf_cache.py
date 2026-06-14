from __future__ import annotations

import os
import tempfile
import threading
import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_HF_CACHE_ENV_KEYS = [
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "HF_MODULES_CACHE",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
]
_HF_CACHE_ENV_LOCK = threading.RLock()


@contextmanager
def local_hf_cache_env() -> Iterator[None]:
    with _HF_CACHE_ENV_LOCK:
        previous = {key: os.environ.get(key) for key in _HF_CACHE_ENV_KEYS}
        cache_root = Path(tempfile.gettempdir()) / "lerobot-viewer-hf-home"
        datasets_cache = Path(tempfile.gettempdir()) / "lerobot-viewer-hf-datasets"
        modules_cache = cache_root / "modules"
        datasets_config = _datasets_config()
        previous_datasets_config = _snapshot_datasets_config(datasets_config)
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            datasets_cache.mkdir(parents=True, exist_ok=True)
            modules_cache.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache_root)
            os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
            os.environ["HF_MODULES_CACHE"] = str(modules_cache)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            _apply_datasets_config(datasets_config, cache_root, datasets_cache, modules_cache)
            yield
        finally:
            _restore_datasets_config(datasets_config, previous_datasets_config)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


writable_hf_cache_env = local_hf_cache_env


def _datasets_config() -> object | None:
    try:
        return importlib.import_module("datasets.config")
    except Exception:
        return None


def _snapshot_datasets_config(config: object | None) -> dict[str, object]:
    if config is None:
        return {}
    names = ("HF_CACHE_HOME", "HF_DATASETS_CACHE", "HF_MODULES_CACHE")
    return {name: getattr(config, name) for name in names if hasattr(config, name)}


def _apply_datasets_config(
    config: object | None,
    cache_root: Path,
    datasets_cache: Path,
    modules_cache: Path,
) -> None:
    if config is None:
        return
    updates = {
        "HF_CACHE_HOME": str(cache_root),
        "HF_DATASETS_CACHE": str(datasets_cache),
        "HF_MODULES_CACHE": str(modules_cache),
    }
    for name, value in updates.items():
        if hasattr(config, name):
            setattr(config, name, value)


def _restore_datasets_config(config: object | None, previous: dict[str, object]) -> None:
    if config is None:
        return
    for name, value in previous.items():
        setattr(config, name, value)
