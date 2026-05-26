from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


APP_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = APP_ROOT / "web"
HISTORY_PATH = APP_ROOT / ".viewer_history.json"
MAX_HISTORY_ITEMS = 20

app = FastAPI(title="LeRobot Dataset v3.0 Local Viewer")
app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")


class OpenDatasetRequest(BaseModel):
    path: str


class DatasetCache:
    def __init__(self, root: Path):
        self.root = root
        self.info = self._load_info()
        self.features = self.info.get("features", {})
        self.stats = self._load_json("meta/stats.json", default={})
        self.tasks = self._load_tasks()
        self.episodes = self._load_episodes()
        self.video_keys = [
            key for key, feature in self.features.items() if feature.get("dtype") == "video"
        ]
        self.numeric_keys = [
            key for key, feature in self.features.items() if is_numeric_feature(feature)
        ]

    def _load_json(self, relative_path: str, default: Any | None = None) -> Any:
        path = self.root / relative_path
        if not path.exists():
            if default is not None:
                return default
            raise HTTPException(status_code=400, detail=f"缺少文件: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_info(self) -> dict[str, Any]:
        info = self._load_json("meta/info.json")
        version = str(info.get("codebase_version", ""))
        if version and version != "v3.0":
            raise HTTPException(
                status_code=400,
                detail=f"当前工具只支持 LeRobot codebase_version v3.0，检测到: {version}",
            )
        required = ["features", "fps", "data_path"]
        missing = [key for key in required if key not in info]
        if missing:
            raise HTTPException(status_code=400, detail=f"info.json 缺少字段: {missing}")
        if self._has_video_features(info) and not info.get("video_path"):
            raise HTTPException(status_code=400, detail="info.json 缺少 video_path")
        return info

    @staticmethod
    def _has_video_features(info: dict[str, Any]) -> bool:
        return any(feature.get("dtype") == "video" for feature in info.get("features", {}).values())

    def _load_tasks(self) -> list[dict[str, Any]]:
        path = self.root / "meta/tasks.parquet"
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"严格 v3.0 需要 meta/tasks.parquet: {path}")
        df = pd.read_parquet(path)
        if df.index.name == "task":
            df = df.reset_index()
        return dataframe_records(df)

    def _load_episodes(self) -> pd.DataFrame:
        episode_dir = self.root / "meta/episodes"
        if not episode_dir.exists():
            raise HTTPException(status_code=400, detail=f"缺少目录: {episode_dir}")
        files = sorted(episode_dir.glob("**/*.parquet"))
        if not files:
            raise HTTPException(status_code=400, detail=f"未找到 episode parquet: {episode_dir}")
        frames = [pd.read_parquet(file) for file in files]
        df = pd.concat(frames, ignore_index=True)
        if "episode_index" not in df.columns:
            raise HTTPException(status_code=400, detail="episode metadata 缺少 episode_index")
        df = df.sort_values("episode_index").reset_index(drop=True)
        return df

    def summary(self) -> dict[str, Any]:
        return {
            "id": dataset_id(self.root),
            "root": str(self.root),
            "codebase_version": self.info.get("codebase_version"),
            "robot_type": self.info.get("robot_type"),
            "fps": self.info.get("fps"),
            "total_episodes": int(self.info.get("total_episodes", len(self.episodes))),
            "total_frames": int(self.info.get("total_frames", 0)),
            "total_tasks": int(self.info.get("total_tasks", len(self.tasks))),
            "data_path": self.info.get("data_path"),
            "video_path": self.info.get("video_path"),
            "video_keys": self.video_keys,
            "numeric_keys": self.numeric_keys,
            "features": self.features,
            "tasks": self.tasks[:100],
        }

    def list_episodes(self) -> list[dict[str, Any]]:
        wanted = [
            "episode_index",
            "length",
            "tasks",
            "task_index",
            "dataset_from_index",
            "dataset_to_index",
            "data/chunk_index",
            "data/file_index",
        ]
        available = [column for column in wanted if column in self.episodes.columns]
        records = dataframe_records(self.episodes[available])
        for record in records:
            record["videos"] = self.video_segments_for_record(record["episode_index"])
        return records

    def episode_record(self, episode_index: int) -> pd.Series:
        match = self.episodes[self.episodes["episode_index"] == episode_index]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"未找到 episode: {episode_index}")
        return match.iloc[0]

    def data_file_for_episode(self, episode: pd.Series) -> Path:
        data_path = self.info["data_path"].format(
            chunk_index=int(episode["data/chunk_index"]),
            file_index=int(episode["data/file_index"]),
        )
        path = self.root / data_path
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"未找到数据文件: {path}")
        return path

    def video_file_for_episode(self, episode: pd.Series, video_key: str) -> Path:
        if video_key not in self.video_keys:
            raise HTTPException(status_code=404, detail=f"未知 video key: {video_key}")
        prefix = f"videos/{video_key}"
        chunk_col = f"{prefix}/chunk_index"
        file_col = f"{prefix}/file_index"
        if chunk_col not in episode or file_col not in episode:
            raise HTTPException(status_code=404, detail=f"episode 缺少视频索引: {video_key}")
        video_path = self.info["video_path"].format(
            video_key=video_key,
            chunk_index=int(episode[chunk_col]),
            file_index=int(episode[file_col]),
        )
        path = self.root / video_path
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"未找到视频文件: {path}")
        return path

    def video_segments_for_record(self, episode_index: int) -> list[dict[str, Any]]:
        episode = self.episode_record(int(episode_index))
        segments = []
        for video_key in self.video_keys:
            prefix = f"videos/{video_key}"
            from_col = f"{prefix}/from_timestamp"
            to_col = f"{prefix}/to_timestamp"
            if from_col not in episode or to_col not in episode:
                continue
        segments.append(
                {
                    "key": video_key,
                    "from_timestamp": as_float(episode[from_col]),
                    "to_timestamp": as_float(episode[to_col]),
                    "url": f"/api/datasets/{dataset_id(self.root)}/video?episode_index={episode_index}&video_key={video_key}",
                }
            )
        return segments

    def load_episode_detail(self, episode_index: int) -> dict[str, Any]:
        episode = self.episode_record(episode_index)
        data_path = self.data_file_for_episode(episode)
        df = pd.read_parquet(data_path)
        df = slice_episode_frame_data(df, episode)
        timeline = build_timeline(df, fps=float(self.info["fps"]))
        series = build_numeric_series(df, self.features)
        return {
            "episode": clean_record(episode.to_dict()),
            "data_file": str(data_path),
            "videos": self.video_segments_for_record(episode_index),
            "timeline": timeline,
            "series": series,
        }


DATASETS: dict[str, DatasetCache] = {}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_ROOT / "index.html").read_text(encoding="utf-8")


@app.get("/api/env")
def env_info() -> dict[str, Any]:
    package_names = ["fastapi", "uvicorn", "pandas", "pyarrow", "numpy", "pydantic"]
    packages = {}
    for name in package_names:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "installed")
        except Exception:
            packages[name] = None
    return {
        "python": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
        "venv": sys.prefix != sys.base_prefix,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "requirements": str(APP_ROOT / "requirements.txt"),
        "packages": packages,
    }


@app.post("/api/env/install-requirements")
def install_requirements() -> dict[str, Any]:
    requirements = APP_ROOT / "requirements.txt"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        cwd=APP_ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-8000:],
        "stderr": result.stderr[-8000:],
    }


@app.post("/api/datasets/open")
def open_dataset(request: OpenDatasetRequest) -> dict[str, Any]:
    root = Path(request.path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail=f"数据集目录不存在: {root}")
    cache = DatasetCache(root)
    key = dataset_id(root)
    DATASETS[key] = cache
    save_history_item(cache.summary())
    return cache.summary()


@app.get("/api/history")
def dataset_history() -> list[dict[str, Any]]:
    return load_history()


@app.get("/api/path/suggest")
def suggest_paths(path: str = Query("", description="Partial local path")) -> dict[str, Any]:
    return build_path_suggestions(path)


@app.get("/api/datasets/{key}")
def get_dataset(key: str) -> dict[str, Any]:
    return get_cache(key).summary()


@app.get("/api/datasets/{key}/episodes")
def list_episodes(key: str) -> list[dict[str, Any]]:
    return get_cache(key).list_episodes()


@app.get("/api/datasets/{key}/episodes/{episode_index}")
def episode_detail(key: str, episode_index: int) -> dict[str, Any]:
    return get_cache(key).load_episode_detail(episode_index)


@app.get("/api/datasets/{key}/video")
def video_file(
    key: str,
    episode_index: int = Query(...),
    video_key: str = Query(...),
) -> FileResponse:
    cache = get_cache(key)
    decoded_video_key = unquote(video_key)
    episode = cache.episode_record(episode_index)
    path = cache.video_file_for_episode(episode, decoded_video_key)
    return FileResponse(path, media_type="video/mp4", filename=path.name)


def get_cache(key: str) -> DatasetCache:
    cache = DATASETS.get(key)
    if not cache:
        raise HTTPException(status_code=404, detail="请先加载数据集")
    return cache


def dataset_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            items = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def save_history_item(summary: dict[str, Any]) -> None:
    path = str(summary["root"])
    item = {
        "id": summary["id"],
        "path": path,
        "name": Path(path).name or path,
        "opened_at": datetime.now().isoformat(timespec="seconds"),
        "total_episodes": summary.get("total_episodes"),
        "total_frames": summary.get("total_frames"),
        "video_keys": summary.get("video_keys", []),
    }
    items = [entry for entry in load_history() if entry.get("path") != path]
    items.insert(0, item)
    try:
        with HISTORY_PATH.open("w", encoding="utf-8") as f:
            json.dump(items[:MAX_HISTORY_ITEMS], f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def build_path_suggestions(raw_path: str) -> dict[str, Any]:
    raw_path = raw_path.strip().strip('"')
    if not raw_path:
        return {"base": "", "query": "", "items": list_path_roots()}

    expanded = os.path.expanduser(raw_path)
    is_trailing_separator = expanded.endswith(("\\", "/"))
    path = Path(expanded)

    if path.exists() and path.is_dir() and is_trailing_separator:
        parent = path
        query = ""
    elif path.exists() and path.is_dir():
        parent = path
        query = ""
    else:
        parent = path.parent
        query = path.name.lower()

    if str(parent) == ".":
        parent = Path.cwd()

    if not parent.exists() or not parent.is_dir():
        return {"base": str(parent), "query": query, "items": []}

    items = []
    try:
        children = sorted(
            [child for child in parent.iterdir() if child.is_dir()],
            key=lambda child: child.name.lower(),
        )
    except OSError:
        children = []

    for child in children:
        if query and not child.name.lower().startswith(query):
            continue
        items.append(
            {
                "name": child.name,
                "path": str(child),
                "has_dataset_marker": (child / "meta" / "info.json").exists(),
            }
        )
        if len(items) >= 30:
            break

    return {"base": str(parent), "query": query, "items": items}


def list_path_roots() -> list[dict[str, Any]]:
    if os.name == "nt":
        return list_windows_drives()
    return list_posix_start_points()


def list_windows_drives() -> list[dict[str, Any]]:
    drives = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = Path(f"{letter}:\\")
        if drive.exists():
            drives.append({"name": f"{letter}:\\", "path": str(drive), "has_dataset_marker": False})
    return drives


def list_posix_start_points() -> list[dict[str, Any]]:
    candidates = [Path.home(), APP_ROOT, Path.cwd(), Path("/")]
    items = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(resolved)
        name = str(resolved) if resolved == Path("/") else resolved.name or str(resolved)
        items.append(
            {
                "name": name,
                "path": str(resolved),
                "has_dataset_marker": (resolved / "meta" / "info.json").exists(),
            }
        )
    return items


def is_numeric_feature(feature: dict[str, Any]) -> bool:
    dtype = str(feature.get("dtype", "")).lower()
    if dtype in {"video", "image", "string"}:
        return False
    return dtype.startswith(("float", "int", "uint", "bool"))


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [clean_record(row) for row in df.to_dict(orient="records")]


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in record.items():
        cleaned[key] = clean_value(value)
    return cleaned


def clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return clean_value(value.tolist())
    if isinstance(value, dict):
        return {key: clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    return float(value)


def slice_episode_frame_data(df: pd.DataFrame, episode: pd.Series) -> pd.DataFrame:
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


def build_timeline(df: pd.DataFrame, fps: float) -> dict[str, list[float]]:
    if "timestamp" in df.columns:
        timestamps = [float(value) for value in df["timestamp"].to_numpy()]
    elif "frame_index" in df.columns:
        timestamps = [float(value) / fps for value in df["frame_index"].to_numpy()]
    else:
        timestamps = [index / fps for index in range(len(df))]
    if timestamps:
        offset = timestamps[0]
        elapsed = [round(ts - offset, 6) for ts in timestamps]
    else:
        elapsed = []
    return {"timestamp": timestamps, "elapsed": elapsed}


def build_numeric_series(df: pd.DataFrame, features: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for key, feature in features.items():
        if key not in df.columns or not is_numeric_feature(feature):
            continue
        expanded = expand_numeric_column(key, df[key])
        result.extend(expanded)

    for key in ["index", "frame_index", "episode_index", "task_index"]:
        if key in df.columns and key not in features and pd.api.types.is_numeric_dtype(df[key]):
            result.append({"name": key, "source": key, "values": numeric_list(df[key])})

    return result


def expand_numeric_column(key: str, series: pd.Series) -> list[dict[str, Any]]:
    first = None
    for value in series:
        if value is not None:
            first = value
            break
    if first is None:
        return []

    if is_scalar_number(first):
        return [{"name": key, "source": key, "values": numeric_list(series)}]

    rows = [flatten_numeric(value) for value in series]
    width = max((len(row) for row in rows), default=0)
    expanded = []
    for index in range(width):
        values = [row[index] if index < len(row) else None for row in rows]
        expanded.append({"name": f"{key}[{index}]", "source": key, "values": values})
    return expanded


def flatten_numeric(value: Any) -> list[float | None]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [as_nullable_float(item) for item in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [as_nullable_float(item) for item in np.array(value, dtype=object).reshape(-1)]
    if is_scalar_number(value):
        return [as_nullable_float(value)]
    return []


def numeric_list(series: pd.Series) -> list[float | None]:
    return [as_nullable_float(value) for value in series]


def is_scalar_number(value: Any) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    return isinstance(value, (int, float, bool)) and not isinstance(value, str)


def as_nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except ValueError:
        return None
    return float(value)
