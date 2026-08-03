from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.collection_store import (
    append_task_log,
    delete_task,
    list_tasks as store_list_tasks,
    load_task,
    now_text,
    read_task_log,
    save_task,
)
from app.operation_log import log_operation


StreamKind = Literal["image", "vector", "scalar", "bool"]
TransportType = Literal["zmq_sub", "zmq_pull"]
ExtractorType = Literal["array", "json_field", "json_fields", "topics", "encoded_image", "raw_image"]
TimestampSource = Literal["message", "receive_time"]
MissingPolicy = Literal["error", "drop_frame", "hold_previous"]
TrimPolicy = Literal["intersection", "reference_full"]
AlignmentPolicy = Literal["nearest", "linear"]


class CollectionTransportConfig(BaseModel):
    type: TransportType = "zmq_sub"
    endpoint: str
    topics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_transport(self) -> "CollectionTransportConfig":
        self.endpoint = self.endpoint.strip()
        if not self.endpoint:
            raise ValueError("ZMQ endpoint is required")
        self.topics = [topic.strip() for topic in self.topics if topic.strip()]
        return self


class CollectionExtractorConfig(BaseModel):
    type: ExtractorType = "array"
    field: str | None = None
    fields: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_extractor(self) -> "CollectionExtractorConfig":
        if self.field is not None:
            self.field = self.field.strip() or None
        self.fields = [field.strip() for field in self.fields if field.strip()]
        self.sources = [source.strip() for source in self.sources if source.strip()]
        if self.type == "json_field" and not self.field:
            raise ValueError("json_field extractor requires field")
        if self.type == "json_fields" and not self.fields:
            raise ValueError("json_fields extractor requires fields")
        if self.type == "topics" and not self.sources:
            raise ValueError("topics extractor requires sources")
        return self


class CollectionStreamConfig(BaseModel):
    enabled: bool = True
    source: str
    feature: str
    kind: StreamKind
    encoding: str = "array"
    dtype: str = "float32"
    shape: list[int] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)
    hz: float | None = Field(default=None, gt=0)
    required: bool = True
    extractor: CollectionExtractorConfig = Field(default_factory=CollectionExtractorConfig)
    max_slop_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_stream(self) -> "CollectionStreamConfig":
        self.source = self.source.strip()
        self.feature = self.feature.strip()
        self.encoding = self.encoding.strip()
        self.dtype = self.dtype.strip()
        self.names = [name.strip() for name in self.names if name.strip()]
        if not self.source:
            raise ValueError("stream source is required")
        if not self.feature:
            raise ValueError("stream feature is required")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"{self.feature} shape must contain positive integers")
        if self.kind == "image":
            if len(self.shape) not in (2, 3):
                raise ValueError(f"{self.feature} image stream requires H,W or H,W,C shape")
            if self.extractor.type not in {"encoded_image", "raw_image"}:
                raise ValueError(f"{self.feature} image stream requires encoded_image or raw_image extractor")
        if self.kind == "vector":
            if len(self.shape) != 1:
                raise ValueError(f"{self.feature} vector stream requires one-dimensional shape")
            if self.names and len(self.names) != self.shape[0]:
                raise ValueError(f"{self.feature} names length must match shape")
        if self.kind in {"scalar", "bool"} and self.shape not in ([], [1]):
            raise ValueError(f"{self.feature} scalar/bool stream shape must be empty or [1]")
        return self


class CollectionAlignmentConfig(BaseModel):
    mode: Literal["reference_stream"] = "reference_stream"
    reference_source: str
    reference_feature: str | None = None
    timestamp_source: TimestampSource = "message"
    max_slop_ms: float = Field(default=15, ge=0)
    trim: TrimPolicy = "intersection"
    image_policy: AlignmentPolicy = "nearest"
    numeric_policy: AlignmentPolicy = "nearest"
    missing_policy: MissingPolicy = "error"

    @model_validator(mode="after")
    def validate_alignment(self) -> "CollectionAlignmentConfig":
        self.reference_source = self.reference_source.strip()
        if not self.reference_source:
            raise ValueError("reference_source is required")
        if self.reference_feature is not None:
            self.reference_feature = self.reference_feature.strip() or None
        if self.image_policy != "nearest":
            raise ValueError("image_policy currently supports nearest only")
        return self


class CollectionTaskCreateRequest(BaseModel):
    dataset_name: str
    repo_id: str
    output_parent: str
    output_path: str | None = None
    episodes: int = Field(ge=1)
    task: str
    fps: float = Field(gt=0)
    overwrite: bool = False
    transport: CollectionTransportConfig
    streams: list[CollectionStreamConfig] = Field(min_length=1)
    alignment: CollectionAlignmentConfig

    @model_validator(mode="after")
    def validate_task(self) -> "CollectionTaskCreateRequest":
        self.dataset_name = self.dataset_name.strip()
        self.repo_id = self.repo_id.strip()
        self.output_parent = self.output_parent.strip()
        self.output_path = self.output_path.strip() if self.output_path else None
        self.task = self.task.strip()
        if not self.dataset_name:
            raise ValueError("dataset_name is required")
        if not self.repo_id:
            raise ValueError("repo_id is required")
        if not self.output_parent:
            raise ValueError("output_parent is required")
        if not self.task:
            raise ValueError("task is required")
        enabled = [stream for stream in self.streams if stream.enabled]
        if not enabled:
            raise ValueError("at least one enabled stream is required")
        features = [stream.feature for stream in enabled]
        if len(features) != len(set(features)):
            raise ValueError("stream feature values must be unique")
        source_or_features = {stream.source for stream in enabled} | {stream.feature for stream in enabled}
        if self.alignment.reference_source not in source_or_features:
            raise ValueError("reference_source must match an enabled stream source or feature")
        subscribed_topics = set(self.transport.topics)
        for stream in enabled:
            if self.transport.type == "zmq_sub":
                for source in stream_sources(stream):
                    subscribed_topics.add(source)
        self.transport.topics = sorted(subscribed_topics)
        return self


def stream_sources(stream: CollectionStreamConfig) -> list[str]:
    if stream.extractor.type == "topics":
        return stream.extractor.sources
    return [stream.source]


def collection_runtime_status() -> dict[str, Any]:
    return {
        "pyzmq_available": importlib.util.find_spec("zmq") is not None,
        "supported_transports": ["zmq_sub", "zmq_pull"],
        "message_contract": ["topic", "header_json", "payload_bytes"],
        "status": "configured",
    }


def create_collection_task(request: CollectionTaskCreateRequest) -> dict[str, Any]:
    output_path = request.output_path or str(Path(request.output_parent) / request.dataset_name)
    task_id = unique_task_id()
    task = {
        "task_id": task_id,
        "status": "ready",
        "created_at": now_text(),
        "updated_at": now_text(),
        "started_at": None,
        "finished_at": None,
        "current_episode": 0,
        "recorded_episodes": [],
        "episode_started_at": None,
        "dataset_name": request.dataset_name,
        "repo_id": request.repo_id,
        "output_parent": request.output_parent,
        "output_path": output_path,
        "episodes": request.episodes,
        "task": request.task,
        "fps": request.fps,
        "overwrite": request.overwrite,
        "config": request.model_dump(),
        "stream_status": initial_stream_status(request),
        "last_error": None,
    }
    save_task(task)
    append_task_log(task_id, f"{now_text()} created collection task")
    log_operation("collection_task_create", "success", target=task_id, details={"repo_id": request.repo_id, "output_path": output_path})
    return public_collection_task(task)


def list_collection_tasks() -> list[dict[str, Any]]:
    return [public_collection_task(task) for task in store_list_tasks()]


def get_collection_task(task_id: str) -> dict[str, Any]:
    return public_collection_task(load_task(task_id))


def delete_collection_task(task_id: str) -> dict[str, Any]:
    task = delete_task(task_id)
    log_operation("collection_task_delete", "success", target=task_id, details={"status": task.get("status"), "output_path": task.get("output_path")})
    return public_collection_task(task)


def start_collection_episode(task_id: str) -> dict[str, Any]:
    task = load_task(task_id)
    if task.get("status") not in {"ready", "episode_complete"}:
        raise ValueError(f"cannot start episode from status {task.get('status')}")
    if int(task.get("current_episode", 0)) >= int(task.get("episodes", 0)):
        raise ValueError("all configured episodes are already recorded")
    task["status"] = "recording"
    task["started_at"] = task.get("started_at") or now_text()
    task["episode_started_at"] = now_text()
    task["updated_at"] = now_text()
    task["last_error"] = None
    save_task(task)
    append_task_log(task_id, f"{now_text()} started episode {task['current_episode']}")
    log_operation("collection_episode_start", "success", target=task_id, details={"episode": task.get("current_episode")})
    return public_collection_task(task)


def end_collection_episode(task_id: str) -> dict[str, Any]:
    task = load_task(task_id)
    if task.get("status") != "recording":
        raise ValueError(f"cannot end episode from status {task.get('status')}")
    episode_index = int(task.get("current_episode", 0))
    recorded = list(task.get("recorded_episodes") or [])
    recorded.append(
        {
            "episode_index": episode_index,
            "status": "pending_writer",
            "started_at": task.get("episode_started_at"),
            "finished_at": now_text(),
            "alignment": task.get("config", {}).get("alignment", {}),
            "note": "ZMQ listener and LeRobot writer will attach buffered frame counts here",
        }
    )
    task["recorded_episodes"] = recorded
    task["current_episode"] = episode_index + 1
    task["episode_started_at"] = None
    task["updated_at"] = now_text()
    task["status"] = "episode_complete" if task["current_episode"] < int(task.get("episodes", 0)) else "ready_to_finish"
    save_task(task)
    append_task_log(task_id, f"{now_text()} ended episode {episode_index}")
    log_operation("collection_episode_end", "success", target=task_id, details={"episode": episode_index})
    return public_collection_task(task)


def retry_collection_episode(task_id: str) -> dict[str, Any]:
    task = load_task(task_id)
    if task.get("status") == "completed":
        raise ValueError("completed collection tasks cannot be retried")
    if task.get("status") == "recording":
        task["episode_started_at"] = None
    task["status"] = "ready" if not task.get("recorded_episodes") else "episode_complete"
    task["updated_at"] = now_text()
    task["last_error"] = None
    save_task(task)
    append_task_log(task_id, f"{now_text()} reset current episode {task.get('current_episode')}")
    log_operation("collection_episode_retry", "success", target=task_id, details={"episode": task.get("current_episode")})
    return public_collection_task(task)


def finish_collection_task(task_id: str) -> dict[str, Any]:
    task = load_task(task_id)
    if task.get("status") == "recording":
        raise ValueError("end the current episode before finishing the dataset")
    if len(task.get("recorded_episodes") or []) < int(task.get("episodes", 0)):
        raise ValueError("recorded episode count is lower than configured episodes")
    task["status"] = "completed"
    task["finished_at"] = now_text()
    task["updated_at"] = now_text()
    save_task(task)
    append_task_log(task_id, f"{now_text()} completed collection task")
    log_operation("collection_task_finish", "success", target=task_id)
    return public_collection_task(task)


def collection_task_log(task_id: str, tail: int = 200) -> dict[str, Any]:
    load_task(task_id)
    return read_task_log(task_id, tail=tail)


def public_collection_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "status": task.get("status"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "current_episode": task.get("current_episode", 0),
        "recorded_episodes": task.get("recorded_episodes", []),
        "dataset_name": task.get("dataset_name"),
        "repo_id": task.get("repo_id"),
        "output_parent": task.get("output_parent"),
        "output_path": task.get("output_path"),
        "episodes": task.get("episodes"),
        "task": task.get("task"),
        "fps": task.get("fps"),
        "overwrite": task.get("overwrite"),
        "config": task.get("config"),
        "stream_status": task.get("stream_status", []),
        "last_error": task.get("last_error"),
    }


def initial_stream_status(request: CollectionTaskCreateRequest) -> list[dict[str, Any]]:
    rows = []
    for stream in request.streams:
        if not stream.enabled:
            continue
        rows.append(
            {
                "source": stream.source,
                "feature": stream.feature,
                "kind": stream.kind,
                "status": "waiting",
                "samples": 0,
                "actual_hz": None,
                "last_timestamp": None,
                "latency_ms": None,
                "preview": None,
            }
        )
    return rows


def unique_task_id() -> str:
    return f"collect_{uuid.uuid4().hex[:12]}"
