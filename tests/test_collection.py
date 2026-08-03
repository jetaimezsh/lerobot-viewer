from __future__ import annotations

from pathlib import Path

import pytest

from app import collection_store
from app.collection import (
    CollectionTaskCreateRequest,
    create_collection_task,
    delete_collection_task,
    end_collection_episode,
    finish_collection_task,
    start_collection_episode,
)


def sample_collection_request(tmp_path: Path, episodes: int = 1) -> CollectionTaskCreateRequest:
    return CollectionTaskCreateRequest.model_validate(
        {
            "dataset_name": "collect_demo",
            "repo_id": "local/collect_demo",
            "output_parent": str(tmp_path),
            "episodes": episodes,
            "task": "pick the object",
            "fps": 30,
            "transport": {
                "type": "zmq_sub",
                "endpoint": "tcp://127.0.0.1:5555",
                "topics": ["/camera/a"],
            },
            "streams": [
                {
                    "source": "/camera/a",
                    "feature": "observation.images.camera_a",
                    "kind": "image",
                    "encoding": "jpeg",
                    "dtype": "uint8",
                    "shape": [480, 640, 3],
                    "hz": 30,
                    "extractor": {"type": "encoded_image"},
                },
                {
                    "source": "/force",
                    "feature": "observation.force",
                    "kind": "vector",
                    "encoding": "json",
                    "dtype": "float32",
                    "shape": [3],
                    "names": ["x", "y", "z"],
                    "hz": 100,
                    "extractor": {"type": "json_fields", "fields": ["x", "y", "z"]},
                },
            ],
            "alignment": {
                "reference_source": "/camera/a",
                "reference_feature": "observation.images.camera_a",
                "max_slop_ms": 15,
            },
        }
    )


def test_collection_request_adds_stream_topics(tmp_path: Path) -> None:
    request = sample_collection_request(tmp_path)

    assert request.transport.topics == ["/camera/a", "/force"]


def test_collection_request_derives_topics_when_transport_topics_empty(tmp_path: Path) -> None:
    data = sample_collection_request(tmp_path).model_dump()
    data["transport"]["topics"] = []
    request = CollectionTaskCreateRequest.model_validate(data)

    assert request.transport.topics == ["/camera/a", "/force"]


def test_collection_request_rejects_unknown_reference(tmp_path: Path) -> None:
    data = sample_collection_request(tmp_path).model_dump()
    data["alignment"]["reference_source"] = "/missing"

    with pytest.raises(ValueError, match="reference_source"):
        CollectionTaskCreateRequest.model_validate(data)


def test_collection_task_state_machine(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(collection_store, "TASK_DIR", tmp_path / "tasks")
    task = create_collection_task(sample_collection_request(tmp_path))

    assert task["status"] == "ready"
    task = start_collection_episode(task["task_id"])
    assert task["status"] == "recording"
    task = end_collection_episode(task["task_id"])
    assert task["status"] == "ready_to_finish"
    assert task["current_episode"] == 1
    assert task["recorded_episodes"][0]["status"] == "pending_writer"
    task = finish_collection_task(task["task_id"])
    assert task["status"] == "completed"


def test_delete_collection_task_removes_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(collection_store, "TASK_DIR", tmp_path / "tasks")
    task = create_collection_task(sample_collection_request(tmp_path))

    deleted = delete_collection_task(task["task_id"])

    assert deleted["task_id"] == task["task_id"]
    with pytest.raises(KeyError):
        collection_store.load_task(task["task_id"])
