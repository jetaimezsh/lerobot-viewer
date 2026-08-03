from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import collection_store
from app.main import app


def test_collection_api_create_and_control(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(collection_store, "TASK_DIR", tmp_path / "tasks")
    client = TestClient(app)
    response = client.post(
        "/api/collection/tasks",
        json={
            "dataset_name": "collect_demo",
            "repo_id": "local/collect_demo",
            "output_parent": str(tmp_path),
            "episodes": 1,
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
                }
            ],
            "alignment": {"reference_source": "/camera/a", "max_slop_ms": 15},
        },
    )
    assert response.status_code == 200
    task = response.json()
    assert task["status"] == "ready"

    started = client.post(f"/api/collection/tasks/{task['task_id']}/start-episode")
    assert started.status_code == 200
    assert started.json()["status"] == "recording"

    ended = client.post(f"/api/collection/tasks/{task['task_id']}/end-episode")
    assert ended.status_code == 200
    assert ended.json()["status"] == "ready_to_finish"

    finished = client.post(f"/api/collection/tasks/{task['task_id']}/finish")
    assert finished.status_code == 200
    assert finished.json()["status"] == "completed"


def test_collection_api_delete_task(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(collection_store, "TASK_DIR", tmp_path / "tasks")
    client = TestClient(app)
    response = client.post(
        "/api/collection/tasks",
        json={
            "dataset_name": "collect_delete",
            "repo_id": "local/collect_delete",
            "output_parent": str(tmp_path),
            "episodes": 1,
            "task": "pick the object",
            "fps": 30,
            "transport": {"type": "zmq_sub", "endpoint": "tcp://127.0.0.1:5555", "topics": []},
            "streams": [
                {
                    "source": "/camera/a",
                    "feature": "observation.images.camera_a",
                    "kind": "image",
                    "encoding": "jpeg",
                    "dtype": "uint8",
                    "shape": [480, 640, 3],
                    "extractor": {"type": "encoded_image"},
                }
            ],
            "alignment": {"reference_source": "/camera/a"},
        },
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    deleted = client.delete(f"/api/collection/tasks/{task_id}")
    assert deleted.status_code == 200
    assert deleted.json()["task_id"] == task_id

    missing = client.get(f"/api/collection/tasks/{task_id}")
    assert missing.status_code == 404
