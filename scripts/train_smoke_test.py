from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.profile_store import PROFILE_DIR, delete_profile, load_profile
from app.trainers.lerobot_train import LeRobotTrainFramework
from app.training_executor import (
    get_training_job,
    list_training_jobs,
    submit_training_job,
)
from app.training_pipeline import create_pipeline, get_training_pipeline
from app.training_store import (
    JOB_DIR,
    RECIPE_DIR,
    create_recipe,
    delete_job_files,
    delete_recipe,
    load_recipe,
    read_job_log,
    update_recipe,
)


TMP_ROOT = APP_ROOT / "state" / "_train_smoke_tmp"


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def cleanup_id(recipe_id: str) -> None:
    try:
        delete_recipe(recipe_id)
    except Exception:
        pass
    for path in JOB_DIR.glob("job_*.json") if JOB_DIR.exists() else []:
        try:
            if recipe_id in path.read_text(encoding="utf-8"):
                delete_job_files(path.stem)
        except Exception:
            pass
    if PROFILE_DIR.exists():
        for path in PROFILE_DIR.glob(f"{recipe_id}*.json"):
            try:
                delete_profile(path.stem)
            except Exception:
                pass


def cleanup_all() -> None:
    for recipe_id in [
        "smoke_train_recipe_crud",
        "smoke_train_success",
        "smoke_train_failure",
        "smoke_train_pipeline",
    ]:
        cleanup_id(recipe_id)
    pipeline_dir = APP_ROOT / "state" / "training_pipelines"
    if pipeline_dir.exists():
        for path in pipeline_dir.glob("pipe_*.json"):
            try:
                if "smoke_train_" in path.read_text(encoding="utf-8"):
                    path.unlink()
            except Exception:
                pass


def create_mock_recipe(
    recipe_id: str,
    epochs: int = 3,
    fail: bool = False,
    launcher: str = "direct",
    gpu_devices: str = "",
    num_processes: int = 1,
    env_vars: dict | None = None,
) -> dict:
    cleanup_id(recipe_id)
    output_dir = TMP_ROOT / recipe_id / "checkpoint"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return create_recipe(
        {
            "id": recipe_id,
            "name": f"Mock {recipe_id}",
            "framework": "mock",
            "dataset_path": str(TMP_ROOT),
            "output_dir": str(output_dir),
            "hyperparams": {
                "policy_type": "act",
                "epochs": epochs,
                "sleep_ms": 5,
                "fail": fail,
            },
            "device": "cpu",
            "launcher": launcher,
            "gpu_devices": gpu_devices,
            "num_processes": num_processes,
            "env_vars": env_vars or {},
            "profile_adapter": "mock",
            "auto_profile_on_complete": True,
        }
    )


def wait_job(job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    latest = get_training_job(job_id)
    while time.time() < deadline:
        latest = get_training_job(job_id)
        if latest["status"] in {"done", "failed", "cancelled"}:
            return latest
        time.sleep(0.05)
    raise TimeoutError(f"job did not finish: {job_id}, latest={latest}")


def check_recipe_crud() -> None:
    recipe_id = "smoke_train_recipe_crud"
    recipe = create_mock_recipe(recipe_id, epochs=1)
    assert_equal(recipe["id"], recipe_id, "created recipe id")
    updated = update_recipe(recipe_id, {"name": "Updated mock recipe"})
    assert_equal(updated["name"], "Updated mock recipe", "updated recipe name")
    loaded = load_recipe(recipe_id)
    assert_equal(loaded["framework"], "mock", "loaded recipe framework")
    updated = update_recipe(
        recipe_id,
        {
            "launcher": "accelerate",
            "gpu_devices": "0,1",
            "num_processes": 2,
            "env_vars": {"LEROBOT_VIEWER_TRAIN_TEST": "crud"},
        },
    )
    assert_equal(updated["launcher"], "accelerate", "updated launcher")
    assert_equal(updated["gpu_devices"], "0,1", "updated gpu devices")
    assert_equal(updated["num_processes"], 2, "updated process count")
    assert_equal(updated["env_vars"]["LEROBOT_VIEWER_TRAIN_TEST"], "crud", "updated env vars")
    delete_recipe(recipe_id)


def check_training_job_success() -> None:
    recipe_id = "smoke_train_success"
    recipe = create_mock_recipe(
        recipe_id,
        epochs=3,
        gpu_devices="0,1",
        num_processes=2,
        env_vars={"LEROBOT_VIEWER_TRAIN_TEST": "visible"},
    )
    job = submit_training_job(recipe["id"])
    latest = wait_job(job["job_id"])
    assert_equal(latest["status"], "done", "training job done")
    assert_equal(latest["progress"]["epoch"], 3, "training progress epoch")
    profile_id = latest["auto_generated_profile_id"]
    assert profile_id, "auto generated profile id missing"
    profile = load_profile(profile_id)
    assert_equal(profile["checkpoint_path"], recipe["output_dir"], "profile checkpoint path")
    log = read_job_log(job["job_id"])
    assert "progress" in log["text"], "job log missing progress"
    assert '"cuda_visible_devices": "0,1"' in log["text"], "job log missing CUDA_VISIBLE_DEVICES"
    assert '"train_test_env": "visible"' in log["text"], "job log missing training env var"


def check_lerobot_accelerate_command() -> None:
    framework = LeRobotTrainFramework()
    recipe = {
        "dataset_path": "/data/dataset",
        "output_dir": "/checkpoints/out",
        "device": "cuda",
        "launcher": "accelerate",
        "num_processes": 2,
        "hyperparams": {"policy_type": "act", "batch_size": 32},
    }
    command = framework.build_command(recipe)
    assert_equal(command[:5], ["accelerate", "launch", "--num_processes", "2", "--multi_gpu"], "accelerate command prefix")
    assert_equal(command[5], "lerobot-train", "accelerate wraps lerobot-train")
    assert "device=cuda" in command, "accelerate command keeps cuda device"
    assert "batch_size=32" in command, "accelerate command keeps hyperparams"


def check_training_job_failure() -> None:
    recipe_id = "smoke_train_failure"
    recipe = create_mock_recipe(recipe_id, epochs=1, fail=True)
    job = submit_training_job(recipe["id"])
    latest = wait_job(job["job_id"])
    assert_equal(latest["status"], "failed", "training job failed")
    assert latest["error"], "failed job missing error"


def check_pipeline_create() -> None:
    recipe_id = "smoke_train_pipeline"
    create_mock_recipe(recipe_id, epochs=1)
    pipeline = create_pipeline(
        {
            "recipe_id": recipe_id,
            "comparison_profile_ids": [],
            "episodes": [{"dataset_path": str(TMP_ROOT), "episode_index": 0}],
        }
    )
    loaded = get_training_pipeline(pipeline["pipeline_id"])
    assert_equal(loaded["recipe_id"], recipe_id, "pipeline recipe id")
    latest = wait_job(loaded["training_job_id"])
    assert_equal(latest["status"], "done", "pipeline training job done")


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    RECIPE_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_all()
    try:
        check_recipe_crud()
        print("ok: training recipe crud passed")
        check_training_job_success()
        print("ok: training job success passed")
        check_lerobot_accelerate_command()
        print("ok: lerobot accelerate command passed")
        check_training_job_failure()
        print("ok: training job failure passed")
        check_pipeline_create()
        print("ok: training pipeline passed")
    finally:
        cleanup_all()
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)


if __name__ == "__main__":
    main()
