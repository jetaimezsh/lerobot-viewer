from __future__ import annotations

import csv
import html
import json
import re
import zipfile
from io import StringIO
from io import BytesIO
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = APP_ROOT / "state"
BACKTEST_DIR = STATE_DIR / "backtests"
BACKTEST_JOB_DIR = STATE_DIR / "backtest_jobs"


def save_backtest_run(run: dict[str, Any]) -> dict[str, Any]:
    run_id = str(run["run_id"])
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    path = run_path(run_id)
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return run


def load_backtest_run(run_id: str) -> dict[str, Any]:
    path = run_path(run_id)
    if not path.exists():
        raise KeyError(f"backtest run not found: {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_backtest_job_record(job: dict[str, Any]) -> dict[str, Any]:
    job_id = str(job["job_id"])
    BACKTEST_JOB_DIR.mkdir(parents=True, exist_ok=True)
    job_path(job_id).write_text(json.dumps(job, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return job


def load_backtest_job_record(job_id: str) -> dict[str, Any]:
    path = job_path(job_id)
    if not path.exists():
        raise KeyError(f"backtest job not found: {job_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_backtest_job_records(limit: int = 200) -> list[dict[str, Any]]:
    if not BACKTEST_JOB_DIR.exists():
        return []
    jobs = []
    for path in sorted(BACKTEST_JOB_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
        if len(jobs) >= limit:
            break
    return jobs


def list_backtest_runs(limit: int = 100) -> list[dict[str, Any]]:
    if not BACKTEST_DIR.exists():
        return []
    runs = []
    for path in sorted(BACKTEST_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        runs.append(public_run_summary(run))
        if len(runs) >= limit:
            break
    return runs


def export_backtest_run(run: dict[str, Any], fmt: str) -> tuple[str, str, str]:
    normalized = fmt.lower()
    if normalized == "json":
        return json.dumps(run, ensure_ascii=False, indent=2), "application/json", f"{run['run_id']}.json"
    if normalized == "csv":
        return export_csv(run), "text/csv; charset=utf-8", f"{run['run_id']}.csv"
    if normalized == "html":
        return export_html(run), "text/html; charset=utf-8", f"{run['run_id']}.html"
    raise ValueError(f"unsupported export format: {fmt}")


def export_action_csv(run: dict[str, Any], result_index: int) -> tuple[str, str, str]:
    item = action_result_at(run, result_index)
    return (
        action_result_csv(run, item),
        "text/csv; charset=utf-8",
        action_result_filename(run, item, result_index),
    )


def export_action_zip(run: dict[str, Any]) -> tuple[bytes, str, str]:
    done = [(index, item) for index, item in enumerate(run.get("results", [])) if action_result_available(item)]
    if not done:
        raise ValueError("backtest run has no action series to export")
    output = BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, item in done:
            filename = unique_zip_name(action_result_filename(run, item, index), used_names)
            archive.writestr(filename, action_result_csv(run, item).encode("utf-8-sig"))
    return output.getvalue(), "application/zip", f"{safe_filename(run.get('run_id') or 'backtest')}_actions.zip"


def action_result_at(run: dict[str, Any], result_index: int) -> dict[str, Any]:
    results = run.get("results", [])
    if result_index < 0 or result_index >= len(results):
        raise KeyError(f"backtest result not found: {result_index}")
    item = results[result_index]
    if not action_result_available(item):
        raise ValueError(f"backtest result has no action series: {result_index}")
    return item


def action_result_available(item: dict[str, Any]) -> bool:
    return item.get("status") == "done" and bool(item.get("series"))


def action_result_csv(run: dict[str, Any], item: dict[str, Any]) -> str:
    series = item.get("series") or []
    action_dims = [int(entry.get("dimension", index)) for index, entry in enumerate(series)]
    frame_count = max((len(entry.get("ground_truth") or []) for entry in series), default=0)
    fieldnames = [
        "run_id",
        "profile_id",
        "profile_name",
        "dataset_id",
        "dataset_name",
        "dataset_path",
        "episode_index",
        "frame_index",
    ]
    for dim in action_dims:
        fieldnames.extend([
            f"ground_truth_action_{dim}",
            f"predicted_action_{dim}",
            f"error_action_{dim}",
        ])
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for frame_index in range(frame_count):
        row = {
            "run_id": run.get("run_id"),
            "profile_id": item.get("profile_id", item.get("model_id")),
            "profile_name": item.get("profile_name"),
            "dataset_id": item.get("dataset_id"),
            "dataset_name": item.get("dataset_name"),
            "dataset_path": item.get("dataset_path"),
            "episode_index": item.get("episode_index"),
            "frame_index": frame_index,
        }
        for dim, entry in zip(action_dims, series):
            row[f"ground_truth_action_{dim}"] = value_at(entry.get("ground_truth"), frame_index)
            row[f"predicted_action_{dim}"] = value_at(entry.get("predicted"), frame_index)
            row[f"error_action_{dim}"] = value_at(entry.get("error"), frame_index)
        writer.writerow(row)
    return output.getvalue()


def value_at(values: Any, index: int) -> Any:
    if not isinstance(values, list) or index >= len(values):
        return ""
    return values[index]


def action_result_filename(run: dict[str, Any], item: dict[str, Any], result_index: int) -> str:
    run_id = safe_filename(run.get("run_id") or "backtest")
    profile = safe_filename(item.get("profile_name") or item.get("profile_id") or item.get("model_id") or "profile")
    dataset = safe_filename(item.get("dataset_name") or "dataset")
    episode = safe_filename(item.get("episode_index"))
    return f"{run_id}_{result_index:03d}_{profile}_{dataset}_episode_{episode}_actions.csv"


def safe_filename(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text[:80] or "item"


def unique_zip_name(filename: str, used_names: set[str]) -> str:
    if filename not in used_names:
        used_names.add(filename)
        return filename
    stem = filename[:-4] if filename.lower().endswith(".csv") else filename
    suffix = ".csv" if filename.lower().endswith(".csv") else ""
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def public_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run.get("run_id"),
        "created_at": run.get("created_at"),
        "dataset_paths": run.get("dataset_paths", []),
        "profile_ids": run.get("profile_ids", run.get("model_ids", [])),
        "model_ids": run.get("model_ids", run.get("profile_ids", [])),
        "profiles": run.get("profiles", []),
        "episodes": run.get("episodes", []),
        "summary": run.get("summary", {}),
    }


def export_csv(run: dict[str, Any]) -> str:
    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "run_id",
            "profile_id",
            "profile_name",
            "dataset_name",
            "dataset_path",
            "episode_index",
            "status",
            "frames",
            "mae",
            "rmse",
            "max_error",
            "error",
        ],
    )
    writer.writeheader()
    for item in run.get("results", []):
        metrics = item.get("metrics") or {}
        writer.writerow(
            {
                "run_id": run.get("run_id"),
                "profile_id": item.get("profile_id", item.get("model_id")),
                "profile_name": item.get("profile_name"),
                "dataset_name": item.get("dataset_name"),
                "dataset_path": item.get("dataset_path"),
                "episode_index": item.get("episode_index"),
                "status": item.get("status"),
                "frames": item.get("frames"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "max_error": metrics.get("max_error"),
                "error": item.get("error", ""),
            }
        )
    return output.getvalue()


def export_html(run: dict[str, Any]) -> str:
    summary = run.get("summary") or {}
    rows = []
    for item in run.get("results", []):
        metrics = item.get("metrics") or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('profile_name') or item.get('profile_id') or item.get('model_id') or ''))}</td>"
            f"<td>{html.escape(str(item.get('dataset_name', '')))}</td>"
            f"<td>{html.escape(str(item.get('episode_index', '')))}</td>"
            f"<td>{html.escape(str(item.get('status', '')))}</td>"
            f"<td>{html.escape(str(item.get('frames', '')))}</td>"
            f"<td>{html.escape(str(metrics.get('mae', '')))}</td>"
            f"<td>{html.escape(str(metrics.get('rmse', '')))}</td>"
            f"<td>{html.escape(str(metrics.get('max_error', '')))}</td>"
            f"<td>{html.escape(str(item.get('error', '')))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Backtest {html.escape(str(run.get('run_id', '')))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172331; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #dbe3ea; padding: 8px; text-align: left; }}
    th {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>Backtest {html.escape(str(run.get('run_id', '')))}</h1>
  <p>Created: {html.escape(str(run.get('created_at', '')))}</p>
  <p>Total: {html.escape(str(summary.get('total', '')))} · Done: {html.escape(str(summary.get('done', '')))} · Failed: {html.escape(str(summary.get('failed', '')))}</p>
  <table>
    <thead><tr><th>Model</th><th>Dataset</th><th>Episode</th><th>Status</th><th>Frames</th><th>MAE</th><th>RMSE</th><th>Max Error</th><th>Error</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>"""


def run_path(run_id: str) -> Path:
    return BACKTEST_DIR / f"{run_id}.json"


def job_path(job_id: str) -> Path:
    return BACKTEST_JOB_DIR / f"{job_id}.json"
