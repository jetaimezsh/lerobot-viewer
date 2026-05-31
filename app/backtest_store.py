from __future__ import annotations

import csv
import html
import json
from io import StringIO
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = APP_ROOT / "state"
BACKTEST_DIR = STATE_DIR / "backtests"


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


def public_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run.get("run_id"),
        "created_at": run.get("created_at"),
        "dataset_paths": run.get("dataset_paths", []),
        "model_ids": run.get("model_ids", []),
        "episodes": run.get("episodes", []),
        "summary": run.get("summary", {}),
    }


def export_csv(run: dict[str, Any]) -> str:
    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "run_id",
            "model_id",
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
                "model_id": item.get("model_id"),
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
            f"<td>{html.escape(str(item.get('model_id', '')))}</td>"
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
