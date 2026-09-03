"""Persist completed reports to a local folder and list them for the website."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

_SAFE = re.compile(r"[^\w\u4e00-\u9fff.-]+")


def _safe_stem(name: str, limit: int = 36) -> str:
    stem = Path(str(name or "video")).stem
    cleaned = _SAFE.sub("_", stem).strip("._")
    return (cleaned or "video")[:limit]


def _stamp(created_at: str | None) -> str:
    raw = (created_at or "").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%Y%m%d-%H%M")
    except ValueError:
        return datetime.now().strftime("%Y%m%d-%H%M")


def archive_slug(job_id: str, created_at: str | None, source_name: str, score) -> str:
    sc = "na" if score is None else str(int(score))
    return f"{_stamp(created_at)}_{_safe_stem(source_name)}_{sc}_{job_id}"


def find_archive(reports_dir: Path, job_id: str) -> Path | None:
    if not reports_dir.is_dir():
        return None
    suffix = f"_{job_id}"
    for path in reports_dir.iterdir():
        if path.is_dir() and path.name.endswith(suffix):
            return path
    return None


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def history_item(job_dir: Path) -> dict | None:
    """One website row for a finished job. None if there is no report."""
    report_path = job_dir / "report.json"
    if not job_dir.is_dir() or not report_path.is_file():
        return None
    status = _load_json(job_dir / "status.json")
    if status.get("status") == "error":
        return None
    report = _load_json(report_path)
    overall = report.get("overall") or {}
    created = status.get("created_at") or report.get("generated_at") or ""
    score = status.get("score")
    if score is None:
        score = overall.get("score")
    n_swings = status.get("n_swings")
    if n_swings is None:
        n_swings = overall.get("n_swings")
    preview = (job_dir / "preview.jpg").is_file() or (job_dir / "cover.jpg").is_file()
    return {
        "id": job_dir.name,
        "title": status.get("title") or report.get("title") or "网球挥拍测评报告",
        "source_name": status.get("source_name") or report.get("source_name") or "",
        "created_at": created,
        "score": score,
        "grade": overall.get("grade"),
        "grade_label": overall.get("grade_label"),
        "n_swings": n_swings,
        "view_label": report.get("view_label") or "",
        "handedness_label": report.get("handedness_label") or "",
        "is_sample": bool(status.get("is_sample")),
        "focus": report.get("focus") or "",
        "has_preview": preview,
    }


def archive_job_id(path: Path) -> str | None:
    meta = _load_json(path / "meta.json")
    job_id = str(meta.get("id") or "").strip()
    if job_id:
        return job_id
    name = path.name
    if "_" not in name:
        return None
    tail = name.rsplit("_", 1)[-1]
    return tail or None


def history_item_from_archive(path: Path) -> dict | None:
    """Website row for a folder in outputs/reports. id is the original job id."""
    job_id = archive_job_id(path)
    if not job_id:
        return None
    item = history_item(path)
    if not item:
        return None
    item["id"] = job_id
    return item


def list_history(jobs_dir: Path, reports_dir: Path | None = None, limit: int = 80) -> list[dict]:
    by_id: dict[str, dict] = {}
    if jobs_dir.is_dir():
        for job_dir in jobs_dir.iterdir():
            item = history_item(job_dir)
            if item:
                by_id[item["id"]] = item
    if reports_dir is not None and reports_dir.is_dir():
        for path in reports_dir.iterdir():
            if not path.is_dir():
                continue
            item = history_item_from_archive(path)
            if item and item["id"] not in by_id:
                by_id[item["id"]] = item
    items = list(by_id.values())
    items.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    return items[:limit]


def archive_report(job_dir: Path, reports_dir: Path, item: dict | None = None) -> Path | None:
    """Copy a finished report into outputs/reports/<readable-name>/."""
    job_id = job_dir.name
    item = item or history_item(job_dir)
    if not item:
        return None
    reports_dir.mkdir(parents=True, exist_ok=True)
    dest = find_archive(reports_dir, job_id)
    slug = archive_slug(job_id, item.get("created_at"), item.get("source_name") or "", item.get("score"))
    wanted = reports_dir / slug
    if dest is None:
        dest = wanted
    elif dest != wanted:
        dest.rename(wanted)
        dest = wanted
    dest.mkdir(parents=True, exist_ok=True)

    src_report = job_dir / "report.json"
    dst_report = dest / "report.json"
    need = (not dst_report.is_file()) or dst_report.stat().st_mtime < src_report.stat().st_mtime
    if need:
        shutil.copy2(src_report, dst_report)
        for name in ("preview.jpg", "cover.jpg", "status.json"):
            src = job_dir / name
            if src.is_file():
                shutil.copy2(src, dest / name)
        overlay = job_dir / "overlay.mp4"
        if overlay.is_file() and overlay.stat().st_size <= 80 * 1024 * 1024:
            shutil.copy2(overlay, dest / "overlay.mp4")
        (dest / "meta.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return dest
