"""Upload a tennis video and return a coaching report."""

from __future__ import annotations

import hashlib
import json
import queue
import shutil
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.cursor_client import warm_agent
from pipeline.oss import refresh_report_urls
from pipeline.session import analyze_video, get_detector, get_estimator, json_default

JOBS_DIR = ROOT / "outputs" / "jobs"
SAMPLE_CACHE_DIR = ROOT / "outputs" / "sample_cache"
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLE_CANDIDATES = [
    ROOT / "samples" / "demo.mp4",
]
SAMPLE_CACHE_VERSION = "2.1-deep"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_queue: queue.Queue[str] = queue.Queue()


def _sample_path() -> Path | None:
    for p in SAMPLE_CANDIDATES:
        if p.exists():
            return p
    return None


def _sample_fingerprint(src: Path, stroke: str) -> str:
    st = src.stat()
    raw = f"{SAMPLE_CACHE_VERSION}|{src.resolve()}|{st.st_mtime_ns}|{st.st_size}|{stroke}|60"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _sample_cache_dir(src: Path, stroke: str) -> Path:
    return SAMPLE_CACHE_DIR / _sample_fingerprint(src, stroke)


def _load_sample_cache(src: Path, stroke: str) -> Path | None:
    cache = _sample_cache_dir(src, stroke)
    if (cache / "report.json").is_file():
        return cache
    return None


def _save_sample_cache(job_id: str, src: Path, stroke: str) -> None:
    job_dir = JOBS_DIR / job_id
    report = job_dir / "report.json"
    if not report.is_file():
        return
    cache = _sample_cache_dir(src, stroke)
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report, cache / "report.json")
    preview = job_dir / "preview.jpg"
    if preview.is_file():
        shutil.copy2(preview, cache / "preview.jpg")


def _install_sample_cache(job_id: str, cache: Path) -> dict:
    out = JOBS_DIR / job_id
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache / "report.json", out / "report.json")
    preview = cache / "preview.jpg"
    if preview.is_file():
        shutil.copy2(preview, out / "preview.jpg")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    overall = report.get("overall") or {}
    return {
        "score": overall.get("score"),
        "n_swings": overall.get("n_swings"),
        "source_name": report.get("source_name") or "demo.mp4",
    }


def _write_status(job_id: str) -> None:
    job = _jobs[job_id]
    path = JOBS_DIR / job_id / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in job.items() if k != "video_path"}
    path.write_text(json.dumps(payload, ensure_ascii=False, default=json_default), encoding="utf-8")


def _set(job_id: str, **kwargs) -> None:
    with _lock:
        _jobs[job_id].update(kwargs)
        _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        )
        _write_status(job_id)


def _run_job(job_id: str) -> None:
    job = _jobs[job_id]
    out_dir = JOBS_DIR / job_id
    try:
        _set(job_id, status="running", step=1, step_name="读取视频", progress=1, message="开始分析你的录像…")

        def progress(payload: dict) -> None:
            patch = {k: v for k, v in payload.items() if k != "preview"}
            if payload.get("preview"):
                patch["preview_rev"] = datetime.now().timestamp()
            _set(job_id, status="running", **patch)

        report = analyze_video(
            Path(job["video_path"]),
            out_dir,
            max_seconds=float(job["max_seconds"]),
            stroke_mode=job["stroke_mode"],
            title=job.get("title") or "网球挥拍测评报告",
            progress=progress,
        )
        _set(
            job_id,
            status="done",
            step=5,
            step_name="教练点评",
            progress=100,
            message="分析完成",
            score=report["overall"]["score"],
            n_swings=report["overall"]["n_swings"],
        )
        if job.get("is_sample"):
            src = _sample_path()
            if src is not None:
                _save_sample_cache(job_id, src, str(job.get("stroke_mode") or "auto"))
    except Exception as exc:
        user_msg = str(exc) if isinstance(exc, RuntimeError) else "分析失败，请稍后重试"
        _set(
            job_id,
            status="error",
            message=user_msg,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )


def _worker() -> None:
    get_estimator()
    get_detector()
    while True:
        job_id = _queue.get()
        try:
            _run_job(job_id)
        finally:
            _queue.task_done()


def _start_worker() -> None:
    t = threading.Thread(target=_worker, name="analyze-worker", daemon=True)
    t.start()
    threading.Thread(target=warm_agent, name="cursor-warm", daemon=True).start()


app = FastAPI(title="网球挥拍测评 2.0")
_start_worker()


@app.get("/api/health")
def health():
    sample = _sample_path()
    gpu = False
    try:
        import torch

        gpu = bool(torch.cuda.is_available())
    except Exception:
        gpu = False
    return {"ok": True, "sample": bool(sample), "gpu": gpu, "version": "2.0"}


@app.get("/api/sample")
def sample_info():
    path = _sample_path()
    if not path:
        raise HTTPException(404, "服务器上还没有样例视频")
    return {
        "name": path.name,
        "label": "1 分钟侧面训练样例",
        "exists": True,
    }


@app.post("/api/analyze")
async def analyze(
    video: UploadFile | None = File(default=None),
    sample: str = Form(default="0"),
    max_seconds: float = Form(default=0),
    stroke: str = Form(default="auto"),
    title: str = Form(default="网球挥拍测评报告 2.0"),
):
    if stroke not in ("auto", "forehand", "backhand"):
        stroke = "auto"

    job_id = uuid.uuid4().hex[:12]
    out_dir = JOBS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    use_sample = sample in ("1", "true", "yes")
    if use_sample:
        src = _sample_path()
        if src is None:
            raise HTTPException(400, "没有可用的样例视频")
        video_path = src
        source_name = src.name
        max_seconds = 60
        cache = _load_sample_cache(src, stroke)
        if cache is not None:
            meta = _install_sample_cache(job_id, cache)
            now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            with _lock:
                _jobs[job_id] = {
                    "id": job_id,
                    "status": "done",
                    "step": 5,
                    "step_name": "教练点评",
                    "progress": 100,
                    "message": "分析完成",
                    "max_seconds": max_seconds,
                    "stroke_mode": stroke,
                    "title": title,
                    "source_name": meta.get("source_name") or source_name,
                    "score": meta.get("score"),
                    "n_swings": meta.get("n_swings"),
                    "cached": True,
                    "is_sample": True,
                    "created_at": now,
                    "updated_at": now,
                }
                _write_status(job_id)
            return {"job_id": job_id, "cached": True}
    else:
        if video is None or not video.filename:
            raise HTTPException(400, "请上传视频，或选择使用样例")
        suffix = Path(video.filename).suffix.lower()
        if suffix not in {".mp4", ".mov", ".webm", ".m4v", ".avi"}:
            raise HTTPException(400, "请上传 mp4 / mov / webm 视频")
        video_path = out_dir / f"source{suffix}"
        data = await video.read()
        if len(data) < 1000:
            raise HTTPException(400, "视频文件太小或已损坏")
        if len(data) > 400 * 1024 * 1024:
            raise HTTPException(400, "视频超过 400MB")
        video_path.write_bytes(data)
        source_name = Path(video.filename).name
        max_seconds = 0

    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "step": 0,
            "step_name": "排队中",
            "progress": 0,
            "message": "排队中，马上开始…",
            "max_seconds": max_seconds,
            "stroke_mode": stroke,
            "title": title,
            "source_name": source_name,
            "video_path": str(video_path),
            "is_sample": use_sample,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
        _write_status(job_id)

    _queue.put(job_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        status_file = JOBS_DIR / job_id / "status.json"
        if status_file.exists():
            return json.loads(status_file.read_text(encoding="utf-8"))
        raise HTTPException(404, "任务不存在")
    return {k: v for k, v in job.items() if k != "video_path"}


@app.get("/api/jobs/{job_id}/preview")
def job_preview(job_id: str):
    root = (JOBS_DIR / job_id).resolve()
    target = (root / "preview.jpg").resolve()
    if target.parent != root or not target.is_file():
        raise HTTPException(404, "预览尚未生成")
    return FileResponse(
        target,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str):
    path = JOBS_DIR / job_id / "report.json"
    if not path.exists():
        raise HTTPException(404, "报告尚未生成")
    data = json.loads(path.read_text(encoding="utf-8"))
    return JSONResponse(refresh_report_urls(data))


@app.get("/jobs/{job_id}/{path:path}")
def job_file(job_id: str, path: str):
    root = (JOBS_DIR / job_id).resolve()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(400, "非法路径")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(target)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
