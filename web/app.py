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
_worker_started = False
_TERMINAL = {"done", "error", "cancelled"}


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


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _public_job(job: dict) -> dict:
    out = {k: v for k, v in job.items() if k != "video_path"}
    out.update(_eta_fields(job))
    return out


def _eta_fields(job: dict) -> dict:
    duration = float(job.get("max_seconds") or 0)
    if duration <= 0:
        duration = 90.0
    pose_s = int(max(90, duration * 4))
    steps = [
        {"step": 1, "name": "读取视频", "seconds": 15, "label": "约 15 秒"},
        {
            "step": 2,
            "name": "识别动作",
            "seconds": pose_s,
            "label": f"约 {max(1, round(pose_s / 60))} 分钟",
        },
        {"step": 3, "name": "找出挥拍", "seconds": 30, "label": "约 30 秒"},
        {"step": 4, "name": "整理数据", "seconds": 50, "label": "约 1 分钟"},
        {
            "step": 5,
            "name": "教练点评",
            "seconds": 150,
            "label": "约 2 分钟，这一步最慢",
        },
    ]
    cur = int(job.get("step") or 0)
    if job.get("status") in _TERMINAL:
        remain = 0
        hint = "已完成" if job.get("status") == "done" else ""
    else:
        remain = sum(s["seconds"] for s in steps if s["step"] >= max(cur, 1))
        mins = max(1, round(remain / 60))
        if cur >= 5:
            hint = f"正在生成报告，大约还要 2 分钟。关掉页面也没关系，回来还能接着看。"
        else:
            hint = f"全部大约还要 {mins} 分钟。关掉页面也不会中断，回来可继续查看进度。"
    return {"eta_steps": steps, "eta_remaining_s": remain, "eta_hint": hint}


def _write_status(job_id: str) -> None:
    job = _jobs[job_id]
    path = JOBS_DIR / job_id / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, ensure_ascii=False, default=json_default), encoding="utf-8")


def _set(job_id: str, **kwargs) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        incoming = kwargs.get("status")
        if job.get("status") in _TERMINAL and incoming not in _TERMINAL:
            return
        job.update(kwargs)
        job["updated_at"] = _now()
        _write_status(job_id)


def _find_source_video(job_dir: Path) -> Path | None:
    for p in sorted(job_dir.glob("source.*")):
        if p.is_file() and p.stat().st_size > 1000:
            return p
    return None


def _load_status_file(job_dir: Path) -> dict | None:
    path = job_dir / "status.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _busy_job() -> dict | None:
    with _lock:
        for job in _jobs.values():
            if job.get("status") in ("queued", "running"):
                return job
    return None


def _recover_jobs() -> None:
    """Reload jobs from disk. Finish ones that already have a report; drop the rest."""
    if not JOBS_DIR.is_dir():
        return
    sample = _sample_path()
    for job_dir in sorted(JOBS_DIR.iterdir()):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        job = _load_status_file(job_dir) or {"id": job_id}
        job["id"] = job_id
        report = job_dir / "report.json"
        video = job.get("video_path")
        if not video:
            if job.get("is_sample") and sample is not None:
                video = str(sample)
            else:
                src = _find_source_video(job_dir)
                video = str(src) if src else ""
        job["video_path"] = video
        status = job.get("status")
        if report.is_file() and status != "error":
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
                overall = data.get("overall") or {}
                job["status"] = "done"
                job["step"] = 5
                job["step_name"] = "教练点评"
                job["progress"] = 100
                job["message"] = "分析完成"
                job["score"] = overall.get("score")
                job["n_swings"] = overall.get("n_swings")
            except Exception:
                pass
        elif status in ("queued", "running"):
            job["status"] = "cancelled"
            job["message"] = "分析已取消，请稍后再试"
        with _lock:
            _jobs[job_id] = job
            _write_status(job_id)
        if job.get("is_sample") and job.get("status") == "done" and sample is not None:
            _save_sample_cache(job_id, sample, str(job.get("stroke_mode") or "auto"))


def _run_job(job_id: str) -> None:
    job = _jobs[job_id]
    out_dir = JOBS_DIR / job_id
    try:
        _set(job_id, status="running", step=1, step_name="读取视频", progress=1, message="开始分析你的录像…")

        def progress(payload: dict) -> None:
            with _lock:
                if _jobs.get(job_id, {}).get("status") in _TERMINAL:
                    return
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
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    t = threading.Thread(target=_worker, name="analyze-worker", daemon=True)
    t.start()
    threading.Thread(target=warm_agent, name="cursor-warm", daemon=True).start()
    _recover_jobs()


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
    return {"ok": True, "sample": bool(sample), "gpu": gpu, "version": "2.0", "busy": bool(_busy_job())}


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

    use_sample = sample in ("1", "true", "yes")
    src = _sample_path() if use_sample else None
    if use_sample:
        if src is None:
            raise HTTPException(400, "没有可用的样例视频")
        cache = _load_sample_cache(src, stroke)
        if cache is not None:
            job_id = uuid.uuid4().hex[:12]
            meta = _install_sample_cache(job_id, cache)
            now = _now()
            with _lock:
                _jobs[job_id] = {
                    "id": job_id,
                    "status": "done",
                    "step": 5,
                    "step_name": "教练点评",
                    "progress": 100,
                    "message": "分析完成",
                    "max_seconds": 60,
                    "stroke_mode": stroke,
                    "title": title,
                    "source_name": meta.get("source_name") or src.name,
                    "score": meta.get("score"),
                    "n_swings": meta.get("n_swings"),
                    "cached": True,
                    "is_sample": True,
                    "created_at": now,
                    "updated_at": now,
                }
                _write_status(job_id)
            return {"job_id": job_id, "cached": True}

    if _busy_job():
        raise HTTPException(409, "正在分析其他录像，请稍后再试")

    job_id = uuid.uuid4().hex[:12]
    out_dir = JOBS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_sample:
        video_path = src
        source_name = src.name
        max_seconds = 60
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
        for job in _jobs.values():
            if job.get("status") in ("queued", "running"):
                raise HTTPException(409, "正在分析其他录像，请稍后再试")
        _jobs[job_id] = {
            "id": job_id,
            "status": "running",
            "step": 0,
            "step_name": "准备中",
            "progress": 0,
            "message": "开始分析…",
            "max_seconds": max_seconds,
            "stroke_mode": stroke,
            "title": title,
            "source_name": source_name,
            "video_path": str(video_path),
            "is_sample": use_sample,
            "created_at": _now(),
        }
        _write_status(job_id)

    _queue.put(job_id)
    return {"job_id": job_id, "cached": False}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            return _public_job(job)
    status_file = JOBS_DIR / job_id / "status.json"
    if status_file.exists():
        data = json.loads(status_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return _public_job(data)
        return data
    raise HTTPException(404, "任务不存在")


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
