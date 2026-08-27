"""Upload a tennis video, watch the pipeline, get a coaching report."""

from __future__ import annotations

import json
import queue
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

from pipeline.session import analyze_video, get_estimator, json_default

JOBS_DIR = ROOT / "outputs" / "jobs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLE_CANDIDATES = [
    ROOT / "samples" / "demo.mp4",
    ROOT / "c487ed1118b3c668be6fe62ab0cbe3f9.mp4",
]

JOBS_DIR.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_queue: queue.Queue[str] = queue.Queue()


def _sample_path() -> Path | None:
    for p in SAMPLE_CANDIDATES:
        if p.exists():
            return p
    return None


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
        _set(job_id, status="running", step=1, step_name="读取视频", progress=1, message="排队完成，开始分析")

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
            media_url_base=f"jobs/{job_id}",
            progress=progress,
        )
        _set(
            job_id,
            status="done",
            step=5,
            step_name="生成报告",
            progress=100,
            message="分析完成",
            score=report["overall"]["score"],
            n_swings=report["overall"]["n_swings"],
        )
    except Exception as exc:
        _set(
            job_id,
            status="error",
            message=str(exc),
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )


def _worker() -> None:
    get_estimator()
    while True:
        job_id = _queue.get()
        try:
            _run_job(job_id)
        finally:
            _queue.task_done()


def _start_worker() -> None:
    t = threading.Thread(target=_worker, name="analyze-worker", daemon=True)
    t.start()


app = FastAPI(title="Tennis Motion Lab")
_start_worker()


@app.get("/api/health")
def health():
    sample = _sample_path()
    return {"ok": True, "sample": bool(sample)}


@app.get("/api/sample")
def sample_info():
    path = _sample_path()
    if not path:
        raise HTTPException(404, "服务器上还没有样例视频")
    return {
        "name": path.name,
        "label": "背面原视频样例（约前 90 秒）",
        "exists": True,
    }


@app.post("/api/analyze")
async def analyze(
    video: UploadFile | None = File(default=None),
    sample: str = Form(default="0"),
    max_seconds: float = Form(default=90),
    stroke: str = Form(default="auto"),
    title: str = Form(default="网球挥拍测评报告"),
):
    max_seconds = float(min(max(max_seconds, 15), 180))
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

    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "step": 0,
            "step_name": "排队中",
            "progress": 0,
            "message": "等待分析线程…",
            "max_seconds": max_seconds,
            "stroke_mode": stroke,
            "title": title,
            "source_name": source_name,
            "video_path": str(video_path),
            "preview_url": f"jobs/{job_id}/preview.jpg",
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


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str):
    path = JOBS_DIR / job_id / "report.json"
    if not path.exists():
        raise HTTPException(404, "报告尚未生成")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


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
