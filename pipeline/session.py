"""Single-camera video analysis: pose → swings → metrics → report JSON."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.analyze import (
    SCORE_AXES,
    build_series,
    classify_swing,
    detect_peaks,
    grade_from_score,
    infer_handedness,
    infer_view,
    measure_swings,
    score_and_write,
    speed_from_xy,
    summarize,
    wrist_track,
)
from pipeline.coach import enrich_with_cursor
from pipeline.detect import ObjectDetector, default_detect_path, draw_objects
from pipeline.oss import object_key, require_configured, upload_file
from pipeline.pose import PoseEstimator, draw_pose

LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def _swap_lr_pose(xy: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy, conf = xy.copy(), conf.copy()
    for a, b in LR_PAIRS:
        xy[[a, b]] = xy[[b, a]]
        conf[[a, b]] = conf[[b, a]]
    return xy, conf


def _fix_backview_laterality(xy_list, conf_list, view: str):
    """YOLO sometimes labels a back-facing player as if they face the camera."""
    if view != "back":
        return xy_list, conf_list
    diffs = []
    for xy, conf in zip(xy_list, conf_list):
        if conf[5] < 0.35 or conf[6] < 0.35:
            continue
        diffs.append(float(xy[5, 0] - xy[6, 0]))  # l_shoulder x - r_shoulder x
    if not diffs:
        return xy_list, conf_list
    # 背面：解剖左肩应在画面左侧（x 更小）。若左肩平均在右，则左右标反了。
    if float(np.median(diffs)) <= 8:
        return xy_list, conf_list
    out_xy, out_conf = [], []
    for xy, conf in zip(xy_list, conf_list):
        a, b = _swap_lr_pose(xy, conf)
        out_xy.append(a)
        out_conf.append(b)
    return out_xy, out_conf

ProgressCb = Callable[[dict], None]

_EST: PoseEstimator | None = None
_DET: ObjectDetector | None = None


def json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def get_estimator(model_path: str | None = None) -> PoseEstimator:
    global _EST
    if _EST is None:
        path = model_path or str(ROOT / "models" / "yolov8n-pose.pt")
        if not Path(path).exists():
            path = "yolov8n-pose.pt"
        _EST = PoseEstimator(path)
    return _EST


def get_detector(model_path: str | None = None) -> ObjectDetector:
    global _DET
    if _DET is None:
        path = model_path or default_detect_path(ROOT)
        _DET = ObjectDetector(path)
    return _DET


def downsample_series(series, every: int = 4) -> dict:
    idx = np.arange(0, len(series.t), every)

    def arr(x):
        y = x[idx]
        return [None if not np.isfinite(v) else round(float(v), 2) for v in y]

    return {
        "t": [round(float(v), 3) for v in series.t[idx]],
        "wrist_speed": arr(series.wrist_speed),
        "elbow": arr(series.elbow),
        "knee": arr(series.knee),
    }


def _encode_h264(src: Path, dst: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-preset",
                "veryfast",
                "-movflags",
                "+faststart",
                str(dst),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("视频处理失败，请稍后重试") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("视频处理失败，请稍后重试") from exc


def _draw_hud(frame: np.ndarray, t: float, ok: bool, _done: int, _total: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (8, 8), (min(w - 8, 420), 54), (12, 16, 22), -1)
    label = f"{t:.1f}s"
    cv2.putText(
        out,
        label,
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255) if ok else (80, 80, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _save_phase(path: Path, view, pose, tag: str, objs=None) -> None:
    vis = draw_pose(view, pose)
    if objs is not None:
        vis = draw_objects(vis, objs)
    h, w = vis.shape[:2]
    scale = 560 / max(w, 1)
    vis = cv2.resize(vis, (560, int(h * scale)))
    cv2.putText(
        vis, tag, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def _publish(job_id: str, rel: str, path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError("保存文件失败，请稍后重试")
    return upload_file(path, object_key(job_id, rel))


def _emit(cb: ProgressCb | None, **payload) -> None:
    if cb:
        cb(payload)


def analyze_video(
    video_path: Path,
    out_dir: Path,
    *,
    max_seconds: float = 0.0,
    stroke_mode: str = "auto",
    title: str | None = None,
    estimator: PoseEstimator | None = None,
    progress: ProgressCb | None = None,
) -> dict:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kf_dir = out_dir / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    job_id = out_dir.name
    require_configured()

    _emit(progress, step=1, step_name="读取视频", progress=3, message="正在打开你的录像…")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("打不开这段视频，请换一个文件再试")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 544)
    if max_seconds and max_seconds > 0:
        max_frames = int(max_seconds * fps)
        if n_total > 0:
            max_frames = min(max_frames, n_total)
    else:
        max_frames = n_total if n_total > 0 else 10**9
    target = max(1, max_frames)

    est = estimator or get_estimator()
    det = get_detector()
    _emit(
        progress,
        step=2,
        step_name="识别动作",
        progress=8,
        message="正在识别球员动作…",
    )

    fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    raw_path = Path(tmp_name)
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    ts: list[float] = []
    xy_list: list[np.ndarray] = []
    conf_list: list[np.ndarray] = []
    ok_flags: list[bool] = []
    obj_list: list = []
    buf: list[np.ndarray] = []
    batch = 4 if getattr(est, "device", "cpu") == "cpu" else 8
    if getattr(est, "device", "cpu") == "cuda":
        try:
            import torch

            # L20-2Q 等 vGPU 只有约 2GB，批量过大容易显存不足
            if torch.cuda.get_device_properties(0).total_memory < 4 * 1024**3:
                batch = 2
        except Exception:
            batch = 2
    preview_path = out_dir / "preview.jpg"
    i = 0

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("这段视频没有可用画面，请换一段再试")
    cv2.imwrite(str(preview_path), first, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    _emit(
        progress,
        step=2,
        step_name="识别动作",
        progress=8,
        message="正在识别球员动作…",
        preview=True,
    )
    buf.append(first)

    def flush():
        nonlocal i
        if not buf:
            return
        poses = est.infer_batch(buf)
        objects = det.infer_batch(buf)
        for frame, pose, objs in zip(buf, poses, objects):
            vis = draw_objects(draw_pose(frame, pose), objs)
            vis = _draw_hud(vis, i / fps, pose.ok, i + 1, target)
            writer.write(vis)
            xy_list.append(pose.xy)
            conf_list.append(pose.conf)
            ok_flags.append(bool(pose.ok))
            obj_list.append(objs)
            ts.append(i / fps)
            if (i % 18) == 0:
                cv2.imwrite(str(preview_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                _emit(
                    progress,
                    step=2,
                    step_name="识别动作",
                    progress=8 + int(62 * (i + 1) / target),
                    message="正在识别球员动作…",
                    preview=True,
                )
            i += 1
        buf.clear()

    while i + len(buf) < target:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        if len(buf) >= batch:
            flush()
    flush()
    cap.release()
    writer.release()

    overlay_path = out_dir / "overlay.mp4"
    _emit(progress, step=2, step_name="识别动作", progress=72, message="正在生成回放…")
    _encode_h264(raw_path, overlay_path)
    overlay_pub = _publish(job_id, "overlay.mp4", overlay_path)
    if not preview_path.is_file():
        raise RuntimeError("无法生成预览，请换一段录像再试")
    preview_pub = _publish(job_id, "preview.jpg", preview_path)
    try:
        raw_path.unlink(missing_ok=True)
    except OSError:
        pass

    if not ts:
        raise RuntimeError("这段视频没有可用画面，请换一段再试")

    t_arr = np.array(ts, dtype=np.float64)
    detect_rate = float(np.mean(ok_flags)) if ok_flags else 0.0
    view = infer_view(xy_list, conf_list)
    xy_list, conf_list = _fix_backview_laterality(xy_list, conf_list, view)
    handed = infer_handedness(xy_list, conf_list, t_arr, fps)
    takeback_mode = "distance"
    enable_late = view == "side"

    _emit(progress, step=3, step_name="找出挥拍", progress=78, message="正在找出每一次挥拍…")

    l_spd = speed_from_xy(wrist_track(xy_list, conf_list, "l_wrist"), t_arr)
    r_spd = speed_from_xy(wrist_track(xy_list, conf_list, "r_wrist"), t_arr)
    combined = np.maximum(l_spd, r_spd)
    peaks = detect_peaks(t_arr, combined, min_gap_s=1.05)

    labels: list[str] = []
    for p in peaks:
        if stroke_mode in ("forehand", "backhand"):
            labels.append(stroke_mode)
        else:
            labels.append(classify_swing(xy_list, conf_list, t_arr, p, handed, fps))

    ball_xy = [o.ball_xy if o is not None else None for o in obj_list]
    racket_xy = [o.racket_xy if o is not None else None for o in obj_list]
    racket_box = [o.racket_box if o is not None else None for o in obj_list]

    _emit(
        progress,
        step=4,
        step_name="整理数据",
        progress=84,
        message=f"已找到 {len(peaks)} 次挥拍，正在整理评分…",
    )

    clips = []
    timeline = []
    all_caveats: list[str] = []
    dummy_series = None

    for stroke in ("forehand", "backhand"):
        stroke_peaks = [p for p, lab in zip(peaks, labels) if lab == stroke]
        if not stroke_peaks:
            continue
        series = build_series(
            t_arr,
            xy_list,
            conf_list,
            xy_list,
            conf_list,
            stroke,
            takeback_mode=takeback_mode,
            hitting=handed if stroke == "forehand" else ("left" if handed == "right" else "right"),
        )
        dummy_series = series
        hitting_wrist = wrist_track(
            xy_list,
            conf_list,
            "r_wrist" if series.hitting == "right" else "l_wrist",
        )
        swings = measure_swings(
            series,
            fps,
            peaks=stroke_peaks,
            enable_late_contact=enable_late,
            ball_xy=ball_xy,
            racket_xy=racket_xy,
            wrist_xy=hitting_wrist,
            racket_box=racket_box,
            view=view,
        )
        summary = summarize(swings, takeback_is_ratio=True)
        written = score_and_write(stroke, summary, view=view, source="original")
        all_caveats = written["caveats"]

        needed: dict[int, list] = {}
        for si, sw in enumerate(swings, 1):
            for name, fi in (
                ("ready", sw.ready_i),
                ("takeback", sw.takeback_i),
                ("contact", sw.contact_i),
                ("follow", sw.follow_i),
            ):
                needed.setdefault(fi, []).append((si, name))

        grabbed: dict[int, tuple] = {}
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or idx >= target:
                break
            if idx in needed:
                grabbed[idx] = (frame, est.infer(frame))
                if len(grabbed) == len(needed):
                    break
            idx += 1
        cap.release()

        swing_payload = []
        clip_id = stroke
        labels_zh = {
            "ready": "准备",
            "takeback": "引拍",
            "contact": "击球",
            "follow": "随挥",
        }
        for si, sw in enumerate(swings, 1):
            phases = {}
            for name, fi in (
                ("ready", sw.ready_i),
                ("takeback", sw.takeback_i),
                ("contact", sw.contact_i),
                ("follow", sw.follow_i),
            ):
                fname = f"{clip_id}_s{si:02d}_{name}.jpg"
                abs_path = kf_dir / fname
                if fi in grabbed:
                    frame, pose = grabbed[fi]
                    objs = obj_list[fi] if fi < len(obj_list) else None
                    _save_phase(abs_path, frame, pose, labels_zh[name], objs=objs)
                if not abs_path.is_file():
                    raise RuntimeError("动作截图生成失败，请换一段更清晰的录像再试")
                published = _publish(job_id, f"keyframes/{fname}", abs_path)
                phases[name] = {
                    "t": round(float(series.t[fi]), 3),
                    "image": published["url"],
                    "oss_key": published["key"],
                }
            swing_payload.append(
                {
                    "index": si,
                    "contact_t": round(sw.contact_t, 3),
                    "elbow_deg": None if sw.elbow_contact is None else round(sw.elbow_contact, 1),
                    "knee_deg": None if sw.knee_contact is None else round(sw.knee_contact, 1),
                    "cog_ratio": None if sw.cog_ready is None else round(sw.cog_ready, 3),
                    "stance_ratio": None if sw.stance_ready is None else round(sw.stance_ready, 3),
                    "takeback_ratio": None
                    if sw.takeback_extent is None
                    else round(float(sw.takeback_extent), 3),
                    "late_contact": bool(sw.late_contact),
                    "contact_source": sw.contact_source,
                    "contact_forward": sw.contact_forward,
                    "cog_stable": sw.cog_stable,
                    "chain_order": sw.chain_order,
                    "racket_speed": None if sw.racket_speed is None else round(sw.racket_speed, 1),
                    "path_lift": sw.path_lift,
                    "phases": phases,
                }
            )
            timeline.append(
                {
                    "t": round(sw.contact_t, 3),
                    "stroke": stroke,
                    "label": written["label"],
                    "clip_id": clip_id,
                    "index": si,
                }
            )

        clips.append(
            {
                "id": clip_id,
                "label": written["label"],
                "stroke": stroke,
                "fps": round(float(fps), 2),
                "n_frames": int(len(ts)),
                "duration_s": round(float(t_arr[-1]), 2),
                "hitting_arm": series.hitting,
                "summary": summary,
                "scores": written["scores"],
                "analysis": {
                    "strengths": written["strengths"],
                    "problems": written["problems"],
                    "drills": written["drills"],
                    "caveats": written["caveats"],
                },
                "series": downsample_series(series),
                "swings": swing_payload,
            }
        )

    timeline.sort(key=lambda x: x["t"])

    if clips:
        total_swings = sum(c["summary"]["n_swings"] for c in clips)
        weights = [max(c["summary"]["n_swings"], 1) for c in clips]
        overall_scores = {}
        for key, _ in SCORE_AXES:
            overall_scores[key] = int(
                round(sum(c["scores"][key] * w for c, w in zip(clips, weights)) / sum(weights))
            )
        overall_scores["综合"] = int(sum(overall_scores[k] for k, _ in SCORE_AXES))
        overall_n = total_swings
    else:
        overall_scores = {k: 0 for k, _ in SCORE_AXES}
        overall_scores["综合"] = 0
        overall_n = 0
        total_swings = 0

    grade, grade_label = grade_from_score(int(overall_scores["综合"]))

    cover_rel = "cover.jpg"
    cover_abs = out_dir / cover_rel
    if clips and clips[0]["swings"]:
        contact = clips[0]["swings"][0]["phases"]["contact"]
        src = kf_dir / Path(str(contact.get("oss_key") or "")).name
        if src.is_file():
            cover_abs.write_bytes(src.read_bytes())
        elif preview_path.is_file():
            cover_abs.write_bytes(preview_path.read_bytes())
    elif preview_path.is_file():
        cover_abs.write_bytes(preview_path.read_bytes())
    if not cover_abs.is_file() and preview_path.is_file():
        cover_abs.write_bytes(preview_path.read_bytes())
    if not cover_abs.is_file():
        raise RuntimeError("无法生成封面，请换一段录像再试")
    cover_pub = _publish(job_id, cover_rel, cover_abs)

    if dummy_series is not None:
        combined_series = {
            "t": [round(float(v), 3) for v in dummy_series.t[::4]],
            "wrist_speed": [
                None if not np.isfinite(v) else round(float(v), 2) for v in combined[::4]
            ],
        }
    else:
        combined_series = {"t": [], "wrist_speed": []}

    report = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "title": title or "网球挥拍测评报告 2.0",
        "app_version": "2.0",
        "source_name": video_path.name,
        "view": view,
        "view_label": "背面" if view == "back" else "侧面",
        "handedness": handed,
        "handedness_label": "右手持拍" if handed == "right" else "左手持拍",
        "duration_s": round(float(t_arr[-1]), 2),
        "n_frames": int(len(ts)),
        "fps": round(float(fps), 2),
        "detect_rate": round(detect_rate, 3),
        "max_seconds": max_seconds,
        "stroke_mode": stroke_mode,
        "tracking": {
            "ball_frames": int(sum(1 for b in ball_xy if b is not None)),
            "racket_frames": int(sum(1 for r in racket_xy if r is not None)),
            "contact_from_ball": int(
                sum(
                    1
                    for c in clips
                    for s in c.get("swings") or []
                    if s.get("contact_source") in ("ball_racket", "ball_wrist")
                )
            ),
        },
        "overlay_video": overlay_pub["url"],
        "overlay_oss_key": overlay_pub["key"],
        "cover_image": cover_pub["url"],
        "cover_oss_key": cover_pub["key"],
        "preview": preview_pub["url"],
        "preview_oss_key": preview_pub["key"],
        "overall": {
            "score": int(overall_scores["综合"]),
            "grade": grade,
            "grade_label": grade_label,
            "n_swings": int(overall_n),
            "scores": overall_scores,
        },
        "series": combined_series,
        "timeline": timeline,
        "clips": clips,
        "caveats": all_caveats
        or [
            "没有识别到完整挥拍。请换一段人更清楚、挥拍更完整的录像再试。",
        ],
    }

    _emit(
        progress,
        step=5,
        step_name="教练点评",
        progress=88,
        message="正在生成教练点评…",
    )
    report = enrich_with_cursor(report, kf_dir, progress=progress)

    _emit(progress, step=5, step_name="教练点评", progress=96, message="正在整理报告…")
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    _emit(progress, step=5, step_name="教练点评", progress=100, message="分析完成")
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a tennis training video")
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "outputs" / "session")
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--stroke", choices=["auto", "forehand", "backhand"], default="auto")
    args = parser.parse_args()

    def cb(d):
        print(f"[{d.get('progress', 0):3d}%] {d.get('step_name', '')} {d.get('message', '')}", flush=True)

    report = analyze_video(
        args.video,
        args.out,
        max_seconds=args.max_seconds,
        stroke_mode=args.stroke,
        progress=cb,
    )
    print("swings", report["overall"]["n_swings"], "score", report["overall"]["score"])


if __name__ == "__main__":
    main()
