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

WORKFLOW = [
    "读取视频并截取分析时长",
    "YOLOv8-pose 逐帧估计 17 点骨架，并烧入叠加视频",
    "用左右手腕速度峰值检测挥拍，自动区分正手 / 反手",
    "计算肘角、膝角、重心高度比、步幅比、引拍幅度",
    "Cursor Cloud Agent（Grok 4.6 Extra High）结合关键帧与指标写评",
]


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


def _encode_h264(src: Path, dst: Path) -> bool:
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
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _draw_hud(frame: np.ndarray, t: float, ok: bool, done: int, total: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (8, 8), (min(w - 8, 420), 54), (12, 16, 22), -1)
    label = f"YOLO pose  t={t:5.2f}s  {done}/{total}"
    if not ok:
        label += "  miss"
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


def _save_phase(path: Path, view, pose, tag: str) -> None:
    vis = draw_pose(view, pose)
    h, w = vis.shape[:2]
    scale = 560 / max(w, 1)
    vis = cv2.resize(vis, (560, int(h * scale)))
    cv2.putText(
        vis, tag, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def _emit(cb: ProgressCb | None, **payload) -> None:
    if cb:
        cb(payload)


def analyze_video(
    video_path: Path,
    out_dir: Path,
    *,
    max_seconds: float = 90.0,
    stroke_mode: str = "auto",
    title: str | None = None,
    media_url_base: str = "",
    estimator: PoseEstimator | None = None,
    progress: ProgressCb | None = None,
) -> dict:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kf_dir = out_dir / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)

    base = media_url_base.rstrip("/")

    def media(name: str) -> str:
        return f"{base}/{name}" if base else name

    _emit(progress, step=1, step_name="读取视频", progress=3, message="打开视频文件…")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 544)
    max_frames = int(max_seconds * fps) if max_seconds > 0 else n_total
    if n_total > 0:
        max_frames = min(max_frames, n_total)
    target = max(1, max_frames)

    est = estimator or get_estimator()
    _emit(
        progress,
        step=2,
        step_name="姿态估计",
        progress=8,
        message=f"加载模型完成，开始逐帧估计（最多 {target} 帧）…",
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
    buf: list[np.ndarray] = []
    batch = 4 if getattr(est, "device", "cpu") == "cpu" else 8
    preview_path = out_dir / "preview.jpg"
    i = 0

    def flush():
        nonlocal i
        if not buf:
            return
        poses = est.infer_batch(buf)
        for frame, pose in zip(buf, poses):
            vis = _draw_hud(draw_pose(frame, pose), i / fps, pose.ok, i + 1, target)
            writer.write(vis)
            xy_list.append(pose.xy)
            conf_list.append(pose.conf)
            ok_flags.append(bool(pose.ok))
            ts.append(i / fps)
            if (i % 18) == 0:
                cv2.imwrite(str(preview_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                _emit(
                    progress,
                    step=2,
                    step_name="姿态估计",
                    progress=8 + int(62 * (i + 1) / target),
                    message=f"已处理 {i + 1} / {target} 帧",
                    preview=True,
                    detected_pct=round(100 * sum(ok_flags) / max(len(ok_flags), 1), 1),
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
    _emit(progress, step=2, step_name="姿态估计", progress=72, message="正在编码叠加视频…")
    if not _encode_h264(raw_path, overlay_path):
        overlay_path.write_bytes(raw_path.read_bytes())
    try:
        raw_path.unlink(missing_ok=True)
    except OSError:
        pass

    if not ts:
        raise RuntimeError("视频没有可读帧")

    t_arr = np.array(ts, dtype=np.float64)
    detect_rate = float(np.mean(ok_flags)) if ok_flags else 0.0
    view = infer_view(xy_list, conf_list)
    xy_list, conf_list = _fix_backview_laterality(xy_list, conf_list, view)
    handed = infer_handedness(xy_list, conf_list, t_arr, fps)
    takeback_mode = "distance"
    enable_late = view == "side"

    _emit(progress, step=3, step_name="检测挥拍", progress=78, message="根据手腕速度检测挥拍峰值…")

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

    _emit(
        progress,
        step=4,
        step_name="计算指标",
        progress=84,
        message=f"检出 {len(peaks)} 次挥拍，正在计算生物力学指标…",
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
        swings = measure_swings(
            series, fps, peaks=stroke_peaks, enable_late_contact=enable_late
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
            "contact": "击球(速度峰)",
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
                    _save_phase(abs_path, frame, pose, labels_zh[name])
                phases[name] = {
                    "t": round(float(series.t[fi]), 3),
                    "image": media(f"keyframes/{fname}"),
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
        for key in clips[0]["scores"]:
            overall_scores[key] = int(
                round(sum(c["scores"][key] * w for c, w in zip(clips, weights)) / sum(weights))
            )
        overall_n = total_swings
    else:
        overall_scores = {
            "综合": 0,
            "重心": 0,
            "动力链": 0,
            "动作框架": 0,
            "步伐": 0,
            "手腕": 0,
        }
        overall_n = 0
        total_swings = 0

    grade, grade_label = grade_from_score(int(overall_scores["综合"]))

    cover_rel = "cover.jpg"
    cover_abs = out_dir / cover_rel
    if clips and clips[0]["swings"]:
        img = clips[0]["swings"][0]["phases"]["contact"]["image"]
        # copy first contact frame as cover
        src = kf_dir / Path(img).name
        if src.exists():
            cover_abs.write_bytes(src.read_bytes())
        elif preview_path.exists():
            cover_abs.write_bytes(preview_path.read_bytes())
    elif preview_path.exists():
        cover_abs.write_bytes(preview_path.read_bytes())

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
        "title": title or "网球挥拍测评报告",
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
        "overlay_video": media("overlay.mp4"),
        "cover_image": media(cover_rel) if cover_abs.exists() else media("preview.jpg"),
        "preview": media("preview.jpg"),
        "workflow": WORKFLOW,
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
            "未检出有效挥拍。请确认画面中球员清晰、挥拍完整，或加长分析时长。",
        ],
    }

    try:
        _emit(
            progress,
            step=5,
            step_name="云端点评",
            progress=88,
            message="准备关键帧，调用 Cursor Cloud Agent…",
        )
        report = enrich_with_cursor(report, kf_dir, progress=progress)
    except Exception as exc:
        report.setdefault("coach", {})
        report["coach"]["status"] = "error"
        report["coach"]["message"] = f"云端点评失败，已回退规则引擎：{exc}"

    _emit(progress, step=5, step_name="生成报告", progress=96, message="写入报告文件…")
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    _emit(progress, step=5, step_name="生成报告", progress=100, message="分析完成")
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a tennis training video")
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "outputs" / "session")
    parser.add_argument("--max-seconds", type=float, default=90)
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
