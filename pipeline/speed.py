"""Convert 2D pixel motion to km/h using body / racket scale."""

from __future__ import annotations

import numpy as np

from pipeline.contact import interpolate_xy, racket_head_xy
from pipeline.pose import KPT

# 成人肩髋约 50cm；手腕到拍头中心约 62cm（整拍约 68.5cm）
_SHOULDER_HIP_M = 0.50
_RACKET_WRIST_TO_HEAD_M = 0.62
_STATURE_M = 1.70
_WRIST_TO_HEAD = 1.38


def _pt(pose_xy, pose_conf, name, min_conf=0.32):
    i = KPT[name]
    if pose_conf[i] < min_conf:
        return None
    return np.asarray(pose_xy[i], dtype=np.float64)


def _mid(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def px_to_kmh(px_per_s: float | None, m_per_px: float | None) -> float | None:
    if px_per_s is None or m_per_px is None:
        return None
    if not np.isfinite(px_per_s) or not np.isfinite(m_per_px) or m_per_px <= 0:
        return None
    kmh = float(px_per_s) * float(m_per_px) * 3.6
    if kmh < 2 or kmh > 220:
        return None
    return round(kmh, 1)


def body_m_per_px(pose_xy, pose_conf, torso_len: float | None = None) -> float | None:
    sh = _mid(_pt(pose_xy, pose_conf, "l_shoulder"), _pt(pose_xy, pose_conf, "r_shoulder"))
    hip = _mid(_pt(pose_xy, pose_conf, "l_hip"), _pt(pose_xy, pose_conf, "r_hip"))
    if sh is not None and hip is not None:
        d = float(np.linalg.norm(sh - hip))
        if d > 18:
            return _SHOULDER_HIP_M / d
    ank = _mid(_pt(pose_xy, pose_conf, "l_ankle"), _pt(pose_xy, pose_conf, "r_ankle"))
    nose = _pt(pose_xy, pose_conf, "nose", 0.25)
    top = nose if nose is not None else sh
    if top is not None and ank is not None:
        d = float(np.linalg.norm(top - ank))
        if d > 40:
            return _STATURE_M * 0.90 / d
    if torso_len is not None and torso_len > 40:
        return 1.20 / float(torso_len)
    return None


def racket_m_per_px(racket_box, wrist) -> float | None:
    if racket_box is None or wrist is None:
        return None
    head = racket_head_xy(racket_box, wrist)
    w = np.asarray(wrist, dtype=np.float64)
    if head is None or not np.isfinite(w).all():
        return None
    L = float(np.linalg.norm(head - w))
    if L < 28:
        return None
    return _RACKET_WRIST_TO_HEAD_M / L


def choose_scale(pose_xy, pose_conf, torso_len, racket_box, wrist) -> float | None:
    rk = racket_m_per_px(racket_box, wrist)
    bd = body_m_per_px(pose_xy, pose_conf, torso_len)
    if rk is not None and bd is not None:
        # 拍长和身体差太多时，相信身体（避免把短框当成整拍）
        if 0.45 * bd <= rk <= 2.2 * bd:
            return rk
        return bd
    return rk or bd


def _xy_at(track, i: int) -> np.ndarray | None:
    if track is None or i < 0:
        return None
    try:
        p = track[i]
    except (IndexError, TypeError, KeyError):
        return None
    if p is None:
        return None
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    if a.size < 2 or not np.isfinite(a[:2]).all():
        return None
    return a[:2]


def track_to_xy(track, n: int | None = None) -> np.ndarray:
    if track is None:
        n = int(n or 0)
        return np.full((n, 2), np.nan)
    if isinstance(track, np.ndarray) and track.ndim == 2 and track.shape[1] >= 2:
        n = n if n is not None else len(track)
        out = np.full((n, 2), np.nan)
        m = min(n, len(track))
        sl = np.asarray(track[:m, :2], dtype=np.float64)
        good = np.isfinite(sl).all(axis=1)
        out[:m][good] = sl[good]
        return out
    n = n if n is not None else len(track)
    out = np.full((n, 2), np.nan)
    for i in range(min(n, len(track))):
        p = _xy_at(track, i)
        if p is not None:
            out[i] = p
    return out


def pairwise_speeds_px(track_xy: np.ndarray, ts: np.ndarray, lo: int, hi: int) -> list[float]:
    """Frame-to-frame speeds in [lo, hi), skipping gaps and tiny jitters."""
    if hi - lo < 2 or track_xy is None or ts is None:
        return []
    vals = []
    prev_i = None
    n = min(len(track_xy), len(ts))
    for i in range(max(0, lo), min(hi, n)):
        if not np.isfinite(track_xy[i]).all():
            continue
        if prev_i is not None:
            dt = float(ts[i] - ts[prev_i])
            if 0.012 <= dt <= 0.20:
                d = float(np.linalg.norm(track_xy[i] - track_xy[prev_i]))
                if d > 1.5:
                    vals.append(d / dt)
        prev_i = i
    return vals


def flight_speed_px(track_xy: np.ndarray, ts: np.ndarray, lo: int, hi: int) -> float | None:
    vals = pairwise_speeds_px(track_xy, ts, lo, hi)
    if not vals:
        return None
    return float(np.median(vals))


def peak_speed_px(track_xy: np.ndarray, ts: np.ndarray, lo: int, hi: int) -> float | None:
    vals = pairwise_speeds_px(track_xy, ts, lo, hi)
    if not vals:
        return None
    return float(np.max(vals))


def racket_head_track(racket_box, wrist_xy, n: int) -> np.ndarray:
    out = np.full((n, 2), np.nan)
    if racket_box is None:
        return out
    for i in range(n):
        box = racket_box[i] if i < len(racket_box) else None
        w = _xy_at(wrist_xy, i)
        head = racket_head_xy(box, w)
        if head is not None and np.isfinite(head).all():
            out[i] = np.asarray(head, dtype=np.float64)[:2]
    return out


def median_m_per_px(
    pose_xy,
    pose_conf,
    torso_len,
    racket_box,
    wrist_xy,
    lo: int,
    hi: int,
    torso_fallback: float | None = None,
) -> float | None:
    scales = []
    n_pose = 0 if pose_xy is None else len(pose_xy)
    for i in range(max(0, lo), max(lo + 1, hi)):
        if pose_xy is None or i >= n_pose:
            continue
        xy = pose_xy[i]
        cf = pose_conf[i] if pose_conf is not None and i < len(pose_conf) else None
        if xy is None or cf is None:
            continue
        tlen = torso_fallback
        if torso_len is not None and i < len(torso_len):
            v = torso_len[i]
            if v is not None and np.isfinite(v) and float(v) > 40:
                tlen = float(v)
        box = racket_box[i] if racket_box is not None and i < len(racket_box) else None
        s = choose_scale(xy, cf, tlen, box, _xy_at(wrist_xy, i))
        if s is not None:
            scales.append(s)
    if not scales:
        return None
    return float(np.median(scales))


def estimate_kmh(
    *,
    m_per_px: float | None,
    wrist_px: float | None,
    racket_px: float | None,
    hip_px: float | None,
    ball_in_px: float | None,
    ball_out_px: float | None,
) -> dict:
    wrist_kmh = px_to_kmh(wrist_px, m_per_px)
    racket_kmh = px_to_kmh(racket_px, m_per_px)
    swing_kmh = racket_kmh
    swing_from = "racket"
    if swing_kmh is None and wrist_px is not None:
        swing_kmh = px_to_kmh(float(wrist_px) * _WRIST_TO_HEAD, m_per_px)
        swing_from = "wrist"
    if swing_kmh is None:
        swing_from = None
    return speeds_dict(
        swing_kmh=swing_kmh,
        wrist_kmh=wrist_kmh,
        hip_kmh=px_to_kmh(hip_px, m_per_px),
        ball_in_kmh=px_to_kmh(ball_in_px, m_per_px),
        ball_out_kmh=px_to_kmh(ball_out_px, m_per_px),
        swing_from=swing_from,
    )


def speeds_dict(
    *,
    swing_kmh,
    wrist_kmh,
    hip_kmh,
    ball_in_kmh,
    ball_out_kmh,
    swing_from=None,
) -> dict:
    return {
        "swing_kmh": swing_kmh,
        "wrist_kmh": wrist_kmh,
        "hip_kmh": hip_kmh,
        "ball_in_kmh": ball_in_kmh,
        "ball_out_kmh": ball_out_kmh,
        "swing_from": swing_from,
    }


def mean_speeds(items: list[dict]) -> dict:
    out = {}
    for key in ("swing_kmh", "wrist_kmh", "hip_kmh", "ball_in_kmh", "ball_out_kmh"):
        xs = [d[key] for d in items if isinstance(d, dict) and d.get(key) is not None]
        out[key] = None if not xs else round(float(np.mean(xs)), 1)
        out[key + "_max"] = None if not xs else round(float(np.max(xs)), 1)
    sources = [d.get("swing_from") for d in items if isinstance(d, dict) and d.get("swing_from")]
    if sources:
        out["swing_from"] = "racket" if sources.count("racket") >= sources.count("wrist") else "wrist"
    else:
        out["swing_from"] = None
    return out


def ball_track_xy(ball_xy, n: int) -> np.ndarray | None:
    if not ball_xy:
        return None
    filled = interpolate_xy(list(ball_xy))
    arr = track_to_xy(filled, n)
    if int(np.isfinite(arr).all(axis=1).sum()) < 3:
        return None
    return arr
