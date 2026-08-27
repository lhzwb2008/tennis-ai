"""Swing events, biomechanical metrics, and rule-based coaching text."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.pose import KPT

# 2.0 四维评分（合计 100）
SCORE_AXES = (
    ("重心", 25),
    ("击球点", 20),
    ("动力链", 30),
    ("击球效果", 25),
)


def _pt(pose_xy, pose_conf, name, min_conf=0.35):
    i = KPT[name]
    if pose_conf[i] < min_conf:
        return None
    return pose_xy[i]


def _mid(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2


def angle_deg(a, b, c) -> float | None:
    if a is None or b is None or c is None:
        return None
    ba = a - b
    bc = c - b
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = np.clip(np.dot(ba, bc) / (n1 * n2), -1, 1)
    return float(np.degrees(np.arccos(cos)))


def smooth(x: np.ndarray, win: int = 5) -> np.ndarray:
    if len(x) < win:
        return x
    k = np.ones(win) / win
    pad = win // 2
    y = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(y, k, mode="valid")[: len(x)]


@dataclass
class ClipSeries:
    t: np.ndarray
    wrist_speed: np.ndarray
    elbow: np.ndarray
    knee: np.ndarray
    cog_ratio: np.ndarray
    stance: np.ndarray
    takeback: np.ndarray
    hitting: str  # "right" or "left"
    wrist_xy: np.ndarray | None = None
    hip_xy: np.ndarray | None = None
    reach_x: np.ndarray | None = None
    torso_len: np.ndarray | None = None
    hip_speed: np.ndarray | None = None
    shoulder_speed: np.ndarray | None = None
    elbow_speed: np.ndarray | None = None


def hitting_side(side_xy: list[np.ndarray], side_conf: list[np.ndarray], stroke: str) -> str:
    """Infer which wrist is the hitting arm from motion energy."""
    speeds = {"left": 0.0, "right": 0.0}
    for name, key in (("left", "l_wrist"), ("right", "r_wrist")):
        pts = []
        for xy, conf in zip(side_xy, side_conf):
            p = _pt(xy, conf, key)
            pts.append(p if p is not None else np.array([np.nan, np.nan]))
        arr = np.vstack(pts)
        d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        speeds[name] = float(np.nanmean(np.nan_to_num(d, nan=0)))
    if stroke == "backhand":
        # 双手反手通常非持拍手（左手，对右手持拍者）运动也很大；取更快的一侧
        return "left" if speeds["left"] >= speeds["right"] * 0.9 else "right"
    return "right" if speeds["right"] >= speeds["left"] else "left"


def infer_handedness(
    xy_list: list[np.ndarray],
    conf_list: list[np.ndarray],
    ts: np.ndarray,
    fps: float,
) -> str:
    """Vote with one-handed swings so 2HBH does not flip the holding hand."""
    l_spd = speed_from_xy(wrist_track(xy_list, conf_list, "l_wrist"), ts)
    r_spd = speed_from_xy(wrist_track(xy_list, conf_list, "r_wrist"), ts)
    peaks = detect_peaks(ts, np.maximum(l_spd, r_spd))
    votes = {"left": 0.0, "right": 0.0}
    half = max(4, int(0.28 * fps))
    for p in peaks:
        lo, hi = max(0, p - half), min(len(ts), p + max(3, int(0.12 * fps)))
        l = float(np.mean(l_spd[lo:hi])) if hi > lo else 0.0
        r = float(np.mean(r_spd[lo:hi])) if hi > lo else 0.0
        if r > l * 1.28:
            votes["right"] += 1.0
        elif l > r * 1.28:
            votes["left"] += 1.0
    if votes["left"] == 0 and votes["right"] == 0:
        return "right" if float(np.mean(r_spd)) >= float(np.mean(l_spd)) else "left"
    return "right" if votes["right"] >= votes["left"] else "left"


def infer_view(xy_list: list[np.ndarray], conf_list: list[np.ndarray]) -> str:
    """back ≈ facing camera/away; side ≈ profile. Uses shoulder width / torso height."""
    ratios = []
    for xy, conf in zip(xy_list, conf_list):
        ls, rs = _pt(xy, conf, "l_shoulder"), _pt(xy, conf, "r_shoulder")
        hip = _mid(_pt(xy, conf, "l_hip"), _pt(xy, conf, "r_hip"))
        sh = _mid(ls, rs)
        if ls is None or rs is None or hip is None or sh is None:
            continue
        height = abs(float(hip[1] - sh[1]))
        width = abs(float(ls[0] - rs[0]))
        if height > 8:
            ratios.append(width / height)
    if not ratios:
        return "back"
    return "back" if float(np.median(ratios)) >= 0.36 else "side"


def wrist_track(xy_list, conf_list, name: str) -> np.ndarray:
    pts = []
    for xy, conf in zip(xy_list, conf_list):
        p = _pt(xy, conf, name)
        pts.append(p if p is not None else np.array([np.nan, np.nan]))
    return np.vstack(pts)


def speed_from_xy(xy: np.ndarray, ts: np.ndarray) -> np.ndarray:
    n = len(ts)
    dt = np.diff(ts, prepend=ts[0] + (ts[1] - ts[0] if n > 1 else 1 / 30))
    dt[dt <= 0] = 1 / 30
    d = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=1)
    d = np.nan_to_num(d, nan=0.0)
    return smooth(d / dt, 5)


def detect_peaks(t: np.ndarray, speed: np.ndarray, min_gap_s: float = 1.05) -> list[int]:
    s = np.asarray(speed, dtype=np.float64)
    valid = np.isfinite(s)
    if valid.sum() < 10:
        return []
    baseline = np.percentile(s[valid], 55)
    peak_th = max(np.percentile(s[valid], 82), baseline * 1.8)
    idxs: list[int] = []
    i = 1
    n = len(s)
    while i < n - 1:
        if s[i] >= peak_th and s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            if not idxs or (t[i] - t[idxs[-1]]) >= min_gap_s:
                idxs.append(i)
            elif s[i] > s[idxs[-1]]:
                idxs[-1] = i
            i += 3
            continue
        i += 1
    return [j for j in idxs if s[j] >= peak_th * 0.85]


def classify_swing(
    xy_list: list[np.ndarray],
    conf_list: list[np.ndarray],
    ts: np.ndarray,
    peak_i: int,
    handed: str,
    fps: float,
) -> str:
    """Righty: FH if right wrist dominates; BH if left wrist joins (two-handed)."""
    half = max(4, int(0.28 * fps))
    lo, hi = max(0, peak_i - half), min(len(ts), peak_i + max(3, int(0.12 * fps)))
    l_xy = wrist_track(xy_list[lo:hi], conf_list[lo:hi], "l_wrist")
    r_xy = wrist_track(xy_list[lo:hi], conf_list[lo:hi], "r_wrist")
    l_s = float(np.mean(speed_from_xy(l_xy, ts[lo:hi])))
    r_s = float(np.mean(speed_from_xy(r_xy, ts[lo:hi])))
    # 正手切削时非持拍手也会摆动，阈值过低会被误判成反手。
    if handed == "right":
        return "backhand" if l_s > r_s * 1.05 and l_s > 55 else "forehand"
    return "backhand" if r_s > l_s * 1.05 and r_s > 55 else "forehand"


def build_series(
    ts: np.ndarray,
    side_xy: list[np.ndarray],
    side_conf: list[np.ndarray],
    back_xy: list[np.ndarray],
    back_conf: list[np.ndarray],
    stroke: str,
    takeback_mode: str = "signed_x",
    hitting: str | None = None,
) -> ClipSeries:
    hit = hitting or hitting_side(side_xy, side_conf, stroke)
    w_name = "r_wrist" if hit == "right" else "l_wrist"
    e_name = "r_elbow" if hit == "right" else "l_elbow"
    s_name = "r_shoulder" if hit == "right" else "l_shoulder"

    n = len(ts)
    wrist = np.full((n, 2), np.nan)
    hip_xy = np.full((n, 2), np.nan)
    sh_xy = np.full((n, 2), np.nan)
    el_xy = np.full((n, 2), np.nan)
    elbow = np.full(n, np.nan)
    knee = np.full(n, np.nan)
    cog = np.full(n, np.nan)
    stance = np.full(n, np.nan)
    takeback = np.full(n, np.nan)
    reach_x = np.full(n, np.nan)
    torso_len = np.full(n, np.nan)

    for i, (xy, conf, bxy, bconf) in enumerate(zip(side_xy, side_conf, back_xy, back_conf)):
        w = _pt(xy, conf, w_name)
        sh = _pt(xy, conf, s_name)
        el = _pt(xy, conf, e_name)
        hip = _mid(_pt(xy, conf, "l_hip"), _pt(xy, conf, "r_hip"))
        l_knee, r_knee = _pt(xy, conf, "l_knee"), _pt(xy, conf, "r_knee")
        l_ank, r_ank = _pt(xy, conf, "l_ankle"), _pt(xy, conf, "r_ankle")
        l_hip, r_hip = _pt(xy, conf, "l_hip"), _pt(xy, conf, "r_hip")
        if w is not None:
            wrist[i] = w
        if hip is not None:
            hip_xy[i] = hip
        if sh is not None:
            sh_xy[i] = sh
        if el is not None:
            el_xy[i] = el
        ang = angle_deg(sh, el, w)
        if ang is not None:
            elbow[i] = ang
        knees = [angle_deg(l_hip, l_knee, l_ank), angle_deg(r_hip, r_knee, r_ank)]
        knees = [k for k in knees if k is not None]
        if knees:
            knee[i] = min(knees)  # more flexed
        sh_mid = _mid(_pt(xy, conf, "l_shoulder"), _pt(xy, conf, "r_shoulder"))
        ank_mid = _mid(l_ank, r_ank)
        torso = None
        if hip is not None and sh_mid is not None and ank_mid is not None:
            torso = float(np.linalg.norm(sh_mid - ank_mid))
            if torso > 1:
                torso_len[i] = torso
                cog[i] = float(np.linalg.norm(hip - ank_mid) / torso)
        if w is not None and hip is not None:
            reach_x[i] = float(w[0] - hip[0])
            if takeback_mode == "distance":
                dist = float(np.linalg.norm(w - hip))
                takeback[i] = dist / torso if torso and torso > 1 else dist
            else:
                # 侧面：x 越大越靠画面右侧。网在左侧时，引拍 = 手腕相对髋更靠右
                takeback[i] = float(w[0] - hip[0])

        la = _pt(bxy, bconf, "l_ankle")
        ra = _pt(bxy, bconf, "r_ankle")
        ls = _pt(bxy, bconf, "l_shoulder")
        rs = _pt(bxy, bconf, "r_shoulder")
        if la is not None and ra is not None and ls is not None and rs is not None:
            sw = abs(la[0] - ra[0])
            shw = abs(ls[0] - rs[0])
            if shw > 1:
                stance[i] = float(sw / shw)

    speed = speed_from_xy(wrist, ts)

    return ClipSeries(
        t=ts,
        wrist_speed=speed,
        elbow=elbow,
        knee=knee,
        cog_ratio=cog,
        stance=stance,
        takeback=takeback,
        hitting=hit,
        wrist_xy=wrist,
        hip_xy=hip_xy,
        reach_x=reach_x,
        torso_len=torso_len,
        hip_speed=speed_from_xy(hip_xy, ts),
        shoulder_speed=speed_from_xy(sh_xy, ts),
        elbow_speed=speed_from_xy(el_xy, ts),
    )


def detect_swings(series: ClipSeries, min_gap_s: float = 1.05) -> list[int]:
    return detect_peaks(series.t, series.wrist_speed, min_gap_s)


def _val_at(arr: np.ndarray | None, idx: int, lo: int, hi: int, how: str) -> float | None:
    if arr is None:
        return None
    sl = arr[lo:hi]
    sl = sl[np.isfinite(sl)]
    if sl.size == 0:
        return None
    if how == "mean":
        return float(np.mean(sl))
    if how == "min":
        return float(np.min(sl))
    if how == "max":
        return float(np.max(sl))
    if how == "at":
        v = arr[idx]
        return None if not np.isfinite(v) else float(v)
    return float(np.median(sl))


@dataclass
class SwingMetrics:
    contact_i: int
    contact_t: float
    ready_i: int
    takeback_i: int
    follow_i: int
    elbow_contact: float | None
    knee_contact: float | None
    cog_ready: float | None
    cog_stable: float | None
    stance_ready: float | None
    takeback_extent: float | None
    wrist_speed: float | None
    racket_speed: float | None
    contact_forward: float | None
    chain_order: float | None
    path_lift: float | None
    face_vert: float | None
    late_contact: bool
    contact_source: str = "wrist"
    shot_kind: str = "topspin"


def _arr_at(arr: np.ndarray | None, i: int) -> np.ndarray | None:
    if arr is None or i < 0 or i >= len(arr):
        return None
    v = arr[i]
    if v is None:
        return None
    x = np.asarray(v, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        return None
    return x


def _peak_index(arr: np.ndarray | None, lo: int, hi: int) -> int | None:
    if arr is None or hi <= lo:
        return None
    sl = arr[lo:hi]
    if not np.isfinite(sl).any():
        return None
    filled = np.where(np.isfinite(sl), sl, -1e9)
    return lo + int(np.argmax(filled))


def refine_contact_index(
    peak_i: int,
    fps: float,
    ball_xy: list,
    racket_xy: list,
    wrist_xy: np.ndarray | None = None,
    max_dist: float = 140.0,
) -> tuple[int, str]:
    """Move the hit frame to where the ball is closest to the racket (or wrist)."""
    n = len(ball_xy)
    if n == 0:
        return peak_i, "wrist"
    lo = max(0, peak_i - int(0.10 * fps))
    hi = min(n - 1, peak_i + int(0.42 * fps))
    best: tuple[float, int, str] | None = None
    for i in range(lo, hi + 1):
        b = ball_xy[i]
        r = racket_xy[i] if i < len(racket_xy) else None
        w = None
        if wrist_xy is not None and i < len(wrist_xy) and np.isfinite(wrist_xy[i]).all():
            w = wrist_xy[i]
        if b is None:
            continue
        if r is not None:
            d = float(np.linalg.norm(b - r))
            src = "ball_racket"
        elif w is not None:
            d = float(np.linalg.norm(b - w))
            src = "ball_wrist"
        else:
            continue
        if best is None or d < best[0]:
            best = (d, i, src)
    if best is None or best[0] > max_dist:
        return peak_i, "wrist"
    return best[1], best[2]


def measure_swings(
    series: ClipSeries,
    fps: float,
    peaks: list[int] | None = None,
    enable_late_contact: bool = True,
    ball_xy: list | None = None,
    racket_xy: list | None = None,
    wrist_xy: np.ndarray | None = None,
    racket_box: list | None = None,
    view: str = "side",
) -> list[SwingMetrics]:
    peaks = detect_swings(series) if peaks is None else peaks
    out: list[SwingMetrics] = []
    n = len(series.t)
    hitting = series.hitting
    for p in peaks:
        source = "wrist"
        if ball_xy is not None:
            p, source = refine_contact_index(
                p, fps, ball_xy, racket_xy or [None] * len(ball_xy), wrist_xy=wrist_xy
            )
        pre = max(0, p - int(0.55 * fps))
        ready = max(0, p - int(0.40 * fps))
        follow = min(n - 1, p + int(0.28 * fps))
        tb_slice = series.takeback[pre:p]
        if np.isfinite(tb_slice).any():
            takeback_i = pre + int(np.nanargmax(np.nan_to_num(tb_slice, nan=-1e9)))
        else:
            takeback_i = max(0, p - int(0.22 * fps))

        torso = _val_at(series.torso_len, p, pre, follow, "median")
        if torso is None or torso < 1:
            torso = 160.0

        hip = _arr_at(series.hip_xy, p)
        contact_pt = None
        if racket_xy is not None and p < len(racket_xy) and racket_xy[p] is not None:
            cand = np.asarray(racket_xy[p], dtype=np.float64)
            if np.isfinite(cand).all():
                contact_pt = cand
        if contact_pt is None:
            contact_pt = _arr_at(series.wrist_xy, p)
        if contact_pt is None and wrist_xy is not None:
            contact_pt = _arr_at(wrist_xy, p)

        load_x = _val_at(series.reach_x, takeback_i, takeback_i, takeback_i + 1, "at") if series.reach_x is not None else None
        load_sign = 1.0 if (load_x is None or load_x >= 0) else -1.0
        contact_forward = None
        if hip is not None and contact_pt is not None:
            offset_x = float(contact_pt[0] - hip[0])
            if view == "back":
                side = 1.0 if hitting == "right" else -1.0
                contact_forward = float(side * offset_x / torso)
            else:
                contact_forward = float(-offset_x * load_sign / torso)

        late = False
        if enable_late_contact and contact_forward is not None:
            late = contact_forward < 0.04

        cog_win = series.cog_ratio[ready:p + 1]
        cog_finite = cog_win[np.isfinite(cog_win)]
        cog_stable = float(np.std(cog_finite)) if cog_finite.size >= 3 else None

        chain_lo = takeback_i
        chain_hi = min(n, p + max(2, int(0.08 * fps)))
        t_hip = _peak_index(series.hip_speed, chain_lo, chain_hi)
        t_sh = _peak_index(series.shoulder_speed, chain_lo, chain_hi)
        t_el = _peak_index(series.elbow_speed, chain_lo, chain_hi)
        t_wr = _peak_index(series.wrist_speed, chain_lo, chain_hi)
        chain_order = None
        peaks_i = [t_hip, t_sh, t_el, t_wr]
        if all(x is not None for x in peaks_i):
            pairs = 0
            for a, b in zip(peaks_i, peaks_i[1:]):
                if a <= b + max(1, int(0.04 * fps)):
                    pairs += 1
            chain_order = pairs / 3.0
            if t_wr < t_hip - max(2, int(0.06 * fps)):
                chain_order = min(chain_order, 0.35)

        r_speed = None
        if racket_xy:
            r_track = np.vstack(
                [
                    np.array(r, dtype=np.float64) if r is not None else np.array([np.nan, np.nan])
                    for r in racket_xy
                ]
            )
            r_spd = speed_from_xy(r_track, series.t)
            r_speed = _val_at(r_spd, p, max(0, p - 1), min(n, p + 2), "max")

        path_lift = None
        src_xy = series.wrist_xy
        if racket_xy:
            rt = np.vstack(
                [
                    np.array(r, dtype=np.float64) if r is not None else np.array([np.nan, np.nan])
                    for r in racket_xy
                ]
            )
            if np.isfinite(rt[max(0, p - 2) : min(n, p + 3)]).all(axis=1).sum() >= 3:
                src_xy = rt
        if src_xy is not None and p >= 1:
            a = _arr_at(src_xy, max(0, p - 2))
            b = _arr_at(src_xy, min(n - 1, p + 1))
            if a is not None and b is not None:
                vel = b - a
                mag = float(np.linalg.norm(vel))
                if mag > 1:
                    path_lift = float(np.clip(-vel[1] / mag, -1, 1))

        face_vert = None
        if racket_box and p < len(racket_box) and racket_box[p] is not None:
            box = np.asarray(racket_box[p], dtype=np.float64)
            if box.size >= 4:
                bw = max(1.0, float(box[2] - box[0]))
                bh = max(1.0, float(box[3] - box[1]))
                face_vert = float(bh / (bw + bh))

        lift_v = None if path_lift is None else round(path_lift, 3)
        shot_kind = "slice" if (lift_v is not None and lift_v < 0.08) else "topspin"
        out.append(
            SwingMetrics(
                contact_i=p,
                contact_t=float(series.t[p]),
                ready_i=ready,
                takeback_i=takeback_i,
                follow_i=follow,
                elbow_contact=_val_at(series.elbow, p, p, min(n, p + 2), "at"),
                knee_contact=_val_at(series.knee, p, max(0, p - 2), min(n, p + 2), "min"),
                cog_ready=_val_at(series.cog_ratio, ready, ready, p, "mean"),
                cog_stable=None if cog_stable is None else round(cog_stable, 4),
                stance_ready=_val_at(series.stance, ready, ready, p, "mean"),
                takeback_extent=_val_at(series.takeback, takeback_i, pre, p, "max"),
                wrist_speed=_val_at(series.wrist_speed, p, p, p + 1, "at"),
                racket_speed=r_speed,
                contact_forward=None if contact_forward is None else round(contact_forward, 3),
                chain_order=None if chain_order is None else round(float(chain_order), 3),
                path_lift=lift_v,
                face_vert=None if face_vert is None else round(face_vert, 3),
                late_contact=late,
                contact_source=source,
                shot_kind=shot_kind,
            )
        )
    return out


def summarize(swings: list[SwingMetrics], takeback_is_ratio: bool = False) -> dict:
    def mean(key):
        xs = [getattr(s, key) for s in swings if getattr(s, key) is not None]
        digits = 3 if takeback_is_ratio and key == "takeback_extent" else 2
        return None if not xs else round(float(np.mean(xs)), digits)

    tb = mean("takeback_extent")
    out = {
        "n_swings": len(swings),
        "elbow_contact_deg": mean("elbow_contact"),
        "knee_contact_deg": mean("knee_contact"),
        "cog_ratio": mean("cog_ready"),
        "cog_stable": mean("cog_stable"),
        "stance_ratio": mean("stance_ready"),
        "takeback_px": None if takeback_is_ratio else tb,
        "takeback_ratio": tb if takeback_is_ratio else None,
        "wrist_speed": mean("wrist_speed"),
        "racket_speed": mean("racket_speed"),
        "contact_forward": mean("contact_forward"),
        "chain_order": mean("chain_order"),
        "path_lift": mean("path_lift"),
        "face_vert": mean("face_vert"),
        "late_contact_rate": float(
            round(sum(1 for s in swings if s.late_contact) / max(len(swings), 1), 2)
        ),
    }
    return out


def _takeback_tier(summary: dict) -> str:
    ratio = summary.get("takeback_ratio")
    px = summary.get("takeback_px") or 0
    if ratio is not None:
        if ratio >= 0.50:
            return "good"
        if ratio >= 0.32:
            return "ok"
        return "shallow"
    if px >= 90:
        return "good"
    if px >= 45:
        return "ok"
    return "shallow"


def score_and_write(
    stroke: str,
    summary: dict,
    *,
    view: str = "side",
    source: str = "overlay",
) -> dict:
    """Map measured stats to 2.0 four-axis scores + coaching text."""
    n = summary["n_swings"] or 1
    cog = summary.get("cog_ratio") or 0.55
    stable = summary.get("cog_stable")
    stance = summary.get("stance_ratio") or 1.2
    late = summary.get("late_contact_rate") or 0
    elbow = summary.get("elbow_contact_deg")
    knee = summary.get("knee_contact_deg")
    speed = summary.get("racket_speed") or summary.get("wrist_speed") or 0
    forward = summary.get("contact_forward")
    chain = summary.get("chain_order")
    lift = summary.get("path_lift")
    face = summary.get("face_vert")
    tb_tier = _takeback_tier(summary)
    ratio = summary.get("takeback_ratio")

    # 重心 25：越低越好 + 越稳越好
    height_pts = float(np.clip(16 - (cog - 0.47) * 90, 4, 16))
    if knee is not None:
        if knee > 165:
            height_pts = min(height_pts, 8)
        elif knee > 155:
            height_pts = min(height_pts, 12)
        elif 125 <= knee <= 150:
            height_pts = min(16, height_pts + 1)
    if stable is None:
        stab_pts = 6.0
    else:
        stab_pts = float(np.clip(9 - stable * 80, 2, 9))
    cog_score = int(np.clip(round(height_pts + stab_pts), 6, 25))

    # 击球点 20：应在身体侧前方，不能贴身偏晚
    if forward is None:
        contact_pts = 11.0 if late < 0.4 else 7.0
    elif 0.10 <= forward <= 0.42:
        contact_pts = 18.0
    elif 0.04 <= forward < 0.10 or 0.42 < forward <= 0.55:
        contact_pts = 14.0
    elif forward > 0.55:
        contact_pts = 11.0
    else:
        contact_pts = 7.0
    if late >= 0.45:
        contact_pts = min(contact_pts, 9)
    elif late >= 0.25:
        contact_pts = min(contact_pts, 13)
    contact_score = int(np.clip(round(contact_pts), 4, 20))

    # 动力链 30：髋→肩→肘→腕；甩胳膊会直接扣分（伤病来源）
    if chain is None:
        seq_pts = 12.0
    else:
        seq_pts = 6 + chain * 16
    if tb_tier == "good":
        tb_pts = 7.0
    elif tb_tier == "ok":
        tb_pts = 5.0
    else:
        tb_pts = 2.0
    knee_pts = 4.0
    if knee is not None and knee > 162:
        knee_pts = 1.0
        seq_pts = min(seq_pts, 12)
    chain_score = int(np.clip(round(seq_pts + tb_pts + knee_pts), 8, 30))

    # 击球效果 25：拍头速度 + 拍面/轨迹（旋转）
    speed_pts = float(np.clip(speed / 28.0, 4, 15))
    spin_pts = 5.0
    if lift is not None:
        if 0.18 <= lift <= 0.75:
            spin_pts += 3.5
        elif 0.05 <= lift < 0.18:
            spin_pts += 1.5
        elif lift < 0:
            spin_pts -= 1.5
    if face is not None:
        if 0.52 <= face <= 0.78:
            spin_pts += 1.5
        elif face < 0.38:
            spin_pts -= 1.0
    spin_pts = float(np.clip(spin_pts, 2, 10))
    effect_score = int(np.clip(round(speed_pts + spin_pts), 6, 25))

    scores = {
        "综合": int(cog_score + contact_score + chain_score + effect_score),
        "重心": cog_score,
        "击球点": contact_score,
        "动力链": chain_score,
        "击球效果": effect_score,
    }

    strengths, problems, drills = [], [], []
    if n >= 8:
        strengths.append(f"连续喂球下识别到约 {n} 次有效挥拍，动作重复性可用。")
    elif n >= 4:
        strengths.append(f"识别到 {n} 次挥拍，样本偏少，结论按趋势看即可。")
    else:
        problems.append("有效挥拍样本过少，以下判断置信度偏低。")

    view_word = "背面" if view == "back" else "侧面"
    if stroke == "backhand":
        strengths.append(f"{view_word}能看到双手反手结构：非持拍手同步参与，不是单手挡球。")
    else:
        if tb_tier != "shallow":
            extra = f"（引拍幅度 / 身高比例 ≈ {ratio:.2f}）" if ratio is not None else ""
            strengths.append(f"{view_word}能看到完整引拍{extra}，不是完全直臂推挡。")
        else:
            problems.append("引拍后摆偏浅，球拍没有充分带到身后，拍头加速会更依赖手臂，动力链容易断。")

    if cog >= 0.56 or (knee is not None and knee > 155):
        problems.append("准备/击球时重心偏高，屈膝加载不够，蹬转空间受限。")
        drills.append(
            "【问题】重心偏高 → 【原因】准备没有屈髋下蹲 → 【训练】无球坐凳准备 8秒×8组，再定点击球要求头肩高度不明显抬起，15球×4组。"
        )
    if stable is not None and stable > 0.06:
        problems.append("重心不稳定：击球前后上下起伏偏大，不是单纯站得高，而是高低在晃。")
        drills.append(
            "【问题】重心不稳 → 【原因】击球时起身过早或步点没踩稳 → 【训练】击球瞬间膝盖保持弯曲，随挥后再站起；连续 15 球头肩高度几乎不变才算过关。"
        )
    if not any("重心" in p for p in problems):
        strengths.append("重心高度和稳定性尚可，没有明显直立挡球。")

    if (forward is not None and forward < 0.06) or late >= 0.4:
        problems.append(
            f"击球点容易偏晚、贴在身体旁边，没有打在身体侧前方（约 {int(late*100)}% 的挥拍有此倾向）。"
        )
        drills.append(
            "【问题】击球点偏晚 → 【原因】引拍完成晚、启动慢 → 【训练】球过网时必须完成引拍；前脚前方放标志物，必须在标志物前击球，15球×4组。"
        )
    elif forward is not None and 0.10 <= forward <= 0.42:
        strengths.append("击球点总体在身体侧前方，没有明显挤在身上。")

    if chain is not None and chain < 0.55:
        problems.append("动力链更像手臂主导：腕或肘先发力，髋肩没有先转，这是伤病高发模式。")
        drills.append(
            "【问题】动力链断裂 → 【原因】手上抢先发力 → 【训练】转体延迟引拍，轻球先转髋再挥臂，20×3 组；感觉肩比手更早动。"
        )
    elif stroke == "forehand" and tb_tier != "good":
        problems.append("正手转体/后摆不足，动力链还偏上肢主导。")
        drills.append(
            "【问题】引拍幅度不足 → 【原因】手臂主动拉拍、肩髋没先转 → 【训练】转体延迟引拍，轻球 20×3 组。"
        )
    elif chain is not None and chain >= 0.75:
        strengths.append("发力顺序比较合理：身体先动，手臂后到。")

    if speed < 180:
        problems.append("拍头速度偏慢，击球威胁不够。")
        drills.append(
            "【问题】球速偏慢 → 【原因】没有用上腿和转体，或击球点太晚只能挡 → 【训练】先把击球点打到身前，再练自抛自打把拍头抽起来，20球×3组。"
        )
    elif speed >= 260:
        strengths.append("挥拍速度够用，能打出一定质量的球。")

    if stroke == "forehand_slice":
        strengths.append("这是正手切削，不是反手；拍头走下切路线。")
        if lift is not None and lift > 0.12:
            problems.append("名义上是切削，但轨迹还在往上刷，削不薄、球容易浮。")
            drills.append(
                "【问题】切削偏浮 → 【原因】还在用上旋的低向高刷 → 【训练】接触点在身侧前方，拍头由高向低送，15 球擦网不过就算过关。"
            )
    elif lift is not None and lift < 0.08:
        problems.append("挥拍轨迹太平、偏向下切，旋转会偏少。若这是切削，应单独按切削练，不要和上旋正手混在一起评价。")
        drills.append(
            "【问题】旋转不足 → 【原因】击球轨迹没有低向高刷 → 【训练】从膝盖高度刷到肩高，强调拍面稳定、轨迹向上，15球×4组。"
        )
    elif lift is not None and 0.2 <= lift <= 0.7:
        strengths.append("挥拍有低向高的轨迹，有利于打出上旋。")

    if stance < 1.25:
        problems.append("准备步幅偏窄，左右开立不够，影响稳定和上步。")
        drills.append(
            "【问题】步幅偏窄 → 【原因】准备站位收着 → 【训练】双脚踩在比肩宽一脚的标志线外准备，20次分腿垫步。"
        )
    if elbow is not None:
        strengths.append(f"击球附近持拍肘角大约 {elbow:.0f}°。")

    if not problems:
        problems.append("没有发现特别明显的问题，建议对照回放再确认击球点和拍面。")
    if not drills:
        drills.append("保持当前框架，增加不同落点的移动击球，再复测重心稳定和击球点。")

    caveats = [
        "本报告根据训练录像自动生成，仅供练习参考，不能替代现场教练。",
        "拍摄角度会影响判断：背面录像较难看清击球点前后位置和拍面开合。",
        "评分来自画面，距离和角度会有一定误差。",
        "能看到球或球拍时，击球画面按球和拍的距离选取；看不到时仍按挥拍动作估计。",
        "旋转根据挥拍轨迹和拍面朝向估计，不是测球的转速。",
    ]

    if stroke == "forehand_slice":
        label = "正手切削"
    elif stroke == "backhand":
        label = "底线反手"
    else:
        label = "底线正手"
    return {
        "label": label,
        "scores": scores,
        "strengths": strengths,
        "problems": problems,
        "drills": drills,
        "caveats": caveats,
    }


def grade_from_score(total: int) -> tuple[str, str]:
    if total >= 80:
        return "A", "优秀"
    if total >= 70:
        return "B", "良好"
    if total >= 60:
        return "C", "及格"
    if total >= 50:
        return "D", "待提高"
    return "E", "需重构"
