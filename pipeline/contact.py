"""Hitting-point geometry, body-relative labels, and overlay drawing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline.pose import KPT

# Poppy：理想击球点 ≈ 胸口高度、持拍侧稍外、身前约 45°
_IDEAL_HEIGHT = 0.02  # 相对胸口：0 为胸口，正值更高（画面上方）
_IDEAL_SIDE = 0.32  # 持拍侧、稍离开身体
_IDEAL_FORWARD = 0.28  # 身前；与侧向接近时大约 45°


def _pt(pose_xy, pose_conf, name, min_conf=0.32):
    i = KPT[name]
    if pose_conf[i] < min_conf:
        return None
    return np.asarray(pose_xy[i], dtype=np.float64)


def _mid(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _finite(p) -> bool:
    if p is None:
        return False
    x = np.asarray(p, dtype=np.float64)
    return x.size >= 2 and bool(np.isfinite(x).all())


def _as2(p) -> np.ndarray | None:
    if not _finite(p):
        return None
    return np.asarray(p, dtype=np.float64).reshape(-1)[:2]


def interpolate_xy(track: list, max_gap: int = 4) -> list:
    """Fill short holes in a ball/racket track so contact search is less jumpy."""
    n = len(track)
    out = [(_as2(p) if _finite(p) else None) for p in track]
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < n and out[j] is None:
            j += 1
        gap = j - i
        left = out[i - 1] if i > 0 else None
        right = out[j] if j < n else None
        if left is not None and right is not None and 1 <= gap <= max_gap:
            for k in range(gap):
                t = (k + 1) / (gap + 1)
                out[i + k] = (1 - t) * left + t * right
        i = j
    return out


def racket_head_xy(box, wrist) -> np.ndarray | None:
    """Head center: the half of the box farther from the wrist (not the handle)."""
    if box is None:
        return None
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    if b.size < 4 or not np.isfinite(b).all():
        return None
    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w = _as2(wrist)
    if w is None:
        return np.array([cx, cy], dtype=np.float64)
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    far = corners[int(np.argmax(np.linalg.norm(corners - w, axis=1)))]
    handle = corners[int(np.argmin(np.linalg.norm(corners - w, axis=1)))]
    # 拍头在远离手腕的一端，用来对准击球帧，不评拍上打点
    return handle + 0.72 * (far - handle)


def chest_xy(pose_xy, pose_conf) -> np.ndarray | None:
    sh = _mid(_pt(pose_xy, pose_conf, "l_shoulder"), _pt(pose_xy, pose_conf, "r_shoulder"))
    hip = _mid(_pt(pose_xy, pose_conf, "l_hip"), _pt(pose_xy, pose_conf, "r_hip"))
    if sh is None:
        return hip
    if hip is None:
        return sh
    return sh + 0.22 * (hip - sh)


def torso_len(pose_xy, pose_conf) -> float | None:
    sh = _mid(_pt(pose_xy, pose_conf, "l_shoulder"), _pt(pose_xy, pose_conf, "r_shoulder"))
    hip = _mid(_pt(pose_xy, pose_conf, "l_hip"), _pt(pose_xy, pose_conf, "r_hip"))
    ank = _mid(_pt(pose_xy, pose_conf, "l_ankle"), _pt(pose_xy, pose_conf, "r_ankle"))
    if sh is not None and ank is not None:
        d = float(np.linalg.norm(sh - ank))
        if d > 8:
            return d
    if sh is not None and hip is not None:
        d = float(np.linalg.norm(sh - hip))
        if d > 8:
            return d * 1.85
    return None


def pick_contact_xy(ball, racket_box, racket_xy, wrist) -> tuple[np.ndarray | None, str]:
    """The hitting point is where the ball is. Fall back to racket head, then wrist."""
    b = _as2(ball)
    if b is not None:
        return b, "ball"
    head = racket_head_xy(racket_box, wrist)
    if head is not None:
        return head, "racket_head"
    r = _as2(racket_xy)
    if r is not None:
        return r, "racket"
    w = _as2(wrist)
    if w is not None:
        return w, "wrist"
    return None, "none"


def refine_contact_index(
    peak_i: int,
    fps: float,
    ball_xy: list,
    racket_xy: list,
    wrist_xy=None,
    racket_box: list | None = None,
    max_dist: float = 140.0,
) -> tuple[int, str]:
    """Hit frame = ball closest to the racket head, with a bonus if the ball turns."""
    n = len(ball_xy)
    if n == 0:
        return peak_i, "wrist"
    lo = max(0, peak_i - int(0.10 * fps))
    hi = min(n - 1, peak_i + int(0.42 * fps))
    best: tuple[float, int, str] | None = None

    def _vel(i0: int, i1: int):
        a, b = _as2(ball_xy[i0]) if 0 <= i0 < n else None, _as2(ball_xy[i1]) if 0 <= i1 < n else None
        if a is None or b is None:
            return None
        return b - a

    for i in range(lo, hi + 1):
        b = _as2(ball_xy[i])
        if b is None:
            continue
        w = None
        if wrist_xy is not None and i < len(wrist_xy) and np.isfinite(wrist_xy[i]).all():
            w = wrist_xy[i]
        box = racket_box[i] if racket_box is not None and i < len(racket_box) else None
        r = racket_xy[i] if i < len(racket_xy) else None
        head = racket_head_xy(box, w)
        if head is not None:
            d = float(np.linalg.norm(b - head))
            src = "ball_racket"
        elif r is not None:
            d = float(np.linalg.norm(b - r))
            src = "ball_racket"
        elif w is not None:
            d = float(np.linalg.norm(b - w))
            src = "ball_wrist"
        else:
            continue
        prev = _vel(i - 2, i)
        nxt = _vel(i, i + 2)
        turn = 0.0
        if prev is not None and nxt is not None:
            np_ = float(np.linalg.norm(prev))
            nn = float(np.linalg.norm(nxt))
            if np_ > 1.5 and nn > 1.5:
                turn = float(-np.dot(prev, nxt) / (np_ * nn))
        score = d - 28.0 * max(0.0, turn)
        if best is None or score < best[0]:
            best = (score, i, src)
    if best is None or best[0] > max_dist:
        return peak_i, "wrist"
    return best[1], best[2]


def _height_label(h: float | None) -> str:
    if h is None:
        return "高度看不清"
    if -0.10 <= h <= 0.12:
        return "胸口高度"
    if h > 0.12:
        return "偏高"
    return "偏低"


def _side_label(s: float | None, hitting: str) -> str:
    if s is None:
        return "左右看不清"
    side = "右侧" if hitting == "right" else "左侧"
    other = "左侧" if hitting == "right" else "右侧"
    if 0.16 <= s <= 0.48:
        return f"{side}稍外"
    if s > 0.48:
        return f"{side}太远"
    if s >= 0.04:
        return f"{side}偏贴身"
    return f"偏{other}/贴身"


def _forward_label(f: float | None) -> str:
    if f is None:
        return "前后看不清"
    if 0.10 <= f <= 0.42:
        return "身前约45°"
    if 0.42 < f <= 0.60:
        return "偏前"
    if 0.04 <= f < 0.10:
        return "稍晚"
    if f > 0.60:
        return "太靠前"
    return "偏晚贴身"


@dataclass
class HitPoint:
    xy: np.ndarray | None = None
    chest: np.ndarray | None = None
    hip: np.ndarray | None = None
    ideal_xy: np.ndarray | None = None
    height: float | None = None
    side: float | None = None
    forward: float | None = None
    angle_deg: float | None = None
    height_label: str = ""
    side_label: str = ""
    forward_label: str = ""
    summary: str = ""
    ok: bool = False
    source: str = "none"
    shoulder_aim: float | None = None
    hand_reaches: bool = False
    both_feet_off: bool = False
    early_step: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        def xy(p):
            if p is None:
                return None
            return [round(float(p[0]), 1), round(float(p[1]), 1)]

        return {
            "xy": xy(self.xy),
            "chest": xy(self.chest),
            "ideal_xy": xy(self.ideal_xy),
            "height": None if self.height is None else round(float(self.height), 3),
            "side": None if self.side is None else round(float(self.side), 3),
            "forward": None if self.forward is None else round(float(self.forward), 3),
            "angle_deg": None if self.angle_deg is None else round(float(self.angle_deg), 1),
            "height_label": self.height_label,
            "side_label": self.side_label,
            "forward_label": self.forward_label,
            "summary": self.summary,
            "ok": self.ok,
            "source": self.source,
            "shoulder_aim": None if self.shoulder_aim is None else round(float(self.shoulder_aim), 3),
            "hand_reaches": bool(self.hand_reaches),
            "both_feet_off": bool(self.both_feet_off),
            "early_step": bool(self.early_step),
            "notes": list(self.notes),
        }


def _lead_names(hitting: str) -> tuple[str, str, str, str]:
    """Non-hitting side is the 'front' shoulder that should face the incoming ball."""
    if hitting == "right":
        return "l_shoulder", "r_shoulder", "l_wrist", "l_ankle"
    return "r_shoulder", "l_shoulder", "r_wrist", "r_ankle"


def measure_hit_point(
    pose_xy,
    pose_conf,
    ball,
    racket_box,
    racket_xy,
    wrist,
    hitting: str,
    view: str,
    load_sign: float = 1.0,
    ball_takeback=None,
    pose_takeback=None,
    conf_takeback=None,
    pose_ready=None,
    conf_ready=None,
    pose_follow=None,
    conf_follow=None,
    enable_forward: bool = True,
) -> HitPoint:
    hp = HitPoint()
    chest = chest_xy(pose_xy, pose_conf)
    hip = _mid(_pt(pose_xy, pose_conf, "l_hip"), _pt(pose_xy, pose_conf, "r_hip"))
    torso = torso_len(pose_xy, pose_conf) or 160.0
    hp.chest, hp.hip = chest, hip

    contact, src = pick_contact_xy(ball, racket_box, racket_xy, wrist)
    hp.xy, hp.source = contact, src

    if contact is not None and chest is not None:
        hp.height = float((chest[1] - contact[1]) / torso)

    if contact is not None and hip is not None:
        offset_x = float(contact[0] - hip[0])
        if view == "back":
            side_sign = 1.0 if hitting == "right" else -1.0
            hp.side = float(side_sign * offset_x / torso)
            # 背面前后方向几乎看不清，不用 x 冒充「身前」
        else:
            hp.forward = float(-offset_x * load_sign / torso)
            # 侧面左右方向几乎看不清

    if hp.side is not None and hp.forward is not None:
        hp.angle_deg = float(np.degrees(np.arctan2(abs(hp.side), max(hp.forward, 1e-3))))

    hp.height_label = _height_label(hp.height)
    hp.side_label = _side_label(hp.side, hitting) if view == "back" else "侧面看不清左右"
    if view == "side" and enable_forward:
        hp.forward_label = _forward_label(hp.forward)
    elif view == "side":
        hp.forward_label = _forward_label(hp.forward)
    else:
        hp.forward_label = "背面看不清前后"

    bits = [hp.height_label]
    if view == "back":
        bits.append(hp.side_label)
    else:
        bits.append(hp.forward_label)
    hp.summary = " · ".join(bits)

    height_ok = hp.height is not None and -0.12 <= hp.height <= 0.14
    if view == "back":
        place_ok = hp.side is not None and 0.14 <= hp.side <= 0.50
    else:
        place_ok = hp.forward is not None and 0.08 <= hp.forward <= 0.48
    hp.ok = bool(height_ok and place_ok)

    if chest is not None and hip is not None:
        ideal = chest.copy()
        if view == "back":
            side_sign = 1.0 if hitting == "right" else -1.0
            ideal[0] = hip[0] + side_sign * _IDEAL_SIDE * torso
            ideal[1] = chest[1] - _IDEAL_HEIGHT * torso
        else:
            ideal[0] = hip[0] - load_sign * _IDEAL_FORWARD * torso
            ideal[1] = chest[1] - _IDEAL_HEIGHT * torso
        hp.ideal_xy = ideal

    _fill_prep_and_feet(
        hp,
        pose_xy,
        pose_conf,
        hitting,
        view,
        load_sign,
        torso,
        ball_takeback=ball_takeback,
        pose_takeback=pose_takeback,
        conf_takeback=conf_takeback,
        pose_ready=pose_ready,
        conf_ready=conf_ready,
        pose_follow=pose_follow,
        conf_follow=conf_follow,
    )
    return hp


def _fill_prep_and_feet(
    hp: HitPoint,
    pose_xy,
    pose_conf,
    hitting: str,
    view: str,
    load_sign: float,
    torso: float,
    *,
    ball_takeback,
    pose_takeback,
    conf_takeback,
    pose_ready,
    conf_ready,
    pose_follow,
    conf_follow,
) -> None:
    lead_sh, trail_sh, lead_wr, lead_ank = _lead_names(hitting)
    notes: list[str] = []

    if pose_takeback is not None and conf_takeback is not None:
        b = _as2(ball_takeback)
        ls = _pt(pose_takeback, conf_takeback, lead_sh)
        ts = _pt(pose_takeback, conf_takeback, trail_sh)
        lw = _pt(pose_takeback, conf_takeback, lead_wr)
        if b is not None and ls is not None and ts is not None:
            sh_line = ls - ts
            to_ball = b - ls
            n1, n2 = float(np.linalg.norm(sh_line)), float(np.linalg.norm(to_ball))
            if n1 > 4 and n2 > 4:
                hp.shoulder_aim = float(np.clip(np.dot(sh_line, to_ball) / (n1 * n2), -1, 1))
        if b is not None and ls is not None and lw is not None:
            d_sh = float(np.linalg.norm(b - ls))
            d_wr = float(np.linalg.norm(b - lw))
            # 左手伸去够球、左肩没对准
            hp.hand_reaches = d_wr + 12 < d_sh * 0.78
        if hp.hand_reaches:
            notes.append("准备时左手伸去够球了，应对准来球的是左肩，不是左手。")
        elif hp.shoulder_aim is not None and hp.shoulder_aim >= 0.35:
            notes.append("准备时前肩能对着来球。")
        elif hp.shoulder_aim is not None and hp.shoulder_aim < 0.15:
            notes.append("准备时肩膀没有对着来球，转体不够。")

    la = _pt(pose_xy, pose_conf, "l_ankle")
    ra = _pt(pose_xy, pose_conf, "r_ankle")
    if la is not None and ra is not None and pose_ready is not None and conf_ready is not None:
        la0 = _pt(pose_ready, conf_ready, "l_ankle")
        ra0 = _pt(pose_ready, conf_ready, "r_ankle")
        if la0 is not None and ra0 is not None:
            # 画面 y 向下：脚踝比准备时明显上移 = 离地。后脚脚尖点地只抬一点，允许。
            lift_l = float(la0[1] - la[1]) / torso
            lift_r = float(ra0[1] - ra[1]) / torso
            hp.both_feet_off = bool(lift_l > 0.085 and lift_r > 0.085)
            if hp.both_feet_off:
                notes.append("击球时双脚同时离地了。后脚可以脚尖点地，但不能两脚一起跳起来。")

    if view == "side" and pose_follow is not None and conf_follow is not None:
        hit_w = "r_wrist" if hitting == "right" else "l_wrist"
        opp_sh = "l_shoulder" if hitting == "right" else "r_shoulder"
        w1 = _pt(pose_follow, conf_follow, hit_w)
        sh1 = _pt(pose_follow, conf_follow, opp_sh)
        a0 = _pt(pose_xy, pose_conf, lead_ank)
        a1 = _pt(pose_follow, conf_follow, lead_ank)
        if w1 is not None and sh1 is not None and a0 is not None and a1 is not None:
            over_shoulder = w1[1] < sh1[1] - 0.02 * torso
            step = float((a1[0] - a0[0]) * (-load_sign)) / torso
            if step > 0.12 and not over_shoulder:
                hp.early_step = True
                notes.append("上步早了：随挥还没过肩，前脚已经迈出去。应先随挥过肩，再上步。")

    hp.notes = notes


def score_hit_point(hits: list[HitPoint], view: str, late_rate: float) -> float:
    """Map measured 击球点 quality onto the 20-point axis."""
    if not hits:
        return 7.0
    parts = []
    for h in hits:
        pts = 10.0
        if h.height is None:
            pts -= 1.5
        elif -0.10 <= h.height <= 0.12:
            pts += 3.0
        elif -0.18 <= h.height <= 0.20:
            pts += 1.0
        else:
            pts -= 2.0
        if view == "back":
            v = h.side
            good, ok = (0.16, 0.48), (0.08, 0.58)
        else:
            v = h.forward
            good, ok = (0.10, 0.42), (0.04, 0.55)
        if v is None:
            pts -= 1.0
        elif good[0] <= v <= good[1]:
            pts += 4.0
        elif ok[0] <= v <= ok[1]:
            pts += 1.5
        else:
            pts -= 2.5
        if h.both_feet_off:
            pts -= 1.0
        parts.append(float(np.clip(pts, 3, 20)))
    contact_pts = float(np.mean(parts))
    if late_rate >= 0.45:
        contact_pts = min(contact_pts, 9)
    elif late_rate >= 0.25:
        contact_pts = min(contact_pts, 13)
    return float(np.clip(contact_pts, 4, 20))


_FONT_CANDIDATES = [
    "/usr/share/fonts/google-droid/DroidSansFallback.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _font_path() -> str | None:
    for p in _FONT_CANDIDATES:
        if Path(p).is_file():
            return p
    return None


def _put_cn(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.62, color=(255, 255, 255)) -> np.ndarray:
    path = _font_path()
    if not path or not text:
        cv2.putText(img, text.encode("ascii", "ignore").decode() or "hit", xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return img
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        cv2.putText(img, "hit", xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return img
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    size = max(16, int(28 * scale))
    try:
        font = ImageFont.truetype(path, size)
    except OSError:
        font = ImageFont.load_default()
    x, y = int(xy[0]), int(xy[1])
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _dashed_ellipse(img, center, axes, color, thickness=2, step=12):
    cx, cy = int(center[0]), int(center[1])
    ax, ay = int(axes[0]), int(axes[1])
    for deg in range(0, 360, step * 2):
        cv2.ellipse(img, (cx, cy), (ax, ay), 0, deg, deg + step, color, thickness, cv2.LINE_AA)


def draw_hit_point(
    bgr: np.ndarray,
    hp: HitPoint | None,
    *,
    tag: str = "",
    emphasize: bool = True,
) -> np.ndarray:
    """Draw actual contact (solid reticle) and the ideal zone (dashed)."""
    out = bgr.copy()
    if hp is None:
        if tag:
            out = _put_cn(out, tag, (12, 28), 0.72, (255, 255, 255))
        return out

    if hp.ideal_xy is not None and emphasize:
        ix, iy = int(hp.ideal_xy[0]), int(hp.ideal_xy[1])
        _dashed_ellipse(out, (ix, iy), (28, 22), (60, 200, 255), 2, 10)
        cv2.circle(out, (ix, iy), 4, (60, 200, 255), -1, cv2.LINE_AA)
        out = _put_cn(out, "理想区", (ix + 16, iy - 28), 0.48, (60, 200, 255))

    if hp.chest is not None and emphasize:
        cx, cy = int(hp.chest[0]), int(hp.chest[1])
        cv2.circle(out, (cx, cy), 5, (220, 180, 80), -1, cv2.LINE_AA)
        if hp.xy is not None:
            cv2.line(
                out,
                (cx, cy),
                (int(hp.xy[0]), int(hp.xy[1])),
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

    if hp.xy is not None:
        x, y = int(hp.xy[0]), int(hp.xy[1])
        color = (40, 70, 255) if not hp.ok else (40, 210, 90)
        cv2.circle(out, (x, y), 16, color, 2, cv2.LINE_AA)
        cv2.circle(out, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        out = _put_cn(out, "击球点", (x + 18, y - 10), 0.52, color)

    title = tag or "击球"
    if hp.summary:
        title = f"{title}  {hp.summary}"
    out = _put_cn(out, title, (12, 28), 0.58, (255, 255, 255))
    return out
