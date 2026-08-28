"""Detect tennis racket and ball with a COCO YOLO detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# COCO
SPORTS_BALL = 32
TENNIS_RACKET = 38


@dataclass
class FrameObjects:
    ball_xy: np.ndarray | None = None
    ball_conf: float = 0.0
    racket_xy: np.ndarray | None = None
    racket_box: np.ndarray | None = None
    racket_conf: float = 0.0
    ball_candidates: list = field(default_factory=list)

    @property
    def has_ball(self) -> bool:
        return self.ball_xy is not None

    @property
    def has_racket(self) -> bool:
        return self.racket_xy is not None


class ObjectDetector:
    def __init__(self, model_path: str = "yolov8s.pt"):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        import torch

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def infer(self, bgr: np.ndarray) -> FrameObjects:
        return self.infer_batch([bgr])[0]

    def infer_batch(
        self,
        images: list[np.ndarray],
        *,
        imgsz: int = 640,
        conf: float = 0.12,
        max_ball_diam: float = 70.0,
        min_ball_diam: float = 4.0,
    ) -> list[FrameObjects]:
        if not images:
            return []
        results = self.model.predict(
            images,
            verbose=False,
            imgsz=imgsz,
            conf=conf,
            iou=0.5,
            max_det=20,
            classes=[SPORTS_BALL, TENNIS_RACKET],
            device=self.device,
            half=self.device == "cuda",
        )
        return [_from_result(r, min_ball_diam=min_ball_diam, max_ball_diam=max_ball_diam) for r in results]


def _from_result(result, min_ball_diam: float = 4.0, max_ball_diam: float = 70.0) -> FrameObjects:
    out = FrameObjects()
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    balls: list[tuple[float, np.ndarray]] = []
    rackets: list[tuple[float, np.ndarray]] = []
    for box, c, p in zip(xyxy, cls, conf):
        w = float(box[2] - box[0])
        h = float(box[3] - box[1])
        diam = max(w, h)
        if int(c) == SPORTS_BALL:
            if min_ball_diam <= diam <= max_ball_diam:
                balls.append((float(p), box))
        elif int(c) == TENNIS_RACKET:
            if diam >= 24:
                rackets.append((float(p), box))
    if balls:
        centers = [
            np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float64)
            for _, box in balls
        ]
        out.ball_candidates = centers
        p, box = max(balls, key=lambda x: x[0])
        out.ball_conf = p
        out.ball_xy = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float64)
    if rackets:
        p, box = max(rackets, key=lambda x: x[0] * max(box[2] - box[0], box[3] - box[1]))
        out.racket_conf = p
        out.racket_box = box.astype(np.float64)
        out.racket_xy = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float64)
    return out


def bind_ball_to_player(objs: FrameObjects, pose_xy, pose_conf, max_dist: float = 220.0) -> None:
    """Prefer the ball nearest the racket or either wrist, not a stray ball on court."""
    cands = [np.asarray(b, dtype=np.float64) for b in (objs.ball_candidates or []) if b is not None]
    if objs.ball_xy is not None and not cands:
        cands = [np.asarray(objs.ball_xy, dtype=np.float64)]
    if not cands:
        return
    anchors: list[np.ndarray] = []
    if objs.racket_xy is not None:
        anchors.append(np.asarray(objs.racket_xy, dtype=np.float64))
    if pose_xy is not None and pose_conf is not None:
        for i in (9, 10):
            if float(pose_conf[i]) > 0.3:
                anchors.append(np.asarray(pose_xy[i], dtype=np.float64))
    if not anchors:
        return

    def dist(b):
        return min(float(np.linalg.norm(b - a)) for a in anchors)

    ranked = sorted(cands, key=dist)
    if dist(ranked[0]) <= max_dist:
        objs.ball_xy = ranked[0]
    else:
        objs.ball_xy = None


def player_crop_box(
    frame_shape,
    pose_xy,
    pose_conf,
    reach: float = 1.9,
) -> tuple[int, int, int, int] | None:
    """Box around the player, with room for the racket and incoming ball."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    conf = np.asarray(pose_conf)
    xy = np.asarray(pose_xy, dtype=np.float64)
    vis = conf > 0.28
    if vis.sum() < 4:
        return None
    pts = xy[vis]
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    bw, bh = max(48.0, x2 - x1), max(48.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(bw, bh) * reach
    xa = int(np.clip(cx - side * 0.58, 0, w - 1))
    ya = int(np.clip(cy - side * 0.52, 0, h - 1))
    xb = int(np.clip(cx + side * 0.58, 0, w))
    yb = int(np.clip(cy + side * 0.62, 0, h))
    if xb - xa < 96 or yb - ya < 96:
        return None
    return xa, ya, xb, yb


def shift_objects(objs: FrameObjects, ox: float, oy: float) -> FrameObjects:
    if objs.ball_xy is not None:
        objs.ball_xy = objs.ball_xy + np.array([ox, oy], dtype=np.float64)
    objs.ball_candidates = [
        np.asarray(c, dtype=np.float64) + np.array([ox, oy], dtype=np.float64)
        for c in (objs.ball_candidates or [])
        if c is not None
    ]
    if objs.racket_xy is not None:
        objs.racket_xy = objs.racket_xy + np.array([ox, oy], dtype=np.float64)
    if objs.racket_box is not None:
        objs.racket_box = objs.racket_box + np.array([ox, oy, ox, oy], dtype=np.float64)
    return objs


def merge_objects(base: FrameObjects, extra: FrameObjects) -> FrameObjects:
    """Keep a first-pass detection unless the zoom pass actually saw the object."""
    if extra.has_ball:
        base.ball_xy = extra.ball_xy
        base.ball_conf = extra.ball_conf
        base.ball_candidates = list(extra.ball_candidates or [])
    elif extra.ball_candidates:
        base.ball_candidates = list(extra.ball_candidates)
    if extra.has_racket:
        base.racket_xy = extra.racket_xy
        base.racket_box = extra.racket_box
        base.racket_conf = extra.racket_conf
    return base


def draw_objects(bgr: np.ndarray, objs: FrameObjects) -> np.ndarray:
    out = bgr.copy()
    if objs.has_racket and objs.racket_box is not None:
        x1, y1, x2, y2 = objs.racket_box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 140, 255), 2, cv2.LINE_AA)
    if objs.has_ball and objs.ball_xy is not None:
        cx, cy = objs.ball_xy.astype(int)
        cv2.circle(out, (cx, cy), 8, (80, 255, 80), 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 3, (80, 255, 80), -1, cv2.LINE_AA)
    return out


def default_detect_path(root: Path) -> str:
    for name in ("yolov8s.pt", "yolov8n.pt"):
        local = root / "models" / name
        if local.is_file():
            return str(local)
    return "yolov8n.pt"
