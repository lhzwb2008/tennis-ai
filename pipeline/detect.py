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

    def infer_batch(self, images: list[np.ndarray]) -> list[FrameObjects]:
        if not images:
            return []
        results = self.model.predict(
            images,
            verbose=False,
            imgsz=640,
            conf=0.12,
            iou=0.5,
            max_det=20,
            classes=[SPORTS_BALL, TENNIS_RACKET],
            device=self.device,
            half=self.device == "cuda",
        )
        return [_from_result(r) for r in results]


def _from_result(result) -> FrameObjects:
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
            if 4 <= diam <= 70:
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
