"""2D pose on split back/side views."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# COCO-17 used by YOLO pose
KPT = {
    "nose": 0,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}

SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class FramePose:
    xy: np.ndarray  # (17, 2) pixel coords in the cropped view
    conf: np.ndarray  # (17,)
    ok: bool


class PoseEstimator:
    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        import torch

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def infer(self, bgr: np.ndarray) -> FramePose:
        result = self.model.predict(
            bgr,
            verbose=False,
            imgsz=640,
            conf=0.25,
            max_det=4,
            device=self.device,
            half=self.device == "cuda",
        )[0]
        return self._from_result(result)

    def infer_batch(self, images: list[np.ndarray]) -> list[FramePose]:
        if not images:
            return []
        results = self.model.predict(
            images,
            verbose=False,
            imgsz=640,
            conf=0.25,
            max_det=4,
            device=self.device,
            half=self.device == "cuda",
        )
        return [self._from_result(r) for r in results]

    def _from_result(self, result) -> FramePose:
        if result.keypoints is None or len(result.keypoints) == 0:
            return FramePose(np.zeros((17, 2)), np.zeros(17), False)
        kpts = result.keypoints
        xy = kpts.xy.cpu().numpy()
        if xy.shape[0] == 0:
            return FramePose(np.zeros((17, 2)), np.zeros(17), False)
        conf = (
            kpts.conf.cpu().numpy()
            if kpts.conf is not None
            else np.ones((xy.shape[0], xy.shape[1]))
        )
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
        idx = _pick_primary(xy, conf, boxes)
        return FramePose(xy[idx], conf[idx], True)


def _pick_primary(xy: np.ndarray, conf: np.ndarray, boxes: np.ndarray | None) -> int:
    """Prefer the largest / most complete person (near-court player, not feeder)."""
    n = xy.shape[0]
    scores = []
    for i in range(n):
        vis = conf[i] > 0.3
        if vis.sum() < 6:
            scores.append(-1)
            continue
        pts = xy[i][vis]
        area = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i]
            area = max(area, (x2 - x1) * (y2 - y1))
        scores.append(float(area) * (vis.mean() + 0.2))
    return int(np.argmax(scores))


def view_layout(frame: np.ndarray) -> tuple[int, int, int]:
    h, w = frame.shape[:2]
    mid = w // 2
    top, bot = int(h * 0.06), int(h * 0.96)
    return mid, top, bot


def split_views(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid, top, bot = view_layout(frame)
    back = frame[top:bot, 0:mid]
    side = frame[top:bot, mid : frame.shape[1]]
    return back, side


def paste_views(frame: np.ndarray, back: np.ndarray, side: np.ndarray) -> np.ndarray:
    out = frame.copy()
    mid, top, bot = view_layout(frame)
    out[top:bot, 0:mid] = back
    out[top:bot, mid : frame.shape[1]] = side
    return out


def draw_pose(
    bgr: np.ndarray,
    pose: FramePose,
    color=(255, 210, 0),
    joint_color=(0, 255, 255),
) -> np.ndarray:
    """Draw our skeleton. Default cyan joints / gold bones to distinguish from QIDO yellow."""
    out = bgr.copy()
    if not pose.ok:
        return out
    for a, b in SKELETON:
        if pose.conf[a] > 0.3 and pose.conf[b] > 0.3:
            pa = tuple(pose.xy[a].astype(int))
            pb = tuple(pose.xy[b].astype(int))
            cv2.line(out, pa, pb, color, 3, cv2.LINE_AA)
    for i, (x, y) in enumerate(pose.xy):
        if pose.conf[i] > 0.3:
            cv2.circle(out, (int(x), int(y)), 5, joint_color, -1, cv2.LINE_AA)
    return out
