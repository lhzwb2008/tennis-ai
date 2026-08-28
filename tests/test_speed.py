"""Pixel-to-km/h scale and swing speed estimates. No GPU required."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.analyze import ClipSeries, measure_swings, speed_from_xy
from pipeline.speed import (
    body_m_per_px,
    choose_scale,
    estimate_kmh,
    flight_speed_px,
    mean_speeds,
    peak_speed_px,
    px_to_kmh,
    track_to_xy,
)


def _pose(h=400.0, hip_x=320.0):
    xy = np.zeros((17, 2), dtype=np.float64)
    conf = np.ones(17, dtype=np.float64)
    xy[5] = [hip_x - 40, 200]
    xy[6] = [hip_x + 40, 200]
    xy[11] = [hip_x - 30, 320]
    xy[12] = [hip_x + 30, 320]
    xy[13] = [hip_x - 32, 400]
    xy[14] = [hip_x + 32, 400]
    xy[15] = [hip_x - 34, 480]
    xy[16] = [hip_x + 34, 480]
    xy[9] = [hip_x - 70, 260]
    xy[10] = [hip_x + 90, 240]
    xy[0] = [hip_x, 140]
    return xy, conf


class SpeedTests(unittest.TestCase):
    def test_body_scale_shoulder_hip(self):
        xy, conf = _pose()
        mpp = body_m_per_px(xy, conf)
        self.assertIsNotNone(mpp)
        self.assertAlmostEqual(mpp, 0.50 / 120.0, places=4)

    def test_px_to_kmh(self):
        self.assertEqual(px_to_kmh(1000, 0.005), 18.0)
        self.assertIsNone(px_to_kmh(1000, None))
        self.assertIsNone(px_to_kmh(1e6, 0.005))

    def test_choose_scale_prefers_consistent_racket(self):
        xy, conf = _pose()
        wrist = xy[10]
        # wrist at (410, 240); box ~62cm at body scale
        mpp = body_m_per_px(xy, conf)
        L = 0.62 / mpp
        box = [wrist[0] + 10, wrist[1] - 20, wrist[0] + 10 + L, wrist[1] + 20]
        chosen = choose_scale(xy, conf, 280, box, wrist)
        self.assertIsNotNone(chosen)
        self.assertGreater(chosen, 0.5 * mpp)
        self.assertLess(chosen, 2.2 * mpp)

    def test_peak_and_flight(self):
        ts = np.arange(8) / 30.0
        xy = np.zeros((8, 2))
        for i in range(8):
            xy[i] = [i * 30.0, 0.0]
        peak = peak_speed_px(xy, ts, 0, 8)
        med = flight_speed_px(xy, ts, 0, 8)
        self.assertAlmostEqual(peak, 900.0, places=0)
        self.assertAlmostEqual(med, 900.0, places=0)

    def test_estimate_falls_back_to_wrist(self):
        d = estimate_kmh(
            m_per_px=0.005,
            wrist_px=1000,
            racket_px=None,
            hip_px=200,
            ball_in_px=800,
            ball_out_px=1200,
        )
        self.assertEqual(d["swing_from"], "wrist")
        self.assertEqual(d["wrist_kmh"], 18.0)
        self.assertGreater(d["swing_kmh"], d["wrist_kmh"])
        self.assertEqual(d["ball_out_kmh"], 21.6)

    def test_mean_speeds(self):
        a = {"swing_kmh": 40.0, "wrist_kmh": 30.0, "swing_from": "racket"}
        b = {"swing_kmh": 50.0, "wrist_kmh": None, "swing_from": "racket"}
        m = mean_speeds([a, b])
        self.assertEqual(m["swing_kmh"], 45.0)
        self.assertEqual(m["swing_kmh_max"], 50.0)
        self.assertEqual(m["wrist_kmh"], 30.0)

    def test_measure_swings_fills_kmh(self):
        n, fps = 24, 30.0
        t = np.arange(n) / fps
        pose_xy, pose_conf = [], []
        wrist = np.zeros((n, 2))
        hip = np.zeros((n, 2))
        boxes, racket, balls = [], [], []
        for i in range(n):
            xy, conf = _pose()
            xy[10] = [280 + i * 55.0, 240]
            xy[12] = [320 + i * 4.0, 320]
            xy[11] = [260 + i * 4.0, 320]
            pose_xy.append(xy)
            pose_conf.append(conf)
            wrist[i] = xy[10]
            hip[i] = (xy[11] + xy[12]) / 2
            w = xy[10]
            box = [w[0] + 30, w[1] - 30, w[0] + 150, w[1] + 30]
            boxes.append(box)
            racket.append(np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]))
            balls.append(np.array([120.0 + i * 40.0, 220.0]))
        series = ClipSeries(
            t=t,
            wrist_speed=speed_from_xy(wrist, t),
            elbow=np.full(n, 100.0),
            knee=np.full(n, 140.0),
            cog_ratio=np.full(n, 0.5),
            stance=np.full(n, 1.4),
            takeback=np.linspace(0.2, 0.55, n),
            hitting="right",
            wrist_xy=wrist,
            hip_xy=hip,
            reach_x=wrist[:, 0] - hip[:, 0],
            torso_len=np.full(n, 280.0),
            hip_speed=speed_from_xy(hip, t),
            shoulder_speed=np.zeros(n),
            elbow_speed=np.zeros(n),
        )
        swings = measure_swings(
            series,
            fps,
            peaks=[12],
            enable_late_contact=False,
            ball_xy=balls,
            racket_xy=racket,
            wrist_xy=wrist,
            racket_box=boxes,
            view="side",
            pose_xy=pose_xy,
            pose_conf=pose_conf,
        )
        self.assertEqual(len(swings), 1)
        sp = swings[0].speeds
        self.assertIsNotNone(sp)
        self.assertIsNotNone(sp["swing_kmh"])
        self.assertGreater(sp["swing_kmh"], 15)
        self.assertLess(sp["swing_kmh"], 120)
        self.assertIsNotNone(sp["wrist_kmh"])
        self.assertIsNotNone(sp["ball_out_kmh"])
        self.assertTrue(np.isfinite(track_to_xy(wrist, n)).all())


if __name__ == "__main__":
    unittest.main()
