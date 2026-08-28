"""Geometry tests for hitting-point labels. No GPU required."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.contact import (
    interpolate_xy,
    measure_hit_point,
    pick_contact_xy,
    racket_head_xy,
    refine_contact_index,
)


def _pose(h=400.0, hip_x=320.0):
    xy = np.zeros((17, 2), dtype=np.float64)
    conf = np.ones(17, dtype=np.float64)
    # y down. shoulders ~ 200, hips ~ 320, ankles ~ 480
    xy[5] = [hip_x - 40, 200]  # l_shoulder
    xy[6] = [hip_x + 40, 200]  # r_shoulder
    xy[11] = [hip_x - 30, 320]  # l_hip
    xy[12] = [hip_x + 30, 320]  # r_hip
    xy[13] = [hip_x - 32, 400]
    xy[14] = [hip_x + 32, 400]
    xy[15] = [hip_x - 34, 480]  # l_ankle
    xy[16] = [hip_x + 34, 480]
    xy[9] = [hip_x - 70, 260]  # l_wrist
    xy[10] = [hip_x + 90, 240]  # r_wrist
    xy[7] = [hip_x - 55, 230]
    xy[8] = [hip_x + 70, 220]
    return xy, conf


class ContactTests(unittest.TestCase):
    def test_pick_ball_as_hitting_point(self):
        ball = np.array([400.0, 230.0])
        pt, src = pick_contact_xy(ball, [350, 180, 430, 280], None, [300, 260])
        self.assertEqual(src, "ball")
        self.assertAlmostEqual(pt[0], 400)

    def test_racket_head_far_from_wrist(self):
        wrist = np.array([100.0, 200.0])
        box = [180, 160, 280, 240]
        head = racket_head_xy(box, wrist)
        self.assertGreater(head[0], 180)

    def test_side_view_chest_and_forward(self):
        xy, conf = _pose()
        # ball at chest height, in front (smaller x if load_sign=+1)
        ball = np.array([250.0, 230.0])
        hp = measure_hit_point(
            xy, conf, ball, [220, 190, 290, 270], None, xy[10],
            hitting="right", view="side", load_sign=1.0,
        )
        self.assertIsNotNone(hp.height)
        self.assertEqual(hp.height_label, "胸口高度")
        self.assertIsNotNone(hp.forward)
        self.assertGreater(hp.forward, 0.05)
        self.assertIn("身前", hp.forward_label)

    def test_back_view_hitting_side(self):
        xy, conf = _pose()
        ball = np.array([420.0, 228.0])  # right of hip
        hp = measure_hit_point(
            xy, conf, ball, None, None, xy[10],
            hitting="right", view="back", load_sign=1.0,
        )
        self.assertIsNotNone(hp.side)
        self.assertGreater(hp.side, 0.1)
        self.assertIn("右", hp.side_label)

    def test_interpolate_short_gap(self):
        a = np.array([0.0, 0.0])
        b = np.array([4.0, 0.0])
        filled = interpolate_xy([a, None, None, b])
        self.assertIsNotNone(filled[1])
        self.assertAlmostEqual(filled[1][0], 4.0 / 3.0, places=2)

    def test_refine_prefers_closest_ball(self):
        balls = [None] * 20
        rackets = [None] * 20
        wrists = np.zeros((20, 2))
        for i in range(20):
            wrists[i] = [100, 200]
            rackets[i] = np.array([180.0, 200.0])
        balls[8] = np.array([250.0, 200.0])
        balls[10] = np.array([185.0, 200.0])
        balls[12] = np.array([240.0, 200.0])
        idx, src = refine_contact_index(9, 30, balls, rackets, wrists)
        self.assertEqual(idx, 10)
        self.assertEqual(src, "ball_racket")

    def test_hand_reaches_flag(self):
        xy, conf = _pose()
        tb = xy.copy()
        tb[9] = [200.0, 210.0]  # left wrist closer to incoming ball
        tb[5] = [280.0, 200.0]
        ball = np.array([160.0, 200.0])
        hp = measure_hit_point(
            xy, conf, np.array([400.0, 230.0]), None, None, xy[10],
            hitting="right", view="side", load_sign=1.0,
            ball_takeback=ball, pose_takeback=tb, conf_takeback=conf,
        )
        self.assertTrue(hp.hand_reaches)


if __name__ == "__main__":
    unittest.main()
