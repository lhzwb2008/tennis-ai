"""Coach checklist, video-observable flags, and prompt wiring. No GPU."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pipeline.analyze import infer_view, score_and_write, swing_clip_span, view_is_side, view_label
from pipeline.coach import _build_prompt, _pick_keyframes, _slim_report
from pipeline.oss import refresh_report_urls
from pipeline.pose import KPT
from pipeline.technique import extra_findings, flags_from_values


class FlagTests(unittest.TestCase):
    def test_wipe_glass_when_high_takeback_does_not_drop(self):
        flags = flags_from_values(
            "forehand",
            slot_drop=0.02,
            takeback_height=0.22,
        )
        self.assertIn("wipe_glass", flags)

    def test_no_wipe_glass_when_racket_drops_into_slot(self):
        flags = flags_from_values(
            "forehand",
            slot_drop=0.12,
            takeback_height=0.22,
        )
        self.assertNotIn("wipe_glass", flags)

    def test_wipe_glass_skips_slice_and_backhand(self):
        kwargs = dict(slot_drop=0.01, takeback_height=0.25)
        self.assertNotIn("wipe_glass", flags_from_values("forehand_slice", **kwargs))
        self.assertNotIn("wipe_glass", flags_from_values("backhand", **kwargs))

    def test_arm_only_when_wrist_goes_back_without_turn(self):
        flags = flags_from_values("forehand", wrist_back=0.22, body_turn=0.01)
        self.assertIn("arm_only", flags)
        flags_ok = flags_from_values("forehand", wrist_back=0.22, body_turn=0.12)
        self.assertNotIn("arm_only", flags_ok)

    def test_weight_shift_only_on_side_view(self):
        self.assertIn(
            "no_weight_shift",
            flags_from_values("forehand", view="side", weight_shift=0.0),
        )
        self.assertNotIn(
            "no_weight_shift",
            flags_from_values("forehand", view="back", weight_shift=0.0),
        )

    def test_missing_numbers_do_not_invent_flags(self):
        self.assertEqual(flags_from_values("forehand"), [])

    def test_side_only_proxies_skip_back_view(self):
        kwargs = dict(slot_drop=0.01, takeback_height=0.25, wrist_back=0.22, body_turn=0.0)
        self.assertIn("wipe_glass", flags_from_values("forehand", view="side", **kwargs))
        back = flags_from_values("forehand", view="back", **kwargs)
        self.assertNotIn("wipe_glass", back)
        self.assertNotIn("arm_only", back)

    def test_hand_reach_skips_non_side(self):
        self.assertIn("hand_reach", flags_from_values("forehand", view="side", hand_reaches=True))
        self.assertIn("hand_reach", flags_from_values("forehand", view="oblique", hand_reaches=True))
        self.assertNotIn("hand_reach", flags_from_values("forehand", view="back", hand_reaches=True))


class FindingsTests(unittest.TestCase):
    def test_extra_findings_name_wipe_glass_drill(self):
        _s, problems, drills = extra_findings(
            "forehand",
            {"flag_rates": {"wipe_glass": 0.6}, "slot_drop": 0.02},
        )
        blob = " ".join(problems + drills)
        self.assertIn("横着扫", blob)
        self.assertNotIn("°", blob)
        self.assertEqual(drills, [])

    def test_score_and_write_surfaces_coach_errors(self):
        base = score_and_write("forehand", {"n_swings": 8, "cog_ratio": 0.5, "flag_rates": {}})
        written = score_and_write(
            "forehand",
            {
                "n_swings": 8,
                "cog_ratio": 0.5,
                "flag_rates": {"wipe_glass": 0.5, "arm_only": 0.5},
                "slot_drop": 0.02,
                "body_turn": 0.01,
            },
        )
        text = " ".join(written["problems"] + written["drills"] + written["strengths"])
        self.assertIn("横着扫", text)
        self.assertIn("手在动", text)
        self.assertNotIn("km/h", text)
        self.assertNotIn("°", text)
        self.assertLess(written["scores"]["击球效果"], base["scores"]["击球效果"])
        self.assertLess(written["scores"]["动力链"], base["scores"]["动力链"])

    def test_back_view_still_writes_coaching(self):
        written = score_and_write(
            "forehand",
            {
                "n_swings": 8,
                "cog_ratio": 0.62,
                "flag_rates": {},
            },
            view="back",
        )
        blob = " ".join(written["problems"] + written["drills"] + written["strengths"])
        self.assertIn("站得偏高", blob)
        self.assertNotIn("不下判断", blob)
        self.assertNotIn("评价范围", blob)

    def test_side_low_lift_does_not_call_it_slice(self):
        written = score_and_write(
            "forehand",
            {"n_swings": 6, "cog_ratio": 0.5, "path_lift": -0.2, "flag_rates": {}},
            view="side",
        )
        blob = " ".join(written["problems"] + written["drills"] + written["strengths"])
        self.assertNotIn("正手切削", blob)
        self.assertIn("轨迹偏平", blob)


class PromptTests(unittest.TestCase):
    def test_build_prompt_trusts_the_picture(self):
        report = {
            "player_level": "3.0",
            "handedness": "right",
            "handedness_label": "右手持拍",
            "view_label": "侧面",
            "clips": [
                {
                    "id": "forehand",
                    "label": "底线正手",
                    "hitting_arm": "right",
                    "scores": {"综合": 70, "重心": 18, "击球点": 14, "动力链": 20, "击球效果": 18},
                    "analysis": {"problems": ["这句规则稿不该进提示词"]},
                    "summary": {"n_swings": 6, "elbow_contact_deg": 140},
                    "swings": [
                        {
                            "index": 1,
                            "contact_t": 1.2,
                            "shot_kind": "topspin",
                            "tech_flags": ["wipe_glass"],
                            "flag_notes": ["擦玻璃倾向"],
                            "elbow_deg": 140,
                        }
                    ],
                }
            ],
            "overall": {"shot_mix": {"forehand": 6}},
        }
        prompt = _build_prompt(report, ["底线正手 挥拍#1 引拍 t=1.2"])
        self.assertIn("以画面为准", prompt)
        self.assertIn("不要写角度", prompt)
        self.assertIn("3.0 进阶初级", prompt)
        self.assertNotIn("rule_hints", prompt)
        self.assertNotIn("这句规则稿", prompt)
        slim = _slim_report(report)
        self.assertNotIn("elbow", str(slim))
        self.assertEqual(slim["clips"][0]["swings"][0]["notes"], ["擦玻璃倾向"])

    def test_pick_keyframes_prefers_full_cycle(self):
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw)
            for name in (
                "forehand_s01_takeback.jpg",
                "forehand_s01_contact.jpg",
                "forehand_s01_follow.jpg",
                "forehand_s01_ready.jpg",
            ):
                (d / name).write_bytes(b"x" * 20)
            clips = [
                {
                    "label": "底线正手",
                    "swings": [
                        {
                            "index": 1,
                            "contact_t": 2.0,
                            "phases": {
                                "ready": {"oss_key": "forehand_s01_ready.jpg"},
                                "takeback": {"oss_key": "forehand_s01_takeback.jpg"},
                                "contact": {"oss_key": "forehand_s01_contact.jpg"},
                                "follow": {"oss_key": "forehand_s01_follow.jpg"},
                            },
                        }
                    ],
                }
            ]
            picks = _pick_keyframes(clips, d, limit=5)
            labels = [lab for lab, _ in picks]
            self.assertGreaterEqual(len(picks), 4)
            self.assertTrue(any("引拍" in lab for lab in labels))
            self.assertTrue(any("随挥" in lab for lab in labels))
            self.assertTrue(any("准备" in lab for lab in labels))


class ViewTests(unittest.TestCase):
    def _frames(self, shoulder_width: float, torso=100.0, n=5):
        xy_list, conf_list = [], []
        for _ in range(n):
            xy = np.zeros((17, 2), dtype=np.float64)
            conf = np.ones(17, dtype=np.float64)
            xy[KPT["l_shoulder"]] = [0.0, 0.0]
            xy[KPT["r_shoulder"]] = [shoulder_width, 0.0]
            xy[KPT["l_hip"]] = [shoulder_width / 2, torso]
            xy[KPT["r_hip"]] = [shoulder_width / 2, torso]
            xy_list.append(xy)
            conf_list.append(conf)
        return xy_list, conf_list

    def test_true_profile_is_side(self):
        xy, conf = self._frames(20)
        self.assertEqual(infer_view(xy, conf), "side")
        self.assertTrue(view_is_side("side"))

    def test_three_quarter_is_oblique_still_coached(self):
        xy, conf = self._frames(36)
        self.assertEqual(infer_view(xy, conf), "oblique")
        self.assertEqual(view_label("oblique"), "侧面")
        self.assertTrue(view_is_side("oblique"))

    def test_facing_camera_is_back(self):
        xy, conf = self._frames(55)
        self.assertEqual(infer_view(xy, conf), "back")
        self.assertEqual(view_label("back"), "背面")


class SwingClipSpanTests(unittest.TestCase):
    def test_pads_ready_to_follow(self):
        start, end = swing_clip_span(ready_i=10, takeback_i=14, contact_i=20, follow_i=27, n=80, fps=25)
        self.assertLessEqual(start, 10)
        self.assertGreaterEqual(end, 27)
        self.assertGreaterEqual(end - start + 1, 21)

    def test_expands_short_window(self):
        start, end = swing_clip_span(ready_i=40, takeback_i=40, contact_i=40, follow_i=41, n=80, fps=25)
        self.assertGreaterEqual(end - start + 1, 21)
        self.assertGreaterEqual(start, 0)
        self.assertLess(end, 80)

    def test_stays_inside_video(self):
        start, end = swing_clip_span(ready_i=0, takeback_i=1, contact_i=2, follow_i=3, n=12, fps=25)
        self.assertEqual(start, 0)
        self.assertEqual(end, 11)


class ReportUrlRefreshTests(unittest.TestCase):
    def test_resigns_swing_video_and_phase_images(self):
        report = {
            "clips": [
                {
                    "swings": [
                        {
                            "video_oss_key": "clips/forehand_s01.mp4",
                            "phases": {"contact": {"oss_key": "keyframes/c.jpg"}},
                        }
                    ]
                }
            ]
        }
        with mock.patch("pipeline.oss.configured", return_value=True), mock.patch(
            "pipeline.oss.sign_url", side_effect=lambda key: "https://cdn/" + key
        ):
            out = refresh_report_urls(report)
        swing = out["clips"][0]["swings"][0]
        self.assertEqual(swing["video"], "https://cdn/clips/forehand_s01.mp4")
        self.assertEqual(swing["phases"]["contact"]["image"], "https://cdn/keyframes/c.jpg")


if __name__ == "__main__":
    unittest.main()
