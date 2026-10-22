"""Player-declared NTRP level is context, not a script. No GPU."""

from __future__ import annotations

import unittest

from pipeline.analyze import score_and_write
from pipeline.coach import _build_prompt, _slim_report
from pipeline.level import level_context, parse_level, public_levels


class ParseLevelTests(unittest.TestCase):
    def test_default_and_aliases(self):
        self.assertEqual(parse_level(None).code, "3.0")
        self.assertEqual(parse_level("").code, "3.0")
        self.assertEqual(parse_level("3.5 中级").code, "3.5")
        self.assertEqual(parse_level(4).code, "4.0")
        self.assertEqual(parse_level("5.0+").code, "5.0")
        self.assertEqual(parse_level("1.0").code, "2.0")
        self.assertEqual(parse_level("9").code, "5.0")
        self.assertEqual(parse_level("ntrp2.5").ntrp, 2.5)

    def test_catalog(self):
        codes = [item["code"] for item in public_levels()]
        self.assertEqual(codes, ["2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"])
        self.assertEqual(parse_level("3.0").label, "3.0 进阶初级")


class CoachingTests(unittest.TestCase):
    def test_level_context_does_not_hand_out_a_script(self):
        text = level_context("2.0")
        self.assertIn("2.0 入门", text)
        self.assertIn("自己的话", text)
        self.assertNotIn("拍凳子", text)
        self.assertNotIn("【训练】", text)

    def test_same_swing_same_score_across_levels(self):
        summary = {
            "n_swings": 8,
            "cog_ratio": 0.5,
            "late_contact_rate": 0.6,
            "contact_forward": 0.02,
            "flag_rates": {},
        }
        low = score_and_write("forehand", summary, player_level="2.0")
        high = score_and_write("forehand", summary, player_level="5.0")
        self.assertEqual(low["scores"]["综合"], high["scores"]["综合"])
        blob = " ".join(low["problems"] + low["drills"] + high["problems"] + high["drills"])
        self.assertNotIn("km/h", blob)
        self.assertNotIn("°", blob)
        self.assertNotIn("%", blob)

    def test_prompt_asks_the_model_to_look_and_skip_numbers(self):
        report = {
            "player_level": "5.0",
            "player_level_label": "5.0 准专业",
            "handedness": "right",
            "handedness_label": "右手持拍",
            "view_label": "侧面",
            "clips": [
                {
                    "id": "forehand",
                    "label": "底线正手",
                    "hitting_arm": "right",
                    "scores": {"综合": 70, "重心": 18, "击球点": 14, "动力链": 20, "击球效果": 18},
                    "analysis": {"problems": ["这句不该被要求照抄"]},
                    "summary": {"n_swings": 6, "elbow_contact_deg": 151, "swing_kmh": 80},
                    "swings": [
                        {
                            "index": 1,
                            "contact_t": 1.2,
                            "late_contact": True,
                            "elbow_deg": 151,
                            "speeds": {"swing_kmh": 80},
                            "flag_notes": ["擦玻璃倾向"],
                            "hit_point": {
                                "summary": "偏低、偏晚贴身",
                                "height": -0.4,
                                "angle_deg": 12,
                                "forward_label": "偏晚贴身",
                            },
                        }
                    ],
                }
            ],
            "overall": {"shot_mix": {"forehand": 6}, "scores": {"综合": 70}},
        }
        prompt = _build_prompt(report, ["底线正手 挥拍#1"])
        self.assertIn("5.0 准专业", prompt)
        self.assertIn("用你自己的话", prompt)
        self.assertIn("不要写角度", prompt)
        self.assertNotIn("rule_hints", prompt)
        self.assertNotIn("必须写进点评", prompt)
        self.assertNotIn("拍凳子", prompt)
        slim = _slim_report(report)
        dumped = str(slim)
        self.assertNotIn("151", dumped)
        self.assertNotIn("swing_kmh", dumped)
        self.assertNotIn("angle_deg", dumped)
        self.assertEqual(slim["clips"][0]["swings"][0]["notes"], ["擦玻璃倾向"])
        self.assertIn("偏晚贴身", dumped)


if __name__ == "__main__":
    unittest.main()
