"""History listing and local report archive. No GPU."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from web.history import archive_report, archive_slug, history_item, list_history


class HistoryTests(unittest.TestCase):
    def _job(self, root: Path, job_id: str, *, score=72, source="side-wide.mp4", error=False):
        d = root / job_id
        d.mkdir()
        report = {
            "title": "网球挥拍测评报告 2.0",
            "source_name": source,
            "generated_at": "2026-09-03T09:12:00+08:00",
            "view_label": "侧面",
            "handedness_label": "右手持拍",
            "overall": {"score": score, "grade": "B", "grade_label": "良好", "n_swings": 11},
            "focus": "先把击球点打到身前",
        }
        (d / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        (d / "preview.jpg").write_bytes(b"\xff\xd8preview")
        status = {
            "id": job_id,
            "status": "error" if error else "done",
            "source_name": source,
            "score": score,
            "n_swings": 11,
            "created_at": "2026-09-03T09:12:00+08:00",
            "is_sample": False,
            "title": "网球挥拍测评报告 2.0",
        }
        (d / "status.json").write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
        return d

    def test_list_skips_errors_and_sorts_newest_first(self):
        with tempfile.TemporaryDirectory() as raw:
            jobs = Path(raw) / "jobs"
            jobs.mkdir()
            self._job(jobs, "oldjob12abcd", score=60, source="a.mp4")
            newer = self._job(jobs, "newjob34efgh", score=80, source="b.mp4")
            status = json.loads((newer / "status.json").read_text())
            status["created_at"] = "2026-09-03T10:00:00+08:00"
            (newer / "status.json").write_text(json.dumps(status), encoding="utf-8")
            self._job(jobs, "badjob56ijkl", error=True)
            items = list_history(jobs)
            self.assertEqual([x["id"] for x in items], ["newjob34efgh", "oldjob12abcd"])
            self.assertEqual(items[0]["score"], 80)
            self.assertTrue(items[0]["has_preview"])

    def test_archive_writes_readable_folder(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            jobs = root / "jobs"
            reports = root / "reports"
            jobs.mkdir()
            job = self._job(jobs, "abc123def456", score=72, source="court A.mp4")
            dest = archive_report(job, reports)
            self.assertIsNotNone(dest)
            self.assertTrue((dest / "report.json").is_file())
            self.assertTrue((dest / "preview.jpg").is_file())
            self.assertTrue((dest / "meta.json").is_file())
            self.assertIn("abc123def456", dest.name)
            self.assertIn("72", dest.name)
            self.assertTrue(dest.name.startswith("20260903-0912_"))
            again = archive_report(job, reports)
            self.assertEqual(again, dest)
            self.assertEqual(len(list(reports.iterdir())), 1)

    def test_slug_strips_odd_filename_chars(self):
        slug = archive_slug("id1", "2026-09-03T09:12:00+08:00", "../weird name?.mp4", 9)
        self.assertNotIn("?", slug)
        self.assertNotIn("..", slug)
        self.assertTrue(slug.endswith("_id1"))

    def test_history_item_none_without_report(self):
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw) / "empty"
            d.mkdir()
            self.assertIsNone(history_item(d))

    def test_list_includes_archive_when_job_gone(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            jobs = root / "jobs"
            reports = root / "reports"
            jobs.mkdir()
            job = self._job(jobs, "gonejob78mnop", score=66, source="far-cam.mp4")
            dest = archive_report(job, reports)
            self.assertIsNotNone(dest)
            shutil.rmtree(job)
            items = list_history(jobs, reports)
            self.assertEqual([x["id"] for x in items], ["gonejob78mnop"])
            self.assertEqual(items[0]["score"], 66)
            self.assertEqual(items[0]["source_name"], "far-cam.mp4")


if __name__ == "__main__":
    unittest.main()
