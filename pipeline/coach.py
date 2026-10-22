"""Call Cursor Cloud Agent to write the coaching report."""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Callable

from pipeline.analyze import SCORE_AXES, grade_from_score
from pipeline.level import level_context, parse_level
from pipeline.cursor_client import available, model_id, model_params, run_with_stream, start_prompt

ProgressCb = Callable[[dict], None]

_SCORE_KEYS = ("综合",) + tuple(k for k, _ in SCORE_AXES)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S)
    if fence:
        raw = fence.group(1)
    else:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _encode_image(path: Path, max_side: int = 720) -> dict | None:
    if not path.exists() or path.stat().st_size < 200:
        return None
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 68])
        if not ok:
            return None
        data = buf.tobytes()
    except Exception:
        data = path.read_bytes()
        if len(data) > 2 * 1024 * 1024:
            return None
    if len(data) > 2 * 1024 * 1024:
        return None
    return {
        "data": base64.b64encode(data).decode("ascii"),
        "mimeType": "image/jpeg",
    }


def _phase_filename(info: dict) -> str:
    raw = str(info.get("oss_key") or info.get("image") or "").split("?", 1)[0]
    name = Path(raw).name
    if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        return name
    return ""


_PHASE_HINT = {
    "ready": "准备（重心、步幅、前肩）",
    "takeback": "引拍（转体还是只动手，拍头是否后下方）",
    "contact": "击球（击球点、膝盖）",
    "follow": "随挥（过肩还是垂直上拎）",
}


def _pick_keyframes(clips: list[dict], kf_dir: Path, limit: int = 5) -> list[tuple[str, Path]]:
    picks: list[tuple[str, Path]] = []
    seen: set[str] = set()
    kf_dir = Path(kf_dir)

    def try_add(clip: dict, sw: dict, phase: str) -> bool:
        if len(picks) >= limit:
            return False
        info = (sw.get("phases") or {}).get(phase) or {}
        name = _phase_filename(info)
        if not name or name in seen:
            return False
        path = kf_dir / name
        if not path.is_file():
            return False
        seen.add(name)
        hint = _PHASE_HINT.get(phase, phase)
        label = f"{clip.get('label')} 挥拍#{sw.get('index')} {hint} t={sw.get('contact_t')}"
        picks.append((label, path))
        return True

    # 同一拍的引拍→击球→随挥→准备，才能看出擦玻璃和收拍方向。
    for clip in clips:
        swings = clip.get("swings") or []
        if not swings:
            continue
        for phase in ("takeback", "contact", "follow", "ready"):
            try_add(clip, swings[0], phase)
            if len(picks) >= limit:
                return picks
    for clip in clips:
        for sw in (clip.get("swings") or [])[1:]:
            try_add(clip, sw, "contact")
            if len(picks) >= limit:
                return picks
    if picks:
        return picks
    for path in sorted(kf_dir.glob("*.jpg"))[:limit]:
        picks.append((path.stem, path))
    return picks


def _seen_hit(hit: dict | None) -> dict | None:
    if not isinstance(hit, dict):
        return None
    seen = []
    for key in ("summary", "height_label", "side_label", "forward_label"):
        text = str(hit.get(key) or "").strip()
        if text and "看不清" not in text and text not in seen:
            seen.append(text)
    extra = []
    if hit.get("hand_reaches"):
        extra.append("空着的手伸向来球")
    if hit.get("both_feet_off"):
        extra.append("击球时双脚离地")
    if hit.get("early_step"):
        extra.append("随挥未结束就上步")
    if not seen and not extra:
        return None
    out = {}
    if seen:
        out["位置"] = seen
    if extra:
        out["另外"] = extra
    return out


def _slim_swing(swing: dict) -> dict:
    notes = [str(x).strip() for x in (swing.get("flag_notes") or []) if str(x).strip()]
    item = {
        "index": swing.get("index"),
        "contact_t": swing.get("contact_t"),
        "shot_kind": swing.get("shot_kind"),
        "late_contact": bool(swing.get("late_contact")),
    }
    hit = _seen_hit(swing.get("hit_point") if isinstance(swing.get("hit_point"), dict) else None)
    if hit:
        item["hit"] = hit
    if notes:
        item["notes"] = notes[:4]
    return item


def _slim_report(report: dict) -> dict:
    clips = []
    for c in report.get("clips") or []:
        summary = c.get("summary") or {}
        clips.append(
            {
                "id": c.get("id"),
                "label": c.get("label"),
                "hitting_arm": c.get("hitting_arm"),
                "n_swings": summary.get("n_swings"),
                "swings": [_slim_swing(s) for s in (c.get("swings") or [])[:6]],
            }
        )
    overall = report.get("overall") or {}
    return {
        "player_level": report.get("player_level_label") or report.get("player_level"),
        "view": report.get("view_label"),
        "handedness": report.get("handedness_label"),
        "duration_s": report.get("duration_s"),
        "shot_mix": overall.get("shot_mix") or {},
        "clips": clips,
    }


def _build_prompt(report: dict, captions: list[str]) -> str:
    payload = json.dumps(_slim_report(report), ensure_ascii=False, indent=2)
    caps = "\n".join(f"- 图片{i+1}: {c}" for i, c in enumerate(captions)) or "（无附图）"
    ids = "、".join(c.get("id") or "" for c in report.get("clips") or []) or "forehand"
    level = parse_level(report.get("player_level"))
    context = level_context(level)
    return f"""你是网球私教。看着这几张挥拍画面，用你自己的话写一份点评。不要改文件、不要跑命令、不要读仓库。

{context}

附图：
{caps}

【怎么写】
- 以画面为准。下面的 JSON 只说明这次是什么技术、击球大概在身前还是偏晚，不是要你照抄的稿子。画面和它不一致时，信画面。
- 按 clips.id 声明的技术来写。没有反手就不要写反手，声明了发球就写发球，不要改成别的技术。看不清的细节直接跳过，不要编。
- 这是单机位录像，看不到完整三维。不要写角度、肘角、膝角、步幅比例、公里每小时、百分比、钟点。这些数字从侧面或背面推出来往往不准，写出来会被质疑。
- 用能看出来的现象说话：人站得高不高、击球在身前还是贴着身体、引拍有没有转开、随挥是送出去还是往上拎。不要判断球打在拍面哪里。
- 用短句、大白话。不要套固定口令和固定组数，也不要把同一句话换个说法凑条数。
- 只点评这些 clip id：{ids}

【四维满分】重心 25、击球点 20、动力链 30、击球效果 25。按这次画面的完成度打分。

【篇幅】
- summary 120–180 字：先说这次最明显的一点，再写一两个真正值得改的地方，带上你看到的原因。
- focus：这次最该抓的一件事，40 字以内。
- improvements：2–3 条，按这个人的水平写怎么练。
- 每个 clip：strengths 2 条左右，problems 2–4 条，drills 2–3 条。

【参考】
{payload}

只输出 JSON：
{{
  "summary": "总评",
  "focus": "本次最该改的一件事",
  "improvements": ["怎么练"],
  "scores": {{"综合": 0-100整数, "重心": 0-25, "击球点": 0-20, "动力链": 0-30, "击球效果": 0-25}},
  "clips": [
    {{
      "id": "{ids.split('、')[0]}",
      "strengths": ["优点"],
      "problems": ["问题"],
      "drills": ["怎么练"],
      "scores": {{"综合": 整数, "重心": 整数, "击球点": 整数, "动力链": 整数, "击球效果": 整数}}
    }}
  ]
}}
"""


def _require_scores(src: dict | None) -> dict:
    if not isinstance(src, dict):
        raise RuntimeError("点评结果不完整，请稍后重试")
    out = {}
    for key, cap in SCORE_AXES:
        v = src.get(key)
        if not isinstance(v, (int, float)):
            raise RuntimeError("点评结果不完整，请稍后重试")
        out[key] = int(min(cap, max(0, round(v))))
    out["综合"] = int(sum(out[k] for k, _ in SCORE_AXES))
    return out


def _require_lines(src: dict, key: str) -> list[str]:
    val = src.get(key)
    if not isinstance(val, list) or not val:
        raise RuntimeError("点评结果不完整，请稍后重试")
    lines = [str(x).strip() for x in val if str(x).strip()]
    if not lines:
        raise RuntimeError("点评结果不完整，请稍后重试")
    return lines


def enrich_with_cursor(report: dict, kf_dir: Path, progress: ProgressCb | None = None) -> dict:
    if not report.get("clips"):
        report["coach"] = {"status": "unused"}
        return report

    if not available():
        raise RuntimeError("点评服务未就绪，请稍后重试")

    picks = _pick_keyframes(report["clips"], Path(kf_dir))
    images = []
    captions = []
    for cap, path in picks:
        enc = _encode_image(path)
        if enc:
            images.append(enc)
            captions.append(cap)
    if not images:
        raise RuntimeError("动作截图读取失败，请稍后重试")

    prompt = _build_prompt(report, captions)
    if progress:
        progress(
            {
                "step": 5,
                "step_name": "教练点评",
                "progress": 90,
                "message": "正在写练习建议，大约还要 2 分钟…",
            }
        )

    t0 = time.time()
    try:
        agent_id, run_id, reused = start_prompt(prompt, images=images)
    except Exception as exc:
        raise RuntimeError("教练点评失败，请稍后重试") from exc

    coach = {
        "model": model_id(),
        "params": model_params(),
        "status": "running",
        "agent_id": agent_id,
        "run_id": run_id,
        "reused": reused,
    }
    report["coach"] = coach

    def _delta(_t: str) -> None:
        if progress:
            elapsed = time.time() - t0
            progress(
                {
                    "step": 5,
                    "step_name": "教练点评",
                    "progress": min(95, 90 + int(elapsed / 8)),
                    "message": "正在写练习建议，大约还要 2 分钟…",
                }
            )

    try:
        text, status = run_with_stream(agent_id, run_id, on_assistant=_delta)
    except Exception as exc:
        raise RuntimeError("教练点评失败，请稍后重试") from exc

    elapsed = time.time() - t0
    coach["elapsed_s"] = round(elapsed, 1)
    coach["status"] = status
    parsed = _extract_json(text)
    print(f"[coach] {status} in {elapsed:.1f}s reused={reused} json={bool(parsed)}", flush=True)
    if not parsed:
        raise RuntimeError("教练点评未完成，请稍后重试")

    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("点评结果不完整，请稍后重试")
    report["summary"] = summary.strip()
    coach["summary"] = summary.strip()

    focus = parsed.get("focus")
    if isinstance(focus, str) and focus.strip():
        report["focus"] = focus.strip()
        coach["focus"] = focus.strip()

    improvements = parsed.get("improvements")
    if isinstance(improvements, list):
        improvements = [str(x).strip() for x in improvements if str(x).strip()]
    else:
        improvements = []
    if len(improvements) < 2:
        extra = []
        for clip in report["clips"]:
            extra.extend((clip.get("analysis") or {}).get("drills") or [])
        improvements = (improvements + extra)[:3]
    if improvements:
        report["improvements"] = improvements[:4]
        coach["improvements"] = report["improvements"]

    overall_scores = _require_scores(parsed.get("scores"))
    report["overall"]["scores"] = overall_scores
    report["overall"]["score"] = int(overall_scores["综合"])
    grade, grade_label = grade_from_score(int(report["overall"]["score"]))
    report["overall"]["grade"] = grade
    report["overall"]["grade_label"] = grade_label

    by_id = {c["id"]: c for c in parsed.get("clips") or [] if isinstance(c, dict) and c.get("id")}
    for clip in report["clips"]:
        extra = by_id.get(clip["id"])
        if not extra:
            raise RuntimeError("点评结果不完整，请稍后重试")
        clip["scores"] = _require_scores(extra.get("scores"))
        analysis = clip.setdefault("analysis", {})
        analysis["strengths"] = _require_lines(extra, "strengths")
        analysis["problems"] = _require_lines(extra, "problems")
        analysis["drills"] = _require_lines(extra, "drills")

    coach["status"] = "FINISHED"
    coach["message"] = "点评完成"
    return report
