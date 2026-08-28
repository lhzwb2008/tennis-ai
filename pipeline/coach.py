"""Call Cursor Cloud Agent to write the coaching report."""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Callable

from pipeline.analyze import SCORE_AXES, grade_from_score
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


def _pick_keyframes(clips: list[dict], kf_dir: Path, limit: int = 4) -> list[tuple[str, Path]]:
    picks: list[tuple[str, Path]] = []
    seen: set[str] = set()
    order = ("contact", "takeback")
    kf_dir = Path(kf_dir)
    for clip in clips:
        for sw in clip.get("swings") or []:
            for phase in order:
                info = (sw.get("phases") or {}).get(phase) or {}
                name = _phase_filename(info)
                if not name or name in seen:
                    continue
                path = kf_dir / name
                if not path.is_file():
                    continue
                seen.add(name)
                label = f"{clip.get('label')} 挥拍#{sw.get('index')} {phase} t={sw.get('contact_t')}"
                picks.append((label, path))
                if len(picks) >= limit:
                    return picks
    if picks:
        return picks
    for path in sorted(kf_dir.glob("*.jpg"))[:limit]:
        picks.append((path.stem, path))
    return picks


def _slim_report(report: dict) -> dict:
    clips = []
    for c in report.get("clips") or []:
        swings = []
        for s in (c.get("swings") or [])[:6]:
            swings.append(
                {
                    "index": s.get("index"),
                    "contact_t": s.get("contact_t"),
                    "shot_kind": s.get("shot_kind"),
                    "late_contact": s.get("late_contact"),
                    "contact_forward": s.get("contact_forward"),
                    "hit_point": s.get("hit_point"),
                    "cog_stable": s.get("cog_stable"),
                    "chain_order": s.get("chain_order"),
                    "racket_speed": s.get("racket_speed"),
                    "path_lift": s.get("path_lift"),
                    "cog_ratio": s.get("cog_ratio"),
                    "knee_deg": s.get("knee_deg"),
                    "contact_source": s.get("contact_source"),
                }
            )
        clips.append(
            {
                "id": c.get("id"),
                "label": c.get("label"),
                "hitting_arm": c.get("hitting_arm"),
                "summary": c.get("summary"),
                "scores_rule": c.get("scores"),
                "swings": swings,
            }
        )
    return {
        "view": report.get("view_label"),
        "handedness": report.get("handedness_label"),
        "duration_s": report.get("duration_s"),
        "detect_rate": report.get("detect_rate"),
        "rule_overall": report.get("overall"),
        "shot_mix": (report.get("overall") or {}).get("shot_mix") or {},
        "clips": clips,
    }


def _build_prompt(report: dict, captions: list[str]) -> str:
    payload = json.dumps(_slim_report(report), ensure_ascii=False, indent=2)
    caps = "\n".join(f"- 图片{i+1}: {c}" for i, c in enumerate(captions)) or "（无附图）"
    ids = "、".join(c.get("id") or "" for c in report.get("clips") or []) or "forehand"
    return f"""你是有执教经验的网球私教，写一份给认真练球的球友看的深度点评。不要改文件、不要跑命令、不要读仓库。

附图：
{caps}

【判断动作，不要想当然】
- shot_mix 和 clips.id 是测量结果。没有 backhand 就不要写反手；正手切削（forehand_slice，path_lift 低或为负）不是反手。
- 重心要拆开写：偏高（直立挡球）和不稳定（击球时头肩上下晃）可以同时存在，不要只写其中一个。
- 对照画面说具体现象（哪类球、准备/击球/随挥），不要空泛的「继续努力」。
- 击球点看 hit_point：只谈相对身体的位置（胸口高度、持拍一侧稍外、身前大约 45°）。用「高了/低了、偏左/偏右、偏前/偏晚」说话。不要写球拍甜区、拍框、拍柄，也不要判断球打在拍面哪里。
- 准备时对准来球的是前肩（右手持拍是左肩），不是前手。禁止写「左手指向来球」当优点。
- 击球瞬间不能双脚同时离地（后脚脚尖点地可以）。随挥过肩之后才能上步。
- 用短句、大白话。少用「动力链」「加载」这种词，改成「腰先转、手后到」「先蹲再打」。

【四维满分】重心 25、击球点 20、动力链 30、击球效果 25。分数可按画面微调，不要编造没出现的动作。

【写深一点】
- summary 160–220 字：先点最大亮点，再写两个最要紧的问题，带一点因果。句子要短，球友扫一眼就能懂。
- focus：这次练球最该抓的一件事，40 字以内。
- improvements：3 条可执行训练，写清口令、组数和过关标准。
- 每个 clip 的 strengths / problems / drills 各 3 条；drills 用「【问题】… → 【原因】… → 【训练】…」。
- 只点评这些 clip id：{ids}

【测量 JSON】
{payload}

只输出 JSON：
{{
  "summary": "总评",
  "focus": "本次最该改的一件事",
  "improvements": ["【问题】… → 【原因】… → 【训练】…"],
  "scores": {{"综合": 0-100整数, "重心": 0-25, "击球点": 0-20, "动力链": 0-30, "击球效果": 0-25}},
  "clips": [
    {{
      "id": "{ids.split('、')[0]}",
      "strengths": ["优点"],
      "problems": ["问题"],
      "drills": ["【问题】… → 【原因】… → 【训练】…"],
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
