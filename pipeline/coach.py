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
from pipeline.technique import prompt_knowledge

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


_SWING_KEYS = (
    "index",
    "contact_t",
    "shot_kind",
    "late_contact",
    "contact_forward",
    "hit_point",
    "cog_stable",
    "chain_order",
    "racket_speed",
    "speeds",
    "path_lift",
    "cog_ratio",
    "knee_deg",
    "contact_source",
    "elbow_deg",
    "elbow_takeback_deg",
    "elbow_follow_deg",
    "takeback_ratio",
    "stance_ratio",
    "slot_drop",
    "body_turn",
    "weight_shift",
    "follow_forward",
    "follow_up",
    "takeback_dt",
    "tech_flags",
    "flag_notes",
)

_METRIC_KEYS = (
    "n_swings",
    "elbow_contact_deg",
    "elbow_takeback_deg",
    "elbow_follow_deg",
    "knee_contact_deg",
    "cog_ratio",
    "cog_stable",
    "stance_ratio",
    "takeback_ratio",
    "late_contact_rate",
    "chain_order",
    "path_lift",
    "slot_drop",
    "body_turn",
    "weight_shift",
    "follow_forward",
    "follow_up",
    "flag_rates",
    "hit_height",
    "shoulder_aim",
    "hand_reaches_rate",
    "both_feet_off_rate",
    "early_step_rate",
)


def _slim_report(report: dict) -> dict:
    clips = []
    for c in report.get("clips") or []:
        swings = []
        for s in (c.get("swings") or [])[:6]:
            swings.append({k: s.get(k) for k in _SWING_KEYS})
        analysis = c.get("analysis") or {}
        summary = c.get("summary") or {}
        clips.append(
            {
                "id": c.get("id"),
                "label": c.get("label"),
                "hitting_arm": c.get("hitting_arm"),
                "metrics": {k: summary.get(k) for k in _METRIC_KEYS},
                "scores_rule": c.get("scores"),
                "rule_hints": {
                    "strengths": (analysis.get("strengths") or [])[:5],
                    "problems": (analysis.get("problems") or [])[:6],
                },
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
    handed = report.get("handedness") or "right"
    knowledge = prompt_knowledge(
        [c.get("id") or "" for c in report.get("clips") or []],
        handed,
    )
    return f"""你是有执教经验的网球私教，写一份给认真练球的球友看的深度点评。不要改文件、不要跑命令、不要读仓库。

附图：
{caps}

{knowledge}

【判断动作，不要想当然】
- shot_mix 和 clips.id 是测量结果。没有 backhand 就不要写反手；正手切削（forehand_slice，path_lift 低或为负）不是反手。
- 重心要拆开写：偏高（直立挡球）和不稳定（击球时头肩上下晃）可以同时存在，不要只写其中一个。
- 对照画面说具体现象（哪类球、准备/引拍/击球/随挥），不要空泛的「继续努力」。
- 击球点看 hit_point：只谈相对身体的位置（胸口高度、持拍一侧稍外、身前大约 45°）。用「高了/低了、偏左/偏右、偏前/偏晚」说话。不要写球拍甜区、拍框、拍柄，也不要判断球打在拍面哪里。
- 准备时对准来球的是前肩（右手持拍是左肩），不是前手。禁止写「左手指向来球」当优点。
- 击球瞬间不能双脚同时离地（后脚脚尖点地可以）。随挥过肩之后才能上步。
- 用短句、大白话。少用「动力链」「加载」这种词，改成「腰先转、手后到」「先蹲再打」。
- speeds 里的 *_kmh 是画面比例估算（拍头、手腕、转髋、来球、出球），单位公里每小时。可以写「拍头大约 xx」，不要写成测速雷达，不要和职业球员雷达数据硬比。没有数字就不要编。
- 钟表方向（掌心 6:00 等）是教练口令，单路视频通常看不准。能看清手腕/前臂就写「拍头偏后下方」或「拍面朝侧面」；看不清不要编钟点。
- 擦玻璃 vs 拍凳子：看引拍结束后球拍有没有先由高往低落到腰再向前，以及球是否往天上飞。slot_drop / tech_flags 是旁证，要和附图对照。
- rule_hints 和 tech_flags 是测量层已经抓到的点，必须写进点评，不要只反复写「重心」「击球点」两句空话。对照技术清单把准备、引拍、击球、随挥都点到。
- 不要把同一句换说法凑数。画面和测量都看不清的细节不要编。

【四维满分】重心 25、击球点 20、动力链 30、击球效果 25。分数可按画面微调，不要编造没出现的动作。

【写深一点】
- summary 160–220 字：先点最大亮点，再写两个最要紧的问题，带一点因果。用清单里的说法（擦玻璃、拍凳子、前肩对球），但必须结合这次测量和画面。
- focus：这次练球最该抓的一件事，40 字以内。
- improvements：3 条可执行训练，写清口令、组数和过关标准。
- 每个 clip 的 strengths 2–3 条；problems 3–5 条，至少覆盖引拍和击球两段；drills 3 条，用「【问题】… → 【原因】… → 【训练】…」。
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
