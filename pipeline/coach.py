"""Call Cursor Cloud Agent to write the coaching report."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Callable

from pipeline.analyze import grade_from_score
from pipeline.cursor_client import available, create_agent, model_id, model_params, run_with_stream

ProgressCb = Callable[[dict], None]

_SCORE_KEYS = ("综合", "重心", "动力链", "动作框架", "步伐", "手腕")


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


def _encode_image(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size < 200:
        return None
    data = path.read_bytes()
    if len(data) > 12 * 1024 * 1024:
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


def _pick_keyframes(clips: list[dict], kf_dir: Path, limit: int = 5) -> list[tuple[str, Path]]:
    picks: list[tuple[str, Path]] = []
    seen: set[str] = set()
    order = ("contact", "takeback", "ready", "follow")
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
        for s in (c.get("swings") or [])[:8]:
            swings.append(
                {
                    "index": s.get("index"),
                    "contact_t": s.get("contact_t"),
                    "elbow_deg": s.get("elbow_deg"),
                    "knee_deg": s.get("knee_deg"),
                    "cog_ratio": s.get("cog_ratio"),
                    "stance_ratio": s.get("stance_ratio"),
                    "takeback_ratio": s.get("takeback_ratio"),
                    "late_contact": s.get("late_contact"),
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
        "clips": clips,
    }


def _build_prompt(report: dict, captions: list[str]) -> str:
    payload = json.dumps(_slim_report(report), ensure_ascii=False, indent=2)
    caps = "\n".join(f"- 图片{i+1}: {c}" for i, c in enumerate(captions)) or "（无附图）"
    return f"""你是网球私教。这是一次训练视频测评。

【硬性要求】
- 不要修改仓库里的任何文件，不要 git commit / 开 PR。
- 不要打开无关代码。直接根据测量数据和附图给出中文点评。
- 附图最多 5 张，顺序如下：
{caps}
- 测量来自录像画面上的人体关键点：击球帧是手腕速度峰值，可能比真实触球略晚；背面机位无法直接看击球点前后距离和拍面。
- 下列测量分数只是参考，你可以按画面改分数，但不要编造视频里没有的动作。
- 点评写给普通球友，不要出现模型名、算法名、规则引擎、像素、关键点、流水线等技术词。
- 训练建议用「【问题】… → 【原因】… → 【训练】…」句式。

【测量 JSON】
{payload}

【输出】
只输出一个 JSON 对象（不要 markdown 解释），字段：
{{
  "summary": "总评，120 字以内",
  "scores": {{"综合": 0-100整数, "重心": 0-30, "动力链": 0-25, "动作框架": 0-25, "步伐": 0-15, "手腕": 0-5}},
  "clips": [
    {{
      "id": "forehand 或 backhand，必须与输入 clips.id 对应",
      "strengths": ["优点"],
      "problems": ["问题"],
      "drills": ["【问题】… → 【原因】… → 【训练】…"],
      "scores": {{"综合": 整数, "重心": 整数, "动力链": 整数, "动作框架": 整数, "步伐": 整数, "手腕": 整数}}
    }}
  ]
}}
"""


def _require_scores(src: dict | None) -> dict:
    if not isinstance(src, dict):
        raise RuntimeError("点评结果不完整，请稍后重试")
    out = {}
    for k in _SCORE_KEYS:
        v = src.get(k)
        if not isinstance(v, (int, float)):
            raise RuntimeError("点评结果不完整，请稍后重试")
        out[k] = int(round(v))
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
                "message": "正在生成教练点评…",
            }
        )

    try:
        agent_id, run_id = create_agent(prompt, images=images)
    except Exception as exc:
        raise RuntimeError("教练点评失败，请稍后重试") from exc

    coach = {
        "model": model_id(),
        "params": model_params(),
        "status": "running",
        "agent_id": agent_id,
        "run_id": run_id,
    }
    report["coach"] = coach

    def _delta(_t: str) -> None:
        if progress:
            progress(
                {
                    "step": 5,
                    "step_name": "教练点评",
                    "progress": 93,
                    "message": "正在撰写练习建议…",
                }
            )

    try:
        text, status = run_with_stream(agent_id, run_id, on_assistant=_delta)
    except Exception as exc:
        raise RuntimeError("教练点评失败，请稍后重试") from exc

    coach["status"] = status
    parsed = _extract_json(text)
    if status != "FINISHED" or not parsed:
        raise RuntimeError("教练点评未完成，请稍后重试")

    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("点评结果不完整，请稍后重试")
    report["summary"] = summary.strip()
    coach["summary"] = summary.strip()

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
