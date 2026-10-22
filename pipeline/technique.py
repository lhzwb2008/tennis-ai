"""Coach-authored checkpoints, video-observable proxies, and rule hints.

Clock-face palm directions are coaching cues, not pixel measurements. Flags are
side-view proxies (path, turn, weight) for the LLM to confirm against keyframes.
"""

from __future__ import annotations

from collections import Counter


# 2D 投影会偏，只把明显越界当旁证。
ELBOW_RANGES = {
    "forehand": {"takeback": (130, 150), "contact": (140, 160), "follow": (100, 120)},
    "backhand": {"takeback": (120, 140), "contact": (130, 150), "follow": (100, 130)},
    "backhand_one": {"takeback": (140, 160), "contact": (150, 170), "follow": (120, 140)},
    "forehand_slice": {"takeback": (130, 150), "contact": (130, 150), "follow": (100, 130)},
    "backhand_slice": {"takeback": (120, 140), "contact": (120, 140), "follow": (100, 120)},
    "volley_fh": {"takeback": (120, 140), "contact": (130, 150), "follow": (100, 120)},
    "volley_bh": {"takeback": (120, 140), "contact": (140, 160), "follow": (100, 120)},
    "serve": {"takeback": (90, 110), "contact": (150, 170), "follow": (100, 120)},
    "smash": {"takeback": (90, 110), "contact": (150, 170), "follow": (100, 120)},
}

FLAG_SHORT = {
    "wipe_glass": "擦玻璃倾向",
    "arm_only": "只动手不转体",
    "no_weight_shift": "重心未前移",
    "hand_reach": "辅助手指球",
    "late_takeback": "引拍偏晚",
    "lean_back": "击球后仰",
    "follow_vertical": "随挥垂直上收",
    "follow_too_high": "反手收拍过高",
    "head_below_wrist": "拍头低于手腕",
}

_KIND_ALIAS = {
    "forehand": "forehand",
    "topspin": "forehand",
    "forehand_slice": "forehand_slice",
    "slice": "forehand_slice",
    "backhand": "backhand",
    "backhand_slice": "backhand_slice",
}


def normalize_kind(stroke: str) -> str:
    return _KIND_ALIAS.get(stroke, stroke or "forehand")


def elbow_in_range(kind: str, phase: str, deg: float | None, slack: float = 18.0) -> bool | None:
    if deg is None:
        return None
    spec = ELBOW_RANGES.get(normalize_kind(kind)) or ELBOW_RANGES["forehand"]
    lo, hi = spec.get(phase) or (0, 180)
    return (lo - slack) <= float(deg) <= (hi + slack)


def flags_from_values(
    kind: str,
    *,
    view: str = "side",
    slot_drop: float | None = None,
    takeback_height: float | None = None,
    body_turn: float | None = None,
    wrist_back: float | None = None,
    weight_shift: float | None = None,
    torso_lean: float | None = None,
    follow_forward: float | None = None,
    follow_up: float | None = None,
    takeback_dt: float | None = None,
    elbow_takeback: float | None = None,
    elbow_contact: float | None = None,
    elbow_follow: float | None = None,
    head_above_wrist: float | None = None,
    hand_reaches: bool = False,
) -> list[str]:
    """Conservative proxies. Missing numbers do not vote yes. Side-view groundstrokes only."""
    kind = normalize_kind(kind)
    flags: list[str] = []
    side = view in ("side", "oblique")
    ground = kind in ("forehand", "backhand")

    if side and kind == "forehand" and slot_drop is not None and takeback_height is not None:
        # 拍凳子：引拍有高度，再落到腰。擦玻璃：有高度却横扫不落。
        if takeback_height >= 0.12 and slot_drop < 0.05:
            flags.append("wipe_glass")

    if side and ground and wrist_back is not None and body_turn is not None:
        if wrist_back >= 0.16 and body_turn < 0.04:
            flags.append("arm_only")

    if side and ground and weight_shift is not None and weight_shift < 0.015:
        flags.append("no_weight_shift")

    if side and ground and hand_reaches:
        flags.append("hand_reach")

    if side and ground and takeback_dt is not None and takeback_dt < 0.10:
        flags.append("late_takeback")

    if side and ground and torso_lean is not None and torso_lean < -0.08:
        flags.append("lean_back")

    if side and ground and follow_up is not None and (follow_forward is None or follow_forward < 0.06) and follow_up > 0.22:
        flags.append("follow_vertical")

    if side and kind == "backhand" and follow_up is not None and follow_up > 0.35:
        flags.append("follow_too_high")

    if side and kind == "backhand" and head_above_wrist is not None:
        if head_above_wrist < -0.02:
            flags.append("head_below_wrist")

    if side and ground:
        off = 0
        for phase, deg in (
            ("takeback", elbow_takeback),
            ("contact", elbow_contact),
            ("follow", elbow_follow),
        ):
            ok = elbow_in_range(kind, phase, deg, slack=22.0)
            if ok is False:
                off += 1
        if off >= 2:
            flags.append("elbow_range")

    return flags


def flag_rates(flag_lists: list[list[str]]) -> dict[str, float]:
    n = max(len(flag_lists), 1)
    counts: Counter[str] = Counter()
    for row in flag_lists:
        counts.update(row or [])
    return {k: round(v / n, 2) for k, v in counts.items()}


def flag_notes(flags: list[str]) -> list[str]:
    return [FLAG_SHORT[f] for f in flags if f in FLAG_SHORT]


def extra_findings(
    stroke: str,
    summary: dict,
    player_level: str = "3.0",
) -> tuple[list[str], list[str], list[str]]:
    """Short qualitative notes. The written report comes from the model, not these lines."""
    del player_level
    kind = normalize_kind(stroke)
    rates = summary.get("flag_rates") or {}
    strengths: list[str] = []
    problems: list[str] = []

    def rate(key: str) -> float:
        return float(rates.get(key) or 0.0)

    if kind == "forehand":
        if rate("wipe_glass") >= 0.3:
            problems.append("引拍之后球拍横着扫，不太像先落下来再往前送。")
        elif (summary.get("slot_drop") or 0) >= 0.08:
            strengths.append("引拍后能看到球拍先落下来再往前送。")

    if rate("arm_only") >= 0.3:
        problems.append("拉拍时主要是手在动，身体转开得不多。")
    elif (summary.get("body_turn") or 0) >= 0.08:
        strengths.append("引拍能看到身体跟着转，不是只甩胳膊。")

    if rate("no_weight_shift") >= 0.35:
        problems.append("击球时人几乎留在后面，没有往前送。")
    elif (summary.get("weight_shift") or 0) >= 0.05:
        strengths.append("击球时人能往前送，不是钉在原地。")

    if rate("late_takeback") >= 0.35:
        problems.append("引拍完成偏晚，球挤上来才动手。")

    if rate("lean_back") >= 0.3:
        problems.append("击球时身体有后仰。")

    if rate("follow_vertical") >= 0.3:
        problems.append("随挥更像往上拎，而不是向前送出去。")

    if kind == "backhand" and rate("follow_too_high") >= 0.3:
        problems.append("反手随挥收得偏高。")

    if rate("head_below_wrist") >= 0.35:
        problems.append("引拍时拍头掉到手腕下面。")

    return strengths, problems, []
