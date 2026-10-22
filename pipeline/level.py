"""Player-declared NTRP level. Context for the coach, not a script."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_LEVEL = "3.0"


@dataclass(frozen=True)
class LevelInfo:
    code: str
    ntrp: float
    title: str
    blurb: str
    band: str

    @property
    def label(self) -> str:
        return f"{self.code} {self.title}"

    def as_public(self) -> dict:
        return {
            "code": self.code,
            "ntrp": self.ntrp,
            "title": self.title,
            "blurb": self.blurb,
            "band": self.band,
            "label": self.label,
        }


LEVELS: tuple[LevelInfo, ...] = (
    LevelInfo("2.0", 2.0, "入门", "刚能打到球，动作还在学", "beginner"),
    LevelInfo("2.5", 2.5, "初级", "能慢慢对打几拍，动作还不稳定", "beginner"),
    LevelInfo("3.0", 3.0, "进阶初级", "能稳定对拉，技术开始有框架", "developing"),
    LevelInfo("3.5", 3.5, "中级", "能变线，有一定旋转和脚步", "intermediate"),
    LevelInfo("4.0", 4.0, "中高级", "能主动进攻，动作比较完整", "advanced"),
    LevelInfo("4.5", 4.5, "高级", "能变化节奏和落点", "advanced"),
    LevelInfo("5.0", 5.0, "准专业", "接近竞技水平，技术完整", "elite"),
)

_BY_CODE = {item.code: item for item in LEVELS}


def parse_level(raw) -> LevelInfo:
    """Accept 3.5 / '3.5 中级' / 4. Unknown or empty falls back to 3.0."""
    if raw is None or raw == "" or isinstance(raw, bool):
        return _BY_CODE[DEFAULT_LEVEL]
    if isinstance(raw, (int, float)):
        n = float(raw)
    else:
        text = str(raw).strip()
        if text in _BY_CODE:
            return _BY_CODE[text]
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return _BY_CODE[DEFAULT_LEVEL]
        n = float(match.group(1))
    n = min(5.0, max(2.0, n))
    code = f"{round(n * 2.0) / 2.0:.1f}"
    return _BY_CODE.get(code, _BY_CODE[DEFAULT_LEVEL])


def public_levels() -> list[dict]:
    return [item.as_public() for item in LEVELS]


def level_context(level: str | LevelInfo | None = None) -> str:
    info = level if isinstance(level, LevelInfo) else parse_level(level)
    return (
        f"球员自己选的水平是 {info.label}。{info.blurb}。"
        "这只说明建议该说到哪一层：入门少讲概念，水平高就不要当成还不会打球。"
        "具体看到什么、怎么改，都按这次画面用你自己的话写，不要套固定段落，也不要另估一个等级。"
        "分数按这次动作的完成度给，不要因为等级故意抬高或压低。"
    )


def level_caveat(level: str | LevelInfo | None = None) -> str:
    info = parse_level(level if not isinstance(level, LevelInfo) else level.code)
    return f"练习建议按你选择的 {info.label} 来写，依据是这次画面，不是固定话术。"
