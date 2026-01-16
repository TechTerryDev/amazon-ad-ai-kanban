# -*- coding: utf-8 -*-
"""
配置：尽量少的“理论指标”，用一组可执行、可复盘的阈值来驱动建议输出。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageConfig:
    """
    分阶段目标（可按你的运营习惯调整）。

    - target_acos：目标 ACoS（0~1）
    - waste_spend：无单浪费花费阈值（$）
    - min_clicks：做判断的最小点击数（避免样本太小）
    - max_change_pct：单次建议调价幅度上限（0~1）
    """

    name: str
    target_acos: float
    waste_spend: float
    min_clicks: int
    max_change_pct: float


STAGES: dict[str, StageConfig] = {
    "launch": StageConfig("launch", target_acos=0.35, waste_spend=15.0, min_clicks=25, max_change_pct=0.20),
    "growth": StageConfig("growth", target_acos=0.25, waste_spend=10.0, min_clicks=20, max_change_pct=0.15),
    "profit": StageConfig("profit", target_acos=0.18, waste_spend=8.0, min_clicks=15, max_change_pct=0.10),
}


def get_stage_config(stage: str) -> StageConfig:
    key = (stage or "").strip().lower()
    if key in STAGES:
        return STAGES[key]
    # 默认用 growth（大多数店铺都能用）
    return STAGES["growth"]

