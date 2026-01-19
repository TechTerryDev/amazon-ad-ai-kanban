# -*- coding: utf-8 -*-
"""
风险评分与趋势信号（轻量级 P0 版）。

说明：
- 内部计算使用 Sigmoid 作为软阈值映射
- 输出提供风险档位（高/中/低）便于运营快速决策
- 提供趋势信号与置信度的通用工具函数
"""

from __future__ import annotations

import math
from typing import Dict, Optional


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _sigmoid(x: float) -> float:
    """
    标准 Sigmoid（数值稳定版）。
    """
    try:
        z = float(x)
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)
    except Exception:
        return 0.5


def sigmoid_risk(x: float, midpoint: float = 0.0, steepness: float = 1.0, inverse: bool = False) -> float:
    """
    通用 Sigmoid 风险映射：
    - inverse=True 表示“值越小风险越高”（例如覆盖天数）
    """
    try:
        z = (_safe_float(x) - _safe_float(midpoint)) * _safe_float(steepness, 1.0)
        p = _sigmoid(z)
        return 1.0 - p if bool(inverse) else p
    except Exception:
        return 0.0


def oos_risk_probability(days_coverage: float, midpoint: float = 7.0, steepness: float = 0.5) -> float:
    """
    库存覆盖天数 -> 断货风险概率（覆盖越低风险越高）。
    """
    return sigmoid_risk(days_coverage, midpoint=midpoint, steepness=steepness, inverse=True)


def acos_risk_probability(
    current_acos: float,
    target_acos: float,
    tolerance: float = 0.1,
    steepness: float = 10.0,
) -> float:
    """
    ACoS 偏离度 -> 超标风险概率（高于目标越多风险越高）。
    """
    try:
        cur = _safe_float(current_acos, 0.0)
        tgt = _safe_float(target_acos, 0.0)
        if tgt <= 0:
            return 0.0
        deviation = (cur - tgt) / max(tgt, 1e-6)
        return sigmoid_risk(deviation, midpoint=_safe_float(tolerance, 0.0), steepness=_safe_float(steepness, 10.0), inverse=False)
    except Exception:
        return 0.0


def cvr_drop_risk_probability(delta_cvr: float, midpoint: float = 0.0, steepness: float = 50.0) -> float:
    """
    CVR 下降风险：delta_cvr 为负时风险上升。
    """
    return sigmoid_risk(delta_cvr, midpoint=midpoint, steepness=steepness, inverse=True)


def calculate_overall_risk(
    oos_risk: Optional[float] = None,
    acos_risk: Optional[float] = None,
    cvr_risk: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    综合风险评分（只使用可用信号；权重会自动归一）。
    """
    try:
        base = {"oos": 0.4, "acos": 0.35, "cvr": 0.25}
        if isinstance(weights, dict):
            base.update({k: float(v) for k, v in weights.items() if k in base})

        items = []
        for key, val in (("oos", oos_risk), ("acos", acos_risk), ("cvr", cvr_risk)):
            if val is None:
                continue
            try:
                v = float(val)
                if math.isnan(v) or math.isinf(v):
                    continue
            except Exception:
                continue
            items.append((base.get(key, 0.0), float(v)))

        if not items:
            return 0.0
        w_sum = sum(w for w, _ in items)
        if w_sum <= 0:
            return float(sum(v for _, v in items) / max(1, len(items)))
        score = sum(w * v for w, v in items) / w_sum
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return 0.0


def risk_level(score: float, high: float = 0.7, mid: float = 0.3) -> str:
    """
    风险档位（高/中/低）。
    """
    try:
        v = _safe_float(score, 0.0)
        if v >= float(high):
            return "高"
        if v >= float(mid):
            return "中"
        return "低"
    except Exception:
        return "低"


def trend_signal_label(delta_value: float, delta_delta_value: float, eps: float = 1e-6) -> str:
    """
    趋势信号：结合一阶变化与二阶变化判断加速/减速方向。
    """
    try:
        d1 = _safe_float(delta_value, 0.0)
        d2 = _safe_float(delta_delta_value, 0.0)
        if abs(d1) <= float(eps):
            return "平稳"
        if d1 < 0 and d2 < 0:
            return "加速下降"
        if d1 < 0 and d2 >= 0:
            return "减速下降"
        if d1 > 0 and d2 > 0:
            return "加速上升"
        if d1 > 0 and d2 <= 0:
            return "减速上升"
        return "平稳"
    except Exception:
        return "平稳"


def signal_confidence(actual_days: float, window_days: float) -> float:
    """
    信号置信度：数据覆盖天数 / 窗口天数（0~1）。
    """
    try:
        a = _safe_float(actual_days, 0.0)
        w = _safe_float(window_days, 0.0)
        if w <= 0:
            return 0.0
        return max(0.0, min(1.0, a / w))
    except Exception:
        return 0.0
