# -*- coding: utf-8 -*-
"""
规则工具箱（把关键口径做成可复用函数，避免“同一个概念多套写法”）。

本文件只放“确定性规则”，不依赖外部服务，不做模型推断。
"""

from __future__ import annotations

from typing import Optional


def _to_float(v: object) -> float:
    try:
        if v is None:
            return 0.0
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def is_waste_spend(
    *,
    orders: object,
    sales: object,
    spend: object,
    clicks: Optional[object] = None,
    min_clicks: int = 0,
    min_spend: float = 0.0,
) -> bool:
    """
    判断一条记录是否属于“浪费花费”（用于否词/收口/主题聚合）。

    口径（运营可解释、可复盘）：
    - spend > 0 且 spend >= min_spend
    - orders <= 0 且 sales <= 0
    - 如果 clicks 字段存在（clicks is not None），则要求 clicks >= min_clicks（避免样本太小的误判）

    说明：
    - 这里的 orders/sales/spend/clicks 都用 object 兼容 pandas/None/字符串；
    - 对缺失 clicks 的报表（或聚合表）会自动跳过点击门槛；
    - sales 的存在用于修复一个常见口径问题：orders=0 但 sales>0（例如统计滞后/退款口径），不应计为浪费。
    """
    spend_v = _to_float(spend)
    if spend_v <= 0:
        return False
    if spend_v < float(min_spend or 0.0):
        return False

    orders_v = _to_float(orders)
    sales_v = _to_float(sales)
    if orders_v > 0 or sales_v > 0:
        return False

    if clicks is not None:
        clicks_v = _to_float(clicks)
        if clicks_v < float(min_clicks or 0):
            return False

    return True

