# -*- coding: utf-8 -*-
"""
确定性指标计算：同一份数据，多次计算结果应完全一致。
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.core.schema import CAN
from src.core.utils import safe_div


def add_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    给明细表增加核心指标列（CTR/CPC/CVR/ACOS/ROAS/CPA）。
    """
    df = df.copy()
    for col in (CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders):
        if col not in df.columns:
            df[col] = 0.0

    df["ctr"] = df.apply(lambda r: safe_div(r[CAN.clicks], r[CAN.impressions]), axis=1)
    df["cpc"] = df.apply(lambda r: safe_div(r[CAN.spend], r[CAN.clicks]), axis=1)
    df["cvr"] = df.apply(lambda r: safe_div(r[CAN.orders], r[CAN.clicks]), axis=1)
    df["acos"] = df.apply(lambda r: safe_div(r[CAN.spend], r[CAN.sales]), axis=1)
    df["roas"] = df.apply(lambda r: safe_div(r[CAN.sales], r[CAN.spend]), axis=1)
    df["cpa"] = df.apply(lambda r: safe_div(r[CAN.spend], r[CAN.orders]), axis=1)
    return df


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    聚合计算：sum 原始数值，再计算指标（避免先算比例再平均导致偏差）。
    """
    if not group_cols:
        # 全局汇总：直接 sum 成一行
        base = {
            CAN.impressions: float(df[CAN.impressions].sum()) if CAN.impressions in df.columns else 0.0,
            CAN.clicks: float(df[CAN.clicks].sum()) if CAN.clicks in df.columns else 0.0,
            CAN.spend: float(df[CAN.spend].sum()) if CAN.spend in df.columns else 0.0,
            CAN.sales: float(df[CAN.sales].sum()) if CAN.sales in df.columns else 0.0,
            CAN.orders: float(df[CAN.orders].sum()) if CAN.orders in df.columns else 0.0,
        }
        return add_core_metrics(pd.DataFrame([base]))

    # 防御性：缺失分组列时，创建空列以便“降级聚合”（避免直接崩溃）
    d = df.copy()
    for c in group_cols:
        if c not in d.columns:
            d[c] = ""
    # 防御性：赛狐导出/清洗过程中可能缺失核心数值列；缺失时补 0，确保 groupby 不崩
    for col in (CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders):
        if col not in d.columns:
            d[col] = 0.0

    g = d.groupby(group_cols, dropna=False, as_index=False).agg(
        impressions=(CAN.impressions, "sum"),
        clicks=(CAN.clicks, "sum"),
        spend=(CAN.spend, "sum"),
        sales=(CAN.sales, "sum"),
        orders=(CAN.orders, "sum"),
    )
    return add_core_metrics(g)


def to_summary_dict(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"impressions": 0.0, "clicks": 0.0, "spend": 0.0, "sales": 0.0, "orders": 0.0}
    s = summarize(df, group_cols=[]).iloc[0].to_dict()
    # 只输出常用字段
    keep = [
        "impressions",
        "clicks",
        "spend",
        "sales",
        "orders",
        "ctr",
        "cpc",
        "cvr",
        "acos",
        "roas",
        "cpa",
    ]
    return {k: float(s.get(k, 0.0) or 0.0) for k in keep}
