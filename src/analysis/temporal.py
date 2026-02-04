# -*- coding: utf-8 -*-
"""
时间窗口分析（核心目标：不是硬阈值筛选，而是“窗口对比 + 增量效率”）。

方法（确定性、可复盘）：
1) 同一实体（campaign/targeting/...）在“最近N天 vs 前N天”做对比
2) 计算增量效率：
   - marginal_acos = Δspend / Δsales（只在 Δsales>0 时有意义）
   - marginal_cpa  = Δspend / Δorders（只在 Δorders>0 时有意义）
3) 给出信号标签（accelerating / decaying / spend_spike_no_sales / stable）

不引入额外统计依赖（保持可维护），但输出足够结构化给 AI/运营使用。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.core.metrics import summarize
from src.core.risk_scoring import signal_confidence, trend_signal_label
from src.core.schema import CAN
from src.core.utils import safe_div


@dataclass(frozen=True)
class WindowSpec:
    days: int
    recent_start: dt.date
    recent_end: dt.date
    prev_start: dt.date
    prev_end: dt.date


def build_windows(end_date: dt.date, days_list: List[int]) -> List[WindowSpec]:
    out: List[WindowSpec] = []
    for n in sorted({int(x) for x in days_list if int(x) > 0}):
        recent_end = end_date
        recent_start = recent_end - dt.timedelta(days=n - 1)
        prev_end = recent_start - dt.timedelta(days=1)
        prev_start = prev_end - dt.timedelta(days=n - 1)
        out.append(WindowSpec(days=n, recent_start=recent_start, recent_end=recent_end, prev_start=prev_start, prev_end=prev_end))
    return out


def _filter_range(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    if df is None or df.empty or CAN.date not in df.columns:
        return pd.DataFrame()
    x = df[df[CAN.date].notna()].copy()
    x = x[(x[CAN.date] >= start) & (x[CAN.date] <= end)].copy()
    return x


def _infer_shop_end_date(dfs: List[pd.DataFrame]) -> Optional[dt.date]:
    dmax: Optional[dt.date] = None
    for df in dfs:
        if df is None or df.empty or CAN.date not in df.columns:
            continue
        try:
            v = df[CAN.date].max()
        except Exception:
            v = None
        if v is None:
            continue
        if dmax is None or v > dmax:
            dmax = v
    return dmax


def window_compare(
    df: pd.DataFrame,
    group_cols: List[str],
    window: WindowSpec,
    min_spend: float = 5.0,
) -> pd.DataFrame:
    """
    输出每个 group 的 prev/recent 指标与差值，并计算增量效率。
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if any(c not in df.columns for c in group_cols):
        return pd.DataFrame()

    prev_df = _filter_range(df, window.prev_start, window.prev_end)
    rec_df = _filter_range(df, window.recent_start, window.recent_end)
    if prev_df.empty and rec_df.empty:
        return pd.DataFrame()

    prev_s = summarize(prev_df, group_cols).rename(
        columns={
            "impressions": "impr_prev",
            "clicks": "clicks_prev",
            "spend": "spend_prev",
            "sales": "sales_prev",
            "orders": "orders_prev",
            "acos": "acos_prev",
            "cvr": "cvr_prev",
            "ctr": "ctr_prev",
            "cpc": "cpc_prev",
        }
    )
    rec_s = summarize(rec_df, group_cols).rename(
        columns={
            "impressions": "impr_recent",
            "clicks": "clicks_recent",
            "spend": "spend_recent",
            "sales": "sales_recent",
            "orders": "orders_recent",
            "acos": "acos_recent",
            "cvr": "cvr_recent",
            "ctr": "ctr_recent",
            "cpc": "cpc_recent",
        }
    )

    # 前前窗口（用于计算二阶变化）
    preprev_s = pd.DataFrame()
    try:
        preprev_end = window.prev_start - dt.timedelta(days=1)
        preprev_start = preprev_end - dt.timedelta(days=window.days - 1)
        preprev_df = _filter_range(df, preprev_start, preprev_end)
        if preprev_df is not None and not preprev_df.empty:
            preprev_s = summarize(preprev_df, group_cols).rename(
                columns={
                    "spend": "spend_preprev",
                    "sales": "sales_preprev",
                    "orders": "orders_preprev",
                }
            )
    except Exception:
        preprev_s = pd.DataFrame()

    merged = prev_s.merge(rec_s, on=group_cols, how="outer").fillna(0.0)
    if merged.empty:
        return pd.DataFrame()
    if preprev_s is not None and not preprev_s.empty:
        merged = merged.merge(preprev_s, on=group_cols, how="left").fillna(0.0)
    else:
        # 兜底：缺失前前窗口时按 0 处理
        merged["spend_preprev"] = 0.0
        merged["sales_preprev"] = 0.0
        merged["orders_preprev"] = 0.0

    merged["delta_spend"] = merged["spend_recent"] - merged["spend_prev"]
    merged["delta_sales"] = merged["sales_recent"] - merged["sales_prev"]
    merged["delta_orders"] = merged["orders_recent"] - merged["orders_prev"]
    merged["delta_clicks"] = merged["clicks_recent"] - merged["clicks_prev"]

    # 二阶变化（趋势加速度）：delta - delta_prev
    try:
        merged["delta_delta_sales"] = merged["delta_sales"] - (merged["sales_prev"] - merged["sales_preprev"])
    except Exception:
        merged["delta_delta_sales"] = 0.0

    merged["marginal_acos"] = merged.apply(lambda r: safe_div(r["delta_spend"], r["delta_sales"]) if r["delta_sales"] > 0 else 0.0, axis=1)
    merged["marginal_cpa"] = merged.apply(lambda r: safe_div(r["delta_spend"], r["delta_orders"]) if r["delta_orders"] > 0 else 0.0, axis=1)

    # 信号置信度（窗口天数覆盖率）
    try:
        recent_days = rec_df.groupby(group_cols, dropna=False)[CAN.date].nunique().reset_index().rename(columns={CAN.date: "recent_days"})
        merged = merged.merge(recent_days, on=group_cols, how="left").fillna({"recent_days": 0.0})
        merged["signal_confidence"] = (
            pd.to_numeric(merged.get("recent_days", 0.0), errors="coerce").fillna(0.0) / float(window.days)
            if float(window.days) > 0
            else 0.0
        )
        merged["signal_confidence"] = merged["signal_confidence"].clip(lower=0.0, upper=1.0)
        merged = merged.drop(columns=["recent_days"], errors="ignore")
    except Exception:
        merged["signal_confidence"] = 0.0

    # 信号标签：用“增量”判断，而不是静态阈值
    def signal(row: pd.Series) -> str:
        spend_r = float(row.get("spend_recent", 0.0) or 0.0)
        spend_p = float(row.get("spend_prev", 0.0) or 0.0)
        ds = float(row.get("delta_spend", 0.0) or 0.0)
        dsa = float(row.get("delta_sales", 0.0) or 0.0)
        dor = float(row.get("delta_orders", 0.0) or 0.0)
        # 花费突增但没增量 -> 止血优先
        if spend_r >= min_spend and ds > max(min_spend, spend_p * 0.3) and dsa <= 0 and dor <= 0:
            return "spend_spike_no_sales"
        # 增量销售明显>0 且增量效率好
        if dsa > 0 and ds > 0 and float(row.get("marginal_acos", 0.0) or 0.0) > 0:
            return "accelerating"
        # 花费增加但销售不增/下降
        if ds > 0 and dsa <= 0:
            return "decaying"
        # 花费下降但销售不降：可能是提效
        if ds < 0 and dsa >= 0:
            return "efficiency_gain"
        return "stable"

    merged["signal"] = merged.apply(signal, axis=1)

    # 趋势信号：结合一阶与二阶变化
    try:
        merged["trend_signal"] = merged.apply(
            lambda r: trend_signal_label(r.get("delta_sales", 0.0), r.get("delta_delta_sales", 0.0)),
            axis=1,
        )
    except Exception:
        merged["trend_signal"] = ""

    # score：用于排序，越大越需要关注/或越值得加码
    def score(row: pd.Series) -> float:
        sig = str(row.get("signal", ""))
        ds = float(row.get("delta_spend", 0.0) or 0.0)
        dsa = float(row.get("delta_sales", 0.0) or 0.0)
        dor = float(row.get("delta_orders", 0.0) or 0.0)
        spend_r = float(row.get("spend_recent", 0.0) or 0.0)
        # 关注度：止血类用花费权重；加码类用增量订单/销售权重
        if sig in {"spend_spike_no_sales", "decaying"}:
            return min(100.0, max(0.0, abs(ds) * 2 + spend_r))
        if sig in {"accelerating", "efficiency_gain"}:
            return min(100.0, max(0.0, dsa * 0.5 + dor * 5))
        return min(60.0, abs(ds) + abs(dsa))

    merged["score"] = merged.apply(score, axis=1)
    merged["window_days"] = int(window.days)
    merged["recent_start"] = str(window.recent_start)
    merged["recent_end"] = str(window.recent_end)
    merged["prev_start"] = str(window.prev_start)
    merged["prev_end"] = str(window.prev_end)

    return merged


def build_temporal_insights(
    camp: pd.DataFrame,
    tgt: pd.DataFrame,
    windows_days: List[int],
    min_spend: float = 5.0,
) -> Dict[str, object]:
    """
    生成店铺级的多窗口洞察（campaign/targeting）。
    """
    end_date = _infer_shop_end_date([camp, tgt])
    if end_date is None:
        return {"shop_end_date": "", "windows": []}
    windows = build_windows(end_date, windows_days)
    out: Dict[str, object] = {
        "shop_end_date": str(end_date),
        "windows": [
            {
                "days": w.days,
                "recent_start": str(w.recent_start),
                "recent_end": str(w.recent_end),
                "prev_start": str(w.prev_start),
                "prev_end": str(w.prev_end),
            }
            for w in windows
        ],
    }

    camp_tables = []
    tgt_tables = []
    for w in windows:
        if camp is not None and not camp.empty and CAN.campaign in camp.columns and CAN.ad_type in camp.columns:
            t = window_compare(camp, [CAN.ad_type, CAN.campaign], w, min_spend=min_spend)
            if not t.empty:
                camp_tables.append(t)
        if tgt is not None and not tgt.empty and CAN.targeting in tgt.columns and CAN.ad_type in tgt.columns:
            # targeting 先用 ad_type + targeting（更通用）；campaign 字段如果有，会在报告/动作里引用原字段
            t = window_compare(tgt, [CAN.ad_type, CAN.targeting], w, min_spend=min_spend)
            if not t.empty:
                tgt_tables.append(t)

    out["campaign_windows"] = pd.concat(camp_tables, ignore_index=True).to_dict(orient="records") if camp_tables else []
    out["targeting_windows"] = pd.concat(tgt_tables, ignore_index=True).to_dict(orient="records") if tgt_tables else []
    return out
