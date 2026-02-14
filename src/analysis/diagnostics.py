# -*- coding: utf-8 -*-
"""
更“数据驱动”的诊断（不是简单规则表）：

目标：
- 利用你私有报表里的结构与时间维度，做“趋势/对比/根因”分析
- 输出结构化结果给 AI：AI 只负责解释与写建议，不参与算数

不额外引入 scipy/statsmodels 等依赖（保持可维护性）。
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.core.config import StageConfig
from src.core.policy import OpsPolicy
from src.core.metrics import summarize, to_summary_dict
from src.core.schema import CAN
from src.core.utils import safe_div


def _build_shop_daily_metrics_with_promo_adjustment(
    product_analysis_shop: pd.DataFrame,
    policy: Optional[OpsPolicy] = None,
) -> pd.DataFrame:
    """
    按日聚合店铺经营指标，并做“促销尖峰”轻量校正（不覆盖原始值）。

    说明：
    - 只用于 compare 的补充字段，原始 delta 仍保留；
    - 促销识别采用“相对历史中位数”的规则化方法，避免引入外部依赖。
    """
    if product_analysis_shop is None or product_analysis_shop.empty or CAN.date not in product_analysis_shop.columns:
        return pd.DataFrame()

    pa = product_analysis_shop.copy()
    pa = pa[pa[CAN.date].notna()].copy()
    if pa.empty:
        return pd.DataFrame()

    num_cols = ["销售额", "订单量", "Sessions", "广告花费", "广告销售额", "毛利润"]
    keep_cols: List[str] = []
    for c in num_cols:
        if c in pa.columns:
            pa[c] = pd.to_numeric(pa[c], errors="coerce").fillna(0.0)
            keep_cols.append(c)
    if not keep_cols:
        return pd.DataFrame()

    daily = pa.groupby(CAN.date, dropna=False, as_index=False)[keep_cols].sum().sort_values(CAN.date).copy()
    if daily.empty:
        return pd.DataFrame()

    for c in ("销售额", "广告花费", "毛利润"):
        if c not in daily.columns:
            daily[c] = 0.0

    pa_cfg = getattr(policy, "dashboard_promo_adjustment", None) if isinstance(policy, OpsPolicy) else None
    enabled = bool(getattr(pa_cfg, "enabled", True) if pa_cfg is not None else True)
    if not enabled:
        daily["promo_spike"] = 0
        daily["sales_spike_ratio"] = 1.0
        daily["spend_spike_ratio"] = 1.0
        daily["sales_corrected"] = daily["销售额"]
        daily["ad_spend_corrected"] = daily["广告花费"]
        daily["profit_corrected"] = daily["毛利润"]
        return daily

    lookback_days = int(getattr(pa_cfg, "baseline_lookback_days", 28) if pa_cfg is not None else 28)
    min_periods = int(getattr(pa_cfg, "baseline_min_periods", 7) if pa_cfg is not None else 7)
    lookback_days = max(7, lookback_days)
    min_periods = max(1, min(min_periods, lookback_days))
    sales_spike_thr = float(getattr(pa_cfg, "sales_spike_threshold", 2.0) if pa_cfg is not None else 2.0)
    spend_spike_thr = float(getattr(pa_cfg, "spend_spike_threshold", 1.5) if pa_cfg is not None else 1.5)
    sales_spike_thr_alt = float(getattr(pa_cfg, "sales_spike_threshold_alt", 1.6) if pa_cfg is not None else 1.6)
    spend_spike_thr_alt = float(getattr(pa_cfg, "spend_spike_threshold_alt", 2.0) if pa_cfg is not None else 2.0)
    damp = float(getattr(pa_cfg, "damp_ratio", 0.35) if pa_cfg is not None else 0.35)
    damp = max(0.0, min(1.0, damp))

    def _baseline_col(s: pd.Series, lookback_days: int = 28, min_periods: int = 7) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").fillna(0.0)
        # 只用“过去”数据做基线，避免未来信息泄漏
        hist = x.shift(1).rolling(window=int(lookback_days), min_periods=int(min_periods)).median()
        hist2 = x.shift(1).expanding(min_periods=1).median()
        base = hist.fillna(hist2)
        if base.isna().all():
            med = float(x.median()) if len(x) > 0 else 0.0
            base = pd.Series([med] * len(x), index=x.index)
        base = base.fillna(float(base.median()) if not base.empty else 0.0)
        return base.clip(lower=0.0)

    daily["baseline_sales"] = _baseline_col(daily["销售额"], lookback_days=lookback_days, min_periods=min_periods)
    daily["baseline_ad_spend"] = _baseline_col(daily["广告花费"], lookback_days=lookback_days, min_periods=min_periods)
    daily["baseline_profit"] = _baseline_col(daily["毛利润"], lookback_days=lookback_days, min_periods=min_periods)

    sales_intensity = daily.apply(
        lambda r: safe_div(float(r.get("销售额", 0.0) or 0.0), max(float(r.get("baseline_sales", 0.0) or 0.0), 1.0)),
        axis=1,
    )
    spend_intensity = daily.apply(
        lambda r: safe_div(float(r.get("广告花费", 0.0) or 0.0), max(float(r.get("baseline_ad_spend", 0.0) or 0.0), 1.0)),
        axis=1,
    )

    # 促销/大促候选：销量与广告花费同步显著抬升
    daily["promo_spike"] = (
        ((sales_intensity >= sales_spike_thr) & (spend_intensity >= spend_spike_thr))
        | ((sales_intensity >= sales_spike_thr_alt) & (spend_intensity >= spend_spike_thr_alt))
    ).astype(int)
    daily["sales_spike_ratio"] = sales_intensity.round(4)
    daily["spend_spike_ratio"] = spend_intensity.round(4)

    # 轻量校正：对促销尖峰只保留部分增量，避免环比被单次活动“拉爆”
    sales_delta = (daily["销售额"] - daily["baseline_sales"]).clip(lower=0.0)
    spend_delta = (daily["广告花费"] - daily["baseline_ad_spend"]).clip(lower=0.0)
    profit_delta = (daily["毛利润"] - daily["baseline_profit"]).clip(lower=0.0)
    daily["sales_corrected"] = daily["销售额"]
    daily["ad_spend_corrected"] = daily["广告花费"]
    daily["profit_corrected"] = daily["毛利润"]
    mask = daily["promo_spike"] > 0
    daily.loc[mask, "sales_corrected"] = daily.loc[mask, "baseline_sales"] + (sales_delta[mask] * damp)
    daily.loc[mask, "ad_spend_corrected"] = daily.loc[mask, "baseline_ad_spend"] + (spend_delta[mask] * damp)
    daily.loc[mask, "profit_corrected"] = daily.loc[mask, "baseline_profit"] + (profit_delta[mask] * damp)
    daily["sales_corrected"] = daily["sales_corrected"].clip(lower=0.0)
    daily["ad_spend_corrected"] = daily["ad_spend_corrected"].clip(lower=0.0)
    return daily


def _shop_roll_compare_from_pa(
    product_analysis_shop: pd.DataFrame,
    window_days: int,
    policy: Optional[OpsPolicy] = None,
) -> Dict[str, object]:
    """
    店铺层滚动环比（优先用产品分析的“经营底座”，因为它包含自然流量+广告合计）。
    recent N 天 vs 前 N 天：
    - Δ销售额/订单/Sessions/广告花费
    - marginal_tacos = Δ广告花费 / Δ销售额（Δ销售额>0 时有意义）
    """
    if product_analysis_shop is None or product_analysis_shop.empty or CAN.date not in product_analysis_shop.columns:
        return {}
    pa = product_analysis_shop.copy()
    pa = pa[pa[CAN.date].notna()].copy()
    if pa.empty:
        return {}
    try:
        dmax = pa[CAN.date].max()
    except Exception:
        return {}
    if not isinstance(dmax, dt.date):
        return {}

    n = int(window_days)
    if n <= 0:
        return {}
    recent_end = dmax
    recent_start = recent_end - dt.timedelta(days=n - 1)
    prev_end = recent_start - dt.timedelta(days=1)
    prev_start = prev_end - dt.timedelta(days=n - 1)

    def _sum_range(start: dt.date, end: dt.date) -> Dict[str, float]:
        x = pa[(pa[CAN.date] >= start) & (pa[CAN.date] <= end)].copy()
        out: Dict[str, float] = {}
        for col, key in [
            ("销售额", "sales"),
            ("订单量", "orders"),
            ("Sessions", "sessions"),
            ("广告花费", "ad_spend"),
            ("广告销售额", "ad_sales"),
            ("毛利润", "profit"),
        ]:
            if col in x.columns:
                try:
                    out[key] = float(pd.to_numeric(x[col], errors="coerce").fillna(0.0).sum())
                except Exception:
                    out[key] = 0.0
        return out

    prev = _sum_range(prev_start, prev_end)
    rec = _sum_range(recent_start, recent_end)

    daily = _build_shop_daily_metrics_with_promo_adjustment(pa, policy=policy)
    promo_days_prev = 0
    promo_days_recent = 0
    sales_prev_corr = 0.0
    sales_recent_corr = 0.0
    ad_spend_prev_corr = 0.0
    ad_spend_recent_corr = 0.0
    profit_prev_corr = 0.0
    profit_recent_corr = 0.0
    if daily is not None and not daily.empty:
        d_prev = daily[(daily[CAN.date] >= prev_start) & (daily[CAN.date] <= prev_end)].copy()
        d_rec = daily[(daily[CAN.date] >= recent_start) & (daily[CAN.date] <= recent_end)].copy()
        if not d_prev.empty:
            promo_days_prev = int(pd.to_numeric(d_prev.get("promo_spike", 0), errors="coerce").fillna(0).sum())
            sales_prev_corr = float(pd.to_numeric(d_prev.get("sales_corrected", 0.0), errors="coerce").fillna(0.0).sum())
            ad_spend_prev_corr = float(pd.to_numeric(d_prev.get("ad_spend_corrected", 0.0), errors="coerce").fillna(0.0).sum())
            profit_prev_corr = float(pd.to_numeric(d_prev.get("profit_corrected", 0.0), errors="coerce").fillna(0.0).sum())
        if not d_rec.empty:
            promo_days_recent = int(pd.to_numeric(d_rec.get("promo_spike", 0), errors="coerce").fillna(0).sum())
            sales_recent_corr = float(pd.to_numeric(d_rec.get("sales_corrected", 0.0), errors="coerce").fillna(0.0).sum())
            ad_spend_recent_corr = float(pd.to_numeric(d_rec.get("ad_spend_corrected", 0.0), errors="coerce").fillna(0.0).sum())
            profit_recent_corr = float(pd.to_numeric(d_rec.get("profit_corrected", 0.0), errors="coerce").fillna(0.0).sum())

    sales_prev = float(prev.get("sales", 0.0))
    sales_recent = float(rec.get("sales", 0.0))
    ad_spend_prev = float(prev.get("ad_spend", 0.0))
    ad_spend_recent = float(rec.get("ad_spend", 0.0))
    orders_prev = float(prev.get("orders", 0.0))
    orders_recent = float(rec.get("orders", 0.0))
    sessions_prev = float(prev.get("sessions", 0.0))
    sessions_recent = float(rec.get("sessions", 0.0))
    ad_sales_prev = float(prev.get("ad_sales", 0.0))
    ad_sales_recent = float(rec.get("ad_sales", 0.0))
    profit_prev = float(prev.get("profit", 0.0))
    profit_recent = float(rec.get("profit", 0.0))

    delta_sales = float(sales_recent - sales_prev)
    delta_ad_spend = float(ad_spend_recent - ad_spend_prev)
    delta_orders = float(orders_recent - orders_prev)
    delta_sessions = float(sessions_recent - sessions_prev)
    delta_profit = float(profit_recent - profit_prev)
    delta_sales_corr = float(sales_recent_corr - sales_prev_corr)
    delta_ad_spend_corr = float(ad_spend_recent_corr - ad_spend_prev_corr)
    delta_profit_corr = float(profit_recent_corr - profit_prev_corr)
    correction_applied = bool((promo_days_prev + promo_days_recent) > 0)

    out = {
        "window_days": n,
        "recent_start": str(recent_start),
        "recent_end": str(recent_end),
        "prev_start": str(prev_start),
        "prev_end": str(prev_end),
        "sales_prev": round(sales_prev, 2),
        "sales_recent": round(sales_recent, 2),
        "ad_spend_prev": round(ad_spend_prev, 2),
        "ad_spend_recent": round(ad_spend_recent, 2),
        "orders_prev": round(orders_prev, 2),
        "orders_recent": round(orders_recent, 2),
        "sessions_prev": round(sessions_prev, 2),
        "sessions_recent": round(sessions_recent, 2),
        "ad_sales_prev": round(ad_sales_prev, 2),
        "ad_sales_recent": round(ad_sales_recent, 2),
        "profit_prev": round(profit_prev, 2),
        "profit_recent": round(profit_recent, 2),
        "delta_sales": round(delta_sales, 2),
        "delta_ad_spend": round(delta_ad_spend, 2),
        "delta_orders": round(delta_orders, 2),
        "delta_sessions": round(delta_sessions, 2),
        "delta_profit": round(delta_profit, 2),
        "sales_prev_corrected": round(sales_prev_corr, 2),
        "sales_recent_corrected": round(sales_recent_corr, 2),
        "ad_spend_prev_corrected": round(ad_spend_prev_corr, 2),
        "ad_spend_recent_corrected": round(ad_spend_recent_corr, 2),
        "profit_prev_corrected": round(profit_prev_corr, 2),
        "profit_recent_corrected": round(profit_recent_corr, 2),
        "delta_sales_corrected": round(delta_sales_corr, 2),
        "delta_ad_spend_corrected": round(delta_ad_spend_corr, 2),
        "delta_profit_corrected": round(delta_profit_corr, 2),
        "marginal_tacos_corrected": round(safe_div(delta_ad_spend_corr, delta_sales_corr) if delta_sales_corr > 0 else 0.0, 4),
        "promo_days_prev": int(promo_days_prev),
        "promo_days_recent": int(promo_days_recent),
        "promo_or_seasonality_adjusted": bool(correction_applied),
        "promo_or_seasonality_note": (
            f"promo_spike_detected(prev={int(promo_days_prev)},recent={int(promo_days_recent)})"
            if correction_applied
            else ""
        ),
        "tacos_prev": round(safe_div(ad_spend_prev, sales_prev) if sales_prev > 0 else 0.0, 4),
        "tacos_recent": round(safe_div(ad_spend_recent, sales_recent) if sales_recent > 0 else 0.0, 4),
        "ad_sales_share_prev": round(safe_div(ad_sales_prev, sales_prev) if sales_prev > 0 else 0.0, 4),
        "ad_sales_share_recent": round(safe_div(ad_sales_recent, sales_recent) if sales_recent > 0 else 0.0, 4),
        "ad_acos_prev": round(safe_div(ad_spend_prev, ad_sales_prev) if ad_sales_prev > 0 else 0.0, 4),
        "ad_acos_recent": round(safe_div(ad_spend_recent, ad_sales_recent) if ad_sales_recent > 0 else 0.0, 4),
        "cvr_prev": round(safe_div(orders_prev, sessions_prev) if sessions_prev > 0 else 0.0, 4),
        "cvr_recent": round(safe_div(orders_recent, sessions_recent) if sessions_recent > 0 else 0.0, 4),
        "aov_prev": round(safe_div(sales_prev, orders_prev) if orders_prev > 0 else 0.0, 4),
        "aov_recent": round(safe_div(sales_recent, orders_recent) if orders_recent > 0 else 0.0, 4),
        "marginal_tacos": round(safe_div(delta_ad_spend, delta_sales) if delta_sales > 0 else 0.0, 4),
    }
    return out


def _asin_roll_drivers_from_pa(
    product_analysis_shop: pd.DataFrame,
    lifecycle_board: pd.DataFrame,
    window_days: int,
    top_n: int = 12,
) -> Dict[str, List[Dict[str, object]]]:
    """
    店铺“变化来源”（drivers）：把店铺滚动环比拆到 ASIN 级别，回答：
    - 最近 N 天销售额/广告花费的变化，主要由哪些 ASIN 驱动？

    说明：
    - 这里优先用产品分析（经营底座）做拆分，因为它同时包含自然+广告口径；
    - 返回两张 Top 表：按 delta_sales 与按 delta_ad_spend 排序。
    """
    if product_analysis_shop is None or product_analysis_shop.empty or CAN.date not in product_analysis_shop.columns:
        return {"by_delta_sales": [], "by_delta_ad_spend": []}
    if "ASIN" not in product_analysis_shop.columns:
        return {"by_delta_sales": [], "by_delta_ad_spend": []}

    pa = product_analysis_shop.copy()
    pa = pa[pa[CAN.date].notna()].copy()
    if pa.empty:
        return {"by_delta_sales": [], "by_delta_ad_spend": []}
    try:
        dmax = pa[CAN.date].max()
    except Exception:
        return {"by_delta_sales": [], "by_delta_ad_spend": []}
    if not isinstance(dmax, dt.date):
        return {"by_delta_sales": [], "by_delta_ad_spend": []}

    n = int(window_days)
    if n <= 0:
        return {"by_delta_sales": [], "by_delta_ad_spend": []}
    recent_end = dmax
    recent_start = recent_end - dt.timedelta(days=n - 1)
    prev_end = recent_start - dt.timedelta(days=1)
    prev_start = prev_end - dt.timedelta(days=n - 1)

    pa["asin_norm"] = pa["ASIN"].astype(str).str.upper().str.strip()
    pa = pa[(pa["asin_norm"] != "") & (pa["asin_norm"].str.lower() != "nan")].copy()
    if pa.empty:
        return {"by_delta_sales": [], "by_delta_ad_spend": []}

    use_cols = []
    for col in ("销售额", "订单量", "Sessions", "广告花费", "广告销售额", "毛利润"):
        if col in pa.columns:
            use_cols.append(col)
            pa[col] = pd.to_numeric(pa[col], errors="coerce").fillna(0.0)

    def _agg_range(start: dt.date, end: dt.date) -> pd.DataFrame:
        x = pa[(pa[CAN.date] >= start) & (pa[CAN.date] <= end)].copy()
        if x.empty:
            return pd.DataFrame(columns=["asin_norm"] + use_cols)
        agg_map = {c: "sum" for c in use_cols}
        return x.groupby("asin_norm", dropna=False, as_index=False).agg(agg_map).copy()

    prev = _agg_range(prev_start, prev_end).rename(columns={c: f"{c}_prev" for c in use_cols})
    rec = _agg_range(recent_start, recent_end).rename(columns={c: f"{c}_recent" for c in use_cols})
    merged = prev.merge(rec, on="asin_norm", how="outer").fillna(0.0)
    if merged.empty:
        return {"by_delta_sales": [], "by_delta_ad_spend": []}

    # delta
    if "销售额_prev" in merged.columns and "销售额_recent" in merged.columns:
        merged["delta_sales"] = merged["销售额_recent"] - merged["销售额_prev"]
    else:
        merged["delta_sales"] = 0.0
    if "广告花费_prev" in merged.columns and "广告花费_recent" in merged.columns:
        merged["delta_ad_spend"] = merged["广告花费_recent"] - merged["广告花费_prev"]
    else:
        merged["delta_ad_spend"] = 0.0

    # marginal_tacos（Δ广告花费/Δ销售额）
    merged["marginal_tacos"] = merged.apply(
        lambda r: safe_div(float(r.get("delta_ad_spend", 0.0) or 0.0), float(r.get("delta_sales", 0.0) or 0.0))
        if float(r.get("delta_sales", 0.0) or 0.0) > 0
        else 0.0,
        axis=1,
    )

    # 拼接产品信息（phase/inventory/product_name）
    try:
        b = lifecycle_board.copy() if lifecycle_board is not None else pd.DataFrame()
        if b is not None and not b.empty and "asin" in b.columns:
            b["asin_norm"] = b["asin"].astype(str).str.upper().str.strip()
            keep = ["asin_norm"]
            for c in ("product_name", "current_phase", "inventory", "flag_low_inventory", "flag_oos"):
                if c in b.columns:
                    keep.append(c)
            b = b[keep].drop_duplicates("asin_norm")
            merged = merged.merge(b, on="asin_norm", how="left")
    except Exception:
        pass

    # 输出列控制（更适合给 AI/运营看）
    out_cols = [
        "asin_norm",
        "product_name",
        "current_phase",
        "inventory",
        "flag_low_inventory",
        "flag_oos",
        "delta_sales",
        "delta_ad_spend",
        "marginal_tacos",
    ]
    for c in ("订单量_prev", "订单量_recent", "Sessions_prev", "Sessions_recent", "毛利润_prev", "毛利润_recent"):
        if c in merged.columns:
            out_cols.append(c)

    view = merged[out_cols].copy()
    # 可读性：四舍五入
    for c in ("delta_sales", "delta_ad_spend", "marginal_tacos"):
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0.0).round(4 if "marginal" in c else 2)
    for c in ("订单量_prev", "订单量_recent", "Sessions_prev", "Sessions_recent"):
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0.0).round(0)
    for c in ("毛利润_prev", "毛利润_recent"):
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0.0).round(2)

    by_sales = view.sort_values("delta_sales", ascending=False).head(int(top_n)).copy()
    by_spend = view.sort_values("delta_ad_spend", ascending=False).head(int(top_n)).copy()

    # 写明窗口（方便报告/AI 引用）
    def _with_meta(df: pd.DataFrame) -> List[Dict[str, object]]:
        rows = df.copy()
        rows = rows.rename(columns={"asin_norm": "asin"})
        rows["window_days"] = int(n)
        rows["recent_start"] = str(recent_start)
        rows["recent_end"] = str(recent_end)
        rows["prev_start"] = str(prev_start)
        rows["prev_end"] = str(prev_end)
        try:
            return rows.to_dict(orient="records")
        except Exception:
            return []

    return {"by_delta_sales": _with_meta(by_sales), "by_delta_ad_spend": _with_meta(by_spend)}


def diagnose_shop_scorecard(
    cfg: StageConfig,
    camp: pd.DataFrame,
    product_analysis_shop: pd.DataFrame,
    lifecycle_board: pd.DataFrame,
    windows_days: Optional[List[int]] = None,
    policy: Optional[OpsPolicy] = None,
) -> Dict[str, object]:
    """
    店铺诊断（scorecard）：把“店铺到底哪里出问题/哪里能放量”变成固定口径的结构化输出。

    输出包含四类信息：
    1) kpi：广告口径 + 经营口径（TACOS/广告依赖度/利润）
    2) phase：生命周期阶段分布（数量/花费占比）
    3) concentration：花费集中度（Top1/3/5/10 ASIN 花费占比）
    4) compares：7/14/30 滚动环比（基于产品分析经营底座）
    """
    windows_days = windows_days or [7, 14, 30]

    # 1) 广告 KPI（以 campaign 报告为主口径）
    ad_kpi = {}
    try:
        ad_kpi = to_summary_dict(camp) if camp is not None and not camp.empty else {}
    except Exception:
        ad_kpi = {}

    # 2) 经营 KPI（以产品分析为主口径：自然+广告合计）
    biz = {}
    try:
        pa = product_analysis_shop.copy() if product_analysis_shop is not None else pd.DataFrame()
        if pa is not None and not pa.empty:
            sales = float(pa["销售额"].sum()) if "销售额" in pa.columns else 0.0
            ad_spend = float(pa["广告花费"].sum()) if "广告花费" in pa.columns else 0.0
            ad_sales = float(pa["广告销售额"].sum()) if "广告销售额" in pa.columns else 0.0
            ad_orders = float(pa["广告订单量"].sum()) if "广告订单量" in pa.columns else 0.0
            profit = float(pa["毛利润"].sum()) if "毛利润" in pa.columns else 0.0
            orders = float(pa["订单量"].sum()) if "订单量" in pa.columns else 0.0
            sessions = float(pa["Sessions"].sum()) if "Sessions" in pa.columns else 0.0
            organic_sales = max(0.0, sales - ad_sales) if (sales > 0 and ad_sales > 0) else 0.0
            organic_orders = max(0.0, orders - ad_orders) if (orders > 0 and ad_orders > 0) else 0.0
            biz = {
                "sales_total": round(sales, 2),
                "orders_total": round(orders, 2),
                "sessions_total": round(sessions, 2),
                "profit_total": round(profit, 2),
                "ad_spend_total": round(ad_spend, 2),
                "ad_sales_total": round(ad_sales, 2),
                "ad_orders_total": round(ad_orders, 2),
                "organic_sales_total": round(organic_sales, 2),
                "organic_orders_total": round(organic_orders, 2),
                "tacos_total": round(safe_div(ad_spend, sales), 4) if sales > 0 else 0.0,
                "ad_acos_total": round(safe_div(ad_spend, ad_sales), 4) if ad_sales > 0 else 0.0,
                "ad_sales_share_total": round(safe_div(ad_sales, sales), 4) if sales > 0 else 0.0,
                "organic_sales_share_total": round(safe_div(organic_sales, sales), 4) if sales > 0 else 0.0,
                "ad_orders_share_total": round(safe_div(ad_orders, orders), 4) if orders > 0 else 0.0,
                "organic_orders_share_total": round(safe_div(organic_orders, orders), 4) if orders > 0 else 0.0,
                "cvr_total": round(safe_div(orders, sessions), 4) if sessions > 0 else 0.0,
                "aov_total": round(safe_div(sales, orders), 4) if orders > 0 else 0.0,
                "sales_per_session_total": round(safe_div(sales, sessions), 4) if sessions > 0 else 0.0,
            }
    except Exception:
        biz = {}

    # 3) 生命周期阶段分布 & 花费占比（用 lifecycle_board 的 7天滚动花费做“运营优先级口径”）
    phase = {"counts": [], "spend_share": []}
    try:
        board = lifecycle_board.copy() if lifecycle_board is not None else pd.DataFrame()
        if board is not None and not board.empty and "current_phase" in board.columns:
            counts = (
                board.groupby("current_phase", dropna=False)
                .size()
                .reset_index(name="asin_count")
                .sort_values("asin_count", ascending=False)
                .to_dict(orient="records")
            )
            phase["counts"] = counts

            if "ad_spend_roll" in board.columns:
                b2 = board.copy()
                b2["ad_spend_roll"] = pd.to_numeric(b2["ad_spend_roll"], errors="coerce").fillna(0.0)
                total = float(b2["ad_spend_roll"].sum())
                spend_by = (
                    b2.groupby("current_phase", dropna=False)["ad_spend_roll"]
                    .sum()
                    .reset_index(name="ad_spend_roll_sum")
                    .sort_values("ad_spend_roll_sum", ascending=False)
                )
                spend_by["ad_spend_share"] = spend_by["ad_spend_roll_sum"].apply(lambda x: safe_div(float(x), total) if total > 0 else 0.0)
                phase["spend_share"] = spend_by.to_dict(orient="records")
    except Exception:
        phase = {"counts": [], "spend_share": []}

    # 4) 花费集中度（Top N ASIN 花费占比）
    concentration = {}
    try:
        board = lifecycle_board.copy() if lifecycle_board is not None else pd.DataFrame()
        if board is not None and not board.empty and "ad_spend_roll" in board.columns and "asin" in board.columns:
            b = board.copy()
            b["ad_spend_roll"] = pd.to_numeric(b["ad_spend_roll"], errors="coerce").fillna(0.0)
            b = b.sort_values("ad_spend_roll", ascending=False)
            total = float(b["ad_spend_roll"].sum())
            def _share(n: int) -> float:
                if total <= 0:
                    return 0.0
                return float(b.head(n)["ad_spend_roll"].sum()) / total

            concentration = {
                "total_ad_spend_roll": round(total, 2),
                "top1_share": round(_share(1), 4),
                "top3_share": round(_share(3), 4),
                "top5_share": round(_share(5), 4),
                "top10_share": round(_share(10), 4),
                "top20_share": round(_share(20), 4),
                "asin_count": int(b["asin"].nunique()),
            }
    except Exception:
        concentration = {}

    # 5) 风险摘要（库存/断货）
    risks = {}
    try:
        board = lifecycle_board.copy() if lifecycle_board is not None else pd.DataFrame()
        if board is not None and not board.empty:
            low_inv = int(pd.to_numeric(board.get("flag_low_inventory"), errors="coerce").fillna(0).sum()) if "flag_low_inventory" in board.columns else 0
            oos = int(pd.to_numeric(board.get("flag_oos"), errors="coerce").fillna(0).sum()) if "flag_oos" in board.columns else 0
            risks = {"asin_low_inventory_count": low_inv, "asin_oos_count": oos}
    except Exception:
        risks = {}

    # 6) 店铺层滚动环比（7/14/30）
    compares: List[Dict[str, object]] = []
    for w in windows_days:
        try:
            c = _shop_roll_compare_from_pa(product_analysis_shop, int(w), policy=policy)
            if c:
                compares.append(c)
        except Exception:
            continue

    # 7) 变化来源（drivers）：默认取 7 天一版（可按需扩展到 14/30）
    drivers: Dict[str, object] = {}
    try:
        d7 = _asin_roll_drivers_from_pa(product_analysis_shop, lifecycle_board, window_days=7, top_n=12)
        drivers["window_7d"] = d7
    except Exception:
        drivers = {}

    return {
        "target_acos": float(cfg.target_acos),
        "waste_spend_threshold": float(cfg.waste_spend),
        "ad_kpi": ad_kpi,
        "biz_kpi": biz,
        "phase": phase,
        "concentration": concentration,
        "risks": risks,
        "compares": compares,
        "drivers": drivers,
    }


def _pick_period_dates(dates: List[dt.date], recent_days: int = 7) -> Tuple[List[dt.date], List[dt.date]]:
    """
    按日期切两个窗口：
    - 如果日期>=2*recent_days：最近 recent_days vs 前 recent_days
    - 否则：按一半一半切
    """
    if not dates:
        return ([], [])
    ds = sorted(dates)
    if len(ds) >= recent_days * 2:
        recent = ds[-recent_days:]
        prev = ds[-recent_days * 2 : -recent_days]
        return (prev, recent)
    mid = max(1, len(ds) // 2)
    prev = ds[:mid]
    recent = ds[mid:]
    return (prev, recent)


def _fmt_pct(x: float) -> float:
    # 输出给 AI/报表用：保留为 ratio（0~1），但避免 nan
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def diagnose_campaign_trends(camp: pd.DataFrame, cfg: StageConfig, recent_days: int = 7) -> List[Dict[str, object]]:
    """
    活动趋势诊断：用“最近窗口 vs 前窗口”的对比，判断结构性变化（不是单点规则）。

    输出字段（每条一条活动告警/机会）：
    - type: "worsening_scale" / "improving_scale" / "spend_spike_no_sales"
    - severity: 0~100（用于排序）
    - suggestion: 一句话可执行建议（后续 AI 可扩写）
    """
    if camp is None or camp.empty or CAN.date not in camp.columns or CAN.campaign not in camp.columns:
        return []

    try:
        dates = [d for d in camp[CAN.date].dropna().unique().tolist() if isinstance(d, dt.date)]
    except Exception:
        dates = []
    prev_dates, recent_dates = _pick_period_dates(dates, recent_days=recent_days)
    if not prev_dates or not recent_dates:
        return []

    prev = camp[camp[CAN.date].isin(prev_dates)].copy()
    recent = camp[camp[CAN.date].isin(recent_dates)].copy()
    if prev.empty or recent.empty:
        return []

    gcols = [CAN.ad_type, CAN.campaign]
    prev_s = summarize(prev, gcols).rename(
        columns={
            "impressions": "impr_prev",
            "clicks": "clicks_prev",
            "spend": "spend_prev",
            "sales": "sales_prev",
            "orders": "orders_prev",
            "acos": "acos_prev",
            "cvr": "cvr_prev",
            "ctr": "ctr_prev",
        }
    )
    rec_s = summarize(recent, gcols).rename(
        columns={
            "impressions": "impr_rec",
            "clicks": "clicks_rec",
            "spend": "spend_rec",
            "sales": "sales_rec",
            "orders": "orders_rec",
            "acos": "acos_rec",
            "cvr": "cvr_rec",
            "ctr": "ctr_rec",
        }
    )

    merged = prev_s.merge(rec_s, on=gcols, how="outer").fillna(0.0)
    if merged.empty:
        return []

    out: List[Dict[str, object]] = []

    for _, r in merged.iterrows():
        spend_prev = float(r.get("spend_prev", 0.0) or 0.0)
        spend_rec = float(r.get("spend_rec", 0.0) or 0.0)
        sales_prev = float(r.get("sales_prev", 0.0) or 0.0)
        sales_rec = float(r.get("sales_rec", 0.0) or 0.0)
        acos_prev = float(r.get("acos_prev", 0.0) or 0.0)
        acos_rec = float(r.get("acos_rec", 0.0) or 0.0)
        orders_prev = float(r.get("orders_prev", 0.0) or 0.0)
        orders_rec = float(r.get("orders_rec", 0.0) or 0.0)

        spend_delta = spend_rec - spend_prev
        spend_delta_pct = safe_div(spend_delta, spend_prev) if spend_prev > 0 else (1.0 if spend_rec > 0 else 0.0)

        # 1) 花费显著上升但销售/订单不跟：典型“扩量扩错了”
        if spend_rec >= cfg.waste_spend * 2 and spend_delta_pct >= 0.35 and (sales_rec <= sales_prev * 1.05) and (orders_rec <= orders_prev * 1.05):
            severity = min(100.0, max(0.0, (spend_rec) / (cfg.waste_spend * 2) * 30 + spend_delta_pct * 70))
            out.append(
                {
                    "type": "spend_spike_no_sales",
                    "severity": round(severity, 2),
                    "ad_type": str(r.get(CAN.ad_type, "")),
                    "campaign": str(r.get(CAN.campaign, "")),
                    "spend_prev": round(spend_prev, 2),
                    "spend_recent": round(spend_rec, 2),
                    "sales_prev": round(sales_prev, 2),
                    "sales_recent": round(sales_rec, 2),
                    "acos_prev": _fmt_pct(acos_prev),
                    "acos_recent": _fmt_pct(acos_rec),
                    "suggestion": "最近窗口花费明显上升但销售/订单不跟，优先检查新增投放/加价/广告位倾斜，先止血再定位原因。",
                }
            )

        # 2) 扩量导致 ACoS 明显变差：扩大但效率掉了
        if spend_rec >= cfg.waste_spend * 2 and spend_delta_pct >= 0.25 and acos_prev > 0 and acos_rec > acos_prev * 1.25 and acos_rec > cfg.target_acos * 1.1:
            severity = min(100.0, max(0.0, (acos_rec / max(cfg.target_acos, 1e-6) - 1) * 60 + spend_delta_pct * 40))
            out.append(
                {
                    "type": "worsening_scale",
                    "severity": round(severity, 2),
                    "ad_type": str(r.get(CAN.ad_type, "")),
                    "campaign": str(r.get(CAN.campaign, "")),
                    "spend_prev": round(spend_prev, 2),
                    "spend_recent": round(spend_rec, 2),
                    "acos_prev": _fmt_pct(acos_prev),
                    "acos_recent": _fmt_pct(acos_rec),
                    "suggestion": "扩量后 ACoS 明显变差：优先回查最近窗口的新增词/投放/广告位，先小幅回撤出价或限额，再用搜索词/投放层定位浪费来源。",
                }
            )

        # 3) 扩量且效率变好：可以作为“预算迁移/放量”优先级
        if spend_rec >= cfg.waste_spend and spend_delta_pct >= 0.20 and acos_rec > 0 and acos_rec < cfg.target_acos * 0.85 and (orders_rec >= max(3.0, orders_prev)):
            severity = min(100.0, max(0.0, (safe_div(orders_rec, max(orders_prev, 1.0)) - 1) * 50 + (cfg.target_acos / max(acos_rec, 1e-6) - 1) * 50))
            out.append(
                {
                    "type": "improving_scale",
                    "severity": round(severity, 2),
                    "ad_type": str(r.get(CAN.ad_type, "")),
                    "campaign": str(r.get(CAN.campaign, "")),
                    "spend_prev": round(spend_prev, 2),
                    "spend_recent": round(spend_rec, 2),
                    "acos_prev": _fmt_pct(acos_prev),
                    "acos_recent": _fmt_pct(acos_rec),
                    "suggestion": "扩量且效率优于目标：可作为预算迁移优先级候选（确认库存/转化稳定后再放量）。",
                }
            )

    # 同一活动可能同时命中多个类型：只保留“最值得关注”的一条（避免报表刷屏）
    type_weight = {"spend_spike_no_sales": 3, "worsening_scale": 2, "improving_scale": 1}
    best: Dict[Tuple[str, str], Dict[str, object]] = {}
    for item in out:
        key = (str(item.get("ad_type", "")), str(item.get("campaign", "")))
        cur = best.get(key)
        if cur is None:
            best[key] = item
            continue
        cur_score = float(cur.get("severity", 0.0) or 0.0) + type_weight.get(str(cur.get("type", "")), 0) * 5
        new_score = float(item.get("severity", 0.0) or 0.0) + type_weight.get(str(item.get("type", "")), 0) * 5
        if new_score > cur_score:
            best[key] = item

    final = list(best.values())
    final.sort(key=lambda x: float(x.get("severity", 0.0) or 0.0), reverse=True)
    return final[:50]


def diagnose_asin_root_causes(
    product_analysis_shop: pd.DataFrame,
    product_listing_shop: pd.DataFrame,
    cfg: StageConfig,
) -> List[Dict[str, object]]:
    """
    ASIN 根因诊断：结合“经营结果 + 广告结果 + 库存/评分/退款”，识别广告调参之外的关键瓶颈。

    注意：这里不做“学术指标”，只输出运营能用的分类与证据。
    """
    if product_analysis_shop is None or product_analysis_shop.empty:
        return []
    if "ASIN" not in product_analysis_shop.columns:
        return []

    pa = product_analysis_shop.copy()
    # 我们只用到这些字段，缺哪个就跳过（不崩溃）
    cols = [
        "ASIN",
        CAN.date,
        "销量",
        "订单量",
        "销售额",
        "Sessions",
        "转化率",
        "退款率",
        "星级评分",
        "评分数",
        "Review中差评数",
        "广告花费",
        "广告销售额",
        "广告订单量",
        "广告点击量",
        "FBA可售",
    ]
    cols = [c for c in cols if c in pa.columns]
    pa = pa[cols].copy()

    # baseline：店铺整体转化（用订单量/Sessions）
    base_sessions = float(pa["Sessions"].sum()) if "Sessions" in pa.columns else 0.0
    base_orders = float(pa["订单量"].sum()) if "订单量" in pa.columns else 0.0
    base_cvr = safe_div(base_orders, base_sessions) if base_sessions > 0 else 0.0

    # ASIN 汇总
    agg = pa.groupby("ASIN", dropna=False, as_index=False).agg(
        sessions=("Sessions", "sum") if "Sessions" in pa.columns else ("ASIN", "size"),
        orders=("订单量", "sum") if "订单量" in pa.columns else ("ASIN", "size"),
        sales=("销售额", "sum") if "销售额" in pa.columns else ("ASIN", "size"),
        refund_rate=("退款率", "mean") if "退款率" in pa.columns else ("ASIN", "size"),
        rating=("星级评分", "mean") if "星级评分" in pa.columns else ("ASIN", "size"),
        rating_cnt=("评分数", "max") if "评分数" in pa.columns else ("ASIN", "size"),
        bad_reviews=("Review中差评数", "max") if "Review中差评数" in pa.columns else ("ASIN", "size"),
        ad_spend=("广告花费", "sum") if "广告花费" in pa.columns else ("ASIN", "size"),
        ad_sales=("广告销售额", "sum") if "广告销售额" in pa.columns else ("ASIN", "size"),
        ad_orders=("广告订单量", "sum") if "广告订单量" in pa.columns else ("ASIN", "size"),
        ad_clicks=("广告点击量", "sum") if "广告点击量" in pa.columns else ("ASIN", "size"),
        fba_avail=("FBA可售", "min") if "FBA可售" in pa.columns else ("ASIN", "size"),
    )
    agg["cvr"] = agg.apply(lambda r: safe_div(r["orders"], r["sessions"]), axis=1)
    agg["ad_acos"] = agg.apply(lambda r: safe_div(r["ad_spend"], r["ad_sales"]), axis=1)
    agg["ad_cvr"] = agg.apply(lambda r: safe_div(r["ad_orders"], r["ad_clicks"]), axis=1)

    # 结合 productListing 的“可售/品名/分类”（如果存在）
    inv = pd.DataFrame()
    if product_listing_shop is not None and not product_listing_shop.empty and "ASIN" in product_listing_shop.columns:
        inv_cols = ["ASIN"]
        for c in ("可售", "品名", "商品分类"):
            if c in product_listing_shop.columns:
                inv_cols.append(c)
        inv = product_listing_shop[inv_cols].copy()
        inv["ASIN_norm"] = inv["ASIN"].astype(str).str.upper()
        inv = inv.drop_duplicates("ASIN_norm")

    agg["ASIN_norm"] = agg["ASIN"].astype(str).str.upper()
    if not inv.empty:
        agg = agg.merge(inv.drop(columns=["ASIN"]).rename(columns={"ASIN_norm": "ASIN_norm"}), on="ASIN_norm", how="left")

    # 辅助：把“转化率/退款率”统一成 ratio（赛狐常见是 12.3 表示 12.3%）
    def pct_to_ratio(v: float) -> float:
        try:
            v = float(v or 0.0)
            if v <= 0:
                return 0.0
            return v / 100.0 if v > 1.0 else v
        except Exception:
            return 0.0

    if "refund_rate" in agg.columns:
        agg["refund_rate"] = agg["refund_rate"].apply(pct_to_ratio)
    if "cvr" in agg.columns:
        agg["cvr"] = agg["cvr"].apply(lambda x: float(x or 0.0))

    # 用分位数做“高流量/高花费”的定义（更贴近每个店铺自己的结构，属于私有定制）
    sessions_p75 = float(agg["sessions"].quantile(0.75)) if "sessions" in agg.columns and len(agg) >= 10 else 0.0
    spend_p75 = float(agg["ad_spend"].quantile(0.75)) if "ad_spend" in agg.columns and len(agg) >= 10 else 0.0

    out: List[Dict[str, object]] = []
    for _, r in agg.iterrows():
        asin = str(r.get("ASIN", ""))
        if not asin or asin.lower() == "nan":
            continue

        sessions = float(r.get("sessions", 0.0) or 0.0)
        cvr = float(r.get("cvr", 0.0) or 0.0)
        ad_spend = float(r.get("ad_spend", 0.0) or 0.0)
        ad_sales = float(r.get("ad_sales", 0.0) or 0.0)
        ad_orders = float(r.get("ad_orders", 0.0) or 0.0)
        ad_acos = float(r.get("ad_acos", 0.0) or 0.0)
        refund_rate = float(r.get("refund_rate", 0.0) or 0.0)
        rating = float(r.get("rating", 0.0) or 0.0)

        # 库存：优先用“可售”，否则用 FBA可售
        inv_val = None
        if "可售" in r.index:
            try:
                inv_val = float(r.get("可售") or 0.0)
            except Exception:
                inv_val = 0.0
        elif "fba_avail" in r.index:
            inv_val = float(r.get("fba_avail", 0.0) or 0.0)

        # 分类逻辑（多维证据）
        tags: List[str] = []
        if inv_val is not None and inv_val <= 20 and ad_spend > 0:
            tags.append("库存风险")
        if sessions >= sessions_p75 and base_cvr > 0 and cvr < base_cvr * 0.7:
            tags.append("转化偏低")
        # 口径统一：广告订单=0 且 广告销售额=0 才计为“无单浪费”
        if ad_spend >= max(cfg.waste_spend, spend_p75) and ad_orders <= 0 and ad_sales <= 0:
            tags.append("广告花费无单")
        if ad_spend >= cfg.waste_spend and ad_acos > cfg.target_acos * 1.2 and ad_orders >= 1:
            tags.append("广告投产偏差")
        if refund_rate >= 0.08:
            tags.append("退款偏高")
        if 0 < rating < 4.2:
            tags.append("评分偏低")

        if not tags:
            continue

        # severity：按“影响面（流量/花费）+ 距离目标”粗排
        sev = 0.0
        sev += min(50.0, safe_div(ad_spend, max(cfg.waste_spend, 1e-6)) * 10)
        if base_cvr > 0:
            sev += min(30.0, max(0.0, (base_cvr - cvr) / base_cvr) * 30)
        if ad_acos > 0:
            sev += min(20.0, max(0.0, (ad_acos - cfg.target_acos) / max(cfg.target_acos, 1e-6)) * 20)

        out.append(
            {
                "asin": asin,
                "tags": ",".join(tags),
                "severity": round(sev, 2),
                "sessions": int(sessions),
                "cvr": _fmt_pct(cvr),
                "ad_spend": round(ad_spend, 2),
                "ad_orders": int(ad_orders),
                "ad_acos": _fmt_pct(ad_acos),
                "refund_rate": _fmt_pct(refund_rate),
                "rating": round(rating, 2),
                "inventory": None if inv_val is None else int(inv_val),
                "name": str(r.get("品名", "")) if "品名" in r.index else "",
                "category": str(r.get("商品分类", "")) if "商品分类" in r.index else "",
                "suggestion": "优先按标签排查：库存→控量/停投；转化偏低/评分退款→优化Listing；广告花费无单/投产偏差→用搜索词/投放层定位浪费并收口。",
            }
        )

    out.sort(key=lambda x: float(x.get("severity", 0.0) or 0.0), reverse=True)
    return out[:80]


def infer_asin_stage_by_profit(
    product_analysis_shop: pd.DataFrame,
    product_listing_shop: pd.DataFrame,
    cfg: StageConfig,
) -> List[Dict[str, object]]:
    """
    用“毛利润”做 ASIN 阶段判断（店铺拆分、ASIN 级别）。

    你给的阶段逻辑（落地版）：
    - 前期（launch）：以流量/数据积累为主，可接受更高广告占比，但要有止血线（不能一直亏）
    - 中期（growth）：加大推广，关注“扩量是否有效”（花费↑同时订单/销售↑，且不把利润打穿）
    - 后期（profit）：缩减推广，把广告控制在毛利承受范围内，优先保利润与稳定性

    输出的是“阶段标签 + 证据 + 基于毛利的广告预算承受度”，供 AI 写建议。
    """
    if product_analysis_shop is None or product_analysis_shop.empty:
        return []
    if "ASIN" not in product_analysis_shop.columns:
        return []
    if "销售额" not in product_analysis_shop.columns or "毛利润" not in product_analysis_shop.columns:
        return []

    pa = product_analysis_shop.copy()

    # 只保留需要列（缺哪个就跳过，但至少要有这些）
    cols = [
        "ASIN",
        CAN.date,
        "销售额",
        "毛利润",
        "毛利率",
        "订单量",
        "Sessions",
        "广告花费",
        "广告销售额",
        "广告订单量",
        "广告点击量",
        "退款率",
        "星级评分",
        "评分数",
        "Review中差评数",
        "FBA可售",
    ]
    cols = [c for c in cols if c in pa.columns]
    pa = pa[cols].copy()

    # ASIN 汇总（按当前窗口）
    agg = pa.groupby("ASIN", dropna=False, as_index=False).agg(
        active_days=(CAN.date, "nunique") if CAN.date in pa.columns else ("ASIN", "size"),
        sales=("销售额", "sum"),
        gross_profit=("毛利润", "sum"),
        orders=("订单量", "sum") if "订单量" in pa.columns else ("ASIN", "size"),
        sessions=("Sessions", "sum") if "Sessions" in pa.columns else ("ASIN", "size"),
        ad_spend=("广告花费", "sum") if "广告花费" in pa.columns else ("ASIN", "size"),
        ad_sales=("广告销售额", "sum") if "广告销售额" in pa.columns else ("ASIN", "size"),
        ad_orders=("广告订单量", "sum") if "广告订单量" in pa.columns else ("ASIN", "size"),
        ad_clicks=("广告点击量", "sum") if "广告点击量" in pa.columns else ("ASIN", "size"),
        refund_rate=("退款率", "mean") if "退款率" in pa.columns else ("ASIN", "size"),
        rating=("星级评分", "mean") if "星级评分" in pa.columns else ("ASIN", "size"),
        rating_cnt=("评分数", "max") if "评分数" in pa.columns else ("ASIN", "size"),
        bad_reviews=("Review中差评数", "max") if "Review中差评数" in pa.columns else ("ASIN", "size"),
        fba_avail=("FBA可售", "min") if "FBA可售" in pa.columns else ("ASIN", "size"),
    )

    # --- 利润口径推断（解决“毛利润是否已包含广告费”导致全员不可放量的问题） ---
    # 赛狐/ERP 里“毛利润”可能是：
    # - 模式A：不含广告费（则“广告后利润”=毛利润-广告花费）
    # - 模式B：已含广告费（则“广告后利润”=毛利润；“广告前利润”≈毛利润+广告花费）
    #
    # 我们用“店铺整体分布”做一个保守推断：
    # - 如果绝大多数 ASIN 的毛利润<=0，但 (毛利润+广告花费)>0 的数量明显更多，则倾向模式B
    agg["profit_addback_ads"] = agg.apply(lambda r: float(r.get("gross_profit", 0.0) or 0.0) + float(r.get("ad_spend", 0.0) or 0.0), axis=1)
    profitable_raw = int((agg["gross_profit"] > 0).sum())
    profitable_addback = int((agg["profit_addback_ads"] > 0).sum())
    # 默认模式：A
    profit_mode = "profit_excludes_ads"
    if len(agg) >= 8 and profitable_raw <= max(1, int(profitable_addback * 0.3)) and profitable_addback >= 3:
        profit_mode = "profit_includes_ads"

    # 基础派生（全部用确定性算数）
    # gross_margin：始终用“报表给的毛利润/销售额”作为展示口径（便于你对账），但承受度用 profit_before_ads
    agg["gross_margin"] = agg.apply(lambda r: safe_div(float(r.get("gross_profit", 0.0) or 0.0), float(r.get("sales", 0.0) or 0.0)), axis=1)
    if profit_mode == "profit_includes_ads":
        # 广告后利润≈毛利润；广告前利润≈毛利润+广告花费
        agg["profit_before_ads"] = agg["profit_addback_ads"]
        agg["profit_after_ads"] = agg["gross_profit"]
    else:
        agg["profit_before_ads"] = agg["gross_profit"]
        agg["profit_after_ads"] = agg.apply(lambda r: float(r.get("gross_profit", 0.0) or 0.0) - float(r.get("ad_spend", 0.0) or 0.0), axis=1)

    agg["tacos"] = agg.apply(lambda r: safe_div(float(r.get("ad_spend", 0.0) or 0.0), float(r.get("sales", 0.0) or 0.0)), axis=1)
    agg["ad_share"] = agg.apply(lambda r: safe_div(float(r.get("ad_sales", 0.0) or 0.0), float(r.get("sales", 0.0) or 0.0)), axis=1)
    agg["shop_cvr"] = agg.apply(lambda r: safe_div(float(r.get("orders", 0.0) or 0.0), float(r.get("sessions", 0.0) or 0.0)), axis=1)
    agg["ad_cvr"] = agg.apply(lambda r: safe_div(float(r.get("ad_orders", 0.0) or 0.0), float(r.get("ad_clicks", 0.0) or 0.0)), axis=1)

    # 把退款率/毛利率统一成 ratio（赛狐通常已经是 0~1）
    def pct_to_ratio(v: float) -> float:
        try:
            v = float(v or 0.0)
            if v <= 0:
                return 0.0
            return v / 100.0 if v > 1.0 else v
        except Exception:
            return 0.0

    agg["refund_rate"] = agg["refund_rate"].apply(pct_to_ratio) if "refund_rate" in agg.columns else 0.0

    # 引入 productListing 的可售/品名/分类（可选）
    inv = pd.DataFrame()
    if product_listing_shop is not None and not product_listing_shop.empty and "ASIN" in product_listing_shop.columns:
        inv_cols = ["ASIN"]
        for c in ("可售", "品名", "商品分类"):
            if c in product_listing_shop.columns:
                inv_cols.append(c)
        inv = product_listing_shop[inv_cols].copy()
        inv["ASIN_norm"] = inv["ASIN"].astype(str).str.upper()
        inv = inv.drop_duplicates("ASIN_norm")
    agg["ASIN_norm"] = agg["ASIN"].astype(str).str.upper()
    if not inv.empty:
        agg = agg.merge(inv.drop(columns=["ASIN"]).rename(columns={"ASIN_norm": "ASIN_norm"}), on="ASIN_norm", how="left")

    # 用“店铺私有分布”定义门槛（深度定制点：不写死阈值）
    sales_p40 = float(agg["sales"].quantile(0.40)) if len(agg) >= 10 else 0.0
    sales_p75 = float(agg["sales"].quantile(0.75)) if len(agg) >= 10 else 0.0
    orders_p75 = float(agg["orders"].quantile(0.75)) if len(agg) >= 10 else 0.0
    sessions_p40 = float(agg["sessions"].quantile(0.40)) if len(agg) >= 10 else 0.0

    # 阶段的“毛利承受度”（把广告从“目标ACoS”转成“最大可花费”）
    stage_factor = {
        "launch": {"max_ad_spend_gross_profit_ratio": 1.00, "target_tacos_ratio_to_margin": 0.95},
        "growth": {"max_ad_spend_gross_profit_ratio": 0.80, "target_tacos_ratio_to_margin": 0.80},
        # 你确认：后期最多吃掉 50% 毛利润
        "profit": {"max_ad_spend_gross_profit_ratio": 0.50, "target_tacos_ratio_to_margin": 0.50},
    }

    out: List[Dict[str, object]] = []
    for _, r in agg.iterrows():
        asin = str(r.get("ASIN", "")).strip()
        if not asin or asin.lower() == "nan":
            continue

        sales = float(r.get("sales", 0.0) or 0.0)
        orders = float(r.get("orders", 0.0) or 0.0)
        sessions = float(r.get("sessions", 0.0) or 0.0)
        ad_spend = float(r.get("ad_spend", 0.0) or 0.0)
        gross_profit = float(r.get("gross_profit", 0.0) or 0.0)
        gross_margin = float(r.get("gross_margin", 0.0) or 0.0)
        profit_before_ads = float(r.get("profit_before_ads", 0.0) or 0.0)
        profit_after_ads = float(r.get("profit_after_ads", 0.0) or 0.0)
        ad_share = float(r.get("ad_share", 0.0) or 0.0)
        tacos = float(r.get("tacos", 0.0) or 0.0)
        active_days = int(r.get("active_days", 0) or 0)

        reasons: List[str] = []

        # 库存：优先用“可售”，否则用 FBA可售（用于决定是否允许放量）
        inv_val = None
        if "可售" in r.index:
            try:
                inv_val = int(float(r.get("可售") or 0.0))
            except Exception:
                inv_val = None
        elif "fba_avail" in r.index:
            try:
                inv_val = int(float(r.get("fba_avail") or 0.0))
            except Exception:
                inv_val = None

        # 1) Launch：样本小/销售规模小（但不等于无脑亏；仍要用毛利做止血）
        is_launch = (
            (active_days > 0 and active_days <= 7)
            or (sales_p40 > 0 and sales < sales_p40 and sessions_p40 > 0 and sessions < sessions_p40)
            or (orders < 10 and sales < max(200.0, sales_p40))
        )
        if is_launch:
            stage = "launch"
            reasons.append("样本/规模偏小，优先拉流量与积累数据")
        else:
            # 2) Profit：成熟且更多依赖自然（广告占比低）+ 广告不应打穿毛利
            is_mature = (sales_p75 > 0 and sales >= sales_p75) or (orders_p75 > 0 and orders >= orders_p75)
            is_organic_led = ad_share <= 0.5  # 广告销售占比低 -> 更偏自然
            is_profitable_after_ads = profit_after_ads > 0
            if is_mature and is_organic_led and is_profitable_after_ads:
                stage = "profit"
                reasons.append("相对成熟且自然占比更高，后期以控投放保利润为主")
            else:
                stage = "growth"
                reasons.append("已有一定规模/样本，中期以放量为主（关注扩量是否有效）")

        # 基于毛利的预算承受度（关键定制点）：用 profit_before_ads
        factors = stage_factor[stage]
        max_ad_spend = max(0.0, profit_before_ads * float(factors["max_ad_spend_gross_profit_ratio"]))
        target_tacos = max(0.0, gross_margin * float(factors["target_tacos_ratio_to_margin"]))

        direction = "hold"
        if profit_before_ads <= 0:
            direction = "reduce"
            reasons.append("利润承受度≤0：不建议继续加大投放（先修复成本/定价/Listing/退货/广告结构）")
        else:
            if ad_spend > max_ad_spend and ad_spend >= cfg.waste_spend:
                direction = "reduce"
                reasons.append("广告花费已接近/超过毛利承受度，建议先收口/降价/限额")
            else:
                # 放量判定：即使是 profit 阶段，也允许“在承受度内的小幅迁移加码”
                # 这样才能把预算从亏损结构迁移到真正赚钱的结构（仍然符合后期“控量提效”）
                scale_ratio = 0.7 if stage in {"launch", "growth"} else 0.5
                if profit_after_ads > 0 and ad_spend < max_ad_spend * scale_ratio and ad_spend >= cfg.waste_spend:
                    direction = "scale"
                    reasons.append("利润承受度内且广告后利润为正，可作为预算迁移/加码候选（先确认库存）")

        # 库存风险：优先控量（无论哪个阶段）
        if inv_val is not None and inv_val <= 20 and ad_spend > 0:
            direction = "reduce"
            reasons.append("库存偏低：先控量/停投，避免断货拖累转化")

        out.append(
            {
                "asin": asin,
                "stage": stage,
                "direction": direction,
                "reasons": "；".join(reasons),
                "sales": round(sales, 2),
                "gross_profit": round(gross_profit, 2),
                "gross_margin": _fmt_pct(gross_margin),
                "ad_spend": round(ad_spend, 2),
                "tacos": _fmt_pct(tacos),
                "ad_share": _fmt_pct(ad_share),
                "profit_after_ads": round(profit_after_ads, 2),
                "profit_mode": profit_mode,
                "profit_before_ads": round(profit_before_ads, 2),
                "max_ad_spend_by_profit": round(max_ad_spend, 2),
                "target_tacos_by_margin": _fmt_pct(target_tacos),
                "orders": int(orders),
                "sessions": int(sessions),
                "active_days": active_days,
                "inventory": inv_val,
                "refund_rate": _fmt_pct(float(r.get("refund_rate", 0.0) or 0.0)),
                "rating": round(float(r.get("rating", 0.0) or 0.0), 2),
                "name": str(r.get("品名", "")) if "品名" in r.index else "",
                "category": str(r.get("商品分类", "")) if "商品分类" in r.index else "",
            }
        )

    # 展示优先级：先看广告花费大的（更贴近“运营要调广告”）
    out.sort(key=lambda x: float(x.get("ad_spend", 0.0) or 0.0), reverse=True)
    return out[:120]


def diagnose_campaign_budget_map_from_asin(
    camp: pd.DataFrame,
    advertised_product: pd.DataFrame,
    asin_stages: List[Dict[str, object]],
    cfg: StageConfig,
    temporal: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """
    预算迁移图谱（Campaign 级别）：
    - 利用广告产品报告把 campaign 的花费拆到 ASIN
    - 再用 asin_stages（毛利承受度/方向）来判断该 campaign 更适合“控量/放量/观察”

    这是“深度定制”的关键点：预算建议不再只看活动 ACoS，而是看它主要在推哪些 ASIN，
    而这些 ASIN 的毛利承受度/库存/广告后利润是否允许继续加码。
    """
    if camp is None or camp.empty or advertised_product is None or advertised_product.empty:
        return []
    if not asin_stages:
        return []
    if CAN.campaign not in advertised_product.columns or CAN.asin not in advertised_product.columns:
        return []
    if CAN.spend not in advertised_product.columns:
        return []

    # temporal 信号（7/14/30窗口）：用于增强“止血/放量”的置信度（不是硬阈值）
    temporal_map: Dict[Tuple[str, str], Dict[str, object]] = {}
    try:
        if temporal and isinstance(temporal, dict):
            rows = temporal.get("campaign_windows")
            if isinstance(rows, list) and rows:
                df = pd.DataFrame(rows)
                # 只取 window_days=7 的信号作为“最近趋势”锚点
                if "window_days" in df.columns:
                    df = df[df["window_days"] == 7].copy()
                if not df.empty and CAN.ad_type in df.columns and CAN.campaign in df.columns:
                    df = df.sort_values("score", ascending=False)
                    for _, r in df.iterrows():
                        key = (str(r.get(CAN.ad_type, "")), str(r.get(CAN.campaign, "")))
                        if key not in temporal_map:
                            temporal_map[key] = {
                                "signal": str(r.get("signal", "")),
                                "marginal_acos": float(r.get("marginal_acos", 0.0) or 0.0),
                                "score": float(r.get("score", 0.0) or 0.0),
                                "delta_sales": float(r.get("delta_sales", 0.0) or 0.0),
                                "delta_spend": float(r.get("delta_spend", 0.0) or 0.0),
                            }
    except Exception:
        temporal_map = {}

    # ASIN 阶段表（来自产品分析+毛利推导）
    asin_df = pd.DataFrame(asin_stages).copy()
    if asin_df.empty or "asin" not in asin_df.columns or "direction" not in asin_df.columns:
        return []
    asin_df["asin_norm"] = asin_df["asin"].astype(str).str.upper().str.strip()
    asin_df = asin_df.drop_duplicates("asin_norm")

    ap = advertised_product.copy()
    ap = ap[ap[CAN.campaign].notna() & ap[CAN.asin].notna()].copy()
    ap[CAN.campaign] = ap[CAN.campaign].astype(str).str.strip()
    ap[CAN.asin] = ap[CAN.asin].astype(str).str.upper().str.strip()

    # campaign x asin：花费拆分（用广告产品报告的 spend 作为归因权重）
    mix = (
        ap.groupby([CAN.ad_type, CAN.campaign, CAN.asin], dropna=False, as_index=False)
        .agg(spend=(CAN.spend, "sum"), sales=(CAN.sales, "sum"), orders=(CAN.orders, "sum"))
        .copy()
    )
    mix = mix.rename(columns={CAN.asin: "asin_norm"})
    mix = mix.merge(asin_df, on="asin_norm", how="left")
    # 未匹配到阶段的 asin：标为 unknown（不胡编）
    mix["direction"] = mix["direction"].fillna("unknown")
    mix["stage"] = mix["stage"].fillna("unknown")

    # campaign 总览（来自 campaign 报告，更完整）
    camp_sum = summarize(camp, [CAN.ad_type, CAN.campaign]).copy()
    camp_sum = camp_sum.rename(columns={"spend": "camp_spend", "sales": "camp_sales", "orders": "camp_orders", "acos": "camp_acos"})

    # 计算每个 campaign 的“方向占比”
    out: List[Dict[str, object]] = []
    for (ad_type, campaign), g in mix.groupby([CAN.ad_type, CAN.campaign], dropna=False):
        total_spend = float(g["spend"].sum()) if not g.empty else 0.0
        if total_spend <= 0:
            continue

        def spend_share(direction: str) -> float:
            return float(g[g["direction"] == direction]["spend"].sum()) / total_spend

        reduce_share = spend_share("reduce")
        scale_share = spend_share("scale")
        unknown_share = spend_share("unknown")

        # 方向判断：用“花费结构”而不是单条规则
        action = "hold"
        # 若主要花费都在 reduce 的 ASIN 上，且金额达到阈值 -> 优先控量
        if reduce_share >= 0.55 and total_spend >= cfg.waste_spend * 3:
            action = "reduce"
        # 若主要花费在 scale 的 ASIN 上 -> 可放量（仍需库存确认）
        # 放量判定稍微放宽：只要 scale 占比明显高于 reduce，就允许进入“加码候选”
        elif scale_share >= 0.40 and scale_share > reduce_share and total_spend >= cfg.waste_spend:
            action = "scale"

        # 建议幅度（可回滚的小步）
        if action == "reduce":
            pct = -min(30, int(10 + 20 * reduce_share))
        elif action == "scale":
            pct = min(25, int(10 + 15 * scale_share))
        else:
            pct = 0

        # Top 贡献 ASIN（证据）
        top_reduce = (
            g[g["direction"] == "reduce"].sort_values("spend", ascending=False).head(5)[["asin_norm", "spend", "stage", "profit_after_ads", "max_ad_spend_by_profit", "inventory"]]
        )
        top_scale = (
            g[g["direction"] == "scale"].sort_values("spend", ascending=False).head(5)[["asin_norm", "spend", "stage", "profit_after_ads", "max_ad_spend_by_profit", "inventory"]]
        )
        top_reduce_hint = ""
        if not top_reduce.empty:
            tr = top_reduce.iloc[0].to_dict()
            top_reduce_hint = f"{tr.get('asin_norm','')} (${float(tr.get('spend',0.0) or 0.0):.2f})"
        top_scale_hint = ""
        if not top_scale.empty:
            ts = top_scale.iloc[0].to_dict()
            top_scale_hint = f"{ts.get('asin_norm','')} (${float(ts.get('spend',0.0) or 0.0):.2f})"

        # 合并 campaign 指标
        row = camp_sum[(camp_sum[CAN.ad_type] == ad_type) & (camp_sum[CAN.campaign] == campaign)]
        camp_spend = float(row["camp_spend"].iloc[0]) if not row.empty else total_spend
        camp_sales = float(row["camp_sales"].iloc[0]) if not row.empty else float(g["sales"].sum())
        camp_orders = float(row["camp_orders"].iloc[0]) if not row.empty else float(g["orders"].sum())
        camp_acos = float(row["camp_acos"].iloc[0]) if not row.empty else safe_div(camp_spend, camp_sales)

        severity = 0.0
        if action == "reduce":
            severity = min(100.0, (reduce_share * 70 + min(30.0, safe_div(camp_spend, cfg.waste_spend * 3) * 30)))
        elif action == "scale":
            severity = min(100.0, (scale_share * 60 + min(40.0, safe_div(camp_orders, 10.0) * 20)))
        else:
            severity = min(60.0, unknown_share * 60)

        # 结合 temporal：最近7天如果出现 spend_spike_no_sales/decaying，加重 reduce；若 accelerating，增强 scale
        tkey = (str(ad_type), str(campaign))
        tinfo = temporal_map.get(tkey)
        if tinfo:
            tsig = str(tinfo.get("signal", ""))
            if action == "reduce" and tsig in {"spend_spike_no_sales", "decaying"}:
                severity = min(100.0, severity + 15.0)
            if action == "scale" and tsig in {"accelerating", "efficiency_gain"}:
                severity = min(100.0, severity + 10.0)

        suggestion = ""
        if action == "reduce":
            suggestion = "该活动主要在推“毛利承受度不足/广告后亏损”的 ASIN，建议先控量/限额，把预算迁移到可放量的结构。"
        elif action == "scale":
            suggestion = "该活动主要在推“可放量(毛利承受度内)”的 ASIN，可作为预算迁移/加码候选（先确认库存）。"
        else:
            suggestion = "该活动 ASIN 结构信息不足或方向不明显，建议先观察并补齐 ASIN↔活动的归因口径。"

        out.append(
            {
                "ad_type": str(ad_type),
                "campaign": str(campaign),
                "action": action,  # reduce/scale/hold
                "suggested_budget_change_pct": int(pct),
                "severity": round(float(severity), 2),
                "camp_spend": round(camp_spend, 2),
                "camp_sales": round(camp_sales, 2),
                "camp_orders": int(camp_orders),
                "camp_acos": float(camp_acos),
                "reduce_spend_share": round(reduce_share, 4),
                "scale_spend_share": round(scale_share, 4),
                "unknown_spend_share": round(unknown_share, 4),
                "top_reduce_asins": top_reduce.to_dict(orient="records") if not top_reduce.empty else [],
                "top_scale_asins": top_scale.to_dict(orient="records") if not top_scale.empty else [],
                "top_reduce_asin_hint": top_reduce_hint,
                "top_scale_asin_hint": top_scale_hint,
                "temporal_signal_7d": str(tinfo.get("signal", "")) if tinfo else "",
                "temporal_marginal_acos_7d": float(tinfo.get("marginal_acos", 0.0)) if tinfo else 0.0,
                "suggestion": suggestion,
            }
        )

    out.sort(key=lambda x: float(x.get("severity", 0.0) or 0.0), reverse=True)
    return out[:80]


def build_budget_transfer_plan(
    campaign_budget_map: List[Dict[str, object]],
    min_transfer_usd: float = 5.0,
    max_transfers: int = 200,
) -> Dict[str, object]:
    """
    把 campaign_budget_map（活动控量/放量方向）合成“净迁移表”：
    - 从哪些活动（reduce）挪出预算
    - 挪到哪些活动（scale）

    重要说明（避免误解）：
    - 赛狐报表里通常拿不到“当前预算值”，所以这里用“本期花费 camp_spend”做 proxy，
      把 suggested_budget_change_pct 转成“建议迁移金额（估算）”。
    - 这是 L0 的执行清单，不生成批量上传文件；执行仍需人工确认。
    """
    reduce_pool: List[Dict[str, object]] = []
    scale_demand: List[Dict[str, object]] = []

    for item in campaign_budget_map or []:
        try:
            action = str(item.get("action", "") or "")
            pct = float(item.get("suggested_budget_change_pct", 0) or 0)
            spend = float(item.get("camp_spend", 0.0) or 0.0)
            sev = float(item.get("severity", 0.0) or 0.0)
            if spend <= 0 or pct == 0:
                continue
            # 估算迁移金额（基于本期花费）
            delta = spend * abs(pct) / 100.0
            if delta < min_transfer_usd:
                continue

            base = {
                "ad_type": str(item.get("ad_type", "") or ""),
                "campaign": str(item.get("campaign", "") or ""),
                "severity": sev,
                "camp_spend": spend,
                "suggested_budget_change_pct": float(pct),
                # 初始建议金额（用于 cuts/adds 表）
                "delta_usd_initial": float(delta),
                # 工作变量：用于匹配迁移时扣减
                "delta_usd": float(delta),
                "top_reduce_asin_hint": str(item.get("top_reduce_asin_hint", "") or ""),
                "top_scale_asin_hint": str(item.get("top_scale_asin_hint", "") or ""),
            }
            if action == "reduce" and pct < 0:
                reduce_pool.append(base)
            elif action == "scale" and pct > 0:
                scale_demand.append(base)
        except Exception:
            continue

    # 排序：优先处理更“确定”的（severity 高）
    reduce_pool.sort(key=lambda x: float(x.get("severity", 0.0) or 0.0), reverse=True)
    scale_demand.sort(key=lambda x: float(x.get("severity", 0.0) or 0.0), reverse=True)

    def _copy_pool(pool: List[Dict[str, object]]) -> List[Dict[str, object]]:
        # 避免在匹配过程中破坏“初始建议金额”，让 cuts/adds 更直观
        return [dict(x) for x in pool]

    def match_transfers(src_list: List[Dict[str, object]], dst_list: List[Dict[str, object]]) -> List[Dict[str, object]]:
        transfers_local: List[Dict[str, object]] = []
        i = 0
        j = 0
        while i < len(src_list) and j < len(dst_list) and len(transfers_local) < max_transfers:
            src = src_list[i]
            dst = dst_list[j]

            src_left = float(src.get("delta_usd", 0.0) or 0.0)
            dst_need = float(dst.get("delta_usd", 0.0) or 0.0)
            if src_left <= 0:
                i += 1
                continue
            if dst_need <= 0:
                j += 1
                continue

            amt = min(src_left, dst_need)
            if amt < min_transfer_usd:
                if src_left <= dst_need:
                    i += 1
                else:
                    j += 1
                continue

            transfers_local.append(
                {
                    "from_ad_type": src.get("ad_type", ""),
                    "from_campaign": src.get("campaign", ""),
                    "from_severity": round(float(src.get("severity", 0.0) or 0.0), 2),
                    "from_spend": round(float(src.get("camp_spend", 0.0) or 0.0), 2),
                    "from_asin_hint": src.get("top_reduce_asin_hint", ""),
                    "to_ad_type": dst.get("ad_type", ""),
                    "to_campaign": dst.get("campaign", ""),
                    "to_severity": round(float(dst.get("severity", 0.0) or 0.0), 2),
                    "to_spend": round(float(dst.get("camp_spend", 0.0) or 0.0), 2),
                    "to_asin_hint": dst.get("top_scale_asin_hint", ""),
                    "amount_usd_estimated": round(float(amt), 2),
                    "note": "金额为估算（基于本期花费×建议百分比），执行时以实际预算/花费节奏校准。",
                }
            )

            src["delta_usd"] = src_left - amt
            dst["delta_usd"] = dst_need - amt
            if src["delta_usd"] <= min_transfer_usd / 2:
                i += 1
            if dst["delta_usd"] <= min_transfer_usd / 2:
                j += 1
        return transfers_local

    # 先同广告类型匹配（更符合运营实际：SP预算尽量挪给SP，SB挪给SB）
    transfers: List[Dict[str, object]] = []
    reduce_work = _copy_pool(reduce_pool)
    scale_work = _copy_pool(scale_demand)
    for ad_type in sorted({str(x.get("ad_type", "") or "") for x in reduce_pool + scale_demand}):
        src_list = [x for x in reduce_work if str(x.get("ad_type", "") or "") == ad_type]
        dst_list = [x for x in scale_work if str(x.get("ad_type", "") or "") == ad_type]
        if not src_list or not dst_list:
            continue
        transfers.extend(match_transfers(src_list, dst_list))
        if len(transfers) >= max_transfers:
            break

    # 再跨类型补齐（如果你希望严格不跨类型，可以把这段关掉）
    if len(transfers) < max_transfers:
        transfers.extend(match_transfers(reduce_work, scale_work))

    reduce_left_total = round(sum(float(x.get("delta_usd", 0.0) or 0.0) for x in reduce_work), 2)
    scale_left_total = round(sum(float(x.get("delta_usd", 0.0) or 0.0) for x in scale_work), 2)

    # 给运营的“控量清单/放量清单”（即使没有可迁移组合也有用）
    cuts: List[Dict[str, object]] = []
    for src in reduce_pool:
        amt = float(src.get("delta_usd_initial", 0.0) or 0.0)
        if amt < min_transfer_usd:
            continue
        cuts.append(
            {
                "ad_type": src.get("ad_type", ""),
                "campaign": src.get("campaign", ""),
                "severity": round(float(src.get("severity", 0.0) or 0.0), 2),
                "camp_spend": round(float(src.get("camp_spend", 0.0) or 0.0), 2),
                "cut_usd_estimated": round(float(amt), 2),
                "asin_hint": src.get("top_reduce_asin_hint", ""),
                "note": "金额为估算（基于本期花费×建议百分比），执行时以实际预算/花费节奏校准。",
            }
        )
    adds: List[Dict[str, object]] = []
    for dst in scale_demand:
        amt = float(dst.get("delta_usd_initial", 0.0) or 0.0)
        if amt < min_transfer_usd:
            continue
        adds.append(
            {
                "ad_type": dst.get("ad_type", ""),
                "campaign": dst.get("campaign", ""),
                "severity": round(float(dst.get("severity", 0.0) or 0.0), 2),
                "camp_spend": round(float(dst.get("camp_spend", 0.0) or 0.0), 2),
                "add_usd_estimated": round(float(amt), 2),
                "asin_hint": dst.get("top_scale_asin_hint", ""),
                "note": "金额为估算（基于本期花费×建议百分比），执行时以实际预算/花费节奏校准。",
            }
        )

    cuts.sort(key=lambda x: float(x.get("severity", 0.0) or 0.0), reverse=True)
    adds.sort(key=lambda x: float(x.get("severity", 0.0) or 0.0), reverse=True)

    # 如果没有可放量池：把 cuts 作为“预算回收/降档”清单（目的地=RESERVE）
    savings: List[Dict[str, object]] = []
    if len(scale_demand) == 0 and cuts:
        for c in cuts[:200]:
            try:
                savings.append(
                    {
                        "from_ad_type": c.get("ad_type", ""),
                        "from_campaign": c.get("campaign", ""),
                        "from_asin_hint": c.get("asin_hint", ""),
                        "to_bucket": "RESERVE",
                        "amount_usd_estimated": float(c.get("cut_usd_estimated", 0.0) or 0.0),
                        "note": "本期无可放量池：建议先回收预算，待利润/库存修复后再分配。",
                    }
                )
            except Exception:
                continue

    return {
        "min_transfer_usd": float(min_transfer_usd),
        "reduce_candidates": len(reduce_pool),
        "scale_candidates": len(scale_demand),
        "transfers": transfers,
        "cuts": cuts[:120],
        "adds": adds[:120],
        "savings": savings,
        "unallocated_reduce_usd_estimated": float(reduce_left_total),
        "unmet_scale_usd_estimated": float(scale_left_total),
    }


def summarize_profit_health(asin_stages: List[Dict[str, object]]) -> Dict[str, object]:
    """
    把 asin_stages 汇总成“利润健康度”诊断，解释为什么可能没有可放量池。
    """
    if not asin_stages:
        return {"asin_count": 0}
    df = pd.DataFrame(asin_stages).copy()
    if df.empty:
        return {"asin_count": 0}

    def cnt(mask: pd.Series) -> int:
        try:
            return int(mask.sum())
        except Exception:
            return 0

    asin_count = int(len(df))
    profit_mode = str(df["profit_mode"].iloc[0]) if "profit_mode" in df.columns and len(df) > 0 else ""

    profit_before_pos = cnt(pd.to_numeric(df.get("profit_before_ads", 0), errors="coerce").fillna(0) > 0)
    profit_after_pos = cnt(pd.to_numeric(df.get("profit_after_ads", 0), errors="coerce").fillna(0) > 0)
    reduce_cnt = cnt(df.get("direction", "") == "reduce")
    scale_cnt = cnt(df.get("direction", "") == "scale")

    # 关键阻塞类型（按常见原因分类）
    blockers = {"承受度≤0": 0, "广告超承受": 0, "库存风险": 0}
    if "profit_before_ads" in df.columns:
        blockers["承受度≤0"] = cnt(pd.to_numeric(df["profit_before_ads"], errors="coerce").fillna(0) <= 0)
    if "ad_spend" in df.columns and "max_ad_spend_by_profit" in df.columns:
        blockers["广告超承受"] = cnt(
            (pd.to_numeric(df["ad_spend"], errors="coerce").fillna(0) > pd.to_numeric(df["max_ad_spend_by_profit"], errors="coerce").fillna(0))
            & (pd.to_numeric(df["ad_spend"], errors="coerce").fillna(0) > 0)
        )
    if "inventory" in df.columns:
        blockers["库存风险"] = cnt(pd.to_numeric(df["inventory"], errors="coerce").fillna(999999) <= 20)

    # 输出 Top 阻塞 ASIN（按广告花费排序，最值得先修）
    top_cols = [c for c in ["asin", "direction", "stage", "ad_spend", "profit_before_ads", "profit_after_ads", "max_ad_spend_by_profit", "inventory", "reasons"] if c in df.columns]
    top_blocked = (
        df.sort_values("ad_spend", ascending=False)[top_cols].head(15).to_dict(orient="records")
        if "ad_spend" in df.columns
        else df[top_cols].head(15).to_dict(orient="records")
    )

    return {
        "asin_count": asin_count,
        "profit_mode_inferred": profit_mode,
        "profit_before_ads_positive_asin": profit_before_pos,
        "profit_after_ads_positive_asin": profit_after_pos,
        "reduce_asin": reduce_cnt,
        "scale_asin": scale_cnt,
        "blockers": blockers,
        "top_blocked_asins": top_blocked,
    }


def build_unlock_scale_plan(
    asin_stages: List[Dict[str, object]],
    min_ad_spend: float = 10.0,
) -> List[Dict[str, object]]:
    """
    “解锁放量池”清单：当 scale 很少/为0 时，告诉运营应该先修哪些 ASIN 才能恢复可放量结构。

    输出策略（按优先级）：
    1) 广告超承受（可通过降预算/降价/收口立刻修复）
    2) 承受度≤0 但广告花费较高（需要修利润：价格/成本/退货/Listing 转化）
    3) 库存风险（先控量）
    """
    if not asin_stages:
        return []
    df = pd.DataFrame(asin_stages).copy()
    if df.empty or "asin" not in df.columns:
        return []

    for c in ("ad_spend", "profit_before_ads", "profit_after_ads", "max_ad_spend_by_profit", "inventory"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df = df[df.get("ad_spend", 0.0) >= float(min_ad_spend)].copy()
    if df.empty:
        return []

    # 类型1：广告超承受
    df1 = df[(df.get("ad_spend", 0.0) > df.get("max_ad_spend_by_profit", 0.0)) & (df.get("max_ad_spend_by_profit", 0.0) > 0)].copy()
    if not df1.empty:
        df1["gap_usd"] = (df1["ad_spend"] - df1["max_ad_spend_by_profit"]).clip(lower=0.0)
        df1["fix"] = df1.apply(lambda r: f"把该ASIN相关活动预算/出价收口到约${r['max_ad_spend_by_profit']:.2f}（估算需减少${r['gap_usd']:.2f}）", axis=1)
        df1["priority"] = "P0"

    # 类型2：承受度≤0
    df2 = df[(df.get("profit_before_ads", 0.0) <= 0)].copy()
    if not df2.empty:
        df2["gap_usd"] = (-df2["profit_before_ads"]).clip(lower=0.0)
        df2["fix"] = df2.apply(lambda r: f"先把利润承受度修到>0：至少提升${r['gap_usd']:.2f}（价格/成本/退货/Listing转化），否则不建议放量", axis=1)
        df2["priority"] = "P1"

    # 类型3：库存风险
    df3 = df[df.get("inventory", 999999) <= 20].copy()
    if not df3.empty:
        df3["gap_usd"] = 0.0
        df3["fix"] = "先控量/停投并补货，避免断货拉低转化"
        df3["priority"] = "P0"

    parts = []
    for part in (df1 if "df1" in locals() else pd.DataFrame(), df3 if "df3" in locals() else pd.DataFrame(), df2 if "df2" in locals() else pd.DataFrame()):
        if part is not None and not part.empty:
            parts.append(part)
    if not parts:
        return []

    merged = pd.concat(parts, ignore_index=True)
    if merged.empty:
        return []

    # 合并同一 ASIN 的多条 fix（避免重复刷屏）
    merged["fix"] = merged["fix"].astype(str)
    merged["priority_rank"] = merged["priority"].map({"P0": 0, "P1": 1, "P2": 2}).fillna(9)
    merged = merged.sort_values(["priority_rank", "ad_spend"], ascending=[True, False])

    grouped = []
    for asin, g in merged.groupby("asin", dropna=False):
        g = g.sort_values(["priority_rank", "ad_spend"], ascending=[True, False])
        top = g.iloc[0].to_dict()
        fixes = [s for s in g["fix"].dropna().astype(str).unique().tolist() if s.strip()]
        top["fix"] = "；".join(fixes[:3])
        # gap_usd 取最大（最困难的那个）
        try:
            top["gap_usd"] = float(pd.to_numeric(g.get("gap_usd", 0), errors="coerce").fillna(0).max())
        except Exception:
            pass
        grouped.append(top)

    df_out = pd.DataFrame(grouped)
    df_out = df_out.sort_values(["priority_rank", "ad_spend"], ascending=[True, False])
    cols = [c for c in ["priority", "asin", "stage", "direction", "ad_spend", "profit_before_ads", "profit_after_ads", "max_ad_spend_by_profit", "inventory", "gap_usd", "fix", "reasons"] if c in df_out.columns]
    return df_out[cols].head(60).to_dict(orient="records")


def build_unlock_tasks(
    asin_stages: List[Dict[str, object]],
    min_sessions_for_cvr: int = 200,
) -> List[Dict[str, object]]:
    """
    把 unlock_scale_plan 进一步拆成“可分工的任务列表”，并给出尽量可量化的缺口估算。

    设计原则：
    - 只输出确定性/可解释的计算（预算缺口、利润缺口、库存风险、转化短板）
    - 不做“拍脑袋”的收益预测；用‘需要修到什么程度’的目标表达
    """
    if not asin_stages:
        return []
    df = pd.DataFrame(asin_stages).copy()
    if df.empty or "asin" not in df.columns:
        return []

    # 数值列清洗
    for c in (
        "ad_spend",
        "profit_before_ads",
        "profit_after_ads",
        "max_ad_spend_by_profit",
        "inventory",
        "refund_rate",
        "rating",
        "orders",
        "sessions",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # 店铺基线：转化率/毛利率（用总体数据，属于“私有口径”）
    total_sessions = float(df["sessions"].sum()) if "sessions" in df.columns else 0.0
    total_orders = float(df["orders"].sum()) if "orders" in df.columns else 0.0
    baseline_cvr = safe_div(total_orders, total_sessions) if total_sessions > 0 else 0.0

    total_sales = float(df["sales"].sum()) if "sales" in df.columns else 0.0
    total_profit_before = float(df["profit_before_ads"].sum()) if "profit_before_ads" in df.columns else 0.0
    baseline_margin = safe_div(total_profit_before, total_sales) if total_sales > 0 else 0.0

    tasks: List[Dict[str, object]] = []

    for _, r in df.iterrows():
        asin = str(r.get("asin", "")).strip()
        if not asin or asin.lower() == "nan":
            continue

        stage = str(r.get("stage", "") or "")
        direction = str(r.get("direction", "") or "")
        ad_spend = float(r.get("ad_spend", 0.0) or 0.0)
        cap = float(r.get("max_ad_spend_by_profit", 0.0) or 0.0)
        profit_before = float(r.get("profit_before_ads", 0.0) or 0.0)
        profit_after = float(r.get("profit_after_ads", 0.0) or 0.0)
        inv = float(r.get("inventory", 0.0) or 0.0)
        refund_rate = float(r.get("refund_rate", 0.0) or 0.0)
        rating = float(r.get("rating", 0.0) or 0.0)
        sessions = int(r.get("sessions", 0.0) or 0.0)
        orders = int(r.get("orders", 0.0) or 0.0)
        reasons = str(r.get("reasons", "") or "")

        # 0) 库存优先（P0）
        if inv > 0 and inv <= 20 and ad_spend > 0:
            tasks.append(
                {
                    "priority": "P0",
                    "asin": asin,
                    "task_type": "库存补货/控量",
                    "owner": "供应链/运营",
                    "need": f"库存≤20 且仍有广告花费，先补货或控量/停投避免断货拖累转化",
                    "budget_gap_usd_est": 0.0,
                    "profit_gap_usd_est": 0.0,
                    "target": "库存恢复后再评估放量",
                    "stage": stage,
                    "direction": direction,
                    "evidence": reasons,
                }
            )

        # 1) 广告超承受度（P0）：可以直接通过“降预算/降价/收口”修复
        if ad_spend > 0 and cap > 0 and ad_spend > cap:
            gap = ad_spend - cap
            tasks.append(
                {
                    "priority": "P0",
                    "asin": asin,
                    "task_type": "预算/出价收口",
                    "owner": "广告运营",
                    "need": "广告花费超过毛利承受度（按阶段比例），先把相关活动预算/出价收口到承受范围内",
                    "budget_gap_usd_est": round(gap, 2),
                    "profit_gap_usd_est": 0.0,
                    "target": f"把该ASIN相关广告花费降到≈${cap:.2f}（估算需减少≈${gap:.2f}）",
                    "stage": stage,
                    "direction": direction,
                    "evidence": reasons,
                }
            )

        # 2) 利润承受度<=0（P1）：需要修利润（价格/成本/退货/Listing）
        if profit_before <= 0 and ad_spend > 0:
            need_profit = -profit_before
            tasks.append(
                {
                    "priority": "P1",
                    "asin": asin,
                    "task_type": "修利润（价格/成本/退货）",
                    "owner": "运营/财务/供应链",
                    "need": "利润承受度≤0，当前不具备放量基础",
                    "budget_gap_usd_est": 0.0,
                    "profit_gap_usd_est": round(need_profit, 2),
                    "target": f"至少把利润承受度提升到>0（需提升≈${need_profit:.2f}）",
                    "stage": stage,
                    "direction": direction,
                    "evidence": reasons,
                }
            )

        # 3) 转化短板（P1）：用店铺基线 CVR 对比（只在样本够大时触发）
        asin_cvr = safe_div(float(orders), float(sessions)) if sessions > 0 else 0.0
        if sessions >= min_sessions_for_cvr and baseline_cvr > 0 and asin_cvr < baseline_cvr * 0.7:
            tasks.append(
                {
                    "priority": "P1",
                    "asin": asin,
                    "task_type": "Listing转化优化",
                    "owner": "运营/美工",
                    "need": "ASIN 转化率显著低于店铺基线，广告放量会放大亏损",
                    "budget_gap_usd_est": 0.0,
                    "profit_gap_usd_est": 0.0,
                    "target": f"把CVR从{asin_cvr:.2%}提升到≥{baseline_cvr:.2%}（基于店铺基线）",
                    "stage": stage,
                    "direction": direction,
                    "evidence": f"sessions={sessions}, orders={orders}, baseline_cvr={baseline_cvr:.2%}",
                }
            )

        # 4) 退款/评分（P2）：偏产品/售后问题，不给金额预测
        if refund_rate >= 0.08:
            tasks.append(
                {
                    "priority": "P2",
                    "asin": asin,
                    "task_type": "降低退款/退货",
                    "owner": "产品/售后/运营",
                    "need": "退款率偏高会直接侵蚀利润并影响转化",
                    "budget_gap_usd_est": 0.0,
                    "profit_gap_usd_est": 0.0,
                    "target": "排查差评/退货原因并把退款率拉回≤5%（经验目标）",
                    "stage": stage,
                    "direction": direction,
                    "evidence": f"refund_rate={refund_rate:.2%}, baseline_margin={baseline_margin:.2%}",
                }
            )
        if 0 < rating < 4.2:
            tasks.append(
                {
                    "priority": "P2",
                    "asin": asin,
                    "task_type": "提升评分/Review",
                    "owner": "运营/产品",
                    "need": "评分偏低会拉低CTR/CVR，广告放量难起量且更贵",
                    "budget_gap_usd_est": 0.0,
                    "profit_gap_usd_est": 0.0,
                    "target": "把评分提升到≥4.3 并控制新增差评",
                    "stage": stage,
                    "direction": direction,
                    "evidence": f"rating={rating:.2f}",
                }
            )

    # 去重：同ASIN同task_type只保留最高优先级那条
    pr_rank = {"P0": 0, "P1": 1, "P2": 2}
    tasks.sort(key=lambda x: (pr_rank.get(str(x.get("priority", "")), 9), -float(x.get("budget_gap_usd_est", 0.0) or 0.0)))
    best = {}
    for t in tasks:
        key = (t.get("asin"), t.get("task_type"))
        cur = best.get(key)
        if cur is None:
            best[key] = t
            continue
        if pr_rank.get(t.get("priority"), 9) < pr_rank.get(cur.get("priority"), 9):
            best[key] = t
    out = list(best.values())
    out.sort(key=lambda x: (pr_rank.get(str(x.get("priority", "")), 9), -float(x.get("budget_gap_usd_est", 0.0) or 0.0), -float(x.get("profit_gap_usd_est", 0.0) or 0.0)))
    return out[:200]
