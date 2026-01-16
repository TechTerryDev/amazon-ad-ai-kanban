# -*- coding: utf-8 -*-
"""
把“计算层输出”整理成给 AI/子 agent 直接消费的 JSON（每店铺一份）。

目标：
- 结构稳定、字段少但信息密度高
- 以 ASIN 为中心：生命周期（动态窗口） + 广告结构（Top campaigns/search terms/targetings）

注意：
- 这里不生成“动作建议”，只整理数据输入；
- AI 侧可以用不同岗位视角（投手/产品/供应链/财务）读取同一份 JSON 出报告。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def _to_records(df: Optional[pd.DataFrame], limit: int = 0) -> List[Dict[str, object]]:
    if df is None or df.empty:
        return []
    view = df.copy()
    if limit and limit > 0:
        view = view.head(int(limit)).copy()
    try:
        return view.to_dict(orient="records")
    except Exception:
        return []


def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def _safe_div(a: object, b: object) -> float:
    """
    安全除法：任何异常或除数为0时返回0.0（AI输入包必须稳定，不得因缺口崩溃）。
    """
    try:
        bb = _safe_float(b)
        if bb == 0:
            return 0.0
        return _safe_float(a) / bb
    except Exception:
        return 0.0


def build_ai_input_bundle(
    shop: str,
    stage: str,
    date_start: str,
    date_end: str,
    summary_total: Dict[str, object],
    product_analysis_summary: Dict[str, object],
    lifecycle_board: Optional[pd.DataFrame],
    lifecycle_windows: Optional[pd.DataFrame],
    asin_top_campaigns: Optional[pd.DataFrame],
    asin_top_search_terms: Optional[pd.DataFrame],
    asin_top_targetings: Optional[pd.DataFrame],
    asin_top_placements: Optional[pd.DataFrame],
    shop_scorecard: Optional[Dict[str, object]] = None,
    asin_limit: int = 120,
) -> Dict[str, object]:
    """
    生成单店铺 AI 输入包（可直接喂给 Claude skill / 子 agent）。

    asin_limit：
    - 默认只保留“最需要关注”的前 N 个 ASIN（通常按生命周期看板里的 ad_spend_roll 排序）
    """
    bundle: Dict[str, object] = {
        "shop": shop,
        "stage_profile": stage,
        "date_range": {"date_start": date_start, "date_end": date_end},
        "summary_total": summary_total or {},
        "product_analysis_summary": product_analysis_summary or {},
        "shop_scorecard": shop_scorecard or {},
        "definitions": {
            "window_type": {
                "since_first_stock_to_date": "主口径：从首次可售日期到当前周期末（更贴近运营调广告的起点）",
                "since_first_sale_to_date": "辅助口径：从首单到当前周期末（兼容预售/测评导致的早出单）",
                "cycle_to_date": "当前补货周期累计（断货>=阈值后到货算新周期）",
                "current_phase_to_date": "当前生命周期阶段累计（launch/growth/mature/decline 等）",
                "compare_7d/14d/30d": "滚动环比：最近N天 vs 前N天（Δ与增量效率：marginal_*）",
            },
            "derived": {
                "aov": "客单价（Average Order Value）= sales / orders（产品侧订单均价，不是广告端）",
                "gross_margin": "毛利率（gross_margin）= profit / sales（profit 来自产品分析的“毛利润”字段）",
            },
        },
        "asins": [],
    }

    # ========== 选择需要重点分析的 ASIN ==========
    focus_asins: List[str] = []
    if lifecycle_board is not None and not lifecycle_board.empty and "asin" in lifecycle_board.columns:
        view = lifecycle_board.copy()
        # 优先按最近 rolling 广告花费排序（更贴近“运营要先看谁”）
        if "ad_spend_roll" in view.columns:
            try:
                view = view.sort_values("ad_spend_roll", ascending=False)
            except Exception:
                pass
        focus_asins = [str(x).strip().upper() for x in view["asin"].dropna().tolist() if str(x).strip()]
    focus_asins = [a for a in focus_asins if a and a.lower() != "nan"]
    if asin_limit and asin_limit > 0:
        focus_asins = focus_asins[: int(asin_limit)]

    # ========== 生命周期窗口：按 ASIN 建索引 ==========
    main_windows = pd.DataFrame()
    compare_windows = pd.DataFrame()
    if lifecycle_windows is not None and not lifecycle_windows.empty and "asin" in lifecycle_windows.columns:
        w = lifecycle_windows.copy()
        w["asin_norm"] = w["asin"].astype(str).str.upper().str.strip()
        # 主口径窗口
        main_windows = w[w["window_type"] == "since_first_stock_to_date"].copy()
        if main_windows.empty:
            main_windows = w[w["window_type"] == "cycle_to_date"].copy()
        # compare 窗口（7/14/30）
        compare_windows = w[w["window_type"].astype(str).str.startswith("compare_")].copy()

    # ========== 广告结构表：按 ASIN 建索引 ==========
    def _pick(df: Optional[pd.DataFrame], asin: str) -> pd.DataFrame:
        if df is None or df.empty or "asin" not in df.columns:
            return pd.DataFrame()
        try:
            s = df["asin"].astype(str).str.upper().str.strip()
            return df[s == asin].copy()
        except Exception:
            return pd.DataFrame()

    # ========== 逐 ASIN 组装 ==========
    for asin in focus_asins:
        item: Dict[str, object] = {"asin": asin}

        # 生命周期：current board（当前状态）
        if lifecycle_board is not None and not lifecycle_board.empty and "asin" in lifecycle_board.columns:
            try:
                b = lifecycle_board.copy()
                b["asin_norm"] = b["asin"].astype(str).str.upper().str.strip()
                row = b[b["asin_norm"] == asin].head(1)
                if not row.empty:
                    r0 = row.iloc[0].to_dict()
                    # 只保留关键字段，避免 JSON 过大且难维护
                    item["lifecycle_current"] = {
                        "cycle_id": r0.get("cycle_id"),
                        "product_category": r0.get("product_category", ""),
                        "current_phase": r0.get("current_phase"),
                        "prev_phase": r0.get("prev_phase"),
                        "phase_change": r0.get("phase_change"),
                        "phase_change_days_ago": r0.get("phase_change_days_ago"),
                        "phase_changed_recent_14d": r0.get("phase_changed_recent_14d"),
                        "phase_trend_14d": r0.get("phase_trend_14d"),
                        "date": r0.get("date"),
                        "sales_roll": r0.get("sales_roll"),
                        "sessions_roll": r0.get("sessions_roll"),
                        "ad_spend_roll": r0.get("ad_spend_roll"),
                        "profit_roll": r0.get("profit_roll"),
                        "tacos_roll": r0.get("tacos_roll"),
                        "inventory": r0.get("inventory"),
                        "flag_low_inventory": r0.get("flag_low_inventory"),
                        "flag_oos": r0.get("flag_oos"),
                        "product_name": r0.get("product_name", ""),
                    }
            except Exception:
                pass

        # 生命周期：主口径窗口（累计）
        try:
            if not main_windows.empty:
                row = main_windows[main_windows["asin_norm"] == asin].head(1)
                if not row.empty:
                    r0 = row.iloc[0].to_dict()
                    # 让 AI “只做解释，不做算数”：在输入包里提前派生关键指标
                    sales = _safe_float(r0.get("sales", 0.0))
                    orders = _safe_float(r0.get("orders", 0.0))
                    profit = _safe_float(r0.get("profit", 0.0))
                    item["lifecycle_main_window"] = {
                        "window_type": r0.get("window_type"),
                        "phase": r0.get("phase"),
                        "date_start": r0.get("date_start"),
                        "date_end": r0.get("date_end"),
                        "sales": sales,
                        "orders": orders,
                        "aov": _safe_div(sales, orders),
                        "sessions": r0.get("sessions"),
                        "ad_spend": r0.get("ad_spend"),
                        "ad_sales": r0.get("ad_sales"),
                        "ad_orders": r0.get("ad_orders"),
                        "profit": profit,
                        "gross_margin": _safe_div(profit, sales),
                        "tacos": r0.get("tacos"),
                        "ad_acos": r0.get("ad_acos"),
                        "cvr": r0.get("cvr"),
                        # 自然/广告拆分（产品语境关键字段）
                        "ad_sales_share": r0.get("ad_sales_share"),
                        "ad_orders_share": r0.get("ad_orders_share"),
                        "organic_sales": r0.get("organic_sales"),
                        "organic_orders": r0.get("organic_orders"),
                        "organic_sales_share": r0.get("organic_sales_share"),
                        "in_stock_days": r0.get("in_stock_days"),
                        "oos_days": r0.get("oos_days"),
                    }
        except Exception:
            pass

        # 生命周期：滚动环比窗口（7/14/30）
        try:
            if not compare_windows.empty:
                rows = compare_windows[compare_windows["asin_norm"] == asin].copy()
                # 按 window_days 从小到大排序
                if "window_days" in rows.columns:
                    rows = rows.sort_values("window_days")
                keep_cols = [
                    "window_type",
                    "window_days",
                    "recent_start",
                    "recent_end",
                    "prev_start",
                    "prev_end",
                    # prev/recent：让 AI 直接引用，不做算数
                    "spend_prev",
                    "spend_recent",
                    "sales_prev",
                    "sales_recent",
                    "orders_prev",
                    "orders_recent",
                    "sessions_prev",
                    "sessions_recent",
                    # 产品语境信号（生命周期窗口已计算）
                    "cvr_prev",
                    "cvr_recent",
                    "delta_cvr",
                    "organic_sales_prev",
                    "organic_sales_recent",
                    "delta_organic_sales",
                    "organic_sales_share_prev",
                    "organic_sales_share_recent",
                    "delta_organic_sales_share",
                    "delta_spend",
                    "delta_sales",
                    "delta_orders",
                    "delta_sessions",
                    "marginal_tacos",
                    "marginal_ad_acos",
                ]
                keep_cols = [c for c in keep_cols if c in rows.columns]
                view = rows[keep_cols].copy() if keep_cols else rows.copy()

                # AOV 环比派生：让 AI 不需要用 sales/orders 自己计算
                if ("sales_prev" in view.columns) and ("orders_prev" in view.columns):
                    view["aov_prev"] = 0.0
                    m = pd.to_numeric(view["orders_prev"], errors="coerce").fillna(0.0) > 0
                    view.loc[m, "aov_prev"] = (
                        pd.to_numeric(view.loc[m, "sales_prev"], errors="coerce").fillna(0.0)
                        / pd.to_numeric(view.loc[m, "orders_prev"], errors="coerce").fillna(0.0)
                    ).fillna(0.0)
                if ("sales_recent" in view.columns) and ("orders_recent" in view.columns):
                    view["aov_recent"] = 0.0
                    m = pd.to_numeric(view["orders_recent"], errors="coerce").fillna(0.0) > 0
                    view.loc[m, "aov_recent"] = (
                        pd.to_numeric(view.loc[m, "sales_recent"], errors="coerce").fillna(0.0)
                        / pd.to_numeric(view.loc[m, "orders_recent"], errors="coerce").fillna(0.0)
                    ).fillna(0.0)
                if ("aov_prev" in view.columns) and ("aov_recent" in view.columns):
                    view["delta_aov"] = pd.to_numeric(view["aov_recent"], errors="coerce").fillna(0.0) - pd.to_numeric(view["aov_prev"], errors="coerce").fillna(0.0)
                    view["ratio_aov"] = 1.0
                    base = pd.to_numeric(view["aov_prev"], errors="coerce").fillna(0.0)
                    m = base > 0
                    view.loc[m, "ratio_aov"] = (pd.to_numeric(view.loc[m, "aov_recent"], errors="coerce").fillna(0.0) / base[m]).fillna(1.0)

                item["lifecycle_compares"] = _to_records(view, limit=0)
        except Exception:
            pass

        # 广告结构：Top campaigns / search terms / targetings / placements
        item["ads"] = {
            "top_campaigns": _to_records(_pick(asin_top_campaigns, asin), limit=10),
            "top_search_terms": _to_records(_pick(asin_top_search_terms, asin), limit=20),
            "top_targetings": _to_records(_pick(asin_top_targetings, asin), limit=20),
            "top_placements": _to_records(_pick(asin_top_placements, asin), limit=10),
        }

        bundle["asins"].append(item)

    return bundle
