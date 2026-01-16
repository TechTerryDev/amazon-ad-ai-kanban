# -*- coding: utf-8 -*-
"""
ASIN 生命周期识别（面向运营的“动态周期”方法，不是硬阈值筛选）。

数据源：
- 产品分析（按日）：销售额/订单量/Sessions/广告花费/广告销售额/广告订单量/FBA可售/毛利润/退款率/评分等

输出：
- daily：每天的生命周期阶段 + 关键滚动指标（可用于复盘/可视化）
- segments：把 daily 连续相同阶段压缩成“阶段段落”（更适合运营阅读）

设计目标：
- 支持“一个ASIN多个周期”（例如断货/补货后重新上架/重推）
- 周期边界优先用“断货(FBA可售=0)持续一段时间后再次到货”来切分（更贴近你的运营场景）
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.schema import CAN
from core.utils import safe_div, to_float


@dataclass(frozen=True)
class LifecycleConfig:
    # 多久“不活跃”后，重新出现算新周期
    new_cycle_inactive_days: int = 28
    # 多久“断货(FBA可售=0)”后，再次到货算新周期（更贴近你的运营：断货->到货=新一轮）
    new_cycle_oos_days: int = 14
    # rolling 窗口（天）
    roll_days: int = 7
    # 启动期（首单后多少天视为 launch）
    launch_days: int = 14
    # 判定成熟/衰退阈值（相对峰值的比例）
    mature_ratio: float = 0.85
    decline_ratio: float = 0.65
    # 库存阈值（用于 flag）
    low_inventory: int = 20


_PHASE_TREND_RANK: Dict[str, int] = {
    # 0=最弱/未启动，2=最强/更健康（只用于“趋势方向”判断，不是模型评分）
    "pre_launch": 0,
    "launch": 1,
    "growth": 2,
    "stable": 2,
    "mature": 2,
    "decline": 1,
    "inactive": 0,
}


def _phase_trend(prev_phase: str, current_phase: str) -> str:
    """
    生命周期阶段趋势（仅用于“阶段迁移是否走弱/走强”的可解释标签）。
    - up：阶段走强（如 launch→growth）
    - down：阶段走弱（如 growth→decline）
    - flat：阶段变化但强弱相同（如 stable→mature）
    - unknown：无法判定
    """
    try:
        p = str(prev_phase or "").strip().lower()
        c = str(current_phase or "").strip().lower()
        pr = _PHASE_TREND_RANK.get(p, -1)
        cr = _PHASE_TREND_RANK.get(c, -1)
        if pr < 0 or cr < 0:
            return "unknown"
        if cr > pr:
            return "up"
        if cr < pr:
            return "down"
        return "flat"
    except Exception:
        return "unknown"


def _to_date(x: object) -> Optional[dt.date]:
    """
    把各种“日期类型”尽量转成 dt.date（兼容 pandas Timestamp / str）。
    """
    if isinstance(x, dt.date):
        return x
    try:
        if hasattr(x, "date"):
            return x.date()  # type: ignore[no-any-return]
    except Exception:
        pass
    try:
        s = str(x).strip()
        return dt.date.fromisoformat(s) if s else None
    except Exception:
        return None


def _first_date_where(df: pd.DataFrame, mask: pd.Series) -> Optional[dt.date]:
    try:
        if df is None or df.empty or CAN.date not in df.columns:
            return None
        idx = df.index[mask.fillna(False)].tolist()
        if not idx:
            return None
        v = df.loc[idx[0], CAN.date]
        return _to_date(v)
    except Exception:
        return None


def _last_date_where(df: pd.DataFrame, mask: pd.Series) -> Optional[dt.date]:
    try:
        if df is None or df.empty or CAN.date not in df.columns:
            return None
        idx = df.index[mask.fillna(False)].tolist()
        if not idx:
            return None
        v = df.loc[idx[-1], CAN.date]
        return _to_date(v)
    except Exception:
        return None


def _ensure_daily_index(df: pd.DataFrame, date_min: dt.date, date_max: dt.date) -> pd.DataFrame:
    all_days = pd.date_range(date_min, date_max, freq="D").date
    out = df.set_index(CAN.date).reindex(all_days).reset_index().rename(columns={"index": CAN.date})
    return out


def _coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 常用列名（产品分析保持原表头；这里只做数值化）
    for col in ("销售额", "订单量", "Sessions", "广告花费", "广告销售额", "广告订单量", "毛利润", "退款率", "星级评分", "FBA可售"):
        if col in df.columns:
            df[col] = df[col].apply(to_float)
    return df


def _active_flag(df: pd.DataFrame) -> pd.Series:
    sales = df["销售额"] if "销售额" in df.columns else 0.0
    sessions = df["Sessions"] if "Sessions" in df.columns else 0.0
    ad_spend = df["广告花费"] if "广告花费" in df.columns else 0.0
    orders = df["订单量"] if "订单量" in df.columns else 0.0
    return (sales > 0) | (orders > 0) | (sessions > 0) | (ad_spend > 0)


def _sum_window(df: pd.DataFrame) -> Dict[str, float]:
    """
    汇总一个窗口内的关键指标（用于“动态日期范围”输出）。
    """
    if df is None or df.empty:
        return {
            "sales": 0.0,
            "orders": 0.0,
            "sessions": 0.0,
            "ad_spend": 0.0,
            "ad_sales": 0.0,
            "ad_orders": 0.0,
            "profit": 0.0,
            "tacos": 0.0,
            "ad_acos": 0.0,
            "cvr": 0.0,
            "ad_impressions": 0.0,
            "ad_clicks": 0.0,
            "ad_ctr": 0.0,
            "ad_cvr": 0.0,
            "organic_orders": 0.0,
            "organic_sales": 0.0,
            "ad_sales_share": 0.0,
            "organic_sales_share": 0.0,
            "ad_orders_share": 0.0,
            "sp_spend": 0.0,
            "sb_spend": 0.0,
            "sd_spend": 0.0,
            "in_stock_days": 0.0,
            "oos_days": 0.0,
            "oos_with_sessions_days": 0.0,
            "oos_with_ad_spend_days": 0.0,
            "presale_order_days": 0.0,
        }

    def s(col: str) -> float:
        try:
            if col not in df.columns:
                return 0.0
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
        except Exception:
            return 0.0

    def s_like(prefix: str, suffix: str) -> float:
        """
        汇总匹配 prefix/suffix 的列（赛狐产品分析里 SB 常拆成多类：商品集/视频/旗舰店）。
        """
        try:
            cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix) and c.endswith(suffix)]
            if not cols:
                return 0.0
            return float(pd.to_numeric(df[cols].stack(), errors="coerce").fillna(0.0).sum())
        except Exception:
            return 0.0

    sales = s("销售额")
    orders = s("订单量")
    sessions = s("Sessions")
    ad_spend = s("广告花费")
    ad_sales = s("广告销售额")
    ad_orders = s("广告订单量")
    profit = s("毛利润")
    ad_impr = s("广告曝光量")
    ad_clicks = s("广告点击量")
    organic_orders = s("自然订单量")
    organic_sales = s("自然销售额")

    # 广告花费按类型拆分（用于“SB/SD 没投也要能看出来”）
    sp_spend = s_like("SP", "广告花费")
    sb_spend = s_like("SB", "广告花费")
    sd_spend = s_like("SD", "广告花费")

    # 如果没有“自然销售额”列，就用总销售额 - 广告销售额推一个（做下限裁剪，避免负数）
    if organic_sales <= 0 and sales > 0 and ad_sales > 0:
        organic_sales = max(0.0, sales - ad_sales)

    tacos = safe_div(ad_spend, sales) if sales > 0 else 0.0
    ad_acos = safe_div(ad_spend, ad_sales) if ad_sales > 0 else 0.0
    cvr = safe_div(orders, sessions) if sessions > 0 else 0.0
    ad_ctr = safe_div(ad_clicks, ad_impr) if ad_impr > 0 else 0.0
    ad_cvr = safe_div(ad_orders, ad_clicks) if ad_clicks > 0 else 0.0
    ad_orders_share = safe_div(ad_orders, orders) if orders > 0 else 0.0
    ad_sales_share = safe_div(ad_sales, sales) if sales > 0 else 0.0
    organic_sales_share = safe_div(organic_sales, sales) if sales > 0 else 0.0

    # 库存/断货天数（用于动态窗口解释）
    try:
        fba = pd.to_numeric(df.get("FBA可售", 0.0), errors="coerce").fillna(0.0)
        in_stock_days = float((fba > 0).sum())
        oos_days = float((fba == 0).sum())
        sess = pd.to_numeric(df.get("Sessions", 0.0), errors="coerce").fillna(0.0)
        spend = pd.to_numeric(df.get("广告花费", 0.0), errors="coerce").fillna(0.0)
        oos_with_sessions_days = float(((fba == 0) & (sess > 0)).sum())
        oos_with_ad_spend_days = float(((fba == 0) & (spend > 0)).sum())
        presale_order_days = float(
            ((fba == 0) & ((pd.to_numeric(df.get("销售额", 0.0), errors="coerce").fillna(0.0) > 0) | (pd.to_numeric(df.get("订单量", 0.0), errors="coerce").fillna(0.0) > 0))).sum()
        )
    except Exception:
        in_stock_days = 0.0
        oos_days = 0.0
        oos_with_sessions_days = 0.0
        oos_with_ad_spend_days = 0.0
        presale_order_days = 0.0
    return {
        "sales": sales,
        "orders": orders,
        "sessions": sessions,
        "ad_spend": ad_spend,
        "ad_sales": ad_sales,
        "ad_orders": ad_orders,
        "profit": profit,
        "tacos": tacos,
        "ad_acos": ad_acos,
        "cvr": cvr,
        "ad_impressions": ad_impr,
        "ad_clicks": ad_clicks,
        "ad_ctr": ad_ctr,
        "ad_cvr": ad_cvr,
        "organic_orders": organic_orders,
        "organic_sales": organic_sales,
        "ad_sales_share": ad_sales_share,
        "organic_sales_share": organic_sales_share,
        "ad_orders_share": ad_orders_share,
        "sp_spend": sp_spend,
        "sb_spend": sb_spend,
        "sd_spend": sd_spend,
        "in_stock_days": in_stock_days,
        "oos_days": oos_days,
        "oos_with_sessions_days": oos_with_sessions_days,
        "oos_with_ad_spend_days": oos_with_ad_spend_days,
        "presale_order_days": presale_order_days,
    }


def _compare_recent_prev(ts: pd.DataFrame, end_date: dt.date, window_days: int) -> Dict[str, object]:
    """
    最近N天 vs 前N天：输出一个对比行（更符合“动态日期范围”，不是逐日标签）。
    """
    n = int(window_days or 0)
    if n <= 0 or ts is None or ts.empty or CAN.date not in ts.columns:
        return {}
    d_end = end_date
    recent_start = d_end - dt.timedelta(days=n - 1)
    prev_end = recent_start - dt.timedelta(days=1)
    prev_start = prev_end - dt.timedelta(days=n - 1)

    prev = ts[(ts[CAN.date] >= prev_start) & (ts[CAN.date] <= prev_end)].copy()
    recent = ts[(ts[CAN.date] >= recent_start) & (ts[CAN.date] <= d_end)].copy()

    p = _sum_window(prev)
    r = _sum_window(recent)

    delta_sales = float(r["sales"]) - float(p["sales"])
    delta_ad_spend = float(r["ad_spend"]) - float(p["ad_spend"])
    delta_ad_sales = float(r["ad_sales"]) - float(p["ad_sales"])
    delta_orders = float(r["orders"]) - float(p["orders"])
    delta_sessions = float(r["sessions"]) - float(p["sessions"])
    delta_ad_clicks = float(r["ad_clicks"]) - float(p["ad_clicks"])
    delta_cvr = float(r.get("cvr", 0.0) or 0.0) - float(p.get("cvr", 0.0) or 0.0)
    delta_organic_sales = float(r.get("organic_sales", 0.0) or 0.0) - float(p.get("organic_sales", 0.0) or 0.0)
    delta_organic_sales_share = float(r.get("organic_sales_share", 0.0) or 0.0) - float(p.get("organic_sales_share", 0.0) or 0.0)

    return {
        "window_days": int(n),
        "recent_start": str(recent_start),
        "recent_end": str(d_end),
        "prev_start": str(prev_start),
        "prev_end": str(prev_end),
        "spend_prev": float(p["ad_spend"]),
        "spend_recent": float(r["ad_spend"]),
        "sales_prev": float(p["sales"]),
        "sales_recent": float(r["sales"]),
        "orders_prev": float(p["orders"]),
        "orders_recent": float(r["orders"]),
        "sessions_prev": float(p["sessions"]),
        "sessions_recent": float(r["sessions"]),
        # 产品侧 CVR（orders/sessions）：用于识别“流量上升但转化下滑”等可解释信号
        "cvr_prev": float(p.get("cvr", 0.0) or 0.0),
        "cvr_recent": float(r.get("cvr", 0.0) or 0.0),
        # 自然端：用于把“销量变化”拆成自然 vs 广告（更贴近运营决策）
        "organic_sales_prev": float(p.get("organic_sales", 0.0) or 0.0),
        "organic_sales_recent": float(r.get("organic_sales", 0.0) or 0.0),
        "organic_sales_share_prev": float(p.get("organic_sales_share", 0.0) or 0.0),
        "organic_sales_share_recent": float(r.get("organic_sales_share", 0.0) or 0.0),
        "ad_clicks_prev": float(p["ad_clicks"]),
        "ad_clicks_recent": float(r["ad_clicks"]),
        "delta_spend": float(delta_ad_spend),
        "delta_sales": float(delta_sales),
        "delta_orders": float(delta_orders),
        "delta_sessions": float(delta_sessions),
        "delta_ad_clicks": float(delta_ad_clicks),
        "delta_cvr": float(delta_cvr),
        "delta_organic_sales": float(delta_organic_sales),
        "delta_organic_sales_share": float(delta_organic_sales_share),
        # 增量效率：给 2 个口径（总销售额口径 / 广告销售额口径），你后续可选用更合适的
        "marginal_tacos": safe_div(delta_ad_spend, delta_sales) if delta_sales != 0 else 0.0,
        "marginal_ad_acos": safe_div(delta_ad_spend, delta_ad_sales) if delta_ad_sales != 0 else 0.0,
    }


def _assign_cycle_id(active: pd.Series, cfg: LifecycleConfig) -> pd.Series:
    """
    用“不活跃连续天数”切分周期：inactive>=N 后再次活跃 -> new cycle。
    """
    cycle = []
    cur = 1
    inactive_streak = 0
    for v in active.tolist():
        if not bool(v):
            inactive_streak += 1
        else:
            if inactive_streak >= cfg.new_cycle_inactive_days and len(cycle) > 0:
                cur += 1
            inactive_streak = 0
        cycle.append(cur)
    return pd.Series(cycle)


def _assign_cycle_id_by_inventory(ts: pd.DataFrame, cfg: LifecycleConfig) -> Optional[pd.Series]:
    """
    以“断货->到货”为周期切分（更贴近你的实际）：连续断货>=N天后再次到货 => 新周期。

    说明：
    - 只在存在 FBA可售 列时启用
    - 断货阈值用 cfg.new_cycle_oos_days（默认 14 天）
    """
    if ts is None or ts.empty or "FBA可售" not in ts.columns:
        return None

    try:
        in_stock = pd.to_numeric(ts["FBA可售"], errors="coerce").fillna(0.0) > 0
    except Exception:
        return None

    n = int(cfg.new_cycle_oos_days or 0)
    if n <= 0:
        n = 14

    cycle: List[int] = []
    cur = 1
    oos_streak = 0
    seen_stock = False
    for v in in_stock.tolist():
        if bool(v):
            if seen_stock and oos_streak >= n and len(cycle) > 0:
                cur += 1
            seen_stock = True
            oos_streak = 0
        else:
            if seen_stock:
                oos_streak += 1
        cycle.append(cur)
    return pd.Series(cycle)


def _rolling(series: pd.Series, n: int) -> pd.Series:
    try:
        return series.rolling(n, min_periods=1).mean()
    except Exception:
        return series


def label_lifecycle_for_asin(ts: pd.DataFrame, cfg: LifecycleConfig) -> pd.DataFrame:
    """
    输入：单个 ASIN 的按日数据（必须包含 date）
    输出：每天的 lifecycle_phase + flags + rolling 指标
    """
    if ts is None or ts.empty or CAN.date not in ts.columns:
        return pd.DataFrame()

    ts = _coerce_cols(ts)
    ts = ts.sort_values(CAN.date).copy()
    dmin = ts[CAN.date].min()
    dmax = ts[CAN.date].max()
    if not isinstance(dmin, dt.date) or not isinstance(dmax, dt.date):
        return pd.DataFrame()

    ts = _ensure_daily_index(ts, dmin, dmax)
    # 缺失填0（代表当天无数据/无行为）
    for col in ("销售额", "订单量", "Sessions", "广告花费", "广告销售额", "广告订单量", "毛利润", "退款率", "星级评分", "FBA可售"):
        if col in ts.columns:
            ts[col] = pd.to_numeric(ts[col], errors="coerce").fillna(0.0)
        else:
            ts[col] = 0.0

    active = _active_flag(ts)
    ts["active"] = active.astype(int)
    # 周期切分优先按库存（断货->到货），缺库存列时再回退到 active（历史兼容）
    inv_cycle = _assign_cycle_id_by_inventory(ts, cfg)
    if inv_cycle is not None:
        ts["cycle_id"] = inv_cycle.astype(int)
    else:
        ts["cycle_id"] = _assign_cycle_id(active, cfg).astype(int)

    # rolling 指标（用于动态阶段判断）
    ts["sales_roll"] = _rolling(ts["销售额"], cfg.roll_days)
    ts["sessions_roll"] = _rolling(ts["Sessions"], cfg.roll_days)
    ts["ad_spend_roll"] = _rolling(ts["广告花费"], cfg.roll_days)
    ts["profit_roll"] = _rolling(ts["毛利润"], cfg.roll_days)

    ts["tacos_roll"] = ts.apply(lambda r: safe_div(r["广告花费"], r["销售额"]) if r["销售额"] > 0 else 0.0, axis=1)
    ts["cvr_roll"] = ts.apply(lambda r: safe_div(r["订单量"], r["Sessions"]) if r["Sessions"] > 0 else 0.0, axis=1)

    # slope：rolling_sales 的变化（近似导数）
    ts["sales_slope"] = ts["sales_roll"].diff().fillna(0.0)

    # 阶段：按 cycle 内的峰值动态归一化
    phases: List[str] = []
    for cid, g in ts.groupby("cycle_id", dropna=False, sort=False):
        g = g.copy()
        # cycle 内峰值（用 rolling sales 更稳定）
        peak = float(g["sales_roll"].max() or 0.0)
        peak = peak if peak > 0 else float(g["销售额"].max() or 0.0)
        # 第一次“有效销售”日期：
        # - 优先用“可售>0 且出单”（更贴近你们的业务：正式开售/到货后启动）
        # - 否则回退到“任意出单”（例如预售/测评/数据口径滞后）
        first_sale_any_idx = g.index[(g["销售额"] > 0) | (g["订单量"] > 0)]
        first_sale_any_date = g.loc[first_sale_any_idx[0], CAN.date] if len(first_sale_any_idx) > 0 else None
        first_sale_in_stock_date = None
        try:
            in_stock_idx = g.index[(g["FBA可售"] > 0) & ((g["销售额"] > 0) | (g["订单量"] > 0))]
            first_sale_in_stock_date = g.loc[in_stock_idx[0], CAN.date] if len(in_stock_idx) > 0 else None
        except Exception:
            first_sale_in_stock_date = None
        first_sale_date = first_sale_in_stock_date if first_sale_in_stock_date is not None else first_sale_any_date

        for _, r in g.iterrows():
            d = r[CAN.date]
            is_active = bool(r["active"])
            sales_r = float(r["sales_roll"] or 0.0)
            slope = float(r["sales_slope"] or 0.0)

            if not is_active:
                phase = "inactive"
            else:
                # 正式开售/启动前：pre_launch（以“可售>0 且出单”的首单为主锚点）
                if first_sale_date is None or d < first_sale_date:
                    phase = "pre_launch"
                else:
                    # 相对峰值比例（动态周期）
                    ratio = 0.0 if peak <= 0 else sales_r / peak
                    days_since_first_sale = (d - first_sale_date).days if isinstance(first_sale_date, dt.date) else 0

                    # launch：刚开始出单的一段时间（默认 14 天，可配置）
                    if days_since_first_sale <= int(cfg.launch_days or 14) and ratio < cfg.mature_ratio:
                        phase = "launch"
                    # growth：销售在爬坡（斜率为正），且未到成熟段
                    elif ratio < cfg.mature_ratio and slope >= 0:
                        phase = "growth"
                    # mature：接近峰值且变化不大
                    elif ratio >= cfg.mature_ratio and abs(slope) < max(1e-6, peak * 0.02):
                        phase = "mature"
                    # decline：明显低于峰值且斜率为负
                    elif ratio <= cfg.decline_ratio and slope < 0:
                        phase = "decline"
                    else:
                        phase = "stable"

            phases.append(phase)

    ts["lifecycle_phase"] = phases

    # flags：库存/断货风险（不改 phase，只做标记）
    ts["flag_low_inventory"] = ((ts["FBA可售"] > 0) & (ts["FBA可售"] <= cfg.low_inventory)).astype(int)
    # 断货判定标准：只看库存（FBA可售==0）
    ts["flag_oos"] = (ts["FBA可售"] == 0).astype(int)
    # 断货异常：断货但仍有 Sessions/仍有广告花费（用于排查变体/配送设置/广告异常等）
    ts["flag_oos_with_sessions"] = ((ts["FBA可售"] == 0) & (ts["Sessions"] > 0)).astype(int)
    ts["flag_oos_with_ad_spend"] = ((ts["FBA可售"] == 0) & (ts["广告花费"] > 0)).astype(int)
    # 预售/测评等场景：未可售但已出单（用于后续解释与口径对齐）
    ts["flag_presale_order"] = ((ts["FBA可售"] == 0) & ((ts["销售额"] > 0) | (ts["订单量"] > 0))).astype(int)
    return ts


def compress_segments(labeled: pd.DataFrame, asin: str, shop: str) -> pd.DataFrame:
    """
    把 daily phase 压缩成分段表（连续相同 phase 合并）。
    """
    if labeled is None or labeled.empty:
        return pd.DataFrame()
    cols_need = [CAN.date, "lifecycle_phase", "cycle_id"]
    if any(c not in labeled.columns for c in cols_need):
        return pd.DataFrame()

    x = labeled.sort_values(CAN.date).copy()
    # segment id：phase 或 cycle 变化就开新段
    x["seg_break"] = ((x["lifecycle_phase"] != x["lifecycle_phase"].shift(1)) | (x["cycle_id"] != x["cycle_id"].shift(1))).astype(int)
    x["segment_id"] = x["seg_break"].cumsum()

    segs = []
    for sid, g in x.groupby("segment_id", dropna=False):
        d0 = g[CAN.date].min()
        d1 = g[CAN.date].max()
        segs.append(
            {
                "shop": shop,
                "asin": asin,
                "cycle_id": int(g["cycle_id"].iloc[0]),
                "segment_id": int(sid),
                "phase": str(g["lifecycle_phase"].iloc[0]),
                "date_start": str(d0),
                "date_end": str(d1),
                "days": int((d1 - d0).days + 1) if isinstance(d0, dt.date) and isinstance(d1, dt.date) else int(len(g)),
                "sales_sum": float(g["销售额"].sum()),
                "orders_sum": float(g["订单量"].sum()),
                "sessions_sum": float(g["Sessions"].sum()),
                "ad_spend_sum": float(g["广告花费"].sum()),
                "profit_sum": float(g["毛利润"].sum()),
                "tacos": safe_div(float(g["广告花费"].sum()), float(g["销售额"].sum())) if float(g["销售额"].sum()) > 0 else 0.0,
                "cvr": safe_div(float(g["订单量"].sum()), float(g["Sessions"].sum())) if float(g["Sessions"].sum()) > 0 else 0.0,
                "inv_min": float(g["FBA可售"].min()),
                "low_inv_days": int(g["flag_low_inventory"].sum()) if "flag_low_inventory" in g.columns else 0,
                "oos_days": int(g["flag_oos"].sum()) if "flag_oos" in g.columns else 0,
                "oos_with_sessions_days": int(g["flag_oos_with_sessions"].sum()) if "flag_oos_with_sessions" in g.columns else 0,
                "oos_with_ad_spend_days": int(g["flag_oos_with_ad_spend"].sum()) if "flag_oos_with_ad_spend" in g.columns else 0,
            }
        )
    return pd.DataFrame(segs)


def build_lifecycle_for_shop(
    product_analysis_shop: pd.DataFrame,
    shop: str,
    cfg: Optional[LifecycleConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    返回：
    - daily_all：每个 ASIN 每天（含 phase/flags/rolling）
    - segments_all：每个 ASIN 的分段
    - current_board：当前日期的“当前阶段看板”
    """
    cfg = cfg or LifecycleConfig()
    if product_analysis_shop is None or product_analysis_shop.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if "ASIN" not in product_analysis_shop.columns or CAN.date not in product_analysis_shop.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pa = product_analysis_shop.copy()
    pa = pa[pa[CAN.date].notna()].copy()
    if pa.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    daily_frames = []
    seg_frames = []
    board_rows = []
    for asin, g in pa.groupby("ASIN", dropna=False):
        asin_str = str(asin).strip()
        if not asin_str or asin_str.lower() == "nan":
            continue
        ts = g[[c for c in g.columns]].copy()
        labeled = label_lifecycle_for_asin(ts[[CAN.date] + [c for c in ts.columns if c != CAN.date]].copy(), cfg)
        if labeled.empty:
            continue
        # 注意：产品分析原表可能已经包含 shop/ASIN 列，避免重复 insert 报错
        if "shop" in labeled.columns:
            labeled["shop"] = shop
        else:
            labeled.insert(0, "shop", shop)
        if "asin" in labeled.columns:
            labeled["asin"] = asin_str
        else:
            # 让 asin 紧跟在 shop 后面，便于查看
            labeled.insert(1 if "shop" in labeled.columns else 0, "asin", asin_str)
        daily_frames.append(labeled)
        seg = compress_segments(labeled, asin_str, shop)
        if not seg.empty:
            seg_frames.append(seg)

        # 当前看板：取最后一天
        last = labeled.sort_values(CAN.date).iloc[-1].to_dict()
        # ===== 生命周期迁移信号（让“动态生命周期”可直接驱动优先级）=====
        prev_phase = ""
        phase_change = ""
        phase_change_days_ago: object = ""
        phase_changed_recent_14d = 0
        phase_trend_14d = ""
        try:
            if seg is not None and not seg.empty and "cycle_id" in seg.columns:
                cur_cycle_id = int(last.get("cycle_id", 0) or 0)
                seg2 = seg[seg["cycle_id"] == cur_cycle_id].copy()
                if seg2 is not None and not seg2.empty and "segment_id" in seg2.columns:
                    seg2 = seg2.sort_values(["segment_id"], ascending=[True]).copy()
                    cur_seg = seg2.iloc[-1].to_dict()
                    cur_phase = str(cur_seg.get("phase", "") or "").strip().lower()
                    # 上一段（同一周期内）
                    if len(seg2) >= 2:
                        prev_seg = seg2.iloc[-2].to_dict()
                        prev_phase = str(prev_seg.get("phase", "") or "").strip().lower()
                    if prev_phase and cur_phase and prev_phase != cur_phase:
                        phase_change = f"{prev_phase}→{cur_phase}"
                        end_date = _to_date(last.get(CAN.date))
                        start_date = _to_date(cur_seg.get("date_start"))
                        if end_date is not None and start_date is not None:
                            diff = max(0, int((end_date - start_date).days))
                            phase_change_days_ago = diff
                            # 近14天：用“当前段持续天数<=14”做判定（包含今天）
                            if int(diff) + 1 <= 14:
                                phase_changed_recent_14d = 1
                                phase_trend_14d = _phase_trend(prev_phase, cur_phase)
        except Exception:
            prev_phase = ""
            phase_change = ""
            phase_change_days_ago = ""
            phase_changed_recent_14d = 0
            phase_trend_14d = ""
        board_rows.append(
            {
                "shop": shop,
                "asin": asin_str,
                "cycle_id": int(last.get("cycle_id", 0) or 0),
                "current_phase": str(last.get("lifecycle_phase", "")),
                "prev_phase": str(prev_phase),
                "phase_change": str(phase_change),
                "phase_change_days_ago": phase_change_days_ago,
                "phase_changed_recent_14d": int(phase_changed_recent_14d),
                "phase_trend_14d": str(phase_trend_14d),
                "date": str(last.get(CAN.date, "")),
                "sales_roll": float(last.get("sales_roll", 0.0) or 0.0),
                "sessions_roll": float(last.get("sessions_roll", 0.0) or 0.0),
                "ad_spend_roll": float(last.get("ad_spend_roll", 0.0) or 0.0),
                "profit_roll": float(last.get("profit_roll", 0.0) or 0.0),
                "tacos_roll": float(last.get("tacos_roll", 0.0) or 0.0),
                "cvr_roll": float(last.get("cvr_roll", 0.0) or 0.0),
                "inventory": float(last.get("FBA可售", 0.0) or 0.0),
                "flag_low_inventory": int(last.get("flag_low_inventory", 0) or 0),
                "flag_oos": int(last.get("flag_oos", 0) or 0),
            }
        )

    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    segments_all = pd.concat(seg_frames, ignore_index=True) if seg_frames else pd.DataFrame()
    board = pd.DataFrame(board_rows)
    if not board.empty:
        board = board.sort_values(["ad_spend_roll", "sales_roll"], ascending=False)
    return daily_all, segments_all, board


def build_lifecycle_windows_for_shop(
    lifecycle_daily: pd.DataFrame,
    lifecycle_segments: pd.DataFrame,
    lifecycle_board: pd.DataFrame,
    windows_days: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    你要的“动态日期范围”输出：以 ASIN 为单位、以“当前周期”为单位，生成窗口对比表。

    输出行（每个 ASIN）：
    - cycle_to_date：当前周期累计（cycle_start~cycle_end）
    - current_phase_to_date：当前阶段累计（phase_start~cycle_end）
    - compare_{N}d：最近N天 vs 前N天（N 默认 7/14/30，可与 CLI --windows 保持一致）
    """
    windows_days = windows_days or [7, 14, 30]
    if lifecycle_daily is None or lifecycle_daily.empty:
        return pd.DataFrame()
    if lifecycle_board is None or lifecycle_board.empty:
        return pd.DataFrame()
    if "asin" not in lifecycle_board.columns:
        return pd.DataFrame()
    if CAN.date not in lifecycle_daily.columns or "cycle_id" not in lifecycle_daily.columns:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for _, b in lifecycle_board.iterrows():
        asin = str(b.get("asin", "")).strip()
        if not asin or asin.lower() == "nan":
            continue
        shop = str(b.get("shop", "")).strip()
        cycle_id = int(b.get("cycle_id", 0) or 0)
        end_date = _to_date(b.get("date")) or None
        if end_date is None:
            continue

        ts = lifecycle_daily[(lifecycle_daily.get("asin").astype(str) == asin) & (lifecycle_daily["cycle_id"] == cycle_id)].copy()
        if ts.empty:
            continue
        ts = ts[ts[CAN.date].notna()].copy()
        ts = ts.sort_values(CAN.date)
        cycle_start = ts[CAN.date].min()
        cycle_end = ts[CAN.date].max()
        if not isinstance(cycle_start, dt.date) or not isinstance(cycle_end, dt.date):
            continue

        phase = str(b.get("current_phase", "")).strip()
        phase_start = cycle_start
        try:
            seg = lifecycle_segments[(lifecycle_segments["asin"].astype(str) == asin) & (lifecycle_segments["cycle_id"] == cycle_id)].copy()
            if not seg.empty and "date_end" in seg.columns and "date_start" in seg.columns:
                seg = seg.sort_values("date_end")
                last_seg = seg.iloc[-1].to_dict()
                if str(last_seg.get("phase", "")).strip():
                    phase = str(last_seg.get("phase", phase))
                ps = _to_date(last_seg.get("date_start"))
                if ps is not None:
                    phase_start = ps
        except Exception:
            phase_start = cycle_start

        # 关键锚点日期（用于“动态窗口”的解释性与可运营性）
        first_stock_date = _first_date_where(ts, (ts.get("FBA可售", 0.0) > 0) if "FBA可售" in ts.columns else pd.Series([False] * len(ts)))
        first_active_date = _first_date_where(ts, (ts.get("active", 0) == 1) if "active" in ts.columns else pd.Series([False] * len(ts)))
        first_ad_spend_date = _first_date_where(ts, (ts.get("广告花费", 0.0) > 0) if "广告花费" in ts.columns else pd.Series([False] * len(ts)))
        first_sale_date = _first_date_where(ts, ((ts.get("销售额", 0.0) > 0) | (ts.get("订单量", 0.0) > 0)) if ("销售额" in ts.columns and "订单量" in ts.columns) else pd.Series([False] * len(ts)))
        first_sale_in_stock_date = _first_date_where(
            ts,
            ((ts.get("FBA可售", 0.0) > 0) & ((ts.get("销售额", 0.0) > 0) | (ts.get("订单量", 0.0) > 0)))
            if ("FBA可售" in ts.columns and "销售额" in ts.columns and "订单量" in ts.columns)
            else pd.Series([False] * len(ts)),
        )
        last_sale_date = _last_date_where(ts, ((ts.get("销售额", 0.0) > 0) | (ts.get("订单量", 0.0) > 0)) if ("销售额" in ts.columns and "订单量" in ts.columns) else pd.Series([False] * len(ts)))

        peak_sales_roll_date: Optional[dt.date] = None
        try:
            if "sales_roll" in ts.columns:
                peak_idx = ts["sales_roll"].astype(float).fillna(0.0).idxmax()
                peak_sales_roll_date = _to_date(ts.loc[peak_idx, CAN.date]) if peak_idx is not None else None
        except Exception:
            peak_sales_roll_date = None

        cycle_days = int((cycle_end - cycle_start).days + 1)
        phase_days = int((cycle_end - phase_start).days + 1) if isinstance(phase_start, dt.date) else cycle_days

        # pre-launch 窗口：从“首次active/首次可售”到“首次出单前一天”的消耗与流量质量（你业务里很关键）
        prelaunch_days = 0
        prelaunch_ad_spend = 0.0
        prelaunch_sessions = 0.0
        prelaunch_ad_clicks = 0.0
        try:
            if "lifecycle_phase" in ts.columns:
                pl_ts = ts[ts["lifecycle_phase"].astype(str) == "pre_launch"].copy()
            else:
                pl_ts = pd.DataFrame()
            if pl_ts is not None and not pl_ts.empty:
                prelaunch_days = int(len(pl_ts))
                prelaunch_ad_spend = float(pd.to_numeric(pl_ts.get("广告花费", 0.0), errors="coerce").fillna(0.0).sum())
                prelaunch_sessions = float(pd.to_numeric(pl_ts.get("Sessions", 0.0), errors="coerce").fillna(0.0).sum())
                prelaunch_ad_clicks = float(pd.to_numeric(pl_ts.get("广告点击量", 0.0), errors="coerce").fillna(0.0).sum())
        except Exception:
            prelaunch_days = 0
            prelaunch_ad_spend = 0.0
            prelaunch_sessions = 0.0
            prelaunch_ad_clicks = 0.0

        # 关键“耗时”指标：库存->首单 / active->首单 / 广告花费->首单
        days_stock_to_sale: Optional[int] = None
        days_active_to_sale: Optional[int] = None
        days_ad_to_sale: Optional[int] = None
        flag_stock_after_sale = 0
        sale_for_stock = first_sale_in_stock_date if first_sale_in_stock_date is not None else first_sale_date
        if sale_for_stock is not None:
            if first_stock_date is not None:
                diff = int((sale_for_stock - first_stock_date).days)
                if diff < 0:
                    # 数据口径常见：FBA可售为0但仍有销售/订单；这里不让“耗时”出现负数，改用 flag 标记
                    flag_stock_after_sale = 1
                    diff = 0
                days_stock_to_sale = diff
            if first_active_date is not None:
                days_active_to_sale = max(0, int((sale_for_stock - first_active_date).days))
            if first_ad_spend_date is not None:
                days_ad_to_sale = max(0, int((sale_for_stock - first_ad_spend_date).days))

        # 1) cycle_to_date
        cycle_sum = _sum_window(ts)
        flag_ad_sales_gt_total = 1 if (cycle_sum.get("ad_sales", 0.0) > (cycle_sum.get("sales", 0.0) or 0.0) * 1.02 and (cycle_sum.get("sales", 0.0) or 0.0) > 0) else 0
        flag_ad_orders_gt_total = 1 if (cycle_sum.get("ad_orders", 0.0) > (cycle_sum.get("orders", 0.0) or 0.0) * 1.02 and (cycle_sum.get("orders", 0.0) or 0.0) > 0) else 0
        rows.append(
            {
                "shop": shop,
                "asin": asin,
                "cycle_id": cycle_id,
                "window_type": "cycle_to_date",
                "date_start": str(cycle_start),
                "date_end": str(cycle_end),
                "cycle_days": cycle_days,
                "phase_days": phase_days,
                "first_stock_date": str(first_stock_date) if first_stock_date else "",
                "first_active_date": str(first_active_date) if first_active_date else "",
                "first_ad_spend_date": str(first_ad_spend_date) if first_ad_spend_date else "",
                "first_sale_date": str(first_sale_date) if first_sale_date else "",
                "first_sale_in_stock_date": str(first_sale_in_stock_date) if first_sale_in_stock_date else "",
                "last_sale_date": str(last_sale_date) if last_sale_date else "",
                "peak_sales_roll_date": str(peak_sales_roll_date) if peak_sales_roll_date else "",
                "prelaunch_days": int(prelaunch_days),
                "prelaunch_ad_spend": round(float(prelaunch_ad_spend), 2),
                "prelaunch_sessions": round(float(prelaunch_sessions), 2),
                "prelaunch_ad_clicks": round(float(prelaunch_ad_clicks), 2),
                "days_stock_to_first_sale": days_stock_to_sale if days_stock_to_sale is not None else "",
                "days_active_to_first_sale": days_active_to_sale if days_active_to_sale is not None else "",
                "days_ad_to_first_sale": days_ad_to_sale if days_ad_to_sale is not None else "",
                "flag_stock_after_sale": int(flag_stock_after_sale),
                "flag_ad_sales_gt_total": int(flag_ad_sales_gt_total),
                "flag_ad_orders_gt_total": int(flag_ad_orders_gt_total),
                **{k: v for k, v in cycle_sum.items()},
            }
        )

        # 2) current_phase_to_date
        phase_ts = ts[(ts[CAN.date] >= phase_start) & (ts[CAN.date] <= cycle_end)].copy()
        phase_sum = _sum_window(phase_ts)
        flag_ad_sales_gt_total = 1 if (phase_sum.get("ad_sales", 0.0) > (phase_sum.get("sales", 0.0) or 0.0) * 1.02 and (phase_sum.get("sales", 0.0) or 0.0) > 0) else 0
        flag_ad_orders_gt_total = 1 if (phase_sum.get("ad_orders", 0.0) > (phase_sum.get("orders", 0.0) or 0.0) * 1.02 and (phase_sum.get("orders", 0.0) or 0.0) > 0) else 0
        rows.append(
            {
                "shop": shop,
                "asin": asin,
                "cycle_id": cycle_id,
                "window_type": "current_phase_to_date",
                "phase": phase,
                "date_start": str(phase_start),
                "date_end": str(cycle_end),
                "cycle_days": cycle_days,
                "phase_days": phase_days,
                "first_stock_date": str(first_stock_date) if first_stock_date else "",
                "first_active_date": str(first_active_date) if first_active_date else "",
                "first_ad_spend_date": str(first_ad_spend_date) if first_ad_spend_date else "",
                "first_sale_date": str(first_sale_date) if first_sale_date else "",
                "first_sale_in_stock_date": str(first_sale_in_stock_date) if first_sale_in_stock_date else "",
                "last_sale_date": str(last_sale_date) if last_sale_date else "",
                "peak_sales_roll_date": str(peak_sales_roll_date) if peak_sales_roll_date else "",
                "prelaunch_days": int(prelaunch_days),
                "prelaunch_ad_spend": round(float(prelaunch_ad_spend), 2),
                "prelaunch_sessions": round(float(prelaunch_sessions), 2),
                "prelaunch_ad_clicks": round(float(prelaunch_ad_clicks), 2),
                "days_stock_to_first_sale": days_stock_to_sale if days_stock_to_sale is not None else "",
                "days_active_to_first_sale": days_active_to_sale if days_active_to_sale is not None else "",
                "days_ad_to_first_sale": days_ad_to_sale if days_ad_to_sale is not None else "",
                "flag_stock_after_sale": int(flag_stock_after_sale),
                "flag_ad_sales_gt_total": int(flag_ad_sales_gt_total),
                "flag_ad_orders_gt_total": int(flag_ad_orders_gt_total),
                **{k: v for k, v in phase_sum.items()},
            }
        )

        # 2.1) since_first_stock（更贴近“上架可售后处于什么阶段”的运营口径）
        if first_stock_date is not None:
            stock_ts = ts[(ts[CAN.date] >= first_stock_date) & (ts[CAN.date] <= cycle_end)].copy()
            stock_sum = _sum_window(stock_ts)
            flag_ad_sales_gt_total = 1 if (stock_sum.get("ad_sales", 0.0) > (stock_sum.get("sales", 0.0) or 0.0) * 1.02 and (stock_sum.get("sales", 0.0) or 0.0) > 0) else 0
            flag_ad_orders_gt_total = 1 if (stock_sum.get("ad_orders", 0.0) > (stock_sum.get("orders", 0.0) or 0.0) * 1.02 and (stock_sum.get("orders", 0.0) or 0.0) > 0) else 0
            rows.append(
                {
                    "shop": shop,
                    "asin": asin,
                    "cycle_id": cycle_id,
                    "window_type": "since_first_stock_to_date",
                    "phase": phase,
                    "date_start": str(first_stock_date),
                    "date_end": str(cycle_end),
                    "cycle_days": cycle_days,
                    "phase_days": phase_days,
                    "first_stock_date": str(first_stock_date),
                    "first_active_date": str(first_active_date) if first_active_date else "",
                    "first_ad_spend_date": str(first_ad_spend_date) if first_ad_spend_date else "",
                    "first_sale_date": str(first_sale_date) if first_sale_date else "",
                    "first_sale_in_stock_date": str(first_sale_in_stock_date) if first_sale_in_stock_date else "",
                    "last_sale_date": str(last_sale_date) if last_sale_date else "",
                    "peak_sales_roll_date": str(peak_sales_roll_date) if peak_sales_roll_date else "",
                    "prelaunch_days": int(prelaunch_days),
                    "prelaunch_ad_spend": round(float(prelaunch_ad_spend), 2),
                    "prelaunch_sessions": round(float(prelaunch_sessions), 2),
                    "prelaunch_ad_clicks": round(float(prelaunch_ad_clicks), 2),
                    "days_stock_to_first_sale": days_stock_to_sale if days_stock_to_sale is not None else "",
                    "days_active_to_first_sale": days_active_to_sale if days_active_to_sale is not None else "",
                    "days_ad_to_first_sale": days_ad_to_sale if days_ad_to_sale is not None else "",
                    "flag_stock_after_sale": int(flag_stock_after_sale),
                    "flag_ad_sales_gt_total": int(flag_ad_sales_gt_total),
                    "flag_ad_orders_gt_total": int(flag_ad_orders_gt_total),
                    **{k: v for k, v in stock_sum.items()},
                }
            )

        # 2.2) since_first_sale（新品期/成长期经常以“首次出单”作为起点）
        if first_sale_date is not None:
            sale_ts = ts[(ts[CAN.date] >= first_sale_date) & (ts[CAN.date] <= cycle_end)].copy()
            sale_sum = _sum_window(sale_ts)
            flag_ad_sales_gt_total = 1 if (sale_sum.get("ad_sales", 0.0) > (sale_sum.get("sales", 0.0) or 0.0) * 1.02 and (sale_sum.get("sales", 0.0) or 0.0) > 0) else 0
            flag_ad_orders_gt_total = 1 if (sale_sum.get("ad_orders", 0.0) > (sale_sum.get("orders", 0.0) or 0.0) * 1.02 and (sale_sum.get("orders", 0.0) or 0.0) > 0) else 0
            rows.append(
                {
                    "shop": shop,
                    "asin": asin,
                    "cycle_id": cycle_id,
                    "window_type": "since_first_sale_to_date",
                    "phase": phase,
                    "date_start": str(first_sale_date),
                    "date_end": str(cycle_end),
                    "cycle_days": cycle_days,
                    "phase_days": phase_days,
                    "first_stock_date": str(first_stock_date) if first_stock_date else "",
                    "first_active_date": str(first_active_date) if first_active_date else "",
                    "first_ad_spend_date": str(first_ad_spend_date) if first_ad_spend_date else "",
                    "first_sale_date": str(first_sale_date),
                    "first_sale_in_stock_date": str(first_sale_in_stock_date) if first_sale_in_stock_date else "",
                    "last_sale_date": str(last_sale_date) if last_sale_date else "",
                    "peak_sales_roll_date": str(peak_sales_roll_date) if peak_sales_roll_date else "",
                    "prelaunch_days": int(prelaunch_days),
                    "prelaunch_ad_spend": round(float(prelaunch_ad_spend), 2),
                    "prelaunch_sessions": round(float(prelaunch_sessions), 2),
                    "prelaunch_ad_clicks": round(float(prelaunch_ad_clicks), 2),
                    "days_stock_to_first_sale": days_stock_to_sale if days_stock_to_sale is not None else "",
                    "days_active_to_first_sale": days_active_to_sale if days_active_to_sale is not None else "",
                    "days_ad_to_first_sale": days_ad_to_sale if days_ad_to_sale is not None else "",
                    "flag_stock_after_sale": int(flag_stock_after_sale),
                    "flag_ad_sales_gt_total": int(flag_ad_sales_gt_total),
                    "flag_ad_orders_gt_total": int(flag_ad_orders_gt_total),
                    **{k: v for k, v in sale_sum.items()},
                }
            )

        # 3) compare windows
        for n in windows_days:
            comp = _compare_recent_prev(ts, cycle_end, int(n))
            if not comp:
                continue
            # compare 窗口也带上一个一致的“口径异常”标记（方便 AI 解释）
            try:
                ad_sales_recent = float(comp.get("ad_sales_recent", 0.0) or 0.0)  # 兼容未来扩展
                sales_recent = float(comp.get("sales_recent", 0.0) or 0.0)
            except Exception:
                ad_sales_recent = 0.0
                sales_recent = float(comp.get("sales_recent", 0.0) or 0.0)
            flag_ad_sales_gt_total = 1 if (ad_sales_recent > sales_recent * 1.02 and sales_recent > 0) else 0
            rows.append(
                {
                    "shop": shop,
                    "asin": asin,
                    "cycle_id": cycle_id,
                    "window_type": f"compare_{int(n)}d",
                    "phase": phase,
                    "cycle_days": cycle_days,
                    "phase_days": phase_days,
                    "first_stock_date": str(first_stock_date) if first_stock_date else "",
                    "first_active_date": str(first_active_date) if first_active_date else "",
                    "first_ad_spend_date": str(first_ad_spend_date) if first_ad_spend_date else "",
                    "first_sale_date": str(first_sale_date) if first_sale_date else "",
                    "first_sale_in_stock_date": str(first_sale_in_stock_date) if first_sale_in_stock_date else "",
                    "last_sale_date": str(last_sale_date) if last_sale_date else "",
                    "peak_sales_roll_date": str(peak_sales_roll_date) if peak_sales_roll_date else "",
                    "prelaunch_days": int(prelaunch_days),
                    "prelaunch_ad_spend": round(float(prelaunch_ad_spend), 2),
                    "prelaunch_sessions": round(float(prelaunch_sessions), 2),
                    "prelaunch_ad_clicks": round(float(prelaunch_ad_clicks), 2),
                    "days_stock_to_first_sale": days_stock_to_sale if days_stock_to_sale is not None else "",
                    "days_active_to_first_sale": days_active_to_sale if days_active_to_sale is not None else "",
                    "days_ad_to_first_sale": days_ad_to_sale if days_ad_to_sale is not None else "",
                    "flag_stock_after_sale": int(flag_stock_after_sale),
                    "flag_ad_sales_gt_total": int(flag_ad_sales_gt_total),
                    "flag_ad_orders_gt_total": 0,
                    **comp,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("sales", "ad_spend", "ad_sales", "profit", "delta_spend", "delta_sales"):
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
            except Exception:
                pass
    for c in ("tacos", "ad_acos", "cvr", "marginal_tacos", "marginal_ad_acos"):
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(4)
            except Exception:
                pass
    for c in ("cvr_prev", "cvr_recent", "delta_cvr"):
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(6)
            except Exception:
                pass
    for c in ("organic_sales_prev", "organic_sales_recent", "delta_organic_sales"):
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
            except Exception:
                pass
    for c in ("organic_sales_share_prev", "organic_sales_share_recent", "delta_organic_sales_share"):
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(6)
            except Exception:
                pass
    for c in ("ad_ctr", "ad_cvr", "ad_orders_share", "ad_sales_share", "organic_sales_share"):
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(6)
            except Exception:
                pass
    return df
