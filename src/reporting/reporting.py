# -*- coding: utf-8 -*-
"""
合成版：复用老项目的“分析流程/可视化/报告输出”思路，
但数据源改成你现在的 `reports/`（赛狐导出，字段更全；也可通过 `--input-dir` 指定）。

输出目标（每店铺一份）：
- figures/*.png（趋势/结构/矩阵图）
- ai/report.md（全量深挖版：指标罗列与图表，主要给 AI/分析用）
- reports/dashboard.md（聚焦版：主要给运营用，见 dashboard/ 模块）
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import matplotlib

matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns
import warnings
import glob

from ads.actions import ActionCandidate
from analysis.temporal import build_windows, window_compare, _infer_shop_end_date
from core.config import StageConfig
from core.md import df_to_md_table
from core.metrics import summarize
from core.policy import OpsPolicy, load_ops_policy
from core.schema import CAN
from core.utils import safe_div


def _ensure_dirs(shop_dir: Path) -> Tuple[Path, Path, Path]:
    figures = shop_dir / "figures"
    reports = shop_dir / "reports"
    ai_dir = shop_dir / "ai"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)
    return figures, reports, ai_dir


_CN_STYLE_READY: bool = False
_CN_FONT_CHOSEN: Optional[str] = None
_CN_FONT_PATH: Optional[str] = None


def _discover_macos_cn_font_files() -> List[Path]:
    """
    在 macOS 上尽可能“稳健”地找到可用的中文字体文件。

    说明：
    - 很多机器上 PingFang 并不在 /System/Library/Fonts 里，而是在 /System/Library/AssetsV2 的字体资产中。
    - matplotlib 有时扫描不到系统字体，需要手动 addfont() 才能用来画图。
    """
    fixed = [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ]

    # PingFang 资产路径（会带 hash 子目录），用 glob 兜底找一遍
    patterns = [
        # 优先：只扫字体资产目录，速度更快
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font*/**/AssetData/PingFang.ttc",
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font*/**/AssetData/PingFangHK.ttc",
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font*/**/AssetData/PingFangSC.ttc",
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font*/**/AssetData/PingFangTC.ttc",
        # 兜底：个别系统路径不一致时再放开范围
        "/System/Library/AssetsV2/**/PingFang.ttc",
    ]

    out: List[Path] = []
    for p in fixed:
        pp = Path(p)
        if pp.exists():
            out.append(pp)

    try:
        for pat in patterns:
            for hit in glob.glob(pat, recursive=True):
                hp = Path(hit)
                if hp.exists():
                    out.append(hp)
    except Exception:
        # glob 失败也不要影响主流程
        pass

    # 去重（保持顺序）
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in out:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def _try_add_font(path: Path) -> bool:
    try:
        font_manager.fontManager.addfont(str(path))
        return True
    except Exception:
        return False


def _set_cn_style() -> None:
    """
    matplotlib 中文字体设置（macOS 优先）。

    目标：
    - 优先用 PingFang（你之前用 PingFang HK 能显示中文）
    - 如果系统/环境扫描不到 PingFang，就尝试手动加载字体文件
    - 如果仍不可用，就降级到 Hiragino Sans GB / STHeiti / Arial Unicode MS
    """
    global _CN_STYLE_READY, _CN_FONT_CHOSEN, _CN_FONT_PATH

    if _CN_STYLE_READY:
        return

    # 注意：seaborn.set_style 会覆盖 matplotlib 的 font.sans-serif 等 rcParams。
    # 所以必须先 set_style，再设置字体，否则会出现中文变成方块。
    sns.set_style("whitegrid")

    # 先尝试“名字直配”，再尝试 addfont()，最后再决定 chosen
    preferred_names = [
        "PingFang HK",
        "PingFang SC",
        "PingFang TC",
        "Hiragino Sans GB",
        "STHeiti",
        "Heiti SC",
        "Heiti TC",
        "Arial Unicode MS",
    ]

    chosen_path: Optional[str] = None

    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    # 如果 PingFang 系列不在可用字体列表里，先尝试从系统字体资产里“手动加载”一次，再选字体。
    pingfang_names = {"PingFang HK", "PingFang SC", "PingFang TC"}
    if not (available & pingfang_names):
        for font_path in _discover_macos_cn_font_files():
            ok = _try_add_font(font_path)
            if ok:
                # 优先记录 PingFang 来源（便于你验收）
                if "PingFang" in font_path.name:
                    chosen_path = str(font_path)
                elif chosen_path is None:
                    chosen_path = str(font_path)
        try:
            available = {f.name for f in font_manager.fontManager.ttflist}
        except Exception:
            available = set()

    chosen = next((n for n in preferred_names if n in available), None)

    # 仍没选到时，再做一次加载兜底（避免上面因为异常中断导致没加载成功）
    if not chosen:
        for font_path in _discover_macos_cn_font_files():
            ok = _try_add_font(font_path)
            if ok:
                if "PingFang" in font_path.name:
                    chosen_path = str(font_path)
                elif chosen_path is None:
                    chosen_path = str(font_path)
        try:
            available = {f.name for f in font_manager.fontManager.ttflist}
        except Exception:
            available = set()
        chosen = next((n for n in preferred_names if n in available), None)

    # 兜底：不管 chosen 有没有，给一个稳定的候选列表
    fallbacks = [n for n in preferred_names if n != chosen]
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + fallbacks
    else:
        plt.rcParams["font.sans-serif"] = fallbacks
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    # 不同 matplotlib 版本的提示文案略有差异，这里都忽略，避免运行输出被刷屏
    warnings.filterwarnings("ignore", message=r"Glyph .* missing from font.*")
    warnings.filterwarnings("ignore", message=r"Glyph .* missing from current font.*")

    _CN_STYLE_READY = True
    _CN_FONT_CHOSEN = chosen
    _CN_FONT_PATH = chosen_path


def _cn_font_info() -> str:
    """
    用于写进报告，方便你验收“图表中文是否可用、用的是什么字体”。
    """
    if _CN_FONT_CHOSEN and _CN_FONT_PATH:
        return f"{_CN_FONT_CHOSEN}（from {Path(_CN_FONT_PATH).name}）"
    if _CN_FONT_CHOSEN:
        return _CN_FONT_CHOSEN
    return "未识别（已设置降级字体列表）"


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _date_range(df: pd.DataFrame) -> Tuple[str, str]:
    if df is None or df.empty or CAN.date not in df.columns:
        return "", ""
    dmin = df[CAN.date].min()
    dmax = df[CAN.date].max()
    return (str(dmin) if dmin else "", str(dmax) if dmax else "")


def _ts_from_campaign(camp: pd.DataFrame) -> pd.DataFrame:
    if camp.empty or CAN.date not in camp.columns:
        return pd.DataFrame()
    ts = (
        camp.groupby(CAN.date, as_index=True)
        .agg(
            impressions=(CAN.impressions, "sum"),
            clicks=(CAN.clicks, "sum"),
            spend=(CAN.spend, "sum"),
            sales=(CAN.sales, "sum"),
            orders=(CAN.orders, "sum"),
        )
        .sort_index()
    )
    ts["acos"] = ts.apply(lambda r: safe_div(r["spend"], r["sales"]), axis=1)
    ts["roas"] = ts.apply(lambda r: safe_div(r["sales"], r["spend"]), axis=1)
    ts["ctr"] = ts.apply(lambda r: safe_div(r["clicks"], r["impressions"]), axis=1)
    ts["cvr"] = ts.apply(lambda r: safe_div(r["orders"], r["clicks"]), axis=1)
    return ts


def _plot_trends(ts: pd.DataFrame, title_prefix: str, out_path: Path) -> Optional[Path]:
    if ts.empty:
        return None
    _set_cn_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    ts[["spend", "sales"]].plot(ax=axes[0])
    axes[0].set_title(f"{title_prefix} 花费 vs 销售额")
    axes[0].set_xlabel("日期")

    ts[["orders", "clicks"]].plot(ax=axes[1])
    axes[1].set_title(f"{title_prefix} 订单量 vs 点击量")
    axes[1].set_xlabel("日期")

    ts[["acos", "roas"]].plot(ax=axes[2])
    axes[2].set_title(f"{title_prefix} ACoS / ROAS")
    axes[2].set_xlabel("日期")

    ts[["ctr", "cvr"]].plot(ax=axes[3])
    axes[3].set_title(f"{title_prefix} CTR / CVR")
    axes[3].set_xlabel("日期")

    _save_fig(fig, out_path)
    return out_path


def _plot_cn_font_smoke_test(out_path: Path) -> Optional[Path]:
    """
    中文字体自检图（用于你验收“中文是否显示正常”）。
    - 目的：避免因为 seaborn/matplotlib 配置覆盖导致中文变成方块。
    - 内容：包含常见中文字符与符号。
    """
    try:
        _set_cn_style()
        fig, ax = plt.subplots(figsize=(9, 2.4))
        ax.plot([1, 2, 3], [1, 4, 2], linewidth=2)
        ax.set_title("中文字体自检：商品分类（- 负号测试）")
        ax.set_xlabel("日期")
        ax.set_ylabel("花费")
        _save_fig(fig, out_path)
        return out_path
    except Exception:
        return None


def _plot_top_bar(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, top_n: int = 15) -> Optional[Path]:
    if df.empty or x not in df.columns or y not in df.columns:
        return None
    _set_cn_style()
    view = df.sort_values(y, ascending=False).head(top_n).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    # seaborn 0.13+：palette 无 hue 会产生 FutureWarning，这里用单色即可
    sns.barplot(data=view, x=y, y=x, ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(y)
    ax.set_ylabel(x)
    _save_fig(fig, out_path)
    return out_path


def _plot_matrix_search_terms(st: pd.DataFrame, out_path: Path, cfg: StageConfig) -> Optional[Path]:
    """
    用搜索词做一个“矩阵图”：
    - x: clicks
    - y: acos
    - size: spend
    - hue: cvr
    """
    need = [CAN.search_term, CAN.clicks, CAN.spend, CAN.sales, CAN.orders]
    if st.empty or any(c not in st.columns for c in need):
        return None

    _set_cn_style()
    gcols = [CAN.search_term]
    agg = (
        st.groupby(gcols, dropna=False, as_index=False)
        .agg(
            clicks=(CAN.clicks, "sum"),
            spend=(CAN.spend, "sum"),
            sales=(CAN.sales, "sum"),
            orders=(CAN.orders, "sum"),
        )
        .copy()
    )
    agg["acos"] = agg.apply(lambda r: safe_div(r["spend"], r["sales"]), axis=1)
    agg["cvr"] = agg.apply(lambda r: safe_div(r["orders"], r["clicks"]), axis=1)
    # 过滤极小样本，避免图太乱
    view = agg[(agg["clicks"] >= max(cfg.min_clicks, 10)) & (agg["spend"] > 0)].copy()
    if view.empty:
        return None
    view = view.sort_values("spend", ascending=False).head(300)

    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(
        view["clicks"],
        view["acos"],
        s=(view["spend"].clip(0, view["spend"].quantile(0.95)) + 1) * 10,
        c=view["cvr"],
        cmap="RdYlGn",
        alpha=0.6,
        edgecolors="none",
    )
    ax.axhline(cfg.target_acos, color="red", linestyle="--", linewidth=1, label=f"目标ACoS={cfg.target_acos:.0%}")
    ax.set_title("搜索词矩阵：Clicks vs ACoS（点越大花费越高，颜色越绿CVR越高）")
    ax.set_xlabel("Clicks")
    ax.set_ylabel("ACoS")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="CVR")
    _save_fig(fig, out_path)
    return out_path


def _plot_phase_counts(board: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """
    生命周期阶段分布（当前）：ASIN 数量。
    """
    if board is None or board.empty or "current_phase" not in board.columns:
        return None
    _set_cn_style()
    view = (
        board.groupby("current_phase", dropna=False)
        .size()
        .reset_index(name="asin_count")
        .sort_values("asin_count", ascending=False)
        .copy()
    )
    if view.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=view, x="current_phase", y="asin_count", ax=ax, color="#55A868")
    ax.set_title("生命周期阶段分布（当前）：ASIN 数量")
    ax.set_xlabel("阶段")
    ax.set_ylabel("ASIN 数量")
    _save_fig(fig, out_path)
    return out_path


def _plot_phase_spend_share(board: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """
    生命周期阶段分布（当前）：7天滚动广告花费占比。
    """
    if board is None or board.empty or "current_phase" not in board.columns or "ad_spend_roll" not in board.columns:
        return None
    _set_cn_style()
    b = board.copy()
    b["ad_spend_roll"] = pd.to_numeric(b["ad_spend_roll"], errors="coerce").fillna(0.0)
    total = float(b["ad_spend_roll"].sum())
    if total <= 0:
        return None
    view = (
        b.groupby("current_phase", dropna=False)["ad_spend_roll"]
        .sum()
        .reset_index(name="ad_spend_roll_sum")
        .sort_values("ad_spend_roll_sum", ascending=False)
        .copy()
    )
    view["share"] = view["ad_spend_roll_sum"].apply(lambda x: safe_div(float(x), total))
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=view, x="current_phase", y="share", ax=ax, color="#4C72B0")
    ax.set_title("生命周期阶段分布（当前）：7天滚动广告花费占比")
    ax.set_xlabel("阶段")
    ax.set_ylabel("花费占比")
    ax.set_ylim(0, min(1.0, max(0.35, float(view["share"].max()) * 1.2)))
    _save_fig(fig, out_path)
    return out_path


def _plot_asin_spend_concentration(board: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """
    花费集中度（7天滚动广告花费）：按 ASIN 排序的累计占比曲线。
    """
    if board is None or board.empty or "asin" not in board.columns or "ad_spend_roll" not in board.columns:
        return None
    _set_cn_style()
    b = board.copy()
    b["ad_spend_roll"] = pd.to_numeric(b["ad_spend_roll"], errors="coerce").fillna(0.0)
    b = b[b["ad_spend_roll"] > 0].copy()
    if b.empty:
        return None
    b = b.sort_values("ad_spend_roll", ascending=False)
    total = float(b["ad_spend_roll"].sum())
    if total <= 0:
        return None
    b["cum_share"] = b["ad_spend_roll"].cumsum().apply(lambda x: safe_div(float(x), total))
    b["rank"] = range(1, len(b) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(b["rank"], b["cum_share"], linewidth=2, color="#C44E52")
    ax.set_title("花费集中度（7天滚动广告花费）：Top N ASIN 累计占比")
    ax.set_xlabel("Top N（按广告花费排序）")
    ax.set_ylabel("累计花费占比")
    ax.set_ylim(0, 1.0)
    _save_fig(fig, out_path)
    return out_path


def _short_label(asin: str, name: str, max_len: int = 18) -> str:
    a = str(asin or "").strip()
    n = str(name or "").strip()
    if not n:
        return a
    n2 = n[:max_len] + ("…" if len(n) > max_len else "")
    return f"{a} {n2}"


def _category_summary(product_listing_shop: pd.DataFrame, product_analysis_shop: pd.DataFrame) -> pd.DataFrame:
    """
    店铺维度：按“商品分类”汇总（横向对比同类产品）。

    数据来源：
    - 分类：productListing.xlsx 的“商品分类”
    - 指标：产品分析（按日）汇总到自然月口径
    """
    if product_listing_shop is None or product_listing_shop.empty:
        return pd.DataFrame()
    if product_analysis_shop is None or product_analysis_shop.empty:
        return pd.DataFrame()
    if "ASIN" not in product_listing_shop.columns or "商品分类" not in product_listing_shop.columns:
        return pd.DataFrame()
    if "ASIN" not in product_analysis_shop.columns:
        return pd.DataFrame()

    try:
        pl = product_listing_shop.copy()
        pl["asin_norm"] = pl["ASIN"].astype(str).str.upper().str.strip()
        pl["product_category"] = pl["商品分类"].astype(str).fillna("").str.strip()
        pl.loc[pl["product_category"].str.lower() == "nan", "product_category"] = ""
        pl = pl[["asin_norm", "product_category"]].drop_duplicates("asin_norm")

        pa = product_analysis_shop.copy()
        pa["asin_norm"] = pa["ASIN"].astype(str).str.upper().str.strip()
        metrics_cols = []
        for col in ("销售额", "订单量", "Sessions", "广告花费", "广告销售额", "广告订单量", "毛利润"):
            if col in pa.columns:
                metrics_cols.append(col)
        if not metrics_cols:
            return pd.DataFrame()
        agg_map = {c: "sum" for c in metrics_cols}
        pa_asin = pa.groupby("asin_norm", dropna=False, as_index=False).agg(agg_map).copy()

        merged = pa_asin.merge(pl, on="asin_norm", how="left")
        merged["product_category"] = merged["product_category"].astype(str).fillna("").str.strip()
        merged.loc[merged["product_category"].str.lower() == "nan", "product_category"] = ""
        merged.loc[merged["product_category"] == "", "product_category"] = "未分类"

        g = merged.groupby("product_category", dropna=False, as_index=False).agg(
            asin_count=("asin_norm", "nunique"),
            sales_total=("销售额", "sum") if "销售额" in merged.columns else ("asin_norm", "size"),
            orders_total=("订单量", "sum") if "订单量" in merged.columns else ("asin_norm", "size"),
            sessions_total=("Sessions", "sum") if "Sessions" in merged.columns else ("asin_norm", "size"),
            ad_spend_total=("广告花费", "sum") if "广告花费" in merged.columns else ("asin_norm", "size"),
            ad_sales_total=("广告销售额", "sum") if "广告销售额" in merged.columns else ("asin_norm", "size"),
            profit_total=("毛利润", "sum") if "毛利润" in merged.columns else ("asin_norm", "size"),
        )
        # 经营口径：TACOS/CVR/广告ACoS
        g["tacos_total"] = g.apply(lambda r: safe_div(r.get("ad_spend_total", 0.0), r.get("sales_total", 0.0)), axis=1)
        g["cvr_total"] = g.apply(lambda r: safe_div(r.get("orders_total", 0.0), r.get("sessions_total", 0.0)), axis=1)
        g["ad_acos_total"] = g.apply(lambda r: safe_div(r.get("ad_spend_total", 0.0), r.get("ad_sales_total", 0.0)), axis=1)

        for c in ("sales_total", "ad_spend_total", "ad_sales_total", "profit_total"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(2)
        for c in ("tacos_total", "cvr_total", "ad_acos_total"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(4)

        # 默认按销售额排序（也方便你看“哪个类是主力”）
        if "sales_total" in g.columns:
            g = g.sort_values("sales_total", ascending=False)
        return g
    except Exception:
        return pd.DataFrame()


def _shop_monthly_biz_dashboard(product_analysis_shop: pd.DataFrame) -> pd.DataFrame:
    """
    店铺大盘（月度）：主口径=产品分析（自然+广告合计）。
    """
    if product_analysis_shop is None or product_analysis_shop.empty:
        return pd.DataFrame()
    if CAN.date not in product_analysis_shop.columns:
        return pd.DataFrame()
    try:
        pa = product_analysis_shop.copy()
        pa[CAN.date] = pd.to_datetime(pa[CAN.date], errors="coerce")
        pa = pa[pa[CAN.date].notna()].copy()
        if pa.empty:
            return pd.DataFrame()
        pa["month"] = pa[CAN.date].dt.to_period("M").astype(str)

        # 产品分析常用列（按你们赛狐表头）
        cols = {
            "sales_total": "销售额",
            "orders_total": "订单量",
            "sessions_total": "Sessions",
            "profit_total": "毛利润",
            "ad_spend_total": "广告花费",
            "ad_sales_total": "广告销售额",
            "ad_orders_total": "广告订单量",
        }
        agg_map = {v: "sum" for v in cols.values() if v in pa.columns}
        if not agg_map:
            return pd.DataFrame()

        g = pa.groupby("month", dropna=False, as_index=False).agg(agg_map).copy()
        # 统一列名
        rename = {v: k for k, v in cols.items() if v in g.columns}
        g = g.rename(columns=rename)

        # 衍生指标
        if "sales_total" in g.columns and "ad_spend_total" in g.columns:
            g["tacos_total"] = g.apply(lambda r: safe_div(r.get("ad_spend_total", 0.0), r.get("sales_total", 0.0)), axis=1)
        if "orders_total" in g.columns and "sessions_total" in g.columns:
            g["cvr_total"] = g.apply(lambda r: safe_div(r.get("orders_total", 0.0), r.get("sessions_total", 0.0)), axis=1)
        if "sales_total" in g.columns and "ad_sales_total" in g.columns:
            g["ad_sales_share_total"] = g.apply(lambda r: safe_div(r.get("ad_sales_total", 0.0), r.get("sales_total", 0.0)), axis=1)

        # 排序 + 四舍五入
        g = g.sort_values("month")
        for c in ("sales_total", "profit_total", "ad_spend_total", "ad_sales_total"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(2)
        for c in ("tacos_total", "cvr_total", "ad_sales_share_total"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(4)

        # 总计行
        total = {}
        total["month"] = "TOTAL"
        for c in ("sales_total", "orders_total", "sessions_total", "profit_total", "ad_spend_total", "ad_sales_total", "ad_orders_total"):
            if c in g.columns:
                total[c] = float(pd.to_numeric(g[c], errors="coerce").fillna(0.0).sum())
        if "sales_total" in total and "ad_spend_total" in total:
            total["tacos_total"] = safe_div(total.get("ad_spend_total", 0.0), total.get("sales_total", 0.0))
        if "orders_total" in total and "sessions_total" in total:
            total["cvr_total"] = safe_div(total.get("orders_total", 0.0), total.get("sessions_total", 0.0))
        if "sales_total" in total and "ad_sales_total" in total:
            total["ad_sales_share_total"] = safe_div(total.get("ad_sales_total", 0.0), total.get("sales_total", 0.0))

        g2 = pd.concat([g, pd.DataFrame([total])], ignore_index=True)
        # 再做一次格式化
        for c in ("sales_total", "profit_total", "ad_spend_total", "ad_sales_total"):
            if c in g2.columns:
                g2[c] = pd.to_numeric(g2[c], errors="coerce").round(2)
        for c in ("tacos_total", "cvr_total", "ad_sales_share_total"):
            if c in g2.columns:
                g2[c] = pd.to_numeric(g2[c], errors="coerce").round(4)
        return g2
    except Exception:
        return pd.DataFrame()


def _shop_monthly_ad_dashboard(camp: pd.DataFrame) -> pd.DataFrame:
    """
    店铺大盘（月度）：广告口径=广告活动报告（campaign）。
    """
    if camp is None or camp.empty:
        return pd.DataFrame()
    if CAN.date not in camp.columns:
        return pd.DataFrame()
    try:
        df = camp.copy()
        df[CAN.date] = pd.to_datetime(df[CAN.date], errors="coerce")
        df = df[df[CAN.date].notna()].copy()
        if df.empty:
            return pd.DataFrame()
        df["month"] = df[CAN.date].dt.to_period("M").astype(str)

        g = (
            df.groupby("month", dropna=False, as_index=False)
            .agg(
                impressions=(CAN.impressions, "sum") if CAN.impressions in df.columns else ("month", "size"),
                clicks=(CAN.clicks, "sum") if CAN.clicks in df.columns else ("month", "size"),
                spend=(CAN.spend, "sum") if CAN.spend in df.columns else ("month", "size"),
                sales=(CAN.sales, "sum") if CAN.sales in df.columns else ("month", "size"),
                orders=(CAN.orders, "sum") if CAN.orders in df.columns else ("month", "size"),
            )
            .copy()
        )
        g = _add_derived_kpis(g)
        g = g.sort_values("month")

        for c in ("spend", "sales"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(2)
        for c in ("acos", "ctr", "cvr"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(4)
        if "cpc" in g.columns:
            g["cpc"] = pd.to_numeric(g["cpc"], errors="coerce").round(2)

        total = {"month": "TOTAL"}
        for c in ("impressions", "clicks", "spend", "sales", "orders"):
            if c in g.columns:
                total[c] = float(pd.to_numeric(g[c], errors="coerce").fillna(0.0).sum())
        total["ctr"] = safe_div(total.get("clicks", 0.0), total.get("impressions", 0.0))
        total["cpc"] = safe_div(total.get("spend", 0.0), total.get("clicks", 0.0))
        total["cvr"] = safe_div(total.get("orders", 0.0), total.get("clicks", 0.0))
        total["acos"] = safe_div(total.get("spend", 0.0), total.get("sales", 0.0))

        g2 = pd.concat([g, pd.DataFrame([total])], ignore_index=True)
        for c in ("spend", "sales"):
            if c in g2.columns:
                g2[c] = pd.to_numeric(g2[c], errors="coerce").round(2)
        for c in ("acos", "ctr", "cvr"):
            if c in g2.columns:
                g2[c] = pd.to_numeric(g2[c], errors="coerce").round(4)
        if "cpc" in g2.columns:
            g2["cpc"] = pd.to_numeric(g2["cpc"], errors="coerce").round(2)
        return g2
    except Exception:
        return pd.DataFrame()


def _shop_monthly_category_dashboard(product_listing_shop: pd.DataFrame, product_analysis_shop: pd.DataFrame) -> pd.DataFrame:
    """
    店铺大盘（月度 × 商品分类）：主口径=产品分析（自然+广告合计）。
    """
    if product_listing_shop is None or product_listing_shop.empty:
        return pd.DataFrame()
    if product_analysis_shop is None or product_analysis_shop.empty:
        return pd.DataFrame()
    if CAN.date not in product_analysis_shop.columns or "ASIN" not in product_analysis_shop.columns:
        return pd.DataFrame()
    if "ASIN" not in product_listing_shop.columns or "商品分类" not in product_listing_shop.columns:
        return pd.DataFrame()

    try:
        pl = product_listing_shop.copy()
        pl["asin_norm"] = pl["ASIN"].astype(str).str.upper().str.strip()
        pl["product_category"] = pl["商品分类"].astype(str).fillna("").str.strip()
        pl.loc[pl["product_category"].str.lower() == "nan", "product_category"] = ""
        pl.loc[pl["product_category"] == "", "product_category"] = "未分类"
        pl = pl[["asin_norm", "product_category"]].drop_duplicates("asin_norm")

        pa = product_analysis_shop.copy()
        pa[CAN.date] = pd.to_datetime(pa[CAN.date], errors="coerce")
        pa = pa[pa[CAN.date].notna()].copy()
        if pa.empty:
            return pd.DataFrame()
        pa["asin_norm"] = pa["ASIN"].astype(str).str.upper().str.strip()
        pa["month"] = pa[CAN.date].dt.to_period("M").astype(str)
        pa = pa.merge(pl, on="asin_norm", how="left")
        pa["product_category"] = pa["product_category"].fillna("").astype(str).str.strip()
        pa.loc[pa["product_category"].str.lower() == "nan", "product_category"] = ""
        pa.loc[pa["product_category"] == "", "product_category"] = "未分类"

        metrics = {
            "sales_total": "销售额",
            "orders_total": "订单量",
            "sessions_total": "Sessions",
            "profit_total": "毛利润",
            "ad_spend_total": "广告花费",
            "ad_sales_total": "广告销售额",
        }
        agg_map = {v: "sum" for v in metrics.values() if v in pa.columns}
        if not agg_map:
            return pd.DataFrame()

        g = (
            pa.groupby(["month", "product_category"], dropna=False, as_index=False)
            .agg({**agg_map, "asin_norm": "nunique"})
            .rename(columns={"asin_norm": "asin_count"})
            .copy()
        )

        rename = {v: k for k, v in metrics.items() if v in g.columns}
        g = g.rename(columns=rename)
        if "sales_total" in g.columns and "ad_spend_total" in g.columns:
            g["tacos_total"] = g.apply(lambda r: safe_div(r.get("ad_spend_total", 0.0), r.get("sales_total", 0.0)), axis=1)
        if "ad_spend_total" in g.columns and "ad_sales_total" in g.columns:
            g["ad_acos_total"] = g.apply(lambda r: safe_div(r.get("ad_spend_total", 0.0), r.get("ad_sales_total", 0.0)), axis=1)

        for c in ("sales_total", "profit_total", "ad_spend_total", "ad_sales_total"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(2)
        for c in ("tacos_total", "ad_acos_total"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(4)

        g = g.sort_values(["month", "sales_total"], ascending=[True, False]) if "sales_total" in g.columns else g.sort_values(["month", "asin_count"], ascending=[True, False])

        # 每个月增加 TOTAL 行（分类汇总）
        out_rows = []
        for month, gg in g.groupby("month", dropna=False):
            out_rows.append(gg)
            tot = {"month": month, "product_category": "TOTAL"}
            tot["asin_count"] = int(pd.to_numeric(gg["asin_count"], errors="coerce").fillna(0).sum()) if "asin_count" in gg.columns else 0
            for c in ("sales_total", "orders_total", "sessions_total", "profit_total", "ad_spend_total", "ad_sales_total"):
                if c in gg.columns:
                    tot[c] = float(pd.to_numeric(gg[c], errors="coerce").fillna(0.0).sum())
            tot["tacos_total"] = safe_div(tot.get("ad_spend_total", 0.0), tot.get("sales_total", 0.0)) if ("ad_spend_total" in tot and "sales_total" in tot) else 0.0
            tot["ad_acos_total"] = safe_div(tot.get("ad_spend_total", 0.0), tot.get("ad_sales_total", 0.0)) if ("ad_spend_total" in tot and "ad_sales_total" in tot) else 0.0
            out_rows.append(pd.DataFrame([tot]))

        out = pd.concat(out_rows, ignore_index=True) if out_rows else g
        for c in ("sales_total", "profit_total", "ad_spend_total", "ad_sales_total"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
        for c in ("tacos_total", "ad_acos_total"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
        return out
    except Exception:
        return pd.DataFrame()


def _md_month_blocks(
    df: pd.DataFrame,
    cols: List[str],
    month_col: str = "month",
    top_n: int = 12,
    keep_total_row: bool = True,
    total_key_col: str = "product_category",
    total_key_value: str = "TOTAL",
) -> str:
    """
    把“月度×明细”表按月份拆成多个小表，避免单个大表在 Markdown 里太难读。
    """
    if df is None or df.empty or month_col not in df.columns:
        return ""
    try:
        lines: List[str] = []
        months = [m for m in df[month_col].dropna().astype(str).unique().tolist() if m.strip()]
        months = sorted(months)
        for m in months:
            g = df[df[month_col].astype(str) == str(m)].copy()
            if g.empty:
                continue
            # 分离 TOTAL 行
            tot = pd.DataFrame()
            if keep_total_row and total_key_col in g.columns:
                tot = g[g[total_key_col].astype(str) == total_key_value].copy()
                g = g[g[total_key_col].astype(str) != total_key_value].copy()

            # 排序并截断（默认按 sales_total ；没有就按 ad_spend_total）
            sort_col = "sales_total" if "sales_total" in g.columns else ("ad_spend_total" if "ad_spend_total" in g.columns else None)
            if sort_col and sort_col in g.columns:
                g[sort_col] = pd.to_numeric(g[sort_col], errors="coerce").fillna(0.0)
                g = g.sort_values(sort_col, ascending=False)
            if top_n > 0 and len(g) > top_n:
                g = g.head(top_n).copy()

            view = pd.concat([g, tot], ignore_index=True) if (keep_total_row and not tot.empty) else g
            keep_cols = [c for c in cols if c in view.columns]
            lines.append(f"#### {m}\n")
            lines.append(df_to_md_table(view[keep_cols], max_rows=top_n + 2))
            lines.append("\n")
        return "\n".join(lines).strip() + "\n"
    except Exception:
        return ""


def _plot_monthly_category_stacked(
    monthly_cat: pd.DataFrame,
    metric_col: str,
    title: str,
    out_path: Path,
    top_n_categories: int = 8,
) -> Optional[Path]:
    """
    月度 × 商品分类：堆叠柱状图（更适合大盘对账，不在报告里铺长表）。
    - 只画 TopN 分类，其余合并为“其它”（避免 legend 过长）。
    """
    if monthly_cat is None or monthly_cat.empty:
        return None
    need = {"month", "product_category", metric_col}
    if any(c not in monthly_cat.columns for c in need):
        return None
    try:
        _set_cn_style()
        df = monthly_cat.copy()
        df["month"] = df["month"].astype(str).str.strip()
        df["product_category"] = df["product_category"].astype(str).str.strip()
        df = df[(df["month"] != "") & (df["product_category"] != "")].copy()
        df = df[df["product_category"] != "TOTAL"].copy()
        if df.empty:
            return None

        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0)

        # Top 分类（按全期 metric 求和）
        cat_sum = df.groupby("product_category", dropna=False)[metric_col].sum().sort_values(ascending=False)
        top_cats = [c for c in cat_sum.head(max(1, int(top_n_categories))).index.tolist() if str(c).strip()]
        df["category2"] = df["product_category"].where(df["product_category"].isin(top_cats), "其它")

        pv = (
            df.pivot_table(index="month", columns="category2", values=metric_col, aggfunc="sum", fill_value=0.0)
            .sort_index()
        )
        if pv.empty:
            return None

        # 列顺序：按总量降序，“其它”放最后
        col_sum = pv.sum(axis=0).sort_values(ascending=False)
        cols = [c for c in col_sum.index.tolist() if c != "其它"]
        if "其它" in pv.columns:
            cols = cols + ["其它"]
        pv = pv[cols]

        fig, ax = plt.subplots(figsize=(12, 5))
        pv.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_title(title)
        ax.set_xlabel("月份")
        ax.set_ylabel(metric_col)
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="商品分类", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        _save_fig(fig, out_path)
        return out_path
    except Exception:
        return None


def _plot_monthly_category_dashboard(
    monthly_cat: pd.DataFrame,
    out_path: Path,
    top_n_categories: int = 8,
) -> Optional[Path]:
    """
    月度×商品分类（主口径）趋势看板：把多个指标放到一张图里（2 行）。
    - 上：销售额（分类堆叠面积图）
    - 下：广告花费（分类堆叠面积图）

    说明：
    - 这两项是“可加总”的，因此适合堆叠（能同时看趋势 + 结构）。
    - TACOS/ACoS/利润等是比率/可为负，强行堆叠会误导；这些留在其它章节单独展示更清晰。
    """
    if monthly_cat is None or monthly_cat.empty:
        return None
    for c in ("month", "product_category", "sales_total", "ad_spend_total"):
        if c not in monthly_cat.columns:
            return None
    try:
        _set_cn_style()
        df = monthly_cat.copy()
        df["month"] = df["month"].astype(str).str.strip()
        df["product_category"] = df["product_category"].astype(str).str.strip()
        df = df[(df["month"] != "") & (df["product_category"] != "")].copy()
        df = df[df["product_category"] != "TOTAL"].copy()
        if df.empty:
            return None

        df["sales_total"] = pd.to_numeric(df["sales_total"], errors="coerce").fillna(0.0)
        df["ad_spend_total"] = pd.to_numeric(df["ad_spend_total"], errors="coerce").fillna(0.0)

        # Top 分类（按全期销售额求和），其余归为“其它”
        cat_sum = df.groupby("product_category", dropna=False)["sales_total"].sum().sort_values(ascending=False)
        top_cats = [c for c in cat_sum.head(max(1, int(top_n_categories))).index.tolist() if str(c).strip()]
        df["category2"] = df["product_category"].where(df["product_category"].isin(top_cats), "其它")

        pv_sales = (
            df.pivot_table(index="month", columns="category2", values="sales_total", aggfunc="sum", fill_value=0.0)
            .sort_index()
        )
        pv_spend = (
            df.pivot_table(index="month", columns="category2", values="ad_spend_total", aggfunc="sum", fill_value=0.0)
            .sort_index()
        )
        if pv_sales.empty and pv_spend.empty:
            return None

        # 列顺序：按销售额总量降序，“其它”放最后
        col_sum = pv_sales.sum(axis=0).sort_values(ascending=False)
        cols = [c for c in col_sum.index.tolist() if c != "其它"]
        if "其它" in pv_sales.columns:
            cols = cols + ["其它"]
        if not pv_sales.empty:
            pv_sales = pv_sales[[c for c in cols if c in pv_sales.columns]]
        if not pv_spend.empty:
            pv_spend = pv_spend[[c for c in cols if c in pv_spend.columns]]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax1, ax2 = axes[0], axes[1]

        if not pv_sales.empty:
            pv_sales.plot(kind="area", stacked=True, ax=ax1, colormap="tab20", alpha=0.95)
        ax1.set_title("月度×商品分类：销售额（Top 分类堆叠趋势）")
        ax1.set_xlabel("")
        ax1.set_ylabel("销售额")

        if not pv_spend.empty:
            pv_spend.plot(kind="area", stacked=True, ax=ax2, colormap="tab20", alpha=0.95)
        ax2.set_title("月度×商品分类：广告花费（Top 分类堆叠趋势）")
        ax2.set_xlabel("月份")
        ax2.set_ylabel("广告花费")
        ax2.tick_params(axis="x", rotation=0)

        # Legend 放到右侧（取上图的即可）
        handles, labels = ax1.get_legend_handles_labels()
        if handles and labels:
            ax1.legend_.remove()
            if getattr(ax2, "legend_", None) is not None:
                ax2.legend_.remove()
            fig.legend(handles, labels, title="商品分类", bbox_to_anchor=(1.02, 0.98), loc="upper left", borderaxespad=0.0)

        _save_fig(fig, out_path)
        return out_path
    except Exception:
        return None


def _plot_shop_monthly_kpi_dashboard(monthly_biz: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """
    店铺月度 KPI 趋势图（主口径：产品分析）。

    说明：
    - TACOS / 广告销售占比是“广告强度”最直观的两条线，便于店铺诊断。
    - 用月度而不是按日，避免波动太大导致读不出趋势。
    """
    if monthly_biz is None or monthly_biz.empty:
        return None
    need = {"month", "tacos_total", "ad_sales_share_total"}
    if any(c not in monthly_biz.columns for c in need):
        return None
    try:
        _set_cn_style()
        df = monthly_biz.copy()
        df["month"] = df["month"].astype(str).str.strip()
        df = df[(df["month"] != "") & (df["month"] != "TOTAL")].copy()
        if df.empty:
            return None

        df["tacos_total"] = pd.to_numeric(df["tacos_total"], errors="coerce").fillna(0.0)
        df["ad_sales_share_total"] = pd.to_numeric(df["ad_sales_share_total"], errors="coerce").fillna(0.0)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1, ax2 = axes[0], axes[1]

        ax1.plot(df["month"], df["tacos_total"], marker="o", linewidth=2, color="#C44E52")
        ax1.set_title("店铺月度趋势：TACOS（广告花费/总销售额）")
        ax1.set_ylabel("TACOS")
        ax1.set_ylim(0, max(0.05, float(df["tacos_total"].max()) * 1.25))
        ax1.grid(True, axis="y", alpha=0.3)

        ax2.plot(df["month"], df["ad_sales_share_total"], marker="o", linewidth=2, color="#4C72B0")
        ax2.set_title("店铺月度趋势：广告销售占比（广告销售额/总销售额）")
        ax2.set_ylabel("占比")
        ax2.set_xlabel("月份")
        ax2.set_ylim(0, min(1.0, max(0.1, float(df["ad_sales_share_total"].max()) * 1.25)))
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.tick_params(axis="x", rotation=0)

        _save_fig(fig, out_path)
        return out_path
    except Exception:
        return None


def _plot_keyword_funnel_dashboard(
    asin_top_targetings: Optional[pd.DataFrame],
    asin_top_search_terms: Optional[pd.DataFrame],
    asins_in_category: List[str],
    category_name: str,
    out_path: Path,
    top_n: int = 12,
) -> Optional[Path]:
    """
    商品分类的“关键词漏斗看板”（TopN + 其它）：
    - Targeting vs Search Term 两层
    - 指标：Spend / Sales / Orders / ACoS

    说明：
    - 这里用的是 pipeline 已经“分摊到 ASIN”的 TopN 明细（asin_top_*），避免在 report 阶段重复做分摊计算。
    - 该看板用于快速判断：问题更像“投放词结构”还是“搜索词结构”。
    """
    if not asins_in_category:
        return None
    asin_set = {str(a).strip().upper() for a in asins_in_category if str(a).strip()}
    if not asin_set:
        return None

    def _layer_summary(df: Optional[pd.DataFrame], entity_col: str) -> Optional[dict]:
        if df is None or df.empty:
            return None
        need = {CAN.asin, entity_col, "spend", "sales", "orders"}
        if any(c not in df.columns for c in need):
            return None
        x = df.copy()
        x["asin_norm"] = x[CAN.asin].astype(str).str.upper().str.strip()
        x = x[x["asin_norm"].isin(asin_set)].copy()
        if x.empty:
            return None
        x["spend"] = pd.to_numeric(x["spend"], errors="coerce").fillna(0.0)
        x["sales"] = pd.to_numeric(x["sales"], errors="coerce").fillna(0.0)
        x["orders"] = pd.to_numeric(x["orders"], errors="coerce").fillna(0.0)
        x[entity_col] = x[entity_col].astype(str).str.strip()
        x = x[(x[entity_col] != "") & (x[entity_col].str.lower() != "nan")].copy()
        if x.empty:
            return None

        g = x.groupby(entity_col, dropna=False, as_index=False).agg(spend=("spend", "sum"), sales=("sales", "sum"), orders=("orders", "sum")).copy()
        g["acos"] = g.apply(lambda r: safe_div(r["spend"], r["sales"]), axis=1)
        g = g.sort_values("spend", ascending=False)

        n = max(1, int(top_n))
        top = g.head(n).copy()
        other = g.iloc[n:].copy()

        def _agg(df2: pd.DataFrame) -> dict:
            spend = float(df2["spend"].sum()) if not df2.empty else 0.0
            sales = float(df2["sales"].sum()) if not df2.empty else 0.0
            orders = float(df2["orders"].sum()) if not df2.empty else 0.0
            acos = safe_div(spend, sales)
            return {"spend": spend, "sales": sales, "orders": orders, "acos": acos}

        return {"top": _agg(top), "other": _agg(other), "top_n": int(n), "entities": int(len(g))}

    try:
        _set_cn_style()
        tgt_sum = _layer_summary(asin_top_targetings, CAN.targeting)
        st_sum = _layer_summary(asin_top_search_terms, CAN.search_term)
        if not tgt_sum and not st_sum:
            return None

        rows = [
            ("Targeting", tgt_sum),
            ("Search Term", st_sum),
        ]

        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        metrics = [("spend", "花费"), ("sales", "销售额"), ("orders", "订单量"), ("acos", "ACoS")]
        for r_idx, (layer_name, s) in enumerate(rows):
            for c_idx, (m, m_label) in enumerate(metrics):
                ax = axes[r_idx][c_idx]
                if not s:
                    ax.axis("off")
                    continue
                vals = [float(s["top"].get(m, 0.0)), float(s["other"].get(m, 0.0))]
                ax.bar(["TopN", "其它"], vals, color=["#4C72B0", "#DD8452"])
                ax.set_title(f"{layer_name} · {m_label}")
                if m == "acos":
                    ax.set_ylim(0, max(0.05, max(vals) * 1.25))
                else:
                    ax.set_ylim(0, max(1.0, max(vals) * 1.25))
                ax.grid(True, axis="y", alpha=0.25)
                # 简单标注（避免读数困难）
                for i, v in enumerate(vals):
                    if v == 0:
                        continue
                    ax.text(i, v, f"{v:.2f}" if m != "orders" else f"{v:.0f}", ha="center", va="bottom", fontsize=9)

        fig.suptitle(f"关键词漏斗看板（分类：{category_name}，TopN={top_n}）", y=1.02)
        _save_fig(fig, out_path)
        return out_path
    except Exception:
        return None


def _build_asin_compares_wide(lifecycle_windows: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    把 lifecycle_windows 里的 compare_7d/14d/30d 行，整理成“宽表”方便 join 到运营动作表。
    输出列示例：
      - asin
      - c7_delta_sales / c14_delta_sales / c30_delta_sales
      - c7_marginal_tacos / c14_marginal_tacos / c30_marginal_tacos
    """
    if lifecycle_windows is None or lifecycle_windows.empty:
        return pd.DataFrame()
    if "asin" not in lifecycle_windows.columns or "window_type" not in lifecycle_windows.columns:
        return pd.DataFrame()
    try:
        lw = lifecycle_windows.copy()
        lw["asin_norm"] = lw["asin"].astype(str).str.upper().str.strip()
        lw = lw[(lw["asin_norm"] != "") & (lw["asin_norm"].str.lower() != "nan")].copy()
        lw["window_type"] = lw["window_type"].astype(str).str.strip()
        lw = lw[lw["window_type"].str.startswith("compare_")].copy()
        if lw.empty:
            return pd.DataFrame()

        cols_keep = [
            "asin_norm",
            "window_days",
            "delta_spend",
            "delta_sales",
            "delta_orders",
            "delta_sessions",
            "delta_ad_clicks",
            "marginal_tacos",
            "marginal_ad_acos",
            "spend_prev",
            "spend_recent",
            "sales_prev",
            "sales_recent",
        ]
        cols_keep = [c for c in cols_keep if c in lw.columns]
        lw = lw[cols_keep].copy()
        lw["window_days"] = pd.to_numeric(lw.get("window_days", 0), errors="coerce").fillna(0).astype(int)

        out = pd.DataFrame({"asin": sorted(lw["asin_norm"].unique().tolist())})
        for n in [7, 14, 30]:
            s = lw[lw["window_days"] == n].copy()
            if s.empty:
                continue
            # 每个 asin 只留一行（同 asin 多行时取第一行即可，因为 compare 是唯一的）
            s = s.drop_duplicates("asin_norm").copy()
            s = s.rename(columns={"asin_norm": "asin"}).drop(columns=["window_days"], errors="ignore")
            rename = {}
            for c in s.columns:
                if c == "asin":
                    continue
                rename[c] = f"c{n}_{c}"
            s = s.rename(columns=rename)
            out = out.merge(s, on="asin", how="left")
        return out
    except Exception:
        return pd.DataFrame()


def _build_campaign_ops(
    shop: str,
    camp: pd.DataFrame,
    cfg: StageConfig,
    lifecycle_board: Optional[pd.DataFrame],
    product_listing_shop: pd.DataFrame,
    asin_top_campaigns: Optional[pd.DataFrame],
    asin_top_targetings: Optional[pd.DataFrame],
    asin_top_search_terms: Optional[pd.DataFrame],
    policy: OpsPolicy,
    windows_days: List[int] = [7, 14, 30],
) -> pd.DataFrame:
    """
    生成 Campaign 维度的可执行清单（调广告的主入口通常在 campaign）。

    输出：
    - 每个 window_days 一行（最近N天 vs 前N天）
    - 带 signal/score/marginal_acos，帮助判断该“加码/控量/排查”
    - 附带 top_asins / top_categories 作为“产品侧上下文”
    """
    if camp is None or camp.empty:
        return pd.DataFrame()
    need = [CAN.date, CAN.ad_type, CAN.campaign]
    if any(c not in camp.columns for c in need):
        return pd.DataFrame()
    try:
        end_date = _infer_shop_end_date([camp])
        if end_date is None:
            return pd.DataFrame()
        # windows_days 默认用 policy，可被调用方覆盖
        days_list = windows_days or list(policy.campaign_windows_days)
        wins = build_windows(end_date, days_list)
        tables = []
        for w in wins:
            t = window_compare(
                camp,
                [CAN.ad_type, CAN.campaign],
                w,
                min_spend=max(float(policy.campaign_min_spend or 0.0), float(cfg.waste_spend or 5.0) / 2.0),
            )
            if t is None or t.empty:
                continue
            tables.append(t)
        if not tables:
            return pd.DataFrame()
        df = pd.concat(tables, ignore_index=True).copy()
        df["shop"] = shop

        # 关联 ASIN / 分类 / 阶段 / 库存（优先 advertised_product 汇总；不全时用 targeting/search_term 的分摊结果兜底）
        asin_map = pd.DataFrame()
        try:
            frames = []
            def _take(df0: Optional[pd.DataFrame]) -> None:
                if df0 is None or df0.empty:
                    return
                if CAN.ad_type not in df0.columns or CAN.campaign not in df0.columns or CAN.asin not in df0.columns:
                    return
                x = df0.copy()
                x["asin_norm"] = x[CAN.asin].astype(str).str.upper().str.strip()
                x = x[(x["asin_norm"] != "") & (x["asin_norm"].str.lower() != "nan")].copy()
                x["spend"] = pd.to_numeric(x.get("spend", 0.0), errors="coerce").fillna(0.0)
                frames.append(x[[CAN.ad_type, CAN.campaign, "asin_norm", "spend"]])

            _take(asin_top_campaigns)
            _take(asin_top_targetings)
            _take(asin_top_search_terms)

            if frames:
                m = pd.concat(frames, ignore_index=True)
                m = m.groupby([CAN.ad_type, CAN.campaign, "asin_norm"], dropna=False, as_index=False).agg(spend=("spend", "sum")).copy()

                # asin -> meta（分类/阶段/库存）
                meta = pd.DataFrame()
                try:
                    if lifecycle_board is not None and not lifecycle_board.empty and "asin" in lifecycle_board.columns:
                        meta = lifecycle_board.copy()
                        meta["asin_norm"] = meta["asin"].astype(str).str.upper().str.strip()
                        keep = ["asin_norm"]
                        for c in ("product_category", "current_phase", "inventory", "flag_low_inventory", "flag_oos"):
                            if c in meta.columns:
                                keep.append(c)
                        meta = meta[keep].drop_duplicates("asin_norm").copy()
                    elif product_listing_shop is not None and not product_listing_shop.empty and "ASIN" in product_listing_shop.columns:
                        meta = product_listing_shop.copy()
                        meta["asin_norm"] = meta["ASIN"].astype(str).str.upper().str.strip()
                        meta["product_category"] = meta["商品分类"].astype(str).fillna("").str.strip() if "商品分类" in meta.columns else ""
                        meta = meta[["asin_norm", "product_category"]].drop_duplicates("asin_norm").copy()
                except Exception:
                    meta = pd.DataFrame()

                # top asins per campaign
                rows = []
                k_top = max(1, int(policy.campaign_top_asins_per_campaign or 3))
                for (ad_type, campaign), g in m.groupby([CAN.ad_type, CAN.campaign], dropna=False):
                    g2 = g.sort_values("spend", ascending=False).head(k_top).copy()
                    asins = [str(x).strip() for x in g2["asin_norm"].tolist() if str(x).strip()]
                    cats = []
                    phases = []
                    inv_risk = 0
                    if meta is not None and not meta.empty:
                        mm = meta[meta["asin_norm"].isin(asins)].copy()
                        if "product_category" in mm.columns:
                            cats = [str(x).strip() for x in mm["product_category"].fillna("").tolist()]
                            cats = [c if c else "未分类" for c in cats]
                        if "current_phase" in mm.columns:
                            phases = [str(x).strip().lower() for x in mm["current_phase"].fillna("").tolist() if str(x).strip()]
                        # 库存风险：任一 top asin 断货/低库存
                        try:
                            if "flag_oos" in mm.columns:
                                if (pd.to_numeric(mm["flag_oos"], errors="coerce").fillna(0).astype(int) > 0).any():
                                    inv_risk = 1
                            if inv_risk == 0 and "inventory" in mm.columns:
                                inv = pd.to_numeric(mm["inventory"], errors="coerce").fillna(999999)
                                if (inv <= int(policy.low_inventory_threshold)).any():
                                    inv_risk = 1
                        except Exception:
                            inv_risk = 0

                    dom_phase = phases[0] if phases else ""
                    if phases:
                        # 简单众数：按出现次数排序
                        from collections import Counter
                        dom_phase = Counter(phases).most_common(1)[0][0]
                    rows.append(
                        {
                            CAN.ad_type: ad_type,
                            CAN.campaign: campaign,
                            "top_asins": ",".join(asins),
                            "top_categories": ",".join(sorted({c for c in cats if c})),
                            "dominant_phase": dom_phase,
                            "inventory_risk": int(inv_risk),
                        }
                    )
                asin_map = pd.DataFrame(rows)
        except Exception:
            asin_map = pd.DataFrame()

        if asin_map is not None and not asin_map.empty:
            df = df.merge(asin_map, on=[CAN.ad_type, CAN.campaign], how="left")
        for c in ("top_asins", "top_categories", "dominant_phase"):
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str).str.strip()
                df.loc[df[c].str.lower() == "nan", c] = ""

        # suggestion：仍然不做“硬编码调参”，只输出方向标签；但会结合阶段/库存做约束
        def _allowed_acos(phase: str) -> float:
            p = str(phase or "").strip().lower()
            mult = float(policy.phase_acos_multiplier.get(p, 1.0)) if isinstance(policy.phase_acos_multiplier, dict) else 1.0
            return float(cfg.target_acos) * float(mult)

        def _suggest(row: pd.Series) -> str:
            sig = str(row.get("signal", "") or "")
            spend_r = float(row.get("spend_recent", 0.0) or 0.0)
            acos_r = float(row.get("acos_recent", 0.0) or 0.0)
            marg = float(row.get("marginal_acos", 0.0) or 0.0)
            phase = str(row.get("dominant_phase", "") or "")
            inv_raw = row.get("inventory_risk", 0)
            try:
                inv_risk = 0 if pd.isna(inv_raw) else int(float(inv_raw))
            except Exception:
                inv_risk = 0
            allowed = _allowed_acos(phase)

            if sig == "spend_spike_no_sales" and spend_r >= float(cfg.waste_spend or 10.0):
                return "CUT_OR_NEGATE"
            if sig in {"decaying"} and spend_r >= float(cfg.waste_spend or 10.0):
                return "REVIEW"

            # 加码：需要效率满足阶段目标，且库存不风险（默认阻断）
            if sig in {"accelerating", "efficiency_gain"}:
                if allowed > 0 and acos_r > 0 and acos_r <= allowed:
                    if bool(policy.block_scale_when_low_inventory) and inv_risk > 0:
                        return "CHECK_INVENTORY"
                    return "SCALE"

            # 增量效率很差：即使不在 spike，也建议排查
            if allowed > 0 and marg > 0 and marg >= allowed * 1.2 and spend_r >= float(cfg.waste_spend or 10.0):
                return "REVIEW"

            return "MONITOR"

        df["suggestion"] = df.apply(_suggest, axis=1)

        # 列顺序（更像运营表）
        cols = [
            "shop",
            "window_days",
            "recent_start",
            "recent_end",
            "prev_start",
            "prev_end",
            CAN.ad_type,
            CAN.campaign,
            "suggestion",
            "signal",
            "score",
            "spend_prev",
            "spend_recent",
            "delta_spend",
            "sales_prev",
            "sales_recent",
            "delta_sales",
            "orders_prev",
            "orders_recent",
            "delta_orders",
            "acos_prev",
            "acos_recent",
            "marginal_acos",
            "marginal_cpa",
            "dominant_phase",
            "inventory_risk",
            "top_asins",
            "top_categories",
        ]
        cols = [c for c in cols if c in df.columns]
        for c in ("spend_prev", "spend_recent", "delta_spend", "sales_prev", "sales_recent", "delta_sales"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
        for c in ("acos_prev", "acos_recent", "marginal_acos", "marginal_cpa"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(4)
        return df[cols].copy()
    except Exception:
        return pd.DataFrame()

def _category_monthly_biz_dashboard(product_analysis_shop: pd.DataFrame, asin_to_category: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    单个商品分类（月度）：主口径=产品分析（自然+广告合计）。
    """
    if product_analysis_shop is None or product_analysis_shop.empty or asin_to_category is None or asin_to_category.empty:
        return pd.DataFrame()
    if CAN.date not in product_analysis_shop.columns or "ASIN" not in product_analysis_shop.columns:
        return pd.DataFrame()
    try:
        pa = product_analysis_shop.copy()
        pa[CAN.date] = pd.to_datetime(pa[CAN.date], errors="coerce")
        pa = pa[pa[CAN.date].notna()].copy()
        pa["asin_norm"] = pa["ASIN"].astype(str).str.upper().str.strip()
        pa = pa.merge(asin_to_category, on="asin_norm", how="left")
        pa["product_category"] = pa["product_category"].fillna("").astype(str).str.strip()
        pa.loc[pa["product_category"].str.lower() == "nan", "product_category"] = ""
        pa.loc[pa["product_category"] == "", "product_category"] = "未分类"
        pa = pa[pa["product_category"] == str(category)].copy()
        if pa.empty:
            return pd.DataFrame()
        pa["month"] = pa[CAN.date].dt.to_period("M").astype(str)

        cols = ["销售额", "订单量", "Sessions", "毛利润", "广告花费", "广告销售额", "广告订单量"]
        agg_map = {c: "sum" for c in cols if c in pa.columns}
        if not agg_map:
            return pd.DataFrame()

        g = pa.groupby("month", dropna=False, as_index=False).agg(agg_map).copy()
        rename = {
            "销售额": "sales_total",
            "订单量": "orders_total",
            "Sessions": "sessions_total",
            "毛利润": "profit_total",
            "广告花费": "ad_spend_total",
            "广告销售额": "ad_sales_total",
            "广告订单量": "ad_orders_total",
        }
        g = g.rename(columns={k: v for k, v in rename.items() if k in g.columns})
        if "sales_total" in g.columns and "ad_spend_total" in g.columns:
            g["tacos_total"] = g.apply(lambda r: safe_div(r.get("ad_spend_total", 0.0), r.get("sales_total", 0.0)), axis=1)
        if "orders_total" in g.columns and "sessions_total" in g.columns:
            g["cvr_total"] = g.apply(lambda r: safe_div(r.get("orders_total", 0.0), r.get("sessions_total", 0.0)), axis=1)
        if "ad_spend_total" in g.columns and "ad_sales_total" in g.columns:
            g["ad_acos_total"] = g.apply(lambda r: safe_div(r.get("ad_spend_total", 0.0), r.get("ad_sales_total", 0.0)), axis=1)
        g = g.sort_values("month")
        for c in ("sales_total", "profit_total", "ad_spend_total", "ad_sales_total"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(2)
        for c in ("tacos_total", "cvr_total", "ad_acos_total"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(4)

        total = {"month": "TOTAL"}
        for c in ("sales_total", "orders_total", "sessions_total", "profit_total", "ad_spend_total", "ad_sales_total", "ad_orders_total"):
            if c in g.columns:
                total[c] = float(pd.to_numeric(g[c], errors="coerce").fillna(0.0).sum())
        total["tacos_total"] = safe_div(total.get("ad_spend_total", 0.0), total.get("sales_total", 0.0))
        total["cvr_total"] = safe_div(total.get("orders_total", 0.0), total.get("sessions_total", 0.0))
        total["ad_acos_total"] = safe_div(total.get("ad_spend_total", 0.0), total.get("ad_sales_total", 0.0))
        out = pd.concat([g, pd.DataFrame([total])], ignore_index=True)
        for c in ("sales_total", "profit_total", "ad_spend_total", "ad_sales_total"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
        for c in ("tacos_total", "cvr_total", "ad_acos_total"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
        return out
    except Exception:
        return pd.DataFrame()


def _category_product_monthly_pivot(
    product_analysis_shop: pd.DataFrame,
    asin_to_category: pd.DataFrame,
    category: str,
    value_col: str,
    product_name_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    分类内：产品月度透视表（行=ASIN，列=月份 + TOTAL），用于“每月拆开再汇总”的产品层复盘。
    """
    if product_analysis_shop is None or product_analysis_shop.empty or asin_to_category is None or asin_to_category.empty:
        return pd.DataFrame()
    if CAN.date not in product_analysis_shop.columns or "ASIN" not in product_analysis_shop.columns:
        return pd.DataFrame()
    if value_col not in product_analysis_shop.columns:
        return pd.DataFrame()
    try:
        pa = product_analysis_shop.copy()
        pa[CAN.date] = pd.to_datetime(pa[CAN.date], errors="coerce")
        pa = pa[pa[CAN.date].notna()].copy()
        pa["asin_norm"] = pa["ASIN"].astype(str).str.upper().str.strip()
        pa["month"] = pa[CAN.date].dt.to_period("M").astype(str)
        pa = pa.merge(asin_to_category, on="asin_norm", how="left")
        pa["product_category"] = pa["product_category"].fillna("").astype(str).str.strip()
        pa.loc[pa["product_category"].str.lower() == "nan", "product_category"] = ""
        pa.loc[pa["product_category"] == "", "product_category"] = "未分类"
        pa = pa[pa["product_category"] == str(category)].copy()
        if pa.empty:
            return pd.DataFrame()

        g = pa.groupby(["asin_norm", "month"], dropna=False, as_index=False).agg(val=(value_col, "sum")).copy()
        pv = g.pivot(index="asin_norm", columns="month", values="val").fillna(0.0)
        pv["TOTAL"] = pv.sum(axis=1)
        pv = pv.reset_index().rename(columns={"asin_norm": "ASIN"})

        # 补产品名
        if product_name_map is not None and not product_name_map.empty and "ASIN" in product_name_map.columns:
            try:
                pv = pv.merge(product_name_map, on="ASIN", how="left")
            except Exception:
                pass
        if "product_name" in pv.columns:
            cols = ["ASIN", "product_name"] + [c for c in pv.columns if c not in ("ASIN", "product_name")]
            pv = pv[cols]

        # 列顺序：月份升序 + TOTAL
        # 注意：这里的正则是匹配类似 "2025-12" 的月份列名（不要写成 "\\d" 否则会匹配字面反斜杠）
        month_cols = [c for c in pv.columns if re.match(r"^\d{4}-\d{2}$", str(c))]
        month_cols = sorted(month_cols)
        keep = [c for c in ["ASIN", "product_name"] if c in pv.columns] + month_cols + (["TOTAL"] if "TOTAL" in pv.columns else [])
        pv = pv[keep]
        for c in month_cols + (["TOTAL"] if "TOTAL" in pv.columns else []):
            pv[c] = pd.to_numeric(pv[c], errors="coerce").round(2)
        return pv
    except Exception:
        return pd.DataFrame()


def _phase_scale_acos_max(phase: str, target_acos: float) -> float:
    """
    不同生命周期阶段对“可接受 ACoS”的上限不同（粗粒度、可解释）。
    这里只用于生成“候选队列”，最终建议仍由你后续 prompt/AI 写成。
    """
    p = str(phase or "").strip().lower()
    if p in {"pre_launch", "launch", "growth"}:
        return float(target_acos) * 1.2
    if p in {"mature", "stable"}:
        return float(target_acos) * 1.0
    if p in {"decline"}:
        return float(target_acos) * 0.9
    return float(target_acos) * 1.0


def _category_keyword_queues(
    asin_top_targetings: Optional[pd.DataFrame],
    asin_top_search_terms: Optional[pd.DataFrame],
    board: pd.DataFrame,
    asins_in_category: List[str],
    cfg: StageConfig,
) -> dict:
    """
    分类级关键词候选队列（给 AI/运营做决策）：
    - Targeting：放量/控量/否词
    - Search term：加词/否词
    """
    out: dict = {
        "tgt_scale": pd.DataFrame(),
        "tgt_cut": pd.DataFrame(),
        "tgt_neg": pd.DataFrame(),
        "st_add": pd.DataFrame(),
        "st_neg": pd.DataFrame(),
    }
    if not asins_in_category:
        return out

    asin_set = {str(a).strip().upper() for a in asins_in_category if str(a).strip()}
    if not asin_set:
        return out

    # asin -> (product_name, phase)
    meta = {}
    try:
        if board is not None and not board.empty and "asin" in board.columns:
            b = board.copy()
            b["asin_norm"] = b["asin"].astype(str).str.upper().str.strip()
            for _, r in b.iterrows():
                a = str(r.get("asin_norm", "")).strip()
                if not a:
                    continue
                meta[a] = {
                    "product_name": str(r.get("product_name", "") or ""),
                    "current_phase": str(r.get("current_phase", "") or ""),
                }
    except Exception:
        meta = {}

    def _enrich(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.copy()
        if "asin" not in d.columns:
            return pd.DataFrame()
        d["asin_norm"] = d["asin"].astype(str).str.upper().str.strip()
        d = d[d["asin_norm"].isin(asin_set)].copy()
        if d.empty:
            return pd.DataFrame()
        d["product_name"] = d["asin_norm"].apply(lambda a: meta.get(a, {}).get("product_name", ""))
        d["current_phase"] = d["asin_norm"].apply(lambda a: meta.get(a, {}).get("current_phase", ""))
        # 清洗 match_type
        if "match_type" in d.columns:
            d["match_type"] = d["match_type"].fillna("").astype(str).str.strip()
            d.loc[(d["match_type"] == "") | (d["match_type"].str.lower() == "nan"), "match_type"] = "N/A"
        # 数值列
        for c in ("spend", "sales", "orders", "clicks", "impressions", "acos", "cvr", "ctr"):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        if "cpc" not in d.columns and "spend" in d.columns and "clicks" in d.columns:
            d["cpc"] = d.apply(lambda r: safe_div(r.get("spend", 0.0), r.get("clicks", 0.0)), axis=1)
        # 统一实体列
        if entity_col in d.columns:
            d[entity_col] = d[entity_col].fillna("").astype(str).str.strip()
            d = d[(d[entity_col] != "") & (d[entity_col].str.lower() != "nan")].copy()
        return d

    # ---- Targeting queues ----
    tgt_all = _enrich(asin_top_targetings, "targeting")
    if not tgt_all.empty:
        # 否词候选：花费高 + 无单 + 点击达标
        spend_thr = float(cfg.waste_spend or 0.0)
        min_clicks = int(cfg.min_clicks or 0)
        neg = tgt_all[(tgt_all["spend"] >= spend_thr) & (tgt_all.get("orders", 0.0) <= 0.0)].copy()
        if "clicks" in neg.columns and min_clicks > 0:
            neg = neg[neg["clicks"] >= float(min_clicks)].copy()
        if not neg.empty:
            neg["signal"] = "NEGATE_CANDIDATE"
            neg = neg.sort_values(["spend", "clicks"], ascending=False).head(60)
            out["tgt_neg"] = neg

        # 控量候选：有单但 ACoS 高（按阶段收紧）
        cut = tgt_all[(tgt_all.get("orders", 0.0) >= 1.0) & (tgt_all["spend"] >= max(5.0, spend_thr / 2.0))].copy()
        if not cut.empty:
            def _is_bad(row) -> bool:
                phase = str(row.get("current_phase", "") or "")
                acos = float(row.get("acos", 0.0) or 0.0)
                # decline/mature 更严格；launch/growth 更宽松
                max_ok = _phase_scale_acos_max(phase, float(cfg.target_acos)) * (1.15 if str(phase).lower() in {"launch", "growth", "pre_launch"} else 1.05)
                return acos >= max_ok and acos > 0

            cut = cut[cut.apply(_is_bad, axis=1)].copy()
            if not cut.empty:
                cut["signal"] = "CUT_CANDIDATE"
                cut = cut.sort_values(["acos", "spend"], ascending=False).head(60)
                out["tgt_cut"] = cut

        # 放量候选：有单 + ACoS 可接受（按阶段）+ CVR 不太差
        scale = tgt_all[(tgt_all.get("orders", 0.0) >= 1.0) & (tgt_all["spend"] >= 5.0)].copy()
        if not scale.empty:
            def _is_good(row) -> bool:
                phase = str(row.get("current_phase", "") or "")
                acos = float(row.get("acos", 0.0) or 0.0)
                cvr = float(row.get("cvr", 0.0) or 0.0)
                return (acos > 0) and (acos <= _phase_scale_acos_max(phase, float(cfg.target_acos))) and (cvr >= 0.03 or float(row.get("orders", 0.0) or 0.0) >= 2.0)

            scale = scale[scale.apply(_is_good, axis=1)].copy()
            if not scale.empty:
                scale["signal"] = "SCALE_CANDIDATE"
                scale = scale.sort_values(["sales", "orders", "spend"], ascending=False).head(60)
                out["tgt_scale"] = scale

    # ---- Search term queues ----
    st_all = _enrich(asin_top_search_terms, "search_term")
    if not st_all.empty:
        spend_thr = float(cfg.waste_spend or 0.0)
        min_clicks = int(cfg.min_clicks or 0)

        neg = st_all[(st_all["spend"] >= spend_thr) & (st_all.get("orders", 0.0) <= 0.0)].copy()
        if "clicks" in neg.columns and min_clicks > 0:
            neg = neg[neg["clicks"] >= float(min_clicks)].copy()
        if not neg.empty:
            neg["signal"] = "NEGATE_CANDIDATE"
            neg = neg.sort_values(["spend", "clicks"], ascending=False).head(60)
            out["st_neg"] = neg

        add = st_all[(st_all.get("orders", 0.0) >= 1.0) & (st_all["spend"] >= 5.0)].copy()
        if not add.empty:
            def _is_good(row) -> bool:
                phase = str(row.get("current_phase", "") or "")
                acos = float(row.get("acos", 0.0) or 0.0)
                return (acos > 0) and (acos <= _phase_scale_acos_max(phase, float(cfg.target_acos)))

            add = add[add.apply(_is_good, axis=1)].copy()
            if not add.empty:
                add["signal"] = "ADD_TO_TARGETING"
                add = add.sort_values(["sales", "orders", "spend"], ascending=False).head(60)
                out["st_add"] = add

    return out


def _add_derived_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    给汇总表补充常用 KPI（基于 canonical 列计算，避免依赖报表自带口径）。
    - ctr = clicks / impressions
    - cpc = spend / clicks
    - cvr = orders / clicks
    - acos = spend / sales
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "impressions" in out.columns and "clicks" in out.columns and "ctr" not in out.columns:
        out["ctr"] = out.apply(lambda r: safe_div(r.get("clicks", 0.0), r.get("impressions", 0.0)), axis=1)
    if "spend" in out.columns and "clicks" in out.columns and "cpc" not in out.columns:
        out["cpc"] = out.apply(lambda r: safe_div(r.get("spend", 0.0), r.get("clicks", 0.0)), axis=1)
    if "orders" in out.columns and "clicks" in out.columns and "cvr" not in out.columns:
        out["cvr"] = out.apply(lambda r: safe_div(r.get("orders", 0.0), r.get("clicks", 0.0)), axis=1)
    if "spend" in out.columns and "sales" in out.columns and "acos" not in out.columns:
        out["acos"] = out.apply(lambda r: safe_div(r.get("spend", 0.0), r.get("sales", 0.0)), axis=1)
    return out


def _shop_targeting_tables(tgt: pd.DataFrame, cfg: StageConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    店铺层：Targeting（投放）主视角。

    输出两张表：
    - top：投放词排行榜（用于看“主动投放词”的整体表现）
    - waste：浪费投放词（用于否词/降价/暂停）
    """
    if tgt is None or tgt.empty or CAN.targeting not in tgt.columns:
        return pd.DataFrame(), pd.DataFrame()

    try:
        # 清掉空投放词（否则会出现 targeting=nan 的一大行，影响可读性）
        t0 = tgt.copy()
        t0 = t0[t0[CAN.targeting].notna()].copy()
        t0[CAN.targeting] = t0[CAN.targeting].astype(str).str.strip()
        t0 = t0[(t0[CAN.targeting] != "") & (t0[CAN.targeting].str.lower() != "nan")].copy()
        if CAN.match_type in t0.columns:
            t0[CAN.match_type] = t0[CAN.match_type].fillna("").astype(str).str.strip()
            t0.loc[(t0[CAN.match_type] == "") | (t0[CAN.match_type].str.lower() == "nan"), CAN.match_type] = "N/A"

        dims = [c for c in [CAN.ad_type, CAN.targeting, CAN.match_type] if c in tgt.columns]
        g = (
            t0.groupby(dims, dropna=False, as_index=False)
            .agg(
                impressions=(CAN.impressions, "sum") if CAN.impressions in t0.columns else (dims[0], "size"),
                clicks=(CAN.clicks, "sum") if CAN.clicks in t0.columns else (dims[0], "size"),
                spend=(CAN.spend, "sum") if CAN.spend in t0.columns else (dims[0], "size"),
                sales=(CAN.sales, "sum") if CAN.sales in t0.columns else (dims[0], "size"),
                orders=(CAN.orders, "sum") if CAN.orders in t0.columns else (dims[0], "size"),
                campaign_count=(CAN.campaign, "nunique") if CAN.campaign in t0.columns else (dims[0], "size"),
                ad_group_count=(CAN.ad_group, "nunique") if CAN.ad_group in t0.columns else (dims[0], "size"),
            )
            .copy()
        )
        g = _add_derived_kpis(g)

        # 统一小数展示
        for c in ("spend", "sales"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(2)
        for c in ("acos", "ctr", "cvr"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(4)
        if "cpc" in g.columns:
            g["cpc"] = pd.to_numeric(g["cpc"], errors="coerce").round(2)

        top = g.sort_values(["spend", "sales", "orders"], ascending=False).head(40).copy()

        waste = pd.DataFrame()
        try:
            spend_thr = float(cfg.waste_spend or 0.0)
            waste = g[(pd.to_numeric(g["spend"], errors="coerce").fillna(0.0) >= spend_thr) & (pd.to_numeric(g["orders"], errors="coerce").fillna(0.0) <= 0)].copy()
            waste = waste.sort_values(["spend", "clicks", "impressions"], ascending=False).head(40)
        except Exception:
            waste = pd.DataFrame()

        return top, waste
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _shop_search_term_tables(st: pd.DataFrame, cfg: StageConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    店铺层：Search Term（用户真实搜索词）视角。

    输出两张表：
    - winners：新增/强化候选（高转化/低 ACoS 的真实搜索词）
    - waste：否词候选（花费高但无订单/无销售）
    """
    if st is None or st.empty or CAN.search_term not in st.columns:
        return pd.DataFrame(), pd.DataFrame()
    try:
        # 清掉空搜索词（避免 search_term=nan 影响判断）
        s0 = st.copy()
        s0 = s0[s0[CAN.search_term].notna()].copy()
        s0[CAN.search_term] = s0[CAN.search_term].astype(str).str.strip()
        s0 = s0[(s0[CAN.search_term] != "") & (s0[CAN.search_term].str.lower() != "nan")].copy()
        if CAN.match_type in s0.columns:
            s0[CAN.match_type] = s0[CAN.match_type].fillna("").astype(str).str.strip()
            s0.loc[(s0[CAN.match_type] == "") | (s0[CAN.match_type].str.lower() == "nan"), CAN.match_type] = "N/A"

        dims = [c for c in [CAN.ad_type, CAN.search_term, CAN.match_type] if c in st.columns]
        g = (
            s0.groupby(dims, dropna=False, as_index=False)
            .agg(
                impressions=(CAN.impressions, "sum") if CAN.impressions in s0.columns else (dims[0], "size"),
                clicks=(CAN.clicks, "sum") if CAN.clicks in s0.columns else (dims[0], "size"),
                spend=(CAN.spend, "sum") if CAN.spend in s0.columns else (dims[0], "size"),
                sales=(CAN.sales, "sum") if CAN.sales in s0.columns else (dims[0], "size"),
                orders=(CAN.orders, "sum") if CAN.orders in s0.columns else (dims[0], "size"),
                campaign_count=(CAN.campaign, "nunique") if CAN.campaign in s0.columns else (dims[0], "size"),
            )
            .copy()
        )
        g = _add_derived_kpis(g)

        for c in ("spend", "sales"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(2)
        for c in ("acos", "ctr", "cvr"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(4)
        if "cpc" in g.columns:
            g["cpc"] = pd.to_numeric(g["cpc"], errors="coerce").round(2)

        # winners：有订单 + acos 不离谱（先用目标ACoS做一个粗筛）
        winners = pd.DataFrame()
        try:
            winners = g[(pd.to_numeric(g["orders"], errors="coerce").fillna(0.0) >= 1) & (pd.to_numeric(g["acos"], errors="coerce").fillna(999.0) <= float(cfg.target_acos) * 1.2)].copy()
            winners = winners.sort_values(["sales", "orders", "spend"], ascending=False).head(40)
        except Exception:
            winners = pd.DataFrame()

        waste = pd.DataFrame()
        try:
            spend_thr = float(cfg.waste_spend or 0.0)
            waste = g[(pd.to_numeric(g["spend"], errors="coerce").fillna(0.0) >= spend_thr) & (pd.to_numeric(g["orders"], errors="coerce").fillna(0.0) <= 0)].copy()
            waste = waste.sort_values(["spend", "clicks", "impressions"], ascending=False).head(40)
        except Exception:
            waste = pd.DataFrame()

        return winners, waste
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _shop_campaign_tables(camp: pd.DataFrame, pl: pd.DataFrame, cfg: StageConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    店铺层：Campaign 控制面板（最后落地用）。
    - top：活动排行榜（看钱花在哪）
    - high_acos：高 ACoS 活动（优先排查）
    """
    if camp is None or camp.empty or CAN.campaign not in camp.columns:
        return pd.DataFrame(), pd.DataFrame()
    try:
        dims = [c for c in [CAN.ad_type, CAN.campaign] if c in camp.columns]
        g = (
            camp.groupby(dims, dropna=False, as_index=False)
            .agg(
                impressions=(CAN.impressions, "sum") if CAN.impressions in camp.columns else (dims[0], "size"),
                clicks=(CAN.clicks, "sum") if CAN.clicks in camp.columns else (dims[0], "size"),
                spend=(CAN.spend, "sum") if CAN.spend in camp.columns else (dims[0], "size"),
                sales=(CAN.sales, "sum") if CAN.sales in camp.columns else (dims[0], "size"),
                orders=(CAN.orders, "sum") if CAN.orders in camp.columns else (dims[0], "size"),
            )
            .copy()
        )
        g = _add_derived_kpis(g)
        for c in ("spend", "sales"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(2)
        for c in ("acos", "ctr", "cvr"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").round(4)
        if "cpc" in g.columns:
            g["cpc"] = pd.to_numeric(g["cpc"], errors="coerce").round(2)

        top = g.sort_values(["spend", "sales", "orders"], ascending=False).head(30).copy()

        high_acos = pd.DataFrame()
        try:
            spend_min = float(cfg.waste_spend or 0.0)
            high_acos = g[(pd.to_numeric(g["spend"], errors="coerce").fillna(0.0) >= spend_min) & (pd.to_numeric(g["acos"], errors="coerce").fillna(0.0) >= float(cfg.target_acos) * 1.2)].copy()
            high_acos = high_acos.sort_values(["acos", "spend"], ascending=False).head(30)
        except Exception:
            high_acos = pd.DataFrame()

        return top, high_acos
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def _plot_drivers_bar(rows: List[dict], value_key: str, title: str, out_path: Path) -> Optional[Path]:
    """
    drivers bar：Top ASIN 的 delta_sales / delta_ad_spend。
    """
    if not rows:
        return None
    df = pd.DataFrame(rows).copy()
    if df.empty or "asin" not in df.columns or value_key not in df.columns:
        return None
    _set_cn_style()
    df["value"] = pd.to_numeric(df[value_key], errors="coerce").fillna(0.0)
    df = df.sort_values("value", ascending=False).head(12).copy()
    if df.empty:
        return None
    df["label"] = df.apply(lambda r: _short_label(r.get("asin"), r.get("product_name", "")), axis=1)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df, x="value", y="label", ax=ax, color="#8172B2")
    ax.set_title(title)
    ax.set_xlabel(value_key)
    ax.set_ylabel("ASIN")
    _save_fig(fig, out_path)
    return out_path


def _inventory_risk_table(ap: pd.DataFrame, product_listing: pd.DataFrame) -> pd.DataFrame:
    """
    Top 广告ASIN + 可售，用于“库存风险”提示。
    """
    if ap.empty or product_listing.empty:
        return pd.DataFrame()
    if CAN.asin not in ap.columns or "ASIN" not in product_listing.columns:
        return pd.DataFrame()

    asin_spend = summarize(ap, [CAN.asin]).sort_values("spend", ascending=False).head(50)
    pl = product_listing.copy()
    # 统一 asin 大小写
    asin_spend["asin_norm"] = asin_spend[CAN.asin].astype(str).str.upper()
    pl["asin_norm"] = pl["ASIN"].astype(str).str.upper()
    cols = ["asin_norm"]
    if "可售" in pl.columns:
        cols.append("可售")
    if "品名" in pl.columns:
        cols.append("品名")
    if "商品分类" in pl.columns:
        cols.append("商品分类")
    pl2 = pl[cols].drop_duplicates("asin_norm")
    merged = asin_spend.merge(pl2, on="asin_norm", how="left")
    # 输出列
    out_cols = ["asin_norm", "spend", "sales", "orders"]
    for c in ("可售", "品名", "商品分类"):
        if c in merged.columns:
            out_cols.append(c)
    merged = merged[out_cols].rename(columns={"asin_norm": "ASIN"})
    # 风险排序：可售升序 + 花费降序
    if "可售" in merged.columns:
        merged = merged.sort_values(["可售", "spend"], ascending=[True, False])
    return merged.head(30)


def generate_shop_report(
    shop_dir: Path,
    shop: str,
    cfg: StageConfig,
    summary_total: dict,
    st: pd.DataFrame,
    tgt: pd.DataFrame,
    camp: pd.DataFrame,
    pl: pd.DataFrame,
    ap: pd.DataFrame,
    pp: pd.DataFrame,
    product_listing_shop: pd.DataFrame,
    product_analysis_shop: pd.DataFrame,
    actions: List[ActionCandidate],
    diagnostics: Optional[dict] = None,
    lifecycle_board: Optional[pd.DataFrame] = None,
    lifecycle_segments: Optional[pd.DataFrame] = None,
    lifecycle_windows: Optional[pd.DataFrame] = None,
    asin_top_campaigns: Optional[pd.DataFrame] = None,
    asin_top_search_terms: Optional[pd.DataFrame] = None,
    asin_top_targetings: Optional[pd.DataFrame] = None,
    asin_top_placements: Optional[pd.DataFrame] = None,
) -> Path:
    """
    生成单店铺报告（Markdown + 图）。
    """
    figures_dir, reports_dir, ai_dir = _ensure_dirs(shop_dir)
    ops_dir = shop_dir / "ops"
    ops_dir.mkdir(parents=True, exist_ok=True)
    # 运营动作清单（按 分类→ASIN）：在 report 生成过程中累积，最后输出 CSV
    ops_rows: List[dict] = []
    # ops 策略：从仓库根目录 config/ops_policy.json 读取（读不到就用默认）
    try:
        repo_root = Path(__file__).resolve().parents[2]
        policy = load_ops_policy(repo_root / "config" / "ops_policy.json")
    except Exception:
        policy = OpsPolicy()

    # 字体尽早初始化 + 自检图（避免中文方块）
    font_smoke_png = None
    try:
        _set_cn_style()
        font_smoke_png = _plot_cn_font_smoke_test(figures_dir / "cn_font_test.png")
    except Exception:
        font_smoke_png = None

    # 1) 图表（用 campaign 日级趋势作为“主时间轴”）
    ts = _ts_from_campaign(camp)
    trends_png = _plot_trends(ts, title_prefix=shop, out_path=figures_dir / "trend_overview.png")

    # Top campaigns（花费）
    top_camp_png = None
    if not camp.empty and CAN.campaign in camp.columns:
        top_camp = summarize(camp, [CAN.campaign]).sort_values("spend", ascending=False)
        top_camp_png = _plot_top_bar(
            top_camp,
            x=CAN.campaign,
            y="spend",
            title="Top Campaign（按花费）",
            out_path=figures_dir / "top_campaign_spend.png",
            top_n=15,
        )

    # Top placements（花费）
    top_place_png = None
    if not pl.empty and CAN.placement in pl.columns:
        top_place = summarize(pl, [CAN.placement]).sort_values("spend", ascending=False)
        top_place_png = _plot_top_bar(
            top_place,
            x=CAN.placement,
            y="spend",
            title="Top Placement（按花费）",
            out_path=figures_dir / "top_placement_spend.png",
            top_n=10,
        )

    # Top targetings（花费）：投放层结构
    top_tgt_png = None
    if tgt is not None and not tgt.empty and CAN.targeting in tgt.columns:
        top_tgt = summarize(tgt, [CAN.targeting]).sort_values("spend", ascending=False)
        top_tgt_png = _plot_top_bar(
            top_tgt,
            x=CAN.targeting,
            y="spend",
            title="Top Targeting（按花费）",
            out_path=figures_dir / "top_targeting_spend.png",
            top_n=15,
        )

    # 搜索词矩阵
    matrix_png = _plot_matrix_search_terms(st, figures_dir / "search_term_matrix.png", cfg)

    # 库存风险表
    inv_df = _inventory_risk_table(ap, product_listing_shop)

    # 产品分析（整体经营）趋势：TACOS（广告花费/销售额）
    tacos_png = None
    if not product_analysis_shop.empty and CAN.date in product_analysis_shop.columns and "销售额" in product_analysis_shop.columns and "广告花费" in product_analysis_shop.columns:
        _set_cn_style()
        pa = product_analysis_shop.copy()
        pa = pa[pa[CAN.date].notna()].copy()
        if not pa.empty:
            ts2 = (
                pa.groupby(CAN.date, as_index=True)
                .agg(sales=("销售额", "sum"), ad_spend=("广告花费", "sum"))
                .sort_index()
            )
            ts2["tacos"] = ts2.apply(lambda r: safe_div(r["ad_spend"], r["sales"]), axis=1)
            fig, ax = plt.subplots(figsize=(12, 4))
            ts2[["tacos"]].plot(ax=ax)
            ax.set_title("产品分析：TACOS（广告花费/总销售额）趋势")
            ax.set_xlabel("日期")
            tacos_png = figures_dir / "tacos_trend.png"
            _save_fig(fig, tacos_png)

    # 店铺诊断可视化：阶段分布 & 花费集中度
    phase_cnt_png = _plot_phase_counts(lifecycle_board, figures_dir / "phase_asin_count.png") if lifecycle_board is not None else None
    phase_share_png = _plot_phase_spend_share(lifecycle_board, figures_dir / "phase_spend_share.png") if lifecycle_board is not None else None
    conc_png = _plot_asin_spend_concentration(lifecycle_board, figures_dir / "asin_spend_concentration.png") if lifecycle_board is not None else None

    # 店铺变化来源（drivers）可视化：优先用 shop_scorecard.window_7d
    diag = diagnostics or {}
    drivers_png_sales = None
    drivers_png_spend = None
    try:
        shop_score = diag.get("shop_scorecard") if isinstance(diag, dict) else None
        d7 = (shop_score.get("drivers", {}) if isinstance(shop_score, dict) else {}).get("window_7d") if isinstance(shop_score, dict) else None
        if isinstance(d7, dict):
            by_sales = d7.get("by_delta_sales")
            by_spend = d7.get("by_delta_ad_spend")
            if isinstance(by_sales, list) and by_sales:
                drivers_png_sales = _plot_drivers_bar(by_sales, "delta_sales", "变化来源（7天）：Top ASIN 销售额增量", figures_dir / "drivers_delta_sales_7d.png")
            if isinstance(by_spend, list) and by_spend:
                drivers_png_spend = _plot_drivers_bar(by_spend, "delta_ad_spend", "变化来源（7天）：Top ASIN 广告花费增量", figures_dir / "drivers_delta_ad_spend_7d.png")
    except Exception:
        drivers_png_sales = None
        drivers_png_spend = None

    # 2) Markdown 报告（全量深挖：主要给 AI/分析用）
    report_path = ai_dir / "report.md"
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ad_dmin, ad_dmax = _date_range(camp if not camp.empty else st)

    # 动作候选表
    act_rows = [asdict(a) for a in actions]
    act_df = pd.DataFrame(act_rows) if act_rows else pd.DataFrame()
    p0_df = act_df[act_df["priority"] == "P0"].copy() if not act_df.empty and "priority" in act_df.columns else pd.DataFrame()
    p1_df = act_df[act_df["priority"] == "P1"].copy() if not act_df.empty and "priority" in act_df.columns else pd.DataFrame()

    # 关键提醒：目标ACoS偏离
    caution_lines: List[str] = []
    acos_val = float(summary_total.get("acos", 0.0) or 0.0)
    if acos_val > 0 and acos_val > cfg.target_acos * 1.15:
        caution_lines.append(f"- 本期 ACoS={acos_val:.0%} 高于目标({cfg.target_acos:.0%})较多，建议优先处理 P0 否词与高 ACoS 词/投放。")
    if inv_df is not None and not inv_df.empty and "可售" in inv_df.columns:
        low = inv_df[inv_df["可售"].fillna(999999) <= 20]
        if not low.empty:
            caution_lines.append(f"- Top 广告ASIN 中有 {len(low)} 个可售≤20，投放前先确认补货/断货风险。")

    # 诊断结构（供 shop_scorecard 与后续章节复用）
    diag = diagnostics or {}

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# {shop} 广告分析报告（合成版）\n\n")
        f.write(f"- 生成时间：{generated_at}\n")
        if ad_dmin and ad_dmax:
            f.write(f"- 广告数据范围：{ad_dmin} ~ {ad_dmax}\n")
        # 产品分析范围（主口径）
        try:
            if product_analysis_shop is not None and not product_analysis_shop.empty and CAN.date in product_analysis_shop.columns:
                pa_dmin = str(product_analysis_shop[CAN.date].min())
                pa_dmax = str(product_analysis_shop[CAN.date].max())
                if pa_dmin and pa_dmax:
                    f.write(f"- 产品分析数据范围：{pa_dmin} ~ {pa_dmax}\n")
        except Exception:
            pass
        f.write(f"- 阶段配置：{cfg.name}（目标ACoS={cfg.target_acos:.0%}，浪费阈值=${cfg.waste_spend}，最小点击={cfg.min_clicks}）\n\n")
        # 图表字体：用于验收中文显示
        try:
            _set_cn_style()
            f.write(f"- 图表字体：{_cn_font_info()}\n\n")
        except Exception:
            f.write("- 图表字体：设置失败（不影响指标计算）\n\n")
        if font_smoke_png:
            f.write(f"![cn_font_test](../figures/{font_smoke_png.name})\n\n")

        # 指标说明：避免运营/对账时被“字段名”困扰
        f.write("## 0) 指标说明（窗口/对比/Delta）\n\n")
        f.write("- `phase`：生命周期阶段（pre_launch/launch/growth/mature/decline/stable/inactive）。\n")
        f.write("- `window_type`：指标聚合窗口类型；`window_type` 与 `phase` 是两条独立维度（同一 ASIN 会有多个 window_type 行）。\n")
        f.write("  - `since_first_stock_to_date`：从“首次可售/首次有库存”到当前日期的累计。\n")
        f.write("  - `since_first_sale_to_date`：从“首次出单”到当前日期的累计。\n")
        f.write("  - `cycle_to_date`：从“当前周期起点”到当前日期的累计；`cycle_id`=2 通常表示经历过一次断货(>=阈值)后到货，进入第 2 个周期。\n")
        f.write("  - `current_phase_to_date`：从“当前阶段起点”到当前日期的累计（用于看当前阶段的投入产出）。\n")
        f.write("- `compare_7d/compare_14d/compare_30d`：滚动环比（最近 N 天 vs 前 N 天）。\n")
        f.write("  - `delta_*`：最近N天 - 前N天（正值=上升，负值=下降）。\n")
        f.write("  - `marginal_tacos`：增量 TACOS = delta_ad_spend / delta_sales（用于判断“新增花费换来的新增销售是否划算”）。\n")
        f.write("  - `marginal_ad_acos`：增量 ACoS = delta_ad_spend / delta_ad_sales（用于判断“新增花费换来的新增广告销售是否划算”）。\n\n")

        # 店铺诊断（scorecard）：把“结构/集中度/阶段/变化”变得可读
        shop_score = (diag.get("shop_scorecard") if isinstance(diag, dict) else {}) if diag is not None else {}
        biz = shop_score.get("biz_kpi") if isinstance(shop_score, dict) else None

        # 1) 店铺总览：做“月度大盘看板”，并附带总计行（TOTAL）
        f.write("## 1) 店铺总览（月度大盘看板）\n\n")

        f.write("### 1.1 主口径：产品分析（自然+广告合计，按月）\n\n")
        f.write("- 数据源：`reports/产品分析/`（按日）汇总到月份\n\n")
        monthly_biz = _shop_monthly_biz_dashboard(product_analysis_shop=product_analysis_shop)
        if monthly_biz is not None and not monthly_biz.empty:
            cols = [c for c in ["month", "sales_total", "orders_total", "sessions_total", "profit_total", "ad_spend_total", "ad_sales_total", "tacos_total", "ad_sales_share_total", "cvr_total"] if c in monthly_biz.columns]
            f.write(df_to_md_table(monthly_biz[cols], max_rows=24))
            f.write("\n\n")
            # 月度趋势图：一眼看投放强度变化（TACOS / 广告销售占比）
            try:
                kpi_png = _plot_shop_monthly_kpi_dashboard(monthly_biz, figures_dir / "monthly_shop_kpis.png")
                if kpi_png:
                    f.write(f"![monthly_shop_kpis](../figures/{kpi_png.name})\n\n")
            except Exception:
                pass
        else:
            f.write("（缺少产品分析数据，无法生成月度看板）\n\n")

        # 月度 × 商品分类（主口径）
        try:
            monthly_cat = _shop_monthly_category_dashboard(product_listing_shop=product_listing_shop, product_analysis_shop=product_analysis_shop)
            if monthly_cat is not None and not monthly_cat.empty:
                f.write("### 1.2 月度 × 商品分类（主口径：产品分析）\n\n")
                f.write("- 说明：此处改为可视化（不再在报告里铺长表）。如需明细，可切换 `--output-profile full` 输出中间表再深挖。\n\n")
                dash_png = _plot_monthly_category_dashboard(
                    monthly_cat=monthly_cat,
                    out_path=figures_dir / "monthly_category_dashboard.png",
                    top_n_categories=8,
                )
                if dash_png:
                    f.write(f"![monthly_category_dashboard](../figures/{dash_png.name})\n\n")
        except Exception:
            pass

        f.write("### 1.3 广告口径：Campaign 报告（按月）\n\n")
        f.write("- 数据源：`reports/ad/**/广告活动报告*.xlsx`\n\n")
        monthly_ad = _shop_monthly_ad_dashboard(camp=camp)
        if monthly_ad is not None and not monthly_ad.empty:
            cols = [c for c in ["month", "impressions", "clicks", "spend", "sales", "orders", "ctr", "cpc", "cvr", "acos"] if c in monthly_ad.columns]
            f.write(df_to_md_table(monthly_ad[cols], max_rows=24))
            f.write("\n\n")
        else:
            f.write("（缺少广告活动报告数据，无法生成月度看板）\n\n")

        if shop_score and isinstance(shop_score, dict):
            f.write("## 2) 店铺诊断（健康度/结构/变化）\n\n")

            # KPI：经营口径优先
            if isinstance(biz, dict) and biz:
                f.write("### 2.1 经营口径 KPI（自然+广告合计）\n\n")
                f.write(df_to_md_table(pd.DataFrame([biz])))
                f.write("\n\n")

            # 阶段分布图
            if phase_cnt_png:
                f.write(f"![phase_asin_count](../figures/{phase_cnt_png.name})\n\n")
            if phase_share_png:
                f.write(f"![phase_spend_share](../figures/{phase_share_png.name})\n\n")

            # 花费集中度图
            if conc_png:
                f.write(f"![asin_spend_concentration](../figures/{conc_png.name})\n\n")

            # 变化来源图
            if drivers_png_sales:
                f.write(f"![drivers_delta_sales_7d](../figures/{drivers_png_sales.name})\n\n")
            if drivers_png_spend:
                f.write(f"![drivers_delta_ad_spend_7d](../figures/{drivers_png_spend.name})\n\n")

            # 集中度表
            conc = shop_score.get("concentration") if isinstance(shop_score, dict) else None
            if isinstance(conc, dict) and conc:
                f.write("### 2.2 花费集中度（7天滚动广告花费）\n\n")
                f.write(df_to_md_table(pd.DataFrame([conc])))
                f.write("\n\n")

            # 滚动环比（店铺层）
            cmp = shop_score.get("compares") if isinstance(shop_score, dict) else None
            if isinstance(cmp, list) and cmp:
                f.write("### 2.3 店铺滚动环比（最近N天 vs 前N天）\n\n")
                f.write(df_to_md_table(pd.DataFrame(cmp), max_rows=10))
                f.write("\n\n")

            # 变化来源表（7天）
            try:
                drivers = shop_score.get("drivers") if isinstance(shop_score, dict) else None
                d7 = (drivers.get("window_7d") if isinstance(drivers, dict) else None) if drivers is not None else None
                if isinstance(d7, dict):
                    by_sales = d7.get("by_delta_sales")
                    by_spend = d7.get("by_delta_ad_spend")
                    if isinstance(by_sales, list) and by_sales:
                        f.write("### 2.4 变化来源（7天）：哪些产品在拉动/拖累销售\n\n")
                        dfv = pd.DataFrame(by_sales)
                        try:
                            if "product_category" not in dfv.columns and lifecycle_board is not None and (not lifecycle_board.empty):
                                lb = lifecycle_board.copy()
                                if "asin" in lb.columns and "product_category" in lb.columns:
                                    lb["asin_norm"] = lb["asin"].astype(str).str.upper().str.strip()
                                    lb["product_category"] = lb["product_category"].fillna("").astype(str).str.strip()
                                    lb.loc[lb["product_category"].str.lower() == "nan", "product_category"] = ""
                                    lb.loc[lb["product_category"] == "", "product_category"] = "未分类"
                                    m = lb[["asin_norm", "product_category"]].drop_duplicates("asin_norm")
                                    dfv["asin_norm"] = dfv["asin"].astype(str).str.upper().str.strip()
                                    dfv = dfv.merge(m, on="asin_norm", how="left").drop(columns=["asin_norm"])
                            # 把分类列尽量放前面，方便扫读
                            if "product_category" in dfv.columns:
                                cols = ["product_category"] + [c for c in dfv.columns if c != "product_category"]
                                dfv = dfv[cols]
                        except Exception:
                            pass
                        f.write(df_to_md_table(dfv, max_rows=12))
                        f.write("\n\n")
                    if isinstance(by_spend, list) and by_spend:
                        f.write("### 2.5 变化来源（7天）：哪些产品在推动花费变化\n\n")
                        dfv = pd.DataFrame(by_spend)
                        try:
                            if "product_category" not in dfv.columns and lifecycle_board is not None and (not lifecycle_board.empty):
                                lb = lifecycle_board.copy()
                                if "asin" in lb.columns and "product_category" in lb.columns:
                                    lb["asin_norm"] = lb["asin"].astype(str).str.upper().str.strip()
                                    lb["product_category"] = lb["product_category"].fillna("").astype(str).str.strip()
                                    lb.loc[lb["product_category"].str.lower() == "nan", "product_category"] = ""
                                    lb.loc[lb["product_category"] == "", "product_category"] = "未分类"
                                    m = lb[["asin_norm", "product_category"]].drop_duplicates("asin_norm")
                                    dfv["asin_norm"] = dfv["asin"].astype(str).str.upper().str.strip()
                                    dfv = dfv.merge(m, on="asin_norm", how="left").drop(columns=["asin_norm"])
                            if "product_category" in dfv.columns:
                                cols = ["product_category"] + [c for c in dfv.columns if c != "product_category"]
                                dfv = dfv[cols]
                        except Exception:
                            pass
                        f.write(df_to_md_table(dfv, max_rows=12))
                        f.write("\n\n")
            except Exception:
                pass

        # 产品侧变化摘要（自然/流量）
        try:
            if lifecycle_windows is not None and not lifecycle_windows.empty and "window_type" in lifecycle_windows.columns:
                cmp7 = lifecycle_windows[lifecycle_windows["window_type"].astype(str) == "compare_7d"].copy()
            else:
                cmp7 = pd.DataFrame()
            if cmp7 is not None and not cmp7.empty and "asin" in cmp7.columns:
                cmp7["asin_norm"] = cmp7["asin"].astype(str).str.upper().str.strip()
                if lifecycle_board is not None and not lifecycle_board.empty and "asin" in lifecycle_board.columns:
                    lb = lifecycle_board.copy()
                    lb["asin_norm"] = lb["asin"].astype(str).str.upper().str.strip()
                    keep_cols = ["asin_norm"]
                    for c in ("product_category", "product_name", "current_phase"):
                        if c in lb.columns:
                            keep_cols.append(c)
                    lb = lb[keep_cols].drop_duplicates("asin_norm")
                    cmp7 = cmp7.merge(lb, on="asin_norm", how="left")
            if cmp7 is not None and not cmp7.empty:
                f.write("### 2.6 产品侧变化摘要（近7天 vs 前7天）\n\n")

                # 自然销售变化 Top
                if "delta_organic_sales" in cmp7.columns:
                    view = cmp7.copy()
                    for c in ("organic_sales_prev", "organic_sales_recent", "delta_organic_sales"):
                        if c in view.columns:
                            view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0.0)
                    view["_abs_delta"] = pd.to_numeric(view["delta_organic_sales"], errors="coerce").fillna(0.0).abs()
                    view = view.sort_values("_abs_delta", ascending=False).head(12).drop(columns=["_abs_delta"], errors="ignore")
                    cols = [
                        c
                        for c in [
                            "product_category",
                            "product_name",
                            "asin",
                            "organic_sales_prev",
                            "organic_sales_recent",
                            "delta_organic_sales",
                            "current_phase",
                        ]
                        if c in view.columns
                    ]
                    f.write("#### 自然销售变化 Top\n\n")
                    f.write(df_to_md_table(view[cols], max_rows=12))
                    f.write("\n\n")

                # Sessions 变化 Top
                if "delta_sessions" in cmp7.columns:
                    view = cmp7.copy()
                    for c in ("sessions_prev", "sessions_recent", "delta_sessions"):
                        if c in view.columns:
                            view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0.0)
                    view["_abs_delta"] = pd.to_numeric(view["delta_sessions"], errors="coerce").fillna(0.0).abs()
                    view = view.sort_values("_abs_delta", ascending=False).head(12).drop(columns=["_abs_delta"], errors="ignore")
                    cols = [
                        c
                        for c in [
                            "product_category",
                            "product_name",
                            "asin",
                            "sessions_prev",
                            "sessions_recent",
                            "delta_sessions",
                            "current_phase",
                        ]
                        if c in view.columns
                    ]
                    f.write("#### Sessions 变化 Top\n\n")
                    f.write(df_to_md_table(view[cols], max_rows=12))
                    f.write("\n\n")
            else:
                f.write("### 2.6 产品侧变化摘要（近7天 vs 前7天）\n\n")
                f.write("（无）\n\n")
        except Exception:
            f.write("### 2.6 产品侧变化摘要（近7天 vs 前7天）\n\n")
            f.write("（生成失败）\n\n")

        # 店铺维度：商品分类汇总（横向对比同类产品）
        cat_df = pd.DataFrame()
        try:
            cat_df = _category_summary(product_listing_shop=product_listing_shop, product_analysis_shop=product_analysis_shop)
            if cat_df is not None and not cat_df.empty:
                f.write("### 2.7 商品分类汇总（横向对比同类产品）\n\n")
                cols = [c for c in ["product_category", "asin_count", "sales_total", "profit_total", "ad_spend_total", "tacos_total", "ad_sales_total", "ad_acos_total", "cvr_total"] if c in cat_df.columns]
                f.write(df_to_md_table(cat_df[cols], max_rows=30))
                f.write("\n\n")
        except Exception:
            pass

        if caution_lines:
            f.write("## 3) 风险/提醒\n\n")
            f.write("\n".join(caution_lines) + "\n\n")

        # ========== 产品画像（核心：以产品为单位，把经营+生命周期+广告结构放在一起） ==========
        has_portrait = lifecycle_board is not None and not lifecycle_board.empty
        if has_portrait:
            f.write("## 4) 商品分类 → 产品 → 关键词（核心）\n\n")

            board = lifecycle_board.copy()
            # 确保带上商品分类（优先用 pipeline enrichment 的 product_category；没有就从 productListing 补一次）
            try:
                if "product_category" not in board.columns and product_listing_shop is not None and not product_listing_shop.empty:
                    if "ASIN" in product_listing_shop.columns and "商品分类" in product_listing_shop.columns:
                        plm = product_listing_shop.copy()
                        plm["asin_norm"] = plm["ASIN"].astype(str).str.upper().str.strip()
                        plm["product_category"] = plm["商品分类"].astype(str).fillna("").str.strip()
                        plm.loc[plm["product_category"].str.lower() == "nan", "product_category"] = ""
                        plm = plm[["asin_norm", "product_category"]].drop_duplicates("asin_norm")
                        board["asin_norm"] = board["asin"].astype(str).str.upper().str.strip()
                        board = board.merge(plm, on="asin_norm", how="left").drop(columns=["asin_norm"])
            except Exception:
                pass
            # 可读性：做一次四舍五入
            for c in ("sales_roll", "sessions_roll", "ad_spend_roll", "profit_roll", "tacos_roll", "cvr_roll"):
                if c in board.columns:
                    try:
                        board[c] = pd.to_numeric(board[c], errors="coerce").round(4 if "roll" in c and ("tacos" in c or "cvr" in c) else 2)
                    except Exception:
                        pass
            # 优先看“当前花费高/最需要关注”的产品
            try:
                if "ad_spend_roll" in board.columns:
                    board = board.sort_values("ad_spend_roll", ascending=False)
            except Exception:
                pass

            cols_focus = [
                c
                for c in [
                    "asin",
                    "product_category",
                    "product_name",
                    "cycle_id",
                    "current_phase",
                    "inventory",
                    "ad_spend_roll",
                    "tacos_roll",
                    "profit_roll",
                    "sales_roll",
                    "sessions_roll",
                    "flag_low_inventory",
                    "flag_oos",
                ]
                if c in board.columns
            ]
            # 注意：这里不再先输出“全店 Top ASIN 表”，避免打散“商品分类 -> 产品”阅读顺序；
            # 产品清单会在下面按“商品分类”分组输出。

            # 生命周期窗口：主口径 + compare
            main_win = pd.DataFrame()
            cmp_win = pd.DataFrame()
            if lifecycle_windows is not None and not lifecycle_windows.empty and "asin" in lifecycle_windows.columns:
                w = lifecycle_windows.copy()
                w["asin_norm"] = w["asin"].astype(str).str.upper().str.strip()
                main_win = w[w["window_type"] == "since_first_stock_to_date"].copy()
                if main_win.empty:
                    main_win = w[w["window_type"] == "cycle_to_date"].copy()
                cmp_win = w[w["window_type"].astype(str).str.startswith("compare_")].copy()

            # 产品分析（自然月全量）：按 ASIN 汇总一份，作为“经营底座”
            pa_asin = pd.DataFrame()
            if product_analysis_shop is not None and (not product_analysis_shop.empty) and "ASIN" in product_analysis_shop.columns:
                pa = product_analysis_shop.copy()
                pa["asin_norm"] = pa["ASIN"].astype(str).str.upper().str.strip()
                metrics_cols = []
                for col in ("销售额", "订单量", "Sessions", "广告花费", "广告销售额", "广告订单量", "毛利润"):
                    if col in pa.columns:
                        metrics_cols.append(col)
                if metrics_cols:
                    agg_map = {c: "sum" for c in metrics_cols}
                    pa_asin = pa.groupby("asin_norm", dropna=False, as_index=False).agg(agg_map).copy()
                    # 经营口径：TACOS（广告花费/销售额）
                    if "销售额" in pa_asin.columns and "广告花费" in pa_asin.columns:
                        pa_asin["tacos_total"] = pa_asin.apply(lambda r: safe_div(r.get("广告花费", 0.0), r.get("销售额", 0.0)), axis=1)
                    # CVR（订单量/Sessions）
                    if "订单量" in pa_asin.columns and "Sessions" in pa_asin.columns:
                        pa_asin["cvr_total"] = pa_asin.apply(lambda r: safe_div(r.get("订单量", 0.0), r.get("Sessions", 0.0)), axis=1)

            # 广告结构：这些表是“按 ASIN 汇总”的（我们在 pipeline 里已经用 advertised_product 做了权重分摊）
            def _pick(df: Optional[pd.DataFrame], asin: str) -> pd.DataFrame:
                if df is None or df.empty or "asin" not in df.columns:
                    return pd.DataFrame()
                try:
                    s = df["asin"].astype(str).str.upper().str.strip()
                    return df[s == asin.upper()].copy()
                except Exception:
                    return pd.DataFrame()

            """
            DEPRECATED（保留供回溯，不再执行）：
            旧版本是“全店 Top 15，再按分类分组”，不满足你要求的阅读顺序（必须：商品分类 -> 产品 -> targeting）。
            """

            # ✅ 新版本：所有广告呈现都按“商品分类 -> 产品 -> targeting/search term/campaign/placement”组织
            # 先列出每个分类的“全量产品清单”，再对每个分类挑 Top N 产品给出关键词主线明细。

            # 1) 合并“生命周期看板 + 产品分析底座”，用于做“分类->产品清单”
            b2 = board.copy()
            try:
                b2["asin_norm"] = b2["asin"].astype(str).str.upper().str.strip()
            except Exception:
                b2["asin_norm"] = ""
            if "product_category" in b2.columns:
                b2["product_category"] = b2["product_category"].fillna("").astype(str).str.strip()
                b2.loc[b2["product_category"].str.lower() == "nan", "product_category"] = ""
            else:
                b2["product_category"] = ""
            b2.loc[b2["product_category"] == "", "product_category"] = "未分类"

            if pa_asin is not None and not pa_asin.empty and "asin_norm" in pa_asin.columns:
                try:
                    b2 = b2.merge(pa_asin, on="asin_norm", how="left", suffixes=("", "_pa"))
                except Exception:
                    pass

            # 分类映射（供“月度×分类/产品月度透视/关键词队列”复用）
            asin_to_category = pd.DataFrame()
            try:
                asin_to_category = b2[["asin_norm", "product_category"]].dropna().drop_duplicates("asin_norm").copy()
            except Exception:
                asin_to_category = pd.DataFrame(columns=["asin_norm", "product_category"])

            # 产品名映射（用于月度透视表展示）
            product_name_map = pd.DataFrame()
            try:
                product_name_map = (
                    b2[["asin_norm", "product_name"]]
                    .dropna()
                    .drop_duplicates("asin_norm")
                    .rename(columns={"asin_norm": "ASIN"})
                    .copy()
                )
            except Exception:
                product_name_map = pd.DataFrame(columns=["ASIN", "product_name"])

            # 2) 分类顺序：优先用“商品分类汇总”的排序；没有则按分类广告花费/滚动花费排序
            cat_order: List[str] = []
            try:
                if cat_df is not None and not cat_df.empty and "product_category" in cat_df.columns:
                    cat_order = [str(x).strip() for x in cat_df["product_category"].tolist() if str(x).strip()]
            except Exception:
                cat_order = []

            if not cat_order:
                try:
                    if "广告花费" in b2.columns:
                        cat_order = (
                            b2.groupby("product_category", dropna=False)["广告花费"]
                            .sum()
                            .reset_index(name="ad_spend_total_sum")
                            .sort_values("ad_spend_total_sum", ascending=False)["product_category"]
                            .astype(str)
                            .tolist()
                        )
                    elif "ad_spend_roll" in b2.columns:
                        cat_order = (
                            b2.groupby("product_category", dropna=False)["ad_spend_roll"]
                            .sum()
                            .reset_index(name="ad_spend_roll_sum")
                            .sort_values("ad_spend_roll_sum", ascending=False)["product_category"]
                            .astype(str)
                            .tolist()
                        )
                    else:
                        cat_order = sorted(b2["product_category"].dropna().astype(str).unique().tolist())
                except Exception:
                    cat_order = ["未分类"]

            detail_top_n_per_category = 6
            f.write("### 4.1 商品分类 → 产品清单（全量）\n\n")
            f.write("- 说明：产品清单用 `reports/产品分析/` 汇总口径（自然+广告合计），并带上生命周期当前阶段与库存等信息。\n\n")

            for cat in cat_order:
                cat_name = str(cat).strip() or "未分类"
                f.write(f"#### 分类：{cat_name}\n\n")

                cat_view = b2[b2["product_category"].astype(str) == cat_name].copy()
                if cat_view.empty:
                    continue

                # 产品清单（全量）
                try:
                    if "广告花费" in cat_view.columns:
                        cat_view = cat_view.sort_values("广告花费", ascending=False)
                    elif "ad_spend_roll" in cat_view.columns:
                        cat_view = cat_view.sort_values("ad_spend_roll", ascending=False)
                except Exception:
                    pass

                cols_list = [
                    c
                    for c in [
                        "asin",
                        "product_name",
                        "current_phase",
                        "inventory",
                        "ad_spend_roll",
                        "tacos_roll",
                        "sales_roll",
                        "profit_roll",
                        "销售额",
                        "广告花费",
                        "广告销售额",
                        "毛利润",
                        "tacos_total",
                        "cvr_total",
                        "flag_low_inventory",
                        "flag_oos",
                    ]
                    if c in cat_view.columns
                ]

                # 简单四舍五入，避免表格太乱
                for c in ("ad_spend_roll", "sales_roll", "profit_roll", "销售额", "广告花费", "广告销售额", "毛利润"):
                    if c in cat_view.columns:
                        cat_view[c] = pd.to_numeric(cat_view[c], errors="coerce").round(2)
                for c in ("tacos_roll", "tacos_total", "cvr_total"):
                    if c in cat_view.columns:
                        cat_view[c] = pd.to_numeric(cat_view[c], errors="coerce").round(4)

                f.write("产品清单（全量）：\n\n")
                f.write(df_to_md_table(cat_view[cols_list], max_rows=200))
                f.write("\n\n")

                # 分类月度看板（主口径）
                try:
                    cat_monthly = _category_monthly_biz_dashboard(
                        product_analysis_shop=product_analysis_shop,
                        asin_to_category=asin_to_category,
                        category=cat_name,
                    )
                    if cat_monthly is not None and not cat_monthly.empty:
                        f.write("分类月度看板（主口径：产品分析）：\n\n")
                        cols = [c for c in ["month", "sales_total", "profit_total", "ad_spend_total", "tacos_total", "ad_sales_total", "ad_acos_total", "cvr_total"] if c in cat_monthly.columns]
                        f.write(df_to_md_table(cat_monthly[cols], max_rows=24))
                        f.write("\n\n")
                except Exception:
                    pass

                # 关键词漏斗（分类级）：Targeting vs Search Term（TopN + 其它）
                try:
                    asins_all = [str(x).strip().upper() for x in cat_view["asin"].dropna().tolist() if str(x).strip()] if "asin" in cat_view.columns else []
                    if asins_all:
                        # 分类名通常是中文，直接 slug 会变成空串，导致图片被覆盖；用 hash 保证稳定且唯一
                        safe_key = hashlib.md5(cat_name.encode("utf-8")).hexdigest()[:8]
                        funnel_png = _plot_keyword_funnel_dashboard(
                            asin_top_targetings=asin_top_targetings,
                            asin_top_search_terms=asin_top_search_terms,
                            asins_in_category=asins_all,
                            category_name=cat_name,
                            out_path=figures_dir / f"keyword_funnel_{safe_key}.png",
                            top_n=int(policy.keyword_funnel_top_n or 12),
                        )
                        if funnel_png:
                            f.write("关键词漏斗看板（分类级：Targeting vs Search Term）\n\n")
                            f.write(f"![keyword_funnel](../figures/{funnel_png.name})\n\n")
                except Exception:
                    pass

                # 产品月度透视（全量）：销售额/广告花费
                try:
                    pv_sales = _category_product_monthly_pivot(
                        product_analysis_shop=product_analysis_shop,
                        asin_to_category=asin_to_category,
                        category=cat_name,
                        value_col="销售额",
                        product_name_map=product_name_map,
                    )
                    if pv_sales is not None and not pv_sales.empty:
                        f.write("产品月度销售额（全量）：\n\n")
                        f.write(df_to_md_table(pv_sales, max_rows=200))
                        f.write("\n\n")
                except Exception:
                    pass

                try:
                    pv_spend = _category_product_monthly_pivot(
                        product_analysis_shop=product_analysis_shop,
                        asin_to_category=asin_to_category,
                        category=cat_name,
                        value_col="广告花费",
                        product_name_map=product_name_map,
                    )
                    if pv_spend is not None and not pv_spend.empty:
                        f.write("产品月度广告花费（全量）：\n\n")
                        f.write(df_to_md_table(pv_spend, max_rows=200))
                        f.write("\n\n")
                except Exception:
                    pass

                # 分类级关键词候选队列（可执行队列：给 AI/运营做决策）
                try:
                    asins_all = [str(x).strip().upper() for x in cat_view["asin"].dropna().tolist() if str(x).strip()] if "asin" in cat_view.columns else []
                    queues = _category_keyword_queues(
                        asin_top_targetings=asin_top_targetings,
                        asin_top_search_terms=asin_top_search_terms,
                        board=board,
                        asins_in_category=asins_all,
                        cfg=cfg,
                    )
                    if isinstance(queues, dict):
                        any_rows = any((isinstance(v, pd.DataFrame) and (not v.empty)) for v in queues.values())
                        if any_rows:
                            f.write("关键词候选队列（分类级：给 AI/运营做决策）\n\n")

                            # 同步落地：写入 ops/actions.csv（便于运营筛选/分配）
                            def _emit_ops(dfq: pd.DataFrame, layer: str, action_group: str) -> None:
                                try:
                                    if dfq is None or dfq.empty:
                                        return
                                    view = dfq.copy()
                                    if "spend" in view.columns:
                                        view["spend"] = pd.to_numeric(view["spend"], errors="coerce").fillna(0.0)
                                        view = view.sort_values("spend", ascending=False)
                                    view = view.head(200)
                                    for _, r in view.iterrows():
                                        ops_rows.append(
                                            {
                                                "shop": shop,
                                                "product_category": cat_name,
                                                "layer": layer,
                                                "action_group": action_group,
                                                "priority": "P0" if "NEG" in str(r.get("signal", "")).upper() else "P1",
                                                "asin": str(r.get("asin_norm", "") or ""),
                                                "product_name": str(r.get("product_name", "") or ""),
                                                "current_phase": str(r.get("current_phase", "") or ""),
                                                "ad_type": str(r.get("ad_type", "") or ""),
                                                "campaign": str(r.get("campaign", "") or ""),
                                                "match_type": str(r.get("match_type", "") or ""),
                                                "targeting": str(r.get("targeting", "") or ""),
                                                "search_term": str(r.get("search_term", "") or ""),
                                                "spend": float(r.get("spend", 0.0) or 0.0),
                                                "sales": float(r.get("sales", 0.0) or 0.0),
                                                "orders": float(r.get("orders", 0.0) or 0.0),
                                                "clicks": float(r.get("clicks", 0.0) or 0.0),
                                                "acos": float(r.get("acos", 0.0) or 0.0),
                                                "cvr": float(r.get("cvr", 0.0) or 0.0),
                                                "cpc": float(r.get("cpc", 0.0) or 0.0),
                                                "signal": str(r.get("signal", "") or ""),
                                            }
                                        )
                                except Exception:
                                    return

                            def _write_q(title: str, dfq: pd.DataFrame, cols_pref: List[str]) -> None:
                                if dfq is None or dfq.empty:
                                    return
                                view = dfq.copy()
                                cols = [c for c in cols_pref if c in view.columns]
                                # 统一展示列
                                for c in ("spend", "sales"):
                                    if c in view.columns:
                                        view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
                                for c in ("acos", "ctr", "cvr"):
                                    if c in view.columns:
                                        view[c] = pd.to_numeric(view[c], errors="coerce").round(4)
                                if "cpc" in view.columns:
                                    view["cpc"] = pd.to_numeric(view["cpc"], errors="coerce").round(2)
                                f.write(f"- {title}\n\n")
                                f.write(df_to_md_table(view[cols], max_rows=30))
                                f.write("\n\n")

                            _write_q(
                                "Targeting 放量候选（SCALE）",
                                queues.get("tgt_scale"),
                                ["signal", "asin_norm", "product_name", "current_phase", "ad_type", "targeting", "match_type", "spend", "sales", "orders", "acos", "cvr", "cpc", "campaign"],
                            )
                            _emit_ops(queues.get("tgt_scale"), layer="targeting", action_group="SCALE")
                            _write_q(
                                "Targeting 控量候选（CUT）",
                                queues.get("tgt_cut"),
                                ["signal", "asin_norm", "product_name", "current_phase", "ad_type", "targeting", "match_type", "spend", "sales", "orders", "acos", "cvr", "cpc", "campaign"],
                            )
                            _emit_ops(queues.get("tgt_cut"), layer="targeting", action_group="CUT")
                            _write_q(
                                "Targeting 否词候选（NEGATE）",
                                queues.get("tgt_neg"),
                                ["signal", "asin_norm", "product_name", "current_phase", "ad_type", "targeting", "match_type", "spend", "clicks", "orders", "sales", "acos", "cpc", "campaign"],
                            )
                            _emit_ops(queues.get("tgt_neg"), layer="targeting", action_group="NEGATE")
                            _write_q(
                                "Search Term 加词候选（ADD）",
                                queues.get("st_add"),
                                ["signal", "asin_norm", "product_name", "current_phase", "ad_type", "search_term", "match_type", "spend", "sales", "orders", "acos", "cvr", "campaign"],
                            )
                            _emit_ops(queues.get("st_add"), layer="search_term", action_group="ADD")
                            _write_q(
                                "Search Term 否词候选（NEGATE）",
                                queues.get("st_neg"),
                                ["signal", "asin_norm", "product_name", "current_phase", "ad_type", "search_term", "match_type", "spend", "clicks", "orders", "sales", "acos", "campaign"],
                            )
                            _emit_ops(queues.get("st_neg"), layer="search_term", action_group="NEGATE")
                except Exception:
                    pass

                # 3) 该分类的 Top 产品（给出关键词主线明细）
                f.write(f"产品关键词明细（Top {detail_top_n_per_category}，按7天滚动广告花费）：\n\n")

                detail_view = cat_view.copy()
                try:
                    if "ad_spend_roll" in detail_view.columns:
                        detail_view = detail_view.sort_values("ad_spend_roll", ascending=False)
                    elif "广告花费" in detail_view.columns:
                        detail_view = detail_view.sort_values("广告花费", ascending=False)
                except Exception:
                    pass

                detail_asins: List[str] = []
                try:
                    detail_asins = [str(x).strip().upper() for x in detail_view["asin"].dropna().tolist() if str(x).strip()]
                except Exception:
                    detail_asins = []
                detail_asins = detail_asins[: int(detail_top_n_per_category)]

                for asin_norm in detail_asins:
                    if not asin_norm or asin_norm.lower() == "nan":
                        continue

                    pname = ""
                    try:
                        row0 = board[board["asin"].astype(str).str.upper().str.strip() == asin_norm].head(1)
                        if not row0.empty:
                            pname = str(row0.iloc[0].get("product_name", "") or "")
                    except Exception:
                        pname = ""

                    title = f"##### {asin_norm}"
                    if pname:
                        title += f" — {pname}"
                    f.write(title + "\n\n")

                    # 当前状态（生命周期看板行）
                    try:
                        row = board[board["asin"].astype(str).str.upper().str.strip() == asin_norm].copy()
                        if not row.empty:
                            cols_now = [c for c in cols_focus if c in row.columns]
                            f.write("当前状态：\n\n")
                            f.write(df_to_md_table(row[cols_now].head(1), max_rows=1))
                            f.write("\n\n")
                    except Exception:
                        pass

                    # 经营底座（自然月汇总）
                    if pa_asin is not None and not pa_asin.empty:
                        try:
                            row = pa_asin[pa_asin["asin_norm"] == asin_norm].copy()
                            if not row.empty:
                                cols = [c for c in ["销售额", "订单量", "Sessions", "广告花费", "广告销售额", "广告订单量", "毛利润", "tacos_total", "cvr_total"] if c in row.columns]
                                for c in cols:
                                    if c in {"tacos_total", "cvr_total"}:
                                        row[c] = pd.to_numeric(row[c], errors="coerce").round(4)
                                    else:
                                        row[c] = pd.to_numeric(row[c], errors="coerce").round(2)
                                f.write("经营底座（自然月汇总口径）：\n\n")
                                f.write(df_to_md_table(row[["asin_norm"] + cols].rename(columns={"asin_norm": "ASIN"}), max_rows=1))
                                f.write("\n\n")
                        except Exception:
                            pass

                    # 生命周期主口径窗口（动态周期）
                    if main_win is not None and not main_win.empty:
                        try:
                            row = main_win[main_win["asin_norm"] == asin_norm].copy()
                            if not row.empty:
                                keep = [
                                    c
                                    for c in [
                                        "window_type",
                                        "phase",
                                        "date_start",
                                        "date_end",
                                        "first_stock_date",
                                        "first_sale_in_stock_date",
                                        "prelaunch_days",
                                        "prelaunch_ad_spend",
                                        "oos_days",
                                        "sessions",
                                        "sales",
                                        "ad_spend",
                                        "profit",
                                        "tacos",
                                        "ad_acos",
                                        "ad_sales_share",
                                    ]
                                    if c in row.columns
                                ]
                                for c in ("sales", "ad_spend", "profit", "prelaunch_ad_spend"):
                                    if c in row.columns:
                                        row[c] = pd.to_numeric(row[c], errors="coerce").round(2)
                                for c in ("tacos", "ad_acos", "ad_sales_share"):
                                    if c in row.columns:
                                        row[c] = pd.to_numeric(row[c], errors="coerce").round(4)
                                f.write("生命周期主口径窗口（动态起点：首次可售）：\n\n")
                                f.write(df_to_md_table(row[keep].head(1), max_rows=1))
                                f.write("\n\n")
                        except Exception:
                            pass

                    # 动态日期范围：最近7/14/30天 vs 前7/14/30天（按 ASIN）
                    if cmp_win is not None and not cmp_win.empty:
                        try:
                            rows = cmp_win[cmp_win["asin_norm"] == asin_norm].copy()
                            if not rows.empty:
                                keep = [c for c in ["window_days", "delta_spend", "delta_sales", "delta_orders", "delta_sessions", "marginal_tacos", "marginal_ad_acos"] if c in rows.columns]
                                for c in keep:
                                    rows[c] = pd.to_numeric(rows[c], errors="coerce").round(4 if "marginal" in c else 2)
                                f.write("滚动环比（最近N天 vs 前N天）：\n\n")
                                f.write(df_to_md_table(rows[keep], max_rows=10))
                                f.write("\n\n")
                        except Exception:
                            pass

                    # 广告结构（关键词主线）：Targeting -> Search Term -> Campaign -> Placement
                    top_tg = _pick(asin_top_targetings, asin_norm)
                    if not top_tg.empty:
                        try:
                            view = top_tg.sort_values("spend", ascending=False).head(12)
                            cols = [c for c in ["ad_type", "campaign", "targeting", "match_type", "spend", "orders", "acos"] if c in view.columns]
                            if "match_type" in view.columns:
                                view["match_type"] = view["match_type"].fillna("").astype(str).str.strip()
                                view.loc[(view["match_type"] == "") | (view["match_type"].str.lower() == "nan"), "match_type"] = "N/A"
                            for c in ("spend",):
                                if c in view.columns:
                                    view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
                            if "orders" in view.columns:
                                view["orders"] = pd.to_numeric(view["orders"], errors="coerce").round(2)
                            if "acos" in view.columns:
                                view["acos"] = pd.to_numeric(view["acos"], errors="coerce").round(4)
                            f.write("广告关键词主线：Top Targeting（按花费，已映射到本 ASIN）：\n\n")
                            f.write(df_to_md_table(view[cols], max_rows=12))
                            f.write("\n\n")
                        except Exception:
                            pass

                    top_st = _pick(asin_top_search_terms, asin_norm)
                    if not top_st.empty:
                        try:
                            view = top_st.sort_values("spend", ascending=False).head(12)
                            cols = [c for c in ["ad_type", "campaign", "search_term", "match_type", "spend", "orders", "acos"] if c in view.columns]
                            if "match_type" in view.columns:
                                view["match_type"] = view["match_type"].fillna("").astype(str).str.strip()
                                view.loc[(view["match_type"] == "") | (view["match_type"].str.lower() == "nan"), "match_type"] = "N/A"
                            for c in ("spend",):
                                if c in view.columns:
                                    view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
                            if "orders" in view.columns:
                                view["orders"] = pd.to_numeric(view["orders"], errors="coerce").round(2)
                            if "acos" in view.columns:
                                view["acos"] = pd.to_numeric(view["acos"], errors="coerce").round(4)
                            f.write("广告结构：Top Search Term（按花费，已映射到本 ASIN）：\n\n")
                            f.write(df_to_md_table(view[cols], max_rows=12))
                            f.write("\n\n")
                        except Exception:
                            pass

                    top_camps = _pick(asin_top_campaigns, asin_norm)
                    if not top_camps.empty:
                        try:
                            view = top_camps.sort_values("spend", ascending=False).head(8)
                            cols = [c for c in ["ad_type", "campaign", "spend", "sales", "orders", "acos"] if c in view.columns]
                            for c in ("spend", "sales"):
                                if c in view.columns:
                                    view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
                            if "acos" in view.columns:
                                view["acos"] = pd.to_numeric(view["acos"], errors="coerce").round(4)
                            f.write("落地到活动：Top Campaign（按花费，已映射到本 ASIN）：\n\n")
                            f.write(df_to_md_table(view[cols], max_rows=8))
                            f.write("\n\n")
                        except Exception:
                            pass

                    top_pl = _pick(asin_top_placements, asin_norm)
                    if not top_pl.empty:
                        try:
                            view = top_pl.sort_values("spend", ascending=False).head(8)
                            cols = [c for c in ["ad_type", "campaign", "placement", "spend", "orders", "acos"] if c in view.columns]
                            for c in ("spend",):
                                if c in view.columns:
                                    view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
                            if "acos" in view.columns:
                                view["acos"] = pd.to_numeric(view["acos"], errors="coerce").round(4)
                            f.write("广告位：Top Placement（按花费，已映射到本 ASIN）：\n\n")
                            f.write(df_to_md_table(view[cols], max_rows=8))
                            f.write("\n\n")
                        except Exception:
                            pass

            """
            groups: List[Tuple[str, List[str]]] = []
            try:
                b2 = board.copy()
                b2["asin_norm"] = b2["asin"].astype(str).str.upper().str.strip()
                b2["product_category"] = b2.get("product_category", "").astype(str) if "product_category" in b2.columns else ""
                b2["product_category"] = b2["product_category"].fillna("").astype(str).str.strip()
                b2.loc[b2["product_category"].str.lower() == "nan", "product_category"] = ""
                b2.loc[b2["product_category"] == "", "product_category"] = "未分类"

                # 只对 focus_asins 分组，顺序保持“花费优先”的排序
                focus_norm = [str(a).strip().upper() for a in focus_asins if str(a).strip()]
                b2 = b2[b2["asin_norm"].isin(focus_norm)].copy()
                # category 顺序：按这 15 个里累计 ad_spend_roll 排序
                if "ad_spend_roll" in b2.columns:
                    cat_order = (
                        b2.groupby("product_category", dropna=False)["ad_spend_roll"]
                        .sum()
                        .reset_index(name="ad_spend_roll_sum")
                        .sort_values("ad_spend_roll_sum", ascending=False)["product_category"]
                        .tolist()
                    )
                else:
                    cat_order = sorted(b2["product_category"].dropna().unique().tolist())
                for cat in cat_order:
                    asins_in_cat = []
                    for a in focus_norm:
                        try:
                            r = b2[b2["asin_norm"] == a].head(1)
                            if r.empty:
                                continue
                            if str(r.iloc[0].get("product_category", "未分类")) == str(cat):
                                asins_in_cat.append(a)
                        except Exception:
                            continue
                    if asins_in_cat:
                        groups.append((str(cat), asins_in_cat))
            except Exception:
                groups = [("未分类", [str(a).strip().upper() for a in focus_asins if str(a).strip()])]

            for cat, asins_in_cat in groups:
                f.write(f"#### 分类：{cat}\n\n")
                for asin in asins_in_cat:
                    asin_norm = str(asin).strip().upper()
                    if not asin_norm or asin_norm.lower() == "nan":
                        continue
                    pname = ""
                    pcat = ""
                    try:
                        row = board[board["asin"].astype(str).str.upper().str.strip() == asin_norm].head(1)
                        if not row.empty:
                            pname = str(row.iloc[0].get("product_name", "") or "")
                            pcat = str(row.iloc[0].get("product_category", "") or "")
                    except Exception:
                        pname = ""
                        pcat = ""

                    title = f"##### {asin_norm}"
                    if pname:
                        title += f" — {pname}"
                    if pcat:
                        title += f"（{pcat}）"
                    f.write(title + "\n\n")

                # 当前状态（生命周期看板行）
                try:
                    row = board[board["asin"].astype(str).str.upper().str.strip() == asin_norm].copy()
                    if not row.empty:
                        cols_now = [c for c in cols_focus if c in row.columns]
                        f.write("当前状态：\n\n")
                        f.write(df_to_md_table(row[cols_now].head(1), max_rows=1))
                        f.write("\n\n")
                except Exception:
                    pass

                # 经营底座（自然月汇总）
                if pa_asin is not None and not pa_asin.empty:
                    try:
                        row = pa_asin[pa_asin["asin_norm"] == asin_norm].copy()
                        if not row.empty:
                            cols = [c for c in ["销售额", "订单量", "Sessions", "广告花费", "广告销售额", "广告订单量", "毛利润", "tacos_total", "cvr_total"] if c in row.columns]
                            # 四舍五入
                            for c in cols:
                                if c in {"tacos_total", "cvr_total"}:
                                    row[c] = pd.to_numeric(row[c], errors="coerce").round(4)
                                else:
                                    row[c] = pd.to_numeric(row[c], errors="coerce").round(2)
                            f.write("经营底座（自然月汇总口径）：\n\n")
                            f.write(df_to_md_table(row[["asin_norm"] + cols].rename(columns={"asin_norm": "ASIN"}), max_rows=1))
                            f.write("\n\n")
                    except Exception:
                        pass

                # 生命周期主口径窗口（动态周期）
                if main_win is not None and not main_win.empty:
                    try:
                        row = main_win[main_win["asin_norm"] == asin_norm].copy()
                        if not row.empty:
                            keep = [
                                c
                                for c in [
                                    "window_type",
                                    "phase",
                                    "date_start",
                                    "date_end",
                                    "first_stock_date",
                                    "first_sale_in_stock_date",
                                    "prelaunch_days",
                                    "prelaunch_ad_spend",
                                    "oos_days",
                                    "sessions",
                                    "sales",
                                    "ad_spend",
                                    "profit",
                                    "tacos",
                                    "ad_acos",
                                    "ad_sales_share",
                                ]
                                if c in row.columns
                            ]
                            for c in ("sales", "ad_spend", "profit", "prelaunch_ad_spend"):
                                if c in row.columns:
                                    row[c] = pd.to_numeric(row[c], errors="coerce").round(2)
                            for c in ("tacos", "ad_acos", "ad_sales_share"):
                                if c in row.columns:
                                    row[c] = pd.to_numeric(row[c], errors="coerce").round(4)
                            f.write("生命周期主口径窗口（动态起点：首次可售）：\n\n")
                            f.write(df_to_md_table(row[keep].head(1), max_rows=1))
                            f.write("\n\n")
                    except Exception:
                        pass

                # 生命周期滚动环比（7/14/30）
                if cmp_win is not None and not cmp_win.empty:
                    try:
                        rows = cmp_win[cmp_win["asin_norm"] == asin_norm].copy()
                        if not rows.empty:
                            if "window_days" in rows.columns:
                                rows = rows.sort_values("window_days")
                            keep = [c for c in ["window_days", "delta_spend", "delta_sales", "delta_orders", "delta_sessions", "marginal_tacos", "marginal_ad_acos"] if c in rows.columns]
                            for c in keep:
                                rows[c] = pd.to_numeric(rows[c], errors="coerce").round(4 if "marginal" in c else 2)
                            f.write("滚动环比（最近N天 vs 前N天）：\n\n")
                            f.write(df_to_md_table(rows[keep], max_rows=10))
                            f.write("\n\n")
                    except Exception:
                        pass

                # 广告结构（关键词主线）：Targeting -> Search Term -> Campaign -> Placement
                top_tg = _pick(asin_top_targetings, asin_norm)
                if not top_tg.empty:
                    try:
                        view = top_tg.sort_values("spend", ascending=False).head(12)
                        cols = [c for c in ["ad_type", "campaign", "targeting", "match_type", "spend", "orders", "acos"] if c in view.columns]
                        if "match_type" in view.columns:
                            view["match_type"] = view["match_type"].fillna("").astype(str).str.strip()
                            view.loc[(view["match_type"] == "") | (view["match_type"].str.lower() == "nan"), "match_type"] = "N/A"
                        for c in ("spend",):
                            if c in view.columns:
                                view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
                        if "orders" in view.columns:
                            view["orders"] = pd.to_numeric(view["orders"], errors="coerce").round(2)
                        if "acos" in view.columns:
                            view["acos"] = pd.to_numeric(view["acos"], errors="coerce").round(4)
                        f.write("广告关键词主线：Top Targeting（按花费，已映射到本 ASIN）：\n\n")
                        f.write(df_to_md_table(view[cols], max_rows=12))
                        f.write("\n\n")
                    except Exception:
                        pass

                top_st = _pick(asin_top_search_terms, asin_norm)
                if not top_st.empty:
                    try:
                        view = top_st.sort_values("spend", ascending=False).head(12)
                        cols = [c for c in ["ad_type", "campaign", "search_term", "match_type", "spend", "orders", "acos"] if c in view.columns]
                        if "match_type" in view.columns:
                            view["match_type"] = view["match_type"].fillna("").astype(str).str.strip()
                            view.loc[(view["match_type"] == "") | (view["match_type"].str.lower() == "nan"), "match_type"] = "N/A"
                        for c in ("spend",):
                            if c in view.columns:
                                view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
                        if "orders" in view.columns:
                            view["orders"] = pd.to_numeric(view["orders"], errors="coerce").round(2)
                        if "acos" in view.columns:
                            view["acos"] = pd.to_numeric(view["acos"], errors="coerce").round(4)
                        f.write("广告结构：Top Search Term（按花费，已映射到本 ASIN）：\n\n")
                        f.write(df_to_md_table(view[cols], max_rows=12))
                        f.write("\n\n")
                    except Exception:
                        pass

                top_camps = _pick(asin_top_campaigns, asin_norm)
                if not top_camps.empty:
                    try:
                        view = top_camps.sort_values("spend", ascending=False).head(8)
                        cols = [c for c in ["ad_type", "campaign", "spend", "sales", "orders", "acos"] if c in view.columns]
                        for c in ("spend", "sales"):
                            if c in view.columns:
                                view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
                        if "acos" in view.columns:
                            view["acos"] = pd.to_numeric(view["acos"], errors="coerce").round(4)
                        f.write("落地到活动：Top Campaign（按花费，已映射到本 ASIN）：\n\n")
                        f.write(df_to_md_table(view[cols], max_rows=8))
                        f.write("\n\n")
                    except Exception:
                        pass

                top_pl = _pick(asin_top_placements, asin_norm)
                if not top_pl.empty:
                    try:
                        view = top_pl.sort_values("spend", ascending=False).head(8)
                        cols = [c for c in ["ad_type", "campaign", "placement", "spend", "orders", "acos"] if c in view.columns]
                        for c in ("spend",):
                            if c in view.columns:
                                view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
                        if "acos" in view.columns:
                            view["acos"] = pd.to_numeric(view["acos"], errors="coerce").round(4)
                        f.write("广告位：Top Placement（按花费，已映射到本 ASIN）：\n\n")
                        f.write(df_to_md_table(view[cols], max_rows=8))
                        f.write("\n\n")
                    except Exception:
                        pass

            f.write("\n")

            """  # end of DEPRECATED block (see above)

        f.write("## 5) 趋势与结构（图表）\n\n")
        if trends_png:
            f.write(f"![trend_overview](../figures/{trends_png.name})\n\n")
        if tacos_png:
            f.write(f"![tacos_trend](../figures/{tacos_png.name})\n\n")
        if top_camp_png:
            f.write(f"![top_campaign_spend](../figures/{top_camp_png.name})\n\n")
        if top_place_png:
            f.write(f"![top_placement_spend](../figures/{top_place_png.name})\n\n")
        if top_tgt_png:
            f.write(f"![top_targeting_spend](../figures/{top_tgt_png.name})\n\n")
        if matrix_png:
            f.write(f"![search_term_matrix](../figures/{matrix_png.name})\n\n")

        # 动作候选：为了避免“广告组罗列太散”，这里做极简展示（不展示 ad_group）
        f.write("## 6) 动作候选清单（极简版：避免广告组刷屏）\n\n")
        if not p0_df.empty:
            f.write("### P0（优先处理：通常是浪费花费 -> 否词）\n\n")
            cols = [c for c in ["ad_type", "level", "action_type", "object_name", "campaign", "reason"] if c in p0_df.columns]
            f.write(df_to_md_table(p0_df[cols], max_rows=30))
            f.write("\n\n")
        if not p1_df.empty:
            f.write("### P1（效率优化/扩量：降价/加价/预算建议）\n\n")
            cols = [c for c in ["ad_type", "level", "action_type", "action_value", "object_name", "campaign", "reason"] if c in p1_df.columns]
            f.write(df_to_md_table(p1_df[cols], max_rows=30))
            f.write("\n\n")

        # 7) 更数据驱动的诊断（趋势/根因）
        camp_trends = diag.get("campaign_trends") if isinstance(diag, dict) else None
        asin_causes = diag.get("asin_root_causes") if isinstance(diag, dict) else None
        asin_stages = diag.get("asin_stages") if isinstance(diag, dict) else None
        camp_budget_map = diag.get("campaign_budget_map") if isinstance(diag, dict) else None
        transfer_plan = diag.get("budget_transfer_plan") if isinstance(diag, dict) else None
        has_lifecycle = lifecycle_board is not None and not lifecycle_board.empty
        if (
            (camp_trends and isinstance(camp_trends, list) and len(camp_trends) > 0)
            or (asin_causes and isinstance(asin_causes, list) and len(asin_causes) > 0)
            or (asin_stages and isinstance(asin_stages, list) and len(asin_stages) > 0)
            or (camp_budget_map and isinstance(camp_budget_map, list) and len(camp_budget_map) > 0)
            or (transfer_plan and isinstance(transfer_plan, dict) and isinstance(transfer_plan.get("transfers"), list) and len(transfer_plan.get("transfers")) > 0)
            or has_lifecycle
        ):
            f.write("## 7) 结构诊断（数据驱动：趋势/根因）\n\n")

            # 生命周期看板（动态周期）：帮助运营按“产品阶段”决定 KPI/动作重点
            if has_lifecycle:
                board = lifecycle_board.copy()
                for c in ("sales_roll", "sessions_roll", "ad_spend_roll", "profit_roll"):
                    if c in board.columns:
                        try:
                            board[c] = pd.to_numeric(board[c], errors="coerce").round(2)
                        except Exception:
                            pass
                for c in ("tacos_roll", "cvr_roll"):
                    if c in board.columns:
                        try:
                            board[c] = pd.to_numeric(board[c], errors="coerce").round(4)
                        except Exception:
                            pass
                cols = [
                    c
                    for c in [
                        "asin",
                        "current_phase",
                        "cycle_id",
                        "ad_spend_roll",
                        "sales_roll",
                        "tacos_roll",
                        "cvr_roll",
                        "inventory",
                        "flag_oos",
                    ]
                    if c in board.columns
                ]
                f.write("### 7.1 生命周期看板（按 ASIN 动态周期：当前处于什么阶段）\n\n")
                # 先给一个“阶段分布”，再给 top ASIN 明细，避免只看单品
                if "current_phase" in board.columns:
                    phase_cnt = board.groupby("current_phase", dropna=False).size().reset_index(name="asin_count").sort_values("asin_count", ascending=False)
                    f.write("#### 阶段分布（当前）\n\n")
                    f.write(df_to_md_table(phase_cnt, max_rows=50))
                    f.write("\n\n")
                f.write("#### Top ASIN（按 7天滚动广告花费/销售）\n\n")
                f.write(df_to_md_table(board, columns=cols, max_rows=25))
                f.write("\n\n")

                # 最近阶段切换（segments）：用来复盘“什么时候从 launch -> growth / mature -> decline”
                if lifecycle_segments is not None and not lifecycle_segments.empty:
                    seg = lifecycle_segments.copy()
                    cols2 = [
                        c
                        for c in [
                            "asin",
                            "cycle_id",
                            "phase",
                            "date_start",
                            "date_end",
                            "days",
                            "sales_sum",
                            "ad_spend_sum",
                            "tacos",
                            "inv_min",
                            "oos_days",
                        ]
                        if c in seg.columns
                    ]
                    try:
                        seg = seg.sort_values(["date_end", "ad_spend_sum"], ascending=[False, False])
                    except Exception:
                        pass
                    f.write("#### 最近阶段段落（便于复盘：按 date_end 倒序）\n\n")
                    f.write(df_to_md_table(seg, columns=cols2, max_rows=30))
                    f.write("\n\n")

                # 动态日期范围：最近7/14/30天 vs 前7/14/30天（按 ASIN）
                if lifecycle_windows is not None and not lifecycle_windows.empty and "window_type" in lifecycle_windows.columns:
                    win = lifecycle_windows.copy()

                    # 主口径：since_first_stock_to_date（更贴近“上架可售后”的生命周期）
                    main = win[win["window_type"] == "since_first_stock_to_date"].copy()
                    if main.empty:
                        main = win[win["window_type"] == "cycle_to_date"].copy()
                    if not main.empty:
                        f.write("#### 主口径窗口：since_first_stock_to_date（上架可售起点）\n\n")
                        try:
                            for c in ("sales", "ad_spend", "ad_sales", "profit", "prelaunch_ad_spend"):
                                if c in main.columns:
                                    main[c] = pd.to_numeric(main[c], errors="coerce").round(2)
                            for c in ("tacos", "ad_acos", "ad_orders_share", "ad_sales_share", "ad_ctr", "ad_cvr"):
                                if c in main.columns:
                                    main[c] = pd.to_numeric(main[c], errors="coerce").round(4)
                        except Exception:
                            pass
                        try:
                            if "ad_spend" in main.columns:
                                main = main.sort_values("ad_spend", ascending=False)
                        except Exception:
                            pass
                        cols_main = [
                            c
                            for c in [
                                "asin",
                                "product_name",
                                "phase",
                                "first_stock_date",
                                "first_sale_date",
                                "first_sale_in_stock_date",
                                "days_stock_to_first_sale",
                                "prelaunch_days",
                                "prelaunch_ad_spend",
                                "oos_days",
                                "oos_with_ad_spend_days",
                                "oos_with_sessions_days",
                                "presale_order_days",
                                "sessions",
                                "sales",
                                "ad_spend",
                                "tacos",
                                "ad_orders_share",
                                "ad_sales_share",
                                "ad_ctr",
                                "ad_cvr",
                            ]
                            if c in main.columns
                        ]
                        f.write(df_to_md_table(main, columns=cols_main, max_rows=25))
                        f.write("\n\n")

                    cmp = win[win["window_type"].astype(str).str.startswith("compare_")].copy()
                    if not cmp.empty:
                        # 只展示“最近7/14/30天”的 top 变化（按 delta_spend 倒序 + marginal_tacos）
                        f.write("#### 动态日期范围：ASIN 最近N天 vs 前N天（N=7/14/30）\n\n")
                        try:
                            cmp["delta_spend"] = pd.to_numeric(cmp.get("delta_spend"), errors="coerce")
                            cmp["marginal_tacos"] = pd.to_numeric(cmp.get("marginal_tacos"), errors="coerce")
                        except Exception:
                            pass
                        for n in sorted(cmp["window_days"].dropna().unique().tolist()):
                            try:
                                n_int = int(n)
                            except Exception:
                                continue
                            view = cmp[cmp["window_days"] == n].copy()
                            if view.empty:
                                continue
                            # 先看“花费变动最大的”，便于抓重点
                            view = view.sort_values(["delta_spend"], ascending=False).head(20)
                            cols3 = [
                                c
                                for c in [
                                    "asin",
                                    "phase",
                                    "delta_spend",
                                    "delta_sales",
                                    "delta_sessions",
                                    "delta_ad_clicks",
                                    "marginal_tacos",
                                    "marginal_ad_acos",
                                    "spend_prev",
                                    "spend_recent",
                                    "sales_prev",
                                    "sales_recent",
                                ]
                                if c in view.columns
                            ]
                            f.write(f"##### 最近{n_int}天 vs 前{n_int}天（ASIN）\n\n")
                            f.write(df_to_md_table(view, columns=cols3, max_rows=20))
                            f.write("\n\n")

            if camp_trends and isinstance(camp_trends, list):
                df = pd.DataFrame(camp_trends)
                # 控制展示列（避免太长）
                cols = [c for c in ["type", "severity", "ad_type", "campaign", "spend_prev", "spend_recent", "acos_prev", "acos_recent", "suggestion"] if c in df.columns]
                f.write("### 7.2 活动趋势告警/机会（最近窗口 vs 前窗口）\n\n")
                f.write(df_to_md_table(df, columns=cols, max_rows=20))
                f.write("\n\n")

            if asin_causes and isinstance(asin_causes, list):
                df = pd.DataFrame(asin_causes)
                cols = [c for c in ["asin", "tags", "severity", "inventory", "sessions", "cvr", "ad_spend", "ad_orders", "ad_acos", "refund_rate", "rating", "category"] if c in df.columns]
                f.write("### 7.3 ASIN 根因诊断（结合库存/转化/评分/退款/广告）\n\n")
                f.write(df_to_md_table(df, columns=cols, max_rows=25))
                f.write("\n\n")

            # 利润健康度：解释“为什么没有可放量池”
            profit_health = diag.get("profit_health") if isinstance(diag, dict) else None
            unlock_scale = diag.get("unlock_scale_plan") if isinstance(diag, dict) else None
            if profit_health and isinstance(profit_health, dict):
                f.write("### 7.3.1 利润健康度（解释放量池是否存在）\n\n")
                try:
                    rows = [
                        {
                            "asin_count": profit_health.get("asin_count", 0),
                            "profit_mode_inferred": profit_health.get("profit_mode_inferred", ""),
                            "profit_before_ads_positive_asin": profit_health.get("profit_before_ads_positive_asin", 0),
                            "profit_after_ads_positive_asin": profit_health.get("profit_after_ads_positive_asin", 0),
                            "reduce_asin": profit_health.get("reduce_asin", 0),
                            "scale_asin": profit_health.get("scale_asin", 0),
                            "blockers": str(profit_health.get("blockers", {})),
                        }
                    ]
                    f.write(df_to_md_table(pd.DataFrame(rows)))
                    f.write("\n\n")
                except Exception:
                    f.write("_利润健康度生成失败_\n\n")

                if unlock_scale and isinstance(unlock_scale, list) and len(unlock_scale) > 0:
                    f.write("### 7.3.2 解锁放量池：优先修哪些 ASIN\n\n")
                    dfu = pd.DataFrame(unlock_scale)
                    cols = [c for c in ["priority", "asin", "ad_spend", "profit_before_ads", "profit_after_ads", "max_ad_spend_by_profit", "inventory", "gap_usd", "fix"] if c in dfu.columns]
                    f.write(df_to_md_table(dfu, columns=cols, max_rows=25))
                    f.write("\n\n")
                unlock_tasks = diag.get("unlock_tasks") if isinstance(diag, dict) else None
                if unlock_tasks and isinstance(unlock_tasks, list) and len(unlock_tasks) > 0:
                    f.write("### 7.3.3 解锁任务清单（可分工执行）\n\n")
                    dft = pd.DataFrame(unlock_tasks)
                    for c in ("budget_gap_usd_est", "profit_gap_usd_est"):
                        if c in dft.columns:
                            try:
                                dft[c] = pd.to_numeric(dft[c], errors="coerce").round(2)
                            except Exception:
                                pass
                    cols = [c for c in ["priority", "asin", "task_type", "owner", "budget_gap_usd_est", "profit_gap_usd_est", "target"] if c in dft.columns]
                    f.write(df_to_md_table(dft, columns=cols, max_rows=30))
                    f.write("\n\n")

            # 多窗口（7/14/30）增量效率洞察
            temporal = diag.get("temporal") if isinstance(diag, dict) else None
            if temporal and isinstance(temporal, dict):
                camp_rows = temporal.get("campaign_windows")
                tgt_rows = temporal.get("targeting_windows")
                if isinstance(camp_rows, list) and len(camp_rows) > 0:
                    f.write("### 7.3.4 多窗口对比（7/14/30天）：增量效率与趋势信号\n\n")
                    dfc = pd.DataFrame(camp_rows)
                    # 只展示最近窗口（window_days=7/14/30）的 Top 信号（按 score）
                    for w in sorted(dfc["window_days"].dropna().unique().tolist())[:10]:
                        view = dfc[dfc["window_days"] == w].copy()
                        if view.empty:
                            continue
                        view = view.sort_values("score", ascending=False).head(15)
                        cols = [
                            c
                            for c in [
                                "window_days",
                                "ad_type",
                                "campaign",
                                "signal",
                                "score",
                                "spend_prev",
                                "spend_recent",
                                "sales_prev",
                                "sales_recent",
                                "delta_spend",
                                "delta_sales",
                                "marginal_acos",
                            ]
                            if c in view.columns
                        ]
                        f.write(f"#### 最近{int(w)}天 vs 前{int(w)}天（Campaign）\n\n")
                        f.write(df_to_md_table(view, columns=cols, max_rows=15))
                        f.write("\n\n")
                if isinstance(tgt_rows, list) and len(tgt_rows) > 0:
                    dft = pd.DataFrame(tgt_rows)
                    for w in sorted(dft["window_days"].dropna().unique().tolist())[:10]:
                        view = dft[dft["window_days"] == w].copy()
                        if view.empty:
                            continue
                        view = view.sort_values("score", ascending=False).head(15)
                        cols = [
                            c
                            for c in [
                                "window_days",
                                "ad_type",
                                "targeting",
                                "signal",
                                "score",
                                "spend_prev",
                                "spend_recent",
                                "sales_prev",
                                "sales_recent",
                                "delta_spend",
                                "delta_sales",
                                "marginal_acos",
                            ]
                            if c in view.columns
                        ]
                        f.write(f"#### 最近{int(w)}天 vs 前{int(w)}天（Targeting）\n\n")
                        f.write(df_to_md_table(view, columns=cols, max_rows=15))
                        f.write("\n\n")

            if asin_stages and isinstance(asin_stages, list):
                df = pd.DataFrame(asin_stages)
                # 可读性：把比例类字段做一次四舍五入，避免报告里小数太长
                for c in ("tacos", "target_tacos_by_margin", "gross_margin", "ad_share", "refund_rate"):
                    if c in df.columns:
                        try:
                            df[c] = pd.to_numeric(df[c], errors="coerce").round(4)
                        except Exception:
                            pass
                for c in ("ad_spend", "max_ad_spend_by_profit", "profit_after_ads"):
                    if c in df.columns:
                        try:
                            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
                        except Exception:
                            pass
                cols = [
                    c
                    for c in [
                        "asin",
                        "stage",
                        "direction",
                        "ad_spend",
                        "max_ad_spend_by_profit",
                        "tacos",
                        "target_tacos_by_margin",
                        "gross_margin",
                        "profit_after_ads",
                        "ad_share",
                        "inventory",
                        "reasons",
                    ]
                    if c in df.columns
                ]
                f.write("### 7.4 ASIN 阶段（按毛利承受度：前期拉流量/中期放量/后期控投放）\n\n")
                f.write(df_to_md_table(df, columns=cols, max_rows=25))
                f.write("\n\n")

            if camp_budget_map and isinstance(camp_budget_map, list):
                df = pd.DataFrame(camp_budget_map)
                # 可读性：比例/金额做简化
                for c in ("camp_acos", "reduce_spend_share", "scale_spend_share", "unknown_spend_share"):
                    if c in df.columns:
                        try:
                            df[c] = pd.to_numeric(df[c], errors="coerce").round(4)
                        except Exception:
                            pass
                for c in ("camp_spend", "camp_sales"):
                    if c in df.columns:
                        try:
                            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
                        except Exception:
                            pass
                cols = [
                    c
                    for c in [
                        "ad_type",
                        "campaign",
                        "action",
                        "suggested_budget_change_pct",
                        "severity",
                        "camp_spend",
                        "camp_sales",
                        "camp_orders",
                        "camp_acos",
                        "reduce_spend_share",
                        "scale_spend_share",
                        "unknown_spend_share",
                        "top_reduce_asin_hint",
                        "top_scale_asin_hint",
                        "suggestion",
                    ]
                    if c in df.columns
                ]
                f.write("### 7.5 预算迁移图谱（ASIN毛利承受度 → Campaign 预算方向）\n\n")
                f.write(df_to_md_table(df, columns=cols, max_rows=25))
                f.write("\n\n")

            # 预算净迁移表（reduce -> scale）
            if transfer_plan and isinstance(transfer_plan, dict):
                transfers = transfer_plan.get("transfers")
                f.write("### 7.6 预算净迁移表（从控量活动挪出 → 加到可放量活动）\n\n")
                if isinstance(transfers, list) and len(transfers) > 0:
                    df = pd.DataFrame(transfers)
                    for c in ("amount_usd_estimated", "from_spend", "to_spend"):
                        if c in df.columns:
                            try:
                                df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
                            except Exception:
                                pass
                    cols = [
                        c
                        for c in [
                            "from_ad_type",
                            "from_campaign",
                            "from_asin_hint",
                            "to_ad_type",
                            "to_campaign",
                            "to_asin_hint",
                            "amount_usd_estimated",
                            "note",
                        ]
                        if c in df.columns
                    ]
                    f.write(df_to_md_table(df, columns=cols, max_rows=30))
                    f.write("\n\n")
                else:
                    # 没有可迁移组合时也给出原因/汇总，避免运营困惑
                    reduce_n = int(transfer_plan.get("reduce_candidates", 0) or 0)
                    scale_n = int(transfer_plan.get("scale_candidates", 0) or 0)
                    f.write(f"_本期没有生成可迁移组合：reduce候选={reduce_n}，scale候选={scale_n}。_\n\n")

                    # 给出“控量清单/放量清单”作为替代执行入口
                    cuts = transfer_plan.get("cuts")
                    adds = transfer_plan.get("adds")
                    if isinstance(cuts, list) and len(cuts) > 0:
                        f.write("#### 7.6.1 控量清单（建议先降/限额）\n\n")
                        dfc = pd.DataFrame(cuts)
                        cols = [c for c in ["ad_type", "campaign", "cut_usd_estimated", "camp_spend", "severity", "asin_hint"] if c in dfc.columns]
                        f.write(df_to_md_table(dfc, columns=cols, max_rows=20))
                        f.write("\n\n")
                    if isinstance(adds, list) and len(adds) > 0:
                        f.write("#### 7.6.2 放量清单（可加码候选）\n\n")
                        dfa = pd.DataFrame(adds)
                        cols = [c for c in ["ad_type", "campaign", "add_usd_estimated", "camp_spend", "severity", "asin_hint"] if c in dfa.columns]
                        f.write(df_to_md_table(dfa, columns=cols, max_rows=20))
                        f.write("\n\n")
                    savings = transfer_plan.get("savings")
                    if isinstance(savings, list) and len(savings) > 0:
                        f.write("#### 7.6.3 预算回收清单（回收到 RESERVE）\n\n")
                        dfs = pd.DataFrame(savings)
                        cols = [c for c in ["from_ad_type", "from_campaign", "amount_usd_estimated", "to_bucket", "from_asin_hint"] if c in dfs.columns]
                        f.write(df_to_md_table(dfs, columns=cols, max_rows=20))
                        f.write("\n\n")

                # 汇总提示
                try:
                    unalloc = float(transfer_plan.get("unallocated_reduce_usd_estimated", 0.0) or 0.0)
                    unmet = float(transfer_plan.get("unmet_scale_usd_estimated", 0.0) or 0.0)
                    f.write(f"_未分配的可挪出金额(估算)：${unalloc:.2f}；未满足的放量需求(估算)：${unmet:.2f}_\n\n")
                except Exception:
                    f.write("\n\n")

        f.write("---\n")
        f.write(f"输出目录：{shop_dir}\n")

    # report 写完后，再落地运营动作表（避免影响 report 生成主流程）
    try:
        if ops_rows:
            ops_df = pd.DataFrame(ops_rows).copy()
            if not ops_df.empty:
                # 清洗：空 asin 去掉
                if "asin" in ops_df.columns:
                    ops_df["asin"] = ops_df["asin"].astype(str).str.upper().str.strip()
                    ops_df = ops_df[(ops_df["asin"] != "") & (ops_df["asin"].str.lower() != "nan")].copy()
                if "spend" in ops_df.columns:
                    ops_df["spend"] = pd.to_numeric(ops_df["spend"], errors="coerce").fillna(0.0)
                # 排序：分类 -> 优先级 -> 花费
                if all(c in ops_df.columns for c in ["product_category", "priority", "spend"]):
                    ops_df = ops_df.sort_values(["product_category", "priority", "spend"], ascending=[True, True, False])
                out_path = ops_dir / "actions.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ops_df.to_csv(out_path, index=False, encoding="utf-8-sig")

                # 运营主手册（Excel）：把动作 + ASIN 动态窗口对比拼在一起，方便运营筛选
                try:
                    compares_wide = _build_asin_compares_wide(lifecycle_windows)
                    playbook = ops_df.copy()
                    if compares_wide is not None and not compares_wide.empty and "asin" in playbook.columns:
                        playbook = playbook.merge(compares_wide, on="asin", how="left")

                    # 补充 ASIN 当前状态（库存/近7天经营滚动等），让运营不用回看 report
                    try:
                        if lifecycle_board is not None and (not lifecycle_board.empty) and "asin" in lifecycle_board.columns:
                            st = lifecycle_board.copy()
                            st["asin"] = st["asin"].astype(str).str.upper().str.strip()
                            keep = ["asin"]
                            for c in (
                                "inventory",
                                "ad_spend_roll",
                                "sales_roll",
                                "profit_roll",
                                "tacos_roll",
                                "flag_low_inventory",
                                "flag_oos",
                                "cycle_id",
                            ):
                                if c in st.columns:
                                    keep.append(c)
                            st = st[keep].drop_duplicates("asin").copy()
                            playbook = playbook.merge(st, on="asin", how="left")
                    except Exception:
                        pass

                    # 分类汇总（便于运营先看“哪一类问题最多/花费最大”）
                    try:
                        sum_df = playbook.copy()
                        for c in ("spend", "sales", "orders", "clicks"):
                            if c in sum_df.columns:
                                sum_df[c] = pd.to_numeric(sum_df[c], errors="coerce").fillna(0.0)
                        if "acos" in sum_df.columns:
                            sum_df["acos"] = pd.to_numeric(sum_df["acos"], errors="coerce").fillna(0.0)
                        gcols = [c for c in ["product_category", "layer", "action_group", "priority"] if c in sum_df.columns]
                        if gcols:
                            cat_summary = (
                                sum_df.groupby(gcols, dropna=False, as_index=False)
                                .agg(
                                    row_count=("shop", "size") if "shop" in sum_df.columns else ("asin", "size"),
                                    spend_sum=("spend", "sum") if "spend" in sum_df.columns else ("asin", "size"),
                                    sales_sum=("sales", "sum") if "sales" in sum_df.columns else ("asin", "size"),
                                    orders_sum=("orders", "sum") if "orders" in sum_df.columns else ("asin", "size"),
                                )
                                .copy()
                            )
                            cat_summary["acos_weighted"] = cat_summary.apply(lambda r: safe_div(r["spend_sum"], r["sales_sum"]), axis=1)
                            cat_summary = cat_summary.sort_values(["spend_sum", "row_count"], ascending=False)
                        else:
                            cat_summary = pd.DataFrame()
                    except Exception:
                        cat_summary = pd.DataFrame()
                    # 常用列放前面
                    preferred = [
                        "shop",
                        "product_category",
                        "asin",
                        "product_name",
                        "current_phase",
                        "cycle_id",
                        "inventory",
                        "flag_low_inventory",
                        "flag_oos",
                        "sales_roll",
                        "ad_spend_roll",
                        "tacos_roll",
                        "profit_roll",
                        "layer",
                        "action_group",
                        "priority",
                        "ad_type",
                        "campaign",
                        "match_type",
                        "targeting",
                        "search_term",
                        "spend",
                        "clicks",
                        "orders",
                        "sales",
                        "acos",
                        "cpc",
                        "cvr",
                        "signal",
                    ]
                    rest = [c for c in playbook.columns if c not in preferred]
                    playbook = playbook[[c for c in preferred if c in playbook.columns] + rest]

                    xlsx_path = ops_dir / "keyword_playbook.xlsx"
                    # 写一个简短说明 sheet（减少误解）
                    note = pd.DataFrame(
                        [
                            {
                                "说明": "这是给运营用的主手册：按 商品分类→ASIN→关键词层（Targeting/SearchTerm）汇总，并拼接 ASIN 的 7/14/30 滚动环比（delta/marginal）。",
                                "TopN口径": "关键词层是候选队列（TopN+规则筛出）；ASIN滚动环比来自 lifecycle_windows（compare_7d/14d/30d）。",
                                "字段提示": "c7_/c14_/c30_ 前缀表示对应窗口的 delta_* / marginal_* 等。",
                            }
                        ]
                    )
                    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
                        playbook.to_excel(w, index=False, sheet_name="playbook")
                        note.to_excel(w, index=False, sheet_name="README")
                        if "cat_summary" in locals() and cat_summary is not None and (not cat_summary.empty):
                            cat_summary.to_excel(w, index=False, sheet_name="category_summary")
                except Exception:
                    # xlsx 写失败也不要影响主流程：退化输出 csv
                    try:
                        pb = playbook if "playbook" in locals() else ops_df
                        pb.to_csv(ops_dir / "keyword_playbook.csv", index=False, encoding="utf-8-sig")
                    except Exception:
                        pass

        # Campaign 可执行清单（CSV）：调广告按 campaign 的入口
        try:
            camp_ops = _build_campaign_ops(
                shop=shop,
                camp=camp,
                cfg=cfg,
                lifecycle_board=lifecycle_board,
                product_listing_shop=product_listing_shop,
                asin_top_campaigns=asin_top_campaigns,
                asin_top_targetings=asin_top_targetings,
                asin_top_search_terms=asin_top_search_terms,
                policy=policy,
                windows_days=list(policy.campaign_windows_days),
            )
            if camp_ops is not None and not camp_ops.empty:
                camp_ops.to_csv(ops_dir / "campaign_ops.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass
    except Exception:
        pass

    # 兼容旧路径：之前的全量报告在 reports/report.md
    # 现在把它放到 ai/report.md，reports/report.md 写一个“跳转说明”，避免你/运营误打开长文档。
    try:
        if report_path is not None and report_path.exists():
            stub_path = reports_dir / "report.md"
            stub_path.write_text(
                "\n".join(
                    [
                        f"# {shop} 全量深挖报告（已迁移）",
                        "",
                        "- 新路径：`../ai/report.md`",
                        "- 运营建议只看：`dashboard.md`（聚焦版）",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
    except Exception:
        pass

    return report_path
