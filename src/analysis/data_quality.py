# -*- coding: utf-8 -*-
"""
数据质量与维度覆盖盘点（给 AI/分析用）。

为什么需要：
- 你当前的痛点是“数据太多抓不到重点”，但反过来 AI 也容易“看错口径/臆造缺失维度”。
- 该模块把“能用哪些维度、缺哪些字段、哪些字段覆盖率低”显式化，作为后续优化规则/仪表盘的依据。

设计原则：
- 只做轻量统计（列存在性/空值占比/日期范围），避免跑数变慢
- 失败不崩：任何异常都不影响主流程输出
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.core.md import df_to_md_table
from src.core.schema import CAN


def _safe_date_range(df: Optional[pd.DataFrame], date_col: str) -> Tuple[str, str]:
    if df is None or df.empty or date_col not in df.columns:
        return ("", "")
    try:
        dmin = df[date_col].min()
        dmax = df[date_col].max()
        return (str(dmin) if dmin else "", str(dmax) if dmax else "")
    except Exception:
        return ("", "")


def _is_missing_series(s: pd.Series) -> pd.Series:
    """
    判定缺失：
    - NaN
    - 空字符串
    - "nan" 字符串
    """
    try:
        if s is None:
            return pd.Series([], dtype=bool)
        if s.dtype == object:
            ss = s.astype(str).fillna("").map(lambda x: "" if str(x).strip().lower() == "nan" else str(x))
            ss = ss.astype(str).str.strip()
            return s.isna() | (ss == "")
        return s.isna()
    except Exception:
        try:
            return s.isna()
        except Exception:
            return pd.Series([True] * int(len(s) if s is not None else 0))


def _field_stats(df: Optional[pd.DataFrame], col: str) -> Dict[str, object]:
    """
    返回字段统计：
    - exists: 是否存在
    - coverage: 非缺失占比（0~1）
    - missing_pct: 缺失占比（0~1）
    - sample_values: 取 3 个样例值（便于 AI 理解字段长什么样）
    """
    if df is None or df.empty or col not in df.columns:
        return {"exists": False, "coverage": 0.0, "missing_pct": 1.0, "sample_values": []}
    try:
        s = df[col]
        miss = _is_missing_series(s)
        total = int(len(s))
        missing = int(miss.sum()) if total else 0
        missing_pct = float(missing) / float(total) if total else 1.0
        coverage = 1.0 - missing_pct
        # 取样例（去掉缺失）
        try:
            samples = df.loc[~miss, col].head(3).tolist()
            samples = [str(x) for x in samples]
        except Exception:
            samples = []
        return {
            "exists": True,
            "coverage": round(float(coverage), 6),
            "missing_pct": round(float(missing_pct), 6),
            "sample_values": samples,
        }
    except Exception:
        return {"exists": True, "coverage": 0.0, "missing_pct": 1.0, "sample_values": []}


def _dataset_summary(
    name: str,
    df: Optional[pd.DataFrame],
    date_col: str = "",
    fields: Optional[List[str]] = None,
) -> Dict[str, object]:
    out: Dict[str, object] = {"name": name}
    try:
        rows = int(len(df)) if df is not None else 0
    except Exception:
        rows = 0
    out["rows"] = rows
    if date_col:
        ds, de = _safe_date_range(df, date_col)
        out["date_start"] = ds
        out["date_end"] = de
    if fields:
        out["fields"] = {c: _field_stats(df, c) for c in fields}
    else:
        out["fields"] = {}
    return out


def build_data_quality_report(
    shop: str,
    st: Optional[pd.DataFrame],
    tgt: Optional[pd.DataFrame],
    camp: Optional[pd.DataFrame],
    plc: Optional[pd.DataFrame],
    ap: Optional[pd.DataFrame],
    pp: Optional[pd.DataFrame],
    product_analysis_shop: Optional[pd.DataFrame],
    product_listing_shop: Optional[pd.DataFrame],
    lifecycle_board: Optional[pd.DataFrame],
) -> Dict[str, object]:
    """
    生成单店铺的数据质量/维度覆盖报告（JSON）。
    """
    now = dt.datetime.now().isoformat(timespec="seconds")

    # 关键维度：你当前的目标是“广告调整要结合产品立体数据”
    # 因此这里优先检查：销量/订单/Sessions、广告花费/广告销售/广告订单、库存、分类、生命周期输出是否完整。
    dims: List[Dict[str, object]] = []
    try:
        # 产品分析侧（经营结果）
        pa_cols = ["ASIN", "品名", "销售额", "订单量", "Sessions", "广告花费", "广告销售额", "广告订单量", "自然销售额", "自然订单量", "毛利润", CAN.date]
        dims.append(_dataset_summary("product_analysis", product_analysis_shop, date_col=CAN.date, fields=[c for c in pa_cols if c]))

        # productListing（库存/分类）
        pl_cols = ["ASIN", "品名", "商品分类", "可售", "采购成本(CNY)", "头程费用(CNY)", CAN.shop]
        dims.append(_dataset_summary("product_listing", product_listing_shop, date_col="", fields=[c for c in pl_cols if c]))

        # 广告侧（canonical schema）
        ad_base = [CAN.shop, CAN.date, CAN.ad_type, CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders]
        dims.append(_dataset_summary("ad_search_term", st, date_col=CAN.date, fields=ad_base + [CAN.search_term, CAN.match_type, CAN.campaign, CAN.ad_group]))
        dims.append(_dataset_summary("ad_targeting", tgt, date_col=CAN.date, fields=ad_base + [CAN.targeting, CAN.match_type, CAN.campaign, CAN.ad_group]))
        dims.append(_dataset_summary("ad_campaign", camp, date_col=CAN.date, fields=ad_base + [CAN.campaign]))
        dims.append(_dataset_summary("ad_placement", plc, date_col=CAN.date, fields=ad_base + [CAN.placement, CAN.campaign]))
        dims.append(_dataset_summary("ad_advertised_product", ap, date_col=CAN.date, fields=ad_base + [CAN.asin, CAN.sku, CAN.campaign]))
        # 已购买商品报告是“跨ASIN关联成交”视角，字段与其它广告报表不同（通常不含 impressions/clicks/spend/sales/orders）
        pp_cols = [CAN.shop, CAN.date, CAN.ad_type, CAN.asin, CAN.other_asin, CAN.campaign, "其他SKU销量", "其他SKU销售额"]
        dims.append(_dataset_summary("ad_purchased_product", pp, date_col=CAN.date, fields=pp_cols))

        # 生命周期输出（计算结果）
        lc_cols = ["asin", "product_name", "product_category", "current_phase", "cycle_id", "inventory", "flag_low_inventory", "flag_oos", "ad_spend_roll", "sales_roll", "tacos_roll", "profit_roll", "date"]
        dims.append(_dataset_summary("lifecycle_board", lifecycle_board, date_col="date", fields=lc_cols))
    except Exception:
        dims = []

    # 维度可用性（更适合快速判断“是否符合预期目标”）
    availability: Dict[str, object] = {}
    try:
        def _has(df: Optional[pd.DataFrame], cols: List[str]) -> bool:
            if df is None or df.empty:
                return False
            return all(c in df.columns for c in cols)

        availability = {
            "has_product_analysis": bool(product_analysis_shop is not None and not product_analysis_shop.empty),
            "has_product_listing": bool(product_listing_shop is not None and not product_listing_shop.empty),
            "has_category": _has(product_listing_shop, ["商品分类"]),
            "has_inventory": _has(product_listing_shop, ["可售"]),
            "has_sessions": _has(product_analysis_shop, ["Sessions"]),
            "has_sales_orders": _has(product_analysis_shop, ["销售额", "订单量"]),
            "has_ad_spend_sales_orders": _has(product_analysis_shop, ["广告花费", "广告销售额", "广告订单量"]),
            "has_organic_split_fields": _has(product_analysis_shop, ["自然销售额", "自然订单量"]),
            "has_search_term_report": bool(st is not None and not st.empty),
            "has_targeting_report": bool(tgt is not None and not tgt.empty),
            "has_campaign_report": bool(camp is not None and not camp.empty),
            "has_placement_report": bool(plc is not None and not plc.empty),
            "has_advertised_product_report": bool(ap is not None and not ap.empty),
            "has_purchased_product_report": bool(pp is not None and not pp.empty),
            "has_lifecycle_board": bool(lifecycle_board is not None and not lifecycle_board.empty),
        }
    except Exception:
        availability = {}

    return {
        "shop": shop,
        "generated_at": now,
        "availability": availability,
        "datasets": dims,
    }


def extract_data_quality_summary_lines(report: Dict[str, object], max_lines: int = 5) -> List[str]:
    """
    从 data_quality report 中提取 0~max_lines 条“口径提示/缺口风险”摘要。

    目标：
    - 让人不用读整张表，也能快速判断“哪些缺口可能影响结论”；
    - 这里仅做提示，不做强制判定；不应阻断主流程。
    """
    try:
        if report is None or not isinstance(report, dict):
            return []
        avail = report.get("availability", {}) if isinstance(report.get("availability", {}), dict) else {}
        ds = report.get("datasets", []) if isinstance(report.get("datasets", []), list) else []

        out: List[str] = []

        # 1) 自然拆分字段（自然销售额/自然订单量）缺失：会影响“自然 vs 广告拆分”解释
        if avail.get("has_product_analysis") and (not bool(avail.get("has_organic_split_fields"))):
            out.append("自然拆分字段不完整：将用 `总销售额-广告销售额` 推导 `organic_sales`（下限估算），并继续使用 `自然订单量`（如存在）。")

        # 2) 产品分析 vs 广告报表日期范围不一致：会导致“展示的时间范围”和“产品侧窗口”的上限不一致
        # 典型场景：你有广告日报到本周，但产品分析是按月导出，只到上月末。
        try:
            def _parse_date(s: object) -> Optional[dt.date]:
                try:
                    ss = str(s or "").strip()
                    if not ss or ss.lower() == "nan":
                        return None
                    return dt.date.fromisoformat(ss)
                except Exception:
                    return None

            ds_map = {str(x.get("name")): x for x in ds if isinstance(x, dict) and x.get("name")}
            pa = ds_map.get("product_analysis")
            pa_end = _parse_date(pa.get("date_end") if isinstance(pa, dict) else None)

            ad_ends: List[dt.date] = []
            for name, item in ds_map.items():
                if not isinstance(name, str) or not name.startswith("ad_"):
                    continue
                if not isinstance(item, dict):
                    continue
                de = _parse_date(item.get("date_end"))
                if de is not None:
                    ad_ends.append(de)

            ad_end = max(ad_ends) if ad_ends else None
            if pa_end is not None and ad_end is not None and pa_end < ad_end:
                out.append(
                    f"产品分析与广告报表日期范围不一致：产品分析截至 `{pa_end}`，广告报表截至 `{ad_end}`；涉及生命周期/自然拆分/利润等产品侧指标时，最近窗口以上限为 `{pa_end}`。"
                )
        except Exception:
            pass

        # 2) product_listing 分类/品名覆盖率低：会导致类目/产品展示大量落入“（未分类）”
        pl = next((x for x in ds if isinstance(x, dict) and x.get("name") == "product_listing"), None)
        if isinstance(pl, dict):
            fields = pl.get("fields", {}) if isinstance(pl.get("fields", {}), dict) else {}
            for key in ["商品分类", "品名"]:
                try:
                    v = fields.get(key, {}) if isinstance(fields.get(key, {}), dict) else {}
                    cov = float(v.get("coverage", 0.0) or 0.0)
                    if cov > 0 and cov < 0.6:
                        out.append(f"product_listing `{key}` 覆盖率偏低（coverage={cov:.3f}）：类目/品名展示会更多走兜底值。")
                        break
                except Exception:
                    continue

        # 3) search_term/targeting match_type 缺失：按 N/A 降级（仅影响拆分/筛选，不影响核心花费/销量口径）
        st = next((x for x in ds if isinstance(x, dict) and x.get("name") == "ad_search_term"), None)
        if isinstance(st, dict):
            fields = st.get("fields", {}) if isinstance(st.get("fields", {}), dict) else {}
            v = fields.get("match_type", {}) if isinstance(fields.get("match_type", {}), dict) else {}
            if v and (v.get("exists") is False):
                out.append("ad_search_term 缺少 `match_type` 字段：已按 N/A 降级（仅影响按匹配类型拆分/筛选）。")

        tgt = next((x for x in ds if isinstance(x, dict) and x.get("name") == "ad_targeting"), None)
        if isinstance(tgt, dict):
            fields = tgt.get("fields", {}) if isinstance(tgt.get("fields", {}), dict) else {}
            v = fields.get("match_type", {}) if isinstance(fields.get("match_type", {}), dict) else {}
            if v and (v.get("exists") is False):
                out.append("ad_targeting 缺少 `match_type` 字段：已按 N/A 降级（仅影响按匹配类型拆分/筛选）。")

        # 4) purchased_product：halo_candidates 依赖 “其他SKU销售额”
        pp = next((x for x in ds if isinstance(x, dict) and x.get("name") == "ad_purchased_product"), None)
        if isinstance(pp, dict):
            fields = pp.get("fields", {}) if isinstance(pp.get("fields", {}), dict) else {}
            v = fields.get("其他SKU销售额", {}) if isinstance(fields.get("其他SKU销售额", {}), dict) else {}
            if v and (v.get("exists") is False):
                out.append("ad_purchased_product 缺少 `其他SKU销售额`：将跳过 halo_candidates（跨ASIN关联成交）输出。")

        # 控制在 0~max_lines
        ml = int(max_lines or 0)
        if ml <= 0:
            return []
        out2: List[str] = []
        for x in out:
            s = str(x).strip()
            if not s:
                continue
            out2.append(s)
            if len(out2) >= ml:
                break
        return out2
    except Exception:
        return []


def write_data_quality_files(ai_dir: Path, report: Dict[str, object]) -> Tuple[Optional[Path], Optional[Path]]:
    """
    写入：
    - ai/data_quality.json（机器可读）
    - ai/data_quality.md（人类/AI 可读）
    """
    json_path = None
    md_path = None
    try:
        ai_dir.mkdir(parents=True, exist_ok=True)
        json_path = ai_dir / "data_quality.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        json_path = None

    try:
        md_path = ai_dir / "data_quality.md"
        shop = str(report.get("shop", "") or "")
        avail = report.get("availability", {}) if isinstance(report.get("availability", {}), dict) else {}
        ds = report.get("datasets", []) if isinstance(report.get("datasets", []), list) else []

        lines: List[str] = []
        lines.append(f"# {shop} 数据质量与维度覆盖盘点")
        lines.append("")
        lines.append(f"- generated_at: `{report.get('generated_at')}`")
        lines.append("")
        summary = extract_data_quality_summary_lines(report, max_lines=5)
        if summary:
            lines.append("## 0) 摘要（优先关注）")
            lines.append("")
            for s in summary:
                lines.append(f"- {s}")
            lines.append("")
        lines.append("## 1) 维度可用性（快速判断是否符合预期目标）")
        lines.append("")
        if avail:
            df = pd.DataFrame([avail])
            # 列太多时会很宽，但这是给 AI/你看的；保留即可
            lines.append(df_to_md_table(df, max_rows=5))
        else:
            lines.append("_无数据_")
        lines.append("")
        lines.append("## 2) 数据集概览（行数/日期范围/关键字段覆盖率）")
        lines.append("")
        rows = []
        for item in ds:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "name": item.get("name"),
                    "rows": item.get("rows"),
                    "date_start": item.get("date_start", ""),
                    "date_end": item.get("date_end", ""),
                }
            )
        lines.append(df_to_md_table(pd.DataFrame(rows), columns=["name", "rows", "date_start", "date_end"], max_rows=50))
        lines.append("")
        lines.append("## 3) 关键字段覆盖率（Top）")
        lines.append("")
        flat_rows: List[Dict[str, object]] = []
        for item in ds:
            if not isinstance(item, dict):
                continue
            fields = item.get("fields", {}) if isinstance(item.get("fields", {}), dict) else {}
            for k, v in fields.items():
                if not isinstance(v, dict):
                    continue
                flat_rows.append(
                    {
                        "dataset": item.get("name"),
                        "field": k,
                        "exists": v.get("exists"),
                        "coverage": v.get("coverage"),
                        "missing_pct": v.get("missing_pct"),
                    }
                )
        if flat_rows:
            df = pd.DataFrame(flat_rows)
            try:
                df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce").fillna(0.0)
                df = df.sort_values(["coverage"], ascending=[True])
            except Exception:
                pass
            lines.append(df_to_md_table(df, columns=["dataset", "field", "exists", "coverage", "missing_pct"], max_rows=80))
        else:
            lines.append("_无数据_")
        lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        md_path = None

    return (json_path, md_path)
