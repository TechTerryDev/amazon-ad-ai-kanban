# -*- coding: utf-8 -*-
"""
ASIN × 广告结构联动（把广告“按活动/投放/搜索词”与产品 ASIN 绑定起来）。

为什么需要这一层？
- 广告报表（search_term/targeting/placement）多数没有直接 ASIN 列；
- 但 advertised_product 报表有 ASIN ↔ campaign/ad_group 的关系；
- 我们用 advertised_product 来做“权重分摊”，把 search_term/targeting 的指标合理分配到 ASIN，
  这样后续就能在“单品生命周期窗口”里讨论“该 ASIN 的广告结构怎么调”。

重要原则：
- 这里不做任何“主观动作建议”，只做确定性计算与结构化输出；
- 分摊逻辑可复盘：默认用 spend 权重（同一 campaign(+ad_group) 下各 ASIN 的 spend 占比）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from core.schema import CAN
from core.utils import safe_div


_NUM_RE = re.compile(r"^\d+$")


def _norm_asin(value: object) -> str:
    try:
        s = str(value).strip().upper()
        return "" if not s or s.lower() == "nan" else s
    except Exception:
        return ""


def _norm_str(value: object) -> str:
    try:
        s = str(value).strip()
        return "" if not s or s.lower() == "nan" else s
    except Exception:
        return ""


def _coerce_required(df: pd.DataFrame, required: Sequence[str]) -> bool:
    return df is not None and (not df.empty) and all(c in df.columns for c in required)


def _is_numeric_like(s: str) -> bool:
    return bool(_NUM_RE.match(str(s).strip()))


def build_ad_product_daily(
    advertised_product: pd.DataFrame,
) -> pd.DataFrame:
    """
    把 advertised_product（广告产品报告）转成“按日、按 ASIN、按活动/广告组”的明细聚合。

    输出列（至少）：
    - shop, date, ad_type, campaign, ad_group, asin
    - impressions, clicks, spend, sales, orders
    """
    need = [CAN.shop, CAN.date, CAN.ad_type, CAN.campaign, CAN.ad_group, CAN.asin]
    if not _coerce_required(advertised_product, need):
        return pd.DataFrame()

    df = advertised_product.copy()
    df[CAN.asin] = df[CAN.asin].apply(_norm_asin)
    df[CAN.campaign] = df[CAN.campaign].apply(_norm_str)
    df[CAN.ad_group] = df[CAN.ad_group].apply(_norm_str)
    df = df[(df[CAN.asin] != "") & (df[CAN.campaign] != "")]

    # 数值列缺失就补 0，避免 groupby 报错
    for col in (CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders):
        if col not in df.columns:
            df[col] = 0.0

    gcols = [CAN.shop, CAN.date, CAN.ad_type, CAN.campaign, CAN.ad_group, CAN.asin]
    out = (
        df.groupby(gcols, dropna=False, as_index=False)
        .agg(
            impressions=(CAN.impressions, "sum"),
            clicks=(CAN.clicks, "sum"),
            spend=(CAN.spend, "sum"),
            sales=(CAN.sales, "sum"),
            orders=(CAN.orders, "sum"),
        )
        .copy()
    )
    return out


def build_asin_campaign_map(ad_product_daily: pd.DataFrame) -> pd.DataFrame:
    """
    ASIN ↔ campaign/ad_group 的累计映射表（用于人工查看/给 AI 做上下文）。
    """
    need = [CAN.shop, CAN.ad_type, CAN.campaign, CAN.ad_group, CAN.asin]
    if not _coerce_required(ad_product_daily, need):
        return pd.DataFrame()

    df = ad_product_daily.copy()
    for col in (CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders):
        if col not in df.columns:
            df[col] = 0.0

    gcols = [CAN.shop, CAN.ad_type, CAN.campaign, CAN.ad_group, CAN.asin]
    agg = (
        df.groupby(gcols, dropna=False, as_index=False)
        .agg(
            impressions=(CAN.impressions, "sum"),
            clicks=(CAN.clicks, "sum"),
            spend=(CAN.spend, "sum"),
            sales=(CAN.sales, "sum"),
            orders=(CAN.orders, "sum"),
        )
        .copy()
    )
    agg["acos"] = agg.apply(lambda r: safe_div(r["spend"], r["sales"]), axis=1)
    agg["ctr"] = agg.apply(lambda r: safe_div(r["clicks"], r["impressions"]), axis=1)
    agg["cvr"] = agg.apply(lambda r: safe_div(r["orders"], r["clicks"]), axis=1)
    return agg


def build_weight_table(
    ad_product_daily: pd.DataFrame,
    key_cols: Sequence[str],
    weight_col: str = CAN.spend,
) -> pd.DataFrame:
    """
    从 advertised_product 的明细聚合里构造“分摊权重表”：
    - key_cols: 用哪些维度来定义一个“可分摊桶”（例如 shop+ad_type+date+campaign+ad_group）
    - weight_col: 默认按 spend 权重

    输出列：
    - key_cols + asin + weight
    """
    need = list(key_cols) + [CAN.asin]
    if not _coerce_required(ad_product_daily, need):
        return pd.DataFrame()
    df = ad_product_daily.copy()
    if weight_col not in df.columns:
        df[weight_col] = 0.0

    gcols = list(key_cols) + [CAN.asin]
    agg = df.groupby(gcols, dropna=False, as_index=False).agg(weight_base=(weight_col, "sum")).copy()
    if agg.empty:
        return pd.DataFrame()

    # 每个 key 的总权重基数
    k = list(key_cols)
    totals = agg.groupby(k, dropna=False, as_index=False).agg(total_base=("weight_base", "sum"), asin_cnt=(CAN.asin, "nunique")).copy()
    merged = agg.merge(totals, on=k, how="left")

    def _weight(row: pd.Series) -> float:
        total = float(row.get("total_base", 0.0) or 0.0)
        base = float(row.get("weight_base", 0.0) or 0.0)
        cnt = int(row.get("asin_cnt", 0) or 0)
        if total > 0:
            return safe_div(base, total)
        # total=0：退化为平均分摊（避免全部丢失）
        return safe_div(1.0, float(cnt)) if cnt > 0 else 0.0

    merged["weight"] = merged.apply(_weight, axis=1)
    keep = list(key_cols) + [CAN.asin, "weight"]
    return merged[keep].copy()


@dataclass(frozen=True)
class WeightJoinSpec:
    """
    一次分摊尝试：
    - join_cols: 明细表与权重表的 join key
    - weights: 权重表（包含 join_cols + asin + weight）
    """

    join_cols: Tuple[str, ...]
    weights: pd.DataFrame


def allocate_detail_to_asin(
    detail: pd.DataFrame,
    specs: List[WeightJoinSpec],
    metric_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    把 search_term/targeting/placement 这类“无 ASIN 列”的明细，按 specs 顺序做权重分摊。

    算法：
    - 先按最精细的 join 试一次（例如包含 date + campaign + ad_group）
    - 对“没匹配到权重”的行，再按更粗的 join 逐层回退
    - 最终输出包含 asin 的明细，并把指标列乘以 weight
    """
    if detail is None or detail.empty:
        return pd.DataFrame()

    metric_cols = list(metric_cols or [CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders])
    df = detail.copy()

    for c in metric_cols:
        if c not in df.columns:
            df[c] = 0.0

    # 只保留 join 需要的列 + 指标列，减少内存
    out_frames: List[pd.DataFrame] = []
    remaining = df.copy()

    for spec in specs:
        if remaining.empty:
            break
        join_cols = list(spec.join_cols)
        weights = spec.weights
        if weights is None or weights.empty:
            continue
        if any(c not in remaining.columns for c in join_cols):
            continue
        if any(c not in weights.columns for c in (list(join_cols) + [CAN.asin, "weight"])):
            continue

        merged = remaining.merge(weights, on=join_cols, how="left")
        matched = merged[merged[CAN.asin].notna()].copy()
        if not matched.empty:
            matched["weight"] = matched["weight"].fillna(0.0)
            for c in metric_cols:
                matched[c] = matched[c].astype(float) * matched["weight"].astype(float)
            out_frames.append(matched.drop(columns=["weight"]))

        # 未匹配的行留给下一轮回退
        remaining = merged[merged[CAN.asin].isna()].copy()
        # 清理 merge 引入列，避免下一轮重复列
        drop_cols = [c for c in [CAN.asin, "weight"] if c in remaining.columns]
        if drop_cols:
            remaining = remaining.drop(columns=drop_cols)

    if out_frames:
        out = pd.concat(out_frames, ignore_index=True)
    else:
        out = pd.DataFrame()
    # asin 规范化
    if not out.empty and CAN.asin in out.columns:
        out[CAN.asin] = out[CAN.asin].apply(_norm_asin)
        out = out[out[CAN.asin] != ""].copy()
    return out


def build_weight_join_specs(
    ad_product_daily: pd.DataFrame,
) -> List[WeightJoinSpec]:
    """
    默认的多层回退策略：
    1) shop + ad_type + date + campaign + ad_group（最精细，能处理“同一活动多 ASIN”）
    2) shop + ad_type + date + campaign（广告组不一致时回退）
    3) shop + ad_type + campaign + ad_group（跨日回退）
    4) shop + ad_type + campaign（最粗回退）
    """
    # 如果 advertised_product 没有 ad_group（理论上会有），就跳过包含 ad_group 的层级
    has_ad_group = ad_product_daily is not None and (not ad_product_daily.empty) and (CAN.ad_group in ad_product_daily.columns)
    specs: List[WeightJoinSpec] = []

    lvl1 = (CAN.shop, CAN.ad_type, CAN.date, CAN.campaign, CAN.ad_group) if has_ad_group else (CAN.shop, CAN.ad_type, CAN.date, CAN.campaign)
    lvl2 = (CAN.shop, CAN.ad_type, CAN.date, CAN.campaign)
    lvl3 = (CAN.shop, CAN.ad_type, CAN.campaign, CAN.ad_group) if has_ad_group else (CAN.shop, CAN.ad_type, CAN.campaign)
    lvl4 = (CAN.shop, CAN.ad_type, CAN.campaign)

    for join_cols in [lvl1, lvl2, lvl3, lvl4]:
        wt = build_weight_table(ad_product_daily, join_cols, weight_col=CAN.spend)
        if wt is not None and not wt.empty:
            specs.append(WeightJoinSpec(join_cols=tuple(join_cols), weights=wt))
    return specs


def top_n_entities_by_asin(
    allocated: pd.DataFrame,
    entity_cols: Sequence[str],
    top_n: int = 20,
    min_spend: float = 1.0,
) -> pd.DataFrame:
    """
    给“分摊到 ASIN 的明细”做 TopN 聚合（search_term/targeting/placement 通用）。
    """
    need = [CAN.shop, CAN.ad_type, CAN.asin]
    if not _coerce_required(allocated, need):
        return pd.DataFrame()
    if any(c not in allocated.columns for c in entity_cols):
        return pd.DataFrame()

    df = allocated.copy()
    for col in (CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders):
        if col not in df.columns:
            df[col] = 0.0

    gcols = [CAN.shop, CAN.ad_type, CAN.asin] + list(entity_cols)
    agg = (
        df.groupby(gcols, dropna=False, as_index=False)
        .agg(
            impressions=(CAN.impressions, "sum"),
            clicks=(CAN.clicks, "sum"),
            spend=(CAN.spend, "sum"),
            sales=(CAN.sales, "sum"),
            orders=(CAN.orders, "sum"),
        )
        .copy()
    )
    agg = agg[agg["spend"] >= float(min_spend)].copy()
    if agg.empty:
        return pd.DataFrame()
    agg["acos"] = agg.apply(lambda r: safe_div(r["spend"], r["sales"]), axis=1)
    agg["ctr"] = agg.apply(lambda r: safe_div(r["clicks"], r["impressions"]), axis=1)
    agg["cvr"] = agg.apply(lambda r: safe_div(r["orders"], r["clicks"]), axis=1)

    # TopN（按 ASIN 分组）
    rows = []
    for (shop, ad_type, asin), g in agg.groupby([CAN.shop, CAN.ad_type, CAN.asin], dropna=False):
        view = g.sort_values(["spend", "sales", "orders"], ascending=False).head(int(top_n)).copy()
        rows.append(view)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
