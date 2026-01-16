# -*- coding: utf-8 -*-
"""
把“可执行动作”抽象成结构化候选清单。

重要原则：
- 动作建议可回滚（给出观察窗口/回滚条件）
- 不让 AI 参与算数（AI 只负责解释/排序/写报告）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from core.config import StageConfig
from core.schema import CAN
from core.utils import json_dumps, safe_div


@dataclass(frozen=True)
class ActionCandidate:
    shop: str
    ad_type: str
    level: str  # search_term / targeting / campaign / placement / asin
    action_type: str  # NEGATE / BID_UP / BID_DOWN / BUDGET_UP / REVIEW
    action_value: str  # '+15%' / '-10%' / '' ...
    priority: str  # P0/P1/P2
    object_name: str
    campaign: str
    ad_group: str
    match_type: str
    date_start: str
    date_end: str
    reason: str
    evidence_json: str


def _date_range(df: pd.DataFrame) -> tuple[str, str]:
    if CAN.date not in df.columns or df.empty:
        return ("", "")
    dmin = df[CAN.date].min()
    dmax = df[CAN.date].max()
    return (str(dmin) if dmin else "", str(dmax) if dmax else "")


def generate_search_term_actions(df_search_term: pd.DataFrame, cfg: StageConfig) -> List[ActionCandidate]:
    """
    搜索词层面的三类动作：
    - 否词（浪费：有花费无单）
    - 降价（高 ACoS 且点击足够）
    - 扩量（低 ACoS 且有单）
    """
    if df_search_term.empty:
        return []

    # 清理：避免把 NaN/空串当成“可执行对象”
    if CAN.search_term in df_search_term.columns:
        df_search_term = df_search_term.copy()
        df_search_term[CAN.search_term] = df_search_term[CAN.search_term].astype(str).str.strip()
        df_search_term = df_search_term[(df_search_term[CAN.search_term] != "") & (df_search_term[CAN.search_term].str.lower() != "nan")]

    required = [CAN.shop, CAN.ad_type, CAN.date, CAN.search_term, CAN.campaign, CAN.ad_group]
    for col in required:
        if col not in df_search_term.columns:
            return []
    # 关键数值列缺失时，宁可不输出动作（避免误导），同时避免 groupby 直接崩溃
    essential_metrics = [CAN.clicks, CAN.spend, CAN.sales, CAN.orders]
    for col in essential_metrics:
        if col not in df_search_term.columns:
            return []
    if CAN.impressions not in df_search_term.columns:
        df_search_term = df_search_term.copy()
        df_search_term[CAN.impressions] = 0.0

    # match_type：部分报表/自动投放场景可能为空或缺列；此处统一成 N/A，避免输出 "nan"
    df_search_term = df_search_term.copy()
    if CAN.match_type not in df_search_term.columns:
        df_search_term[CAN.match_type] = "N/A"
    else:
        try:
            df_search_term[CAN.match_type] = df_search_term[CAN.match_type].fillna("").astype(str).str.strip()
            df_search_term.loc[
                (df_search_term[CAN.match_type] == "") | (df_search_term[CAN.match_type].str.lower() == "nan"),
                CAN.match_type,
            ] = "N/A"
        except Exception:
            df_search_term[CAN.match_type] = "N/A"

    gcols = [CAN.shop, CAN.ad_type, CAN.search_term, CAN.match_type, CAN.campaign, CAN.ad_group]
    agg = (
        df_search_term.groupby(gcols, dropna=False, as_index=False)
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
    agg["cvr"] = agg.apply(lambda r: safe_div(r["orders"], r["clicks"]), axis=1)

    date_start, date_end = _date_range(df_search_term)
    out: List[ActionCandidate] = []

    # 1) 否词：浪费（口径统一：订单=0 且 销售额=0；并要求点击达到门槛，避免样本太小）
    waste = agg[
        (agg["orders"] <= 0)
        & (agg["sales"] <= 0)
        & (agg["spend"] >= cfg.waste_spend)
        & (agg["clicks"] >= cfg.min_clicks)
    ]
    for _, r in waste.sort_values(["spend"], ascending=False).head(200).iterrows():
        out.append(
            ActionCandidate(
                shop=str(r[CAN.shop]),
                ad_type=str(r[CAN.ad_type]),
                level="search_term",
                action_type="NEGATE",
                action_value="",
                priority="P0",
                object_name=str(r[CAN.search_term]),
                campaign=str(r.get(CAN.campaign, "")),
                ad_group=str(r.get(CAN.ad_group, "")),
                match_type=str(r.get(CAN.match_type, "")),
                date_start=date_start,
                date_end=date_end,
                reason=f"点击≥{cfg.min_clicks}且花费≥{cfg.waste_spend}但订单=0且销售额=0（浪费花费）",
                evidence_json=json_dumps(
                    {
                        "impressions": r["impressions"],
                        "clicks": r["clicks"],
                        "spend": r["spend"],
                        "sales": r["sales"],
                        "orders": r["orders"],
                    }
                ),
            )
        )

    # 2) 降价：高 ACoS
    high_acos = agg[(agg["orders"] > 0) & (agg["clicks"] >= cfg.min_clicks) & (agg["acos"] > cfg.target_acos)]
    for _, r in high_acos.sort_values(["acos", "spend"], ascending=False).head(200).iterrows():
        out.append(
            ActionCandidate(
                shop=str(r[CAN.shop]),
                ad_type=str(r[CAN.ad_type]),
                level="search_term",
                action_type="REVIEW",
                action_value="",
                priority="P1",
                object_name=str(r[CAN.search_term]),
                campaign=str(r.get(CAN.campaign, "")),
                ad_group=str(r.get(CAN.ad_group, "")),
                match_type=str(r.get(CAN.match_type, "")),
                date_start=date_start,
                date_end=date_end,
                reason=f"搜索词无法直接调价，建议回到对应投放词/关键词处理（ACoS>{cfg.target_acos:.0%} 且点击充足）",
                evidence_json=json_dumps(
                    {
                        "clicks": r["clicks"],
                        "spend": r["spend"],
                        "sales": r["sales"],
                        "orders": r["orders"],
                        "acos": float(r["acos"]),
                        "cvr": float(r["cvr"]),
                    }
                ),
            )
        )

    # 3) 扩量：低 ACoS 且有单
    good = agg[(agg["orders"] >= 2) & (agg["clicks"] >= cfg.min_clicks) & (agg["acos"] > 0) & (agg["acos"] < cfg.target_acos * 0.7)]
    for _, r in good.sort_values(["orders", "sales"], ascending=False).head(200).iterrows():
        out.append(
            ActionCandidate(
                shop=str(r[CAN.shop]),
                ad_type=str(r[CAN.ad_type]),
                level="search_term",
                action_type="REVIEW",
                action_value="",
                priority="P1",
                object_name=str(r[CAN.search_term]),
                campaign=str(r.get(CAN.campaign, "")),
                ad_group=str(r.get(CAN.ad_group, "")),
                match_type=str(r.get(CAN.match_type, "")),
                date_start=date_start,
                date_end=date_end,
                reason=f"搜索词无法直接调价，建议回到对应投放词/关键词处理（低 ACoS 且已出单）",
                evidence_json=json_dumps(
                    {
                        "clicks": r["clicks"],
                        "spend": r["spend"],
                        "sales": r["sales"],
                        "orders": r["orders"],
                        "acos": float(r["acos"]),
                        "cvr": float(r["cvr"]),
                    }
                ),
            )
        )

    return out


def generate_campaign_budget_suggestions(df_campaign: pd.DataFrame, cfg: StageConfig) -> List[ActionCandidate]:
    """
    预算建议（不依赖“当前预算字段”也能给出优先级）：
    - 好活动：订单/销售多 + ACoS 低 -> 建议优先加预算/放量
    - 差活动：花费高 + ACoS 高 -> 建议先排查/限额
    """
    if df_campaign.empty:
        return []
    required = [CAN.shop, CAN.ad_type, CAN.campaign, CAN.date]
    for col in required:
        if col not in df_campaign.columns:
            return []
    # 关键数值列缺失时不输出预算建议（避免把缺字段当 0 导致误判）
    essential_metrics = [CAN.spend, CAN.sales, CAN.orders]
    for col in essential_metrics:
        if col not in df_campaign.columns:
            return []
    # impressions/clicks 仅用于展示，不作为决策关键；缺失时补 0 防止 groupby 报错
    if CAN.impressions not in df_campaign.columns or CAN.clicks not in df_campaign.columns:
        df_campaign = df_campaign.copy()
        if CAN.impressions not in df_campaign.columns:
            df_campaign[CAN.impressions] = 0.0
        if CAN.clicks not in df_campaign.columns:
            df_campaign[CAN.clicks] = 0.0

    gcols = [CAN.shop, CAN.ad_type, CAN.campaign]
    agg = (
        df_campaign.groupby(gcols, dropna=False, as_index=False)
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

    date_start, date_end = _date_range(df_campaign)
    out: List[ActionCandidate] = []

    good = agg[(agg["orders"] >= 3) & (agg["acos"] > 0) & (agg["acos"] < cfg.target_acos)]
    for _, r in good.sort_values(["orders", "sales"], ascending=False).head(80).iterrows():
        out.append(
            ActionCandidate(
                shop=str(r[CAN.shop]),
                ad_type=str(r[CAN.ad_type]),
                level="campaign",
                action_type="BUDGET_UP",
                action_value="+20%",
                priority="P1",
                object_name=str(r[CAN.campaign]),
                campaign=str(r[CAN.campaign]),
                ad_group="",
                match_type="",
                date_start=date_start,
                date_end=date_end,
                reason="活动投产达标且有量，建议优先分配更多预算（人工确认后执行）",
                evidence_json=json_dumps(
                    {
                        "spend": r["spend"],
                        "sales": r["sales"],
                        "orders": r["orders"],
                        "acos": float(r["acos"]),
                    }
                ),
            )
        )

    bad = agg[(agg["spend"] >= cfg.waste_spend * 2) & (agg["acos"] > cfg.target_acos * 1.2)]
    for _, r in bad.sort_values(["spend", "acos"], ascending=False).head(80).iterrows():
        out.append(
            ActionCandidate(
                shop=str(r[CAN.shop]),
                ad_type=str(r[CAN.ad_type]),
                level="campaign",
                action_type="REVIEW",
                action_value="",
                priority="P1",
                object_name=str(r[CAN.campaign]),
                campaign=str(r[CAN.campaign]),
                ad_group="",
                match_type="",
                date_start=date_start,
                date_end=date_end,
                reason="活动花费高且 ACoS 明显偏高，建议优先排查（结构/出价/否词/Listing）",
                evidence_json=json_dumps(
                    {
                        "spend": r["spend"],
                        "sales": r["sales"],
                        "orders": r["orders"],
                        "acos": float(r["acos"]),
                    }
                ),
            )
        )

    return out


def generate_campaign_budget_actions_from_map(
    shop: str,
    camp_budget_map: List[Dict[str, object]],
    date_start: str,
    date_end: str,
) -> List[ActionCandidate]:
    """
    把“预算迁移图谱”（diagnostics.campaign_budget_map）转成可执行的 ActionCandidate。

    注意：
    - 这不是批量上传文件（L1），只是 L0 的“方向/幅度候选”
    - 预算调整一定要人工确认（尤其是 SB/SD 预算策略不同）
    """
    out: List[ActionCandidate] = []
    if not camp_budget_map:
        return out

    for item in camp_budget_map[:200]:
        try:
            ad_type = str(item.get("ad_type", "") or "")
            campaign = str(item.get("campaign", "") or "")
            action = str(item.get("action", "") or "")
            pct = int(item.get("suggested_budget_change_pct", 0) or 0)
            severity = float(item.get("severity", 0.0) or 0.0)
            if not campaign or campaign.lower() == "nan":
                continue
            if action not in {"reduce", "scale"}:
                continue

            if action == "reduce" and pct < 0:
                action_type = "BUDGET_DOWN"
                action_value = f"{pct}%"
                priority = "P1" if severity >= 70 else "P2"
                reason = "预算图谱：该活动主要花费在“毛利承受度不足/广告后亏损”的 ASIN 上，建议先控量再定位浪费来源。"
            elif action == "scale" and pct > 0:
                action_type = "BUDGET_UP"
                action_value = f"+{pct}%"
                priority = "P2"
                reason = "预算图谱：该活动主要花费在“可放量(毛利承受度内)”的 ASIN 上，可作为预算迁移/加码候选（先确认库存）。"
            else:
                continue

            out.append(
                ActionCandidate(
                    shop=shop,
                    ad_type=ad_type,
                    level="campaign",
                    action_type=action_type,
                    action_value=action_value,
                    priority=priority,
                    object_name=campaign,
                    campaign=campaign,
                    ad_group="",
                    match_type="",
                    date_start=date_start,
                    date_end=date_end,
                    reason=reason,
                    evidence_json=json_dumps(item),
                )
            )
        except Exception:
            continue

    return out


def generate_targeting_actions(df_targeting: pd.DataFrame, cfg: StageConfig) -> List[ActionCandidate]:
    """
    投放（Targeting）层面的动作候选。

    说明（务实版）：
    - 赛狐“投放报告”里的“投放”可能是关键词 / ASIN / 类目等
    - 我们不在代码里强行区分（避免写成论文），而是给运营/AI明确提示：该怎么执行

    动作类型：
    - NEGATE：花费有点击但无单（若为关键词 -> 否词；若为ASIN/类目 -> 否定投放/暂停）
    - BID_DOWN：有单但 ACoS 偏高 -> 小步降价观察
    - BID_UP：投产明显好且有单 -> 小步加价扩量（确认库存）
    """
    if df_targeting.empty:
        return []

    # 清理：避免把 NaN/空串当成“投放对象”
    if CAN.targeting in df_targeting.columns:
        df_targeting = df_targeting.copy()
        df_targeting[CAN.targeting] = df_targeting[CAN.targeting].astype(str).str.strip()
        df_targeting = df_targeting[(df_targeting[CAN.targeting] != "") & (df_targeting[CAN.targeting].str.lower() != "nan")]

    required = [CAN.shop, CAN.ad_type, CAN.date, CAN.targeting]
    for col in required:
        if col not in df_targeting.columns:
            return []
    # 关键数值列缺失时，宁可不输出动作（避免误导）
    essential_metrics = [CAN.clicks, CAN.spend, CAN.sales, CAN.orders]
    for col in essential_metrics:
        if col not in df_targeting.columns:
            return []
    if CAN.impressions not in df_targeting.columns:
        df_targeting = df_targeting.copy()
        df_targeting[CAN.impressions] = 0.0

    # 有的报表包含 campaign/ad_group/match_type，有的没有；尽量保留以便可执行
    if CAN.match_type in df_targeting.columns:
        try:
            df_targeting = df_targeting.copy()
            df_targeting[CAN.match_type] = df_targeting[CAN.match_type].fillna("").astype(str).str.strip()
            df_targeting.loc[
                (df_targeting[CAN.match_type] == "") | (df_targeting[CAN.match_type].str.lower() == "nan"),
                CAN.match_type,
            ] = "N/A"
        except Exception:
            pass
    gcols = [CAN.shop, CAN.ad_type]
    for c in (CAN.campaign, CAN.ad_group, CAN.targeting, CAN.match_type):
        if c in df_targeting.columns and c not in gcols:
            gcols.append(c)

    agg = (
        df_targeting.groupby(gcols, dropna=False, as_index=False)
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
    agg["cvr"] = agg.apply(lambda r: safe_div(r["orders"], r["clicks"]), axis=1)

    date_start, date_end = _date_range(df_targeting)
    out: List[ActionCandidate] = []

    # 1) 否定/暂停：浪费（口径统一：订单=0 且 销售额=0；并要求点击达到门槛，避免样本太小）
    waste = agg[
        (agg["orders"] <= 0)
        & (agg["sales"] <= 0)
        & (agg["spend"] >= cfg.waste_spend)
        & (agg["clicks"] >= cfg.min_clicks)
    ]
    for _, r in waste.sort_values(["spend"], ascending=False).head(200).iterrows():
        out.append(
            ActionCandidate(
                shop=str(r[CAN.shop]),
                ad_type=str(r[CAN.ad_type]),
                level="targeting",
                action_type="NEGATE",
                action_value="",
                priority="P0",
                object_name=str(r.get(CAN.targeting, "")),
                campaign=str(r.get(CAN.campaign, "")),
                ad_group=str(r.get(CAN.ad_group, "")),
                match_type=str(r.get(CAN.match_type, "")),
                date_start=date_start,
                date_end=date_end,
                reason=f"投放层：点击≥{cfg.min_clicks}且花费≥{cfg.waste_spend}但订单=0且销售额=0；若为关键词->否词，若为ASIN/类目->否定投放/暂停",
                evidence_json=json_dumps(
                    {
                        "impressions": r["impressions"],
                        "clicks": r["clicks"],
                        "spend": r["spend"],
                        "sales": r["sales"],
                        "orders": r["orders"],
                    }
                ),
            )
        )

    # 2) 降价：高 ACoS
    high_acos = agg[(agg["orders"] > 0) & (agg["clicks"] >= cfg.min_clicks) & (agg["acos"] > cfg.target_acos)]
    for _, r in high_acos.sort_values(["acos", "spend"], ascending=False).head(200).iterrows():
        pct = int(cfg.max_change_pct * 100)
        out.append(
            ActionCandidate(
                shop=str(r[CAN.shop]),
                ad_type=str(r[CAN.ad_type]),
                level="targeting",
                action_type="BID_DOWN",
                action_value=f"-{pct}%",
                priority="P1",
                object_name=str(r.get(CAN.targeting, "")),
                campaign=str(r.get(CAN.campaign, "")),
                ad_group=str(r.get(CAN.ad_group, "")),
                match_type=str(r.get(CAN.match_type, "")),
                date_start=date_start,
                date_end=date_end,
                reason=f"投放层：ACoS>{cfg.target_acos:.0%} 且点击充足，有单但效率偏差，建议小步降价观察（48-72h，可回滚）",
                evidence_json=json_dumps(
                    {
                        "clicks": r["clicks"],
                        "spend": r["spend"],
                        "sales": r["sales"],
                        "orders": r["orders"],
                        "acos": float(r["acos"]),
                        "cvr": float(r["cvr"]),
                    }
                ),
            )
        )

    # 3) 扩量：低 ACoS 且有单
    good = agg[(agg["orders"] >= 2) & (agg["clicks"] >= cfg.min_clicks) & (agg["acos"] > 0) & (agg["acos"] < cfg.target_acos * 0.7)]
    for _, r in good.sort_values(["orders", "sales"], ascending=False).head(200).iterrows():
        pct = int(cfg.max_change_pct * 100)
        out.append(
            ActionCandidate(
                shop=str(r[CAN.shop]),
                ad_type=str(r[CAN.ad_type]),
                level="targeting",
                action_type="BID_UP",
                action_value=f"+{pct}%",
                priority="P1",
                object_name=str(r.get(CAN.targeting, "")),
                campaign=str(r.get(CAN.campaign, "")),
                ad_group=str(r.get(CAN.ad_group, "")),
                match_type=str(r.get(CAN.match_type, "")),
                date_start=date_start,
                date_end=date_end,
                reason="投放层：投产明显优于目标且已出单，建议小步加价扩量（先确认库存/转化是否稳定）",
                evidence_json=json_dumps(
                    {
                        "clicks": r["clicks"],
                        "spend": r["spend"],
                        "sales": r["sales"],
                        "orders": r["orders"],
                        "acos": float(r["acos"]),
                        "cvr": float(r["cvr"]),
                    }
                ),
            )
        )

    return out


def generate_placement_actions(df_placement: pd.DataFrame, cfg: StageConfig) -> List[ActionCandidate]:
    """
    广告位（Placement）层面的动作候选：用于“加价系数/倾斜预算”的方向建议。

    注意：
    - Placement 报告不一定包含广告组；以 campaign + placement 为主
    - 我们输出的是“候选动作”，实际执行时由运营判断是否有对应的加价入口
    """
    if df_placement.empty:
        return []

    # 清理：避免把 NaN/空串当成“广告位对象”
    if CAN.placement in df_placement.columns:
        df_placement = df_placement.copy()
        df_placement[CAN.placement] = df_placement[CAN.placement].astype(str).str.strip()
        df_placement = df_placement[(df_placement[CAN.placement] != "") & (df_placement[CAN.placement].str.lower() != "nan")]

    required = [CAN.shop, CAN.ad_type, CAN.date, CAN.placement]
    for col in required:
        if col not in df_placement.columns:
            return []
    # 关键数值列缺失时，宁可不输出动作（避免误导）；同时补齐 clicks 列防止 groupby 报错
    essential_metrics = [CAN.spend, CAN.sales, CAN.orders]
    for col in essential_metrics:
        if col not in df_placement.columns:
            return []
    if CAN.clicks not in df_placement.columns or CAN.impressions not in df_placement.columns:
        df_placement = df_placement.copy()
        if CAN.clicks not in df_placement.columns:
            df_placement[CAN.clicks] = 0.0
        if CAN.impressions not in df_placement.columns:
            df_placement[CAN.impressions] = 0.0

    gcols = [CAN.shop, CAN.ad_type, CAN.placement]
    if CAN.campaign in df_placement.columns:
        gcols.insert(2, CAN.campaign)

    agg = (
        df_placement.groupby(gcols, dropna=False, as_index=False)
        .agg(
            clicks=(CAN.clicks, "sum"),
            spend=(CAN.spend, "sum"),
            sales=(CAN.sales, "sum"),
            orders=(CAN.orders, "sum"),
        )
        .copy()
    )
    agg["acos"] = agg.apply(lambda r: safe_div(r["spend"], r["sales"]), axis=1)

    date_start, date_end = _date_range(df_placement)
    out: List[ActionCandidate] = []

    good = agg[(agg["orders"] >= 3) & (agg["acos"] > 0) & (agg["acos"] < cfg.target_acos * 0.75)]
    for _, r in good.sort_values(["orders", "sales"], ascending=False).head(120).iterrows():
        out.append(
            ActionCandidate(
                shop=str(r[CAN.shop]),
                ad_type=str(r[CAN.ad_type]),
                level="placement",
                action_type="BID_UP",
                action_value="+20%",
                priority="P2",
                object_name=str(r.get(CAN.placement, "")),
                campaign=str(r.get(CAN.campaign, "")),
                ad_group="",
                match_type="",
                date_start=date_start,
                date_end=date_end,
                reason="广告位投产明显优于目标，可考虑提高该广告位加价系数或倾斜预算（小步调整）",
                evidence_json=json_dumps(
                    {"clicks": r["clicks"], "spend": r["spend"], "sales": r["sales"], "orders": r["orders"], "acos": float(r["acos"])}
                ),
            )
        )

    bad = agg[(agg["spend"] >= cfg.waste_spend) & (agg["acos"] > cfg.target_acos * 1.3)]
    for _, r in bad.sort_values(["spend", "acos"], ascending=False).head(120).iterrows():
        out.append(
            ActionCandidate(
                shop=str(r[CAN.shop]),
                ad_type=str(r[CAN.ad_type]),
                level="placement",
                action_type="BID_DOWN",
                action_value="-20%",
                priority="P2",
                object_name=str(r.get(CAN.placement, "")),
                campaign=str(r.get(CAN.campaign, "")),
                ad_group="",
                match_type="",
                date_start=date_start,
                date_end=date_end,
                reason="广告位花费较高且 ACoS 明显偏高，可考虑下调该广告位加价系数/把预算移回更优位置",
                evidence_json=json_dumps(
                    {"clicks": r["clicks"], "spend": r["spend"], "sales": r["sales"], "orders": r["orders"], "acos": float(r["acos"])}
                ),
            )
        )

    return out


def generate_product_side_actions(
    product_analysis_shop: pd.DataFrame,
    product_listing_shop: pd.DataFrame,
    cfg: StageConfig,
) -> List[ActionCandidate]:
    """
    产品侧（Listing/库存/结构）动作候选，优先解决“能指导运营调广告”的问题：
    - 库存低：先控量/停投，避免断货浪费
    - 有广告花费但无广告单：优先排查投放与Listing转化
    - 投产好且库存安全：提示“可放量”

    依赖：
    - 产品分析（按日）里常见列：ASIN/广告花费/广告销售额/广告订单量/销售额/Sessions/转化率/FBA可售
    - productListing 里常见列：ASIN/可售/品名/商品分类
    """
    if product_analysis_shop is None or product_analysis_shop.empty:
        return []
    if "ASIN" not in product_analysis_shop.columns:
        return []

    pa = product_analysis_shop.copy()
    # 只保留我们会用到的列，避免内存大
    keep_cols = ["ASIN", "广告花费", "广告销售额", "广告订单量", "销售额", "Sessions", "转化率", CAN.shop, CAN.date, "FBA可售"]
    keep_cols = [c for c in keep_cols if c in pa.columns]
    pa = pa[keep_cols].copy()

    # 按 ASIN 汇总
    agg = (
        pa.groupby("ASIN", dropna=False, as_index=False)
        .agg(
            ad_spend=("广告花费", "sum") if "广告花费" in pa.columns else ("ASIN", "size"),
            ad_sales=("广告销售额", "sum") if "广告销售额" in pa.columns else ("ASIN", "size"),
            ad_orders=("广告订单量", "sum") if "广告订单量" in pa.columns else ("ASIN", "size"),
            total_sales=("销售额", "sum") if "销售额" in pa.columns else ("ASIN", "size"),
            sessions=("Sessions", "sum") if "Sessions" in pa.columns else ("ASIN", "size"),
            fba_avail=("FBA可售", "min") if "FBA可售" in pa.columns else ("ASIN", "size"),
            conv_pct=("转化率", "mean") if "转化率" in pa.columns else ("ASIN", "size"),
        )
        .copy()
    )
    agg["ad_acos"] = agg.apply(lambda r: safe_div(r["ad_spend"], r["ad_sales"]), axis=1)
    agg["tacos"] = agg.apply(lambda r: safe_div(r["ad_spend"], r["total_sales"]), axis=1)

    # 转化率：赛狐表通常是“百分比数字”（如 12.3 表示 12.3%）
    def pct_to_ratio(v: float) -> float:
        try:
            if v <= 0:
                return 0.0
            return v / 100.0 if v > 1.0 else v
        except Exception:
            return 0.0

    agg["conv_rate"] = agg["conv_pct"].apply(pct_to_ratio) if "conv_pct" in agg.columns else 0.0

    # 补一个可售库存：优先用 productListing 的“可售”，否则用产品分析的 FBA可售
    inv = pd.DataFrame()
    if product_listing_shop is not None and not product_listing_shop.empty and "ASIN" in product_listing_shop.columns:
        cols = ["ASIN"]
        for c in ("可售", "品名", "商品分类"):
            if c in product_listing_shop.columns:
                cols.append(c)
        inv = product_listing_shop[cols].copy()
        if "可售" in inv.columns:
            inv["可售"] = inv["可售"].apply(lambda x: x if isinstance(x, (int, float)) else 0.0)
        inv["ASIN"] = inv["ASIN"].astype(str).str.upper()
    agg["ASIN_norm"] = agg["ASIN"].astype(str).str.upper()
    if not inv.empty:
        inv2 = inv.drop_duplicates("ASIN").rename(columns={"ASIN": "ASIN_norm"})
        agg = agg.merge(inv2, on="ASIN_norm", how="left")

    # 报告日期范围：用产品分析范围
    shop = ""
    if CAN.shop in pa.columns and not pa[CAN.shop].dropna().empty:
        shop = str(pa[CAN.shop].dropna().astype(str).iloc[0])
    date_start, date_end = _date_range(pa.rename(columns={"日期": CAN.date}) if CAN.date not in pa.columns else pa)

    out: List[ActionCandidate] = []

    # 1) 库存低：P0
    inv_col = "可售" if "可售" in agg.columns else ("fba_avail" if "fba_avail" in agg.columns else "")
    if inv_col:
        low = agg[(agg[inv_col].fillna(999999) <= 20) & (agg["ad_spend"] > 0)].copy()
        for _, r in low.sort_values(["ad_spend"], ascending=False).head(120).iterrows():
            out.append(
                ActionCandidate(
                    shop=shop,
                    ad_type="ALL",
                    level="asin",
                    action_type="REVIEW",
                    action_value="",
                    priority="P0",
                    object_name=str(r.get("ASIN", "")),
                    campaign="",
                    ad_group="",
                    match_type="",
                    date_start=date_start,
                    date_end=date_end,
                    reason="产品侧：库存偏低且仍有广告花费，优先控量/停投，避免断货导致转化下降与浪费",
                    evidence_json=json_dumps(
                        {
                            "asin": str(r.get("ASIN", "")),
                            "inventory": float(r.get(inv_col, 0.0) or 0.0),
                            "ad_spend": float(r.get("ad_spend", 0.0) or 0.0),
                            "tacos": float(r.get("tacos", 0.0) or 0.0),
                        }
                    ),
                )
            )

    # 2) 有广告花费但无广告单：P0（口径统一：广告订单=0 且 广告销售额=0）
    waste = agg[(agg["ad_spend"] >= cfg.waste_spend) & (agg["ad_orders"] <= 0) & (agg["ad_sales"] <= 0)].copy()
    for _, r in waste.sort_values(["ad_spend"], ascending=False).head(200).iterrows():
        out.append(
            ActionCandidate(
                shop=shop,
                ad_type="ALL",
                level="asin",
                action_type="REVIEW",
                action_value="",
                priority="P0",
                object_name=str(r.get("ASIN", "")),
                campaign="",
                ad_group="",
                match_type="",
                date_start=date_start,
                date_end=date_end,
                reason="产品侧：ASIN 有广告花费但广告订单=0且广告销售额=0，优先排查投放结构/否词/Listing（主图价格评价）",
                evidence_json=json_dumps(
                    {
                        "asin": str(r.get("ASIN", "")),
                        "ad_spend": float(r.get("ad_spend", 0.0) or 0.0),
                        "ad_sales": float(r.get("ad_sales", 0.0) or 0.0),
                        "sessions": float(r.get("sessions", 0.0) or 0.0),
                        "conv_rate": float(r.get("conv_rate", 0.0) or 0.0),
                    }
                ),
            )
        )

    # 3) 投产好且库存安全：P2（提示可放量）
    good = agg[(agg["ad_orders"] >= 3) & (agg["ad_acos"] > 0) & (agg["ad_acos"] < cfg.target_acos * 0.7)].copy()
    for _, r in good.sort_values(["ad_orders", "ad_sales"], ascending=False).head(120).iterrows():
        out.append(
            ActionCandidate(
                shop=shop,
                ad_type="ALL",
                level="asin",
                action_type="REVIEW",
                action_value="",
                priority="P2",
                object_name=str(r.get("ASIN", "")),
                campaign="",
                ad_group="",
                match_type="",
                date_start=date_start,
                date_end=date_end,
                reason="产品侧：广告投产明显优于目标且已出单，可在相关活动/投放里适度加价或加预算放量（先确认库存）",
                evidence_json=json_dumps(
                    {
                        "asin": str(r.get("ASIN", "")),
                        "ad_spend": float(r.get("ad_spend", 0.0) or 0.0),
                        "ad_sales": float(r.get("ad_sales", 0.0) or 0.0),
                        "ad_orders": float(r.get("ad_orders", 0.0) or 0.0),
                        "ad_acos": float(r.get("ad_acos", 0.0) or 0.0),
                        "tacos": float(r.get("tacos", 0.0) or 0.0),
                    }
                ),
            )
        )

    return out
