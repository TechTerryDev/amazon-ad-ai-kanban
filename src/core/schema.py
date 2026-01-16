# -*- coding: utf-8 -*-
"""
赛狐导出报表（xlsx）字段映射到统一内部字段（canonical schema）。

注意：
- 这里不追求覆盖所有字段，只抽出“做建议必需”的字段。
- 其它字段可以后续按需加入（不要一开始就做成论文指标库）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalCols:
    shop: str = "shop"
    date: str = "date"
    ad_type: str = "ad_type"  # SP/SB/SD

    # 状态字段（用于过滤暂停广告的“行动清单”）
    status: str = "status"

    # 维度字段
    campaign: str = "campaign"
    ad_group: str = "ad_group"
    match_type: str = "match_type"
    targeting: str = "targeting"
    search_term: str = "search_term"
    placement: str = "placement"
    asin: str = "asin"
    sku: str = "sku"
    other_asin: str = "other_asin"  # 已购买商品报告的“其他ASIN”

    # 核心数值字段（建议层基本都靠这几列）
    impressions: str = "impressions"
    clicks: str = "clicks"
    spend: str = "spend"
    sales: str = "sales"
    orders: str = "orders"


CAN = CanonicalCols()


# 常用中文列名 -> canonical
BASE_COLUMN_ALIASES: dict[str, str] = {
    "店铺": CAN.shop,
    "日期": CAN.date,
    "广告活动": CAN.campaign,
    "广告组": CAN.ad_group,
    "匹配类型": CAN.match_type,
    "投放": CAN.targeting,
    "用户搜索词": CAN.search_term,
    "广告位": CAN.placement,
    # 运行状态（赛狐常见列名）
    "广告活动运行状态": CAN.status,
    "广告活动状态": CAN.status,
    "广告组运行状态": CAN.status,
    "广告组状态": CAN.status,
    "投放运行状态": CAN.status,
    "投放状态": CAN.status,
    "广告产品运行状态": CAN.status,
    "广告产品状态": CAN.status,
    "状态": CAN.status,
    "Status": CAN.status,
    "Campaign Status": CAN.status,
    "Ad Group Status": CAN.status,
    "Targeting Status": CAN.status,
    # 广告产品报告里是小写 asin/sku
    "ASIN": CAN.asin,
    "asin": CAN.asin,
    "SKU": CAN.sku,
    "sku": CAN.sku,
    "其他ASIN": CAN.other_asin,
    # 指标列
    "广告曝光量": CAN.impressions,
    "曝光量": CAN.impressions,
    "Impressions": CAN.impressions,
    "广告点击量": CAN.clicks,
    "点击量": CAN.clicks,
    "Clicks": CAN.clicks,
    "广告花费": CAN.spend,
    "花费": CAN.spend,
    "Spend": CAN.spend,
    "广告销售额": CAN.sales,
    "销售额": CAN.sales,
    "Sales": CAN.sales,
    "广告订单量": CAN.orders,
    "订单量": CAN.orders,
    "Orders": CAN.orders,
}


# 识别报表类型（从文件名）
REPORT_KEYWORDS: dict[str, str] = {
    "搜索词报告": "search_term",
    "投放报告": "targeting",
    "广告位报告": "placement",
    "广告活动报告": "campaign",
    "广告组报告": "ad_group",
    "广告产品报告": "advertised_product",
    "已购买商品报告": "purchased_product",
    "匹配的目标报告": "matched_target",
}
