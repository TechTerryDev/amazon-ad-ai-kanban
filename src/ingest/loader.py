# -*- coding: utf-8 -*-
"""
从 `reports/`（或你运行时指定的 `--input-dir`）读取赛狐导出的 xlsx 报表，并做最小归一化：
- 统一列名（canonical schema）
- 统一日期格式（date）
- 统一数值列为 float

说明：
你的环境里可能还没装 pandas/openpyxl，先按仓库根目录 `requirements.txt` 安装即可。
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.core.schema import BASE_COLUMN_ALIASES, CAN, REPORT_KEYWORDS
from src.core.utils import parse_date, to_float, normalize_status

# 赛狐导出的 xlsx 经常触发这个 openpyxl 警告，属于“文件样式缺省”，不影响读取。
warnings.filterwarnings("ignore", message="Workbook contains no default style, apply openpyxl's default")


@dataclass(frozen=True)
class LoadedReport:
    ad_type: str  # SP/SB/SD
    report_type: str  # search_term/targeting/...
    file: str
    df: pd.DataFrame


def discover_ad_reports(ad_root: Path) -> List[Tuple[str, str, Path]]:
    """
    扫描 ad_root 下所有 xlsx，基于文件名识别：
    - ad_type: SP/SB/SD
    - report_type: search_term / targeting / ...
    """
    out: List[Tuple[str, str, Path]] = []
    for p in sorted(ad_root.rglob("*.xlsx")):
        name = p.name
        ad_type = "UNKNOWN"
        if name.startswith("SP"):
            ad_type = "SP"
        elif name.startswith("SB"):
            ad_type = "SB"
        elif name.startswith("SD"):
            ad_type = "SD"

        report_type = ""
        for kw, typ in REPORT_KEYWORDS.items():
            if kw in name:
                report_type = typ
                break
        if not report_type:
            continue
        out.append((ad_type, report_type, p))
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        c2 = str(c).strip()
        cols.append(c2)
    df = df.copy()
    df.columns = cols

    # alias -> canonical
    rename: Dict[str, str] = {}
    for c in df.columns:
        if c in BASE_COLUMN_ALIASES:
            rename[c] = BASE_COLUMN_ALIASES[c]
    df = df.rename(columns=rename)
    return df


def _coerce_core_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 日期
    if CAN.date in df.columns:
        df[CAN.date] = df[CAN.date].apply(parse_date)

    # 数值列
    for col in (CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders):
        if col in df.columns:
            df[col] = df[col].apply(to_float)

    # shop 统一成字符串
    if CAN.shop in df.columns:
        df[CAN.shop] = df[CAN.shop].astype(str).str.strip()

    # 状态字段清洗：统一空值/空白，保留原始文案用于后续判断
    if CAN.status in df.columns:
        df[CAN.status] = df[CAN.status].apply(normalize_status)

    # 维度字段清洗：把 NaN/空格统一掉，避免后续 groupby / 字符串拼接出现 "nan"
    for col in (
        CAN.campaign,
        CAN.ad_group,
        CAN.match_type,
        CAN.targeting,
        CAN.search_term,
        CAN.placement,
        CAN.asin,
        CAN.sku,
        CAN.other_asin,
    ):
        if col in df.columns:
            try:
                df[col] = df[col].fillna("").astype(str).str.strip()
                df.loc[df[col].str.lower() == "nan", col] = ""
            except Exception:
                continue

    # match_type 特殊：很多报表（尤其自动投放/某些报告类型）会为空，此时用 N/A 显式标记“不可用/不适用”
    if CAN.match_type in df.columns:
        try:
            df.loc[df[CAN.match_type] == "", CAN.match_type] = "N/A"
        except Exception:
            pass
    return df


def load_xlsx(file_path: Path) -> pd.DataFrame:
    """
    读取单个 xlsx。默认取第一张 sheet（赛狐导出通常只有一张）。
    """
    try:
        # 注：pandas 的 openpyxl engine 内部已按需使用 read_only/data_only；
        # 这里不要显式传 engine_kwargs，否则容易与 pandas 默认参数冲突（出现重复关键字错误）。
        df = pd.read_excel(file_path, engine="openpyxl")
        return df
    except Exception as e:
        # 给出更友好的报错信息
        raise RuntimeError(f"读取失败: {file_path}，原因: {e}") from e


def _load_xlsx_selective(file_path: Path, usecols: Optional[Callable[[str], bool]] = None) -> pd.DataFrame:
    """
    性能优化版读取：优先按 usecols 只读必需列；若失败则回退为全量读取。
    """
    try:
        if usecols is None:
            return load_xlsx(file_path)
        return pd.read_excel(
            file_path,
            engine="openpyxl",
            usecols=usecols,
        )
    except Exception:
        # 回退：避免因列名差异导致直接跳过文件
        return load_xlsx(file_path)


def load_ad_reports(ad_root: Path) -> List[LoadedReport]:
    loaded: List[LoadedReport] = []
    # 只读“做分析必需”的列：避免赛狐导出列很多导致内存暴涨
    allowed_cols = set(BASE_COLUMN_ALIASES.keys()) | {"其他SKU销量", "其他SKU销售额"}
    usecols = lambda c: str(c).strip() in allowed_cols
    for ad_type, report_type, path in discover_ad_reports(ad_root):
        try:
            raw = _load_xlsx_selective(path, usecols=usecols)
            df = _coerce_core_types(_normalize_columns(raw))
            df[CAN.ad_type] = ad_type
            loaded.append(LoadedReport(ad_type=ad_type, report_type=report_type, file=str(path), df=df))
        except Exception:
            # 单个文件失败不影响整体
            continue
    return loaded


def load_product_listing(path: Path) -> pd.DataFrame:
    """
    在线产品/成本/可售等底座（productListing.xlsx）
    """
    # 注意：productListing 不是“广告报表”，不要强行把“销售额/花费”等字段改成 canonical，
    # 否则会影响后续与运营表头对齐。这里只做最小处理：列名清洗 + 额外补一列 shop 供过滤。
    # 只读“类目/库存/成本”相关列（productListing 常见列很多）
    allowed_cols = {"店铺", "ASIN", "品名", "商品分类", "可售", "采购成本(CNY)", "头程费用(CNY)"}
    df = _load_xlsx_selective(path, usecols=lambda c: str(c).strip() in allowed_cols)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 兼容：如果表里是“店铺”，就新增 canonical shop 列（不改原列名）
    if CAN.shop not in df.columns and "店铺" in df.columns:
        df[CAN.shop] = df["店铺"].astype(str).str.strip()
    elif CAN.shop in df.columns:
        df[CAN.shop] = df[CAN.shop].astype(str).str.strip()

    # 常用字段：ASIN/MSKU/品名/分类/可售/采购成本/头程费用
    for col in ("可售", "采购成本(CNY)", "头程费用(CNY)"):
        if col in df.columns:
            df[col] = df[col].apply(to_float)
    return df


def discover_product_analysis_files(product_analysis_dir: Path) -> List[Path]:
    # 只处理“按日”的 ASIN 列表
    return sorted(product_analysis_dir.rglob("产品分析-ASIN-列表-按日-*.xlsx"))


def load_product_analysis(product_analysis_dir: Path) -> pd.DataFrame:
    """
    产品分析（按日）是“产品经营结果”的统一底座。
    我们主要取：店铺/日期/ASIN/销量/销售额/Sessions/PV/转化率/广告花费/广告销售额/广告订单量/毛利润等。
    """
    files = discover_product_analysis_files(product_analysis_dir)
    if not files:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    # 只读“生命周期/销量/广告/利润/库存”必需列（产品分析表列非常多）
    allowed_cols = {
        "日期",
        "店铺",
        "ASIN",
        "品名",
        "销量",
        "订单量",
        "销售额",
        "Sessions",
        "PV",
        "转化率",
        "广告花费",
        "广告销售额",
        "广告订单量",
        "自然销售额",
        "自然订单量",
        "毛利润",
        "FBA可售",
        "退款率",
        "星级评分",
    }
    usecols = lambda c: str(c).strip() in allowed_cols
    for p in files:
        try:
            df = _load_xlsx_selective(p, usecols=usecols)
            # 产品分析是“经营底座表”，列非常多；不要用广告报表的 alias 映射去改列名，
            # 否则会把“销售额/广告花费”等字段改成 canonical，导致你肉眼对不上表头。
            df = df.copy()
            df.columns = [str(c).strip() for c in df.columns]

            # 最小改动：把“日期/店铺”补成 canonical 列，方便按店铺切分与按日聚合
            if CAN.date not in df.columns and "日期" in df.columns:
                df[CAN.date] = df["日期"]
            if CAN.shop not in df.columns and "店铺" in df.columns:
                df[CAN.shop] = df["店铺"]
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    # 统一类型
    if CAN.date in df_all.columns:
        df_all[CAN.date] = df_all[CAN.date].apply(parse_date)
    if CAN.shop in df_all.columns:
        df_all[CAN.shop] = df_all[CAN.shop].astype(str).str.strip()
    # 常用数值字段（只把我们会用到的列转数字，避免太慢）
    for col in ("销量", "订单量", "销售额", "Sessions", "PV", "转化率", "广告花费", "广告销售额", "广告订单量", "毛利润"):
        if col in df_all.columns:
            df_all[col] = df_all[col].apply(to_float)
    return df_all
