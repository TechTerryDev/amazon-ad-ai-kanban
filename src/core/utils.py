# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any, Optional

import pandas as pd


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


_NUM_CLEAN_RE = re.compile(r"[,$￥¥%\\s]")


def to_float(value: Any) -> float:
    """
    把各种“看起来像数字”的内容转成 float：
    - '1,234.56' / '$12.3' / '  9.9 ' / None / '--'
    """
    try:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return 0.0
            return float(value)
        s = str(value).strip()
        if not s or s in {"--", "-", "nan", "NaN", "None"}:
            return 0.0
        s = _NUM_CLEAN_RE.sub("", s)
        return float(s) if s else 0.0
    except Exception:
        return 0.0


def parse_date(value: Any) -> Optional[dt.date]:
    """
    支持：
    - 2026-01-04 / 2026/01/04
    - 20260104
    - Excel serial: 45998.0
    - pandas datetime
    """
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            return value
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, (int, float)):
            serial = int(float(value))
            # 合理范围：2000~2100
            if 36526 <= serial <= 73051:
                base = dt.date(1899, 12, 30)
                return base + dt.timedelta(days=serial)
            return None
        s = str(value).strip()
        if not s:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return dt.datetime.strptime(s, fmt).date()
            except Exception:
                pass
        if re.fullmatch(r"\\d{8}", s):
            try:
                return dt.datetime.strptime(s, "%Y%m%d").date()
            except Exception:
                return None
        return None
    except Exception:
        return None


def json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return "{}"


_STATUS_EMPTY = {"", "nan", "none", "null", "na", "n/a", "--", "-"}
_STATUS_PAUSED_KEYWORDS_ZH = ("暂停", "停用", "已暂停", "已停用", "停投", "已停投", "关闭", "已关闭", "已结束", "已终止", "终止", "停止")
_STATUS_PAUSED_KEYWORDS_EN = ("paused", "pause", "stopped", "disabled", "inactive", "ended", "archived")


def normalize_status(value: Any) -> str:
    """
    统一状态字段的空值与格式（仅清洗，不做强语义归类）。
    """
    try:
        if value is None:
            return ""
        s = str(value).strip()
        if not s:
            return ""
        if s.lower() in _STATUS_EMPTY:
            return ""
        return s
    except Exception:
        return ""


def is_paused_status(value: Any) -> bool:
    """
    判断“是否为暂停/停用/终止”等状态，用于过滤行动清单。
    """
    s = normalize_status(value)
    if not s:
        return False
    s_lower = s.lower()
    for kw in _STATUS_PAUSED_KEYWORDS_EN:
        if kw in s_lower:
            return True
    for kw in _STATUS_PAUSED_KEYWORDS_ZH:
        if kw in s:
            return True
    return False
