# -*- coding: utf-8 -*-
"""
关键词/搜索词主题聚合（n-gram）。

目标（面向你的业务痛点）：
- 赛狐 search_term 报表往往很长，运营难以快速抓到“在烧什么主题/哪些主题在带量”；
- 用 n-gram（1~3gram）把大量搜索词压缩成可解释的“主题词/意图”列表；
- 输出 CSV 供运营筛选/排序，不引入 TF-IDF（解释性更强、无需新增依赖）。

注意：
- n-gram 是“主题归并/线索”，不是精确归因；同一条搜索词会贡献给多个 n-gram，因此各主题的 spend 会有重复计数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.core.config import get_stage_config
from src.core.policy import KeywordTopicsPolicy
from src.core.rules import is_waste_spend
from src.core.schema import CAN
from src.core.utils import safe_div


_RE_WORD = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
_RE_ASIN_LIKE = re.compile(r"^[a-z0-9]{10}$", flags=re.IGNORECASE)


# 轻量停用词：只做最基本的噪声过滤（避免 “for/with/the” 霸榜）
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "your",
}


def _is_asin_like(token: str) -> bool:
    """
    简单识别 ASIN/类似串：避免把一堆 B0XXXXXXX 变成“主题词”。
    """
    try:
        s = str(token or "").strip().lower()
        if not s:
            return False
        if not _RE_ASIN_LIKE.match(s):
            return False
        # ASIN 通常同时含字母和数字（纯数字/纯字母意义不大）
        has_alpha = any(c.isalpha() for c in s)
        has_digit = any(c.isdigit() for c in s)
        return has_alpha and has_digit
    except Exception:
        return False


def _tokenize(text: str) -> List[str]:
    """
    把搜索词拆成 token。

    说明：
    - 仅保留字母数字（适配 US 站为主）；中文/符号会被忽略（后续如需要可扩展）。
    - 去停用词、去纯数字、去 ASIN-like token。
    """
    try:
        s = str(text or "").strip().lower()
        if not s or s == "nan":
            return []
        tokens = [t.lower() for t in _RE_WORD.findall(s)]
        out: List[str] = []
        for t in tokens:
            tt = t.strip().lower()
            if not tt:
                continue
            if tt in _STOPWORDS:
                continue
            if tt.isdigit():
                continue
            if len(tt) <= 1:
                continue
            if _is_asin_like(tt):
                continue
            out.append(tt)
        return out
    except Exception:
        return []


def _iter_ngrams(tokens: List[str], n: int) -> Iterable[str]:
    try:
        nn = int(n or 0)
        if nn <= 0:
            return []
        if len(tokens) < nn:
            return []
        return (" ".join(tokens[i : i + nn]) for i in range(0, len(tokens) - nn + 1))
    except Exception:
        return []


@dataclass
class _Agg:
    term_count: int = 0
    spend: float = 0.0
    sales: float = 0.0
    orders: float = 0.0
    clicks: float = 0.0
    impressions: float = 0.0
    waste_spend: float = 0.0
    waste_term_count: int = 0
    # 用于解释：该 n-gram 对应的“样例搜索词”（按 spend 排序，最多 K 条）
    top_terms: List[Tuple[float, str]] = field(default_factory=list)


def _push_top_term(agg: _Agg, term: str, spend: float, top_k: int) -> None:
    """
    维护 top_terms：按 spend 降序去重保留前 K 条。
    """
    try:
        k = int(top_k or 0)
        if k <= 0:
            return
        t = str(term or "").strip()
        if not t or t.lower() == "nan":
            return
        s = float(spend or 0.0)
        # 去重
        existing = [x for x in agg.top_terms if str(x[1]) == t]
        if existing:
            # 如果已有且新 spend 更大，更新后再排序
            agg.top_terms = [(max(float(x[0]), s), t) if str(x[1]) == t else x for x in agg.top_terms]
        else:
            agg.top_terms.append((s, t))
        # 排序 + 截断
        agg.top_terms = sorted(agg.top_terms, key=lambda x: float(x[0]), reverse=True)[:k]
    except Exception:
        return


def build_keyword_topics(
    search_term_report: Optional[pd.DataFrame],
    n_values: Optional[List[int]] = None,
    min_term_spend: float = 0.0,
    waste_min_clicks: int = 0,
    waste_min_spend: float = 0.0,
    max_terms: int = 5000,
    max_rows: int = 2000,
    top_terms_per_ngram: int = 3,
) -> pd.DataFrame:
    """
    从 search_term 报表生成 n-gram 主题表。

    输入：
    - search_term_report：canonical schema 的 search_term DataFrame（含 search_term + spend/sales/orders 等）

    输出（CSV 友好）：
    - ad_type, n, ngram, term_count, spend, sales, orders, acos, waste_spend, waste_term_count, top_terms

    关键口径：
    - waste_spend：orders=0 且 sales=0 的花费（若 clicks 字段存在，则进一步要求 clicks>=waste_min_clicks）
    """
    st = search_term_report.copy() if isinstance(search_term_report, pd.DataFrame) else pd.DataFrame()
    if st is None or st.empty:
        return pd.DataFrame()
    if CAN.search_term not in st.columns:
        return pd.DataFrame()

    has_clicks = CAN.clicks in st.columns

    # 数值列兜底：缺列就补 0，避免聚合报错
    for c in (CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders):
        if c not in st.columns:
            st[c] = 0.0

    if CAN.ad_type not in st.columns:
        st[CAN.ad_type] = ""

    # 先按 ad_type + search_term 汇总（减少按日噪声）
    try:
        st2 = st.copy()
        st2[CAN.ad_type] = st2[CAN.ad_type].fillna("").astype(str).str.strip()
        st2[CAN.search_term] = st2[CAN.search_term].fillna("").astype(str).str.strip()
        st2 = st2[(st2[CAN.search_term] != "") & (st2[CAN.search_term].str.lower() != "nan")].copy()
        for c in (CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders):
            st2[c] = pd.to_numeric(st2[c], errors="coerce").fillna(0.0)

        g = (
            st2.groupby([CAN.ad_type, CAN.search_term], dropna=False, as_index=False)
            .agg(
                impressions=(CAN.impressions, "sum"),
                clicks=(CAN.clicks, "sum"),
                spend=(CAN.spend, "sum"),
                sales=(CAN.sales, "sum"),
                orders=(CAN.orders, "sum"),
            )
            .copy()
        )
    except Exception:
        return pd.DataFrame()

    try:
        min_spend = float(min_term_spend or 0.0)
        if min_spend > 0:
            g = g[g["spend"] >= min_spend].copy()
    except Exception:
        pass
    if g.empty:
        return pd.DataFrame()

    # 性能/聚焦：搜索词可能非常多（尤其是长尾），先按 spend 取 TopN 再做 n-gram 聚合
    try:
        mt = int(max_terms or 0)
        if mt > 0 and len(g) > mt and "spend" in g.columns:
            g = g.sort_values(["spend"], ascending=[False]).head(mt).copy()
    except Exception:
        pass

    n_list = n_values if isinstance(n_values, list) and n_values else [1, 2, 3]
    n_list2: List[int] = []
    for x in n_list:
        try:
            v = int(x)
            if 1 <= v <= 5:
                n_list2.append(v)
        except Exception:
            continue
    if not n_list2:
        n_list2 = [1, 2, 3]

    aggs: Dict[Tuple[str, int, str], _Agg] = {}
    try:
        for _, r in g.iterrows():
            ad_type = str(r.get(CAN.ad_type, "") or "").strip()
            term = str(r.get(CAN.search_term, "") or "").strip()
            tokens = _tokenize(term)
            if not tokens:
                continue
            spend = float(r.get("spend", 0.0) or 0.0)
            sales = float(r.get("sales", 0.0) or 0.0)
            orders = float(r.get("orders", 0.0) or 0.0)
            clicks = float(r.get("clicks", 0.0) or 0.0)
            impressions = float(r.get("impressions", 0.0) or 0.0)
            is_waste = is_waste_spend(
                orders=orders,
                sales=sales,
                spend=spend,
                clicks=(clicks if has_clicks else None),
                min_clicks=int(waste_min_clicks or 0),
                min_spend=float(waste_min_spend or 0.0),
            )

            # 同一搜索词内，同一 n-gram 只计一次（避免重复 token 造成放大）
            for n in n_list2:
                uniq = set(_iter_ngrams(tokens, n))
                for ng in uniq:
                    key = (ad_type, int(n), str(ng))
                    agg = aggs.get(key)
                    if agg is None:
                        agg = _Agg()
                        aggs[key] = agg
                    agg.term_count += 1
                    agg.spend += spend
                    agg.sales += sales
                    agg.orders += orders
                    agg.clicks += clicks
                    agg.impressions += impressions
                    if is_waste:
                        agg.waste_spend += spend
                        agg.waste_term_count += 1
                    _push_top_term(agg, term=term, spend=spend, top_k=top_terms_per_ngram)
    except Exception:
        return pd.DataFrame()

    if not aggs:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for (ad_type, n, ng), agg in aggs.items():
        spend = float(agg.spend or 0.0)
        sales = float(agg.sales or 0.0)
        orders = float(agg.orders or 0.0)
        clicks = float(agg.clicks or 0.0)
        impressions = float(agg.impressions or 0.0)
        rows.append(
            {
                "ad_type": ad_type,
                "n": int(n),
                "ngram": str(ng),
                "term_count": int(agg.term_count or 0),
                "spend": round(spend, 2),
                "sales": round(sales, 2),
                "orders": int(round(orders)),
                "acos": round(safe_div(spend, sales), 4),
                "clicks": int(round(clicks)),
                "impressions": int(round(impressions)),
                "ctr": round(safe_div(clicks, impressions), 4),
                "cvr": round(safe_div(orders, clicks), 4),
                "waste_spend": round(float(agg.waste_spend or 0.0), 2),
                "waste_term_count": int(agg.waste_term_count or 0),
                "top_terms": " | ".join([t for _, t in agg.top_terms][: int(top_terms_per_ngram or 0)]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 默认排序：spend -> sales -> waste_spend（方便扫主题强弱）
    try:
        df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0.0)
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0)
        df["waste_spend"] = pd.to_numeric(df["waste_spend"], errors="coerce").fillna(0.0)
        df = df.sort_values(["spend", "sales", "waste_spend"], ascending=[False, False, False]).copy()
    except Exception:
        pass

    try:
        mr = int(max_rows or 0)
        if mr > 0:
            df = df.head(mr).copy()
    except Exception:
        pass

    # 列顺序稳定（方便 Excel/透视）
    cols = [
        "ad_type",
        "n",
        "ngram",
        "term_count",
        "spend",
        "sales",
        "orders",
        "acos",
        "clicks",
        "impressions",
        "ctr",
        "cvr",
        "waste_spend",
        "waste_term_count",
        "top_terms",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].reset_index(drop=True)


@dataclass
class _EntityAgg:
    spend: float = 0.0
    sales: float = 0.0
    orders: float = 0.0
    waste_spend: float = 0.0


def _norm_text_cell(x: object) -> str:
    """
    清理文本单元格，避免换行/表格符号影响 CSV/Markdown 可读性。
    """
    try:
        s = str(x or "").replace("\n", " ").replace("\r", " ").replace("|", "｜").strip()
        return s
    except Exception:
        return ""


def _fmt_top_entities(items: List[Tuple[float, str]], top_k: int) -> str:
    """
    items: [(score, text), ...] 已排序
    """
    try:
        k = int(top_k or 0)
        if k <= 0:
            return ""
        out = []
        for _, t in items[:k]:
            tt = _norm_text_cell(t)
            if tt:
                out.append(tt)
        return " | ".join(out)
    except Exception:
        return ""


def _build_topic_entity_aggs(
    search_term_report: pd.DataFrame,
    topic_keys: set[Tuple[str, int, str]],
    n_values: List[int],
    min_term_spend: float,
    max_terms: int,
    waste_min_clicks: int = 0,
    waste_min_spend: float = 0.0,
) -> Tuple[
    Dict[Tuple[str, int, str], Dict[str, _EntityAgg]],
    Dict[Tuple[str, int, str], Dict[str, _EntityAgg]],
    Dict[Tuple[str, int, str], Dict[str, _EntityAgg]],
]:
    """
    为选中的 topic_keys 生成 “落地定位”聚合：
    - topic -> campaign 维度聚合
    - topic -> campaign/ad_group 维度聚合
    - topic -> match_type 维度聚合（用于判断是否主要来自 broad/phrase/exact）
    """
    camp_aggs: Dict[Tuple[str, int, str], Dict[str, _EntityAgg]] = {}
    ag_aggs: Dict[Tuple[str, int, str], Dict[str, _EntityAgg]] = {}
    mt_aggs: Dict[Tuple[str, int, str], Dict[str, _EntityAgg]] = {}

    if search_term_report is None or search_term_report.empty:
        return camp_aggs, ag_aggs, mt_aggs
    if CAN.search_term not in search_term_report.columns:
        return camp_aggs, ag_aggs, mt_aggs
    if not topic_keys:
        return camp_aggs, ag_aggs, mt_aggs

    has_clicks = CAN.clicks in search_term_report.columns

    st = search_term_report.copy()
    for c in (CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders):
        if c not in st.columns:
            st[c] = 0.0
    if CAN.ad_type not in st.columns:
        st[CAN.ad_type] = ""
    if CAN.campaign not in st.columns:
        st[CAN.campaign] = ""
    if CAN.ad_group not in st.columns:
        st[CAN.ad_group] = ""
    if CAN.match_type not in st.columns:
        st[CAN.match_type] = ""

    try:
        st2 = st.copy()
        st2[CAN.ad_type] = st2[CAN.ad_type].fillna("").astype(str).str.strip()
        st2[CAN.search_term] = st2[CAN.search_term].fillna("").astype(str).str.strip()
        st2[CAN.campaign] = st2[CAN.campaign].fillna("").astype(str).str.strip()
        st2[CAN.ad_group] = st2[CAN.ad_group].fillna("").astype(str).str.strip()
        st2[CAN.match_type] = st2[CAN.match_type].fillna("").astype(str).str.strip()

        st2 = st2[(st2[CAN.search_term] != "") & (st2[CAN.search_term].str.lower() != "nan")].copy()
        if st2.empty:
            return camp_aggs, ag_aggs, mt_aggs

        for c in (CAN.impressions, CAN.clicks, CAN.spend, CAN.sales, CAN.orders):
            st2[c] = pd.to_numeric(st2[c], errors="coerce").fillna(0.0)

        # 聚合到“搜索词×活动×广告组×匹配类型”粒度（减少重复）
        g = (
            st2.groupby([CAN.ad_type, CAN.campaign, CAN.ad_group, CAN.match_type, CAN.search_term], dropna=False, as_index=False)
            .agg(
                spend=(CAN.spend, "sum"),
                sales=(CAN.sales, "sum"),
                orders=(CAN.orders, "sum"),
                clicks=(CAN.clicks, "sum"),
            )
            .copy()
        )
    except Exception:
        return camp_aggs, ag_aggs, mt_aggs

    try:
        min_spend = float(min_term_spend or 0.0)
        if min_spend > 0 and "spend" in g.columns:
            g = g[g["spend"] >= min_spend].copy()
    except Exception:
        pass

    # 性能保护：只取 spend TopN 的搜索词条目（长尾太大时防止爆炸）
    try:
        mt = int(max_terms or 0)
        if mt > 0 and len(g) > mt and "spend" in g.columns:
            g = g.sort_values(["spend"], ascending=[False]).head(mt).copy()
    except Exception:
        pass

    if g.empty:
        return camp_aggs, ag_aggs, mt_aggs

    n_list: List[int] = []
    for x in n_values:
        try:
            n = int(x)
            if 1 <= n <= 5:
                n_list.append(n)
        except Exception:
            continue
    if not n_list:
        n_list = [1, 2, 3]

    try:
        for _, r in g.iterrows():
            ad_type = str(r.get(CAN.ad_type, "") or "").strip()
            term = str(r.get(CAN.search_term, "") or "").strip()
            campaign = str(r.get(CAN.campaign, "") or "").strip()
            ad_group = str(r.get(CAN.ad_group, "") or "").strip()
            match_type = str(r.get(CAN.match_type, "") or "").strip()

            spend = float(r.get("spend", 0.0) or 0.0)
            sales = float(r.get("sales", 0.0) or 0.0)
            orders = float(r.get("orders", 0.0) or 0.0)
            clicks = float(r.get("clicks", 0.0) or 0.0)
            is_waste = is_waste_spend(
                orders=orders,
                sales=sales,
                spend=spend,
                clicks=(clicks if has_clicks else None),
                min_clicks=int(waste_min_clicks or 0),
                min_spend=float(waste_min_spend or 0.0),
            )

            tokens = _tokenize(term)
            if not tokens:
                continue

            for n in n_list:
                uniq = set(_iter_ngrams(tokens, n))
                for ng in uniq:
                    tkey = (ad_type, int(n), str(ng))
                    if tkey not in topic_keys:
                        continue

                    # campaign 聚合
                    if campaign:
                        bag = camp_aggs.get(tkey)
                        if bag is None:
                            bag = {}
                            camp_aggs[tkey] = bag
                        ea = bag.get(campaign)
                        if ea is None:
                            ea = _EntityAgg()
                            bag[campaign] = ea
                        ea.spend += spend
                        ea.sales += sales
                        ea.orders += orders
                        if is_waste:
                            ea.waste_spend += spend

                    # ad_group 聚合（带上 campaign，避免同名 ad_group 混淆）
                    if campaign or ad_group:
                        ag_key = f"{campaign} / {ad_group}".strip(" /")
                        if ag_key:
                            bag = ag_aggs.get(tkey)
                            if bag is None:
                                bag = {}
                                ag_aggs[tkey] = bag
                            ea = bag.get(ag_key)
                            if ea is None:
                                ea = _EntityAgg()
                                bag[ag_key] = ea
                            ea.spend += spend
                            ea.sales += sales
                            ea.orders += orders
                            if is_waste:
                                ea.waste_spend += spend

                    # match_type 聚合
                    if match_type:
                        bag = mt_aggs.get(tkey)
                        if bag is None:
                            bag = {}
                            mt_aggs[tkey] = bag
                        ea = bag.get(match_type)
                        if ea is None:
                            ea = _EntityAgg()
                            bag[match_type] = ea
                        ea.spend += spend
                        ea.sales += sales
                        ea.orders += orders
                        if is_waste:
                            ea.waste_spend += spend
    except Exception:
        return camp_aggs, ag_aggs, mt_aggs

    return camp_aggs, ag_aggs, mt_aggs


def build_keyword_topic_action_hints(
    search_term_report: Optional[pd.DataFrame],
    stage: str,
    policy: Optional[KeywordTopicsPolicy] = None,
    topics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    输出“关键词主题→可执行建议”清单（CSV）。

    说明：
    - 这是“主题线索”层，不做回写；用于帮助运营更快定位要处理的方向与位置；
    - 同一搜索词会贡献多个 n-gram，主题 spend 会重复计数；不要当成精确归因。
    """
    st = search_term_report.copy() if isinstance(search_term_report, pd.DataFrame) else pd.DataFrame()
    if st is None or st.empty:
        return pd.DataFrame()
    if CAN.search_term not in st.columns:
        return pd.DataFrame()

    ktp = policy if isinstance(policy, KeywordTopicsPolicy) else KeywordTopicsPolicy()
    if not bool(getattr(ktp, "action_hints_enabled", True)):
        return pd.DataFrame()

    # 阶段配置：用于“浪费口径”的噪声过滤（min_clicks/waste_spend）
    cfg = get_stage_config(stage)

    # 1) 主题表：优先复用外部已算好的 topics（避免重复 groupby）；否则自行生成
    topics_df = topics.copy() if isinstance(topics, pd.DataFrame) else pd.DataFrame()
    if topics_df is None or topics_df.empty:
        topics_df = build_keyword_topics(
            search_term_report=st,
            n_values=list(getattr(ktp, "n_values", [1, 2, 3]) or [1, 2, 3]),
            min_term_spend=float(getattr(ktp, "min_term_spend", 0.0) or 0.0),
            waste_min_clicks=int(getattr(cfg, "min_clicks", 0) or 0),
            waste_min_spend=float(getattr(cfg, "waste_spend", 0.0) or 0.0),
            max_terms=int(getattr(ktp, "max_terms", 5000) or 5000),
            max_rows=int(getattr(ktp, "max_rows", 2000) or 2000),
            top_terms_per_ngram=int(getattr(ktp, "top_terms_per_ngram", 3) or 3),
        )
    if topics_df is None or topics_df.empty:
        return pd.DataFrame()

    # 2) 计算派生指标与阈值
    target_acos = float(cfg.target_acos or 0.0)
    min_waste_spend = max(float(getattr(ktp, "action_hints_min_waste_spend", 0.0) or 0.0), float(cfg.waste_spend or 0.0))
    min_waste_ratio = float(getattr(ktp, "action_hints_min_waste_ratio", 0.0) or 0.0)
    max_scale_acos = target_acos * float(getattr(ktp, "action_hints_scale_acos_multiplier", 1.0) or 1.0)
    min_scale_sales = float(getattr(ktp, "action_hints_min_sales", 0.0) or 0.0)

    # n 过滤：默认与 dashboard 展示一致（md_min_n）
    min_n = int(getattr(ktp, "md_min_n", 2) or 2)
    if "n" in topics_df.columns:
        topics_df = topics_df[pd.to_numeric(topics_df["n"], errors="coerce").fillna(0).astype(int) >= int(min_n)].copy()
    if topics_df.empty:
        return pd.DataFrame()

    try:
        topics_df["spend"] = pd.to_numeric(topics_df.get("spend", 0.0), errors="coerce").fillna(0.0)
        topics_df["sales"] = pd.to_numeric(topics_df.get("sales", 0.0), errors="coerce").fillna(0.0)
        topics_df["orders"] = pd.to_numeric(topics_df.get("orders", 0.0), errors="coerce").fillna(0.0)
        topics_df["acos"] = pd.to_numeric(topics_df.get("acos", 0.0), errors="coerce").fillna(0.0)
        topics_df["waste_spend"] = pd.to_numeric(topics_df.get("waste_spend", 0.0), errors="coerce").fillna(0.0)
    except Exception:
        pass

    def _waste_ratio_row(r: pd.Series) -> float:
        try:
            return round(safe_div(float(r.get("waste_spend", 0.0) or 0.0), float(r.get("spend", 0.0) or 0.0)), 4)
        except Exception:
            return 0.0

    try:
        topics_df["waste_ratio"] = topics_df.apply(_waste_ratio_row, axis=1)
    except Exception:
        topics_df["waste_ratio"] = 0.0

    # 3) 选择要输出的 Top 主题 key（减少后续“落地定位”聚合成本）
    try:
        top_waste = int(getattr(ktp, "action_hints_top_waste", 20) or 20)
        top_scale = int(getattr(ktp, "action_hints_top_scale", 20) or 20)
        top_waste = max(0, min(200, top_waste))
        top_scale = max(0, min(200, top_scale))
    except Exception:
        top_waste = 20
        top_scale = 20

    waste_candidates = topics_df[
        (topics_df["waste_spend"] >= float(min_waste_spend))
        & (topics_df["waste_ratio"] >= float(min_waste_ratio))
        & (topics_df["spend"] > 0)
    ].copy()
    if not waste_candidates.empty:
        waste_candidates = waste_candidates.sort_values(["waste_spend", "spend"], ascending=[False, False]).head(top_waste).copy()

    scale_candidates = topics_df[
        (topics_df["sales"] >= float(min_scale_sales))
        & (topics_df["sales"] > 0)
        & (topics_df["acos"] <= float(max_scale_acos))
    ].copy()
    if not scale_candidates.empty:
        scale_candidates = scale_candidates.sort_values(["sales", "spend"], ascending=[False, False]).head(top_scale).copy()

    if (waste_candidates is None or waste_candidates.empty) and (scale_candidates is None or scale_candidates.empty):
        return pd.DataFrame()

    # 选中 topics keys
    topic_keys: set[Tuple[str, int, str]] = set()
    try:
        for df in (waste_candidates, scale_candidates):
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                ad_type = str(r.get("ad_type", "") or "").strip()
                n = int(r.get("n", 0) or 0)
                ng = str(r.get("ngram", "") or "").strip()
                if ad_type and n > 0 and ng:
                    topic_keys.add((ad_type, n, ng))
    except Exception:
        topic_keys = set()

    # 4) 构建落地定位（campaign/ad_group/match_type）
    camp_aggs, ag_aggs, mt_aggs = _build_topic_entity_aggs(
        search_term_report=st,
        topic_keys=topic_keys,
        n_values=list(getattr(ktp, "n_values", [1, 2, 3]) or [1, 2, 3]),
        min_term_spend=float(getattr(ktp, "min_term_spend", 0.0) or 0.0),
        max_terms=int(getattr(ktp, "max_terms", 5000) or 5000),
        waste_min_clicks=int(getattr(cfg, "min_clicks", 0) or 0),
        waste_min_spend=float(getattr(cfg, "waste_spend", 0.0) or 0.0),
    )

    top_k = int(getattr(ktp, "action_hints_top_entities", 3) or 3)
    if top_k < 1:
        top_k = 1
    if top_k > 10:
        top_k = 10

    def _fmt_campaigns(tkey: Tuple[str, int, str], direction: str) -> str:
        bag = camp_aggs.get(tkey, {})
        if not bag:
            return ""
        items: List[Tuple[float, str]] = []
        for name, ea in bag.items():
            if direction == "reduce":
                score = float(ea.waste_spend or 0.0)
                txt = f"{name} (waste={ea.waste_spend:.0f}, spend={ea.spend:.0f})"
            else:
                score = float(ea.sales or 0.0)
                acos = safe_div(float(ea.spend or 0.0), float(ea.sales or 0.0))
                txt = f"{name} (sales={ea.sales:.0f}, spend={ea.spend:.0f}, acos={acos:.2f})"
            items.append((score, txt))
        items = sorted(items, key=lambda x: float(x[0]), reverse=True)
        return _fmt_top_entities(items, top_k=top_k)

    def _fmt_ad_groups(tkey: Tuple[str, int, str], direction: str) -> str:
        bag = ag_aggs.get(tkey, {})
        if not bag:
            return ""
        items: List[Tuple[float, str]] = []
        for name, ea in bag.items():
            if direction == "reduce":
                score = float(ea.waste_spend or 0.0)
                txt = f"{name} (waste={ea.waste_spend:.0f})"
            else:
                score = float(ea.sales or 0.0)
                txt = f"{name} (sales={ea.sales:.0f})"
            items.append((score, txt))
        items = sorted(items, key=lambda x: float(x[0]), reverse=True)
        return _fmt_top_entities(items, top_k=top_k)

    def _fmt_match_types(tkey: Tuple[str, int, str], direction: str) -> str:
        bag = mt_aggs.get(tkey, {})
        if not bag:
            return ""
        items: List[Tuple[float, str]] = []
        for name, ea in bag.items():
            if direction == "reduce":
                score = float(ea.waste_spend or 0.0)
                txt = f"{name} (waste={ea.waste_spend:.0f})"
            else:
                score = float(ea.sales or 0.0)
                txt = f"{name} (sales={ea.sales:.0f})"
            items.append((score, txt))
        items = sorted(items, key=lambda x: float(x[0]), reverse=True)
        return _fmt_top_entities(items, top_k=top_k)

    # 5) 组装 action_hints 表（稳定列顺序，便于运营 Excel 筛选）
    rows: List[Dict[str, object]] = []
    # P0 阈值（随 stage 调整，避免太激进）
    p0_waste_spend = float(cfg.waste_spend or 0.0) * 3.0

    def _build_human_step(
        direction: str,
        risk_level: str,
        top_match_types: str,
        waste_ratio: float,
        acos: float,
        target_acos_val: float,
    ) -> Tuple[str, str, str, str]:
        """
        生成更“可执行/有人味”的建议四件套：
        - execution_style: 执行风格
        - expected_signal: 观察成功信号
        - rollback_guard: 回滚护栏
        - next_step: 具体下一步
        """
        mt = str(top_match_types or "")
        has_broad = "广泛匹配" in mt
        target = max(float(target_acos_val or 0.0), 1e-6)
        if direction == "reduce":
            if str(risk_level) == "high":
                style = "先止损后保量"
                step = (
                    "先从高浪费词和广泛匹配词开始处理：先否词3-5个并小步降价，"
                    "当天只做一轮，避免把还在出单的词一次性误杀。"
                )
            elif str(risk_level) == "medium":
                style = "稳健降噪"
                step = "先按 top_campaigns 定位主题词，优先清理低意图词；降价优先于大范围否词。"
            else:
                style = "精修长尾"
                step = "把该主题当作低优先级清理项：先处理长尾词，再观察是否需要继续收口。"
            if has_broad:
                step += " 先处理广泛匹配，再看词组/精确，通常更稳。"
            expected = "24-48h 内 waste_spend 下降且订单基本稳定，即可继续下一批。"
            rollback = "若订单下滑>15%或广告销售下滑>20%，回滚最近一轮否词/降价。"
            return style, expected, rollback, step

        # scale
        acos_ratio = safe_div(float(acos), target)
        if str(risk_level) == "high" or acos_ratio > 1.0:
            style = "谨慎试投"
            step = (
                "先做小预算测试：只扩1-2个精确词并小步提价，"
                "确认转化稳定后再加预算，避免直接放大造成成本失控。"
            )
        elif str(risk_level) == "medium":
            style = "小步放量"
            step = "优先把高转化词加精确并提价 5%~10%，连续两天稳定后再加预算。"
        else:
            style = "顺势放量"
            step = "该主题可作为优先放量池：先加精确词，再按转化表现逐步加预算。"
        if waste_ratio > 0.4:
            step += " 同时保留否词检查，避免放量时把低意图词一并放大。"
        expected = "48h 内订单/销售提升且 ACoS 不劣化，再进入第二轮加码。"
        rollback = "若 ACoS 连续两天高于目标且订单无提升，撤回最近一次提价/加预算。"
        return style, expected, rollback, step

    def _append_rows(df: pd.DataFrame, direction: str) -> None:
        for _, r in df.iterrows():
            ad_type = str(r.get("ad_type", "") or "").strip()
            n = int(r.get("n", 0) or 0)
            ng = str(r.get("ngram", "") or "").strip()
            tkey = (ad_type, n, ng)
            spend = float(r.get("spend", 0.0) or 0.0)
            sales = float(r.get("sales", 0.0) or 0.0)
            orders = int(float(r.get("orders", 0) or 0))
            acos = float(r.get("acos", 0.0) or 0.0)
            waste_spend = float(r.get("waste_spend", 0.0) or 0.0)
            waste_ratio = float(r.get("waste_ratio", 0.0) or 0.0)
            term_count = int(float(r.get("term_count", 0) or 0))
            waste_term_count = int(float(r.get("waste_term_count", 0) or 0))
            top_terms = _norm_text_cell(r.get("top_terms", ""))

            target_base = max(float(target_acos or 0.0), 1e-6)
            # 统一优先级分：用于跨主题快速排序，不替代原始口径
            if direction == "reduce":
                waste_pressure = safe_div(waste_spend, max(float(cfg.waste_spend or 0.0), 1.0))
                over_target = max(0.0, acos - float(target_acos or 0.0))
                hint_priority_score = (
                    60.0
                    + waste_pressure * 10.0
                    + waste_ratio * 80.0
                    + min(40.0, over_target * 80.0)
                    + min(20.0, float(waste_term_count))
                )
            else:
                efficiency_gain = max(0.0, safe_div(float(target_acos or 0.0) - acos, target_base))
                hint_priority_score = (
                    40.0
                    + min(60.0, sales * 0.12)
                    + min(40.0, float(orders) * 5.0)
                    + efficiency_gain * 60.0
                )
            hint_priority_score = round(float(hint_priority_score), 2)

            if direction == "reduce":
                priority = "P0" if waste_spend >= p0_waste_spend else "P1"
                hint_action = "否词/降价/收预算"
                owner = "广告运营"
                if waste_spend >= p0_waste_spend * 1.5 or waste_ratio >= 0.85:
                    risk_level = "high"
                elif waste_spend >= p0_waste_spend or waste_ratio >= 0.7:
                    risk_level = "medium"
                else:
                    risk_level = "low"
            else:
                priority = "P1"
                hint_action = "加精确词/提价/加预算"
                owner = "广告运营"
                # 放量主题按 ACoS 接近目标线给风险分层，避免“好词误加码”
                if acos > float(target_acos or 0.0):
                    risk_level = "high"
                elif acos > float(target_acos or 0.0) * 0.85:
                    risk_level = "medium"
                else:
                    risk_level = "low"

            top_campaigns = _fmt_campaigns(tkey, direction=direction)
            top_ad_groups = _fmt_ad_groups(tkey, direction=direction)
            top_match_types = _fmt_match_types(tkey, direction=direction)
            execution_style, expected_signal, rollback_guard, next_step = _build_human_step(
                direction=direction,
                risk_level=risk_level,
                top_match_types=top_match_types,
                waste_ratio=waste_ratio,
                acos=acos,
                target_acos_val=float(target_acos or 0.0),
            )

            rows.append(
                {
                    "priority": priority,
                    "hint_priority_score": hint_priority_score,
                    "owner": owner,
                    "risk_level": risk_level,
                    "direction": direction,
                    "hint_action": hint_action,
                    "ad_type": ad_type,
                    "n": n,
                    "ngram": ng,
                    "spend": round(spend, 2),
                    "sales": round(sales, 2),
                    "orders": orders,
                    "acos": round(acos, 4),
                    "waste_spend": round(waste_spend, 2),
                    "waste_ratio": round(waste_ratio, 4),
                    "term_count": term_count,
                    "waste_term_count": waste_term_count,
                    "top_terms": top_terms,
                    "top_campaigns": top_campaigns,
                    "top_ad_groups": top_ad_groups,
                    "top_match_types": top_match_types,
                    "execution_style": execution_style,
                    "expected_signal": expected_signal,
                    "rollback_guard": rollback_guard,
                    "filter_contains": ng,
                    "next_step": next_step,
                }
            )

    if waste_candidates is not None and not waste_candidates.empty:
        _append_rows(waste_candidates, direction="reduce")
    if scale_candidates is not None and not scale_candidates.empty:
        _append_rows(scale_candidates, direction="scale")

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # 排序：priority -> 方向 -> 影响（浪费/销售）
    try:
        out["_p"] = out["priority"].map(lambda x: 0 if str(x).upper() == "P0" else 1)
        out["_dir"] = out["direction"].map(lambda x: 0 if str(x) == "reduce" else 1)
        out["_score"] = pd.to_numeric(out.get("hint_priority_score", 0.0), errors="coerce").fillna(0.0)
        out["_impact"] = out.apply(
            lambda rr: float(rr.get("waste_spend", 0.0) or 0.0) if str(rr.get("direction", "")) == "reduce" else float(rr.get("sales", 0.0) or 0.0),
            axis=1,
        )
        out = out.sort_values(["_p", "_dir", "_score", "_impact"], ascending=[True, True, False, False]).copy()
        out = out.drop(columns=["_p", "_dir", "_score", "_impact"], errors="ignore")
    except Exception:
        pass

    cols = [
        "priority",
        "hint_priority_score",
        "owner",
        "risk_level",
        "direction",
        "hint_action",
        "ad_type",
        "n",
        "ngram",
        "spend",
        "sales",
        "orders",
        "acos",
        "waste_spend",
        "waste_ratio",
        "term_count",
        "waste_term_count",
        "top_terms",
        "top_campaigns",
        "top_ad_groups",
        "top_match_types",
        "execution_style",
        "expected_signal",
        "rollback_guard",
        "filter_contains",
        "next_step",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


@dataclass
class _TopicAsinAgg:
    """
    topic -> asin 的聚合器（用于“主题→产品语境”）。
    """

    term_count: int = 0
    spend: float = 0.0
    sales: float = 0.0
    orders: float = 0.0
    waste_spend: float = 0.0
    waste_term_count: int = 0
    confidence_spend_sum: float = 0.0
    # 解释用样例
    top_terms: List[Tuple[float, str]] = field(default_factory=list)
    top_campaigns: List[Tuple[float, str]] = field(default_factory=list)
    top_match_types: List[Tuple[float, str]] = field(default_factory=list)


def _push_top_text(items: List[Tuple[float, str]], text: str, score: float, top_k: int) -> List[Tuple[float, str]]:
    """
    维护一个按 score 降序的 TopK 文本列表（去重）。
    """
    try:
        k = int(top_k or 0)
        if k <= 0:
            return items
        t = _norm_text_cell(text)
        if not t:
            return items
        s = float(score or 0.0)
        # 去重（同文本保留最大分）
        out = []
        found = False
        for sc, tt in items:
            if tt == t:
                out.append((max(float(sc), s), t))
                found = True
            else:
                out.append((float(sc), tt))
        if not found:
            out.append((s, t))
        out = sorted(out, key=lambda x: float(x[0]), reverse=True)[:k]
        return out
    except Exception:
        return items


def _norm_category(x: object) -> str:
    try:
        s = str(x or "").strip()
        if not s or s.lower() == "nan":
            return "（未分类）"
        if s in {"未分类", "(未分类)", "（未分类）"}:
            return "（未分类）"
        return s
    except Exception:
        return "（未分类）"


def _norm_phase(x: object) -> str:
    try:
        s = str(x or "").strip().lower()
        if not s or s == "nan":
            return "unknown"
        return s
    except Exception:
        return "unknown"


def _build_high_confidence_term_asin_top1(
    asin_top_search_terms: pd.DataFrame,
    min_confidence: float,
) -> pd.DataFrame:
    """
    从 asin_top_search_terms 反推 term→asin 的 top1 映射，并按 share>=min_confidence 过滤。

    输出列（至少）：
    - ad_type, campaign, match_type, search_term, asin, spend, sales, orders, share
    """
    st = asin_top_search_terms.copy() if isinstance(asin_top_search_terms, pd.DataFrame) else pd.DataFrame()
    if st is None or st.empty:
        return pd.DataFrame()
    if CAN.search_term not in st.columns or CAN.asin not in st.columns:
        return pd.DataFrame()

    try:
        df = st.copy()
        if CAN.ad_type not in df.columns:
            df[CAN.ad_type] = ""
        if CAN.campaign not in df.columns:
            df[CAN.campaign] = ""
        if CAN.match_type not in df.columns:
            df[CAN.match_type] = ""

        for c in (CAN.ad_type, CAN.campaign, CAN.match_type, CAN.search_term, CAN.asin):
            df[c] = df[c].fillna("").astype(str).str.strip()
        df[CAN.asin] = df[CAN.asin].astype(str).str.upper().str.strip()
        df = df[(df[CAN.search_term] != "") & (df[CAN.search_term].str.lower() != "nan")].copy()
        df = df[(df[CAN.asin] != "") & (df[CAN.asin].str.lower() != "nan")].copy()
        if df.empty:
            return pd.DataFrame()

        for c in ("spend", "sales", "orders"):
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        group_cols = [CAN.ad_type, CAN.campaign, CAN.match_type, CAN.search_term]
        totals = df.groupby(group_cols, dropna=False, as_index=False).agg(total_spend=("spend", "sum")).copy()
        merged = df.merge(totals, on=group_cols, how="left")
        merged["total_spend"] = pd.to_numeric(merged.get("total_spend", 0.0), errors="coerce").fillna(0.0)
        merged["share"] = 0.0
        mask = merged["total_spend"] > 0
        merged.loc[mask, "share"] = (merged.loc[mask, "spend"] / merged.loc[mask, "total_spend"]).fillna(0.0)

        # 选 top1 asin（按 spend 排序稳定）
        merged = merged.sort_values(["spend", "sales", "orders"], ascending=[False, False, False]).copy()
        merged["_rank"] = merged.groupby(group_cols, dropna=False).cumcount() + 1

        mc = max(0.0, min(1.0, float(min_confidence)))
        top1 = merged[(merged["_rank"] == 1) & (merged["share"] >= mc)].copy()
        if top1.empty:
            return pd.DataFrame()

        keep_cols = group_cols + [CAN.asin, "spend", "sales", "orders", "share"]
        keep_cols2 = [c for c in keep_cols if c in top1.columns]
        return top1[keep_cols2].copy()
    except Exception:
        return pd.DataFrame()


def build_keyword_topic_asin_context(
    asin_top_search_terms: Optional[pd.DataFrame],
    asin_cockpit: Optional[pd.DataFrame],
    topic_hints: Optional[pd.DataFrame],
    stage: str,
    policy: Optional[KeywordTopicsPolicy] = None,
) -> pd.DataFrame:
    """
    生成“关键词主题 → 产品语境（ASIN/类目/生命周期/库存覆盖）”表。

    输入：
    - asin_top_search_terms：已分摊到 ASIN 的 search_term 明细（来自 pipeline 的 asin_top_search_terms_df）
    - asin_cockpit：ASIN 总览（focus + drivers + 产品维度）
    - topic_hints：主题建议清单（keyword_topics_action_hints.csv 的 DataFrame；用于限定主题范围与方向）

    输出：
    - 每行 = 1 个 (topic, asin)
    - 只使用“高置信”的 search_term→ASIN 映射（top1 spend share >= asin_context_min_confidence）
    """
    st = asin_top_search_terms.copy() if isinstance(asin_top_search_terms, pd.DataFrame) else pd.DataFrame()
    if st is None or st.empty:
        return pd.DataFrame()
    if CAN.search_term not in st.columns or CAN.asin not in st.columns:
        return pd.DataFrame()

    hints = topic_hints.copy() if isinstance(topic_hints, pd.DataFrame) else pd.DataFrame()
    if hints is None or hints.empty:
        return pd.DataFrame()

    ktp = policy if isinstance(policy, KeywordTopicsPolicy) else KeywordTopicsPolicy()
    if not bool(getattr(ktp, "asin_context_enabled", True)):
        return pd.DataFrame()

    # 阶段阈值：用于“浪费口径”的噪声过滤（主要用 waste_spend）
    cfg = get_stage_config(stage)

    # 主题范围：默认只对“已进入 action_hints 的主题”做产品语境（避免输出过大）
    topic_set: set[Tuple[str, int, str]] = set()
    topic_meta: Dict[Tuple[str, int, str], Dict[str, str]] = {}
    try:
        for _, r in hints.iterrows():
            ad_type = str(r.get("ad_type", "") or "").strip()
            n = int(float(r.get("n", 0) or 0))
            ng = str(r.get("ngram", "") or "").strip()
            if not ad_type or n <= 0 or not ng:
                continue
            key = (ad_type, n, ng)
            topic_set.add(key)
            topic_meta[key] = {
                "priority": str(r.get("priority", "") or "").strip(),
                "direction": str(r.get("direction", "") or "").strip(),
                "hint_action": str(r.get("hint_action", "") or "").strip(),
            }
    except Exception:
        topic_set = set()
        topic_meta = {}
    if not topic_set:
        return pd.DataFrame()

    # ===== 1) 计算 search_term→ASIN 的“高置信 top1”映射 =====
    min_conf = float(getattr(ktp, "asin_context_min_confidence", 0.6) or 0.6)
    top1 = _build_high_confidence_term_asin_top1(st, min_conf)
    if top1 is None or top1.empty:
        return pd.DataFrame()

    # ===== 2) 基于 top1 term→asin，把 n-gram 主题落到 ASIN =====
    # 只处理在 topic_set 里的主题（来自 action_hints），避免输出爆炸。
    n_list: List[int] = []
    try:
        for x in list(getattr(ktp, "n_values", [1, 2, 3]) or [1, 2, 3]):
            n = int(x)
            if 1 <= n <= 5:
                n_list.append(n)
    except Exception:
        n_list = [1, 2, 3]
    if not n_list:
        n_list = [1, 2, 3]

    # Top 样例条数：复用已有参数，避免新增过多配置项
    top_terms_k = int(getattr(ktp, "top_terms_per_ngram", 3) or 3)
    top_entities_k = int(getattr(ktp, "action_hints_top_entities", 3) or 3)

    aggs: Dict[Tuple[str, int, str, str], _TopicAsinAgg] = {}
    try:
        for _, r in top1.iterrows():
            ad_type = str(r.get(CAN.ad_type, "") or "").strip()
            campaign = str(r.get(CAN.campaign, "") or "").strip()
            match_type = str(r.get(CAN.match_type, "") or "").strip()
            term = str(r.get(CAN.search_term, "") or "").strip()
            asin = str(r.get(CAN.asin, "") or "").strip().upper()
            if not ad_type or not term or not asin:
                continue

            spend = float(r.get("spend", 0.0) or 0.0)
            sales = float(r.get("sales", 0.0) or 0.0)
            orders = float(r.get("orders", 0.0) or 0.0)
            confidence = float(r.get("share", 0.0) or 0.0)
            is_waste = is_waste_spend(
                orders=orders,
                sales=sales,
                spend=spend,
                clicks=None,
                min_clicks=int(getattr(cfg, "min_clicks", 0) or 0),
                min_spend=float(getattr(cfg, "waste_spend", 0.0) or 0.0),
            )

            tokens = _tokenize(term)
            if not tokens:
                continue

            for n in n_list:
                uniq = set(_iter_ngrams(tokens, n))
                for ng in uniq:
                    tkey = (ad_type, int(n), str(ng))
                    if tkey not in topic_set:
                        continue
                    key2 = (ad_type, int(n), str(ng), asin)
                    agg = aggs.get(key2)
                    if agg is None:
                        agg = _TopicAsinAgg()
                        aggs[key2] = agg
                    agg.term_count += 1
                    agg.spend += spend
                    agg.sales += sales
                    agg.orders += orders
                    if is_waste:
                        agg.waste_spend += spend
                        agg.waste_term_count += 1
                    agg.confidence_spend_sum += confidence * spend
                    agg.top_terms = _push_top_text(agg.top_terms, text=term, score=spend, top_k=top_terms_k)
                    agg.top_campaigns = _push_top_text(agg.top_campaigns, text=campaign, score=spend, top_k=top_entities_k)
                    agg.top_match_types = _push_top_text(agg.top_match_types, text=match_type, score=spend, top_k=top_entities_k)
    except Exception:
        return pd.DataFrame()

    if not aggs:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for (ad_type, n, ng, asin), agg in aggs.items():
        spend = float(agg.spend or 0.0)
        sales = float(agg.sales or 0.0)
        waste_spend = float(agg.waste_spend or 0.0)
        direction = topic_meta.get((ad_type, int(n), str(ng)), {}).get("direction", "")
        priority = topic_meta.get((ad_type, int(n), str(ng)), {}).get("priority", "")
        hint_action = topic_meta.get((ad_type, int(n), str(ng)), {}).get("hint_action", "")
        rows.append(
            {
                "priority": priority,
                "direction": direction,
                "hint_action": hint_action,
                "ad_type": ad_type,
                "n": int(n),
                "ngram": str(ng),
                "asin": str(asin),
                "topic_spend": round(spend, 2),
                "topic_sales": round(sales, 2),
                "topic_orders": int(round(float(agg.orders or 0.0))),
                "topic_acos": round(safe_div(spend, sales), 4),
                "topic_waste_spend": round(waste_spend, 2),
                "topic_waste_ratio": round(safe_div(waste_spend, spend), 4),
                "term_count": int(agg.term_count or 0),
                "waste_term_count": int(agg.waste_term_count or 0),
                "avg_term_confidence": round(safe_div(float(agg.confidence_spend_sum or 0.0), spend), 4) if spend > 0 else 0.0,
                "top_terms": " | ".join([t for _, t in agg.top_terms]),
                "top_campaigns": " | ".join([t for _, t in agg.top_campaigns]),
                "top_match_types": " | ".join([t for _, t in agg.top_match_types]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # ===== 3) 合并 ASIN 语境（类目/生命周期/库存覆盖）=====
    ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
    if ac is not None and not ac.empty and "asin" in ac.columns:
        try:
            ac2 = ac.copy()
            ac2["asin_norm"] = ac2["asin"].astype(str).str.upper().str.strip()
            out["asin_norm"] = out["asin"].astype(str).str.upper().str.strip()
            # 只挑核心列（避免输出太长）
            keep_cols = [
                "asin_norm",
                "product_category",
                "product_name",
                "current_phase",
                "cycle_id",
                "inventory",
                "inventory_cover_days_7d",
                "inventory_cover_days_14d",
                "inventory_cover_days_30d",
                "sales_per_day_7d",
                "ad_spend_roll",
                "profit_direction",
                "focus_score",
            ]
            keep_cols2 = [c for c in keep_cols if c in ac2.columns]
            ac2 = ac2[keep_cols2].drop_duplicates("asin_norm", keep="first").copy() if keep_cols2 else pd.DataFrame()
            if ac2 is not None and not ac2.empty and "asin_norm" in ac2.columns:
                out = out.merge(ac2, on="asin_norm", how="left")
        except Exception:
            pass
        try:
            out = out.drop(columns=["asin_norm"], errors="ignore")
        except Exception:
            pass
        # 统一类目/阶段兜底值，避免出现空值导致分组/筛选困难
        try:
            if "product_category" in out.columns:
                out["product_category"] = out["product_category"].map(_norm_category)
            if "current_phase" in out.columns:
                out["current_phase"] = out["current_phase"].map(_norm_phase)
        except Exception:
            pass

    # ===== 4) 每个主题只保留 Top ASIN（避免输出过大）=====
    top_asins = int(getattr(ktp, "asin_context_top_asins_per_topic", 10) or 10)
    if top_asins < 1:
        top_asins = 1
    if top_asins > 50:
        top_asins = 50

    try:
        def _rank_metric(df: pd.DataFrame) -> str:
            d = str(df.get("direction", "").iloc[0] if len(df) else "")
            return "topic_waste_spend" if d == "reduce" else "topic_sales"

        parts: List[pd.DataFrame] = []
        for (ad_type, n, ng), grp in out.groupby(["ad_type", "n", "ngram"], dropna=False):
            g = grp.copy()
            metric = _rank_metric(g)
            if metric in g.columns:
                g[metric] = pd.to_numeric(g.get(metric, 0.0), errors="coerce").fillna(0.0)
            g["topic_spend"] = pd.to_numeric(g.get("topic_spend", 0.0), errors="coerce").fillna(0.0)
            # reduce: waste_spend -> spend; scale: sales -> spend
            if metric == "topic_waste_spend":
                g = g.sort_values(["topic_waste_spend", "topic_spend"], ascending=[False, False])
            else:
                g = g.sort_values(["topic_sales", "topic_spend"], ascending=[False, False])
            parts.append(g.head(top_asins).copy())
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    except Exception:
        pass

    # 列顺序稳定（方便 Excel/透视）
    cols = [
        "priority",
        "direction",
        "hint_action",
        "ad_type",
        "n",
        "ngram",
        "product_category",
        "asin",
        "product_name",
        "current_phase",
        "cycle_id",
        "inventory",
        "inventory_cover_days_7d",
        "sales_per_day_7d",
        "profit_direction",
        "focus_score",
        "topic_spend",
        "topic_sales",
        "topic_orders",
        "topic_acos",
        "topic_waste_spend",
        "topic_waste_ratio",
        "term_count",
        "waste_term_count",
        "avg_term_confidence",
        "top_terms",
        "top_campaigns",
        "top_match_types",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True) if cols else out.reset_index(drop=True)


@dataclass
class _TopicSegmentAgg:
    term_count: int = 0
    spend: float = 0.0
    sales: float = 0.0
    orders: float = 0.0
    waste_spend: float = 0.0
    waste_term_count: int = 0
    confidence_spend_sum: float = 0.0
    asins: set[str] = field(default_factory=set)
    top_terms: List[Tuple[float, str]] = field(default_factory=list)
    top_campaigns: List[Tuple[float, str]] = field(default_factory=list)
    top_match_types: List[Tuple[float, str]] = field(default_factory=list)
    top_asins: List[Tuple[float, str]] = field(default_factory=list)


def build_keyword_topic_category_phase_summary(
    asin_top_search_terms: Optional[pd.DataFrame],
    asin_cockpit: Optional[pd.DataFrame],
    topic_hints: Optional[pd.DataFrame],
    stage: str,
    policy: Optional[KeywordTopicsPolicy] = None,
) -> pd.DataFrame:
    """
    生成“主题→类目/生命周期”汇总表：
    - 先按 类目(product_category) 与 阶段(current_phase) 看该主题的 spend/sales/waste_spend
    - 再下钻到 ASIN（配合 keyword_topics_asin_context.csv）
    """
    st = asin_top_search_terms.copy() if isinstance(asin_top_search_terms, pd.DataFrame) else pd.DataFrame()
    if st is None or st.empty:
        return pd.DataFrame()
    if CAN.search_term not in st.columns or CAN.asin not in st.columns:
        return pd.DataFrame()

    hints = topic_hints.copy() if isinstance(topic_hints, pd.DataFrame) else pd.DataFrame()
    if hints is None or hints.empty:
        return pd.DataFrame()

    ktp = policy if isinstance(policy, KeywordTopicsPolicy) else KeywordTopicsPolicy()
    if not bool(getattr(ktp, "asin_context_enabled", True)):
        return pd.DataFrame()

    # 阶段阈值：用于“浪费口径”的噪声过滤（主要用 waste_spend）
    cfg = get_stage_config(stage)

    # 主题集合（仅对已进入 action_hints 的主题做汇总，避免输出爆炸）
    topic_set: set[Tuple[str, int, str]] = set()
    topic_meta: Dict[Tuple[str, int, str], Dict[str, str]] = {}
    try:
        for _, r in hints.iterrows():
            ad_type = str(r.get("ad_type", "") or "").strip()
            n = int(float(r.get("n", 0) or 0))
            ng = str(r.get("ngram", "") or "").strip()
            if not ad_type or n <= 0 or not ng:
                continue
            key = (ad_type, n, ng)
            topic_set.add(key)
            topic_meta[key] = {
                "priority": str(r.get("priority", "") or "").strip(),
                "direction": str(r.get("direction", "") or "").strip(),
                "hint_action": str(r.get("hint_action", "") or "").strip(),
            }
    except Exception:
        topic_set = set()
        topic_meta = {}
    if not topic_set:
        return pd.DataFrame()

    # 计算 term→asin 的高置信 top1 映射
    min_conf = float(getattr(ktp, "asin_context_min_confidence", 0.6) or 0.6)
    top1 = _build_high_confidence_term_asin_top1(st, min_conf)
    if top1 is None or top1.empty:
        return pd.DataFrame()

    # asin -> (category, phase) 映射（来自 asin_cockpit）
    asin_seg: Dict[str, Tuple[str, str]] = {}
    ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
    if ac is not None and not ac.empty and "asin" in ac.columns:
        try:
            ac2 = ac.copy()
            ac2["asin_norm"] = ac2["asin"].astype(str).str.upper().str.strip()
            if "product_category" not in ac2.columns:
                ac2["product_category"] = "（未分类）"
            if "current_phase" not in ac2.columns:
                ac2["current_phase"] = "unknown"
            ac2["product_category"] = ac2["product_category"].map(_norm_category)
            ac2["current_phase"] = ac2["current_phase"].map(_norm_phase)
            ac2 = ac2.drop_duplicates("asin_norm", keep="first")
            for _, r in ac2.iterrows():
                asin = str(r.get("asin_norm", "") or "").strip().upper()
                if not asin:
                    continue
                asin_seg[asin] = (
                    _norm_category(r.get("product_category", "")),
                    _norm_phase(r.get("current_phase", "")),
                )
        except Exception:
            asin_seg = {}

    # n 列表（仍然按 policy 的 n_values，但最终会被 topic_set 过滤）
    n_list: List[int] = []
    try:
        for x in list(getattr(ktp, "n_values", [1, 2, 3]) or [1, 2, 3]):
            n = int(x)
            if 1 <= n <= 5:
                n_list.append(n)
    except Exception:
        n_list = [1, 2, 3]
    if not n_list:
        n_list = [1, 2, 3]

    top_terms_k = int(getattr(ktp, "top_terms_per_ngram", 3) or 3)
    top_entities_k = int(getattr(ktp, "action_hints_top_entities", 3) or 3)

    aggs: Dict[Tuple[str, int, str, str, str], _TopicSegmentAgg] = {}
    try:
        for _, r in top1.iterrows():
            ad_type = str(r.get(CAN.ad_type, "") or "").strip()
            campaign = str(r.get(CAN.campaign, "") or "").strip()
            match_type = str(r.get(CAN.match_type, "") or "").strip()
            term = str(r.get(CAN.search_term, "") or "").strip()
            asin = str(r.get(CAN.asin, "") or "").strip().upper()
            if not ad_type or not term or not asin:
                continue

            # segment：类目/生命周期（缺失就兜底）
            cat, ph = asin_seg.get(asin, ("（未分类）", "unknown"))
            cat = _norm_category(cat)
            ph = _norm_phase(ph)

            spend = float(r.get("spend", 0.0) or 0.0)
            sales = float(r.get("sales", 0.0) or 0.0)
            orders = float(r.get("orders", 0.0) or 0.0)
            confidence = float(r.get("share", 0.0) or 0.0)
            is_waste = is_waste_spend(
                orders=orders,
                sales=sales,
                spend=spend,
                clicks=None,
                min_clicks=int(getattr(cfg, "min_clicks", 0) or 0),
                min_spend=float(getattr(cfg, "waste_spend", 0.0) or 0.0),
            )

            tokens = _tokenize(term)
            if not tokens:
                continue

            for n in n_list:
                uniq = set(_iter_ngrams(tokens, n))
                for ng in uniq:
                    tkey = (ad_type, int(n), str(ng))
                    if tkey not in topic_set:
                        continue
                    key2 = (ad_type, int(n), str(ng), cat, ph)
                    agg = aggs.get(key2)
                    if agg is None:
                        agg = _TopicSegmentAgg()
                        aggs[key2] = agg
                    agg.term_count += 1
                    agg.spend += spend
                    agg.sales += sales
                    agg.orders += orders
                    if is_waste:
                        agg.waste_spend += spend
                        agg.waste_term_count += 1
                    agg.confidence_spend_sum += confidence * spend
                    agg.asins.add(asin)
                    agg.top_terms = _push_top_text(agg.top_terms, text=term, score=spend, top_k=top_terms_k)
                    agg.top_campaigns = _push_top_text(agg.top_campaigns, text=campaign, score=spend, top_k=top_entities_k)
                    agg.top_match_types = _push_top_text(agg.top_match_types, text=match_type, score=spend, top_k=top_entities_k)
                    # Top ASIN：reduce 看 waste_spend（仅无单花费），scale 看 sales
                    direction = topic_meta.get(tkey, {}).get("direction", "")
                    score = (spend if is_waste else 0.0) if direction == "reduce" else sales
                    agg.top_asins = _push_top_text(agg.top_asins, text=asin, score=score, top_k=top_entities_k)
    except Exception:
        return pd.DataFrame()

    if not aggs:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for (ad_type, n, ng, cat, ph), agg in aggs.items():
        spend = float(agg.spend or 0.0)
        sales = float(agg.sales or 0.0)
        waste_spend = float(agg.waste_spend or 0.0)
        tkey = (ad_type, int(n), str(ng))
        rows.append(
            {
                "priority": topic_meta.get(tkey, {}).get("priority", ""),
                "direction": topic_meta.get(tkey, {}).get("direction", ""),
                "hint_action": topic_meta.get(tkey, {}).get("hint_action", ""),
                "ad_type": ad_type,
                "n": int(n),
                "ngram": str(ng),
                "product_category": _norm_category(cat),
                "current_phase": _norm_phase(ph),
                "asin_count": int(len(agg.asins)),
                "topic_spend": round(spend, 2),
                "topic_sales": round(sales, 2),
                "topic_orders": int(round(float(agg.orders or 0.0))),
                "topic_acos": round(safe_div(spend, sales), 4),
                "topic_waste_spend": round(waste_spend, 2),
                "topic_waste_ratio": round(safe_div(waste_spend, spend), 4),
                "term_count": int(agg.term_count or 0),
                "waste_term_count": int(agg.waste_term_count or 0),
                "avg_term_confidence": round(safe_div(float(agg.confidence_spend_sum or 0.0), spend), 4) if spend > 0 else 0.0,
                "top_asins": " | ".join([t for _, t in agg.top_asins]),
                "top_terms": " | ".join([t for _, t in agg.top_terms]),
                "top_campaigns": " | ".join([t for _, t in agg.top_campaigns]),
                "top_match_types": " | ".join([t for _, t in agg.top_match_types]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # 排序：先按方向（reduce 优先看 waste），再按指标
    try:
        out["_dir"] = out["direction"].map(lambda x: 0 if str(x) == "reduce" else 1)
        out["topic_waste_spend"] = pd.to_numeric(out.get("topic_waste_spend", 0.0), errors="coerce").fillna(0.0)
        out["topic_sales"] = pd.to_numeric(out.get("topic_sales", 0.0), errors="coerce").fillna(0.0)
        out["topic_spend"] = pd.to_numeric(out.get("topic_spend", 0.0), errors="coerce").fillna(0.0)
        out = out.sort_values(["_dir", "topic_waste_spend", "topic_sales", "topic_spend"], ascending=[True, False, False, False]).copy()
        out = out.drop(columns=["_dir"], errors="ignore")
    except Exception:
        pass

    cols = [
        "priority",
        "direction",
        "hint_action",
        "product_category",
        "current_phase",
        "ad_type",
        "n",
        "ngram",
        "asin_count",
        "topic_spend",
        "topic_sales",
        "topic_orders",
        "topic_acos",
        "topic_waste_spend",
        "topic_waste_ratio",
        "term_count",
        "waste_term_count",
        "avg_term_confidence",
        "top_asins",
        "top_terms",
        "top_campaigns",
        "top_match_types",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True) if cols else out.reset_index(drop=True)


def build_keyword_topic_segment_top(
    category_phase_summary: Optional[pd.DataFrame],
    policy: Optional[KeywordTopicsPolicy] = None,
) -> pd.DataFrame:
    """
    生成“Segment Top（类目×生命周期→Top 主题）”表。

    目标：
    - 运营先从「类目×阶段」看到 Top 浪费主题 / Top 贡献主题（各 TopN）；
    - 再下钻到 keyword_topics_action_hints / keyword_topics_asin_context 做具体执行。

    输入：
    - category_phase_summary：`build_keyword_topic_category_phase_summary(...)` 的输出（包含 product_category/current_phase/direction/topic_*）

    输出：
    - 每行 = 1 个 (product_category, current_phase)
    - reduce_top_topics / scale_top_topics：用一列字符串列出 Top 主题（便于 Excel 直接筛选/复制）
    """
    df = category_phase_summary.copy() if isinstance(category_phase_summary, pd.DataFrame) else pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    ktp = policy if isinstance(policy, KeywordTopicsPolicy) else KeywordTopicsPolicy()
    top_n = int(getattr(ktp, "md_top_n", 5) or 5)
    min_n = int(getattr(ktp, "md_min_n", 2) or 2)
    if top_n < 1:
        top_n = 1
    if top_n > 20:
        top_n = 20
    if min_n < 1:
        min_n = 1
    if min_n > 5:
        min_n = 5

    # 统一兜底：避免空类目/空阶段导致分组混乱
    try:
        if "product_category" in df.columns:
            df["product_category"] = df["product_category"].map(_norm_category)
        else:
            df["product_category"] = "（未分类）"
        if "current_phase" in df.columns:
            df["current_phase"] = df["current_phase"].map(_norm_phase)
        else:
            df["current_phase"] = "unknown"
    except Exception:
        return pd.DataFrame()

    # n 过滤：与 dashboard.md 展示口径一致（优先短语，避免 1-gram 太宽泛）
    try:
        if "n" in df.columns:
            df["n"] = pd.to_numeric(df.get("n", 0), errors="coerce").fillna(0).astype(int)
            df = df[df["n"] >= int(min_n)].copy()
    except Exception:
        pass
    if df.empty:
        return pd.DataFrame()

    # 数值列兜底：缺列就补 0，避免排序/汇总报错
    for c in ("topic_spend", "topic_sales", "topic_waste_spend", "topic_acos"):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "direction" not in df.columns:
        df["direction"] = ""
    if "ad_type" not in df.columns:
        df["ad_type"] = ""
    if "ngram" not in df.columns:
        df["ngram"] = ""

    def _fmt_reduce(r: pd.Series) -> str:
        try:
            ad_type = str(r.get("ad_type", "") or "").strip()
            n = int(r.get("n", 0) or 0)
            ng = str(r.get("ngram", "") or "").strip()
            waste = float(r.get("topic_waste_spend", 0.0) or 0.0)
            spend = float(r.get("topic_spend", 0.0) or 0.0)
            head = f"{ad_type} n={n} {ng}".strip()
            return f"{head} (waste={waste:.0f}, spend={spend:.0f})".strip()
        except Exception:
            return ""

    def _fmt_scale(r: pd.Series) -> str:
        try:
            ad_type = str(r.get("ad_type", "") or "").strip()
            n = int(r.get("n", 0) or 0)
            ng = str(r.get("ngram", "") or "").strip()
            sales = float(r.get("topic_sales", 0.0) or 0.0)
            spend = float(r.get("topic_spend", 0.0) or 0.0)
            acos = float(r.get("topic_acos", 0.0) or 0.0)
            head = f"{ad_type} n={n} {ng}".strip()
            return f"{head} (sales={sales:.0f}, spend={spend:.0f}, acos={acos:.2f})".strip()
        except Exception:
            return ""

    rows: List[Dict[str, object]] = []
    try:
        for (cat, ph), grp in df.groupby(["product_category", "current_phase"], dropna=False):
            g = grp.copy()

            reduce_df = g[g["direction"].astype(str) == "reduce"].copy()
            scale_df = g[g["direction"].astype(str) == "scale"].copy()

            # 统计值（用于排序/快速扫）
            reduce_waste_sum = float(reduce_df["topic_waste_spend"].sum()) if not reduce_df.empty else 0.0
            scale_sales_sum = float(scale_df["topic_sales"].sum()) if not scale_df.empty else 0.0
            reduce_topic_count = int(reduce_df[["ad_type", "n", "ngram"]].drop_duplicates().shape[0]) if not reduce_df.empty else 0
            scale_topic_count = int(scale_df[["ad_type", "n", "ngram"]].drop_duplicates().shape[0]) if not scale_df.empty else 0

            reduce_top_topics = ""
            scale_top_topics = ""
            if not reduce_df.empty:
                reduce_top = reduce_df.sort_values(["topic_waste_spend", "topic_spend"], ascending=[False, False]).head(top_n).copy()
                reduce_top_topics = " | ".join([t for t in [ _fmt_reduce(r) for _, r in reduce_top.iterrows() ] if t])
            if not scale_df.empty:
                scale_top = scale_df.sort_values(["topic_sales", "topic_spend"], ascending=[False, False]).head(top_n).copy()
                scale_top_topics = " | ".join([t for t in [ _fmt_scale(r) for _, r in scale_top.iterrows() ] if t])

            rows.append(
                {
                    "product_category": _norm_category(cat),
                    "current_phase": _norm_phase(ph),
                    "reduce_topic_count": reduce_topic_count,
                    "reduce_waste_spend_sum": round(reduce_waste_sum, 2),
                    "reduce_top_topics": reduce_top_topics,
                    "scale_topic_count": scale_topic_count,
                    "scale_sales_sum": round(scale_sales_sum, 2),
                    "scale_top_topics": scale_top_topics,
                }
            )
    except Exception:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # 排序：优先看“浪费总量”与“贡献总量”
    try:
        out["reduce_waste_spend_sum"] = pd.to_numeric(out.get("reduce_waste_spend_sum", 0.0), errors="coerce").fillna(0.0)
        out["scale_sales_sum"] = pd.to_numeric(out.get("scale_sales_sum", 0.0), errors="coerce").fillna(0.0)
        out = out.sort_values(["reduce_waste_spend_sum", "scale_sales_sum"], ascending=[False, False]).copy()
    except Exception:
        pass

    cols = [
        "product_category",
        "current_phase",
        "reduce_topic_count",
        "reduce_waste_spend_sum",
        "reduce_top_topics",
        "scale_topic_count",
        "scale_sales_sum",
        "scale_top_topics",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True) if cols else out.reset_index(drop=True)


def annotate_keyword_topic_action_hints(
    topic_hints: Optional[pd.DataFrame],
    asin_context: Optional[pd.DataFrame],
    low_inventory_threshold: int = 20,
    block_scale_when_low_inventory: bool = True,
    block_scale_when_cover_days_below: float = 7.0,
) -> pd.DataFrame:
    """
    在 keyword_topics_action_hints 上补充“产品语境”与放量阻断标记。

    设计原则：
    - 不改变主题筛选/排序逻辑（仍由 build_keyword_topic_action_hints 决定 Top 主题）；
    - 只对 direction=scale 的主题，基于库存约束做 blocked 标记，避免误导运营加码；
    - 利润方向（profit_direction）仅做标注，不强制阻断（避免误杀可放量主题）。
    """
    hints = topic_hints.copy() if isinstance(topic_hints, pd.DataFrame) else pd.DataFrame()
    if hints is None or hints.empty:
        return pd.DataFrame() if hints is None else hints

    # 需要补齐到 action_hints 的“语境/阻断”列（无语境时也保留空列，避免 schema 漂移）
    extra_cols_defaults: Dict[str, object] = {
        "context_asin_count": 0,
        "context_top_asins": "",
        "context_profit_directions": "",
        "context_min_inventory": "",
        "context_min_cover_days_7d": "",
        "blocked": 0,
        "blocked_reason": "",
    }

    ctx = asin_context.copy() if isinstance(asin_context, pd.DataFrame) else pd.DataFrame()
    if ctx is None or ctx.empty:
        # 仍然补齐列，保持 CSV 表头稳定
        for c, v in extra_cols_defaults.items():
            if c not in hints.columns:
                hints[c] = v
        return hints

    # 统一 key 类型，避免 merge 失败（尤其是 n 可能是 float）
    for c in ("ad_type", "ngram", "direction"):
        if c in hints.columns:
            hints[c] = hints[c].fillna("").astype(str).str.strip()
    if "n" in hints.columns:
        hints["n"] = pd.to_numeric(hints.get("n", 0), errors="coerce").fillna(0).astype(int)

    # 只对 hints 内的 topic 做 enrich（避免 ctx 太大）
    topic_keys: set[Tuple[str, int, str]] = set()
    try:
        for _, r in hints.iterrows():
            ad_type = str(r.get("ad_type", "") or "").strip()
            n = int(r.get("n", 0) or 0)
            ng = str(r.get("ngram", "") or "").strip()
            if ad_type and n > 0 and ng:
                topic_keys.add((ad_type, n, ng))
    except Exception:
        topic_keys = set()
    if not topic_keys:
        return hints

    ctx2 = ctx.copy()
    for c in ("ad_type", "ngram", "direction"):
        if c in ctx2.columns:
            ctx2[c] = ctx2[c].fillna("").astype(str).str.strip()
    if "n" in ctx2.columns:
        ctx2["n"] = pd.to_numeric(ctx2.get("n", 0), errors="coerce").fillna(0).astype(int)
    if "asin" in ctx2.columns:
        ctx2["asin"] = ctx2["asin"].fillna("").astype(str).str.upper().str.strip()

    # 过滤到目标 topics
    try:
        ctx2["_key"] = list(
            zip(
                ctx2.get("ad_type", "").astype(str),
                ctx2.get("n", 0).astype(int),
                ctx2.get("ngram", "").astype(str),
            )
        )
        ctx2 = ctx2[ctx2["_key"].map(lambda x: x in topic_keys)].copy()
        ctx2 = ctx2.drop(columns=["_key"], errors="ignore")
    except Exception:
        pass
    if ctx2.empty:
        for c, v in extra_cols_defaults.items():
            if c not in hints.columns:
                hints[c] = v
        return hints

    # 数值列
    for c in ("inventory", "inventory_cover_days_7d", "topic_sales", "topic_waste_spend", "topic_spend"):
        if c not in ctx2.columns:
            ctx2[c] = 0.0
        ctx2[c] = pd.to_numeric(ctx2.get(c, 0.0), errors="coerce")

    if "profit_direction" not in ctx2.columns:
        ctx2["profit_direction"] = ""
    ctx2["profit_direction"] = ctx2["profit_direction"].fillna("").astype(str).str.strip().str.lower()

    # ===== 1) 每个 topic 聚合出“产品语境摘要” =====
    def _fmt_asin_cell(r: pd.Series) -> str:
        try:
            asin = str(r.get("asin", "") or "").strip().upper()
            if not asin:
                return ""
            cover = r.get("inventory_cover_days_7d", None)
            profit = str(r.get("profit_direction", "") or "").strip()
            ph = str(r.get("current_phase", "") or "").strip()
            parts: List[str] = []
            try:
                if cover is not None and str(cover) != "nan":
                    parts.append(f"cover7d={float(cover):.0f}")
            except Exception:
                pass
            if profit:
                parts.append(f"profit={profit}")
            if ph:
                parts.append(f"ph={ph}")
            return f"{asin}({','.join(parts)})" if parts else asin
        except Exception:
            return ""

    ctx_rows: List[Dict[str, object]] = []
    try:
        for (ad_type, n, ng), grp in ctx2.groupby(["ad_type", "n", "ngram"], dropna=False):
            g = grp.copy()
            asin_count = int(g["asin"].nunique()) if "asin" in g.columns else int(len(g))

            # min 值（用于阻断）
            try:
                min_inv = float(pd.to_numeric(g.get("inventory", 0.0), errors="coerce").min())
            except Exception:
                min_inv = float("nan")
            try:
                min_cover = float(pd.to_numeric(g.get("inventory_cover_days_7d", 0.0), errors="coerce").min())
            except Exception:
                min_cover = float("nan")

            # profit_directions（用于标注）
            try:
                dirs = sorted({str(x).strip().lower() for x in g.get("profit_direction", "").tolist() if str(x).strip()})
                profit_dirs = " | ".join(dirs)
            except Exception:
                profit_dirs = ""

            # Top ASIN 样例（按主题方向：reduce 看 waste_spend；scale 看 sales）
            top_asins = ""
            try:
                direction = str(g.get("direction", "").iloc[0] if "direction" in g.columns and len(g) else "").strip()
                if direction == "reduce":
                    g2 = g.sort_values(["topic_waste_spend", "topic_spend"], ascending=[False, False]).copy()
                elif direction == "scale":
                    g2 = g.sort_values(["topic_sales", "topic_spend"], ascending=[False, False]).copy()
                else:
                    g2 = g.sort_values(["topic_spend"], ascending=[False]).copy()
                top_asins = " | ".join([t for t in [_fmt_asin_cell(r) for _, r in g2.head(3).iterrows()] if t])
            except Exception:
                top_asins = ""

            ctx_rows.append(
                {
                    "ad_type": str(ad_type),
                    "n": int(n),
                    "ngram": str(ng),
                    "context_asin_count": asin_count,
                    "context_top_asins": top_asins,
                    "context_profit_directions": profit_dirs,
                    "context_min_inventory": min_inv,
                    "context_min_cover_days_7d": min_cover,
                }
            )
    except Exception:
        return hints

    topic_ctx = pd.DataFrame(ctx_rows)
    if topic_ctx is None or topic_ctx.empty:
        for c, v in extra_cols_defaults.items():
            if c not in hints.columns:
                hints[c] = v
        return hints

    out = hints.merge(topic_ctx, on=["ad_type", "n", "ngram"], how="left")

    # ===== 2) 放量阻断：只对 scale 方向生效 =====
    def _blocked_row(r: pd.Series) -> Tuple[int, str]:
        try:
            direction = str(r.get("direction", "") or "").strip()
            if direction != "scale":
                return (0, "")

            reasons: List[str] = []

            # 低库存阻断
            if bool(block_scale_when_low_inventory):
                try:
                    inv = float(r.get("context_min_inventory", float("nan")))
                    if str(inv) != "nan" and inv <= float(low_inventory_threshold):
                        reasons.append(f"低库存(min_inventory≤{int(low_inventory_threshold)})")
                except Exception:
                    pass

            # 覆盖天数阻断
            try:
                cover_th = float(block_scale_when_cover_days_below or 0.0)
                if cover_th > 0:
                    cover = float(r.get("context_min_cover_days_7d", float("nan")))
                    if str(cover) != "nan" and cover < cover_th:
                        reasons.append(f"覆盖不足(min_cover7d<{cover_th:.0f}d)")
            except Exception:
                pass

            if not reasons:
                return (0, "")
            return (1, "；".join(reasons))
        except Exception:
            return (0, "")

    try:
        blocked_vals: List[int] = []
        blocked_reasons: List[str] = []
        for _, rr in out.iterrows():
            b, reason = _blocked_row(rr)
            blocked_vals.append(int(b))
            blocked_reasons.append(str(reason))
        out["blocked"] = blocked_vals
        out["blocked_reason"] = blocked_reasons
    except Exception:
        out["blocked"] = 0
        out["blocked_reason"] = ""

    # blocked=1 时，把 next_step 里也提示一下（仍保持单元格可读）
    try:
        if "next_step" in out.columns:
            out["next_step"] = out.apply(
                lambda rr: (
                    str(rr.get("next_step", "") or "")
                    + (f"（放量阻断：{rr.get('blocked_reason', '')}）" if str(rr.get("direction", "")) == "scale" and int(rr.get("blocked", 0) or 0) == 1 else "")
                ),
                axis=1,
            )
    except Exception:
        pass

    return out
