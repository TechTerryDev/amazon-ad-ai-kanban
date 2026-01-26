# -*- coding: utf-8 -*-
"""
把赛狐导出的报表 -> 指标汇总(JSON) + 动作候选(CSV)。

设计目标（符合你现在阶段）：
- 代码只做确定性计算与结构化输出
- AI（Claude skill）只做“解释与写报告”，不参与算数
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ads.actions import (
    ActionCandidate,
    generate_campaign_budget_suggestions,
    generate_campaign_budget_actions_from_map,
    generate_placement_actions,
    generate_product_side_actions,
    generate_search_term_actions,
    generate_targeting_actions,
)
from core.config import StageConfig, get_stage_config
from ingest.loader import LoadedReport, load_ad_reports, load_product_analysis, load_product_listing
from core.metrics import summarize, to_summary_dict
from analysis.diagnostics import (
    diagnose_asin_root_causes,
    diagnose_campaign_budget_map_from_asin,
    diagnose_campaign_trends,
    diagnose_shop_scorecard,
    build_budget_transfer_plan,
    build_unlock_scale_plan,
    build_unlock_tasks,
    infer_asin_stage_by_profit,
    summarize_profit_health,
)
from reporting.reporting import generate_shop_report
from core.schema import CAN
from core.utils import json_dumps, parse_date, to_float, is_paused_status
from analysis.temporal import build_temporal_insights
from lifecycle.lifecycle import build_lifecycle_for_shop, build_lifecycle_windows_for_shop, LifecycleConfig
from ads.ad_linkage import (
    allocate_detail_to_asin,
    build_ad_product_daily,
    build_asin_campaign_map,
    build_weight_join_specs,
    top_n_entities_by_asin,
)
from analysis.ai_bundle import build_ai_input_bundle
from analysis.data_quality import build_data_quality_report, extract_data_quality_summary_lines, write_data_quality_files
from core.policy import (
    OpsPolicy,
    OpsProfile,
    load_ops_policy,
    load_ops_policy_with_overrides,
    load_ops_profile,
    ops_policy_effective_to_dict,
    ops_profile_to_overrides,
    validate_ops_policy_path,
)
from dashboard.outputs import write_dashboard_outputs, write_report_html_from_md
from dashboard.execution_review import load_execution_log, write_action_review, write_execution_log_template


def _concat_reports(reports: List[LoadedReport], report_type: str) -> pd.DataFrame:
    frames = [r.df for r in reports if r.report_type == report_type and not r.df.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # 去掉“日期为空”的异常行，避免后续 date_range 不准确
    if CAN.date in df.columns:
        df = df[df[CAN.date].notna()]
    # 你可能会下载“上个月”+“最近30天”造成重叠，先做一次去重（完全一致的行会被移除）
    try:
        df = df.drop_duplicates()
    except Exception:
        pass
    return df


def _data_gaps(reports: List[LoadedReport]) -> Dict[str, Dict[str, bool]]:
    """
    给 AI 的“缺口清单”：哪些报表类型存在/缺失，避免它胡编。
    """
    have = {r.report_type for r in reports}
    expected = [
        "search_term",
        "targeting",
        "placement",
        "campaign",
        "ad_group",
        "advertised_product",
        "purchased_product",
        "matched_target",
    ]
    return {"reports": {k: (k in have) for k in expected}}


def _shop_list(reports: List[LoadedReport]) -> List[str]:
    shops: set[str] = set()
    for r in reports:
        if CAN.shop in r.df.columns:
            shops.update({s for s in r.df[CAN.shop].dropna().astype(str).unique().tolist() if s.strip()})
    return sorted(shops)


def _infer_max_date(dfs: List[pd.DataFrame]) -> Optional[dt.date]:
    """
    用于 --days 过滤：以报表内的最大日期为准，而不是用“今天”，避免历史月报被误剪。
    """
    dmax: Optional[dt.date] = None
    for df in dfs:
        if df is None or df.empty or CAN.date not in df.columns:
            continue
        try:
            v = df[CAN.date].max()
        except Exception:
            v = None
        if v is None:
            continue
        if dmax is None or v > dmax:
            dmax = v
    return dmax


def _filter_by_date(df: pd.DataFrame, date_start: Optional[dt.date], date_end: Optional[dt.date]) -> pd.DataFrame:
    if df is None or df.empty or CAN.date not in df.columns:
        return df
    if date_start is None and date_end is None:
        return df
    out = df.copy()
    out = out[out[CAN.date].notna()]
    if date_start is not None:
        out = out[out[CAN.date] >= date_start]
    if date_end is not None:
        out = out[out[CAN.date] <= date_end]
    return out


def _resolve_ad_root(reports_root: Path) -> Path:
    """
    兼容数据目录：
    - reports/ad（旧结构）
    - data/input/广告数据（新结构）
    """
    candidates = [
        reports_root / "ad",
        reports_root / "广告数据",
        reports_root / "广告",
    ]
    for p in candidates:
        if p.exists():
            return p
    return reports_root / "ad"


def _resolve_product_analysis_dir(reports_root: Path) -> Path:
    """
    兼容数据目录：
    - reports/产品分析（旧结构）
    - data/input/产品分析（新结构）
    """
    candidates = [
        reports_root / "产品分析",
        reports_root / "product_analysis",
    ]
    for p in candidates:
        if p.exists():
            return p
    return reports_root / "产品分析"


def _resolve_product_listing_path(reports_root: Path) -> Optional[Path]:
    """
    兼容数据目录：
    - reports/productListing.xlsx（旧结构）
    - data/input/产品映射/*.xlsx（新结构，优先匹配“商品列表”）
    """
    direct = reports_root / "productListing.xlsx"
    if direct.exists():
        return direct

    mapping_dir = reports_root / "产品映射"
    if mapping_dir.exists():
        # 优先匹配“商品列表”关键字
        preferred = []
        others = []
        for p in mapping_dir.rglob("*.xlsx"):
            name = p.name
            if "商品列表" in name or "productListing" in name or "productlisting" in name:
                preferred.append(p)
            else:
                others.append(p)
        if preferred:
            return sorted(preferred)[0]
        if others:
            return sorted(others)[0]
    return None


def _filter_paused_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    仅用于“行动清单”：如果报表自带状态列，则过滤暂停/停用/终止的行。
    """
    if df is None or df.empty or CAN.status not in df.columns:
        return df
    try:
        out = df.copy()
        out = out[~out[CAN.status].apply(is_paused_status)]
        return out
    except Exception:
        return df


def _paused_key_set(df: pd.DataFrame, key_cols: List[str]) -> set[tuple]:
    """
    生成“暂停对象”的键集合（用于过滤行动清单）。
    """
    if df is None or df.empty or CAN.status not in df.columns:
        return set()
    for col in key_cols:
        if col not in df.columns:
            return set()
    try:
        sub = df[df[CAN.status].apply(is_paused_status)]
        if sub.empty:
            return set()
        keys = sub[key_cols].fillna("").astype(str)
        return set(tuple(r) for r in keys.itertuples(index=False, name=None))
    except Exception:
        return set()


def _filter_actions_by_paused(
    actions: List[ActionCandidate],
    paused_campaigns: set[tuple],
    paused_ad_groups: set[tuple],
    paused_targetings: set[tuple],
) -> List[ActionCandidate]:
    """
    过滤暂停广告相关动作（仅影响行动清单，不影响其他分析汇总）。
    """
    if not actions:
        return actions
    out: List[ActionCandidate] = []
    for a in actions:
        # campaign 级（预算/活动/广告位）
        if a.campaign:
            key_camp = (a.ad_type, a.campaign)
            if key_camp in paused_campaigns:
                continue
        # ad_group 级（搜索词/投放等）
        if a.campaign and a.ad_group:
            key_ag = (a.ad_type, a.campaign, a.ad_group)
            if key_ag in paused_ad_groups:
                continue
        # targeting 级（投放报告带状态时可过滤）
        if a.level == "targeting" and a.campaign and a.ad_group and a.object_name:
            key_tgt = (a.ad_type, a.campaign, a.ad_group, a.object_name, a.match_type)
            if key_tgt in paused_targetings:
                continue
        out.append(a)
    return out


def _write_action_candidates(path: Path, actions: List[ActionCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in actions:
        rows.append(
            {
                "shop": a.shop,
                "ad_type": a.ad_type,
                "level": a.level,
                "action_type": a.action_type,
                "action_value": a.action_value,
                "priority": a.priority,
                "object_name": a.object_name,
                "campaign": a.campaign,
                "ad_group": a.ad_group,
                "match_type": a.match_type,
                "date_start": a.date_start,
                "date_end": a.date_end,
                "reason": a.reason,
                "evidence_json": a.evidence_json,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_budget_transfers(path: Path, plan: Dict[str, object]) -> None:
    """
    预算净迁移表（L0）：方便运营直接照着挪预算。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # 即使无迁移，也输出表头，方便运营查看“为什么为空”
    transfers = plan.get("transfers") if isinstance(plan, dict) else None
    rows = transfers if isinstance(transfers, list) else []
    cols = [
        "from_ad_type",
        "from_campaign",
        "from_asin_hint",
        "to_ad_type",
        "to_campaign",
        "to_asin_hint",
        "amount_usd_estimated",
        "note",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_budget_table(path: Path, rows: object, columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows if isinstance(rows, list) else [], columns=columns)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_start_here(
    shop_dir: Path,
    shop: str,
    output_profile: str,
    render_dashboard_md: bool,
    render_full_report: bool,
    policy: Optional[OpsPolicy] = None,
) -> None:
    """
    验收入口：把“要看哪个文件”明确出来，减少输出文件带来的困扰。
    """
    try:
        p = shop_dir / "START_HERE.md"
        prof = (output_profile or "minimal").strip().lower()
        def L(rel_path: str) -> str:
            # 用代码样式展示路径，同时支持点击跳转（大多数 Markdown 预览器可用）
            return f"[`{rel_path}`]({rel_path})"

        lines = [f"# {shop} 输出验收入口\n", ""]
        lines += [
            "快速导航：",
            "- [运营入口](#ops)",
            "- [AI/分析入口](#ai)",
            "",
            '<a id="ops"></a>',
            "## A) 给运营看（聚焦抓重点）",
            "",
        ]
        if render_dashboard_md:
            lines += [
                "建议阅读顺序：",
                f"- 先看 {L('reports/dashboard.html')}（更好读：浏览器打开；结论 + Top 动作）",
                f"- 口径提示：如遇“结论与直觉不一致”，先扫一眼 {L('ai/data_quality.html')} 的摘要（缺失字段/覆盖率会影响解读）",
                f"- 配置提示：如调参后结论异常，先看 {L('../ops_policy_warnings.html')}（ops_policy 配置校验/默认值提示）",
                f"- 再看 {L('dashboard/action_board.csv')}（可筛选/分派）",
                f"- 最后按需下钻到 ASIN/类目/生命周期明细",
                "",
                "口径与提示（集中说明）：",
                "- 未标注的累计指标=主窗口；标注 compare/Δ 的为近N天 vs 前N天（日期见表内 recent/prev）",
                "- 表头含(7d/14d/30d)=近窗；含Δ=对比窗口；含roll=滚动窗口",
                "- Drilldown：类目/ASIN/生命周期字段可点击跳转（Dashboard/Drilldown 均支持）",
                "- Δ 指标默认 compare_7d（近7天 vs 前7天），如需换口径请看对应 CSV/字段",
                "- HTML 支持目录/表头排序（离线可用），如需追溯口径请以 CSV/JSON 为准",
            ]
            try:
                ignore_last = int(getattr(policy, "dashboard_compare_ignore_last_days", 0) or 0) if policy is not None else 0
                if ignore_last > 0:
                    lines.append(f"- compare 忽略最近 {ignore_last} 天（规避归因滞后噪声）")
            except Exception:
                pass
            try:
                low_thr = 0.35
                if isinstance(policy, OpsPolicy):
                    asp = getattr(policy, "dashboard_action_scoring", None)
                    if asp is not None:
                        low_thr = float(getattr(asp, "low_hint_confidence_threshold", low_thr) or low_thr)
                low_thr = max(0.0, min(1.0, float(low_thr)))
                lines.append(
                    f"- Action Board：`asin_hint` 为弱关联定位；当 `asin_hint_confidence<{low_thr:.2f}` 时建议先人工确认（可在 action_board.csv 里筛选/对照候选 ASIN）"
                )
            except Exception:
                lines.append("- Action Board：`asin_hint` 为弱关联定位；低置信度时建议先人工确认")
            try:
                cover_days_thr = float(getattr(policy, "block_scale_when_cover_days_below", 7.0) or 7.0) if isinstance(policy, OpsPolicy) else 7.0
                lines.append(f"- 库存告急仍投放：`inventory_cover_days_7d ≤ {int(cover_days_thr)}d` 且 `ad_spend_roll ≥ 10`")
            except Exception:
                pass
            lines.append("- 库存调速（Sigmoid）：基于 `inventory_cover_days_7d` 计算调速系数，仅建议不自动执行")
            lines.append("- 利润护栏（Break-even）：安全ACOS = 毛利率 - 目标净利率，超线仅提示")
            lines.append("- 关键词主题：n-gram 会重复计数 spend；仅用于线索与聚焦，不做精确归因")
            try:
                dq_path = shop_dir / "ai" / "data_quality.md"
                if dq_path.exists():
                    hints: List[str] = []
                    with dq_path.open("r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            t = line.strip()
                            if not t:
                                continue
                            if t.startswith("- "):
                                hints.append(t[2:].strip())
                            if len(hints) >= 2:
                                break
                    if hints:
                        lines.append("- 数据质量提示（自动摘取）:")
                        for h in hints:
                            if h:
                                lines.append(f"  - {h}")
            except Exception:
                pass
            lines.append("")
        else:
            lines += [
                "建议阅读顺序：",
                "- （当前未生成 reports/*.md/html：你可能使用了 `--no-report`）",
                f"- 先看 {L('ai/data_quality.html')}（缺失字段/覆盖率会影响解读）",
                f"- 配置提示：如调参后结论异常，先看 {L('../ops_policy_warnings.html')}（ops_policy 配置校验/默认值提示）",
                f"- 再看 {L('dashboard/action_board.csv')}（可筛选/分派）",
                f"- 再看 {L('dashboard/shop_scorecard.json')}（店铺 KPI/诊断：结构化）",
                "",
            ]
        if render_dashboard_md:
            lines.append("文件导航（Dashboard 常用）：")
            lines.append(f"- {L('reports/dashboard.html')}：聚焦版（HTML，更好读）")
            lines.append(f"- {L('reports/asin_drilldown.html')}：ASIN Drilldown（HTML）")
            lines.append(f"- {L('reports/category_drilldown.html')}：Category Drilldown（HTML）")
            lines.append(f"- {L('reports/phase_drilldown.html')}：Phase Drilldown（HTML）")
            lines.append(f"- {L('reports/lifecycle_overview.html')}：Lifecycle Overview（HTML：类目→ASIN 生命周期时间轴）")
            lines.append(f"- {L('reports/keyword_topics.html')}：Keyword Topics Drilldown（HTML）")
        lines.append(f"- {L('dashboard/category_summary.csv')}：类目总览（先看商品分类，再下钻产品）")
        lines.append(f"- {L('dashboard/category_cockpit.csv')}：类目总览（focus + drivers + 动作量汇总）")
        lines.append(f"- {L('dashboard/category_asin_compare.csv')}：类目→产品对比（同类 Top ASIN：速度/覆盖/利润承受度/风险一张表，可筛选）")
        lines.append(f"- {L('dashboard/phase_cockpit.csv')}：生命周期总览（按 current_phase 汇总 focus/变化/动作量）")
        lines.append(f"- {L('dashboard/asin_focus.csv')}：ASIN Focus List（按 focus_score 排序，可筛选/分派）")
        lines.append(f"- {L('dashboard/asin_cockpit.csv')}：ASIN 总览（focus + drivers + 动作量汇总，一行一个 ASIN）")
        lines.append(f"- {L('dashboard/compare_summary.csv')}：店铺环比摘要（7/14/30，销售/利润/花费/自然/转化）")
        lines.append(f"- {L('dashboard/lifecycle_timeline.csv')}：生命周期时间轴摘要（每 ASIN 一行，供复盘/筛选）")
        lines.append(f"- {L('dashboard/task_summary.csv')}：任务汇总（本周行动/Shop Alerts/Action Board 汇聚，可筛选复盘）")
        lines.append(f"- {L('dashboard/profit_reduce_watchlist.csv')}：利润控量 Watchlist（profit_direction=reduce 且仍在烧钱：优先止血/收口）")
        lines.append(f"- {L('dashboard/oos_with_ad_spend_watchlist.csv')}：断货仍烧钱 Watchlist（oos_with_ad_spend_days>0 且仍在投放：优先止损）")
        lines.append(f"- {L('dashboard/spend_up_no_sales_watchlist.csv')}：加花费但销量不增 Watchlist（delta_spend>0 且 delta_sales<=0：优先排查）")
        lines.append(f"- {L('dashboard/phase_down_recent_watchlist.csv')}：阶段走弱 Watchlist（近14天阶段走弱 down 且仍在花费：优先排查根因）")
        lines.append(f"- {L('dashboard/scale_opportunity_watchlist.csv')}：机会 Watchlist（可放量窗口/低花费高潜；用于预算迁移/加码）")
        lines.append(f"- {L('dashboard/opportunity_action_board.csv')}：机会→可执行动作（只保留 BID_UP/BUDGET_UP 且未阻断）")
        lines.append(f"- {L('dashboard/inventory_sigmoid_watchlist.csv')}：库存调速建议（Sigmoid，仅建议，不影响排序）")
        lines.append(f"- {L('dashboard/profit_guard_watchlist.csv')}：利润护栏 Watchlist（Break-even：安全ACOS/CPC 超线提示）")
        lines.append(f"- {L('dashboard/budget_transfer_plan.csv')}：预算迁移净表（估算金额；执行时以实际预算/花费节奏校准）")
        lines.append(f"- {L('dashboard/unlock_scale_tasks.csv')}：放量解锁任务表（可分工：广告/供应链/运营/美工）")
        lines.append(f"- {L('dashboard/unlock_scale_tasks_full.csv')}：放量解锁任务表全量（含更多任务/优先级，便于追溯）")
        lines.append(f"- {L('dashboard/drivers_top_asins.csv')}：变化来源（近7天 vs 前7天 Top ASIN）")
        lines.append(f"- {L('dashboard/keyword_topics.csv')}：关键词主题（n-gram，压缩 search_term 长表，快速看“在烧什么/在带量什么”）")
        lines.append(f"- {L('dashboard/keyword_topics_segment_top.csv')}：关键词主题 Segment Top（先按 类目×阶段 看 Top 浪费/贡献主题，再下钻）")
        lines.append(f"- {L('dashboard/keyword_topics_action_hints.csv')}：关键词主题建议清单（Top 浪费→否词/降价；Top 贡献→加精确/提价；含 top_campaigns/top_ad_groups；scale 方向会标注/阻断库存风险）")
        lines.append(f"- {L('dashboard/keyword_topics_asin_context.csv')}：关键词主题→产品语境（只用高置信 term→asin；可看到类目/ASIN/生命周期/库存覆盖）")
        lines.append(f"- {L('dashboard/keyword_topics_category_phase_summary.csv')}：关键词主题→类目/生命周期汇总（先按类目/阶段看主题，再下钻到 ASIN）")
        lines.append(f"- {L('dashboard/campaign_action_view.csv')}：Campaign 行动聚合（从 Action Board 归并，方便先按 campaign 排查）")
        lines.append(f"- {L('dashboard/action_board.csv')}：动作看板（去重后的运营视图；P0/P1 优先；可按类目/生命周期筛选）")
        lines.append(f"- {L('dashboard/action_board_full.csv')}：动作看板全量（含重复，便于追溯）")
        lines.append(f"- {L('dashboard/shop_scorecard.json')}：店铺 KPI/诊断（结构化）")
        lines.append(f"- {L('ops/execution_log_template.xlsx')}：L0+ 执行回填模板（手工执行后回填，用于下次复盘）")
        if (shop_dir / "ops" / "action_review.csv").exists():
            lines.append(f"- {L('ops/action_review.csv')}：L0+ 动作复盘（已生成：基于历史 execution_log 复盘 7/14 天效果）")
        else:
            lines.append("- （可选）`ops/action_review.csv`：L0+ 动作复盘（需要你提供已回填的 execution_log，并运行时指定 `--ops-log-root <目录>`）")
        if render_full_report:
            lines.append(f"- {L('ops/actions.csv')}：运营动作清单（按 商品分类→ASIN；用于筛选/分配给运营）")
            lines.append(f"- {L('ops/keyword_playbook.xlsx')}：运营主手册（动作 + ASIN 7/14/30 滚动环比拼表）")
            lines.append(f"- {L('ops/campaign_ops.csv')}：Campaign 可执行清单（含 7/14/30 窗口信号 + 关联 ASIN/分类）")
        else:
            lines.append("- （提示）本次未生成 `ops/actions.csv/keyword_playbook.xlsx/campaign_ops.csv`：你可能使用了 `--no-full-report` 或 `--no-report`")
        lines.append("")
        lines += [
            '<a id="ai"></a>',
            "## B) 给 AI/分析看（全量指标与证据）",
            "",
            f"- {L('ai/ai_input_bundle.json')}：AI 输入包（结构化、优先喂这个）",
            f"- {L('ai/data_quality.html')}：数据质量摘要（HTML，推荐先看）",
            f"- {L('ai/data_quality.json')}：维度覆盖与数据质量盘点（JSON）",
        ]
        # AI 自动写建议报告（可选）：仅在文件实际生成后才展示可点击链接，避免 START_HERE 出现“死链”。
        try:
            ai_dir = shop_dir / "ai"
            if (ai_dir / "ai_suggestions.md").exists():
                if (ai_dir / "ai_suggestions.html").exists():
                    lines.append(f"- {L('ai/ai_suggestions.html')}：AI 建议报告（HTML，自动生成；事实口径仍以 CSV/JSON 为准）")
                else:
                    lines.append("- AI 建议报告：已生成（HTML 未生成）")
            else:
                lines.append("- AI 建议报告：可选（运行时加 `--ai-report`，并配置 `LLM_*` 环境变量）")
        except Exception:
            lines.append("- AI 建议报告：可选（运行时加 `--ai-report`，并配置 `LLM_*` 环境变量）")

        if render_full_report:
            lines.append("- AI 深挖版：已生成（Markdown 备份见下方折叠）")
        else:
            lines.append("- AI 深挖版：未生成（你可能使用了 `--no-full-report` 或 `--no-report`）")

        # Markdown 备份：折叠放置（给 AI/追溯用，运营默认不看）
        if render_dashboard_md:
            lines += [
                "",
                "<details>",
                "<summary>AI/备份（Markdown 版本，默认折叠）</summary>",
                "",
                f"- {L('reports/dashboard.md')}：聚焦版（Markdown 备份）",
                f"- {L('reports/asin_drilldown.md')}：ASIN Drilldown（Markdown 备份）",
                f"- {L('reports/category_drilldown.md')}：Category Drilldown（Markdown 备份）",
                f"- {L('reports/phase_drilldown.md')}：Phase Drilldown（Markdown 备份）",
                f"- {L('reports/lifecycle_overview.md')}：Lifecycle Overview（Markdown 备份）",
                f"- {L('reports/keyword_topics.md')}：Keyword Topics（Markdown 备份）",
                f"- {L('ai/data_quality.md')}：数据质量（Markdown 备份）",
            ]
            if (shop_dir / "ai" / "ai_suggestions.md").exists():
                lines.append(f"- {L('ai/ai_suggestions.md')}：AI 建议报告（Markdown 备份）")
            if (shop_dir / "ai" / "ai_suggestions_prompt.md").exists():
                lines.append(f"- {L('ai/ai_suggestions_prompt.md')}：AI 提示词留档（Markdown）")
            if render_full_report:
                lines.append(f"- {L('ai/report.md')}：全量深挖版（Markdown）")
            lines += ["", "</details>"]

        if prof == "full":
            lines += [
                "",
                "full 档位还会输出这些（便于 AI 深挖/排查）：",
                f"- {L('ai/metrics_bundle.json')}：更全的结构化计算结果（含 diagnostics）",
                f"- {L('ai/data_gaps.json')}：报表缺口清单（避免误解）",
                f"- {L('action_candidates.csv')}：动作候选（P0/P1）",
                f"- {L('lifecycle_current_board.csv')} / {L('lifecycle_segments.csv')} / {L('lifecycle_windows.csv')}：生命周期明细",
                f"- {L('asin_top_campaigns.csv')} / {L('asin_top_search_terms.csv')} / {L('asin_top_targetings.csv')} / {L('asin_top_placements.csv')}：广告结构按 ASIN 汇总",
                f"- {L('budget_transfers.csv')} / {L('budget_adds.csv')} / {L('budget_cuts.csv')} / {L('budget_savings.csv')}：预算迁移与控量/放量清单",
                f"- {L('unlock_scale.csv')} / {L('unlock_tasks.csv')}：放量池与任务清单",
                f"- {L('temporal_campaign_windows.csv')} / {L('temporal_targeting_windows.csv')}：多窗口增量效率与信号",
                f"- {L('asin_campaign_map.csv')}：ASIN↔活动/广告组映射全表",
            ]
        lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8")
        # 同目录生成 HTML（更好读；不改变链接/口径）
        try:
            write_report_html_from_md(md_path=p, out_path=shop_dir / "START_HERE.html")
        except Exception:
            pass
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, object]:
    try:
        if path is None or not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_ops_policy_warnings(
    out_dir: Path,
    policy_path: Path,
    profile_path: Optional[Path],
    profile: Optional[OpsProfile],
    profile_warnings: List[str],
) -> None:
    """
    把 ops_policy.json 的“结构校验/默认值提示”输出到本次 run 目录，避免调参后不知道是否生效。

    注意：只提示，不阻断跑数。
    """
    try:
        if out_dir is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        warns = validate_ops_policy_path(policy_path)
        md_path = out_dir / "ops_policy_warnings.md"
        html_path = out_dir / "ops_policy_warnings.html"

        lines: List[str] = []
        lines.append("# ops_policy 配置校验（本次运行）")
        lines.append("")
        lines.append("本文件只做提示，不影响跑数；用于减少调参误配导致的“结论不可信/不知是否生效”。")
        lines.append("")
        lines.append(f"- 配置文件：`{policy_path}`")
        lines.append("- 本次配置快照：`config/ops_policy.json`")
        lines.append("- 本次生效配置：`config/ops_policy_effective.json`（默认值/兜底/截断后的实际值）")
        if profile_path is not None and profile_path.exists():
            lines.append(f"- 总选项（ops_profile）：`{profile_path}`")
            lines.append("- 本次总选项快照：`config/ops_profile.json`")
        lines.append("- 调参说明：仓库内 `helloagents/wiki/ops_policy.md`")
        lines.append("")

        # ops_profile 总选项（让你只改少数几个开关）
        try:
            if profile is not None and bool(getattr(profile, "enabled", False)) and profile_path is not None and profile_path.exists():
                lines.append("## 总选项（ops_profile）")
                lines.append("")
                lines.append(f"- enabled: `{bool(getattr(profile, 'enabled', False))}`")
                lines.append(f"- preset: `{str(getattr(profile, 'preset', '') or '')}`（guardrail/balanced/growth）")
                lines.append(f"- density: `{str(getattr(profile, 'density', '') or '')}`（compact/normal/deep）")
                lines.append(f"- keyword_topics: `{str(getattr(profile, 'keyword_topics', '') or '')}`（off/standard/deep）")
                if profile_warnings:
                    for w in profile_warnings:
                        lines.append(f"- ⚠️ {w}")
                lines.append("")
        except Exception:
            pass

        if warns:
            lines.append("## 提醒清单")
            lines.append("")
            for w in warns:
                lines.append(f"- {w}")
            lines.append("")
        else:
            lines.append("## 提醒清单")
            lines.append("")
            lines.append("- ✅ 未发现明显配置问题。")
            lines.append("")

        md_path.write_text("\n".join(lines), encoding="utf-8")
        try:
            write_report_html_from_md(md_path=md_path, out_path=html_path)
        except Exception:
            # 极端兜底：HTML 转换失败时仍输出一个可打开的简易 HTML，避免 START_HERE 出现死链
            try:
                body = "\n".join(lines)
                html_path.write_text(
                    "\n".join(
                        [
                            "<!doctype html>",
                            "<html lang=\"zh-CN\">",
                            "<head>",
                            "  <meta charset=\"utf-8\" />",
                            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
                            "  <title>ops_policy 配置校验</title>",
                            "  <style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial;max-width:980px;margin:24px auto;padding:0 16px;line-height:1.5}pre{background:#f6f8fa;padding:12px;border-radius:8px;white-space:pre-wrap}</style>",
                            "</head>",
                            "<body>",
                            "  <h1>ops_policy 配置校验（fallback）</h1>",
                            "  <p>HTML 渲染失败，已降级为纯文本展示（不影响跑数）。</p>",
                            f"  <pre>{html.escape(body)}</pre>",
                            "</body>",
                            "</html>",
                        ]
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
    except Exception:
        return


def _write_ops_policy_snapshot(
    out_dir: Path,
    policy_path: Path,
    policy: OpsPolicy,
    profile_path: Optional[Path],
) -> None:
    """
    写出本次运行的 ops_policy 配置快照（raw + effective），用于可追溯/可复现。
    """
    try:
        if out_dir is None or policy_path is None:
            return
        cfg_dir = out_dir / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)

        # 1) raw copy（保持与仓库内一致）
        try:
            raw_bytes = Path(policy_path).read_bytes()
            (cfg_dir / "ops_policy.json").write_bytes(raw_bytes)
        except Exception:
            # 兜底：用文本方式（UTF-8）复制
            try:
                raw = Path(policy_path).read_text(encoding="utf-8")
                (cfg_dir / "ops_policy.json").write_text(raw, encoding="utf-8")
            except Exception:
                pass

        # 1.5) ops_profile.json（可选：总选项快照）
        try:
            if profile_path is not None and profile_path.exists():
                try:
                    raw_bytes = Path(profile_path).read_bytes()
                    (cfg_dir / "ops_profile.json").write_bytes(raw_bytes)
                except Exception:
                    try:
                        raw = Path(profile_path).read_text(encoding="utf-8")
                        (cfg_dir / "ops_profile.json").write_text(raw, encoding="utf-8")
                    except Exception:
                        pass
        except Exception:
            pass

        # 2) effective（实际生效值：默认/兜底/截断后的结果）
        try:
            effective = ops_policy_effective_to_dict(policy)
            (cfg_dir / "ops_policy_effective.json").write_text(
                json.dumps(effective, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    except Exception:
        return


def _write_run_start_here(out_dir: Path, shops: List[str], render_dashboard_md: bool, ai_report: bool) -> None:
    """
    run 级入口：方便你在 output/<run>/ 下快速找到每个店铺的 dashboard 与 AI 输出。
    """
    try:
        if out_dir is None:
            return

        def L(rel_path: str) -> str:
            return f"[`{rel_path}`]({rel_path})"

        def NL(n: object, rel_path: str) -> str:
            """
            数字型单元格：既保留可排序的数字展示，又提供可点击跳转。
            """
            try:
                if n is None:
                    return ""
                s = str(n).strip()
                if not s:
                    return ""
                return f"[{s}]({rel_path})"
            except Exception:
                return ""

        lines: List[str] = ["# 本次运行输出入口\n", ""]
        lines += [
            "快速导航：",
            "- [本次重点](#focus)",
            "- [店铺列表](#shops)",
            "- [配置校验](#policy)",
            "- [轻量汇总](#summary)",
            "- [Owner 汇总](OWNER_OVERVIEW.html)",
            "",
            '<a id="shops"></a>',
            "## 店铺列表（快速入口）",
            "",
        ]

        if shops:
            for s in shops:
                s2 = str(s or "").strip()
                if not s2:
                    continue
                parts: List[str] = []
                if render_dashboard_md and (out_dir / s2 / "reports" / "dashboard.html").exists():
                    parts.append(L(f"{s2}/reports/dashboard.html"))
                # HTML-first：run 级入口默认指向更好读的 HTML（避免在 HTML 中展示 md 路径）
                if (out_dir / s2 / "START_HERE.html").exists():
                    parts.append(L(f"{s2}/START_HERE.html"))
                elif (out_dir / s2 / "START_HERE.md").exists():
                    parts.append(L(f"{s2}/START_HERE.md"))
                # AI：如果已生成（ai-report 或 ai-prompt-only），在 run 入口也直接给出链接（方便批量验收/复制提示词）
                try:
                    ai_dir = out_dir / s2 / "ai"
                    if (ai_dir / "ai_suggestions.html").exists():
                        parts.append(L(f"{s2}/ai/ai_suggestions.html"))
                except Exception:
                    pass
                joined = " | ".join(parts)
                lines.append(f"- {s2}: {joined}")
            lines.append("")
        else:
            lines.append("- （无店铺输出）")
            lines.append("")

        # 本次重点：用“动作数/问题池”做一个非常粗的排序，帮助先抓重点（不改变任何口径）
        try:
            def _asin_anchor_id(asin: str) -> str:
                try:
                    s = str(asin or "").strip().lower()
                    s2 = "".join([c for c in s if c.isalnum()])
                    return f"asin-{s2}" if s2 else ""
                except Exception:
                    return ""

            def _read_csv_first_row(path: Path) -> Dict[str, str]:
                try:
                    if path is None or (not path.exists()):
                        return {}
                    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                        reader = csv.DictReader(f)
                        row = next(reader, None)
                        return {str(k): str(v or "") for k, v in (row or {}).items()} if isinstance(row, dict) else {}
                except Exception:
                    return {}

            def _build_asin_link(shop: str, asin: str, fallback_rel: str) -> str:
                """
                优先链接到 asin_drilldown（HTML）；如果未生成报告，则回退到 watchlist CSV。
                """
                try:
                    shop2 = str(shop or "").strip()
                    a = str(asin or "").strip().upper()
                    aid = _asin_anchor_id(a)
                    html_path = out_dir / shop2 / "reports" / "asin_drilldown.html"
                    if html_path.exists() and aid:
                        return f"{shop2}/reports/asin_drilldown.html#{aid}"
                    return fallback_rel
                except Exception:
                    return fallback_rel

            def _cat_anchor_id(category: str) -> str:
                """
                生成稳定的 Category 锚点 id（与 dashboard.outputs._cat_anchor_id 保持同算法）。
                """
                try:
                    s = str(category or "").strip()
                    if not s or s.lower() == "nan":
                        s = "（未分类）"
                    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
                    return f"cat-{h}"
                except Exception:
                    return "cat-unknown"

            def _build_category_link(shop: str, category: str, fallback_rel: str) -> str:
                """
                优先链接到 category_drilldown（HTML）；如果未生成报告，则回退到 cockpit CSV。
                """
                try:
                    shop2 = str(shop or "").strip()
                    c = str(category or "").strip()
                    if not c or c.lower() == "nan":
                        c = "（未分类）"
                    cid = _cat_anchor_id(c)

                    html_path = out_dir / shop2 / "reports" / "category_drilldown.html"
                    if html_path.exists() and cid:
                        return f"{shop2}/reports/category_drilldown.html#{cid}"

                    md_path = out_dir / shop2 / "reports" / "category_drilldown.md"
                    if md_path.exists() and cid:
                        return f"{shop2}/reports/category_drilldown.md#{cid}"

                    return fallback_rel
                except Exception:
                    return fallback_rel

            def _read_csv_head(path: Path, n: int = 3) -> List[Dict[str, str]]:
                try:
                    if path is None or (not path.exists()) or n <= 0:
                        return []
                    out_rows: List[Dict[str, str]] = []
                    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                        reader = csv.DictReader(f)
                        for _ in range(int(n)):
                            row = next(reader, None)
                            if not isinstance(row, dict):
                                break
                            out_rows.append({str(k): str(v or "") for k, v in row.items()})
                    return out_rows
                except Exception:
                    return []

            def _md_cell(v: object) -> str:
                """
                Markdown 表格单元格安全化：避免 `|` 破坏表格结构。
                """
                try:
                    s = str(v or "")
                    s = s.replace("\r", " ").replace("\n", " ").replace("|", "｜").strip()
                    return s
                except Exception:
                    return str(v or "").strip()

            def _short(v: object, max_len: int = 60) -> str:
                """
                入口摘要避免过长：超出长度时截断（不影响原始 CSV/报告内容）。
                """
                try:
                    s = str(v or "")
                    s = s.replace("\r", " ").replace("\n", " ").strip()
                    if not s:
                        return ""
                    if len(s) <= int(max_len):
                        return s
                    return s[: int(max_len)].rstrip() + "…"
                except Exception:
                    return str(v or "").strip()

            def _pick(row: Dict[str, str], primary: str, fallback: str) -> str:
                try:
                    v = str(row.get(primary) or "").strip()
                    if v and v.lower() != "nan":
                        return v
                    v2 = str(row.get(fallback) or "").strip()
                    return v2 if v2.lower() != "nan" else ""
                except Exception:
                    return ""

            focus_rows: List[Dict[str, object]] = []
            for s in shops:
                s2 = str(s or "").strip()
                if not s2:
                    continue
                sc_path = out_dir / s2 / "dashboard" / "shop_scorecard.json"
                sc = _read_json(sc_path)
                scorecard = sc.get("scorecard") if isinstance(sc.get("scorecard"), dict) else {}
                actions = scorecard.get("actions") if isinstance(scorecard.get("actions"), dict) else {}
                watchlists = scorecard.get("watchlists") if isinstance(scorecard.get("watchlists"), dict) else {}
                try:
                    p0 = int(actions.get("p0_count") or 0) if isinstance(actions, dict) else 0
                except Exception:
                    p0 = 0
                try:
                    risk = 0
                    for k in ("profit_reduce_count", "oos_with_ad_spend_count", "spend_up_no_sales_count", "phase_down_recent_count"):
                        try:
                            risk += int(watchlists.get(k) or 0) if isinstance(watchlists, dict) else 0
                        except Exception:
                            pass
                except Exception:
                    risk = 0
                try:
                    opp = int(watchlists.get("scale_opportunity_count") or 0) if isinstance(watchlists, dict) else 0
                except Exception:
                    opp = 0

                focus_rows.append({"shop": s2, "p0_actions": p0, "risk_pool": risk, "opp_pool": opp})

            # 仅展示 Top 3（避免入口变长）；优先看“P0动作多 + 风险池大”的店铺
            focus_rows = sorted(
                focus_rows,
                key=lambda r: (
                    int(r.get("p0_actions") or 0),
                    int(r.get("risk_pool") or 0),
                    int(r.get("opp_pool") or 0),
                ),
                reverse=True,
            )
            top = [r for r in focus_rows if (int(r.get("p0_actions") or 0) + int(r.get("risk_pool") or 0) + int(r.get("opp_pool") or 0)) > 0][
                :3
            ]
            if top:
                lines += [
                    '<a id="focus"></a>',
                    "## 本次重点（先看这几个）",
                    "",
                    "- 排序规则：按 `P0动作数` + `风险池条数` 粗排（仅用于抓重点，不改变任何口径）。",
                    "",
                ]
                for i, r in enumerate(top, start=1):
                    shop2 = str(r.get("shop") or "").strip()
                    # 优先链接到 dashboard.html（更聚焦），不存在则退回 shop START_HERE
                    dash_rel = f"{shop2}/reports/dashboard.html"
                    if not (out_dir / dash_rel).exists():
                        dash_rel = f"{shop2}/START_HERE.html" if (out_dir / shop2 / "START_HERE.html").exists() else f"{shop2}/START_HERE.md"
                    lines += [
                        f"### {i}) {shop2}",
                        "",
                        f"- 汇总：P0动作={r.get('p0_actions', 0)}，风险池={r.get('risk_pool', 0)}，机会池={r.get('opp_pool', 0)}",
                        f"- {L(dash_rel)} | {L(f'{shop2}/dashboard/action_board.csv')}",
                    ]

                    # --- Top ASIN 线索（风险/机会）---
                    try:
                        dash_dir = out_dir / shop2 / "dashboard"

                        # 风险 Top1：按优先级依次取第一行（尽量给出可执行线索）
                        risk_candidates = [
                            ("断货仍烧钱", dash_dir / "oos_with_ad_spend_watchlist.csv", "oos_with_ad_spend_days", "ad_spend_roll"),
                            ("加花费无增量", dash_dir / "spend_up_no_sales_watchlist.csv", "delta_sales", "delta_spend"),
                            ("利润控量", dash_dir / "profit_reduce_watchlist.csv", "profit_after_ads", "ad_spend_roll"),
                            ("阶段走弱", dash_dir / "phase_down_recent_watchlist.csv", "phase_trend_14d", "ad_spend_roll"),
                        ]
                        risk_line = ""
                        for label, path, k1, k2 in risk_candidates:
                            row = _read_csv_first_row(path)
                            if not row:
                                continue
                            asin = str(row.get("asin") or "").strip().upper()
                            if not asin:
                                continue
                            pname = str(row.get("product_name") or "").strip()
                            pcat = str(row.get("product_category") or "").strip() or "（未分类）"
                            reason = str(row.get("reason_1") or "").strip()
                            v1 = str(row.get(k1) or "").strip()
                            v2 = str(row.get(k2) or "").strip()
                            fallback = f"{shop2}/dashboard/{path.name}"
                            link = _build_asin_link(shop2, asin, fallback)
                            extra = []
                            if reason:
                                extra.append(reason)
                            if v1:
                                extra.append(f"{k1}={v1}")
                            if v2:
                                extra.append(f"{k2}={v2}")
                            extra_s = ("；" + "，".join(extra)) if extra else ""
                            risk_line = f"- 风险Top1（{label}）：[{_md_cell(asin)}]({link}) {_md_cell(pcat)} / {_md_cell(pname)}{extra_s}（{L(fallback)}）"
                            break
                        if risk_line:
                            lines.append(risk_line)

                        # 机会 Top1：来自 scale_opportunity_watchlist 第一行
                        opp_path = dash_dir / "scale_opportunity_watchlist.csv"
                        opp = _read_csv_first_row(opp_path)
                        if opp:
                            asin = str(opp.get("asin") or "").strip().upper()
                            if asin:
                                pname = str(opp.get("product_name") or "").strip()
                                pcat = str(opp.get("product_category") or "").strip() or "（未分类）"
                                ds = str(opp.get("delta_sales") or "").strip()
                                ms = str(opp.get("marginal_tacos") or "").strip()
                                cv = str(opp.get("inventory_cover_days_7d") or "").strip()
                                fallback = f"{shop2}/dashboard/scale_opportunity_watchlist.csv"
                                link = _build_asin_link(shop2, asin, fallback)
                                extra = []
                                if ds:
                                    extra.append(f"delta_sales={ds}")
                                if ms:
                                    extra.append(f"marginal_tacos={ms}")
                                if cv:
                                    extra.append(f"cover7d={cv}")
                                extra_s = ("；" + "，".join(extra)) if extra else ""
                                lines.append(
                                    f"- 机会Top1（可放量窗口）：[{_md_cell(asin)}]({link}) {_md_cell(pcat)} / {_md_cell(pname)}{extra_s}（{L(fallback)}）"
                                )

                        # --- Top 类目（3）：来自 category_cockpit ---
                        try:
                            cat_rows = _read_csv_head(dash_dir / "category_cockpit.csv", n=3)
                            lines += ["", "#### Top 类目（3）", ""]
                            if cat_rows:
                                headers = ["类目", "ASIN数", "利润", "广告花费", "TACOS", "P0动作", "OOS烧钱ASIN"]
                                lines.append("| " + " | ".join(headers) + " |")
                                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                                for row in cat_rows:
                                    cat = str(row.get("product_category") or "").strip() or "（未分类）"
                                    fallback = f"{shop2}/dashboard/category_cockpit.csv"
                                    link2 = _build_category_link(shop2, cat, fallback)
                                    cell_cat = f"[{_md_cell(cat)}]({link2})"
                                    vals = [
                                        cell_cat,
                                        _md_cell(row.get("asin_count")),
                                        _md_cell(row.get("profit_total")),
                                        _md_cell(row.get("ad_spend_total")),
                                        _md_cell(row.get("tacos_total")),
                                        _md_cell(row.get("category_p0_action_count")),
                                        _md_cell(row.get("oos_with_ad_spend_asin_count")),
                                    ]
                                    lines.append("| " + " | ".join(vals) + " |")
                            else:
                                lines.append("- （暂无：`dashboard/category_cockpit.csv` 为空或未生成）")
                        except Exception:
                            pass

                        # --- Top ASIN（3）：来自 asin_cockpit ---
                        try:
                            asin_rows = _read_csv_head(dash_dir / "asin_cockpit.csv", n=3)
                            lines += ["", "#### Top ASIN（3）", ""]
                            if asin_rows:
                                headers = ["ASIN", "品名", "类目", "阶段", "focus", "cover7d", "OOS烧钱天数", "ΔSales", "ΔSpend"]
                                lines.append("| " + " | ".join(headers) + " |")
                                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                                for row in asin_rows:
                                    asin2 = str(row.get("asin") or "").strip().upper()
                                    if not asin2:
                                        continue
                                    fallback = f"{shop2}/dashboard/asin_cockpit.csv"
                                    link2 = _build_asin_link(shop2, asin2, fallback)
                                    cell_asin = f"[{_md_cell(asin2)}]({link2})"
                                    pname2 = _md_cell(row.get("product_name"))
                                    pcat2 = _md_cell(row.get("product_category") or "（未分类）")
                                    phase2 = _md_cell(row.get("current_phase"))
                                    focus2 = _md_cell(row.get("focus_score"))
                                    cover7d = _pick(row, "inventory_cover_days_7d", "inventory_cover_days_14d")
                                    oos_days = _pick(row, "oos_with_ad_spend_days", "flag_oos")
                                    ds = _pick(row, "delta_sales", "drivers_delta_sales")
                                    dsp = _pick(row, "delta_spend", "drivers_delta_ad_spend")
                                    vals = [
                                        cell_asin,
                                        pname2,
                                        pcat2,
                                        phase2,
                                        focus2,
                                        _md_cell(cover7d),
                                        _md_cell(oos_days),
                                        _md_cell(ds),
                                        _md_cell(dsp),
                                    ]
                                    lines.append("| " + " | ".join(vals) + " |")
                            else:
                                lines.append("- （暂无：`dashboard/asin_cockpit.csv` 为空或未生成）")
                        except Exception:
                            pass

                        # --- 本周行动（Top 3）：来自 unlock_scale_tasks ---
                        try:
                            task_rows = _read_csv_head(dash_dir / "unlock_scale_tasks.csv", n=3)
                            lines += ["", "#### 本周行动（Top 3）", ""]
                            # 入口跳转：优先 dashboard.html#weekly；否则回退到 shop START_HERE
                            weekly_rel = f"{shop2}/reports/dashboard.html#weekly"
                            if not (out_dir / shop2 / "reports" / "dashboard.html").exists():
                                weekly_rel = f"{shop2}/reports/dashboard.md#weekly" if (out_dir / shop2 / "reports" / "dashboard.md").exists() else dash_rel
                            lines.append(f"- [跳转到本周行动]({weekly_rel}) | {L(f'{shop2}/dashboard/unlock_scale_tasks.csv')}")
                            if task_rows:
                                for row in task_rows:
                                    pr = str(row.get("priority") or "").strip().upper()
                                    ttype = str(row.get("task_type") or "").strip()
                                    direction = str(row.get("direction") or "").strip()
                                    asin3 = str(row.get("asin") or "").strip().upper()
                                    cat3 = str(row.get("product_category") or "").strip() or "（未分类）"
                                    name3 = str(row.get("product_name") or "").strip()
                                    need3 = str(row.get("need") or "").strip() or str(row.get("evidence") or "").strip()

                                    fallback3 = f"{shop2}/dashboard/unlock_scale_tasks.csv"
                                    asin_link = _build_asin_link(shop2, asin3, fallback3) if asin3 else fallback3
                                    cat_link = _build_category_link(shop2, cat3, fallback3)
                                    cat_cell = f"[{_md_cell(cat3)}]({cat_link})"

                                    parts = []
                                    if pr:
                                        parts.append(pr)
                                    if ttype:
                                        parts.append(ttype)
                                    if direction:
                                        parts.append(direction)
                                    head = " ".join(parts).strip() or "任务"
                                    asin_cell = f"[{_md_cell(asin3)}]({asin_link})" if asin3 else ""
                                    line = f"- {head}：{asin_cell} {cat_cell} / {_md_cell(_short(name3, 28))}"
                                    if need3:
                                        line += f" — {_md_cell(_short(need3, 60))}"
                                    lines.append(line)
                            else:
                                lines.append("- （暂无：`dashboard/unlock_scale_tasks.csv` 为空或未生成）")
                        except Exception:
                            pass
                    except Exception:
                        pass
                    lines.append("")
            else:
                lines += [
                    '<a id="focus"></a>',
                    "## 本次重点（先看这几个）",
                    "",
                    "- （暂无可收敛的重点：可能 action_board/watchlists 未生成或均为空。建议先看各店铺的 Dashboard。）",
                    "",
                ]
        except Exception:
            pass

        # ops_policy 配置校验：避免调参误配导致“结论不可信/不知是否生效”
        try:
            if (out_dir / "ops_policy_warnings.html").exists():
                lines += [
                    '<a id="policy"></a>',
                    "## 配置校验（ops_policy）",
                    "",
                    f"- {L('ops_policy_warnings.html')}（推荐：更好读）",
                    f"- {L('ops_policy_warnings.md')}（Markdown 源文件）",
                ]
                if (out_dir / "config" / "ops_policy.json").exists():
                    lines.append(f"- {L('config/ops_policy.json')}（本次配置快照）")
                if (out_dir / "config" / "ops_profile.json").exists():
                    lines.append(f"- {L('config/ops_profile.json')}（本次总选项快照：少数几个开关）")
                if (out_dir / "config" / "ops_policy_effective.json").exists():
                    lines.append(f"- {L('config/ops_policy_effective.json')}（本次生效配置：默认/兜底/截断后的实际值）")
                lines.append("")
        except Exception:
            pass

        # 轻量汇总表：只给你快速扫（不做复杂横向对比）
        rows_total: List[Dict[str, object]] = []
        rows_recent: List[Dict[str, object]] = []
        rows_ops: List[Dict[str, object]] = []
        for s in shops:
            s2 = str(s or "").strip()
            if not s2:
                continue
            sc_path = out_dir / s2 / "dashboard" / "shop_scorecard.json"
            sc = _read_json(sc_path)
            dr = sc.get("date_range") if isinstance(sc.get("date_range"), dict) else {}
            scorecard = sc.get("scorecard") if isinstance(sc.get("scorecard"), dict) else {}
            biz = scorecard.get("biz_kpi") if isinstance(scorecard.get("biz_kpi"), dict) else {}
            risks = scorecard.get("risks") if isinstance(scorecard.get("risks"), dict) else {}
            actions = scorecard.get("actions") if isinstance(scorecard.get("actions"), dict) else {}
            watchlists = scorecard.get("watchlists") if isinstance(scorecard.get("watchlists"), dict) else {}
            rows_total.append(
                {
                    "shop": s2,
                    "date_start": dr.get("date_start") if isinstance(dr, dict) else "",
                    "date_end": dr.get("date_end") if isinstance(dr, dict) else "",
                    "sales_total": biz.get("sales_total") if isinstance(biz, dict) else "",
                    "profit_total": biz.get("profit_total") if isinstance(biz, dict) else "",
                    "ad_spend_total": biz.get("ad_spend_total") if isinstance(biz, dict) else "",
                    "tacos_total": biz.get("tacos_total") if isinstance(biz, dict) else "",
                    "ad_acos_total": biz.get("ad_acos_total") if isinstance(biz, dict) else "",
                    "asin_oos_count": risks.get("asin_oos_count") if isinstance(risks, dict) else "",
                    "asin_low_inventory_count": risks.get("asin_low_inventory_count") if isinstance(risks, dict) else "",
                }
            )

            # 近期窗口（默认取 7 天）：更贴近“当下要怎么调”
            try:
                compares = scorecard.get("compares") if isinstance(scorecard.get("compares"), list) else []
            except Exception:
                compares = []
            c7 = None
            if isinstance(compares, list):
                for c in compares:
                    if not isinstance(c, dict):
                        continue
                    try:
                        if int(c.get("window_days") or 0) == 7:
                            c7 = c
                            break
                    except Exception:
                        continue
            rows_recent.append(
                {
                    "shop": s2,
                    "recent_start_7d": c7.get("recent_start") if isinstance(c7, dict) else "",
                    "recent_end_7d": c7.get("recent_end") if isinstance(c7, dict) else "",
                    "sales_recent_7d": c7.get("sales_recent") if isinstance(c7, dict) else "",
                    "profit_recent_7d": c7.get("profit_recent") if isinstance(c7, dict) else "",
                    "ad_spend_recent_7d": c7.get("ad_spend_recent") if isinstance(c7, dict) else "",
                    "tacos_recent_7d": c7.get("tacos_recent") if isinstance(c7, dict) else "",
                    "delta_sales_7d": c7.get("delta_sales") if isinstance(c7, dict) else "",
                    "delta_ad_spend_7d": c7.get("delta_ad_spend") if isinstance(c7, dict) else "",
                    "delta_profit_7d": c7.get("delta_profit") if isinstance(c7, dict) else "",
                    "marginal_tacos_7d": c7.get("marginal_tacos") if isinstance(c7, dict) else "",
                }
            )

            # 动作/问题池：给你“更贴近执行”的轻量统计（不做复杂横向对比）
            rows_ops.append(
                {
                    "shop": s2,
                    "actions_total": NL(
                        (actions.get("total") if isinstance(actions, dict) else ""),
                        f"{s2}/dashboard/action_board.csv",
                    ),
                    "actions_p0": NL(
                        (actions.get("p0_count") if isinstance(actions, dict) else ""),
                        f"{s2}/dashboard/action_board.csv",
                    ),
                    "actions_p1": NL(
                        (actions.get("p1_count") if isinstance(actions, dict) else ""),
                        f"{s2}/dashboard/action_board.csv",
                    ),
                    "actions_blocked": NL(
                        (actions.get("blocked_count") if isinstance(actions, dict) else ""),
                        f"{s2}/dashboard/action_board.csv",
                    ),
                    "actions_manual_confirm": NL(
                        (actions.get("needs_manual_confirm_count") if isinstance(actions, dict) else ""),
                        f"{s2}/dashboard/action_board.csv",
                    ),
                    "wl_profit_reduce": NL(
                        (watchlists.get("profit_reduce_count") if isinstance(watchlists, dict) else ""),
                        f"{s2}/dashboard/profit_reduce_watchlist.csv",
                    ),
                    "wl_oos_spend": NL(
                        (watchlists.get("oos_with_ad_spend_count") if isinstance(watchlists, dict) else ""),
                        f"{s2}/dashboard/oos_with_ad_spend_watchlist.csv",
                    ),
                    "wl_spend_up_no_sales": NL(
                        (watchlists.get("spend_up_no_sales_count") if isinstance(watchlists, dict) else ""),
                        f"{s2}/dashboard/spend_up_no_sales_watchlist.csv",
                    ),
                    "wl_phase_down_recent": NL(
                        (watchlists.get("phase_down_recent_count") if isinstance(watchlists, dict) else ""),
                        f"{s2}/dashboard/phase_down_recent_watchlist.csv",
                    ),
                    "wl_scale_opportunity": NL(
                        (watchlists.get("scale_opportunity_count") if isinstance(watchlists, dict) else ""),
                        f"{s2}/dashboard/scale_opportunity_watchlist.csv",
                    ),
                }
            )

        lines += [
            '<a id="summary"></a>',
            "## 店铺汇总（轻量）",
            "",
        ]
        if rows_total:
            lines += ["### 全量（历史汇总）", ""]
            headers = [
                "shop",
                "date_start",
                "date_end",
                "sales_total",
                "profit_total",
                "ad_spend_total",
                "tacos_total",
                "ad_acos_total",
                "asin_oos_count",
                "asin_low_inventory_count",
            ]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for r in rows_total:
                lines.append("| " + " | ".join([str(r.get(h, "")) for h in headers]) + " |")
            lines.append("")

        if rows_recent:
            lines += ["### 近期（近7天）", ""]
            headers = [
                "shop",
                "recent_start_7d",
                "recent_end_7d",
                "sales_recent_7d",
                "profit_recent_7d",
                "ad_spend_recent_7d",
                "tacos_recent_7d",
                "delta_sales_7d",
                "delta_ad_spend_7d",
                "delta_profit_7d",
                "marginal_tacos_7d",
            ]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for r in rows_recent:
                lines.append("| " + " | ".join([str(r.get(h, "")) for h in headers]) + " |")
            lines.append("")

        if rows_ops:
            lines += ["### 动作与问题池（轻量）", ""]
            lines.append("说明：动作数= `dashboard/action_board.csv`（TopN 运营视图）；Watchlists 行数= `dashboard/*watchlist.csv`。")
            lines.append("")
            headers = [
                "shop",
                "actions_total",
                "actions_p0",
                "actions_p1",
                "actions_blocked",
                "actions_manual_confirm",
                "wl_profit_reduce",
                "wl_oos_spend",
                "wl_spend_up_no_sales",
                "wl_phase_down_recent",
                "wl_scale_opportunity",
            ]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for r in rows_ops:
                lines.append("| " + " | ".join([str(r.get(h, "")) for h in headers]) + " |")
            lines.append("")

        if (not rows_total) and (not rows_recent) and (not rows_ops):
            lines.append("- （无汇总数据：可能 shop_scorecard.json 未生成或被清理）")
            lines.append("")

        md_path = out_dir / "START_HERE.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")

        # HTML 版本：更好读（不改变链接/口径；不重写 shop/START_HERE.md 链接）
        try:
            write_report_html_from_md(md_path=md_path, out_path=out_dir / "START_HERE.html")
        except Exception:
            pass

        # Owner 汇总页（多店铺）
        try:
            _write_owner_overview(out_dir=out_dir, shops=shops)
        except Exception:
            pass
    except Exception:
        return


def _write_owner_overview(out_dir: Path, shops: List[str]) -> None:
    """
    L2：多店铺 Owner 汇总页（Top 风险 / Top 机会 / 本周行动）。
    - 不引入交互
    - 仅做“入口级聚合”，不改变任何口径
    """
    try:
        if out_dir is None or not shops:
            return

        def L(rel_path: str) -> str:
            return f"[`{rel_path}`]({rel_path})"

        def _read_csv_rows(path: Path, n: int = 1) -> List[Dict[str, str]]:
            try:
                if path is None or (not path.exists()):
                    return []
                rows: List[Dict[str, str]] = []
                with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if isinstance(row, dict):
                            rows.append({str(k): str(v or "") for k, v in row.items()})
                        if len(rows) >= n:
                            break
                return rows
            except Exception:
                return []

        def _pick(row: Dict[str, str], keys: List[str]) -> str:
            for k in keys:
                v = str(row.get(k, "") or "").strip()
                if v:
                    return v
            return ""

        lines: List[str] = [
            "# Owner 汇总（多店铺）",
            "",
            "说明：只做“入口级聚合”，便于负责人快速定位重点；口径来源保持与各店铺 Dashboard 一致。",
            "",
        ]

        # ===== 1) Top 风险 =====
        risk_sources = [
            ("断货仍烧钱", "dashboard/oos_with_ad_spend_watchlist.csv", "供应链/运营", 0, ["asin", "asin_hint"], ["product_name"], ["oos_with_ad_spend_days", "ad_spend_roll"]),
            ("加花费但销量不增", "dashboard/spend_up_no_sales_watchlist.csv", "广告运营", 1, ["asin", "asin_hint"], ["product_name"], ["delta_spend", "delta_sales"]),
            ("利润控量", "dashboard/profit_reduce_watchlist.csv", "运营/财务", 1, ["asin", "asin_hint"], ["product_name"], ["profit_direction", "max_ad_spend_by_profit"]),
            ("阶段走弱", "dashboard/phase_down_recent_watchlist.csv", "运营/广告运营", 1, ["asin", "asin_hint"], ["product_name"], ["phase_trend_14d", "ad_spend_roll"]),
        ]
        risk_items: List[Tuple[int, str]] = []
        for s in shops:
            shop = str(s or "").strip()
            if not shop:
                continue
            for title, rel, owner, pr, asin_keys, name_keys, detail_keys in risk_sources:
                rel_path = f"{shop}/{rel}"
                rows = _read_csv_rows(out_dir / rel_path, n=1)
                if not rows:
                    continue
                row = rows[0]
                asin = _pick(row, asin_keys)
                name = _pick(row, name_keys)
                detail = _pick(row, detail_keys)
                parts = [p for p in [asin, name] if p]
                extra = f" / {detail}" if detail else ""
                label = " ".join(parts).strip()
                label = f" - {label}" if label else ""
                pr_tag = "P0" if int(pr) <= 0 else "P1"
                risk_items.append((pr, f"- `{pr_tag}` `{owner}` `{shop}`: {title}{label}{extra}（{L(rel_path)}）"))
        risk_items.sort(key=lambda x: x[0])
        lines += ["## Top 风险", ""]
        if risk_items:
            for _, line in risk_items[:12]:
                lines.append(line)
        else:
            lines.append("- （暂无数据）")
        lines.append("")

        # ===== 2) Top 机会 =====
        opp_sources = [
            ("放量机会", "dashboard/scale_opportunity_watchlist.csv", "广告运营", 0, ["asin", "asin_hint"], ["product_name"], ["delta_sales", "inventory_cover_days_7d"]),
            ("可执行动作", "dashboard/opportunity_action_board.csv", "广告运营", 1, ["asin_hint"], ["object_name"], ["action_type", "action_value"]),
        ]
        opp_items: List[Tuple[int, str]] = []
        for s in shops:
            shop = str(s or "").strip()
            if not shop:
                continue
            for title, rel, owner, pr, asin_keys, name_keys, detail_keys in opp_sources:
                rel_path = f"{shop}/{rel}"
                rows = _read_csv_rows(out_dir / rel_path, n=1)
                if not rows:
                    continue
                row = rows[0]
                asin = _pick(row, asin_keys)
                name = _pick(row, name_keys)
                detail = _pick(row, detail_keys)
                parts = [p for p in [asin, name] if p]
                extra = f" / {detail}" if detail else ""
                label = " ".join(parts).strip()
                label = f" - {label}" if label else ""
                pr_tag = "P0" if int(pr) <= 0 else "P1"
                opp_items.append((pr, f"- `{pr_tag}` `{owner}` `{shop}`: {title}{label}{extra}（{L(rel_path)}）"))
        opp_items.sort(key=lambda x: x[0])
        lines += ["## Top 机会", ""]
        if opp_items:
            for _, line in opp_items[:12]:
                lines.append(line)
        else:
            lines.append("- （暂无数据）")
        lines.append("")

        # ===== 3) 本周行动 =====
        action_items: List[Tuple[int, str]] = []
        action_items_fallback: List[Tuple[int, str]] = []
        pr_rank = {"P0": 0, "P1": 1, "P2": 2}
        seen_asin: set = set()
        seen_task: set = set()
        for s in shops:
            shop = str(s or "").strip()
            if not shop:
                continue
            rel_path = f"{shop}/dashboard/unlock_scale_tasks.csv"
            rows = _read_csv_rows(out_dir / rel_path, n=10)
            for row in rows:
                p = str(row.get("priority", "") or "").strip().upper()
                pr = pr_rank.get(p, 9)
                owner = str(row.get("owner", "") or "").strip() or "运营"
                asin = _pick(row, ["asin"])
                task = _pick(row, ["task_type"])
                need = _pick(row, ["need"])
                label = f"{task}".strip() if task else ""
                extra_parts = [x for x in [asin, need] if x]
                extra = " / " + " · ".join(extra_parts) if extra_parts else ""
                prefix = " ".join([x for x in [f"`{p}`" if p else "", f"`{owner}`", f"`{shop}`"] if x])
                line = f"- {prefix}: {label}{extra}（{L(rel_path)}）"
                if (asin and asin in seen_asin) or (task and task in seen_task):
                    action_items_fallback.append((pr, line))
                else:
                    action_items.append((pr, line))
                    if asin:
                        seen_asin.add(asin)
                    if task:
                        seen_task.add(task)
        action_items = action_items + action_items_fallback
        action_items.sort(key=lambda x: x[0])
        lines += ["## 本周行动", ""]
        if action_items:
            for _, line in action_items[:18]:
                lines.append(line)
        else:
            lines.append("- （暂无数据）")
        lines.append("")

        md_path = out_dir / "OWNER_OVERVIEW.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        try:
            write_report_html_from_md(md_path=md_path, out_path=out_dir / "OWNER_OVERVIEW.html")
        except Exception:
            pass
    except Exception:
        return


def _build_asin_meta(pa_shop: pd.DataFrame, pl_shop: pd.DataFrame) -> pd.DataFrame:
    """
    给生命周期输出补充“可读字段”：
    - product_name：产品名（品名）
    - product_category：商品分类（来自 productListing）

    说明：你后续希望报告按“商品分类 -> ASIN”组织，所以这里把分类一并带上。
    """
    rows = []
    try:
        if pa_shop is not None and not pa_shop.empty and "ASIN" in pa_shop.columns:
            g = pa_shop.copy()
            g["asin_norm"] = g["ASIN"].astype(str).str.upper().str.strip()
            # 取最近一天的“品名/标题”作为展示（避免跨月变更导致显示混乱）
            if CAN.date in g.columns:
                g = g.sort_values(CAN.date)
            for asin_norm, gg in g.groupby("asin_norm", dropna=False):
                a = str(asin_norm).strip()
                if not a or a.lower() == "nan":
                    continue
                last = gg.iloc[-1].to_dict()
                rows.append(
                    {
                        "asin_norm": a,
                        "product_name": str(last.get("品名", "") or ""),
                    }
                )
    except Exception:
        rows = []

    df = pd.DataFrame(rows)

    # 用 productListing 补全缺失（例如产品分析里品名为空）
    try:
        if pl_shop is not None and not pl_shop.empty and "ASIN" in pl_shop.columns:
            pl = pl_shop.copy()
            pl["asin_norm"] = pl["ASIN"].astype(str).str.upper().str.strip()
            keep = ["asin_norm"]
            for c in ("品名", "商品分类"):
                if c in pl.columns:
                    keep.append(c)
            pl = pl[keep].drop_duplicates("asin_norm")

            if df.empty:
                df = pl.rename(columns={"品名": "product_name", "商品分类": "product_category"}).copy()
            else:
                df = df.merge(pl.rename(columns={"品名": "product_name_pl", "商品分类": "product_category"}), on="asin_norm", how="left")
                if "product_name" in df.columns and "product_name_pl" in df.columns:
                    df["product_name"] = df["product_name"].astype(str)
                    df.loc[(df["product_name"].str.strip() == "") | (df["product_name"].str.lower() == "nan"), "product_name"] = df["product_name_pl"]
                df = df.drop(columns=[c for c in ["product_name_pl"] if c in df.columns])
    except Exception:
        pass

    if df is None or df.empty:
        return pd.DataFrame(columns=["asin_norm", "product_name", "product_category"])
    # 清洗
    for c in ("product_name", "product_category"):
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()
            df.loc[df[c].str.lower() == "nan", c] = ""
    return df


def run(
    reports_root: Path,
    out_dir: Path,
    stage: str = "growth",
    only_shops: Optional[List[str]] = None,
    days: int = 0,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    windows_days: Optional[List[int]] = None,
    render_report: bool = True,
    render_dashboard_md: bool = True,
    lifecycle_daily: bool = False,
    lifecycle_daily_days: int = 0,
    lifecycle_cfg: Optional[LifecycleConfig] = None,
    output_profile: str = "minimal",
    ops_log_root: Optional[Path] = None,
    action_review_windows: Optional[List[int]] = None,
    ai_report: bool = False,
    ai_prompt_only: bool = False,
    ai_prefix: str = "LLM",
    ai_max_asins: int = 40,
    ai_max_actions: int = 200,
    ai_timeout: int = 180,
) -> None:
    """
    reports_root 结构约定：
    - {reports_root}/ad/**.xlsx  （可以按店铺分目录，例如 ad/SH/xxx.xlsx）
    - {reports_root}/productListing.xlsx
    - {reports_root}/产品分析/*.xlsx （按月分文件）
    """
    cfg = get_stage_config(stage)

    ad_root = _resolve_ad_root(reports_root)
    reports = load_ad_reports(ad_root)

    product_listing_path = _resolve_product_listing_path(reports_root)
    product_analysis_dir = _resolve_product_analysis_dir(reports_root)

    product_listing = pd.DataFrame()
    product_analysis = pd.DataFrame()
    try:
        if product_listing_path is not None and product_listing_path.exists():
            product_listing = load_product_listing(product_listing_path)
    except Exception:
        product_listing = pd.DataFrame()
    try:
        if product_analysis_dir.exists():
            product_analysis = load_product_analysis(product_analysis_dir)
    except Exception:
        product_analysis = pd.DataFrame()

    shops = _shop_list(reports)
    if only_shops:
        only = {s.strip() for s in only_shops if s.strip()}
        shops = [s for s in shops if s in only]

    gaps = _data_gaps(reports)

    # 合并各类型报表
    df_search_term = _concat_reports(reports, "search_term")
    df_targeting = _concat_reports(reports, "targeting")
    df_campaign = _concat_reports(reports, "campaign")
    df_ad_group = _concat_reports(reports, "ad_group")
    df_placement = _concat_reports(reports, "placement")
    df_ad_product = _concat_reports(reports, "advertised_product")
    df_purchased = _concat_reports(reports, "purchased_product")

    # 内存优化：concat 完成后不再需要单文件级 DataFrame，释放 LoadedReport 列表，避免“双份常驻”
    try:
        reports = []
    except Exception:
        pass

    # 日期过滤策略：
    # - date_start/date_end：全局过滤（所有店铺一致）
    # - days：每个店铺按“自己数据的最大日期”截取最近 N 天（避免某店铺只下载上个月导致被全局窗口剪空）
    days_int = int(days or 0)
    ds_arg = parse_date(date_start) if date_start else None
    de_arg = parse_date(date_end) if date_end else None

    if ds_arg is not None or de_arg is not None:
        df_search_term = _filter_by_date(df_search_term, ds_arg, de_arg)
        df_targeting = _filter_by_date(df_targeting, ds_arg, de_arg)
        df_campaign = _filter_by_date(df_campaign, ds_arg, de_arg)
        df_ad_group = _filter_by_date(df_ad_group, ds_arg, de_arg)
        df_placement = _filter_by_date(df_placement, ds_arg, de_arg)
        df_ad_product = _filter_by_date(df_ad_product, ds_arg, de_arg)
        df_purchased = _filter_by_date(df_purchased, ds_arg, de_arg)
        if not product_analysis.empty:
            product_analysis = _filter_by_date(product_analysis, ds_arg, de_arg)

    now = dt.datetime.now().isoformat(timespec="seconds")
    windows_days = windows_days or [7, 14, 30]
    action_review_windows = action_review_windows or [7, 14]
    output_profile = str(output_profile or "minimal").strip().lower()
    lifecycle_daily_days = int(lifecycle_daily_days or 0)

    # ops 策略（读不到就用默认）：dashboard TopN 等也从这里取
    policy_path: Optional[Path] = None
    profile_path: Optional[Path] = None
    profile: Optional[OpsProfile] = None
    profile_warns: List[str] = []
    try:
        repo_root = Path(__file__).resolve().parents[2]
        policy_path = repo_root / "config" / "ops_policy.json"
        profile_path = repo_root / "config" / "ops_profile.json"
        profile, profile_warns = load_ops_profile(profile_path)
        overrides = ops_profile_to_overrides(profile)
        policy = load_ops_policy_with_overrides(policy_path, overrides)
    except Exception:
        policy = OpsPolicy()
        policy_path = None
        profile_path = None
        profile = None
        profile_warns = []

    # ops_policy 结构校验：只提示，不阻断跑数（写到 output/<run>/ops_policy_warnings.*）
    try:
        if policy_path is not None:
            _write_ops_policy_warnings(
                out_dir=out_dir,
                policy_path=policy_path,
                profile_path=profile_path,
                profile=profile,
                profile_warnings=profile_warns,
            )
            _write_ops_policy_snapshot(
                out_dir=out_dir,
                policy_path=policy_path,
                policy=policy,
                profile_path=profile_path,
            )
    except Exception:
        pass

    for shop in shops:
        shop_dir = out_dir / shop
        shop_dir.mkdir(parents=True, exist_ok=True)
        _write_start_here(
            shop_dir,
            shop,
            output_profile=output_profile,
            render_dashboard_md=bool(render_dashboard_md),
            render_full_report=bool(render_report),
            policy=policy,
        )

        def pick(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or CAN.shop not in df.columns:
                return pd.DataFrame()
            return df[df[CAN.shop] == shop].copy()

        st = pick(df_search_term)
        tgt = pick(df_targeting)
        camp = pick(df_campaign)
        ag = pick(df_ad_group)
        pl = pick(df_placement)
        ap = pick(df_ad_product)
        pp = pick(df_purchased)

        # --days：按“当前店铺数据的最大日期”截取窗口（更符合运营直觉）
        shop_ds = ds_arg
        shop_de = de_arg
        if days_int > 0:
            inferred_end = shop_de or _infer_max_date([camp, st, tgt, pl, ap, pp])
            if inferred_end is not None:
                shop_de = inferred_end
                if shop_ds is None:
                    shop_ds = shop_de - dt.timedelta(days=days_int - 1)
            st = _filter_by_date(st, shop_ds, shop_de)
            tgt = _filter_by_date(tgt, shop_ds, shop_de)
            camp = _filter_by_date(camp, shop_ds, shop_de)
            ag = _filter_by_date(ag, shop_ds, shop_de)
            pl = _filter_by_date(pl, shop_ds, shop_de)
            ap = _filter_by_date(ap, shop_ds, shop_de)
            pp = _filter_by_date(pp, shop_ds, shop_de)

        # dashboard 展示用的“实际口径日期范围”（即使你没传 --days/--date-start，也尽量推断）
        def _infer_min_date(dfs: List[pd.DataFrame]) -> Optional[dt.date]:
            dmin: Optional[dt.date] = None
            for df in dfs:
                if df is None or df.empty or CAN.date not in df.columns:
                    continue
                try:
                    v = df[CAN.date].min()
                except Exception:
                    v = None
                if v is None:
                    continue
                if dmin is None or v < dmin:
                    dmin = v
            return dmin

        view_date_end = shop_de or _infer_max_date([camp, st, tgt, pl, ap, pp]) or None
        view_date_start = shop_ds or _infer_min_date([camp, st, tgt, pl, ap, pp]) or None
        view_date_start_s = str(view_date_start) if view_date_start else ""
        view_date_end_s = str(view_date_end) if view_date_end else ""

        # 指标汇总（给 AI 的结构化输入）
        # 注意：不要把 search_term/targeting/placement 等不同粒度表直接相加，会重复统计；
        # 全局汇总以“广告活动报告（campaign）”为主（没有的话再降级用其它表）。
        summary_base = camp if not camp.empty else (tgt if not tgt.empty else (st if not st.empty else ap))
        bundle: Dict[str, object] = {
            "shop": shop,
            "generated_at": now,
            "stage": {"name": cfg.name, "target_acos": cfg.target_acos, "waste_spend": cfg.waste_spend, "min_clicks": cfg.min_clicks},
            "data_gaps": gaps,
            "date_filter": {"days": days_int, "date_start": str(shop_ds) if shop_ds else "", "date_end": str(shop_de) if shop_de else ""},
            "summary_total": to_summary_dict(summary_base),
        }

        # 按广告类型汇总（SP/SB/SD）
        if not st.empty and CAN.ad_type in st.columns:
            by_type = summarize(st, [CAN.ad_type]).sort_values("spend", ascending=False)
            bundle["search_term_by_ad_type"] = by_type.to_dict(orient="records")
        if not camp.empty and CAN.ad_type in camp.columns:
            by_type = summarize(camp, [CAN.ad_type]).sort_values("spend", ascending=False)
            bundle["campaign_by_ad_type"] = by_type.to_dict(orient="records")

        # Top campaigns（按花费/按销售额）
        if not camp.empty and CAN.campaign in camp.columns:
            top_camp = summarize(camp, [CAN.ad_type, CAN.campaign]).sort_values("spend", ascending=False).head(30)
            bundle["top_campaigns"] = top_camp.to_dict(orient="records")

        # Top placements（用来做“广告位倾斜”）
        if not pl.empty and CAN.placement in pl.columns:
            top_place = summarize(pl, [CAN.ad_type, CAN.placement]).sort_values("spend", ascending=False).head(30)
            bundle["top_placements"] = top_place.to_dict(orient="records")

        # Top search terms（浪费/高效）
        if not st.empty and CAN.search_term in st.columns:
            st_sum = summarize(st, [CAN.ad_type, CAN.search_term, CAN.match_type]).sort_values("spend", ascending=False).head(50)
            bundle["top_search_terms"] = st_sum.to_dict(orient="records")

        # ASIN 维度（广告产品）
        if not ap.empty and CAN.asin in ap.columns:
            asin_sum = summarize(ap, [CAN.ad_type, CAN.asin]).sort_values("spend", ascending=False).head(50)
            bundle["top_ad_asins"] = asin_sum.to_dict(orient="records")

        # 已购买商品（跨ASIN关联成交）：只做提示，不做复杂归因
        if not pp.empty and CAN.asin in pp.columns and CAN.other_asin in pp.columns and "其他SKU销售额" in pp.columns:
            pp = pp.copy()
            pp["其他SKU销售额"] = pp["其他SKU销售额"].apply(to_float)
            halo = (
                pp.groupby([CAN.ad_type, CAN.asin, CAN.other_asin], dropna=False, as_index=False)
                .agg(other_sales=("其他SKU销售额", "sum"))
                .sort_values("other_sales", ascending=False)
                .head(50)
            )
            bundle["halo_candidates"] = halo.to_dict(orient="records")

        # 产品分析（全店铺一起下载的）：按 shop 切一份基础摘要给 AI
        pa_shop = pd.DataFrame()
        if not product_analysis.empty and CAN.shop in product_analysis.columns:
            pa_shop = product_analysis[product_analysis[CAN.shop] == shop].copy()
        if not pa_shop.empty and (shop_ds is not None or shop_de is not None):
            pa_shop = _filter_by_date(pa_shop, shop_ds, shop_de)

        # 产品底座（在线商品/可售/成本等）：按 shop 切一份
        pl_shop = pd.DataFrame()
        if not product_listing.empty and CAN.shop in product_listing.columns:
            pl_shop = product_listing[product_listing[CAN.shop] == shop].copy()

        if not pa_shop.empty:
            # 只做“产品层面的广告/自然”汇总，避免过度理论化
            try:
                dmin = str(pa_shop[CAN.date].min())
                dmax = str(pa_shop[CAN.date].max())
            except Exception:
                dmin, dmax = "", ""
            bundle["product_analysis_range"] = {"date_min": dmin, "date_max": dmax}
            # 全局：销售额、广告花费、广告销售额、广告订单量、Sessions、转化率、毛利润
            keys = ["销售额", "广告花费", "广告销售额", "广告订单量", "Sessions", "转化率", "毛利润"]
            sums = {k: float(pa_shop[k].sum()) for k in keys if k in pa_shop.columns}
            # TACOS（广告花费/总销售额）
            if "销售额" in sums and "广告花费" in sums:
                sums["tacos"] = 0.0 if sums["销售额"] == 0 else float(sums["广告花费"]) / float(sums["销售额"])
            bundle["product_analysis_summary"] = sums

            # 产品侧：Top ASIN（按广告花费）
            if "ASIN" in pa_shop.columns and "广告花费" in pa_shop.columns:
                asin_view = (
                    pa_shop.groupby("ASIN", dropna=False, as_index=False)
                    .agg(ad_spend=("广告花费", "sum"), ad_sales=("广告销售额", "sum") if "广告销售额" in pa_shop.columns else ("广告花费", "size"))
                    .copy()
                )
                asin_view["ad_acos"] = asin_view.apply(lambda r: 0.0 if r["ad_sales"] == 0 else float(r["ad_spend"]) / float(r["ad_sales"]), axis=1)
                bundle["product_top_asins_by_ad_spend"] = asin_view.sort_values("ad_spend", ascending=False).head(50).to_dict(orient="records")

        # 生命周期（按 ASIN 的动态周期）：默认输出 segments + current_board，daily 可选
        lifecycle_daily_df = pd.DataFrame()
        lifecycle_segments_df = pd.DataFrame()
        lifecycle_board_df = pd.DataFrame()
        lifecycle_windows_df = pd.DataFrame()
        try:
            if not pa_shop.empty:
                # 生命周期参数：从 CLI/配置传入，否则用默认
                cfg_lc = lifecycle_cfg or LifecycleConfig()
                lifecycle_daily_df, lifecycle_segments_df, lifecycle_board_df = build_lifecycle_for_shop(
                    product_analysis_shop=pa_shop,
                    shop=shop,
                    cfg=cfg_lc,
                )
                ignore_last_days = 0
                try:
                    ignore_last_days = int(getattr(policy, "dashboard_compare_ignore_last_days", 0) or 0)
                except Exception:
                    ignore_last_days = 0
                lifecycle_windows_df = build_lifecycle_windows_for_shop(
                    lifecycle_daily=lifecycle_daily_df,
                    lifecycle_segments=lifecycle_segments_df,
                    lifecycle_board=lifecycle_board_df,
                    windows_days=windows_days,
                    ignore_last_days=ignore_last_days,
                )
        except Exception:
            lifecycle_daily_df = pd.DataFrame()
            lifecycle_segments_df = pd.DataFrame()
            lifecycle_board_df = pd.DataFrame()
            lifecycle_windows_df = pd.DataFrame()

        # 生命周期输出补充“品名/SKU”等元信息（按 ASIN 映射）
        try:
            asin_meta = _build_asin_meta(pa_shop, pl_shop)
            if asin_meta is not None and not asin_meta.empty:
                def _enrich(df: pd.DataFrame, asin_col: str = "asin") -> pd.DataFrame:
                    if df is None or df.empty or asin_col not in df.columns:
                        return df
                    out = df.copy()
                    out["asin_norm"] = out[asin_col].astype(str).str.upper().str.strip()
                    merged = out.merge(asin_meta, on="asin_norm", how="left")
                    return merged.drop(columns=["asin_norm"])

                lifecycle_board_df = _enrich(lifecycle_board_df, "asin")
                lifecycle_segments_df = _enrich(lifecycle_segments_df, "asin")
                lifecycle_windows_df = _enrich(lifecycle_windows_df, "asin")
        except Exception:
            pass

        # ========= ASIN × 广告结构联动（把 search_term/targeting/placement 分摊到 ASIN） =========
        # 说明：这一步是为了把“生命周期窗口（按 ASIN）”与“广告结构（按活动/投放/搜索词）”打通。
        # 输出给 AI/运营的是结构化表，不在这里硬编码“调参动作”。
        asin_campaign_map_df = pd.DataFrame()
        asin_top_campaigns_df = pd.DataFrame()
        asin_top_search_terms_df = pd.DataFrame()
        asin_top_targetings_df = pd.DataFrame()
        asin_top_placements_df = pd.DataFrame()
        try:
            ad_product_daily = build_ad_product_daily(ap)
            if ad_product_daily is not None and not ad_product_daily.empty:
                asin_campaign_map_df = build_asin_campaign_map(ad_product_daily)

                # 分摊权重：默认先用最精细维度（含 date/campaign/ad_group），不命中再回退到 campaign-only
                specs = build_weight_join_specs(ad_product_daily)

                # 1) search_term -> ASIN
                st_alloc = allocate_detail_to_asin(st, specs)
                asin_top_search_terms_df = top_n_entities_by_asin(
                    st_alloc,
                    entity_cols=[CAN.search_term, CAN.match_type, CAN.campaign],
                    top_n=20,
                    min_spend=float(cfg.waste_spend) / 5.0 if float(cfg.waste_spend) > 0 else 1.0,
                )

                # 2) targeting -> ASIN
                tgt_alloc = allocate_detail_to_asin(tgt, specs)
                asin_top_targetings_df = top_n_entities_by_asin(
                    tgt_alloc,
                    entity_cols=[CAN.targeting, CAN.match_type, CAN.campaign],
                    top_n=20,
                    min_spend=float(cfg.waste_spend) / 5.0 if float(cfg.waste_spend) > 0 else 1.0,
                )

                # 3) placement -> ASIN（可选）
                pl_alloc = allocate_detail_to_asin(pl, specs)
                asin_top_placements_df = top_n_entities_by_asin(
                    pl_alloc,
                    entity_cols=[CAN.placement, CAN.campaign],
                    top_n=10,
                    min_spend=float(cfg.waste_spend) / 5.0 if float(cfg.waste_spend) > 0 else 1.0,
                )

                # 4) ASIN Top campaigns（从 advertised_product 直接汇总，最可信）
                if asin_campaign_map_df is not None and not asin_campaign_map_df.empty:
                    g = asin_campaign_map_df.groupby([CAN.shop, CAN.ad_type, CAN.asin, CAN.campaign], dropna=False, as_index=False).agg(
                        impressions=("impressions", "sum"),
                        clicks=("clicks", "sum"),
                        spend=("spend", "sum"),
                        sales=("sales", "sum"),
                        orders=("orders", "sum"),
                    )
                    g["acos"] = g.apply(lambda r: 0.0 if r["sales"] == 0 else float(r["spend"]) / float(r["sales"]), axis=1)
                    g["ctr"] = g.apply(lambda r: 0.0 if r["impressions"] == 0 else float(r["clicks"]) / float(r["impressions"]), axis=1)
                    g["cvr"] = g.apply(lambda r: 0.0 if r["clicks"] == 0 else float(r["orders"]) / float(r["clicks"]), axis=1)
                    rows = []
                    for (shop2, ad_type2, asin2), gg in g.groupby([CAN.shop, CAN.ad_type, CAN.asin], dropna=False):
                        rows.append(gg.sort_values(["spend", "sales", "orders"], ascending=False).head(10))
                    asin_top_campaigns_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        except Exception:
            asin_campaign_map_df = pd.DataFrame()
            asin_top_campaigns_df = pd.DataFrame()
            asin_top_search_terms_df = pd.DataFrame()
            asin_top_targetings_df = pd.DataFrame()
            asin_top_placements_df = pd.DataFrame()

        # 把“联动摘要”写入 bundle（保持小体积；完整表在 CSV）
        try:
            bundle["asin_ad_linkage"] = {
                "has_ad_product": bool(ap is not None and not ap.empty),
                "campaign_map_rows": int(len(asin_campaign_map_df)) if asin_campaign_map_df is not None else 0,
                "top_campaigns_rows": int(len(asin_top_campaigns_df)) if asin_top_campaigns_df is not None else 0,
                "top_search_terms_rows": int(len(asin_top_search_terms_df)) if asin_top_search_terms_df is not None else 0,
                "top_targetings_rows": int(len(asin_top_targetings_df)) if asin_top_targetings_df is not None else 0,
                "top_placements_rows": int(len(asin_top_placements_df)) if asin_top_placements_df is not None else 0,
            }
        except Exception:
            bundle["asin_ad_linkage"] = {}

        # lifecycle 摘要写入 metrics_bundle.json，供 AI/报告引用（避免只看 CSV）
        try:
            if lifecycle_board_df is not None and not lifecycle_board_df.empty and "current_phase" in lifecycle_board_df.columns:
                phase_counts = (
                    lifecycle_board_df.groupby("current_phase", dropna=False)
                    .size()
                    .reset_index(name="asin_count")
                    .sort_values("asin_count", ascending=False)
                )
                # 主口径窗口：默认优先 since_first_stock_to_date（更贴近“上架可售后”）
                main_rows = []
                if lifecycle_windows_df is not None and not lifecycle_windows_df.empty and "window_type" in lifecycle_windows_df.columns:
                    main = lifecycle_windows_df[lifecycle_windows_df["window_type"] == "since_first_stock_to_date"].copy()
                    if main.empty:
                        main = lifecycle_windows_df[lifecycle_windows_df["window_type"] == "cycle_to_date"].copy()
                    if not main.empty:
                        try:
                            if "ad_spend" in main.columns:
                                main = main.sort_values("ad_spend", ascending=False)
                        except Exception:
                            pass
                        main_rows = main.head(50).to_dict(orient="records")

                bundle["lifecycle"] = {
                    "phase_counts": phase_counts.to_dict(orient="records"),
                    "top_by_ad_spend_roll": lifecycle_board_df.head(30).to_dict(orient="records"),
                    "main_windows_top": main_rows,
                }
            else:
                bundle["lifecycle"] = {"phase_counts": [], "top_by_ad_spend_roll": [], "main_windows_top": []}
        except Exception:
            bundle["lifecycle"] = {"phase_counts": [], "top_by_ad_spend_roll": [], "main_windows_top": []}

        # Top targetings（用于“投放层结构”提示）
        if not tgt.empty and CAN.targeting in tgt.columns:
            tcols = [CAN.ad_type, CAN.targeting]
            if CAN.match_type in tgt.columns:
                tcols.append(CAN.match_type)
            if CAN.campaign in tgt.columns:
                tcols.insert(1, CAN.campaign)
            top_tgt = summarize(tgt, tcols).sort_values("spend", ascending=False).head(50)
            bundle["top_targetings"] = top_tgt.to_dict(orient="records")

        # 更数据驱动的诊断（趋势/根因）：这部分会用到你的“私有结构”（每店铺自己的分位数/基线）
        diagnostics: Dict[str, object] = {}
        try:
            diagnostics["campaign_trends"] = diagnose_campaign_trends(camp, cfg)
        except Exception:
            diagnostics["campaign_trends"] = []
        try:
            diagnostics["asin_root_causes"] = diagnose_asin_root_causes(pa_shop, pl_shop, cfg)
        except Exception:
            diagnostics["asin_root_causes"] = []
        try:
            diagnostics["asin_stages"] = infer_asin_stage_by_profit(pa_shop, pl_shop, cfg)
        except Exception:
            diagnostics["asin_stages"] = []
        try:
            diagnostics["profit_health"] = summarize_profit_health(diagnostics.get("asin_stages", []) if isinstance(diagnostics, dict) else [])
        except Exception:
            diagnostics["profit_health"] = {}
        try:
            diagnostics["unlock_scale_plan"] = build_unlock_scale_plan(diagnostics.get("asin_stages", []) if isinstance(diagnostics, dict) else [])
        except Exception:
            diagnostics["unlock_scale_plan"] = []
        try:
            diagnostics["unlock_tasks"] = build_unlock_tasks(diagnostics.get("asin_stages", []) if isinstance(diagnostics, dict) else [])
        except Exception:
            diagnostics["unlock_tasks"] = []
        # 多窗口（7/14/30）对比：增量效率 + 信号
        try:
            diagnostics["temporal"] = build_temporal_insights(camp=camp, tgt=tgt, windows_days=list(windows_days), min_spend=float(cfg.waste_spend))
        except Exception:
            diagnostics["temporal"] = {"shop_end_date": "", "windows": [], "campaign_windows": [], "targeting_windows": []}
        # 店铺诊断 scorecard（店铺层固定口径）
        try:
            diagnostics["shop_scorecard"] = diagnose_shop_scorecard(
                cfg=cfg,
                camp=camp,
                product_analysis_shop=pa_shop,
                lifecycle_board=lifecycle_board_df,
                windows_days=list(windows_days),
            )
        except Exception:
            diagnostics["shop_scorecard"] = {}
        try:
            diagnostics["campaign_budget_map"] = diagnose_campaign_budget_map_from_asin(
                camp=camp,
                advertised_product=ap,
                asin_stages=diagnostics.get("asin_stages", []) if isinstance(diagnostics, dict) else [],
                cfg=cfg,
                temporal=diagnostics.get("temporal") if isinstance(diagnostics, dict) else None,
            )
        except Exception:
            diagnostics["campaign_budget_map"] = []
        try:
            diagnostics["budget_transfer_plan"] = build_budget_transfer_plan(
                diagnostics.get("campaign_budget_map", []) if isinstance(diagnostics, dict) else []
            )
        except Exception:
            diagnostics["budget_transfer_plan"] = {}
        bundle["diagnostics"] = diagnostics

        # 动作候选：L0（结构化、可执行、可回滚）
        # 暂停状态映射（仅用于过滤行动清单，不影响其它分析）
        paused_campaigns = _paused_key_set(camp, [CAN.ad_type, CAN.campaign])
        paused_ad_groups = _paused_key_set(ag, [CAN.ad_type, CAN.campaign, CAN.ad_group])
        tgt_for_status = tgt.copy()
        if not tgt_for_status.empty and CAN.match_type not in tgt_for_status.columns:
            tgt_for_status[CAN.match_type] = ""
        paused_targetings = _paused_key_set(tgt_for_status, [CAN.ad_type, CAN.campaign, CAN.ad_group, CAN.targeting, CAN.match_type])

        # 先过滤“暂停行”再生成动作（避免把已暂停广告加入行动清单）
        st_act = _filter_paused_rows(st)
        tgt_act = _filter_paused_rows(tgt)
        camp_act = _filter_paused_rows(camp)
        pl_act = _filter_paused_rows(pl)

        actions: List[ActionCandidate] = []
        actions.extend(generate_search_term_actions(st_act, cfg))
        actions.extend(generate_targeting_actions(tgt_act, cfg))
        actions.extend(generate_placement_actions(pl_act, cfg))
        actions.extend(generate_campaign_budget_suggestions(camp_act, cfg))
        actions.extend(generate_product_side_actions(pa_shop, pl_shop, cfg))
        # 预算迁移图谱 -> 活动预算方向（更贴近“按ASIN毛利承受度”）
        try:
            dstart = str(shop_ds) if shop_ds else ""
            dend = str(shop_de) if shop_de else ""
            actions.extend(
                generate_campaign_budget_actions_from_map(
                    shop=shop,
                    camp_budget_map=diagnostics.get("campaign_budget_map", []) if isinstance(diagnostics, dict) else [],
                    date_start=dstart,
                    date_end=dend,
                )
            )
        except Exception:
            pass

        # 二次过滤：如活动/广告组为暂停，即使来自其它来源（预算迁移等）也不入行动清单
        actions = _filter_actions_by_paused(actions, paused_campaigns, paused_ad_groups, paused_targetings)

        # ========= 数据质量/维度覆盖（用于 dashboard 口径提示 + ai/data_quality.* 输出） =========
        dq_report: Optional[Dict[str, object]] = None
        dq_hints: List[str] = []
        try:
            dq_report = build_data_quality_report(
                shop=shop,
                st=st,
                tgt=tgt,
                camp=camp,
                plc=pl,
                ap=ap,
                pp=pp,
                product_analysis_shop=pa_shop,
                product_listing_shop=pl_shop,
                lifecycle_board=lifecycle_board_df,
            )
            # dashboard.md 只带 1-2 条“口径提示”，避免运营阅读时信息爆炸
            dq_hints = extract_data_quality_summary_lines(dq_report, max_lines=2)
        except Exception:
            dq_report = None
            dq_hints = []

        # ========= L0+ 执行回填模板 + 复盘输出（可选，没回填也不影响） =========
        # 说明：先生成 action_review（如果有历史回填），再写 dashboard.md（这样 dashboard 第一屏可以引用复盘结果）
        action_review_df = pd.DataFrame()
        try:
            write_execution_log_template(shop_dir=shop_dir, actions=actions)
        except Exception:
            pass
        try:
            exec_log = load_execution_log(ops_log_root=ops_log_root, reports_root=reports_root, shop=shop)
            if exec_log is not None and not exec_log.empty:
                out_path = write_action_review(
                    shop_dir=shop_dir,
                    execution_log=exec_log,
                    st=st,
                    tgt=tgt,
                    camp=camp,
                    pl=pl,
                    windows_days=[int(x) for x in (action_review_windows or []) if int(x) > 0],
                )
                # 读回 DataFrame：用于在 dashboard.md 里给出“执行复盘”摘要（CSV 是给 Excel 用）
                if out_path is not None and Path(out_path).exists():
                    try:
                        action_review_df = pd.read_csv(Path(out_path), encoding="utf-8-sig")
                    except Exception:
                        action_review_df = pd.DataFrame()
        except Exception:
            action_review_df = pd.DataFrame()

        # ========= dashboard 聚焦层输出（总是生成，解决“report.md 太长抓不到重点”） =========
        try:
            write_dashboard_outputs(
                shop_dir=shop_dir,
                shop=shop,
                stage=str(cfg.name),
                date_start=view_date_start_s,
                date_end=view_date_end_s,
                diagnostics=diagnostics,
                product_analysis_shop=pa_shop,
                lifecycle_board=lifecycle_board_df,
                lifecycle_segments=lifecycle_segments_df,
                lifecycle_windows=lifecycle_windows_df,
                asin_campaign_map=asin_campaign_map_df,
                asin_top_search_terms=asin_top_search_terms_df,
                asin_top_targetings=asin_top_targetings_df,
                asin_top_placements=asin_top_placements_df,
                search_term_report=st,
                actions=actions,
                policy=policy,
                render_md=bool(render_dashboard_md),
                data_quality_hints=dq_hints,
                action_review=action_review_df if (action_review_df is not None and not action_review_df.empty) else None,
            )
        except Exception:
            pass

        # ========= 输出（按档位控制，避免目录过乱） =========
        # AI 输入包：无论 minimal/full 都生成（这是主文件）
        try:
            ai_dir = shop_dir / "ai"
            ai_dir.mkdir(parents=True, exist_ok=True)
            ai_bundle = build_ai_input_bundle(
                shop=shop,
                stage=stage,
                date_start=str(shop_ds) if shop_ds else "",
                date_end=str(shop_de) if shop_de else "",
                summary_total=bundle.get("summary_total", {}) if isinstance(bundle, dict) else {},
                product_analysis_summary=bundle.get("product_analysis_summary", {}) if isinstance(bundle, dict) else {},
                shop_scorecard=(diagnostics.get("shop_scorecard") if isinstance(diagnostics, dict) else {}) or {},
                lifecycle_board=lifecycle_board_df,
                lifecycle_windows=lifecycle_windows_df,
                asin_top_campaigns=asin_top_campaigns_df,
                asin_top_search_terms=asin_top_search_terms_df,
                asin_top_targetings=asin_top_targetings_df,
                asin_top_placements=asin_top_placements_df,
                asin_limit=120,
            )
            # 兼容旧路径：根目录也保留一份（便于脚本/工具直接读取）
            (shop_dir / "ai_input_bundle.json").write_text(json_dumps(ai_bundle), encoding="utf-8")
            (ai_dir / "ai_input_bundle.json").write_text(json_dumps(ai_bundle), encoding="utf-8")
        except Exception:
            pass

        # 数据质量/维度覆盖：给 AI/分析用（总是生成，不依赖 report.md）
        try:
            ai_dir = shop_dir / "ai"
            ai_dir.mkdir(parents=True, exist_ok=True)
            dq = dq_report or build_data_quality_report(
                shop=shop,
                st=st,
                tgt=tgt,
                camp=camp,
                plc=pl,
                ap=ap,
                pp=pp,
                product_analysis_shop=pa_shop,
                product_listing_shop=pl_shop,
                lifecycle_board=lifecycle_board_df,
            )
            write_data_quality_files(ai_dir=ai_dir, report=dq)
            # 展示层：生成 HTML（更好读，不改变口径）
            try:
                write_report_html_from_md(md_path=ai_dir / "data_quality.md", out_path=ai_dir / "data_quality.html")
            except Exception:
                pass
        except Exception:
            pass

        # ========= AI 建议报告 / 提示词留档（可选，不影响主流程） =========
        # 说明：
        # - 这里默认不开启，避免误耗 token；
        # - 你启用后，需要配置环境变量（默认前缀 LLM_），例如：
        #   LLM_PROVIDER=oai_http
        #   LLM_API_KEY=...
        #   LLM_MODEL=...
        try:
            if bool(ai_report) or bool(ai_prompt_only):
                from analysis.ai_report import write_ai_suggestions_for_shop

                write_ai_suggestions_for_shop(
                    shop_dir=shop_dir,
                    stage=str(stage or "").strip(),
                    prefix=str(ai_prefix or "LLM").strip() or "LLM",
                    max_asins=int(ai_max_asins or 0),
                    max_actions=int(ai_max_actions or 0),
                    timeout=int(ai_timeout or 180),
                    prompt_only=bool(ai_prompt_only),
                )
                # 展示层：生成 HTML（更好读，不改变口径）
                try:
                    ai_dir = shop_dir / "ai"
                    if (ai_dir / "ai_suggestions.md").exists():
                        write_report_html_from_md(md_path=ai_dir / "ai_suggestions.md", out_path=ai_dir / "ai_suggestions.html")
                except Exception:
                    pass
        except Exception:
            pass

        if output_profile == "full":
            # full：保留所有中间表，方便你深挖/对照/排查
            try:
                ai_dir = shop_dir / "ai"
                ai_dir.mkdir(parents=True, exist_ok=True)
                # 兼容旧路径：根目录仍保留一份；AI 推荐读 ai/ 下的
                (shop_dir / "metrics_bundle.json").write_text(json_dumps(bundle), encoding="utf-8")
                (shop_dir / "data_gaps.json").write_text(json_dumps(gaps), encoding="utf-8")
                (ai_dir / "metrics_bundle.json").write_text(json_dumps(bundle), encoding="utf-8")
                (ai_dir / "data_gaps.json").write_text(json_dumps(gaps), encoding="utf-8")
            except Exception:
                pass
            _write_action_candidates(shop_dir / "action_candidates.csv", actions)
            _write_budget_transfers(
                shop_dir / "budget_transfers.csv",
                diagnostics.get("budget_transfer_plan", {}) if isinstance(diagnostics, dict) else {},
            )
            _write_budget_table(
                shop_dir / "budget_cuts.csv",
                (diagnostics.get("budget_transfer_plan", {}) if isinstance(diagnostics, dict) else {}).get("cuts"),
                ["ad_type", "campaign", "severity", "camp_spend", "cut_usd_estimated", "asin_hint", "note"],
            )
            _write_budget_table(
                shop_dir / "budget_adds.csv",
                (diagnostics.get("budget_transfer_plan", {}) if isinstance(diagnostics, dict) else {}).get("adds"),
                ["ad_type", "campaign", "severity", "camp_spend", "add_usd_estimated", "asin_hint", "note"],
            )
            _write_budget_table(
                shop_dir / "budget_savings.csv",
                (diagnostics.get("budget_transfer_plan", {}) if isinstance(diagnostics, dict) else {}).get("savings"),
                ["from_ad_type", "from_campaign", "from_asin_hint", "to_bucket", "amount_usd_estimated", "note"],
            )
            _write_budget_table(
                shop_dir / "unlock_scale.csv",
                diagnostics.get("unlock_scale_plan", []) if isinstance(diagnostics, dict) else [],
                [
                    "priority",
                    "asin",
                    "stage",
                    "direction",
                    "ad_spend",
                    "profit_before_ads",
                    "profit_after_ads",
                    "max_ad_spend_by_profit",
                    "inventory",
                    "gap_usd",
                    "fix",
                    "reasons",
                ],
            )
            _write_budget_table(
                shop_dir / "unlock_tasks.csv",
                diagnostics.get("unlock_tasks", []) if isinstance(diagnostics, dict) else [],
                ["priority", "asin", "task_type", "owner", "need", "target", "budget_gap_usd_est", "profit_gap_usd_est", "stage", "direction", "evidence"],
            )
            _write_budget_table(
                shop_dir / "temporal_campaign_windows.csv",
                ((diagnostics.get("temporal", {}) if isinstance(diagnostics, dict) else {}).get("campaign_windows")),
                [
                    "window_days",
                    "recent_start",
                    "recent_end",
                    "prev_start",
                    "prev_end",
                    "ad_type",
                    "campaign",
                    "spend_prev",
                    "spend_recent",
                    "sales_prev",
                    "sales_recent",
                    "orders_prev",
                    "orders_recent",
                    "delta_spend",
                    "delta_sales",
                    "delta_orders",
                    "marginal_acos",
                    "marginal_cpa",
                    "signal",
                    "score",
                ],
            )
            _write_budget_table(
                shop_dir / "temporal_targeting_windows.csv",
                ((diagnostics.get("temporal", {}) if isinstance(diagnostics, dict) else {}).get("targeting_windows")),
                [
                    "window_days",
                    "recent_start",
                    "recent_end",
                    "prev_start",
                    "prev_end",
                    "ad_type",
                    "targeting",
                    "spend_prev",
                    "spend_recent",
                    "sales_prev",
                    "sales_recent",
                    "orders_prev",
                    "orders_recent",
                    "delta_spend",
                    "delta_sales",
                    "delta_orders",
                    "marginal_acos",
                    "marginal_cpa",
                    "signal",
                    "score",
                ],
            )
            # ASIN × 广告结构联动：落盘
            try:
                _write_budget_table(
                    shop_dir / "asin_campaign_map.csv",
                    asin_campaign_map_df.to_dict(orient="records") if asin_campaign_map_df is not None and not asin_campaign_map_df.empty else [],
                    [CAN.shop, CAN.ad_type, CAN.asin, CAN.campaign, CAN.ad_group, "impressions", "clicks", "spend", "sales", "orders", "ctr", "cvr", "acos"],
                )
                _write_budget_table(
                    shop_dir / "asin_top_campaigns.csv",
                    asin_top_campaigns_df.to_dict(orient="records") if asin_top_campaigns_df is not None and not asin_top_campaigns_df.empty else [],
                    [CAN.shop, CAN.ad_type, CAN.asin, CAN.campaign, "impressions", "clicks", "spend", "sales", "orders", "ctr", "cvr", "acos"],
                )
                _write_budget_table(
                    shop_dir / "asin_top_search_terms.csv",
                    asin_top_search_terms_df.to_dict(orient="records") if asin_top_search_terms_df is not None and not asin_top_search_terms_df.empty else [],
                    [CAN.shop, CAN.ad_type, CAN.asin, CAN.campaign, CAN.search_term, CAN.match_type, "impressions", "clicks", "spend", "sales", "orders", "ctr", "cvr", "acos"],
                )
                _write_budget_table(
                    shop_dir / "asin_top_targetings.csv",
                    asin_top_targetings_df.to_dict(orient="records") if asin_top_targetings_df is not None and not asin_top_targetings_df.empty else [],
                    [CAN.shop, CAN.ad_type, CAN.asin, CAN.campaign, CAN.targeting, CAN.match_type, "impressions", "clicks", "spend", "sales", "orders", "ctr", "cvr", "acos"],
                )
                _write_budget_table(
                    shop_dir / "asin_top_placements.csv",
                    asin_top_placements_df.to_dict(orient="records") if asin_top_placements_df is not None and not asin_top_placements_df.empty else [],
                    [CAN.shop, CAN.ad_type, CAN.asin, CAN.campaign, CAN.placement, "impressions", "clicks", "spend", "sales", "orders", "ctr", "cvr", "acos"],
                )
            except Exception:
                pass

        # 生命周期落盘（segments + current_board 默认输出；daily 仅在 lifecycle_daily=True 时输出）
        try:
            if output_profile == "full":
                _write_budget_table(
                    shop_dir / "lifecycle_current_board.csv",
                    lifecycle_board_df.to_dict(orient="records") if lifecycle_board_df is not None and not lifecycle_board_df.empty else [],
                    [
                        "shop",
                        "asin",
                        "product_name",
                        "cycle_id",
                        "current_phase",
                        "date",
                        "sales_roll",
                        "sessions_roll",
                        "ad_spend_roll",
                        "profit_roll",
                        "tacos_roll",
                        "cvr_roll",
                        "inventory",
                        "flag_low_inventory",
                        "flag_oos",
                    ],
                )
                _write_budget_table(
                    shop_dir / "lifecycle_segments.csv",
                    lifecycle_segments_df.to_dict(orient="records") if lifecycle_segments_df is not None and not lifecycle_segments_df.empty else [],
                    [
                        "shop",
                        "asin",
                        "product_name",
                        "cycle_id",
                        "segment_id",
                        "phase",
                        "date_start",
                        "date_end",
                        "days",
                        "sales_sum",
                        "orders_sum",
                        "sessions_sum",
                        "ad_spend_sum",
                        "profit_sum",
                        "tacos",
                        "cvr",
                        "inv_min",
                        "low_inv_days",
                        "oos_days",
                        "oos_with_sessions_days",
                        "oos_with_ad_spend_days",
                    ],
                )
                _write_budget_table(
                    shop_dir / "lifecycle_windows.csv",
                    lifecycle_windows_df.to_dict(orient="records") if lifecycle_windows_df is not None and not lifecycle_windows_df.empty else [],
                    [
                        "shop",
                        "asin",
                        "product_name",
                        "cycle_id",
                        "window_type",
                        "phase",
                        "date_start",
                        "date_end",
                        "cycle_days",
                        "phase_days",
                        "first_stock_date",
                        "first_active_date",
                        "first_ad_spend_date",
                        "first_sale_date",
                        "first_sale_in_stock_date",
                        "last_sale_date",
                        "peak_sales_roll_date",
                        "sales",
                        "orders",
                        "sessions",
                        "ad_spend",
                        "ad_sales",
                        "ad_orders",
                        "profit",
                        "tacos",
                        "ad_acos",
                        "cvr",
                        "ad_impressions",
                        "ad_clicks",
                        "ad_ctr",
                        "ad_cvr",
                        "organic_orders",
                        "organic_sales",
                        "ad_sales_share",
                        "organic_sales_share",
                        "ad_orders_share",
                        "sp_spend",
                        "sb_spend",
                        "sd_spend",
                        "in_stock_days",
                        "oos_days",
                        "oos_with_sessions_days",
                        "oos_with_ad_spend_days",
                        "presale_order_days",
                        "prelaunch_days",
                        "prelaunch_ad_spend",
                        "prelaunch_sessions",
                        "prelaunch_ad_clicks",
                        "days_stock_to_first_sale",
                        "days_active_to_first_sale",
                        "days_ad_to_first_sale",
                        "flag_stock_after_sale",
                        "flag_ad_sales_gt_total",
                        "flag_ad_orders_gt_total",
                        "window_days",
                        "recent_start",
                        "recent_end",
                        "prev_start",
                        "prev_end",
                        "spend_prev",
                        "spend_recent",
                        "sales_prev",
                        "sales_recent",
                        "orders_prev",
                        "orders_recent",
                        "sessions_prev",
                        "sessions_recent",
                        "cvr_prev",
                        "cvr_recent",
                        "organic_sales_prev",
                        "organic_sales_recent",
                        "organic_sales_share_prev",
                        "organic_sales_share_recent",
                        "ad_clicks_prev",
                        "ad_clicks_recent",
                        "delta_spend",
                        "delta_sales",
                        "delta_orders",
                        "delta_sessions",
                        "delta_cvr",
                        "delta_ad_clicks",
                        "delta_organic_sales",
                        "delta_organic_sales_share",
                        "marginal_tacos",
                        "marginal_ad_acos",
                    ],
                )
            if lifecycle_daily and lifecycle_daily_df is not None and not lifecycle_daily_df.empty:
                view = lifecycle_daily_df.copy()
                if lifecycle_daily_days > 0 and CAN.date in view.columns:
                    try:
                        dmax = view[CAN.date].max()
                        if isinstance(dmax, dt.date):
                            dmin = dmax - dt.timedelta(days=lifecycle_daily_days - 1)
                            view = view[view[CAN.date] >= dmin].copy()
                    except Exception:
                        pass
                _write_budget_table(
                    shop_dir / "lifecycle_daily.csv",
                    view.to_dict(orient="records"),
                    [
                        "shop",
                        "asin",
                        CAN.date,
                        "cycle_id",
                        "active",
                        "lifecycle_phase",
                        "sales_roll",
                        "sessions_roll",
                        "ad_spend_roll",
                        "profit_roll",
                        "tacos_roll",
                        "cvr_roll",
                        "sales_slope",
                        "flag_low_inventory",
                        "flag_oos",
                        "FBA可售",
                        "销售额",
                        "订单量",
                        "Sessions",
                        "广告花费",
                        "广告销售额",
                        "广告订单量",
                        "毛利润",
                    ],
                )
        except Exception:
            pass

        # 合成版报告（Markdown + 图）
        if render_report:
            generate_shop_report(
                shop_dir=shop_dir,
                shop=shop,
                cfg=cfg,
                summary_total=bundle.get("summary_total", {}),
                st=st,
                tgt=tgt,
                camp=camp,
                pl=pl,
                ap=ap,
                pp=pp,
                product_listing_shop=pl_shop,
                product_analysis_shop=pa_shop,
                lifecycle_board=lifecycle_board_df,
                lifecycle_segments=lifecycle_segments_df,
                lifecycle_windows=lifecycle_windows_df,
                asin_top_campaigns=asin_top_campaigns_df,
                asin_top_search_terms=asin_top_search_terms_df,
                asin_top_targetings=asin_top_targetings_df,
                asin_top_placements=asin_top_placements_df,
                actions=actions,
                diagnostics=diagnostics,
            )

        # 最后再写一次 START_HERE（此时文件已生成，可判断哪些文件存在）
        _write_start_here(
            shop_dir,
            shop,
            output_profile=output_profile,
            render_dashboard_md=bool(render_dashboard_md),
            render_full_report=bool(render_report),
            policy=policy,
        )

    # run 级入口（方便你一次性查看所有店铺）
    try:
        _write_run_start_here(
            out_dir=out_dir,
            shops=shops,
            render_dashboard_md=bool(render_dashboard_md),
            ai_report=bool(ai_report),
        )
    except Exception:
        pass


def run_for_sh_example() -> None:
    """
    方便你快速试跑的默认入口（按你目前的目录结构）。
    """
    run(reports_root=Path("reports"), out_dir=Path(".outputs/ppc_ai"), stage="growth")
