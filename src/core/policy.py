# -*- coding: utf-8 -*-
"""
运营策略配置（把“经验参数”从代码里抽出来，便于你按业务不断校准）。

设计原则：
- 失败不崩：JSON 配置读不到/格式不对时，用默认值兜底
- 尽量少字段：先覆盖“最影响建议方向”的参数
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


@dataclass(frozen=True)
class ScaleWindowPolicy:
    """
    “可放量窗口”筛选阈值（可配置）。

    说明：
    - 用于机会池（scale_opportunity_watchlist）与 Action Board 的 asin_scale_window 标记；
    - 目标是把“可放量/可迁移预算”的判断从硬编码变成可校准参数。
    """

    # 增量/速度（>0 表示需要至少有增长/有销量）
    min_sales_per_day_7d: float = 0.0
    min_delta_sales: float = 0.0

    # 库存覆盖（>=）
    min_inventory_cover_days_7d: float = 30.0

    # 效率阈值（<=）
    max_tacos_roll: float = 0.25
    max_marginal_tacos: float = 0.25

    # 生命周期排除（默认不在 decline/inactive 里做“放量窗口”）
    exclude_phases: List[str] = field(default_factory=lambda: ["decline", "inactive"])

    # 风险约束（默认都要求“没有明显库存/断货风险”）
    require_no_oos: bool = True
    require_no_low_inventory: bool = True
    require_oos_with_ad_spend_days_zero: bool = True


@dataclass(frozen=True)
class FocusScoringPolicy:
    """
    ASIN Focus 的评分规则（可配置）。

    说明：
    - 这里的分数只用于“排序抓重点”，不是模型预测，也不是精确归因。
    - 你可以把它理解为“运营优先级加权器”：越需要处理，分越高。
    """

    # 基础项：按“最近 rolling 广告花费”给基础分（越花钱越先看）
    base_spend_log_multiplier: float = 12.0
    base_spend_score_cap: float = 60.0

    # 风险/异常权重（加分）
    weight_flag_oos: float = 15.0
    weight_flag_low_inventory: float = 10.0
    weight_oos_with_ad_spend_days: float = 25.0
    weight_oos_with_sessions_days: float = 8.0
    weight_presale_order_days: float = 6.0

    # 增量效率（加分）
    weight_spend_up_no_sales: float = 18.0
    weight_marginal_tacos_worse: float = 12.0

    # 生命周期语境（加分）
    weight_decline_or_inactive_spend: float = 8.0

    # 生命周期迁移信号（加分）
    # - 近14天阶段走弱（例如 growth→decline）：优先关注原因（断货/差评/投放结构/利润承受度等）
    weight_phase_down_recent: float = 6.0

    # 广告依赖/库存（加分）
    weight_high_ad_dependency: float = 6.0
    weight_inventory_zero_still_spend: float = 10.0

    # 阈值
    high_ad_dependency_threshold: float = 0.8
    marginal_tacos_worse_ratio: float = 1.2

    # ===== 产品侧转化异常（Sessions↑但 CVR↓）=====
    # 说明：
    # - 该信号用于“抓重点排序/告警/阶段走弱原因标签”的可解释线索；
    # - 默认做得比较保守：只在样本量足够时触发，避免小样本噪声。
    weight_sessions_up_cvr_down: float = 8.0
    # 前一窗口 sessions 至少多少才认为“样本量足够”
    cvr_signal_min_sessions_prev: float = 100.0
    # sessions 增量至少多少才认为“流量显著上升”
    cvr_signal_min_delta_sessions: float = 50.0
    # CVR 下降至少多少（绝对值，例如 0.02 = 下降 2 个百分点）
    cvr_signal_min_cvr_drop: float = 0.02
    # 为避免“无广告花费但 CVR 波动”的噪声，只在 ad_spend_roll 达到该值才触发
    cvr_signal_min_ad_spend_roll: float = 10.0

    # ===== 自然端回落信号（7d vs prev7d）=====
    # 说明：
    # - 用于识别“自然端变弱导致的销量压力”，优先回到 Listing/价格/评价/变体/促销等产品语境；
    # - 默认保守：避免小体量产品/小样本的波动噪声。
    weight_organic_down: float = 6.0
    # 前一窗口“自然销售额”至少多少才认为样本量足够（美元）
    organic_signal_min_organic_sales_prev: float = 100.0
    # 自然销售额至少下降多少才触发（绝对值，美元）
    organic_signal_min_delta_organic_sales: float = 20.0
    # 自然销售额下降比例阈值（recent/prev <= 该值触发；0.8=下降≥20%）
    organic_signal_drop_ratio: float = 0.8

    # ===== 客单价 AOV 信号（7d vs prev7d）=====
    # 说明：
    # - AOV = sales / orders（产品侧“订单均价”），用于识别“销量压力是否来自客单变化”
    # - 默认保守：样本量约束 + 下降比例 + 绝对值双阈值，避免小样本噪声
    # - 该信号主要用于阶段走弱 Watchlist 的原因标签（客单价下滑）
    aov_signal_min_orders_prev: float = 10.0
    aov_signal_min_delta_aov: float = 2.0
    # AOV 下降比例阈值（recent/prev <= 该值触发；0.9=下降≥10%）
    aov_signal_drop_ratio: float = 0.9

    # ===== 毛利率信号（gross_margin=profit/sales）=====
    # 说明：用于提示“利润空间过窄”导致的控量/提价/降成本优先级（避免只在广告里调结构）
    gross_margin_signal_min_sales: float = 200.0
    gross_margin_signal_low_threshold: float = 0.15


@dataclass(frozen=True)
class SignalScoringPolicy:
    """
    产品/广告侧 Sigmoid 信号评分（用于排序参考，可配置）。

    说明：
    - product_* 用于产品侧变化（销量/流量/自然/利润方向）
    - ad_* 用于广告效率风险（ACoS/CVR/加花费无增量）
    """

    # 产品侧权重
    product_sales_weight: float = 0.4
    product_sessions_weight: float = 0.2
    product_organic_sales_weight: float = 0.3
    product_profit_weight: float = 0.1
    product_steepness: float = 4.0

    # 广告侧权重
    ad_acos_weight: float = 0.6
    ad_cvr_weight: float = 0.3
    ad_spend_up_no_sales_weight: float = 0.1
    ad_steepness: float = 4.0


@dataclass(frozen=True)
class InventorySigmoidPolicy:
    """
    库存调速（Sigmoid）建议：仅用于“建议/提示”，不直接改价与不影响排序。
    """

    enabled: bool = True
    # DoS 口径：默认使用 inventory_cover_days_7d
    optimal_cover_days: float = 45.0
    steepness: float = 0.1
    min_modifier: float = 0.5
    max_modifier: float = 1.5
    # 只有当 |modifier-1| >= 该值时才输出建议
    min_change_ratio: float = 0.1
    # 只对仍在投放（有花费）者给出建议
    min_ad_spend_roll: float = 10.0


@dataclass(frozen=True)
class ProfitGuardPolicy:
    """
    利润护栏（Break-even）：基于毛利率与目标净利率计算安全 ACOS/CPC。
    """

    enabled: bool = True
    target_net_margin: float = 0.05
    min_sales_7d: float = 50.0
    min_ad_spend_roll: float = 10.0


@dataclass(frozen=True)
class StageScoringPolicy:
    """
    阶段化指标与权重（新品期/成熟期/衰退期）。

    说明：
    - 用于把“指标关注点”按阶段拆开（新品重流量效率；成熟重稳定与成本）。
    - 以“阶段内中位数”做相对判断，减少绝对阈值误判。
    """

    # 阶段划分（可配置）
    # 说明：launch/growth 可单独拆分；未配置时会回退到 new_phases 兼容旧配置。
    launch_phases: List[str] = field(default_factory=lambda: ["launch"])
    growth_phases: List[str] = field(default_factory=lambda: ["growth"])
    new_phases: List[str] = field(default_factory=lambda: ["launch", "growth"])
    mature_phases: List[str] = field(default_factory=lambda: ["mature", "stable"])
    decline_phases: List[str] = field(default_factory=lambda: ["decline", "inactive"])

    # 中位数计算最小样本（过小则不触发阶段对比）
    median_min_samples: int = 8

    # 新品期样本量门槛（避免小样本噪声）
    min_impressions_7d: float = 200.0
    min_clicks_7d: float = 30.0
    min_orders_7d: float = 3.0

    # 成熟期样本量门槛
    min_sales_7d: float = 50.0
    min_orders_7d_mature: float = 5.0

    # 新品期相对阈值（相对阶段中位数）
    new_ctr_low_ratio: float = 0.7
    new_cvr_low_ratio: float = 0.7
    new_cpa_high_ratio: float = 1.3

    # 成熟期相对阈值
    mature_cpa_high_ratio: float = 1.2
    mature_acos_high_ratio: float = 1.2

    # 成熟期稳定性阈值（绝对变化）
    mature_ad_share_shift_abs: float = 0.15
    mature_spend_shift_ratio: float = 0.3

    # 权重（用于阶段化加分）
    weight_new_low_ctr: float = 6.0
    weight_new_low_cvr: float = 6.0
    weight_new_high_cpa: float = 8.0
    weight_mature_high_cpa: float = 6.0
    weight_mature_high_acos: float = 6.0
    weight_mature_ad_share_shift: float = 5.0
    weight_mature_spend_shift: float = 5.0

    # 展示上最多保留多少条阶段标签
    max_stage_tags: int = 3


@dataclass(frozen=True)
class ActionScoringPolicy:
    """
    Action Board 的优先级打分规则（可配置）。

    目标：
    - 把“P0/P1/P2 + 证据花费”升级为“更运营化”的排序：融合 ASIN 重点度、弱关联可信度、风险阻断。
    - 分数只用于排序抓重点，不代表精确归因或预测。
    """

    # 基础分（来自 priority）
    base_score_p0: float = 100.0
    base_score_p1: float = 70.0
    base_score_p2: float = 40.0
    base_score_other: float = 30.0

    # 证据花费（动作层证据）log 缩放权重
    spend_log_multiplier: float = 8.0

    # ASIN 重点度（来自 ASIN Focus 的 focus_score）
    weight_focus_score: float = 0.5

    # 弱关联可信度（0~1）
    weight_hint_confidence: float = 20.0

    # 低可信度放量惩罚（避免“拿不准还放量”）
    low_hint_confidence_threshold: float = 0.35
    low_hint_scale_penalty: float = 15.0

    # 生命周期语境：衰退/不活跃时放量惩罚（不强制阻断）
    phase_scale_penalty_decline: float = 10.0
    phase_scale_penalty_inactive: float = 20.0

    # 利润承受度方向：只对“放量类动作”做轻量加权（不强制阻断）
    # - reduce：当前更建议控量/止血，避免排序靠前误导运营
    # - scale：利润承受度内，可作为预算迁移/加码候选（仍需库存确认）
    profit_reduce_scale_penalty: float = 15.0
    profit_scale_scale_boost: float = 5.0


@dataclass(frozen=True)
class BudgetTransferOpportunityPolicy:
    """
    “机会池 → 预算迁移”联动策略（可配置）。

    背景：
    - 预算迁移（budget_transfer_plan）默认依赖 asin_stages 的利润方向（reduce/scale）。
    - 但你现在的核心目标是“调广告要结合产品立体数据”，而机会池（scale_opportunity_watchlist）
      可能在利润方向仍偏保守时依然存在“可放量窗口”。

    因此提供一个“软联动”：
    - 当本期没有 scale 侧 campaign（导致预算只能回收/RESERVE）时，
      用机会 ASIN → Campaign 映射，把一部分回收预算迁移到“能承接机会”的 Campaign。

    注意：
    - 由于赛狐报表通常拿不到“当前预算值”，金额仍是估算（基于本期花费/结构信号）。
    - 该策略不生成批量回写文件，仅提供运营执行清单与证据列。
    """

    enabled: bool = True
    # 估算“加码需求”的比例（基于机会 ASIN 在该 Campaign 下的 spend proxy）
    suggested_add_pct: float = 20.0
    # 最多输出多少个“承接机会”的 Campaign（避免输出过多导致运营迷失）
    max_target_campaigns: int = 25
    # Campaign 被纳入目标池的最低机会 spend proxy（过小可能是噪声）
    min_target_opp_spend: float = 5.0
    # 优先在同广告类型内迁移（SP→SP/SB→SB/SD→SD）
    prefer_same_ad_type: bool = True


@dataclass(frozen=True)
class UnlockTasksPolicy:
    """
    放量解锁任务（unlock_scale_tasks.csv）的收敛策略（可配置）。

    背景：
    - unlock_tasks 是“放量池很少/为0 时”的关键解释，但全量任务会很长；
    - 运营更需要一个“本周先修什么”的 Top 列表；
    - 仍保留 full 表用于追溯/深挖，Top 表用于分派与执行。
    """

    # 输出 Top 表最多多少行
    top_n: int = 30
    # Top 表包含哪些优先级（默认只保留 P0/P1）
    include_priorities: List[str] = field(default_factory=lambda: ["P0", "P1"])
    # Dashboard 第一屏“本周先修”展示多少条（<=5 建议）
    dashboard_top_n: int = 5
    # Dashboard 第一屏“本周先修”是否优先保证 ASIN 不重复（更便于分工覆盖面）
    dashboard_prefer_unique_asin: bool = True


@dataclass(frozen=True)
class KeywordTopicsPolicy:
    """
    关键词主题（n-gram）聚合策略（可配置）。

    背景：
    - search_term 报表通常非常长，运营难以快速抓到“在烧什么主题/哪些主题在带量”；
    - 用 n-gram（1~3gram）把大量搜索词压缩成可解释的主题列表；
    - 注意：同一条搜索词会贡献多个 n-gram，因此主题 spend 会有重复计数（只用于线索，不做精确归因）。
    """

    enabled: bool = True
    # n-gram 的 n 列表（建议 1~3）
    n_values: List[int] = field(default_factory=lambda: [1, 2, 3])
    # 过滤噪声：单个搜索词累计花费小于该值会被忽略（美元）
    min_term_spend: float = 1.0
    # 性能/聚焦：按 spend 取 TopN 搜索词再做 n-gram（长尾非常大时建议 2000~10000）
    max_terms: int = 5000
    # 输出 CSV 最多保留多少行主题（按 spend 排序）
    max_rows: int = 2000
    # 每个主题保留多少条“代表搜索词样例”（按 spend 排序），用于解释
    top_terms_per_ngram: int = 3
    # reports/dashboard.md 里展示 Top N（建议 ≤5）
    md_top_n: int = 5
    # reports/dashboard.md 里展示主题的最小 n（2 表示优先展示短语，避免 1-gram 太宽泛）
    md_min_n: int = 2

    # ===== 主题建议（可执行清单）=====
    # 是否输出 keyword_topics_action_hints.csv
    action_hints_enabled: bool = True
    # action_hints 输出：Top 浪费主题条数 / Top 贡献主题条数
    action_hints_top_waste: int = 20
    action_hints_top_scale: int = 20
    # 浪费主题过滤阈值：waste_spend >= min_waste_spend 且 waste_ratio >= min_waste_ratio
    action_hints_min_waste_spend: float = 10.0
    action_hints_min_waste_ratio: float = 0.6
    # 放量主题过滤阈值：acos <= target_acos * scale_acos_multiplier 且 sales >= min_sales
    action_hints_scale_acos_multiplier: float = 1.2
    action_hints_min_sales: float = 0.0
    # 每个主题保留多少条“落地定位”（Top campaigns / Top ad_groups）
    action_hints_top_entities: int = 3

    # ===== 主题 → 产品语境（Topic → ASIN context）=====
    # 是否输出 keyword_topics_asin_context.csv
    asin_context_enabled: bool = True
    # 仅使用“高置信”的 search_term→ASIN 映射（top1 share >= 该阈值）
    asin_context_min_confidence: float = 0.6
    # 每个主题最多输出多少个 ASIN（按方向：reduce 看 waste_spend，scale 看 sales）
    asin_context_top_asins_per_topic: int = 10


@dataclass(frozen=True)
class PhaseDownRecentAlertPolicy:
    """
    Shop Alerts：阶段走弱（近14天）告警阈值（可配置）。

    触发条件（在代码里固定，不在这里配置）：
    - phase_changed_recent_14d = 1
    - phase_trend_14d = 'down'
    - ad_spend_roll > 0

    本 Policy 只负责“P0/P1 优先级”的判定阈值，便于你按店铺规模校准。
    """

    enabled: bool = True
    # 直接金额路线：阶段走弱池的 ad_spend_roll_sum >= p0_spend_sum 时升为 P0
    p0_spend_sum: float = 200.0
    # 占比路线：阶段走弱池 spend_share >= p0_spend_share 且满足下方两个最小约束时升为 P0
    p0_spend_share: float = 0.25
    p0_spend_sum_min_when_share: float = 50.0
    p0_asin_count_min: int = 5


@dataclass(frozen=True)
class ShopAlertsPolicy:
    """
    Shop Alerts 总配置（可扩展）：不同规则的阈值集合。
    """

    phase_down_recent: PhaseDownRecentAlertPolicy = field(default_factory=PhaseDownRecentAlertPolicy)


@dataclass(frozen=True)
class OpsPolicy:
    # 库存相关
    low_inventory_threshold: int = 20
    block_scale_when_low_inventory: bool = True
    # 库存覆盖天数阻断：用于“库存不低但速度太快”的场景（0 表示关闭）
    block_scale_when_cover_days_below: float = 7.0

    # 关键词漏斗
    keyword_funnel_top_n: int = 12

    # campaign_ops
    campaign_windows_days: List[int] = None  # type: ignore[assignment]
    campaign_min_spend: float = 5.0
    campaign_top_asins_per_campaign: int = 3
    phase_acos_multiplier: Dict[str, float] = None  # type: ignore[assignment]

    # dashboard（聚焦层）
    dashboard_top_asins: int = 50
    dashboard_top_actions: int = 60
    # compare 窗口忽略最近 N 天（默认 0=不忽略）
    dashboard_compare_ignore_last_days: int = 0
    dashboard_scale_window: ScaleWindowPolicy = field(default_factory=ScaleWindowPolicy)
    dashboard_focus_scoring: FocusScoringPolicy = field(default_factory=FocusScoringPolicy)
    dashboard_signal_scoring: SignalScoringPolicy = field(default_factory=SignalScoringPolicy)
    dashboard_stage_scoring: StageScoringPolicy = field(default_factory=StageScoringPolicy)
    dashboard_action_scoring: ActionScoringPolicy = field(default_factory=ActionScoringPolicy)
    dashboard_inventory_sigmoid: InventorySigmoidPolicy = field(default_factory=InventorySigmoidPolicy)
    dashboard_profit_guard: ProfitGuardPolicy = field(default_factory=ProfitGuardPolicy)
    dashboard_budget_transfer_opportunity: BudgetTransferOpportunityPolicy = field(default_factory=BudgetTransferOpportunityPolicy)
    dashboard_unlock_tasks: UnlockTasksPolicy = field(default_factory=UnlockTasksPolicy)
    dashboard_shop_alerts: ShopAlertsPolicy = field(default_factory=ShopAlertsPolicy)
    dashboard_keyword_topics: KeywordTopicsPolicy = field(default_factory=KeywordTopicsPolicy)

    def __post_init__(self) -> None:
        # dataclass frozen 下的默认可变对象处理：用 object.__setattr__
        if self.campaign_windows_days is None:
            object.__setattr__(self, "campaign_windows_days", [7, 14, 30])
        if self.phase_acos_multiplier is None:
            object.__setattr__(
                self,
                "phase_acos_multiplier",
                {
                    "pre_launch": 1.2,
                    "launch": 1.2,
                    "growth": 1.1,
                    "mature": 1.0,
                    "stable": 1.0,
                    "decline": 0.9,
                    "inactive": 0.0,
                },
            )


def _to_int(v: object, default: int) -> int:
    try:
        x = int(float(v))  # 支持 "20" / 20 / 20.0
        return x
    except Exception:
        return int(default)


def _to_float(v: object, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_bool(v: object, default: bool) -> bool:
    try:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)
    except Exception:
        return bool(default)


def load_ops_policy(path: Path) -> OpsPolicy:
    """
    从 JSON 读取 ops 策略；失败则返回默认。
    """
    base = OpsPolicy()
    try:
        if path is None or (not Path(path).exists()):
            return base
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return base

        inv = data.get("inventory") if isinstance(data.get("inventory"), dict) else {}
        kw = data.get("keyword_funnel") if isinstance(data.get("keyword_funnel"), dict) else {}
        cop = data.get("campaign_ops") if isinstance(data.get("campaign_ops"), dict) else {}
        dash = data.get("dashboard") if isinstance(data.get("dashboard"), dict) else {}

        low_thr = max(0, _to_int(inv.get("low_inventory_threshold"), base.low_inventory_threshold))
        block_scale = _to_bool(inv.get("block_scale_when_low_inventory"), base.block_scale_when_low_inventory)
        cover_below = max(
            0.0,
            _to_float(inv.get("block_scale_when_cover_days_below"), base.block_scale_when_cover_days_below),
        )

        top_n = max(1, _to_int(kw.get("top_n"), base.keyword_funnel_top_n))

        windows = cop.get("windows_days")
        if isinstance(windows, list):
            windows2 = []
            for x in windows:
                n = _to_int(x, 0)
                if n > 0:
                    windows2.append(n)
            windows2 = sorted({int(x) for x in windows2})
        else:
            windows2 = list(base.campaign_windows_days)

        min_spend = max(0.0, _to_float(cop.get("min_spend"), base.campaign_min_spend))
        top_asins = max(1, _to_int(cop.get("top_asins_per_campaign"), base.campaign_top_asins_per_campaign))

        mult = cop.get("phase_acos_multiplier")
        mult2 = dict(base.phase_acos_multiplier)
        if isinstance(mult, dict):
            for k, v in mult.items():
                kk = str(k).strip().lower()
                if not kk:
                    continue
                mult2[kk] = _to_float(v, mult2.get(kk, 1.0))

        dash_top_asins = max(1, _to_int(dash.get("top_asins"), base.dashboard_top_asins))
        dash_top_actions = max(1, _to_int(dash.get("top_actions"), base.dashboard_top_actions))
        dash_ignore_last_days = max(0, _to_int(dash.get("compare_ignore_last_days"), base.dashboard_compare_ignore_last_days))
        dash_ignore_last_days = max(0, _to_int(dash.get("compare_ignore_last_days"), base.dashboard_compare_ignore_last_days))

        # “可放量窗口”阈值（可配置；没有则用默认）
        sw_base = base.dashboard_scale_window
        sw_data = dash.get("scale_window") if isinstance(dash.get("scale_window"), dict) else {}
        exclude_phases = list(sw_base.exclude_phases)
        if isinstance(sw_data.get("exclude_phases"), list):
            tmp: List[str] = []
            for x in sw_data.get("exclude_phases"):
                s = str(x or "").strip().lower()
                if s:
                    tmp.append(s)
            if tmp:
                exclude_phases = tmp
        scale_window = ScaleWindowPolicy(
            min_sales_per_day_7d=max(
                0.0,
                _to_float(sw_data.get("min_sales_per_day_7d"), sw_base.min_sales_per_day_7d),
            ),
            min_delta_sales=max(
                0.0,
                _to_float(sw_data.get("min_delta_sales"), sw_base.min_delta_sales),
            ),
            min_inventory_cover_days_7d=max(
                0.0,
                _to_float(
                    sw_data.get("min_inventory_cover_days_7d"),
                    sw_base.min_inventory_cover_days_7d,
                ),
            ),
            max_tacos_roll=max(0.0, _to_float(sw_data.get("max_tacos_roll"), sw_base.max_tacos_roll)),
            max_marginal_tacos=max(0.0, _to_float(sw_data.get("max_marginal_tacos"), sw_base.max_marginal_tacos)),
            exclude_phases=exclude_phases,
            require_no_oos=_to_bool(sw_data.get("require_no_oos"), sw_base.require_no_oos),
            require_no_low_inventory=_to_bool(sw_data.get("require_no_low_inventory"), sw_base.require_no_low_inventory),
            require_oos_with_ad_spend_days_zero=_to_bool(
                sw_data.get("require_oos_with_ad_spend_days_zero"),
                sw_base.require_oos_with_ad_spend_days_zero,
            ),
        )

        # ASIN Focus 评分（可配置；没有则用默认）
        fs_base = base.dashboard_focus_scoring
        fs_data = dash.get("focus_scoring") if isinstance(dash.get("focus_scoring"), dict) else {}
        fs = FocusScoringPolicy(
            base_spend_log_multiplier=max(
                0.0,
                _to_float(fs_data.get("base_spend_log_multiplier"), fs_base.base_spend_log_multiplier),
            ),
            base_spend_score_cap=max(
                0.0,
                _to_float(fs_data.get("base_spend_score_cap"), fs_base.base_spend_score_cap),
            ),
            weight_flag_oos=_to_float(fs_data.get("weight_flag_oos"), fs_base.weight_flag_oos),
            weight_flag_low_inventory=_to_float(fs_data.get("weight_flag_low_inventory"), fs_base.weight_flag_low_inventory),
            weight_oos_with_ad_spend_days=_to_float(fs_data.get("weight_oos_with_ad_spend_days"), fs_base.weight_oos_with_ad_spend_days),
            weight_oos_with_sessions_days=_to_float(fs_data.get("weight_oos_with_sessions_days"), fs_base.weight_oos_with_sessions_days),
            weight_presale_order_days=_to_float(fs_data.get("weight_presale_order_days"), fs_base.weight_presale_order_days),
            weight_spend_up_no_sales=_to_float(fs_data.get("weight_spend_up_no_sales"), fs_base.weight_spend_up_no_sales),
            weight_marginal_tacos_worse=_to_float(fs_data.get("weight_marginal_tacos_worse"), fs_base.weight_marginal_tacos_worse),
            weight_decline_or_inactive_spend=_to_float(fs_data.get("weight_decline_or_inactive_spend"), fs_base.weight_decline_or_inactive_spend),
            weight_phase_down_recent=_to_float(fs_data.get("weight_phase_down_recent"), fs_base.weight_phase_down_recent),
            weight_high_ad_dependency=_to_float(fs_data.get("weight_high_ad_dependency"), fs_base.weight_high_ad_dependency),
            weight_inventory_zero_still_spend=_to_float(fs_data.get("weight_inventory_zero_still_spend"), fs_base.weight_inventory_zero_still_spend),
            high_ad_dependency_threshold=max(
                0.0,
                min(
                    1.0,
                    _to_float(fs_data.get("high_ad_dependency_threshold"), fs_base.high_ad_dependency_threshold),
                ),
            ),
            marginal_tacos_worse_ratio=max(
                0.0,
                _to_float(fs_data.get("marginal_tacos_worse_ratio"), fs_base.marginal_tacos_worse_ratio),
            ),
            weight_sessions_up_cvr_down=_to_float(fs_data.get("weight_sessions_up_cvr_down"), fs_base.weight_sessions_up_cvr_down),
            cvr_signal_min_sessions_prev=max(
                0.0,
                _to_float(fs_data.get("cvr_signal_min_sessions_prev"), fs_base.cvr_signal_min_sessions_prev),
            ),
            cvr_signal_min_delta_sessions=max(
                0.0,
                _to_float(fs_data.get("cvr_signal_min_delta_sessions"), fs_base.cvr_signal_min_delta_sessions),
            ),
            cvr_signal_min_cvr_drop=max(
                0.0,
                min(
                    1.0,
                    _to_float(fs_data.get("cvr_signal_min_cvr_drop"), fs_base.cvr_signal_min_cvr_drop),
                ),
            ),
            cvr_signal_min_ad_spend_roll=max(
                0.0,
                _to_float(fs_data.get("cvr_signal_min_ad_spend_roll"), fs_base.cvr_signal_min_ad_spend_roll),
            ),
            weight_organic_down=_to_float(fs_data.get("weight_organic_down"), fs_base.weight_organic_down),
            organic_signal_min_organic_sales_prev=max(
                0.0,
                _to_float(fs_data.get("organic_signal_min_organic_sales_prev"), fs_base.organic_signal_min_organic_sales_prev),
            ),
            organic_signal_min_delta_organic_sales=max(
                0.0,
                _to_float(fs_data.get("organic_signal_min_delta_organic_sales"), fs_base.organic_signal_min_delta_organic_sales),
            ),
            organic_signal_drop_ratio=max(
                0.0,
                min(
                    1.0,
                    _to_float(fs_data.get("organic_signal_drop_ratio"), fs_base.organic_signal_drop_ratio),
                ),
            ),
            aov_signal_min_orders_prev=max(
                0.0,
                _to_float(fs_data.get("aov_signal_min_orders_prev"), fs_base.aov_signal_min_orders_prev),
            ),
            aov_signal_min_delta_aov=max(
                0.0,
                _to_float(fs_data.get("aov_signal_min_delta_aov"), fs_base.aov_signal_min_delta_aov),
            ),
            aov_signal_drop_ratio=max(
                0.0,
                min(
                    1.0,
                    _to_float(fs_data.get("aov_signal_drop_ratio"), fs_base.aov_signal_drop_ratio),
                ),
            ),
            gross_margin_signal_min_sales=max(
                0.0,
                _to_float(fs_data.get("gross_margin_signal_min_sales"), fs_base.gross_margin_signal_min_sales),
            ),
            gross_margin_signal_low_threshold=max(
                -1.0,
                min(
                    1.0,
                    _to_float(fs_data.get("gross_margin_signal_low_threshold"), fs_base.gross_margin_signal_low_threshold),
                ),
            ),
        )

        # Sigmoid 多维评分（可配置；没有则用默认）
        sigs_base = base.dashboard_signal_scoring
        sigs_data = dash.get("signal_scoring") if isinstance(dash.get("signal_scoring"), dict) else {}
        signal_scoring = SignalScoringPolicy(
            product_sales_weight=max(
                0.0,
                _to_float(sigs_data.get("product_sales_weight"), sigs_base.product_sales_weight),
            ),
            product_sessions_weight=max(
                0.0,
                _to_float(sigs_data.get("product_sessions_weight"), sigs_base.product_sessions_weight),
            ),
            product_organic_sales_weight=max(
                0.0,
                _to_float(sigs_data.get("product_organic_sales_weight"), sigs_base.product_organic_sales_weight),
            ),
            product_profit_weight=max(
                0.0,
                _to_float(sigs_data.get("product_profit_weight"), sigs_base.product_profit_weight),
            ),
            product_steepness=max(
                0.0,
                _to_float(sigs_data.get("product_steepness"), sigs_base.product_steepness),
            ),
            ad_acos_weight=max(
                0.0,
                _to_float(sigs_data.get("ad_acos_weight"), sigs_base.ad_acos_weight),
            ),
            ad_cvr_weight=max(
                0.0,
                _to_float(sigs_data.get("ad_cvr_weight"), sigs_base.ad_cvr_weight),
            ),
            ad_spend_up_no_sales_weight=max(
                0.0,
                _to_float(sigs_data.get("ad_spend_up_no_sales_weight"), sigs_base.ad_spend_up_no_sales_weight),
            ),
            ad_steepness=max(
                0.0,
                _to_float(sigs_data.get("ad_steepness"), sigs_base.ad_steepness),
            ),
        )

        # 阶段化指标权重（可配置；没有则用默认）
        ss_base = base.dashboard_stage_scoring
        ss_data = dash.get("stage_scoring") if isinstance(dash.get("stage_scoring"), dict) else {}

        def _parse_phase_list(v: object, default: List[str]) -> List[str]:
            try:
                if not isinstance(v, list):
                    return list(default)
                out: List[str] = []
                for x in v:
                    s = str(x or "").strip().lower()
                    if s:
                        out.append(s)
                return out if out else list(default)
            except Exception:
                return list(default)

        stage_scoring = StageScoringPolicy(
            launch_phases=_parse_phase_list(ss_data.get("launch_phases"), ss_base.launch_phases),
            growth_phases=_parse_phase_list(ss_data.get("growth_phases"), ss_base.growth_phases),
            new_phases=_parse_phase_list(ss_data.get("new_phases"), ss_base.new_phases),
            mature_phases=_parse_phase_list(ss_data.get("mature_phases"), ss_base.mature_phases),
            decline_phases=_parse_phase_list(ss_data.get("decline_phases"), ss_base.decline_phases),
            median_min_samples=max(
                1,
                _to_int(ss_data.get("median_min_samples"), ss_base.median_min_samples),
            ),
            min_impressions_7d=max(
                0.0,
                _to_float(ss_data.get("min_impressions_7d"), ss_base.min_impressions_7d),
            ),
            min_clicks_7d=max(
                0.0,
                _to_float(ss_data.get("min_clicks_7d"), ss_base.min_clicks_7d),
            ),
            min_orders_7d=max(
                0.0,
                _to_float(ss_data.get("min_orders_7d"), ss_base.min_orders_7d),
            ),
            min_sales_7d=max(
                0.0,
                _to_float(ss_data.get("min_sales_7d"), ss_base.min_sales_7d),
            ),
            min_orders_7d_mature=max(
                0.0,
                _to_float(ss_data.get("min_orders_7d_mature"), ss_base.min_orders_7d_mature),
            ),
            new_ctr_low_ratio=max(
                0.0,
                _to_float(ss_data.get("new_ctr_low_ratio"), ss_base.new_ctr_low_ratio),
            ),
            new_cvr_low_ratio=max(
                0.0,
                _to_float(ss_data.get("new_cvr_low_ratio"), ss_base.new_cvr_low_ratio),
            ),
            new_cpa_high_ratio=max(
                0.0,
                _to_float(ss_data.get("new_cpa_high_ratio"), ss_base.new_cpa_high_ratio),
            ),
            mature_cpa_high_ratio=max(
                0.0,
                _to_float(ss_data.get("mature_cpa_high_ratio"), ss_base.mature_cpa_high_ratio),
            ),
            mature_acos_high_ratio=max(
                0.0,
                _to_float(ss_data.get("mature_acos_high_ratio"), ss_base.mature_acos_high_ratio),
            ),
            mature_ad_share_shift_abs=max(
                0.0,
                _to_float(ss_data.get("mature_ad_share_shift_abs"), ss_base.mature_ad_share_shift_abs),
            ),
            mature_spend_shift_ratio=max(
                0.0,
                _to_float(ss_data.get("mature_spend_shift_ratio"), ss_base.mature_spend_shift_ratio),
            ),
            weight_new_low_ctr=_to_float(ss_data.get("weight_new_low_ctr"), ss_base.weight_new_low_ctr),
            weight_new_low_cvr=_to_float(ss_data.get("weight_new_low_cvr"), ss_base.weight_new_low_cvr),
            weight_new_high_cpa=_to_float(ss_data.get("weight_new_high_cpa"), ss_base.weight_new_high_cpa),
            weight_mature_high_cpa=_to_float(ss_data.get("weight_mature_high_cpa"), ss_base.weight_mature_high_cpa),
            weight_mature_high_acos=_to_float(ss_data.get("weight_mature_high_acos"), ss_base.weight_mature_high_acos),
            weight_mature_ad_share_shift=_to_float(
                ss_data.get("weight_mature_ad_share_shift"),
                ss_base.weight_mature_ad_share_shift,
            ),
            weight_mature_spend_shift=_to_float(
                ss_data.get("weight_mature_spend_shift"),
                ss_base.weight_mature_spend_shift,
            ),
            max_stage_tags=max(1, _to_int(ss_data.get("max_stage_tags"), ss_base.max_stage_tags)),
        )

        # Action Board 优先级（可配置；没有则用默认）
        as_base = base.dashboard_action_scoring
        as_data = dash.get("action_scoring") if isinstance(dash.get("action_scoring"), dict) else {}
        action_scoring = ActionScoringPolicy(
            base_score_p0=_to_float(as_data.get("base_score_p0"), as_base.base_score_p0),
            base_score_p1=_to_float(as_data.get("base_score_p1"), as_base.base_score_p1),
            base_score_p2=_to_float(as_data.get("base_score_p2"), as_base.base_score_p2),
            base_score_other=_to_float(as_data.get("base_score_other"), as_base.base_score_other),
            spend_log_multiplier=_to_float(as_data.get("spend_log_multiplier"), as_base.spend_log_multiplier),
            weight_focus_score=_to_float(as_data.get("weight_focus_score"), as_base.weight_focus_score),
            weight_hint_confidence=_to_float(as_data.get("weight_hint_confidence"), as_base.weight_hint_confidence),
            low_hint_confidence_threshold=max(
                0.0,
                min(
                    1.0,
                    _to_float(
                        as_data.get("low_hint_confidence_threshold"),
                        as_base.low_hint_confidence_threshold,
                    ),
                ),
            ),
            low_hint_scale_penalty=_to_float(as_data.get("low_hint_scale_penalty"), as_base.low_hint_scale_penalty),
            phase_scale_penalty_decline=_to_float(
                as_data.get("phase_scale_penalty_decline"),
                as_base.phase_scale_penalty_decline,
            ),
            phase_scale_penalty_inactive=_to_float(
                as_data.get("phase_scale_penalty_inactive"),
                as_base.phase_scale_penalty_inactive,
            ),
            profit_reduce_scale_penalty=_to_float(
                as_data.get("profit_reduce_scale_penalty"),
                as_base.profit_reduce_scale_penalty,
            ),
            profit_scale_scale_boost=_to_float(
                as_data.get("profit_scale_scale_boost"),
                as_base.profit_scale_scale_boost,
            ),
        )

        # 库存调速（Sigmoid）建议（只用于提示）
        sig_base = base.dashboard_inventory_sigmoid
        sig_data = dash.get("inventory_sigmoid") if isinstance(dash.get("inventory_sigmoid"), dict) else {}
        sig_min = max(0.0, _to_float(sig_data.get("min_modifier"), sig_base.min_modifier))
        sig_max = max(sig_min, _to_float(sig_data.get("max_modifier"), sig_base.max_modifier))
        inventory_sigmoid = InventorySigmoidPolicy(
            enabled=_to_bool(sig_data.get("enabled"), sig_base.enabled),
            optimal_cover_days=max(0.0, _to_float(sig_data.get("optimal_cover_days"), sig_base.optimal_cover_days)),
            steepness=max(0.0, _to_float(sig_data.get("steepness"), sig_base.steepness)),
            min_modifier=sig_min,
            max_modifier=sig_max,
            min_change_ratio=max(0.0, _to_float(sig_data.get("min_change_ratio"), sig_base.min_change_ratio)),
            min_ad_spend_roll=max(0.0, _to_float(sig_data.get("min_ad_spend_roll"), sig_base.min_ad_spend_roll)),
        )

        # 利润护栏（Break-even）
        pg_base = base.dashboard_profit_guard
        pg_data = dash.get("profit_guard") if isinstance(dash.get("profit_guard"), dict) else {}
        profit_guard = ProfitGuardPolicy(
            enabled=_to_bool(pg_data.get("enabled"), pg_base.enabled),
            target_net_margin=max(
                -1.0,
                min(1.0, _to_float(pg_data.get("target_net_margin"), pg_base.target_net_margin)),
            ),
            min_sales_7d=max(0.0, _to_float(pg_data.get("min_sales_7d"), pg_base.min_sales_7d)),
            min_ad_spend_roll=max(0.0, _to_float(pg_data.get("min_ad_spend_roll"), pg_base.min_ad_spend_roll)),
        )

        # 库存调速（Sigmoid）建议（只用于提示）
        sig_base = base.dashboard_inventory_sigmoid
        sig_data = dash.get("inventory_sigmoid") if isinstance(dash.get("inventory_sigmoid"), dict) else {}
        sig_min = max(0.0, _to_float(sig_data.get("min_modifier"), sig_base.min_modifier))
        sig_max = max(sig_min, _to_float(sig_data.get("max_modifier"), sig_base.max_modifier))
        inventory_sigmoid = InventorySigmoidPolicy(
            enabled=_to_bool(sig_data.get("enabled"), sig_base.enabled),
            optimal_cover_days=max(0.0, _to_float(sig_data.get("optimal_cover_days"), sig_base.optimal_cover_days)),
            steepness=max(0.0, _to_float(sig_data.get("steepness"), sig_base.steepness)),
            min_modifier=sig_min,
            max_modifier=sig_max,
            min_change_ratio=max(0.0, _to_float(sig_data.get("min_change_ratio"), sig_base.min_change_ratio)),
            min_ad_spend_roll=max(0.0, _to_float(sig_data.get("min_ad_spend_roll"), sig_base.min_ad_spend_roll)),
        )

        # 利润护栏（Break-even）
        pg_base = base.dashboard_profit_guard
        pg_data = dash.get("profit_guard") if isinstance(dash.get("profit_guard"), dict) else {}
        profit_guard = ProfitGuardPolicy(
            enabled=_to_bool(pg_data.get("enabled"), pg_base.enabled),
            target_net_margin=max(
                -1.0,
                min(1.0, _to_float(pg_data.get("target_net_margin"), pg_base.target_net_margin)),
            ),
            min_sales_7d=max(0.0, _to_float(pg_data.get("min_sales_7d"), pg_base.min_sales_7d)),
            min_ad_spend_roll=max(0.0, _to_float(pg_data.get("min_ad_spend_roll"), pg_base.min_ad_spend_roll)),
        )

        # 机会池 -> 预算迁移（可配置；没有则用默认）
        bto_base = base.dashboard_budget_transfer_opportunity
        bto_data = dash.get("budget_transfer_opportunity") if isinstance(dash.get("budget_transfer_opportunity"), dict) else {}
        budget_transfer_opportunity = BudgetTransferOpportunityPolicy(
            enabled=_to_bool(bto_data.get("enabled"), bto_base.enabled),
            suggested_add_pct=max(0.0, _to_float(bto_data.get("suggested_add_pct"), bto_base.suggested_add_pct)),
            max_target_campaigns=max(1, _to_int(bto_data.get("max_target_campaigns"), bto_base.max_target_campaigns)),
            min_target_opp_spend=max(0.0, _to_float(bto_data.get("min_target_opp_spend"), bto_base.min_target_opp_spend)),
            prefer_same_ad_type=_to_bool(bto_data.get("prefer_same_ad_type"), bto_base.prefer_same_ad_type),
        )

        # 放量解锁任务收敛策略（可配置；没有则用默认）
        ut_base = base.dashboard_unlock_tasks
        ut_data = dash.get("unlock_tasks") if isinstance(dash.get("unlock_tasks"), dict) else {}
        include_pr = list(ut_base.include_priorities)
        if isinstance(ut_data.get("include_priorities"), list):
            tmp: List[str] = []
            for x in ut_data.get("include_priorities"):
                s = str(x or "").strip().upper()
                if s:
                    tmp.append(s)
            if tmp:
                include_pr = tmp
        unlock_tasks_policy = UnlockTasksPolicy(
            top_n=max(1, _to_int(ut_data.get("top_n"), ut_base.top_n)),
            include_priorities=include_pr,
            dashboard_top_n=max(1, _to_int(ut_data.get("dashboard_top_n"), ut_base.dashboard_top_n)),
            dashboard_prefer_unique_asin=_to_bool(
                ut_data.get("dashboard_prefer_unique_asin"),
                ut_base.dashboard_prefer_unique_asin,
            ),
        )

        # Shop Alerts（规则化告警）阈值（可配置；没有则用默认）
        sa_base = base.dashboard_shop_alerts
        sa_data = dash.get("shop_alerts") if isinstance(dash.get("shop_alerts"), dict) else {}
        pdr_base = sa_base.phase_down_recent
        pdr_data = sa_data.get("phase_down_recent") if isinstance(sa_data.get("phase_down_recent"), dict) else {}
        p0_spend_sum = max(0.0, _to_float(pdr_data.get("p0_spend_sum"), pdr_base.p0_spend_sum))
        p0_spend_share = _to_float(pdr_data.get("p0_spend_share"), pdr_base.p0_spend_share)
        if p0_spend_share < 0:
            p0_spend_share = 0.0
        if p0_spend_share > 1:
            p0_spend_share = 1.0
        p0_spend_sum_min_when_share = max(
            0.0,
            _to_float(pdr_data.get("p0_spend_sum_min_when_share"), pdr_base.p0_spend_sum_min_when_share),
        )
        p0_asin_count_min = max(1, _to_int(pdr_data.get("p0_asin_count_min"), pdr_base.p0_asin_count_min))
        phase_down_recent_policy = PhaseDownRecentAlertPolicy(
            enabled=_to_bool(pdr_data.get("enabled"), pdr_base.enabled),
            p0_spend_sum=p0_spend_sum,
            p0_spend_share=p0_spend_share,
            p0_spend_sum_min_when_share=p0_spend_sum_min_when_share,
            p0_asin_count_min=p0_asin_count_min,
        )
        shop_alerts_policy = ShopAlertsPolicy(phase_down_recent=phase_down_recent_policy)

        # 关键词主题（n-gram）（可配置；没有则用默认）
        kt_base = base.dashboard_keyword_topics
        kt_data = dash.get("keyword_topics") if isinstance(dash.get("keyword_topics"), dict) else {}
        n_values = list(kt_base.n_values)
        if isinstance(kt_data.get("n_values"), list):
            tmp: List[int] = []
            for x in kt_data.get("n_values"):
                n = _to_int(x, 0)
                if 1 <= n <= 5:
                    tmp.append(int(n))
            tmp2 = sorted({int(x) for x in tmp if int(x) > 0})
            if tmp2:
                n_values = tmp2
        md_min_n = _to_int(kt_data.get("md_min_n"), kt_base.md_min_n)
        if md_min_n < 1:
            md_min_n = 1
        if md_min_n > 5:
            md_min_n = 5
        keyword_topics_policy = KeywordTopicsPolicy(
            enabled=_to_bool(kt_data.get("enabled"), kt_base.enabled),
            n_values=n_values,
            min_term_spend=max(0.0, _to_float(kt_data.get("min_term_spend"), kt_base.min_term_spend)),
            max_terms=max(1, _to_int(kt_data.get("max_terms"), kt_base.max_terms)),
            max_rows=max(1, _to_int(kt_data.get("max_rows"), kt_base.max_rows)),
            top_terms_per_ngram=max(
                1,
                _to_int(kt_data.get("top_terms_per_ngram"), kt_base.top_terms_per_ngram),
            ),
            md_top_n=max(1, _to_int(kt_data.get("md_top_n"), kt_base.md_top_n)),
            md_min_n=md_min_n,
            action_hints_enabled=_to_bool(kt_data.get("action_hints_enabled"), kt_base.action_hints_enabled),
            action_hints_top_waste=max(
                0,
                _to_int(kt_data.get("action_hints_top_waste"), kt_base.action_hints_top_waste),
            ),
            action_hints_top_scale=max(
                0,
                _to_int(kt_data.get("action_hints_top_scale"), kt_base.action_hints_top_scale),
            ),
            action_hints_min_waste_spend=max(
                0.0,
                _to_float(kt_data.get("action_hints_min_waste_spend"), kt_base.action_hints_min_waste_spend),
            ),
            action_hints_min_waste_ratio=max(
                0.0,
                min(
                    1.0,
                    _to_float(kt_data.get("action_hints_min_waste_ratio"), kt_base.action_hints_min_waste_ratio),
                ),
            ),
            action_hints_scale_acos_multiplier=max(
                0.0,
                _to_float(kt_data.get("action_hints_scale_acos_multiplier"), kt_base.action_hints_scale_acos_multiplier),
            ),
            action_hints_min_sales=max(
                0.0,
                _to_float(kt_data.get("action_hints_min_sales"), kt_base.action_hints_min_sales),
            ),
            action_hints_top_entities=max(
                1,
                _to_int(kt_data.get("action_hints_top_entities"), kt_base.action_hints_top_entities),
            ),
            asin_context_enabled=_to_bool(kt_data.get("asin_context_enabled"), kt_base.asin_context_enabled),
            asin_context_min_confidence=max(
                0.0,
                min(
                    1.0,
                    _to_float(kt_data.get("asin_context_min_confidence"), kt_base.asin_context_min_confidence),
                ),
            ),
            asin_context_top_asins_per_topic=max(
                1,
                _to_int(kt_data.get("asin_context_top_asins_per_topic"), kt_base.asin_context_top_asins_per_topic),
            ),
        )

        return OpsPolicy(
            low_inventory_threshold=low_thr,
            block_scale_when_low_inventory=block_scale,
            block_scale_when_cover_days_below=cover_below,
            keyword_funnel_top_n=top_n,
            campaign_windows_days=windows2,
            campaign_min_spend=min_spend,
            campaign_top_asins_per_campaign=top_asins,
            phase_acos_multiplier=mult2,
            dashboard_top_asins=dash_top_asins,
            dashboard_top_actions=dash_top_actions,
            dashboard_compare_ignore_last_days=dash_ignore_last_days,
            dashboard_scale_window=scale_window,
            dashboard_focus_scoring=fs,
            dashboard_signal_scoring=signal_scoring,
            dashboard_stage_scoring=stage_scoring,
            dashboard_action_scoring=action_scoring,
            dashboard_inventory_sigmoid=inventory_sigmoid,
            dashboard_profit_guard=profit_guard,
            dashboard_budget_transfer_opportunity=budget_transfer_opportunity,
            dashboard_unlock_tasks=unlock_tasks_policy,
            dashboard_shop_alerts=shop_alerts_policy,
            dashboard_keyword_topics=keyword_topics_policy,
        )
    except Exception:
        return base


def validate_ops_policy_path(path: Path) -> List[str]:
    """
    校验 `ops_policy.json`（只做提醒，不阻断跑数）。

    目标：
    - 降低调参误配的成本：字段名写错/类型写错/明显越界时给出提示；
    - 不引入新依赖（先不使用 Pydantic）。
    """
    try:
        if path is None or (not Path(path).exists()):
            return [f"未找到 `{path}`，将使用默认策略（可忽略）。"]
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return [f"`{path}` 不是 JSON object（应为 {{...}}），将使用默认策略。"]
        return validate_ops_policy_dict(data)
    except Exception as e:
        return [f"解析 `{path}` 失败：{type(e).__name__}，将使用默认策略。"]


def validate_ops_policy_dict(data: Dict[str, object]) -> List[str]:
    """
    对 ops_policy 的 dict 做轻量校验（只提示，不阻断）。

    说明：
    - “未知字段”意味着：你可能拼写写错了；这些字段会被忽略；
    - “类型不匹配”意味着：该字段无法被解析为期望类型，会回退默认值；
    - “越界”意味着：该字段超出建议范围，运行时会被截断到合理范围。
    """
    warnings: List[str] = []
    base = OpsPolicy()

    def join(prefix: str, key: str) -> str:
        return f"{prefix}.{key}" if prefix else key

    def warn_unknown_keys(obj: Dict[str, object], allowed: set[str], prefix: str) -> None:
        try:
            for k in obj.keys():
                kk = str(k)
                if kk.startswith("_"):
                    continue
                if kk not in allowed:
                    warnings.append(f"未知字段：`{join(prefix, kk)}`（将被忽略）")
        except Exception:
            return

    def parse_int(v: object) -> Optional[int]:
        try:
            return int(float(v))
        except Exception:
            return None

    def parse_float(v: object) -> Optional[float]:
        try:
            return float(v)
        except Exception:
            return None

    def clamp_float(x: float, min_v: Optional[float], max_v: Optional[float]) -> float:
        y = float(x)
        if min_v is not None and y < float(min_v):
            y = float(min_v)
        if max_v is not None and y > float(max_v):
            y = float(max_v)
        return y

    def clamp_int(x: int, min_v: Optional[int], max_v: Optional[int]) -> int:
        y = int(x)
        if min_v is not None and y < int(min_v):
            y = int(min_v)
        if max_v is not None and y > int(max_v):
            y = int(max_v)
        return y

    def check_int(obj: Dict[str, object], key: str, default: int, prefix: str, *, min_v: Optional[int] = None, max_v: Optional[int] = None) -> None:
        if key not in obj:
            return
        v = obj.get(key)
        x = parse_int(v)
        if x is None:
            warnings.append(f"类型不匹配：`{join(prefix, key)}` 期望整数，但得到 {v!r}；将使用默认值 {default}")
            return
        y = clamp_int(int(x), min_v, max_v)
        if y != int(x):
            warnings.append(f"越界：`{join(prefix, key)}`={x} 超出范围 [{min_v},{max_v}]；将被截断为 {y}")

    def check_float(obj: Dict[str, object], key: str, default: float, prefix: str, *, min_v: Optional[float] = None, max_v: Optional[float] = None) -> None:
        if key not in obj:
            return
        v = obj.get(key)
        x = parse_float(v)
        if x is None:
            warnings.append(f"类型不匹配：`{join(prefix, key)}` 期望数字，但得到 {v!r}；将使用默认值 {default}")
            return
        y = clamp_float(float(x), min_v, max_v)
        if y != float(x):
            warnings.append(f"越界：`{join(prefix, key)}`={x} 超出范围 [{min_v},{max_v}]；将被截断为 {y}")

    def check_bool(obj: Dict[str, object], key: str, default: bool, prefix: str) -> None:
        if key not in obj:
            return
        v = obj.get(key)
        if isinstance(v, bool):
            return
        s = str(v).strip().lower()
        if s in {"1", "0", "true", "false", "yes", "no", "y", "n", "on", "off"}:
            return
        warnings.append(f"类型不匹配：`{join(prefix, key)}` 期望布尔值 true/false，但得到 {v!r}；将使用默认值 {default}")

    # ===== 顶层 =====
    if not isinstance(data, dict):
        return ["ops_policy 不是 dict（应为 JSON object）。"]
    warn_unknown_keys(data, {"inventory", "keyword_funnel", "campaign_ops", "dashboard"}, "")

    # ===== inventory =====
    inv = data.get("inventory")
    if inv is not None and not isinstance(inv, dict):
        warnings.append("类型不匹配：`inventory` 期望为 object/dict；将使用默认策略。")
        inv = {}
    inv = inv if isinstance(inv, dict) else {}
    warn_unknown_keys(inv, {"low_inventory_threshold", "block_scale_when_low_inventory", "block_scale_when_cover_days_below"}, "inventory")
    check_int(inv, "low_inventory_threshold", base.low_inventory_threshold, "inventory", min_v=0)
    check_bool(inv, "block_scale_when_low_inventory", base.block_scale_when_low_inventory, "inventory")
    check_float(inv, "block_scale_when_cover_days_below", base.block_scale_when_cover_days_below, "inventory", min_v=0.0)

    # ===== keyword_funnel =====
    kw = data.get("keyword_funnel")
    if kw is not None and not isinstance(kw, dict):
        warnings.append("类型不匹配：`keyword_funnel` 期望为 object/dict；将使用默认策略。")
        kw = {}
    kw = kw if isinstance(kw, dict) else {}
    warn_unknown_keys(kw, {"top_n"}, "keyword_funnel")
    check_int(kw, "top_n", base.keyword_funnel_top_n, "keyword_funnel", min_v=1)

    # ===== campaign_ops =====
    cop = data.get("campaign_ops")
    if cop is not None and not isinstance(cop, dict):
        warnings.append("类型不匹配：`campaign_ops` 期望为 object/dict；将使用默认策略。")
        cop = {}
    cop = cop if isinstance(cop, dict) else {}
    warn_unknown_keys(cop, {"windows_days", "min_spend", "top_asins_per_campaign", "phase_acos_multiplier"}, "campaign_ops")
    if "windows_days" in cop and not isinstance(cop.get("windows_days"), list):
        warnings.append("类型不匹配：`campaign_ops.windows_days` 期望为数组，例如 [7,14,30]。")
    check_float(cop, "min_spend", base.campaign_min_spend, "campaign_ops", min_v=0.0)
    check_int(cop, "top_asins_per_campaign", base.campaign_top_asins_per_campaign, "campaign_ops", min_v=1)
    if "phase_acos_multiplier" in cop and not isinstance(cop.get("phase_acos_multiplier"), dict):
        warnings.append("类型不匹配：`campaign_ops.phase_acos_multiplier` 期望为 object/dict。")

    # ===== dashboard =====
    dash = data.get("dashboard")
    if dash is not None and not isinstance(dash, dict):
        warnings.append("类型不匹配：`dashboard` 期望为 object/dict；将使用默认策略。")
        dash = {}
    dash = dash if isinstance(dash, dict) else {}
    warn_unknown_keys(
        dash,
        {
            "top_asins",
            "top_actions",
            "compare_ignore_last_days",
            "scale_window",
            "focus_scoring",
            "signal_scoring",
            "stage_scoring",
            "action_scoring",
            "inventory_sigmoid",
            "profit_guard",
            "budget_transfer_opportunity",
            "unlock_tasks",
            "shop_alerts",
            "keyword_topics",
        },
        "dashboard",
    )
    check_int(dash, "top_asins", base.dashboard_top_asins, "dashboard", min_v=1)
    check_int(dash, "top_actions", base.dashboard_top_actions, "dashboard", min_v=1)
    check_int(dash, "compare_ignore_last_days", base.dashboard_compare_ignore_last_days, "dashboard", min_v=0, max_v=14)

    # ===== dashboard.scale_window =====
    sw = dash.get("scale_window")
    if sw is not None and not isinstance(sw, dict):
        warnings.append("类型不匹配：`dashboard.scale_window` 期望为 object/dict；将使用默认策略。")
        sw = {}
    sw = sw if isinstance(sw, dict) else {}
    warn_unknown_keys(sw, {f.name for f in fields(ScaleWindowPolicy)}, "dashboard.scale_window")
    check_float(sw, "min_sales_per_day_7d", base.dashboard_scale_window.min_sales_per_day_7d, "dashboard.scale_window", min_v=0.0)
    check_float(sw, "min_delta_sales", base.dashboard_scale_window.min_delta_sales, "dashboard.scale_window", min_v=0.0)
    check_float(
        sw,
        "min_inventory_cover_days_7d",
        base.dashboard_scale_window.min_inventory_cover_days_7d,
        "dashboard.scale_window",
        min_v=0.0,
    )
    check_float(sw, "max_tacos_roll", base.dashboard_scale_window.max_tacos_roll, "dashboard.scale_window", min_v=0.0)
    check_float(sw, "max_marginal_tacos", base.dashboard_scale_window.max_marginal_tacos, "dashboard.scale_window", min_v=0.0)

    # ===== dashboard.focus_scoring =====
    fs = dash.get("focus_scoring")
    if fs is not None and not isinstance(fs, dict):
        warnings.append("类型不匹配：`dashboard.focus_scoring` 期望为 object/dict；将使用默认策略。")
        fs = {}
    fs = fs if isinstance(fs, dict) else {}
    warn_unknown_keys(fs, {f.name for f in fields(FocusScoringPolicy)}, "dashboard.focus_scoring")
    check_float(fs, "base_spend_log_multiplier", base.dashboard_focus_scoring.base_spend_log_multiplier, "dashboard.focus_scoring", min_v=0.0)
    check_float(fs, "base_spend_score_cap", base.dashboard_focus_scoring.base_spend_score_cap, "dashboard.focus_scoring", min_v=0.0)
    check_float(fs, "high_ad_dependency_threshold", base.dashboard_focus_scoring.high_ad_dependency_threshold, "dashboard.focus_scoring", min_v=0.0, max_v=1.0)
    check_float(fs, "cvr_signal_min_cvr_drop", base.dashboard_focus_scoring.cvr_signal_min_cvr_drop, "dashboard.focus_scoring", min_v=0.0, max_v=1.0)
    check_float(fs, "organic_signal_drop_ratio", base.dashboard_focus_scoring.organic_signal_drop_ratio, "dashboard.focus_scoring", min_v=0.0, max_v=1.0)
    check_float(fs, "aov_signal_drop_ratio", base.dashboard_focus_scoring.aov_signal_drop_ratio, "dashboard.focus_scoring", min_v=0.0, max_v=1.0)
    check_float(
        fs,
        "gross_margin_signal_low_threshold",
        base.dashboard_focus_scoring.gross_margin_signal_low_threshold,
        "dashboard.focus_scoring",
        min_v=-1.0,
        max_v=1.0,
    )

    # ===== dashboard.signal_scoring =====
    sigs = dash.get("signal_scoring")
    if sigs is not None and not isinstance(sigs, dict):
        warnings.append("类型不匹配：`dashboard.signal_scoring` 期望为 object/dict；将使用默认策略。")
        sigs = {}
    sigs = sigs if isinstance(sigs, dict) else {}
    warn_unknown_keys(sigs, {f.name for f in fields(SignalScoringPolicy)}, "dashboard.signal_scoring")
    check_float(
        sigs,
        "product_sales_weight",
        base.dashboard_signal_scoring.product_sales_weight,
        "dashboard.signal_scoring",
        min_v=0.0,
    )
    check_float(
        sigs,
        "product_sessions_weight",
        base.dashboard_signal_scoring.product_sessions_weight,
        "dashboard.signal_scoring",
        min_v=0.0,
    )
    check_float(
        sigs,
        "product_organic_sales_weight",
        base.dashboard_signal_scoring.product_organic_sales_weight,
        "dashboard.signal_scoring",
        min_v=0.0,
    )
    check_float(
        sigs,
        "product_profit_weight",
        base.dashboard_signal_scoring.product_profit_weight,
        "dashboard.signal_scoring",
        min_v=0.0,
    )
    check_float(
        sigs,
        "product_steepness",
        base.dashboard_signal_scoring.product_steepness,
        "dashboard.signal_scoring",
        min_v=0.0,
    )
    check_float(
        sigs,
        "ad_acos_weight",
        base.dashboard_signal_scoring.ad_acos_weight,
        "dashboard.signal_scoring",
        min_v=0.0,
    )
    check_float(
        sigs,
        "ad_cvr_weight",
        base.dashboard_signal_scoring.ad_cvr_weight,
        "dashboard.signal_scoring",
        min_v=0.0,
    )
    check_float(
        sigs,
        "ad_spend_up_no_sales_weight",
        base.dashboard_signal_scoring.ad_spend_up_no_sales_weight,
        "dashboard.signal_scoring",
        min_v=0.0,
    )
    check_float(
        sigs,
        "ad_steepness",
        base.dashboard_signal_scoring.ad_steepness,
        "dashboard.signal_scoring",
        min_v=0.0,
    )

    # ===== dashboard.stage_scoring =====
    ss = dash.get("stage_scoring")
    if ss is not None and not isinstance(ss, dict):
        warnings.append("类型不匹配：`dashboard.stage_scoring` 期望为 object/dict；将使用默认策略。")
        ss = {}
    ss = ss if isinstance(ss, dict) else {}
    warn_unknown_keys(ss, {f.name for f in fields(StageScoringPolicy)}, "dashboard.stage_scoring")
    check_int(ss, "median_min_samples", base.dashboard_stage_scoring.median_min_samples, "dashboard.stage_scoring", min_v=1)
    check_float(ss, "min_impressions_7d", base.dashboard_stage_scoring.min_impressions_7d, "dashboard.stage_scoring", min_v=0.0)
    check_float(ss, "min_clicks_7d", base.dashboard_stage_scoring.min_clicks_7d, "dashboard.stage_scoring", min_v=0.0)
    check_float(ss, "min_orders_7d", base.dashboard_stage_scoring.min_orders_7d, "dashboard.stage_scoring", min_v=0.0)
    check_float(ss, "min_sales_7d", base.dashboard_stage_scoring.min_sales_7d, "dashboard.stage_scoring", min_v=0.0)
    check_float(ss, "min_orders_7d_mature", base.dashboard_stage_scoring.min_orders_7d_mature, "dashboard.stage_scoring", min_v=0.0)
    check_float(ss, "new_ctr_low_ratio", base.dashboard_stage_scoring.new_ctr_low_ratio, "dashboard.stage_scoring", min_v=0.0)
    check_float(ss, "new_cvr_low_ratio", base.dashboard_stage_scoring.new_cvr_low_ratio, "dashboard.stage_scoring", min_v=0.0)
    check_float(ss, "new_cpa_high_ratio", base.dashboard_stage_scoring.new_cpa_high_ratio, "dashboard.stage_scoring", min_v=0.0)
    check_float(ss, "mature_cpa_high_ratio", base.dashboard_stage_scoring.mature_cpa_high_ratio, "dashboard.stage_scoring", min_v=0.0)
    check_float(ss, "mature_acos_high_ratio", base.dashboard_stage_scoring.mature_acos_high_ratio, "dashboard.stage_scoring", min_v=0.0)
    check_float(ss, "mature_ad_share_shift_abs", base.dashboard_stage_scoring.mature_ad_share_shift_abs, "dashboard.stage_scoring", min_v=0.0, max_v=1.0)
    check_float(ss, "mature_spend_shift_ratio", base.dashboard_stage_scoring.mature_spend_shift_ratio, "dashboard.stage_scoring", min_v=0.0)
    check_int(ss, "max_stage_tags", base.dashboard_stage_scoring.max_stage_tags, "dashboard.stage_scoring", min_v=1, max_v=10)

    # ===== dashboard.inventory_sigmoid =====
    sig = dash.get("inventory_sigmoid")
    if sig is not None and not isinstance(sig, dict):
        warnings.append("类型不匹配：`dashboard.inventory_sigmoid` 期望为 object/dict；将使用默认策略。")
        sig = {}
    sig = sig if isinstance(sig, dict) else {}
    warn_unknown_keys(sig, {f.name for f in fields(InventorySigmoidPolicy)}, "dashboard.inventory_sigmoid")
    check_bool(sig, "enabled", base.dashboard_inventory_sigmoid.enabled, "dashboard.inventory_sigmoid")
    check_float(sig, "optimal_cover_days", base.dashboard_inventory_sigmoid.optimal_cover_days, "dashboard.inventory_sigmoid", min_v=0.0)
    check_float(sig, "steepness", base.dashboard_inventory_sigmoid.steepness, "dashboard.inventory_sigmoid", min_v=0.0)
    check_float(sig, "min_modifier", base.dashboard_inventory_sigmoid.min_modifier, "dashboard.inventory_sigmoid", min_v=0.0)
    check_float(sig, "max_modifier", base.dashboard_inventory_sigmoid.max_modifier, "dashboard.inventory_sigmoid", min_v=0.0)
    check_float(sig, "min_change_ratio", base.dashboard_inventory_sigmoid.min_change_ratio, "dashboard.inventory_sigmoid", min_v=0.0)
    check_float(sig, "min_ad_spend_roll", base.dashboard_inventory_sigmoid.min_ad_spend_roll, "dashboard.inventory_sigmoid", min_v=0.0)

    # ===== dashboard.profit_guard =====
    pg = dash.get("profit_guard")
    if pg is not None and not isinstance(pg, dict):
        warnings.append("类型不匹配：`dashboard.profit_guard` 期望为 object/dict；将使用默认策略。")
        pg = {}
    pg = pg if isinstance(pg, dict) else {}
    warn_unknown_keys(pg, {f.name for f in fields(ProfitGuardPolicy)}, "dashboard.profit_guard")
    check_bool(pg, "enabled", base.dashboard_profit_guard.enabled, "dashboard.profit_guard")
    check_float(pg, "target_net_margin", base.dashboard_profit_guard.target_net_margin, "dashboard.profit_guard", min_v=-1.0, max_v=1.0)
    check_float(pg, "min_sales_7d", base.dashboard_profit_guard.min_sales_7d, "dashboard.profit_guard", min_v=0.0)
    check_float(pg, "min_ad_spend_roll", base.dashboard_profit_guard.min_ad_spend_roll, "dashboard.profit_guard", min_v=0.0)

    # ===== dashboard.action_scoring =====
    ac = dash.get("action_scoring")
    if ac is not None and not isinstance(ac, dict):
        warnings.append("类型不匹配：`dashboard.action_scoring` 期望为 object/dict；将使用默认策略。")
        ac = {}
    ac = ac if isinstance(ac, dict) else {}
    warn_unknown_keys(ac, {f.name for f in fields(ActionScoringPolicy)}, "dashboard.action_scoring")
    check_float(
        ac,
        "low_hint_confidence_threshold",
        base.dashboard_action_scoring.low_hint_confidence_threshold,
        "dashboard.action_scoring",
        min_v=0.0,
        max_v=1.0,
    )

    # ===== dashboard.budget_transfer_opportunity =====
    bto = dash.get("budget_transfer_opportunity")
    if bto is not None and not isinstance(bto, dict):
        warnings.append("类型不匹配：`dashboard.budget_transfer_opportunity` 期望为 object/dict；将使用默认策略。")
        bto = {}
    bto = bto if isinstance(bto, dict) else {}
    warn_unknown_keys(bto, {f.name for f in fields(BudgetTransferOpportunityPolicy)}, "dashboard.budget_transfer_opportunity")
    check_float(bto, "suggested_add_pct", base.dashboard_budget_transfer_opportunity.suggested_add_pct, "dashboard.budget_transfer_opportunity", min_v=0.0)
    check_int(bto, "max_target_campaigns", base.dashboard_budget_transfer_opportunity.max_target_campaigns, "dashboard.budget_transfer_opportunity", min_v=1)
    check_float(bto, "min_target_opp_spend", base.dashboard_budget_transfer_opportunity.min_target_opp_spend, "dashboard.budget_transfer_opportunity", min_v=0.0)

    # ===== dashboard.unlock_tasks =====
    ut = dash.get("unlock_tasks")
    if ut is not None and not isinstance(ut, dict):
        warnings.append("类型不匹配：`dashboard.unlock_tasks` 期望为 object/dict；将使用默认策略。")
        ut = {}
    ut = ut if isinstance(ut, dict) else {}
    warn_unknown_keys(ut, {f.name for f in fields(UnlockTasksPolicy)}, "dashboard.unlock_tasks")
    check_int(ut, "top_n", base.dashboard_unlock_tasks.top_n, "dashboard.unlock_tasks", min_v=1)
    check_int(ut, "dashboard_top_n", base.dashboard_unlock_tasks.dashboard_top_n, "dashboard.unlock_tasks", min_v=1)

    # ===== dashboard.shop_alerts =====
    sa = dash.get("shop_alerts")
    if sa is not None and not isinstance(sa, dict):
        warnings.append("类型不匹配：`dashboard.shop_alerts` 期望为 object/dict；将使用默认策略。")
        sa = {}
    sa = sa if isinstance(sa, dict) else {}
    warn_unknown_keys(sa, {"phase_down_recent"}, "dashboard.shop_alerts")
    pdr = sa.get("phase_down_recent")
    if pdr is not None and not isinstance(pdr, dict):
        warnings.append("类型不匹配：`dashboard.shop_alerts.phase_down_recent` 期望为 object/dict；将使用默认策略。")
        pdr = {}
    pdr = pdr if isinstance(pdr, dict) else {}
    warn_unknown_keys(pdr, {f.name for f in fields(PhaseDownRecentAlertPolicy)}, "dashboard.shop_alerts.phase_down_recent")
    check_float(
        pdr,
        "p0_spend_share",
        base.dashboard_shop_alerts.phase_down_recent.p0_spend_share,
        "dashboard.shop_alerts.phase_down_recent",
        min_v=0.0,
        max_v=1.0,
    )
    check_int(
        pdr,
        "p0_asin_count_min",
        base.dashboard_shop_alerts.phase_down_recent.p0_asin_count_min,
        "dashboard.shop_alerts.phase_down_recent",
        min_v=1,
    )

    # ===== dashboard.keyword_topics =====
    kt = dash.get("keyword_topics")
    if kt is not None and not isinstance(kt, dict):
        warnings.append("类型不匹配：`dashboard.keyword_topics` 期望为 object/dict；将使用默认策略。")
        kt = {}
    kt = kt if isinstance(kt, dict) else {}
    warn_unknown_keys(kt, {f.name for f in fields(KeywordTopicsPolicy)}, "dashboard.keyword_topics")
    if "n_values" in kt and not isinstance(kt.get("n_values"), list):
        warnings.append("类型不匹配：`dashboard.keyword_topics.n_values` 期望为数组，例如 [1,2,3]。")
    check_int(kt, "max_terms", base.dashboard_keyword_topics.max_terms, "dashboard.keyword_topics", min_v=1)
    check_int(kt, "max_rows", base.dashboard_keyword_topics.max_rows, "dashboard.keyword_topics", min_v=1)
    check_int(kt, "md_top_n", base.dashboard_keyword_topics.md_top_n, "dashboard.keyword_topics", min_v=1)
    check_int(kt, "md_min_n", base.dashboard_keyword_topics.md_min_n, "dashboard.keyword_topics", min_v=1, max_v=5)
    check_float(kt, "min_term_spend", base.dashboard_keyword_topics.min_term_spend, "dashboard.keyword_topics", min_v=0.0)
    check_float(kt, "action_hints_min_waste_ratio", base.dashboard_keyword_topics.action_hints_min_waste_ratio, "dashboard.keyword_topics", min_v=0.0, max_v=1.0)
    check_float(kt, "asin_context_min_confidence", base.dashboard_keyword_topics.asin_context_min_confidence, "dashboard.keyword_topics", min_v=0.0, max_v=1.0)

    # 去重，避免刷屏
    try:
        uniq: List[str] = []
        seen: set[str] = set()
        for w in warnings:
            if w not in seen:
                uniq.append(w)
                seen.add(w)
        return uniq
    except Exception:
        return warnings


def ops_policy_effective_to_dict(policy: OpsPolicy) -> Dict[str, object]:
    """
    把 “OpsPolicy（dataclass）” 转成与 `ops_policy.json` 结构一致的 dict（用于写出“实际生效值”）。

    说明：
    - raw 配置（`config/ops_policy.json`）可能缺字段/有非法值；`load_ops_policy()` 会做默认兜底与截断；
    - 该函数输出的是 **本次运行实际生效** 的参数，便于你对比/复盘不同 runs 的调参效果。
    """
    try:
        if policy is None:
            return {}
        return {
            "inventory": {
                "low_inventory_threshold": int(policy.low_inventory_threshold),
                "block_scale_when_low_inventory": bool(policy.block_scale_when_low_inventory),
                "block_scale_when_cover_days_below": float(policy.block_scale_when_cover_days_below),
            },
            "keyword_funnel": {
                "top_n": int(policy.keyword_funnel_top_n),
            },
            "campaign_ops": {
                "windows_days": list(policy.campaign_windows_days or []),
                "min_spend": float(policy.campaign_min_spend),
                "top_asins_per_campaign": int(policy.campaign_top_asins_per_campaign),
                "phase_acos_multiplier": dict(policy.phase_acos_multiplier or {}),
            },
            "dashboard": {
                "top_asins": int(policy.dashboard_top_asins),
                "top_actions": int(policy.dashboard_top_actions),
                "compare_ignore_last_days": int(policy.dashboard_compare_ignore_last_days),
                "scale_window": asdict(policy.dashboard_scale_window),
                "focus_scoring": asdict(policy.dashboard_focus_scoring),
                "signal_scoring": asdict(policy.dashboard_signal_scoring),
                "stage_scoring": asdict(policy.dashboard_stage_scoring),
                "action_scoring": asdict(policy.dashboard_action_scoring),
                "inventory_sigmoid": asdict(policy.dashboard_inventory_sigmoid),
                "profit_guard": asdict(policy.dashboard_profit_guard),
                "budget_transfer_opportunity": asdict(policy.dashboard_budget_transfer_opportunity),
                "unlock_tasks": asdict(policy.dashboard_unlock_tasks),
                "shop_alerts": {
                    "phase_down_recent": asdict(policy.dashboard_shop_alerts.phase_down_recent),
                },
                "keyword_topics": asdict(policy.dashboard_keyword_topics),
            },
        }
    except Exception:
        return {}


@dataclass(frozen=True)
class OpsProfile:
    """
    运营“阶段/档位”总选项（少数几个开关，帮你快速切换整体风格）。

    使用方式：
    - 日常只改 `config/ops_profile.json`（这里字段很少）
    - 需要更细调时，再去改 `config/ops_policy.json` 的具体参数
    """

    enabled: bool = True
    # 总体风格：更保守/更积极/平衡
    preset: str = "balanced"  # guardrail|balanced|growth
    # 输出密度：更少/默认/更多（只影响 TopN 与展示规模）
    density: str = "normal"  # compact|normal|deep
    # 关键词主题（n-gram）深度：关闭/默认/深挖
    keyword_topics: str = "standard"  # off|standard|deep


def load_ops_profile(path: Path) -> Tuple[OpsProfile, List[str]]:
    """
    读取 `ops_profile.json`（失败不崩：返回默认）。
    """
    warns: List[str] = []
    default = OpsProfile(enabled=False)  # 文件不存在时默认不启用 profile 覆盖
    try:
        if path is None or (not Path(path).exists()):
            return default, warns
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            warns.append(f"`{path}` 不是 JSON object（应为 {{...}}），将忽略 ops_profile。")
            return default, warns

        enabled = bool(data.get("enabled")) if "enabled" in data else True

        preset = str(data.get("preset") or "balanced").strip().lower()
        density = str(data.get("density") or "normal").strip().lower()
        keyword_topics = str(data.get("keyword_topics") or "standard").strip().lower()

        if preset not in {"guardrail", "balanced", "growth"}:
            warns.append(f"ops_profile.preset={preset!r} 不在允许范围（guardrail/balanced/growth），将回退为 'balanced'。")
            preset = "balanced"
        if density not in {"compact", "normal", "deep"}:
            warns.append(f"ops_profile.density={density!r} 不在允许范围（compact/normal/deep），将回退为 'normal'。")
            density = "normal"
        if keyword_topics not in {"off", "standard", "deep"}:
            warns.append(f"ops_profile.keyword_topics={keyword_topics!r} 不在允许范围（off/standard/deep），将回退为 'standard'。")
            keyword_topics = "standard"

        return OpsProfile(enabled=enabled, preset=preset, density=density, keyword_topics=keyword_topics), warns
    except Exception as e:
        warns.append(f"解析 `{path}` 失败：{type(e).__name__}，将忽略 ops_profile。")
        return default, warns


def ops_profile_to_overrides(profile: OpsProfile) -> Dict[str, object]:
    """
    把“总选项”映射为 `ops_policy.json` 的局部覆盖 dict。

    说明：
    - 仅覆盖少量“最影响抓重点/放量判断/输出规模”的参数；
    - 其它细节参数仍以 `config/ops_policy.json` 为准（便于你按业务继续微调）。
    """
    try:
        if profile is None or (not bool(getattr(profile, "enabled", False))):
            return {}

        out: Dict[str, object] = {}

        # ===== 1) preset：整体风格（更保守/更积极）=====
        preset = str(getattr(profile, "preset", "") or "").strip().lower()
        if preset == "guardrail":
            out = deep_merge_dict(
                out,
                {
                    "inventory": {
                        "low_inventory_threshold": 30,
                        "block_scale_when_cover_days_below": 21,
                    },
                    "dashboard": {
                        "scale_window": {
                            "min_inventory_cover_days_7d": 45,
                            "max_tacos_roll": 0.22,
                            "max_marginal_tacos": 0.22,
                        },
                        "action_scoring": {
                            "low_hint_confidence_threshold": 0.55,
                            "low_hint_scale_penalty": 30,
                        },
                        "budget_transfer_opportunity": {
                            "suggested_add_pct": 10.0,
                            "min_target_opp_spend": 20.0,
                        },
                    },
                },
            )
        elif preset == "growth":
            out = deep_merge_dict(
                out,
                {
                    "inventory": {
                        "low_inventory_threshold": 15,
                        "block_scale_when_cover_days_below": 7,
                    },
                    "dashboard": {
                        "scale_window": {
                            "min_inventory_cover_days_7d": 21,
                            "max_tacos_roll": 0.30,
                            "max_marginal_tacos": 0.30,
                        },
                        "action_scoring": {
                            "low_hint_confidence_threshold": 0.35,
                            "low_hint_scale_penalty": 10,
                        },
                        "budget_transfer_opportunity": {
                            "suggested_add_pct": 20.0,
                            "min_target_opp_spend": 10.0,
                        },
                    },
                },
            )
        else:
            # balanced：不额外覆盖（以 ops_policy.json 为准）
            pass

        # ===== 2) density：输出规模（TopN）=====
        density = str(getattr(profile, "density", "") or "").strip().lower()
        if density == "compact":
            out = deep_merge_dict(
                out,
                {
                    "keyword_funnel": {"top_n": 8},
                    "dashboard": {
                        "top_asins": 30,
                        "top_actions": 40,
                        "unlock_tasks": {"top_n": 20, "dashboard_top_n": 3},
                        "keyword_topics": {"md_top_n": 3, "max_terms": 3000, "max_rows": 1000},
                    },
                },
            )
        elif density == "deep":
            out = deep_merge_dict(
                out,
                {
                    "keyword_funnel": {"top_n": 20},
                    "dashboard": {
                        "top_asins": 80,
                        "top_actions": 120,
                        "unlock_tasks": {"top_n": 50, "dashboard_top_n": 5},
                        "keyword_topics": {"md_top_n": 8, "max_terms": 10000, "max_rows": 5000},
                    },
                },
            )

        # ===== 3) keyword_topics：关键词主题深度 =====
        kt = str(getattr(profile, "keyword_topics", "") or "").strip().lower()
        if kt == "off":
            out = deep_merge_dict(out, {"dashboard": {"keyword_topics": {"enabled": False}}})
        elif kt == "deep":
            out = deep_merge_dict(
                out,
                {
                    "dashboard": {
                        "keyword_topics": {
                            "enabled": True,
                            "top_terms_per_ngram": 5,
                            "action_hints_top_waste": 30,
                            "action_hints_top_scale": 30,
                        }
                    }
                },
            )

        return out
    except Exception:
        return {}


def deep_merge_dict(base: Dict[str, object], overrides: Dict[str, object]) -> Dict[str, object]:
    """
    递归合并 dict：overrides 覆盖 base。
    """
    try:
        if base is None:
            base = {}
        if overrides is None:
            return base
        for k, v in overrides.items():
            if k in base and isinstance(base.get(k), dict) and isinstance(v, dict):
                base[k] = deep_merge_dict(base.get(k) or {}, v)  # type: ignore[arg-type]
            else:
                base[k] = v
        return base
    except Exception:
        return base


def load_ops_policy_dict(data: Dict[str, object]) -> OpsPolicy:
    """
    从 dict 解析 ops_policy（失败不崩：返回默认）。
    """
    base = OpsPolicy()
    try:
        if not isinstance(data, dict):
            return base

        inv = data.get("inventory") if isinstance(data.get("inventory"), dict) else {}
        kw = data.get("keyword_funnel") if isinstance(data.get("keyword_funnel"), dict) else {}
        cop = data.get("campaign_ops") if isinstance(data.get("campaign_ops"), dict) else {}
        dash = data.get("dashboard") if isinstance(data.get("dashboard"), dict) else {}

        low_thr = max(0, _to_int(inv.get("low_inventory_threshold"), base.low_inventory_threshold))
        block_scale = _to_bool(inv.get("block_scale_when_low_inventory"), base.block_scale_when_low_inventory)
        cover_below = max(
            0.0,
            _to_float(inv.get("block_scale_when_cover_days_below"), base.block_scale_when_cover_days_below),
        )

        top_n = max(1, _to_int(kw.get("top_n"), base.keyword_funnel_top_n))

        windows = cop.get("windows_days")
        if isinstance(windows, list):
            windows2 = []
            for x in windows:
                n = _to_int(x, 0)
                if n > 0:
                    windows2.append(n)
            windows2 = sorted({int(x) for x in windows2})
        else:
            windows2 = list(base.campaign_windows_days)

        min_spend = max(0.0, _to_float(cop.get("min_spend"), base.campaign_min_spend))
        top_asins = max(1, _to_int(cop.get("top_asins_per_campaign"), base.campaign_top_asins_per_campaign))

        mult = cop.get("phase_acos_multiplier")
        mult2 = dict(base.phase_acos_multiplier)
        if isinstance(mult, dict):
            for k, v in mult.items():
                kk = str(k).strip().lower()
                if not kk:
                    continue
                mult2[kk] = _to_float(v, mult2.get(kk, 1.0))

        dash_top_asins = max(1, _to_int(dash.get("top_asins"), base.dashboard_top_asins))
        dash_top_actions = max(1, _to_int(dash.get("top_actions"), base.dashboard_top_actions))

        # “可放量窗口”阈值（可配置；没有则用默认）
        sw_base = base.dashboard_scale_window
        sw_data = dash.get("scale_window") if isinstance(dash.get("scale_window"), dict) else {}
        exclude_phases = list(sw_base.exclude_phases)
        if isinstance(sw_data.get("exclude_phases"), list):
            tmp: List[str] = []
            for x in sw_data.get("exclude_phases"):
                s = str(x or "").strip().lower()
                if s:
                    tmp.append(s)
            if tmp:
                exclude_phases = tmp
        scale_window = ScaleWindowPolicy(
            min_sales_per_day_7d=max(
                0.0,
                _to_float(sw_data.get("min_sales_per_day_7d"), sw_base.min_sales_per_day_7d),
            ),
            min_delta_sales=max(
                0.0,
                _to_float(sw_data.get("min_delta_sales"), sw_base.min_delta_sales),
            ),
            min_inventory_cover_days_7d=max(
                0.0,
                _to_float(
                    sw_data.get("min_inventory_cover_days_7d"),
                    sw_base.min_inventory_cover_days_7d,
                ),
            ),
            max_tacos_roll=max(0.0, _to_float(sw_data.get("max_tacos_roll"), sw_base.max_tacos_roll)),
            max_marginal_tacos=max(0.0, _to_float(sw_data.get("max_marginal_tacos"), sw_base.max_marginal_tacos)),
            exclude_phases=exclude_phases,
            require_no_oos=_to_bool(sw_data.get("require_no_oos"), sw_base.require_no_oos),
            require_no_low_inventory=_to_bool(sw_data.get("require_no_low_inventory"), sw_base.require_no_low_inventory),
            require_oos_with_ad_spend_days_zero=_to_bool(
                sw_data.get("require_oos_with_ad_spend_days_zero"),
                sw_base.require_oos_with_ad_spend_days_zero,
            ),
        )

        # ASIN Focus 评分（可配置；没有则用默认）
        fs_base = base.dashboard_focus_scoring
        fs_data = dash.get("focus_scoring") if isinstance(dash.get("focus_scoring"), dict) else {}
        fs = FocusScoringPolicy(
            base_spend_log_multiplier=max(
                0.0,
                _to_float(fs_data.get("base_spend_log_multiplier"), fs_base.base_spend_log_multiplier),
            ),
            base_spend_score_cap=max(
                0.0,
                _to_float(fs_data.get("base_spend_score_cap"), fs_base.base_spend_score_cap),
            ),
            weight_flag_oos=max(0.0, _to_float(fs_data.get("weight_flag_oos"), fs_base.weight_flag_oos)),
            weight_flag_low_inventory=max(
                0.0,
                _to_float(fs_data.get("weight_flag_low_inventory"), fs_base.weight_flag_low_inventory),
            ),
            weight_oos_with_ad_spend_days=max(
                0.0,
                _to_float(fs_data.get("weight_oos_with_ad_spend_days"), fs_base.weight_oos_with_ad_spend_days),
            ),
            weight_oos_with_sessions_days=max(
                0.0,
                _to_float(fs_data.get("weight_oos_with_sessions_days"), fs_base.weight_oos_with_sessions_days),
            ),
            weight_presale_order_days=max(
                0.0,
                _to_float(fs_data.get("weight_presale_order_days"), fs_base.weight_presale_order_days),
            ),
            weight_spend_up_no_sales=max(
                0.0,
                _to_float(fs_data.get("weight_spend_up_no_sales"), fs_base.weight_spend_up_no_sales),
            ),
            weight_marginal_tacos_worse=max(
                0.0,
                _to_float(fs_data.get("weight_marginal_tacos_worse"), fs_base.weight_marginal_tacos_worse),
            ),
            weight_decline_or_inactive_spend=max(
                0.0,
                _to_float(fs_data.get("weight_decline_or_inactive_spend"), fs_base.weight_decline_or_inactive_spend),
            ),
            weight_phase_down_recent=max(
                0.0,
                _to_float(fs_data.get("weight_phase_down_recent"), fs_base.weight_phase_down_recent),
            ),
            weight_high_ad_dependency=max(
                0.0,
                _to_float(fs_data.get("weight_high_ad_dependency"), fs_base.weight_high_ad_dependency),
            ),
            weight_inventory_zero_still_spend=max(
                0.0,
                _to_float(fs_data.get("weight_inventory_zero_still_spend"), fs_base.weight_inventory_zero_still_spend),
            ),
            high_ad_dependency_threshold=max(
                0.0,
                min(
                    1.0,
                    _to_float(fs_data.get("high_ad_dependency_threshold"), fs_base.high_ad_dependency_threshold),
                ),
            ),
            marginal_tacos_worse_ratio=max(
                0.0,
                _to_float(fs_data.get("marginal_tacos_worse_ratio"), fs_base.marginal_tacos_worse_ratio),
            ),
            weight_sessions_up_cvr_down=max(
                0.0,
                _to_float(fs_data.get("weight_sessions_up_cvr_down"), fs_base.weight_sessions_up_cvr_down),
            ),
            cvr_signal_min_sessions_prev=max(
                0.0,
                _to_float(fs_data.get("cvr_signal_min_sessions_prev"), fs_base.cvr_signal_min_sessions_prev),
            ),
            cvr_signal_min_delta_sessions=max(
                0.0,
                _to_float(fs_data.get("cvr_signal_min_delta_sessions"), fs_base.cvr_signal_min_delta_sessions),
            ),
            cvr_signal_min_cvr_drop=max(
                0.0,
                min(
                    1.0,
                    _to_float(fs_data.get("cvr_signal_min_cvr_drop"), fs_base.cvr_signal_min_cvr_drop),
                ),
            ),
            cvr_signal_min_ad_spend_roll=max(
                0.0,
                _to_float(fs_data.get("cvr_signal_min_ad_spend_roll"), fs_base.cvr_signal_min_ad_spend_roll),
            ),
            weight_organic_down=max(0.0, _to_float(fs_data.get("weight_organic_down"), fs_base.weight_organic_down)),
            organic_signal_min_organic_sales_prev=max(
                0.0,
                _to_float(fs_data.get("organic_signal_min_organic_sales_prev"), fs_base.organic_signal_min_organic_sales_prev),
            ),
            organic_signal_min_delta_organic_sales=max(
                0.0,
                _to_float(fs_data.get("organic_signal_min_delta_organic_sales"), fs_base.organic_signal_min_delta_organic_sales),
            ),
            organic_signal_drop_ratio=max(
                0.0,
                min(
                    1.0,
                    _to_float(fs_data.get("organic_signal_drop_ratio"), fs_base.organic_signal_drop_ratio),
                ),
            ),
            aov_signal_min_orders_prev=max(
                0.0,
                _to_float(fs_data.get("aov_signal_min_orders_prev"), fs_base.aov_signal_min_orders_prev),
            ),
            aov_signal_min_delta_aov=max(
                0.0,
                _to_float(fs_data.get("aov_signal_min_delta_aov"), fs_base.aov_signal_min_delta_aov),
            ),
            aov_signal_drop_ratio=max(
                0.0,
                min(
                    1.0,
                    _to_float(fs_data.get("aov_signal_drop_ratio"), fs_base.aov_signal_drop_ratio),
                ),
            ),
            gross_margin_signal_min_sales=max(
                0.0,
                _to_float(fs_data.get("gross_margin_signal_min_sales"), fs_base.gross_margin_signal_min_sales),
            ),
            gross_margin_signal_low_threshold=max(
                -1.0,
                min(
                    1.0,
                    _to_float(fs_data.get("gross_margin_signal_low_threshold"), fs_base.gross_margin_signal_low_threshold),
                ),
            ),
        )

        # Sigmoid 多维评分（可配置；没有则用默认）
        sigs_base = base.dashboard_signal_scoring
        sigs_data = dash.get("signal_scoring") if isinstance(dash.get("signal_scoring"), dict) else {}
        signal_scoring = SignalScoringPolicy(
            product_sales_weight=max(
                0.0,
                _to_float(sigs_data.get("product_sales_weight"), sigs_base.product_sales_weight),
            ),
            product_sessions_weight=max(
                0.0,
                _to_float(sigs_data.get("product_sessions_weight"), sigs_base.product_sessions_weight),
            ),
            product_organic_sales_weight=max(
                0.0,
                _to_float(sigs_data.get("product_organic_sales_weight"), sigs_base.product_organic_sales_weight),
            ),
            product_profit_weight=max(
                0.0,
                _to_float(sigs_data.get("product_profit_weight"), sigs_base.product_profit_weight),
            ),
            product_steepness=max(
                0.0,
                _to_float(sigs_data.get("product_steepness"), sigs_base.product_steepness),
            ),
            ad_acos_weight=max(
                0.0,
                _to_float(sigs_data.get("ad_acos_weight"), sigs_base.ad_acos_weight),
            ),
            ad_cvr_weight=max(
                0.0,
                _to_float(sigs_data.get("ad_cvr_weight"), sigs_base.ad_cvr_weight),
            ),
            ad_spend_up_no_sales_weight=max(
                0.0,
                _to_float(sigs_data.get("ad_spend_up_no_sales_weight"), sigs_base.ad_spend_up_no_sales_weight),
            ),
            ad_steepness=max(
                0.0,
                _to_float(sigs_data.get("ad_steepness"), sigs_base.ad_steepness),
            ),
        )

        # 阶段化指标权重（可配置；没有则用默认）
        ss_base = base.dashboard_stage_scoring
        ss_data = dash.get("stage_scoring") if isinstance(dash.get("stage_scoring"), dict) else {}

        def _parse_phase_list(v: object, default: List[str]) -> List[str]:
            try:
                if not isinstance(v, list):
                    return list(default)
                out: List[str] = []
                for x in v:
                    s = str(x or "").strip().lower()
                    if s:
                        out.append(s)
                return out if out else list(default)
            except Exception:
                return list(default)

        stage_scoring = StageScoringPolicy(
            launch_phases=_parse_phase_list(ss_data.get("launch_phases"), ss_base.launch_phases),
            growth_phases=_parse_phase_list(ss_data.get("growth_phases"), ss_base.growth_phases),
            new_phases=_parse_phase_list(ss_data.get("new_phases"), ss_base.new_phases),
            mature_phases=_parse_phase_list(ss_data.get("mature_phases"), ss_base.mature_phases),
            decline_phases=_parse_phase_list(ss_data.get("decline_phases"), ss_base.decline_phases),
            median_min_samples=max(
                1,
                _to_int(ss_data.get("median_min_samples"), ss_base.median_min_samples),
            ),
            min_impressions_7d=max(
                0.0,
                _to_float(ss_data.get("min_impressions_7d"), ss_base.min_impressions_7d),
            ),
            min_clicks_7d=max(
                0.0,
                _to_float(ss_data.get("min_clicks_7d"), ss_base.min_clicks_7d),
            ),
            min_orders_7d=max(
                0.0,
                _to_float(ss_data.get("min_orders_7d"), ss_base.min_orders_7d),
            ),
            min_sales_7d=max(
                0.0,
                _to_float(ss_data.get("min_sales_7d"), ss_base.min_sales_7d),
            ),
            min_orders_7d_mature=max(
                0.0,
                _to_float(ss_data.get("min_orders_7d_mature"), ss_base.min_orders_7d_mature),
            ),
            new_ctr_low_ratio=max(
                0.0,
                _to_float(ss_data.get("new_ctr_low_ratio"), ss_base.new_ctr_low_ratio),
            ),
            new_cvr_low_ratio=max(
                0.0,
                _to_float(ss_data.get("new_cvr_low_ratio"), ss_base.new_cvr_low_ratio),
            ),
            new_cpa_high_ratio=max(
                0.0,
                _to_float(ss_data.get("new_cpa_high_ratio"), ss_base.new_cpa_high_ratio),
            ),
            mature_cpa_high_ratio=max(
                0.0,
                _to_float(ss_data.get("mature_cpa_high_ratio"), ss_base.mature_cpa_high_ratio),
            ),
            mature_acos_high_ratio=max(
                0.0,
                _to_float(ss_data.get("mature_acos_high_ratio"), ss_base.mature_acos_high_ratio),
            ),
            mature_ad_share_shift_abs=max(
                0.0,
                _to_float(ss_data.get("mature_ad_share_shift_abs"), ss_base.mature_ad_share_shift_abs),
            ),
            mature_spend_shift_ratio=max(
                0.0,
                _to_float(ss_data.get("mature_spend_shift_ratio"), ss_base.mature_spend_shift_ratio),
            ),
            weight_new_low_ctr=_to_float(ss_data.get("weight_new_low_ctr"), ss_base.weight_new_low_ctr),
            weight_new_low_cvr=_to_float(ss_data.get("weight_new_low_cvr"), ss_base.weight_new_low_cvr),
            weight_new_high_cpa=_to_float(ss_data.get("weight_new_high_cpa"), ss_base.weight_new_high_cpa),
            weight_mature_high_cpa=_to_float(ss_data.get("weight_mature_high_cpa"), ss_base.weight_mature_high_cpa),
            weight_mature_high_acos=_to_float(ss_data.get("weight_mature_high_acos"), ss_base.weight_mature_high_acos),
            weight_mature_ad_share_shift=_to_float(
                ss_data.get("weight_mature_ad_share_shift"),
                ss_base.weight_mature_ad_share_shift,
            ),
            weight_mature_spend_shift=_to_float(
                ss_data.get("weight_mature_spend_shift"),
                ss_base.weight_mature_spend_shift,
            ),
            max_stage_tags=max(1, _to_int(ss_data.get("max_stage_tags"), ss_base.max_stage_tags)),
        )

        # Action Board 优先级（可配置；没有则用默认）
        as_base = base.dashboard_action_scoring
        as_data = dash.get("action_scoring") if isinstance(dash.get("action_scoring"), dict) else {}
        action_scoring = ActionScoringPolicy(
            base_score_p0=_to_float(as_data.get("base_score_p0"), as_base.base_score_p0),
            base_score_p1=_to_float(as_data.get("base_score_p1"), as_base.base_score_p1),
            base_score_p2=_to_float(as_data.get("base_score_p2"), as_base.base_score_p2),
            base_score_other=_to_float(as_data.get("base_score_other"), as_base.base_score_other),
            spend_log_multiplier=_to_float(as_data.get("spend_log_multiplier"), as_base.spend_log_multiplier),
            weight_focus_score=_to_float(as_data.get("weight_focus_score"), as_base.weight_focus_score),
            weight_hint_confidence=_to_float(as_data.get("weight_hint_confidence"), as_base.weight_hint_confidence),
            low_hint_confidence_threshold=max(
                0.0,
                min(
                    1.0,
                    _to_float(
                        as_data.get("low_hint_confidence_threshold"),
                        as_base.low_hint_confidence_threshold,
                    ),
                ),
            ),
            low_hint_scale_penalty=_to_float(as_data.get("low_hint_scale_penalty"), as_base.low_hint_scale_penalty),
            phase_scale_penalty_decline=_to_float(
                as_data.get("phase_scale_penalty_decline"),
                as_base.phase_scale_penalty_decline,
            ),
            phase_scale_penalty_inactive=_to_float(
                as_data.get("phase_scale_penalty_inactive"),
                as_base.phase_scale_penalty_inactive,
            ),
            profit_reduce_scale_penalty=_to_float(
                as_data.get("profit_reduce_scale_penalty"),
                as_base.profit_reduce_scale_penalty,
            ),
            profit_scale_scale_boost=_to_float(
                as_data.get("profit_scale_scale_boost"),
                as_base.profit_scale_scale_boost,
            ),
        )

        # 机会池 -> 预算迁移（可配置；没有则用默认）
        bto_base = base.dashboard_budget_transfer_opportunity
        bto_data = dash.get("budget_transfer_opportunity") if isinstance(dash.get("budget_transfer_opportunity"), dict) else {}
        budget_transfer_opportunity = BudgetTransferOpportunityPolicy(
            enabled=_to_bool(bto_data.get("enabled"), bto_base.enabled),
            suggested_add_pct=max(0.0, _to_float(bto_data.get("suggested_add_pct"), bto_base.suggested_add_pct)),
            max_target_campaigns=max(1, _to_int(bto_data.get("max_target_campaigns"), bto_base.max_target_campaigns)),
            min_target_opp_spend=max(0.0, _to_float(bto_data.get("min_target_opp_spend"), bto_base.min_target_opp_spend)),
            prefer_same_ad_type=_to_bool(bto_data.get("prefer_same_ad_type"), bto_base.prefer_same_ad_type),
        )

        # 放量解锁任务收敛策略（可配置；没有则用默认）
        ut_base = base.dashboard_unlock_tasks
        ut_data = dash.get("unlock_tasks") if isinstance(dash.get("unlock_tasks"), dict) else {}
        include_pr = list(ut_base.include_priorities)
        if isinstance(ut_data.get("include_priorities"), list):
            tmp: List[str] = []
            for x in ut_data.get("include_priorities"):
                s = str(x or "").strip().upper()
                if s:
                    tmp.append(s)
            if tmp:
                include_pr = tmp
        unlock_tasks_policy = UnlockTasksPolicy(
            top_n=max(1, _to_int(ut_data.get("top_n"), ut_base.top_n)),
            include_priorities=include_pr,
            dashboard_top_n=max(1, _to_int(ut_data.get("dashboard_top_n"), ut_base.dashboard_top_n)),
            dashboard_prefer_unique_asin=_to_bool(
                ut_data.get("dashboard_prefer_unique_asin"),
                ut_base.dashboard_prefer_unique_asin,
            ),
        )

        # Shop Alerts（规则化告警）阈值（可配置；没有则用默认）
        sa_base = base.dashboard_shop_alerts
        sa_data = dash.get("shop_alerts") if isinstance(dash.get("shop_alerts"), dict) else {}
        pdr_base = sa_base.phase_down_recent
        pdr_data = sa_data.get("phase_down_recent") if isinstance(sa_data.get("phase_down_recent"), dict) else {}
        p0_spend_sum = max(0.0, _to_float(pdr_data.get("p0_spend_sum"), pdr_base.p0_spend_sum))
        p0_spend_share = _to_float(pdr_data.get("p0_spend_share"), pdr_base.p0_spend_share)
        if p0_spend_share < 0:
            p0_spend_share = 0.0
        if p0_spend_share > 1:
            p0_spend_share = 1.0
        p0_spend_sum_min_when_share = max(
            0.0,
            _to_float(pdr_data.get("p0_spend_sum_min_when_share"), pdr_base.p0_spend_sum_min_when_share),
        )
        p0_asin_count_min = max(1, _to_int(pdr_data.get("p0_asin_count_min"), pdr_base.p0_asin_count_min))
        phase_down_recent_policy = PhaseDownRecentAlertPolicy(
            enabled=_to_bool(pdr_data.get("enabled"), pdr_base.enabled),
            p0_spend_sum=p0_spend_sum,
            p0_spend_share=p0_spend_share,
            p0_spend_sum_min_when_share=p0_spend_sum_min_when_share,
            p0_asin_count_min=p0_asin_count_min,
        )
        shop_alerts_policy = ShopAlertsPolicy(phase_down_recent=phase_down_recent_policy)

        # 关键词主题（n-gram）（可配置；没有则用默认）
        kt_base = base.dashboard_keyword_topics
        kt_data = dash.get("keyword_topics") if isinstance(dash.get("keyword_topics"), dict) else {}
        n_values = list(kt_base.n_values)
        if isinstance(kt_data.get("n_values"), list):
            tmp: List[int] = []
            for x in kt_data.get("n_values"):
                n = _to_int(x, 0)
                if 1 <= n <= 5:
                    tmp.append(int(n))
            tmp2 = sorted({int(x) for x in tmp if int(x) > 0})
            if tmp2:
                n_values = tmp2
        md_min_n = _to_int(kt_data.get("md_min_n"), kt_base.md_min_n)
        if md_min_n < 1:
            md_min_n = 1
        if md_min_n > 5:
            md_min_n = 5
        keyword_topics_policy = KeywordTopicsPolicy(
            enabled=_to_bool(kt_data.get("enabled"), kt_base.enabled),
            n_values=n_values,
            min_term_spend=max(0.0, _to_float(kt_data.get("min_term_spend"), kt_base.min_term_spend)),
            max_terms=max(1, _to_int(kt_data.get("max_terms"), kt_base.max_terms)),
            max_rows=max(1, _to_int(kt_data.get("max_rows"), kt_base.max_rows)),
            top_terms_per_ngram=max(
                1,
                _to_int(kt_data.get("top_terms_per_ngram"), kt_base.top_terms_per_ngram),
            ),
            md_top_n=max(1, _to_int(kt_data.get("md_top_n"), kt_base.md_top_n)),
            md_min_n=md_min_n,
            action_hints_enabled=_to_bool(kt_data.get("action_hints_enabled"), kt_base.action_hints_enabled),
            action_hints_top_waste=max(
                0,
                _to_int(kt_data.get("action_hints_top_waste"), kt_base.action_hints_top_waste),
            ),
            action_hints_top_scale=max(
                0,
                _to_int(kt_data.get("action_hints_top_scale"), kt_base.action_hints_top_scale),
            ),
            action_hints_min_waste_spend=max(
                0.0,
                _to_float(kt_data.get("action_hints_min_waste_spend"), kt_base.action_hints_min_waste_spend),
            ),
            action_hints_min_waste_ratio=max(
                0.0,
                min(
                    1.0,
                    _to_float(kt_data.get("action_hints_min_waste_ratio"), kt_base.action_hints_min_waste_ratio),
                ),
            ),
            action_hints_scale_acos_multiplier=max(
                0.0,
                _to_float(kt_data.get("action_hints_scale_acos_multiplier"), kt_base.action_hints_scale_acos_multiplier),
            ),
            action_hints_min_sales=max(
                0.0,
                _to_float(kt_data.get("action_hints_min_sales"), kt_base.action_hints_min_sales),
            ),
            action_hints_top_entities=max(
                1,
                _to_int(kt_data.get("action_hints_top_entities"), kt_base.action_hints_top_entities),
            ),
            asin_context_enabled=_to_bool(kt_data.get("asin_context_enabled"), kt_base.asin_context_enabled),
            asin_context_min_confidence=max(
                0.0,
                min(
                    1.0,
                    _to_float(kt_data.get("asin_context_min_confidence"), kt_base.asin_context_min_confidence),
                ),
            ),
            asin_context_top_asins_per_topic=max(
                1,
                _to_int(kt_data.get("asin_context_top_asins_per_topic"), kt_base.asin_context_top_asins_per_topic),
            ),
        )

        return OpsPolicy(
            low_inventory_threshold=low_thr,
            block_scale_when_low_inventory=block_scale,
            block_scale_when_cover_days_below=cover_below,
            keyword_funnel_top_n=top_n,
            campaign_windows_days=windows2,
            campaign_min_spend=min_spend,
            campaign_top_asins_per_campaign=top_asins,
            phase_acos_multiplier=mult2,
            dashboard_top_asins=dash_top_asins,
            dashboard_top_actions=dash_top_actions,
            dashboard_scale_window=scale_window,
            dashboard_focus_scoring=fs,
            dashboard_signal_scoring=signal_scoring,
            dashboard_stage_scoring=stage_scoring,
            dashboard_action_scoring=action_scoring,
            dashboard_budget_transfer_opportunity=budget_transfer_opportunity,
            dashboard_unlock_tasks=unlock_tasks_policy,
            dashboard_shop_alerts=shop_alerts_policy,
            dashboard_keyword_topics=keyword_topics_policy,
        )
    except Exception:
        return base


def load_ops_policy_with_overrides(path: Path, overrides: Dict[str, object]) -> OpsPolicy:
    """
    从文件读取 ops_policy.json，并应用 overrides（overrides 优先）。
    """
    try:
        base = OpsPolicy()
        if path is None or (not Path(path).exists()):
            return base
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return base
        merged = deep_merge_dict(dict(data), overrides or {})
        return load_ops_policy_dict(merged)
    except Exception:
        return OpsPolicy()
