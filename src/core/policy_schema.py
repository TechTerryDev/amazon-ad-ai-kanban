# -*- coding: utf-8 -*-
"""ops_policy.json 的 Pydantic 校验（仅做类型校验，不改变现有逻辑）。"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from pydantic import BaseModel, ConfigDict
    PYDANTIC_AVAILABLE = True
except Exception:  # pragma: no cover - 兼容未安装依赖时的导入
    BaseModel = object  # type: ignore
    ConfigDict = dict  # type: ignore
    PYDANTIC_AVAILABLE = False


class _BaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class InventorySchema(_BaseModel):
    low_inventory_threshold: Optional[int] = None
    block_scale_when_low_inventory: Optional[bool] = None
    block_scale_when_cover_days_below: Optional[float] = None
    category_default: Optional[Dict[str, object]] = None
    category_overrides: Optional[List[Dict[str, object]]] = None


class KeywordFunnelSchema(_BaseModel):
    top_n: Optional[int] = None


class CampaignOpsSchema(_BaseModel):
    windows_days: Optional[List[int]] = None
    min_spend: Optional[float] = None
    top_asins_per_campaign: Optional[int] = None
    phase_acos_multiplier: Optional[Dict[str, float]] = None


class ScaleWindowSchema(_BaseModel):
    min_sales_per_day_7d: Optional[float] = None
    min_delta_sales: Optional[float] = None
    min_inventory_cover_days_7d: Optional[float] = None
    max_tacos_roll: Optional[float] = None
    max_marginal_tacos: Optional[float] = None
    exclude_phases: Optional[List[str]] = None
    require_no_oos: Optional[bool] = None
    require_no_low_inventory: Optional[bool] = None
    require_oos_with_ad_spend_days_zero: Optional[bool] = None
    category_default: Optional[Dict[str, object]] = None
    category_overrides: Optional[List[Dict[str, object]]] = None


class FocusScoringSchema(_BaseModel):
    base_spend_log_multiplier: Optional[float] = None
    base_spend_score_cap: Optional[float] = None
    weight_flag_oos: Optional[float] = None
    weight_flag_low_inventory: Optional[float] = None
    weight_oos_with_ad_spend_days: Optional[float] = None
    weight_oos_with_sessions_days: Optional[float] = None
    weight_presale_order_days: Optional[float] = None
    weight_spend_up_no_sales: Optional[float] = None
    weight_marginal_tacos_worse: Optional[float] = None
    weight_decline_or_inactive_spend: Optional[float] = None
    weight_phase_down_recent: Optional[float] = None
    weight_high_ad_dependency: Optional[float] = None
    weight_inventory_zero_still_spend: Optional[float] = None
    high_ad_dependency_threshold: Optional[float] = None
    marginal_tacos_worse_ratio: Optional[float] = None
    weight_sessions_up_cvr_down: Optional[float] = None
    cvr_signal_min_sessions_prev: Optional[float] = None
    cvr_signal_min_delta_sessions: Optional[float] = None
    cvr_signal_min_cvr_drop: Optional[float] = None
    cvr_signal_min_ad_spend_roll: Optional[float] = None
    weight_organic_down: Optional[float] = None
    organic_signal_min_organic_sales_prev: Optional[float] = None
    organic_signal_min_delta_organic_sales: Optional[float] = None
    organic_signal_drop_ratio: Optional[float] = None
    aov_signal_min_orders_prev: Optional[float] = None
    aov_signal_min_delta_aov: Optional[float] = None
    aov_signal_drop_ratio: Optional[float] = None
    gross_margin_signal_min_sales: Optional[float] = None
    gross_margin_signal_low_threshold: Optional[float] = None


class SignalScoringSchema(_BaseModel):
    product_sales_weight: Optional[float] = None
    product_sessions_weight: Optional[float] = None
    product_organic_sales_weight: Optional[float] = None
    product_profit_weight: Optional[float] = None
    product_steepness: Optional[float] = None
    ad_acos_weight: Optional[float] = None
    ad_cvr_weight: Optional[float] = None
    ad_spend_up_no_sales_weight: Optional[float] = None
    ad_steepness: Optional[float] = None


class StageScoringSchema(_BaseModel):
    launch_phases: Optional[List[str]] = None
    growth_phases: Optional[List[str]] = None
    new_phases: Optional[List[str]] = None
    mature_phases: Optional[List[str]] = None
    decline_phases: Optional[List[str]] = None
    median_min_samples: Optional[int] = None
    min_impressions_7d: Optional[float] = None
    min_clicks_7d: Optional[float] = None
    min_orders_7d: Optional[float] = None
    min_sales_7d: Optional[float] = None
    min_orders_7d_mature: Optional[float] = None
    new_ctr_low_ratio: Optional[float] = None
    new_cvr_low_ratio: Optional[float] = None
    new_cpa_high_ratio: Optional[float] = None
    mature_cpa_high_ratio: Optional[float] = None
    mature_acos_high_ratio: Optional[float] = None
    mature_ad_share_shift_abs: Optional[float] = None
    mature_spend_shift_ratio: Optional[float] = None
    weight_new_low_ctr: Optional[float] = None
    weight_new_low_cvr: Optional[float] = None
    weight_new_high_cpa: Optional[float] = None
    weight_mature_high_cpa: Optional[float] = None
    weight_mature_high_acos: Optional[float] = None
    weight_mature_ad_share_shift: Optional[float] = None
    weight_mature_spend_shift: Optional[float] = None
    max_stage_tags: Optional[int] = None


class ActionScoringSchema(_BaseModel):
    base_score_p0: Optional[float] = None
    base_score_p1: Optional[float] = None
    base_score_p2: Optional[float] = None
    base_score_other: Optional[float] = None
    spend_log_multiplier: Optional[float] = None
    weight_focus_score: Optional[float] = None
    weight_hint_confidence: Optional[float] = None
    low_hint_confidence_threshold: Optional[float] = None
    low_hint_scale_penalty: Optional[float] = None
    phase_scale_penalty_decline: Optional[float] = None
    phase_scale_penalty_inactive: Optional[float] = None
    profit_reduce_scale_penalty: Optional[float] = None
    profit_scale_scale_boost: Optional[float] = None
    weight_action_loop_score: Optional[float] = None
    action_loop_min_support: Optional[int] = None
    action_loop_recent_window_days: Optional[int] = None


class PromoAdjustmentSchema(_BaseModel):
    enabled: Optional[bool] = None
    baseline_lookback_days: Optional[int] = None
    baseline_min_periods: Optional[int] = None
    sales_spike_threshold: Optional[float] = None
    spend_spike_threshold: Optional[float] = None
    sales_spike_threshold_alt: Optional[float] = None
    spend_spike_threshold_alt: Optional[float] = None
    damp_ratio: Optional[float] = None


class InventorySigmoidSchema(_BaseModel):
    enabled: Optional[bool] = None
    optimal_cover_days: Optional[float] = None
    steepness: Optional[float] = None
    min_modifier: Optional[float] = None
    max_modifier: Optional[float] = None
    min_change_ratio: Optional[float] = None
    min_ad_spend_roll: Optional[float] = None


class ProfitGuardSchema(_BaseModel):
    enabled: Optional[bool] = None
    target_net_margin: Optional[float] = None
    min_sales_7d: Optional[float] = None
    min_ad_spend_roll: Optional[float] = None


class BudgetTransferOpportunitySchema(_BaseModel):
    enabled: Optional[bool] = None
    suggested_add_pct: Optional[float] = None
    max_target_campaigns: Optional[int] = None
    min_target_opp_spend: Optional[float] = None
    prefer_same_ad_type: Optional[bool] = None


class UnlockTasksSchema(_BaseModel):
    top_n: Optional[int] = None
    include_priorities: Optional[List[str]] = None
    dashboard_top_n: Optional[int] = None
    dashboard_prefer_unique_asin: Optional[bool] = None


class PhaseDownRecentSchema(_BaseModel):
    enabled: Optional[bool] = None
    p0_spend_sum: Optional[float] = None
    p0_spend_share: Optional[float] = None
    p0_spend_sum_min_when_share: Optional[float] = None
    p0_asin_count_min: Optional[int] = None


class ShopAlertsSchema(_BaseModel):
    phase_down_recent: Optional[PhaseDownRecentSchema] = None


class KeywordTopicsSchema(_BaseModel):
    enabled: Optional[bool] = None
    n_values: Optional[List[int]] = None
    min_term_spend: Optional[float] = None
    max_terms: Optional[int] = None
    max_rows: Optional[int] = None
    top_terms_per_ngram: Optional[int] = None
    md_top_n: Optional[int] = None
    md_min_n: Optional[int] = None
    action_hints_enabled: Optional[bool] = None
    action_hints_top_waste: Optional[int] = None
    action_hints_top_scale: Optional[int] = None
    action_hints_min_waste_spend: Optional[float] = None
    action_hints_min_waste_ratio: Optional[float] = None
    action_hints_scale_acos_multiplier: Optional[float] = None
    action_hints_min_sales: Optional[float] = None
    action_hints_top_entities: Optional[int] = None
    asin_context_enabled: Optional[bool] = None
    asin_context_min_confidence: Optional[float] = None
    asin_context_top_asins_per_topic: Optional[int] = None


class DashboardSchema(_BaseModel):
    top_asins: Optional[int] = None
    top_actions: Optional[int] = None
    compare_ignore_last_days: Optional[int] = None
    promo_adjustment: Optional[PromoAdjustmentSchema] = None
    scale_window: Optional[ScaleWindowSchema] = None
    focus_scoring: Optional[FocusScoringSchema] = None
    signal_scoring: Optional[SignalScoringSchema] = None
    stage_scoring: Optional[StageScoringSchema] = None
    action_scoring: Optional[ActionScoringSchema] = None
    inventory_sigmoid: Optional[InventorySigmoidSchema] = None
    profit_guard: Optional[ProfitGuardSchema] = None
    budget_transfer_opportunity: Optional[BudgetTransferOpportunitySchema] = None
    unlock_tasks: Optional[UnlockTasksSchema] = None
    shop_alerts: Optional[ShopAlertsSchema] = None
    keyword_topics: Optional[KeywordTopicsSchema] = None


class OpsPolicySchema(_BaseModel):
    inventory: Optional[InventorySchema] = None
    keyword_funnel: Optional[KeywordFunnelSchema] = None
    campaign_ops: Optional[CampaignOpsSchema] = None
    dashboard: Optional[DashboardSchema] = None
