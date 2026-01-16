# -*- coding: utf-8 -*-
"""
AI 自动写报告（可选）。

把本项目已经产出的结构化文件（ai_input_bundle + data_quality + action_board）
组装成“只做解释与写作”的提示词，然后通过 ai_providers 调用大模型，生成：
- ai/ai_suggestions.md：AI 输出报告（给你/分析用，不给一线运营当作事实）
- ai/ai_suggestions_prompt.md：本次提示词留档（便于复现/调参）

设计目标（对齐你的项目目标）：
- 先把数据分析逻辑跑通：AI 只负责解释，不参与算数
- 默认不开启，避免误耗 token；开启后失败也不影响主流程
- 尽量不新增依赖：默认推荐用 ai_providers 的 oai_http（标准库 HTTP）

实现说明（为什么要这么做）：
- 你的痛点是“数据太多抓不到重点”，所以 AI 报告的输入必须强收敛（否则 token 太大、反而更难读）
- 报告只允许基于本项目已经算好的字段写作，避免 AI“看错口径/臆造缺失维度”
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _repo_root() -> Path:
    # src/analysis/ai_report.py -> src/analysis -> src -> repo_root
    return Path(__file__).resolve().parents[2]


def _load_dotenv_if_present(dotenv_path: Path) -> None:
    """
    轻量加载 .env（不依赖 python-dotenv）。

    规则：
    - 仅支持 KEY=VALUE
    - 忽略空行与 # 注释
    - 不覆盖已有环境变量（便于你在 shell 里临时 export）
    """
    try:
        # 单元测试/CI 场景：允许显式禁用自动读取 .env，避免测试与本机环境耦合
        # 使用方式：export HELLOAGENTS_DISABLE_DOTENV=1
        if str(os.getenv("HELLOAGENTS_DISABLE_DOTENV") or "").strip().lower() in ("1", "true", "yes", "y"):
            return
        if dotenv_path is None or not dotenv_path.exists():
            return
        for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = str(raw or "").strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = str(k).strip()
            val = str(v).strip().strip('"').strip("'")
            if not key:
                continue
            if os.getenv(key) is not None:
                continue
            os.environ[key] = val
    except Exception:
        # .env 读取失败不应该影响主流程
        return


def _read_text(path: Path, max_chars: int = 0) -> str:
    try:
        if path is None or not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        if max_chars and max_chars > 0 and len(text) > max_chars:
            return text[: int(max_chars)] + "\n...(truncated)...\n"
        return text
    except Exception:
        return ""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path is None or not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_data_quality_snippet(md_text: str, max_chars: int = 6000) -> str:
    """
    从 data_quality.md 中提取更适合喂给 AI 的片段：
    - 保留：标题 + 摘要 + 维度可用性（到 ## 2) 前）
    - 避免把“超长覆盖率表”塞给模型导致 token 爆炸
    """
    try:
        if not md_text:
            return ""
        lines = md_text.splitlines()
        out: List[str] = []
        for line in lines:
            out.append(line)
            if str(line).strip().startswith("## 2)"):
                break
        text = "\n".join(out).strip()
        if max_chars and len(text) > max_chars:
            return text[: int(max_chars)] + "\n...(truncated)...\n"
        return text
    except Exception:
        return md_text[:max_chars] if md_text else ""


def _compact_ai_bundle(
    bundle: Dict[str, Any],
    max_asins: int = 40,
    keep_compare_windows: Optional[List[int]] = None,
    include_ads: bool = False,
    top_campaigns: int = 3,
    top_search_terms: int = 0,
    top_targetings: int = 0,
    top_placements: int = 0,
) -> Dict[str, Any]:
    """
    压缩 ai_input_bundle.json，避免 prompt 过大。

    原则：
    - 保留“产品立体语境”（生命周期/库存/自然拆分/利润派生）
    - 默认不带广告结构（action_board/keyword_topics 已足够写执行清单）；需要时再开启 include_ads
    """
    keep_compare_windows = keep_compare_windows or [7, 14]
    out: Dict[str, Any] = {}
    try:
        for k in ("shop", "stage_profile", "date_range", "summary_total", "product_analysis_summary", "shop_scorecard", "definitions"):
            if k in bundle:
                out[k] = bundle.get(k)

        asins = bundle.get("asins")
        if not isinstance(asins, list):
            out["asins"] = []
            return out

        max_n = int(max_asins or 0)
        if max_n <= 0:
            max_n = 40

        compacted: List[Dict[str, Any]] = []
        for item in asins[:max_n]:
            if not isinstance(item, dict):
                continue
            x: Dict[str, Any] = {"asin": item.get("asin")}
            if "lifecycle_current" in item:
                x["lifecycle_current"] = item.get("lifecycle_current")
            if "lifecycle_main_window" in item:
                x["lifecycle_main_window"] = item.get("lifecycle_main_window")

            compares = item.get("lifecycle_compares")
            if isinstance(compares, list) and compares:
                kept = []
                for r in compares:
                    if not isinstance(r, dict):
                        continue
                    wd = r.get("window_days")
                    try:
                        wdi = int(wd)
                    except Exception:
                        wdi = -1
                    if wdi in keep_compare_windows:
                        kept.append(r)
                x["lifecycle_compares"] = kept

            if bool(include_ads):
                ads = item.get("ads")
                if isinstance(ads, dict):
                    def _slice_list(v: Any, n: int) -> List[Any]:
                        try:
                            nn = int(n or 0)
                            if nn <= 0:
                                return []
                            return v[:nn] if isinstance(v, list) else []
                        except Exception:
                            return []

                    x["ads"] = {
                        "top_campaigns": _slice_list(ads.get("top_campaigns"), top_campaigns),
                        "top_search_terms": _slice_list(ads.get("top_search_terms"), top_search_terms),
                        "top_targetings": _slice_list(ads.get("top_targetings"), top_targetings),
                        "top_placements": _slice_list(ads.get("top_placements"), top_placements),
                    }

            compacted.append(x)

        out["asins"] = compacted
        return out
    except Exception:
        # 任何异常都不要影响主流程
        return out


def _action_board_snippet(action_board_csv: Path, max_rows: int = 200) -> str:
    try:
        if action_board_csv is None or not action_board_csv.exists():
            return ""
        n = int(max_rows or 0)
        if n <= 0:
            n = 200
        df = pd.read_csv(action_board_csv)
        if df is None or df.empty:
            return ""
        # 只保留“写报告必需列”，避免 evidence_json 等超长字段把 prompt 撑爆
        prefer_cols = [
            "priority",
            "blocked",
            "blocked_reason",
            "needs_manual_confirm",
            "product_category",
            "asin_hint",
            "asin_hint_confidence",
            "current_phase",
            "action_type",
            "action_value",
            "level",
            "ad_type",
            "campaign",
            "ad_group",
            "match_type",
            "object_name",
            "e_spend",
            "e_clicks",
            "e_orders",
            "e_sales",
            "e_acos",
            "reason",
            "priority_reason",
            # 产品语境（抓重点）
            "inventory",
            "flag_oos",
            "flag_low_inventory",
            "asin_inventory_cover_days_7d",
            "asin_ad_sales_share",
            "asin_tacos",
            "asin_delta_sales",
            "asin_delta_spend",
            "asin_marginal_tacos",
            "asin_phase_change",
            "asin_phase_trend_14d",
            "asin_profit_direction",
            "asin_max_ad_spend_by_profit",
        ]
        cols = [c for c in prefer_cols if c in df.columns]
        view = df[cols].copy() if cols else df.copy()

        # 结构化排序：确保 P0 不被截断，P1 再取 TopN
        try:
            if "priority" in view.columns:
                p0 = view[view["priority"] == "P0"].copy()
                p1 = view[view["priority"] == "P1"].copy()
                other = view[(view["priority"] != "P0") & (view["priority"] != "P1")].copy()
                sort_cols = []
                if "e_spend" in view.columns:
                    sort_cols = ["e_spend"]
                if sort_cols:
                    try:
                        p0 = p0.sort_values(sort_cols, ascending=False)
                        p1 = p1.sort_values(sort_cols, ascending=False)
                        other = other.sort_values(sort_cols, ascending=False)
                    except Exception:
                        pass
                view = pd.concat([p0, p1, other], ignore_index=True)
        except Exception:
            pass

        view = view.head(n).copy()
        return view.to_csv(index=False)
    except Exception:
        return ""


def _asin_focus_snippet(asin_focus_csv: Path, max_rows: int = 40) -> str:
    try:
        if asin_focus_csv is None or not asin_focus_csv.exists():
            return ""
        n = int(max_rows or 0)
        if n <= 0:
            n = 40
        df = pd.read_csv(asin_focus_csv)
        if df is None or df.empty:
            return ""
        prefer_cols = [
            "product_category",
            "asin",
            "product_name",
            "current_phase",
            "phase_change",
            "phase_trend_14d",
            "inventory",
            "inventory_cover_days_7d",
            "sales_per_day_7d",
            "ad_spend_roll",
            "tacos_roll",
            "ad_sales_share",
            "ad_acos",
            "gross_margin",
            "profit_direction",
            "max_ad_spend_by_profit",
            "focus_score",
            "focus_reasons",
            "delta_sales",
            "delta_spend",
            "marginal_tacos",
            "delta_cvr",
            "delta_organic_sales",
            "delta_aov_7d",
        ]
        cols = [c for c in prefer_cols if c in df.columns]
        view = df[cols].copy() if cols else df.copy()
        if "focus_score" in view.columns:
            try:
                view = view.sort_values("focus_score", ascending=False)
            except Exception:
                pass
        return view.head(n).to_csv(index=False)
    except Exception:
        return ""


def _unlock_scale_tasks_snippet(unlock_tasks_csv: Path, max_rows: int = 30) -> str:
    try:
        if unlock_tasks_csv is None or not unlock_tasks_csv.exists():
            return ""
        n = int(max_rows or 0)
        if n <= 0:
            n = 30
        df = pd.read_csv(unlock_tasks_csv)
        if df is None or df.empty:
            return ""
        prefer_cols = [
            "priority",
            "owner",
            "task_type",
            "product_category",
            "asin",
            "product_name",
            "current_phase",
            "inventory",
            "inventory_cover_days_7d",
            "sales_per_day_7d",
            "budget_gap_usd_est",
            "profit_gap_usd_est",
            "need",
            "target",
            "evidence",
        ]
        cols = [c for c in prefer_cols if c in df.columns]
        view = df[cols].copy() if cols else df.copy()
        if "priority" in view.columns:
            order = {"P0": 0, "P1": 1, "P2": 2}
            try:
                view["_p"] = view["priority"].map(lambda x: order.get(str(x).strip(), 9))
                view = view.sort_values(["_p"])
                view = view.drop(columns=["_p"], errors="ignore")
            except Exception:
                pass
        return view.head(n).to_csv(index=False)
    except Exception:
        return ""


def _budget_transfer_plan_snippet(plan_csv: Path, max_rows: int = 50) -> str:
    try:
        if plan_csv is None or not plan_csv.exists():
            return ""
        n = int(max_rows or 0)
        if n <= 0:
            n = 50
        df = pd.read_csv(plan_csv)
        if df is None or df.empty:
            return ""
        try:
            # 按金额降序（兜底：确保 AI 看到的是最重要的 TopN）
            if "amount_usd_estimated" in df.columns:
                df["amount_usd_estimated"] = pd.to_numeric(df["amount_usd_estimated"], errors="coerce").fillna(0.0)
                df = df.sort_values(["amount_usd_estimated"], ascending=[False]).copy()
        except Exception:
            pass
        prefer_cols = [
            "strategy",
            "transfer_type",
            "from_ad_type",
            "from_campaign",
            "from_spend",
            "from_asin_hint",
            "to_ad_type",
            "to_campaign",
            "to_spend",
            "to_asin_hint",
            "amount_usd_estimated",
            "note",
        ]
        cols = [c for c in prefer_cols if c in df.columns]
        view = df[cols].copy() if cols else df.copy()
        return view.head(n).to_csv(index=False)
    except Exception:
        return ""


def _summarize_budget_transfer_plan(plan_csv: Path, top_n: int = 5) -> Dict[str, Any]:
    """
    从 dashboard/budget_transfer_plan.csv 生成“短而有效”的摘要，给 AI 抓重点用。

    说明：
    - 预算迁移表在部分店铺可能非常长（例如回收/RESERVE 很多行），
      运营/AI 都更需要“总览 + TopN”而不是全文。
    - 金额是估算值（基于本期花费 proxy），只用于线索与排序。
    """
    try:
        if plan_csv is None or not plan_csv.exists():
            return {}
        df = pd.read_csv(plan_csv)
        if df is None or df.empty:
            return {}

        for c in ("amount_usd_estimated", "from_spend", "to_spend"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "amount_usd_estimated" in df.columns:
            df["amount_usd_estimated"] = df["amount_usd_estimated"].fillna(0.0)
        else:
            df["amount_usd_estimated"] = 0.0

        transfer_df = df[df.get("transfer_type", "") == "transfer"].copy() if "transfer_type" in df.columns else pd.DataFrame()
        reserve_df = df[df.get("transfer_type", "") == "reserve"].copy() if "transfer_type" in df.columns else pd.DataFrame()

        def _top_rows(x: pd.DataFrame) -> List[Dict[str, Any]]:
            if x is None or x.empty:
                return []
            x = x.sort_values(["amount_usd_estimated"], ascending=[False]).copy()
            keep_cols = [
                "strategy",
                "transfer_type",
                "from_ad_type",
                "from_campaign",
                "from_severity",
                "from_spend",
                "from_asin_hint",
                "to_ad_type",
                "to_campaign",
                "to_severity",
                "to_spend",
                "to_asin_hint",
                "to_confidence",
                "to_bucket",
                "amount_usd_estimated",
                "note",
            ]
            cols = [c for c in keep_cols if c in x.columns]
            out = x[cols].head(max(1, int(top_n or 0))).to_dict(orient="records")
            # 转成 python 标量，避免 numpy 类型在 json.dumps 时变成奇怪格式
            cleaned: List[Dict[str, Any]] = []
            for r in out:
                rr: Dict[str, Any] = {}
                for k, v in (r or {}).items():
                    if v is None:
                        rr[k] = None
                        continue
                    if hasattr(v, "item"):
                        try:
                            vv = v.item()
                            if isinstance(vv, float) and vv != vv:  # NaN
                                rr[k] = None
                            else:
                                rr[k] = vv
                            continue
                        except Exception:
                            pass
                    if isinstance(v, float) and v != v:  # NaN
                        rr[k] = None
                    else:
                        rr[k] = v
                cleaned.append(rr)
            return cleaned

        total_amount = float(df["amount_usd_estimated"].sum())
        transfer_amount = float(transfer_df["amount_usd_estimated"].sum()) if not transfer_df.empty else 0.0
        reserve_amount = float(reserve_df["amount_usd_estimated"].sum()) if not reserve_df.empty else 0.0

        summary: Dict[str, Any] = {
            "total_rows": int(len(df)),
            "transfer_rows": int(len(transfer_df)) if not transfer_df.empty else 0,
            "reserve_rows": int(len(reserve_df)) if not reserve_df.empty else 0,
            "total_amount_usd_estimated": total_amount,
            "transfer_amount_usd_estimated": transfer_amount,
            "reserve_amount_usd_estimated": reserve_amount,
            "top_transfers": _top_rows(transfer_df),
            "top_reserves": _top_rows(reserve_df),
        }
        return summary
    except Exception:
        return {}


def _keyword_topics_action_hints_snippet(hints_csv: Path, top_reduce: int = 8, top_scale: int = 8) -> str:
    try:
        if hints_csv is None or not hints_csv.exists():
            return ""
        df = pd.read_csv(hints_csv)
        if df is None or df.empty:
            return ""
        reduce_df = df[df["direction"] == "reduce"].copy() if "direction" in df.columns else pd.DataFrame()
        scale_df = df[df["direction"] == "scale"].copy() if "direction" in df.columns else pd.DataFrame()
        try:
            if not reduce_df.empty and "waste_spend" in reduce_df.columns:
                reduce_df["waste_spend"] = pd.to_numeric(reduce_df["waste_spend"], errors="coerce").fillna(0.0)
                reduce_df = reduce_df.sort_values(["priority", "waste_spend"], ascending=[True, False])
        except Exception:
            pass
        try:
            if not scale_df.empty:
                if "sales" in scale_df.columns:
                    scale_df["sales"] = pd.to_numeric(scale_df["sales"], errors="coerce").fillna(0.0)
                if "acos" in scale_df.columns:
                    scale_df["acos"] = pd.to_numeric(scale_df["acos"], errors="coerce").fillna(0.0)
                # sales 大优先，acos 小优先
                sort_cols = [c for c in ["priority", "sales", "acos"] if c in scale_df.columns]
                if sort_cols:
                    scale_df = scale_df.sort_values(sort_cols, ascending=[True, False, True][: len(sort_cols)])
        except Exception:
            pass

        view = pd.concat([reduce_df.head(int(top_reduce or 0)), scale_df.head(int(top_scale or 0))], ignore_index=True)
        if view.empty:
            return ""
        prefer_cols = [
            "priority",
            "direction",
            "hint_action",
            "n",
            "ngram",
            "spend",
            "sales",
            "orders",
            "acos",
            "waste_spend",
            "waste_ratio",
            "blocked",
            "blocked_reason",
            "top_terms",
            "top_campaigns",
            "top_ad_groups",
            "filter_contains",
            "next_step",
            "context_top_asins",
            "context_min_cover_days_7d",
        ]
        cols = [c for c in prefer_cols if c in view.columns]
        return (view[cols] if cols else view).to_csv(index=False)
    except Exception:
        return ""


def _build_run_context_json(shop_dir: Path) -> Dict[str, Any]:
    """
    用更“短而有效”的结构化摘要，帮助 AI 抓重点（减少喂整份大 JSON/CSV）。
    """
    ctx: Dict[str, Any] = {}
    try:
        sc_path = shop_dir / "dashboard" / "shop_scorecard.json"
        sc = _read_json(sc_path)
        if sc:
            ctx["shop"] = sc.get("shop")
            ctx["stage_profile"] = sc.get("stage_profile")
            ctx["date_range"] = sc.get("date_range")
            scorecard = sc.get("scorecard", {}) if isinstance(sc.get("scorecard", {}), dict) else {}
            biz_kpi = scorecard.get("biz_kpi", {}) if isinstance(scorecard.get("biz_kpi", {}), dict) else {}
            ctx["biz_kpi"] = {
                k: biz_kpi.get(k)
                for k in [
                    "sales_total",
                    "orders_total",
                    "sessions_total",
                    "profit_total",
                    "ad_spend_total",
                    "ad_sales_total",
                    "organic_sales_total",
                    "tacos_total",
                    "ad_acos_total",
                    "ad_sales_share_total",
                    "cvr_total",
                    "aov_total",
                ]
                if k in biz_kpi
            }
            compares = scorecard.get("compares", []) if isinstance(scorecard.get("compares", []), list) else []
            keep: List[Dict[str, Any]] = []
            for c in compares:
                if not isinstance(c, dict):
                    continue
                try:
                    wd = int(c.get("window_days") or 0)
                except Exception:
                    wd = 0
                if wd in {7, 14}:
                    keep.append(
                        {
                            k: c.get(k)
                            for k in [
                                "window_days",
                                "recent_start",
                                "recent_end",
                                "prev_start",
                                "prev_end",
                                "sales_recent",
                                "ad_spend_recent",
                                "profit_recent",
                                "tacos_recent",
                                "ad_sales_share_recent",
                                "delta_sales",
                                "delta_ad_spend",
                                "delta_profit",
                                "marginal_tacos",
                            ]
                        }
                    )
            if keep:
                ctx["roll_compares"] = keep

            drivers = scorecard.get("drivers", {}) if isinstance(scorecard.get("drivers", {}), dict) else {}
            win7 = drivers.get("window_7d", {}) if isinstance(drivers.get("window_7d", {}), dict) else {}

            def _pick_driver(rows: Any) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                if not isinstance(rows, list):
                    return out
                for r in rows[:5]:
                    if not isinstance(r, dict):
                        continue
                    out.append(
                        {
                            "asin": r.get("asin"),
                            "product_name": r.get("product_name"),
                            "current_phase": r.get("current_phase"),
                            "inventory": r.get("inventory"),
                            "flag_oos": r.get("flag_oos"),
                            "flag_low_inventory": r.get("flag_low_inventory"),
                            "delta_sales": r.get("delta_sales"),
                            "delta_ad_spend": r.get("delta_ad_spend"),
                            "marginal_tacos": r.get("marginal_tacos"),
                        }
                    )
                return out

            ctx["drivers_7d"] = {
                "by_delta_sales_top": _pick_driver(win7.get("by_delta_sales")),
                "by_delta_ad_spend_top": _pick_driver(win7.get("by_delta_ad_spend")),
            }
    except Exception:
        pass

    try:
        ab = pd.read_csv(shop_dir / "dashboard" / "action_board.csv")
        if ab is not None and not ab.empty and "priority" in ab.columns:
            ctx["action_board_counts"] = {
                "total": int(len(ab)),
                "p0": int((ab["priority"] == "P0").sum()),
                "p1": int((ab["priority"] == "P1").sum()),
                "blocked": int(pd.to_numeric(ab.get("blocked", 0), errors="coerce").fillna(0).astype(int).sum()) if "blocked" in ab.columns else 0,
                "needs_manual_confirm": int(pd.to_numeric(ab.get("needs_manual_confirm", 0), errors="coerce").fillna(0).astype(int).sum()) if "needs_manual_confirm" in ab.columns else 0,
            }
    except Exception:
        pass

    try:
        wl = {
            "profit_reduce_watchlist": "dashboard/profit_reduce_watchlist.csv",
            "oos_with_ad_spend_watchlist": "dashboard/oos_with_ad_spend_watchlist.csv",
            "spend_up_no_sales_watchlist": "dashboard/spend_up_no_sales_watchlist.csv",
            "phase_down_recent_watchlist": "dashboard/phase_down_recent_watchlist.csv",
            "scale_opportunity_watchlist": "dashboard/scale_opportunity_watchlist.csv",
        }
        counts: Dict[str, int] = {}
        for k, rel in wl.items():
            p = shop_dir / rel
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
                counts[k] = int(len(df)) if df is not None else 0
            except Exception:
                continue
        if counts:
            ctx["watchlist_counts"] = counts
    except Exception:
        pass

    try:
        ut = pd.read_csv(shop_dir / "dashboard" / "unlock_scale_tasks.csv")
        if ut is not None and not ut.empty and "priority" in ut.columns:
            ctx["unlock_tasks_counts"] = {
                "total": int(len(ut)),
                "p0": int((ut["priority"] == "P0").sum()),
                "p1": int((ut["priority"] == "P1").sum()),
            }
    except Exception:
        pass

    try:
        bt_path = shop_dir / "dashboard" / "budget_transfer_plan.csv"
        bt_summary = _summarize_budget_transfer_plan(bt_path, top_n=5)
        if bt_summary:
            ctx["budget_transfer_summary"] = bt_summary
    except Exception:
        pass

    return ctx


def build_ai_suggestions_prompt(
    shop: str,
    stage: str,
    run_context: Dict[str, Any],
    data_quality_snippet_md: str,
    asin_focus_csv_snippet: str,
    unlock_tasks_csv_snippet: str,
    budget_transfer_csv_snippet: str,
    keyword_topics_csv_snippet: str,
    action_board_csv_snippet: str,
) -> Tuple[str, str]:
    """
    返回 (system_prompt, user_prompt)。
    """
    system_prompt = (
        "你是一个“亚马逊广告运营分析助手”。你的任务是根据用户提供的数据文件生成可执行、可复盘、可回滚的广告优化建议。\n\n"
        "硬性规则：\n"
        "1) 你只能使用输入里出现的字段与数值，不得臆造任何缺失维度。\n"
        "2) 你不参与算数；如需数值，只能引用输入中现成的数值（aov/gross_margin/delta_* 等已提供）。\n"
        "3) 必须遵守 data_quality 的缺口提示：缺失某类报表/字段时，不得输出依赖该字段的绝对值建议。\n"
        "4) 输出必须包含：口径说明、关键结论（<=5条）、动作清单（按 P0/P1）、风险提示。\n"
        "5) 每条结论/每条动作都必须写“证据”，用“来源文件:字段=数值”的形式（例如 dashboard/action_board.csv:e_spend=66.07）。\n"
        "6) 必须把“动态生命周期”作为语境：至少引用 current_phase 与 phase_change/phase_trend_14d；并用库存/利润承受度阻断放量。\n\n"
        "格式强约束（为了解决“前置杂项/数据太多抓不到重点”）：\n"
        "7) 你的输出正文必须从 `## 0) 口径与数据覆盖` 开始；在此之前禁止出现任何标题/前言/分隔线/元信息。\n"
        "8) 六个小节内，所有内容都必须使用 Markdown 无序列表：每条以 `- ` 开头；禁止使用编号列表（`1.`/`2.`）。\n"
        "9) 每条 bullet 必须紧跟一行证据行：以 `  证据：` 开头，并至少包含 1 个 `来源文件:字段=数值`（必须带 `=`）。\n"
        "10) 证据来源文件只能来自：run_context、ai/data_quality.md、dashboard/shop_scorecard.json、dashboard/asin_focus.csv、dashboard/action_board.csv、dashboard/budget_transfer_plan.csv、dashboard/unlock_scale_tasks.csv、dashboard/keyword_topics_action_hints.csv。\n"
        "11) 如果证据不足/缺失字段：必须把该条写成“待人工确认/需补报表”，并在证据行引用 data_quality 的缺口描述。\n\n"
        "输出格式：Markdown。\n"
        "\n"
        "推荐输出结构（必须严格遵守小节标题）：\n"
        "## 0) 口径与数据覆盖\n"
        "## 1) 关键结论（<=5）\n"
        "## 2) P0 动作清单（先做）\n"
        "## 3) P1 动作清单（随后做）\n"
        "## 4) 关键词主题（n-gram）抓重点\n"
        "## 5) 风险与回滚条件\n"
    )

    user_prompt = (
        f"请基于以下输入生成《店铺广告优化建议报告》：\n\n"
        f"- shop: {shop}\n"
        f"- stage_profile: {stage}\n\n"
        "【run_context（结构化摘要，优先用它抓重点）】\n"
        "```json\n"
        + json.dumps(run_context or {}, ensure_ascii=False)
        + "\n```\n\n"
        "【ai/data_quality.md（摘要+维度可用性）】\n"
        "```md\n"
        + (data_quality_snippet_md or "").strip()
        + "\n```\n\n"
        "【dashboard/asin_focus.csv（Top 列表：产品立体语境）】\n"
        "```csv\n"
        + (asin_focus_csv_snippet or "").strip()
        + "\n```\n\n"
        "【dashboard/unlock_scale_tasks.csv（放量解锁任务：跨部门）】\n"
        "```csv\n"
        + (unlock_tasks_csv_snippet or "").strip()
        + "\n```\n\n"
        "【dashboard/budget_transfer_plan.csv（预算迁移净表：估算）】\n"
        "```csv\n"
        + (budget_transfer_csv_snippet or "").strip()
        + "\n```\n\n"
        "【dashboard/keyword_topics_action_hints.csv（关键词主题建议：reduce/scale）】\n"
        "```csv\n"
        + (keyword_topics_csv_snippet or "").strip()
        + "\n```\n\n"
        "【dashboard/action_board.csv（Top 列表）】\n"
        "```csv\n"
        + (action_board_csv_snippet or "").strip()
        + "\n```\n\n"
        "要求（严格遵守；不符合则视为失败）：\n"
        "1) 只输出六个小节（`## 0)`~`## 5)`），不得添加任何其它标题/前言/分隔线。\n"
        "2) 每个小节必须是 bullet 列表；每条 bullet 必须紧跟一行 `证据：`（带来源文件与 `=`）。\n"
        "3) `## 0)` 第一条必须写清楚：店铺/时间范围/阶段配置（目标ACoS等，如有）。\n"
        "4) `## 1)` 关键结论最多 5 条；`## 2)` P0 最多 8 条；`## 3)` P1 最多 8 条（宁缺毋滥）。\n"
        "5) `## 4)` 必须给出 3~5 个 reduce 主题 + 3~5 个 scale 主题，并为每个主题写 next_step。\n"
        "6) 预算迁移请优先使用 run_context.budget_transfer_summary 的总览与 TopN（避免被 budget_transfer_plan 全量淹没）。\n"
        "\n"
        "请严格按以下模板输出（不要增加/删除标题）：\n\n"
        "## 0) 口径与数据覆盖\n"
        "- 店铺/时间范围/阶段配置：...\n"
        "  证据：run_context:date_range=...; run_context:stage_profile=...; dashboard/shop_scorecard.json:target_acos=...\n"
        "- 数据缺口与口径提醒：...\n"
        "  证据：ai/data_quality.md:...=...\n"
        "\n"
        "## 1) 关键结论（<=5）\n"
        "- 结论：...\n"
        "  证据：dashboard/asin_focus.csv:...=...; dashboard/action_board.csv:...=...\n"
        "\n"
        "## 2) P0 动作清单（先做）\n"
        "- 动作：...（范围/对象/预期/阻断/人工确认点）\n"
        "  证据：dashboard/action_board.csv:...=...; dashboard/unlock_scale_tasks.csv:...=...\n"
        "\n"
        "## 3) P1 动作清单（随后做）\n"
        "- 动作：...（范围/对象/预期/阻断/人工确认点）\n"
        "  证据：dashboard/action_board.csv:...=...\n"
        "\n"
        "## 4) 关键词主题（n-gram）抓重点\n"
        "- reduce 主题：...（next_step=...）\n"
        "  证据：dashboard/keyword_topics_action_hints.csv:ngram=...; waste_spend=...\n"
        "- scale 主题：...（next_step=...）\n"
        "  证据：dashboard/keyword_topics_action_hints.csv:ngram=...; sales=...; acos=...\n"
        "\n"
        "## 5) 风险与回滚条件\n"
        "- 风险/回滚：...\n"
        "  证据：ai/data_quality.md:...=...; dashboard/asin_focus.csv:inventory_cover_days_7d=...\n"
    )
    return system_prompt, user_prompt


def _normalize_ai_suggestions_body(md: str) -> str:
    """
    轻量“清洗”AI输出，避免：
    - 在 `## 0)` 前输出一堆前置杂项（标题/分隔线/自我介绍）
    - 用编号列表导致可读性差（运营习惯看 bullet + 证据行）
    - 部分 bullet 缺少证据行，无法复盘
    - 证据行被截断/不含 `=`（无法被机器与人同时复核）
    """
    try:
        if not md:
            return ""
        text = str(md).replace("\r\n", "\n").strip()
        if not text:
            return ""

        # 注意：这里不要包含分号（; / ；），否则会被证据行分段逻辑误拆
        placeholder_evidence = "ai/data_quality.md:missing_evidence=1（需人工确认，请回看 data_quality 与各 CSV 证据列）"

        def _fix_evidence_line(line: str) -> str:
            """
            证据行规范化（尽量不改变原意，只让它更可复盘/更稳定）：
            - 把 `≥/≤` 统一为 `>=/<=`（避免缺少 `=` 导致校验失败）
            - 对不含 `=` 的片段补 `note=`（例如 ai/data_quality.md:自然拆分字段不完整）
            - 若整行仍不含 `=`（例如被截断成 `证据：ai/data`），回退到占位证据
            """
            try:
                s = str(line or "")
                if "证据" not in s:
                    return s

                # 统一冒号与不等号符号
                s = s.replace("证据:", "证据：").replace("≥", ">=").replace("≤", "<=")
                if "证据：" not in s:
                    return s

                prefix, rest = s.split("证据：", 1)
                rest = str(rest or "").strip()
                if not rest:
                    return prefix + "证据：" + placeholder_evidence

                parts = re.split(r"[;；]", rest)
                cleaned_parts: List[str] = []
                for part in parts:
                    part = str(part or "").strip()
                    if not part:
                        continue
                    if "=" in part:
                        cleaned_parts.append(part)
                        continue
                    # 形如 source:xxx -> source:note=xxx（补齐 `=`，保留语义）
                    m = re.match(r"^([^:：]+)[:：](.+)$", part)
                    if m:
                        src = str(m.group(1) or "").strip()
                        val = str(m.group(2) or "").strip()
                        if src and val:
                            cleaned_parts.append(f"{src}:note={val}")
                            continue
                    cleaned_parts.append(f"note={part}")

                cleaned = "; ".join([p for p in cleaned_parts if p]).strip()
                # 必须至少包含 1 个“来源文件:字段=值”，否则视为不可复盘（例如被截断成 note=ai/data）
                if ("run_context" not in cleaned) and ("ai/data_quality.md" not in cleaned) and ("dashboard/" not in cleaned):
                    cleaned = placeholder_evidence
                if "=" not in cleaned:
                    cleaned = placeholder_evidence
                return prefix + "证据：" + cleaned
            except Exception:
                return str(line or "")

        # 1) 去掉 `## 0)` 之前的前置杂项
        anchors = [
            "## 0) 口径与数据覆盖",
            "##0) 口径与数据覆盖",
        ]
        for a in anchors:
            idx = text.find(a)
            if idx >= 0:
                text = text[idx:]
                break

        # 2) 把“编号列表”改成 bullet，保证一致性
        lines0 = text.splitlines()
        lines: List[str] = []
        in_code_block = False
        for raw in lines0:
            line = str(raw or "").rstrip()
            if not line.strip():
                lines.append("")
                continue
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                lines.append(line)
                continue
            if in_code_block:
                lines.append(line)
                continue
            # 证据行：先做一次规范化（避免后面被截断/不含 `=` 导致不可复盘）
            if "证据" in line:
                line = _fix_evidence_line(line)
            # 1. xxx / 1) xxx
            m = re.match(r"^\s*\d+[\.\)]\s+(.*)$", line)
            if m and not line.lstrip().startswith("##"):
                lines.append("- " + str(m.group(1) or "").strip())
                continue
            # 一、xxx / 二.xxx
            m2 = re.match(r"^\s*[一二三四五六七八九十]+[、\.\)]\s*(.*)$", line)
            if m2 and not line.lstrip().startswith("##"):
                lines.append("- " + str(m2.group(1) or "").strip())
                continue
            lines.append(line)

        # 3) 确保每条 bullet 后都有“证据行”
        out: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            out.append(line)
            if line.lstrip().startswith("- "):
                # 若 bullet 自带证据（同一行），跳过
                if "证据：" in line:
                    i += 1
                    continue

                # lookahead：在该 bullet 的“块”内寻找证据行（直到下一个 bullet/标题）
                j = i + 1
                has_evidence = False
                while j < len(lines):
                    nxt = str(lines[j] or "")
                    if nxt.lstrip().startswith("- ") or nxt.lstrip().startswith("##"):
                        break
                    if "证据：" in nxt:
                        # 如果证据行存在但被截断/不含 `=`，也视为“不可复盘”，改为占位证据
                        fixed = _fix_evidence_line(nxt)
                        if fixed != nxt:
                            lines[j] = fixed
                        has_evidence = ("=" in str(lines[j] or ""))
                        break
                    j += 1
                if not has_evidence:
                    out.append("  证据：" + placeholder_evidence)
            i += 1

        cleaned = "\n".join(out).strip()
        return cleaned + "\n"
    except Exception:
        return str(md or "").strip() + "\n"


def write_ai_suggestions_for_shop(
    shop_dir: Path,
    stage: str,
    prefix: str = "LLM",
    max_asins: int = 40,
    max_actions: int = 200,
    timeout: int = 180,
    prompt_only: bool = False,
) -> bool:
    """
    为单店铺生成 ai/ai_suggestions.md。

    返回：
    - True：成功生成（或已写入）
    - False：跳过/失败（但不抛异常）
    """

    def _write_skipped(ai_dir: Path, reason: str) -> None:
        """
        当用户显式开启 --ai-report 但未配置环境变量时，输出一个“可点击可解释”的占位文件，
        避免 START_HERE 的链接指向空文件导致困惑。
        """
        try:
            ai_dir.mkdir(parents=True, exist_ok=True)
            msg = (
                "# AI 建议报告（未生成）\n\n"
                f"- generated_at: `{dt.datetime.now().isoformat(timespec='seconds')}`\n"
                f"- reason: {reason}\n\n"
                "## 如何启用\n\n"
                "1) 复制并填写 `.env.example`：\n\n"
                "```bash\n"
                "cp .env.example .env\n"
                "# 编辑 .env：填入 LLM_API_KEY / LLM_MODEL\n"
                "```\n\n"
                "2) 重新运行（示例）：\n\n"
                "```bash\n"
                "python main.py --input-dir reports --out-dir output --ai-report\n"
                "```\n\n"
                "3) 如果你暂时不想配置密钥：可以直接打开 `ai/ai_suggestions_prompt.md`，把里面的 system/user 提示词复制到 ChatGPT/Claude 手工生成。\n\n"
                "> 注意：本文件仅说明“未生成原因”。当配置齐全后，会被真实的 AI 输出覆盖。\n"
            )
            (ai_dir / "ai_suggestions.md").write_text(msg, encoding="utf-8")
        except Exception:
            return

    try:
        if shop_dir is None or not shop_dir.exists():
            return False

        repo_root = _repo_root()
        _load_dotenv_if_present(repo_root / ".env")

        # 读取输入文件
        ai_dir = shop_dir / "ai"
        ai_bundle_path = ai_dir / "ai_input_bundle.json"
        if not ai_bundle_path.exists():
            ai_bundle_path = shop_dir / "ai_input_bundle.json"
        bundle = _read_json(ai_bundle_path)
        if not bundle:
            return False

        shop = str(bundle.get("shop") or shop_dir.name).strip()

        dq_md = _read_text(ai_dir / "data_quality.md", max_chars=0)
        dq_snippet = _extract_data_quality_snippet(dq_md, max_chars=6000)

        # 构建更“短而有效”的输入包（避免 token 爆炸）
        run_ctx = _build_run_context_json(shop_dir=shop_dir)
        asin_focus_snip = _asin_focus_snippet(shop_dir / "dashboard" / "asin_focus.csv", max_rows=int(max_asins or 0))
        unlock_tasks_snip = _unlock_scale_tasks_snippet(shop_dir / "dashboard" / "unlock_scale_tasks.csv", max_rows=30)
        # 预算迁移表在部分店铺可能很长，这里只喂 TopN + 总览（总览在 run_context.budget_transfer_summary）
        budget_transfer_snip = _budget_transfer_plan_snippet(shop_dir / "dashboard" / "budget_transfer_plan.csv", max_rows=25)
        keyword_topics_snip = _keyword_topics_action_hints_snippet(
            shop_dir / "dashboard" / "keyword_topics_action_hints.csv", top_reduce=8, top_scale=8
        )
        action_snippet = _action_board_snippet(shop_dir / "dashboard" / "action_board.csv", max_rows=int(max_actions or 0))

        system_prompt, user_prompt = build_ai_suggestions_prompt(
            shop=shop,
            stage=str(stage or bundle.get("stage_profile") or "").strip(),
            run_context=run_ctx,
            data_quality_snippet_md=dq_snippet,
            asin_focus_csv_snippet=asin_focus_snip,
            unlock_tasks_csv_snippet=unlock_tasks_snip,
            budget_transfer_csv_snippet=budget_transfer_snip,
            keyword_topics_csv_snippet=keyword_topics_snip,
            action_board_csv_snippet=action_snippet,
        )

        # 先写 prompt 留档（即使后面没配置 key/model，你也能手工复制到 ChatGPT/Claude 用）
        ai_dir.mkdir(parents=True, exist_ok=True)
        prompt_md_static = (
            "# AI 建议报告提示词（自动生成留档）\n\n"
            f"- generated_at: `{dt.datetime.now().isoformat(timespec='seconds')}`\n"
            f"- env_prefix: `{prefix}`\n\n"
            "## system\n\n"
            "```text\n"
            + system_prompt.strip()
            + "\n```\n\n"
            "## user\n\n"
            "```text\n"
            + user_prompt.strip()
            + "\n```\n"
        )
        (ai_dir / "ai_suggestions_prompt.md").write_text(prompt_md_static, encoding="utf-8")

        # prompt-only：只生成提示词留档，不调用 LLM（即使已配置 key/model 也不请求）
        if bool(prompt_only):
            _write_skipped(ai_dir, reason="prompt_only=1（本次仅生成提示词留档，不调用 LLM）")
            return True

        # 构建 provider（通过环境变量）
        try:
            from ai_providers import build_chat_provider  # type: ignore
        except Exception:
            _write_skipped(ai_dir, reason="未找到 ai_providers 模块（请确认仓库根目录存在 ai_providers/）")
            return False

        provider = build_chat_provider(prefix=str(prefix or "LLM").strip() or "LLM")
        if provider is None:
            _write_skipped(ai_dir, reason="Provider 构建失败（请检查 LLM_PROVIDER 配置）")
            return False

        # 关键校验：未配置 key/model 时直接跳过，避免无意义的 401/报错输出
        api_key = getattr(provider, "api_key", "") or ""
        model = getattr(provider, "model", "") or ""
        if not str(api_key).strip() or not str(model).strip():
            _write_skipped(ai_dir, reason=f"未配置 {prefix}_API_KEY 或 {prefix}_MODEL")
            return False

        # 调用 LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        content = provider.generate(prompt=user_prompt, messages=messages, temperature=0.1, timeout=int(timeout or 180))

        # provider 信息补齐到 prompt 留档（方便你溯源）
        try:
            prompt_md_with_provider = (
                prompt_md_static
                + "\n"
                + "## provider\n\n"
                + f"- provider: `{type(provider).__name__}`\n"
                + f"- model: `{model}`\n"
            )
            (ai_dir / "ai_suggestions_prompt.md").write_text(prompt_md_with_provider, encoding="utf-8")
        except Exception:
            pass

        # 失败识别：把明显的 HTTP/SDK 错误转换成“生成失败”文件，避免误以为是正文
        txt = str(content or "").strip()
        err = ""
        if not txt:
            err = "返回为空"
        elif txt.startswith("HTTP "):
            err = txt[:200]
        elif ("调用失败" in txt) or ("请求失败" in txt) or ("响应解析失败" in txt):
            err = txt[:200]

        if err:
            _write_skipped(ai_dir, reason=f"AI 生成失败：{err}")
            return False

        # 清洗：去掉前置杂项/编号列表，补齐缺失证据行（避免“看着乱/难复盘”）
        txt = _normalize_ai_suggestions_body(txt)

        report_md = (
            "# AI 建议报告（自动生成）\n\n"
            f"- generated_at: `{dt.datetime.now().isoformat(timespec='seconds')}`\n"
            f"- provider: `{type(provider).__name__}`\n"
            f"- model: `{model}`\n\n"
            "> 注意：本文件由 AI 生成，仅用于“解释与写作”。事实口径请以 `dashboard/*.csv` 与 `ai/*.json` 为准。\n\n"
            + txt
            + "\n"
        )
        (ai_dir / "ai_suggestions.md").write_text(report_md, encoding="utf-8")
        return True
    except Exception:
        return False
