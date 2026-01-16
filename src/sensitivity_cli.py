# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from lifecycle.lifecycle import LifecycleConfig, build_lifecycle_for_shop, build_lifecycle_windows_for_shop
from lifecycle.lifecycle_settings import load_lifecycle_config, merge_lifecycle_overrides
from ingest.loader import load_product_analysis
from core.md import df_to_md_table
from core.schema import CAN


def _parse_int_list(s: str, default: List[int]) -> List[int]:
    try:
        xs = [int(x.strip()) for x in str(s).split(",") if x.strip()]
        return xs if xs else default
    except Exception:
        return default


def _parse_float_list(s: str, default: List[float]) -> List[float]:
    try:
        xs = [float(x.strip()) for x in str(s).split(",") if x.strip()]
        return xs if xs else default
    except Exception:
        return default


def _grid(
    roll_days: List[int],
    launch_days: List[int],
    mature_ratio: List[float],
    decline_ratio: List[float],
    cycle_oos_days: List[int],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for rd in roll_days:
        for ld in launch_days:
            for mr in mature_ratio:
                for dr in decline_ratio:
                    for co in cycle_oos_days:
                        out.append(
                            {
                                "roll_days": int(rd),
                                "launch_days": int(ld),
                                "mature_ratio": float(mr),
                                "decline_ratio": float(dr),
                                "new_cycle_oos_days": int(co),
                            }
                        )
    return out


def _summarize_combo(
    shop: str,
    pa_shop: pd.DataFrame,
    windows_days: List[int],
    cfg: LifecycleConfig,
) -> Dict[str, object]:
    """
    对单个参数组合输出“可比”的汇总指标（用于敏感性检查）。
    """
    try:
        daily, seg, board = build_lifecycle_for_shop(pa_shop, shop=shop, cfg=cfg)
        win = build_lifecycle_windows_for_shop(daily, seg, board, windows_days=windows_days)
    except Exception:
        return {
            "shop": shop,
            **asdict(cfg),
            "asin_count": 0,
            "cycle_count": 0,
            "phase_counts_current_json": "[]",
            "phase_days_json": "[]",
            "stage_switches_avg": 0.0,
            "prelaunch_days_avg": 0.0,
            "prelaunch_spend_avg": 0.0,
            "days_stock_to_first_sale_avg": 0.0,
            "oos_with_ad_spend_asin_pct": 0.0,
            "oos_with_sessions_asin_pct": 0.0,
            "presale_order_asin_pct": 0.0,
        }

    asin_count = int(board["asin"].nunique()) if board is not None and not board.empty and "asin" in board.columns else 0
    cycle_count = 0
    try:
        if daily is not None and not daily.empty and "asin" in daily.columns and "cycle_id" in daily.columns:
            cycle_count = int(daily[["asin", "cycle_id"]].drop_duplicates().shape[0])
    except Exception:
        cycle_count = 0

    # 当前阶段分布
    phase_counts = []
    try:
        if board is not None and not board.empty and "current_phase" in board.columns:
            phase_counts = (
                board.groupby("current_phase", dropna=False).size().reset_index(name="asin_count").sort_values("asin_count", ascending=False).to_dict(orient="records")
            )
    except Exception:
        phase_counts = []

    # 阶段累计天数（segments）
    phase_days = []
    stage_switches_avg = 0.0
    try:
        if seg is not None and not seg.empty and "phase" in seg.columns and "days" in seg.columns:
            phase_days = (
                seg.groupby("phase", dropna=False)["days"].sum().reset_index().rename(columns={"days": "days_sum"}).sort_values("days_sum", ascending=False).to_dict(orient="records")
            )
        # 阶段跳变：每个 asin+cycle 的 segment 数 - 1
        if seg is not None and not seg.empty and {"asin", "cycle_id", "segment_id"}.issubset(seg.columns):
            grp = seg.groupby(["asin", "cycle_id"], dropna=False)["segment_id"].nunique()
            stage_switches_avg = float((grp - 1).clip(lower=0).mean()) if len(grp) else 0.0
    except Exception:
        phase_days = []
        stage_switches_avg = 0.0

    # 主口径窗口指标（since_first_stock_to_date）
    prelaunch_days_avg = 0.0
    prelaunch_spend_avg = 0.0
    days_stock_to_first_sale_avg = 0.0
    oos_with_ad_spend_asin_pct = 0.0
    oos_with_sessions_asin_pct = 0.0
    presale_order_asin_pct = 0.0
    try:
        main = pd.DataFrame()
        if win is not None and not win.empty and "window_type" in win.columns:
            main = win[win["window_type"] == "since_first_stock_to_date"].copy()
            if main.empty:
                main = win[win["window_type"] == "cycle_to_date"].copy()
        if not main.empty:
            # 平均 pre_launch 消耗/耗时
            if "prelaunch_days" in main.columns:
                prelaunch_days_avg = float(pd.to_numeric(main["prelaunch_days"], errors="coerce").fillna(0.0).mean())
            if "prelaunch_ad_spend" in main.columns:
                prelaunch_spend_avg = float(pd.to_numeric(main["prelaunch_ad_spend"], errors="coerce").fillna(0.0).mean())
            if "days_stock_to_first_sale" in main.columns:
                days_stock_to_first_sale_avg = float(pd.to_numeric(main["days_stock_to_first_sale"], errors="coerce").fillna(0.0).mean())

            # 断货异常/预售：按 ASIN 的占比（更适合“敏感性检查”）
            def _pct(col: str) -> float:
                if col not in main.columns:
                    return 0.0
                x = pd.to_numeric(main[col], errors="coerce").fillna(0.0)
                if len(x) == 0:
                    return 0.0
                return float((x > 0).mean())

            oos_with_ad_spend_asin_pct = _pct("oos_with_ad_spend_days")
            oos_with_sessions_asin_pct = _pct("oos_with_sessions_days")
            presale_order_asin_pct = _pct("presale_order_days")
    except Exception:
        pass

    return {
        "shop": shop,
        "roll_days": int(cfg.roll_days),
        "launch_days": int(cfg.launch_days),
        "mature_ratio": float(cfg.mature_ratio),
        "decline_ratio": float(cfg.decline_ratio),
        "new_cycle_oos_days": int(cfg.new_cycle_oos_days),
        "asin_count": asin_count,
        "cycle_count": cycle_count,
        "stage_switches_avg": round(stage_switches_avg, 4),
        "prelaunch_days_avg": round(prelaunch_days_avg, 4),
        "prelaunch_spend_avg": round(prelaunch_spend_avg, 4),
        "days_stock_to_first_sale_avg": round(days_stock_to_first_sale_avg, 4),
        "oos_with_ad_spend_asin_pct": round(oos_with_ad_spend_asin_pct, 6),
        "oos_with_sessions_asin_pct": round(oos_with_sessions_asin_pct, 6),
        "presale_order_asin_pct": round(presale_order_asin_pct, 6),
        "phase_counts_current_json": json.dumps(phase_counts, ensure_ascii=False),
        "phase_days_json": json.dumps(phase_days, ensure_ascii=False),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="reports", help="输入目录（包含 productListing.xlsx 与 产品分析/）")
    ap.add_argument("--out-dir", default="output/lifecycle_sensitivity", help="输出目录")
    ap.add_argument("--only-shop", action="append", default=[], help="只处理指定店铺（可重复传参）")
    ap.add_argument("--windows", default="7,14,30", help="窗口对比（逗号分隔），默认 7,14,30")
    ap.add_argument("--lifecycle-config", default="", help="生命周期参数 JSON（不传则用包内默认 lifecycle_config.json）")

    # 网格参数（默认给一组“轻量但够用”的组合，避免跑太久）
    ap.add_argument("--grid-roll-days", default="7", help="rolling 天数列表，如 7 或 7,14")
    ap.add_argument("--grid-launch-days", default="7,14", help="launch 天数列表，如 7,14,21")
    ap.add_argument("--grid-mature-ratio", default="0.8,0.85,0.9", help="成熟阈值列表，如 0.8,0.85,0.9")
    ap.add_argument("--grid-decline-ratio", default="0.6,0.65", help="衰退阈值列表，如 0.6,0.65,0.7")
    ap.add_argument("--grid-cycle-oos-days", default="7,14", help="断货新周期阈值列表，如 7,14,21")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows_days = _parse_int_list(args.windows, [7, 14, 30])

    # 载入产品分析
    try:
        pa = load_product_analysis(Path(args.input_dir) / "产品分析")
    except Exception as e:
        print(f"[ERR] 读取产品分析失败: {e}")
        return 2
    if pa.empty or CAN.shop not in pa.columns:
        print("[ERR] 产品分析为空或缺少店铺列")
        return 2

    shops = sorted({str(x).strip() for x in pa[CAN.shop].dropna().astype(str).tolist() if str(x).strip()})
    if args.only_shop:
        only = {s.strip() for s in args.only_shop if s.strip()}
        shops = [s for s in shops if s in only]
    if not shops:
        print("[ERR] 未找到可处理的店铺")
        return 2

    default_cfg_path = Path(__file__).resolve().parent / "lifecycle" / "lifecycle_config.json"
    cfg_path = Path(str(args.lifecycle_config).strip()) if str(args.lifecycle_config).strip() else default_cfg_path
    base_cfg = load_lifecycle_config(cfg_path)

    grid = _grid(
        roll_days=_parse_int_list(args.grid_roll_days, [7]),
        launch_days=_parse_int_list(args.grid_launch_days, [7, 14]),
        mature_ratio=_parse_float_list(args.grid_mature_ratio, [0.8, 0.85, 0.9]),
        decline_ratio=_parse_float_list(args.grid_decline_ratio, [0.6, 0.65]),
        cycle_oos_days=_parse_int_list(args.grid_cycle_oos_days, [7, 14]),
    )

    all_rows: List[Dict[str, object]] = []
    for shop in shops:
        pa_shop = pa[pa[CAN.shop] == shop].copy()
        if pa_shop.empty:
            continue
        for ov in grid:
            cfg = merge_lifecycle_overrides(base_cfg, {k: (v if v is not None else None) for k, v in ov.items()})  # type: ignore[arg-type]
            row = _summarize_combo(shop=shop, pa_shop=pa_shop, windows_days=windows_days, cfg=cfg)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[ERR] 未生成任何敏感性结果")
        return 2

    out_csv = out_dir / "lifecycle_sensitivity_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 简单的阅读入口：按“阶段跳变更少 + prelaunch花费更低”排序展示 top
    try:
        view = df.copy()
        view["score"] = (
            pd.to_numeric(view.get("stage_switches_avg"), errors="coerce").fillna(0.0) * 10
            + pd.to_numeric(view.get("prelaunch_spend_avg"), errors="coerce").fillna(0.0) * 0.1
        )
        view = view.sort_values(["shop", "score"], ascending=[True, True])
        md = out_dir / "README.md"
        lines = [
            "# 生命周期阈值敏感性检查（输出）",
            "",
            f"- 汇总表：`{out_csv.name}`",
            "",
            "## 你应该怎么用",
            "- 先按 `shop` 过滤，看同一家店不同参数组合下：阶段分布是否符合运营直觉、pre_launch耗时/花费是否合理。",
            "- 重点关注：`stage_switches_avg`（阶段抖动）、`days_stock_to_first_sale_avg`（到货后启动速度）、`prelaunch_spend_avg`（点火成本）。",
            "",
            "## 每店铺 Top 5（按一个简单 score 排序，仅供快速浏览）",
            "",
        ]
        for shop, g in view.groupby("shop", dropna=False, sort=False):
            lines.append(f"### {shop}")
            top = g.head(5)
            cols = ["roll_days", "launch_days", "mature_ratio", "decline_ratio", "new_cycle_oos_days", "stage_switches_avg", "prelaunch_spend_avg", "days_stock_to_first_sale_avg"]
            lines.append(df_to_md_table(top, columns=cols, max_rows=5))
            lines.append("")
        md.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

    print(f"[OK] wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
