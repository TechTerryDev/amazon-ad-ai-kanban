# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from src.analysis.pipeline import run
from src.lifecycle.lifecycle_settings import load_lifecycle_config, merge_lifecycle_overrides


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        default="data/input",
        help="输入目录（包含 广告数据/ 产品分析/ 产品映射/；兼容 ad/ productListing.xlsx 旧结构）",
    )
    ap.add_argument("--out-dir", default="data/output", help="输出目录（默认会在其下创建一个时间戳 run 子目录）")
    ap.add_argument("--no-timestamp", action="store_true", help="禁用时间戳 run 目录（直接输出到 --out-dir）")
    ap.add_argument("--stage", default="growth", choices=["launch", "growth", "profit"], help="阶段配置：影响阈值与动作幅度")
    ap.add_argument("--only-shop", action="append", default=[], help="只处理指定店铺（可重复传参）")
    ap.add_argument("--days", type=int, default=0, help="只分析最近 N 天（以报表内最大日期为准；0 表示不过滤）")
    ap.add_argument("--date-start", default="", help="开始日期（YYYY-MM-DD，可选）")
    ap.add_argument("--date-end", default="", help="结束日期（YYYY-MM-DD，可选）")
    ap.add_argument("--windows", default="7,14,30", help="多窗口对比（逗号分隔），例如 7,14,30")
    ap.add_argument("--lifecycle-daily", action="store_true", help="输出 lifecycle_daily.csv（按日明细；文件可能很大）")
    ap.add_argument("--lifecycle-daily-days", type=int, default=0, help="lifecycle_daily 只输出最近 N 天（0 表示不裁剪）")
    ap.add_argument("--lifecycle-config", default="", help="生命周期参数 JSON（不传则用包内默认 lifecycle_config.json）")
    ap.add_argument("--roll-days", type=int, default=0, help="生命周期 rolling 天数（覆盖配置；0 表示不覆盖）")
    ap.add_argument("--launch-days", type=int, default=0, help="生命周期 launch 天数（覆盖配置；0 表示不覆盖）")
    ap.add_argument("--mature-ratio", type=float, default=0.0, help="成熟阈值（相对峰值比例，例如 0.85；0 表示不覆盖）")
    ap.add_argument("--decline-ratio", type=float, default=0.0, help="衰退阈值（相对峰值比例，例如 0.65；0 表示不覆盖）")
    ap.add_argument("--cycle-oos-days", type=int, default=0, help="断货>=N天后到货算新周期（覆盖配置；0 表示不覆盖）")
    ap.add_argument(
        "--no-report",
        action="store_true",
        help="不生成任何报告类输出（reports/*.md + reports/*.html + ai/report.md 等都不生成；仍会输出 CSV/JSON）",
    )
    ap.add_argument(
        "--no-full-report",
        action="store_true",
        help="跳过 reporting 合成版（ai/report.md + figures/ + ops/keyword_playbook.xlsx 等），仍生成 reports/dashboard.* 与 CSV/JSON",
    )
    ap.add_argument(
        "--ai-report",
        action="store_true",
        help="生成 ai/ai_suggestions.md（需要配置 LLM_* 环境变量；默认不启用，避免误耗 token）",
    )
    ap.add_argument(
        "--ai-dashboard",
        action="store_true",
        help="生成 ai/ai_dashboard_suggestions.json（双Agent；用于 dashboard 决策建议，可选）",
    )
    ap.add_argument(
        "--ai-dashboard-multiagent",
        action="store_true",
        help="生成 ai/ai_dashboard_suggestions.json（LangGraph+Guardrails+Promptfoo 方案；多轮思考）",
    )
    ap.add_argument(
        "--ai-prompt-only",
        action="store_true",
        help="仅生成 AI 提示词留档（ai/ai_suggestions_prompt.md / ai_dashboard_prompt.md / ai_dashboard_multiagent_prompt.md），不调用 LLM",
    )
    ap.add_argument("--ai-prefix", default="LLM", help="AI 环境变量前缀（默认 LLM；对应 {PREFIX}_API_KEY/{PREFIX}_MODEL 等）")
    ap.add_argument("--ai-max-asins", type=int, default=40, help="喂给 AI 的 ASIN 数量上限（默认 40）")
    ap.add_argument("--ai-max-actions", type=int, default=200, help="喂给 AI 的 action_board 行数上限（默认 200）")
    ap.add_argument("--ai-timeout", type=int, default=180, help="AI 请求超时（秒，默认 180）")
    ap.add_argument(
        "--output-profile",
        default="minimal",
        choices=["minimal", "full"],
        help="输出档位：minimal=只保留验收必需文件；full=输出所有中间表（便于深挖/排查）",
    )
    ap.add_argument(
        "--ops-log-root",
        default="",
        help="L0+ 执行回填目录（可选）。建议结构：reports/ops_logs/<shop>/execution_log.xlsx",
    )
    ap.add_argument(
        "--action-review-windows",
        default="7,14",
        help="L0+ 复盘窗口（逗号分隔）。例如 7,14（默认），或 7,14,30",
    )
    args = ap.parse_args()

    windows_days = []
    try:
        windows_days = [int(x.strip()) for x in str(args.windows).split(",") if x.strip()]
    except Exception:
        windows_days = [7, 14, 30]

    review_windows = []
    try:
        review_windows = [int(x.strip()) for x in str(args.action_review_windows).split(",") if x.strip()]
    except Exception:
        review_windows = [7, 14]

    # 生命周期配置：默认用包内的 lifecycle_config.json，CLI 可覆盖
    default_cfg_path = Path(__file__).resolve().parent / "lifecycle" / "lifecycle_config.json"
    cfg_path = Path(str(args.lifecycle_config).strip()) if str(args.lifecycle_config).strip() else default_cfg_path
    lifecycle_cfg = load_lifecycle_config(cfg_path)
    overrides = {
        "roll_days": int(args.roll_days) if int(args.roll_days or 0) > 0 else None,
        "launch_days": int(args.launch_days) if int(args.launch_days or 0) > 0 else None,
        "mature_ratio": float(args.mature_ratio) if float(args.mature_ratio or 0.0) > 0 else None,
        "decline_ratio": float(args.decline_ratio) if float(args.decline_ratio or 0.0) > 0 else None,
        "new_cycle_oos_days": int(args.cycle_oos_days) if int(args.cycle_oos_days or 0) > 0 else None,
    }
    lifecycle_cfg = merge_lifecycle_overrides(lifecycle_cfg, overrides)

    base_out_dir = Path(args.out_dir)
    if bool(args.no_timestamp):
        out_dir = base_out_dir
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base_out_dir / ts

    render_dashboard_md = not bool(args.no_report)
    render_full_report = (not bool(args.no_report)) and (not bool(args.no_full_report))
    run(
        reports_root=Path(args.input_dir),
        out_dir=out_dir,
        stage=args.stage,
        only_shops=args.only_shop or None,
        days=int(args.days or 0),
        date_start=str(args.date_start or "").strip() or None,
        date_end=str(args.date_end or "").strip() or None,
        windows_days=windows_days or [7, 14, 30],
        render_report=bool(render_full_report),
        render_dashboard_md=bool(render_dashboard_md),
        lifecycle_daily=bool(args.lifecycle_daily),
        lifecycle_daily_days=int(args.lifecycle_daily_days or 0),
        lifecycle_cfg=lifecycle_cfg,
        output_profile=str(args.output_profile or "minimal").strip().lower(),
        ops_log_root=Path(str(args.ops_log_root).strip()) if str(args.ops_log_root).strip() else None,
        action_review_windows=review_windows or [7, 14],
        ai_report=bool(args.ai_report),
        ai_prompt_only=bool(args.ai_prompt_only),
        ai_dashboard=bool(args.ai_dashboard),
        ai_dashboard_multiagent=bool(args.ai_dashboard_multiagent),
        ai_prefix=str(args.ai_prefix or "LLM").strip() or "LLM",
        ai_max_asins=int(args.ai_max_asins or 0),
        ai_max_actions=int(args.ai_max_actions or 0),
        ai_timeout=int(args.ai_timeout or 0),
    )
    print(f"[OK] outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
