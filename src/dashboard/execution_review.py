# -*- coding: utf-8 -*-
"""
L0+ 运营闭环（无唯一 ID 也能做）：

1) 输出 execution_log_template（运营可勾选“是否执行”并回填执行时间与备注）
2) 如果提供了上次/历史 execution_log，则在本次输出中生成 action_review（7/14天效果对比）

注意：
- 没有唯一 ID 时，只能用“组合键”匹配（shop/ad_type/level/action_type/object_name/campaign/ad_group/match_type）
- 匹配失败时不报错，只在 review 里标注原因
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.ads.actions import ActionCandidate
from src.core.schema import CAN
from src.core.utils import parse_date, safe_div


def _safe_str(x: object) -> str:
    try:
        if x is None:
            return ""
        # pandas 读 csv/xlsx 常把空值变成 NaN
        try:
            if pd.isna(x):  # type: ignore[arg-type]
                return ""
        except Exception:
            pass
        s = str(x).strip()
        return "" if s.lower() == "nan" else s
    except Exception:
        return ""


def _safe_int(x: object) -> int:
    try:
        return int(float(x))  # type: ignore[arg-type]
    except Exception:
        return 0


def _action_key_from_row(r: Dict[str, object]) -> str:
    parts = [
        _safe_str(r.get("shop")),
        _safe_str(r.get("ad_type")),
        _safe_str(r.get("level")),
        _safe_str(r.get("action_type")),
        _safe_str(r.get("object_name")),
        _safe_str(r.get("campaign")),
        _safe_str(r.get("ad_group")),
        _safe_str(r.get("match_type")),
    ]
    return "|".join(parts)


def build_execution_log_template(actions: List[ActionCandidate]) -> pd.DataFrame:
    """
    生成运营回填模板（不依赖外部 ID）。
    """
    rows: List[Dict[str, object]] = []
    for a in actions:
        d = asdict(a)
        d["action_key"] = _action_key_from_row(d)
        # 运营回填字段（默认空）
        d["executed"] = 0
        d["executed_at"] = ""
        d["operator"] = ""
        d["note"] = ""
        d["rollback_condition"] = ""
        d["rollback_done"] = 0
        rows.append(d)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    # 输出列顺序（更贴近运营填表）
    cols = [
        "action_key",
        "executed",
        "executed_at",
        "operator",
        "note",
        "rollback_condition",
        "rollback_done",
        # 建议内容
        "priority",
        "level",
        "ad_type",
        "action_type",
        "action_value",
        "object_name",
        "campaign",
        "ad_group",
        "match_type",
        "date_start",
        "date_end",
        "reason",
        "evidence_json",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def write_execution_log_template(shop_dir: Path, actions: List[ActionCandidate]) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    写出运营回填模板（csv + xlsx）与说明文件。
    """
    try:
        ops_dir = shop_dir / "ops"
        ops_dir.mkdir(parents=True, exist_ok=True)

        df = build_execution_log_template(actions)
        csv_path = ops_dir / "execution_log_template.csv"
        xlsx_path = ops_dir / "execution_log_template.xlsx"
        readme_path = ops_dir / "execution_log_README.md"

        if df is None or df.empty:
            pd.DataFrame(columns=["action_key"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            # xlsx 失败也不影响主流程
            try:
                with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
                    df.to_excel(w, index=False, sheet_name="execution_log")
            except Exception:
                xlsx_path = None

        # 简明说明：告诉运营怎么用
        try:
            lines = []
            lines.append("# execution_log 使用说明（L0+）")
            lines.append("")
            lines.append("用途：把“建议动作”变成可复盘闭环。你不需要唯一 ID 也能做：执行靠人工，复盘靠窗口对比。")
            lines.append("")
            lines.append("## 1) 怎么填")
            lines.append("- `executed`：执行填 1，不执行留 0")
            lines.append("- `executed_at`：执行日期（YYYY-MM-DD），用于后续 7/14 天效果对比")
            lines.append("- `operator/note`：执行人/备注（可选）")
            lines.append("- `rollback_condition/rollback_done`：回滚条件与是否已回滚（可选）")
            lines.append("")
            lines.append("## 2) 下次怎么复盘")
            lines.append("1. 把你填好的表另存为 `execution_log.xlsx`（或 csv）")
            lines.append("2. 运行脚本时指定 `--ops-log-root <目录>`（里面按店铺放文件），脚本会输出 `ops/action_review.csv`")
            lines.append("3. 复盘需要“前后窗口”都在数据范围内：如果你运行时加了 `--days 14`，可能导致 before 窗口被裁掉，从而出现 `insufficient_data`。")
            lines.append("")
            lines.append("## 3) 目录约定（建议）")
            lines.append("- 建议你把回填文件放到：`reports/ops_logs/<shop>/execution_log.xlsx`")
            lines.append("- 这样每次跑数都能自动读到上次执行记录")
            lines.append("")
            readme_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

        return (csv_path, xlsx_path, readme_path)
    except Exception:
        return (None, None, None)


def _read_table(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path, engine="openpyxl")
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def find_ops_log_file(ops_log_root: Path, shop: str) -> Optional[Path]:
    """
    在 ops_log_root 下寻找该 shop 的执行回填文件。
    支持：
    - <root>/<shop>/execution_log.xlsx
    - <root>/<shop>/execution_log.csv
    - <root>/execution_log_<shop>.xlsx
    - <root>/execution_log_<shop>.csv
    """
    try:
        if ops_log_root is None:
            return None
        root = Path(ops_log_root)
        if not root.exists():
            return None
        cand = [
            root / shop / "execution_log.xlsx",
            root / shop / "execution_log.csv",
            root / f"execution_log_{shop}.xlsx",
            root / f"execution_log_{shop}.csv",
        ]
        for p in cand:
            if p.exists():
                return p
        return None
    except Exception:
        return None


def load_execution_log(ops_log_root: Optional[Path], reports_root: Optional[Path], shop: str) -> pd.DataFrame:
    """
    读取运营回填表。
    - 优先使用 --ops-log-root
    - 不传时，如果存在 reports/ops_logs，也尝试读取
    """
    try:
        root = Path(ops_log_root) if ops_log_root else None
        if root:
            p = find_ops_log_file(root, shop)
            if p:
                return _read_table(p)
        if reports_root:
            rr = Path(reports_root) / "ops_logs"
            p2 = find_ops_log_file(rr, shop) if rr.exists() else None
            if p2:
                return _read_table(p2)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _match_entity_df(level: str, st: pd.DataFrame, tgt: pd.DataFrame, camp: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    lv = (level or "").strip().lower()
    if lv == "search_term":
        return st
    if lv == "targeting":
        return tgt
    if lv == "campaign":
        return camp
    if lv == "placement":
        return pl
    return pd.DataFrame()


def _filter_entity(df: pd.DataFrame, r: Dict[str, object]) -> pd.DataFrame:
    """
    用组合键过滤到“建议对象”的明细行。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    # 基础：ad_type
    ad_type = _safe_str(r.get("ad_type"))
    if ad_type and CAN.ad_type in out.columns:
        out = out[out[CAN.ad_type].astype(str).str.strip() == ad_type]

    level = _safe_str(r.get("level")).lower()
    obj = _safe_str(r.get("object_name"))
    campaign = _safe_str(r.get("campaign"))
    ad_group = _safe_str(r.get("ad_group"))
    match_type = _safe_str(r.get("match_type"))

    # 按 level 过滤对象字段
    if level == "search_term" and CAN.search_term in out.columns:
        out = out[out[CAN.search_term].astype(str).str.strip() == obj]
        if campaign and CAN.campaign in out.columns:
            out = out[out[CAN.campaign].astype(str).str.strip() == campaign]
        if ad_group and CAN.ad_group in out.columns:
            out = out[out[CAN.ad_group].astype(str).str.strip() == ad_group]
        if match_type and CAN.match_type in out.columns:
            out = out[out[CAN.match_type].astype(str).str.strip() == match_type]

    elif level == "targeting" and CAN.targeting in out.columns:
        out = out[out[CAN.targeting].astype(str).str.strip() == obj]
        if campaign and CAN.campaign in out.columns:
            out = out[out[CAN.campaign].astype(str).str.strip() == campaign]
        if ad_group and CAN.ad_group in out.columns:
            out = out[out[CAN.ad_group].astype(str).str.strip() == ad_group]
        if match_type and CAN.match_type in out.columns:
            out = out[out[CAN.match_type].astype(str).str.strip() == match_type]

    elif level == "campaign" and CAN.campaign in out.columns:
        out = out[out[CAN.campaign].astype(str).str.strip() == (campaign or obj)]

    elif level == "placement" and CAN.placement in out.columns:
        out = out[out[CAN.placement].astype(str).str.strip() == obj]
        if campaign and CAN.campaign in out.columns:
            out = out[out[CAN.campaign].astype(str).str.strip() == campaign]

    return out


def _sum_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {"impressions": 0.0, "clicks": 0.0, "spend": 0.0, "sales": 0.0, "orders": 0.0, "acos": 0.0, "cvr": 0.0}
    imp = float(pd.to_numeric(df.get(CAN.impressions, 0.0), errors="coerce").fillna(0.0).sum())
    clk = float(pd.to_numeric(df.get(CAN.clicks, 0.0), errors="coerce").fillna(0.0).sum())
    spend = float(pd.to_numeric(df.get(CAN.spend, 0.0), errors="coerce").fillna(0.0).sum())
    sales = float(pd.to_numeric(df.get(CAN.sales, 0.0), errors="coerce").fillna(0.0).sum())
    orders = float(pd.to_numeric(df.get(CAN.orders, 0.0), errors="coerce").fillna(0.0).sum())
    return {
        "impressions": round(imp, 2),
        "clicks": round(clk, 2),
        "spend": round(spend, 2),
        "sales": round(sales, 2),
        "orders": round(orders, 2),
        "acos": round(safe_div(spend, sales) if sales > 0 else 0.0, 4),
        "cvr": round(safe_div(orders, clk) if clk > 0 else 0.0, 4),
    }


def _slice_by_date(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    if df is None or df.empty or CAN.date not in df.columns:
        return pd.DataFrame()
    x = df[df[CAN.date].notna()].copy()
    return x[(x[CAN.date] >= start) & (x[CAN.date] <= end)].copy()


def build_action_review(
    execution_log: pd.DataFrame,
    st: pd.DataFrame,
    tgt: pd.DataFrame,
    camp: pd.DataFrame,
    pl: pd.DataFrame,
    windows_days: List[int],
) -> pd.DataFrame:
    """
    对已执行动作做 7/14 天效果对比（after vs before）。
    """
    if execution_log is None or execution_log.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []

    # 标准化列名（容错）
    ex = execution_log.copy()
    ex.columns = [str(c).strip() for c in ex.columns]
    if "executed" not in ex.columns:
        return pd.DataFrame()
    if "executed_at" not in ex.columns:
        return pd.DataFrame()

    for _, rr in ex.iterrows():
        r = rr.to_dict()
        if _safe_int(r.get("executed", 0)) <= 0:
            continue
        d0 = parse_date(r.get("executed_at"))
        if d0 is None:
            continue

        level = _safe_str(r.get("level")).lower()
        base_df = _match_entity_df(level, st, tgt, camp, pl)
        if base_df is None or base_df.empty:
            # 不支持的 level（例如 asin/product_side）先跳过
            continue

        entity_df = _filter_entity(base_df, r)
        if entity_df is None or entity_df.empty:
            # 匹配不到对象：也输出一行，方便排查
            for n in windows_days:
                rows.append(
                    {
                        "action_key": _safe_str(r.get("action_key")) or _action_key_from_row(r),
                        "window_days": int(n),
                        "status": "not_found",
                        "reason": "无法用组合键匹配到对象（可能是名称变更/字段缺失）",
                        "executed_at": str(d0),
                        "level": _safe_str(r.get("level")),
                        "ad_type": _safe_str(r.get("ad_type")),
                        "action_type": _safe_str(r.get("action_type")),
                        "object_name": _safe_str(r.get("object_name")),
                        "campaign": _safe_str(r.get("campaign")),
                        "ad_group": _safe_str(r.get("ad_group")),
                        "match_type": _safe_str(r.get("match_type")),
                    }
                )
            continue

        # 对每个 window_days 计算 before/after
        for n in windows_days:
            n2 = int(n or 0)
            if n2 <= 0:
                continue
            after_start = d0
            after_end = d0 + dt.timedelta(days=n2 - 1)
            before_end = d0 - dt.timedelta(days=1)
            before_start = before_end - dt.timedelta(days=n2 - 1)

            before = _slice_by_date(entity_df, before_start, before_end)
            after = _slice_by_date(entity_df, after_start, after_end)

            m_before = _sum_metrics(before)
            m_after = _sum_metrics(after)

            rows.append(
                {
                    "action_key": _safe_str(r.get("action_key")) or _action_key_from_row(r),
                    "window_days": int(n2),
                    "status": "ok" if (not before.empty and not after.empty) else ("insufficient_data" if (before.empty or after.empty) else "unknown"),
                    "reason": "",
                    "executed_at": str(d0),
                    "level": _safe_str(r.get("level")),
                    "ad_type": _safe_str(r.get("ad_type")),
                    "action_type": _safe_str(r.get("action_type")),
                    "action_value": _safe_str(r.get("action_value")),
                    "object_name": _safe_str(r.get("object_name")),
                    "campaign": _safe_str(r.get("campaign")),
                    "ad_group": _safe_str(r.get("ad_group")),
                    "match_type": _safe_str(r.get("match_type")),
                    # before
                    "before_spend": m_before["spend"],
                    "before_sales": m_before["sales"],
                    "before_orders": m_before["orders"],
                    "before_clicks": m_before["clicks"],
                    "before_acos": m_before["acos"],
                    "before_cvr": m_before["cvr"],
                    # after
                    "after_spend": m_after["spend"],
                    "after_sales": m_after["sales"],
                    "after_orders": m_after["orders"],
                    "after_clicks": m_after["clicks"],
                    "after_acos": m_after["acos"],
                    "after_cvr": m_after["cvr"],
                    # delta
                    "delta_spend": round(m_after["spend"] - m_before["spend"], 2),
                    "delta_sales": round(m_after["sales"] - m_before["sales"], 2),
                    "delta_orders": round(m_after["orders"] - m_before["orders"], 2),
                    "delta_acos": round(m_after["acos"] - m_before["acos"], 4),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def write_action_review(
    shop_dir: Path,
    execution_log: pd.DataFrame,
    st: pd.DataFrame,
    tgt: pd.DataFrame,
    camp: pd.DataFrame,
    pl: pd.DataFrame,
    windows_days: List[int],
) -> Optional[Path]:
    """
    写入复盘输出：ops/action_review.csv
    """
    try:
        ops_dir = shop_dir / "ops"
        ops_dir.mkdir(parents=True, exist_ok=True)
        review = build_action_review(execution_log, st=st, tgt=tgt, camp=camp, pl=pl, windows_days=windows_days)
        out_path = ops_dir / "action_review.csv"
        if review is None or review.empty:
            pd.DataFrame(columns=["action_key", "status"]).to_csv(out_path, index=False, encoding="utf-8-sig")
        else:
            review.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path
    except Exception:
        return None
