# -*- coding: utf-8 -*-
"""
Markdown 工具（避免 pandas.to_markdown 依赖 tabulate）。
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd


def md_escape(text: object) -> str:
    s = "" if text is None else str(text)
    s = s.replace("|", "\\|")
    s = s.replace("\n", " ")
    return s


def df_to_md_table(df: pd.DataFrame, columns: Optional[List[str]] = None, max_rows: int = 20) -> str:
    """
    把 DataFrame 转成简易 Markdown 表格（最多 max_rows 行）。
    """
    if df is None or df.empty:
        return "_无数据_"

    view = df.copy()
    if columns:
        cols = [c for c in columns if c in view.columns]
        view = view[cols] if cols else view

    if len(view) > max_rows:
        view = view.head(max_rows)
        truncated = True
    else:
        truncated = False

    cols = list(view.columns)
    header = "| " + " | ".join(md_escape(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in view.iterrows():
        rows.append("| " + " | ".join(md_escape(r.get(c)) for c in cols) + " |")

    out = "\n".join([header, sep] + rows)
    if truncated:
        out += f"\n\n_仅显示前 {max_rows} 行_"
    return out


def md_list(items: Iterable[str], max_items: int = 20, numbered: bool = True) -> str:
    lst = list(items)
    out: List[str] = []
    for i, item in enumerate(lst[:max_items], start=1):
        if numbered:
            out.append(f"{i}. {item}")
        else:
            out.append(f"- {item}")
    if len(lst) > max_items:
        out.append(f"\n_共 {len(lst)} 项，仅显示前 {max_items} 项_")
    return "\n".join(out) if out else "_无数据_"

