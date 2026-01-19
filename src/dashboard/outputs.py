# -*- coding: utf-8 -*-
"""
dashboard/ 聚焦层输出（解决“报告太长抓不到重点”的问题）。

设计原则：
- 不依赖额外第三方库（只用 pandas / 标准库）
- 失败不崩：任何一步失败不影响主流程（pipeline 继续输出其它文件）
- 产物“可排序/可筛选/可复盘”：优先输出 CSV/JSON
"""

from __future__ import annotations

import datetime as dt
import csv
import hashlib
import html
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from ads.actions import ActionCandidate
from core.config import StageConfig, get_stage_config
from core.policy import (
    ActionScoringPolicy,
    FocusScoringPolicy,
    InventorySigmoidPolicy,
    OpsPolicy,
    ProfitGuardPolicy,
    SignalScoringPolicy,
    StageScoringPolicy,
)
from core.risk_scoring import (
    acos_risk_probability,
    ad_signal_score,
    calculate_overall_risk,
    cvr_drop_risk_probability,
    oos_risk_probability,
    product_signal_score,
    risk_level,
)
from core.schema import CAN
from core.utils import json_dumps
from dashboard.keyword_topics import (
    annotate_keyword_topic_action_hints,
    build_keyword_topic_action_hints,
    build_keyword_topic_asin_context,
    build_keyword_topic_category_phase_summary,
    build_keyword_topic_segment_top,
    build_keyword_topics,
)


def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def _safe_int(x: object) -> int:
    try:
        return int(float(x))  # type: ignore[arg-type]
    except Exception:
        return 0


def _is_percent_col(col_name: str) -> bool:
    try:
        s = str(col_name or "")
        n = s.lower()
        if any(k in n for k in ("ctr", "cvr", "tacos", "acos", "share", "ratio", "margin", "rate", "pct", "percent")):
            return True
        if any(k in s for k in ("率", "占比", "比例", "毛利率")):
            return True
        return False
    except Exception:
        return False


def _is_int_col(col_name: str) -> bool:
    try:
        s = str(col_name or "")
        n = s.lower()
        if any(
            k in n
            for k in (
                "orders",
                "order",
                "sessions",
                "impressions",
                "clicks",
                "days",
                "day",
                "count",
                "qty",
                "inventory",
                "rank",
                "asin_count",
                "cycle_id",
                "segment_id",
                "num",
            )
        ):
            return True
        if any(k in s for k in ("订单", "点击", "曝光", "天数", "次数", "库存", "数量", "排名", "排行", "ASIN数", "店铺数", "类目数")):
            return True
        return False
    except Exception:
        return False


def _format_number_for_md(col_name: str, value: float) -> str:
    try:
        if _is_percent_col(col_name):
            return f"{float(value) * 100:.2f}%"
        if _is_int_col(col_name):
            if abs(float(value) - round(float(value))) < 1e-6:
                return str(int(round(float(value))))
            return f"{float(value):.2f}"
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _format_md_cell(col_name: str, value: object) -> str:
    try:
        if value is None:
            return ""
        num = pd.to_numeric(value, errors="coerce")
        if pd.notna(num):
            s = _format_number_for_md(col_name, float(num))
        else:
            s = str(value)
    except Exception:
        s = str(value)
    return s.replace("\n", " ").replace("|", "｜")


def _norm_product_category(x: object) -> str:
    """
    统一商品分类口径，避免出现“未分类”和“（未分类）”两套兜底值导致分组重复。
    """
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
    """
    统一生命周期阶段口径（current_phase）。
    """
    try:
        s = str(x or "").strip()
        if not s or s.lower() == "nan":
            return "unknown"
        return s.strip().lower()
    except Exception:
        return "unknown"


def _phase_anchor_id(phase: str) -> str:
    """
    生成稳定的 Phase 锚点 id（用于文件内跳转）。
    例：phase=growth -> phase-growth
    """
    try:
        s = _norm_phase(phase)
        s2 = "".join([c for c in s if c.isalnum() or c in ("_", "-")])
        if not s2:
            s2 = "unknown"
        return f"phase-{s2}"
    except Exception:
        return "phase-unknown"


def _phase_md_link(phase: str, target_md_path: str) -> str:
    """
    生成指向 phase_drilldown 的链接：`[growth](./phase_drilldown.md#phase-growth)`
    """
    p = _norm_phase(phase)
    pid = _phase_anchor_id(p)
    return f"[{p}]({target_md_path}#{pid})"


def _parse_evidence_json(s: str) -> Dict[str, object]:
    try:
        if not s:
            return {}
        return json.loads(s)
    except Exception:
        return {}


def _asin_anchor_id(asin: str) -> str:
    """
    生成稳定的 Markdown 锚点 id（用于文件内跳转）。
    例：ASIN=B0ABC123 -> asin-b0abc123
    """
    try:
        s = str(asin or "").strip().lower()
        s = "".join([c for c in s if c.isalnum()])
        return f"asin-{s}" if s else ""
    except Exception:
        return ""


def _asin_md_link(asin: str, target_md_path: str) -> str:
    """
    生成指向 drilldown 的链接：`[ASIN](./asin_drilldown.md#asin-xxxx)`
    """
    a = str(asin or "").strip().upper()
    aid = _asin_anchor_id(a)
    if not a or not aid:
        return a
    return f"[{a}]({target_md_path}#{aid})"


def _cat_anchor_id(category: str) -> str:
    """
    生成稳定的 Category 锚点 id（用于文件内跳转）。

    类目可能包含中文/空格/特殊字符，直接做 slug 容易不稳定，
    因此采用 md5(category) 的短哈希，保证“可链接 + 稳定 + 不报错”。
    """
    try:
        s = str(category or "").strip()
        if not s:
            s = "（未分类）"
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
        return f"cat-{h}"
    except Exception:
        return "cat-unknown"


def _cat_md_link(category: str, target_md_path: str) -> str:
    """
    生成指向 category_drilldown 的链接：`[类目](./category_drilldown.md#cat-xxxx)`
    """
    c = str(category or "").strip()
    if not c or c.lower() == "nan":
        c = "（未分类）"
    cid = _cat_anchor_id(c)
    return f"[{c}]({target_md_path}#{cid})"


def _rewrite_md_href_to_html_if_exists(href: str, base_dir: Optional[Path]) -> str:
    """
    展示层链接重写（HTML-first）：
    - 当 href 指向本地相对路径的 *.md，且同路径存在 *.html 时，将链接改为 *.html
    - 保留 query / fragment（例如 #anchor），避免破坏文件内跳转

    说明：
    - 这是“展示层”改写，不影响 Markdown 源文件本身。
    - 使用存在性判断，避免生成死链（比如某些 md 没有生成 html）。
    """
    try:
        href_s = str(href or "").strip()
        if not href_s:
            return href_s
        if base_dir is None:
            return href_s

        parts = urlsplit(href_s)
        # 非本地链接（http/https/mailto/...）不处理
        if parts.scheme or parts.netloc:
            return href_s

        path_part = str(parts.path or "")
        if not path_part:
            return href_s

        # 只处理相对路径（绝对路径可能指向系统文件，不应在报告中改写）
        if path_part.startswith("/") or path_part.startswith("\\"):
            return href_s

        p = Path(path_part)
        if p.suffix.lower() != ".md":
            return href_s

        # HTML-first：已知“本项目会生成 HTML 的 Markdown 文件”，允许直接改写（不依赖文件已存在）。
        # 说明：reports/*.html 的生成顺序会先写 dashboard.html，再写其它 report html。
        # 如果这里强依赖 exists()，会导致 dashboard.html 内链仍指向 .md（体验差）。
        force_names = {
            "start_here.md",
            "dashboard.md",
            "asin_drilldown.md",
            "category_drilldown.md",
            "phase_drilldown.md",
            "keyword_topics.md",
            "data_quality.md",
            "ai_suggestions.md",
        }
        base_name = str(p.name or "").strip().lower()
        p_html = p.with_suffix(".html")
        if base_name in force_names:
            return urlunsplit((parts.scheme, parts.netloc, str(p_html), parts.query, parts.fragment))

        # 其他 md：只有在确实存在对应 html 时才改写（避免生成死链）
        if (base_dir / p_html).exists():
            return urlunsplit((parts.scheme, parts.netloc, str(p_html), parts.query, parts.fragment))

        return href_s
    except Exception:
        return str(href or "")


def _md_inline_to_html(s: str, base_dir: Optional[Path] = None) -> str:
    """
    极简 Markdown inline → HTML（只覆盖本项目报告里用到的子集）：
    - `code`
    - [text](href)（reports/*.md 会自动重写为 *.html，保持 HTML 报告内跳转一致）

    说明：
    - 不引入第三方 markdown 库，保持“离线可跑 + 依赖最少”。
    - 这是“展示层”转换：不会改变原始 Markdown 的产物与口径。
    """
    try:
        def _render_link_label(label: str) -> str:
            """
            链接文本里也允许出现 `code`（例如 [`path`](...)），提高可读性。
            说明：这里仅解析 code span，不解析嵌套链接，避免生成非法的 <a><a>...</a></a>。
            """
            try:
                t = str(label or "")
                out2: List[str] = []
                j2 = 0
                while j2 < len(t):
                    ch2 = t[j2]
                    if ch2 == "`":
                        k2 = t.find("`", j2 + 1)
                        if k2 >= 0:
                            code = t[j2 + 1 : k2]
                            out2.append("<code>" + html.escape(code) + "</code>")
                            j2 = k2 + 1
                            continue
                    out2.append(html.escape(ch2))
                    j2 += 1
                return "".join(out2)
            except Exception:
                return html.escape(str(label or ""))

        text = str(s or "")
        out: List[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            # code span
            if ch == "`":
                j = text.find("`", i + 1)
                if j >= 0:
                    code = text[i + 1 : j]
                    out.append("<code>" + html.escape(code) + "</code>")
                    i = j + 1
                    continue
            # link: [text](href)
            if ch == "[":
                j = text.find("]", i + 1)
                if j >= 0 and (j + 1) < len(text) and text[j + 1] == "(":
                    k = text.find(")", j + 2)
                    if k >= 0:
                        label = text[i + 1 : j]
                        href = text[j + 2 : k]
                        href2 = str(href or "").strip()
                        # HTML-first：如果存在对应的 .html，就把 .md 链接改为 .html（避免浏览器打开 .md 很难读）
                        href2 = _rewrite_md_href_to_html_if_exists(href2, base_dir=base_dir)
                        out.append(
                            '<a href="' + html.escape(href2, quote=True) + '">' + _render_link_label(label) + "</a>"
                        )
                        i = k + 1
                        continue
            out.append(html.escape(ch))
            i += 1
        return "".join(out)
    except Exception:
        return html.escape(str(s or ""))


def _md_to_html_body(md_text: str, base_dir: Optional[Path] = None) -> str:
    """
    极简 Markdown block → HTML body（只覆盖本项目 reports/*.md 的结构）。
    支持：
    - #/##/###/#### heading
    - 无序列表（- item）
    - 表格（pipe table）
    - <a id="..."></a> 原样保留（用于锚点跳转）
    """
    try:
        lines = str(md_text or "").replace("\r\n", "\n").splitlines()
        out: List[str] = []
        in_ul = False
        in_table = False

        def _close_ul() -> None:
            nonlocal in_ul
            if in_ul:
                out.append("</ul>")
                in_ul = False

        def _close_table() -> None:
            nonlocal in_table
            if in_table:
                out.append("</tbody></table></div>")
                in_table = False

        def _emit_table_header(header_line: str) -> None:
            nonlocal in_table
            cells = [c.strip() for c in str(header_line).strip().strip("|").split("|")]
            out.append('<div class="table-wrap"><table><thead><tr>')
            for c in cells:
                out.append("<th>" + _md_inline_to_html(c, base_dir=base_dir) + "</th>")
            out.append("</tr></thead><tbody>")
            in_table = True

        def _emit_table_row(row_line: str) -> None:
            cells = [c.strip() for c in str(row_line).strip().strip("|").split("|")]
            out.append("<tr>")
            for c in cells:
                out.append("<td>" + _md_inline_to_html(c, base_dir=base_dir) + "</td>")
            out.append("</tr>")

        i = 0
        while i < len(lines):
            raw = str(lines[i] or "").rstrip()
            line = raw.strip()

            # anchor lines: keep raw HTML for stable in-file jump
            if re.match(r'^<a\s+id="[^"]+"\s*></a>\s*$', line):
                _close_ul()
                _close_table()
                out.append(line)
                i += 1
                continue

            # blank line: close blocks
            if not line:
                _close_ul()
                _close_table()
                out.append("")
                i += 1
                continue

            # allow specific raw HTML blocks (for layout cards/flows in reports)
            if re.match(r"^</?(div|details|summary|span)(\s|>)", line):
                _close_ul()
                _close_table()
                out.append(raw)
                i += 1
                continue

            # horizontal rule
            if line == "---":
                _close_ul()
                _close_table()
                out.append("<hr/>")
                i += 1
                continue

            # heading
            m = re.match(r"^(#{1,6})\s+(.*)$", raw)
            if m:
                _close_ul()
                _close_table()
                lvl = len(m.group(1))
                txt = str(m.group(2) or "").strip()
                out.append(f"<h{lvl}>" + _md_inline_to_html(txt, base_dir=base_dir) + f"</h{lvl}>")
                i += 1
                continue

            # table detection: header + separator line
            if ("|" in raw) and (i + 1) < len(lines):
                sep = str(lines[i + 1] or "").strip()
                if re.match(r"^\|?\s*---", sep) and "|" in sep:
                    _close_ul()
                    _close_table()
                    _emit_table_header(raw)
                    i += 2  # skip header+separator
                    # rows until blank or non-table
                    while i < len(lines):
                        r = str(lines[i] or "").rstrip()
                        if not r.strip():
                            break
                        if "|" not in r:
                            break
                        _emit_table_row(r)
                        i += 1
                    _close_table()
                    continue

            # list item
            if raw.lstrip().startswith("- "):
                _close_table()
                if not in_ul:
                    out.append("<ul>")
                    in_ul = True
                item = raw.lstrip()[2:].strip()
                out.append("<li>" + _md_inline_to_html(item, base_dir=base_dir) + "</li>")
                i += 1
                continue

            # paragraph
            _close_ul()
            _close_table()
            out.append("<p>" + _md_inline_to_html(line, base_dir=base_dir) + "</p>")
            i += 1

        _close_ul()
        _close_table()
        return "\n".join([x for x in out if x is not None])
    except Exception:
        return "<pre>" + html.escape(str(md_text or "")) + "</pre>"


def write_report_html_from_md(md_path: Path, out_path: Path) -> None:
    """
    将 reports/*.md 转成离线可打开的 reports/*.html（保留相对链接与锚点跳转）。
    """
    try:
        if md_path is None or not md_path.exists():
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md_text = md_path.read_text(encoding="utf-8", errors="replace")
        body = _md_to_html_body(md_text, base_dir=out_path.parent)
        title = md_path.stem
        # 顶部导航：尽量在不依赖外部资源的前提下，让 HTML 更像“离线仪表盘”
        start_here_href = ""
        try:
            p0 = out_path.parent / "START_HERE.html"
            p1 = out_path.parent.parent / "START_HERE.html"
            if p0.exists() and p0.resolve() != out_path.resolve():
                start_here_href = "START_HERE.html"
            elif p1.exists():
                start_here_href = "../START_HERE.html"
        except Exception:
            start_here_href = ""

        nav_links: List[str] = []
        if start_here_href:
            nav_links.append(
                f'<a class="btn" href="{html.escape(start_here_href, quote=True)}" title="返回入口">← START_HERE</a>'
            )
        # 展示增强：展开/收起全部（不影响口径与算数，只是更好阅读）
        nav_links.append('<button class="btn" id="btnExpandAll" type="button" title="展开所有可折叠区块/表格">展开全部</button>')
        nav_links.append('<button class="btn" id="btnCollapseAll" type="button" title="收起所有可折叠区块/表格">收起全部</button>')
        nav_html = "\n".join(nav_links)

        css = """
:root{
  --bg:#f6f7fb;
  --card:#ffffff;
  --text:#0f172a;
  --muted:#64748b;
  --border:#e5e7eb;
  --link:#2563eb;
  --shadow:0 1px 2px rgba(0,0,0,.04);
  --radius:14px;
  --mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0b1220;
    --card:#0f172a;
    --text:#e5e7eb;
    --muted:#94a3b8;
    --border:#1f2937;
    --link:#60a5fa;
    --shadow:0 1px 2px rgba(0,0,0,.3);
  }
}
*{box-sizing:border-box;}
html,body{height:100%;}
body{
  margin:0;
  background:var(--bg);
  color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;
}
a{color:var(--link);text-decoration:none;}
a:hover{text-decoration:underline;}
.wrap{max-width:1200px;margin:0 auto;padding:18px 16px;}
.topbar{
  position:sticky;
  top:0;
  z-index:20;
  background:rgba(255,255,255,.72);
  border-bottom:1px solid var(--border);
  backdrop-filter:saturate(180%) blur(10px);
}
@media (prefers-color-scheme: dark){
  .topbar{background:rgba(15,23,42,.72);}
}
.topbar .wrap{padding:12px 16px;}
.topbar-inner{display:flex;align-items:center;justify-content:space-between;gap:12px;}
.title{font-weight:700;letter-spacing:.2px;}
.actions{display:flex;gap:8px;flex-wrap:wrap;}
.btn{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:7px 10px;
  border:1px solid var(--border);
  border-radius:10px;
  background:rgba(255,255,255,.7);
  color:var(--text);
  box-shadow:var(--shadow);
  cursor:pointer;
  font:inherit;
}
@media (prefers-color-scheme: dark){
  .btn{background:rgba(2,6,23,.4);}
}
.btn:hover{text-decoration:none;border-color:rgba(37,99,235,.45);}
.btn:active{transform:translateY(1px);}
.layout{
  display:grid;
  grid-template-columns:300px 1fr;
  gap:14px;
  align-items:start;
}
.layout.single{grid-template-columns:1fr;}
@media (max-width: 980px){
  .layout{grid-template-columns:1fr;}
}
.card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:18px 16px;
  box-shadow:var(--shadow);
}
.page-title-bar{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  flex-wrap:wrap;
}
.page-title-bar h1{margin:0;}
.shop-filter{
  display:flex;
  align-items:center;
  gap:8px;
  font-size:12px;
  color:var(--muted);
}
.shop-filter select{
  padding:6px 10px;
  border-radius:10px;
  border:1px solid var(--border);
  background:rgba(255,255,255,.7);
  color:var(--text);
  font:inherit;
}
@media (prefers-color-scheme: dark){
  .shop-filter select{background:rgba(2,6,23,.4);}
}
.shop-filter-hint{color:var(--muted);font-size:12px;}
.anchor-spacer{height:1px;}
.toc{
  position:sticky;
  top:68px;
  max-height:calc(100vh - 84px);
  overflow:auto;
}
@media (max-width: 980px){
  .toc{position:relative;top:auto;max-height:none;}
}
.toc-title{font-weight:700;margin-bottom:8px;}
.toc-body{display:flex;flex-direction:column;gap:6px;}
.toc-item{display:block;padding:6px 8px;border-radius:10px;color:var(--text);border:1px solid transparent;}
.toc-item:hover{border-color:rgba(37,99,235,.35);background:rgba(37,99,235,.06);text-decoration:none;}
.toc-item.active{border-color:rgba(37,99,235,.55);background:rgba(37,99,235,.10);}
.toc-lvl2{font-weight:600;}
.toc-lvl3{padding-left:18px;color:var(--muted);font-weight:500;}
.content h1,.content h2,.content h3,.content h4{margin:18px 0 10px;line-height:1.25;scroll-margin-top:84px;}
.content h1{font-size:22px;}
.content h2{font-size:18px;margin-top:22px;padding-top:8px;border-top:1px solid var(--border);}
.content h3{font-size:16px;}
.content h4{font-size:14px;color:var(--muted);}
/* 区块标题一致化：在 H2/H3 前加统一标签（不改原文） */
.h-labeled{position:relative;}
.h-labeled::before{
  content:attr(data-label);
  display:inline-block;
  font-size:11px;
  font-weight:800;
  padding:2px 8px;
  margin-right:8px;
  border-radius:999px;
  border:1px solid var(--border);
  vertical-align:middle;
}
.label-overview::before{background:rgba(59,130,246,.12);color:#1d4ed8;}
.label-action::before{background:rgba(34,197,94,.12);color:#15803d;}
.label-risk::before{background:rgba(239,68,68,.12);color:#b91c1c;}
.label-top::before{background:rgba(245,158,11,.12);color:#b45309;}
.label-drill::before{background:rgba(100,116,139,.16);color:var(--muted);}
.label-topic::before{background:rgba(168,85,247,.12);color:#7c3aed;}
.label-phase::before{background:rgba(14,165,233,.12);color:#0ea5e9;}
.label-category::before{background:rgba(6,182,212,.12);color:#0891b2;}
.label-asin::before{background:rgba(22,163,74,.10);color:#15803d;}
.label-change::before{background:rgba(99,102,241,.12);color:#4338ca;}
.content p{margin:10px 0;line-height:1.7;}
.content ul{margin:10px 0 14px 20px;line-height:1.7;}
.content li{margin:6px 0;}
/* 分离原则：把每个 H2 章节包成“段落卡片”，减少内容堆叠感（不改口径，只改展示） */
.section{
  margin:14px 0;
  padding:14px 14px;
  border:1px solid var(--border);
  border-radius:16px;
  background:rgba(255,255,255,.55);
}
@media (prefers-color-scheme: dark){
  .section{background:rgba(2,6,23,.22);}
}
.section:first-of-type{margin-top:0;}
.section h2{
  position:relative;
  margin:0 0 10px;
  padding:0 0 0 14px;
  border-top:none;
  font-size:18px;
}
.section h2::before{
  content:'';
  position:absolute;
  left:0;
  top:.18em;
  bottom:.18em;
  width:4px;
  border-radius:999px;
  background:linear-gradient(180deg, rgba(37,99,235,.85), rgba(168,85,247,.70));
}
.section h3{margin-top:16px;}
.section .hint{margin-top:12px;}
/* dashboard/下钻页常见“快速入口”段落：做成更像工具栏的视觉层 */
.quick-entry{
  padding:10px 12px;
  border:1px solid var(--border);
  border-radius:14px;
  background:rgba(37,99,235,.03);
  margin:10px 0 12px;
}
.quick-entry a{
  display:inline-block;
  padding:3px 8px;
  margin:2px 4px 2px 0;
  border:1px solid var(--border);
  border-radius:999px;
  background:rgba(255,255,255,.6);
}
@media (prefers-color-scheme: dark){
  .quick-entry a{background:rgba(2,6,23,.35);}
}
/* dashboard 第一屏：概览 KPI 先单独横排，行动/预警在下方三栏 */
.hero-grid{
  display:grid;
  grid-template-columns:repeat(3, minmax(0, 1fr));
  grid-template-areas:
    "overview overview overview"
    "weekly alerts campaign";
  gap:12px;
  margin:10px 0 14px;
}
@media (max-width: 1400px){
  .hero-grid{
    grid-template-columns:1fr 1fr;
    grid-template-areas:
      "overview overview"
      "weekly alerts"
      "campaign campaign";
  }
}
@media (max-width: 1100px){
  .hero-grid{
    grid-template-columns:1fr;
    grid-template-areas:
      "overview"
      "weekly"
      "alerts"
      "campaign";
  }
}
.hero-panel{
  border:1px solid var(--border);
  border-radius:16px;
  padding:12px 12px;
  background:rgba(255,255,255,.60);
}
@media (prefers-color-scheme: dark){
  .hero-panel{background:rgba(2,6,23,.22);}
}
.hero-title{
  font-weight:800;
  font-size:13px;
  color:var(--muted);
  letter-spacing:.2px;
  text-transform:none;
  margin:0 0 10px;
}
.hero-panel h3{
  margin:0 0 10px;
  font-weight:800;
  font-size:13px;
  color:var(--muted);
  letter-spacing:.2px;
}
.hero-panel .cards{margin:8px 0 0;}
.hero-panel.overview{grid-area:overview;}
.hero-panel.weekly{grid-area:weekly;}
.hero-panel.alerts{grid-area:alerts;}
.hero-panel.campaign{grid-area:campaign;}
.kpi-grid{
  display:grid;
  grid-template-columns:repeat(4, minmax(150px, 1fr));
  gap:8px;
  margin:0 0 10px;
}
@media (max-width: 1400px){
  .kpi-grid{grid-template-columns:repeat(3, minmax(150px, 1fr));}
}
@media (max-width: 1100px){
  .kpi-grid{grid-template-columns:repeat(2, minmax(150px, 1fr));}
}
@media (max-width: 700px){
  .kpi-grid{grid-template-columns:1fr;}
}
.kpi{
  border:1px solid var(--border);
  border-radius:14px;
  padding:10px 10px;
  background:rgba(37,99,235,.03);
}
.kpi .k{font-size:12px;color:var(--muted);margin-bottom:6px;}
.kpi .v{font-size:18px;font-weight:800;letter-spacing:.2px;line-height:1.2;font-variant-numeric:tabular-nums;}
.kpi .v.neg{color:#b91c1c;}
.kpi .v.pos{color:#16a34a;}
.kpi .v.zero{color:var(--muted);}
@media (prefers-color-scheme: dark){
  .kpi .v.neg{color:#f87171;}
  .kpi .v.pos{color:#34d399;}
}
.kpi .s{font-size:11px;color:var(--muted);margin-top:4px;}
.raw-details{margin-top:6px;}
.raw-details summary{
  cursor:pointer;
  font-weight:700;
  color:var(--muted);
  list-style:none;
}
.raw-details summary::-webkit-details-marker{display:none;}
.raw-details summary:before{content:'＋';display:inline-block;margin-right:6px;color:var(--muted);}
.raw-details[open] summary:before{content:'－';}
.raw-details .raw-body{margin-top:8px;color:var(--muted);font-size:12px;line-height:1.6;}
.fold-section{margin:8px 0 12px;}
.fold-section summary{
  cursor:pointer;
  font-weight:800;
  color:var(--muted);
  list-style:none;
}
.fold-section summary::-webkit-details-marker{display:none;}
.fold-section summary:before{content:'＋';display:inline-block;margin-right:6px;color:var(--muted);}
.fold-section[open] summary:before{content:'－';}
.fold-section .fold-body{margin-top:10px;}
.cards{
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(260px, 1fr));
  gap:10px;
  margin:12px 0 16px;
}
.kanban{
  display:grid;
  grid-template-columns:repeat(3, minmax(0, 1fr));
  gap:12px;
  margin:12px 0 16px;
}
@media (max-width: 1100px){
  .kanban{grid-template-columns:1fr;}
}
.kanban-col{
  border:1px solid var(--border);
  border-radius:16px;
  background:rgba(255,255,255,.60);
  display:flex;
  flex-direction:column;
  min-height:160px;
  box-shadow:var(--shadow);
}
@media (prefers-color-scheme: dark){
  .kanban-col{background:rgba(2,6,23,.22);}
}
.kanban-head{
  padding:10px 12px;
  font-weight:800;
  border-bottom:1px solid var(--border);
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
}
.kanban-count{
  font-size:12px;
  color:var(--muted);
  font-weight:700;
}
.kanban-meta{
  display:flex;
  flex-wrap:wrap;
  gap:6px;
  margin-bottom:6px;
}
.kanban-tag{
  font-size:11px;
  font-weight:800;
  padding:2px 8px;
  border-radius:999px;
  border:1px solid var(--border);
  background:rgba(148,163,184,.10);
  color:var(--text);
}
.kanban-tag.p0{background:rgba(239,68,68,.16);border-color:rgba(239,68,68,.40);color:#b91c1c;}
.kanban-tag.p1{background:rgba(245,158,11,.16);border-color:rgba(245,158,11,.40);color:#b45309;}
.kanban-tag.p2{background:rgba(34,197,94,.14);border-color:rgba(34,197,94,.35);color:#15803d;}
.kanban-tag.owner{background:rgba(14,165,233,.10);border-color:rgba(14,165,233,.25);color:#0ea5e9;}
.kanban-tag.shop{background:rgba(99,102,241,.10);border-color:rgba(99,102,241,.25);color:#4f46e5;}
.kanban-body{
  padding:10px 12px 12px;
  display:flex;
  flex-direction:column;
  gap:8px;
}
.kanban-card{
  padding:10px 10px;
  border:1px solid var(--border);
  border-radius:12px;
  background:rgba(255,255,255,.55);
  line-height:1.65;
  word-break:break-word;
  white-space:normal;
}
.kanban-card code{margin-right:4px;}
.kanban-card a{display:inline-block;margin-top:6px;}
.kanban-desc{line-height:1.65;}
.kanban-link a{font-size:12px;}
.kanban-group{padding-top:6px;margin-top:6px;border-top:1px dashed var(--border);}
.kanban-group:first-child{padding-top:0;margin-top:0;border-top:none;}
.kanban-group-head{display:flex;justify-content:space-between;gap:8px;font-size:12px;font-weight:800;color:var(--muted);margin-bottom:6px;}
.kanban-card.prio-p0{border-left:6px solid rgba(239,68,68,.55);}
.kanban-card.prio-p1{border-left:6px solid rgba(245,158,11,.55);}
.kanban-card.prio-p2{border-left:6px solid rgba(34,197,94,.50);}
@media (prefers-color-scheme: dark){
  .kanban-card{background:rgba(2,6,23,.25);}
}
.loop-flow{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  align-items:center;
  margin:8px 0 12px;
}
.loop-step{
  padding:6px 10px;
  border:1px solid var(--border);
  border-radius:12px;
  background:rgba(99,102,241,.06);
  font-weight:800;
  font-size:12px;
}
.loop-arrow{color:var(--muted);font-weight:800;}
.loop-metrics{
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
  gap:8px;
  margin:6px 0 12px;
}
.loop-metric{
  border:1px solid var(--border);
  border-radius:12px;
  padding:8px 10px;
  background:rgba(37,99,235,.03);
}
.loop-metric .k{font-size:12px;color:var(--muted);}
.loop-metric .v{font-size:16px;font-weight:800;font-variant-numeric:tabular-nums;}
.timeline-cards{
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(260px, 1fr));
  gap:10px;
  margin:8px 0 14px;
}
.timeline-card{
  border:1px solid var(--border);
  border-radius:14px;
  padding:10px 10px;
  background:rgba(255,255,255,.60);
  display:flex;
  flex-direction:column;
  gap:6px;
}
@media (prefers-color-scheme: dark){
  .timeline-card{background:rgba(2,6,23,.22);}
}
.timeline-card.risk{border-left:4px solid rgba(239,68,68,.55);}
.timeline-card.opp{border-left:4px solid rgba(34,197,94,.55);}
.timeline-card .timeline{width:100%;max-width:100%;}
.timeline-card .timeline-row{
  display:grid;
  grid-template-columns:minmax(180px, 1.2fr) minmax(140px, 1fr);
  gap:8px;
  align-items:center;
}
.timeline-card .timeline-wrap{min-width:0;}
.timeline-card .metrics{
  display:flex;
  flex-wrap:wrap;
  gap:6px;
  font-size:12px;
  color:var(--muted);
}
@media (max-width: 720px){
  .timeline-card .timeline-row{grid-template-columns:1fr;}
}
.timeline-card .title{font-weight:800;font-size:13px;}
.timeline-card .sub{font-size:12px;color:var(--muted);}
.timeline-card .badges{display:flex;gap:6px;flex-wrap:wrap;}
.phase-badge{
  font-size:11px;
  font-weight:800;
  padding:2px 8px;
  border-radius:999px;
  border:1px solid var(--border);
  background:rgba(148,163,184,.10);
}
.phase-badge.phase-pre_launch{background:rgba(100,116,139,.12);}
.phase-badge.phase-launch{background:rgba(168,85,247,.14);}
.phase-badge.phase-growth{background:rgba(37,99,235,.14);}
.phase-badge.phase-stable{background:rgba(34,197,94,.14);}
.phase-badge.phase-mature{background:rgba(34,197,94,.14);}
.phase-badge.phase-decline{background:rgba(249,115,22,.14);}
.phase-badge.phase-inactive{background:rgba(239,68,68,.14);}
.phase-badge.strategy{background:rgba(14,165,233,.10);}
.kanban-col.risk .kanban-head{background:rgba(239,68,68,.08);}
.kanban-col.opp .kanban-head{background:rgba(34,197,94,.08);}
.kanban-col.action .kanban-head{background:rgba(59,130,246,.08);}
.card-item{
  padding:12px 12px;
  border:1px solid var(--border);
  border-radius:14px;
  background:rgba(255,255,255,.55);
}
@media (prefers-color-scheme: dark){
  .card-item{background:rgba(2,6,23,.25);}
}
.card-item:hover{border-color:rgba(37,99,235,.35);}
.card-item.prio-p0{border-left:6px solid rgba(239,68,68,.55);}
.card-item.prio-p1{border-left:6px solid rgba(245,158,11,.55);}
.card-item.prio-p2{border-left:6px solid rgba(34,197,94,.50);}
/* 风险/机会/提示的统一视觉语义 */
.tone-risk{border-left:4px solid rgba(239,68,68,.55);background:rgba(239,68,68,.06);}
.tone-opp{border-left:4px solid rgba(34,197,94,.55);background:rgba(34,197,94,.06);}
.tone-hint{border-left:4px solid rgba(245,158,11,.55);background:rgba(245,158,11,.06);}
@media (prefers-color-scheme: dark){
  .tone-risk{background:rgba(239,68,68,.12);}
  .tone-opp{background:rgba(34,197,94,.10);}
  .tone-hint{background:rgba(245,158,11,.12);}
}
.card-item.tone-risk,.card-item.tone-opp,.card-item.tone-hint{border-left-width:6px;}
.block-details.tone-risk summary{background:rgba(239,68,68,.08);}
.block-details.tone-opp summary{background:rgba(34,197,94,.08);}
.block-details.tone-hint summary{background:rgba(245,158,11,.08);}
.content li.tone-risk,.content li.tone-opp,.content li.tone-hint{
  padding:6px 8px;
  border-radius:10px;
}
.read-guide,.logic-note{
  margin:10px 0 12px;
  padding:10px 12px;
  border:1px solid var(--border);
  border-radius:14px;
  background:rgba(59,130,246,.04);
}
@media (prefers-color-scheme: dark){
  .read-guide,.logic-note{background:rgba(59,130,246,.08);}
}
.read-guide .rg-title,.logic-note .ln-title{
  font-weight:800;
  color:var(--muted);
  margin:0 0 6px;
}
.read-guide ol,.logic-note ul{margin:0 0 0 18px;line-height:1.6;}
.read-guide li,.logic-note li{margin:4px 0;}
.content code{
  background:rgba(148,163,184,.15);
  border:1px solid var(--border);
  padding:0 6px;
  border-radius:8px;
  font-family:var(--mono);
  font-size:.95em;
}
.content a code{color:var(--link);}
.content code.badge{
  padding:2px 10px;
  border-radius:999px;
  font-weight:800;
  letter-spacing:.2px;
}
.content code.badge.p0{background:rgba(239,68,68,.16);border-color:rgba(239,68,68,.40);color:#b91c1c;}
.content code.badge.p1{background:rgba(245,158,11,.16);border-color:rgba(245,158,11,.40);color:#b45309;}
.content code.badge.p2{background:rgba(34,197,94,.14);border-color:rgba(34,197,94,.35);color:#15803d;}
.content code.badge.tag-stop{background:rgba(239,68,68,.10);border-color:rgba(239,68,68,.25);color:#b91c1c;}
.content code.badge.tag-scale{background:rgba(37,99,235,.10);border-color:rgba(37,99,235,.25);color:#1d4ed8;}
.content code.badge.tag-review{background:rgba(100,116,139,.12);border-color:rgba(100,116,139,.25);color:var(--muted);}
.content code.badge.tag-blocked{background:rgba(239,68,68,.10);border-color:rgba(239,68,68,.25);color:#b91c1c;}
.content code.badge.tag-owner{background:rgba(14,165,233,.10);border-color:rgba(14,165,233,.25);color:#0ea5e9;}
.content code.badge.tag-shop{background:rgba(99,102,241,.10);border-color:rgba(99,102,241,.25);color:#4f46e5;}
@media (prefers-color-scheme: dark){
  .content code.badge.p0{color:#fecaca;}
  .content code.badge.p1{color:#fde68a;}
  .content code.badge.p2{color:#bbf7d0;}
  .content code.badge.tag-stop{color:#fecaca;}
  .content code.badge.tag-scale{color:#bfdbfe;}
  .content code.badge.tag-review{color:#cbd5e1;}
  .content code.badge.tag-blocked{color:#fecaca;}
}
.content pre{
  background:rgba(148,163,184,.12);
  border:1px solid var(--border);
  padding:12px 12px;
  border-radius:12px;
  overflow:auto;
  font-family:var(--mono);
  font-size:13px;
  line-height:1.5;
}
hr{border:none;border-top:1px solid var(--border);margin:16px 0;}
.table-wrap{
  overflow:auto;
  border:1px solid var(--border);
  border-radius:12px;
  background:rgba(255,255,255,.35);
}
@media (prefers-color-scheme: dark){
  .table-wrap{background:rgba(2,6,23,.2);}
}
/* 超大表格：默认折叠，避免“所有东西都堆在一起” */
.table-details{margin:10px 0 14px;}
.table-details summary{
  cursor:pointer;
  padding:10px 12px;
  border:1px solid var(--border);
  border-radius:12px;
  background:rgba(37,99,235,.04);
  font-weight:700;
  color:var(--text);
  user-select:none;
}
.table-details summary:hover{border-color:rgba(37,99,235,.35);}
.table-details summary::-webkit-details-marker{display:none;}
.table-details summary:before{
  content:'▸';
  display:inline-block;
  margin-right:8px;
  color:var(--muted);
  transform:translateY(-1px);
}
.table-details[open] summary:before{content:'▾';}
.table-details[open] summary{border-bottom-left-radius:0;border-bottom-right-radius:0;}
.table-details[open] .table-wrap{border-top:none;border-top-left-radius:0;border-top-right-radius:0;}
/* 章节内的“块级折叠”：对大量 H3 区块做默认收起（按重要性决定是否收起），降低页面长度 */
.block-details{
  margin:10px 0 14px;
  border:1px solid var(--border);
  border-radius:16px;
  background:rgba(255,255,255,.55);
  overflow:hidden;
}
@media (prefers-color-scheme: dark){
  .block-details{background:rgba(2,6,23,.22);}
}
.block-details summary{
  cursor:pointer;
  padding:11px 12px;
  border-bottom:1px solid var(--border);
  background:rgba(37,99,235,.03);
  font-weight:800;
  color:var(--text);
  user-select:none;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
}
.block-details summary::-webkit-details-marker{display:none;}
.block-details summary:before{
  content:'▸';
  display:inline-block;
  margin-right:8px;
  color:var(--muted);
  transform:translateY(-1px);
}
.block-details[open] summary:before{content:'▾';}
.block-details[open] summary{border-bottom-left-radius:0;border-bottom-right-radius:0;}
.block-details .sum-meta{font-size:12px;color:var(--muted);font-weight:700;white-space:nowrap;}
.block-details .block-body{padding:12px 12px 2px;}
/* 保留 h3 的 anchor（TOC/跳转用），但避免与 summary 文案重复：把 h3 压缩为“不可见占位” */
.block-details .block-body > h3{
  margin:0;
  height:1px;
  overflow:hidden;
  opacity:0;
}
		table{border-collapse:collapse;width:100%;min-width:760px;}
		th,td{
		  border-bottom:1px solid var(--border);
		  padding:8px 10px;
		  text-align:left;
		  vertical-align:top;
		  font-size:13px;
		  white-space:nowrap;
		}
		/* 数字列自动右对齐（JS 识别列类型后加 class），更利于扫表 */
		th.col-num, td.col-num{text-align:right;font-variant-numeric:tabular-nums;}
		/* 长文本列允许换行（减少横向滚动） */
		th.col-wrap, td.col-wrap{white-space:normal;}
		td.col-wrap{max-width:420px;word-break:break-word;line-height:1.55;}
	thead th{
	  position:sticky;
	  top:0;
	  z-index:2;
	  background:rgba(249,250,251,.9);
  font-weight:700;
  cursor:pointer;
}
@media (prefers-color-scheme: dark){
  thead th{background:rgba(15,23,42,.92);}
}
tbody tr:hover td{background:rgba(37,99,235,.06);}
	tr:nth-child(even) td{background:rgba(100,116,139,.03);}
	.hint{color:var(--muted);font-size:12px;margin-top:10px;}
	/* lifecycle timeline：用 code 标记 tl:...，在 JS 中渲染成条形时间轴 */
	.timeline{
	  display:flex;
	  align-items:stretch;
	  width:360px;
	  max-width:520px;
	  height:18px;
	  border:1px solid var(--border);
	  border-radius:999px;
	  overflow:hidden;
	  background:rgba(100,116,139,.08);
	}
	.timeline.recent{
	  box-shadow:0 0 0 2px rgba(239,68,68,.25);
	  border-color:rgba(239,68,68,.45);
	}
	.timeline .seg{
	  display:flex;
	  align-items:center;
	  justify-content:center;
	  height:100%;
	  min-width:3px;
	  color:#ffffff;
	  font-size:11px;
	  line-height:1;
	}
	.timeline .seg .lbl{padding:0 6px;opacity:.95;white-space:nowrap;}
	.timeline .phase-pre_launch{background:rgba(100,116,139,.85);}
	.timeline .phase-launch{background:rgba(168,85,247,.85);}
	.timeline .phase-growth{background:rgba(37,99,235,.85);}
	.timeline .phase-stable{background:rgba(34,197,94,.75);}
	.timeline .phase-mature{background:rgba(34,197,94,.75);}
	.timeline .phase-decline{background:rgba(249,115,22,.85);}
	.timeline .phase-inactive{background:rgba(239,68,68,.85);}
	.timeline .phase-unknown{background:rgba(100,116,139,.55);}
	.to-top{
	  position:fixed;
	  right:16px;
	  bottom:16px;
	  width:42px;
  height:42px;
  border-radius:14px;
  border:1px solid var(--border);
  background:rgba(255,255,255,.85);
  box-shadow:var(--shadow);
  color:var(--text);
  display:none;
  cursor:pointer;
}
@media (prefers-color-scheme: dark){
  .to-top{background:rgba(2,6,23,.5);}
}
	.to-top.show{display:block;}
"""

        js = r"""
function _qs(sel, root){return (root||document).querySelector(sel);}
function _slugify(s){
  const t=String(s||'').trim().toLowerCase().replace(/\s+/g,'-');
  const cleaned=t.replace(/[^a-z0-9\u4e00-\u9fff_\-]+/g,'');
  return cleaned;
}

// 轻量表格排序：点击表头即可按该列排序（数字/文本自动识别）
function _parseNum(t){
  const s=String(t||'').replace(/,/g,'').trim();
  const n=parseFloat(s);
  return Number.isFinite(n) ? n : null;
}
function _sortTable(table, col, asc){
  const tbody=table.tBodies && table.tBodies[0];
  if(!tbody) return;
  const rows=Array.from(tbody.rows);
  rows.sort((a,b)=>{
    const ta=a.cells[col] ? a.cells[col].innerText : '';
    const tb=b.cells[col] ? b.cells[col].innerText : '';
    const na=_parseNum(ta);
    const nb=_parseNum(tb);
    if(na!==null && nb!==null){
      return asc ? (na-nb) : (nb-na);
    }
    const sa=String(ta||'');
    const sb=String(tb||'');
    return asc ? sa.localeCompare(sb,'zh') : sb.localeCompare(sa,'zh');
  });
  rows.forEach(r=>tbody.appendChild(r));
}

function _initTableSort(){
  document.querySelectorAll('table').forEach((table)=>{
    const ths=table.querySelectorAll('thead th');
    ths.forEach((th, idx)=>{
      th.title='点击排序';
      th.addEventListener('click', ()=>{
        const key='data-sort-col';
        const dirKey='data-sort-dir';
        const prevCol=parseInt(table.getAttribute(key)||'-1',10);
        const prevDir=table.getAttribute(dirKey)||'desc';
        const asc=(prevCol===idx) ? (prevDir!=='asc') : true;
        table.setAttribute(key, String(idx));
        table.setAttribute(dirKey, asc ? 'asc' : 'desc');
        _sortTable(table, idx, asc);
      });
    });
  });
}

function _normalizeAnchorsAndBuildToc(){
  const content=_qs('#content');
  const toc=_qs('#toc');
  const tocCard=_qs('#tocCard');
  if(!content || !toc || !tocCard) return;

  // 1) 把 <a id="..."></a> 的 id 迁移到紧随其后的 heading 上（保持锚点跳转更准确）
  const headings=Array.from(content.querySelectorAll('h1,h2,h3,h4'));
  headings.forEach((h, idx)=>{
    const prev=h.previousElementSibling;
    if(prev && prev.tagName==='A' && prev.id && (prev.textContent||'').trim()===''){
      if(!h.id) h.id = prev.id;
      prev.remove();
    }
    if(!h.id){
      let base=_slugify(h.textContent||'') || ('sec-'+(idx+1));
      let id=base;
      let n=2;
      while(document.getElementById(id)){
        id = base + '-' + (n++);
      }
      h.id=id;
    }
  });

  // 2) 生成目录（优先 h2/h3；h1 通常是标题，不放进目录）
  const hs=Array.from(content.querySelectorAll('h2,h3'));
  if(hs.length===0){
    tocCard.style.display='none';
    return;
  }
  const frag=document.createDocumentFragment();
  hs.forEach((h)=>{
    const a=document.createElement('a');
    a.href='#'+(h.id||'');
    a.textContent=(h.textContent||'').trim();
    a.className='toc-item ' + (h.tagName==='H3' ? 'toc-lvl3' : 'toc-lvl2');
    frag.appendChild(a);
  });
  toc.innerHTML='';
  toc.appendChild(frag);
}

function _initTocActive(){
  const toc=_qs('#toc');
  const content=_qs('#content');
  if(!toc || !content) return;
  const links=Array.from(toc.querySelectorAll('a.toc-item'));
  if(links.length===0) return;
  const headings=Array.from(content.querySelectorAll('h2,h3'));
  if(headings.length===0) return;
  let activeId='';
  function setActive(id){
    if(!id || id===activeId) return;
    activeId=id;
    links.forEach((a)=>{
      const href=a.getAttribute('href')||'';
      a.classList.toggle('active', href==='#'+id);
    });
  }
  // 点击目录时立即高亮
  links.forEach((a)=>{
    a.addEventListener('click', ()=>{
      const href=a.getAttribute('href')||'';
      if(href.startsWith('#')) setActive(href.slice(1));
    });
  });
  // 滚动时跟随：优先使用 IntersectionObserver（性能更好）
  if('IntersectionObserver' in window){
    const observer=new IntersectionObserver((entries)=>{
      const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio);
      if(visible.length>0){
        const id=visible[0].target && visible[0].target.id;
        if(id) setActive(id);
      }
    }, {root:null, rootMargin:'-15% 0px -70% 0px', threshold:[0,0.1,0.25,0.5,1]});
    headings.forEach((h)=>{ if(h.id) observer.observe(h); });
  }
}

// 让“所有东西都堆在一起”的感觉更弱：把每个 H2 章节包进 section 卡片
function _wrapSections(){
  const content=_qs('#content');
  if(!content) return;

  // 保留末尾提示（生成说明），避免被包进最后一个 section
  const tail=[];
  try{
    while(content.lastElementChild && content.lastElementChild.classList && content.lastElementChild.classList.contains('hint')){
      tail.unshift(content.lastElementChild);
      content.removeChild(content.lastElementChild);
    }
  }catch(e){}

  const kids=Array.from(content.children);
  let current=null;
  kids.forEach((el)=>{
    if(el && el.tagName==='H2'){
      current=document.createElement('section');
      current.className='section';
      content.insertBefore(current, el);
      current.appendChild(el);
      return;
    }
    if(current){
      current.appendChild(el);
    }
  });

  try{ tail.forEach((el)=>content.appendChild(el)); }catch(e){}
}

function _decorateQuickEntry(){
  const content=_qs('#content');
  if(!content) return;
  try{
    Array.from(content.querySelectorAll('p')).forEach((p)=>{
      const t=String(p.innerText||'').trim();
      if(t.startsWith('快速入口：')) p.classList.add('quick-entry');
    });
  }catch(e){}
}

function _openParentDetails(el){
  try{
    let cur=el;
    while(cur){
      if(cur.tagName==='DETAILS' && !cur.open){ cur.open=true; }
      cur=cur.parentElement;
    }
  }catch(e){}
}

function _openDetailsForHash(){
  try{
    const h=String(location.hash||'').trim();
    if(!h || !h.startsWith('#') || h.length<2) return;
    const id=h.slice(1);
    const el=document.getElementById(id);
    if(!el) return;
    _openParentDetails(el);
    // hash 可能在 details 打开前就滚动了，补一次定位更稳定
    try{ el.scrollIntoView({block:'start'}); }catch(e){}
  }catch(e){}
}

function _toggleAllDetails(open){
  try{
    const ds=Array.from(document.querySelectorAll('details'));
    ds.forEach((d)=>{ d.open=!!open; });
  }catch(e){}
}

function _initExpandCollapseAll(){
  try{
    const openBtn=_qs('#btnExpandAll');
    const closeBtn=_qs('#btnCollapseAll');
    if(openBtn){
      openBtn.addEventListener('click', ()=>_toggleAllDetails(true));
    }
    if(closeBtn){
      closeBtn.addEventListener('click', ()=>_toggleAllDetails(false));
    }
  }catch(e){}
}

function _detectTone(text){
  try{
    const s=String(text||'').toLowerCase();
    if(!s) return '';
    // 机会优先（避免被 P0/P1 覆盖）
    if(/机会|放量|scale|增长|增量/.test(s)) return 'tone-opp';
    if(/p0|止损|断货|风险|亏损|烧钱|走弱|下滑|阻断/.test(s)) return 'tone-risk';
    if(/p1|排查|提示|注意|review|观察/.test(s)) return 'tone-hint';
    return '';
  }catch(e){
    return '';
  }
}

function _decorateTones(){
  const content=_qs('#content');
  if(!content) return;
  try{
    // 1) 卡片
    content.querySelectorAll('.card-item').forEach((el)=>{
      const tone=_detectTone(el.innerText||'');
      if(tone) el.classList.add(tone);
    });
    // 2) 列表项（例如本周行动/告警）
    content.querySelectorAll('li').forEach((el)=>{
      const tone=_detectTone(el.innerText||'');
      if(tone) el.classList.add(tone);
    });
    // 3) 折叠块（以 summary 文案判定）
    content.querySelectorAll('.block-details').forEach((el)=>{
      const sum=el.querySelector('summary');
      const tone=_detectTone(sum ? (sum.textContent||'') : '');
      if(tone) el.classList.add(tone);
    });
  }catch(e){}
}

function _insertAfter(ref, node){
  try{
    if(!ref || !ref.parentNode || !node) return;
    ref.parentNode.insertBefore(node, ref.nextSibling);
  }catch(e){}
}

function _findMetaAnchor(content){
  try{
    const h1=content.querySelector('h1');
    if(!h1) return null;
    let el=h1.nextElementSibling;
    while(el){
      const tag=String(el.tagName||'');
      if(tag==='UL') return el;
      if(tag==='H2') break;
      el=el.nextElementSibling;
    }
    return h1;
  }catch(e){
    return null;
  }
}

function _buildReadGuide(title){
  const t=String(title||'');
  let steps=null;
  if(/Dashboard/i.test(t)){
    steps=[
      '先看「本期结论/概览卡片」确定重点方向',
      '再看「本周行动 / Shop Alerts」决定先做什么',
      '然后看 Watchlists / ASIN Focus 进行排查或分派',
      '需要细节时再进入类目/ASIN/阶段下钻'
    ];
  }else if(/ASIN Drilldown/i.test(t)){
    steps=[
      '先用索引定位 ASIN',
      '先看 Drivers（近7天 vs 前7天）确认变化来源',
      '再看 Top Actions 按优先级执行或复核'
    ];
  }else if(/Category Drilldown|类目下钻/i.test(t)){
    steps=[
      '先看类目总览判断优先级',
      '再看 Top ASIN 找到主要贡献/风险产品',
      '需要放量/控量时再查关键词主题'
    ];
  }else if(/Phase Drilldown|生命周期下钻/i.test(t)){
    steps=[
      '先看生命周期总览判断阶段风险',
      '再看阶段内类目 Top',
      '最后看阶段内 Top ASIN'
    ];
  }else if(/Keyword Topics/i.test(t)){
    steps=[
      '先选类目×阶段（Segment Top）',
      '再筛主题并进入 Action Hints',
      '执行前查看 ASIN Context 的库存/生命周期语境'
    ];
  }else if(/生命周期时间轴|Lifecycle Overview/i.test(t)){
    steps=[
      '先看近期重点（3-5条）',
      '再看阶段分布与类目结构 Top5',
      '最后进入时间轴细节'
    ];
  }else{
    return null;
  }
  const box=document.createElement('div');
  box.className='read-guide';
  const titleEl=document.createElement('div');
  titleEl.className='rg-title';
  titleEl.textContent='阅读顺序（建议）';
  const ol=document.createElement('ol');
  steps.forEach((s)=>{
    const li=document.createElement('li');
    li.textContent=s;
    ol.appendChild(li);
  });
  box.appendChild(titleEl);
  box.appendChild(ol);
  return box;
}

function _buildLogicNote(title){
  const t=String(title||'');
  let notes=null;
  if(/Dashboard/i.test(t)){
    notes=[
      '“放量/控量”受库存与利润承受度约束；blocked=1 表示需先解决阻断',
      '阶段/Δ指标默认来自近7天 vs 前7天 compare 窗口（解释用）',
      '执行以 Action Board / 解锁任务 / Watchlists 为准'
    ];
  }else if(/生命周期时间轴|Lifecycle Overview/i.test(t)){
    notes=[
      '时间轴为展示层：短碎片段“平滑合并”不改变算数口径',
      '阶段变化提示用于聚焦，不代表必须执行动作',
      '执行仍以 Action Board / 解锁任务为准'
    ];
  }else if(/Keyword Topics/i.test(t)){
    notes=[
      '关键词主题基于 n-gram 聚合，仅用于归类与排查',
      '放量只对 direction=scale 且 blocked=0 生效',
      '执行前建议结合 ASIN Context 看库存/生命周期'
    ];
  }else if(/ASIN Drilldown|Category Drilldown|Phase Drilldown/i.test(t)){
    notes=[
      'Δ 指标默认来自近7天 vs 前7天 compare 窗口',
      'Top Actions 已按优先级排序，但需结合库存/利润阻断判断',
      '如动作关联置信度低，先回看类目/生命周期语境'
    ];
  }else{
    return null;
  }
  const box=document.createElement('div');
  box.className='logic-note';
  const titleEl=document.createElement('div');
  titleEl.className='ln-title';
  titleEl.textContent='口径提示';
  const ul=document.createElement('ul');
  notes.forEach((s)=>{
    const li=document.createElement('li');
    li.textContent=s;
    ul.appendChild(li);
  });
  box.appendChild(titleEl);
  box.appendChild(ul);
  return box;
}

function _injectReadGuideAndNotes(){
  const content=_qs('#content');
  if(!content) return;
  const h1=content.querySelector('h1');
  if(!h1) return;
  const title=String(h1.textContent||'').trim();
  const anchor=_findMetaAnchor(content);
  if(!anchor) return;

  const guide=_buildReadGuide(title);
  if(guide) _insertAfter(anchor, guide);

  const note=_buildLogicNote(title);
  if(note){
    if(guide){
      _insertAfter(guide, note);
    }else{
      _insertAfter(anchor, note);
    }
  }
}

function _labelForHeading(text){
  const t=String(text||'').trim();
  if(!t) return null;
  const rules=[
    {label:'总览', cls:'label-overview', re:/(本期结论|本次重点|总览|概览|阶段分布|类目结构|近期重点|怎么读|怎么用)/i},
    {label:'行动', cls:'label-action', re:/(本周行动|行动清单|任务|解锁|Action Board|Action|执行|复盘)/i},
    {label:'风险', cls:'label-risk', re:/(Watchlist|Alerts|告警|风险|止损|走弱|断货|亏损)/i},
    {label:'Top', cls:'label-top', re:/(Top\s*\d*|Top ASIN|Top Actions|Top\s*(?:3|5|7|8|10|12|15|20))/i},
    {label:'下钻', cls:'label-drill', re:/(下钻|Drilldown|索引)/i},
    {label:'主题', cls:'label-topic', re:/(关键词主题|Keyword Topics|Topic)/i},
    {label:'生命周期', cls:'label-phase', re:/(生命周期|Phase)/i},
    {label:'类目', cls:'label-category', re:/(类目|Category)/i},
    {label:'ASIN', cls:'label-asin', re:/\bASIN\b/i},
    {label:'变化', cls:'label-change', re:/(Drivers|变化来源|delta)/i},
  ];
  for(let i=0;i<rules.length;i++){
    if(rules[i].re.test(t)) return rules[i];
  }
  return null;
}

function _decorateHeadingLabels(){
  const content=_qs('#content');
  if(!content) return;
  try{
    const hs=Array.from(content.querySelectorAll('h2,h3'));
    hs.forEach((h)=>{
      if(h.classList.contains('h-labeled')) return;
      const info=_labelForHeading(h.textContent||'');
      if(!info) return;
      h.classList.add('h-labeled');
      h.classList.add(info.cls);
      h.setAttribute('data-label', info.label);
    });
  }catch(e){}
}

function _countTablesAndRows(root){
  try{
    let tables=0;
    let rows=0;
    const nodes=[];
    if(root && root.querySelectorAll){
      nodes.push(root);
    }
    nodes.forEach((r)=>{
      (r.querySelectorAll('table')||[]).forEach((t)=>{
        tables++;
        try{
          const tb=t.tBodies && t.tBodies[0];
          if(tb && tb.rows) rows += tb.rows.length;
        }catch(e){}
      });
    });
    return {tables:tables, rows:rows};
  }catch(e){
    return {tables:0, rows:0};
  }
}

function _collapseDenseH3Blocks(){
  // 经验规则：当一个 H2 章节内出现很多 H3 区块时，默认折叠“非总览/非Top/非摘要”的 H3，避免页面过长。
  const content=_qs('#content');
  if(!content) return;
  const sections=Array.from(content.querySelectorAll('.section'));
  sections.forEach((sec)=>{
    // 只处理“直接子级 H3”（避免误伤 hero 三栏内部）
    const h3s=Array.from(sec.querySelectorAll(':scope > h3'));
    let secTitle='';
    try{
      const h2=sec.querySelector(':scope > h2');
      secTitle=String(h2 ? (h2.textContent||'') : '').trim();
    }catch(e){ secTitle=''; }
    const force = /ASIN\s*Focus|类目总览|类目下钻|阶段下钻|ASIN\s*下钻|关键词主题|生命周期/i.test(secTitle);
    if(!force && h3s.length < 5) return;

    h3s.forEach((h3)=>{
      try{
        if(!h3 || h3.closest('.hero-grid')) return;
        const txt=String(h3.textContent||'').trim();
        if(!txt) return;
        // 重要区块：保持展开（总览/Top/摘要等）
        const keepOpen = /总览|Top|摘要|关键|结论|Shop Alerts|本周行动/.test(txt);
        if(keepOpen) return;

        // 收集该 H3 block：h3 + 直到下一个 h3
        const nodes=[h3];
        let el=h3.nextElementSibling;
        while(el){
          if(el.tagName==='H3') break;
          nodes.push(el);
          el=el.nextElementSibling;
        }

        // 如果这个 block 很小（既没表也没 cards），就不折叠，避免“点来点去”
        let hasTable=false;
        let hasCards=false;
        nodes.forEach((n)=>{
          try{
            if(n && n.querySelector){
              if(n.querySelector('table')) hasTable=true;
              if(n.querySelector('.cards')) hasCards=true;
            }
          }catch(e){}
        });
        if(!(hasTable || hasCards)) return;

        const details=document.createElement('details');
        details.className='block-details';
        // 默认关闭（open=false）
        const summary=document.createElement('summary');
        const sumText=document.createElement('span');
        sumText.className='sum-text';
        sumText.textContent=txt;
        const meta=document.createElement('span');
        meta.className='sum-meta';
        // meta 内容稍后基于 block-body 计算
        meta.textContent='';
        summary.appendChild(sumText);
        summary.appendChild(meta);
        const body=document.createElement('div');
        body.className='block-body';
        // 先插入 details 再移动节点
        sec.insertBefore(details, h3);
        details.appendChild(summary);
        details.appendChild(body);
        nodes.forEach((n)=>body.appendChild(n));

        // meta：表/行数（可选）
        try{
          const cnt=_countTablesAndRows(body);
          if(cnt.tables>0){
            meta.textContent=''+cnt.tables+'表 · '+cnt.rows+'行';
          }else if(hasCards){
            meta.textContent='卡片区块';
          }
        }catch(e){}
      }catch(e){}
    });
  });
}

function _parseMaybeNum(t){
  const s=String(t||'')
    .replace(/[,￥¥$]/g,'')
    .replace(/%/g,'')
    .trim();
  if(!s) return null;
  const n=parseFloat(s);
  return Number.isFinite(n) ? n : null;
}

// 表格可读性增强：
// 1) 数字列右对齐（更容易扫对比）
// 2) 长文本列允许换行（减少横向滚动）
// 3) 超大表格默认折叠（减少信息轰炸）
function _autoStyleTables(){
  document.querySelectorAll('table').forEach((table)=>{
    const tbody=table.tBodies && table.tBodies[0];
    if(!tbody) return;
    const bodyRows=Array.from(tbody.rows||[]);
    if(bodyRows.length===0) return;

    const sample=bodyRows.slice(0, 25);
    const headRow=(table.tHead && table.tHead.rows && table.tHead.rows[0]) ? table.tHead.rows[0] : null;
    const colCount=Math.max(
      headRow ? headRow.cells.length : 0,
      ...sample.map(r=>r.cells.length)
    );
    for(let c=0;c<colCount;c++){
      let total=0, num=0, textCount=0, textLen=0, maxLen=0;
      sample.forEach((r)=>{
        const cell=r.cells[c];
        if(!cell) return;
        const t=(cell.innerText||'').trim();
        if(!t) return;
        total++;
        const n=_parseMaybeNum(t);
        if(n!==null){ num++; return; }
        const l=t.length;
        textCount++; textLen+=l; if(l>maxLen) maxLen=l;
      });
      if(total===0) continue;
      const numRatio=num/total;
      const isNum=(num>=3 && numRatio>=0.6);
      const avgText=(textCount>0)?(textLen/textCount):0;
      const isWrap=(!isNum && textCount>=3 && avgText>=24 && maxLen>=40);
      if(!(isNum || isWrap)) continue;
      const cls=isNum ? 'col-num' : 'col-wrap';
      try{
        if(headRow && headRow.cells && headRow.cells[c]) headRow.cells[c].classList.add(cls);
      }catch(e){}
      try{
        bodyRows.forEach((r)=>{ const cell=r.cells[c]; if(cell) cell.classList.add(cls); });
      }catch(e){}
    }

    // 仅折叠“超大表格”（避免隐藏 Top 表）
    try{
      const wrap=table.closest('.table-wrap');
      const rowsCount=bodyRows.length;
      if(wrap && rowsCount>=80){
        const details=document.createElement('details');
        details.className='table-details';
        const summary=document.createElement('summary');
        summary.className='table-summary';
        summary.textContent='展开/收起表格（'+rowsCount+' 行）';
        details.appendChild(summary);
        wrap.parentNode.insertBefore(details, wrap);
        details.appendChild(wrap);
      }
    }catch(e){}
  });
}

function _nthCapture(text, re, n){
  // 返回第 n 次匹配的第 1 个捕获组（没有捕获组则回退为完整匹配）
  try{
    if(!text) return null;
    const flags=re && re.flags ? re.flags : '';
    const gflags=flags.includes('g') ? flags : (flags + 'g');
    const r=new RegExp(re.source, gflags);
    const s=String(text||'');
    let m=null;
    let i=0;
    while((m=r.exec(s))!==null){
      i++;
      if(i===n){
        if(m.length>=2 && m[1]!==undefined) return m[1];
        return m[0];
      }
    }
    return null;
  }catch(e){
    return null;
  }
}

function _extractDashboardKpis(text){
  // 从 dashboard.md 的“大盘：...|...”行中抽取少量 KPI（可读性优先）
  try{
    const t=String(text||'');
    const range=_nthCapture(t, /近7天\(([^)]+)\)/, 1);

    const _toNum=(x)=>{
      try{
        const v=parseFloat(String(x||'').replace(/,/g,''));
        return Number.isFinite(v) ? v : null;
      }catch(e){ return null; }
    };
    const _fmtNum=(x, nd=2)=>{
      const v=_toNum(x);
      if(v===null) return String(x||'');
      try{
        return v.toLocaleString(undefined, {minimumFractionDigits: nd, maximumFractionDigits: nd});
      }catch(e){
        return v.toFixed(nd);
      }
    };
    const _fmtUSD=(x)=>{
      const v=_toNum(x);
      if(v===null) return String(x||'');
      const sign=v<0 ? '-' : '';
      return `${sign}$${_fmtNum(Math.abs(v), 2)}`;
    };
    const _fmtPct=(x, nd=1)=>{
      const v=_toNum(x);
      if(v===null) return String(x||'');
      return `${(v*100).toFixed(nd)}%`;
    };

    const sales1=_nthCapture(t, /Sales\s*=\s*([-\d.,]+)/, 1);
    const sales2=_nthCapture(t, /Sales\s*=\s*([-\d.,]+)/, 2);
    const spend1=_nthCapture(t, /AdSpend\s*=\s*([-\d.,]+)/, 1);
    const spend2=_nthCapture(t, /AdSpend\s*=\s*([-\d.,]+)/, 2);
    const tacos=_nthCapture(t, /TACOS\s*=\s*([-\d.,]+)/, 1);
    const profit7=_nthCapture(t, /Profit\s*=\s*([-\d.,]+)/, 1);
    const dprofit=_nthCapture(t, /ΔProfit\s*=\s*([-\d.,]+)/, 1);
    const mtacos=_nthCapture(t, /marginal_tacos\s*=\s*([-\d.,]+)/, 1);

    const out=[];
    if(sales1) out.push({k:'Sales（总）', v:_fmtUSD(sales1), s:''});
    if(spend1) out.push({k:'AdSpend（总）', v:_fmtUSD(spend1), s:''});
    if(tacos) out.push({k:'TACoS（总）', v:_fmtPct(tacos, 2), s:''});
    if(sales2) out.push({k:'Sales（近7天）', v:_fmtUSD(sales2), s:range?('近7天 '+range):'近7天'});
    if(spend2) out.push({k:'AdSpend（近7天）', v:_fmtUSD(spend2), s:range?('近7天 '+range):'近7天'});
    if(profit7) out.push({k:'Profit（近7天）', v:_fmtUSD(profit7), s:range?('近7天 '+range):'近7天'});
    if(dprofit) out.push({k:'ΔProfit（近7天vs前7天）', v:_fmtUSD(dprofit), s:''});
    if(mtacos) out.push({k:'边际TACoS', v:_fmtPct(mtacos, 2), s:''});
    return out;
  }catch(e){
    return [];
  }
}

function _cloneBlock(block){
  const frag=document.createElement('div');
  try{
    block.forEach((node)=>{
      const cloned=node.cloneNode(true);
      if(cloned && cloned.nodeType===1 && cloned.hasAttribute('id')) cloned.removeAttribute('id');
      if(cloned && cloned.querySelectorAll){
        cloned.querySelectorAll('[id]').forEach(el=>el.removeAttribute('id'));
      }
      frag.appendChild(cloned);
    });
  }catch(e){}
  return frag;
}

function _limitList(container, maxItems, filterFn, moreHref){
  if(!container) return;
  const list=container.querySelector('ul,ol');
  if(!list) return;
  const items=Array.from(list.children).filter(el=>String(el.tagName||'').toLowerCase()==='li');
  let kept=0;
  let totalMatch=0;
  items.forEach((li)=>{
    const keep=filterFn ? !!filterFn(li) : true;
    if(keep) totalMatch++;
    if(keep && kept<maxItems){
      kept++;
    }else{
      li.style.display='none';
    }
  });
  if(kept===0){
    const hint=document.createElement('div');
    hint.className='hint';
    hint.textContent='（暂无匹配项）';
    container.appendChild(hint);
    return;
  }
  if(totalMatch>kept){
    const hint=document.createElement('div');
    hint.className='hint';
    hint.innerHTML=moreHref ? `<a href=\"${moreHref}\">更多见下方</a>` : '更多见下方';
    container.appendChild(hint);
  }
}

function _buildDashboardHero(){
  // 把 dashboard 第一屏变成“概览KPI + 本周行动 + Shop Alerts”三栏（更像仪表盘）
  const content=_qs('#content');
  if(!content) return;
  const h2s=Array.from(content.querySelectorAll('h2'));
  const h2=h2s.find(x=>String(x.textContent||'').includes('本期结论'));
  if(!h2) return;
  const sec=h2.closest('.section');
  if(!sec) return;

  // 1) 概览 cards（来自“本期结论”下的列表，已被 cardize 成 .cards）
  let overviewCards=null;
  try{
    let el=h2.nextElementSibling;
    while(el){
      if(el.classList && el.classList.contains('cards')){ overviewCards=el; break; }
      if(el.tagName && String(el.tagName).match(/^H[2-3]$/)) break;
      el=el.nextElementSibling;
    }
  }catch(e){}
  if(!overviewCards) return;

  // 2) 抽 KPI：找到 “大盘：...” 这张卡
  let kpiCard=null;
  try{
    const cards=Array.from(overviewCards.querySelectorAll('.card-item'));
    kpiCard=cards.find(c=>String(c.innerText||'').trim().startsWith('大盘：')) || null;
  }catch(e){ kpiCard=null; }

  const kpis=_extractDashboardKpis(kpiCard ? (kpiCard.innerText||'') : '');
  const kpiGrid=document.createElement('div');
  kpiGrid.className='kpi-grid';
  if(kpis && kpis.length>0){
    kpis.forEach((it)=>{
      const card=document.createElement('div');
      card.className='kpi';
      const k=document.createElement('div');
      k.className='k';
      k.textContent=String(it.k||'');
      const v=document.createElement('div');
      v.className='v';
      const vText=String(it.v||'');
      v.textContent=vText;
      if(vText.trim().startsWith('-')){
        v.classList.add('neg');
      }else if(vText.trim()==='0' || vText.trim()==='0.0' || vText.trim()==='0.00'){
        v.classList.add('zero');
      }else{
        v.classList.add('pos');
      }
      card.appendChild(k);
      card.appendChild(v);
      if(it.s){
        const s=document.createElement('div');
        s.className='s';
        s.textContent=String(it.s||'');
        card.appendChild(s);
      }
      kpiGrid.appendChild(card);
    });
  }

  // 3) 不再在“本期结论”区展示“大盘全量指标”，避免与 Hero 重复

  // 4) 收集 “本周行动 / Shop Alerts / Campaign 优先排查” 三个区块（heading + cards）
  function findH3ByTextOrId(root, id, text){
    if(!root) return null;
    try{
      const byId=root.querySelector('#'+id);
      if(byId){
        const tag=String(byId.tagName||'');
        if(tag.match(/^H[2-4]$/)) return byId;
        const next=byId.nextElementSibling;
        if(next && String(next.tagName||'').match(/^H[2-4]$/)) return next;
      }
    }catch(e){}
    try{
      const hs=Array.from(root.querySelectorAll('h2,h3,h4'));
      return hs.find(x=>String(x.textContent||'').includes(text)) || null;
    }catch(e){
      return null;
    }
  }
  function pickBlock(h3, anchorId){
    const items=[];
    if(!h3) return items;
    try{
      const prev=h3.previousElementSibling;
      if(prev && anchorId && String(prev.id||'')===anchorId){
        items.push(prev);
      }
    }catch(e){}
    items.push(h3);
    let el=h3.nextElementSibling;
    while(el){
      if(el.tagName && String(el.tagName).match(/^H[2-3]$/)) break;
      items.push(el);
      el=el.nextElementSibling;
    }
    return items;
  }
  const weeklyH3=findH3ByTextOrId(sec,'weekly','本周行动') || findH3ByTextOrId(content,'weekly','本周行动');
  const alertsH3=findH3ByTextOrId(sec,'alerts','Shop Alerts') || findH3ByTextOrId(content,'alerts','Shop Alerts');
  const campaignH3=findH3ByTextOrId(sec,'campaign','Campaign 优先排查') || findH3ByTextOrId(content,'campaign','Campaign 优先排查');
  const weeklyBlock=pickBlock(weeklyH3, 'weekly');
  const alertsBlock=pickBlock(alertsH3, 'alerts');
  const campaignBlock=pickBlock(campaignH3, 'campaign');

  // 5) 构建 hero 四栏并插入到“概览 cards”原位置（避免打乱“快速入口”段落）
  const hero=document.createElement('div');
  hero.className='hero-grid';
  const col1=document.createElement('div');
  col1.className='hero-panel overview';
  const col2=document.createElement('div');
  col2.className='hero-panel weekly';
  const col3=document.createElement('div');
  col3.className='hero-panel alerts';
  const col4=document.createElement('div');
  col4.className='hero-panel campaign';
  hero.appendChild(col1);
  hero.appendChild(col2);
  hero.appendChild(col3);
  hero.appendChild(col4);

  try{
    // 先插入再移动节点（避免 reference node 已被移动导致 insertBefore 报错）
    sec.insertBefore(hero, overviewCards);
  }catch(e){ return; }

  // 5.1) 概览列：KPI + 概览 cards
  const t1=document.createElement('div');
  t1.className='hero-title';
  t1.textContent='概览（先抓重点）';
  col1.appendChild(t1);
  if(kpis && kpis.length>0) col1.appendChild(kpiGrid);
  // 本期结论卡片移到下方折叠区，避免与行动/预警重复

  // 5.2) 本周行动列
  if(weeklyBlock.length===0){
    const t2=document.createElement('div');
    t2.className='hero-title';
    t2.textContent='本周行动（Top 3）';
    col2.appendChild(t2);
    const hint=document.createElement('div');
    hint.className='hint';
    hint.textContent='（暂无本周行动清单）';
    col2.appendChild(hint);
  }else{
    weeklyBlock.forEach((x)=>col2.appendChild(x));
  }

  // 5.3) Shop Alerts 列
  if(alertsBlock.length===0){
    const t3=document.createElement('div');
    t3.className='hero-title';
    t3.textContent='Shop Alerts（规则化）';
    col3.appendChild(t3);
    const hint=document.createElement('div');
    hint.className='hint';
    hint.textContent='（暂无 Shop Alerts）';
    col3.appendChild(hint);
  }else{
    alertsBlock.forEach((x)=>col3.appendChild(x));
  }

  // 5.4) Campaign 优先排查列
  if(campaignBlock.length===0){
    const t4=document.createElement('div');
    t4.className='hero-title';
    t4.textContent='Campaign 优先排查（Top 3）';
    col4.appendChild(t4);
    const hint=document.createElement('div');
    hint.className='hint';
    hint.textContent='（暂无 Campaign 聚合）';
    col4.appendChild(hint);
  }else{
    campaignBlock.forEach((x)=>col4.appendChild(x));
  }

  let insertAfter=hero;

  // Hero 内只显示 Top 2（P0/P1 优先），减少第一屏噪音
  try{
    _limitList(col2, 2, null, '');
    _limitList(col3, 2, (li)=>/\\bP0\\b|\\bP1\\b/.test(String(li.textContent||'')), '');
    _limitList(col4, 2, null, '');
  }catch(e){}

  // 5.6) 本期结论补充已移除（避免重复占用第一屏空间）

  // 6) 移除本期结论区头部（避免重复模块），快速入口移入概览列
  try{
    if(h2){ h2.remove(); }
    const qe=sec.querySelector('.quick-entry');
    if(qe){ col1.appendChild(qe); }
    if(overviewCards && overviewCards.parentNode){
      overviewCards.parentNode.removeChild(overviewCards);
    }
  }catch(e){}

  // 7) 修复目录锚点：把本周行动/Shop Alerts 的 TOC 指向“hero 上下两个锚点”
  try{
    const weeklyAnchor=document.createElement('div');
    weeklyAnchor.id='weekly-anchor';
    weeklyAnchor.className='anchor-spacer anchor-weekly';
    hero.insertAdjacentElement('beforebegin', weeklyAnchor);
    const alertsAnchor=document.createElement('div');
    alertsAnchor.id='alerts-anchor';
    alertsAnchor.className='anchor-spacer anchor-alerts';
    hero.insertAdjacentElement('afterend', alertsAnchor);

    const toc=_qs('#toc');
    if(toc){
      Array.from(toc.querySelectorAll('a.toc-item')).forEach((a)=>{
        const t=String(a.textContent||'');
        if(t.includes('本期结论')){
          const li=a.closest('li');
          if(li) li.remove();
          return;
        }
        if(t.includes('本周行动')) a.setAttribute('href','#weekly-anchor');
        if(t.includes('Shop Alerts')) a.setAttribute('href','#alerts-anchor');
      });
    }
  }catch(e){}
}

function _cardizeListAfterHeading(content, headingText){
  try{
    const hs=Array.from(content.querySelectorAll('h1,h2,h3,h4'));
    const h=hs.find(x=>String(x.textContent||'').includes(headingText));
    if(!h) return;
    let el=h.nextElementSibling;
    while(el && !(String(el.tagName||'').match(/^H[1-4]$/))){
      if(el.tagName==='UL' || el.tagName==='OL'){
        const ul=el;
        const items=Array.from(ul.querySelectorAll(':scope > li'));
        if(items.length===0) return;
        const grid=document.createElement('div');
        grid.className='cards';
        items.forEach((li)=>{
          const card=document.createElement('div');
          card.className='card-item';
          card.innerHTML=li.innerHTML;
          grid.appendChild(card);
        });
        ul.replaceWith(grid);
        return;
      }
      el=el.nextElementSibling;
    }
  }catch(e){}
}

function _cardizeKeyLists(){
  const content=_qs('#content');
  if(!content) return;
  // 让“第一屏结论/告警”更像仪表盘卡片（不改口径，只改展示）
  _cardizeListAfterHeading(content, '本次重点');
  _cardizeListAfterHeading(content, '本期结论');
  _cardizeListAfterHeading(content, '近期重点');
  _cardizeListAfterHeading(content, '本周行动清单');
  _cardizeListAfterHeading(content, 'Shop Alerts');
  _cardizeListAfterHeading(content, 'Campaign 优先排查');
}

function _buildOwnerKanban(){
  const content=_qs('#content');
  if(!content) return;
  const h1=content.querySelector('h1');
  if(!h1 || !String(h1.textContent||'').includes('Owner 汇总')) return;

  const layout=_qs('.layout');
  if(layout) layout.classList.add('single');
  const toc=_qs('#tocCard');
  if(toc) toc.style.display='none';

  const insertAfter=(h1.nextElementSibling && h1.nextElementSibling.tagName==='P') ? h1.nextElementSibling : h1;
  const board=document.createElement('div');
  board.className='kanban';

  const buildCol=(title, cls)=>{
    const col=document.createElement('div');
    col.className='kanban-col ' + cls;
    const head=document.createElement('div');
    head.className='kanban-head';
    const headLabel=document.createElement('div');
    headLabel.textContent=title;
    const headCount=document.createElement('div');
    headCount.className='kanban-count';
    head.appendChild(headLabel);
    head.appendChild(headCount);
    const body=document.createElement('div');
    body.className='kanban-body';
    const hs=Array.from(content.querySelectorAll('h2'));
    const h=hs.find(x=>String(x.textContent||'').trim()===title || String(x.textContent||'').includes(title));
    const groups={};
    const stats={total:0,p0:0,p1:0,p2:0};
    const makeTag=(text, cls)=>{
      const t=document.createElement('span');
      t.className='kanban-tag' + (cls ? (' '+cls) : '');
      t.textContent=text;
      return t;
    };
    const parseMeta=(li)=>{
      try{
        const tmp=document.createElement('div');
        tmp.innerHTML=li.innerHTML;
        const codes=Array.from(tmp.querySelectorAll('code'));
        let pr='', owner='', shop='';
        codes.forEach((c)=>{
          const t=String(c.textContent||'').trim();
          if(!pr && /^P[0-9]$/.test(t)){ pr=t; return; }
          if(!owner && /运营|供应链|财务|美工|产品/.test(t)){ owner=t; return; }
          if(!shop && /^[0-9A-Z]+-[A-Z]{2,}$/.test(t)){ shop=t; return; }
        });
        const linkEl=tmp.querySelector('a');
        const linkHtml=linkEl ? linkEl.outerHTML : '';
        if(linkEl) linkEl.remove();
        codes.forEach((c)=>c.remove());
        let desc=(tmp.textContent||'').replace(/^[：:\\s-]+/,'').trim();
        desc=desc.replace(/（\\s*）/g,'').replace(/\\(\\s*\\)/g,'').trim();
        return {pr, owner, shop, desc, linkHtml};
      }catch(e){
        return {pr:'', owner:'', shop:'', desc: li.textContent||'', linkHtml:''};
      }
    };
    const buildCard=(li)=>{
      const meta=parseMeta(li);
      const card=document.createElement('div');
      card.className='kanban-card';
      if(meta.pr==='P0') card.classList.add('prio-p0');
      if(meta.pr==='P1') card.classList.add('prio-p1');
      if(meta.pr==='P2') card.classList.add('prio-p2');
      const metaRow=document.createElement('div');
      metaRow.className='kanban-meta';
      if(meta.pr) metaRow.appendChild(makeTag(meta.pr, meta.pr.toLowerCase()));
      if(meta.owner) metaRow.appendChild(makeTag(meta.owner, 'owner'));
      if(meta.shop) metaRow.appendChild(makeTag(meta.shop, 'shop'));
      if(metaRow.children.length>0) card.appendChild(metaRow);
      const desc=document.createElement('div');
      desc.className='kanban-desc';
      desc.textContent=meta.desc || '';
      card.appendChild(desc);
      if(meta.linkHtml){
        const link=document.createElement('div');
        link.className='kanban-link';
        link.innerHTML=meta.linkHtml;
        card.appendChild(link);
      }
      if(meta.shop) card.dataset.shop = meta.shop;
      stats.total += 1;
      if(meta.pr==='P0') stats.p0 += 1;
      if(meta.pr==='P1') stats.p1 += 1;
      if(meta.pr==='P2') stats.p2 += 1;
      return {card, meta};
    };
    if(h){
      let el=h.nextElementSibling;
      while(el && !(String(el.tagName||'').match(/^H[1-4]$/))){
        if(el.tagName==='UL' || el.tagName==='OL'){
          const items=Array.from(el.querySelectorAll(':scope > li'));
          if(items.length){
            items.forEach((li)=>{
              const built=buildCard(li);
              const ownerKey=built.meta.owner || '其他';
              if(!groups[ownerKey]) groups[ownerKey]=[];
              groups[ownerKey].push(built.card);
            });
          }
          const sec=(h.parentElement && h.parentElement.classList && h.parentElement.classList.contains('section')) ? h.parentElement : null;
          if(sec){ sec.style.display='none'; } else { h.style.display='none'; el.style.display='none'; }
          break;
        }
        el=el.nextElementSibling;
      }
    }
    const ownerKeys=Object.keys(groups);
    if(ownerKeys.length){
      ownerKeys.sort((a,b)=>a.localeCompare(b));
      ownerKeys.forEach((k)=>{
        const group=document.createElement('div');
        group.className='kanban-group';
        const gh=document.createElement('div');
        gh.className='kanban-group-head';
        const lh=document.createElement('div');
        lh.textContent=k;
        const rh=document.createElement('div');
        rh.textContent=String(groups[k].length);
        gh.appendChild(lh);
        gh.appendChild(rh);
        group.appendChild(gh);
        groups[k].forEach((card)=>group.appendChild(card));
        body.appendChild(group);
      });
    }else{
      const empty=document.createElement('div');
      empty.className='hint';
      empty.textContent='（暂无数据）';
      body.appendChild(empty);
    }
    headCount.textContent=stats.total>0 ? `${stats.total} · P0 ${stats.p0} · P1 ${stats.p1}` : '0';
    col.appendChild(head);
    col.appendChild(body);
    return col;
  };

  board.appendChild(buildCol('Top 风险', 'risk'));
  board.appendChild(buildCol('Top 机会', 'opp'));
  board.appendChild(buildCol('本周行动', 'action'));

  insertAfter.insertAdjacentElement('afterend', board);

  _initOwnerShopSelect(board, h1);
}

function _initOwnerShopSelect(board, h1){
  try{
    if(!board || !h1) return;
    const cards=Array.from(board.querySelectorAll('.kanban-card'));
    const shops=[];
    cards.forEach((c)=>{
      const s=String(c.dataset.shop||'').trim();
      if(s && shops.indexOf(s)<0) shops.push(s);
    });
    if(shops.length===0){
      return;
    }
    shops.sort((a,b)=>a.localeCompare(b));
    const bar=document.createElement('div');
    bar.className='page-title-bar';
    h1.parentNode.insertBefore(bar, h1);
    bar.appendChild(h1);

    const filterWrap=document.createElement('div');
    filterWrap.className='shop-filter';
    const label=document.createElement('span');
    label.textContent='店铺筛选';
    const select=document.createElement('select');
    const optAll=document.createElement('option');
    optAll.value='__ALL__';
    optAll.textContent='全部';
    select.appendChild(optAll);
    shops.forEach((s)=>{
      const opt=document.createElement('option');
      opt.value=s;
      opt.textContent=s;
      select.appendChild(opt);
    });
    filterWrap.appendChild(label);
    filterWrap.appendChild(select);
    const hint=document.createElement('span');
    hint.className='shop-filter-hint';
    hint.textContent='（仅影响展示）';
    filterWrap.appendChild(hint);
    bar.appendChild(filterWrap);

    const apply=(selected)=>{
      const sel=String(selected||'__ALL__');
      const cols=Array.from(board.querySelectorAll('.kanban-col'));
      cols.forEach((col)=>{
        let visible=0, p0=0, p1=0;
        const groups=Array.from(col.querySelectorAll('.kanban-group'));
        groups.forEach((g)=>{
          let gVisible=0;
          Array.from(g.querySelectorAll('.kanban-card')).forEach((card)=>{
            const shop=String(card.dataset.shop||'').trim();
            const show = (sel==='__ALL__') || !shop || (shop===sel);
            card.style.display = show ? '' : 'none';
            if(show){
              gVisible += 1;
              visible += 1;
              if(card.classList.contains('prio-p0')) p0 += 1;
              if(card.classList.contains('prio-p1')) p1 += 1;
            }
          });
          g.style.display = gVisible>0 ? '' : 'none';
        });
        let empty=col.querySelector('.kanban-empty');
        if(visible===0){
          if(!empty){
            empty=document.createElement('div');
            empty.className='kanban-empty hint';
            empty.textContent='（无匹配店铺）';
            col.querySelector('.kanban-body')?.appendChild(empty);
          }
        }else{
          if(empty) empty.remove();
        }
        const headCount=col.querySelector('.kanban-count');
        if(headCount) headCount.textContent = visible>0 ? `${visible} · P0 ${p0} · P1 ${p1}` : '0';
      });
    };
    select.addEventListener('change', ()=>apply(select.value));
    apply(select.value);
  }catch(e){}
}

function _decorateBadges(){
  const content=_qs('#content');
  if(!content) return;
  content.querySelectorAll('code').forEach((el)=>{
    const t=(el.innerText||'').trim();
    if(t==='P0' || t==='P1' || t==='P2'){
      el.classList.add('badge', t.toLowerCase());
      return;
    }
    // “本周行动/概览”标签：让运营更容易扫到类型
    if(t==='止损'){
      el.classList.add('badge','tag-stop');
      return;
    }
    if(t==='放量'){
      el.classList.add('badge','tag-scale');
      return;
    }
    if(t==='排查'){
      el.classList.add('badge','tag-review');
      return;
    }
    if(t==='放量被阻断'){
      el.classList.add('badge','tag-blocked');
      return;
    }
  });
}

	function _decorateCardPriority(){
	  const content=_qs('#content');
	  if(!content) return;
	  content.querySelectorAll('.card-item,.kanban-card').forEach((card)=>{
	    if(card.querySelector('code.badge.p0')){ card.classList.add('prio-p0'); return; }
	    if(card.querySelector('code.badge.p1')){ card.classList.add('prio-p1'); return; }
	    if(card.querySelector('code.badge.p2')){ card.classList.add('prio-p2'); return; }
	  });
	}

	function _prettyPhase(p){
	  const s=String(p||'').trim().toLowerCase();
	  const map={
	    'pre_launch':'pre',
	    'launch':'launch',
	    'growth':'growth',
	    'stable':'stable',
	    'mature':'mature',
	    'decline':'decline',
	    'inactive':'inactive',
	    'unknown':'unknown',
	  };
	  return map[s] || s.replace(/_/g,'');
	}
	function _parseTimelineText(t){
	  const raw=String(t||'').trim();
	  if(!raw.startsWith('tl:')) return null;
	  const s=raw.slice(3);
	  const parts=s.split('|').map(x=>String(x||'').trim()).filter(Boolean);
	  const segText=parts[0] || '';
	  const flags=parts.slice(1);
	  const items=segText.split(';').map(x=>String(x||'').trim()).filter(Boolean);
	  const segs=[];
	  items.forEach((it)=>{
	    const segParts=it.split('=');
	    if(segParts.length!==2) return;
	    const phase=String(segParts[0]||'').trim().toLowerCase();
	    const days=parseFloat(String(segParts[1]||'').trim());
	    if(!phase || !Number.isFinite(days) || days<=0) return;
	    segs.push({phase, days});
	  });
	  if(segs.length<=0) return null;
	  return {segs, recent: flags.includes('chg14')};
	}
	function _renderTimelines(){
	  const content=_qs('#content');
	  if(!content) return;
	  content.querySelectorAll('code').forEach((el)=>{
	    const txt=(el.innerText||'').trim();
	    if(!txt.startsWith('tl:')) return;
	    const parsed=_parseTimelineText(txt);
	    if(!parsed) return;
	    const segs=parsed.segs || [];
	    if(segs.length<=0) return;
	    const total=segs.reduce((a,b)=>a+(b.days||0),0);
	    const bar=document.createElement('div');
	    bar.className=parsed.recent ? 'timeline recent' : 'timeline';
	    segs.forEach((seg)=>{
	      const d=seg.days||0;
	      const part=document.createElement('div');
	      const phaseCls=String(seg.phase||'unknown').replace(/[^a-z0-9_\-]+/g,'');
	      part.className='seg phase-'+(phaseCls || 'unknown');
	      part.style.flex=String(d);
	      part.title=String(seg.phase||'unknown') + ' ' + String(d) + 'd';
	      if(total>0 && (d/total)>=0.18){
	        const lbl=document.createElement('span');
	        lbl.className='lbl';
	        lbl.textContent=_prettyPhase(seg.phase);
	        part.appendChild(lbl);
	      }
	      bar.appendChild(part);
	    });
	    el.replaceWith(bar);
	  });
	}

	function _initToTop(){
	  const btn=_qs('#toTop');
	  if(!btn) return;
	  const toggle=()=>{
    if(window.scrollY>420){ btn.classList.add('show'); }
    else{ btn.classList.remove('show'); }
  };
  btn.addEventListener('click', ()=>window.scrollTo({top:0, behavior:'smooth'}));
  window.addEventListener('scroll', toggle, {passive:true});
  toggle();
}

document.addEventListener('DOMContentLoaded', ()=>{
  _initTableSort();
  _decorateHeadingLabels();
  _normalizeAnchorsAndBuildToc();
  _wrapSections();
  _buildOwnerKanban();
  _decorateQuickEntry();
  _initTocActive();
  _cardizeKeyLists();
  _buildDashboardHero();
  _collapseDenseH3Blocks();
  _decorateTones();
  _injectReadGuideAndNotes();
  _decorateBadges();
  _decorateCardPriority();
  _renderTimelines();
  _autoStyleTables();
  _initToTop();
  _initExpandCollapseAll();
  _openDetailsForHash();
  window.addEventListener('hashchange', _openDetailsForHash);
});
"""

        doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="topbar">
    <div class="wrap">
      <div class="topbar-inner">
        <div class="title">{html.escape(title)}</div>
        <div class="actions">{nav_html}</div>
      </div>
    </div>
  </div>
  <div class="wrap">
    <div class="layout">
      <aside class="card toc" id="tocCard">
        <div class="toc-title">目录</div>
        <div class="toc-body" id="toc"></div>
        <div class="hint">提示：点击目录/表头可跳转与排序（离线可用）。</div>
      </aside>
      <main class="card content" id="content">
{body}
        <div class="hint">本页由程序从 Markdown 自动生成（离线可打开）。如需编辑/追溯口径，请以同目录的 .md/.csv 为准。</div>
      </main>
    </div>
  </div>
  <button class="to-top" id="toTop" title="回到顶部">↑</button>
  <script>{js}</script>
</body>
</html>
"""
        out_path.write_text(doc, encoding="utf-8")
    except Exception:
        return


def _read_csv_header(path: Path) -> List[str]:
    """
    读取 CSV 第一行作为列名（避免加载整表）。

    说明：
    - 输出 CSV 使用 utf-8-sig（带 BOM）兼容 Excel，这里也用 utf-8-sig 读取。
    """
    try:
        if path is None or (not path.exists()):
            return []
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                return [str(x or "") for x in row]
        return []
    except Exception:
        return []


def write_dashboard_schema_manifest(
    dashboard_dir: Path,
    shop: str,
    stage: str,
    date_start: str,
    date_end: str,
) -> Optional[Path]:
    """
    输出 dashboard/schema_manifest.json：用于 SSOT（口径/字段）一致性审计与回归。

    内容：
    - 记录 dashboard/ 下每个 CSV 的列名列表
    - 记录 JSON 文件的顶层 keys（仅用于快速定位结构变化）

    说明：
    - 这是“审计/治理”产物，不参与任何算数逻辑。
    - 失败不崩：写失败不影响主流程其它输出。
    """
    try:
        if dashboard_dir is None:
            return None
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        out_path = dashboard_dir / "schema_manifest.json"

        items: List[Dict[str, object]] = []
        try:
            for p in sorted(dashboard_dir.glob("*")):
                if not p.is_file():
                    continue
                suf = p.suffix.lower()
                if suf == ".csv":
                    items.append(
                        {
                            "path": p.name,
                            "type": "csv",
                            "columns": _read_csv_header(p),
                        }
                    )
                elif suf == ".json":
                    try:
                        obj = json.loads(p.read_text(encoding="utf-8"))
                        keys = sorted(list(obj.keys())) if isinstance(obj, dict) else []
                    except Exception:
                        keys = []
                    items.append(
                        {
                            "path": p.name,
                            "type": "json",
                            "keys": keys,
                        }
                    )
        except Exception:
            items = []

        manifest = {
            "schema_manifest_version": 1,
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "shop": str(shop or "").strip(),
            "stage": str(stage or "").strip(),
            "date_start": str(date_start or "").strip(),
            "date_end": str(date_end or "").strip(),
            "dashboard_dir": "dashboard",
            "items": items,
        }
        out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path
    except Exception:
        return None


def _pick_main_window(lifecycle_windows: pd.DataFrame) -> pd.DataFrame:
    """
    为每个 asin 选“主口径窗口”，优先 since_first_stock_to_date，不存在则回退 cycle_to_date。
    """
    if lifecycle_windows is None or lifecycle_windows.empty:
        return pd.DataFrame()
    if "asin" not in lifecycle_windows.columns or "window_type" not in lifecycle_windows.columns:
        return pd.DataFrame()
    w = lifecycle_windows.copy()
    w["asin_norm"] = w["asin"].astype(str).str.upper().str.strip()
    w["window_type"] = w["window_type"].astype(str)
    main = w[w["window_type"] == "since_first_stock_to_date"].copy()
    if main.empty:
        main = w[w["window_type"] == "cycle_to_date"].copy()
    if main.empty:
        return pd.DataFrame()
    # 每个 asin 取 1 行（理论上就 1 行），这里做防御性去重
    main = main.sort_values(["asin_norm"])
    main = main.drop_duplicates("asin_norm", keep="first")
    return main


def _pick_compare_window(lifecycle_windows: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    选 compare_N 的滚动环比窗口（最近 N 天 vs 前 N 天）。
    """
    if lifecycle_windows is None or lifecycle_windows.empty:
        return pd.DataFrame()
    if "asin" not in lifecycle_windows.columns or "window_type" not in lifecycle_windows.columns:
        return pd.DataFrame()
    w = lifecycle_windows.copy()
    w["asin_norm"] = w["asin"].astype(str).str.upper().str.strip()
    w["window_type"] = w["window_type"].astype(str)
    wdays = int(window_days or 0)
    if wdays <= 0:
        return pd.DataFrame()
    # 防御性：window_days 列可能缺失（例如测试/精简数据）
    try:
        if "window_days" in w.columns:
            wd = pd.to_numeric(w["window_days"], errors="coerce").fillna(0).astype(int)
        else:
            wd = pd.Series([0] * int(len(w)), index=w.index)
        mask = w["window_type"].str.startswith("compare_") & (wd == wdays)
        out = w[mask].copy()
    except Exception:
        return pd.DataFrame()
    if out.empty:
        return pd.DataFrame()
    out = out.sort_values(["asin_norm"])
    out = out.drop_duplicates("asin_norm", keep="first")
    return out


def build_shop_scorecard_json(
    shop: str,
    stage: str,
    date_start: str,
    date_end: str,
    diagnostics: Dict[str, object],
) -> Dict[str, object]:
    """
    生成店铺级 scorecard JSON（给 dashboard 用）。
    """
    sc = diagnostics.get("shop_scorecard") if isinstance(diagnostics, dict) else None
    sc2 = sc if isinstance(sc, dict) else {}
    return {
        "shop": shop,
        "stage_profile": stage,
        "date_range": {"date_start": date_start, "date_end": date_end},
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "scorecard": sc2,
    }


def _count_truthy_series(s: pd.Series) -> int:
    """
    统计“真值”数量（兼容 0/1、True/False、"true"/"1" 等常见写法）。
    """
    try:
        if s is None:
            return 0
        ss = s.copy()
        try:
            nums = pd.to_numeric(ss, errors="coerce")
            mask_num = nums.fillna(0).astype(float) > 0
        except Exception:
            mask_num = pd.Series([False] * int(len(ss)), index=ss.index)
        try:
            txt = ss.astype(str).fillna("").str.strip().str.lower()
            mask_txt = txt.isin({"1", "true", "t", "yes", "y"})
        except Exception:
            mask_txt = pd.Series([False] * int(len(ss)), index=ss.index)
        try:
            return int((mask_num | mask_txt).sum())
        except Exception:
            return 0
    except Exception:
        return 0


def build_actions_summary(action_board: Optional[pd.DataFrame]) -> Dict[str, int]:
    """
    Action Board 计数摘要（只用于“入口/抓重点”，不参与任何算数口径）。

    说明：
    - 统计对象：`dashboard/action_board.csv`（去重后的运营视图，TopN）
    - 输出用途：run 级 `START_HERE.*` 的轻量汇总表
    """
    out = {
        "total": 0,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 0,
        "blocked_count": 0,
        "needs_manual_confirm_count": 0,
    }
    try:
        df = action_board.copy() if isinstance(action_board, pd.DataFrame) else pd.DataFrame()
        if df is None or df.empty:
            return out
        out["total"] = int(len(df))

        # priority
        if "priority" in df.columns:
            p = df["priority"].astype(str).fillna("").str.strip().str.upper()
            vc = p.value_counts(dropna=False).to_dict()
            out["p0_count"] = int(vc.get("P0", 0))
            out["p1_count"] = int(vc.get("P1", 0))
            out["p2_count"] = int(vc.get("P2", 0))

        # blocked / needs_manual_confirm
        if "blocked" in df.columns:
            out["blocked_count"] = _count_truthy_series(df["blocked"])
        if "needs_manual_confirm" in df.columns:
            out["needs_manual_confirm_count"] = _count_truthy_series(df["needs_manual_confirm"])

        return out
    except Exception:
        return out


def build_watchlists_summary(
    profit_reduce_watchlist: Optional[pd.DataFrame],
    inventory_risk_watchlist: Optional[pd.DataFrame],
    inventory_sigmoid_watchlist: Optional[pd.DataFrame],
    profit_guard_watchlist: Optional[pd.DataFrame],
    oos_with_ad_spend_watchlist: Optional[pd.DataFrame],
    spend_up_no_sales_watchlist: Optional[pd.DataFrame],
    phase_down_recent_watchlist: Optional[pd.DataFrame],
    scale_opportunity_watchlist: Optional[pd.DataFrame],
) -> Dict[str, int]:
    """
    Watchlists 行数计数（只用于“入口/抓重点”，不参与任何算数口径）。
    """
    out = {
        "profit_reduce_count": 0,
        "inventory_risk_count": 0,
        "inventory_sigmoid_count": 0,
        "profit_guard_count": 0,
        "oos_with_ad_spend_count": 0,
        "spend_up_no_sales_count": 0,
        "phase_down_recent_count": 0,
        "scale_opportunity_count": 0,
    }
    try:
        out["profit_reduce_count"] = int(len(profit_reduce_watchlist)) if isinstance(profit_reduce_watchlist, pd.DataFrame) else 0
        out["inventory_risk_count"] = int(len(inventory_risk_watchlist)) if isinstance(inventory_risk_watchlist, pd.DataFrame) else 0
        out["inventory_sigmoid_count"] = int(len(inventory_sigmoid_watchlist)) if isinstance(inventory_sigmoid_watchlist, pd.DataFrame) else 0
        out["profit_guard_count"] = int(len(profit_guard_watchlist)) if isinstance(profit_guard_watchlist, pd.DataFrame) else 0
        out["oos_with_ad_spend_count"] = int(len(oos_with_ad_spend_watchlist)) if isinstance(oos_with_ad_spend_watchlist, pd.DataFrame) else 0
        out["spend_up_no_sales_count"] = int(len(spend_up_no_sales_watchlist)) if isinstance(spend_up_no_sales_watchlist, pd.DataFrame) else 0
        out["phase_down_recent_count"] = int(len(phase_down_recent_watchlist)) if isinstance(phase_down_recent_watchlist, pd.DataFrame) else 0
        out["scale_opportunity_count"] = int(len(scale_opportunity_watchlist)) if isinstance(scale_opportunity_watchlist, pd.DataFrame) else 0
        return out
    except Exception:
        return out


def build_category_summary(
    product_analysis_shop: Optional[pd.DataFrame],
    lifecycle_board: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    生成商品分类汇总表（category_summary.csv）。

    目标：
    - 让运营先看“类目维度”再下钻到产品（同类对比更快）
    - 把动态生命周期(current_phase/cycle_id)作为解释语境的一部分（至少在类目里体现 phase 分布）
    """
    if product_analysis_shop is None or product_analysis_shop.empty:
        return pd.DataFrame()

    pa = product_analysis_shop.copy()
    if "ASIN" not in pa.columns:
        return pd.DataFrame()

    # 数值化（只转我们会用到的列，避免太慢）
    for col in ("销售额", "订单量", "Sessions", "广告花费", "广告销售额", "广告订单量", "毛利润"):
        if col in pa.columns:
            pa[col] = pd.to_numeric(pa[col], errors="coerce").fillna(0.0)

    pa["asin_norm"] = pa["ASIN"].astype(str).str.upper().str.strip()
    pa = pa[(pa["asin_norm"] != "") & (pa["asin_norm"].str.lower() != "nan")].copy()
    if pa.empty:
        return pd.DataFrame()

    # 先按 ASIN 汇总
    agg_map = {}
    if "销售额" in pa.columns:
        agg_map["销售额"] = "sum"
    if "订单量" in pa.columns:
        agg_map["订单量"] = "sum"
    if "Sessions" in pa.columns:
        agg_map["Sessions"] = "sum"
    if "广告花费" in pa.columns:
        agg_map["广告花费"] = "sum"
    if "广告销售额" in pa.columns:
        agg_map["广告销售额"] = "sum"
    if "广告订单量" in pa.columns:
        agg_map["广告订单量"] = "sum"
    if "毛利润" in pa.columns:
        agg_map["毛利润"] = "sum"

    asin_sum = pa.groupby("asin_norm", dropna=False, as_index=False).agg(agg_map).copy()
    if asin_sum.empty:
        return pd.DataFrame()

    # 生命周期/产品信息（来自 lifecycle_board，已包含 product_name/product_category/current_phase/cycle_id 等）
    meta = pd.DataFrame()
    try:
        b = lifecycle_board.copy() if lifecycle_board is not None else pd.DataFrame()
        if b is not None and not b.empty and "asin" in b.columns:
            meta = b.copy()
            meta["asin_norm"] = meta["asin"].astype(str).str.upper().str.strip()
            # 只保留需要的字段
            keep = ["asin_norm"]
            for c in (
                "product_name",
                "product_category",
                "current_phase",
                "cycle_id",
                "inventory",
                "flag_low_inventory",
                "flag_oos",
            ):
                if c in meta.columns:
                    keep.append(c)
            meta = meta[keep].drop_duplicates("asin_norm", keep="first").copy()
    except Exception:
        meta = pd.DataFrame()

    merged = asin_sum.merge(meta, on="asin_norm", how="left")
    # 分类兜底：空/缺失/未分类统一归为“（未分类）”
    if "product_category" in merged.columns:
        merged["product_category"] = merged["product_category"].map(_norm_product_category)
    else:
        merged["product_category"] = "（未分类）"

    # 类目汇总（避免 groupby.apply 的 FutureWarning，且更快）
    for col in ("销售额", "订单量", "Sessions", "广告花费", "广告销售额", "广告订单量", "毛利润"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        else:
            merged[col] = 0.0
    for col in ("flag_low_inventory", "flag_oos"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
        else:
            merged[col] = 0

    g = merged.groupby("product_category", dropna=False)
    out = (
        g.agg(
            asin_count=("asin_norm", "size"),
            sales_total=("销售额", "sum"),
            orders_total=("订单量", "sum"),
            sessions_total=("Sessions", "sum"),
            ad_spend_total=("广告花费", "sum"),
            ad_sales_total=("广告销售额", "sum"),
            ad_orders_total=("广告订单量", "sum"),
            profit_total=("毛利润", "sum"),
            low_inventory_asin_count=("flag_low_inventory", "sum"),
            oos_asin_count=("flag_oos", "sum"),
        )
        .reset_index()
    )

    # 派生指标（类目级）
    # 自然 vs 广告拆分（类目级）：在无法拿到“自然销售额/订单量”字段时，用 总-广告 做确定性推导兜底
    try:
        if "sales_total" in out.columns and "ad_sales_total" in out.columns:
            out["organic_sales_total"] = (pd.to_numeric(out["sales_total"], errors="coerce").fillna(0.0) - pd.to_numeric(out["ad_sales_total"], errors="coerce").fillna(0.0)).clip(lower=0.0)
        else:
            out["organic_sales_total"] = 0.0
        if "orders_total" in out.columns and "ad_orders_total" in out.columns:
            out["organic_orders_total"] = (pd.to_numeric(out["orders_total"], errors="coerce").fillna(0.0) - pd.to_numeric(out["ad_orders_total"], errors="coerce").fillna(0.0)).clip(lower=0.0)
        else:
            out["organic_orders_total"] = 0.0
        out["organic_sales_share_total"] = out.apply(lambda r: (r["organic_sales_total"] / r["sales_total"]) if float(r.get("sales_total", 0.0) or 0.0) > 0 else 0.0, axis=1)
        out["organic_orders_share_total"] = out.apply(lambda r: (r["organic_orders_total"] / r["orders_total"]) if float(r.get("orders_total", 0.0) or 0.0) > 0 else 0.0, axis=1)
    except Exception:
        out["organic_sales_total"] = 0.0
        out["organic_orders_total"] = 0.0
        out["organic_sales_share_total"] = 0.0
        out["organic_orders_share_total"] = 0.0

    out["tacos_total"] = out.apply(lambda r: (r["ad_spend_total"] / r["sales_total"]) if float(r["sales_total"] or 0.0) > 0 else 0.0, axis=1)
    out["ad_acos_total"] = out.apply(lambda r: (r["ad_spend_total"] / r["ad_sales_total"]) if float(r["ad_sales_total"] or 0.0) > 0 else 0.0, axis=1)
    out["ad_sales_share_total"] = out.apply(lambda r: (r["ad_sales_total"] / r["sales_total"]) if float(r["sales_total"] or 0.0) > 0 else 0.0, axis=1)
    out["ad_orders_share_total"] = out.apply(lambda r: (r["ad_orders_total"] / r["orders_total"]) if float(r["orders_total"] or 0.0) > 0 else 0.0, axis=1)
    out["cvr_total"] = out.apply(lambda r: (r["orders_total"] / r["sessions_total"]) if float(r["sessions_total"] or 0.0) > 0 else 0.0, axis=1)
    out["aov_total"] = out.apply(lambda r: (r["sales_total"] / r["orders_total"]) if float(r["orders_total"] or 0.0) > 0 else 0.0, axis=1)

    # 生命周期阶段分布（字符串：phase:count;phase:count）
    try:
        if "current_phase" in merged.columns:
            def _phase_counts(s: pd.Series) -> str:
                vc = s.astype(str).fillna("").map(lambda x: "" if str(x).strip().lower() == "nan" else str(x).strip())
                vc = vc[vc != ""]
                if vc.empty:
                    return ""
                cnt = vc.value_counts().to_dict()
                # 按数量降序，最多 6 个，避免太长
                items = sorted(cnt.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
                items = items[:6]
                return ";".join([f"{k}:{v}" for k, v in items])

            pc = merged.groupby("product_category", dropna=False)["current_phase"].apply(_phase_counts)
            out = out.merge(pc.rename("phase_counts"), left_on="product_category", right_index=True, how="left")
            out["phase_counts"] = out["phase_counts"].fillna("")
        else:
            out["phase_counts"] = ""
    except Exception:
        out["phase_counts"] = ""

    # 排序：按 ad_spend_total（更贴近“广告要先看哪里”），再按 sales_total
    out = out.sort_values(["ad_spend_total", "sales_total"], ascending=[False, False]).copy()

    # 格式化：保留常用小数位
    for c in ("sales_total", "ad_spend_total", "ad_sales_total", "profit_total", "organic_sales_total"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(2)
    for c in (
        "tacos_total",
        "ad_acos_total",
        "ad_sales_share_total",
        "ad_orders_share_total",
        "organic_sales_share_total",
        "organic_orders_share_total",
        "cvr_total",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(4)
    if "aov_total" in out.columns:
        out["aov_total"] = pd.to_numeric(out["aov_total"], errors="coerce").fillna(0.0).round(4)

    return out.reset_index(drop=True)


def build_drivers_top_asins(
    scorecard: Dict[str, object],
    lifecycle_board: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    从 shop_scorecard 里提取“变化来源 drivers”，生成结构化表（用于运营筛选/排序）。

    数据来源：
    - scorecard["drivers"]["window_7d"]["by_delta_sales"/"by_delta_ad_spend"]

    输出：
    - driver_type：delta_sales / delta_ad_spend
    - rank：同一 driver_type 内的排序（按 diagnostics 已排序）
    - 其它字段：asin/product_name/current_phase/inventory/flag_* 以及 window 元信息等
    """
    try:
        sc = scorecard if isinstance(scorecard, dict) else {}
        drivers = sc.get("drivers") if isinstance(sc.get("drivers"), dict) else {}
        w7 = drivers.get("window_7d") if isinstance(drivers.get("window_7d"), dict) else {}
        rows_sales = w7.get("by_delta_sales") if isinstance(w7.get("by_delta_sales"), list) else []
        rows_spend = w7.get("by_delta_ad_spend") if isinstance(w7.get("by_delta_ad_spend"), list) else []

        def _to_df(rows: List[object], driver_type: str) -> pd.DataFrame:
            items = [r for r in rows if isinstance(r, dict)]
            df = pd.DataFrame(items)
            if df.empty:
                return df
            df.insert(0, "rank", list(range(1, int(len(df)) + 1)))
            df.insert(0, "driver_type", str(driver_type))
            return df

        d1 = _to_df(rows_sales, "delta_sales")
        d2 = _to_df(rows_spend, "delta_ad_spend")
        out = pd.concat([d1, d2], ignore_index=True, sort=False)
        if out.empty:
            return out

        # 兼容：ASIN 大写标准化
        if "asin" in out.columns:
            out["asin"] = out["asin"].astype(str).str.upper().str.strip()

        # 补充商品分类（便于对比同类产品）
        try:
            b = lifecycle_board.copy() if lifecycle_board is not None else pd.DataFrame()
            if b is not None and not b.empty and "asin" in b.columns and "product_category" in b.columns:
                meta = b[["asin", "product_category"]].copy()
                meta["asin"] = meta["asin"].astype(str).str.upper().str.strip()
                meta["product_category"] = (
                    meta["product_category"]
                    .astype(str)
                    .fillna("")
                    .map(lambda x: "" if str(x).strip().lower() == "nan" else str(x).strip())
                )
                meta.loc[meta["product_category"] == "", "product_category"] = "（未分类）"
                meta = meta.drop_duplicates("asin", keep="first")
                out = out.merge(meta, on="asin", how="left")
        except Exception:
            pass

        if "product_category" in out.columns:
            out["product_category"] = (
                out["product_category"]
                .astype(str)
                .fillna("")
                .map(lambda x: "" if str(x).strip().lower() == "nan" else str(x).strip())
            )
            out.loc[out["product_category"] == "", "product_category"] = "（未分类）"

        # 数值化（方便运营排序）
        for c in (
            "delta_sales",
            "delta_ad_spend",
            "marginal_tacos",
            "inventory",
            "flag_low_inventory",
            "flag_oos",
        ):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

        # 稳定排序：按 driver_type（delta_sales 优先）再按 rank
        dt_rank = {"delta_sales": 0, "delta_ad_spend": 1}
        out["_driver_rank"] = out.get("driver_type", "").astype(str).map(lambda x: dt_rank.get(str(x), 9))
        out["rank"] = pd.to_numeric(out.get("rank", 0), errors="coerce").fillna(0).astype(int)
        out = out.sort_values(["_driver_rank", "rank"], ascending=[True, True]).drop(columns=["_driver_rank"], errors="ignore")

        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def enrich_drivers_with_action_counts(
    drivers_top_asins: pd.DataFrame,
    action_board_dedup_all: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    给 drivers_top_asins 补充“该 ASIN 对应 Top 动作数/阻断数”。

    口径（Top 动作）：
    - 以 action_board_dedup_all（全量动作候选去重后的集合）为准
    - Top 指优先级为 P0/P1 的动作数量（更符合运营“先做什么”的直觉）
    """
    if drivers_top_asins is None or drivers_top_asins.empty:
        return drivers_top_asins

    out = drivers_top_asins.copy()
    if "asin" not in out.columns:
        out["top_action_count"] = 0
        out["top_blocked_action_count"] = 0
        return out

    # 默认补齐列（避免 merge 失败）
    if "top_action_count" not in out.columns:
        out["top_action_count"] = 0
    if "top_blocked_action_count" not in out.columns:
        out["top_blocked_action_count"] = 0

    ab = action_board_dedup_all.copy() if action_board_dedup_all is not None else pd.DataFrame()
    if ab is None or ab.empty or "asin_hint" not in ab.columns:
        return out

    try:
        ab2 = ab.copy()
        ab2["asin_hint"] = ab2["asin_hint"].astype(str).str.upper().str.strip()
        ab2 = ab2[ab2["asin_hint"] != ""].copy()
        if ab2.empty:
            return out

        if "priority" not in ab2.columns:
            ab2["priority"] = ""
        ab2["priority"] = ab2["priority"].astype(str).str.upper().str.strip()

        if "blocked" not in ab2.columns:
            ab2["blocked"] = 0
        ab2["blocked"] = pd.to_numeric(ab2["blocked"], errors="coerce").fillna(0).astype(int)

        ab2["_is_top"] = ab2["priority"].isin(["P0", "P1"]).astype(int)
        ab2["_is_top_blocked"] = ((ab2["_is_top"] == 1) & (ab2["blocked"] == 1)).astype(int)

        stats = (
            ab2.groupby("asin_hint", dropna=False, as_index=False)
            .agg(
                top_action_count=("_is_top", "sum"),
                top_blocked_action_count=("_is_top_blocked", "sum"),
            )
            .copy()
        )
        stats = stats.rename(columns={"asin_hint": "asin"})

        out["asin"] = out["asin"].astype(str).str.upper().str.strip()
        out = out.merge(stats, on="asin", how="left", suffixes=("", "_y"))

        # merge 后可能产生 _y 列，优先用统计值
        for c in ("top_action_count", "top_blocked_action_count"):
            if f"{c}_y" in out.columns:
                out[c] = pd.to_numeric(out[f"{c}_y"], errors="coerce").fillna(out[c]).fillna(0).astype(int)
                out = out.drop(columns=[f"{c}_y"], errors="ignore")
            else:
                out[c] = pd.to_numeric(out.get(c, 0), errors="coerce").fillna(0).astype(int)

        return out
    except Exception:
        out["top_action_count"] = pd.to_numeric(out.get("top_action_count", 0), errors="coerce").fillna(0).astype(int)
        out["top_blocked_action_count"] = (
            pd.to_numeric(out.get("top_blocked_action_count", 0), errors="coerce").fillna(0).astype(int)
        )
        return out


def build_asin_cockpit(
    asin_focus_all: Optional[pd.DataFrame],
    drivers_top_asins: Optional[pd.DataFrame],
    action_board_dedup_all: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    ASIN Cockpit：把“运营抓重点”的关键维度汇总到每个 ASIN 一行。

    汇总维度（当前版本）：
    - ASIN Focus：focus_score、生命周期、库存、滚动花费等（来自 asin_focus_all）
    - Drivers：近7天 vs 前7天的变化来源信息（来自 drivers_top_asins 的 Top 列表）
    - 动作量：P0/P1 动作数、阻断数等（来自 action_board_dedup_all）

    设计原则：
    - 不依赖新增数据源/唯一ID
    - 防御性：任何缺列/空表都不崩，输出尽量完整
    """
    base = asin_focus_all.copy() if asin_focus_all is not None else pd.DataFrame()
    if base is None:
        base = pd.DataFrame()

    # 基础行集：优先用 asin_focus_all；如果没有，则从动作里取 asin_hint 做兜底
    if base.empty or "asin" not in base.columns:
        base = pd.DataFrame(columns=["asin"])

    base = base.copy()
    if "asin" not in base.columns:
        base["asin"] = ""
    base["asin"] = base["asin"].astype(str).str.upper().str.strip()
    base = base[base["asin"] != ""].copy()

    # 动作统计（全量去重动作集合）
    ab = action_board_dedup_all.copy() if action_board_dedup_all is not None else pd.DataFrame()
    if ab is None:
        ab = pd.DataFrame()

    action_stats = pd.DataFrame()
    try:
        if not ab.empty and "asin_hint" in ab.columns:
            ab2 = ab.copy()
            ab2["asin_hint"] = ab2["asin_hint"].astype(str).str.upper().str.strip()
            ab2 = ab2[ab2["asin_hint"] != ""].copy()
            if "priority" not in ab2.columns:
                ab2["priority"] = ""
            ab2["priority"] = ab2["priority"].astype(str).str.upper().str.strip()
            if "blocked" not in ab2.columns:
                ab2["blocked"] = 0
            ab2["blocked"] = pd.to_numeric(ab2["blocked"], errors="coerce").fillna(0).astype(int)

            ab2["_p0"] = (ab2["priority"] == "P0").astype(int)
            ab2["_p1"] = (ab2["priority"] == "P1").astype(int)
            ab2["_p2"] = (ab2["priority"] == "P2").astype(int)
            ab2["_is_top"] = ab2["priority"].isin(["P0", "P1"]).astype(int)
            ab2["_is_top_blocked"] = ((ab2["_is_top"] == 1) & (ab2["blocked"] == 1)).astype(int)

            action_stats = (
                ab2.groupby("asin_hint", dropna=False, as_index=False)
                .agg(
                    total_action_count=("asin_hint", "size"),
                    p0_action_count=("_p0", "sum"),
                    p1_action_count=("_p1", "sum"),
                    p2_action_count=("_p2", "sum"),
                    top_action_count=("_is_top", "sum"),
                    top_blocked_action_count=("_is_top_blocked", "sum"),
                    blocked_action_count=("blocked", "sum"),
                )
                .copy()
            )
            action_stats = action_stats.rename(columns={"asin_hint": "asin"})
    except Exception:
        action_stats = pd.DataFrame()

    # drivers（Top 列表转成 per-asin 的宽表）
    drivers_metrics = pd.DataFrame()
    drivers_ranks = pd.DataFrame()
    try:
        d = drivers_top_asins.copy() if isinstance(drivers_top_asins, pd.DataFrame) else pd.DataFrame()
        if d is not None and not d.empty and "asin" in d.columns:
            d2 = d.copy()
            d2["asin"] = d2["asin"].astype(str).str.upper().str.strip()
            d2["driver_type"] = d2.get("driver_type", "").astype(str)
            d2["rank"] = pd.to_numeric(d2.get("rank", 0), errors="coerce").fillna(0).astype(int)

            # ranks（delta_sales / delta_ad_spend）
            r1 = d2[d2["driver_type"] == "delta_sales"][["asin", "rank"]].copy()
            r1 = r1.rename(columns={"rank": "drivers_rank_delta_sales"})
            r2 = d2[d2["driver_type"] == "delta_ad_spend"][["asin", "rank"]].copy()
            r2 = r2.rename(columns={"rank": "drivers_rank_delta_ad_spend"})
            drivers_ranks = r1.merge(r2, on="asin", how="outer")

            # metrics：优先取 delta_sales 列表（同一 ASIN 的数值一致，仅排序不同）
            pref = {"delta_sales": 0, "delta_ad_spend": 1}
            d2["_pref"] = d2["driver_type"].map(lambda x: pref.get(str(x), 9))
            d2 = d2.sort_values(["_pref", "rank"], ascending=[True, True])
            pick_cols = [
                "asin",
                "window_days",
                "recent_start",
                "recent_end",
                "prev_start",
                "prev_end",
                "delta_sales",
                "delta_ad_spend",
                "marginal_tacos",
            ]
            pick_cols = [c for c in pick_cols if c in d2.columns]
            drivers_metrics = d2[pick_cols].drop_duplicates("asin", keep="first").copy()
            drivers_metrics = drivers_metrics.rename(
                columns={
                    "window_days": "drivers_window_days",
                    "recent_start": "drivers_recent_start",
                    "recent_end": "drivers_recent_end",
                    "prev_start": "drivers_prev_start",
                    "prev_end": "drivers_prev_end",
                    "delta_sales": "drivers_delta_sales",
                    "delta_ad_spend": "drivers_delta_ad_spend",
                    "marginal_tacos": "drivers_marginal_tacos",
                }
            )
            # 数值化
            for c in ("drivers_delta_sales", "drivers_delta_ad_spend", "drivers_marginal_tacos"):
                if c in drivers_metrics.columns:
                    drivers_metrics[c] = pd.to_numeric(drivers_metrics[c], errors="coerce").fillna(0.0)
    except Exception:
        drivers_metrics = pd.DataFrame()
        drivers_ranks = pd.DataFrame()

    # union：把动作里出现但 asin_focus_all 不存在的 asin 也纳入 cockpit（兜底）
    try:
        if not action_stats.empty and "asin" in action_stats.columns:
            base_asins = set(base["asin"].tolist()) if not base.empty and "asin" in base.columns else set()
            extra_asins = [a for a in action_stats["asin"].astype(str).str.upper().str.strip().tolist() if a and a not in base_asins]
            if extra_asins:
                extra = pd.DataFrame({"asin": extra_asins})
                extra["shop"] = ""
                extra["product_name"] = ""
                extra["product_category"] = "（未分类）"
                extra["current_phase"] = ""
                extra["cycle_id"] = ""
                extra["inventory"] = 0.0
                extra["focus_score"] = 0.0
                extra["focus_reasons"] = ""
                base = pd.concat([base, extra], ignore_index=True, sort=False)
    except Exception as _e:
        raise

    out = base.copy()
    if not action_stats.empty:
        out = out.merge(action_stats, on="asin", how="left")
    if not drivers_metrics.empty:
        out = out.merge(drivers_metrics, on="asin", how="left")
    if not drivers_ranks.empty:
        out = out.merge(drivers_ranks, on="asin", how="left")

    # 兜底列与数值化
    if "product_category" in out.columns:
        out["product_category"] = out["product_category"].map(_norm_product_category)
    else:
        out["product_category"] = "（未分类）"

    for c in (
        "focus_score",
        "ad_spend_roll",
        "tacos_roll",
        "inventory",
        "total_action_count",
        "p0_action_count",
        "p1_action_count",
        "p2_action_count",
        "top_action_count",
        "top_blocked_action_count",
        "blocked_action_count",
        "drivers_rank_delta_sales",
        "drivers_rank_delta_ad_spend",
        "drivers_window_days",
        "drivers_delta_sales",
        "drivers_delta_ad_spend",
        "drivers_marginal_tacos",
    ):
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # 排序：先 focus_score，再 Top 动作量，再 ad_spend_roll（更贴近运营抓重点）
    out = out.sort_values(
        ["focus_score", "top_action_count", "ad_spend_roll"],
        ascending=[False, False, False],
    ).copy()

    return out.reset_index(drop=True)


def build_category_cockpit(
    category_summary: Optional[pd.DataFrame],
    asin_cockpit: Optional[pd.DataFrame],
    action_board_dedup_all: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Category Cockpit：把“运营抓重点”的关键维度汇总到每个类目一行。

    汇总维度（当前版本）：
    - 类目汇总：sales/orders/sessions/ad_spend/TACOS/广告依赖等（来自 category_summary）
    - 变化来源：ΔSales/ΔAdSpend（来自 asin_cockpit 的 drivers_* 汇总）
    - 动作量：P0/P1 动作数、阻断数等（来自 action_board_dedup_all）

    设计原则：
    - 不引入新数据源/唯一ID
    - 防御性：任何缺列/空表都不崩，输出尽量完整
    """
    # 1) base：优先用 category_summary（它包含销量/广告/生命周期分布）
    base = category_summary.copy() if isinstance(category_summary, pd.DataFrame) else pd.DataFrame()
    if base is None or base.empty or "product_category" not in base.columns:
        base = pd.DataFrame(columns=["product_category"])
    else:
        base = base.copy()

    base["product_category"] = base.get("product_category", "").map(_norm_product_category)
    base = base.drop_duplicates("product_category", keep="first").copy()

    # 2) 动作统计（按类目）
    action_stats = pd.DataFrame()
    try:
        ab = action_board_dedup_all.copy() if isinstance(action_board_dedup_all, pd.DataFrame) else pd.DataFrame()
        if ab is not None and not ab.empty:
            ab2 = ab.copy()
            ab2["product_category"] = ab2.get("product_category", "").map(_norm_product_category)
            if "priority" not in ab2.columns:
                ab2["priority"] = ""
            ab2["priority"] = ab2["priority"].astype(str).str.upper().str.strip()
            if "blocked" not in ab2.columns:
                ab2["blocked"] = 0
            ab2["blocked"] = pd.to_numeric(ab2["blocked"], errors="coerce").fillna(0).astype(int)

            ab2["_p0"] = (ab2["priority"] == "P0").astype(int)
            ab2["_p1"] = (ab2["priority"] == "P1").astype(int)
            ab2["_p2"] = (ab2["priority"] == "P2").astype(int)
            ab2["_is_top"] = ab2["priority"].isin(["P0", "P1"]).astype(int)
            ab2["_is_top_blocked"] = ((ab2["_is_top"] == 1) & (ab2["blocked"] == 1)).astype(int)

            action_stats = (
                ab2.groupby("product_category", dropna=False, as_index=False)
                .agg(
                    category_action_count=("product_category", "size"),
                    category_p0_action_count=("_p0", "sum"),
                    category_p1_action_count=("_p1", "sum"),
                    category_p2_action_count=("_p2", "sum"),
                    category_top_action_count=("_is_top", "sum"),
                    category_top_blocked_action_count=("_is_top_blocked", "sum"),
                    category_blocked_action_count=("blocked", "sum"),
                )
                .copy()
            )
    except Exception:
        action_stats = pd.DataFrame()

    # 3) drivers（按类目汇总：用 asin_cockpit 的 drivers_*，避免再次算 window）
    drivers_stats = pd.DataFrame()
    try:
        ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
        if ac is not None and not ac.empty:
            ac2 = ac.copy()
            ac2["product_category"] = ac2.get("product_category", "").map(_norm_product_category)

            # 驱动可用性：drivers_window_days>0（无数据时我们会填 0）
            if "drivers_window_days" in ac2.columns:
                w = pd.to_numeric(ac2["drivers_window_days"], errors="coerce").fillna(0).astype(int)
            else:
                w = pd.Series([0] * int(len(ac2)), index=ac2.index)
            ac2["_has_drivers"] = (w > 0).astype(int)

            for c in ("drivers_delta_sales", "drivers_delta_ad_spend"):
                if c not in ac2.columns:
                    ac2[c] = 0.0
                ac2[c] = pd.to_numeric(ac2[c], errors="coerce").fillna(0.0)

            # 只在 has_drivers=1 的 ASIN 上汇总（避免大量 0 混入导致误读）
            mask = ac2["_has_drivers"] == 1
            ac3 = ac2[mask].copy()

            if ac3.empty:
                drivers_stats = pd.DataFrame(
                    columns=[
                        "product_category",
                        "drivers_asin_count",
                        "drivers_delta_sales_sum",
                        "drivers_delta_ad_spend_sum",
                        "drivers_sales_up_asin_count",
                        "drivers_sales_down_asin_count",
                    ]
                )
            else:
                drivers_stats = (
                    ac3.groupby("product_category", dropna=False, as_index=False)
                    .agg(
                        drivers_asin_count=("asin", "count") if "asin" in ac3.columns else ("product_category", "size"),
                        drivers_delta_sales_sum=("drivers_delta_sales", "sum"),
                        drivers_delta_ad_spend_sum=("drivers_delta_ad_spend", "sum"),
                    )
                    .copy()
                )
                # up/down 计数
                tmp = ac3.copy()
                tmp["_sales_up"] = (tmp["drivers_delta_sales"] > 0).astype(int)
                tmp["_sales_down"] = (tmp["drivers_delta_sales"] < 0).astype(int)
                ud = (
                    tmp.groupby("product_category", dropna=False, as_index=False)
                    .agg(
                        drivers_sales_up_asin_count=("_sales_up", "sum"),
                        drivers_sales_down_asin_count=("_sales_down", "sum"),
                    )
                    .copy()
                )
                drivers_stats = drivers_stats.merge(ud, on="product_category", how="left")
    except Exception:
        drivers_stats = pd.DataFrame()

    # 4) union：把在 action/drivers 出现但 base 缺失的类目也纳入（兜底）
    try:
        cats = set(base["product_category"].tolist()) if not base.empty else set()
        if not action_stats.empty and "product_category" in action_stats.columns:
            cats |= set(action_stats["product_category"].tolist())
        if not drivers_stats.empty and "product_category" in drivers_stats.columns:
            cats |= set(drivers_stats["product_category"].tolist())
        cats = {str(c or "").strip() for c in cats}
        cats = {c for c in cats if c}
        if cats and (base.empty or base["product_category"].nunique() < len(cats)):
            base2 = pd.DataFrame({"product_category": sorted(list(cats))})
            out = base2.merge(base, on="product_category", how="left")
        else:
            out = base.copy()
    except Exception:
        out = base.copy()

    if not action_stats.empty:
        out = out.merge(action_stats, on="product_category", how="left")
    if not drivers_stats.empty:
        out = out.merge(drivers_stats, on="product_category", how="left")

    # 兜底数值列
    for c in (
        "asin_count",
        "sales_total",
        "orders_total",
        "sessions_total",
        "ad_spend_total",
        "ad_sales_total",
        "ad_orders_total",
        "profit_total",
        "tacos_total",
        "ad_acos_total",
        "ad_sales_share_total",
        "ad_orders_share_total",
        "focus_score_sum",
        "focus_score_mean",
        "focus_asin_count",
        "low_inventory_asin_count",
        "oos_asin_count",
        "oos_with_ad_spend_asin_count",
        "spend_up_no_sales_asin_count",
        "high_ad_dependency_asin_count",
        "category_action_count",
        "category_p0_action_count",
        "category_p1_action_count",
        "category_p2_action_count",
        "category_top_action_count",
        "category_top_blocked_action_count",
        "category_blocked_action_count",
        "drivers_asin_count",
        "drivers_delta_sales_sum",
        "drivers_delta_ad_spend_sum",
        "drivers_sales_up_asin_count",
        "drivers_sales_down_asin_count",
    ):
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # 格式化（保持 cockpit 可读性）
    for c in (
        "sales_total",
        "ad_spend_total",
        "ad_sales_total",
        "profit_total",
        "focus_score_sum",
        "focus_score_mean",
        "drivers_delta_sales_sum",
        "drivers_delta_ad_spend_sum",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(2)
    for c in ("tacos_total", "ad_acos_total", "ad_sales_share_total", "ad_orders_share_total"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(4)
    for c in (
        "asin_count",
        "focus_asin_count",
        "low_inventory_asin_count",
        "oos_asin_count",
        "oos_with_ad_spend_asin_count",
        "spend_up_no_sales_asin_count",
        "high_ad_dependency_asin_count",
        "category_action_count",
        "category_p0_action_count",
        "category_p1_action_count",
        "category_p2_action_count",
        "category_top_action_count",
        "category_top_blocked_action_count",
        "category_blocked_action_count",
        "drivers_asin_count",
        "drivers_sales_up_asin_count",
        "drivers_sales_down_asin_count",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # 排序：先 focus_score_sum，再 Top 动作量，再 ad_spend_total（更贴近运营抓重点）
    sort_cols = [c for c in ["focus_score_sum", "category_top_action_count", "ad_spend_total"] if c in out.columns]
    if sort_cols:
        asc = [False] * int(len(sort_cols))
        out = out.sort_values(sort_cols, ascending=asc).copy()

    return out.reset_index(drop=True)


def build_category_asin_compare(
    asin_cockpit: Optional[pd.DataFrame],
    category_cockpit: Optional[pd.DataFrame],
    max_categories: int = 50,
    asins_per_category: int = 30,
) -> pd.DataFrame:
    """
    类目→产品对比表（category_asin_compare.csv）。

    设计目标：
    - 运营按“商品分类→产品(ASIN)”做同类对比时，能在一个表里同时看到：
      速度（7/14/30）、库存覆盖（7/14/30）、利润承受度摘要、风险信号、动作量、变化来源。
    - 这是 CSV（可筛选/可排序），避免在 Markdown 里信息爆炸。
    """
    ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
    if ac is None or ac.empty or "asin" not in ac.columns:
        return pd.DataFrame()

    out = ac.copy()
    out["asin"] = out["asin"].astype(str).str.upper().str.strip()
    out = out[out["asin"] != ""].copy()
    if out.empty:
        return pd.DataFrame()

    out["product_category"] = out.get("product_category", "").map(_norm_product_category)

    # 类目 rank（优先用 category_cockpit 的排序；否则用 asin_cockpit 自己汇总兜底）
    cat_rank = pd.DataFrame()
    try:
        cc = category_cockpit.copy() if isinstance(category_cockpit, pd.DataFrame) else pd.DataFrame()
        if cc is not None and not cc.empty and "product_category" in cc.columns:
            cc = cc.copy()
            cc["product_category"] = cc["product_category"].map(_norm_product_category)
            for c in ("focus_score_sum", "category_top_action_count", "ad_spend_total"):
                if c not in cc.columns:
                    cc[c] = 0.0
                cc[c] = pd.to_numeric(cc[c], errors="coerce").fillna(0.0)
            cc = cc.sort_values(["focus_score_sum", "category_top_action_count", "ad_spend_total"], ascending=[False, False, False]).copy()
            cc["category_rank"] = list(range(1, int(len(cc)) + 1))
            cat_rank = cc[["product_category", "category_rank"]].drop_duplicates("product_category", keep="first").copy()
    except Exception:
        cat_rank = pd.DataFrame()

    if cat_rank is None or cat_rank.empty:
        try:
            tmp = out.copy()
            if "focus_score" not in tmp.columns:
                tmp["focus_score"] = 0.0
            if "ad_spend_roll" not in tmp.columns:
                tmp["ad_spend_roll"] = 0.0
            tmp["focus_score"] = pd.to_numeric(tmp.get("focus_score", 0.0), errors="coerce").fillna(0.0)
            tmp["ad_spend_roll"] = pd.to_numeric(tmp.get("ad_spend_roll", 0.0), errors="coerce").fillna(0.0)
            g = (
                tmp.groupby("product_category", dropna=False, as_index=False)
                .agg(
                    focus_score_sum=("focus_score", "sum"),
                    ad_spend_roll_sum=("ad_spend_roll", "sum"),
                    asin_count=("asin", "count"),
                )
                .copy()
            )
            g = g.sort_values(["focus_score_sum", "ad_spend_roll_sum", "asin_count"], ascending=[False, False, False]).copy()
            g["category_rank"] = list(range(1, int(len(g)) + 1))
            cat_rank = g[["product_category", "category_rank"]].copy()
        except Exception:
            cat_rank = pd.DataFrame()

    if cat_rank is not None and not cat_rank.empty:
        out = out.merge(cat_rank, on="product_category", how="left")
    if "category_rank" not in out.columns:
        out["category_rank"] = 999
    out["category_rank"] = pd.to_numeric(out.get("category_rank", 999), errors="coerce").fillna(999).astype(int)

    # 数值化（用于排序/稳定输出）
    for c in (
        "focus_score",
        "top_action_count",
        "top_blocked_action_count",
        "ad_spend_roll",
        "drivers_delta_sales",
        "drivers_delta_ad_spend",
        "drivers_marginal_tacos",
        "sales_per_day_7d",
        "sales_per_day_14d",
        "sales_per_day_30d",
        "orders_per_day_7d",
        "orders_per_day_14d",
        "orders_per_day_30d",
        "aov_recent_7d",
        "delta_aov_7d",
        "inventory_cover_days_7d",
        "inventory_cover_days_14d",
        "inventory_cover_days_30d",
        "gross_margin",
        "profit_before_ads",
        "profit_after_ads",
        "max_ad_spend_by_profit",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # 排序：类目 rank → focus_score → top_action_count → ad_spend_roll
    out = out.sort_values(
        ["category_rank", "focus_score", "top_action_count", "ad_spend_roll"],
        ascending=[True, False, False, False],
    ).copy()

    # 类内 rank（同类对比）
    try:
        out["asin_rank_in_category"] = out.groupby("product_category", dropna=False).cumcount() + 1
    except Exception:
        out["asin_rank_in_category"] = 0

    # 控制输出规模：Top 类目 + 每类 Top K
    try:
        mc = max(1, int(max_categories or 0))
        k = max(1, int(asins_per_category or 0))
        out2 = out[out["category_rank"] <= mc].copy()
        out2 = out2[out2["asin_rank_in_category"] <= k].copy()
        out = out2
    except Exception as _e:
        raise

    cols = [
        "category_rank",
        "product_category",
        "asin_rank_in_category",
        "asin",
        "product_name",
        "current_phase",
        "phase_change",
        "phase_changed_recent_14d",
        "phase_trend_14d",
        "cycle_id",
        "inventory",
        # 速度/覆盖（7/14/30）
        "sales_per_day_7d",
        "sales_per_day_14d",
        "sales_per_day_30d",
        "orders_per_day_7d",
        "orders_per_day_14d",
        "orders_per_day_30d",
        # 客单价（7d）与变化（便于同类对比“价格/变体/促销”语境）
        "aov_recent_7d",
        "delta_aov_7d",
        "inventory_cover_days_7d",
        "inventory_cover_days_14d",
        "inventory_cover_days_30d",
        # 广告依赖/效率（产品侧）
        "ad_sales_share",
        "tacos_roll",
        # 毛利率（主窗口口径）：用于同类对比“利润空间”语境
        "gross_margin",
        # 利润承受度摘要（利润视角）
        "profit_stage",
        "profit_direction",
        "profit_before_ads",
        "profit_after_ads",
        "max_ad_spend_by_profit",
        "target_tacos_by_margin",
        # 抓重点
        "focus_score",
        "focus_reasons",
        # 动作/变化来源
        "top_action_count",
        "top_blocked_action_count",
        "drivers_delta_sales",
        "drivers_delta_ad_spend",
        "drivers_marginal_tacos",
        "ad_spend_roll",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


def _pick_phase_down_reasons(row: Dict[str, object], policy: Optional[OpsPolicy] = None) -> List[str]:
    """
    “阶段走弱且仍在花费”的可解释原因标签（最多 3 个）。

    设计原则：
    - 标签要可解释、稳定（尽量少、优先覆盖大多数场景）
    - 不引入新数据源：只依赖 asin_cockpit 现有字段
    - 不是结论：只是“优先排查方向”
    """
    reasons: List[str] = []

    try:
        oos_with_spend_days = int(float(row.get("oos_with_ad_spend_days", 0) or 0))
    except Exception:
        oos_with_spend_days = 0
    try:
        flag_oos = int(float(row.get("flag_oos", 0) or 0))
    except Exception:
        flag_oos = 0
    try:
        ad_spend_roll = float(row.get("ad_spend_roll", 0.0) or 0.0)
    except Exception:
        ad_spend_roll = 0.0

    # 1) 断货风险（最优先；只做“方向提示”，不判定根因）
    if (oos_with_spend_days > 0) or (flag_oos > 0 and ad_spend_roll > 0):
        reasons.append("断货风险")

    # 2) 库存风险（覆盖不足/低库存）
    cover_below = 0.0
    try:
        cover_below = float(getattr(policy, "block_scale_when_cover_days_below", 0.0) or 0.0) if policy is not None else 0.0
    except Exception:
        cover_below = 0.0
    try:
        cover_7d = float(row.get("inventory_cover_days_7d", 0.0) or 0.0)
    except Exception:
        cover_7d = 0.0
    try:
        flag_low_inv = int(float(row.get("flag_low_inventory", 0) or 0))
    except Exception:
        flag_low_inv = 0

    # 说明：cover_7d=0 可能是“销量很低导致除数为0”，所以这里用阈值判断时要求 cover_7d>0
    if (cover_below > 0 and cover_7d > 0 and cover_7d < cover_below) or (flag_low_inv > 0):
        reasons.append("库存风险")

    # 3.2) 转化下滑（Sessions↑但 CVR↓：更像“产品语境”问题，而不只是投放效率）
    # 说明：只在样本量足够时触发，避免小样本噪声；阈值复用 focus_scoring 的配置入口。
    try:
        fs = getattr(policy, "dashboard_focus_scoring", None) if policy is not None else None
        min_prev_sess = float(getattr(fs, "cvr_signal_min_sessions_prev", 100.0) or 100.0) if fs is not None else 100.0
        min_delta_sess = float(getattr(fs, "cvr_signal_min_delta_sessions", 50.0) or 50.0) if fs is not None else 50.0
        min_cvr_drop = float(getattr(fs, "cvr_signal_min_cvr_drop", 0.02) or 0.02) if fs is not None else 0.02
    except Exception:
        min_prev_sess, min_delta_sess, min_cvr_drop = 100.0, 50.0, 0.02

    try:
        sess_prev_7d = float(row.get("sessions_prev_7d", 0.0) or 0.0)
    except Exception:
        sess_prev_7d = 0.0
    try:
        delta_sessions = float(row.get("delta_sessions", 0.0) or 0.0)
    except Exception:
        delta_sessions = 0.0
    try:
        delta_cvr = float(row.get("delta_cvr", 0.0) or 0.0)
    except Exception:
        delta_cvr = 0.0

    if (sess_prev_7d >= min_prev_sess) and (delta_sessions >= min_delta_sess) and (delta_cvr <= -abs(min_cvr_drop)):
        reasons.append("转化下滑")

    # 3.3) 自然端回落（7d vs prev7d）：优先排查 Listing/价格/评价/变体/促销等“产品语境”
    try:
        fs = getattr(policy, "dashboard_focus_scoring", None) if policy is not None else None
        organic_min_prev = float(getattr(fs, "organic_signal_min_organic_sales_prev", 100.0) or 100.0) if fs is not None else 100.0
        organic_drop_ratio = float(getattr(fs, "organic_signal_drop_ratio", 0.8) or 0.8) if fs is not None else 0.8
        organic_min_drop = float(getattr(fs, "organic_signal_min_delta_organic_sales", 20.0) or 20.0) if fs is not None else 20.0
    except Exception:
        organic_min_prev, organic_drop_ratio, organic_min_drop = 100.0, 0.8, 20.0

    try:
        organic_prev_7d = float(row.get("organic_sales_prev_7d", 0.0) or 0.0)
    except Exception:
        organic_prev_7d = 0.0
    try:
        organic_recent_7d = float(row.get("organic_sales_recent_7d", 0.0) or 0.0)
    except Exception:
        organic_recent_7d = 0.0
    try:
        delta_organic_sales = float(row.get("delta_organic_sales", 0.0) or 0.0)
    except Exception:
        delta_organic_sales = 0.0

    ratio = (organic_recent_7d / organic_prev_7d) if organic_prev_7d > 0 else 1.0
    if (organic_prev_7d >= organic_min_prev) and (delta_organic_sales <= -abs(organic_min_drop)) and (ratio <= organic_drop_ratio):
        reasons.append("自然端回落")

    # 3.3.1) 客单价下滑（AOV：sales/orders）
    # 说明：
    # - 用于提示优先排查价格/变体结构/促销/捆绑等“客单相关”的产品语境；
    # - 默认保守：要求上一窗口订单量足够，且 AOV 下降达到比例+绝对值双阈值，避免小样本噪声。
    try:
        fs = getattr(policy, "dashboard_focus_scoring", None) if policy is not None else None
        aov_min_orders_prev = float(getattr(fs, "aov_signal_min_orders_prev", 10.0) or 10.0) if fs is not None else 10.0
        aov_drop_ratio = float(getattr(fs, "aov_signal_drop_ratio", 0.9) or 0.9) if fs is not None else 0.9
        aov_min_drop_usd = float(getattr(fs, "aov_signal_min_delta_aov", 2.0) or 2.0) if fs is not None else 2.0
        aov_min_orders_prev = max(0.0, aov_min_orders_prev)
        aov_drop_ratio = max(0.0, min(1.0, aov_drop_ratio))
        aov_min_drop_usd = max(0.0, aov_min_drop_usd)
    except Exception:
        aov_min_orders_prev, aov_drop_ratio, aov_min_drop_usd = 10.0, 0.9, 2.0

    try:
        # 优先使用已派生列；若缺失则用 7d recent + delta 推导
        aov_prev = float(row.get("aov_prev_7d", 0.0) or 0.0)
        aov_recent = float(row.get("aov_recent_7d", 0.0) or 0.0)

        # 样本量约束：用 prev 订单量（推导/字段存在则优先用推导值）
        orders_prev = 0.0
        try:
            if ("orders_recent_7d" in row) and ("delta_orders" in row):
                orders_prev = float(row.get("orders_recent_7d", 0.0) or 0.0) - float(row.get("delta_orders", 0.0) or 0.0)
        except Exception:
            orders_prev = 0.0
        orders_prev = max(0.0, orders_prev)

        # 若 aov_* 未命中（可能是旧产物/测试数据），尝试用 sales/orders 现算
        if aov_prev <= 0 and aov_recent <= 0:
            try:
                sales_recent = float(row.get("sales_recent_7d", 0.0) or 0.0)
                orders_recent = float(row.get("orders_recent_7d", 0.0) or 0.0)
                ds = float(row.get("delta_sales", 0.0) or 0.0)
                do = float(row.get("delta_orders", 0.0) or 0.0)
                sales_prev = max(0.0, sales_recent - ds)
                orders_prev = max(0.0, orders_recent - do)
                aov_prev = (sales_prev / orders_prev) if orders_prev > 0 else 0.0
                aov_recent = (sales_recent / orders_recent) if orders_recent > 0 else 0.0
            except Exception:
                pass

        delta_aov = aov_recent - aov_prev
        ratio_aov = (aov_recent / aov_prev) if aov_prev > 0 else 1.0
        if (orders_prev >= aov_min_orders_prev) and (delta_aov <= -abs(aov_min_drop_usd)) and (ratio_aov <= aov_drop_ratio):
            reasons.append("客单价下滑")
    except Exception:
        import traceback
        traceback.print_exc()

    # 3.3.2) 毛利率偏低（gross_margin=profit/sales）
    # 说明：用于提示优先排查“提价/降成本/控量”，而不是只调广告结构。
    try:
        fs = getattr(policy, "dashboard_focus_scoring", None) if policy is not None else None
        gm_min_sales = float(getattr(fs, "gross_margin_signal_min_sales", 200.0) or 200.0) if fs is not None else 200.0
        gm_low_thr = float(getattr(fs, "gross_margin_signal_low_threshold", 0.15) or 0.15) if fs is not None else 0.15
        gm_min_sales = max(0.0, gm_min_sales)
        gm_low_thr = max(-1.0, min(1.0, gm_low_thr))
    except Exception:
        gm_min_sales, gm_low_thr = 200.0, 0.15

    try:
        gm = None
        if "gross_margin" in row:
            gm = float(row.get("gross_margin", 0.0) or 0.0)
        elif "profit_gross_margin" in row:
            gm = float(row.get("profit_gross_margin", 0.0) or 0.0)

        # 用 sales 做样本量约束（主窗口口径）；如缺失则回退 recent_7d
        sales_anchor = float(row.get("sales", 0.0) or 0.0)
        if sales_anchor <= 0 and "sales_recent_7d" in row:
            sales_anchor = float(row.get("sales_recent_7d", 0.0) or 0.0)

        if (gm is not None) and (sales_anchor >= gm_min_sales) and (gm <= gm_low_thr):
            reasons.append("毛利率偏低")
    except Exception:
        pass

    # 3.4) 加花费但销量不增（优先排查结构/词/Listing/价格）
    # 说明：放在“产品语境信号”之后，避免它过于常见导致挤掉更有解释力的原因。
    try:
        delta_spend = float(row.get("delta_spend", 0.0) or 0.0)
    except Exception:
        delta_spend = 0.0
    try:
        delta_sales = float(row.get("delta_sales", 0.0) or 0.0)
    except Exception:
        delta_sales = 0.0
    if delta_spend > 0 and delta_sales <= 0:
        reasons.append("加花费无增量")

    # 3.5) 速度下降（7d 明显低于 30d：阶段走弱时常见，提示优先排查需求/Listing/价格等）
    try:
        spd7 = float(row.get("sales_per_day_7d", 0.0) or 0.0)
    except Exception:
        spd7 = 0.0
    try:
        spd30 = float(row.get("sales_per_day_30d", 0.0) or 0.0)
    except Exception:
        spd30 = 0.0
    if spd30 > 0 and (spd7 / spd30) < 0.8:
        reasons.append("速度下降")

    # 4) 利润方向（优先把投放动作放回“产品语境”）
    try:
        pdirection = str(row.get("profit_direction", "") or "").strip().lower()
    except Exception:
        pdirection = ""
    if pdirection == "reduce":
        reasons.append("利润方向=控量")

    # 5) 广告依赖高（阶段走弱时更要关注“自然端”）
    ad_dep_thr = 0.0
    try:
        fs = getattr(policy, "dashboard_focus_scoring", None) if policy is not None else None
        ad_dep_thr = float(getattr(fs, "high_ad_dependency_threshold", 0.8) or 0.8) if fs is not None else 0.8
    except Exception:
        ad_dep_thr = 0.8
    try:
        ad_share = float(row.get("ad_sales_share", 0.0) or 0.0)
    except Exception:
        ad_share = 0.0
    if ad_dep_thr > 0 and ad_share >= ad_dep_thr:
        reasons.append("广告依赖高")

    # 6) 动作被阻断（说明当前“能做的事”可能受限：库存/低置信等）
    try:
        blocked_top = int(float(row.get("top_blocked_action_count", 0) or 0))
    except Exception:
        blocked_top = 0
    if blocked_top >= 3:
        reasons.append("动作阻断多")

    # 兜底：不让 reason 全空（对你后续维度补齐很重要）
    if not reasons:
        reasons.append("原因待确认")

    return reasons[:3]


def _annotate_phase_down_reasons(df: pd.DataFrame, policy: Optional[OpsPolicy] = None) -> pd.DataFrame:
    """
    给 DataFrame 增加 reason_1/2/3（防御性：失败不崩）。
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    try:
        tags = out.apply(lambda r: _pick_phase_down_reasons(r.to_dict(), policy=policy), axis=1)
        out["reason_1"] = tags.map(lambda xs: xs[0] if isinstance(xs, list) and len(xs) >= 1 else "")
        out["reason_2"] = tags.map(lambda xs: xs[1] if isinstance(xs, list) and len(xs) >= 2 else "")
        out["reason_3"] = tags.map(lambda xs: xs[2] if isinstance(xs, list) and len(xs) >= 3 else "")
    except Exception:
        # 保底补齐列，避免下游列选择报错
        for c in ("reason_1", "reason_2", "reason_3"):
            if c not in out.columns:
                out[c] = ""
    return out


def _pick_profit_reduce_reasons(row: Dict[str, object], policy: Optional[OpsPolicy] = None) -> List[str]:
    """
    利润控量（profit_direction=reduce）Watchlist 的原因标签（最多 3 个）。

    设计目标：
    - 让运营快速知道“为什么控量、先查哪里”，避免只看到一堆数
    - 标签必须可解释、尽量少（稳定集合）
    """
    reasons: List[str] = []

    # 1) 超利润上限（最贴合该 Watchlist）
    try:
        spend_roll = float(row.get("ad_spend_roll", 0.0) or 0.0)
    except Exception:
        spend_roll = 0.0
    try:
        cap = float(row.get("max_ad_spend_by_profit", 0.0) or 0.0)
    except Exception:
        cap = 0.0
    try:
        over = float(row.get("over_profit_cap", 0.0) or 0.0)
    except Exception:
        over = 0.0
    if over <= 0 and spend_roll > 0 and cap > 0:
        over = max(0.0, spend_roll - cap)
    if over > 0:
        reasons.append("超利润上限")

    # 2) 断货风险 / 库存风险（控量时优先排查供给侧）
    try:
        oos_with_spend_days = int(float(row.get("oos_with_ad_spend_days", 0) or 0))
    except Exception:
        oos_with_spend_days = 0
    try:
        flag_oos = int(float(row.get("flag_oos", 0) or 0))
    except Exception:
        flag_oos = 0
    if oos_with_spend_days > 0:
        reasons.append(f"近期断货风险({oos_with_spend_days}天)")
    elif (flag_oos > 0 and spend_roll > 0):
        reasons.append("当前断货风险")

    cover_below = 0.0
    try:
        cover_below = float(getattr(policy, "block_scale_when_cover_days_below", 0.0) or 0.0) if policy is not None else 0.0
    except Exception:
        cover_below = 0.0
    try:
        cover_7d = float(row.get("inventory_cover_days_7d", 0.0) or 0.0)
    except Exception:
        cover_7d = 0.0
    try:
        flag_low_inv = int(float(row.get("flag_low_inventory", 0) or 0))
    except Exception:
        flag_low_inv = 0
    if (cover_below > 0 and cover_7d > 0 and cover_7d < cover_below) or (flag_low_inv > 0):
        reasons.append("库存风险")

    # 3) 加花费无增量 / 速度下降（把“控量”放回增长语境）
    try:
        delta_spend = float(row.get("delta_spend", 0.0) or 0.0)
    except Exception:
        delta_spend = 0.0
    try:
        delta_sales = float(row.get("delta_sales", 0.0) or 0.0)
    except Exception:
        delta_sales = 0.0
    if delta_spend > 0 and delta_sales <= 0:
        reasons.append("加花费无增量")

    try:
        spd7 = float(row.get("sales_per_day_7d", 0.0) or 0.0)
    except Exception:
        spd7 = 0.0
    try:
        spd30 = float(row.get("sales_per_day_30d", 0.0) or 0.0)
    except Exception:
        spd30 = 0.0
    if spd30 > 0 and (spd7 / spd30) < 0.8:
        reasons.append("速度下降")

    # 4) 广告依赖高（控量阶段更需要关注自然端）
    ad_dep_thr = 0.0
    try:
        fs = getattr(policy, "dashboard_focus_scoring", None) if policy is not None else None
        ad_dep_thr = float(getattr(fs, "high_ad_dependency_threshold", 0.8) or 0.8) if fs is not None else 0.8
    except Exception:
        ad_dep_thr = 0.8
    try:
        ad_share = float(row.get("ad_sales_share", 0.0) or 0.0)
    except Exception:
        ad_share = 0.0
    if ad_dep_thr > 0 and ad_share >= ad_dep_thr:
        reasons.append("广告依赖高")

    # 5) 动作阻断多（提示先处理阻断原因）
    try:
        blocked_top = int(float(row.get("top_blocked_action_count", 0) or 0))
    except Exception:
        blocked_top = 0
    if blocked_top >= 3:
        reasons.append("动作阻断多")

    # 兜底：不让 reason 全空（对你后续维度补齐很重要）
    if not reasons:
        reasons.append("利润方向=控量")
    return reasons[:3]


def _annotate_profit_reduce_reasons(df: pd.DataFrame, policy: Optional[OpsPolicy] = None) -> pd.DataFrame:
    """
    给利润控量 Watchlist 增加 reason_1/2/3（防御性：失败不崩）。
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    try:
        tags = out.apply(lambda r: _pick_profit_reduce_reasons(r.to_dict(), policy=policy), axis=1)
        out["reason_1"] = tags.map(lambda xs: xs[0] if isinstance(xs, list) and len(xs) >= 1 else "")
        out["reason_2"] = tags.map(lambda xs: xs[1] if isinstance(xs, list) and len(xs) >= 2 else "")
        out["reason_3"] = tags.map(lambda xs: xs[2] if isinstance(xs, list) and len(xs) >= 3 else "")
    except Exception:
        for c in ("reason_1", "reason_2", "reason_3"):
            if c not in out.columns:
                out[c] = ""
    return out


def _pick_oos_with_ad_spend_reasons(row: Dict[str, object], policy: Optional[OpsPolicy] = None) -> List[str]:
    """
    断货仍烧钱 Watchlist 的原因标签（最多 3 个）。
    """
    reasons: List[str] = []

    try:
        days = int(float(row.get("oos_with_ad_spend_days", 0) or 0))
    except Exception:
        days = 0
    if days > 0:
        reasons.append(f"累计断货仍烧钱({days}天)")
    if days >= 3:
        reasons.append("断货天数多")

    cover_below = 0.0
    try:
        cover_below = float(getattr(policy, "block_scale_when_cover_days_below", 0.0) or 0.0) if policy is not None else 0.0
    except Exception:
        cover_below = 0.0
    try:
        cover_7d = float(row.get("inventory_cover_days_7d", 0.0) or 0.0)
    except Exception:
        cover_7d = 0.0
    try:
        flag_low_inv = int(float(row.get("flag_low_inventory", 0) or 0))
    except Exception:
        flag_low_inv = 0
    if (cover_below > 0 and cover_7d > 0 and cover_7d < cover_below) or (flag_low_inv > 0):
        reasons.append("库存风险")

    try:
        blocked_top = int(float(row.get("top_blocked_action_count", 0) or 0))
    except Exception:
        blocked_top = 0
    if blocked_top >= 3:
        reasons.append("动作阻断多")

    if not reasons:
        reasons.append("断货风险")
    return [x for x in reasons if x][:3]


def _annotate_oos_with_ad_spend_reasons(df: pd.DataFrame, policy: Optional[OpsPolicy] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    try:
        tags = out.apply(lambda r: _pick_oos_with_ad_spend_reasons(r.to_dict(), policy=policy), axis=1)
        out["reason_1"] = tags.map(lambda xs: xs[0] if isinstance(xs, list) and len(xs) >= 1 else "")
        out["reason_2"] = tags.map(lambda xs: xs[1] if isinstance(xs, list) and len(xs) >= 2 else "")
        out["reason_3"] = tags.map(lambda xs: xs[2] if isinstance(xs, list) and len(xs) >= 3 else "")
    except Exception:
        for c in ("reason_1", "reason_2", "reason_3"):
            if c not in out.columns:
                out[c] = ""
    return out


def _pick_spend_up_no_sales_reasons(row: Dict[str, object], policy: Optional[OpsPolicy] = None) -> List[str]:
    """
    加花费但销量不增 Watchlist 的原因标签（最多 3 个）。
    """
    reasons: List[str] = ["加花费无增量"]

    # 断货/库存风险（优先排查供给侧）
    try:
        oos_with_spend_days = int(float(row.get("oos_with_ad_spend_days", 0) or 0))
    except Exception:
        oos_with_spend_days = 0
    try:
        flag_oos = int(float(row.get("flag_oos", 0) or 0))
    except Exception:
        flag_oos = 0
    try:
        spend_roll = float(row.get("ad_spend_roll", 0.0) or 0.0)
    except Exception:
        spend_roll = 0.0
    if (oos_with_spend_days > 0) or (flag_oos > 0 and spend_roll > 0):
        reasons.append("断货风险")

    cover_below = 0.0
    try:
        cover_below = float(getattr(policy, "block_scale_when_cover_days_below", 0.0) or 0.0) if policy is not None else 0.0
    except Exception:
        cover_below = 0.0
    try:
        cover_7d = float(row.get("inventory_cover_days_7d", 0.0) or 0.0)
    except Exception:
        cover_7d = 0.0
    try:
        flag_low_inv = int(float(row.get("flag_low_inventory", 0) or 0))
    except Exception:
        flag_low_inv = 0
    if (cover_below > 0 and cover_7d > 0 and cover_7d < cover_below) or (flag_low_inv > 0):
        reasons.append("库存风险")

    # 销量走弱（趋势/速度）
    try:
        delta_sales = float(row.get("delta_sales", 0.0) or 0.0)
    except Exception:
        delta_sales = 0.0
    if delta_sales < 0:
        reasons.append("销量下降")

    try:
        spd7 = float(row.get("sales_per_day_7d", 0.0) or 0.0)
    except Exception:
        spd7 = 0.0
    try:
        spd30 = float(row.get("sales_per_day_30d", 0.0) or 0.0)
    except Exception:
        spd30 = 0.0
    if spd30 > 0 and (spd7 / spd30) < 0.8:
        reasons.append("速度下降")

    # 广告依赖高
    ad_dep_thr = 0.0
    try:
        fs = getattr(policy, "dashboard_focus_scoring", None) if policy is not None else None
        ad_dep_thr = float(getattr(fs, "high_ad_dependency_threshold", 0.8) or 0.8) if fs is not None else 0.8
    except Exception:
        ad_dep_thr = 0.8
    try:
        ad_share = float(row.get("ad_sales_share", 0.0) or 0.0)
    except Exception:
        ad_share = 0.0
    if ad_dep_thr > 0 and ad_share >= ad_dep_thr:
        reasons.append("广告依赖高")

    # 利润方向=控量（提示投放要回到产品语境）
    try:
        pdirection = str(row.get("profit_direction", "") or "").strip().lower()
    except Exception:
        pdirection = ""
    if pdirection == "reduce":
        reasons.append("利润方向=控量")

    # 动作阻断多
    try:
        blocked_top = int(float(row.get("top_blocked_action_count", 0) or 0))
    except Exception:
        blocked_top = 0
    if blocked_top >= 3:
        reasons.append("动作阻断多")

    return [x for x in reasons if x][:3]


def _annotate_spend_up_no_sales_reasons(df: pd.DataFrame, policy: Optional[OpsPolicy] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    try:
        tags = out.apply(lambda r: _pick_spend_up_no_sales_reasons(r.to_dict(), policy=policy), axis=1)
        out["reason_1"] = tags.map(lambda xs: xs[0] if isinstance(xs, list) and len(xs) >= 1 else "")
        out["reason_2"] = tags.map(lambda xs: xs[1] if isinstance(xs, list) and len(xs) >= 2 else "")
        out["reason_3"] = tags.map(lambda xs: xs[2] if isinstance(xs, list) and len(xs) >= 3 else "")
    except Exception:
        for c in ("reason_1", "reason_2", "reason_3"):
            if c not in out.columns:
                out[c] = ""
    return out


def build_profit_reduce_watchlist(
    asin_cockpit: Optional[pd.DataFrame],
    max_rows: int = 200,
    policy: Optional[OpsPolicy] = None,
) -> pd.DataFrame:
    """
    利润控量 Watchlist（profit_reduce_watchlist.csv）。

    设计目标：
    - 把“profit_direction=reduce（建议控量）”从概念提示变成可筛选清单；
    - 优先关注“仍在烧钱”的 ASIN：ad_spend_roll > 0；
    - 一行覆盖：类目/ASIN/生命周期/速度/覆盖/利润承受度/动作量/变化来源（尽量一眼能判断）。

    注意：
    - 不引入新数据源/唯一 ID；
    - 防御性：缺列/空表不崩，返回空表即可。
    """
    ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
    if ac is None or ac.empty or "asin" not in ac.columns:
        return pd.DataFrame()

    out = ac.copy()
    out["asin"] = out["asin"].astype(str).str.upper().str.strip()
    out = out[out["asin"] != ""].copy()
    if out.empty:
        return pd.DataFrame()

    if "profit_direction" not in out.columns:
        return pd.DataFrame()
    out["profit_direction"] = out.get("profit_direction", "").astype(str).str.strip().str.lower()

    # 数值化（用于筛选/排序/稳定输出）
    for c in (
        "focus_score",
        "ad_spend_roll",
        "tacos_roll",
        "inventory",
        "flag_oos",
        "flag_low_inventory",
        "oos_with_ad_spend_days",
        "ad_sales_share",
        "sales_per_day_7d",
        "sales_per_day_14d",
        "sales_per_day_30d",
        "orders_per_day_7d",
        "orders_per_day_14d",
        "orders_per_day_30d",
        "inventory_cover_days_7d",
        "inventory_cover_days_14d",
        "inventory_cover_days_30d",
        "delta_sales",
        "delta_spend",
        "profit_before_ads",
        "profit_after_ads",
        "max_ad_spend_by_profit",
        "target_tacos_by_margin",
        "top_action_count",
        "top_blocked_action_count",
        "drivers_delta_sales",
        "drivers_delta_ad_spend",
        "drivers_marginal_tacos",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    if "ad_spend_roll" not in out.columns:
        out["ad_spend_roll"] = 0.0
    out["ad_spend_roll"] = pd.to_numeric(out["ad_spend_roll"], errors="coerce").fillna(0.0)

    # 过滤：控量且仍在花钱
    out = out[(out["profit_direction"] == "reduce") & (out["ad_spend_roll"] > 0)].copy()
    if out.empty:
        return pd.DataFrame()

    # 规范化：分类/生命周期（便于筛选）
    out["product_category"] = out.get("product_category", "").map(_norm_product_category)
    if "current_phase" in out.columns:
        out["current_phase"] = out["current_phase"].map(_norm_phase)
    else:
        out["current_phase"] = ""

    # 超额花费：滚动花费 - 利润可承受上限（>0 越大越需要优先止血）
    try:
        if "max_ad_spend_by_profit" in out.columns:
            out["max_ad_spend_by_profit"] = pd.to_numeric(out["max_ad_spend_by_profit"], errors="coerce").fillna(0.0)
            out["over_profit_cap"] = (out["ad_spend_roll"] - out["max_ad_spend_by_profit"]).clip(lower=0.0)
        else:
            out["over_profit_cap"] = 0.0
    except Exception:
        out["over_profit_cap"] = 0.0

    # 排序：花费大优先 → focus → 超额 → Top 动作量（更贴近“止血/收口”）
    sort_cols: List[str] = []
    if "ad_spend_roll" in out.columns:
        sort_cols.append("ad_spend_roll")
    if "focus_score" in out.columns:
        sort_cols.append("focus_score")
    if "over_profit_cap" in out.columns:
        sort_cols.append("over_profit_cap")
    if "top_action_count" in out.columns:
        sort_cols.append("top_action_count")
    if sort_cols:
        try:
            out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()
        except Exception:
            pass

    # 控制输出规模
    try:
        mr = max(1, int(max_rows or 0))
        out = out.head(mr).copy()
    except Exception:
        pass

    # 可解释原因标签（最多 3 个）：帮助运营快速判断先查哪里
    out = _annotate_profit_reduce_reasons(out, policy=policy)

    cols = [
        "product_category",
        "asin",
        "product_name",
        "current_phase",
        "cycle_id",
        "inventory",
        "flag_oos",
        "flag_low_inventory",
        "oos_with_ad_spend_days",
        # 速度/覆盖（先用 7d/30d 两档，避免表太宽）
        "sales_per_day_7d",
        "orders_per_day_7d",
        "inventory_cover_days_7d",
        "sales_per_day_30d",
        "orders_per_day_30d",
        "inventory_cover_days_30d",
        "delta_sales",
        "delta_spend",
        "ad_sales_share",
        # 利润承受度摘要
        "profit_stage",
        "profit_direction",
        "profit_before_ads",
        "profit_after_ads",
        "max_ad_spend_by_profit",
        "over_profit_cap",
        "target_tacos_by_margin",
        "reason_1",
        "reason_2",
        "reason_3",
        # 抓重点
        "focus_score",
        # 动作/变化来源
        "top_action_count",
        "top_blocked_action_count",
        "drivers_delta_sales",
        "drivers_delta_ad_spend",
        "drivers_marginal_tacos",
        "ad_spend_roll",
        "tacos_roll",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


def build_oos_with_ad_spend_watchlist(
    asin_cockpit: Optional[pd.DataFrame],
    max_rows: int = 200,
    policy: Optional[OpsPolicy] = None,
) -> pd.DataFrame:
    """
    断货仍烧钱 Watchlist（oos_with_ad_spend_watchlist.csv）。

    设计目标：
    - 运营优先处理“断货仍在投放”的浪费（库存=0 但广告仍消耗）；
    - 这是强信号：通常无需再纠结广告细节，先止损/关停/补货再谈放量。

    过滤口径（优先使用生命周期主窗口的派生字段）：
    - oos_with_ad_spend_days > 0 且 ad_spend_roll > 0

    注意：
    - 不引入新数据源/唯一ID
    - 防御性：缺列/空表不崩，返回空表即可
    """
    ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
    if ac is None or ac.empty or "asin" not in ac.columns:
        return pd.DataFrame()

    out = ac.copy()
    out["asin"] = out["asin"].astype(str).str.upper().str.strip()
    out = out[out["asin"] != ""].copy()
    if out.empty:
        return pd.DataFrame()

    if "oos_with_ad_spend_days" not in out.columns:
        return pd.DataFrame()

    out["oos_with_ad_spend_days"] = pd.to_numeric(out.get("oos_with_ad_spend_days", 0.0), errors="coerce").fillna(0.0)
    if "ad_spend_roll" not in out.columns:
        out["ad_spend_roll"] = 0.0
    out["ad_spend_roll"] = pd.to_numeric(out.get("ad_spend_roll", 0.0), errors="coerce").fillna(0.0)

    out = out[(out["oos_with_ad_spend_days"] > 0) & (out["ad_spend_roll"] > 0)].copy()
    if out.empty:
        return pd.DataFrame()

    # 规范化：分类/生命周期（便于筛选）
    out["product_category"] = out.get("product_category", "").map(_norm_product_category)
    if "current_phase" in out.columns:
        out["current_phase"] = out["current_phase"].map(_norm_phase)
    else:
        out["current_phase"] = ""

    # 数值化（用于排序/稳定输出）
    for c in (
        "focus_score",
        "inventory",
        "inventory_cover_days_7d",
        "inventory_cover_days_30d",
        "sales_per_day_7d",
        "sales_per_day_30d",
        "flag_oos",
        "flag_low_inventory",
        "top_action_count",
        "top_blocked_action_count",
        "drivers_delta_sales",
        "drivers_delta_ad_spend",
        "drivers_marginal_tacos",
        "tacos_roll",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # 排序：断货仍烧钱天数 → 花费 → focus → Top 动作量
    sort_cols: List[str] = []
    for c in ("oos_with_ad_spend_days", "ad_spend_roll", "focus_score", "top_action_count"):
        if c in out.columns:
            sort_cols.append(c)
    if sort_cols:
        try:
            out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()
        except Exception:
            pass

    # 控制输出规模
    try:
        mr = max(1, int(max_rows or 0))
        out = out.head(mr).copy()
    except Exception:
        pass

    # 可解释原因标签（最多 3 个）：帮助运营快速判断先查哪里
    out = _annotate_oos_with_ad_spend_reasons(out, policy=policy)

    cols = [
        "product_category",
        "asin",
        "product_name",
        "current_phase",
        "cycle_id",
        "inventory",
        "flag_oos",
        "flag_low_inventory",
        "inventory_cover_days_7d",
        "inventory_cover_days_30d",
        "oos_with_ad_spend_days",
        "reason_1",
        "reason_2",
        "reason_3",
        "focus_score",
        "top_action_count",
        "top_blocked_action_count",
        "drivers_delta_sales",
        "drivers_delta_ad_spend",
        "drivers_marginal_tacos",
        "ad_spend_roll",
        "tacos_roll",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


def build_spend_up_no_sales_watchlist(
    asin_cockpit: Optional[pd.DataFrame],
    max_rows: int = 200,
    policy: Optional[OpsPolicy] = None,
) -> pd.DataFrame:
    """
    加花费但销量不增 Watchlist（spend_up_no_sales_watchlist.csv）。

    设计目标：
    - 运营快速定位“花费上升但销量不增/下降”的可疑池子（通常需要否词/降价/结构调整/止损）；
    - 默认基于 compare_7d 的增量口径：delta_spend / delta_sales。

    过滤规则：
    - delta_spend > 0 且 delta_sales <= 0

    注意：
    - 不引入新数据源/唯一ID
    - 防御性：缺列/空表不崩，返回空表即可
    """
    ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
    if ac is None or ac.empty or "asin" not in ac.columns:
        return pd.DataFrame()

    out = ac.copy()
    out["asin"] = out["asin"].astype(str).str.upper().str.strip()
    out = out[out["asin"] != ""].copy()
    if out.empty:
        return pd.DataFrame()

    if "delta_spend" not in out.columns or "delta_sales" not in out.columns:
        return pd.DataFrame()

    out["delta_spend"] = pd.to_numeric(out.get("delta_spend", 0.0), errors="coerce").fillna(0.0)
    out["delta_sales"] = pd.to_numeric(out.get("delta_sales", 0.0), errors="coerce").fillna(0.0)
    if "ad_spend_roll" not in out.columns:
        out["ad_spend_roll"] = 0.0
    out["ad_spend_roll"] = pd.to_numeric(out.get("ad_spend_roll", 0.0), errors="coerce").fillna(0.0)

    out = out[(out["delta_spend"] > 0) & (out["delta_sales"] <= 0) & (out["ad_spend_roll"] > 0)].copy()
    if out.empty:
        return pd.DataFrame()

    # 规范化：分类/生命周期（便于筛选）
    out["product_category"] = out.get("product_category", "").map(_norm_product_category)
    if "current_phase" in out.columns:
        out["current_phase"] = out["current_phase"].map(_norm_phase)
    else:
        out["current_phase"] = ""

    # 数值化（用于排序/稳定输出）
    for c in (
        "focus_score",
        "inventory",
        "flag_oos",
        "flag_low_inventory",
        "oos_with_ad_spend_days",
        "ad_sales_share",
        "inventory_cover_days_7d",
        "inventory_cover_days_30d",
        "sales_per_day_7d",
        "sales_per_day_30d",
        "marginal_tacos",
        "marginal_ad_acos",
        "top_action_count",
        "top_blocked_action_count",
        "drivers_delta_sales",
        "drivers_delta_ad_spend",
        "drivers_marginal_tacos",
        "tacos_roll",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # 排序：Δspend 大优先 → Δsales 更差优先（更负优先）→ focus → Top 动作量
    sort_cols: List[str] = []
    sort_asc: List[bool] = []
    if "delta_spend" in out.columns:
        sort_cols.append("delta_spend")
        sort_asc.append(False)
    if "delta_sales" in out.columns:
        sort_cols.append("delta_sales")
        sort_asc.append(True)  # 更负更靠前
    if "focus_score" in out.columns:
        sort_cols.append("focus_score")
        sort_asc.append(False)
    if "top_action_count" in out.columns:
        sort_cols.append("top_action_count")
        sort_asc.append(False)
    if "ad_spend_roll" in out.columns:
        sort_cols.append("ad_spend_roll")
        sort_asc.append(False)
    if sort_cols:
        try:
            out = out.sort_values(sort_cols, ascending=sort_asc).copy()
        except Exception:
            pass

    # 控制输出规模
    try:
        mr = max(1, int(max_rows or 0))
        out = out.head(mr).copy()
    except Exception:
        pass

    # 可解释原因标签（最多 3 个）：帮助运营快速判断先查哪里
    out = _annotate_spend_up_no_sales_reasons(out, policy=policy)

    cols = [
        "product_category",
        "asin",
        "product_name",
        "current_phase",
        "cycle_id",
        "inventory",
        "flag_oos",
        "flag_low_inventory",
        "oos_with_ad_spend_days",
        "sales_per_day_7d",
        "inventory_cover_days_7d",
        "sales_per_day_30d",
        "inventory_cover_days_30d",
        "delta_sales",
        "delta_spend",
        "ad_sales_share",
        "profit_direction",
        "reason_1",
        "reason_2",
        "reason_3",
        "marginal_tacos",
        "marginal_ad_acos",
        "focus_score",
        "top_action_count",
        "top_blocked_action_count",
        "drivers_delta_sales",
        "drivers_delta_ad_spend",
        "drivers_marginal_tacos",
        "ad_spend_roll",
        "tacos_roll",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


def build_phase_down_recent_watchlist(
    asin_cockpit: Optional[pd.DataFrame],
    max_rows: int = 200,
    policy: Optional[OpsPolicy] = None,
) -> pd.DataFrame:
    """
    阶段下滑 Watchlist（phase_down_recent_watchlist.csv）。

    设计目标：
    - 把“动态生命周期”真正落到运营工作流：当 ASIN 近14天阶段走弱（down）且仍在花费时，优先排查根因；
    - 该信号本身不判定原因（可能是库存/评价/Listing/投放结构/利润承受度等），只负责把它从海量数据里捞出来。

    过滤规则：
    - phase_changed_recent_14d = 1
    - phase_trend_14d = 'down'
    - ad_spend_roll > 0

    注意：
    - 不引入新数据源/唯一ID
    - 防御性：缺列/空表不崩，返回空表即可
    """
    ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
    if ac is None or ac.empty or "asin" not in ac.columns:
        return pd.DataFrame()

    out = ac.copy()
    out["asin"] = out["asin"].astype(str).str.upper().str.strip()
    out = out[out["asin"] != ""].copy()
    if out.empty:
        return pd.DataFrame()

    need_cols = {"phase_changed_recent_14d", "phase_trend_14d", "ad_spend_roll"}
    if any(c not in out.columns for c in need_cols):
        return pd.DataFrame()

    out["phase_changed_recent_14d"] = pd.to_numeric(out.get("phase_changed_recent_14d", 0), errors="coerce").fillna(0).astype(int)
    out["phase_trend_14d"] = out.get("phase_trend_14d", "").astype(str).fillna("").str.strip().str.lower()
    out["ad_spend_roll"] = pd.to_numeric(out.get("ad_spend_roll", 0.0), errors="coerce").fillna(0.0)

    out = out[(out["phase_changed_recent_14d"] > 0) & (out["phase_trend_14d"] == "down") & (out["ad_spend_roll"] > 0)].copy()
    if out.empty:
        return pd.DataFrame()

    # 规范化：分类/生命周期（便于筛选）
    out["product_category"] = out.get("product_category", "").map(_norm_product_category)
    if "current_phase" in out.columns:
        out["current_phase"] = out["current_phase"].map(_norm_phase)
    if "prev_phase" in out.columns:
        out["prev_phase"] = out["prev_phase"].map(_norm_phase)

    # 数值化（用于排序/稳定输出）
    for c in (
        "focus_score",
        "inventory",
        "flag_low_inventory",
        "flag_oos",
        "oos_with_ad_spend_days",
        "ad_sales_share",
        "aov_prev_7d",
        "aov_recent_7d",
        "delta_aov_7d",
        "gross_margin",
        "sessions_prev_7d",
        "sessions_recent_7d",
        "cvr_prev_7d",
        "cvr_recent_7d",
        "organic_sales_prev_7d",
        "organic_sales_recent_7d",
        "organic_sales_share_prev_7d",
        "organic_sales_share_recent_7d",
        "inventory_cover_days_7d",
        "inventory_cover_days_30d",
        "sales_per_day_7d",
        "sales_per_day_30d",
        "delta_sales",
        "delta_spend",
        "delta_sessions",
        "delta_cvr",
        "delta_organic_sales",
        "delta_organic_sales_share",
        "drivers_delta_sales",
        "drivers_delta_ad_spend",
        "drivers_marginal_tacos",
        "top_action_count",
        "top_blocked_action_count",
        "tacos_roll",
        "phase_change_days_ago",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # 排序：花费大优先 → focus 高优先 → Top 动作量
    sort_cols: List[str] = []
    sort_asc: List[bool] = []
    if "ad_spend_roll" in out.columns:
        sort_cols.append("ad_spend_roll")
        sort_asc.append(False)
    if "focus_score" in out.columns:
        sort_cols.append("focus_score")
        sort_asc.append(False)
    if "top_action_count" in out.columns:
        sort_cols.append("top_action_count")
        sort_asc.append(False)
    if sort_cols:
        try:
            out = out.sort_values(sort_cols, ascending=sort_asc).copy()
        except Exception:
            pass

    # 控制输出规模
    try:
        mr = max(1, int(max_rows or 0))
        out = out.head(mr).copy()
    except Exception:
        pass

    # 可解释原因标签（最多 3 个）：帮助运营快速判断先查哪里
    out = _annotate_phase_down_reasons(out, policy=policy)

    cols = [
        "product_category",
        "asin",
        "product_name",
        "prev_phase",
        "current_phase",
        "phase_change",
        "phase_change_days_ago",
        "phase_trend_14d",
        "reason_1",
        "reason_2",
        "reason_3",
        "cycle_id",
        "inventory",
        "flag_oos",
        "flag_low_inventory",
        "oos_with_ad_spend_days",
        # CVR 信号证据列：让“转化下滑”原因更可解释
        "sessions_prev_7d",
        "sessions_recent_7d",
        "cvr_prev_7d",
        "cvr_recent_7d",
        # 客单价/毛利率证据列：让“客单价下滑/毛利率偏低”更可解释
        "aov_prev_7d",
        "aov_recent_7d",
        "delta_aov_7d",
        "gross_margin",
        # 自然端证据列：让“自然端回落”原因更可解释
        "organic_sales_prev_7d",
        "organic_sales_recent_7d",
        "organic_sales_share_prev_7d",
        "organic_sales_share_recent_7d",
        "sales_per_day_7d",
        "inventory_cover_days_7d",
        "sales_per_day_30d",
        "inventory_cover_days_30d",
        "delta_sales",
        "delta_spend",
        "delta_sessions",
        "delta_cvr",
        "delta_organic_sales",
        "delta_organic_sales_share",
        "ad_sales_share",
        "profit_direction",
        "focus_score",
        "top_action_count",
        "top_blocked_action_count",
        "drivers_delta_sales",
        "drivers_delta_ad_spend",
        "drivers_marginal_tacos",
        "ad_spend_roll",
        "tacos_roll",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


def build_scale_opportunity_watchlist(
    asin_cockpit: Optional[pd.DataFrame],
    max_rows: int = 200,
    policy: Optional[OpsPolicy] = None,
) -> pd.DataFrame:
    """
    机会 Watchlist（scale_opportunity_watchlist.csv）：可放量窗口 / 低花费高潜候选。

    设计目标：
    - 运营不只看“风险止损”，还需要一个“可加码/迁移预算”的候选池；
    - 仍然不依赖唯一 ID：完全基于现有 ASIN 维度与 rolling/compare 指标；
    - 规则化且可解释：便于你后续按业务迭代阈值。

    当前过滤规则（保守版，阈值可配置）：
    - 有销量：sales_per_day_7d > min_sales_per_day_7d
    - 有增长：delta_sales > min_delta_sales（默认 compare_7d）
    - 有库存：inventory_cover_days_7d >= min_inventory_cover_days_7d
    - 效率不错：tacos_roll <= max_tacos_roll 且 marginal_tacos <= max_marginal_tacos
    - 风险约束（可开关）：不触发断货/低库存/断货仍烧钱
    - 生命周期排除：exclude_phases（默认 decline/inactive）

    排序（更贴近“值得先看”）：
    - ΔSales desc → 速度 desc → TACOS asc → 库存覆盖 desc
    """
    ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
    if ac is None or ac.empty or "asin" not in ac.columns:
        return pd.DataFrame()

    out = ac.copy()
    out["asin"] = out["asin"].astype(str).str.upper().str.strip()
    out = out[out["asin"] != ""].copy()
    if out.empty:
        return pd.DataFrame()

    # 阈值：优先从 policy.dashboard_scale_window 读取；否则用默认
    sw = getattr(policy, "dashboard_scale_window", None) if policy is not None else None
    try:
        min_sales_pd = float(getattr(sw, "min_sales_per_day_7d", 0.0) or 0.0)
    except Exception:
        min_sales_pd = 0.0
    try:
        min_delta_sales = float(getattr(sw, "min_delta_sales", 0.0) or 0.0)
    except Exception:
        min_delta_sales = 0.0
    try:
        min_cover_days = float(getattr(sw, "min_inventory_cover_days_7d", 30.0) or 30.0)
    except Exception:
        min_cover_days = 30.0
    try:
        max_tacos_roll = float(getattr(sw, "max_tacos_roll", 0.25) or 0.25)
    except Exception:
        max_tacos_roll = 0.25
    try:
        max_marginal_tacos = float(getattr(sw, "max_marginal_tacos", 0.25) or 0.25)
    except Exception:
        max_marginal_tacos = 0.25
    try:
        exclude_phases = [str(x or "").strip().lower() for x in getattr(sw, "exclude_phases", []) or [] if str(x or "").strip()]
    except Exception:
        exclude_phases = []
    if not exclude_phases:
        exclude_phases = ["decline", "inactive"]
    require_no_oos = bool(getattr(sw, "require_no_oos", True)) if sw is not None else True
    require_no_low_inventory = bool(getattr(sw, "require_no_low_inventory", True)) if sw is not None else True
    require_oos_spend_days_zero = bool(getattr(sw, "require_oos_with_ad_spend_days_zero", True)) if sw is not None else True

    # 必需列：没有就直接返回空表（避免“规则失真”）
    required_cols = [
        "sales_per_day_7d",
        "inventory_cover_days_7d",
        "tacos_roll",
        "delta_sales",
        "marginal_tacos",
    ]
    for c in required_cols:
        if c not in out.columns:
            return pd.DataFrame()

    # 数值化（用于过滤/排序/稳定输出）
    for c in (
        "sales_per_day_7d",
        "inventory_cover_days_7d",
        "tacos_roll",
        "delta_sales",
        "delta_spend",
        "marginal_tacos",
        "ad_spend_roll",
        "flag_oos",
        "flag_low_inventory",
        "oos_with_ad_spend_days",
        "focus_score",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # 规范化：分类/生命周期（便于筛选）
    out["product_category"] = out.get("product_category", "").map(_norm_product_category)
    if "current_phase" in out.columns:
        out["current_phase"] = out["current_phase"].map(_norm_phase)
    else:
        out["current_phase"] = ""

    # 过滤：可放量窗口（保守版）
    cover_days = pd.to_numeric(out.get("inventory_cover_days_7d", 0.0), errors="coerce").fillna(0.0)
    sales_pd = pd.to_numeric(out.get("sales_per_day_7d", 0.0), errors="coerce").fillna(0.0)
    tacos_roll = pd.to_numeric(out.get("tacos_roll", 0.0), errors="coerce").fillna(0.0)
    delta_sales = pd.to_numeric(out.get("delta_sales", 0.0), errors="coerce").fillna(0.0)
    marginal_tacos = pd.to_numeric(out.get("marginal_tacos", 0.0), errors="coerce").fillna(0.0)

    flag_oos = pd.to_numeric(out.get("flag_oos", 0.0), errors="coerce").fillna(0.0)
    flag_low = pd.to_numeric(out.get("flag_low_inventory", 0.0), errors="coerce").fillna(0.0)
    oos_spend_days = pd.to_numeric(out.get("oos_with_ad_spend_days", 0.0), errors="coerce").fillna(0.0)
    phase = out.get("current_phase", "").astype(str).str.strip().str.lower()

    cond = (sales_pd > float(min_sales_pd)) & (delta_sales > float(min_delta_sales)) & (cover_days >= float(min_cover_days))
    cond = cond & (tacos_roll <= float(max_tacos_roll)) & (marginal_tacos <= float(max_marginal_tacos))
    if require_no_oos:
        cond = cond & (flag_oos <= 0)
    if require_no_low_inventory:
        cond = cond & (flag_low <= 0)
    if require_oos_spend_days_zero:
        cond = cond & (oos_spend_days <= 0)
    if exclude_phases:
        cond = cond & (~phase.isin(exclude_phases))
    out = out[cond].copy()
    if out.empty:
        return pd.DataFrame()

    # 排序：ΔSales desc → 速度 desc → tacos_roll asc → 库存覆盖 desc
    sort_cols: List[str] = []
    sort_asc: List[bool] = []
    for c, asc in (
        ("delta_sales", False),
        ("sales_per_day_7d", False),
        ("tacos_roll", True),
        ("inventory_cover_days_7d", False),
    ):
        if c in out.columns:
            sort_cols.append(c)
            sort_asc.append(bool(asc))
    if sort_cols:
        try:
            out = out.sort_values(sort_cols, ascending=sort_asc).copy()
        except Exception:
            pass

    # 控制输出规模
    try:
        mr = max(1, int(max_rows or 0))
        out = out.head(mr).copy()
    except Exception:
        pass

    cols = [
        "product_category",
        "asin",
        "product_name",
        "current_phase",
        "cycle_id",
        "sales_per_day_7d",
        "inventory_cover_days_7d",
        "delta_sales",
        "delta_spend",
        "marginal_tacos",
        "tacos_roll",
        "ad_spend_roll",
        "focus_score",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


def build_opportunity_action_board(
    action_board_dedup_all: Optional[pd.DataFrame],
    scale_opportunity_watchlist: Optional[pd.DataFrame],
    max_rows: int = 500,
) -> pd.DataFrame:
    """
    Opportunity Action Board（opportunity_action_board.csv）：把“机会池”映射到可执行动作清单。

    设计目标：
    - 运营从机会 ASIN 直接看到“可放量动作”（BID_UP/BUDGET_UP），减少来回筛选；
    - 只输出未阻断（blocked=0）的动作，避免“库存/断货约束”下误导加码；
    - 不引入新数据源/唯一ID，完全基于现有 Action Board + 机会 Watchlist。
    """
    ab = action_board_dedup_all.copy() if isinstance(action_board_dedup_all, pd.DataFrame) else pd.DataFrame()
    wl = scale_opportunity_watchlist.copy() if isinstance(scale_opportunity_watchlist, pd.DataFrame) else pd.DataFrame()
    if ab is None or ab.empty or wl is None or wl.empty:
        return pd.DataFrame()
    if "asin_hint" not in ab.columns or "asin" not in wl.columns:
        return pd.DataFrame()

    scale_asins = {str(x or "").strip().upper() for x in wl["asin"].tolist()}
    scale_asins = {x for x in scale_asins if x}
    if not scale_asins:
        return pd.DataFrame()

    out = ab.copy()
    out["asin_hint"] = out["asin_hint"].astype(str).str.upper().str.strip()
    out = out[out["asin_hint"].isin(scale_asins)].copy()
    if out.empty:
        return pd.DataFrame()

    # 只保留“放量类动作”（与 blocked 口径一致）
    scale_actions = {"BID_UP", "BUDGET_UP"}
    out["action_type"] = out.get("action_type", "").astype(str).str.upper().str.strip()
    out = out[out["action_type"].isin(scale_actions)].copy()
    if out.empty:
        return pd.DataFrame()

    # 只保留未阻断（blocked=0）
    out["blocked"] = pd.to_numeric(out.get("blocked", 0), errors="coerce").fillna(0).astype(int)
    out = out[out["blocked"] <= 0].copy()
    if out.empty:
        return pd.DataFrame()

    # 可选：如果存在 asin_scale_window 标记，则要求为 1（避免口径不一致）
    if "asin_scale_window" in out.columns:
        try:
            out["asin_scale_window"] = pd.to_numeric(out["asin_scale_window"], errors="coerce").fillna(0).astype(int)
            out = out[out["asin_scale_window"] >= 1].copy()
        except Exception:
            pass
    if out.empty:
        return pd.DataFrame()

    # 排序：优先级分 → 证据花费
    out["action_priority_score"] = pd.to_numeric(out.get("action_priority_score", 0.0), errors="coerce").fillna(0.0)
    out["e_spend"] = pd.to_numeric(out.get("e_spend", 0.0), errors="coerce").fillna(0.0)
    try:
        out = out.sort_values(["action_priority_score", "e_spend"], ascending=[False, False]).copy()
    except Exception:
        pass

    # 控制输出规模
    try:
        mr = max(1, int(max_rows or 0))
        out = out.head(mr).copy()
    except Exception:
        pass

    # 操作手册联动（不改口径）
    try:
        out = enrich_action_board_with_playbook_scene(out)
    except Exception:
        pass

    cols = [
        "priority",
        "action_priority_score",
        "product_category",
        "asin_hint",
        "asin_hint_confidence",
        "asin_scale_window_reason",
        "current_phase",
        "action_type",
        "action_value",
        "object_name",
        "e_spend",
        "priority_reason",
        "priority_reason_1",
        "priority_reason_2",
        "priority_reason_3",
        "playbook_scene",
        "playbook_url",
        # 便于机会判断的关键产品维度
        "asin_delta_sales",
        "asin_delta_spend",
        "asin_sales_per_day_7d",
        "asin_inventory_cover_days_7d",
        "asin_tacos_roll",
        "asin_marginal_tacos",
        # 便于定位（可选）
        "ad_type",
        "level",
        "campaign",
        "ad_group",
        "match_type",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


def build_budget_transfer_plan_table(
    budget_transfer_plan: Optional[Dict[str, object]],
    max_rows: int = 500,
) -> pd.DataFrame:
    """
    预算迁移净表（budget_transfer_plan.csv）。

    设计目标：
    - 输出“从哪里挪到哪里”的净迁移（transfers），运营可直接照着执行；
    - 若本期没有可放量池，则输出“预算回收/降档”（savings，to_bucket=RESERVE）；
    - 不依赖“当前预算值”（赛狐通常拿不到），金额是估算值（基于本期花费×建议百分比）。
    """
    plan = budget_transfer_plan if isinstance(budget_transfer_plan, dict) else {}
    transfers = plan.get("transfers") if isinstance(plan, dict) else None
    savings = plan.get("savings") if isinstance(plan, dict) else None
    cuts = plan.get("cuts") if isinstance(plan, dict) else None

    pb_doc = "../../../../docs/OPS_PLAYBOOK.md"

    def _pick_scene(transfer_type: str) -> str:
        t = str(transfer_type or "").strip().lower()
        if t == "transfer":
            return "scene-scale-opportunity"
        return "scene-profit-reduce"

    # savings 默认缺少 severity/spend，这里用 cuts 元信息补齐（如果存在）
    cut_meta: Dict[Tuple[str, str], Dict[str, object]] = {}
    if isinstance(cuts, list):
        for c in cuts:
            if not isinstance(c, dict):
                continue
            ad_type = str(c.get("ad_type", "") or "").strip()
            campaign = str(c.get("campaign", "") or "").strip()
            if not ad_type or not campaign:
                continue
            cut_meta[(ad_type, campaign)] = {
                "from_severity": c.get("severity", ""),
                "from_spend": c.get("camp_spend", ""),
            }

    rows: List[Dict[str, object]] = []
    if isinstance(transfers, list):
        for t in transfers:
            if not isinstance(t, dict):
                continue
            scene = _pick_scene("transfer")
            rows.append(
                {
                    "strategy": (t.get("strategy", "") or "profit"),
                    "transfer_type": "transfer",
                    "from_ad_type": t.get("from_ad_type", ""),
                    "from_campaign": t.get("from_campaign", ""),
                    "from_severity": t.get("from_severity", ""),
                    "from_spend": t.get("from_spend", ""),
                    "from_asin_hint": t.get("from_asin_hint", ""),
                    "to_ad_type": t.get("to_ad_type", ""),
                    "to_campaign": t.get("to_campaign", ""),
                    "to_severity": t.get("to_severity", ""),
                    "to_spend": t.get("to_spend", ""),
                    "to_asin_hint": t.get("to_asin_hint", ""),
                    "to_confidence": t.get("to_confidence", ""),
                    "to_opp_asin_count": t.get("to_opp_asin_count", ""),
                    "to_opp_asins_top": t.get("to_opp_asins_top", ""),
                    "to_opp_spend": t.get("to_opp_spend", ""),
                    "to_opp_spend_share": t.get("to_opp_spend_share", ""),
                    "to_opp_action_count": t.get("to_opp_action_count", ""),
                    "to_bucket": "",
                    "amount_usd_estimated": t.get("amount_usd_estimated", 0.0),
                    "note": t.get("note", ""),
                    "playbook_scene": scene,
                    "playbook_url": f"{pb_doc}#{scene}" if scene else "",
                }
            )
    if isinstance(savings, list):
        for s in savings:
            if not isinstance(s, dict):
                continue
            from_ad_type = str(s.get("from_ad_type", "") or "").strip()
            from_campaign = str(s.get("from_campaign", "") or "").strip()
            meta = cut_meta.get((from_ad_type, from_campaign), {})
            scene = _pick_scene("reserve")
            rows.append(
                {
                    "strategy": (s.get("strategy", "") or "profit"),
                    "transfer_type": "reserve",
                    "from_ad_type": from_ad_type,
                    "from_campaign": from_campaign,
                    "from_severity": meta.get("from_severity", ""),
                    "from_spend": meta.get("from_spend", ""),
                    "from_asin_hint": s.get("from_asin_hint", ""),
                    "to_ad_type": "",
                    "to_campaign": "",
                    "to_severity": "",
                    "to_spend": "",
                    "to_asin_hint": "",
                    "to_confidence": "",
                    "to_opp_asin_count": "",
                    "to_opp_asins_top": "",
                    "to_opp_spend": "",
                    "to_opp_spend_share": "",
                    "to_opp_action_count": "",
                    "to_bucket": s.get("to_bucket", "RESERVE"),
                    "amount_usd_estimated": s.get("amount_usd_estimated", 0.0),
                    "note": s.get("note", ""),
                    "playbook_scene": scene,
                    "playbook_url": f"{pb_doc}#{scene}" if scene else "",
                }
            )

    cols = [
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
        "to_opp_asin_count",
        "to_opp_asins_top",
        "to_opp_spend",
        "to_opp_spend_share",
        "to_opp_action_count",
        "to_bucket",
        "amount_usd_estimated",
        "playbook_scene",
        "playbook_url",
        "note",
    ]
    out = pd.DataFrame(rows, columns=cols)
    if out is None or out.empty:
        return pd.DataFrame(columns=cols)

    for c in (
        "from_severity",
        "from_spend",
        "to_severity",
        "to_spend",
        "to_opp_asin_count",
        "to_opp_spend",
        "to_opp_spend_share",
        "to_opp_action_count",
        "amount_usd_estimated",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # 排序：先看净迁移（transfer），再看回收（reserve），每组内按金额降序
    try:
        out["amount_usd_estimated"] = out["amount_usd_estimated"].fillna(0.0)
        out["transfer_rank"] = out["transfer_type"].astype(str).map({"transfer": 0, "reserve": 1}).fillna(9).astype(int)
        out = out.sort_values(["transfer_rank", "amount_usd_estimated"], ascending=[True, False]).copy()
        out = out.drop(columns=["transfer_rank"], errors="ignore")
    except Exception:
        pass

    try:
        mr = max(1, int(max_rows or 0))
        out = out.head(mr).copy()
    except Exception:
        pass

    return out.reset_index(drop=True)


def build_inventory_risk_watchlist(
    asin_cockpit: Optional[pd.DataFrame],
    max_rows: int = 200,
    policy: Optional[OpsPolicy] = None,
    spend_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    库存告急仍投放 Watchlist（预警型）：
    - 近7天覆盖天数较低，但仍在花费
    - 用于提前减速/控量，不等到断货
    """
    if asin_cockpit is None or asin_cockpit.empty:
        return pd.DataFrame()
    try:
        out = asin_cockpit.copy()
        if "asin" in out.columns:
            out["asin"] = out["asin"].astype(str).str.upper().str.strip()
        if "product_category" in out.columns:
            out["product_category"] = out["product_category"].map(_norm_product_category)
        # 数值化
        for c in ("ad_spend_roll", "inventory_cover_days_7d", "inventory_cover_days_14d", "sales_per_day_7d"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

        cover_thr = 7.0
        try:
            if isinstance(policy, OpsPolicy):
                cover_thr = float(getattr(policy, "block_scale_when_cover_days_below", cover_thr) or cover_thr)
        except Exception:
            cover_thr = 7.0

        # 核心筛选：覆盖天数较低 + 仍在花费
        out = out[(out.get("ad_spend_roll", 0.0) >= float(spend_threshold)) & (out.get("inventory_cover_days_7d", 0.0) > 0)].copy()
        out = out[out.get("inventory_cover_days_7d", 0.0) <= float(cover_thr)].copy()
        if out.empty:
            return out

        # reason 标签
        def _reasons(row: Dict[str, object]) -> List[str]:
            rs: List[str] = []
            try:
                cover7 = float(row.get("inventory_cover_days_7d", 0.0) or 0.0)
            except Exception:
                cover7 = 0.0
            try:
                cover14 = float(row.get("inventory_cover_days_14d", 0.0) or 0.0)
            except Exception:
                cover14 = 0.0
            if cover7 > 0 and cover7 <= 3:
                rs.append("库存告急≤3天")
            elif cover7 > 0:
                rs.append(f"库存告急≤{int(cover_thr)}天")
            if cover14 > 0 and cover7 > 0 and cover7 < cover14:
                rs.append("消耗加速")
            rs.append("仍在投放")
            return rs[:3]

        try:
            tags = out.apply(lambda r: _reasons(r.to_dict()), axis=1)
            out["reason_1"] = tags.map(lambda xs: xs[0] if isinstance(xs, list) and len(xs) >= 1 else "")
            out["reason_2"] = tags.map(lambda xs: xs[1] if isinstance(xs, list) and len(xs) >= 2 else "")
            out["reason_3"] = tags.map(lambda xs: xs[2] if isinstance(xs, list) and len(xs) >= 3 else "")
        except Exception:
            for c in ("reason_1", "reason_2", "reason_3"):
                if c not in out.columns:
                    out[c] = ""

        # 排序：覆盖天数↑紧急度 + 花费
        out = out.sort_values(["inventory_cover_days_7d", "ad_spend_roll"], ascending=[True, False]).copy()
        if max_rows and int(max_rows) > 0:
            out = out.head(int(max_rows))
        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _sigmoid_modifier(dos: float, optimal: float, steepness: float, min_m: float, max_m: float) -> float:
    try:
        if dos <= 0 or max_m <= 0:
            return 1.0
        k = float(steepness or 0.0)
        if k <= 0:
            return 1.0
        min_v = float(min_m)
        max_v = float(max_m)
        if max_v < min_v:
            max_v = min_v
        rng = max_v - min_v
        if rng <= 0:
            return 1.0
        core = 1.0 / (1.0 + math.exp(-k * (float(dos) - float(optimal))))
        return float(rng * core + min_v)
    except Exception:
        return 1.0


def build_inventory_sigmoid_watchlist(
    asin_cockpit: Optional[pd.DataFrame],
    max_rows: int = 200,
    policy: Optional[OpsPolicy] = None,
) -> pd.DataFrame:
    """
    库存调速建议（Sigmoid）：只给建议，不影响排序/不自动执行。
    """
    if asin_cockpit is None or asin_cockpit.empty:
        return pd.DataFrame()
    try:
        sig = getattr(policy, "dashboard_inventory_sigmoid", None) if isinstance(policy, OpsPolicy) else None
        if sig is None:
            sig = InventorySigmoidPolicy()
        if not bool(getattr(sig, "enabled", True)):
            return pd.DataFrame()

        out = asin_cockpit.copy()
        if "asin" in out.columns:
            out["asin"] = out["asin"].astype(str).str.upper().str.strip()
        if "product_category" in out.columns:
            out["product_category"] = out["product_category"].map(_norm_product_category)

        for c in ("ad_spend_roll", "inventory_cover_days_7d", "sales_per_day_7d"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

        min_spend = float(getattr(sig, "min_ad_spend_roll", 10.0) or 10.0)
        out = out[out.get("ad_spend_roll", 0.0) >= float(min_spend)].copy()
        out = out[out.get("inventory_cover_days_7d", 0.0) > 0].copy()
        if out.empty:
            return out

        opt = float(getattr(sig, "optimal_cover_days", 45.0) or 45.0)
        steep = float(getattr(sig, "steepness", 0.1) or 0.1)
        min_m = float(getattr(sig, "min_modifier", 0.5) or 0.5)
        max_m = float(getattr(sig, "max_modifier", 1.5) or 1.5)
        min_change = float(getattr(sig, "min_change_ratio", 0.1) or 0.1)

        out["sigmoid_modifier"] = out["inventory_cover_days_7d"].map(lambda x: _sigmoid_modifier(float(x or 0.0), opt, steep, min_m, max_m))
        out["sigmoid_delta"] = (out["sigmoid_modifier"] - 1.0).abs()
        out = out[out["sigmoid_delta"] >= float(min_change)].copy()
        if out.empty:
            return out

        def _reasons(row: Dict[str, object]) -> List[str]:
            rs: List[str] = []
            try:
                cover7 = float(row.get("inventory_cover_days_7d", 0.0) or 0.0)
            except Exception:
                cover7 = 0.0
            if cover7 > 0 and cover7 < opt:
                rs.append(f"覆盖偏紧({cover7:.1f}d<目标{opt:.0f}d)")
            elif cover7 > 0 and cover7 > opt:
                rs.append(f"覆盖偏高({cover7:.1f}d>目标{opt:.0f}d)")
            rs.append(f"建议系数x{float(row.get('sigmoid_modifier', 1.0)):.2f}")
            rs.append("仍在投放")
            return rs[:3]

        try:
            tags = out.apply(lambda r: _reasons(r.to_dict()), axis=1)
            out["reason_1"] = tags.map(lambda xs: xs[0] if isinstance(xs, list) and len(xs) >= 1 else "")
            out["reason_2"] = tags.map(lambda xs: xs[1] if isinstance(xs, list) and len(xs) >= 2 else "")
            out["reason_3"] = tags.map(lambda xs: xs[2] if isinstance(xs, list) and len(xs) >= 3 else "")
        except Exception:
            for c in ("reason_1", "reason_2", "reason_3"):
                if c not in out.columns:
                    out[c] = ""

        def _action(v: float) -> str:
            try:
                return "控量" if float(v) < 1.0 else "放量"
            except Exception:
                return ""

        out["sigmoid_action"] = out["sigmoid_modifier"].map(_action)
        out = out.sort_values(["sigmoid_delta", "ad_spend_roll"], ascending=[False, False]).copy()
        if max_rows and int(max_rows) > 0:
            out = out.head(int(max_rows))
        cols = [
            "asin",
            "product_name",
            "product_category",
            "current_phase",
            "inventory_cover_days_7d",
            "ad_spend_roll",
            "sigmoid_modifier",
            "sigmoid_action",
            "reason_1",
            "reason_2",
        ]
        cols = [c for c in cols if c in out.columns]
        return out[cols].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def build_profit_guard_watchlist(
    asin_cockpit: Optional[pd.DataFrame],
    max_rows: int = 200,
    policy: Optional[OpsPolicy] = None,
) -> pd.DataFrame:
    """
    利润护栏（Break-even）：当广告 ACOS/CPC 超过“安全线”时给出提示。
    """
    if asin_cockpit is None or asin_cockpit.empty:
        return pd.DataFrame()
    try:
        pg = getattr(policy, "dashboard_profit_guard", None) if isinstance(policy, OpsPolicy) else None
        if pg is None:
            pg = ProfitGuardPolicy()
        if not bool(getattr(pg, "enabled", True)):
            return pd.DataFrame()

        out = asin_cockpit.copy()
        if "asin" in out.columns:
            out["asin"] = out["asin"].astype(str).str.upper().str.strip()
        if "product_category" in out.columns:
            out["product_category"] = out["product_category"].map(_norm_product_category)

        for c in (
            "gross_margin",
            "ad_acos_recent_7d",
            "ad_acos",
            "ad_spend_roll",
            "sales_recent_7d",
            "aov_recent_7d",
            "ad_cvr_recent_7d",
            "ad_spend_recent_7d",
            "ad_clicks_recent_7d",
        ):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

        min_sales = float(getattr(pg, "min_sales_7d", 50.0) or 50.0)
        min_spend = float(getattr(pg, "min_ad_spend_roll", 10.0) or 10.0)
        target_net = float(getattr(pg, "target_net_margin", 0.05) or 0.05)

        out["safe_acos"] = (out.get("gross_margin", 0.0) - float(target_net)).clip(lower=0.0)
        out["ad_acos_recent"] = out.get("ad_acos_recent_7d", 0.0)
        out.loc[out["ad_acos_recent"] <= 0, "ad_acos_recent"] = out.get("ad_acos", 0.0)

        out["ad_cpc_recent_7d"] = 0.0
        try:
            m = out.get("ad_clicks_recent_7d", 0.0) > 0
            out.loc[m, "ad_cpc_recent_7d"] = (out.loc[m, "ad_spend_recent_7d"] / out.loc[m, "ad_clicks_recent_7d"]).fillna(0.0)
        except Exception:
            pass

        out["safe_cpc"] = 0.0
        try:
            m2 = (out.get("aov_recent_7d", 0.0) > 0) & (out.get("ad_cvr_recent_7d", 0.0) > 0) & (out["safe_acos"] > 0)
            out.loc[m2, "safe_cpc"] = (out.loc[m2, "aov_recent_7d"] * out.loc[m2, "ad_cvr_recent_7d"] * out.loc[m2, "safe_acos"]).fillna(0.0)
        except Exception:
            pass

        out = out[
            (out.get("sales_recent_7d", 0.0) >= float(min_sales))
            & (out.get("ad_spend_roll", 0.0) >= float(min_spend))
            & (out.get("safe_acos", 0.0) > 0)
            & (out.get("ad_acos_recent", 0.0) > out.get("safe_acos", 0.0))
        ].copy()
        if out.empty:
            return out

        def _reasons(row: Dict[str, object]) -> List[str]:
            rs: List[str] = []
            try:
                sa = float(row.get("safe_acos", 0.0) or 0.0)
                aa = float(row.get("ad_acos_recent", 0.0) or 0.0)
            except Exception:
                sa, aa = 0.0, 0.0
            rs.append(f"ACOS>{sa:.2f}")
            rs.append(f"实际ACOS={aa:.2f}")
            rs.append("利润护栏")
            return rs[:3]

        try:
            tags = out.apply(lambda r: _reasons(r.to_dict()), axis=1)
            out["reason_1"] = tags.map(lambda xs: xs[0] if isinstance(xs, list) and len(xs) >= 1 else "")
            out["reason_2"] = tags.map(lambda xs: xs[1] if isinstance(xs, list) and len(xs) >= 2 else "")
            out["reason_3"] = tags.map(lambda xs: xs[2] if isinstance(xs, list) and len(xs) >= 3 else "")
        except Exception:
            for c in ("reason_1", "reason_2", "reason_3"):
                if c not in out.columns:
                    out[c] = ""

        out = out.sort_values(["ad_acos_recent", "ad_spend_roll"], ascending=[False, False]).copy()
        if max_rows and int(max_rows) > 0:
            out = out.head(int(max_rows))
        cols = [
            "asin",
            "product_name",
            "product_category",
            "gross_margin",
            "safe_acos",
            "ad_acos_recent",
            "ad_cpc_recent_7d",
            "safe_cpc",
            "ad_spend_roll",
            "reason_1",
            "reason_2",
        ]
        cols = [c for c in cols if c in out.columns]
        return out[cols].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def build_budget_transfer_plan_with_opportunities(
    budget_transfer_plan: Optional[Dict[str, object]],
    scale_opportunity_watchlist: Optional[pd.DataFrame],
    opportunity_action_board: Optional[pd.DataFrame],
    asin_campaign_map: Optional[pd.DataFrame],
    policy: Optional[OpsPolicy] = None,
) -> Dict[str, object]:
    """
    用“机会池（scale_opportunity_watchlist）”补齐预算迁移表。

    解决的问题：
    - 预算迁移图谱默认依赖 asin_stages 的利润方向（reduce/scale），当本期没有 scale 侧 campaign 时，
      预算只能全部回收/RESERVE；
    - 但机会池可能已经识别出“可放量窗口”（效率/增量/库存满足），运营希望把回收预算迁移到能承接机会的结构上。

    实现策略（保守）：
    - 仅当“原计划没有净迁移（transfers=0）且 cuts 存在”时才启用该联动；
    - 用两条证据生成“承接机会的 Campaign 目标池”：
      1) `asin_campaign_map`：机会 ASIN 在各 Campaign 下的 spend proxy（更可信）
      2) `opportunity_action_board`：机会 ASIN 对应的可执行动作里出现的 Campaign（更贴近落地）
    - 输出 transfer 行会带 `to_confidence/to_opp_*` 证据字段，便于运营人工确认。
    """
    plan0 = budget_transfer_plan if isinstance(budget_transfer_plan, dict) else {}
    # 不破坏原对象（避免上层复用时被意外修改）
    plan: Dict[str, object] = dict(plan0)

    # 开关（默认开启；可在 ops_policy.json 里关闭）
    bto = getattr(policy, "dashboard_budget_transfer_opportunity", None) if isinstance(policy, OpsPolicy) else None
    enabled = bool(getattr(bto, "enabled", True))
    if not enabled:
        return plan

    transfers0 = plan.get("transfers")
    # 如果原本已经有净迁移，不抢占（避免两套逻辑混在一起）
    if isinstance(transfers0, list) and len(transfers0) > 0:
        return plan

    cuts0 = plan.get("cuts")
    cuts = cuts0 if isinstance(cuts0, list) else []
    if not cuts:
        return plan

    wl = scale_opportunity_watchlist.copy() if isinstance(scale_opportunity_watchlist, pd.DataFrame) else pd.DataFrame()
    if wl is None or wl.empty or "asin" not in wl.columns:
        return plan

    # 机会 ASIN 集合
    opp_asins = {str(x or "").strip().upper() for x in wl["asin"].tolist()}
    opp_asins = {x for x in opp_asins if x}
    if not opp_asins:
        return plan

    # 参数
    min_transfer_usd = float(plan.get("min_transfer_usd", 5.0) or 5.0)
    add_pct = float(getattr(bto, "suggested_add_pct", 20.0) or 20.0)
    max_target_campaigns = int(getattr(bto, "max_target_campaigns", 25) or 25)
    min_target_opp_spend = float(getattr(bto, "min_target_opp_spend", 5.0) or 5.0)
    prefer_same_ad_type = bool(getattr(bto, "prefer_same_ad_type", True))

    # 1) 构造 reduce 侧“可回收预算池”
    src: List[Dict[str, object]] = []
    for c in cuts:
        if not isinstance(c, dict):
            continue
        ad_type = str(c.get("ad_type", "") or "").strip()
        campaign = str(c.get("campaign", "") or "").strip()
        if not ad_type or not campaign:
            continue
        amt = float(c.get("cut_usd_estimated", 0.0) or 0.0)
        if amt < min_transfer_usd:
            continue
        src.append(
            {
                "ad_type": ad_type,
                "campaign": campaign,
                "severity": float(c.get("severity", 0.0) or 0.0),
                "camp_spend": float(c.get("camp_spend", 0.0) or 0.0),
                "asin_hint": str(c.get("asin_hint", "") or ""),
                # 工作变量（剩余可迁移金额）
                "_left": float(amt),
            }
        )
    if not src:
        return plan
    # 优先回收更“确定”的（severity 高）
    src.sort(key=lambda x: float(x.get("severity", 0.0) or 0.0), reverse=True)

    # 2) 构造 “承接机会的 Campaign 目标池”
    # 2.1) 从 asin_campaign_map 取 spend proxy（更可信）
    acm = asin_campaign_map.copy() if isinstance(asin_campaign_map, pd.DataFrame) else pd.DataFrame()
    acm_targets = pd.DataFrame()
    if acm is not None and (not acm.empty) and all(c in acm.columns for c in ("ad_type", "campaign", "asin", "spend")):
        try:
            t = acm.copy()
            t["asin_norm"] = t["asin"].astype(str).str.upper().str.strip()
            t["campaign"] = t["campaign"].astype(str).str.strip()
            t["ad_type"] = t["ad_type"].astype(str).str.strip()
            t["spend"] = pd.to_numeric(t["spend"], errors="coerce").fillna(0.0)
            # 机会 ASIN 的 spend 归集到 campaign
            t2 = t[t["asin_norm"].isin(opp_asins) & (t["campaign"] != "")].copy()
            if t2 is not None and not t2.empty:
                acm_targets = (
                    t2.groupby(["ad_type", "campaign"], dropna=False, as_index=False)
                    .agg(
                        to_opp_spend=("spend", "sum"),
                        to_opp_asin_count=("asin_norm", "nunique"),
                    )
                    .copy()
                )
                # top 机会 ASIN（按 spend）
                top_rows = (
                    t2.groupby(["ad_type", "campaign", "asin_norm"], dropna=False, as_index=False)
                    .agg(_asin_spend=("spend", "sum"))
                    .sort_values(["ad_type", "campaign", "_asin_spend"], ascending=[True, True, False])
                    .copy()
                )
                top_asins_map: Dict[Tuple[str, str], str] = {}
                for (ad_type, campaign), g in top_rows.groupby(["ad_type", "campaign"], dropna=False):
                    tops = g.head(3)["asin_norm"].astype(str).tolist()
                    top_asins_map[(str(ad_type), str(campaign))] = ";".join([x for x in tops if x])
                if not acm_targets.empty:
                    acm_targets["to_opp_asins_top"] = acm_targets.apply(
                        lambda r: top_asins_map.get((str(r.get("ad_type", "")), str(r.get("campaign", ""))), ""),
                        axis=1,
                    )
            # campaign 总 spend proxy（用于 share）
            camp_tot = (
                t.groupby(["ad_type", "campaign"], dropna=False, as_index=False)
                .agg(to_spend=("spend", "sum"))
                .copy()
            )
            if acm_targets is not None and not acm_targets.empty:
                acm_targets = acm_targets.merge(camp_tot, on=["ad_type", "campaign"], how="left")
        except Exception:
            acm_targets = pd.DataFrame()

    # 2.2) 从 opportunity_action_board 取“落地侧 Campaign”（更贴近执行）
    oab = opportunity_action_board.copy() if isinstance(opportunity_action_board, pd.DataFrame) else pd.DataFrame()
    oab_targets = pd.DataFrame()
    if oab is not None and (not oab.empty) and "campaign" in oab.columns:
        try:
            t = oab.copy()
            t["campaign"] = t["campaign"].astype(str).str.strip()
            if "ad_type" in t.columns:
                t["ad_type"] = t["ad_type"].astype(str).str.strip()
            else:
                t["ad_type"] = ""
            t = t[t["campaign"] != ""].copy()
            if "e_spend" in t.columns:
                t["e_spend"] = pd.to_numeric(t["e_spend"], errors="coerce").fillna(0.0)
            else:
                t["e_spend"] = 0.0
            oab_targets = (
                t.groupby(["ad_type", "campaign"], dropna=False, as_index=False)
                .agg(
                    to_opp_action_count=("campaign", "size"),
                    _opp_action_spend=("e_spend", "sum"),
                )
                .copy()
            )
            # top 机会 asin_hint（按动作数）
            if "asin_hint" in t.columns:
                try:
                    t["asin_hint_norm"] = t["asin_hint"].astype(str).str.upper().str.strip()
                    g = (
                        t.groupby(["ad_type", "campaign", "asin_hint_norm"], dropna=False, as_index=False)
                        .agg(_cnt=("asin_hint_norm", "size"))
                        .sort_values(["ad_type", "campaign", "_cnt"], ascending=[True, True, False])
                    )
                    top_asin_hint_map: Dict[Tuple[str, str], str] = {}
                    for (ad_type, campaign), gg in g.groupby(["ad_type", "campaign"], dropna=False):
                        tops = gg.head(3)["asin_hint_norm"].astype(str).tolist()
                        top_asin_hint_map[(str(ad_type), str(campaign))] = ";".join([x for x in tops if x])
                    if not oab_targets.empty:
                        oab_targets["to_opp_asins_top"] = oab_targets.apply(
                            lambda r: top_asin_hint_map.get((str(r.get("ad_type", "")), str(r.get("campaign", ""))), ""),
                            axis=1,
                        )
                except Exception:
                    pass
        except Exception:
            oab_targets = pd.DataFrame()

    # 2.3) 合并两类证据
    targets = pd.DataFrame()
    try:
        if acm_targets is not None and not acm_targets.empty:
            targets = acm_targets.copy()
        if oab_targets is not None and not oab_targets.empty:
            if targets is None or targets.empty:
                targets = oab_targets.copy()
                # 没有 asin_campaign_map 时：用动作证据的 spend proxy 兜底
                if "to_opp_spend" not in targets.columns:
                    targets["to_opp_spend"] = targets.get("_opp_action_spend", 0.0)
                if "to_spend" not in targets.columns:
                    targets["to_spend"] = targets.get("_opp_action_spend", 0.0)
                if "to_opp_asin_count" not in targets.columns:
                    targets["to_opp_asin_count"] = 0
            else:
                targets = targets.merge(oab_targets, on=["ad_type", "campaign"], how="outer", suffixes=("", "_oab"))
                # 合并字段（优先 asin_campaign_map 的 spend，缺失则用动作 spend）
                if "to_opp_spend" not in targets.columns and "_opp_action_spend" in targets.columns:
                    targets["to_opp_spend"] = targets["_opp_action_spend"]
                else:
                    try:
                        if "_opp_action_spend" in targets.columns:
                            targets["to_opp_spend"] = pd.to_numeric(targets.get("to_opp_spend", 0.0), errors="coerce").fillna(0.0)
                            targets["_opp_action_spend"] = pd.to_numeric(targets.get("_opp_action_spend", 0.0), errors="coerce").fillna(0.0)
                            targets["to_opp_spend"] = targets["to_opp_spend"].where(targets["to_opp_spend"] > 0, targets["_opp_action_spend"])
                    except Exception:
                        pass
                if "to_spend" not in targets.columns and "_opp_action_spend" in targets.columns:
                    targets["to_spend"] = targets["_opp_action_spend"]
                if "to_opp_asins_top" in targets.columns and "to_opp_asins_top_oab" in targets.columns:
                    # 机会 ASIN 列表：优先 asin_campaign_map 的 top；为空时用 oab 的 top
                    try:
                        targets["to_opp_asins_top"] = targets["to_opp_asins_top"].where(
                            targets["to_opp_asins_top"].astype(str).str.strip() != "",
                            targets["to_opp_asins_top_oab"],
                        )
                    except Exception:
                        pass
        if targets is None:
            targets = pd.DataFrame()
    except Exception:
        targets = pd.DataFrame()

    if targets is None or targets.empty or "campaign" not in targets.columns:
        return plan

    # 规范化/数值化
    try:
        targets["campaign"] = targets["campaign"].astype(str).str.strip()
        targets["ad_type"] = targets.get("ad_type", "").astype(str).str.strip()
        targets = targets[targets["campaign"] != ""].copy()
        targets["to_opp_spend"] = pd.to_numeric(targets.get("to_opp_spend", 0.0), errors="coerce").fillna(0.0)
        targets["to_spend"] = pd.to_numeric(targets.get("to_spend", 0.0), errors="coerce").fillna(0.0)
        targets["to_opp_asin_count"] = pd.to_numeric(targets.get("to_opp_asin_count", 0), errors="coerce").fillna(0).astype(int)
        targets["to_opp_action_count"] = pd.to_numeric(targets.get("to_opp_action_count", 0), errors="coerce").fillna(0).astype(int)
        targets["to_opp_spend_share"] = targets.apply(
            lambda r: 0.0 if float(r.get("to_spend", 0.0) or 0.0) <= 0 else float(r.get("to_opp_spend", 0.0) or 0.0) / float(r.get("to_spend", 0.0) or 1.0),
            axis=1,
        )
    except Exception:
        pass

    # 过滤噪声：opp_spend 太小
    try:
        targets = targets[targets["to_opp_spend"] >= float(min_target_opp_spend)].copy()
    except Exception:
        pass
    if targets is None or targets.empty:
        return plan

    # 置信度：根据 spend_share + 是否存在可执行动作
    def _confidence(row: pd.Series) -> str:
        try:
            share = float(row.get("to_opp_spend_share", 0.0) or 0.0)
            action_cnt = int(row.get("to_opp_action_count", 0) or 0)
            if share >= 0.5 or action_cnt >= 3:
                return "high"
            if share >= 0.25 or action_cnt >= 1:
                return "medium"
            return "low"
        except Exception:
            return "low"

    try:
        targets["to_confidence"] = targets.apply(_confidence, axis=1)
    except Exception:
        targets["to_confidence"] = "low"

    # 如果存在 medium/high，则先排除 low（更保守）；否则 low 也保留
    try:
        has_mid = bool((targets["to_confidence"] != "low").any())
        if has_mid:
            targets = targets[targets["to_confidence"] != "low"].copy()
    except Exception:
        pass

    if targets is None or targets.empty:
        return plan

    # 排序：机会 spend proxy + share + 动作数
    try:
        targets = targets.sort_values(
            ["to_opp_spend", "to_opp_spend_share", "to_opp_action_count"],
            ascending=[False, False, False],
        ).copy()
    except Exception:
        pass
    # 控制目标池规模
    try:
        mt = max(1, int(max_target_campaigns or 0))
        targets = targets.head(mt).copy()
    except Exception:
        pass

    # 生成 dst 列表
    dst: List[Dict[str, object]] = []
    for _, r in targets.iterrows():
        try:
            ad_type = str(r.get("ad_type", "") or "").strip()
            campaign = str(r.get("campaign", "") or "").strip()
            if not campaign:
                continue
            opp_spend = float(r.get("to_opp_spend", 0.0) or 0.0)
            spend_total = float(r.get("to_spend", 0.0) or 0.0)
            share = float(r.get("to_opp_spend_share", 0.0) or 0.0)
            action_cnt = int(r.get("to_opp_action_count", 0) or 0)
            asin_cnt = int(r.get("to_opp_asin_count", 0) or 0)
            conf = str(r.get("to_confidence", "low") or "low")
            tops = str(r.get("to_opp_asins_top", "") or "")

            # 估算“加码需求”（按 opp_spend × pct × confidence factor）
            factor = 1.0 if conf == "high" else (0.7 if conf == "medium" else 0.4)
            need = max(min_transfer_usd, opp_spend * (add_pct / 100.0) * factor)

            # severity（用于排序/匹配）：share 更高 + opp_spend 更大 + 有动作证据
            sev = min(
                100.0,
                (share * 60.0)
                + min(30.0, (opp_spend / 50.0) * 30.0)
                + min(10.0, float(action_cnt) * 2.0),
            )

            dst.append(
                {
                    "ad_type": ad_type,
                    "campaign": campaign,
                    "to_severity": round(float(sev), 2),
                    "to_spend": round(float(spend_total), 2),
                    "to_asin_hint": (tops.split(";")[0] if tops else ""),
                    "to_confidence": conf,
                    "to_opp_asin_count": asin_cnt,
                    "to_opp_asins_top": tops,
                    "to_opp_spend": round(float(opp_spend), 2),
                    "to_opp_spend_share": round(float(share), 4),
                    "to_opp_action_count": action_cnt,
                    # 工作变量（剩余需求）
                    "_need_left": float(need),
                }
            )
        except Exception:
            continue

    if not dst:
        return plan

    # dst 优先级：机会 spend proxy 更大/置信度更高优先
    conf_rank = {"high": 0, "medium": 1, "low": 2}
    dst.sort(
        key=lambda x: (
            conf_rank.get(str(x.get("to_confidence", "low")), 9),
            -float(x.get("to_opp_spend", 0.0) or 0.0),
            -float(x.get("to_opp_spend_share", 0.0) or 0.0),
        )
    )

    # 3) 贪心匹配：reduce -> opportunity scale campaigns
    max_transfers = 200
    transfers: List[Dict[str, object]] = []

    def _greedy_match(src_list: List[Dict[str, object]], dst_list: List[Dict[str, object]]) -> None:
        nonlocal transfers
        i = 0
        j = 0
        while i < len(src_list) and j < len(dst_list) and len(transfers) < max_transfers:
            s = src_list[i]
            d = dst_list[j]
            s_left = float(s.get("_left", 0.0) or 0.0)
            d_need = float(d.get("_need_left", 0.0) or 0.0)
            if s_left <= 0:
                i += 1
                continue
            if d_need <= 0:
                j += 1
                continue
            amt = min(s_left, d_need)
            if amt < min_transfer_usd:
                # 尾部小额：跳过
                if s_left <= d_need:
                    i += 1
                else:
                    j += 1
                continue

            transfers.append(
                {
                    "strategy": "opportunity",
                    "from_ad_type": s.get("ad_type", ""),
                    "from_campaign": s.get("campaign", ""),
                    "from_severity": round(float(s.get("severity", 0.0) or 0.0), 2),
                    "from_spend": round(float(s.get("camp_spend", 0.0) or 0.0), 2),
                    "from_asin_hint": s.get("asin_hint", ""),
                    "to_ad_type": d.get("ad_type", ""),
                    "to_campaign": d.get("campaign", ""),
                    "to_severity": d.get("to_severity", ""),
                    "to_spend": d.get("to_spend", ""),
                    "to_asin_hint": d.get("to_asin_hint", ""),
                    "to_confidence": d.get("to_confidence", ""),
                    "to_opp_asin_count": d.get("to_opp_asin_count", ""),
                    "to_opp_asins_top": d.get("to_opp_asins_top", ""),
                    "to_opp_spend": d.get("to_opp_spend", ""),
                    "to_opp_spend_share": d.get("to_opp_spend_share", ""),
                    "to_opp_action_count": d.get("to_opp_action_count", ""),
                    "amount_usd_estimated": round(float(amt), 2),
                    "note": "机会池联动：把回收预算迁移到能承接机会ASIN的Campaign（无唯一ID；请结合 to_confidence/to_opp_* 人工确认）。",
                }
            )

            s["_left"] = s_left - amt
            d["_need_left"] = d_need - amt
            if float(s.get("_left", 0.0) or 0.0) <= min_transfer_usd / 2:
                i += 1
            if float(d.get("_need_left", 0.0) or 0.0) <= min_transfer_usd / 2:
                j += 1

    if prefer_same_ad_type:
        # 先同类型
        ad_types = sorted({str(x.get("ad_type", "")) for x in src if str(x.get("ad_type", ""))} & {str(x.get("ad_type", "")) for x in dst if str(x.get("ad_type", ""))})
        for at in ad_types:
            _greedy_match([x for x in src if str(x.get("ad_type", "")) == at], [x for x in dst if str(x.get("ad_type", "")) == at])
        # 再跨类型补齐
        _greedy_match(src, dst)
    else:
        _greedy_match(src, dst)

    # 4) 剩余预算：回收/RESERVE
    savings: List[Dict[str, object]] = []
    for s in src:
        left = float(s.get("_left", 0.0) or 0.0)
        if left < min_transfer_usd:
            continue
        savings.append(
            {
                "strategy": "opportunity",
                "from_ad_type": s.get("ad_type", ""),
                "from_campaign": s.get("campaign", ""),
                "from_asin_hint": s.get("asin_hint", ""),
                "to_bucket": "RESERVE",
                "amount_usd_estimated": round(float(left), 2),
                "note": "机会池联动后仍有剩余预算：建议暂时回收或人工分配到更确定的承接结构。",
            }
        )

    plan["transfers"] = transfers
    plan["savings"] = savings
    # 方便 dashboard 第一屏统计
    plan["scale_candidates"] = int(len(dst))
    plan["reduce_candidates"] = int(len(src))
    plan["unallocated_reduce_usd_estimated"] = round(float(sum(float(s.get("_left", 0.0) or 0.0) for s in src)), 2)
    plan["unmet_scale_usd_estimated"] = round(float(sum(float(d.get("_need_left", 0.0) or 0.0) for d in dst)), 2)
    return plan


def build_unlock_scale_tasks_table(
    unlock_tasks: object,
    asin_cockpit: Optional[pd.DataFrame] = None,
    max_rows: int = 500,
) -> pd.DataFrame:
    """
    放量解锁任务表（unlock_scale_tasks.csv）。

    来源：
    - diagnostics["unlock_tasks"]（由 build_unlock_tasks 生成，尽量给出“需要修到什么程度”的目标）

    增强：
    - 如果提供 asin_cockpit，则补齐 product_category/current_phase/库存覆盖等字段，便于筛选分派。
    """
    tasks = unlock_tasks if isinstance(unlock_tasks, list) else []
    df = pd.DataFrame(tasks).copy()

    base_cols = [
        "priority",
        "owner",
        "task_type",
        "playbook_scene",
        "playbook_url",
        "product_category",
        "asin",
        "product_name",
        "current_phase",
        "cycle_id",
        "inventory",
        "inventory_cover_days_7d",
        "inventory_cover_days_30d",
        "sales_per_day_7d",
        "budget_gap_usd_est",
        "profit_gap_usd_est",
        "need",
        "target",
        "stage",
        "direction",
        "evidence",
    ]
    if df is None or df.empty or "asin" not in df.columns:
        return pd.DataFrame(columns=base_cols)

    pb_doc = "../../../../docs/OPS_PLAYBOOK.md"

    def _task_scene(task_type: str) -> str:
        t = str(task_type or "")
        if ("库存" in t) or ("补货" in t):
            return "inventory-first"
        if ("预算" in t) or ("出价" in t) or ("利润" in t):
            return "scene-profit-reduce"
        if ("转化" in t) or ("Listing" in t) or ("评分" in t) or ("退款" in t):
            return "scene-phase-down"
        return "scene-scale-opportunity"

    df["asin_norm"] = df["asin"].astype(str).str.upper().str.strip()

    # 运营筛选维度补齐（可选）
    ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
    if ac is not None and not ac.empty and "asin" in ac.columns:
        try:
            ac2 = ac.copy()
            ac2["asin_norm"] = ac2["asin"].astype(str).str.upper().str.strip()
            enrich_cols = [
                "product_category",
                "product_name",
                "current_phase",
                "cycle_id",
                "inventory",
                "inventory_cover_days_7d",
                "inventory_cover_days_30d",
                "sales_per_day_7d",
            ]
            keep = ["asin_norm"] + [c for c in enrich_cols if c in ac2.columns]
            ac2 = ac2[keep].drop_duplicates("asin_norm", keep="first")
            df = df.merge(ac2, on="asin_norm", how="left", suffixes=("", "_asin"))
        except Exception:
            pass

    # 如果 task 自带维度为空，则用 asin_cockpit 补齐（保持“未分类兜底”逻辑不在这里硬编码）
    for c in ("product_category", "product_name", "current_phase", "cycle_id", "inventory", "inventory_cover_days_7d", "inventory_cover_days_30d", "sales_per_day_7d"):
        if c not in df.columns and f"{c}_asin" in df.columns:
            df[c] = df[f"{c}_asin"]
        else:
            # 优先 task 本身字段，为空时再兜底 asin 维度
            try:
                if c in df.columns and f"{c}_asin" in df.columns:
                    df[c] = df[c].where(df[c].notna() & (df[c].astype(str).str.strip() != ""), df[f"{c}_asin"])
            except Exception:
                pass

    df = df.drop(columns=[c for c in df.columns if c.endswith("_asin")] + ["asin_norm"], errors="ignore")

    for c in ("inventory", "inventory_cover_days_7d", "inventory_cover_days_30d", "sales_per_day_7d", "budget_gap_usd_est", "profit_gap_usd_est"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 排序：优先级 → 缺口（预算/利润）
    pr_rank = {"P0": 0, "P1": 1, "P2": 2}
    df["priority_rank"] = df.get("priority", "").astype(str).map(lambda x: pr_rank.get(str(x).strip().upper(), 9))
    for c in ("budget_gap_usd_est", "profit_gap_usd_est"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    try:
        df = df.sort_values(["priority_rank", "budget_gap_usd_est", "profit_gap_usd_est"], ascending=[True, False, False]).copy()
    except Exception:
        pass

    try:
        mr = max(1, int(max_rows or 0))
        df = df.head(mr).copy()
    except Exception:
        pass

    # 操作手册联动（不改口径）
    try:
        df["playbook_scene"] = df.get("task_type", "").map(_task_scene)
        df["playbook_url"] = df["playbook_scene"].map(lambda s: f"{pb_doc}#{s}" if str(s or "").strip() else "")
    except Exception:
        pass

    cols = [c for c in base_cols if c in df.columns]
    out = df[cols].copy()
    return out.reset_index(drop=True)


def filter_unlock_scale_tasks_for_ops(
    unlock_scale_tasks_full: Optional[pd.DataFrame],
    policy: Optional[OpsPolicy] = None,
) -> pd.DataFrame:
    """
    把“放量解锁任务”收敛成运营可执行的 Top 列表（默认只保留 P0/P1）。

    说明：
    - 全量表仍会落盘为 `unlock_scale_tasks_full.csv`（便于追溯/深挖）
    - Top 表落盘为 `unlock_scale_tasks.csv`（便于分派/执行/看板化）
    """
    df = unlock_scale_tasks_full.copy() if isinstance(unlock_scale_tasks_full, pd.DataFrame) else pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) if isinstance(df, pd.DataFrame) else [])

    ut = getattr(policy, "dashboard_unlock_tasks", None) if isinstance(policy, OpsPolicy) else None
    top_n = 30
    include_priorities = ["P0", "P1"]
    try:
        if ut is not None:
            top_n = int(getattr(ut, "top_n", top_n) or top_n)
            include_priorities = list(getattr(ut, "include_priorities", include_priorities) or include_priorities)
    except Exception:
        top_n = 30
        include_priorities = ["P0", "P1"]

    # 过滤优先级（默认只保留 P0/P1）
    if include_priorities and "priority" in df.columns:
        try:
            keep = {str(x or "").strip().upper() for x in include_priorities}
            keep = {x for x in keep if x}
            df["priority"] = df["priority"].astype(str).str.strip().str.upper()
            if keep:
                df = df[df["priority"].isin(keep)].copy()
        except Exception:
            pass

    if df is None or df.empty:
        return pd.DataFrame(columns=list(unlock_scale_tasks_full.columns) if isinstance(unlock_scale_tasks_full, pd.DataFrame) else [])

    # TopN：保持 build_unlock_scale_tasks_table 的排序（P0→P1→P2，再按缺口金额）
    try:
        n = max(1, int(top_n or 0))
        df = df.head(n).copy()
    except Exception:
        pass
    return df.reset_index(drop=True)


def build_phase_cockpit(
    asin_focus_all: Optional[pd.DataFrame],
    action_board_dedup_all: Optional[pd.DataFrame],
    policy: Optional[OpsPolicy] = None,
    inventory_risk_spend_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Phase Cockpit：把“动态生命周期”的关键维度汇总到每个阶段一行。

    汇总维度（当前版本）：
    - 重点程度：focus_score_sum / focus_score_mean（来自 asin_focus_all）
    - 变化趋势：delta_sales_sum / delta_spend_sum（来自 asin_focus_all 的 compare_7d 派生列）
    - 库存风险：low_inventory/oos/oos_with_ad_spend/库存告急仍投放 等计数（来自 asin_focus_all）
    - 动作量：P0/P1 动作数、阻断数（来自 action_board_dedup_all）

    设计原则：
    - 不引入新数据源/唯一ID
    - 防御性：缺列/空表都不崩，输出尽量完整
    """
    # 1) 基础：asin_focus_all（每个 asin 一行）
    base = asin_focus_all.copy() if isinstance(asin_focus_all, pd.DataFrame) else pd.DataFrame()
    if base is None:
        base = pd.DataFrame()

    if base.empty:
        base = pd.DataFrame(columns=["asin", "current_phase", "product_category"])

    base = base.copy()
    if "asin" not in base.columns:
        base["asin"] = ""
    base["_asin"] = base["asin"].astype(str).str.upper().str.strip()
    base = base[(base["_asin"] != "") & (base["_asin"].str.lower() != "nan")].copy()
    # 防御性去重：每个 asin 只保留一行
    base = base.drop_duplicates("_asin", keep="first").copy()

    if "current_phase" not in base.columns:
        base["current_phase"] = "unknown"
    base["current_phase"] = base["current_phase"].map(_norm_phase)

    if "product_category" not in base.columns:
        base["product_category"] = "（未分类）"
    base["product_category"] = base["product_category"].map(_norm_product_category)

    # 数值列兜底
    for c in (
        "focus_score",
        "ad_spend_roll",
        "delta_sales",
        "delta_spend",
        "oos_with_ad_spend_days",
        "ad_sales_share",
        "inventory_cover_days_7d",
    ):
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)

    # 2) phase 聚合（来自 asin_focus_all）
    phase_stats = pd.DataFrame()
    try:
        g = base.groupby("current_phase", dropna=False)
        phase_stats = g.agg(
            asin_count=("_asin", "nunique"),
            category_count=("product_category", "nunique"),
        ).reset_index()

        # focus
        if "focus_score" in base.columns:
            fs = (
                base.groupby("current_phase", dropna=False, as_index=False)
                .agg(
                    focus_score_sum=("focus_score", "sum"),
                    focus_score_mean=("focus_score", "mean"),
                )
                .copy()
            )
            phase_stats = phase_stats.merge(fs, on="current_phase", how="left")
        else:
            phase_stats["focus_score_sum"] = 0.0
            phase_stats["focus_score_mean"] = 0.0

        # delta（7d vs prev7d）：存在则汇总，否则置 0
        if "delta_sales" in base.columns:
            ds = (
                base.groupby("current_phase", dropna=False, as_index=False)
                .agg(delta_sales_sum=("delta_sales", "sum"))
                .copy()
            )
            phase_stats = phase_stats.merge(ds, on="current_phase", how="left")
        else:
            phase_stats["delta_sales_sum"] = 0.0
        if "delta_spend" in base.columns:
            dd = (
                base.groupby("current_phase", dropna=False, as_index=False)
                .agg(delta_spend_sum=("delta_spend", "sum"))
                .copy()
            )
            phase_stats = phase_stats.merge(dd, on="current_phase", how="left")
        else:
            phase_stats["delta_spend_sum"] = 0.0

        # 风险计数
        if "flag_low_inventory" in base.columns:
            tmp = base.copy()
            tmp["_flag"] = (pd.to_numeric(tmp.get("flag_low_inventory", 0), errors="coerce").fillna(0).astype(int) > 0).astype(int)
            t2 = tmp.groupby("current_phase", dropna=False, as_index=False).agg(low_inventory_asin_count=("_flag", "sum"))
            phase_stats = phase_stats.merge(t2, on="current_phase", how="left")
        else:
            phase_stats["low_inventory_asin_count"] = 0

        if "flag_oos" in base.columns:
            tmp = base.copy()
            tmp["_flag"] = (pd.to_numeric(tmp.get("flag_oos", 0), errors="coerce").fillna(0).astype(int) > 0).astype(int)
            t2 = tmp.groupby("current_phase", dropna=False, as_index=False).agg(oos_asin_count=("_flag", "sum"))
            phase_stats = phase_stats.merge(t2, on="current_phase", how="left")
        else:
            phase_stats["oos_asin_count"] = 0

        if "oos_with_ad_spend_days" in base.columns:
            tmp = base.copy()
            tmp["_flag"] = (pd.to_numeric(tmp.get("oos_with_ad_spend_days", 0.0), errors="coerce").fillna(0.0) > 0).astype(int)
            t2 = tmp.groupby("current_phase", dropna=False, as_index=False).agg(oos_with_ad_spend_asin_count=("_flag", "sum"))
            phase_stats = phase_stats.merge(t2, on="current_phase", how="left")
        else:
            phase_stats["oos_with_ad_spend_asin_count"] = 0

        # 库存告急仍投放：覆盖天数低且仍在花费（提前预警，不等断货）
        try:
            cover_thr = 7.0
            if isinstance(policy, OpsPolicy):
                cover_thr = float(getattr(policy, "block_scale_when_cover_days_below", cover_thr) or cover_thr)
            spend_thr = float(inventory_risk_spend_threshold or 10.0)
            if (cover_thr > 0) and ("inventory_cover_days_7d" in base.columns):
                tmp = base.copy()
                tmp["_cover7"] = pd.to_numeric(tmp.get("inventory_cover_days_7d", 0.0), errors="coerce").fillna(0.0)
                tmp["_spend"] = pd.to_numeric(tmp.get("ad_spend_roll", 0.0), errors="coerce").fillna(0.0)
                tmp["_flag"] = ((tmp["_cover7"] > 0) & (tmp["_cover7"] <= cover_thr) & (tmp["_spend"] >= spend_thr)).astype(int)
                t2 = tmp.groupby("current_phase", dropna=False, as_index=False).agg(inventory_risk_asin_count=("_flag", "sum"))
                phase_stats = phase_stats.merge(t2, on="current_phase", how="left")
            else:
                phase_stats["inventory_risk_asin_count"] = 0
        except Exception:
            phase_stats["inventory_risk_asin_count"] = 0

        # spend_up_no_sales：Δspend>0 且 Δsales<=0
        if "delta_spend" in base.columns and "delta_sales" in base.columns:
            tmp = base.copy()
            tmp["_flag"] = ((tmp["delta_spend"] > 0) & (tmp["delta_sales"] <= 0)).astype(int)
            t2 = tmp.groupby("current_phase", dropna=False, as_index=False).agg(spend_up_no_sales_asin_count=("_flag", "sum"))
            phase_stats = phase_stats.merge(t2, on="current_phase", how="left")
        else:
            phase_stats["spend_up_no_sales_asin_count"] = 0

        # high_ad_dependency：ad_sales_share >= 0.8（如列存在）
        if "ad_sales_share" in base.columns:
            tmp = base.copy()
            tmp["_flag"] = (pd.to_numeric(tmp.get("ad_sales_share", 0.0), errors="coerce").fillna(0.0) >= 0.8).astype(int)
            t2 = tmp.groupby("current_phase", dropna=False, as_index=False).agg(high_ad_dependency_asin_count=("_flag", "sum"))
            phase_stats = phase_stats.merge(t2, on="current_phase", how="left")
        else:
            phase_stats["high_ad_dependency_asin_count"] = 0
    except Exception:
        phase_stats = pd.DataFrame(columns=["current_phase"])

    # 3) 动作统计（按 phase）
    action_stats = pd.DataFrame()
    try:
        ab = action_board_dedup_all.copy() if isinstance(action_board_dedup_all, pd.DataFrame) else pd.DataFrame()
        if ab is not None and not ab.empty:
            ab2 = ab.copy()
            ab2["current_phase"] = ab2.get("current_phase", "unknown").map(_norm_phase)
            if "priority" not in ab2.columns:
                ab2["priority"] = ""
            ab2["priority"] = ab2["priority"].astype(str).str.upper().str.strip()
            if "blocked" not in ab2.columns:
                ab2["blocked"] = 0
            ab2["blocked"] = pd.to_numeric(ab2["blocked"], errors="coerce").fillna(0).astype(int)

            ab2["_p0"] = (ab2["priority"] == "P0").astype(int)
            ab2["_p1"] = (ab2["priority"] == "P1").astype(int)
            ab2["_p2"] = (ab2["priority"] == "P2").astype(int)
            ab2["_is_top"] = ab2["priority"].isin(["P0", "P1"]).astype(int)
            ab2["_is_top_blocked"] = ((ab2["_is_top"] == 1) & (ab2["blocked"] == 1)).astype(int)

            action_stats = (
                ab2.groupby("current_phase", dropna=False, as_index=False)
                .agg(
                    phase_action_count=("current_phase", "size"),
                    phase_p0_action_count=("_p0", "sum"),
                    phase_p1_action_count=("_p1", "sum"),
                    phase_p2_action_count=("_p2", "sum"),
                    phase_top_action_count=("_is_top", "sum"),
                    phase_top_blocked_action_count=("_is_top_blocked", "sum"),
                    phase_blocked_action_count=("blocked", "sum"),
                )
                .copy()
            )
    except Exception:
        action_stats = pd.DataFrame()

    # 4) union phases
    phases = set()
    try:
        if not phase_stats.empty and "current_phase" in phase_stats.columns:
            phases |= set(phase_stats["current_phase"].astype(str).tolist())
        if not action_stats.empty and "current_phase" in action_stats.columns:
            phases |= set(action_stats["current_phase"].astype(str).tolist())
    except Exception:
        phases = set()
    phases = {str(p or "").strip() for p in phases}
    phases = {p for p in phases if p}
    if not phases:
        return pd.DataFrame()

    out = pd.DataFrame({"current_phase": sorted(list(phases))})
    if not phase_stats.empty:
        out = out.merge(phase_stats, on="current_phase", how="left")
    if not action_stats.empty:
        out = out.merge(action_stats, on="current_phase", how="left")

    # 兜底数值列
    for c in (
        "asin_count",
        "category_count",
        "focus_score_sum",
        "focus_score_mean",
        "delta_sales_sum",
        "delta_spend_sum",
        "low_inventory_asin_count",
        "oos_asin_count",
        "oos_with_ad_spend_asin_count",
        "inventory_risk_asin_count",
        "spend_up_no_sales_asin_count",
        "high_ad_dependency_asin_count",
        "phase_action_count",
        "phase_p0_action_count",
        "phase_p1_action_count",
        "phase_p2_action_count",
        "phase_top_action_count",
        "phase_top_blocked_action_count",
        "phase_blocked_action_count",
    ):
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # 格式化
    for c in ("focus_score_sum", "focus_score_mean", "delta_sales_sum", "delta_spend_sum"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(2)
    for c in (
        "asin_count",
        "category_count",
        "low_inventory_asin_count",
        "oos_asin_count",
        "oos_with_ad_spend_asin_count",
        "inventory_risk_asin_count",
        "spend_up_no_sales_asin_count",
        "high_ad_dependency_asin_count",
        "phase_action_count",
        "phase_p0_action_count",
        "phase_p1_action_count",
        "phase_p2_action_count",
        "phase_top_action_count",
        "phase_top_blocked_action_count",
        "phase_blocked_action_count",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # 排序：按生命周期顺序（更贴近“动态生命周期”的阅读顺序）
    order = ["pre_launch", "launch", "growth", "mature", "stable", "decline", "inactive", "unknown"]
    order_map = {p: i for i, p in enumerate(order)}
    out["_order"] = out["current_phase"].map(lambda x: order_map.get(str(x or "").strip().lower(), 999))
    out = out.sort_values(["_order", "focus_score_sum"], ascending=[True, False]).drop(columns=["_order"], errors="ignore")

    return out.reset_index(drop=True)


def build_shop_alerts(
    scorecard: Dict[str, object],
    phase_cockpit: Optional[pd.DataFrame],
    category_cockpit: Optional[pd.DataFrame],
    asin_cockpit: Optional[pd.DataFrame],
    max_items: int = 5,
    policy: Optional[OpsPolicy] = None,
) -> List[Dict[str, str]]:
    """
    Shop Alerts：规则化告警 Top N（不依赖 AI 生成结论）。

    输出结构（用于 dashboard.md 渲染）：
    - priority: P0/P1
    - title: 简短标题
    - detail: 证据/数字（字符串）
    - link: Markdown 链接（字符串，可为空）
    """
    alerts: List[Dict[str, str]] = []
    category_blocked_added = False
    PB_DOC = "../../../../docs/OPS_PLAYBOOK.md"
    asin_name_map: Dict[str, str] = {}
    try:
        ac_map = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
        if ac_map is not None and not ac_map.empty and "asin" in ac_map.columns and "product_name" in ac_map.columns:
            ac_map = ac_map.copy()
            ac_map["asin"] = ac_map["asin"].astype(str).str.upper().str.strip()
            ac_map["product_name"] = ac_map["product_name"].astype(str).str.strip()
            ac_map = ac_map[ac_map["asin"] != ""].drop_duplicates(subset=["asin"], keep="first")
            for _, r in ac_map.iterrows():
                asin = str(r.get("asin", "") or "").strip().upper()
                name = str(r.get("product_name", "") or "").strip()
                if asin and name and name.lower() != "nan" and asin not in asin_name_map:
                    asin_name_map[asin] = name
    except Exception:
        asin_name_map = {}

    def _asin_label(asin: str) -> str:
        a = str(asin or "").strip().upper()
        if not a:
            return ""
        name = str(asin_name_map.get(a, "") or "").strip()
        if name:
            return f"{name}({a})"
        return a

    def _with_playbook(link: str, anchor: str) -> str:
        """
        给告警补一个“操作手册”入口（稳定锚点），让运营点进去就知道怎么查/怎么做。
        """
        try:
            a = str(anchor or "").strip()
            if not a:
                return str(link or "").strip()
            pb = f"[操作手册]({PB_DOC}#{a})"
            l = str(link or "").strip()
            return pb if not l else f"{l} | {pb}"
        except Exception:
            return str(link or "").strip()

    def _add(priority: str, title: str, detail: str, link: str) -> None:
        try:
            p = str(priority or "").strip().upper() or "P1"
            alerts.append(
                {
                    "priority": p,
                    "title": str(title or "").strip(),
                    "detail": str(detail or "").strip(),
                    "link": str(link or "").strip(),
                }
            )
        except Exception:
            return

    # ===== 规则 0：店铺级“花费上升但销量下降”（近7天 vs 前7天） =====
    try:
        c7 = None
        compares = scorecard.get("compares") if isinstance(scorecard, dict) else None
        rows = compares if isinstance(compares, list) else []
        for x in rows:
            if isinstance(x, dict) and int(x.get("window_days", 0) or 0) == 7:
                c7 = x
                break
        if isinstance(c7, dict):
            ds = _safe_float(c7.get("delta_sales", 0.0))
            dsp = _safe_float(c7.get("delta_ad_spend", 0.0))
            mt = _safe_float(c7.get("marginal_tacos", 0.0))
            if ds < 0 and dsp > 0:
                _add(
                    "P0",
                    "近7天销量下降但广告花费上升",
                    f"ΔSales={ds:.2f}，ΔAdSpend={dsp:.2f}，marginal_tacos={mt:.4f}",
                    _with_playbook("[查看变化来源](#drivers)", "scene-spend-up-no-sales"),
                )
    except Exception:
        pass

    # ===== Phase 维度：优先找“库存告急仍投放 / 加花费但销量不增 / 广告依赖高 / 未归类动作过多” =====
    pc = phase_cockpit.copy() if isinstance(phase_cockpit, pd.DataFrame) else pd.DataFrame()
    if pc is None:
        pc = pd.DataFrame()
    if not pc.empty and "current_phase" in pc.columns:
        pc = pc.copy()
        pc["current_phase"] = pc["current_phase"].map(_norm_phase)
        for c in (
            "oos_with_ad_spend_asin_count",
            "inventory_risk_asin_count",
            "spend_up_no_sales_asin_count",
            "high_ad_dependency_asin_count",
            "phase_top_action_count",
            "phase_top_blocked_action_count",
            "asin_count",
            "focus_score_sum",
            "delta_sales_sum",
            "delta_spend_sum",
        ):
            if c not in pc.columns:
                pc[c] = 0
            pc[c] = pd.to_numeric(pc[c], errors="coerce").fillna(0)

        # 1) 库存告急仍投放（优先级最高）
        try:
            sub = pc[pc["inventory_risk_asin_count"] > 0].copy()
            if not sub.empty:
                sub = sub.sort_values(["inventory_risk_asin_count", "focus_score_sum"], ascending=[False, False])
                r = sub.iloc[0].to_dict()
                ph = str(r.get("current_phase", "") or "unknown")
                cnt = int(_safe_int(r.get("inventory_risk_asin_count", 0)))
                _add(
                    "P0",
                    f"库存告急仍投放（phase={ph}）",
                    f"{cnt} 个 ASIN；top_actions={_safe_int(r.get('phase_top_action_count', 0))}，blocked={_safe_int(r.get('phase_top_blocked_action_count', 0))}",
                    _with_playbook(
                        f"[筛选 Watchlist](../dashboard/inventory_risk_watchlist.csv) | [查看该阶段](./phase_drilldown.md#{_phase_anchor_id(ph)})",
                        "inventory-first",
                    ),
                )
        except Exception:
            pass

        # 2) 加花费但销量不增（Δspend>0 且 Δsales<=0）
        try:
            sub = pc[pc["spend_up_no_sales_asin_count"] > 0].copy()
            if not sub.empty:
                sub = sub.sort_values(["spend_up_no_sales_asin_count", "focus_score_sum"], ascending=[False, False])
                r = sub.iloc[0].to_dict()
                ph = str(r.get("current_phase", "") or "unknown")
                cnt = int(_safe_int(r.get("spend_up_no_sales_asin_count", 0)))
                ds = _safe_float(r.get("delta_sales_sum", 0.0))
                dsp = _safe_float(r.get("delta_spend_sum", 0.0))
                _add(
                    "P0",
                    f"加花费但销量不增（phase={ph}）",
                    f"{cnt} 个 ASIN；phase_ΔSales={ds:.2f}，phase_ΔAdSpend={dsp:.2f}",
                    _with_playbook(
                        f"[筛选 Watchlist](../dashboard/spend_up_no_sales_watchlist.csv) | [查看该阶段](./phase_drilldown.md#{_phase_anchor_id(ph)})",
                        "scene-spend-up-no-sales",
                    ),
                )
        except Exception:
            pass

        # 3) 广告依赖高（ad_sales_share>=0.8 的 ASIN 数）
        try:
            sub = pc[pc["high_ad_dependency_asin_count"] > 0].copy()
            if not sub.empty:
                sub = sub.sort_values(["high_ad_dependency_asin_count", "focus_score_sum"], ascending=[False, False])
                r = sub.iloc[0].to_dict()
                ph = str(r.get("current_phase", "") or "unknown")
                cnt = int(_safe_int(r.get("high_ad_dependency_asin_count", 0)))
                _add(
                    "P2",
                    f"广告依赖高（phase={ph}）",
                    f"{cnt} 个 ASIN（广告销售占比≥0.8）",
                    f"[查看该阶段](./phase_drilldown.md#{_phase_anchor_id(ph)})",
                )
        except Exception:
            pass

        # 4) 未归类动作过多（unknown phase 的 Top 动作量很大）
        try:
            sub = pc[pc["current_phase"] == "unknown"].copy()
            if not sub.empty:
                r = sub.iloc[0].to_dict()
                top_actions = int(_safe_int(r.get("phase_top_action_count", 0)))
                asin_cnt = int(_safe_int(r.get("asin_count", 0)))
                # 阈值：避免没必要的噪音（只在 Top 动作>=20 时提示）
                if top_actions >= 20 and asin_cnt <= 0:
                    _add(
                        "P2",
                        "存在未能映射到生命周期/ASIN 的 Top 动作",
                        f"Top 动作={top_actions}（可能为弱关联/缺少产品信息，建议优先看 Action Board 的 asin_hint_confidence）",
                        f"[查看 unknown](./phase_drilldown.md#{_phase_anchor_id('unknown')})",
                    )
        except Exception:
            pass

    # ===== 类目维度：Top 动作阻断集中在哪个类目 =====
    cc = category_cockpit.copy() if isinstance(category_cockpit, pd.DataFrame) else pd.DataFrame()
    if cc is None:
        cc = pd.DataFrame()
    try:
        if not cc.empty and "product_category" in cc.columns:
            cc = cc.copy()
            cc["product_category"] = cc["product_category"].map(_norm_product_category)
            if "category_top_blocked_action_count" not in cc.columns:
                cc["category_top_blocked_action_count"] = 0
            cc["category_top_blocked_action_count"] = pd.to_numeric(cc["category_top_blocked_action_count"], errors="coerce").fillna(0).astype(int)
            sub = cc[cc["category_top_blocked_action_count"] > 0].copy()
            if not sub.empty:
                sub = sub.sort_values(["category_top_blocked_action_count", "focus_score_sum"], ascending=[False, False])
                r = sub.iloc[0].to_dict()
                cat = str(r.get("product_category", "") or "（未分类）")
                cnt = int(_safe_int(r.get("category_top_blocked_action_count", 0)))
                _add(
                    "P1",
                    f"Top 动作阻断集中在类目：{cat}",
                    f"blocked_top_actions={cnt}",
                    _with_playbook(f"[查看该类目](./category_drilldown.md#{_cat_anchor_id(cat)})", "inventory-first"),
                )
                category_blocked_added = True
    except Exception:
        pass

    # ===== ASIN 维度：Top 动作阻断最多的 ASIN =====
    ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
    if ac is None:
        ac = pd.DataFrame()

    # ===== 生命周期迁移：近14天阶段走弱且仍在花费（第二入口 Watchlist）=====
    # 说明：这是“早提醒”，不判定原因；目的是把需要优先排查的 ASIN 从海量数据里捞出来。
    try:
        # 阈值：可从 ops_policy.json 读取；缺失则使用默认值（保持与历史硬编码一致）
        pdr_cfg = None
        if policy is not None:
            try:
                pdr_cfg = getattr(getattr(policy, "dashboard_shop_alerts", None), "phase_down_recent", None)
            except Exception:
                pdr_cfg = None
        pdr_enabled = bool(getattr(pdr_cfg, "enabled", True)) if pdr_cfg is not None else True
        p0_spend_sum = float(getattr(pdr_cfg, "p0_spend_sum", 200.0) or 200.0) if pdr_cfg is not None else 200.0
        p0_spend_share = float(getattr(pdr_cfg, "p0_spend_share", 0.25) or 0.25) if pdr_cfg is not None else 0.25
        p0_spend_sum_min_when_share = (
            float(getattr(pdr_cfg, "p0_spend_sum_min_when_share", 50.0) or 50.0) if pdr_cfg is not None else 50.0
        )
        p0_asin_count_min = int(getattr(pdr_cfg, "p0_asin_count_min", 5) or 5) if pdr_cfg is not None else 5

        if pdr_enabled and (not ac.empty) and ("asin" in ac.columns):
            ac2 = ac.copy()
            ac2["asin"] = ac2["asin"].astype(str).str.upper().str.strip()
            # 防御性补齐列（赛狐模板变动时不崩）
            for c in ("phase_changed_recent_14d", "phase_trend_14d", "ad_spend_roll", "phase_change", "focus_score", "current_phase"):
                if c not in ac2.columns:
                    ac2[c] = 0
            ac2["phase_changed_recent_14d"] = pd.to_numeric(ac2.get("phase_changed_recent_14d", 0), errors="coerce").fillna(0).astype(int)
            ac2["phase_trend_14d"] = ac2.get("phase_trend_14d", "").astype(str).fillna("").str.strip().str.lower()
            ac2["ad_spend_roll"] = pd.to_numeric(ac2.get("ad_spend_roll", 0.0), errors="coerce").fillna(0.0)
            if "focus_score" in ac2.columns:
                ac2["focus_score"] = pd.to_numeric(ac2.get("focus_score", 0.0), errors="coerce").fillna(0.0)
            if "current_phase" in ac2.columns:
                ac2["current_phase"] = ac2.get("current_phase", "").map(_norm_phase)

            sub = ac2[(ac2["phase_changed_recent_14d"] > 0) & (ac2["phase_trend_14d"] == "down") & (ac2["ad_spend_roll"] > 0)].copy()
            if not sub.empty:
                cnt = int(len(sub))
                spend_sum = float(sub["ad_spend_roll"].sum() or 0.0)
                total_spend = float(ac2["ad_spend_roll"].sum() or 0.0)
                spend_share = (spend_sum / total_spend) if total_spend > 0 else 0.0

                # Top 原因标签（Top 3）：用于第一屏“更可解释”，不增加下钻复杂度
                top_reasons: List[str] = []
                try:
                    sub_r = _annotate_phase_down_reasons(sub, policy=policy)
                    vals: List[str] = []
                    for c in ("reason_1", "reason_2", "reason_3"):
                        if c in sub_r.columns:
                            vals += [str(x or "").strip() for x in sub_r[c].tolist()]
                    # 优先展示“可行动”的原因，避免被“原因待确认”占满
                    vals2 = [v for v in vals if v and v != "原因待确认"]
                    if not vals2:
                        vals2 = [v for v in vals if v]
                    if vals2:
                        vc = pd.Series(vals2).value_counts()
                        top_reasons = [str(x or "").strip() for x in vc.head(3).index.tolist() if str(x or "").strip()]
                except Exception:
                    top_reasons = []

                sub = sub.sort_values(["ad_spend_roll", "focus_score"], ascending=[False, False]).copy()
                top = sub.iloc[0].to_dict()
                top_asin = str(top.get("asin", "") or "").strip().upper()
                top_change = str(top.get("phase_change", "") or "").strip()
                top_phase = str(top.get("current_phase", "") or "").strip().lower() or "unknown"

                # 优先级：保守默认 P1；规模/占比很高时再升到 P0，避免把“断货仍烧钱”等更硬风险顶掉。
                priority = "P1"
                if (spend_sum >= p0_spend_sum) or (
                    (spend_share >= p0_spend_share) and (spend_sum >= p0_spend_sum_min_when_share) and (cnt >= p0_asin_count_min)
                ):
                    priority = "P0"

                # detail 里给“规模 + top 例子”，让运营一眼知道为什么会出现这个提示
                top_hint = ""
                if top_asin:
                    top_label = _asin_label(top_asin)
                    if top_change:
                        top_hint = f"top={top_label}({top_change})"
                    else:
                        top_hint = f"top={top_label}(phase={top_phase})"
                detail = f"asin={cnt}，spend_roll_sum={spend_sum:.2f}（占比={spend_share:.1%}）"
                if top_hint:
                    detail = f"{detail}；{top_hint}"
                if top_reasons:
                    detail = f"{detail}；top_reasons={'/'.join(top_reasons)}"

                _add(
                    priority,
                    "近14天阶段走弱且仍在花费（优先排查）",
                    detail,
                    _with_playbook(
                        "[筛选 Watchlist](../dashboard/phase_down_recent_watchlist.csv) | [筛选同类对比表](../dashboard/category_asin_compare.csv)",
                        "scene-phase-down",
                    ),
                )
    except Exception:
        pass

    # ===== 产品侧转化异常：近7天流量上升但转化下滑（优先排查 listing/价格/评价/库存等） =====
    try:
        if not ac.empty and "asin" in ac.columns:
            ac2 = ac.copy()
            ac2["asin"] = ac2["asin"].astype(str).str.upper().str.strip()
            for c in (
                "sessions_prev_7d",
                "sessions_recent_7d",
                "delta_sessions",
                "cvr_prev_7d",
                "cvr_recent_7d",
                "delta_cvr",
                "ad_spend_roll",
                "focus_score",
                "current_phase",
                "product_category",
            ):
                if c not in ac2.columns:
                    ac2[c] = 0

            for c in (
                "sessions_prev_7d",
                "sessions_recent_7d",
                "delta_sessions",
                "cvr_prev_7d",
                "cvr_recent_7d",
                "delta_cvr",
                "ad_spend_roll",
                "focus_score",
            ):
                ac2[c] = pd.to_numeric(ac2.get(c, 0.0), errors="coerce").fillna(0.0)
            ac2["current_phase"] = ac2.get("current_phase", "").map(_norm_phase)
            ac2["product_category"] = ac2.get("product_category", "").map(_norm_product_category)

            # 阈值：复用 focus_scoring 的配置入口（保持“抓重点排序”与“告警”口径一致）
            fs = getattr(policy, "dashboard_focus_scoring", None) if policy is not None else None
            try:
                min_prev_sess = float(getattr(fs, "cvr_signal_min_sessions_prev", 100.0) or 100.0) if fs is not None else 100.0
                min_delta_sess = float(getattr(fs, "cvr_signal_min_delta_sessions", 50.0) or 50.0) if fs is not None else 50.0
                min_cvr_drop = float(getattr(fs, "cvr_signal_min_cvr_drop", 0.02) or 0.02) if fs is not None else 0.02
                min_spend_roll = float(getattr(fs, "cvr_signal_min_ad_spend_roll", 10.0) or 10.0) if fs is not None else 10.0
            except Exception:
                min_prev_sess, min_delta_sess, min_cvr_drop, min_spend_roll = 100.0, 50.0, 0.02, 10.0

            sub = ac2[
                (ac2["ad_spend_roll"] >= min_spend_roll)
                & (ac2["sessions_prev_7d"] >= min_prev_sess)
                & (ac2["delta_sessions"] >= min_delta_sess)
                & (ac2["delta_cvr"] <= -abs(min_cvr_drop))
            ].copy()
            if not sub.empty:
                cnt = int(len(sub))
                spend_sum = float(sub["ad_spend_roll"].sum() or 0.0)
                total_spend = float(ac2["ad_spend_roll"].sum() or 0.0)
                spend_share = (spend_sum / total_spend) if total_spend > 0 else 0.0

                sub = sub.sort_values(["ad_spend_roll", "focus_score"], ascending=[False, False]).copy()
                top = sub.iloc[0].to_dict()
                top_asin = str(top.get("asin", "") or "").strip().upper()
                top_delta_sess = float(top.get("delta_sessions", 0.0) or 0.0)
                top_delta_cvr = float(top.get("delta_cvr", 0.0) or 0.0)
                top_cvr_recent = float(top.get("cvr_recent_7d", 0.0) or 0.0)
                top_phase = str(top.get("current_phase", "") or "").strip().lower() or "unknown"
                top_cat = str(top.get("product_category", "") or "（未分类）").strip()

                priority = "P1"
                if (spend_sum >= 200.0) or (spend_share >= 0.25 and spend_sum >= 50.0 and cnt >= 3):
                    priority = "P0"

                top_hint = ""
                if top_asin:
                    top_hint = f"top={top_asin} Δsessions={top_delta_sess:.0f} Δcvr={top_delta_cvr:.4f} cvr_recent_7d={top_cvr_recent:.4f} phase={top_phase} category={top_cat}"
                detail = f"asin={cnt}，spend_roll_sum={spend_sum:.2f}（占比={spend_share:.1%}）"
                if top_hint:
                    detail = f"{detail}；{top_hint}"

                _add(
                    priority,
                    "近7天流量上升但转化下滑（优先排查）",
                    detail,
                    _with_playbook(
                        "[筛选 ASIN Focus](../dashboard/asin_focus.csv) | [筛选同类对比表](../dashboard/category_asin_compare.csv)",
                        "scene-phase-down",
                    ),
                )
    except Exception:
        pass

    try:
        # 如果已经有“类目阻断集中”告警，则不再重复输出“单个ASIN阻断最多”（避免 Top 5 被同类信息占满）
        if (not category_blocked_added) and (not ac.empty) and "asin" in ac.columns:
            ac = ac.copy()
            ac["asin"] = ac["asin"].astype(str).str.upper().str.strip()
            if "top_blocked_action_count" not in ac.columns:
                ac["top_blocked_action_count"] = 0
            ac["top_blocked_action_count"] = pd.to_numeric(ac["top_blocked_action_count"], errors="coerce").fillna(0).astype(int)
            sub = ac[ac["top_blocked_action_count"] > 0].copy()
            if not sub.empty:
                sub = sub.sort_values(["top_blocked_action_count", "focus_score"], ascending=[False, False])
                r = sub.iloc[0].to_dict()
                asin = str(r.get("asin", "") or "").strip().upper()
                cnt = int(_safe_int(r.get("top_blocked_action_count", 0)))
                ph = _norm_phase(r.get("current_phase", ""))
                cat = _norm_product_category(r.get("product_category", ""))
                _add(
                    "P1",
                    f"Top 动作阻断最多的 ASIN：{asin}",
                    f"blocked_top_actions={cnt}；phase={ph}；category={cat}",
                    _with_playbook(f"[查看该ASIN](./asin_drilldown.md#{_asin_anchor_id(asin)})", "inventory-first"),
                )
    except Exception:
        pass

    # ===== 利润承受度：控量方向仍在烧钱（优先提示运营回到“产品语境”） =====
    try:
        if not ac.empty and "asin" in ac.columns and "profit_direction" in ac.columns:
            ac2 = ac.copy()
            ac2["asin"] = ac2["asin"].astype(str).str.upper().str.strip()
            ac2["profit_direction"] = ac2.get("profit_direction", "").astype(str).str.strip().str.lower()
            for c in ("ad_spend_roll", "profit_before_ads", "profit_after_ads", "max_ad_spend_by_profit", "focus_score"):
                if c in ac2.columns:
                    ac2[c] = pd.to_numeric(ac2[c], errors="coerce").fillna(0.0)
                else:
                    ac2[c] = 0.0

            reduce_pool = ac2[(ac2["profit_direction"] == "reduce") & (ac2["ad_spend_roll"] > 0)].copy()
            scale_pool = ac2[(ac2["profit_direction"] == "scale") & (ac2["ad_spend_roll"] > 0)].copy()
            reduce_cnt = int(len(reduce_pool))
            scale_cnt = int(len(scale_pool))
            if reduce_cnt > 0:
                reduce_pool = reduce_pool.sort_values(["ad_spend_roll", "focus_score"], ascending=[False, False]).copy()
                r = reduce_pool.iloc[0].to_dict()
                asin = str(r.get("asin", "") or "").strip().upper()
                spend_roll = float(r.get("ad_spend_roll", 0.0) or 0.0)
                pba = float(r.get("profit_before_ads", 0.0) or 0.0)
                paa = float(r.get("profit_after_ads", 0.0) or 0.0)
                max_spend = float(r.get("max_ad_spend_by_profit", 0.0) or 0.0)

                detail = f"reduce_asin={reduce_cnt}，scale_asin={scale_cnt}；top={asin} ad_spend_roll={spend_roll:.2f} profit_before_ads={pba:.2f} profit_after_ads={paa:.2f} max_ad_spend={max_spend:.2f}"
                link = ""
                if asin:
                    link = f"[筛选 Watchlist](../dashboard/profit_reduce_watchlist.csv) | [查看该ASIN](./asin_drilldown.md#{_asin_anchor_id(asin)}) | [筛选同类对比表](../dashboard/category_asin_compare.csv)"
                else:
                    link = "[筛选 Watchlist](../dashboard/profit_reduce_watchlist.csv) | [筛选同类对比表](../dashboard/category_asin_compare.csv)"
                _add(
                    "P1",
                    "利润方向=控量但仍在烧钱（优先止血/收口）",
                    detail,
                    _with_playbook(link, "scene-profit-reduce"),
                )
    except Exception:
        pass

    # 去重 + 截断：按 P0→P1，保持输出稳定（按加入顺序）
    seen = set()
    out: List[Dict[str, str]] = []
    pr_order = {"P0": 0, "P1": 1, "P2": 2}
    try:
        # 仅按优先级排序；Python 的排序是稳定的，同优先级保持原插入顺序
        alerts_sorted = sorted(alerts, key=lambda x: pr_order.get(x.get("priority", "P1"), 9))
    except Exception:
        alerts_sorted = alerts

    for a in alerts_sorted:
        k = f"{a.get('priority')}|{a.get('title')}"
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
        if len(out) >= max(1, int(max_items or 5)):
            break

    return out


def enrich_action_board_with_product(
    action_board: pd.DataFrame,
    lifecycle_board: Optional[pd.DataFrame],
    asin_campaign_map: Optional[pd.DataFrame],
    asin_top_search_terms: Optional[pd.DataFrame],
    asin_top_targetings: Optional[pd.DataFrame],
    asin_top_placements: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    给 action_board 增加产品维度（asin/product_category/current_phase...），便于按“类目→产品”筛选动作。

    注意：由于没有唯一 ID，这里的 asin 是“关联提示（asin_hint）”，用于分析与筛选，不能当作精确归因。
    """
    if action_board is None or action_board.empty:
        return action_board

    df = action_board.copy()

    # 1) 构建映射：不同 level -> asin_hint
    def _norm_key(*parts: object) -> str:
        return "|".join([str(x or "").strip() for x in parts])

    # 为“关联提示”补充 candidates/confidence，避免误导运营（无唯一ID时只能做弱关联）
    def _build_hint_map(
        raw_df: Optional[pd.DataFrame],
        group_cols: List[str],
        asin_col: str,
        top_k: int = 3,
    ) -> Dict[str, Tuple[str, str, float, int]]:
        """
        返回:
        - key -> (asin_hint, asin_hint_candidates, asin_hint_confidence, asin_hint_candidate_count)

        candidates 形如: ASIN1(0.62);ASIN2(0.23);ASIN3(0.15)
        confidence 为 top1 spend / group_total_spend
        """
        try:
            base = raw_df.copy() if raw_df is not None else pd.DataFrame()
            if base is None or base.empty:
                return {}
            if asin_col not in base.columns:
                return {}
            for c in group_cols:
                if c not in base.columns:
                    base[c] = ""

            df2 = base.copy()
            df2[asin_col] = df2[asin_col].astype(str).str.upper().str.strip()
            for c in group_cols:
                df2[c] = df2[c].astype(str).str.strip()
            df2["spend"] = pd.to_numeric(df2.get("spend", 0.0), errors="coerce").fillna(0.0)

            g = (
                df2.groupby(group_cols + [asin_col], dropna=False, as_index=False)
                .agg(spend=("spend", "sum"))
                .copy()
            )
            if g.empty:
                return {}
            totals = (
                g.groupby(group_cols, dropna=False, as_index=False)
                .agg(total_spend=("spend", "sum"))
                .copy()
            )
            top = (
                g.sort_values("spend", ascending=False)
                .groupby(group_cols, dropna=False)
                .head(int(top_k))
                .copy()
            )
            top = top.merge(totals, on=group_cols, how="left")
            top["total_spend"] = pd.to_numeric(top.get("total_spend", 0.0), errors="coerce").fillna(0.0)
            top["share"] = 0.0
            mask = top["total_spend"] > 0
            top.loc[mask, "share"] = (top.loc[mask, "spend"] / top.loc[mask, "total_spend"]).fillna(0.0)

            out: Dict[str, Tuple[str, str, float, int]] = {}
            for key, grp in top.groupby(group_cols, dropna=False):
                if isinstance(key, tuple):
                    key_tuple = key
                else:
                    key_tuple = (key,)
                k = _norm_key(*key_tuple)
                grp2 = grp.sort_values("spend", ascending=False)
                candidates: List[str] = []
                for _, rr in grp2.iterrows():
                    asin = str(rr.get(asin_col, "") or "").strip()
                    if not asin or asin.lower() == "nan":
                        continue
                    share = float(rr.get("share", 0.0) or 0.0)
                    candidates.append(f"{asin}({share:.2f})")
                if not candidates:
                    continue
                asin_hint = candidates[0].split("(", 1)[0]
                confidence = float(grp2.iloc[0].get("share", 0.0) or 0.0)
                out[k] = (asin_hint, ";".join(candidates), round(confidence, 4), int(len(candidates)))
            return out
        except Exception:
            return {}

    # campaign -> asin
    camp_map = _build_hint_map(
        raw_df=asin_campaign_map,
        group_cols=[CAN.ad_type, CAN.campaign],
        asin_col=CAN.asin,
        top_k=3,
    )

    # search_term -> asin
    st_map = _build_hint_map(
        raw_df=asin_top_search_terms,
        group_cols=[CAN.ad_type, CAN.campaign, CAN.match_type, CAN.search_term],
        asin_col=CAN.asin,
        top_k=3,
    )

    # targeting -> asin
    tgt_map = _build_hint_map(
        raw_df=asin_top_targetings,
        group_cols=[CAN.ad_type, CAN.campaign, CAN.match_type, CAN.targeting],
        asin_col=CAN.asin,
        top_k=3,
    )

    # placement -> asin
    plc_map = _build_hint_map(
        raw_df=asin_top_placements,
        group_cols=[CAN.ad_type, CAN.campaign, CAN.placement],
        asin_col=CAN.asin,
        top_k=3,
    )

    asin_hint_list: List[str] = []
    asin_hint_candidates_list: List[str] = []
    asin_hint_confidence_list: List[float] = []
    asin_hint_candidate_count_list: List[int] = []
    asin_hint_source_list: List[str] = []
    for _, r in df.iterrows():
        level = str(r.get("level", "") or "").strip().lower()
        ad_type = str(r.get("ad_type", "") or "").strip()
        campaign = str(r.get("campaign", "") or "").strip()
        match_type = str(r.get("match_type", "") or "").strip()
        obj = str(r.get("object_name", "") or "").strip()

        asin_hint = ""
        candidates = ""
        confidence = 0.0
        cand_count = 0
        source = ""
        if level == "asin":
            asin_hint = obj.strip().upper()
            if asin_hint:
                candidates = f"{asin_hint}(1.00)"
                confidence = 1.0
                cand_count = 1
                source = "asin"
        elif level == "campaign":
            info = camp_map.get(_norm_key(ad_type, campaign))
            if info:
                asin_hint, candidates, confidence, cand_count = info
                source = "campaign"
        elif level == "search_term":
            info = st_map.get(_norm_key(ad_type, campaign, match_type, obj))
            if info:
                asin_hint, candidates, confidence, cand_count = info
                source = "search_term"
        elif level == "targeting":
            info = tgt_map.get(_norm_key(ad_type, campaign, match_type, obj))
            if info:
                asin_hint, candidates, confidence, cand_count = info
                source = "targeting"
        elif level == "placement":
            info = plc_map.get(_norm_key(ad_type, campaign, obj))
            if info:
                asin_hint, candidates, confidence, cand_count = info
                source = "placement"

        asin_hint_list.append(asin_hint)
        asin_hint_candidates_list.append(candidates)
        asin_hint_confidence_list.append(float(confidence or 0.0))
        asin_hint_candidate_count_list.append(int(cand_count or 0))
        asin_hint_source_list.append(source)

    df["asin_hint"] = asin_hint_list
    df["asin_hint_candidates"] = asin_hint_candidates_list
    df["asin_hint_confidence"] = asin_hint_confidence_list
    df["asin_hint_candidate_count"] = asin_hint_candidate_count_list
    df["asin_hint_source"] = asin_hint_source_list

    # 2) 用 lifecycle_board 补全产品维度（category/phase/cycle/inventory）
    try:
        b = lifecycle_board.copy() if lifecycle_board is not None else pd.DataFrame()
        if b is not None and not b.empty and "asin" in b.columns:
            meta = b.copy()
            meta["asin_hint"] = meta["asin"].astype(str).str.upper().str.strip()
            keep = ["asin_hint"]
            for c in ("product_name", "product_category", "current_phase", "cycle_id", "inventory", "flag_low_inventory", "flag_oos"):
                if c in meta.columns:
                    keep.append(c)
            meta = meta[keep].drop_duplicates("asin_hint", keep="first")
            df = df.merge(meta, on="asin_hint", how="left")
    except Exception:
        pass

    # 分类兜底（与 ASIN Focus 一致）
    if "product_category" in df.columns:
        df["product_category"] = df["product_category"].astype(str).fillna("").map(lambda x: "" if str(x).strip().lower() == "nan" else str(x).strip())
        df.loc[df["product_category"] == "", "product_category"] = "（未分类）"

    return df


def enrich_action_board_with_playbook_scene(action_board: pd.DataFrame) -> pd.DataFrame:
    """
    给 Action Board 增加“操作手册联动”字段：

    - playbook_scene：稳定锚点（对应 docs/OPS_PLAYBOOK.md 内的 <a id="...">）
    - playbook_url：从 output 报告目录指向 repo 根目录 docs 的相对路径（便于 HTML/Markdown 一键跳转）

    说明：
    - 这是“体验层”字段，不影响任何算数口径/动作生成逻辑；
    - 目标是让运营从动作表也能直接回到“怎么查/怎么做”的固定流程。
    """
    if action_board is None or action_board.empty:
        return action_board

    df = action_board.copy()

    # 从 output/<run>/<shop>/(reports|dashboard)/... 指向仓库根目录 docs/ 的相对路径
    pb_doc = "../../../../docs/OPS_PLAYBOOK.md"

    # 基础字段（尽量兼容缺列）
    act = df.get("action_type", "").astype(str).fillna("").str.upper().str.strip()
    level = df.get("level", "").astype(str).fillna("").str.lower().str.strip()
    blocked = pd.to_numeric(df.get("blocked", 0), errors="coerce").fillna(0).astype(int)
    blocked_reason = df.get("blocked_reason", "").astype(str).fillna("")

    # 产品侧信号（优先用 asin_* 前缀字段；缺失则回退）
    oos_days = (
        pd.to_numeric(df.get("asin_oos_with_ad_spend_days", df.get("oos_with_ad_spend_days", 0.0)), errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    flag_oos = (
        pd.to_numeric(df.get("asin_flag_oos", df.get("flag_oos", 0.0)), errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    phase_changed = (
        pd.to_numeric(
            df.get("asin_phase_changed_recent_14d", df.get("phase_changed_recent_14d", 0)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    phase_trend = df.get("asin_phase_trend_14d", df.get("phase_trend_14d", "")).astype(str).fillna("").str.lower().str.strip()
    profit_dir = df.get("asin_profit_direction", df.get("profit_direction", "")).astype(str).fillna("").str.lower().str.strip()

    # 规则化场景选择（优先级从“更硬风险/更明确语境”到“通用动作解释”）
    # 场景锚点在 docs/OPS_PLAYBOOK.md 中维护：
    # - inventory-first
    # - scene-oos-spend / scene-spend-up-no-sales / scene-phase-down / scene-scale-opportunity / scene-profit-reduce / scene-keyword-topics
    scene = pd.Series([""] * int(len(df)), index=df.index, dtype="object")

    # 1) 断货仍烧钱（强信号：优先止损）
    #    说明：不仅看 oos_days，也允许 flag_oos 做兜底（某些报表缺天数但能识别断货标记）
    cond_oos = (oos_days > 0) | (flag_oos > 0)
    scene.loc[cond_oos] = "scene-oos-spend"

    # 2) 利润控量（把动作放回产品语境）
    cond_profit_reduce = (scene == "") & (profit_dir == "reduce")
    scene.loc[cond_profit_reduce] = "scene-profit-reduce"

    # 3) 阶段走弱（动态生命周期）
    cond_phase_down = (scene == "") & (phase_changed > 0) & (phase_trend == "down")
    scene.loc[cond_phase_down] = "scene-phase-down"

    # 4) 库存/断货/覆盖不足导致的放量阻断（先解锁再放量）
    #    注意：blocked 本身只对放量类动作有意义，但这里依旧作为“先看库存”的强提示。
    cond_blocked = (scene == "") & (blocked > 0)
    scene.loc[cond_blocked] = "inventory-first"

    # 5) 放量类动作（机会池语境）
    scale_actions = {"BID_UP", "BUDGET_UP"}
    cond_scale = (scene == "") & (act.isin(scale_actions))
    scene.loc[cond_scale] = "scene-scale-opportunity"

    # 6) 关键词/投放结构动作（更贴近关键词主题与止损）
    #    - search_term/targeting 层的 NEGATE/BID_DOWN 等，优先引导到“关键词主题”打法
    stop_actions = {"NEGATE", "BID_DOWN", "BUDGET_DOWN", "PAUSE"}
    cond_kw = (scene == "") & (level.isin({"search_term", "targeting"})) & (act.isin(stop_actions))
    scene.loc[cond_kw] = "scene-keyword-topics"

    # 7) 其余止损/排查动作：给一个通用入口（加花费无增量/止损类）
    cond_stop = (scene == "") & (act.isin(stop_actions))
    scene.loc[cond_stop] = "scene-spend-up-no-sales"

    df["playbook_scene"] = scene.astype(str)

    # 生成可跳转 URL（空 scene 则为空）
    try:
        df["playbook_url"] = df["playbook_scene"].map(lambda s: f"{pb_doc}#{s}" if str(s or "").strip() else "")
    except Exception:
        df["playbook_url"] = ""

    # 把 playbook_* 放到 priority_reason_3 后面（更方便筛选/阅读）
    try:
        cols = list(df.columns)
        for c in ("playbook_scene", "playbook_url"):
            if c in cols:
                cols.remove(c)
        if "priority_reason_3" in cols:
            i = cols.index("priority_reason_3") + 1
        elif "priority_reason" in cols:
            i = cols.index("priority_reason") + 1
        else:
            i = len(cols)
        cols.insert(i, "playbook_scene")
        cols.insert(i + 1, "playbook_url")
        df = df[cols].copy()
    except Exception:
        pass

    return df


def score_action_board(
    action_board: pd.DataFrame,
    asin_focus_all: Optional[pd.DataFrame],
    policy: OpsPolicy,
) -> pd.DataFrame:
    """
    给 Action Board 增加“运营化优先级”字段，并调整默认排序。

    新增列：
    - asin_focus_score / asin_focus_reasons：从 ASIN Focus 映射（按 asin_hint）
    - blocked / blocked_reason：库存/断货阻断放量（只对放量类动作生效）
    - action_priority_score：排序分（融合 priority + focus_score + hint_confidence + 证据 spend）
    - priority_reason：1~3 个简短标签（便于运营扫读）
    """
    if action_board is None or action_board.empty:
        return action_board

    df = action_board.copy()

    # 1) 映射 ASIN focus_score（用于“动作与产品重点”联动）
    if "asin_hint" in df.columns:
        df["asin_hint"] = df["asin_hint"].astype(str).str.upper().str.strip()
    else:
        df["asin_hint"] = ""
    try:
        f = asin_focus_all.copy() if asin_focus_all is not None else pd.DataFrame()
        if f is None or f.empty or "asin" not in f.columns:
            raise ValueError("asin_focus_all empty")
        f = f.copy()
        f["asin_hint"] = f["asin"].astype(str).str.upper().str.strip()
        if "focus_score" not in f.columns:
            f["focus_score"] = 0.0
        if "focus_reasons" not in f.columns:
            f["focus_reasons"] = ""
        if "focus_reasons_history" not in f.columns:
            f["focus_reasons_history"] = ""
        # 除 focus_score 外，再映射一些“产品侧 P0 维度”，便于运营在动作表里筛选/判断
        extra_cols = []
        for c in (
            "sales_recent_7d",
            "orders_recent_7d",
            "sales_per_day_7d",
            "orders_per_day_7d",
            "inventory_cover_days_7d",
            "risk_score",
            "risk_level",
            "trend_signal",
            "signal_confidence",
            "product_signal_score",
            "product_signal_level",
            "ad_signal_score",
            "ad_signal_level",
            "sales_recent_14d",
            "orders_recent_14d",
            "sales_per_day_14d",
            "orders_per_day_14d",
            "inventory_cover_days_14d",
            "sales_recent_30d",
            "orders_recent_30d",
            "sales_per_day_30d",
            "orders_per_day_30d",
            "inventory_cover_days_30d",
            "ad_sales_share",
            "tacos",
            # 机会/增量（用于“可放量窗口”筛选）
            "tacos_roll",
            "ad_spend_roll",
            "delta_sales",
            "delta_spend",
            "marginal_tacos",
            "flag_low_inventory",
            "flag_oos",
            "oos_with_ad_spend_days",
            # 生命周期迁移信号（来自 lifecycle current_board -> asin_focus_all）
            "phase_change",
            "phase_changed_recent_14d",
            "phase_trend_14d",
            # 利润承受度（来自 diagnostics["asin_stages"]，已在 write_dashboard_outputs 合并到 asin_focus）
            "profit_stage",
            "profit_direction",
            "profit_before_ads",
            "profit_after_ads",
            "max_ad_spend_by_profit",
            "target_tacos_by_margin",
        ):
            if c in f.columns:
                extra_cols.append(c)
        fmap = f[["asin_hint", "focus_score", "focus_reasons", "focus_reasons_history"] + extra_cols].drop_duplicates(
            "asin_hint", keep="first"
        ).copy()
        fmap = fmap.rename(
            columns={
                "focus_score": "asin_focus_score",
                "focus_reasons": "asin_focus_reasons",
                "focus_reasons_history": "asin_focus_reasons_history",
                "sales_recent_7d": "asin_sales_recent_7d",
                "orders_recent_7d": "asin_orders_recent_7d",
                "sales_per_day_7d": "asin_sales_per_day_7d",
                "orders_per_day_7d": "asin_orders_per_day_7d",
                "inventory_cover_days_7d": "asin_inventory_cover_days_7d",
                "risk_score": "asin_risk_score",
                "risk_level": "asin_risk_level",
                "trend_signal": "asin_trend_signal",
                "signal_confidence": "asin_signal_confidence",
                "product_signal_score": "asin_product_signal_score",
                "product_signal_level": "asin_product_signal_level",
                "ad_signal_score": "asin_ad_signal_score",
                "ad_signal_level": "asin_ad_signal_level",
                "sales_recent_14d": "asin_sales_recent_14d",
                "orders_recent_14d": "asin_orders_recent_14d",
                "sales_per_day_14d": "asin_sales_per_day_14d",
                "orders_per_day_14d": "asin_orders_per_day_14d",
                "inventory_cover_days_14d": "asin_inventory_cover_days_14d",
                "sales_recent_30d": "asin_sales_recent_30d",
                "orders_recent_30d": "asin_orders_recent_30d",
                "sales_per_day_30d": "asin_sales_per_day_30d",
                "orders_per_day_30d": "asin_orders_per_day_30d",
                "inventory_cover_days_30d": "asin_inventory_cover_days_30d",
                "ad_sales_share": "asin_ad_sales_share",
                "tacos": "asin_tacos",
                "tacos_roll": "asin_tacos_roll",
                "ad_spend_roll": "asin_ad_spend_roll",
                "delta_sales": "asin_delta_sales",
                "delta_spend": "asin_delta_spend",
                "marginal_tacos": "asin_marginal_tacos",
                "flag_low_inventory": "asin_flag_low_inventory",
                "flag_oos": "asin_flag_oos",
                "oos_with_ad_spend_days": "asin_oos_with_ad_spend_days",
                "phase_change": "asin_phase_change",
                "phase_changed_recent_14d": "asin_phase_changed_recent_14d",
                "phase_trend_14d": "asin_phase_trend_14d",
                "profit_stage": "asin_profit_stage",
                "profit_direction": "asin_profit_direction",
                "profit_before_ads": "asin_profit_before_ads",
                "profit_after_ads": "asin_profit_after_ads",
                "max_ad_spend_by_profit": "asin_max_ad_spend_by_profit",
                "target_tacos_by_margin": "asin_target_tacos_by_margin",
            }
        )
        df = df.merge(fmap, on="asin_hint", how="left")
    except Exception:
        df["asin_focus_score"] = 0.0
        df["asin_focus_reasons"] = ""
        df["asin_focus_reasons_history"] = ""
        df["asin_sales_recent_7d"] = 0.0
        df["asin_orders_recent_7d"] = 0.0
        df["asin_sales_per_day_7d"] = 0.0
        df["asin_orders_per_day_7d"] = 0.0
        df["asin_inventory_cover_days_7d"] = 0.0
        df["asin_risk_score"] = 0.0
        df["asin_risk_level"] = ""
        df["asin_trend_signal"] = ""
        df["asin_signal_confidence"] = 0.0
        df["asin_product_signal_score"] = 0.0
        df["asin_product_signal_level"] = ""
        df["asin_ad_signal_score"] = 0.0
        df["asin_ad_signal_level"] = ""
        df["asin_sales_recent_14d"] = 0.0
        df["asin_orders_recent_14d"] = 0.0
        df["asin_sales_per_day_14d"] = 0.0
        df["asin_orders_per_day_14d"] = 0.0
        df["asin_inventory_cover_days_14d"] = 0.0
        df["asin_sales_recent_30d"] = 0.0
        df["asin_orders_recent_30d"] = 0.0
        df["asin_sales_per_day_30d"] = 0.0
        df["asin_orders_per_day_30d"] = 0.0
        df["asin_inventory_cover_days_30d"] = 0.0
        df["asin_ad_sales_share"] = 0.0
        df["asin_tacos"] = 0.0
        df["asin_tacos_roll"] = 0.0
        df["asin_ad_spend_roll"] = 0.0
        df["asin_delta_sales"] = 0.0
        df["asin_delta_spend"] = 0.0
        df["asin_marginal_tacos"] = 0.0
        df["asin_flag_low_inventory"] = 0.0
        df["asin_flag_oos"] = 0.0
        df["asin_oos_with_ad_spend_days"] = 0.0
        df["asin_phase_change"] = ""
        df["asin_phase_changed_recent_14d"] = 0
        df["asin_phase_trend_14d"] = ""
        df["asin_profit_stage"] = ""
        df["asin_profit_direction"] = ""
        df["asin_profit_before_ads"] = 0.0
        df["asin_profit_after_ads"] = 0.0
        df["asin_max_ad_spend_by_profit"] = 0.0
        df["asin_target_tacos_by_margin"] = 0.0

    if "asin_focus_score" not in df.columns:
        df["asin_focus_score"] = 0.0
    if "asin_hint_confidence" not in df.columns:
        df["asin_hint_confidence"] = 0.0
    if "e_spend" not in df.columns:
        df["e_spend"] = 0.0

    df["asin_focus_score"] = pd.to_numeric(df["asin_focus_score"], errors="coerce").fillna(0.0)
    df["asin_hint_confidence"] = pd.to_numeric(df["asin_hint_confidence"], errors="coerce").fillna(0.0)
    df["e_spend"] = pd.to_numeric(df["e_spend"], errors="coerce").fillna(0.0)

    # 新增映射字段数值化（缺失时兜底 0）
    for c in (
        "asin_sales_recent_7d",
        "asin_orders_recent_7d",
        "asin_sales_per_day_7d",
        "asin_orders_per_day_7d",
        "asin_inventory_cover_days_7d",
        "asin_risk_score",
        "asin_signal_confidence",
        "asin_product_signal_score",
        "asin_ad_signal_score",
        "asin_sales_recent_14d",
        "asin_orders_recent_14d",
        "asin_sales_per_day_14d",
        "asin_orders_per_day_14d",
        "asin_inventory_cover_days_14d",
        "asin_sales_recent_30d",
        "asin_orders_recent_30d",
        "asin_sales_per_day_30d",
        "asin_orders_per_day_30d",
        "asin_inventory_cover_days_30d",
        "asin_ad_sales_share",
        "asin_tacos",
        "asin_tacos_roll",
        "asin_ad_spend_roll",
        "asin_delta_sales",
        "asin_delta_spend",
        "asin_marginal_tacos",
        "asin_flag_low_inventory",
        "asin_flag_oos",
        "asin_oos_with_ad_spend_days",
        "asin_phase_changed_recent_14d",
        "asin_profit_before_ads",
        "asin_profit_after_ads",
        "asin_max_ad_spend_by_profit",
        "asin_target_tacos_by_margin",
    ):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # 字符字段兜底（利润视角）
    if "asin_profit_stage" not in df.columns:
        df["asin_profit_stage"] = ""
    if "asin_profit_direction" not in df.columns:
        df["asin_profit_direction"] = ""
    df["asin_profit_stage"] = df["asin_profit_stage"].astype(str).fillna("")
    df["asin_profit_direction"] = df["asin_profit_direction"].astype(str).fillna("")

    # 字符字段兜底（风险/趋势）
    if "asin_risk_level" not in df.columns:
        df["asin_risk_level"] = ""
    if "asin_trend_signal" not in df.columns:
        df["asin_trend_signal"] = ""
    df["asin_risk_level"] = df["asin_risk_level"].astype(str).fillna("")
    df["asin_trend_signal"] = df["asin_trend_signal"].astype(str).fillna("")

    # 字符字段兜底（产品/广告信号）
    if "asin_product_signal_level" not in df.columns:
        df["asin_product_signal_level"] = ""
    if "asin_ad_signal_level" not in df.columns:
        df["asin_ad_signal_level"] = ""
    df["asin_product_signal_level"] = df["asin_product_signal_level"].astype(str).fillna("")
    df["asin_ad_signal_level"] = df["asin_ad_signal_level"].astype(str).fillna("")

    # 字符字段兜底（生命周期迁移）
    if "asin_phase_change" not in df.columns:
        df["asin_phase_change"] = ""
    if "asin_phase_trend_14d" not in df.columns:
        df["asin_phase_trend_14d"] = ""
    df["asin_phase_change"] = df["asin_phase_change"].astype(str).fillna("")
    df["asin_phase_trend_14d"] = df["asin_phase_trend_14d"].astype(str).fillna("")

    # 1.5) 机会标记：可放量窗口（用于 Excel/透视筛选；不强制改动作优先级）
    # 说明：
    # - 放量动作本身已经有 blocked 机制（低库存/覆盖天数不足），这里补充“值得考虑加码”的候选池标记；
    # - 规则与 scale_opportunity_watchlist.csv 保持一致口径，方便运营“从表到动作”联动筛选。
    try:
        sw = getattr(policy, "dashboard_scale_window", None)
        try:
            min_sales_pd = float(getattr(sw, "min_sales_per_day_7d", 0.0) or 0.0)
        except Exception:
            min_sales_pd = 0.0
        try:
            min_delta_sales = float(getattr(sw, "min_delta_sales", 0.0) or 0.0)
        except Exception:
            min_delta_sales = 0.0
        try:
            min_cover_days = float(getattr(sw, "min_inventory_cover_days_7d", 30.0) or 30.0)
        except Exception:
            min_cover_days = 30.0
        try:
            max_tacos_roll = float(getattr(sw, "max_tacos_roll", 0.25) or 0.25)
        except Exception:
            max_tacos_roll = 0.25
        try:
            max_marginal_tacos = float(getattr(sw, "max_marginal_tacos", 0.25) or 0.25)
        except Exception:
            max_marginal_tacos = 0.25
        try:
            exclude_phases = [str(x or "").strip().lower() for x in getattr(sw, "exclude_phases", []) or [] if str(x or "").strip()]
        except Exception:
            exclude_phases = []
        if not exclude_phases:
            exclude_phases = ["decline", "inactive"]
        require_no_oos = bool(getattr(sw, "require_no_oos", True)) if sw is not None else True
        require_no_low_inventory = bool(getattr(sw, "require_no_low_inventory", True)) if sw is not None else True
        require_oos_spend_days_zero = bool(getattr(sw, "require_oos_with_ad_spend_days_zero", True)) if sw is not None else True

        phase = df.get("current_phase", "").astype(str).str.strip().str.lower()
        cond = (
            (df["asin_hint"].astype(str).str.strip() != "")
            & (pd.to_numeric(df.get("asin_sales_per_day_7d", 0.0), errors="coerce").fillna(0.0) > float(min_sales_pd))
            & (pd.to_numeric(df.get("asin_delta_sales", 0.0), errors="coerce").fillna(0.0) > float(min_delta_sales))
            & (pd.to_numeric(df.get("asin_inventory_cover_days_7d", 0.0), errors="coerce").fillna(0.0) >= float(min_cover_days))
            & (pd.to_numeric(df.get("asin_tacos_roll", 0.0), errors="coerce").fillna(0.0) <= float(max_tacos_roll))
            & (pd.to_numeric(df.get("asin_marginal_tacos", 0.0), errors="coerce").fillna(0.0) <= float(max_marginal_tacos))
        )
        if require_no_oos:
            cond = cond & (pd.to_numeric(df.get("asin_flag_oos", 0.0), errors="coerce").fillna(0.0) <= 0)
        if require_no_low_inventory:
            cond = cond & (pd.to_numeric(df.get("asin_flag_low_inventory", 0.0), errors="coerce").fillna(0.0) <= 0)
        if require_oos_spend_days_zero:
            cond = cond & (pd.to_numeric(df.get("asin_oos_with_ad_spend_days", 0.0), errors="coerce").fillna(0.0) <= 0)
        if exclude_phases:
            cond = cond & (~phase.isin(exclude_phases))
        df["asin_scale_window"] = cond.astype(int)
        df["asin_scale_window_reason"] = ""
        reason_parts: List[str] = []
        if min_sales_pd > 0:
            reason_parts.append(f"spd7d>{min_sales_pd:g}")
        reason_parts.append(f"ΔSales>{min_delta_sales:g}")
        reason_parts.append(f"tacos_roll<={max_tacos_roll:g}")
        reason_parts.append(f"marginal_tacos<={max_marginal_tacos:g}")
        reason_parts.append(f"cover_days>={min_cover_days:g}天")
        df.loc[cond, "asin_scale_window_reason"] = ";".join(reason_parts)
    except Exception:
        df["asin_scale_window"] = 0
        df["asin_scale_window_reason"] = ""

    # 2) 阻断规则：只对“放量类动作”生效
    scale_actions = {"BID_UP", "BUDGET_UP"}
    block_scale_low_inventory = bool(getattr(policy, "block_scale_when_low_inventory", True))
    # 覆盖天数阈值：0 表示关闭；>0 且 cover_days>0 且 cover_days<thr 时阻断放量
    try:
        cover_days_thr = float(getattr(policy, "block_scale_when_cover_days_below", 0.0) or 0.0)
    except Exception:
        cover_days_thr = 0.0
    cover_days_thr_text = ""
    try:
        if cover_days_thr > 0 and float(cover_days_thr).is_integer():
            cover_days_thr_text = str(int(cover_days_thr))
        elif cover_days_thr > 0:
            cover_days_thr_text = f"{cover_days_thr:g}"
    except Exception:
        cover_days_thr_text = ""

    blocked_list: List[int] = []
    blocked_reason_list: List[str] = []
    for _, r in df.iterrows():
        act = str(r.get("action_type", "") or "").strip().upper()
        if act not in scale_actions:
            blocked_list.append(0)
            blocked_reason_list.append("")
            continue

        # 阻断优先级：断货 > 库存=0 > 低库存 > 覆盖天数不足
        try:
            flag_oos = int(float(r.get("flag_oos", 0) or 0))
        except Exception:
            flag_oos = 0
        inv = _safe_float(r.get("inventory", 0.0))
        try:
            flag_low = int(float(r.get("flag_low_inventory", 0) or 0))
        except Exception:
            flag_low = 0

        if flag_oos > 0:
            blocked_list.append(1)
            blocked_reason_list.append("断货阻断放量")
        elif inv <= 0:
            blocked_list.append(1)
            blocked_reason_list.append("库存=0阻断放量")
        elif block_scale_low_inventory and flag_low > 0:
            blocked_list.append(1)
            blocked_reason_list.append("低库存阻断放量")
        else:
            # 覆盖天数阻断：库存不一定低，但如果销量速度很快，覆盖仍可能很短
            cover_days = _safe_float(r.get("asin_inventory_cover_days_7d", 0.0))
            if cover_days_thr > 0 and cover_days > 0 and cover_days < cover_days_thr:
                blocked_list.append(1)
                if cover_days_thr_text:
                    blocked_reason_list.append(f"库存覆盖<{cover_days_thr_text}天阻断放量")
                else:
                    blocked_reason_list.append("库存覆盖不足阻断放量")
            else:
                blocked_list.append(0)
                blocked_reason_list.append("")

    df["blocked"] = blocked_list
    df["blocked_reason"] = blocked_reason_list

    # 3) 计算 action_priority_score（排序分）
    ap = getattr(policy, "dashboard_action_scoring", None) or ActionScoringPolicy()
    pr_rank = {"P0": 0, "P1": 1, "P2": 2}

    def _base_score(priority: str) -> float:
        p = str(priority or "").strip().upper()
        if p == "P0":
            return float(ap.base_score_p0)
        if p == "P1":
            return float(ap.base_score_p1)
        if p == "P2":
            return float(ap.base_score_p2)
        return float(ap.base_score_other)

    def _action_tag(action_type: str) -> str:
        t = str(action_type or "").strip().upper()
        mapping = {
            "NEGATE": "否词",
            "BID_DOWN": "降价",
            "BID_UP": "加价",
            "BUDGET_UP": "加预算",
            "REVIEW": "排查",
        }
        return mapping.get(t, t)

    def _blocked_tag(reason: str) -> str:
        s = str(reason or "")
        if "断货" in s:
            return "阻断:断货"
        if "库存=0" in s or "库存0" in s:
            return "阻断:库存0"
        if "低库存" in s:
            return "阻断:低库存"
        return "阻断"

    scores: List[float] = []
    reasons: List[str] = []
    for _, r in df.iterrows():
        priority = str(r.get("priority", "") or "")
        act = str(r.get("action_type", "") or "").strip().upper()
        phase = str(r.get("current_phase", "") or "").strip().lower()

        conf = _safe_float(r.get("asin_hint_confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        focus_score = _safe_float(r.get("asin_focus_score", 0.0))
        e_spend = _safe_float(r.get("e_spend", 0.0))

        score = _base_score(priority)
        # 证据花费：log 缩放，避免极端 spend 把列表“顶爆”
        try:
            import math

            score += math.log1p(max(0.0, e_spend)) * float(ap.spend_log_multiplier)
        except Exception:
            score += max(0.0, e_spend) * float(ap.spend_log_multiplier)
        # 重点ASIN：把产品侧重点度融入动作排序
        score += max(0.0, focus_score) * float(ap.weight_focus_score)
        # 弱关联可信度：避免无唯一ID导致的误判
        score += conf * float(ap.weight_hint_confidence)

        # 放量类动作：低可信度/衰退期额外降权（不强制阻断）
        if act in scale_actions:
            if conf < float(ap.low_hint_confidence_threshold):
                score -= float(ap.low_hint_scale_penalty)
            if phase == "inactive":
                score -= float(ap.phase_scale_penalty_inactive)
            elif phase == "decline":
                score -= float(ap.phase_scale_penalty_decline)
            # 利润承受度方向：reduce 降权、scale 轻微加分（仍需结合库存/阻断）
            pdir = str(r.get("asin_profit_direction", "") or "").strip().lower()
            if pdir == "reduce":
                score -= float(ap.profit_reduce_scale_penalty)
            elif pdir == "scale":
                score += float(ap.profit_scale_scale_boost)

        scores.append(round(float(score), 2))

        # priority_reason：短标签（便于运营扫读）
        tags: List[str] = []
        tags.append(_action_tag(act))

        blocked = int(float(r.get("blocked", 0) or 0))
        blocked_reason = str(r.get("blocked_reason", "") or "")
        if blocked > 0 and blocked_reason:
            tags.append(_blocked_tag(blocked_reason))
        elif act in scale_actions and conf < float(ap.low_hint_confidence_threshold):
            tags.append("关联低")

        # 利润方向：补充短标签（把“动作”放回产品语境）
        pdir = str(r.get("asin_profit_direction", "") or "").strip().lower()
        if pdir == "reduce":
            tags.append("利润:控量")
        elif pdir == "scale":
            tags.append("利润:可放量")

        if focus_score >= 60:
            tags.append("ASIN重点")
        elif focus_score >= 50:
            tags.append("ASIN关注")

        if phase in {"decline", "inactive"} and blocked <= 0:
            tags.append("衰退期" if phase == "decline" else "不活跃")

        uniq: List[str] = []
        for t in tags:
            if t and t not in uniq:
                uniq.append(t)
        uniq = uniq[:3]
        reasons.append(";".join(uniq))

    df["action_priority_score"] = scores
    df["priority_reason"] = reasons

    # 3.1) needs_manual_confirm：弱关联动作的“人工确认”标记（便于运营一键筛选）
    # 说明：
    # - 当前仅对“放量类动作”（BID_UP/BUDGET_UP）标记；
    # - 低于阈值或缺少 asin_hint 时，建议先人工确认该动作对应的产品/活动再执行。
    try:
        needs_confirm: List[int] = []
        thr = float(ap.low_hint_confidence_threshold)
        thr = max(0.0, min(1.0, thr))
        for _, r in df.iterrows():
            act = str(r.get("action_type", "") or "").strip().upper()
            if act not in scale_actions:
                needs_confirm.append(0)
                continue
            asin_hint = str(r.get("asin_hint", "") or "").strip().upper()
            conf = _safe_float(r.get("asin_hint_confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
            if (not asin_hint) or (conf < thr):
                needs_confirm.append(1)
            else:
                needs_confirm.append(0)
        df["needs_manual_confirm"] = needs_confirm

        # 尽量把该列放在 asin_hint_confidence 后面（更方便筛选）
        cols = list(df.columns)
        if "asin_hint_confidence" in cols and "needs_manual_confirm" in cols:
            cols.remove("needs_manual_confirm")
            i = cols.index("asin_hint_confidence") + 1
            cols.insert(i, "needs_manual_confirm")
            df = df[cols].copy()
    except Exception:
        pass

    # priority_reason_1/2/3：把短标签拆分成列（便于 Excel 筛选/透视）
    try:
        r1: List[str] = []
        r2: List[str] = []
        r3: List[str] = []
        for x in df["priority_reason"].astype(str).fillna("").tolist():
            parts = [p.strip() for p in str(x or "").split(";") if str(p).strip()]
            r1.append(parts[0] if len(parts) >= 1 else "")
            r2.append(parts[1] if len(parts) >= 2 else "")
            r3.append(parts[2] if len(parts) >= 3 else "")
        df["priority_reason_1"] = r1
        df["priority_reason_2"] = r2
        df["priority_reason_3"] = r3

        # 把拆分列放在 priority_reason 后面（更方便阅读/筛选）
        cols = list(df.columns)
        if "priority_reason" in cols:
            for c in ("priority_reason_1", "priority_reason_2", "priority_reason_3"):
                if c in cols:
                    cols.remove(c)
            i = cols.index("priority_reason") + 1
            for j, c in enumerate(["priority_reason_1", "priority_reason_2", "priority_reason_3"]):
                cols.insert(i + j, c)
            df = df[cols].copy()
    except Exception:
        pass

    # 4) 排序：未阻断优先 → P0/P1/P2 → action_priority_score → 证据 spend
    if "priority" not in df.columns:
        df["priority"] = ""
    df["_priority_rank"] = df["priority"].astype(str).map(lambda x: pr_rank.get(str(x), 9))
    df["_e_spend"] = pd.to_numeric(df.get("e_spend", 0.0), errors="coerce").fillna(0.0)
    df["blocked"] = pd.to_numeric(df.get("blocked", 0), errors="coerce").fillna(0).astype(int)
    df = df.sort_values(
        ["blocked", "_priority_rank", "action_priority_score", "_e_spend"],
        ascending=[True, True, False, False],
    ).copy()
    df = df.drop(columns=["_priority_rank", "_e_spend"], errors="ignore")

    return df.reset_index(drop=True)


def dedup_action_board(action_board: pd.DataFrame) -> pd.DataFrame:
    """
    去重合并 Action Board（减少重复动作噪音）。

    背景：
    - 在无唯一ID的前提下，同一对象可能在 search_term/targeting 两个 level 同时触发相同动作；
    - 去重的目标是“让运营先看到更少但更确定的动作”，并保留全量版便于追溯。

    合并规则（保守）：
    - 仅在 bucket=st_tgt（search_term + targeting）内合并
    - 合并键包含 action_type/action_value/campaign/ad_group/match_type/object_name 等关键字段
    - 选择保留行：未阻断优先 → P0/P1/P2 → search_term 优先 → action_priority_score 高优先 → e_spend 高优先

    输出补充字段：
    - dedup_key：用于在 full/dedup 文件间对齐
    - merged_count：被合并的行数
    - merged_levels：合并来源 level 列表（例如 search_term;targeting）
    """
    if action_board is None or action_board.empty:
        return action_board

    df = action_board.copy()
    for c in (
        "shop",
        "ad_type",
        "level",
        "priority",
        "action_type",
        "action_value",
        "object_name",
        "campaign",
        "ad_group",
        "match_type",
        "action_key",
        "blocked",
        "action_priority_score",
        "e_spend",
    ):
        if c not in df.columns:
            df[c] = ""

    def _clean(x: object) -> str:
        s = str(x or "").strip()
        return "" if s.lower() == "nan" else s

    # 只在 search_term/targeting 之间做合并，避免跨层级误合并
    try:
        lv = df["level"].astype(str).fillna("").map(lambda x: _clean(x).lower())
    except Exception:
        lv = pd.Series([""] * int(len(df)), index=df.index)
    bucket = lv.map(lambda x: "st_tgt" if x in {"search_term", "targeting"} else x)
    df["_dedup_bucket"] = bucket

    # dedup_key：包含 bucket（确保只在 st_tgt 内合并）
    try:
        def _key(r: pd.Series) -> str:
            if str(r.get("_dedup_bucket", "") or "") != "st_tgt":
                return _clean(r.get("action_key", "")) or "|".join(
                    [
                        _clean(r.get("shop", "")),
                        _clean(r.get("ad_type", "")),
                        _clean(r.get("level", "")),
                        _clean(r.get("action_type", "")),
                        _clean(r.get("object_name", "")),
                        _clean(r.get("campaign", "")),
                        _clean(r.get("ad_group", "")),
                        _clean(r.get("match_type", "")),
                    ]
                )
            return "|".join(
                [
                    _clean(r.get("shop", "")),
                    _clean(r.get("ad_type", "")),
                    "st_tgt",
                    _clean(r.get("action_type", "")),
                    _clean(r.get("action_value", "")),
                    _clean(r.get("campaign", "")),
                    _clean(r.get("ad_group", "")),
                    _clean(r.get("match_type", "")),
                    _clean(r.get("object_name", "")),
                ]
            )

        df["dedup_key"] = df.apply(_key, axis=1)
    except Exception:
        df["dedup_key"] = df.get("action_key", "").astype(str)

    # 排序优先级：用于选择“保留哪一行”
    pr_rank = {"P0": 0, "P1": 1, "P2": 2}
    try:
        df["_priority_rank"] = df["priority"].astype(str).map(lambda x: pr_rank.get(str(x), 9))
    except Exception:
        df["_priority_rank"] = 9

    level_rank_map = {"search_term": 0, "targeting": 1, "asin": 2, "placement": 3, "campaign": 4}
    try:
        df["_level_rank"] = lv.map(lambda x: level_rank_map.get(str(x), 9))
    except Exception:
        df["_level_rank"] = 9

    df["blocked"] = pd.to_numeric(df.get("blocked", 0), errors="coerce").fillna(0).astype(int)
    df["action_priority_score"] = pd.to_numeric(df.get("action_priority_score", 0.0), errors="coerce").fillna(0.0)
    df["e_spend"] = pd.to_numeric(df.get("e_spend", 0.0), errors="coerce").fillna(0.0)

    # 聚合信息（仅对 dedup_key 粒度输出）
    def _levels_str(s: pd.Series) -> str:
        pairs = []
        for x in s.astype(str).tolist():
            xx = _clean(x).lower()
            if not xx:
                continue
            pairs.append((level_rank_map.get(xx, 9), xx))
        # 去重 + 按 rank 排序
        uniq: List[str] = []
        for _, name in sorted(set(pairs), key=lambda t: (int(t[0]), str(t[1]))):
            if name not in uniq:
                uniq.append(name)
        return ";".join(uniq)

    try:
        meta = (
            df.groupby("dedup_key", dropna=False, as_index=False)
            .agg(
                merged_count=("dedup_key", "size"),
                merged_levels=("level", _levels_str),
            )
            .copy()
        )
    except Exception:
        meta = pd.DataFrame(columns=["dedup_key", "merged_count", "merged_levels"])

    # 选择保留行（稳定排序）
    try:
        df2 = df.sort_values(
            ["blocked", "_priority_rank", "_level_rank", "action_priority_score", "e_spend"],
            ascending=[True, True, True, False, False],
            kind="mergesort",
        ).copy()
        keep = df2.drop_duplicates("dedup_key", keep="first").copy()
    except Exception:
        keep = df.drop_duplicates("dedup_key", keep="first").copy()

    keep = keep.merge(meta, on="dedup_key", how="left")
    keep["merged_count"] = pd.to_numeric(keep.get("merged_count", 1), errors="coerce").fillna(1).astype(int)
    keep["merged_levels"] = keep.get("merged_levels", "").astype(str).fillna("")

    keep = keep.drop(columns=["_priority_rank", "_level_rank", "_dedup_bucket"], errors="ignore")
    return keep.reset_index(drop=True)


def build_campaign_action_view(
    action_board: Optional[pd.DataFrame],
    max_rows: int = 50,
    min_spend: float = 10.0,
) -> pd.DataFrame:
    """
    Campaign 维度行动聚合（用于 dashboard 优先按 campaign 排查）。

    目标：
    - 把 Action Board 的“词/ASIN 级动作”汇总到 campaign
    - 先给运营一个“从大到小”的入口，再下钻到 Action Board 细节
    """
    if action_board is None or action_board.empty:
        return pd.DataFrame()
    try:
        df = action_board.copy()
        for c in ("ad_type", "campaign", "priority", "blocked", "e_spend", "asin_hint", "asin_delta_sales", "asin_delta_spend"):
            if c not in df.columns:
                df[c] = ""

        df["ad_type"] = df["ad_type"].astype(str).fillna("").str.strip()
        df["campaign"] = df["campaign"].astype(str).fillna("").str.strip()
        df = df[df["campaign"] != ""].copy()
        if df.empty:
            return pd.DataFrame()

        # 数值化
        df["e_spend"] = pd.to_numeric(df.get("e_spend", 0.0), errors="coerce").fillna(0.0)
        df["asin_delta_sales"] = pd.to_numeric(df.get("asin_delta_sales", 0.0), errors="coerce").fillna(0.0)
        df["asin_delta_spend"] = pd.to_numeric(df.get("asin_delta_spend", 0.0), errors="coerce").fillna(0.0)
        df["priority"] = df["priority"].astype(str).fillna("").str.upper().str.strip()
        df["blocked"] = pd.to_numeric(df.get("blocked", 0), errors="coerce").fillna(0).astype(int)

        df["_p0"] = (df["priority"] == "P0").astype(int)
        df["_p1"] = (df["priority"] == "P1").astype(int)
        df["_p2"] = (df["priority"] == "P2").astype(int)
        df["_blocked"] = (df["blocked"] > 0).astype(int)

        agg = (
            df.groupby(["ad_type", "campaign"], dropna=False, as_index=False)
            .agg(
                action_count=("campaign", "size"),
                p0_count=("_p0", "sum"),
                p1_count=("_p1", "sum"),
                p2_count=("_p2", "sum"),
                blocked_count=("_blocked", "sum"),
                spend_sum=("e_spend", "sum"),
                delta_sales_sum=("asin_delta_sales", "sum"),
                delta_spend_sum=("asin_delta_spend", "sum"),
            )
            .copy()
        )

        # Top ASIN（按 spend 聚合）
        if "asin_hint" in df.columns:
            df["_asin"] = df["asin_hint"].astype(str).str.upper().str.strip()
            t = (
                df[df["_asin"] != ""]
                .groupby(["ad_type", "campaign", "_asin"], dropna=False, as_index=False)
                .agg(_spend=("e_spend", "sum"))
                .sort_values(["ad_type", "campaign", "_spend"], ascending=[True, True, False])
                .copy()
            )
            top_map: Dict[Tuple[str, str], str] = {}
            for (ad_type, campaign), g in t.groupby(["ad_type", "campaign"], dropna=False):
                tops = g.head(3)["_asin"].astype(str).tolist()
                top_map[(str(ad_type), str(campaign))] = ";".join([x for x in tops if x])
            agg["top_asins"] = agg.apply(
                lambda r: top_map.get((str(r.get("ad_type", "")), str(r.get("campaign", ""))), ""),
                axis=1,
            )
        else:
            agg["top_asins"] = ""

        # 评分（轻量、可解释）：花费 + 增量 + P0 动作
        agg["_spend_score"] = agg["spend_sum"].map(lambda x: math.log1p(max(float(x or 0.0), 0.0)))
        agg["_delta_score"] = agg["delta_sales_sum"].map(lambda x: math.log1p(max(float(x or 0.0), 0.0)))
        agg["score"] = (0.5 * agg["_spend_score"]) + (0.3 * agg["_delta_score"]) + (0.2 * agg["p0_count"])

        # 过滤噪声：最低花费门槛（但保留有 P0 的 campaign）
        spend_thr = float(min_spend or 0.0)
        if spend_thr > 0:
            agg = agg[(agg["spend_sum"] >= spend_thr) | (agg["p0_count"] > 0)].copy()
        if agg.empty:
            return agg

        agg = agg.sort_values(["score", "spend_sum", "p0_count"], ascending=[False, False, False]).copy()
        if max_rows and int(max_rows) > 0:
            agg = agg.head(int(max_rows)).copy()

        # 清理辅助列
        agg = agg.drop(columns=["_spend_score", "_delta_score"], errors="ignore")
        return agg.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def build_asin_focus(
    lifecycle_board: Optional[pd.DataFrame],
    lifecycle_windows: Optional[pd.DataFrame],
    policy: OpsPolicy,
    stage: Optional[str] = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    生成 ASIN Focus List（用于“先抓重点”）。

    输出为 CSV：适合运营筛选/排序/分派任务。
    """
    if lifecycle_board is None or lifecycle_board.empty or "asin" not in lifecycle_board.columns:
        return pd.DataFrame()

    b = lifecycle_board.copy()
    b["asin_norm"] = b["asin"].astype(str).str.upper().str.strip()
    # 兼容字段：product_name / product_category 可能存在
    # 生命周期迁移信号：prev_phase/phase_change/phase_trend_14d（来自 lifecycle current_board）
    for c in ("product_name", "product_category", "current_phase", "prev_phase", "phase_change", "phase_trend_14d"):
        if c in b.columns:
            b[c] = b[c].astype(str).fillna("").str.strip()
    # 分类兜底：空/缺失统一归为“（未分类）”，保持 dashboard 输出一致
    if "product_category" in b.columns:
        b["product_category"] = b["product_category"].map(lambda x: "" if str(x).strip().lower() == "nan" else str(x).strip())
        b.loc[b["product_category"] == "", "product_category"] = "（未分类）"

    main = _pick_main_window(lifecycle_windows if lifecycle_windows is not None else pd.DataFrame())
    c7 = _pick_compare_window(lifecycle_windows if lifecycle_windows is not None else pd.DataFrame(), window_days=7)
    c14 = _pick_compare_window(lifecycle_windows if lifecycle_windows is not None else pd.DataFrame(), window_days=14)
    c30 = _pick_compare_window(lifecycle_windows if lifecycle_windows is not None else pd.DataFrame(), window_days=30)

    # 合并主窗口与 7d 环比（只取对“抓重点”有帮助的字段）
    keep_main = [
        "asin_norm",
        # --- 产品侧“正常销售字段”（用于立体决策：销量/广告/自然/库存联动） ---
        "sales",
        "orders",
        "sessions",
        # 转化率（产品侧）：orders/sessions（来源：生命周期窗口汇总）
        "cvr",
        "profit",
        "ad_spend",
        "ad_sales",
        "ad_orders",
        "ad_impressions",
        "ad_clicks",
        "ad_ctr",
        "ad_cvr",
        "organic_sales",
        "organic_orders",
        "organic_sales_share",
        "tacos",
        "ad_acos",
        "ad_sales_share",
        "ad_orders_share",
        "oos_with_sessions_days",
        "oos_with_ad_spend_days",
        "presale_order_days",
        "sp_spend",
        "sb_spend",
        "sd_spend",
    ]
    keep_main = [c for c in keep_main if c in main.columns]
    main2 = main[keep_main].copy() if not main.empty else pd.DataFrame(columns=["asin_norm"])

    # 近7天速度：recent/prev 都在 compare_7d 里（这里仅挑“运营最常用”的 recent + 增量口径）
    keep_c7 = [
        "asin_norm",
        "sales_prev",
        "sales_recent",
        "orders_prev",
        "orders_recent",
        "sessions_prev",
        "sessions_recent",
        "spend_prev",
        "spend_recent",
        "ad_orders_prev",
        "ad_orders_recent",
        "ad_sales_prev",
        "ad_sales_recent",
        "ad_impressions_prev",
        "ad_impressions_recent",
        "ad_clicks_prev",
        "ad_clicks_recent",
        "ad_ctr_prev",
        "ad_ctr_recent",
        "ad_cvr_prev",
        "ad_cvr_recent",
        "ad_sales_share_prev",
        "ad_sales_share_recent",
        "oos_with_ad_spend_days_recent",
        "oos_with_sessions_days_recent",
        "presale_order_days_recent",
        "cvr_prev",
        "cvr_recent",
        "organic_sales_prev",
        "organic_sales_recent",
        "organic_orders_prev",
        "organic_orders_recent",
        "organic_sales_share_prev",
        "organic_sales_share_recent",
        "delta_spend",
        "delta_sales",
        "delta_ad_sales",
        "delta_ad_orders",
        "delta_orders",
        "delta_sessions",
        "delta_cvr",
        "delta_ad_ctr",
        "delta_ad_cvr",
        "delta_ad_impressions",
        "delta_organic_sales",
        "delta_organic_orders",
        "delta_organic_sales_share",
        "delta_ad_sales_share",
        "delta_delta_sales",
        "trend_signal",
        "signal_confidence",
        "marginal_tacos",
        "marginal_ad_acos",
    ]
    keep_c7 = [c for c in keep_c7 if c in c7.columns]
    c7_2 = c7[keep_c7].copy() if not c7.empty else pd.DataFrame(columns=["asin_norm"])
    # 字段改名：避免与主窗口字段混淆（主窗口是累计口径，这里是“最近7天”口径）
    try:
        c7_2 = c7_2.rename(
            columns={
                "sales_prev": "sales_prev_7d",
                "sales_recent": "sales_recent_7d",
                "orders_prev": "orders_prev_7d",
                "orders_recent": "orders_recent_7d",
                "sessions_prev": "sessions_prev_7d",
                "sessions_recent": "sessions_recent_7d",
                "spend_prev": "ad_spend_prev_7d",
                "spend_recent": "ad_spend_recent_7d",
                "ad_orders_prev": "ad_orders_prev_7d",
                "ad_orders_recent": "ad_orders_recent_7d",
                "ad_sales_prev": "ad_sales_prev_7d",
                "ad_sales_recent": "ad_sales_recent_7d",
                "ad_impressions_prev": "ad_impressions_prev_7d",
                "ad_impressions_recent": "ad_impressions_recent_7d",
                "ad_clicks_prev": "ad_clicks_prev_7d",
                "ad_clicks_recent": "ad_clicks_recent_7d",
                "ad_ctr_prev": "ad_ctr_prev_7d",
                "ad_ctr_recent": "ad_ctr_recent_7d",
                "ad_cvr_prev": "ad_cvr_prev_7d",
                "ad_cvr_recent": "ad_cvr_recent_7d",
                "ad_sales_share_prev": "ad_sales_share_prev_7d",
                "ad_sales_share_recent": "ad_sales_share_recent_7d",
                "cvr_prev": "cvr_prev_7d",
                "cvr_recent": "cvr_recent_7d",
                "organic_sales_prev": "organic_sales_prev_7d",
                "organic_sales_recent": "organic_sales_recent_7d",
                "organic_orders_prev": "organic_orders_prev_7d",
                "organic_orders_recent": "organic_orders_recent_7d",
                "organic_sales_share_prev": "organic_sales_share_prev_7d",
                "organic_sales_share_recent": "organic_sales_share_recent_7d",
                "oos_with_ad_spend_days_recent": "oos_with_ad_spend_days_7d",
                "oos_with_sessions_days_recent": "oos_with_sessions_days_7d",
                "presale_order_days_recent": "presale_order_days_7d",
            }
        )
    except Exception:
        pass

    # 14/30 天：只取“最近窗口”字段（为速度/覆盖天数服务），避免 focus 表列爆炸
    def _pick_recent(df: pd.DataFrame, days: int, include_prev: bool = False, include_delta: bool = False) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["asin_norm"])
        keep = [
            "asin_norm",
            "sales_recent",
            "orders_recent",
            "sessions_recent",
            "spend_recent",
            "ad_orders_recent",
            "ad_sales_recent",
            "ad_impressions_recent",
            "ad_clicks_recent",
            "ad_ctr_recent",
            "ad_cvr_recent",
            "ad_sales_share_recent",
            "organic_sales_recent",
            "organic_orders_recent",
            "organic_sales_share_recent",
            "oos_with_ad_spend_days_recent",
            "oos_with_sessions_days_recent",
            "presale_order_days_recent",
        ]
        if include_prev:
            keep += [
                "sales_prev",
                "orders_prev",
                "sessions_prev",
                "spend_prev",
                "ad_orders_prev",
                "ad_sales_prev",
                "ad_impressions_prev",
                "ad_clicks_prev",
                "ad_ctr_prev",
                "ad_cvr_prev",
                "ad_sales_share_prev",
                "organic_sales_prev",
                "organic_orders_prev",
                "organic_sales_share_prev",
            ]
        if include_delta:
            keep += [
                "delta_sales",
                "delta_orders",
                "delta_sessions",
                "delta_spend",
                "delta_ad_sales",
                "delta_ad_orders",
                "delta_ad_impressions",
                "delta_ad_ctr",
                "delta_ad_cvr",
                "delta_ad_sales_share",
                "delta_organic_sales",
                "delta_organic_orders",
                "delta_organic_sales_share",
            ]
        keep = [c for c in keep if c in df.columns]
        out2 = df[keep].copy() if keep else pd.DataFrame(columns=["asin_norm"])
        try:
            out2 = out2.rename(
                columns={
                    "sales_recent": f"sales_recent_{int(days)}d",
                    "orders_recent": f"orders_recent_{int(days)}d",
                "sessions_recent": f"sessions_recent_{int(days)}d",
                "spend_recent": f"ad_spend_recent_{int(days)}d",
                "ad_orders_recent": f"ad_orders_recent_{int(days)}d",
                "ad_sales_recent": f"ad_sales_recent_{int(days)}d",
                "ad_impressions_recent": f"ad_impressions_recent_{int(days)}d",
                "ad_clicks_recent": f"ad_clicks_recent_{int(days)}d",
                "ad_ctr_recent": f"ad_ctr_recent_{int(days)}d",
                "ad_cvr_recent": f"ad_cvr_recent_{int(days)}d",
                "ad_sales_share_recent": f"ad_sales_share_recent_{int(days)}d",
                "organic_sales_recent": f"organic_sales_recent_{int(days)}d",
                "organic_orders_recent": f"organic_orders_recent_{int(days)}d",
                "organic_sales_share_recent": f"organic_sales_share_recent_{int(days)}d",
                "oos_with_ad_spend_days_recent": f"oos_with_ad_spend_days_{int(days)}d",
                "oos_with_sessions_days_recent": f"oos_with_sessions_days_{int(days)}d",
                "presale_order_days_recent": f"presale_order_days_{int(days)}d",
                "sales_prev": f"sales_prev_{int(days)}d",
                "orders_prev": f"orders_prev_{int(days)}d",
                "sessions_prev": f"sessions_prev_{int(days)}d",
                "spend_prev": f"ad_spend_prev_{int(days)}d",
                "ad_orders_prev": f"ad_orders_prev_{int(days)}d",
                "ad_sales_prev": f"ad_sales_prev_{int(days)}d",
                "ad_impressions_prev": f"ad_impressions_prev_{int(days)}d",
                "ad_clicks_prev": f"ad_clicks_prev_{int(days)}d",
                "ad_ctr_prev": f"ad_ctr_prev_{int(days)}d",
                "ad_cvr_prev": f"ad_cvr_prev_{int(days)}d",
                "ad_sales_share_prev": f"ad_sales_share_prev_{int(days)}d",
                "organic_sales_prev": f"organic_sales_prev_{int(days)}d",
                "organic_orders_prev": f"organic_orders_prev_{int(days)}d",
                "organic_sales_share_prev": f"organic_sales_share_prev_{int(days)}d",
                "delta_sales": f"delta_sales_{int(days)}d",
                "delta_orders": f"delta_orders_{int(days)}d",
                "delta_sessions": f"delta_sessions_{int(days)}d",
                "delta_spend": f"delta_spend_{int(days)}d",
                "delta_ad_sales": f"delta_ad_sales_{int(days)}d",
                "delta_ad_orders": f"delta_ad_orders_{int(days)}d",
                "delta_ad_impressions": f"delta_ad_impressions_{int(days)}d",
                "delta_ad_ctr": f"delta_ad_ctr_{int(days)}d",
                "delta_ad_cvr": f"delta_ad_cvr_{int(days)}d",
                "delta_ad_sales_share": f"delta_ad_sales_share_{int(days)}d",
                "delta_organic_sales": f"delta_organic_sales_{int(days)}d",
                "delta_organic_orders": f"delta_organic_orders_{int(days)}d",
                "delta_organic_sales_share": f"delta_organic_sales_share_{int(days)}d",
            }
        )
        except Exception:
            pass
        return out2

    c14_2 = _pick_recent(c14, 14, include_prev=True, include_delta=True)
    c30_2 = _pick_recent(c30, 30)

    out = (
        b.merge(main2, on="asin_norm", how="left")
        .merge(c7_2, on="asin_norm", how="left")
        .merge(c14_2, on="asin_norm", how="left")
        .merge(c30_2, on="asin_norm", how="left")
    )
    out = out.fillna(0.0)

    # --- 派生维度：把“销量速度/库存覆盖/自然占比”显式化（运营更容易抓重点） ---
    try:
        # 数值化（只处理我们新加的列，避免影响原有字符串列）
        for c in (
            "sales_recent_7d",
            "sales_prev_7d",
            "orders_recent_7d",
            "orders_prev_7d",
            "sessions_prev_7d",
            "sessions_recent_7d",
            "ad_spend_prev_7d",
            "ad_spend_recent_7d",
            "ad_orders_prev_7d",
            "ad_orders_recent_7d",
            "ad_sales_prev_7d",
            "ad_sales_recent_7d",
            "ad_impressions_prev_7d",
            "ad_impressions_recent_7d",
            "ad_clicks_prev_7d",
            "ad_clicks_recent_7d",
            "ad_ctr_prev_7d",
            "ad_ctr_recent_7d",
            "ad_cvr_prev_7d",
            "ad_cvr_recent_7d",
            "ad_sales_share_prev_7d",
            "ad_sales_share_recent_7d",
            "cvr",
            "cvr_prev_7d",
            "cvr_recent_7d",
            "organic_sales_prev_7d",
            "organic_sales_recent_7d",
            "organic_orders_prev_7d",
            "organic_orders_recent_7d",
            "organic_sales_share_prev_7d",
            "organic_sales_share_recent_7d",
            "delta_cvr",
            "delta_sales",
            "delta_delta_sales",
            "delta_orders",
            "delta_spend",
            "delta_sessions",
            "delta_organic_sales",
            "delta_organic_orders",
            "delta_organic_sales_share",
            "delta_ad_sales",
            "delta_ad_orders",
            "delta_ad_impressions",
            "delta_ad_ctr",
            "delta_ad_cvr",
            "delta_ad_sales_share",
            "signal_confidence",
            "sales_recent_14d",
            "sales_prev_14d",
            "orders_recent_14d",
            "orders_prev_14d",
            "sessions_recent_14d",
            "sessions_prev_14d",
            "ad_spend_recent_14d",
            "ad_spend_prev_14d",
            "ad_orders_recent_14d",
            "ad_orders_prev_14d",
            "ad_sales_recent_14d",
            "ad_sales_prev_14d",
            "ad_impressions_recent_14d",
            "ad_impressions_prev_14d",
            "ad_clicks_recent_14d",
            "ad_clicks_prev_14d",
            "ad_ctr_recent_14d",
            "ad_ctr_prev_14d",
            "ad_cvr_recent_14d",
            "ad_cvr_prev_14d",
            "ad_sales_share_recent_14d",
            "ad_sales_share_prev_14d",
            "organic_sales_recent_14d",
            "organic_orders_recent_14d",
            "organic_sales_share_recent_14d",
            "organic_sales_prev_14d",
            "organic_orders_prev_14d",
            "organic_sales_share_prev_14d",
            "delta_sales_14d",
            "delta_orders_14d",
            "delta_sessions_14d",
            "delta_spend_14d",
            "delta_ad_sales_14d",
            "delta_ad_orders_14d",
            "delta_ad_impressions_14d",
            "delta_ad_ctr_14d",
            "delta_ad_cvr_14d",
            "delta_ad_sales_share_14d",
            "delta_organic_sales_14d",
            "delta_organic_orders_14d",
            "delta_organic_sales_share_14d",
            "sales_recent_30d",
            "orders_recent_30d",
            "sessions_recent_30d",
            "ad_spend_recent_30d",
            "ad_orders_recent_30d",
            "ad_sales_recent_30d",
            "ad_impressions_recent_30d",
            "ad_clicks_recent_30d",
            "ad_ctr_recent_30d",
            "ad_cvr_recent_30d",
            "ad_sales_share_recent_30d",
            "organic_sales_recent_30d",
            "organic_orders_recent_30d",
            "organic_sales_share_recent_30d",
            "inventory",
            "sales",
            "orders",
            "profit",
            "ad_sales",
            "ad_orders",
            "ad_impressions",
            "ad_clicks",
            "ad_ctr",
            "ad_cvr",
            "organic_sales",
            "organic_orders",
            "organic_sales_share",
            "ad_sales_share",
            "ad_orders_share",
        ):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

        # 小工具：安全除法（避免 0/0）
        def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
            try:
                out2 = pd.Series([0.0] * len(a), index=a.index)
                mask = b > 0
                out2.loc[mask] = (a.loc[mask] / b.loc[mask]).astype(float)
                return out2.fillna(0.0)
            except Exception:
                return pd.Series([0.0] * len(a), index=a.index)

        # 1) 近7天/14天/30天派生字段（集中拼接，避免碎片化告警）
        new_cols: Dict[str, pd.Series] = {}
        orders_per_day_7d = None
        orders_per_day_14d = None
        orders_per_day_30d = None

        # 1) 近7天“速度”（日均）
        if "sales_recent_7d" in out.columns:
            new_cols["sales_per_day_7d"] = (out["sales_recent_7d"] / 7.0).fillna(0.0)
        if "orders_recent_7d" in out.columns:
            orders_per_day_7d = (out["orders_recent_7d"] / 7.0).fillna(0.0)
            new_cols["orders_per_day_7d"] = orders_per_day_7d
        if "sessions_recent_7d" in out.columns:
            new_cols["sessions_per_day_7d"] = (out["sessions_recent_7d"] / 7.0).fillna(0.0)
        if "ad_spend_recent_7d" in out.columns:
            new_cols["ad_spend_per_day_7d"] = (out["ad_spend_recent_7d"] / 7.0).fillna(0.0)

        # 1.1) 广告端效率（7d）：CTR/CVR/CPA/ACOS/广告占比
        if ("ad_clicks_recent_7d" in out.columns) and ("ad_impressions_recent_7d" in out.columns):
            new_cols["ad_ctr_recent_7d"] = _safe_div(out["ad_clicks_recent_7d"], out["ad_impressions_recent_7d"])
        if ("ad_clicks_prev_7d" in out.columns) and ("ad_impressions_prev_7d" in out.columns):
            new_cols["ad_ctr_prev_7d"] = _safe_div(out["ad_clicks_prev_7d"], out["ad_impressions_prev_7d"])
        if ("ad_orders_recent_7d" in out.columns) and ("ad_clicks_recent_7d" in out.columns):
            new_cols["ad_cvr_recent_7d"] = _safe_div(out["ad_orders_recent_7d"], out["ad_clicks_recent_7d"])
        if ("ad_orders_prev_7d" in out.columns) and ("ad_clicks_prev_7d" in out.columns):
            new_cols["ad_cvr_prev_7d"] = _safe_div(out["ad_orders_prev_7d"], out["ad_clicks_prev_7d"])
        if ("ad_spend_recent_7d" in out.columns) and ("ad_orders_recent_7d" in out.columns):
            new_cols["ad_cpa_order_recent_7d"] = _safe_div(out["ad_spend_recent_7d"], out["ad_orders_recent_7d"])
        if ("ad_spend_prev_7d" in out.columns) and ("ad_orders_prev_7d" in out.columns):
            new_cols["ad_cpa_order_prev_7d"] = _safe_div(out["ad_spend_prev_7d"], out["ad_orders_prev_7d"])
        if ("ad_spend_recent_7d" in out.columns) and ("ad_sales_recent_7d" in out.columns):
            new_cols["ad_acos_recent_7d"] = _safe_div(out["ad_spend_recent_7d"], out["ad_sales_recent_7d"])
        if ("ad_spend_prev_7d" in out.columns) and ("ad_sales_prev_7d" in out.columns):
            new_cols["ad_acos_prev_7d"] = _safe_div(out["ad_spend_prev_7d"], out["ad_sales_prev_7d"])
        if ("ad_sales_recent_7d" in out.columns) and ("sales_recent_7d" in out.columns):
            new_cols["ad_sales_share_recent_7d"] = _safe_div(out["ad_sales_recent_7d"], out["sales_recent_7d"])
        if ("ad_sales_prev_7d" in out.columns) and ("sales_prev_7d" in out.columns):
            new_cols["ad_sales_share_prev_7d"] = _safe_div(out["ad_sales_prev_7d"], out["sales_prev_7d"])

        # 2) 库存覆盖天数（粗粒度）：库存 / 日均订单
        if "inventory" in out.columns and orders_per_day_7d is not None:
            inv = pd.to_numeric(out["inventory"], errors="coerce").fillna(0.0)
            cover = pd.Series(0.0, index=out.index)
            mask = orders_per_day_7d > 0
            cover.loc[mask] = (inv.loc[mask] / orders_per_day_7d.loc[mask]).fillna(0.0)
            new_cols["inventory_cover_days_7d"] = cover

        # 1.2) 近14天速度（日均）
        if "sales_recent_14d" in out.columns:
            new_cols["sales_per_day_14d"] = (out["sales_recent_14d"] / 14.0).fillna(0.0)
        if "orders_recent_14d" in out.columns:
            orders_per_day_14d = (out["orders_recent_14d"] / 14.0).fillna(0.0)
            new_cols["orders_per_day_14d"] = orders_per_day_14d
        if "sessions_recent_14d" in out.columns:
            new_cols["sessions_per_day_14d"] = (out["sessions_recent_14d"] / 14.0).fillna(0.0)
        if "ad_spend_recent_14d" in out.columns:
            new_cols["ad_spend_per_day_14d"] = (out["ad_spend_recent_14d"] / 14.0).fillna(0.0)

        # 1.2) 广告端效率（14d）：用于成熟期稳定性判断
        if ("ad_clicks_recent_14d" in out.columns) and ("ad_impressions_recent_14d" in out.columns):
            new_cols["ad_ctr_recent_14d"] = _safe_div(out["ad_clicks_recent_14d"], out["ad_impressions_recent_14d"])
        if ("ad_clicks_prev_14d" in out.columns) and ("ad_impressions_prev_14d" in out.columns):
            new_cols["ad_ctr_prev_14d"] = _safe_div(out["ad_clicks_prev_14d"], out["ad_impressions_prev_14d"])
        if ("ad_orders_recent_14d" in out.columns) and ("ad_clicks_recent_14d" in out.columns):
            new_cols["ad_cvr_recent_14d"] = _safe_div(out["ad_orders_recent_14d"], out["ad_clicks_recent_14d"])
        if ("ad_orders_prev_14d" in out.columns) and ("ad_clicks_prev_14d" in out.columns):
            new_cols["ad_cvr_prev_14d"] = _safe_div(out["ad_orders_prev_14d"], out["ad_clicks_prev_14d"])
        if ("ad_spend_recent_14d" in out.columns) and ("ad_orders_recent_14d" in out.columns):
            new_cols["ad_cpa_order_recent_14d"] = _safe_div(out["ad_spend_recent_14d"], out["ad_orders_recent_14d"])
        if ("ad_spend_prev_14d" in out.columns) and ("ad_orders_prev_14d" in out.columns):
            new_cols["ad_cpa_order_prev_14d"] = _safe_div(out["ad_spend_prev_14d"], out["ad_orders_prev_14d"])
        if ("ad_spend_recent_14d" in out.columns) and ("ad_sales_recent_14d" in out.columns):
            new_cols["ad_acos_recent_14d"] = _safe_div(out["ad_spend_recent_14d"], out["ad_sales_recent_14d"])
        if ("ad_spend_prev_14d" in out.columns) and ("ad_sales_prev_14d" in out.columns):
            new_cols["ad_acos_prev_14d"] = _safe_div(out["ad_spend_prev_14d"], out["ad_sales_prev_14d"])
        if ("ad_sales_recent_14d" in out.columns) and ("sales_recent_14d" in out.columns):
            new_cols["ad_sales_share_recent_14d"] = _safe_div(out["ad_sales_recent_14d"], out["sales_recent_14d"])
        if ("ad_sales_prev_14d" in out.columns) and ("sales_prev_14d" in out.columns):
            new_cols["ad_sales_share_prev_14d"] = _safe_div(out["ad_sales_prev_14d"], out["sales_prev_14d"])

        # 2.2) 库存覆盖天数（14d）
        if "inventory" in out.columns and orders_per_day_14d is not None:
            inv = pd.to_numeric(out["inventory"], errors="coerce").fillna(0.0)
            cover = pd.Series(0.0, index=out.index)
            mask = orders_per_day_14d > 0
            cover.loc[mask] = (inv.loc[mask] / orders_per_day_14d.loc[mask]).fillna(0.0)
            new_cols["inventory_cover_days_14d"] = cover

        # 1.3) 近30天速度（日均）
        if "sales_recent_30d" in out.columns:
            new_cols["sales_per_day_30d"] = (out["sales_recent_30d"] / 30.0).fillna(0.0)
        if "orders_recent_30d" in out.columns:
            orders_per_day_30d = (out["orders_recent_30d"] / 30.0).fillna(0.0)
            new_cols["orders_per_day_30d"] = orders_per_day_30d
        if "sessions_recent_30d" in out.columns:
            new_cols["sessions_per_day_30d"] = (out["sessions_recent_30d"] / 30.0).fillna(0.0)
        if "ad_spend_recent_30d" in out.columns:
            new_cols["ad_spend_per_day_30d"] = (out["ad_spend_recent_30d"] / 30.0).fillna(0.0)

        # 2.3) 库存覆盖天数（30d）
        if "inventory" in out.columns and orders_per_day_30d is not None:
            inv = pd.to_numeric(out["inventory"], errors="coerce").fillna(0.0)
            cover = pd.Series(0.0, index=out.index)
            mask = orders_per_day_30d > 0
            cover.loc[mask] = (inv.loc[mask] / orders_per_day_30d.loc[mask]).fillna(0.0)
            new_cols["inventory_cover_days_30d"] = cover

        # 3) 自然 vs 广告拆分（若缺字段则用确定性推导兜底）
        if "organic_sales" not in out.columns and "sales" in out.columns and "ad_sales" in out.columns:
            new_cols["organic_sales"] = (out["sales"] - out["ad_sales"]).clip(lower=0.0)
        if "organic_orders" not in out.columns and "orders" in out.columns and "ad_orders" in out.columns:
            new_cols["organic_orders"] = (out["orders"] - out["ad_orders"]).clip(lower=0.0)
        if "organic_sales_share" not in out.columns and "ad_sales_share" in out.columns:
            new_cols["organic_sales_share"] = (1.0 - pd.to_numeric(out["ad_sales_share"], errors="coerce").fillna(0.0)).clip(lower=0.0)

        # 4) CVR 兜底（极少数情况下生命周期窗口缺列时，用 orders/sessions 现算）
        if "cvr" not in out.columns and ("orders" in out.columns) and ("sessions" in out.columns):
            sessions = pd.to_numeric(out["sessions"], errors="coerce").fillna(0.0)
            orders = pd.to_numeric(out["orders"], errors="coerce").fillna(0.0)
            cvr_series = pd.Series(0.0, index=out.index)
            mask = sessions > 0
            cvr_series.loc[mask] = (orders.loc[mask] / sessions.loc[mask]).fillna(0.0)
            new_cols["cvr"] = cvr_series

        # 4.1) 广告端成本指标（主窗口）：CPA(订单口径)
        if ("ad_spend" in out.columns) and ("ad_orders" in out.columns):
            new_cols["ad_cpa_order"] = _safe_div(out["ad_spend"], out["ad_orders"])

        # 5) 经营侧“正常销售字段”补齐：客单价 AOV / 毛利率 gross_margin
        # 说明：
        # - AOV 用于识别“销量下滑是否来自客单变化”（价格/变体结构/促销/捆绑等）
        # - gross_margin 用于识别“利润空间是否过窄”（更适合控量/提价/降成本，而非盲目加码）
        if ("sales" in out.columns) and ("orders" in out.columns):
            sales = pd.to_numeric(out["sales"], errors="coerce").fillna(0.0)
            orders = pd.to_numeric(out["orders"], errors="coerce").fillna(0.0)
            aov = pd.Series(0.0, index=out.index)
            m = orders > 0
            aov.loc[m] = (sales.loc[m] / orders.loc[m]).fillna(0.0)
            new_cols["aov"] = aov

        # 近7天 AOV（recent vs prev）：prev 由 recent - delta 推导，避免扩列导致 focus 表字段爆炸
        if ("sales_recent_7d" in out.columns) and ("orders_recent_7d" in out.columns) and ("delta_sales" in out.columns) and ("delta_orders" in out.columns):
            sales_recent = pd.to_numeric(out["sales_recent_7d"], errors="coerce").fillna(0.0)
            orders_recent = pd.to_numeric(out["orders_recent_7d"], errors="coerce").fillna(0.0)
            delta_sales = pd.to_numeric(out["delta_sales"], errors="coerce").fillna(0.0)
            delta_orders = pd.to_numeric(out["delta_orders"], errors="coerce").fillna(0.0)

            sales_prev = (sales_recent - delta_sales).clip(lower=0.0)
            orders_prev = (orders_recent - delta_orders).clip(lower=0.0)

            aov_prev = pd.Series(0.0, index=out.index)
            aov_recent = pd.Series(0.0, index=out.index)

            mp = orders_prev > 0
            mr = orders_recent > 0
            aov_prev.loc[mp] = (sales_prev[mp] / orders_prev[mp]).fillna(0.0)
            aov_recent.loc[mr] = (sales_recent[mr] / orders_recent[mr]).fillna(0.0)

            new_cols["aov_prev_7d"] = aov_prev
            new_cols["aov_recent_7d"] = aov_recent
            new_cols["delta_aov_7d"] = (aov_recent - aov_prev).fillna(0.0)

        # 毛利率（gross_margin）：profit/sales（主窗口口径）
        if ("profit" in out.columns) and ("sales" in out.columns):
            profit = pd.to_numeric(out["profit"], errors="coerce").fillna(0.0)
            sales = pd.to_numeric(out["sales"], errors="coerce").fillna(0.0)
            gm = pd.Series(0.0, index=out.index)
            m = sales > 0
            gm.loc[m] = (profit.loc[m] / sales.loc[m]).fillna(0.0)
            gm = pd.to_numeric(gm, errors="coerce").fillna(0.0).clip(lower=-1.0, upper=1.0)
            new_cols["gross_margin"] = gm

        if new_cols:
            try:
                # 覆盖同名列，避免重复列
                to_drop = [c for c in new_cols.keys() if c in out.columns]
                if to_drop:
                    out = out.drop(columns=to_drop, errors="ignore")
                out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
            except Exception:
                pass
    except Exception:
        # 防御性：任何新维度计算失败都不影响主流程
        pass

    # --- Sigmoid 风险评分（P0）：覆盖天数 + ACoS + CVR 下降 ---
    try:
        try:
            cfg = get_stage_config(stage or "")
            target_acos = float(getattr(cfg, "target_acos", 0.25) or 0.25)
        except Exception:
            target_acos = 0.25

        def _calc_risk(row: pd.Series) -> Tuple[float, str]:
            # 覆盖天数（仅在近7天有订单速度时启用，避免缺口误判）
            inv_cover = _safe_float(row.get("inventory_cover_days_7d", 0.0))
            orders_pd = _safe_float(row.get("orders_per_day_7d", 0.0))
            oos_r = oos_risk_probability(inv_cover) if orders_pd > 0 else None

            # ACoS 风险：仅在近期有投放时启用
            ad_spend_recent = _safe_float(row.get("ad_spend_recent_7d", 0.0))
            ad_sales_recent = _safe_float(row.get("ad_sales_recent_7d", 0.0))
            acos_recent = _safe_float(row.get("ad_acos_recent_7d", row.get("ad_acos", 0.0)))
            acos_r = None
            if ad_spend_recent > 0 or ad_sales_recent > 0:
                acos_r = acos_risk_probability(acos_recent, target_acos)

            # CVR 下降风险（7d vs prev7d）
            delta_cvr = _safe_float(row.get("delta_cvr", 0.0))
            if abs(delta_cvr) <= 1e-12:
                cvr_prev = _safe_float(row.get("cvr_prev_7d", 0.0))
                cvr_recent = _safe_float(row.get("cvr_recent_7d", 0.0))
                if cvr_prev != 0.0 or cvr_recent != 0.0:
                    delta_cvr = cvr_recent - cvr_prev
            cvr_r = cvr_drop_risk_probability(delta_cvr)

            score = calculate_overall_risk(oos_risk=oos_r, acos_risk=acos_r, cvr_risk=cvr_r)
            # 置信度折减：覆盖不足时降低风险分
            conf = _safe_float(row.get("signal_confidence", 1.0))
            score = float(score) * max(0.0, min(1.0, conf))
            return float(score), risk_level(score)

        risk_vals = out.apply(_calc_risk, axis=1, result_type="expand")
        if isinstance(risk_vals, pd.DataFrame) and risk_vals.shape[1] >= 2:
            out["risk_score"] = pd.to_numeric(risk_vals.iloc[:, 0], errors="coerce").fillna(0.0)
            out["risk_level"] = risk_vals.iloc[:, 1].astype(str).fillna("")
    except Exception:
        out["risk_score"] = 0.0
        out["risk_level"] = "低"
    if "risk_score" not in out.columns:
        out["risk_score"] = 0.0
    if "risk_level" not in out.columns:
        out["risk_level"] = "低"

    # 去碎片化：避免多次 insert/赋值导致 PerformanceWarning
    try:
        out = out.copy()
    except Exception:
        pass

    # ===== 阶段化标签与中位数（用于新品/成熟期权重）=====
    stage_policy = None
    try:
        stage_policy = getattr(policy, "dashboard_stage_scoring", None)
    except Exception:
        stage_policy = None
    if stage_policy is None:
        stage_policy = StageScoringPolicy()

    def _norm_list(v: object, default: List[str]) -> List[str]:
        try:
            if not isinstance(v, list):
                return list(default)
            out2: List[str] = []
            for x in v:
                s = str(x or "").strip().lower()
                if s:
                    out2.append(s)
            return out2 if out2 else list(default)
        except Exception:
            return list(default)

    launch_phases = _norm_list(getattr(stage_policy, "launch_phases", None), [])
    growth_phases = _norm_list(getattr(stage_policy, "growth_phases", None), [])
    new_phases = _norm_list(getattr(stage_policy, "new_phases", None), [])
    mature_phases = _norm_list(getattr(stage_policy, "mature_phases", None), [])
    decline_phases = _norm_list(getattr(stage_policy, "decline_phases", None), [])

    if not launch_phases and "launch" in new_phases:
        launch_phases = ["launch"]
    if not growth_phases and "growth" in new_phases:
        growth_phases = ["growth"]
    new_other = [p for p in new_phases if p not in launch_phases and p not in growth_phases]

    def _stage_group(phase_value: object) -> str:
        try:
            p = str(phase_value or "").strip().lower()
            if p in launch_phases:
                return "launch"
            if p in growth_phases:
                return "growth"
            if p in mature_phases:
                return "mature"
            if p in decline_phases:
                return "decline"
            if p in new_other:
                return "new"
            return "other"
        except Exception:
            return "other"

    try:
        if "current_phase" in out.columns:
            out["stage_group"] = out["current_phase"].map(_stage_group)
        else:
            out["stage_group"] = "other"
    except Exception:
        out["stage_group"] = "other"

    def _median_by_stage(df: pd.DataFrame, group_names: List[str], col: str) -> float:
        try:
            sub = df[df["stage_group"].isin(group_names)]
            if sub is None or sub.empty or col not in sub.columns:
                return 0.0
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            vals = vals[vals > 0]
            if len(vals) < int(stage_policy.median_min_samples or 0):
                return 0.0
            return float(vals.median())
        except Exception:
            return 0.0

    stage_medians: Dict[Tuple[str, str], float] = {}
    for grp in ("launch", "growth", "mature"):
        stage_medians[(grp, "ad_ctr_recent_7d")] = _median_by_stage(out, [grp], "ad_ctr_recent_7d")
        stage_medians[(grp, "ad_cvr_recent_7d")] = _median_by_stage(out, [grp], "ad_cvr_recent_7d")
        stage_medians[(grp, "ad_cpa_order_recent_7d")] = _median_by_stage(out, [grp], "ad_cpa_order_recent_7d")
        stage_medians[(grp, "ad_acos_recent_7d")] = _median_by_stage(out, [grp], "ad_acos_recent_7d")
    stage_medians[("new", "ad_ctr_recent_7d")] = _median_by_stage(out, ["launch", "growth", "new"], "ad_ctr_recent_7d")
    stage_medians[("new", "ad_cvr_recent_7d")] = _median_by_stage(out, ["launch", "growth", "new"], "ad_cvr_recent_7d")
    stage_medians[("new", "ad_cpa_order_recent_7d")] = _median_by_stage(out, ["launch", "growth", "new"], "ad_cpa_order_recent_7d")

    # 评分策略（可从 config/ops_policy.json.dashboard.focus_scoring 调整）
    fs = None
    try:
        fs = getattr(policy, "dashboard_focus_scoring", None)
    except Exception:
        fs = None
    if fs is None:
        fs = FocusScoringPolicy()

    # 生成“重点原因”标签（1~3 个）+ 历史诊断
    reasons: List[str] = []
    history_reasons: List[str] = []
    stage_reasons: List[str] = []
    scores: List[float] = []
    stage_scores: List[float] = []

    for _, r in out.iterrows():
        recent_tags: List[str] = []
        history_tags: List[str] = []
        score = 0.0

        ad_spend_roll = _safe_float(r.get("ad_spend_roll", 0.0))
        tacos_roll = _safe_float(r.get("tacos_roll", 0.0))
        inv = _safe_float(r.get("inventory", 0.0))
        phase = str(r.get("current_phase", "") or "").strip().lower()
        phase_trend_14d = str(r.get("phase_trend_14d", "") or "").strip().lower()
        phase_changed_recent_14d = _safe_int(r.get("phase_changed_recent_14d", 0))

        # 1) 基础：按“最近 rolling 广告花费”优先（越花钱越先看）
        # 用 log 限制极端值
        try:
            import math

            score += min(float(fs.base_spend_score_cap), math.log1p(max(0.0, ad_spend_roll)) * float(fs.base_spend_log_multiplier))
        except Exception:
            score += min(float(fs.base_spend_score_cap), ad_spend_roll)

        # 2) 库存风险
        if _safe_int(r.get("flag_oos", 0)) > 0:
            recent_tags.append("断货")
            score += float(fs.weight_flag_oos)
        if _safe_int(r.get("flag_low_inventory", 0)) > 0:
            recent_tags.append("低库存")
            score += float(fs.weight_flag_low_inventory)

        # 断货/未可售历史诊断（不纳入近期行动优先级）
        oos_spend_days_7d = _safe_float(r.get("oos_with_ad_spend_days_7d", 0.0))
        oos_spend_days_14d = _safe_float(r.get("oos_with_ad_spend_days_14d", 0.0))
        oos_spend_days = _safe_float(r.get("oos_with_ad_spend_days", 0.0))
        if oos_spend_days_7d > 0:
            history_tags.append(f"近7天断货仍烧钱({int(oos_spend_days_7d)}天)")
        if oos_spend_days_14d > 0:
            history_tags.append(f"近14天断货仍烧钱({int(oos_spend_days_14d)}天)")
        if oos_spend_days > 0:
            history_tags.append(f"累计断货仍烧钱({int(oos_spend_days)}天)")

        oos_sessions_days_7d = _safe_float(r.get("oos_with_sessions_days_7d", 0.0))
        oos_sessions_days_14d = _safe_float(r.get("oos_with_sessions_days_14d", 0.0))
        oos_sessions_days = _safe_float(r.get("oos_with_sessions_days", 0.0))
        if oos_sessions_days_7d > 0:
            history_tags.append(f"近7天断货仍有流量({int(oos_sessions_days_7d)}天)")
        if oos_sessions_days_14d > 0:
            history_tags.append(f"近14天断货仍有流量({int(oos_sessions_days_14d)}天)")
        if oos_sessions_days > 0:
            history_tags.append(f"累计断货仍有流量({int(oos_sessions_days)}天)")

        presale_days_7d = _safe_float(r.get("presale_order_days_7d", 0.0))
        presale_days_14d = _safe_float(r.get("presale_order_days_14d", 0.0))
        presale_days = _safe_float(r.get("presale_order_days", 0.0))
        if presale_days_7d > 0:
            history_tags.append(f"近7天未可售仍出单({int(presale_days_7d)}天)")
        if presale_days_14d > 0:
            history_tags.append(f"近14天未可售仍出单({int(presale_days_14d)}天)")
        if presale_days > 0:
            history_tags.append(f"累计未可售仍出单({int(presale_days)}天)")

        # 3) 增量效率（7天）
        delta_spend = _safe_float(r.get("delta_spend", 0.0))
        delta_sales = _safe_float(r.get("delta_sales", 0.0))
        marginal_tacos = _safe_float(r.get("marginal_tacos", 0.0))
        if delta_spend > 0 and delta_sales <= 0:
            recent_tags.append("加花费但销量不增")
            score += float(fs.weight_spend_up_no_sales)
        if marginal_tacos > 0 and tacos_roll > 0 and marginal_tacos > max(0.01, tacos_roll * float(fs.marginal_tacos_worse_ratio)):
            recent_tags.append("增量效率变差")
            score += float(fs.weight_marginal_tacos_worse)

        # 4) 生命周期语境（衰退/不活跃仍花费）
        if phase in {"decline", "inactive"} and ad_spend_roll > 0:
            recent_tags.append("衰退期仍花费")
            score += float(fs.weight_decline_or_inactive_spend)

        # 4.1) 生命周期迁移信号（近14天阶段走弱）
        if phase_changed_recent_14d > 0 and phase_trend_14d == "down" and ad_spend_roll > 0:
            recent_tags.append("阶段下滑")
            score += float(getattr(fs, "weight_phase_down_recent", 0.0) or 0.0)

        # 4.2) 产品侧转化异常（Sessions↑但 CVR↓）：优先排查 listing/价格/评论/库存等“产品语境”
        # 说明：
        # - 只在样本量足够时触发，避免小样本噪声（阈值可配置）
        try:
            min_prev_sess = float(getattr(fs, "cvr_signal_min_sessions_prev", 100.0) or 100.0)
            min_delta_sess = float(getattr(fs, "cvr_signal_min_delta_sessions", 50.0) or 50.0)
            min_cvr_drop = float(getattr(fs, "cvr_signal_min_cvr_drop", 0.02) or 0.02)
            min_spend_roll = float(getattr(fs, "cvr_signal_min_ad_spend_roll", 10.0) or 10.0)
        except Exception:
            min_prev_sess, min_delta_sess, min_cvr_drop, min_spend_roll = 100.0, 50.0, 0.02, 10.0

        sess_prev_7d = _safe_float(r.get("sessions_prev_7d", 0.0))
        delta_sessions = _safe_float(r.get("delta_sessions", 0.0))
        delta_cvr = _safe_float(r.get("delta_cvr", 0.0))
        if (
            (ad_spend_roll >= min_spend_roll)
            and (sess_prev_7d >= min_prev_sess)
            and (delta_sessions >= min_delta_sess)
            and (delta_cvr <= -abs(min_cvr_drop))
        ):
            recent_tags.append("流量上升但转化下滑")
            score += float(getattr(fs, "weight_sessions_up_cvr_down", 8.0) or 0.0)

        # 4.3) 自然端回落（7d vs prev7d）：优先回到“Listing/价格/评价/变体/促销”等产品语境
        # 说明：默认做得比较保守，避免小样本噪声（阈值可配置）。
        try:
            organic_min_prev = float(getattr(fs, "organic_signal_min_organic_sales_prev", 100.0) or 100.0)
            organic_drop_ratio = float(getattr(fs, "organic_signal_drop_ratio", 0.8) or 0.8)
            organic_min_drop = float(getattr(fs, "organic_signal_min_delta_organic_sales", 20.0) or 20.0)
        except Exception:
            organic_min_prev, organic_drop_ratio, organic_min_drop = 100.0, 0.8, 20.0

        organic_prev_7d = _safe_float(r.get("organic_sales_prev_7d", 0.0))
        organic_recent_7d = _safe_float(r.get("organic_sales_recent_7d", 0.0))
        delta_organic_sales = _safe_float(r.get("delta_organic_sales", 0.0))
        ratio = (organic_recent_7d / organic_prev_7d) if organic_prev_7d > 0 else 1.0
        if (organic_prev_7d >= organic_min_prev) and (delta_organic_sales <= -abs(organic_min_drop)) and (ratio <= organic_drop_ratio):
            recent_tags.append("自然端回落")
            score += float(getattr(fs, "weight_organic_down", 6.0) or 0.0)

        # 5) 广告依赖度（如果有）
        ad_sales_share = _safe_float(r.get("ad_sales_share", 0.0))
        if ad_sales_share >= float(fs.high_ad_dependency_threshold) and ad_spend_roll > 0:
            recent_tags.append("广告依赖高")
            score += float(fs.weight_high_ad_dependency)

        # 6) 兜底：库存数值本身（没有可售也需要注意）
        if inv <= 0 and ad_spend_roll > 0:
            recent_tags.append("库存=0仍投放")
            score += float(fs.weight_inventory_zero_still_spend)

        # 6.1) 阶段化指标（新品/成熟期）
        stage_tags: List[str] = []
        stage_score = 0.0
        stage_group = str(r.get("stage_group", "") or "").strip().lower()
        max_stage_tags = int(getattr(stage_policy, "max_stage_tags", 3) or 3)

        try:
            if stage_group in {"launch", "growth", "new"}:
                ctr = _safe_float(r.get("ad_ctr_recent_7d", 0.0))
                cvr = _safe_float(r.get("ad_cvr_recent_7d", 0.0))
                cpa = _safe_float(r.get("ad_cpa_order_recent_7d", 0.0))
                impr = _safe_float(r.get("ad_impressions_recent_7d", 0.0))
                clicks = _safe_float(r.get("ad_clicks_recent_7d", 0.0))
                orders = _safe_float(r.get("ad_orders_recent_7d", 0.0))

                median_ctr = stage_medians.get((stage_group, "ad_ctr_recent_7d"), 0.0)
                if median_ctr <= 0:
                    median_ctr = stage_medians.get(("new", "ad_ctr_recent_7d"), 0.0)
                median_cvr = stage_medians.get((stage_group, "ad_cvr_recent_7d"), 0.0)
                if median_cvr <= 0:
                    median_cvr = stage_medians.get(("new", "ad_cvr_recent_7d"), 0.0)
                median_cpa = stage_medians.get((stage_group, "ad_cpa_order_recent_7d"), 0.0)
                if median_cpa <= 0:
                    median_cpa = stage_medians.get(("new", "ad_cpa_order_recent_7d"), 0.0)

                if (impr >= float(stage_policy.min_impressions_7d)) and (median_ctr > 0) and (ctr > 0) and (ctr < median_ctr * float(stage_policy.new_ctr_low_ratio)):
                    stage_tags.append("新品CTR偏低")
                    stage_score += float(stage_policy.weight_new_low_ctr)
                if (clicks >= float(stage_policy.min_clicks_7d)) and (median_cvr > 0) and (cvr > 0) and (cvr < median_cvr * float(stage_policy.new_cvr_low_ratio)):
                    stage_tags.append("新品转化偏低")
                    stage_score += float(stage_policy.weight_new_low_cvr)
                if (orders >= float(stage_policy.min_orders_7d)) and (median_cpa > 0) and (cpa > median_cpa * float(stage_policy.new_cpa_high_ratio)):
                    stage_tags.append("新品CPA偏高")
                    stage_score += float(stage_policy.weight_new_high_cpa)

            elif stage_group == "mature":
                cpa = _safe_float(r.get("ad_cpa_order_recent_7d", 0.0))
                acos = _safe_float(r.get("ad_acos_recent_7d", 0.0))
                orders = _safe_float(r.get("ad_orders_recent_7d", 0.0))
                sales_recent = _safe_float(r.get("sales_recent_7d", 0.0))
                ad_share_recent = _safe_float(r.get("ad_sales_share_recent_7d", 0.0))
                ad_share_prev = _safe_float(r.get("ad_sales_share_prev_7d", 0.0))
                spend_prev = _safe_float(r.get("ad_spend_prev_7d", 0.0))
                spend_recent = _safe_float(r.get("ad_spend_recent_7d", 0.0))

                median_cpa = stage_medians.get(("mature", "ad_cpa_order_recent_7d"), 0.0)
                median_acos = stage_medians.get(("mature", "ad_acos_recent_7d"), 0.0)

                if (orders >= float(stage_policy.min_orders_7d_mature)) and (median_cpa > 0) and (cpa > median_cpa * float(stage_policy.mature_cpa_high_ratio)):
                    stage_tags.append("成熟CPA偏高")
                    stage_score += float(stage_policy.weight_mature_high_cpa)
                if (sales_recent >= float(stage_policy.min_sales_7d)) and (median_acos > 0) and (acos > median_acos * float(stage_policy.mature_acos_high_ratio)):
                    stage_tags.append("成熟ACOS偏高")
                    stage_score += float(stage_policy.weight_mature_high_acos)

                if (sales_recent >= float(stage_policy.min_sales_7d)) and (abs(ad_share_recent - ad_share_prev) >= float(stage_policy.mature_ad_share_shift_abs)):
                    stage_tags.append("广告占比波动")
                    stage_score += float(stage_policy.weight_mature_ad_share_shift)

                if (spend_prev > 0) and (abs(spend_recent - spend_prev) / max(spend_prev, 1e-9) >= float(stage_policy.mature_spend_shift_ratio)):
                    stage_tags.append("广告投入波动")
                    stage_score += float(stage_policy.weight_mature_spend_shift)
        except Exception:
            stage_tags = []
            stage_score = 0.0

        # 控制阶段标签数量
        if stage_tags:
            stage_tags = stage_tags[: max_stage_tags if max_stage_tags > 0 else 3]
            score += float(stage_score)

        # 把阶段标签补到“近期原因”里（最多 3 条）
        if stage_tags:
            for t in stage_tags:
                if t not in recent_tags and len(recent_tags) < 3:
                    recent_tags.append(t)

        # 去重并截断到 3 个标签
        uniq_recent = []
        for t in recent_tags:
            if t not in uniq_recent:
                uniq_recent.append(t)
        uniq_recent = uniq_recent[:3]
        uniq_history = []
        for t in history_tags:
            if t not in uniq_history:
                uniq_history.append(t)
        uniq_history = uniq_history[:3]

        reasons.append(";".join(uniq_recent))
        history_reasons.append(";".join(uniq_history))
        stage_reasons.append(";".join(stage_tags))
        scores.append(round(score, 2))
        stage_scores.append(round(float(stage_score), 2))

    out["focus_score"] = scores
    out["focus_reasons"] = reasons
    out["focus_reasons_history"] = history_reasons
    out["stage_focus_score"] = stage_scores
    out["stage_focus_reasons"] = stage_reasons

    # 输出列控制（更偏“仪表盘表格”）
    cols = [
        "shop",
        "asin",
        "product_name",
        "product_category",
        "cycle_id",
        "current_phase",
        "stage_group",
        "prev_phase",
        "phase_change",
        "phase_change_days_ago",
        "phase_changed_recent_14d",
        "phase_trend_14d",
        "date",
        "inventory",
        "flag_low_inventory",
        "flag_oos",
        # 近7天速度/覆盖（P0：帮助运营把“投放动作”放在产品语境里）
        "sales_recent_7d",
        "orders_recent_7d",
        "aov_prev_7d",
        "aov_recent_7d",
        "delta_aov_7d",
        "sessions_prev_7d",
        "sessions_recent_7d",
        "cvr_prev_7d",
        "cvr_recent_7d",
        "ad_ctr_recent_7d",
        "ad_cvr_recent_7d",
        "ad_cpa_order_recent_7d",
        "ad_acos_recent_7d",
        "ad_sales_share_recent_7d",
        "organic_sales_prev_7d",
        "organic_sales_recent_7d",
        "organic_sales_share_prev_7d",
        "organic_sales_share_recent_7d",
        "sales_per_day_7d",
        "orders_per_day_7d",
        "inventory_cover_days_7d",
        # 近14/30天速度/覆盖（P0++：更稳健的速度口径）
        "sales_recent_14d",
        "orders_recent_14d",
        "sales_per_day_14d",
        "orders_per_day_14d",
        "inventory_cover_days_14d",
        "sales_recent_30d",
        "orders_recent_30d",
        "sales_per_day_30d",
        "orders_per_day_30d",
        "inventory_cover_days_30d",
        "ad_spend_roll",
        "tacos_roll",
        "profit_roll",
        "focus_score",
        "focus_reasons",
        "focus_reasons_history",
        "stage_focus_score",
        "stage_focus_reasons",
        # 主窗口（累计口径）
        "sales",
        "orders",
        "aov",
        "gross_margin",
        "sessions",
        "cvr",
        "ad_spend",
        "ad_sales",
        "ad_orders",
        "ad_ctr",
        "ad_cvr",
        "ad_cpa_order",
        "organic_sales",
        "organic_orders",
        "organic_sales_share",
        "tacos",
        "ad_acos",
        "ad_sales_share",
        "ad_orders_share",
        # 7d 环比（增量口径）
        "delta_spend",
        "delta_sales",
        "delta_orders",
        "delta_sessions",
        "delta_cvr",
        "delta_organic_sales",
        "delta_organic_sales_share",
        "marginal_tacos",
        "marginal_ad_acos",
        # 异常信号（来自主窗口）
        "oos_with_sessions_days",
        "oos_with_ad_spend_days",
        "presale_order_days",
        # 广告类型拆分（如存在）
        "sp_spend",
        "sb_spend",
        "sd_spend",
    ]
    cols = [c for c in cols if c in out.columns]
    view = out[cols].copy()

    # 排序：先 focus_score，再 ad_spend_roll
    if "focus_score" in view.columns:
        view = view.sort_values(["focus_score", "ad_spend_roll"], ascending=[False, False])
    elif "ad_spend_roll" in view.columns:
        view = view.sort_values(["ad_spend_roll"], ascending=False)

    n = int(top_n or 0)
    if n <= 0:
        n = int(getattr(policy, "keyword_funnel_top_n", 12) or 12) * 5
    return view.head(n).reset_index(drop=True)


def enrich_asin_focus_with_profit_capacity(
    asin_focus_all: pd.DataFrame,
    asin_stages: object,
) -> pd.DataFrame:
    """
    把“利润承受度/方向”摘要字段合并到 ASIN Focus（用于运营筛选/排序）。

    数据来源：
    - analysis.diagnostics.infer_asin_stage_by_profit() -> diagnostics["asin_stages"]

    注意：
    - 这里的 stage/direction 是“利润视角的阶段/方向”，为避免与 lifecycle 的 current_phase 混淆，
      输出字段统一加 `profit_` 前缀。
    """
    if asin_focus_all is None or asin_focus_all.empty:
        return asin_focus_all
    if not isinstance(asin_stages, list) or not asin_stages:
        return asin_focus_all

    rows = [x for x in asin_stages if isinstance(x, dict)]
    if not rows:
        return asin_focus_all

    df = pd.DataFrame(rows)
    if df.empty or "asin" not in df.columns:
        return asin_focus_all

    df["asin_norm"] = df["asin"].astype(str).str.upper().str.strip()
    df = df[df["asin_norm"] != ""].copy()
    if df.empty:
        return asin_focus_all

    keep = ["asin_norm"]
    # 方向/阶段（利润视角）
    for c in ("stage", "direction", "reasons", "profit_mode"):
        if c in df.columns:
            keep.append(c)
    # 关键可量化字段（用于筛选/排序）
    for c in (
        "profit_before_ads",
        "profit_after_ads",
        "max_ad_spend_by_profit",
        "gross_margin",
        "tacos",
        "ad_share",
        "target_tacos_by_margin",
        "refund_rate",
        "inventory",
    ):
        if c in df.columns:
            keep.append(c)
    df2 = df[keep].drop_duplicates("asin_norm", keep="first").copy()
    df2 = df2.rename(
        columns={
            "stage": "profit_stage",
            "direction": "profit_direction",
            "reasons": "profit_reasons",
            "profit_mode": "profit_mode",
            "profit_before_ads": "profit_before_ads",
            "profit_after_ads": "profit_after_ads",
            "max_ad_spend_by_profit": "max_ad_spend_by_profit",
            "gross_margin": "profit_gross_margin",
            "tacos": "profit_tacos",
            "ad_share": "profit_ad_share",
            "target_tacos_by_margin": "target_tacos_by_margin",
            "refund_rate": "profit_refund_rate",
            "inventory": "profit_inventory",
        }
    )

    # 数值化（让 CSV/筛选更稳定；失败不影响主流程）
    for c in (
        "profit_before_ads",
        "profit_after_ads",
        "max_ad_spend_by_profit",
        "profit_gross_margin",
        "profit_tacos",
        "profit_ad_share",
        "target_tacos_by_margin",
        "profit_refund_rate",
        "profit_inventory",
    ):
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    out = asin_focus_all.copy()
    if "asin" not in out.columns:
        return out
    out["asin_norm"] = out["asin"].astype(str).str.upper().str.strip()
    out = out.merge(df2, on="asin_norm", how="left")
    out = out.drop(columns=["asin_norm"], errors="ignore")
    return out


def enrich_asin_focus_with_signal_scores(
    asin_focus_all: pd.DataFrame,
    policy: Optional[OpsPolicy] = None,
    stage: Optional[str] = None,
) -> pd.DataFrame:
    """
    计算产品侧/广告侧 Sigmoid 信号评分（仅用于排序参考）。
    """
    if asin_focus_all is None or asin_focus_all.empty:
        return asin_focus_all

    out = asin_focus_all.copy()
    sig_policy = None
    try:
        sig_policy = getattr(policy, "dashboard_signal_scoring", None)
    except Exception:
        sig_policy = None
    if sig_policy is None:
        sig_policy = SignalScoringPolicy()

    try:
        cfg = get_stage_config(stage or "")
        target_acos = float(getattr(cfg, "target_acos", 0.25) or 0.25)
    except Exception:
        target_acos = 0.25

    weights_prod = {
        "sales": float(getattr(sig_policy, "product_sales_weight", 0.4) or 0.4),
        "sessions": float(getattr(sig_policy, "product_sessions_weight", 0.2) or 0.2),
        "organic_sales": float(getattr(sig_policy, "product_organic_sales_weight", 0.3) or 0.3),
        "profit": float(getattr(sig_policy, "product_profit_weight", 0.1) or 0.1),
    }
    weights_ad = {
        "acos": float(getattr(sig_policy, "ad_acos_weight", 0.6) or 0.6),
        "cvr": float(getattr(sig_policy, "ad_cvr_weight", 0.3) or 0.3),
        "spend_up_no_sales": float(getattr(sig_policy, "ad_spend_up_no_sales_weight", 0.1) or 0.1),
    }
    prod_steep = float(getattr(sig_policy, "product_steepness", 4.0) or 4.0)
    ad_steep = float(getattr(sig_policy, "ad_steepness", 4.0) or 4.0)

    def _ratio(delta: object, prev: object) -> float:
        try:
            d = _safe_float(delta)
            p = _safe_float(prev)
            denom = p if p > 0 else 1.0
            return float(d / denom)
        except Exception:
            return 0.0

    def _profit_score(direction: object) -> float:
        try:
            s = str(direction or "").strip().lower()
            if s == "reduce":
                return 1.0
            if s == "scale":
                return -0.5
            return 0.0
        except Exception:
            return 0.0

    def _product_score(row: pd.Series) -> float:
        try:
            return float(
                product_signal_score(
                    delta_sales_ratio=_ratio(row.get("delta_sales"), row.get("sales_prev_7d")),
                    delta_sessions_ratio=_ratio(row.get("delta_sessions"), row.get("sessions_prev_7d")),
                    delta_organic_sales_ratio=_ratio(row.get("delta_organic_sales"), row.get("organic_sales_prev_7d")),
                    profit_direction_score=_profit_score(row.get("profit_direction")),
                    weights=weights_prod,
                    steepness=prod_steep,
                )
            )
        except Exception:
            return 0.0

    def _ad_score(row: pd.Series) -> float:
        try:
            ad_spend_recent = _safe_float(row.get("ad_spend_recent_7d"))
            ad_sales_recent = _safe_float(row.get("ad_sales_recent_7d"))
            acos_val = _safe_float(row.get("ad_acos_recent_7d", row.get("ad_acos")))
            acos_r = 0.0
            if ad_spend_recent > 0 or ad_sales_recent > 0:
                acos_r = acos_risk_probability(acos_val, target_acos)

            delta_ad_cvr = _safe_float(row.get("delta_ad_cvr"))
            if abs(delta_ad_cvr) <= 1e-12:
                cvr_prev = _safe_float(row.get("ad_cvr_prev_7d"))
                cvr_recent = _safe_float(row.get("ad_cvr_recent_7d"))
                if cvr_prev != 0.0 or cvr_recent != 0.0:
                    delta_ad_cvr = cvr_recent - cvr_prev
            cvr_r = cvr_drop_risk_probability(delta_ad_cvr)

            spend_up_no_sales = 1.0 if (_safe_float(row.get("delta_spend")) > 0 and _safe_float(row.get("delta_sales")) <= 0) else 0.0
            return float(
                ad_signal_score(
                    acos_risk=acos_r,
                    cvr_risk=cvr_r,
                    spend_up_no_sales=spend_up_no_sales,
                    weights=weights_ad,
                    steepness=ad_steep,
                )
            )
        except Exception:
            return 0.0

    try:
        out["product_signal_score"] = out.apply(_product_score, axis=1)
        out["ad_signal_score"] = out.apply(_ad_score, axis=1)

        if "signal_confidence" in out.columns:
            conf = out["signal_confidence"]
        else:
            conf = pd.Series([1.0] * len(out), index=out.index)
        conf = pd.to_numeric(conf, errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
        out["product_signal_score"] = pd.to_numeric(out["product_signal_score"], errors="coerce").fillna(0.0) * conf
        out["ad_signal_score"] = pd.to_numeric(out["ad_signal_score"], errors="coerce").fillna(0.0) * conf

        out["product_signal_level"] = out["product_signal_score"].map(risk_level)
        out["ad_signal_level"] = out["ad_signal_score"].map(risk_level)
    except Exception:
        out["product_signal_score"] = 0.0
        out["ad_signal_score"] = 0.0
        out["product_signal_level"] = "低"
        out["ad_signal_level"] = "低"

    return out


def build_action_board(actions: List[ActionCandidate], top_n: int = 60) -> pd.DataFrame:
    """
    动作看板（Action Board）：把动作候选变成可排序表格。
    """
    if not actions:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for a in actions:
        d = asdict(a)
        ev = _parse_evidence_json(str(a.evidence_json or ""))
        # 把常用证据字段拉平，方便在表里排序/筛选
        for k in ("impressions", "clicks", "spend", "sales", "orders", "acos", "cvr"):
            if k in ev:
                d[f"e_{k}"] = ev.get(k)
        rows.append(d)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # action_key：用于“执行回填/复盘”稳定匹配（没有唯一 ID 时只能依赖组合键）
    def _key(r: pd.Series) -> str:
        parts = [
            str(r.get("shop", "") or "").strip(),
            str(r.get("ad_type", "") or "").strip(),
            str(r.get("level", "") or "").strip(),
            str(r.get("action_type", "") or "").strip(),
            str(r.get("object_name", "") or "").strip(),
            str(r.get("campaign", "") or "").strip(),
            str(r.get("ad_group", "") or "").strip(),
            str(r.get("match_type", "") or "").strip(),
        ]
        return "|".join(parts)

    try:
        df["action_key"] = df.apply(_key, axis=1)
    except Exception:
        df["action_key"] = ""

    # 排序分：P0>P1>P2，然后按证据 spend/acos 等
    pr_rank = {"P0": 0, "P1": 1, "P2": 2}
    df["priority_rank"] = df["priority"].astype(str).map(lambda x: pr_rank.get(str(x), 9))
    if "e_spend" not in df.columns:
        df["e_spend"] = 0.0
    if "e_acos" not in df.columns:
        df["e_acos"] = 0.0
    df["e_spend"] = pd.to_numeric(df["e_spend"], errors="coerce").fillna(0.0)
    df["e_acos"] = pd.to_numeric(df["e_acos"], errors="coerce").fillna(0.0)
    df["board_score"] = (
        (100 - df["priority_rank"] * 30)
        + df["e_spend"].clip(lower=0.0) * 0.2
        + df["e_acos"].clip(lower=0.0) * 10.0
    )
    df = df.sort_values(["priority_rank", "board_score"], ascending=[True, False])

    keep = [
        "action_key",
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
        "e_impressions",
        "e_clicks",
        "e_spend",
        "e_sales",
        "e_orders",
        "e_acos",
        "e_cvr",
        "evidence_json",
        "board_score",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    n = int(top_n or 0)
    if n > 0:
        out = out.head(n)
    return out.reset_index(drop=True)


def write_dashboard_md(
    out_path: Path,
    shop: str,
    stage: str,
    date_start: str,
    date_end: str,
    scorecard: Dict[str, object],
    category_summary: Optional[pd.DataFrame],
    category_cockpit: Optional[pd.DataFrame],
    phase_cockpit: Optional[pd.DataFrame],
    asin_focus: pd.DataFrame,
    action_board: pd.DataFrame,
    campaign_action_view: Optional[pd.DataFrame] = None,
    drivers_top_asins: Optional[pd.DataFrame] = None,
    keyword_topics: Optional[pd.DataFrame] = None,
    asin_cockpit: Optional[pd.DataFrame] = None,
    policy: Optional[OpsPolicy] = None,
    budget_transfer_plan: Optional[Dict[str, object]] = None,
    unlock_scale_tasks: Optional[pd.DataFrame] = None,
    data_quality_hints: Optional[List[str]] = None,
    action_review: Optional[pd.DataFrame] = None,
) -> None:
    """
    生成短报告（dashboard.md）：严格聚焦（结论≤5，动作≤20，ASIN≤20）。
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _df_to_md_table(df: pd.DataFrame, cols: List[str]) -> str:
            """
            轻量 Markdown 表格渲染（不依赖 tabulate）。
            """
            try:
                if df is None or df.empty:
                    return ""
                view = df.copy()
                cols2 = [c for c in cols if c in view.columns]
                if not cols2:
                    cols2 = list(view.columns)[:8]
                view = view[cols2].copy()
                # 统一把换行/管道符清理掉，避免破坏表格
                for c in cols2:
                    view[c] = view[c].map(lambda x, _c=c: _format_md_cell(_c, x))
                header = "| " + " | ".join(cols2) + " |"
                sep = "| " + " | ".join(["---"] * len(cols2)) + " |"
                body = ["| " + " | ".join(row) + " |" for row in view.values.tolist()]
                return "\n".join([header, sep] + body)
            except Exception:
                return ""

        def _rename_for_display(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
            """
            仅用于展示层：把表头改为中文，不影响 CSV 与算数口径。
            """
            try:
                if df is None or df.empty:
                    return df
                if not mapping:
                    return df
                return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
            except Exception:
                return df

        def _display_table(df: pd.DataFrame, cols: List[str], mapping: Dict[str, str]) -> str:
            try:
                view = _rename_for_display(df, mapping)
                cols2 = [mapping.get(c, c) for c in cols if mapping.get(c, c) in view.columns]
                return _df_to_md_table(view, cols2)
            except Exception:
                return _df_to_md_table(df, cols)

        def _clean_product_name(value: object) -> str:
            name = str(value or "").strip()
            if name.lower() == "nan":
                return ""
            return name

        def _build_asin_name_map() -> Dict[str, str]:
            """
            统一收集 ASIN -> 品名 映射，供展示层优先展示产品名。
            """
            name_map: Dict[str, str] = {}

            def _add(df: Optional[pd.DataFrame], asin_col: str, name_col: str) -> None:
                if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                    return
                if asin_col not in df.columns or name_col not in df.columns:
                    return
                try:
                    view = df[[asin_col, name_col]].copy()
                    view[asin_col] = view[asin_col].astype(str).str.upper().str.strip()
                    view[name_col] = view[name_col].map(_clean_product_name)
                    view = view[view[asin_col] != ""].copy()
                    view = view.drop_duplicates(subset=[asin_col], keep="first")
                    for _, r in view.iterrows():
                        asin = str(r.get(asin_col, "") or "").strip().upper()
                        name = str(r.get(name_col, "") or "").strip()
                        if asin and name and asin not in name_map:
                            name_map[asin] = name
                except Exception:
                    return

            _add(asin_focus, "asin", "product_name")
            _add(asin_cockpit, "asin", "product_name")
            _add(action_board, "asin_hint", "product_name")
            _add(unlock_scale_tasks, "asin", "product_name")
            return name_map

        asin_name_map = _build_asin_name_map()

        def _resolve_product_name(asin: str, product_name: object) -> str:
            name = _clean_product_name(product_name)
            if not name and asin:
                name = str(asin_name_map.get(asin, "") or "").strip()
            return name

        def _format_product_label(asin: str, product_name: object) -> str:
            asin_norm = str(asin or "").strip().upper()
            name = _resolve_product_name(asin_norm, product_name)
            asin_link = _asin_md_link(asin_norm, "./asin_drilldown.md") if asin_norm else ""
            if name:
                name_disp = f"**{name}**"
                if asin_link:
                    return f"{name_disp}（{asin_link}）"
                return name_disp
            return asin_link or asin_norm

        def _compact_product_label(asin: str, product_name: object) -> str:
            asin_norm = str(asin or "").strip().upper()
            name = _resolve_product_name(asin_norm, product_name)
            if name and asin_norm:
                return f"{name}({asin_norm})"
            return name or asin_norm

        lines: List[str] = []
        lines.append(f"# {shop} Dashboard（聚焦版）")
        lines.append("")
        lines.append(f"- 阶段: `{stage}`")
        lines.append(f"- 时间范围: `{date_start} ~ {date_end}`")
        lines.append("- 口径说明: 未标注的累计指标=主窗口；标注 compare/Δ 的为近N天 vs 前N天（日期见表内 recent/prev）")
        lines.append("- 表头含(7d/14d/30d)=近窗；含Δ=对比窗口；含roll=滚动窗口（字段名自带口径提示）")
        try:
            ignore_last = int(getattr(policy, "dashboard_compare_ignore_last_days", 0) or 0) if policy is not None else 0
            if ignore_last > 0:
                lines.append(f"- compare 忽略最近 {ignore_last} 天（规避归因滞后噪声）")
        except Exception:
            pass
        lines.append("")

        # 预计算：Shop Alerts / 机会池（用于“本期结论”与后续章节复用）
        try:
            alerts = build_shop_alerts(
                scorecard=scorecard if isinstance(scorecard, dict) else {},
                phase_cockpit=phase_cockpit if isinstance(phase_cockpit, pd.DataFrame) else None,
                category_cockpit=category_cockpit if isinstance(category_cockpit, pd.DataFrame) else None,
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                max_items=5,
                policy=policy,
            )
        except Exception:
            alerts = []
        try:
            scale_opportunity_all = build_scale_opportunity_watchlist(
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                max_rows=200,
                policy=policy,
            )
        except Exception:
            scale_opportunity_all = pd.DataFrame()

        lines.append("## 1) 本期结论（规则化 3-5 条）")
        lines.append("")
        # 快速入口：有复盘时加一个入口（更贴近“当下执行意义”）
        # 运营操作手册：把“告警/生命周期/关键词主题/动作闭环”串起来，避免只盯广告调
        quick_links = ["[运营操作手册](../../../../docs/OPS_PLAYBOOK.md)", "[Shop Alerts](#alerts)"]
        quick_links.append("[本周行动](#weekly)")
        quick_links.append("[Campaign排查](#campaign)")
        quick_links.append("[Action Board](../dashboard/action_board.csv)")
        quick_links.append("[解锁任务表](../dashboard/unlock_scale_tasks.csv)")
        quick_links.append("[Campaign筛选表](../dashboard/campaign_action_view.csv)")
        if isinstance(action_review, pd.DataFrame) and (not action_review.empty):
            quick_links.append("[执行复盘](#review)")
        quick_links += [
            "[Watchlists](#watchlists)",
            "[生命周期闭环](./lifecycle_overview.md#loop)",
            "[生命周期时间轴](./lifecycle_overview.md)",
            "[关键词主题](#keywords)",
            "[Drivers](#drivers)",
            "[产品侧变化](#product_changes)",
        ]
        lines.append("快速入口：" + " | ".join(quick_links))
        lines.append("")

        biz_kpi = (scorecard.get("biz_kpi") if isinstance(scorecard, dict) else {}) if scorecard else {}
        if isinstance(biz_kpi, dict) and biz_kpi:
            spend = biz_kpi.get("ad_spend_total")
            sales = biz_kpi.get("sales_total")
            orders = biz_kpi.get("orders_total")
            sessions = biz_kpi.get("sessions_total")
            tacos = biz_kpi.get("tacos_total")
            ad_sales = biz_kpi.get("ad_sales_total")
            organic_sales = biz_kpi.get("organic_sales_total")
            ad_share = biz_kpi.get("ad_sales_share_total")
            ad_orders = biz_kpi.get("ad_orders_total")
            organic_orders = biz_kpi.get("organic_orders_total")
            ad_orders_share = biz_kpi.get("ad_orders_share_total")
            cvr = biz_kpi.get("cvr_total")
            aov = biz_kpi.get("aov_total")
            ad_acos = biz_kpi.get("ad_acos_total")
            summary = f"大盘：Sales=`{sales}` | AdSpend=`{spend}` | TACOS=`{tacos}` | 广告依赖(销售)=`{ad_share}`"
            if sessions is not None and cvr is not None:
                summary += f" | Sessions=`{sessions}` | CVR=`{cvr}`"
            if aov is not None:
                summary += f" | AOV=`{aov}`"
            if ad_acos is not None:
                summary += f" | 广告ACOS=`{ad_acos}`"
            lines.append(f"- {summary}")
        compares = (scorecard.get("compares") if isinstance(scorecard, dict) else []) if scorecard else []
        if isinstance(compares, list) and compares:
            # 默认展示 7 天一行
            c7 = None
            for x in compares:
                if isinstance(x, dict) and int(x.get("window_days", 0) or 0) == 7:
                    c7 = x
                    break
            if isinstance(c7, dict):
                # 近期窗口：让运营/你更容易把“全量汇总”映射到“当下该做什么”
                recent_txt = (
                    f"近7天({c7.get('recent_start')}~{c7.get('recent_end')})："
                    f"Sales=`{c7.get('sales_recent')}` | AdSpend=`{c7.get('ad_spend_recent')}`"
                    f" | TACOS=`{c7.get('tacos_recent')}` | Profit=`{c7.get('profit_recent')}`"
                )
                compare_txt = (
                    f"近7天 vs 前7天：ΔSales=`{c7.get('delta_sales')}` | ΔAdSpend=`{c7.get('delta_ad_spend')}`"
                    f" | ΔProfit=`{c7.get('delta_profit')}` | marginal_tacos=`{c7.get('marginal_tacos')}`"
                )
                combined = recent_txt + " | " + compare_txt
                # 尽量把“变化”合并到同一条大盘摘要里，避免结论超过 3-5 条
                if lines and str(lines[-1]).startswith("- 大盘："):
                    lines[-1] = str(lines[-1]) + " | " + combined
                else:
                    lines.append(f"- {combined}")

        # 机会：可放量窗口（第二入口）
        try:
            if scale_opportunity_all is not None and not scale_opportunity_all.empty and "asin" in scale_opportunity_all.columns:
                top_asin = str(scale_opportunity_all.iloc[0].get("asin", "") or "").strip().upper()
                top_asin_md = _format_product_label(top_asin, "")
                cnt = int(len(scale_opportunity_all))
                if top_asin_md:
                    lines.append(
                        f"- 机会：可放量窗口候选 `{cnt}` 个（top={top_asin_md}；[筛选表](../dashboard/scale_opportunity_watchlist.csv)）"
                    )
                else:
                    lines.append(f"- 机会：可放量窗口候选 `{cnt}` 个（[筛选表](../dashboard/scale_opportunity_watchlist.csv)）")
        except Exception:
            pass

        # 预算迁移/放量解锁（第二入口）：只给 1 条结论，避免第一屏信息爆炸
        has_ops_plans = False
        has_weekly_tasks = False
        try:
            plan = budget_transfer_plan if isinstance(budget_transfer_plan, dict) else {}
            transfers = plan.get("transfers") if isinstance(plan, dict) else None
            savings = plan.get("savings") if isinstance(plan, dict) else None
            transfer_cnt = len(transfers) if isinstance(transfers, list) else 0
            savings_cnt = len(savings) if isinstance(savings, list) else 0
            task_cnt = int(len(unlock_scale_tasks)) if isinstance(unlock_scale_tasks, pd.DataFrame) else 0

            if transfer_cnt > 0 or savings_cnt > 0 or task_cnt > 0:
                has_ops_plans = True
                parts = []
                if transfer_cnt > 0:
                    parts.append(f"净迁移 `{transfer_cnt}`")
                if savings_cnt > 0:
                    parts.append(f"回收 `{savings_cnt}`")
                if task_cnt > 0:
                    parts.append(f"解锁任务(P0/P1 Top) `{task_cnt}`")
                joined = " | ".join(parts) if parts else "（无）"
                lines.append(
                    f"- 预算迁移/放量解锁：{joined}（[迁移表](../dashboard/budget_transfer_plan.csv) | [任务表](../dashboard/unlock_scale_tasks.csv)）"
                )
        except Exception:
            has_ops_plans = False

        # 本周行动清单（Top 3）：让运营一眼知道“近7天先做什么”（责任归属+证据+跳转）
        weekly_actions: List[Dict[str, str]] = []
        try:
            def _fmt_num(x: object, nd: int = 2) -> str:
                try:
                    if x is None:
                        return ""
                    v = float(x)  # type: ignore[arg-type]
                    if pd.isna(v):
                        return ""
                    s = f"{v:.{nd}f}"
                    s = s.rstrip("0").rstrip(".")
                    return s
                except Exception:
                    return str(x or "").strip()

            def _fmt_usd(x: object) -> str:
                s = _fmt_num(x, nd=2)
                return f"${s}" if s else ""

            def _fmt_signed_usd(x: object) -> str:
                s = _fmt_num(x, nd=2)
                if not s:
                    return ""
                if s.startswith("-"):
                    return f"-${s.lstrip('-')}"
                return f"${s}"

            def _as_float(x: object) -> float:
                try:
                    v = float(pd.to_numeric(x, errors="coerce"))
                    return 0.0 if (pd.isna(v) if hasattr(pd, "isna") else False) else v  # type: ignore[arg-type]
                except Exception:
                    return 0.0

            def _evidence_text(parts: List[str]) -> str:
                ps = [p for p in parts if str(p or "").strip()]
                return f"证据: {' | '.join(ps)}" if ps else "证据: 见明细表"

            def _priority_rank(p: str) -> int:
                pr = str(p or "").strip().upper()
                if pr == "P0":
                    return 0
                if pr == "P1":
                    return 1
                if pr == "P2":
                    return 2
                return 9

            def _group_tag(group: str) -> str:
                g = str(group or "").strip().lower()
                if g == "stop":
                    return "止损"
                if g == "scale":
                    return "放量"
                return "排查"

            def _group_rank(group: str) -> int:
                g = str(group or "").strip().lower()
                if g == "stop":
                    return 0
                if g == "review":
                    return 1
                if g == "scale":
                    return 2
                return 1

            def _group_from_alert(title: str) -> str:
                """
                轻量分类（只用于“本周行动”展示聚焦，不影响任何算数逻辑）：
                - stop：需要先止损/收口
                - scale：有放量机会（或放量被阻断需要先解锁）
                - review：需要排查根因/修复产品侧问题
                """
                t = str(title or "")
                if ("断货" in t) or ("库存" in t) or ("低库存" in t):
                    return "stop"
                if ("加花费" in t) or ("无销量" in t) or ("浪费" in t) or ("ACOS" in t):
                    return "stop"
                if ("阶段走弱" in t) or ("转化" in t) or ("自然" in t) or ("评价" in t) or ("Listing" in t):
                    return "review"
                if ("放量" in t) or ("机会" in t):
                    return "scale"
                return "review"

            def _group_from_action_type(action_type: str, blocked: int) -> str:
                t = str(action_type or "").strip().upper()
                if t in {"NEGATE", "BID_DOWN"}:
                    return "stop"
                if t in {"BID_UP", "BUDGET_UP"}:
                    # 放量被阻断（库存/断货）时，优先当作“排查/解锁”
                    return "review" if int(blocked or 0) > 0 else "scale"
                return "review"

            def _pick_unlock_tasks(df: pd.DataFrame, n: int) -> pd.DataFrame:
                if df is None or df.empty:
                    return pd.DataFrame()
                if "asin" not in df.columns:
                    return pd.DataFrame()
                n2 = max(1, min(5, int(n)))  # 防御性：最多也就 5 条
                ut = getattr(policy, "dashboard_unlock_tasks", None) if isinstance(policy, OpsPolicy) else None
                prefer_unique_asin = True
                try:
                    prefer_unique_asin = bool(getattr(ut, "dashboard_prefer_unique_asin", True)) if ut is not None else True
                except Exception:
                    prefer_unique_asin = True

                picked_rows: List[Dict[str, object]] = []
                picked_idx: set[int] = set()
                seen_types: set[str] = set()
                seen_asins: set[str] = set()
                for i, (_, r) in enumerate(df.iterrows()):
                    if i > 5000:
                        break
                    task_type = str(r.get("task_type", "") or "").strip()
                    if task_type and task_type not in seen_types:
                        picked_rows.append(r.to_dict())
                        picked_idx.add(i)
                        seen_types.add(task_type)
                        if prefer_unique_asin:
                            asin = str(r.get("asin", "") or "").strip().upper()
                            if asin:
                                seen_asins.add(asin)
                    if len(picked_rows) >= n2:
                        break
                if len(picked_rows) < n2 and prefer_unique_asin:
                    for i, (_, r) in enumerate(df.iterrows()):
                        if i > 5000:
                            break
                        if i in picked_idx:
                            continue
                        asin = str(r.get("asin", "") or "").strip().upper()
                        if asin and asin in seen_asins:
                            continue
                        picked_rows.append(r.to_dict())
                        picked_idx.add(i)
                        if asin:
                            seen_asins.add(asin)
                        if len(picked_rows) >= n2:
                            break
                if len(picked_rows) < n2:
                    for i, (_, r) in enumerate(df.iterrows()):
                        if i > 5000:
                            break
                        if i in picked_idx:
                            continue
                        picked_rows.append(r.to_dict())
                        picked_idx.add(i)
                        if len(picked_rows) >= n2:
                            break
                return pd.DataFrame(picked_rows).copy() if picked_rows else df.head(n2).copy()

            def _owner_from_alert(title: str) -> str:
                t = str(title or "")
                if ("断货" in t) or ("库存" in t):
                    return "供应链/广告运营"
                if ("利润" in t) or ("毛利" in t) or ("客单" in t):
                    return "运营/财务/广告运营"
                if ("转化" in t) or ("自然" in t) or ("评价" in t) or ("Listing" in t):
                    return "运营/广告运营"
                if ("花费" in t) or ("ACOS" in t) or ("广告" in t):
                    return "广告运营"
                return "运营/广告运营"

            # 先把“可执行线索”统一收集成 candidates，再挑 Top3（避免 P0 告警被“放量解锁”淹没）
            candidates: List[Dict[str, object]] = []

            # A) unlock_scale_tasks：更贴近“广告×销量×库存×生命周期×利润”的立体决策
            if isinstance(unlock_scale_tasks, pd.DataFrame) and (not unlock_scale_tasks.empty):
                view = _pick_unlock_tasks(unlock_scale_tasks, n=5)
                for _, r in view.iterrows():
                    p = str(r.get("priority", "P1") or "P1").strip().upper()
                    owner = str(r.get("owner", "") or "").strip() or "广告运营"
                    asin = str(r.get("asin", "") or "").strip().upper()
                    cat = str(r.get("product_category", "") or "").strip()
                    task_type = str(r.get("task_type", "") or "").strip()

                    group = "scale"
                    if ("供应链" in owner) or ("库存" in task_type) or ("断货" in task_type):
                        group = "review"

                    ev_parts = []
                    spend_val = 0.0
                    delta_val = 0.0
                    spd7 = _fmt_num(r.get("sales_per_day_7d"), nd=2)
                    cov7 = _fmt_num(r.get("inventory_cover_days_7d"), nd=1)
                    bg = _fmt_usd(r.get("budget_gap_usd_est"))
                    pg = _fmt_usd(r.get("profit_gap_usd_est"))
                    if spd7:
                        ev_parts.append(f"日销7d=`{spd7}`")
                        delta_val = abs(_as_float(r.get("sales_per_day_7d")))
                    if cov7:
                        ev_parts.append(f"cover7d=`{cov7}`")
                    if bg and bg != "$0":
                        ev_parts.append(f"预算缺口≈`{bg}`")
                        spend_val += abs(_as_float(r.get("budget_gap_usd_est")))
                    if pg and pg != "$0":
                        ev_parts.append(f"利润缺口≈`{pg}`")
                        spend_val += abs(_as_float(r.get("profit_gap_usd_est")))
                    ev_txt = _evidence_text(ev_parts)

                    prefix = f"{cat} " if cat else ""
                    product_label = _format_product_label(asin, r.get("product_name", ""))
                    title = f"{prefix}{product_label} {task_type}".strip()
                    if not title:
                        continue
                    candidates.append(
                        {
                            "priority": p,
                            "group": group,
                            "asin": asin,
                            "spend": float(spend_val),
                            "delta": float(delta_val),
                            "line": f"`{p}` `{_group_tag(group)}` {title} | {ev_txt} | 责任:{owner}",
                        }
                    )

            # B) Shop Alerts：更贴近“近期出现的问题”
            if alerts:
                for a in alerts[:10]:
                    p = str(a.get("priority", "P1") or "P1").strip().upper()
                    title = str(a.get("title", "") or "").strip()
                    detail = str(a.get("detail", "") or "").strip()
                    if not title:
                        continue
                    owner = _owner_from_alert(title)
                    group = _group_from_alert(title)
                    ev_txt = _evidence_text([detail] if detail else [])
                    line = f"`{p}` `{_group_tag(group)}` {title} | {ev_txt} | 责任:{owner}"
                    candidates.append({"priority": p, "group": group, "asin": "", "spend": 0.0, "delta": 0.0, "line": line})

            # C) Action Board：广告端可直接执行（补齐“止损/放量/排查”）
            if isinstance(action_board, pd.DataFrame) and (not action_board.empty):
                ab = action_board.copy()
                if "priority" not in ab.columns:
                    ab["priority"] = "P1"
                ab["_pr"] = ab["priority"].astype(str).map(lambda x: _priority_rank(str(x)))
                if "action_priority_score" in ab.columns:
                    ab["_score"] = pd.to_numeric(ab["action_priority_score"], errors="coerce").fillna(0.0)
                else:
                    ab["_score"] = 0.0
                if "e_spend" in ab.columns:
                    ab["_spend"] = pd.to_numeric(ab["e_spend"], errors="coerce").fillna(0.0)
                else:
                    ab["_spend"] = 0.0
                ab = ab.sort_values(["_pr", "_spend", "_score"], ascending=[True, False, False])
                for _, r in ab.head(30).iterrows():
                    p = str(r.get("priority", "P1") or "P1").strip().upper()
                    action_type = str(r.get("action_type", "") or "").strip()
                    obj = str(r.get("object_name", "") or "").strip()
                    camp = str(r.get("campaign", "") or "").strip()
                    asin = str(r.get("asin_hint", "") or "").strip().upper()
                    product_label = _format_product_label(asin, r.get("product_name", ""))

                    blocked = 0
                    try:
                        blocked = int(float(r.get("blocked", 0) or 0))
                    except Exception:
                        blocked = 0
                    blocked_reason = str(r.get("blocked_reason", "") or "").strip()

                    group = _group_from_action_type(action_type, blocked=blocked)
                    owner = "广告运营" if group != "review" else "运营/广告运营"
                    if blocked > 0 and blocked_reason:
                        owner = "供应链/广告运营"

                    needs_confirm = False
                    try:
                        needs_confirm = bool(int(r.get("needs_manual_confirm") or 0))
                    except Exception:
                        needs_confirm = False

                    e_spend = _fmt_usd(r.get("e_spend"))
                    e_orders = _fmt_num(r.get("e_orders"), nd=0)
                    e_acos = _fmt_num(r.get("e_acos"), nd=4)
                    delta_sales_val = _as_float(r.get("asin_delta_sales"))
                    delta_spend_val = _as_float(r.get("asin_delta_spend"))
                    delta_sales = _fmt_signed_usd(delta_sales_val) if delta_sales_val != 0 else ""
                    delta_spend = _fmt_signed_usd(delta_spend_val) if delta_spend_val != 0 else ""

                    ev_parts = []
                    if e_spend:
                        ev_parts.append(f"花费=`{e_spend}`")
                    if delta_sales:
                        ev_parts.append(f"ΔSales=`{delta_sales}`")
                    if delta_spend:
                        ev_parts.append(f"ΔSpend=`{delta_spend}`")
                    if e_orders:
                        ev_parts.append(f"订单=`{e_orders}`")
                    if e_acos:
                        ev_parts.append(f"ACOS=`{e_acos}`")
                    if blocked > 0 and blocked_reason:
                        ev_parts.append(f"阻断=`{blocked_reason}`")
                    ev_txt = _evidence_text(ev_parts)

                    core = f"{action_type} {obj}".strip() if action_type or obj else "广告动作"
                    if product_label:
                        core = f"{product_label} {core}".strip()
                    if camp:
                        core += f" @ {camp}"
                    if needs_confirm:
                        core += " [需确认]"

                    candidates.append(
                        {
                            "priority": p,
                            "group": group,
                            "asin": asin,
                            "spend": float(_as_float(r.get("e_spend"))),
                            "delta": float(abs(delta_sales_val) if delta_sales_val != 0 else abs(delta_spend_val)),
                            "line": f"`{p}` `{_group_tag(group)}` {core} | {ev_txt} | 责任:{owner}",
                        }
                    )

            # 选 Top3：先取全局最高优先级，再尽量覆盖不同 group（止损/放量/排查）
            remaining = [c for c in candidates if str(c.get("line", "") or "").strip()]
            picked: List[Dict[str, str]] = []
            used_asins: set[str] = set()
            used_groups: set[str] = set()
            used_lines: set[str] = set()

            def _pop_best(predicate) -> None:
                nonlocal remaining
                best_idx = -1
                best_key = None
                for i, c in enumerate(remaining):
                    if not predicate(c):
                        continue
                    line = str(c.get("line", "") or "").strip()
                    if not line or line in used_lines:
                        continue
                    asin = str(c.get("asin", "") or "").strip().upper()
                    if asin and asin in used_asins:
                        continue
                    pr = _priority_rank(str(c.get("priority", "P1") or "P1"))
                    gr = _group_rank(str(c.get("group", "") or ""))
                    spend_v = float(c.get("spend", 0.0) or 0.0)
                    delta_v = float(c.get("delta", 0.0) or 0.0)
                    key = (pr, gr, -spend_v, -delta_v)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = i
                if best_idx < 0:
                    return
                c = remaining.pop(best_idx)
                line = str(c.get("line", "") or "").strip()
                p = str(c.get("priority", "P1") or "P1").strip().upper()
                asin = str(c.get("asin", "") or "").strip().upper()
                group = str(c.get("group", "") or "").strip().lower() or "review"
                used_lines.add(line)
                if asin:
                    used_asins.add(asin)
                used_groups.add(group)
                picked.append({"priority": p, "line": line})

            _pop_best(lambda c: True)
            while len(picked) < 3:
                # 先尝试“不同 group”以保证覆盖面
                before = len(picked)
                _pop_best(lambda c: str(c.get("group", "") or "").strip().lower() not in used_groups)
                if len(picked) == before:
                    _pop_best(lambda c: True)
                if len(picked) == before:
                    break

            weekly_actions = picked[:3]
        except Exception:
            has_weekly_tasks = False
            weekly_actions = []

        if weekly_actions:
            has_weekly_tasks = True
            lines.append(f"- 本周行动清单：Top {len(weekly_actions)}（见下方）")
        else:
            has_weekly_tasks = False

        # 风险/问题：从 Shop Alerts 抽取 P0/P1 作为“结论层”（最多 3 条，避免重复铺满）
        picked_alerts: List[Dict[str, object]] = []
        max_alerts = 1 if has_weekly_tasks else (2 if has_ops_plans else 3)
        try:
            for a in alerts:
                p = str(a.get("priority", "P1") or "P1").strip().upper()
                if p in {"P0", "P1"}:
                    picked_alerts.append(a)
                if len(picked_alerts) >= max_alerts:
                    break
        except Exception:
            picked_alerts = []
        if picked_alerts:
            for a in picked_alerts:
                p = str(a.get("priority", "P1") or "P1").strip().upper()
                title = str(a.get("title", "") or "").strip()
                detail = str(a.get("detail", "") or "").strip()
                link = str(a.get("link", "") or "").strip()
                if title and detail and link:
                    lines.append(f"- `{p}` {title}：证据: {detail}（{link}）")
                elif title and link:
                    lines.append(f"- `{p}` {title}（{link}）")
                elif title and detail:
                    lines.append(f"- `{p}` {title}：证据: {detail}")
                elif title:
                    lines.append(f"- `{p}` {title}")
        else:
            # 防御性兜底：至少给一个“下一步”
            lines.append("- （暂无显著告警；建议先看快速入口）")

        # ===== 阶段化指标区块（新品期 / 成熟期）=====
        lines.append("")
        lines.append("## 2) 阶段化指标（启动/成长/成熟）")
        lines.append("")
        lines.append("- 启动期/成长期：优先看 CTR/CVR/CPA（订单口径）与流量")
        lines.append("- 成熟期：优先看 CPA/ACOS 与广告占比、花费稳定性")
        lines.append("")

        try:
            af = asin_focus.copy() if isinstance(asin_focus, pd.DataFrame) else pd.DataFrame()
            if not af.empty and "stage_group" in af.columns:
                # 只取需要展示的少量列，避免第一屏信息过载
                stage_map = {
                    "asin": "ASIN",
                    "product_name": "商品名",
                    "product_category": "类目",
                    "sessions_recent_7d": "Sessions(7d)",
                    "ad_ctr_recent_7d": "广告CTR(7d,%)",
                    "ad_cvr_recent_7d": "广告CVR(7d,%)",
                    "ad_cpa_order_recent_7d": "CPA(订单,USD)",
                    "ad_sales_share_recent_7d": "广告占比(7d,%)",
                    "ad_acos_recent_7d": "广告ACOS(7d,%)",
                    "ad_spend_recent_7d": "广告花费(7d,USD)",
                    "stage_focus_reasons": "阶段提示",
                }

                def _fmt_num(x: object, nd: int = 2) -> str:
                    try:
                        v = float(pd.to_numeric(x, errors="coerce"))
                        if pd.isna(v):
                            return ""
                        s = f"{v:.{nd}f}"
                        return s.rstrip("0").rstrip(".")
                    except Exception:
                        return str(x or "").strip()

                def _fmt_usd(x: object) -> str:
                    s = _fmt_num(x, nd=2)
                    return f"${s}" if s else ""

                def _fmt_pct(x: object, nd: int = 1) -> str:
                    try:
                        v = float(pd.to_numeric(x, errors="coerce"))
                        if pd.isna(v):
                            return ""
                        return f"{v * 100:.{nd}f}%"
                    except Exception:
                        return str(x or "").strip()

                def _stage_table(df: pd.DataFrame, group_name: str, cols: List[str]) -> str:
                    sub = df[df["stage_group"] == group_name].copy()
                    if sub.empty:
                        return ""
                    # 排序：阶段分>focus_score（更贴近“阶段问题”）
                    if "stage_focus_score" in sub.columns:
                        sub = sub.sort_values(["stage_focus_score", "focus_score"], ascending=[False, False])
                    elif "focus_score" in sub.columns:
                        sub = sub.sort_values(["focus_score"], ascending=[False])
                    sub = sub.head(5).copy()
                    sub_fmt = sub.copy()
                    for c in ("sessions_recent_7d",):
                        if c in sub_fmt.columns:
                            sub_fmt[c] = pd.to_numeric(sub_fmt[c], errors="coerce").fillna(0).astype(int)
                    for c in (
                        "ad_ctr_recent_7d",
                        "ad_cvr_recent_7d",
                        "ad_sales_share_recent_7d",
                        "ad_acos_recent_7d",
                    ):
                        if c in sub_fmt.columns:
                            sub_fmt[c] = sub_fmt[c].map(lambda x: _fmt_pct(x, nd=1))
                    for c in ("ad_cpa_order_recent_7d", "ad_spend_recent_7d"):
                        if c in sub_fmt.columns:
                            sub_fmt[c] = sub_fmt[c].map(_fmt_usd)
                    return _display_table(sub_fmt, cols, stage_map)

                # 新品期 Top
                lines.append("### 启动期 Top（launch）")
                lines.append("")
                launch_cols = [
                    "asin",
                    "product_name",
                    "product_category",
                    "sessions_recent_7d",
                    "ad_ctr_recent_7d",
                    "ad_cvr_recent_7d",
                    "ad_cpa_order_recent_7d",
                    "stage_focus_reasons",
                ]
                t_launch = _stage_table(af, "launch", launch_cols)
                if t_launch:
                    lines.append(t_launch)
                else:
                    lines.append("- （暂无）")

                lines.append("")
                lines.append("### 成长期 Top（growth）")
                lines.append("")
                growth_cols = [
                    "asin",
                    "product_name",
                    "product_category",
                    "sessions_recent_7d",
                    "ad_ctr_recent_7d",
                    "ad_cvr_recent_7d",
                    "ad_cpa_order_recent_7d",
                    "stage_focus_reasons",
                ]
                t_growth = _stage_table(af, "growth", growth_cols)
                if t_growth:
                    lines.append(t_growth)
                else:
                    lines.append("- （暂无）")

                lines.append("")
                lines.append("### 成熟期 Top（mature + stable）")
                lines.append("")
                mature_cols = [
                    "asin",
                    "product_name",
                    "product_category",
                    "ad_sales_share_recent_7d",
                    "ad_acos_recent_7d",
                    "ad_cpa_order_recent_7d",
                    "ad_spend_recent_7d",
                    "stage_focus_reasons",
                ]
                t_mature = _stage_table(af, "mature", mature_cols)
                if t_mature:
                    lines.append(t_mature)
                else:
                    lines.append("- （暂无）")
            else:
                lines.append("- （暂无阶段化指标数据）")
        except Exception:
            lines.append("- （阶段化指标生成失败）")

        # 产品侧变化摘要（自然/流量）
        lines.append("")
        lines.append('<a id="product_changes"></a>')
        lines.append("### 产品侧变化摘要（近7天 vs 前7天）")
        lines.append("")
        lines.append("- 说明：按 |Δ| 排序，仅用于抓重点。")
        try:
            af = asin_focus.copy() if isinstance(asin_focus, pd.DataFrame) else pd.DataFrame()
            if af is None or af.empty:
                lines.append("- （暂无）")
            else:
                def _top_changes(df: pd.DataFrame, delta_col: str, prev_col: str, recent_col: str, title: str, mapping: Dict[str, str]) -> None:
                    if df is None or df.empty or delta_col not in df.columns:
                        lines.append(f"- {title}：暂无")
                        return
                    view = df.copy()
                    for c in (delta_col, prev_col, recent_col, "signal_confidence"):
                        if c in view.columns:
                            view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0.0)
                    if "asin" in view.columns:
                        view["asin"] = view["asin"].map(lambda x: _asin_md_link(str(x or ""), "./asin_drilldown.md"))
                    if "product_category" in view.columns:
                        view["product_category"] = view["product_category"].map(
                            lambda x: _cat_md_link(str(x or ""), "./category_drilldown.md")
                        )
                    view["_abs_delta"] = pd.to_numeric(view[delta_col], errors="coerce").fillna(0.0).abs()
                    view = view.sort_values("_abs_delta", ascending=False).head(5).drop(columns=["_abs_delta"], errors="ignore")
                    show_cols = [c for c in ["asin", "product_name", "product_category", prev_col, recent_col, delta_col, "signal_confidence"] if c in view.columns]
                    lines.append(f"#### {title}")
                    lines.append("")
                    if view.empty:
                        lines.append("- （暂无）")
                    else:
                        lines.append(_display_table(view, show_cols, mapping))
                    lines.append("")

                organic_map = {
                    "asin": "ASIN",
                    "product_name": "品名",
                    "product_category": "类目",
                    "organic_sales_prev_7d": "自然销售(前7天)",
                    "organic_sales_recent_7d": "自然销售(近7天)",
                    "delta_organic_sales": "自然销售Δ",
                    "signal_confidence": "置信度",
                }
                session_map = {
                    "asin": "ASIN",
                    "product_name": "品名",
                    "product_category": "类目",
                    "sessions_prev_7d": "Sessions(前7天)",
                    "sessions_recent_7d": "Sessions(近7天)",
                    "delta_sessions": "SessionsΔ",
                    "signal_confidence": "置信度",
                }
                _top_changes(af, "delta_organic_sales", "organic_sales_prev_7d", "organic_sales_recent_7d", "自然销售变化 Top5", organic_map)
                _top_changes(af, "delta_sessions", "sessions_prev_7d", "sessions_recent_7d", "总流量变化 Top5", session_map)
        except Exception:
            lines.append("- （生成失败）")

        # 本周行动清单：Top 3（每条必须包含：责任归属 + 关键证据 + 跳转链接）
        lines.append("")
        lines.append('<a id="weekly"></a>')
        lines.append("### 本周行动清单（Top 3）")
        lines.append("")
        if weekly_actions:
            for it in weekly_actions[:3]:
                lines.append("- " + str(it.get("line", "") or "").strip())
        else:
            lines.append("- （暂无可收敛的本周行动清单；建议先看快速入口）")

        # Campaign 优先排查：让运营先按“外层 campaign”收口，再下钻到词/ASIN
        lines.append("")
        lines.append('<a id="campaign"></a>')
        lines.append("### Campaign 优先排查（Top 3）")
        lines.append("")
        try:
            camp_view = campaign_action_view.copy() if isinstance(campaign_action_view, pd.DataFrame) else pd.DataFrame()
            if camp_view is None or camp_view.empty:
                lines.append("- （暂无）")
            else:
                cv = camp_view.copy()
                for c in ("action_count", "p0_count", "p1_count", "p2_count", "blocked_count"):
                    if c in cv.columns:
                        cv[c] = pd.to_numeric(cv[c], errors="coerce").fillna(0).astype(int)
                for c in ("spend_sum", "delta_sales_sum", "delta_spend_sum"):
                    if c in cv.columns:
                        cv[c] = pd.to_numeric(cv[c], errors="coerce").fillna(0.0)
                # 排序：先 P0，再花费，再 ΔSales
                try:
                    cv = cv.sort_values(["p0_count", "spend_sum", "delta_sales_sum"], ascending=[False, False, False])
                except Exception:
                    pass

                for _, r in cv.head(3).iterrows():
                    ad_type = str(r.get("ad_type", "") or "").strip()
                    camp = str(r.get("campaign", "") or "").strip()
                    p0 = int(r.get("p0_count", 0) or 0)
                    p1 = int(r.get("p1_count", 0) or 0)
                    cnt = int(r.get("action_count", 0) or 0)
                    blocked_cnt = int(r.get("blocked_count", 0) or 0)
                    spend = _fmt_usd(r.get("spend_sum"))
                    delta_sales = _fmt_signed_usd(r.get("delta_sales_sum"))
                    delta_spend = _fmt_signed_usd(r.get("delta_spend_sum"))
                    top_asins_raw = str(r.get("top_asins", "") or "").strip()
                    top_products = ""
                    if top_asins_raw:
                        tops = [x for x in top_asins_raw.split(";") if x]
                        labels: List[str] = []
                        for a in tops[:2]:
                            label = _compact_product_label(a, "")
                            if label:
                                labels.append(label)
                        top_products = ";".join(labels)

                    parts = []
                    if ad_type:
                        parts.append(f"`{ad_type}`")
                    if camp:
                        parts.append(camp)
                    parts.append(f"P0=`{p0}` P1=`{p1}` 动作=`{cnt}`")
                    ev_parts = []
                    if spend:
                        ev_parts.append(f"花费=`{spend}`")
                    if delta_sales:
                        ev_parts.append(f"ΔSales=`{delta_sales}`")
                    if delta_spend:
                        ev_parts.append(f"ΔSpend=`{delta_spend}`")
                    if blocked_cnt > 0:
                        ev_parts.append(f"阻断=`{blocked_cnt}`")
                    if ev_parts:
                        parts.append(f"证据: {' | '.join(ev_parts)}")
                    if top_products:
                        parts.append(f"Top产品=`{top_products}`")
                    lines.append("- " + " | ".join([p for p in parts if p]))
        except Exception:
            lines.append("- （暂无）")

        # 规则化告警（Top 5）：避免 dashboard 变成“指标堆叠”，先把最关键的风险/机会露出来
        lines.append("")
        lines.append('<a id="alerts"></a>')
        lines.append("### Shop Alerts（规则化 Top 5）")
        lines.append("")
        if not alerts:
            lines.append("- （无显著告警）")
        else:
            for a in alerts:
                p = str(a.get("priority", "P1") or "P1").strip().upper()
                title = str(a.get("title", "") or "").strip()
                detail = str(a.get("detail", "") or "").strip()
                link = str(a.get("link", "") or "").strip()
                if link:
                    lines.append(f"- `{p}` {title}：证据: {detail}（{link}）")
                else:
                    lines.append(f"- `{p}` {title}：证据: {detail}")

        # L0+ 执行复盘（可选）：基于 execution_log 回填，回答“上次做了什么？有没有效果？有什么异常？”
        try:
            if isinstance(action_review, pd.DataFrame) and (not action_review.empty):
                ar = action_review.copy()
                if "window_days" in ar.columns:
                    ar["window_days"] = pd.to_numeric(ar["window_days"], errors="coerce").fillna(0).astype(int)
                else:
                    ar["window_days"] = 0
                if "status" in ar.columns:
                    ar["status"] = ar["status"].astype(str).fillna("")
                else:
                    ar["status"] = ""

                lines.append("")
                lines.append('<a id="review"></a>')
                lines.append("### L0+ 执行复盘（近7/14天） - [打开复盘表](../ops/action_review.csv)")
                lines.append("")
                lines.append("- 说明：只复盘 execution_log 里勾选 `executed=1` 且 `executed_at` 合法的动作（after vs before）。")

                # 1) 汇总：每个窗口一行
                summary_rows: List[Dict[str, object]] = []
                for w in sorted([int(x) for x in ar["window_days"].unique().tolist() if int(x) > 0]):
                    sub = ar[ar["window_days"] == int(w)].copy()
                    try:
                        total_actions = int(sub["action_key"].nunique()) if "action_key" in sub.columns else int(len(sub))
                    except Exception:
                        total_actions = int(len(sub))
                    vc = sub["status"].value_counts().to_dict() if "status" in sub.columns else {}
                    ok_cnt = int(vc.get("ok", 0) or 0)
                    not_found_cnt = int(vc.get("not_found", 0) or 0)
                    insufficient_cnt = int(vc.get("insufficient_data", 0) or 0)
                    ok_df = sub[sub["status"] == "ok"].copy()

                    def _sum_col(df: pd.DataFrame, col: str) -> float:
                        if df is None or df.empty or col not in df.columns:
                            return 0.0
                        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())

                    def _median_col(df: pd.DataFrame, col: str) -> float:
                        if df is None or df.empty or col not in df.columns:
                            return 0.0
                        try:
                            v = pd.to_numeric(df[col], errors="coerce")
                            m = float(v.median()) if (v is not None) else 0.0
                            return 0.0 if (pd.isna(m) if hasattr(pd, "isna") else False) else m  # type: ignore[arg-type]
                        except Exception:
                            return 0.0

                    delta_spend_sum = _sum_col(ok_df, "delta_spend")
                    delta_sales_sum = _sum_col(ok_df, "delta_sales")
                    delta_acos_median = _median_col(ok_df, "delta_acos")
                    risk_cnt = 0
                    try:
                        if (not ok_df.empty) and ("delta_spend" in ok_df.columns) and ("delta_sales" in ok_df.columns):
                            ds = pd.to_numeric(ok_df["delta_spend"], errors="coerce").fillna(0.0)
                            dsl = pd.to_numeric(ok_df["delta_sales"], errors="coerce").fillna(0.0)
                            risk_cnt = int(((ds > 0) & (dsl <= 0)).sum())
                    except Exception:
                        risk_cnt = 0

                    summary_rows.append(
                        {
                            "window_days": int(w),
                            "actions": int(total_actions),
                            "ok": int(ok_cnt),
                            "not_found": int(not_found_cnt),
                            "insufficient": int(insufficient_cnt),
                            "risk_spend_up_no_sales": int(risk_cnt),
                            "delta_spend_sum": round(float(delta_spend_sum), 2),
                            "delta_sales_sum": round(float(delta_sales_sum), 2),
                            "delta_acos_median": round(float(delta_acos_median), 4),
                        }
                    )

                if summary_rows:
                    s_df = pd.DataFrame(summary_rows)
                    show_cols = [
                        "window_days",
                        "actions",
                        "ok",
                        "not_found",
                        "insufficient",
                        "risk_spend_up_no_sales",
                        "delta_spend_sum",
                        "delta_sales_sum",
                        "delta_acos_median",
                    ]
                    lines.append(_df_to_md_table(s_df, show_cols))
                else:
                    lines.append("- （复盘表为空：可能本次 execution_log 没有勾选 executed=1，或 executed_at 不合法）")

                # 2) 异常提示：not_found / insufficient_data
                try:
                    nf = ar[ar["status"] == "not_found"].copy()
                    if nf is not None and not nf.empty:
                        ex = []
                        for _, r in nf.head(3).iterrows():
                            ex.append(f"{r.get('level','')}/{r.get('action_type','')}: {r.get('object_name','')}")
                        ex = [str(x).strip() for x in ex if str(x).strip()]
                        if ex:
                            lines.append(f"- not_found 示例（Top {len(ex)}）：{'；'.join(ex)}（通常是名称变更/字段缺失）")
                except Exception:
                    pass
                try:
                    ins = ar[ar["status"] == "insufficient_data"].copy()
                    if ins is not None and not ins.empty:
                        lines.append(
                            f"- insufficient_data：{int(len(ins))} 条（提示：复盘需要 before/after 窗口都在数据范围内；如果跑数时使用了 `--days`，可能导致 before 窗口被裁掉）"
                        )
                except Exception:
                    pass
        except Exception:
            pass

        # 五大 Watchlist 摘要：风险 4 + 机会 1（只展示 ASIN + 花费/Δ + 1 个关键维度，避免第一屏信息爆炸）
        lines.append("")
        lines.append('<a id="watchlists"></a>')
        lines.append("### 五大 Watchlist 摘要（Top 5）")
        lines.append("")
        lines.append("- 表头释义：`roll`=滚动窗口；`cover7d`=库存覆盖天数(7d)；`原因1/原因2`=规则标签")
        lines.append("")
        try:
            ac = asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None

            def _fmt(df: pd.DataFrame, float_cols: List[str], int_cols: List[str]) -> pd.DataFrame:
                v = df.copy()
                if "asin" in v.columns:
                    v["asin"] = v["asin"].map(lambda x: _asin_md_link(str(x or ""), "./asin_drilldown.md"))
                for c in float_cols:
                    if c in v.columns:
                        v[c] = pd.to_numeric(v[c], errors="coerce").fillna(0.0).round(2)
                for c in int_cols:
                    if c in v.columns:
                        v[c] = pd.to_numeric(v[c], errors="coerce").fillna(0).astype(int)
                return v

            def _ensure_reason_cols(df: pd.DataFrame) -> pd.DataFrame:
                v = df.copy()
                if "reason_1" in v.columns and "reason_2" in v.columns:
                    return v
                reason_candidates = []
                for c in ("reason", "reasons", "focus_reasons", "focus_reasons_history", "stage_focus_reasons"):
                    if c in v.columns:
                        reason_candidates.append(c)
                if not reason_candidates:
                    v["reason_1"] = v.get("reason_1", "")
                    v["reason_2"] = v.get("reason_2", "")
                    return v
                src = reason_candidates[0]
                try:
                    parts = v[src].astype(str).fillna("").map(lambda x: [p for p in str(x).split(";") if str(p).strip()])
                    v["reason_1"] = parts.map(lambda xs: xs[0] if isinstance(xs, list) and len(xs) >= 1 else "")
                    v["reason_2"] = parts.map(lambda xs: xs[1] if isinstance(xs, list) and len(xs) >= 2 else "")
                except Exception:
                    v["reason_1"] = v.get("reason_1", "")
                    v["reason_2"] = v.get("reason_2", "")
                return v

            wl_map = {
                "asin": "ASIN",
                "product_name": "品名",
                "product_category": "商品分类",
                "ad_spend_roll": "广告花费(滚动)",
                "over_profit_cap": "超利润上限",
                "inventory_cover_days_7d": "库存覆盖(7天)",
                "sigmoid_modifier": "调速系数",
                "sigmoid_action": "调速建议",
                "gross_margin": "毛利率",
                "safe_acos": "安全ACOS",
                "ad_acos_recent": "实际ACOS(近7天)",
                "ad_cpc_recent_7d": "实际CPC(近7天)",
                "safe_cpc": "安全CPC",
                "oos_with_ad_spend_days": "断货仍花费天数",
                "delta_spend": "花费Δ",
                "delta_sales": "销售Δ",
                "tacos_roll": "TACOS(滚动)",
                "reason_1": "原因1",
                "reason_2": "原因2",
            }

            # 1) 利润控量：over_profit_cap（越大越需要优先止血）
            lines.append(
                "#### 利润控量（reduce 且仍在烧钱） - [打开筛选表](../dashboard/profit_reduce_watchlist.csv) | [操作手册](../../../../docs/OPS_PLAYBOOK.md#scene-profit-reduce)"
            )
            lines.append("")
            lines.append("- reason 快速指引（先查哪里）：")
            lines.append("- `超利润上限`：优先止血/收口（看 `over_profit_cap/max_ad_spend_by_profit`）")
            lines.append("- `断货风险`/`库存风险`：先查补货节奏与断货（`oos_with_ad_spend_days/flag_oos/inventory_cover_days_7d`）")
            lines.append("- `加花费无增量`/`速度下降`：回看结构与关键词主题（`action_board.csv`/`keyword_topics_action_hints.csv`）")
            lines.append("")
            try:
                w1 = build_profit_reduce_watchlist(asin_cockpit=ac, max_rows=5, policy=policy)
            except Exception:
                w1 = pd.DataFrame()
            if w1 is None or w1.empty:
                lines.append("- （无）")
            else:
                w1v = _ensure_reason_cols(_fmt(w1, float_cols=["ad_spend_roll", "over_profit_cap"], int_cols=[]))
                lines.append(
                    _display_table(
                        w1v,
                        ["asin", "product_name", "product_category", "ad_spend_roll", "over_profit_cap", "reason_1", "reason_2"],
                        wl_map,
                    )
                )
            lines.append("")

            # 2) 库存告急仍投放：cover7d 低且仍在花费
            lines.append(
                "#### 库存告急仍投放（cover7d 低且仍在投放） - [打开筛选表](../dashboard/inventory_risk_watchlist.csv) | [操作手册](../../../../docs/OPS_PLAYBOOK.md#inventory-first)"
            )
            lines.append("")
            try:
                cover_days_thr = float(getattr(policy, "block_scale_when_cover_days_below", 7.0) or 7.0) if isinstance(policy, OpsPolicy) else 7.0
                spend_thr = 10.0
                lines.append(f"- 口径：`inventory_cover_days_7d ≤ {int(cover_days_thr)}d` 且 `ad_spend_roll ≥ {int(spend_thr)}`")
            except Exception:
                pass
            lines.append("- reason 快速指引（先查哪里）：")
            lines.append("- `库存告急`：优先控量/降速，避免烧完库存")
            lines.append("- `消耗加速`：关注补货与实际销量节奏")
            lines.append("- `仍在投放`：排查预算是否需要临时收口")
            lines.append("")
            try:
                w2 = build_inventory_risk_watchlist(asin_cockpit=ac, max_rows=5, policy=policy, spend_threshold=10.0)
            except Exception:
                w2 = pd.DataFrame()
            if w2 is None or w2.empty:
                lines.append("- （无）")
            else:
                w2v = _ensure_reason_cols(_fmt(w2, float_cols=["ad_spend_roll", "inventory_cover_days_7d"], int_cols=[]))
                lines.append(
                    _display_table(
                        w2v,
                        ["asin", "product_name", "product_category", "ad_spend_roll", "inventory_cover_days_7d", "reason_1", "reason_2"],
                        wl_map,
                    )
                )
            lines.append("")

            # 2.5) 补充建议：库存调速（Sigmoid，仅建议）
            lines.append("### 补充建议（可选）")
            lines.append("")
            lines.append(
                "#### 库存调速建议（Sigmoid，仅建议） - [打开筛选表](../dashboard/inventory_sigmoid_watchlist.csv)"
            )
            lines.append("")
            lines.append("- 口径：基于 `inventory_cover_days_7d` 计算调速系数；仅提示，不影响排序/不自动执行。")
            try:
                w2s = build_inventory_sigmoid_watchlist(asin_cockpit=ac, max_rows=5, policy=policy)
            except Exception:
                w2s = pd.DataFrame()
            if w2s is None or w2s.empty:
                lines.append("- （无）")
            else:
                w2sv = _ensure_reason_cols(_fmt(w2s, float_cols=["ad_spend_roll", "inventory_cover_days_7d", "sigmoid_modifier"], int_cols=[]))
                lines.append(
                    _display_table(
                        w2sv,
                        ["asin", "product_name", "product_category", "inventory_cover_days_7d", "sigmoid_modifier", "sigmoid_action", "reason_1"],
                        wl_map,
                    )
                )
            lines.append("")

            # 2.6) 补充建议：利润护栏（Break-even）
            lines.append(
                "#### 利润护栏（Break-even） - [打开筛选表](../dashboard/profit_guard_watchlist.csv)"
            )
            lines.append("")
            lines.append("- 口径：安全ACOS = 毛利率 - 目标净利率；当实际 ACOS 超线时提示。")
            try:
                w2p = build_profit_guard_watchlist(asin_cockpit=ac, max_rows=5, policy=policy)
            except Exception:
                w2p = pd.DataFrame()
            if w2p is None or w2p.empty:
                lines.append("- （无）")
            else:
                w2pv = _ensure_reason_cols(_fmt(w2p, float_cols=["gross_margin", "safe_acos", "ad_acos_recent", "ad_cpc_recent_7d", "safe_cpc", "ad_spend_roll"], int_cols=[]))
                lines.append(
                    _display_table(
                        w2pv,
                        ["asin", "product_name", "product_category", "gross_margin", "safe_acos", "ad_acos_recent", "ad_cpc_recent_7d", "safe_cpc", "reason_1"],
                        wl_map,
                    )
                )
            lines.append("")

            # 3) 加花费但销量不增：delta_spend（默认 compare_7d）
            lines.append(
                "#### 加花费但销量不增（delta_spend>0 且 delta_sales<=0） - [打开筛选表](../dashboard/spend_up_no_sales_watchlist.csv) | [操作手册](../../../../docs/OPS_PLAYBOOK.md#scene-spend-up-no-sales)"
            )
            lines.append("")
            lines.append("- reason 快速指引（先查哪里）：")
            lines.append("- `加花费无增量`：优先看 Action Board 与关键词主题（否词/降价/结构调整）")
            lines.append("- `断货风险`/`库存风险`：先排除供给侧问题（避免“没货/将断货还加码”）")
            lines.append("- `广告依赖高`：关注自然端（`ad_sales_share`），避免只靠广告硬顶")
            lines.append("")
            try:
                w3 = build_spend_up_no_sales_watchlist(asin_cockpit=ac, max_rows=5, policy=policy)
            except Exception:
                w3 = pd.DataFrame()
            if w3 is None or w3.empty:
                lines.append("- （无）")
            else:
                w3v = _ensure_reason_cols(_fmt(w3, float_cols=["ad_spend_roll", "delta_spend", "delta_sales"], int_cols=[]))
                lines.append(
                    _display_table(
                        w3v,
                        ["asin", "product_name", "product_category", "ad_spend_roll", "delta_spend", "delta_sales", "reason_1", "reason_2"],
                        wl_map,
                    )
                )
            lines.append("")

            # 4) 阶段走弱（近14天 down 且仍在花费）
            lines.append(
                "#### 阶段走弱（近14天 down 且仍在花费） - [打开筛选表](../dashboard/phase_down_recent_watchlist.csv) | [操作手册](../../../../docs/OPS_PLAYBOOK.md#scene-phase-down)"
            )
            lines.append("")
            lines.append("- reason 快速指引（先查哪里）：")
            lines.append("- `断货风险`/`库存风险`：先查库存/断货与补货节奏（`oos_with_ad_spend_days`/`flag_oos`/`inventory_cover_days_7d`）")
            lines.append("- `加花费无增量`/`速度下降`：先查转化&结构（`action_board.csv`）+ 关键词主题（`keyword_topics_action_hints.csv`）")
            lines.append("- `利润方向=控量`/`广告依赖高`：先把投放放回产品语境（`category_asin_compare.csv` + `profit_*` 字段）")
            lines.append("- `动作阻断多`：先处理阻断原因（库存风险/低置信/规则阻断）")
            lines.append("")
            try:
                w5 = build_phase_down_recent_watchlist(asin_cockpit=ac, max_rows=5, policy=policy)
            except Exception:
                w5 = pd.DataFrame()
            if w5 is None or w5.empty:
                lines.append("- （无）")
            else:
                w5v = _ensure_reason_cols(_fmt(w5, float_cols=["ad_spend_roll"], int_cols=["flag_oos", "flag_low_inventory", "oos_with_ad_spend_days"]))
                lines.append(
                    _display_table(
                        w5v,
                        ["asin", "product_name", "product_category", "ad_spend_roll", "reason_1", "reason_2"],
                        wl_map,
                    )
                )
            lines.append("")

            # 5) 机会：可放量窗口（低花费高潜候选）
            lines.append(
                "#### 机会：可放量窗口（ΔSales>0 且效率不错 且库存覆盖足够） - [打开筛选表](../dashboard/scale_opportunity_watchlist.csv) | [操作手册](../../../../docs/OPS_PLAYBOOK.md#scene-scale-opportunity)"
            )
            lines.append("")
            try:
                w4 = build_scale_opportunity_watchlist(asin_cockpit=ac, max_rows=5, policy=policy)
            except Exception:
                w4 = pd.DataFrame()
            if w4 is None or w4.empty:
                lines.append("- （无）")
            else:
                w4v = _ensure_reason_cols(_fmt(w4, float_cols=["delta_sales", "tacos_roll", "inventory_cover_days_7d"], int_cols=[]))
                lines.append(
                    _display_table(
                        w4v,
                        ["asin", "product_name", "product_category", "delta_sales", "tacos_roll", "inventory_cover_days_7d"],
                        wl_map,
                    )
                )
            lines.append("")

            # 历史诊断（可选）：断货仍烧钱
            lines.append("### 历史诊断（可选）")
            lines.append("")
            lines.append(
                "#### 断货仍烧钱（oos_with_ad_spend_days>0） - [打开筛选表](../dashboard/oos_with_ad_spend_watchlist.csv) | [操作手册](../../../../docs/OPS_PLAYBOOK.md#scene-oos-spend)"
            )
            lines.append("")
            try:
                w6 = build_oos_with_ad_spend_watchlist(asin_cockpit=ac, max_rows=5, policy=policy)
            except Exception:
                w6 = pd.DataFrame()
            if w6 is None or w6.empty:
                lines.append("- （无）")
            else:
                w6v = _ensure_reason_cols(_fmt(w6, float_cols=["ad_spend_roll"], int_cols=["oos_with_ad_spend_days"]))
                lines.append(
                    _display_table(
                        w6v,
                        ["asin", "product_name", "product_category", "ad_spend_roll", "oos_with_ad_spend_days", "reason_1", "reason_2"],
                        wl_map,
                    )
                )
            lines.append("")
        except Exception:
            lines.append("- （生成失败）")

        # 关键词主题：把海量搜索词压缩为可解释的主题列表（用于快速定位“在烧什么/在带量什么”）
        lines.append("")
        lines.append('<a id="keywords"></a>')
        lines.append(
            "### 关键词主题（n-gram） - [打开筛选表](../dashboard/keyword_topics.csv) | [操作手册](../../../../docs/OPS_PLAYBOOK.md#scene-keyword-topics)"
        )
        lines.append("")
        lines.append("- 口径提示：同一搜索词会贡献多个 n-gram，因此主题 spend 会有重复计数；该表仅用于线索与聚焦，不做精确归因。")
        lines.append("- 字段释义：n=词长；waste_spend=浪费花费；top_terms=代表搜索词；ctr/cvr/acos 为广告口径。")
        lines.append("- 下钻页：`./keyword_topics.md`（按步骤：Segment Top → Action Hints → ASIN Context）")
        lines.append("- Segment Top（先选类目/阶段再下钻）：`../dashboard/keyword_topics_segment_top.csv`（每行=类目×阶段；Top 浪费/贡献主题各 TopN）")
        lines.append("- 主题建议清单：`../dashboard/keyword_topics_action_hints.csv`（Top 浪费主题→否词/降价；Top 贡献主题→加精确/提价/加预算；含 top_campaigns/top_ad_groups；scale 方向会标注/阻断库存风险：blocked/blocked_reason）")
        lines.append("- 主题→产品语境：`../dashboard/keyword_topics_asin_context.csv`（只用高置信 term→asin；可直接看到该主题对应的类目/ASIN/生命周期/库存覆盖）")
        lines.append("- 主题→类目/生命周期汇总：`../dashboard/keyword_topics_category_phase_summary.csv`（先按类目/阶段看主题的 spend/sales/waste_spend，再下钻到 ASIN）")
        lines.append("")
        if keyword_topics is None or keyword_topics.empty:
            lines.append("- （无：缺少 search_term 报表或过滤后无有效搜索词）")
        else:
            try:
                kt = keyword_topics.copy()
                for c in ("spend", "sales", "acos", "waste_spend", "ctr", "cvr"):
                    if c in kt.columns:
                        kt[c] = pd.to_numeric(kt[c], errors="coerce").fillna(0.0).round(4 if c in {"acos", "ctr", "cvr"} else 2)
                for c in ("orders", "clicks", "impressions", "term_count", "waste_term_count", "n"):
                    if c in kt.columns:
                        kt[c] = pd.to_numeric(kt[c], errors="coerce").fillna(0).astype(int)

                # Dashboard 展示参数（默认 Top5 + n>=2；可用 ops_policy.json 调参）
                md_top_n = 5
                md_min_n = 2
                try:
                    if isinstance(policy, OpsPolicy):
                        ktp = getattr(policy, "dashboard_keyword_topics", None)
                        if ktp is not None:
                            md_top_n = int(getattr(ktp, "md_top_n", md_top_n) or md_top_n)
                            md_min_n = int(getattr(ktp, "md_min_n", md_min_n) or md_min_n)
                except Exception:
                    pass
                if md_top_n < 1:
                    md_top_n = 1
                if md_top_n > 20:
                    md_top_n = 20
                if md_min_n < 1:
                    md_min_n = 1
                if md_min_n > 5:
                    md_min_n = 5

                # Markdown 里优先展示更可解释的短语（n>=md_min_n）；如果没有则回退到全部
                kt_view = kt
                if "n" in kt_view.columns:
                    kt2 = kt_view[kt_view["n"] >= md_min_n].copy()
                    if not kt2.empty:
                        kt_view = kt2

                # 1) Top 浪费主题（waste_spend）
                lines.append(f"#### Top 浪费主题（waste_spend，Top {md_top_n}）")
                lines.append("")
                if "waste_spend" in kt_view.columns:
                    w = kt_view.sort_values(["waste_spend", "spend"], ascending=[False, False]).head(md_top_n).copy()
                else:
                    w = kt_view.head(0).copy()
                if w is None or w.empty:
                    lines.append("- （无）")
                else:
                    lines.append(
                        _df_to_md_table(
                            w,
                            [c for c in ["ad_type", "n", "ngram", "waste_spend", "spend", "orders", "acos", "top_terms"] if c in w.columns],
                        )
                    )
                lines.append("")

                # 2) Top 贡献主题（sales）
                lines.append(f"#### Top 贡献主题（sales，Top {md_top_n}）")
                lines.append("")
                if "sales" in kt_view.columns:
                    g = kt_view.sort_values(["sales", "spend"], ascending=[False, False]).head(md_top_n).copy()
                else:
                    g = kt_view.head(0).copy()
                if g is None or g.empty:
                    lines.append("- （无）")
                else:
                    lines.append(
                        _df_to_md_table(
                            g,
                            [c for c in ["ad_type", "n", "ngram", "sales", "spend", "orders", "acos", "top_terms"] if c in g.columns],
                        )
                    )
                lines.append("")
            except Exception:
                lines.append("- （生成失败）")

        # 变化来源 drivers：把“店铺变化”拆到 Top ASIN（运营第一屏先看为什么变了）
        lines.append("")
        lines.append('<a id="drivers"></a>')
        lines.append("### 变化来源（近7天 vs 前7天 Top ASIN） - [打开筛选表](../dashboard/drivers_top_asins.csv)")
        lines.append("")
        lines.append("- 运营版：只保留少列快速扫；更多维度请看 `../dashboard/drivers_top_asins.csv`")
        lines.append("- 字段释义：ΔSales/ΔAdSpend=近7天-前7天；marginal_tacos=ΔAdSpend/ΔSales（销售为0时以原表为准）")
        lines.append("")
        if drivers_top_asins is None or drivers_top_asins.empty:
            lines.append("- （无）")
        else:
            try:
                d = drivers_top_asins.copy()
                d["driver_type"] = d.get("driver_type", "").astype(str)

                # 小表：分别展示 Top ΔSales 与 Top ΔAdSpend
                def _show(df: pd.DataFrame) -> pd.DataFrame:
                    v = df.copy()
                    for c in ("delta_sales", "delta_ad_spend"):
                        if c in v.columns:
                            v[c] = pd.to_numeric(v[c], errors="coerce").fillna(0.0).round(2)
                    if "marginal_tacos" in v.columns:
                        v["marginal_tacos"] = pd.to_numeric(v["marginal_tacos"], errors="coerce").fillna(0.0).round(4)
                    return v

                by_sales = d[d["driver_type"] == "delta_sales"].head(5).copy()
                by_spend = d[d["driver_type"] == "delta_ad_spend"].head(5).copy()

                if by_sales is not None and not by_sales.empty:
                    if "asin" in by_sales.columns:
                        by_sales["asin"] = by_sales["asin"].map(lambda x: _asin_md_link(str(x or ""), "./asin_drilldown.md"))
                    if "product_category" in by_sales.columns:
                        by_sales["product_category"] = by_sales["product_category"].map(
                            lambda x: _cat_md_link(str(x or ""), "./category_drilldown.md")
                        )
                    if "current_phase" in by_sales.columns:
                        by_sales["current_phase"] = by_sales["current_phase"].map(
                            lambda x: _phase_md_link(str(x or ""), "./phase_drilldown.md")
                        )
                    lines.append("#### Top ΔSales")
                    lines.append("")
                    lines.append(
                        _df_to_md_table(
                            _show(by_sales),
                            [
                                "rank",
                                "product_category",
                                "asin",
                                "current_phase",
                                "delta_sales",
                                "delta_ad_spend",
                                "marginal_tacos",
                            ],
                        )
                    )
                    lines.append("")
                if by_spend is not None and not by_spend.empty:
                    if "asin" in by_spend.columns:
                        by_spend["asin"] = by_spend["asin"].map(lambda x: _asin_md_link(str(x or ""), "./asin_drilldown.md"))
                    if "product_category" in by_spend.columns:
                        by_spend["product_category"] = by_spend["product_category"].map(
                            lambda x: _cat_md_link(str(x or ""), "./category_drilldown.md")
                        )
                    if "current_phase" in by_spend.columns:
                        by_spend["current_phase"] = by_spend["current_phase"].map(
                            lambda x: _phase_md_link(str(x or ""), "./phase_drilldown.md")
                        )
                    lines.append("#### Top ΔAdSpend")
                    lines.append("")
                    lines.append(
                        _df_to_md_table(
                            _show(by_spend),
                            [
                                "rank",
                                "product_category",
                                "asin",
                                "current_phase",
                                "delta_ad_spend",
                                "delta_sales",
                                "marginal_tacos",
                            ],
                        )
                    )
            except Exception:
                lines.append("- （生成失败）")

        # ASIN 总览：把 focus + drivers + 动作量聚合到一张表（便于运营快速扫一遍）
        lines.append("")
        lines.append("### ASIN 总览（Top 20）")
        lines.append("")
        lines.append("- 运营版：只保留 8 列快速扫；更多维度请看 `../dashboard/asin_cockpit.csv`")
        lines.append("")
        if asin_cockpit is None or asin_cockpit.empty:
            lines.append("- （无）")
        else:
            try:
                c = asin_cockpit.copy()
                # 链接跳转：asin -> drilldown
                if "asin" in c.columns:
                    c["asin"] = c["asin"].map(lambda x: _asin_md_link(str(x or ""), "./asin_drilldown.md"))
                # 链接跳转：类目/生命周期 -> drilldown（运营更快定位）
                if "product_category" in c.columns:
                    c["product_category"] = c["product_category"].map(lambda x: _cat_md_link(str(x or ""), "./category_drilldown.md"))
                if "current_phase" in c.columns:
                    c["current_phase"] = c["current_phase"].map(lambda x: _phase_md_link(str(x or ""), "./phase_drilldown.md"))
                # 可读性：数值格式化
                for col in ("focus_score", "ad_spend_roll", "tacos_roll", "drivers_delta_sales", "drivers_delta_ad_spend"):
                    if col in c.columns:
                        c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0.0).round(2)
                for col in ("sales_per_day_7d", "orders_per_day_7d"):
                    if col in c.columns:
                        c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0.0).round(2)
                if "inventory_cover_days_7d" in c.columns:
                    c["inventory_cover_days_7d"] = pd.to_numeric(c["inventory_cover_days_7d"], errors="coerce").fillna(0.0).round(1)
                if "drivers_marginal_tacos" in c.columns:
                    c["drivers_marginal_tacos"] = pd.to_numeric(c["drivers_marginal_tacos"], errors="coerce").fillna(0.0).round(4)
                for col in ("inventory", "top_action_count", "top_blocked_action_count"):
                    if col in c.columns:
                        c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0).astype(int)
                for col in ("drivers_rank_delta_sales", "drivers_rank_delta_ad_spend"):
                    if col in c.columns:
                        c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0).astype(int)

                view = c.head(20).copy()
                lines.append(
                    _df_to_md_table(
                        view,
                        [
                            "product_category",
                            "asin",
                            "current_phase",
                            "sales_per_day_7d",
                            "inventory_cover_days_7d",
                            "ad_spend_roll",
                            "profit_direction",
                            "focus_score",
                        ],
                    )
                )
            except Exception:
                lines.append("- （生成失败）")
        lines.append("")
        lines.append("## 2) ASIN Focus（先看最需要处理的 20 个）")
        lines.append("")
        if asin_focus is None or asin_focus.empty:
            lines.append("- （无）")
        else:
            lines.append("- 展示方式：`商品分类 → 产品`（便于同类对比）；生命周期字段 `current_phase/cycle_id` 贯穿解释。")
            lines.append("- 运营版：只保留少列快速扫；更多维度请看 `../dashboard/asin_focus.csv`")
            lines.append("")

            # 2.0 生命周期总览（Top N）
            try:
                pc = phase_cockpit.copy() if phase_cockpit is not None and isinstance(phase_cockpit, pd.DataFrame) else pd.DataFrame()
                if pc is not None and not pc.empty and "current_phase" in pc.columns:
                    lines.append("### 生命周期总览（Top 7） - [打开筛选表](../dashboard/phase_cockpit.csv)")
                    lines.append("")
                    lines.append("- 运营版：只保留少列快速扫；更多维度请看 `../dashboard/phase_cockpit.csv`")
                    lines.append("")
                    view = pc.head(7).copy()
                    view["current_phase"] = view["current_phase"].map(lambda x: _phase_md_link(str(x or ""), "./phase_drilldown.md"))
                    phase_map = {
                        "current_phase": "生命周期",
                        "asin_count": "ASIN数",
                        "focus_score_sum": "关注度总分",
                        "delta_sales_sum": "销售Δ合计",
                        "delta_spend_sum": "花费Δ合计",
                        "oos_asin_count": "断货ASIN数",
                        "low_inventory_asin_count": "低库存ASIN数",
                    }
                    show_cols = [
                        c
                        for c in [
                            "current_phase",
                            "asin_count",
                            "focus_score_sum",
                            "delta_sales_sum",
                            "delta_spend_sum",
                            "oos_asin_count",
                            "low_inventory_asin_count",
                        ]
                        if c in view.columns
                    ]
                    lines.append(_display_table(view, show_cols, phase_map))
                    lines.append("")
            except Exception:
                pass

            # 2.0 类目总览（Top N）
            try:
                # 优先展示 category_cockpit（包含 drivers/action 汇总）；没有则回退 category_summary
                use_cockpit = category_cockpit is not None and isinstance(category_cockpit, pd.DataFrame) and not category_cockpit.empty
                cs = category_cockpit.copy() if use_cockpit else (category_summary.copy() if category_summary is not None else pd.DataFrame())
                if cs is not None and not cs.empty:
                    csv_link = "../dashboard/category_cockpit.csv" if use_cockpit else "../dashboard/category_summary.csv"
                    lines.append(f"### 类目总览（Top 10） - [打开筛选表]({csv_link})")
                    lines.append("")
                    lines.append(f"- 运营版：只保留少列快速扫；更多维度请看 `{csv_link}`")
                    lines.append("")
                    view = cs.head(10).copy()
                    # 类目名称可点击跳转到 category_drilldown
                    if "product_category" in view.columns:
                        view["product_category"] = view["product_category"].map(lambda x: _cat_md_link(str(x or ""), "./category_drilldown.md"))
                    cat_map = {
                        "product_category": "商品分类",
                        "focus_score_sum": "关注度总分",
                        "drivers_delta_sales_sum": "销售Δ合计",
                        "sales_total": "销售额",
                        "ad_spend_total": "广告花费",
                        "tacos_total": "TACOS",
                        "oos_asin_count": "断货ASIN数",
                    }
                    show_cols = [
                        c
                        for c in [
                            "product_category",
                            "focus_score_sum",
                            "drivers_delta_sales_sum",
                            "sales_total",
                            "ad_spend_total",
                            "tacos_total",
                            "oos_asin_count",
                        ]
                        if c in view.columns
                    ]
                    lines.append(_display_table(view, show_cols, cat_map))
                    lines.append("")
            except Exception:
                pass

            base = asin_focus.copy()
            # 规范化分类字段（避免 nan/空串破坏分组）
            if "product_category" in base.columns:
                base["product_category"] = base["product_category"].map(_norm_product_category)
            else:
                base["product_category"] = "（未分类）"

            if "focus_score" in base.columns:
                base["focus_score"] = pd.to_numeric(base["focus_score"], errors="coerce").fillna(0.0)
            else:
                base["focus_score"] = 0.0

            # 选择用于 markdown 展示的 Top ASIN：按“分类优先”，每类最多取若干个，总量控制在 20
            max_rows = 20
            per_cat = 5
            picked_rows: List[pd.DataFrame] = []

            # 分类排序：用该分类的 focus_score 总和作为“类目优先级”
            cat_scores = (
                base.groupby("product_category", dropna=False, as_index=False)
                .agg(cat_score=("focus_score", "sum"))
                .sort_values("cat_score", ascending=False)
            )
            cat_order = [str(x).strip() for x in cat_scores["product_category"].tolist()]

            used = 0
            for cat in cat_order:
                if used >= max_rows:
                    break
                cat_name = cat if cat else "（未分类）"
                sub = base[base["product_category"] == cat].copy()
                if sub.empty:
                    continue
                # 类内排序：先 focus_score，再 ad_spend_roll
                if "ad_spend_roll" in sub.columns:
                    sub["ad_spend_roll"] = pd.to_numeric(sub["ad_spend_roll"], errors="coerce").fillna(0.0)
                    sub = sub.sort_values(["focus_score", "ad_spend_roll"], ascending=[False, False])
                else:
                    sub = sub.sort_values(["focus_score"], ascending=[False])
                take = min(per_cat, max_rows - used)
                view = sub.head(take).copy()
                used += int(len(view))

                lines.append(f"### {cat_name}")
                lines.append("")
                # 链接跳转：asin -> asin_drilldown；phase -> phase_drilldown
                if "asin" in view.columns:
                    view["asin"] = view["asin"].map(lambda x: _asin_md_link(str(x or ""), "./asin_drilldown.md"))
                if "current_phase" in view.columns:
                    view["current_phase"] = view["current_phase"].map(lambda x: _phase_md_link(str(x or ""), "./phase_drilldown.md"))
                # 数值格式化（运营快速扫）
                if "sales_per_day_7d" in view.columns:
                    view["sales_per_day_7d"] = pd.to_numeric(view["sales_per_day_7d"], errors="coerce").fillna(0.0).round(2)
                if "inventory_cover_days_7d" in view.columns:
                    view["inventory_cover_days_7d"] = pd.to_numeric(view["inventory_cover_days_7d"], errors="coerce").fillna(0.0).round(1)
                if "ad_spend_roll" in view.columns:
                    view["ad_spend_roll"] = pd.to_numeric(view["ad_spend_roll"], errors="coerce").fillna(0.0).round(2)
                asin_map = {
                    "asin": "ASIN",
                    "product_name": "品名",
                    "current_phase": "生命周期",
                    "sales_per_day_7d": "日均销售(7天)",
                    "inventory_cover_days_7d": "库存覆盖(7天)",
                    "ad_spend_roll": "广告花费(滚动)",
                    "profit_direction": "利润方向",
                    "focus_reasons": "近期诊断",
                }
                show_cols = [
                    c
                    for c in [
                        "asin",
                        "product_name",
                        "current_phase",
                        "sales_per_day_7d",
                        "inventory_cover_days_7d",
                        "ad_spend_roll",
                        "profit_direction",
                        "focus_reasons",
                    ]
                    if c in view.columns
                ]
                lines.append(_display_table(view, show_cols, asin_map))
                lines.append("")
        lines.append("")
        lines.append("## 3) Campaign 行动聚合（优先按 campaign 排查）")
        lines.append("")
        lines.append("- 说明：从 Action Board 聚合到 Campaign，先定位“哪个 Campaign 最需要处理”。")
        lines.append("- 字段释义：P0/P1=优先级计数；ΔSales/ΔSpend=近7天-前7天；TopASIN=该 Campaign 花费Top ASIN")
        lines.append("- 下钻入口：`../dashboard/action_board.csv`（按 campaign 过滤即可定位细项）")
        lines.append("")
        try:
            camp_view = campaign_action_view
            if camp_view is None or camp_view.empty:
                camp_view = build_campaign_action_view(action_board=action_board, max_rows=20, min_spend=10.0)
            if camp_view is None or camp_view.empty:
                lines.append("- （无）")
            else:
                view = camp_view.head(20).copy()
                for c in ("spend_sum", "delta_sales_sum", "delta_spend_sum", "score"):
                    if c in view.columns:
                        view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0.0).round(2)
                camp_map = {
                    "ad_type": "广告类型",
                    "campaign": "Campaign",
                    "p0_count": "P0数",
                    "p1_count": "P1数",
                    "blocked_count": "阻断数",
                    "spend_sum": "花费合计",
                    "delta_sales_sum": "销售Δ合计",
                    "score": "优先评分",
                    "top_asins": "Top ASIN",
                }
                show_cols = [
                    c
                    for c in [
                        "ad_type",
                        "campaign",
                        "p0_count",
                        "p1_count",
                        "blocked_count",
                        "spend_sum",
                        "delta_sales_sum",
                        "score",
                        "top_asins",
                    ]
                    if c in view.columns
                ]
                lines.append(_display_table(view, show_cols, camp_map))
        except Exception:
            lines.append("- （无）")

        lines.append("")
        lines.append("## 4) Action Board（运营聚焦版）")
        lines.append("")
        lines.append("- 只保留“行动/证据/责任/操作手册”四列，避免词级噪音；更多维度请看 `../dashboard/action_board.csv`")
        # 口径提示：asin_hint 是弱关联，低置信度动作不建议直接执行
        try:
            low_thr = 0.35
            if isinstance(policy, OpsPolicy):
                asp = getattr(policy, "dashboard_action_scoring", None)
                if asp is not None:
                    low_thr = float(getattr(asp, "low_hint_confidence_threshold", low_thr) or low_thr)
            low_thr = max(0.0, min(1.0, float(low_thr)))
            lines.append(
                f"- 口径提示：`asin_hint` 为弱关联定位；当 `asin_hint_confidence<{low_thr:.2f}` 时建议先人工确认（可在 `../dashboard/action_board.csv` 里筛选/对照候选ASIN）"
            )
        except Exception:
            lines.append(
                "- 口径提示：`asin_hint` 为弱关联定位；低置信度时建议先人工确认（可在 `../dashboard/action_board.csv` 里筛选/对照候选ASIN）"
            )
        lines.append("- 字段释义：行动=动作类型+对象+动作值；证据=花费/Δ/订单/ACOS/阻断；责任=建议归属；操作手册=排查路径")
        lines.append("")
        if action_board is None or action_board.empty:
            lines.append("- （无）")
        else:
            # 快速概览：按“止损/放量/排查”聚合 Action Board（不改变口径，只帮助运营抓重点）
            try:
                ab0 = action_board.copy()
                if "action_type" in ab0.columns:
                    ab0["action_type"] = ab0["action_type"].astype(str).fillna("").str.upper().str.strip()
                else:
                    ab0["action_type"] = ""
                if "blocked" in ab0.columns:
                    ab0["blocked"] = pd.to_numeric(ab0["blocked"], errors="coerce").fillna(0).astype(int)
                else:
                    ab0["blocked"] = 0

                stop_types = {"NEGATE", "BID_DOWN"}
                scale_types = {"BID_UP", "BUDGET_UP"}

                stop_cnt = int((ab0["action_type"].isin(stop_types)).sum())
                scale_cnt = int(((ab0["action_type"].isin(scale_types)) & (ab0["blocked"] <= 0)).sum())
                blocked_scale_cnt = int(((ab0["action_type"].isin(scale_types)) & (ab0["blocked"] > 0)).sum())
                review_cnt = int((ab0["action_type"] == "REVIEW").sum())

                # 优先展示“可执行”的放量数量，并把“放量被阻断”单独露出来（通常要先解决库存/断货）
                lines.append(
                    f"- 概览：`止损`={stop_cnt} | `放量`={scale_cnt} | `放量被阻断`={blocked_scale_cnt} | `排查`={review_cnt}（更多见 `../dashboard/action_board.csv`）"
                )
                lines.append("")
            except Exception:
                pass

            view = action_board.head(20).copy()
            # 链接跳转：类目/ASIN/生命周期 -> drilldown
            if "product_category" in view.columns:
                view["product_category"] = view["product_category"].map(lambda x: _cat_md_link(str(x or ""), "./category_drilldown.md"))
            if "asin_hint" in view.columns:
                view["asin_hint"] = view["asin_hint"].map(lambda x: _asin_md_link(str(x or ""), "./asin_drilldown.md"))
            if "current_phase" in view.columns:
                view["current_phase"] = view["current_phase"].map(lambda x: _phase_md_link(str(x or ""), "./phase_drilldown.md"))

            # 操作手册联动：从动作表一键跳回“怎么查/怎么做”
            try:
                if "playbook_url" in view.columns:
                    view["playbook"] = view["playbook_url"].map(
                        lambda x: f"[操作手册]({str(x or '').strip()})" if str(x or "").strip() else ""
                    )
            except Exception:
                pass

            # 数值格式化（运营快速扫）
            for col in (
                "asin_hint_confidence",
                "asin_focus_score",
                "action_priority_score",
                "e_spend",
                "e_sales",
                "e_acos",
                "asin_sales_recent_7d",
                "asin_delta_sales",
                "asin_marginal_tacos",
            ):
                if col in view.columns:
                    view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0)
                    if col in {"asin_marginal_tacos"}:
                        view[col] = view[col].round(4)
                    else:
                        view[col] = view[col].round(2)
            if "e_orders" in view.columns:
                view["e_orders"] = pd.to_numeric(view["e_orders"], errors="coerce").fillna(0).astype(int)
            if "blocked" in view.columns:
                view["blocked"] = pd.to_numeric(view["blocked"], errors="coerce").fillna(0).astype(int)

            # 运营聚焦：合并行动描述
            try:
                def _fmt_action_row(r: pd.Series) -> str:
                    p = str(r.get("priority", "") or "").strip().upper()
                    obj = str(r.get("object_name", "") or "").strip()
                    act = str(r.get("action_type", "") or "").strip().upper()
                    val = str(r.get("action_value", "") or "").strip()
                    if val:
                        return f"`{p}` {act} {obj} {val}".strip()
                    return f"`{p}` {act} {obj}".strip()

                view["action_brief"] = view.apply(_fmt_action_row, axis=1)
            except Exception:
                view["action_brief"] = view.get("action_type", "")

            # 证据列（统一格式）
            try:
                evid = []
                for _, r in view.iterrows():
                    parts = []
                    if "e_spend" in r and r.get("e_spend") is not None:
                        parts.append(f"花费={_fmt_usd(r.get('e_spend'))}")
                    if "e_orders" in r and r.get("e_orders") is not None:
                        parts.append(f"订单={_fmt_num(r.get('e_orders'), nd=0)}")
                    if "e_acos" in r and r.get("e_acos") is not None:
                        parts.append(f"ACOS={_fmt_num(r.get('e_acos'), nd=4)}")
                    if "asin_delta_sales" in r and r.get("asin_delta_sales") is not None:
                        ds = _fmt_signed_usd(r.get("asin_delta_sales"))
                        if ds:
                            parts.append(f"ΔSales={ds}")
                    if "blocked_reason" in r and str(r.get("blocked_reason") or "").strip():
                        parts.append(f"阻断={str(r.get('blocked_reason') or '').strip()}")
                    evid.append(" | ".join([p for p in parts if p]))
                view["evidence_brief"] = evid
            except Exception:
                view["evidence_brief"] = ""

            # 责任归属
            try:
                def _owner_from_row(r: pd.Series) -> str:
                    if int(r.get("blocked", 0) or 0) > 0:
                        return "供应链/广告运营"
                    act = str(r.get("action_type", "") or "").upper()
                    if act in {"NEGATE", "BID_DOWN"}:
                        return "广告运营"
                    if act in {"BID_UP", "BUDGET_UP"}:
                        return "广告运营"
                    return "运营/广告运营"

                view["owner_brief"] = view.apply(_owner_from_row, axis=1)
            except Exception:
                view["owner_brief"] = ""

            ab_map = {
                "action_brief": "行动",
                "evidence_brief": "证据",
                "owner_brief": "责任",
                "playbook": "操作手册",
            }
            show_cols = [c for c in ["action_brief", "evidence_brief", "owner_brief", "playbook"] if c in view.columns]
            lines.append(_display_table(view, show_cols, ab_map))
        lines.append("")
        lines.append("## 5) 文件导航")
        lines.append("")
        lines.append("- `../START_HERE.md`：本次店铺输出入口")
        lines.append("- `../dashboard/budget_transfer_plan.csv`：预算迁移净表（估算金额；执行时以实际预算/花费节奏校准）")
        lines.append("- `../dashboard/unlock_scale_tasks.csv`：放量解锁任务表（可分工：广告/供应链/运营/美工）")
        lines.append("- `../dashboard/unlock_scale_tasks_full.csv`：放量解锁任务表全量（含更多任务/优先级，便于追溯）")
        lines.append("- `../dashboard/profit_reduce_watchlist.csv`：利润控量 Watchlist（profit_direction=reduce 且仍在烧钱，可筛选优先止血）")
        lines.append("- `../dashboard/inventory_risk_watchlist.csv`：库存告急仍投放 Watchlist（cover7d 低且仍在投放，提前控量/预警）")
        lines.append("- `../dashboard/inventory_sigmoid_watchlist.csv`：库存调速建议（Sigmoid，仅建议，不影响排序）")
        lines.append("- `../dashboard/profit_guard_watchlist.csv`：利润护栏 Watchlist（Break-even：安全ACOS/CPC 超线提示）")
        lines.append("- `../dashboard/oos_with_ad_spend_watchlist.csv`：断货仍烧钱 Watchlist（oos_with_ad_spend_days>0 且仍在投放，优先止损）")
        lines.append("- `../dashboard/spend_up_no_sales_watchlist.csv`：加花费但销量不增 Watchlist（delta_spend>0 且 delta_sales<=0，优先排查）")
        lines.append("- `../dashboard/phase_down_recent_watchlist.csv`：阶段走弱 Watchlist（近14天阶段走弱 down 且仍在花费：优先排查根因）")
        lines.append("- `../dashboard/scale_opportunity_watchlist.csv`：机会 Watchlist（可放量窗口/低花费高潜候选；用于筛选预算迁移/加码）")
        lines.append("- `../dashboard/opportunity_action_board.csv`：机会→可执行动作（只保留 BID_UP/BUDGET_UP 且未阻断）")
        lines.append("- `../dashboard/shop_scorecard.json`：店铺 KPI/诊断（结构化）")
        lines.append("- `../dashboard/phase_cockpit.csv`：生命周期总览（按 current_phase 汇总 focus/变化/动作量）")
        lines.append("- `./phase_drilldown.md`：Phase Drilldown（点击生命周期阶段跳转到该阶段的类目/ASIN）")
        lines.append("- `./lifecycle_overview.md`：Lifecycle Overview（类目→ASIN 生命周期时间轴，可直观看阶段轨迹）")
        lines.append("- `../dashboard/category_summary.csv`：类目总览（基础汇总）")
        lines.append("- `../dashboard/category_cockpit.csv`：类目总览（focus + drivers + 动作量汇总）")
        lines.append("- `../dashboard/category_asin_compare.csv`：类目→产品对比（同类 Top ASIN：速度/覆盖/利润承受度/风险一张表，可筛选）")
        lines.append("- `./category_drilldown.md`：Category Drilldown（点击类目跳转到该类目的 Top ASIN）")
        lines.append("- `../dashboard/asin_focus.csv`：ASIN Focus List（可筛选）")
        lines.append("- `../dashboard/asin_cockpit.csv`：ASIN 总览（focus + drivers + 动作量汇总）")
        lines.append("- `../dashboard/drivers_top_asins.csv`：变化来源（近7天 vs 前7天 Top ASIN）")
        lines.append("- `./asin_drilldown.md`：ASIN Drilldown（点击 ASIN 可跳转到该 ASIN 的动作与摘要）")
        lines.append("- `../dashboard/campaign_action_view.csv`：Campaign 行动聚合（从 Action Board 归并，方便先按 campaign 排查）")
        lines.append("- `../dashboard/action_board.csv`：动作看板（去重后的运营视图）")
        lines.append("- `../dashboard/action_board_full.csv`：动作看板全量（含重复，便于追溯）")
        lines.append("- `../ai/ai_input_bundle.json`：给 AI 的结构化输入包（推荐喂这个）")
        lines.append("- `../ai/data_quality.md`：数据质量与维度覆盖盘点（避免口径误读）")
        lines.append("- `../ai/report.md`：全量深挖版（指标罗列 + 图表；主要给 AI/分析用）")
        lines.append("")

        # 口径提示（来自 ai/data_quality.md 的摘要 1-2 条）：避免运营误读
        try:
            hints = data_quality_hints if isinstance(data_quality_hints, list) else []
            hints = [str(x).strip() for x in hints if str(x).strip()]
            if hints:
                lines.append("## 5) 口径提示（数据质量）")
                lines.append("")
                for h in hints[:2]:
                    lines.append(f"- {h}")
                lines.append("- 更多见：`../ai/data_quality.md`")
                lines.append("")
        except Exception:
            pass

        out_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        # 不影响主流程
        return


def write_asin_drilldown_md(
    out_path: Path,
    shop: str,
    stage: str,
    date_start: str,
    date_end: str,
    asin_focus_all: Optional[pd.DataFrame],
    drivers_top_asins: Optional[pd.DataFrame],
    action_board_full: Optional[pd.DataFrame],
    max_asins: int = 30,
) -> None:
    """
    生成 ASIN Drilldown（支持从 dashboard.md 点击 ASIN 跳转）。

    说明：
    - 这是“运营可读”的导航页：每个 ASIN 一段，包含 drivers 摘要 + Top actions
    - Action Board 的 `asin_hint` 属于弱关联，请结合 `asin_hint_confidence` 判断
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _df_to_md_table(df: pd.DataFrame, cols: List[str]) -> str:
            try:
                if df is None or df.empty:
                    return ""
                view = df.copy()
                cols2 = [c for c in cols if c in view.columns]
                if not cols2:
                    cols2 = list(view.columns)[:8]
                view = view[cols2].copy()
                for c in cols2:
                    view[c] = view[c].map(lambda x, _c=c: _format_md_cell(_c, x))
                header = "| " + " | ".join(cols2) + " |"
                sep = "| " + " | ".join(["---"] * len(cols2)) + " |"
                body = ["| " + " | ".join(row) + " |" for row in view.values.tolist()]
                return "\n".join([header, sep] + body)
            except Exception:
                return ""

        # ASIN 元信息（优先从 asin_focus_all 获取）
        meta_map: Dict[str, Dict[str, object]] = {}
        try:
            f = asin_focus_all.copy() if asin_focus_all is not None else pd.DataFrame()
            if f is not None and not f.empty and "asin" in f.columns:
                f = f.copy()
                f["asin_norm"] = f["asin"].astype(str).str.upper().str.strip()
                for _, r in f.iterrows():
                    asin = str(r.get("asin_norm", "") or "").strip().upper()
                    if not asin:
                        continue
                    if asin in meta_map:
                        continue
                    meta_map[asin] = {
                        "product_name": str(r.get("product_name", "") or ""),
                        "product_category": str(r.get("product_category", "") or ""),
                        "current_phase": str(r.get("current_phase", "") or ""),
                        "cycle_id": r.get("cycle_id", ""),
                        "inventory": r.get("inventory", ""),
                        "focus_score": r.get("focus_score", ""),
                        "focus_reasons": str(r.get("focus_reasons", "") or ""),
                        "focus_reasons_history": str(r.get("focus_reasons_history", "") or ""),
                    }
        except Exception:
            meta_map = {}

        # ASIN Focus 明细（用于 Δ窗口校验）
        focus_map: Dict[str, Dict[str, object]] = {}
        try:
            f2 = asin_focus_all.copy() if asin_focus_all is not None else pd.DataFrame()
            if f2 is not None and not f2.empty and "asin" in f2.columns:
                f2 = f2.copy()
                f2["asin_norm"] = f2["asin"].astype(str).str.upper().str.strip()
                for _, r in f2.iterrows():
                    asin = str(r.get("asin_norm", "") or "").strip().upper()
                    if not asin or asin in focus_map:
                        continue
                    focus_map[asin] = r.to_dict()
        except Exception:
            focus_map = {}

        # 选择要输出的 ASIN 列表（顺序：drivers -> action_board -> focus）
        asins: List[str] = []

        def _push(asin: object) -> None:
            a = str(asin or "").strip().upper()
            if not a or a.lower() == "nan":
                return
            if a not in asins:
                asins.append(a)

        # drivers Top
        try:
            d = drivers_top_asins.copy() if drivers_top_asins is not None else pd.DataFrame()
            if d is not None and not d.empty and "asin" in d.columns and "driver_type" in d.columns:
                for t in ("delta_sales", "delta_ad_spend"):
                    sub = d[d["driver_type"].astype(str) == t].copy()
                    if "rank" in sub.columns:
                        sub["rank"] = pd.to_numeric(sub["rank"], errors="coerce").fillna(0).astype(int)
                        sub = sub.sort_values("rank", ascending=True)
                    for a in sub.head(8)["asin"].tolist():
                        _push(a)
        except Exception:
            pass

        # action_board Top（全量中挑一些，避免 drilldown 过长）
        try:
            ab = action_board_full.copy() if action_board_full is not None else pd.DataFrame()
            if ab is not None and not ab.empty and "asin_hint" in ab.columns:
                view = ab.copy()
                if "blocked" not in view.columns:
                    view["blocked"] = 0
                if "action_priority_score" not in view.columns:
                    view["action_priority_score"] = 0.0
                if "e_spend" not in view.columns:
                    view["e_spend"] = 0.0
                view["blocked"] = pd.to_numeric(view["blocked"], errors="coerce").fillna(0).astype(int)
                view["action_priority_score"] = pd.to_numeric(view["action_priority_score"], errors="coerce").fillna(0.0)
                view["e_spend"] = pd.to_numeric(view["e_spend"], errors="coerce").fillna(0.0)
                view = view.sort_values(["blocked", "action_priority_score", "e_spend"], ascending=[True, False, False])
                for a in view.head(80)["asin_hint"].tolist():
                    _push(a)
        except Exception:
            pass

        # focus Top
        try:
            f = asin_focus_all.copy() if asin_focus_all is not None else pd.DataFrame()
            if f is not None and not f.empty and "asin" in f.columns and "focus_score" in f.columns:
                f = f.copy()
                f["asin_norm"] = f["asin"].astype(str).str.upper().str.strip()
                f["focus_score"] = pd.to_numeric(f.get("focus_score", 0.0), errors="coerce").fillna(0.0)
                f = f.sort_values(["focus_score"], ascending=[False])
                for a in f.head(20)["asin_norm"].tolist():
                    _push(a)
        except Exception:
            pass

        asins = asins[: max(1, int(max_asins or 30))]

        # 写文件
        lines: List[str] = []
        lines.append('<a id="top"></a>')
        lines.append(f"# {shop} ASIN Drilldown（点击跳转）")
        lines.append("")
        lines.append(f"- 阶段: `{stage}`")
        lines.append(f"- 时间范围: `{date_start} ~ {date_end}`")
        lines.append("- 口径说明: 未标注的累计指标=主窗口；对比表为近7/14天 vs 前7/14天（日期见表内 recent/prev）")
        lines.append("")
        lines.append("提示：Action Board 的 `asin_hint` 属于弱关联，请结合 `asin_hint_confidence` 判断；不确定时优先回看原始报表与类目/生命周期语境。")
        lines.append("")

        lines.append("## 索引")
        lines.append("")
        if not asins:
            lines.append("- （无）")
        else:
            for a in asins:
                m = meta_map.get(a, {})
                cat = str(m.get("product_category", "") or "")
                cat = "" if cat.strip().lower() == "nan" else cat.strip()
                if not cat:
                    cat = "（未分类）"
                name = str(m.get("product_name", "") or "").strip()
                aid = _asin_anchor_id(a)
                lines.append(f"- [{a}](#{aid}) | {cat} | {name}")
        lines.append("")

        # 准备 drivers/action_board 数据
        d_all = drivers_top_asins.copy() if drivers_top_asins is not None else pd.DataFrame()
        if d_all is None:
            d_all = pd.DataFrame()
        if not d_all.empty and "asin" in d_all.columns:
            d_all["asin"] = d_all["asin"].astype(str).str.upper().str.strip()

        ab_all = action_board_full.copy() if action_board_full is not None else pd.DataFrame()
        if ab_all is None:
            ab_all = pd.DataFrame()
        if not ab_all.empty and "asin_hint" in ab_all.columns:
            ab_all["asin_hint"] = ab_all["asin_hint"].astype(str).str.upper().str.strip()

        for a in asins:
            aid = _asin_anchor_id(a)
            m = meta_map.get(a, {})
            cat = str(m.get("product_category", "") or "").strip()
            cat = "" if cat.lower() == "nan" else cat
            if not cat:
                cat = "（未分类）"
            name = str(m.get("product_name", "") or "").strip()
            phase = str(m.get("current_phase", "") or "").strip()
            cycle_id = str(m.get("cycle_id", "") or "").strip()
            inv = str(m.get("inventory", "") or "").strip()
            fscore = str(m.get("focus_score", "") or "").strip()
            freasons = str(m.get("focus_reasons", "") or "").strip()
            fhistory = str(m.get("focus_reasons_history", "") or "").strip()

            lines.append(f'<a id="{aid}"></a>')
            lines.append(f"## {a}")
            lines.append("")
            lines.append(f"- 商品分类: `{cat}`")
            if name:
                lines.append(f"- 品名: `{name}`")
            if phase or cycle_id:
                lines.append(f"- 生命周期: phase=`{phase}` | cycle_id=`{cycle_id}`")
            if inv:
                lines.append(f"- 库存: `{inv}`")
            if fscore:
                lines.append(f"- focus_score: `{fscore}`")
            if freasons:
                lines.append(f"- 近期诊断: `{freasons}`")
            if fhistory:
                lines.append(f"- 历史诊断: `{fhistory}`")
            lines.append("")

            # drivers
            lines.append("### Drivers（近7天 vs 前7天）")
            lines.append("")
            try:
                sub = d_all[d_all.get("asin", "") == a].copy() if not d_all.empty and "asin" in d_all.columns else pd.DataFrame()
                if sub is None or sub.empty:
                    lines.append("- （无）")
                else:
                    # 只展示关键列
                    show = sub.copy()
                    for c in ("delta_sales", "delta_ad_spend"):
                        if c in show.columns:
                            show[c] = pd.to_numeric(show[c], errors="coerce").fillna(0.0).round(2)
                    if "marginal_tacos" in show.columns:
                        show["marginal_tacos"] = pd.to_numeric(show["marginal_tacos"], errors="coerce").fillna(0.0).round(4)
                    # 仅保留“驱动类型”的对应列，避免误读
                    try:
                        if "driver_type" in show.columns:
                            for c in ("delta_sales", "delta_ad_spend"):
                                if c in show.columns:
                                    show[c] = show[c].astype(object)
                            show["driver_type"] = show["driver_type"].astype(str)
                            show.loc[show["driver_type"] == "delta_sales", "delta_ad_spend"] = ""
                            show.loc[show["driver_type"] == "delta_ad_spend", "delta_sales"] = ""
                    except Exception:
                        pass
                    lines.append(
                        _df_to_md_table(
                            show,
                            ["driver_type", "rank", "delta_sales", "delta_ad_spend", "marginal_tacos", "recent_start", "recent_end", "prev_start", "prev_end"],
                        )
                    )
            except Exception:
                lines.append("- （无）")
            lines.append("")

            # Δ窗口校验（7/14天）
            lines.append("### Δ窗口校验（7/14天：最近 vs 前段）")
            lines.append("")
            try:
                r = focus_map.get(a, {})
                rows = []
                # 7d
                if r:
                    rows.append(
                        {
                            "window_days": 7,
                            "sales_prev": r.get("sales_prev_7d", 0.0),
                            "sales_recent": r.get("sales_recent_7d", 0.0),
                            "delta_sales": r.get("delta_sales", 0.0),
                            "spend_prev": r.get("ad_spend_prev_7d", 0.0),
                            "spend_recent": r.get("ad_spend_recent_7d", 0.0),
                            "delta_spend": r.get("delta_spend", 0.0),
                        }
                    )
                    # 14d
                    if (
                        ("sales_prev_14d" in r)
                        or ("sales_recent_14d" in r)
                        or ("ad_spend_prev_14d" in r)
                        or ("ad_spend_recent_14d" in r)
                        or ("delta_sales_14d" in r)
                        or ("delta_spend_14d" in r)
                    ):
                        rows.append(
                            {
                                "window_days": 14,
                                "sales_prev": r.get("sales_prev_14d", 0.0),
                                "sales_recent": r.get("sales_recent_14d", 0.0),
                                "delta_sales": r.get("delta_sales_14d", 0.0),
                                "spend_prev": r.get("ad_spend_prev_14d", 0.0),
                                "spend_recent": r.get("ad_spend_recent_14d", 0.0),
                                "delta_spend": r.get("delta_spend_14d", 0.0),
                            }
                        )
                if not rows:
                    lines.append("- （无）")
                else:
                    dfv = pd.DataFrame(rows)
                    for c in ("sales_prev", "sales_recent", "delta_sales", "spend_prev", "spend_recent", "delta_spend"):
                        if c in dfv.columns:
                            dfv[c] = pd.to_numeric(dfv[c], errors="coerce").fillna(0.0).round(2)
                    lines.append(
                        _df_to_md_table(
                            dfv,
                            ["window_days", "sales_prev", "sales_recent", "delta_sales", "spend_prev", "spend_recent", "delta_spend"],
                        )
                    )
            except Exception:
                lines.append("- （无）")
            lines.append("")

            # actions
            lines.append("### Top Actions（按 action_priority_score 排序，Top 10）")
            lines.append("")
            try:
                sub = ab_all[ab_all.get("asin_hint", "") == a].copy() if not ab_all.empty and "asin_hint" in ab_all.columns else pd.DataFrame()
                if sub is None or sub.empty:
                    lines.append("- （无）")
                else:
                    if "blocked" not in sub.columns:
                        sub["blocked"] = 0
                    if "action_priority_score" not in sub.columns:
                        sub["action_priority_score"] = 0.0
                    if "e_spend" not in sub.columns:
                        sub["e_spend"] = 0.0
                    sub["blocked"] = pd.to_numeric(sub["blocked"], errors="coerce").fillna(0).astype(int)
                    sub["action_priority_score"] = pd.to_numeric(sub["action_priority_score"], errors="coerce").fillna(0.0)
                    sub["e_spend"] = pd.to_numeric(sub["e_spend"], errors="coerce").fillna(0.0)
                    # 去重（合并 search_term/targeting 重复）
                    dedup = dedup_action_board(sub)
                    dedup = dedup.sort_values(["blocked", "action_priority_score", "e_spend"], ascending=[True, False, False])
                    view = dedup.head(10).copy()
                    # 操作手册联动：让 Top Actions 也能一键跳回固定流程
                    try:
                        if "playbook_url" in view.columns:
                            view["playbook"] = view["playbook_url"].map(
                                lambda x: f"[操作手册]({str(x or '').strip()})" if str(x or "").strip() else ""
                            )
                    except Exception:
                        pass
                    lines.append(
                        _df_to_md_table(
                            view,
                            [
                                "blocked",
                                "blocked_reason",
                                "priority",
                                "action_priority_score",
                                "priority_reason",
                                "asin_hint_confidence",
                                "action_type",
                                "action_value",
                                "campaign",
                                "ad_group",
                                "match_type",
                                "object_name",
                                "e_spend",
                                "e_orders",
                                "e_acos",
                                "reason",
                                "playbook",
                            ],
                        )
                    )
                    lines.append("")
                    lines.append(f"- 全量动作见：`../dashboard/action_board_full.csv`（在表格里按 `asin_hint={a}` 过滤）")
            except Exception:
                lines.append("- （无）")

            lines.append("")
            lines.append("[回到顶部](#top) | [返回 Dashboard](./dashboard.md)")
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        return


def write_category_drilldown_md(
    out_path: Path,
    shop: str,
    stage: str,
    date_start: str,
    date_end: str,
    category_cockpit: Optional[pd.DataFrame],
    asin_cockpit: Optional[pd.DataFrame],
    keyword_segment_top: Optional[pd.DataFrame] = None,
    max_categories: int = 20,
    asins_per_category: int = 10,
) -> None:
    """
    生成 Category Drilldown（支持从 dashboard.md 点击类目跳转）。

    设计目标：
    - 运营先看类目（同类对比），再点 ASIN 跳到 asin_drilldown.md 看动作
    - 控制信息密度：默认只输出 Top N 类目，每类 Top K ASIN
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _df_to_md_table(df: pd.DataFrame, cols: List[str]) -> str:
            try:
                if df is None or df.empty:
                    return ""
                view = df.copy()
                cols2 = [c for c in cols if c in view.columns]
                if not cols2:
                    cols2 = list(view.columns)[:8]
                view = view[cols2].copy()
                for c in cols2:
                    view[c] = view[c].map(lambda x, _c=c: _format_md_cell(_c, x))
                header = "| " + " | ".join(cols2) + " |"
                sep = "| " + " | ".join(["---"] * len(cols2)) + " |"
                body = ["| " + " | ".join(row) + " |" for row in view.values.tolist()]
                return "\n".join([header, sep] + body)
            except Exception:
                return ""

        cc = category_cockpit.copy() if isinstance(category_cockpit, pd.DataFrame) else pd.DataFrame()
        if cc is None:
            cc = pd.DataFrame()
        if cc.empty or "product_category" not in cc.columns:
            lines = [
                f"# {shop} Category Drilldown（类目下钻）",
                "",
                f"- 阶段: `{stage}`",
                f"- 时间范围: `{date_start} ~ {date_end}`",
                "- 口径说明: 未标注的累计指标=主窗口；标注 compare/Δ 的为近N天 vs 前N天（日期见表内 recent/prev）",
                "",
                "- （无）",
                "",
            ]
            out_path.write_text("\n".join(lines), encoding="utf-8")
            return

        cc = cc.copy()
        cc["product_category"] = cc["product_category"].map(_norm_product_category)

        # 排序：优先按 focus_score_sum，再按 Top 动作量，再按 ad_spend_total
        sort_cols = [c for c in ["focus_score_sum", "category_top_action_count", "ad_spend_total"] if c in cc.columns]
        if sort_cols:
            cc = cc.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()

        top_n = max(1, int(max_categories or 20))
        cats = cc["product_category"].tolist()[:top_n]

        ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
        if ac is None:
            ac = pd.DataFrame()
        if not ac.empty:
            ac = ac.copy()
            ac["product_category"] = ac.get("product_category", "").map(_norm_product_category)
            if "asin" in ac.columns:
                ac["asin"] = ac["asin"].astype(str).str.upper().str.strip()
            for c in ("focus_score", "ad_spend_roll", "drivers_delta_sales", "drivers_delta_ad_spend"):
                if c in ac.columns:
                    ac[c] = pd.to_numeric(ac[c], errors="coerce").fillna(0.0)
            for c in ("inventory", "top_action_count", "top_blocked_action_count"):
                if c in ac.columns:
                    ac[c] = pd.to_numeric(ac[c], errors="coerce").fillna(0).astype(int)

        # keyword_topics_segment_top（类目×阶段→Top主题）
        st = keyword_segment_top.copy() if isinstance(keyword_segment_top, pd.DataFrame) else pd.DataFrame()
        if st is None:
            st = pd.DataFrame()
        if not st.empty:
            try:
                st = st.copy()
                st["product_category"] = st.get("product_category", "").map(_norm_product_category)
                st["current_phase"] = st.get("current_phase", "unknown").map(_norm_phase)
                for c in ("reduce_waste_spend_sum", "scale_sales_sum"):
                    if c in st.columns:
                        st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0.0)
                for c in ("reduce_topic_count", "scale_topic_count"):
                    if c in st.columns:
                        st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0).astype(int)
            except Exception:
                st = pd.DataFrame()

        # 写文件
        lines: List[str] = []
        lines.append('<a id="top"></a>')
        lines.append(f"# {shop} Category Drilldown（类目下钻）")
        lines.append("")
        lines.append(f"- 阶段: `{stage}`")
        lines.append(f"- 时间范围: `{date_start} ~ {date_end}`")
        lines.append("- 口径说明: 未标注的累计指标=主窗口；标注 compare/Δ 的为近N天 vs 前N天（日期见表内 recent/prev）")
        lines.append("")
        lines.append("提示：点击 ASIN 会跳到 `asin_drilldown.md`（动作与 drivers 摘要）。")
        lines.append(f"- 可筛选对比表：[`../dashboard/category_asin_compare.csv`](../dashboard/category_asin_compare.csv)")
        lines.append(f"- 关键词主题下钻：[`keyword_topics.md`](./keyword_topics.md)（建议先 segment→topic 再定位执行）")
        lines.append("")

        # 目录
        lines.append("## 索引")
        lines.append("")
        for cat in cats:
            cid = _cat_anchor_id(cat)
            lines.append(f"- [{cat}](#{cid})")
        lines.append("")

        # 总览表（Top N）
        lines.append(f"## 类目总览（Top {top_n}）")
        lines.append("")
        try:
            view = cc.head(top_n).copy()
            view["product_category"] = view["product_category"].map(lambda x: _cat_md_link(str(x or ""), "./category_drilldown.md"))
            show_cols = [
                c
                for c in [
                    "product_category",
                    "focus_score_sum",
                    "drivers_delta_sales_sum",
                    "drivers_delta_ad_spend_sum",
                    "ad_spend_total",
                    "sales_total",
                    "tacos_total",
                    "ad_sales_share_total",
                    "asin_count",
                    "category_top_action_count",
                    "category_top_blocked_action_count",
                    "oos_asin_count",
                    "low_inventory_asin_count",
                    "phase_counts",
                ]
                if c in view.columns
            ]
            lines.append(_df_to_md_table(view, show_cols))
        except Exception:
            lines.append("- （生成失败）")
        lines.append("")

        # 每类目下钻：Top ASIN
        for cat in cats:
            cid = _cat_anchor_id(cat)
            lines.append(f'<a id="{cid}"></a>')
            lines.append(f"## {cat}")
            lines.append("")

            # 类目摘要（尽量一行）
            try:
                r = cc[cc["product_category"] == cat].head(1)
                if not r.empty:
                    rr = r.iloc[0].to_dict()
                    lines.append(
                        f"- sales=`{rr.get('sales_total', 0)}` | ad_spend=`{rr.get('ad_spend_total', 0)}` | tacos=`{rr.get('tacos_total', 0)}` | ad_sales_share=`{rr.get('ad_sales_share_total', 0)}` | focus_sum=`{rr.get('focus_score_sum', 0)}` | top_actions=`{rr.get('category_top_action_count', 0)}`"
                    )
                    if str(rr.get("phase_counts", "") or "").strip():
                        lines.append(f"- phase_counts: `{rr.get('phase_counts')}`")
            except Exception:
                pass
            lines.append("")

            lines.append(f"### Top ASIN（按 focus_score / 动作量排序，Top {int(asins_per_category or 10)}）")
            lines.append("")
            try:
                if ac is None or ac.empty or "product_category" not in ac.columns:
                    lines.append("- （无）")
                else:
                    sub = ac[ac["product_category"] == cat].copy()
                    if sub.empty:
                        lines.append("- （无）")
                    else:
                        # 排序：focus_score -> top_action_count -> ad_spend_roll
                        sort_cols2 = [c for c in ["focus_score", "top_action_count", "ad_spend_roll"] if c in sub.columns]
                        if sort_cols2:
                            sub = sub.sort_values(sort_cols2, ascending=[False] * len(sort_cols2))
                        view = sub.head(max(1, int(asins_per_category or 10))).copy()
                        if "asin" in view.columns:
                            view["asin"] = view["asin"].map(lambda x: _asin_md_link(str(x or ""), "./asin_drilldown.md"))
                        # 数值格式化
                        for col in ("focus_score", "ad_spend_roll", "drivers_delta_sales", "drivers_delta_ad_spend"):
                            if col in view.columns:
                                view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).round(2)
                        if "drivers_marginal_tacos" in view.columns:
                            view["drivers_marginal_tacos"] = pd.to_numeric(view["drivers_marginal_tacos"], errors="coerce").fillna(0.0).round(4)
                        for col in ("sales_per_day_7d", "sales_per_day_30d"):
                            if col in view.columns:
                                view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).round(2)
                        for col in ("aov_recent_7d", "delta_aov_7d"):
                            if col in view.columns:
                                view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).round(2)
                        if "gross_margin" in view.columns:
                            view["gross_margin"] = pd.to_numeric(view["gross_margin"], errors="coerce").fillna(0.0).round(4)
                        for col in ("inventory_cover_days_7d", "inventory_cover_days_30d"):
                            if col in view.columns:
                                view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).round(1)
                        if "max_ad_spend_by_profit" in view.columns:
                            view["max_ad_spend_by_profit"] = pd.to_numeric(view["max_ad_spend_by_profit"], errors="coerce").fillna(0.0).round(2)
                        for col in ("inventory", "top_action_count", "top_blocked_action_count"):
                            if col in view.columns:
                                view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0).astype(int)
                        lines.append(
                            _df_to_md_table(
                                view,
                                [
                                    "asin",
                                    "product_name",
                                    "current_phase",
                                    "inventory",
                                    "sales_per_day_7d",
                                    "sales_per_day_30d",
                                    "aov_recent_7d",
                                    "delta_aov_7d",
                                    "gross_margin",
                                    "inventory_cover_days_7d",
                                    "inventory_cover_days_30d",
                                    "profit_direction",
                                    "focus_score",
                                    "top_action_count",
                                    "ad_spend_roll",
                                ],
                            )
                        )
            except Exception:
                lines.append("- （无）")
            lines.append("")

            # ===== 关键词主题 Top（reduce/scale）=====
            lines.append("### 关键词主题 Top（reduce/scale） - [下钻](./keyword_topics.md)")
            lines.append("")
            lines.append("- 对应 CSV：`../dashboard/keyword_topics_segment_top.csv`（可在 Excel 里筛 `product_category`）")
            lines.append("- 执行入口：`../dashboard/keyword_topics_action_hints.csv`（优先筛 `direction=scale` 且 `blocked=0` 再放量）")
            lines.append("")
            try:
                if st is None or st.empty or "product_category" not in st.columns:
                    lines.append("- （无：本次缺少关键词主题 segment_top 或无有效主题）")
                else:
                    sub2 = st[st["product_category"] == cat].copy()
                    if sub2.empty:
                        lines.append("- （无：该类目本次无有效主题，或缺少 search_term 报表）")
                    else:

                        def _short(x: object, n: int = 120) -> str:
                            try:
                                s = str(x or "").replace("\n", " ").replace("|", "｜").strip()
                                if not s or s.lower() == "nan":
                                    return ""
                                if len(s) <= int(n):
                                    return s
                                return s[: int(n)] + "…"
                            except Exception:
                                return ""

                        if "reduce_top_topics" in sub2.columns:
                            sub2["reduce_top_topics"] = sub2["reduce_top_topics"].map(lambda x: _short(x, 120))
                        if "scale_top_topics" in sub2.columns:
                            sub2["scale_top_topics"] = sub2["scale_top_topics"].map(lambda x: _short(x, 120))

                        # Top 浪费阶段
                        lines.append("#### Top 浪费阶段（reduce，按 waste_spend_sum）")
                        lines.append("")
                        red = sub2.copy()
                        if "reduce_waste_spend_sum" in red.columns:
                            red = red[pd.to_numeric(red.get("reduce_waste_spend_sum", 0.0), errors="coerce").fillna(0.0) > 0].copy()
                            red = red.sort_values(["reduce_waste_spend_sum"], ascending=[False]).copy()
                        red = red.head(6)
                        if red.empty:
                            lines.append("- （无）")
                        else:
                            lines.append(
                                _df_to_md_table(
                                    red,
                                    [c for c in ["current_phase", "reduce_waste_spend_sum", "reduce_topic_count", "reduce_top_topics"] if c in red.columns],
                                )
                            )
                        lines.append("")

                        # Top 贡献阶段
                        lines.append("#### Top 贡献阶段（scale，按 sales_sum）")
                        lines.append("")
                        sc = sub2.copy()
                        if "scale_sales_sum" in sc.columns:
                            sc = sc[pd.to_numeric(sc.get("scale_sales_sum", 0.0), errors="coerce").fillna(0.0) > 0].copy()
                            sc = sc.sort_values(["scale_sales_sum"], ascending=[False]).copy()
                        sc = sc.head(6)
                        if sc.empty:
                            lines.append("- （无）")
                        else:
                            lines.append(
                                _df_to_md_table(
                                    sc,
                                    [c for c in ["current_phase", "scale_sales_sum", "scale_topic_count", "scale_top_topics"] if c in sc.columns],
                                )
                            )
                        lines.append("")
            except Exception:
                lines.append("- （无）")
                lines.append("")

            lines.append("[回到顶部](#top) | [返回 Dashboard](./dashboard.md)")
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        return


def write_phase_drilldown_md(
    out_path: Path,
    shop: str,
    stage: str,
    date_start: str,
    date_end: str,
    phase_cockpit: Optional[pd.DataFrame],
    asin_cockpit: Optional[pd.DataFrame],
    max_phases: int = 20,
    categories_per_phase: int = 8,
    asins_per_phase: int = 12,
) -> None:
    """
    生成 Phase Drilldown（支持从 dashboard.md 点击生命周期阶段跳转）。

    设计目标：
    - “动态生命周期”作为第二条主线入口：phase →（类目/ASIN）→ 动作
    - 控制信息密度：默认只输出 Top N phase；每个 phase 只展示 Top K 类目与 Top M ASIN
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _df_to_md_table(df: pd.DataFrame, cols: List[str]) -> str:
            try:
                if df is None or df.empty:
                    return ""
                view = df.copy()
                cols2 = [c for c in cols if c in view.columns]
                if not cols2:
                    cols2 = list(view.columns)[:8]
                view = view[cols2].copy()
                for c in cols2:
                    view[c] = view[c].map(lambda x, _c=c: _format_md_cell(_c, x))
                header = "| " + " | ".join(cols2) + " |"
                sep = "| " + " | ".join(["---"] * len(cols2)) + " |"
                body = ["| " + " | ".join(row) + " |" for row in view.values.tolist()]
                return "\n".join([header, sep] + body)
            except Exception:
                return ""

        pc = phase_cockpit.copy() if isinstance(phase_cockpit, pd.DataFrame) else pd.DataFrame()
        if pc is None:
            pc = pd.DataFrame()
        if pc.empty or "current_phase" not in pc.columns:
            lines = [
                f"# {shop} Phase Drilldown（生命周期下钻）",
                "",
                f"- 阶段: `{stage}`",
                f"- 时间范围: `{date_start} ~ {date_end}`",
                "- 口径说明: 未标注的累计指标=主窗口；标注 compare/Δ 的为近N天 vs 前N天（日期见表内 recent/prev）",
                "",
                "- （无）",
                "",
            ]
            out_path.write_text("\n".join(lines), encoding="utf-8")
            return

        pc = pc.copy()
        pc["current_phase"] = pc["current_phase"].map(_norm_phase)

        # 排序：按生命周期顺序
        order = ["pre_launch", "launch", "growth", "mature", "stable", "decline", "inactive", "unknown"]
        order_map = {p: i for i, p in enumerate(order)}
        pc["_order"] = pc["current_phase"].map(lambda x: order_map.get(str(x or "").strip().lower(), 999))
        pc = pc.sort_values(["_order", "focus_score_sum"], ascending=[True, False]).drop(columns=["_order"], errors="ignore")

        top_n = max(1, int(max_phases or 20))
        phases = pc["current_phase"].tolist()[:top_n]

        ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
        if ac is None:
            ac = pd.DataFrame()
        if not ac.empty:
            ac = ac.copy()
            ac["current_phase"] = ac.get("current_phase", "unknown").map(_norm_phase)
            ac["product_category"] = ac.get("product_category", "").map(_norm_product_category)
            if "asin" in ac.columns:
                ac["asin"] = ac["asin"].astype(str).str.upper().str.strip()
            for c in ("focus_score", "ad_spend_roll", "delta_sales", "delta_spend", "marginal_tacos"):
                if c in ac.columns:
                    ac[c] = pd.to_numeric(ac[c], errors="coerce").fillna(0.0)
            for c in ("inventory", "top_action_count", "top_blocked_action_count"):
                if c in ac.columns:
                    ac[c] = pd.to_numeric(ac[c], errors="coerce").fillna(0).astype(int)

        # 写文件
        lines: List[str] = []
        lines.append('<a id="top"></a>')
        lines.append(f"# {shop} Phase Drilldown（生命周期下钻）")
        lines.append("")
        lines.append(f"- 阶段: `{stage}`")
        lines.append(f"- 时间范围: `{date_start} ~ {date_end}`")
        lines.append("- 口径说明: 未标注的累计指标=主窗口；标注 compare/Δ 的为近N天 vs 前N天（日期见表内 recent/prev）")
        lines.append("")
        lines.append("提示：这里的 `delta_*` 默认来自 compare_7d（近7天 vs 前7天）。点击 ASIN 会跳到 `asin_drilldown.md`。")
        lines.append("")

        # 索引
        lines.append("## 索引")
        lines.append("")
        for p in phases:
            pid = _phase_anchor_id(p)
            lines.append(f"- [{p}](#{pid})")
        lines.append("")

        # 总览表
        lines.append(f"## 生命周期总览（Top {top_n}）")
        lines.append("")
        try:
            view = pc.head(top_n).copy()
            view["current_phase"] = view["current_phase"].map(lambda x: _phase_md_link(str(x or ""), "./phase_drilldown.md"))
            show_cols = [
                c
                for c in [
                    "current_phase",
                    "asin_count",
                    "category_count",
                    "focus_score_sum",
                    "delta_sales_sum",
                    "delta_spend_sum",
                    "phase_top_action_count",
                    "phase_top_blocked_action_count",
                    "oos_asin_count",
                    "low_inventory_asin_count",
                    "oos_with_ad_spend_asin_count",
                    "inventory_risk_asin_count",
                    "spend_up_no_sales_asin_count",
                    "high_ad_dependency_asin_count",
                ]
                if c in view.columns
            ]
            lines.append(_df_to_md_table(view, show_cols))
        except Exception:
            lines.append("- （生成失败）")
        lines.append("")

        # 每 phase 下钻
        for p in phases:
            pid = _phase_anchor_id(p)
            lines.append(f'<a id="{pid}"></a>')
            lines.append(f"## {p}")
            lines.append("")

            # phase 摘要
            try:
                r = pc[pc["current_phase"] == p].head(1)
                if not r.empty:
                    rr = r.iloc[0].to_dict()
                    lines.append(
                        f"- asins=`{rr.get('asin_count', 0)}` | categories=`{rr.get('category_count', 0)}` | focus_sum=`{rr.get('focus_score_sum', 0)}` | Δsales=`{rr.get('delta_sales_sum', 0)}` | Δad_spend=`{rr.get('delta_spend_sum', 0)}` | top_actions=`{rr.get('phase_top_action_count', 0)}` | top_blocked=`{rr.get('phase_top_blocked_action_count', 0)}`"
                    )
                    lines.append(
                        f"- oos=`{rr.get('oos_asin_count', 0)}` | low_inventory=`{rr.get('low_inventory_asin_count', 0)}` | oos_with_ad_spend=`{rr.get('oos_with_ad_spend_asin_count', 0)}` | spend_up_no_sales=`{rr.get('spend_up_no_sales_asin_count', 0)}`"
                    )
            except Exception:
                pass
            lines.append("")

            # phase -> 类目 Top
            lines.append(f"### 类目 Top（按 focus_sum / 动作量排序，Top {int(categories_per_phase or 8)}）")
            lines.append("")
            try:
                if ac is None or ac.empty or "current_phase" not in ac.columns:
                    lines.append("- （无）")
                else:
                    sub = ac[ac["current_phase"] == p].copy()
                    if sub.empty:
                        lines.append("- （无）")
                    else:
                        # 聚合
                        tmp = sub.copy()
                        for c in ("focus_score", "delta_sales", "delta_spend"):
                            if c in tmp.columns:
                                tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)
                        for c in ("top_action_count", "top_blocked_action_count"):
                            if c in tmp.columns:
                                tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0).astype(int)

                        cat = (
                            tmp.groupby("product_category", dropna=False, as_index=False)
                            .agg(
                                asin_count=("asin", "count") if "asin" in tmp.columns else ("product_category", "size"),
                                focus_score_sum=("focus_score", "sum") if "focus_score" in tmp.columns else ("product_category", "size"),
                                delta_sales_sum=("delta_sales", "sum") if "delta_sales" in tmp.columns else ("product_category", "size"),
                                delta_spend_sum=("delta_spend", "sum") if "delta_spend" in tmp.columns else ("product_category", "size"),
                                top_action_count_sum=("top_action_count", "sum") if "top_action_count" in tmp.columns else ("product_category", "size"),
                                top_blocked_action_count_sum=("top_blocked_action_count", "sum")
                                if "top_blocked_action_count" in tmp.columns
                                else ("product_category", "size"),
                            )
                            .copy()
                        )
                        # 数值化/格式化
                        for c in ("focus_score_sum", "delta_sales_sum", "delta_spend_sum"):
                            if c in cat.columns:
                                cat[c] = pd.to_numeric(cat[c], errors="coerce").fillna(0.0).round(2)
                        for c in ("asin_count", "top_action_count_sum", "top_blocked_action_count_sum"):
                            if c in cat.columns:
                                cat[c] = pd.to_numeric(cat[c], errors="coerce").fillna(0).astype(int)
                        cat = cat.sort_values(["focus_score_sum", "top_action_count_sum"], ascending=[False, False])
                        view = cat.head(max(1, int(categories_per_phase or 8))).copy()
                        view["product_category"] = view["product_category"].map(lambda x: _cat_md_link(str(x or ""), "./category_drilldown.md"))
                        lines.append(
                            _df_to_md_table(
                                view,
                                [
                                    "product_category",
                                    "asin_count",
                                    "focus_score_sum",
                                    "delta_sales_sum",
                                    "delta_spend_sum",
                                    "top_action_count_sum",
                                    "top_blocked_action_count_sum",
                                ],
                            )
                        )
            except Exception:
                lines.append("- （无）")
            lines.append("")

            # phase -> ASIN Top
            lines.append(f"### Top ASIN（按 focus_score / 动作量排序，Top {int(asins_per_phase or 12)}）")
            lines.append("")
            try:
                if ac is None or ac.empty or "current_phase" not in ac.columns:
                    lines.append("- （无）")
                else:
                    sub = ac[ac["current_phase"] == p].copy()
                    if sub.empty:
                        lines.append("- （无）")
                    else:
                        # 排序：focus_score -> top_action_count -> ad_spend_roll
                        sort_cols2 = [c for c in ["focus_score", "top_action_count", "ad_spend_roll"] if c in sub.columns]
                        if sort_cols2:
                            sub = sub.sort_values(sort_cols2, ascending=[False] * len(sort_cols2))
                        view = sub.head(max(1, int(asins_per_phase or 12))).copy()
                        if "asin" in view.columns:
                            view["asin"] = view["asin"].map(lambda x: _asin_md_link(str(x or ""), "./asin_drilldown.md"))
                        if "product_category" in view.columns:
                            view["product_category"] = view["product_category"].map(lambda x: _cat_md_link(str(x or ""), "./category_drilldown.md"))
                        # 格式化
                        for col in ("focus_score", "ad_spend_roll", "delta_sales", "delta_spend", "marginal_tacos"):
                            if col in view.columns:
                                view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0.0).round(2 if col != "marginal_tacos" else 4)
                        for col in ("inventory", "top_action_count", "top_blocked_action_count"):
                            if col in view.columns:
                                view[col] = pd.to_numeric(view[col], errors="coerce").fillna(0).astype(int)

                        lines.append(
                            _df_to_md_table(
                                view,
                                [
                                    "product_category",
                                    "asin",
                                    "product_name",
                                    "inventory",
                                    "focus_score",
                                    "top_action_count",
                                    "top_blocked_action_count",
                                    "delta_sales",
                                    "delta_spend",
                                    "marginal_tacos",
                                    "ad_spend_roll",
                                ],
                            )
                        )
            except Exception:
                lines.append("- （无）")
            lines.append("")
            lines.append("[回到顶部](#top) | [返回 Dashboard](./dashboard.md) | [类目 Drilldown](./category_drilldown.md)")
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        return


def _merge_adjacent_phase_days(parts: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """
    合并相邻同 phase 的 (phase, days) 序列。

    说明：
    - 这是“展示层”的工具函数（用于 lifecycle_overview 的时间轴可视化），不改变任何指标口径/算数逻辑。
    - 会自动忽略 days<=0 的片段，并对 phase 做规范化（_norm_phase）。
    """
    try:
        out: List[Tuple[str, int]] = []
        for ph, days in parts or []:
            ph2 = _norm_phase(ph)
            d = _safe_int(days)
            if d <= 0:
                continue
            if out and out[-1][0] == ph2:
                out[-1] = (ph2, int(out[-1][1]) + int(d))
            else:
                out.append((ph2, int(d)))
        return out
    except Exception:
        return []


def _smooth_lifecycle_timeline_parts(
    parts: List[Tuple[str, int]],
    max_segments: int = 18,
    min_days: int = 2,
    max_min_days: int = 14,
) -> List[Tuple[str, int]]:
    """
    生命周期时间轴“可读性平滑”（仅用于 lifecycle_overview 的展示层）。

    痛点：生命周期按日判定时，phase 可能频繁来回，导致 timeline 出现“条纹爆炸”（很多 1-2 天的碎片段）。

    策略：
    1) 先合并相邻同 phase；
    2) 将短碎片段（<=min_days）合并到相邻段（若夹在同 phase 中间，则三段合并为一段）；
    3) 若段数仍超过 max_segments，则逐步提高阈值（直到 max_min_days）以控制段数。

    注意：只影响展示，不影响任何计算/口径。
    """
    try:
        segs = _merge_adjacent_phase_days(parts)
        if not segs:
            return []

        max_segments2 = max(3, int(max_segments or 0))
        min_days2 = max(1, int(min_days or 0))
        max_min_days2 = max(min_days2, int(max_min_days or 0))

        def _merge_short(segs0: List[Tuple[str, int]], thr: int) -> List[Tuple[str, int]]:
            s = _merge_adjacent_phase_days(segs0)
            if len(s) <= 1:
                return s
            changed = True
            while changed and len(s) > 1:
                changed = False
                for i, (ph, days) in enumerate(list(s)):
                    if int(days) > int(thr):
                        continue

                    # 1) 夹心：A - x - A => 直接合并为 A
                    if 0 < i < (len(s) - 1) and s[i - 1][0] == s[i + 1][0]:
                        new_ph = s[i - 1][0]
                        new_days = int(s[i - 1][1]) + int(days) + int(s[i + 1][1])
                        s = s[: i - 1] + [(new_ph, new_days)] + s[i + 2 :]
                        changed = True
                        break

                    # 2) 非夹心：合并到“更大”的邻居（更稳）
                    if i == 0:
                        # 合并到 next
                        nph, nd = s[i + 1]
                        s = [(nph, int(nd) + int(days))] + s[i + 2 :]
                        changed = True
                        break
                    if i == len(s) - 1:
                        # 合并到 prev
                        pph, pd = s[i - 1]
                        s = s[: i - 1] + [(pph, int(pd) + int(days))]
                        changed = True
                        break

                    left_days = int(s[i - 1][1])
                    right_days = int(s[i + 1][1])
                    if right_days >= left_days:
                        nph, nd = s[i + 1]
                        s = s[:i] + [(nph, int(nd) + int(days))] + s[i + 2 :]
                    else:
                        pph, pd = s[i - 1]
                        s = s[: i - 1] + [(pph, int(pd) + int(days))] + s[i + 1 :]
                    changed = True
                    break
            return _merge_adjacent_phase_days(s)

        thr = int(min_days2)
        while len(segs) > max_segments2 and thr <= max_min_days2:
            segs = _merge_short(segs, thr)
            thr += 1

        return _merge_adjacent_phase_days(segs)
    except Exception:
        return _merge_adjacent_phase_days(parts)


def write_lifecycle_overview_md(
    out_path: Path,
    shop: str,
    stage: str,
    date_start: str,
    date_end: str,
    lifecycle_segments: Optional[pd.DataFrame],
    lifecycle_board: Optional[pd.DataFrame],
    asin_cockpit: Optional[pd.DataFrame] = None,
    max_categories: int = 30,
    asins_per_category: int = 60,
    max_total_asins: int = 800,
) -> None:
    """
    生成“生命周期时间轴总览”：按「类目 → ASIN」展示当前周期的阶段轨迹。

    设计目标：
    - 运营/你一眼看到：每个 ASIN 当前在哪个阶段，以及当前周期内阶段如何演进（时间轴）
    - 不引入外部依赖：纯 Markdown + write_report_html_from_md（HTML 转换时用 JS 渲染时间轴）
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _df_to_md_table(df: pd.DataFrame, cols: List[str]) -> str:
            try:
                if df is None or df.empty:
                    return ""
                view = df.copy()
                cols2 = [c for c in cols if c in view.columns]
                if not cols2:
                    cols2 = list(view.columns)[:8]
                view = view[cols2].copy()
                for c in cols2:
                    view[c] = view[c].map(lambda x, _c=c: _format_md_cell(_c, x))
                header = "| " + " | ".join(cols2) + " |"
                sep = "| " + " | ".join(["---"] * len(cols2)) + " |"
                body = ["| " + " | ".join(row) + " |" for row in view.values.tolist()]
                return "\n".join([header, sep] + body)
            except Exception:
                return ""

        seg = lifecycle_segments.copy() if isinstance(lifecycle_segments, pd.DataFrame) else pd.DataFrame()
        if seg is None:
            seg = pd.DataFrame()
        if seg.empty or "asin" not in seg.columns or "phase" not in seg.columns:
            lines = [
                '<a id="top"></a>',
                f"# {shop} 生命周期时间轴（类目 → ASIN）",
                "",
                f"- 阶段: `{stage}`",
                f"- 时间范围: `{date_start} ~ {date_end}`",
                "- 口径说明: 未标注的累计指标=主窗口；标注 compare/Δ 的为近N天 vs 前N天（日期见表内 recent/prev）",
                "",
                "- （无：缺少 lifecycle_segments；请确认 `reports/产品分析` 是否包含该店铺的按日数据）",
                "",
                "[返回 Dashboard](./dashboard.md)",
                "",
            ]
            out_path.write_text("\n".join(lines), encoding="utf-8")
            return

        seg = seg.copy()
        # 规范化字段
        seg["asin"] = seg["asin"].astype(str).fillna("").str.upper().str.strip()
        if "cycle_id" in seg.columns:
            seg["cycle_id"] = pd.to_numeric(seg["cycle_id"], errors="coerce").fillna(0).astype(int)
        else:
            seg["cycle_id"] = 0
        if "segment_id" in seg.columns:
            seg["segment_id"] = pd.to_numeric(seg["segment_id"], errors="coerce").fillna(0).astype(int)
        else:
            seg["segment_id"] = 0
        seg["phase"] = seg["phase"].map(_norm_phase)
        if "days" in seg.columns:
            seg["days"] = pd.to_numeric(seg["days"], errors="coerce").fillna(0).astype(int)
        else:
            seg["days"] = 0

        if "date_start" in seg.columns:
            seg["date_start"] = seg["date_start"].astype(str).fillna("").str.strip()
        else:
            seg["date_start"] = ""
        if "date_end" in seg.columns:
            seg["date_end"] = seg["date_end"].astype(str).fillna("").str.strip()
        else:
            seg["date_end"] = ""

        if "product_category" in seg.columns:
            seg["product_category"] = seg["product_category"].map(_norm_product_category)
        else:
            seg["product_category"] = "（未分类）"
        if "product_name" in seg.columns:
            seg["product_name"] = seg["product_name"].astype(str).fillna("").str.strip()
        else:
            seg["product_name"] = ""

        # 当前周期/当前阶段：优先用 lifecycle_board（更贴近“当前状态”）
        cycle_map: Dict[str, int] = {}
        phase_map: Dict[str, str] = {}
        if isinstance(lifecycle_board, pd.DataFrame) and (not lifecycle_board.empty) and "asin" in lifecycle_board.columns:
            b = lifecycle_board.copy()
            b["asin"] = b["asin"].astype(str).fillna("").str.upper().str.strip()
            if "cycle_id" in b.columns:
                b["cycle_id"] = pd.to_numeric(b["cycle_id"], errors="coerce").fillna(0).astype(int)
            if "current_phase" in b.columns:
                b["current_phase"] = b["current_phase"].map(_norm_phase)
            for _, r in b.iterrows():
                a = str(r.get("asin", "") or "").strip().upper()
                if not a:
                    continue
                try:
                    cycle_map[a] = int(r.get("cycle_id", 0) or 0)
                except Exception:
                    cycle_map[a] = 0
                try:
                    phase_map[a] = _norm_phase(r.get("current_phase", "unknown"))
                except Exception:
                    phase_map[a] = "unknown"

        if cycle_map:
            map_rows = [{"asin": k, "current_cycle_id": int(v), "current_phase": phase_map.get(k, "unknown")} for k, v in cycle_map.items()]
            map_df = pd.DataFrame(map_rows)
            seg = seg.merge(map_df, on="asin", how="left")
            seg = seg[(seg["current_cycle_id"].isna()) | (seg["cycle_id"] == seg["current_cycle_id"])].copy()
        else:
            # 兜底：没有 board 时，以每个 ASIN 的“最新 date_end”所在 cycle_id 作为当前周期
            try:
                tmp = (
                    seg.groupby(["asin", "cycle_id"], dropna=False, as_index=False)
                    .agg(last_end=("date_end", "max"))
                    .sort_values(["asin", "last_end"], ascending=[True, False])
                )
                pick = tmp.drop_duplicates("asin")[["asin", "cycle_id"]].rename(columns={"cycle_id": "current_cycle_id"})
                seg = seg.merge(pick, on="asin", how="left")
                seg = seg[(seg["current_cycle_id"].isna()) | (seg["cycle_id"] == seg["current_cycle_id"])].copy()
            except Exception:
                seg["current_cycle_id"] = seg["cycle_id"]

        # ASIN cockpit：补齐“当前方向/风险”的轻量字段，并作为排序依据
        cockpit_map: Dict[str, Dict[str, object]] = {}
        try:
            ac = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
            if ac is None:
                ac = pd.DataFrame()
            if not ac.empty and "asin" in ac.columns:
                ac = ac.copy()
                ac["asin"] = ac["asin"].astype(str).fillna("").str.upper().str.strip()
                for _, r in ac.iterrows():
                    a = str(r.get("asin", "") or "").strip().upper()
                    if not a or a in cockpit_map:
                        continue
                    cockpit_map[a] = {
                        "focus_score": r.get("focus_score", 0.0),
                        "profit_direction": str(r.get("profit_direction", "") or "").strip().lower(),
                        "inventory_cover_days_7d": r.get("inventory_cover_days_7d", ""),
                        "sales_per_day_7d": r.get("sales_per_day_7d", ""),
                        # 用于“类目结构 Top5”的影响权重（展示层排序）
                        "ad_spend_roll": r.get("ad_spend_roll", ""),
                        "sales_recent_7d": r.get("sales_recent_7d", ""),
                        # 近期窗口信号（可用于 hint）：更贴近“当下怎么调”
                        "phase_trend_14d": r.get("phase_trend_14d", ""),
                        "phase_change_days_ago": r.get("phase_change_days_ago", ""),
                        "delta_sales": r.get("delta_sales", ""),
                        "delta_spend": r.get("delta_spend", ""),
                        "top_action_count": r.get("top_action_count", ""),
                        "top_blocked_action_count": r.get("top_blocked_action_count", ""),
                    }
        except Exception:
            cockpit_map = {}

        # 汇总到“每 ASIN 一行”
        rows: List[Dict[str, object]] = []
        for asin, g in seg.groupby("asin", dropna=False):
            a = str(asin or "").strip().upper()
            if not a or a.lower() == "nan":
                continue
            gg = g.copy()
            try:
                cid = int(pd.to_numeric(gg.get("cycle_id", 0), errors="coerce").fillna(0).astype(int).iloc[0])
            except Exception:
                cid = 0
            try:
                cat = _norm_product_category(gg.get("product_category", "（未分类）").iloc[0])
            except Exception:
                cat = "（未分类）"
            name = ""
            try:
                names = [str(x or "").strip() for x in gg.get("product_name", "").tolist() if str(x or "").strip() and str(x).strip().lower() != "nan"]
                name = names[0] if names else ""
            except Exception:
                name = ""

            try:
                gg = gg.sort_values(["date_start", "segment_id"], ascending=[True, True]).copy()
            except Exception:
                gg = gg.copy()

            d0 = ""
            d1 = ""
            try:
                d0 = str(gg.get("date_start", "").iloc[0] or "").strip()
                d1 = str(gg.get("date_end", "").iloc[-1] or "").strip()
            except Exception:
                d0, d1 = "", ""

            cur = phase_map.get(a, "")
            if not cur:
                try:
                    cur = _norm_phase(gg.get("phase", "").iloc[-1])
                except Exception:
                    cur = "unknown"

            cm = cockpit_map.get(a, {})
            chg_days_val = _safe_int(cm.get("phase_change_days_ago", 0))
            recent_flag = True if (0 < int(chg_days_val) <= 14) else False

            raw_parts: List[Tuple[str, int]] = []
            total_days = 0
            for _, r in gg.iterrows():
                ph = _norm_phase(r.get("phase", "unknown"))
                days = _safe_int(r.get("days", 0))
                if days <= 0:
                    continue
                total_days += int(days)
                raw_parts.append((ph, int(days)))

            # timeline 平滑：避免“条纹爆炸”（很多 1-2 天碎片段），只影响展示层
            # 规则：周期越长，默认允许更粗一点（更可读）
            max_segments = 18 if int(total_days) >= 120 else 14
            min_days = 3 if int(total_days) >= 120 else 2
            smooth_parts = _smooth_lifecycle_timeline_parts(
                raw_parts,
                max_segments=max_segments,
                min_days=min_days,
            )
            parts2 = [f"{ph}={int(days)}" for ph, days in smooth_parts if int(days) > 0]
            tl = "tl:" + ";".join(parts2) if parts2 else ""
            if tl and recent_flag:
                tl = tl + "|chg14"
            tl_cell = f"`{tl}`" if tl else ""

            strategy_tag = "排查"
            if cur in {"pre_launch", "launch"}:
                strategy_tag = "上新打基础"
            elif cur in {"growth", "stable", "mature"}:
                strategy_tag = "放量/效率"
            elif cur in {"decline", "inactive"}:
                strategy_tag = "止损/收口"

            def _fmt_signed(x: object, nd: int = 1) -> str:
                """格式化带符号数字（用于 hint 展示），例如 +12.3 / -5。"""
                try:
                    v = float(pd.to_numeric(x, errors="coerce"))
                    if pd.isna(v):
                        return ""
                    s = f"{v:+.{int(nd)}f}"
                    s = s.rstrip("0").rstrip(".")
                    return s
                except Exception:
                    return ""

            def _fmt_usd_signed(x: object, nd: int = 1) -> str:
                """格式化带符号金额（用于 hint 展示），例如 +$12.3 / -$5。"""
                s = _fmt_signed(x, nd=nd)
                if not s:
                    return ""
                if s[0] in {"+", "-"}:
                    return s[0] + "$" + s[1:]
                return "$" + s

            def _fmt_num(x: object, nd: int = 1) -> str:
                try:
                    v = float(pd.to_numeric(x, errors="coerce"))
                    if pd.isna(v):
                        return ""
                    s = f"{v:.{int(nd)}f}"
                    s = s.rstrip("0").rstrip(".")
                    return s
                except Exception:
                    return ""

            def _fmt_usd(x: object, nd: int = 1) -> str:
                s = _fmt_num(x, nd=nd)
                return f"${s}" if s else ""

            def _fmt_pct(x: object, nd: int = 1) -> str:
                try:
                    v = float(pd.to_numeric(x, errors="coerce"))
                    if pd.isna(v):
                        return ""
                    return f"{v * 100:.{int(nd)}f}%"
                except Exception:
                    return ""

            hint_parts: List[str] = []
            pdx = str(cm.get("profit_direction", "") or "").strip().lower()
            if pdx in {"reduce", "scale"}:
                hint_parts.append(f"profit={pdx}")

            # 近期趋势：让生命周期页也能看到“最近是不是在走弱/走强”
            trend14 = str(cm.get("phase_trend_14d", "") or "").strip().lower()
            if trend14 in {"up", "down"}:
                hint_parts.append(f"trend14={trend14}")
            chg_days = chg_days_val
            if 0 < int(chg_days) <= 14:
                hint_parts.append(f"⚡chg={int(chg_days)}d")
            ds = _fmt_signed(cm.get("delta_sales", ""), nd=1)
            if ds:
                hint_parts.append(f"ΔSales={ds}")
            dd = _fmt_usd_signed(cm.get("delta_spend", ""), nd=1)
            if dd:
                hint_parts.append(f"ΔSpend={dd}")

            cov7 = cm.get("inventory_cover_days_7d", "")
            try:
                cov7f = float(pd.to_numeric(cov7, errors="coerce"))
                if cov7f > 0 and cov7f < 7:
                    hint_parts.append("cover7d<7")
            except Exception:
                pass
            try:
                act_cnt = int(float(pd.to_numeric(cm.get("top_action_count", 0), errors="coerce") or 0))
                blk_cnt = int(float(pd.to_numeric(cm.get("top_blocked_action_count", 0), errors="coerce") or 0))
                if blk_cnt > 0:
                    hint_parts.append(f"blocked={blk_cnt}")
                elif act_cnt > 0:
                    hint_parts.append(f"top_actions={act_cnt}")
            except Exception:
                pass
            hint = " | ".join(hint_parts)

            sales7 = _fmt_usd(cm.get("sales_recent_7d", ""), nd=1)
            spend_roll = _fmt_usd(cm.get("ad_spend_roll", ""), nd=1)
            tacos_roll = _fmt_pct(cm.get("tacos_roll", ""), nd=1)
            cover7 = _fmt_num(cm.get("inventory_cover_days_7d", ""), nd=1)
            delta_sales = _fmt_usd_signed(cm.get("delta_sales", ""), nd=1)
            delta_spend = _fmt_usd_signed(cm.get("delta_spend", ""), nd=1)

            rows.append(
                {
                    "product_category": cat,
                    "asin": a,
                    "product_name": name,
                    "current_phase": cur,
                    "cycle_id": cid,
                    "cycle_range": f"{d0}~{d1} ({int(total_days)}d)" if (d0 or d1) else f"({int(total_days)}d)",
                    "timeline": tl_cell,
                    "strategy": strategy_tag,
                    "hint": hint,
                    "sales_recent_7d": sales7,
                    "ad_spend_roll": spend_roll,
                    "tacos_roll": tacos_roll,
                    "inventory_cover_days_7d": f"{cover7}d" if cover7 else "",
                    "delta_sales": delta_sales,
                    "delta_spend": delta_spend,
                    "_focus_score": float(pd.to_numeric(cm.get("focus_score", 0.0), errors="coerce") or 0.0),
                }
            )

        if not rows:
            lines = [
                '<a id="top"></a>',
                f"# {shop} 生命周期时间轴（类目 → ASIN）",
                "",
                f"- 阶段: `{stage}`",
                f"- 时间范围: `{date_start} ~ {date_end}`",
                "- 口径说明: 未标注的累计指标=主窗口；标注 compare/Δ 的为近N天 vs 前N天（日期见表内 recent/prev）",
                "",
                "- （无：本次未生成有效 lifecycle_segments；可能产品分析数据为空/缺列）",
                "",
                "[返回 Dashboard](./dashboard.md)",
                "",
            ]
            out_path.write_text("\n".join(lines), encoding="utf-8")
            return

        df = pd.DataFrame(rows)
        try:
            df["product_category"] = df["product_category"].map(_norm_product_category)
            df = df.sort_values(["product_category", "_focus_score", "asin"], ascending=[True, False, True]).copy()
        except Exception:
            pass

        try:
            cat_stat = (
                df.groupby("product_category", dropna=False, as_index=False)
                .agg(asin_count=("asin", "nunique"), focus_sum=("_focus_score", "sum"))
                .sort_values(["asin_count", "focus_sum"], ascending=[False, False])
            )
            cat_list = [str(x).strip() for x in cat_stat["product_category"].tolist() if str(x).strip()]
        except Exception:
            cat_list = sorted([str(x).strip() for x in df["product_category"].unique().tolist() if str(x).strip()])

        cat_list = cat_list[: max(1, int(max_categories or 30))]

        # 生命周期闭环（全链条追踪）+ 近期重点
        loop_lines: List[str] = []
        highlight_lines: List[str] = []
        phase_dist_lines: List[str] = []
        try:
            ac2 = asin_cockpit.copy() if isinstance(asin_cockpit, pd.DataFrame) else pd.DataFrame()
            if ac2 is None:
                ac2 = pd.DataFrame()
            if not ac2.empty and "asin" in ac2.columns:
                ac2 = ac2.copy()
                ac2["asin"] = ac2["asin"].astype(str).fillna("").str.upper().str.strip()
                ac2 = ac2[ac2["asin"] != ""].copy()

                if "product_category" in ac2.columns:
                    ac2["product_category"] = ac2["product_category"].map(_norm_product_category)
                else:
                    ac2["product_category"] = "（未分类）"
                if "product_name" not in ac2.columns:
                    ac2["product_name"] = ""
                if "current_phase" in ac2.columns:
                    ac2["current_phase"] = ac2["current_phase"].map(_norm_phase)
                else:
                    ac2["current_phase"] = ""

                def _num_col(col: str, default: float = 0.0) -> pd.Series:
                    """安全取数值列：缺列时返回同长度默认值 Series。"""
                    try:
                        if col in ac2.columns:
                            return pd.to_numeric(ac2[col], errors="coerce").fillna(default)
                        return pd.Series([default] * int(len(ac2)), index=ac2.index, dtype=float)
                    except Exception:
                        return pd.Series([default] * int(len(ac2)), index=ac2.index, dtype=float)

                def _int_col(col: str, default: int = 0) -> pd.Series:
                    """安全取整型列：缺列时返回同长度默认值 Series。"""
                    try:
                        if col in ac2.columns:
                            return pd.to_numeric(ac2[col], errors="coerce").fillna(default).astype(int)
                        return pd.Series([int(default)] * int(len(ac2)), index=ac2.index, dtype=int)
                    except Exception:
                        return pd.Series([int(default)] * int(len(ac2)), index=ac2.index, dtype=int)

                def _str_col(col: str, default: str = "") -> pd.Series:
                    """安全取字符串列：缺列时返回同长度默认值 Series。"""
                    try:
                        if col in ac2.columns:
                            return ac2[col].astype(str).fillna("").str.strip()
                        return pd.Series([str(default)] * int(len(ac2)), index=ac2.index, dtype=str)
                    except Exception:
                        return pd.Series([str(default)] * int(len(ac2)), index=ac2.index, dtype=str)

                ac2["_ad_spend_roll"] = _num_col("ad_spend_roll", 0.0)
                ac2["_delta_sales"] = _num_col("delta_sales", 0.0)
                ac2["_delta_spend"] = _num_col("delta_spend", 0.0)
                ac2["_focus"] = _num_col("focus_score", 0.0)
                ac2["_cover7"] = _num_col("inventory_cover_days_7d", 0.0)
                ac2["_oos_days"] = _int_col("oos_with_ad_spend_days", 0)
                ac2["_max_ad_spend_by_profit"] = _num_col("max_ad_spend_by_profit", 0.0)
                ac2["_overspend"] = (ac2["_ad_spend_roll"] - ac2["_max_ad_spend_by_profit"]).fillna(0.0)
                ac2["_profit_direction"] = _str_col("profit_direction", "").str.lower()
                ac2["_trend14"] = _str_col("phase_trend_14d", "").str.lower()
                ac2["_chg_days"] = _int_col("phase_change_days_ago", 0)

                def _short_name(x: object, n2: int = 28) -> str:
                    try:
                        s = str(x or "").strip()
                        if not s or s.lower() == "nan":
                            return ""
                        if len(s) <= int(n2):
                            return s
                        return s[: int(n2)] + "…"
                    except Exception:
                        return ""

                def _fmt_num(x: object, nd: int = 1) -> str:
                    try:
                        v = float(pd.to_numeric(x, errors="coerce"))
                        if pd.isna(v):
                            return ""
                        s = f"{v:.{int(nd)}f}"
                        s = s.rstrip("0").rstrip(".")
                        return s
                    except Exception:
                        return ""

                def _fmt_usd(x: object, nd: int = 1) -> str:
                    s = _fmt_num(x, nd=nd)
                    return f"${s}" if s else ""

                def _fmt_signed(x: object, nd: int = 1) -> str:
                    try:
                        v = float(pd.to_numeric(x, errors="coerce"))
                        if pd.isna(v):
                            return ""
                        s = f"{v:+.{int(nd)}f}"
                        s = s.rstrip("0").rstrip(".")
                        return s
                    except Exception:
                        return ""

                def _fmt_usd_signed(x: object, nd: int = 1) -> str:
                    s = _fmt_signed(x, nd=nd)
                    if not s:
                        return ""
                    if s[0] in {"+", "-"}:
                        return s[0] + "$" + s[1:]
                    return "$" + s

                picked_asins: set[str] = set()

                def _mk_line(priority: str, tag: str, title: str, r: pd.Series, extra: List[str]) -> str:
                    try:
                        asin = str(r.get("asin", "") or "").strip().upper()
                        cat = str(r.get("product_category", "（未分类）") or "").strip() or "（未分类）"
                        name = _short_name(r.get("product_name", ""), 28)
                        phase = _norm_phase(r.get("current_phase", ""))
                        asin_link = _asin_md_link(asin, "./asin_drilldown.md") if asin else ""
                        cat_link = _cat_md_link(cat, "./category_drilldown.md") if cat else "（未分类）"
                        phase_link = _phase_md_link(phase, "./phase_drilldown.md") if phase else ""
                        left = f"{cat_link} / {asin_link}"
                        if name:
                            left += f" {name}"
                        details = [x for x in (extra or []) if str(x).strip()]
                        # 默认把 phase 放进“证据”，避免标题重复堆字段
                        if phase_link:
                            details = [f"phase={phase_link}"] + details
                        details_txt = "；".join(details)
                        if details_txt:
                            return f"- `{priority}` `{tag}` {title}：{left}（{details_txt}）"
                        return f"- `{priority}` `{tag}` {title}：{left}"
                    except Exception:
                        return ""

                def _pick_one(
                    df0: pd.DataFrame,
                    sort_cols: List[str],
                    asc: List[bool],
                    priority: str,
                    tag: str,
                    title: str,
                    extra_fn,
                ) -> bool:
                    try:
                        if df0 is None or df0.empty:
                            return False
                        d = df0.sort_values(sort_cols, ascending=asc).copy()
                        for _, rr in d.iterrows():
                            asin = str(rr.get("asin", "") or "").strip().upper()
                            if not asin or asin in picked_asins:
                                continue
                            line = _mk_line(priority, tag, title, rr, extra_fn(rr))
                            line = str(line or "").strip()
                            if not line:
                                continue
                            highlight_lines.append(line)
                            picked_asins.add(asin)
                            return True
                        return False
                    except Exception:
                        return False

                # ===== Top 异常（最多 3 条）=====
                anomalies = 0
                # 1) 断货仍烧钱（P0，优先止损）
                if anomalies < 3:
                    _pick_one(
                        ac2[(ac2["_oos_days"] > 0) & (ac2["_ad_spend_roll"] > 0)].copy(),
                        sort_cols=["_oos_days", "_ad_spend_roll"],
                        asc=[False, False],
                        priority="P0",
                        tag="止损",
                        title="断货仍烧钱",
                        extra_fn=lambda r: [
                            f"oos_days={int(r.get('_oos_days', 0) or 0)}",
                            f"AdSpend={_fmt_usd(r.get('_ad_spend_roll', 0.0), nd=1)}",
                            (f"ΔSales={_fmt_signed(r.get('_delta_sales', 0.0), nd=1)}" if _fmt_signed(r.get('_delta_sales', 0.0), nd=1) else ""),
                            (f"ΔSpend={_fmt_usd_signed(r.get('_delta_spend', 0.0), nd=1)}" if _fmt_usd_signed(r.get('_delta_spend', 0.0), nd=1) else ""),
                        ],
                    )
                    anomalies = len([x for x in highlight_lines if x])

                # 2) 加花费无增量（P0：优先排查/止损）
                if anomalies < 3:
                    _pick_one(
                        ac2[(ac2["_delta_spend"] > 0) & (ac2["_delta_sales"] <= 0) & (ac2["_ad_spend_roll"] > 0)].copy(),
                        sort_cols=["_delta_spend", "_ad_spend_roll"],
                        asc=[False, False],
                        priority="P0",
                        tag="排查",
                        title="加花费无增量",
                        extra_fn=lambda r: [
                            (f"ΔSales={_fmt_signed(r.get('_delta_sales', 0.0), nd=1)}" if _fmt_signed(r.get('_delta_sales', 0.0), nd=1) else ""),
                            (f"ΔSpend={_fmt_usd_signed(r.get('_delta_spend', 0.0), nd=1)}" if _fmt_usd_signed(r.get('_delta_spend', 0.0), nd=1) else ""),
                            f"AdSpend={_fmt_usd(r.get('_ad_spend_roll', 0.0), nd=1)}",
                        ],
                    )
                    anomalies = len([x for x in highlight_lines if x])

                # 3) 阶段走弱仍在花费（P1：优先找根因）
                if anomalies < 3:
                    _pick_one(
                        ac2[(ac2["_trend14"] == "down") & (ac2["_ad_spend_roll"] > 0)].copy(),
                        sort_cols=["_ad_spend_roll", "_focus"],
                        asc=[False, False],
                        priority="P1",
                        tag="排查",
                        title="阶段走弱仍在花费",
                        extra_fn=lambda r: [
                            "trend14=down",
                            (f"chg={int(r.get('_chg_days', 0) or 0)}d" if int(r.get("_chg_days", 0) or 0) > 0 else ""),
                            (f"ΔSales={_fmt_signed(r.get('_delta_sales', 0.0), nd=1)}" if _fmt_signed(r.get('_delta_sales', 0.0), nd=1) else ""),
                            (f"ΔSpend={_fmt_usd_signed(r.get('_delta_spend', 0.0), nd=1)}" if _fmt_usd_signed(r.get('_delta_spend', 0.0), nd=1) else ""),
                            f"AdSpend={_fmt_usd(r.get('_ad_spend_roll', 0.0), nd=1)}",
                        ],
                    )
                    anomalies = len([x for x in highlight_lines if x])

                # 4) 利润承受度超限（P1：优先止损收口；当上面信号不足时兜底补齐）
                if anomalies < 3:
                    _pick_one(
                        ac2[
                            (ac2["_profit_direction"] == "reduce")
                            & (ac2["_max_ad_spend_by_profit"] > 0)
                            & (ac2["_overspend"] > 0)
                            & (ac2["_ad_spend_roll"] > 0)
                        ].copy(),
                        sort_cols=["_overspend", "_ad_spend_roll"],
                        asc=[False, False],
                        priority="P1",
                        tag="止损",
                        title="利润承受度超限",
                        extra_fn=lambda r: [
                            f"超额={_fmt_usd(r.get('_overspend', 0.0), nd=1)}",
                            f"AdSpend={_fmt_usd(r.get('_ad_spend_roll', 0.0), nd=1)}",
                            f"上限={_fmt_usd(r.get('_max_ad_spend_by_profit', 0.0), nd=1)}",
                        ],
                    )
                    anomalies = len([x for x in highlight_lines if x])

                # ===== Top 机会（最多 2 条）=====
                opportunities = 0
                # 1) 可放量候选（利润方向=scale 且 cover7>=21）
                if opportunities < 2:
                    ok_phases = {"growth", "stable", "mature"}
                    _pick_one(
                        ac2[
                            (ac2["_profit_direction"] == "scale")
                            & (ac2["_cover7"] >= 21)
                            & (ac2["current_phase"].astype(str).str.lower().isin(ok_phases))
                        ].copy(),
                        sort_cols=["_focus", "_delta_sales", "_cover7"],
                        asc=[False, False, False],
                        priority="P1",
                        tag="放量",
                        title="可放量候选（库存/利润允许）",
                        extra_fn=lambda r: [
                            (f"cover7d={_fmt_num(r.get('_cover7', 0.0), nd=1)}" if _fmt_num(r.get('_cover7', 0.0), nd=1) else ""),
                            (f"ΔSales={_fmt_signed(r.get('_delta_sales', 0.0), nd=1)}" if _fmt_signed(r.get('_delta_sales', 0.0), nd=1) else ""),
                            (f"ΔSpend={_fmt_usd_signed(r.get('_delta_spend', 0.0), nd=1)}" if _fmt_usd_signed(r.get('_delta_spend', 0.0), nd=1) else ""),
                        ],
                    )
                    opportunities = max(0, len(highlight_lines) - anomalies)

                # 2) 近期走强（trend14=up 且 ΔSales>0）
                if opportunities < 2:
                    _pick_one(
                        ac2[(ac2["_trend14"] == "up") & (ac2["_delta_sales"] > 0)].copy(),
                        sort_cols=["_delta_sales", "_focus"],
                        asc=[False, False],
                        priority="P2",
                        tag="放量",
                        title="近期走强（验证放量节奏）",
                        extra_fn=lambda r: [
                            "trend14=up",
                            (f"chg={int(r.get('_chg_days', 0) or 0)}d" if int(r.get("_chg_days", 0) or 0) > 0 else ""),
                            (f"ΔSales={_fmt_signed(r.get('_delta_sales', 0.0), nd=1)}" if _fmt_signed(r.get('_delta_sales', 0.0), nd=1) else ""),
                            (f"ΔSpend={_fmt_usd_signed(r.get('_delta_spend', 0.0), nd=1)}" if _fmt_usd_signed(r.get('_delta_spend', 0.0), nd=1) else ""),
                        ],
                    )
                    opportunities = max(0, len(highlight_lines) - anomalies)

                # 如果不足 3 条，用“Top focus”兜底补齐到 3（仍然只做聚焦展示）
                while len(highlight_lines) < 3 and (ac2 is not None and not ac2.empty):
                    ok = _pick_one(
                        ac2.copy(),
                        sort_cols=["_focus", "_ad_spend_roll"],
                        asc=[False, False],
                        priority="P2",
                        tag="排查",
                        title="关注（高 focus，优先看 Action Board）",
                        extra_fn=lambda r: [
                            (f"focus={_fmt_num(r.get('_focus', 0.0), nd=1)}" if _fmt_num(r.get('_focus', 0.0), nd=1) else ""),
                            (f"AdSpend={_fmt_usd(r.get('_ad_spend_roll', 0.0), nd=1)}" if _fmt_usd(r.get('_ad_spend_roll', 0.0), nd=1) else ""),
                        ],
                    )
                    if not ok:
                        break

                # 控制上限：最多 5 条
                highlight_lines = [x for x in highlight_lines if str(x).strip()][:5]
        except Exception:
            highlight_lines = []

        # 生命周期闭环（全链条追踪）：阶段流转概览 + ASIN 闭环追踪表
        try:
            lb = lifecycle_board.copy() if isinstance(lifecycle_board, pd.DataFrame) else pd.DataFrame()
            if lb is None:
                lb = pd.DataFrame()
            if not lb.empty and "asin" in lb.columns:
                lb = lb.copy()
                lb["asin"] = lb["asin"].astype(str).fillna("").str.upper().str.strip()
                lb = lb[lb["asin"] != ""].copy()
                if "current_phase" in lb.columns:
                    lb["current_phase"] = lb["current_phase"].map(_norm_phase)
                if "prev_phase" in lb.columns:
                    lb["prev_phase"] = lb["prev_phase"].map(_norm_phase)
                if "phase_change_days_ago" in lb.columns:
                    lb["phase_change_days_ago"] = pd.to_numeric(lb["phase_change_days_ago"], errors="coerce").fillna(0).astype(int)
                if "phase_changed_recent_14d" in lb.columns:
                    lb["phase_changed_recent_14d"] = pd.to_numeric(lb["phase_changed_recent_14d"], errors="coerce").fillna(0).astype(int)

            # 1) 阶段流转统计（近14天）
            trans_table = ""
            try:
                if not lb.empty and ("prev_phase" in lb.columns) and ("current_phase" in lb.columns):
                    t = lb[(lb["prev_phase"] != "") & (lb["current_phase"] != "") & (lb["prev_phase"] != lb["current_phase"])].copy()
                    if not t.empty:
                        if "phase_change_days_ago" in t.columns:
                            t["_recent"] = (t["phase_change_days_ago"] > 0) & (t["phase_change_days_ago"] <= 14)
                        elif "phase_changed_recent_14d" in t.columns:
                            t["_recent"] = t["phase_changed_recent_14d"] > 0
                        else:
                            t["_recent"] = False
                        t["transition"] = t["prev_phase"] + "→" + t["current_phase"]
                        stat = (
                            t.groupby("transition", dropna=False, as_index=False)
                            .agg(total=("asin", "nunique"), recent_14d=("_recent", "sum"))
                            .copy()
                        )
                        stat = stat.sort_values(["recent_14d", "total"], ascending=[False, False]).copy()
                        trans_table = _df_to_md_table(stat, ["transition", "total", "recent_14d"])
            except Exception:
                trans_table = ""

            # 2) ASIN 闭环追踪（Top 30）
            try:
                view = df.copy()
                if not view.empty:
                    prev_map: Dict[str, Dict[str, object]] = {}
                    if not lb.empty and "asin" in lb.columns:
                        for _, r in lb.iterrows():
                            a = str(r.get("asin", "") or "").strip().upper()
                            if not a or a in prev_map:
                                continue
                            prev_map[a] = {
                                "prev_phase": r.get("prev_phase", ""),
                                "phase_change_days_ago": r.get("phase_change_days_ago", 0),
                                "phase_trend_14d": r.get("phase_trend_14d", ""),
                            }
                    view["prev_phase"] = view["asin"].map(lambda a: prev_map.get(str(a or "").strip().upper(), {}).get("prev_phase", ""))
                    view["phase_change_days_ago"] = view["asin"].map(
                        lambda a: prev_map.get(str(a or "").strip().upper(), {}).get("phase_change_days_ago", 0)
                    )
                    view["phase_trend_14d"] = view["asin"].map(
                        lambda a: prev_map.get(str(a or "").strip().upper(), {}).get("phase_trend_14d", "")
                    )

                    def _cm_num(a: object, key: str) -> float:
                        try:
                            aa = str(a or "").strip().upper()
                            v = (cockpit_map.get(aa, {}) or {}).get(key, 0.0)
                            vv = pd.to_numeric(v, errors="coerce")
                            if pd.isna(vv):
                                return 0.0
                            return float(vv)
                        except Exception:
                            return 0.0

                    view["sales_recent_7d"] = view["asin"].map(lambda a: _cm_num(a, "sales_recent_7d"))
                    view["ad_spend_roll"] = view["asin"].map(lambda a: _cm_num(a, "ad_spend_roll"))
                    view["inventory_cover_days_7d"] = view["asin"].map(lambda a: _cm_num(a, "inventory_cover_days_7d"))
                    view["top_action_count"] = view["asin"].map(lambda a: _cm_num(a, "top_action_count"))
                    view["top_blocked_action_count"] = view["asin"].map(lambda a: _cm_num(a, "top_blocked_action_count"))

                    view["prev_phase"] = view["prev_phase"].map(_norm_phase)
                    view["phase_trend_14d"] = view["phase_trend_14d"].astype(str).str.strip().str.lower()
                    view["phase_change_days_ago"] = pd.to_numeric(view["phase_change_days_ago"], errors="coerce").fillna(0).astype(int)
                    view["inventory_cover_days_7d"] = pd.to_numeric(view["inventory_cover_days_7d"], errors="coerce").fillna(0.0)
                    view["sales_recent_7d"] = pd.to_numeric(view["sales_recent_7d"], errors="coerce").fillna(0.0)
                    view["ad_spend_roll"] = pd.to_numeric(view["ad_spend_roll"], errors="coerce").fillna(0.0)
                    view["top_action_count"] = pd.to_numeric(view["top_action_count"], errors="coerce").fillna(0.0).astype(int)
                    view["top_blocked_action_count"] = pd.to_numeric(view["top_blocked_action_count"], errors="coerce").fillna(0.0).astype(int)

                    view_full_raw = view.copy()
                    try:
                        view_focus_raw = view[
                            (view["phase_change_days_ago"] > 0)
                            | (view["phase_trend_14d"].isin(["up", "down"]))
                            | (view["top_action_count"] > 0)
                            | (view["top_blocked_action_count"] > 0)
                            | ((view["inventory_cover_days_7d"] > 0) & (view["inventory_cover_days_7d"] < 7))
                        ].copy()
                    except Exception:
                        view_focus_raw = view.copy()

                    def _short_name(x: object, n2: int = 24) -> str:
                        try:
                            s = str(x or "").strip()
                            if not s or s.lower() == "nan":
                                return ""
                            if len(s) <= int(n2):
                                return s
                            return s[: int(n2)] + "…"
                        except Exception:
                            return ""

                    def _trend_tag(x: object) -> str:
                        s = str(x or "").strip().lower()
                        if s == "down":
                            return "🔻down"
                        if s == "up":
                            return "🔺up"
                        return ""

                    def _decorate_loop_view(df0: pd.DataFrame) -> pd.DataFrame:
                        v = df0.copy()
                        v["product_name"] = v["product_name"].map(lambda x: _short_name(x, 24))
                        v["product_category"] = v["product_category"].map(lambda x: _norm_product_category(x))
                        v["item"] = v.apply(
                            lambda r: (str(r.get("product_name", "") or "").strip() + " / " + str(r.get("product_category", "") or "").strip()).strip(" /"),
                            axis=1,
                        )
                        v["item"] = v["item"].map(lambda x: _short_name(x, 28))
                        v["phase_trend_14d"] = v["phase_trend_14d"].map(_trend_tag)
                        v["phase_change_days_ago"] = v["phase_change_days_ago"].map(lambda x: f"{int(x)}d" if int(x) > 0 else "")
                        v["inventory_cover_days_7d"] = v["inventory_cover_days_7d"].map(lambda x: f"{x:.1f}d" if x else "")
                        v["sales_recent_7d"] = v["sales_recent_7d"].map(lambda x: f"${x:.1f}" if x > 0 else "$0")
                        v["ad_spend_roll"] = v["ad_spend_roll"].map(lambda x: f"${x:.1f}" if x > 0 else "$0")
                        def _fmt_delta(x: object) -> str:
                            try:
                                s = str(x or "").strip()
                                if s:
                                    return s
                                v = float(pd.to_numeric(x, errors="coerce"))
                                if pd.isna(v):
                                    return ""
                                sign = "+" if v > 0 else "-" if v < 0 else ""
                                return f"{sign}${abs(v):.1f}" if sign else ""
                            except Exception:
                                return str(x or "").strip()

                        v["delta_sales"] = v["delta_sales"].map(_fmt_delta)
                        v["delta_spend"] = v["delta_spend"].map(_fmt_delta)
                        v["actions"] = v.apply(
                            lambda r: f"{int(r.get('top_action_count', 0) or 0)}/{int(r.get('top_blocked_action_count', 0) or 0)}",
                            axis=1,
                        )
                        v["asin"] = v["asin"].map(lambda x: _asin_md_link(str(x or ""), "./asin_drilldown.md"))
                        v["current_phase"] = v["current_phase"].map(lambda x: _phase_md_link(str(x or ""), "./phase_drilldown.md"))
                        v["prev_phase"] = v["prev_phase"].map(lambda x: _phase_md_link(str(x or ""), "./phase_drilldown.md"))
                        v["phase_path"] = v.apply(
                            lambda r: f"{r.get('prev_phase','')}→{r.get('current_phase','')}".strip("→"),
                            axis=1,
                        )
                        return v

                    view = view_full_raw.sort_values(["_focus_score", "asin"], ascending=[False, True]).copy().head(30)
                    view = _decorate_loop_view(view)
                    view_focus = view_focus_raw.sort_values(["_focus_score", "asin"], ascending=[False, True]).copy().head(15)
                    view_focus = _decorate_loop_view(view_focus)

                    # 闭环指标卡片
                    total_asins = 0
                    try:
                        total_asins = int(df["asin"].nunique()) if (df is not None and not df.empty and "asin" in df.columns) else 0
                    except Exception:
                        total_asins = 0
                    change_14d = 0
                    try:
                        if not lb.empty and "phase_change_days_ago" in lb.columns:
                            change_14d = int(lb[(lb["phase_change_days_ago"] > 0) & (lb["phase_change_days_ago"] <= 14)]["asin"].nunique())
                        elif not lb.empty and "phase_changed_recent_14d" in lb.columns:
                            change_14d = int(lb[lb["phase_changed_recent_14d"] > 0]["asin"].nunique())
                    except Exception:
                        change_14d = 0
                    down_14d = 0
                    try:
                        if not df.empty and "asin" in df.columns:
                            for a in df["asin"].unique().tolist():
                                aa = str(a or "").strip().upper()
                                if not aa:
                                    continue
                                trend = str((cockpit_map.get(aa, {}) or {}).get("phase_trend_14d", "") or "").strip().lower()
                                if trend == "down":
                                    down_14d += 1
                    except Exception:
                        down_14d = 0
                    action_asins = 0
                    blocked_asins = 0
                    try:
                        if not df.empty and "asin" in df.columns:
                            for a in df["asin"].unique().tolist():
                                aa = str(a or "").strip().upper()
                                if not aa:
                                    continue
                                act = float(pd.to_numeric((cockpit_map.get(aa, {}) or {}).get("top_action_count", 0), errors="coerce") or 0)
                                blk = float(pd.to_numeric((cockpit_map.get(aa, {}) or {}).get("top_blocked_action_count", 0), errors="coerce") or 0)
                                if act > 0:
                                    action_asins += 1
                                if blk > 0:
                                    blocked_asins += 1
                    except Exception:
                        action_asins = 0
                        blocked_asins = 0

                    loop_lines = [
                        '<a id="loop"></a>',
                        "## 0.5) 生命周期闭环（全链条追踪）",
                        "",
                        "- 闭环路径：数据采集 → 阶段诊断 → 动作执行（Action Board/Watchlists）→ 复盘回流",
                        "- 该区块只做“全链条追踪”展示，不改变任何口径或评分。",
                        "",
                        '<div class="loop-flow">',
                        '<div class="loop-step">数据采集</div>',
                        '<div class="loop-arrow">→</div>',
                        '<div class="loop-step">阶段诊断</div>',
                        '<div class="loop-arrow">→</div>',
                        '<div class="loop-step">动作执行</div>',
                        '<div class="loop-arrow">→</div>',
                        '<div class="loop-step">复盘回流</div>',
                        "</div>",
                        '<div class="loop-metrics">',
                        f'<div class="loop-metric"><div class="k">ASIN 总数</div><div class="v">{total_asins}</div></div>',
                        f'<div class="loop-metric"><div class="k">近14天阶段变化</div><div class="v">{change_14d}</div></div>',
                        f'<div class="loop-metric"><div class="k">trend14=down</div><div class="v">{down_14d}</div></div>',
                        f'<div class="loop-metric"><div class="k">有动作 ASIN</div><div class="v">{action_asins}</div></div>',
                        f'<div class="loop-metric"><div class="k">被阻断 ASIN</div><div class="v">{blocked_asins}</div></div>',
                        "</div>",
                    ]
                    if trans_table:
                        loop_lines += [
                            "### 阶段流转概览（近14天）",
                            "",
                            trans_table,
                            "",
                        ]
                    loop_lines += [
                        "### ASIN 闭环追踪（聚焦，Top 15）",
                        "",
                        _df_to_md_table(
                            view_focus,
                            [
                                "asin",
                                "item",
                                "phase_path",
                                "phase_change_days_ago",
                                "phase_trend_14d",
                                "sales_recent_7d",
                                "ad_spend_roll",
                                "inventory_cover_days_7d",
                                "delta_sales",
                                "delta_spend",
                                "actions",
                            ],
                        ),
                        "",
                        "<details>",
                        "<summary>全量追踪（展开）</summary>",
                        "",
                        _df_to_md_table(
                            view,
                            [
                                "asin",
                                "item",
                                "phase_path",
                                "phase_change_days_ago",
                                "phase_trend_14d",
                                "sales_recent_7d",
                                "ad_spend_roll",
                                "inventory_cover_days_7d",
                                "delta_sales",
                                "delta_spend",
                                "actions",
                            ],
                        ),
                        "</details>",
                        "",
                        "- 说明：`phase_path`=上一阶段→当前阶段；`phase_change_days_ago`=阶段变化距今天数；`phase_trend_14d`=近14天趋势；`actions`=动作数/阻断数。",
                        "- 提示：动作执行优先看 `Action Board / Watchlists`，此表用于“全链条复盘”。",
                        "",
                    ]
            except Exception:
                loop_lines = []
        except Exception:
            loop_lines = []

        # 阶段分布小结：让你先判断“结构问题”（down/inactive 占比）再看单品细节
        try:
            try:
                tmp = df.copy()
                if "asin" in tmp.columns:
                    tmp["asin"] = tmp["asin"].astype(str).fillna("").str.upper().str.strip()
                else:
                    tmp["asin"] = ""
                tmp = tmp[tmp["asin"] != ""].copy()

                if "current_phase" in tmp.columns:
                    tmp["current_phase"] = tmp["current_phase"].map(_norm_phase)
                else:
                    tmp["current_phase"] = "unknown"
            except Exception:
                tmp = pd.DataFrame(columns=["asin", "current_phase"])

            total_asins = 0
            try:
                total_asins = int(tmp["asin"].nunique()) if (tmp is not None and not tmp.empty) else 0
            except Exception:
                total_asins = 0

            if total_asins > 0:
                # 1) phase 分布（按 ASIN 去重）
                try:
                    stat = (
                        tmp.groupby("current_phase", dropna=False, as_index=False)
                        .agg(asin_count=("asin", "nunique"))
                        .copy()
                    )
                    stat = stat.rename(columns={"current_phase": "phase"}).copy()
                    stat["share"] = stat["asin_count"].map(lambda x: (float(x) / float(total_asins)) if total_asins > 0 else 0.0)

                    # 排序：按阶段顺序 > 数量
                    order_map = {
                        "pre_launch": 0,
                        "launch": 1,
                        "growth": 2,
                        "stable": 3,
                        "mature": 4,
                        "decline": 5,
                        "inactive": 6,
                        "unknown": 9,
                    }
                    stat["_order"] = stat["phase"].map(lambda x: order_map.get(_norm_phase(x), 9))
                    try:
                        stat = stat.sort_values(["_order", "asin_count"], ascending=[True, False]).copy()
                    except Exception:
                        pass

                    # phase 链接到 phase_drilldown
                    try:
                        stat["phase"] = stat["phase"].map(lambda x: _phase_md_link(_norm_phase(x), "./phase_drilldown.md"))
                    except Exception:
                        pass
                    # share 格式化为百分比（1 位小数）
                    try:
                        stat["share"] = stat["share"].map(lambda x: f"{float(x) * 100:.1f}%")
                    except Exception:
                        stat["share"] = stat["share"].astype(str)

                    phase_table = _df_to_md_table(stat, ["phase", "asin_count", "share"])
                except Exception:
                    phase_table = ""

                # 2) down/inactive 占比（用 cockpit_map 的 trend14；inactive 用 df 的 current_phase）
                inactive_cnt = 0
                down_cnt = 0
                try:
                    inactive_cnt = int(tmp[tmp["current_phase"] == "inactive"]["asin"].nunique())
                except Exception:
                    inactive_cnt = 0
                try:
                    for a in tmp["asin"].unique().tolist():
                        aa = str(a or "").strip().upper()
                        if not aa:
                            continue
                        trend = str((cockpit_map.get(aa, {}) or {}).get("phase_trend_14d", "") or "").strip().lower()
                        if trend == "down":
                            down_cnt += 1
                except Exception:
                    down_cnt = 0

                def _pct(n: int, d: int) -> str:
                    try:
                        if d <= 0:
                            return "0%"
                        return f"{(float(n) / float(d)) * 100:.1f}%"
                    except Exception:
                        return ""

                phase_dist_lines = [
                    '<a id="phase_dist"></a>',
                    "### 阶段分布小结（当前周期）",
                    "",
                    f"- 总 ASIN: `{int(total_asins)}`",
                    f"- trend14=down（近期走弱）: `{int(down_cnt)}` ({_pct(int(down_cnt), int(total_asins))})",
                    f"- current_phase=inactive（停滞）: `{int(inactive_cnt)}` ({_pct(int(inactive_cnt), int(total_asins))})",
                ]
                if phase_table:
                    phase_dist_lines += ["", phase_table]
                phase_dist_lines.append("")
        except Exception:
            phase_dist_lines = []

        # 类目结构 Top5：哪个类目 down/inactive 占比最高（优先排查）
        cat_struct_lines: List[str] = []
        try:
            base = df.copy()
            if base is None:
                base = pd.DataFrame()
            if (not base.empty) and ("asin" in base.columns):
                base["asin"] = base["asin"].astype(str).fillna("").str.upper().str.strip()
                base = base[base["asin"] != ""].copy()
            if base.empty:
                cat_struct_lines = []
            else:
                if "product_category" in base.columns:
                    base["product_category"] = base["product_category"].map(_norm_product_category)
                else:
                    base["product_category"] = "（未分类）"
                if "current_phase" in base.columns:
                    base["current_phase"] = base["current_phase"].map(_norm_phase)
                else:
                    base["current_phase"] = "unknown"

                base["_flag_inactive"] = base["current_phase"].map(lambda x: 1 if _norm_phase(x) == "inactive" else 0)
                base["_flag_down"] = base["asin"].map(
                    lambda a: 1
                    if str((cockpit_map.get(str(a or "").strip().upper(), {}) or {}).get("phase_trend_14d", "") or "").strip().lower()
                    == "down"
                    else 0
                )
                # 影响权重：类目近7天销售 / 滚动花费（用于排序；只影响展示）
                def _cm_num(asin: object, key: str) -> float:
                    try:
                        a = str(asin or "").strip().upper()
                        v = (cockpit_map.get(a, {}) or {}).get(key, 0.0)
                        vv = pd.to_numeric(v, errors="coerce")
                        if pd.isna(vv):
                            return 0.0
                        return float(vv)
                    except Exception:
                        return 0.0

                base["_ad_spend_roll"] = base["asin"].map(lambda a: _cm_num(a, "ad_spend_roll"))
                base["_sales_recent_7d"] = base["asin"].map(lambda a: _cm_num(a, "sales_recent_7d"))
                base["_flag_risk"] = (base["_flag_inactive"].astype(int) > 0) | (base["_flag_down"].astype(int) > 0)
                base["_flag_risk"] = base["_flag_risk"].map(lambda x: 1 if bool(x) else 0)

                stat = (
                    base.groupby("product_category", dropna=False, as_index=False)
                    .agg(
                        asin_count=("asin", "nunique"),
                        risk_any_count=("_flag_risk", "sum"),
                        down_count=("_flag_down", "sum"),
                        inactive_count=("_flag_inactive", "sum"),
                        ad_spend_roll_sum=("_ad_spend_roll", "sum"),
                        sales_recent_7d_sum=("_sales_recent_7d", "sum"),
                    )
                    .copy()
                )
                stat["asin_count"] = pd.to_numeric(stat["asin_count"], errors="coerce").fillna(0).astype(int)
                stat["risk_any_count"] = pd.to_numeric(stat["risk_any_count"], errors="coerce").fillna(0).astype(int)
                stat["down_count"] = pd.to_numeric(stat["down_count"], errors="coerce").fillna(0).astype(int)
                stat["inactive_count"] = pd.to_numeric(stat["inactive_count"], errors="coerce").fillna(0).astype(int)
                stat["ad_spend_roll_sum"] = pd.to_numeric(stat["ad_spend_roll_sum"], errors="coerce").fillna(0.0)
                stat["sales_recent_7d_sum"] = pd.to_numeric(stat["sales_recent_7d_sum"], errors="coerce").fillna(0.0)

                stat["risk_any_share"] = stat.apply(
                    lambda r: (float(r.get("risk_any_count", 0) or 0) / float(r.get("asin_count", 0) or 1))
                    if int(r.get("asin_count", 0) or 0) > 0
                    else 0.0,
                    axis=1,
                )
                stat["down_share"] = stat.apply(
                    lambda r: (float(r.get("down_count", 0) or 0) / float(r.get("asin_count", 0) or 1))
                    if int(r.get("asin_count", 0) or 0) > 0
                    else 0.0,
                    axis=1,
                )
                stat["inactive_share"] = stat.apply(
                    lambda r: (float(r.get("inactive_count", 0) or 0) / float(r.get("asin_count", 0) or 1))
                    if int(r.get("asin_count", 0) or 0) > 0
                    else 0.0,
                    axis=1,
                )

                # 业务影响权重（USD）：Sales7d + AdSpend(roll)
                stat["impact_usd"] = (stat["sales_recent_7d_sum"].clip(lower=0.0) + stat["ad_spend_roll_sum"].clip(lower=0.0)).fillna(0.0)
                stat["risk_weighted"] = stat["risk_any_share"].astype(float) * stat["impact_usd"].astype(float)

                try:
                    stat = stat.sort_values(
                        ["risk_weighted", "risk_any_share", "impact_usd", "risk_any_count", "asin_count"],
                        ascending=[False, False, False, False, False],
                    ).copy()
                except Exception:
                    pass

                top = stat.head(5).copy()
                if top is not None and not top.empty:
                    def _pct(x: object) -> str:
                        try:
                            v = float(pd.to_numeric(x, errors="coerce"))
                            if pd.isna(v):
                                return ""
                            return f"{v * 100:.1f}%"
                        except Exception:
                            return ""

                    def _usd(x: object) -> str:
                        try:
                            v = float(pd.to_numeric(x, errors="coerce"))
                            if pd.isna(v):
                                return "$0"
                            s = f"{v:.1f}"
                            s = s.rstrip("0").rstrip(".")
                            return f"${s}"
                        except Exception:
                            return "$0"

                    view = top.copy()
                    view["类目"] = view["product_category"].map(lambda x: _cat_md_link(_norm_product_category(x), "./category_drilldown.md"))
                    view["ASIN数"] = view["asin_count"].astype(int)
                    view["AdSpend(roll)"] = view["ad_spend_roll_sum"].map(_usd)
                    view["Sales7d"] = view["sales_recent_7d_sum"].map(_usd)
                    view["风险(Down|Inactive)"] = view.apply(
                        lambda r: f"{int(r.get('risk_any_count', 0) or 0)} ({_pct(r.get('risk_any_share', 0.0))})",
                        axis=1,
                    )
                    view["trend14=down"] = view.apply(
                        lambda r: f"{int(r.get('down_count', 0) or 0)} ({_pct(r.get('down_share', 0.0))})",
                        axis=1,
                    )
                    view["inactive"] = view.apply(
                        lambda r: f"{int(r.get('inactive_count', 0) or 0)} ({_pct(r.get('inactive_share', 0.0))})",
                        axis=1,
                    )

                    cat_struct_lines = [
                        '<a id="cat_struct"></a>',
                        "### 类目阶段结构 Top5（优先排查：Down/Inactive 占比最高）",
                        "",
                        "- `风险(Down|Inactive)`=该类目中 `trend14=down` 或 `current_phase=inactive` 的 ASIN 数量与占比。",
                        "- 排序权重：`风险占比 * (Sales7d + AdSpend(roll))`（避免小样本类目因 100% 占比排到最前）。",
                        "",
                        _df_to_md_table(view, ["类目", "ASIN数", "AdSpend(roll)", "Sales7d", "风险(Down|Inactive)", "trend14=down", "inactive"]),
                        "",
                    ]
        except Exception:
            cat_struct_lines = []

        lines: List[str] = []
        lines.append('<a id="top"></a>')
        lines.append(f"# {shop} 生命周期时间轴（类目 → ASIN）")
        lines.append("")
        lines.append(f"- 阶段: `{stage}`")
        lines.append(f"- 时间范围: `{date_start} ~ {date_end}`")
        lines.append("- 口径说明: 未标注的累计指标=主窗口；标注 compare/Δ 的为近N天 vs 前N天（日期见表内 recent/prev）")
        lines.append("")
        lines.append(
            "快速入口：[返回 Dashboard](./dashboard.md) | [近期重点](#highlights) | [生命周期闭环](#loop) | [阶段分布](#phase_dist) | [类目结构](#cat_struct) | [ASIN Drilldown](./asin_drilldown.md) | [Phase Drilldown](./phase_drilldown.md)"
        )
        lines.append("")

        lines.append('<a id="highlights"></a>')
        lines.append("## 0) 近期重点（近7/14天，3-5条）")
        lines.append("")
        if highlight_lines:
            lines.extend(highlight_lines)
        else:
            lines.append("- （暂无足够信号；建议先看 [Action Board](../dashboard/action_board.csv) 与 [Watchlists](./dashboard.md#watchlists)）")
        lines.append("")

        if loop_lines:
            lines.extend(loop_lines)
        if phase_dist_lines:
            lines.extend(phase_dist_lines)
        if cat_struct_lines:
            lines.extend(cat_struct_lines)

        lines.append("## 1) 怎么读")
        lines.append("")
        lines.append("- 每行=一个 ASIN 的“当前补货周期（cycle_id）”生命周期轨迹（时间轴）。")
        lines.append("- `current_phase` 是当前阶段；右侧 `timeline` 展示该周期内各阶段持续时长。")
        lines.append("- “近期重点”基于近7/14天窗口信号（ΔSales/ΔSpend/趋势/库存覆盖等）做聚焦展示。")
        lines.append("- “生命周期闭环”展示阶段流转概览 + ASIN 全链条追踪（便于复盘从诊断到执行的路径）。")
        lines.append("- “阶段分布小结”回答结构问题：各 phase 数量与 down/inactive 占比（用于判断整体是否在走弱/停滞）。")
        lines.append("- “类目阶段结构 Top5”帮助你先定位“最需要优先看的类目”。")
        lines.append("- `timeline` 为了可读性会对短碎片段做“平滑合并”（仅展示层，不改变任何算数/口径）。")
        lines.append("- 时间轴红色描边=近14天发生过阶段变化（chg14）。")
        lines.append("- 阶段判定规则（简述）：以 rolling 销量峰值为参照；首单前为 pre_launch，首单后 <= launch_days 为 launch；未到成熟且斜率≥0 为 growth；接近峰值且斜率≈0 为 mature；低于 decline_ratio 且斜率<0 为 decline；其余为 stable；不活跃为 inactive。")
        lines.append("- 这是解释层/可视化：执行仍以 `Action Board / 解锁任务 / Watchlists` 为准。")
        lines.append("")

        lines.append("## 2) 阶段→策略速查（建议）")
        lines.append("")
        lines.append("| phase | 含义（粗粒度） | 建议策略（默认） |")
        lines.append("| --- | --- | --- |")
        lines.append("| pre_launch | 未可售/未稳定出单 | 少量试投拉数据；优先补 Listing/评价/变体/库存准备 |")
        lines.append("| launch | 刚起量/刚过冷启动 | 放量前先稳转化与库存；重点看关键词主题与结构健康 |")
        lines.append("| growth | 增长期 | 库存与利润允许时放量；优先做预算迁移/出价/扩词；被阻断先解锁 |")
        lines.append("| stable/mature | 稳定期 | 做效率与结构优化（否词/分组/预算结构），保持 TACOS/毛利 |")
        lines.append("| decline | 衰退期 | 优先止损收口（否词/降价/收紧预算），必要时做重建/清库存 |")
        lines.append("| inactive | 停滞/无有效销售 | 默认控量/停投，先排查 Listing/库存/变体/归因口径 |")
        lines.append("")

        lines.append("## 3) 类目索引")
        lines.append("")
        for cat in cat_list:
            cid = _cat_anchor_id(cat)
            lines.append(f"- [{cat}](#{cid})")
        lines.append("")

        lines.append("## 4) 类目 → ASIN 时间轴")
        lines.append("")

        total_written = 0
        for cat in cat_list:
            if total_written >= int(max_total_asins or 0) and int(max_total_asins or 0) > 0:
                break
            cid = _cat_anchor_id(cat)
            lines.append(f'<a id="{cid}"></a>')
            lines.append(f"### {cat}")
            lines.append("")
            sub = df[df["product_category"] == cat].copy()
            if sub.empty:
                lines.append("- （无）")
                lines.append("")
                continue

            try:
                sub = sub.sort_values(["_focus_score", "asin"], ascending=[False, True]).copy()
            except Exception:
                pass
            n = max(1, int(asins_per_category or 60))
            view = sub.head(n).copy()
            total_written += int(len(view))

            try:
                view["asin"] = view["asin"].map(lambda x: _asin_md_link(str(x or ""), "./asin_drilldown.md"))
            except Exception:
                pass
            try:
                view["current_phase"] = view["current_phase"].map(lambda x: _phase_md_link(str(x or ""), "./phase_drilldown.md"))
            except Exception:
                pass

            def _short_name(x: object, n2: int = 36) -> str:
                try:
                    s = str(x or "").strip()
                    if not s or s.lower() == "nan":
                        return ""
                    if len(s) <= int(n2):
                        return s
                    return s[: int(n2)] + "…"
                except Exception:
                    return ""

            try:
                view["product_name"] = view["product_name"].map(lambda x: _short_name(x, 36))
            except Exception:
                pass

            view = view.rename(
                columns={
                    "product_name": "商品",
                    "current_phase": "阶段",
                    "timeline": "时间轴",
                    "strategy": "策略",
                    "sales_recent_7d": "销售7d",
                    "ad_spend_roll": "花费(滚动)",
                    "tacos_roll": "TACOS(滚动)",
                    "inventory_cover_days_7d": "库存覆盖7d",
                    "delta_sales": "ΔSales(7d)",
                    "delta_spend": "ΔSpend(7d)",
                    "hint": "提示",
                }
            )
            # 卡片视图（Top 12）
            try:
                card_limit = min(12, int(len(view)))
                card_rows = view.head(card_limit).copy()
                if card_rows is not None and (not card_rows.empty):
                    lines.append("#### ASIN 时间轴卡片（Top 12）")
                    lines.append("")
                    lines.append("- 说明：红边=近14天阶段变化；`ΔSales/ΔSpend` 为近7天对比前7天。")
                    lines.append("")
                    lines.append('<div class="timeline-cards">')
                    for _, r in card_rows.iterrows():
                        asin_link = str(r.get("asin", "") or "").strip()
                        name = str(r.get("商品", "") or "").strip()
                        phase = str(r.get("阶段", "") or "").strip()
                        strategy = str(r.get("策略", "") or "").strip()
                        tl_raw = str(r.get("时间轴", "") or "").strip().strip("`")
                        tl_html = f"<code>{tl_raw}</code>" if tl_raw else ""
                        sales7 = str(r.get("销售7d", "") or "").strip()
                        spend = str(r.get("花费(滚动)", "") or "").strip()
                        tacos = str(r.get("TACOS(滚动)", "") or "").strip()
                        cover7 = str(r.get("库存覆盖7d", "") or "").strip()
                        d_sales = str(r.get("ΔSales(7d)", "") or "").strip()
                        d_spend = str(r.get("ΔSpend(7d)", "") or "").strip()
                        hint = str(r.get("提示", "") or "").strip()
                        phase_cls = "phase-" + re.sub(r"[^a-z0-9_\\-]+", "", _norm_phase(phase)) if phase else "phase-unknown"
                        card_cls = ""
                        hint_l = hint.lower()
                        if ("trend14=down" in hint_l) or ("止损" in strategy):
                            card_cls = " risk"
                        elif ("trend14=up" in hint_l) or ("放量" in strategy):
                            card_cls = " opp"
                        lines.append(f'<div class="timeline-card{card_cls}">')
                        lines.append(f'<div class="title">{asin_link} {name}</div>')
                        lines.append('<div class="badges">')
                        if phase:
                            lines.append(f'<span class="phase-badge {phase_cls}">{phase}</span>')
                        if strategy:
                            lines.append(f'<span class="phase-badge strategy">{strategy}</span>')
                        lines.append("</div>")
                        lines.append('<div class="timeline-row">')
                        if tl_html:
                            lines.append(f'<div class="timeline-wrap">{tl_html}</div>')
                        lines.append('<div class="metrics">')
                        if sales7:
                            lines.append(f"<span>Sales7d {sales7}</span>")
                        if spend:
                            lines.append(f"<span>花费 {spend}</span>")
                        if tacos:
                            lines.append(f"<span>TACOS {tacos}</span>")
                        if cover7:
                            lines.append(f"<span>Cover {cover7}</span>")
                        if d_sales:
                            lines.append(f"<span>ΔSales {d_sales}</span>")
                        if d_spend:
                            lines.append(f"<span>ΔSpend {d_spend}</span>")
                        lines.append("</div>")
                        lines.append("</div>")
                        if hint:
                            lines.append(f'<div class="sub">{hint}</div>')
                        lines.append("</div>")
                    lines.append("</div>")
            except Exception:
                pass

            # 表格视图（完整）
            show_cols = [
                "asin",
                "商品",
                "阶段",
                "时间轴",
                "策略",
                "销售7d",
                "花费(滚动)",
                "TACOS(滚动)",
                "库存覆盖7d",
                "ΔSales(7d)",
                "ΔSpend(7d)",
                "提示",
            ]
            lines.append("<details>")
            lines.append("<summary>表格视图（展开）</summary>")
            lines.append("")
            lines.append(_df_to_md_table(view, [c for c in show_cols if c in view.columns]))
            lines.append("</details>")
            lines.append("")
            lines.append("[回到顶部](#top) | [返回 Dashboard](./dashboard.md)")
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        return


def write_keyword_topics_drilldown_md(
    out_path: Path,
    shop: str,
    stage: str,
    date_start: str,
    date_end: str,
    keyword_segment_top: Optional[pd.DataFrame],
    keyword_action_hints: Optional[pd.DataFrame],
    keyword_asin_context: Optional[pd.DataFrame],
    policy: Optional[OpsPolicy] = None,
    max_segments: int = 12,
) -> None:
    """
    生成关键词主题（n-gram）下钻页：reports/keyword_topics.md。

    目标：
    - 解决“CSV/MD 太多，运营抓不到重点”的问题；
    - 提供一个固定的使用流程：segment（类目×阶段）→ topic（主题）→ 定位执行位置（campaign/ad_group）；
    - 明确 scale 放量阻断口径（库存/覆盖天数）。
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _df_to_md_table(df: pd.DataFrame, cols: List[str]) -> str:
            try:
                if df is None or df.empty:
                    return ""
                view = df.copy()
                cols2 = [c for c in cols if c in view.columns]
                if not cols2:
                    cols2 = list(view.columns)[:8]
                view = view[cols2].copy()
                for c in cols2:
                    view[c] = view[c].map(lambda x, _c=c: _format_md_cell(_c, x))
                header = "| " + " | ".join(cols2) + " |"
                sep = "| " + " | ".join(["---"] * len(cols2)) + " |"
                body = ["| " + " | ".join(row) + " |" for row in view.values.tolist()]
                return "\n".join([header, sep] + body)
            except Exception:
                return ""

        def _short(x: object, n: int = 120) -> str:
            try:
                s = str(x or "").replace("\n", " ").replace("|", "｜").strip()
                if not s or s.lower() == "nan":
                    return ""
                if len(s) <= int(n):
                    return s
                return s[: int(n)] + "…"
            except Exception:
                return ""

        # 阈值提示（对齐 ops_policy.json.inventory.*）
        low_inv_th = 20
        cover_days_th = 7.0
        try:
            if isinstance(policy, OpsPolicy):
                low_inv_th = int(getattr(policy, "low_inventory_threshold", 20) or 20)
                cover_days_th = float(getattr(policy, "block_scale_when_cover_days_below", 7.0) or 7.0)
        except Exception:
            low_inv_th = 20
            cover_days_th = 7.0

        lines: List[str] = []
        lines.append('<a id="top"></a>')
        lines.append(f"# {shop} Keyword Topics Drilldown（关键词主题下钻）")
        lines.append("")
        lines.append(f"- 阶段: `{stage}`")
        lines.append(f"- 时间范围: `{date_start} ~ {date_end}`")
        lines.append("- 口径说明: 未标注的累计指标=主窗口；标注 compare/Δ 的为近N天 vs 前N天（日期见表内 recent/prev）")
        lines.append("")
        lines.append(
            "快速入口：[返回 Dashboard](./dashboard.md) | "
            "[Segment Top CSV](../dashboard/keyword_topics_segment_top.csv) | "
            "[Action Hints CSV](../dashboard/keyword_topics_action_hints.csv) | "
            "[ASIN Context CSV](../dashboard/keyword_topics_asin_context.csv)"
        )
        lines.append("")

        # 1) 使用流程
        lines.append("## 1) 怎么用（推荐流程）")
        lines.append("")
        lines.append("- 第一步：打开 `../dashboard/keyword_topics_segment_top.csv`，先按「类目×生命周期阶段」选你要看的 segment。")
        lines.append("- 第二步：在该 segment 的 Top 主题里挑 1-3 个主题，去 `../dashboard/keyword_topics_action_hints.csv` 筛选（用 ngram 或 filter_contains）。")
        lines.append("- 第三步：执行前去 `../dashboard/keyword_topics_asin_context.csv` 看该主题对应的 Top ASIN 语境（库存覆盖/生命周期/利润方向）。")
        lines.append("- 放量规则：只对 `direction=scale` 生效，优先筛 `blocked=0`；`blocked_reason` 会告诉你被什么库存约束阻断。")
        lines.append("")

        # 2) Segment Top 摘要
        seg = keyword_segment_top.copy() if isinstance(keyword_segment_top, pd.DataFrame) else pd.DataFrame()
        if seg is None:
            seg = pd.DataFrame()
        lines.append("## 2) Segment Top（先选类目×阶段）")
        lines.append("")
        lines.append("- 说明：这里仅展示 Top 若干个 segment 的摘要；细节请直接打开 CSV 做筛选。")
        lines.append("")
        if seg.empty or "product_category" not in seg.columns or "current_phase" not in seg.columns:
            lines.append("- （无：本次缺少 segment_top 或无有效主题）")
        else:
            try:
                view = seg.copy()
                view["product_category"] = view["product_category"].map(_norm_product_category)
                view["current_phase"] = view["current_phase"].map(_norm_phase)
                for c in ("reduce_waste_spend_sum", "scale_sales_sum"):
                    if c in view.columns:
                        view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0.0).round(2)
                for c in ("reduce_topic_count", "scale_topic_count"):
                    if c in view.columns:
                        view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0).astype(int)
                if "reduce_top_topics" in view.columns:
                    view["reduce_top_topics"] = view["reduce_top_topics"].map(lambda x: _short(x, 120))
                if "scale_top_topics" in view.columns:
                    view["scale_top_topics"] = view["scale_top_topics"].map(lambda x: _short(x, 120))

                top_n = max(1, int(max_segments or 12))

                lines.append(f"### 2.1 浪费优先（按 reduce_waste_spend_sum 排序，Top {top_n}）")
                lines.append("")
                v1 = view.sort_values(["reduce_waste_spend_sum", "scale_sales_sum"], ascending=[False, False]).head(top_n).copy()
                lines.append(
                    _df_to_md_table(
                        v1,
                        ["product_category", "current_phase", "reduce_waste_spend_sum", "reduce_topic_count", "reduce_top_topics"],
                    )
                )
                lines.append("")

                lines.append(f"### 2.2 贡献优先（按 scale_sales_sum 排序，Top {top_n}）")
                lines.append("")
                v2 = view.sort_values(["scale_sales_sum", "reduce_waste_spend_sum"], ascending=[False, False]).head(top_n).copy()
                lines.append(
                    _df_to_md_table(
                        v2,
                        ["product_category", "current_phase", "scale_sales_sum", "scale_topic_count", "scale_top_topics"],
                    )
                )
            except Exception:
                lines.append("- （生成失败）")
        lines.append("")

        # 3) Action Hints 摘要（少量可执行项）
        hints = keyword_action_hints.copy() if isinstance(keyword_action_hints, pd.DataFrame) else pd.DataFrame()
        if hints is None:
            hints = pd.DataFrame()
        lines.append("## 3) 本周优先主题（从 Action Hints 里挑 Top）")
        lines.append("")
        lines.append("- 说明：这里只展示少量 Top 项，便于快速开始；完整清单请打开 CSV。")
        lines.append("")
        if hints.empty or "direction" not in hints.columns:
            lines.append("- （无）")
        else:
            try:
                hv = hints.copy()
                for c in ("priority", "direction", "ad_type", "ngram", "blocked_reason", "top_campaigns", "context_top_asins"):
                    if c in hv.columns:
                        hv[c] = hv[c].fillna("").astype(str)
                for c in ("n", "blocked"):
                    if c in hv.columns:
                        hv[c] = pd.to_numeric(hv[c], errors="coerce").fillna(0).astype(int)
                for c in ("waste_spend", "spend", "sales", "acos"):
                    if c in hv.columns:
                        hv[c] = pd.to_numeric(hv[c], errors="coerce").fillna(0.0)

                pr_rank = {"P0": 0, "P1": 1, "P2": 2}
                hv["_pr"] = hv.get("priority", "").map(lambda x: pr_rank.get(str(x), 9))

                # 3.1 reduce：按 waste_spend
                red = hv[hv["direction"].astype(str) == "reduce"].copy()
                if not red.empty:
                    red = red.sort_values(["_pr", "waste_spend", "spend"], ascending=[True, False, False]).head(8).copy()
                    if "top_campaigns" in red.columns:
                        red["top_campaigns"] = red["top_campaigns"].map(lambda x: _short(x, 120))
                    lines.append("### 3.1 Top 浪费主题（优先否词/降价）")
                    lines.append("")
                    lines.append(_df_to_md_table(red, ["priority", "ad_type", "n", "ngram", "waste_spend", "spend", "top_campaigns"]))
                    lines.append("")

                # 3.2 scale：按 sales，且提示 blocked
                sc = hv[hv["direction"].astype(str) == "scale"].copy()
                if not sc.empty:
                    sc = sc.sort_values(["blocked", "_pr", "sales", "spend"], ascending=[True, True, False, False]).head(8).copy()
                    if "blocked_reason" in sc.columns:
                        sc["blocked_reason"] = sc["blocked_reason"].map(lambda x: _short(x, 80))
                    if "context_top_asins" in sc.columns:
                        sc["context_top_asins"] = sc["context_top_asins"].map(lambda x: _short(x, 120))
                    lines.append("### 3.2 Top 贡献主题（可放量，但需先过库存阻断）")
                    lines.append("")
                    lines.append(
                        _df_to_md_table(
                            sc,
                            ["priority", "blocked", "blocked_reason", "ad_type", "n", "ngram", "sales", "acos", "context_top_asins"],
                        )
                    )
                    lines.append("")
            except Exception:
                lines.append("- （生成失败）")
        lines.append("")

        # 4) 阻断口径说明（把关键阈值写清楚）
        lines.append("## 4) 阻断口径说明（库存/覆盖）")
        lines.append("")
        lines.append("- `blocked/blocked_reason` 仅对 `direction=scale` 生效：用于避免“库存不足时误加码”。")
        lines.append(f"- 低库存阈值：`min_inventory ≤ {int(low_inv_th)}`（来自 `ops_policy.json.inventory.low_inventory_threshold`）")
        if float(cover_days_th or 0.0) > 0:
            lines.append(
                f"- 覆盖天数阈值：`min_cover7d < {float(cover_days_th):.0f}d`（来自 `ops_policy.json.inventory.block_scale_when_cover_days_below`）"
            )
        else:
            lines.append("- 覆盖天数阈值：已关闭（`block_scale_when_cover_days_below=0`）")
        lines.append("- `min_inventory/min_cover7d` 来自该主题的 Top ASIN 语境（`keyword_topics_asin_context.csv`），属于保守口径。")
        lines.append("")

        # 5) ASIN Context 说明（给运营一个“再确认”入口）
        ctx = keyword_asin_context.copy() if isinstance(keyword_asin_context, pd.DataFrame) else pd.DataFrame()
        if ctx is None:
            ctx = pd.DataFrame()
        lines.append("## 5) 主题→产品语境（ASIN Context）怎么用")
        lines.append("")
        lines.append("- 打开 `../dashboard/keyword_topics_asin_context.csv`：按 `ngram` 筛选，看该主题关联的 Top ASIN 是否处于合适阶段/是否有足够库存覆盖。")
        lines.append("- 如果该主题的 Top ASIN 多为 `profit_direction=reduce` 或覆盖天数不足，优先做“否词/收口/结构优化”，不要盲目加预算。")
        lines.append("")

        lines.append("[回到顶部](#top) | [返回 Dashboard](./dashboard.md)")
        lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        return


def write_dashboard_outputs(
    shop_dir: Path,
    shop: str,
    stage: str,
    date_start: str,
    date_end: str,
    diagnostics: Dict[str, object],
    product_analysis_shop: Optional[pd.DataFrame],
    lifecycle_board: Optional[pd.DataFrame],
    lifecycle_segments: Optional[pd.DataFrame],
    lifecycle_windows: Optional[pd.DataFrame],
    asin_campaign_map: Optional[pd.DataFrame],
    asin_top_search_terms: Optional[pd.DataFrame],
    asin_top_targetings: Optional[pd.DataFrame],
    asin_top_placements: Optional[pd.DataFrame],
    search_term_report: Optional[pd.DataFrame],
    actions: List[ActionCandidate],
    policy: OpsPolicy,
    render_md: bool = True,
    data_quality_hints: Optional[List[str]] = None,
    action_review: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """
    写入 dashboard/ 聚焦层文件。

    同时写入：
    - dashboard/action_board_full.csv（全量含重复，便于追溯）
    - dashboard/asin_cockpit.csv（ASIN 总览：focus + drivers + 动作量汇总）
    - dashboard/keyword_topics.csv（关键词主题 n-gram：压缩 search_term 报表的海量搜索词）
    - dashboard/keyword_topics_segment_top.csv（Segment Top：类目×生命周期 → Top 浪费/贡献主题，各 TopN）
    - dashboard/keyword_topics_action_hints.csv（主题建议：把主题落到 top_campaigns/top_ad_groups，形成可分派清单；scale 方向会标注/阻断库存风险）
    - dashboard/keyword_topics_asin_context.csv（主题→产品语境：只用高置信 term→asin，把主题落到类目/ASIN/生命周期/库存覆盖）
    - dashboard/keyword_topics_category_phase_summary.csv（主题→类目/生命周期汇总：先按类目/阶段看主题的 spend/sales/waste_spend，再下钻到 ASIN）

    返回（主要路径）：
    - dashboard/shop_scorecard.json
    - dashboard/asin_focus.csv
    - dashboard/action_board.csv（去重后的运营视图）
    - dashboard/category_summary.csv
    - reports/dashboard.md（可选）
    """
    scorecard_path = None
    asin_focus_path = None
    action_board_path = None
    action_board_full_path = None
    drivers_path = None
    category_summary_path = None
    asin_cockpit_path = None
    profit_reduce_watchlist_path = None
    dash_md_path = None
    keyword_topics_path = None
    keyword_topics_hints_path = None

    # 供 shop_scorecard “抓重点汇总”使用（即便中途异常也避免 UnboundLocalError）
    action_board: pd.DataFrame = pd.DataFrame()
    profit_reduce_watchlist: pd.DataFrame = pd.DataFrame()
    oos_watchlist: pd.DataFrame = pd.DataFrame()
    spend_up_no_sales_watchlist: pd.DataFrame = pd.DataFrame()
    phase_down_recent_watchlist: pd.DataFrame = pd.DataFrame()
    scale_opportunity_watchlist: pd.DataFrame = pd.DataFrame()

    try:
        dashboard_dir = shop_dir / "dashboard"
        dashboard_dir.mkdir(parents=True, exist_ok=True)

        # 1) shop_scorecard.json
        sc_json = build_shop_scorecard_json(shop=shop, stage=stage, date_start=date_start, date_end=date_end, diagnostics=diagnostics)
        scorecard_path = dashboard_dir / "shop_scorecard.json"
        scorecard_path.write_text(json_dumps(sc_json), encoding="utf-8")
        scorecard = (sc_json.get("scorecard") if isinstance(sc_json, dict) else {}) or {}
        # 预算迁移计划（会在后面结合机会池进一步补齐）
        budget_transfer_plan_effective: Dict[str, object] = (
            (diagnostics.get("budget_transfer_plan") if isinstance(diagnostics, dict) else {}) or {}
        )

        # 2) asin_focus.csv（同时生成全量 asin_focus_all，用于类目统计与 Action Board 联动）
        asin_focus_all = build_asin_focus(
            lifecycle_board=lifecycle_board,
            lifecycle_windows=lifecycle_windows,
            policy=policy,
            stage=stage,
            top_n=1000000,  # 足够大：用于映射与统计，不用于展示
        )
        # 2.0) 利润承受度字段前置（来自 diagnostics["asin_stages"]）
        try:
            asin_focus_all = enrich_asin_focus_with_profit_capacity(
                asin_focus_all=asin_focus_all,
                asin_stages=(diagnostics.get("asin_stages") if isinstance(diagnostics, dict) else []) or [],
            )
        except Exception:
            pass
        try:
            asin_focus_all = enrich_asin_focus_with_signal_scores(
                asin_focus_all=asin_focus_all,
                policy=policy,
                stage=stage,
            )
        except Exception:
            pass
        top_asins = int(getattr(policy, "dashboard_top_asins", 50) or 50)
        asin_focus = asin_focus_all.head(top_asins) if asin_focus_all is not None and not asin_focus_all.empty else pd.DataFrame()
        asin_focus_path = dashboard_dir / "asin_focus.csv"
        if asin_focus is not None and not asin_focus.empty:
            asin_focus.to_csv(asin_focus_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["asin"]).to_csv(asin_focus_path, index=False, encoding="utf-8-sig")

        # 2.1) drivers_top_asins.csv（变化来源：近7天 vs 前7天 Top ASIN）
        # 说明：drivers 的“动作数/阻断数”需要依赖 action_board，因此先构建 DataFrame，后续在 action_board 生成后再 enrich 并写文件。
        try:
            drivers_df = build_drivers_top_asins(
                scorecard=scorecard if isinstance(scorecard, dict) else {},
                lifecycle_board=lifecycle_board,
            )
        except Exception:
            drivers_df = pd.DataFrame()

        # 3) action_board.csv（默认去重后的运营视图）+ action_board_full.csv（全量便于追溯）
        action_board_full = build_action_board(actions, top_n=0)  # 先全量，再按新排序截断 TopN
        action_board_full = enrich_action_board_with_product(
            action_board=action_board_full,
            lifecycle_board=lifecycle_board,
            asin_campaign_map=asin_campaign_map,
            asin_top_search_terms=asin_top_search_terms,
            asin_top_targetings=asin_top_targetings,
            asin_top_placements=asin_top_placements,
        )
        action_board_full = score_action_board(action_board=action_board_full, asin_focus_all=asin_focus_all, policy=policy)
        # 操作手册联动：把动作表一键接回“怎么查/怎么做”的固定流程（不影响口径/算数逻辑）
        action_board_full = enrich_action_board_with_playbook_scene(action_board_full)
        # 全量文件（含重复）
        action_board_full_path = dashboard_dir / "action_board_full.csv"
        if action_board_full is not None and not action_board_full.empty:
            action_board_full.to_csv(action_board_full_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["priority", "action_type"]).to_csv(action_board_full_path, index=False, encoding="utf-8-sig")

        # 默认视图：去重后再截断 TopN
        action_board_all = dedup_action_board(action_board_full)
        top_actions = int(getattr(policy, "dashboard_top_actions", 60) or 60)
        action_board = action_board_all
        if action_board is not None and not action_board.empty and top_actions > 0:
            action_board = action_board_all.head(top_actions).reset_index(drop=True)
        action_board_path = dashboard_dir / "action_board.csv"
        if action_board is not None and not action_board.empty:
            action_board.to_csv(action_board_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["priority", "action_type"]).to_csv(action_board_path, index=False, encoding="utf-8-sig")

        # 3.01) campaign_action_view.csv（按 campaign 聚合 Action Board）
        campaign_action_view = None
        try:
            campaign_action_view = build_campaign_action_view(
                action_board=action_board_all if isinstance(action_board_all, pd.DataFrame) else action_board,
                max_rows=500,
                min_spend=10.0,
            )
        except Exception:
            campaign_action_view = pd.DataFrame()
        campaign_action_view_path = dashboard_dir / "campaign_action_view.csv"
        if campaign_action_view is not None and not campaign_action_view.empty:
            campaign_action_view.to_csv(campaign_action_view_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["campaign"]).to_csv(campaign_action_view_path, index=False, encoding="utf-8-sig")

        # drivers 补充“Top动作数/阻断数”（以全量去重 action_board_all 为准）
        try:
            drivers_df = enrich_drivers_with_action_counts(
                drivers_top_asins=drivers_df,
                action_board_dedup_all=action_board_all,
            )
        except Exception:
            pass

        # 写 drivers 文件（包含动作计数）
        drivers_path = dashboard_dir / "drivers_top_asins.csv"
        if drivers_df is not None and not drivers_df.empty:
            # 防御性补齐列
            if "top_action_count" not in drivers_df.columns:
                drivers_df["top_action_count"] = 0
            if "top_blocked_action_count" not in drivers_df.columns:
                drivers_df["top_blocked_action_count"] = 0
            drivers_df.to_csv(drivers_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(
                columns=[
                    "driver_type",
                    "rank",
                    "window_days",
                    "recent_start",
                    "recent_end",
                    "prev_start",
                    "prev_end",
                    "product_category",
                    "asin",
                    "product_name",
                    "current_phase",
                    "inventory",
                    "flag_low_inventory",
                    "flag_oos",
                    "delta_sales",
                    "delta_ad_spend",
                    "marginal_tacos",
                    "top_action_count",
                    "top_blocked_action_count",
                ]
            ).to_csv(drivers_path, index=False, encoding="utf-8-sig")

        # 3.5) asin_cockpit.csv（ASIN 总览：focus + drivers + 动作量汇总）
        asin_cockpit = None
        try:
            asin_cockpit = build_asin_cockpit(
                asin_focus_all=asin_focus_all,
                drivers_top_asins=drivers_df if isinstance(drivers_df, pd.DataFrame) else None,
                action_board_dedup_all=action_board_all,
            )
        except Exception:
            asin_cockpit = pd.DataFrame()
        asin_cockpit_path = dashboard_dir / "asin_cockpit.csv"
        if asin_cockpit is not None and not asin_cockpit.empty:
            asin_cockpit.to_csv(asin_cockpit_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["asin"]).to_csv(asin_cockpit_path, index=False, encoding="utf-8-sig")

        # 3.55) profit_reduce_watchlist.csv（利润方向=控量 且仍在烧钱：第二入口）
        profit_reduce_watchlist = None
        try:
            profit_reduce_watchlist = build_profit_reduce_watchlist(
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                max_rows=500,
                policy=policy,
            )
        except Exception:
            profit_reduce_watchlist = pd.DataFrame()
        profit_reduce_watchlist_path = dashboard_dir / "profit_reduce_watchlist.csv"
        if profit_reduce_watchlist is not None and not profit_reduce_watchlist.empty:
            profit_reduce_watchlist.to_csv(profit_reduce_watchlist_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["asin"]).to_csv(profit_reduce_watchlist_path, index=False, encoding="utf-8-sig")

        # 3.56) inventory_risk_watchlist.csv（库存告急仍投放：主入口预警）
        inventory_risk_watchlist = None
        try:
            inventory_risk_watchlist = build_inventory_risk_watchlist(
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                max_rows=500,
                policy=policy,
                spend_threshold=10.0,
            )
        except Exception:
            inventory_risk_watchlist = pd.DataFrame()
        inventory_risk_watchlist_path = dashboard_dir / "inventory_risk_watchlist.csv"
        if inventory_risk_watchlist is not None and not inventory_risk_watchlist.empty:
            inventory_risk_watchlist.to_csv(inventory_risk_watchlist_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["asin"]).to_csv(inventory_risk_watchlist_path, index=False, encoding="utf-8-sig")

        # 3.565) inventory_sigmoid_watchlist.csv（库存调速建议：Sigmoid）
        inventory_sigmoid_watchlist = None
        try:
            inventory_sigmoid_watchlist = build_inventory_sigmoid_watchlist(
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                max_rows=500,
                policy=policy,
            )
        except Exception:
            inventory_sigmoid_watchlist = pd.DataFrame()
        inventory_sigmoid_watchlist_path = dashboard_dir / "inventory_sigmoid_watchlist.csv"
        if inventory_sigmoid_watchlist is not None and not inventory_sigmoid_watchlist.empty:
            inventory_sigmoid_watchlist.to_csv(inventory_sigmoid_watchlist_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["asin"]).to_csv(inventory_sigmoid_watchlist_path, index=False, encoding="utf-8-sig")

        # 3.566) profit_guard_watchlist.csv（利润护栏：Break-even 提示）
        profit_guard_watchlist = None
        try:
            profit_guard_watchlist = build_profit_guard_watchlist(
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                max_rows=500,
                policy=policy,
            )
        except Exception:
            profit_guard_watchlist = pd.DataFrame()
        profit_guard_watchlist_path = dashboard_dir / "profit_guard_watchlist.csv"
        if profit_guard_watchlist is not None and not profit_guard_watchlist.empty:
            profit_guard_watchlist.to_csv(profit_guard_watchlist_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["asin"]).to_csv(profit_guard_watchlist_path, index=False, encoding="utf-8-sig")

        # 3.57) oos_with_ad_spend_watchlist.csv（断货仍烧钱：历史诊断入口）
        oos_watchlist = None
        try:
            oos_watchlist = build_oos_with_ad_spend_watchlist(
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                max_rows=500,
                policy=policy,
            )
        except Exception:
            oos_watchlist = pd.DataFrame()
        oos_watchlist_path = dashboard_dir / "oos_with_ad_spend_watchlist.csv"
        if oos_watchlist is not None and not oos_watchlist.empty:
            oos_watchlist.to_csv(oos_watchlist_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["asin"]).to_csv(oos_watchlist_path, index=False, encoding="utf-8-sig")

        # 3.58) spend_up_no_sales_watchlist.csv（加花费但销量不增：第二入口）
        spend_up_no_sales_watchlist = None
        try:
            spend_up_no_sales_watchlist = build_spend_up_no_sales_watchlist(
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                max_rows=500,
                policy=policy,
            )
        except Exception:
            spend_up_no_sales_watchlist = pd.DataFrame()
        spend_up_no_sales_watchlist_path = dashboard_dir / "spend_up_no_sales_watchlist.csv"
        if spend_up_no_sales_watchlist is not None and not spend_up_no_sales_watchlist.empty:
            spend_up_no_sales_watchlist.to_csv(spend_up_no_sales_watchlist_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["asin"]).to_csv(spend_up_no_sales_watchlist_path, index=False, encoding="utf-8-sig")

        # 3.585) phase_down_recent_watchlist.csv（近14天阶段走弱且仍在花费：第二入口）
        phase_down_recent_watchlist = None
        try:
            phase_down_recent_watchlist = build_phase_down_recent_watchlist(
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                max_rows=500,
                policy=policy,
            )
        except Exception:
            phase_down_recent_watchlist = pd.DataFrame()
        phase_down_recent_watchlist_path = dashboard_dir / "phase_down_recent_watchlist.csv"
        if phase_down_recent_watchlist is not None and not phase_down_recent_watchlist.empty:
            phase_down_recent_watchlist.to_csv(phase_down_recent_watchlist_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["asin"]).to_csv(phase_down_recent_watchlist_path, index=False, encoding="utf-8-sig")

        # 3.59) scale_opportunity_watchlist.csv（机会：可放量窗口/低花费高潜）
        scale_opportunity_watchlist = None
        try:
            scale_opportunity_watchlist = build_scale_opportunity_watchlist(
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                max_rows=500,
                policy=policy,
            )
        except Exception:
            scale_opportunity_watchlist = pd.DataFrame()
        scale_opportunity_watchlist_path = dashboard_dir / "scale_opportunity_watchlist.csv"
        if scale_opportunity_watchlist is not None and not scale_opportunity_watchlist.empty:
            scale_opportunity_watchlist.to_csv(scale_opportunity_watchlist_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["asin"]).to_csv(scale_opportunity_watchlist_path, index=False, encoding="utf-8-sig")

        # 3.59) opportunity_action_board.csv（机会→可执行动作：只保留可放量且未阻断）
        opportunity_action_board = None
        try:
            opportunity_action_board = build_opportunity_action_board(
                action_board_dedup_all=action_board_all,
                scale_opportunity_watchlist=scale_opportunity_watchlist,
                max_rows=500,
            )
        except Exception:
            opportunity_action_board = pd.DataFrame()
        opportunity_action_board_path = dashboard_dir / "opportunity_action_board.csv"
        if opportunity_action_board is not None and not opportunity_action_board.empty:
            opportunity_action_board.to_csv(opportunity_action_board_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["asin_hint", "action_type"]).to_csv(opportunity_action_board_path, index=False, encoding="utf-8-sig")

        # 3.60) budget_transfer_plan.csv（预算净迁移/回收：运营执行清单）
        budget_transfer_plan_table = None
        try:
            # 机会池联动：当本期没有“scale 侧 campaign”时，用机会 ASIN→Campaign 映射补齐净迁移去向
            budget_transfer_plan_effective = build_budget_transfer_plan_with_opportunities(
                budget_transfer_plan=budget_transfer_plan_effective,
                scale_opportunity_watchlist=scale_opportunity_watchlist,
                opportunity_action_board=opportunity_action_board,
                asin_campaign_map=asin_campaign_map,
                policy=policy,
            )
            budget_transfer_plan_table = build_budget_transfer_plan_table(
                budget_transfer_plan_effective,
                max_rows=500,
            )
        except Exception:
            budget_transfer_plan_table = pd.DataFrame()
        budget_transfer_plan_path = dashboard_dir / "budget_transfer_plan.csv"
        if budget_transfer_plan_table is not None and not budget_transfer_plan_table.empty:
            budget_transfer_plan_table.to_csv(budget_transfer_plan_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(
                columns=[
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
                    "to_opp_asin_count",
                    "to_opp_asins_top",
                    "to_opp_spend",
                    "to_opp_spend_share",
                    "to_opp_action_count",
                    "to_bucket",
                    "amount_usd_estimated",
                    "note",
                ]
            ).to_csv(budget_transfer_plan_path, index=False, encoding="utf-8-sig")

        # 3.61) unlock_scale_tasks.csv（放量解锁任务：可分工）
        unlock_scale_tasks_full_table = None
        try:
            unlock_scale_tasks_full_table = build_unlock_scale_tasks_table(
                (diagnostics.get("unlock_tasks") if isinstance(diagnostics, dict) else []) or [],
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                max_rows=2000,
            )
        except Exception:
            unlock_scale_tasks_full_table = pd.DataFrame()

        # 3.61.1) unlock_scale_tasks_full.csv（全量：便于追溯/深挖）
        unlock_scale_tasks_full_path = dashboard_dir / "unlock_scale_tasks_full.csv"
        if unlock_scale_tasks_full_table is not None and not unlock_scale_tasks_full_table.empty:
            unlock_scale_tasks_full_table.to_csv(unlock_scale_tasks_full_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(
                columns=[
                    "priority",
                    "owner",
                    "task_type",
                    "product_category",
                    "asin",
                    "product_name",
                    "current_phase",
                    "cycle_id",
                    "inventory",
                    "inventory_cover_days_7d",
                    "inventory_cover_days_30d",
                    "sales_per_day_7d",
                    "budget_gap_usd_est",
                    "profit_gap_usd_est",
                    "need",
                    "target",
                    "stage",
                    "direction",
                    "evidence",
                ]
            ).to_csv(unlock_scale_tasks_full_path, index=False, encoding="utf-8-sig")

        # 3.61.2) unlock_scale_tasks.csv（Top：运营分派/执行）
        unlock_scale_tasks_table = None
        try:
            unlock_scale_tasks_table = filter_unlock_scale_tasks_for_ops(unlock_scale_tasks_full_table, policy=policy)
        except Exception:
            unlock_scale_tasks_table = pd.DataFrame()
        unlock_scale_tasks_path = dashboard_dir / "unlock_scale_tasks.csv"
        if unlock_scale_tasks_table is not None and not unlock_scale_tasks_table.empty:
            unlock_scale_tasks_table.to_csv(unlock_scale_tasks_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(
                columns=[
                    "priority",
                    "owner",
                    "task_type",
                    "product_category",
                    "asin",
                    "product_name",
                    "current_phase",
                    "cycle_id",
                    "inventory",
                    "inventory_cover_days_7d",
                    "inventory_cover_days_30d",
                    "sales_per_day_7d",
                    "budget_gap_usd_est",
                    "profit_gap_usd_est",
                    "need",
                    "target",
                    "stage",
                    "direction",
                    "evidence",
                ]
            ).to_csv(unlock_scale_tasks_path, index=False, encoding="utf-8-sig")

        # 4) category_summary.csv（类目汇总：用于先看类目再看产品）
        category_summary = build_category_summary(product_analysis_shop=product_analysis_shop, lifecycle_board=lifecycle_board)
        # 额外补充“抓重点”的类目优先级维度（来自 ASIN Focus 全量评分）
        try:
            if asin_focus_all is not None and not asin_focus_all.empty and "product_category" in asin_focus_all.columns:
                f = asin_focus_all.copy()
                f["product_category"] = f["product_category"].map(_norm_product_category)
                if "focus_score" in f.columns:
                    f["focus_score"] = pd.to_numeric(f["focus_score"], errors="coerce").fillna(0.0)
                else:
                    f["focus_score"] = 0.0

                g = f.groupby("product_category", dropna=False)
                focus_stats = g.agg(
                    focus_score_sum=("focus_score", "sum"),
                    focus_score_mean=("focus_score", "mean"),
                    focus_asin_count=("asin", "count"),
                ).reset_index()

                # 关键风险计数（按类目），用于快速抓重点
                if "oos_with_ad_spend_days" in f.columns:
                    tmp = f.copy()
                    tmp["_flag"] = (pd.to_numeric(tmp.get("oos_with_ad_spend_days", 0.0), errors="coerce").fillna(0.0) > 0).astype(int)
                    tmp2 = tmp.groupby("product_category", dropna=False, as_index=False).agg(oos_with_ad_spend_asin_count=("_flag", "sum"))
                    focus_stats = focus_stats.merge(tmp2, on="product_category", how="left")
                else:
                    focus_stats["oos_with_ad_spend_asin_count"] = 0

                if "delta_spend" in f.columns and "delta_sales" in f.columns:
                    tmp = f.copy()
                    tmp["_flag"] = (
                        (pd.to_numeric(tmp.get("delta_spend", 0.0), errors="coerce").fillna(0.0) > 0)
                        & (pd.to_numeric(tmp.get("delta_sales", 0.0), errors="coerce").fillna(0.0) <= 0)
                    ).astype(int)
                    tmp2 = tmp.groupby("product_category", dropna=False, as_index=False).agg(spend_up_no_sales_asin_count=("_flag", "sum"))
                    focus_stats = focus_stats.merge(tmp2, on="product_category", how="left")
                else:
                    focus_stats["spend_up_no_sales_asin_count"] = 0

                if "ad_sales_share" in f.columns:
                    tmp = f.copy()
                    tmp["_flag"] = (pd.to_numeric(tmp.get("ad_sales_share", 0.0), errors="coerce").fillna(0.0) >= 0.8).astype(int)
                    tmp2 = tmp.groupby("product_category", dropna=False, as_index=False).agg(high_ad_dependency_asin_count=("_flag", "sum"))
                    focus_stats = focus_stats.merge(tmp2, on="product_category", how="left")
                else:
                    focus_stats["high_ad_dependency_asin_count"] = 0

                # 合并进 category_summary
                if category_summary is not None and not category_summary.empty and "product_category" in category_summary.columns:
                    category_summary = category_summary.merge(focus_stats, on="product_category", how="left")
                    for c in (
                        "focus_score_sum",
                        "focus_score_mean",
                        "focus_asin_count",
                        "oos_with_ad_spend_asin_count",
                        "spend_up_no_sales_asin_count",
                        "high_ad_dependency_asin_count",
                    ):
                        if c in category_summary.columns:
                            category_summary[c] = pd.to_numeric(category_summary[c], errors="coerce").fillna(0.0)
                    # 默认排序：先看 focus_score_sum（更贴近“抓重点”），再看广告花费
                    if "focus_score_sum" in category_summary.columns and "ad_spend_total" in category_summary.columns:
                        category_summary = category_summary.sort_values(
                            ["focus_score_sum", "ad_spend_total"],
                            ascending=[False, False],
                        ).copy()
        except Exception:
            pass
        category_summary_path = dashboard_dir / "category_summary.csv"
        if category_summary is not None and not category_summary.empty:
            category_summary.to_csv(category_summary_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["product_category"]).to_csv(category_summary_path, index=False, encoding="utf-8-sig")

        # 4.5) category_cockpit.csv（类目总览：汇总 focus/drivers/动作量）
        category_cockpit = None
        try:
            category_cockpit = build_category_cockpit(
                category_summary=category_summary,
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                action_board_dedup_all=action_board_all,
            )
        except Exception:
            category_cockpit = pd.DataFrame()
        category_cockpit_path = dashboard_dir / "category_cockpit.csv"
        if category_cockpit is not None and not category_cockpit.empty:
            category_cockpit.to_csv(category_cockpit_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["product_category"]).to_csv(category_cockpit_path, index=False, encoding="utf-8-sig")

        # 4.55) category_asin_compare.csv（类目→产品对比：同类产品横向对比）
        category_asin_compare = None
        try:
            category_asin_compare = build_category_asin_compare(
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                category_cockpit=category_cockpit if isinstance(category_cockpit, pd.DataFrame) else None,
                max_categories=50,
                asins_per_category=30,
            )
        except Exception:
            category_asin_compare = pd.DataFrame()
        category_asin_compare_path = dashboard_dir / "category_asin_compare.csv"
        if category_asin_compare is not None and not category_asin_compare.empty:
            category_asin_compare.to_csv(category_asin_compare_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["product_category", "asin"]).to_csv(category_asin_compare_path, index=False, encoding="utf-8-sig")

        # 4.6) phase_cockpit.csv（生命周期总览：按 phase 汇总 focus/变化/动作量）
        phase_cockpit = None
        try:
            phase_cockpit = build_phase_cockpit(
                asin_focus_all=asin_focus_all,
                action_board_dedup_all=action_board_all,
                policy=policy,
                inventory_risk_spend_threshold=10.0,
            )
        except Exception:
            phase_cockpit = pd.DataFrame()
        phase_cockpit_path = dashboard_dir / "phase_cockpit.csv"
        if phase_cockpit is not None and not phase_cockpit.empty:
            phase_cockpit.to_csv(phase_cockpit_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["current_phase"]).to_csv(phase_cockpit_path, index=False, encoding="utf-8-sig")

        # 4.7) keyword_topics.csv（搜索词主题 n-gram）
        keyword_topics = None
        try:
            enabled = True
            n_values = None
            min_term_spend = 0.0
            max_terms = 5000
            max_rows = 2000
            top_terms_per_ngram = 3
            ktp = None
            if isinstance(policy, OpsPolicy):
                ktp = getattr(policy, "dashboard_keyword_topics", None)
                if ktp is not None:
                    enabled = bool(getattr(ktp, "enabled", enabled))
                    n_values = getattr(ktp, "n_values", n_values)
                    min_term_spend = float(getattr(ktp, "min_term_spend", min_term_spend) or min_term_spend)
                    max_terms = int(getattr(ktp, "max_terms", max_terms) or max_terms)
                    max_rows = int(getattr(ktp, "max_rows", max_rows) or max_rows)
                    top_terms_per_ngram = int(getattr(ktp, "top_terms_per_ngram", top_terms_per_ngram) or top_terms_per_ngram)
            if not enabled:
                keyword_topics = pd.DataFrame()
            else:
                keyword_topics = build_keyword_topics(
                    search_term_report=search_term_report,
                    n_values=n_values,
                    min_term_spend=min_term_spend,
                    waste_min_clicks=int(getattr(get_stage_config(stage), "min_clicks", 0) or 0),
                    waste_min_spend=float(getattr(get_stage_config(stage), "waste_spend", 0.0) or 0.0),
                    max_terms=max_terms,
                    max_rows=max_rows,
                    top_terms_per_ngram=top_terms_per_ngram,
                )
        except Exception:
            keyword_topics = pd.DataFrame()
            ktp = getattr(policy, "dashboard_keyword_topics", None) if isinstance(policy, OpsPolicy) else None
        keyword_topics_path = dashboard_dir / "keyword_topics.csv"
        keyword_cols = [
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
        if isinstance(keyword_topics, pd.DataFrame) and not keyword_topics.empty:
            keyword_topics.to_csv(keyword_topics_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=keyword_cols).to_csv(keyword_topics_path, index=False, encoding="utf-8-sig")

        # 4.8) keyword_topics_action_hints.csv（主题建议：可分派清单）
        keyword_hints_cols = [
            "priority",
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
            "context_asin_count",
            "context_top_asins",
            "context_profit_directions",
            "context_min_inventory",
            "context_min_cover_days_7d",
            "blocked",
            "blocked_reason",
            "filter_contains",
            "next_step",
        ]
        keyword_topics_hints_path = dashboard_dir / "keyword_topics_action_hints.csv"
        try:
            hints_df = build_keyword_topic_action_hints(
                search_term_report=search_term_report,
                stage=stage,
                policy=ktp,
                topics=keyword_topics if isinstance(keyword_topics, pd.DataFrame) else None,
            )
        except Exception:
            hints_df = pd.DataFrame()

        # 4.9) keyword_topics_asin_context.csv（主题→产品语境：只用高置信 term→asin）
        keyword_asin_context_cols = [
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
        keyword_asin_context_path = dashboard_dir / "keyword_topics_asin_context.csv"
        try:
            asin_ctx = build_keyword_topic_asin_context(
                asin_top_search_terms=asin_top_search_terms,
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                topic_hints=hints_df if isinstance(hints_df, pd.DataFrame) else None,
                stage=stage,
                policy=ktp,
            )
        except Exception:
            asin_ctx = pd.DataFrame()

        # ===== 用 ASIN 语境对主题建议做“放量阻断/标注”（只影响 hints 输出，不影响 topic 的选取逻辑）=====
        hints_out = hints_df.copy() if isinstance(hints_df, pd.DataFrame) else pd.DataFrame()
        try:
            low_inv_th = int(getattr(policy, "low_inventory_threshold", 20) or 20) if isinstance(policy, OpsPolicy) else 20
            block_low_inv = bool(getattr(policy, "block_scale_when_low_inventory", True)) if isinstance(policy, OpsPolicy) else True
            cover_days_th = float(getattr(policy, "block_scale_when_cover_days_below", 7.0) or 7.0) if isinstance(policy, OpsPolicy) else 7.0
        except Exception:
            low_inv_th = 20
            block_low_inv = True
            cover_days_th = 7.0
        try:
            hints_out = annotate_keyword_topic_action_hints(
                topic_hints=hints_out,
                asin_context=asin_ctx if isinstance(asin_ctx, pd.DataFrame) else None,
                low_inventory_threshold=low_inv_th,
                block_scale_when_low_inventory=block_low_inv,
                block_scale_when_cover_days_below=cover_days_th,
            )
        except Exception:
            hints_out = hints_df.copy() if isinstance(hints_df, pd.DataFrame) else pd.DataFrame()

        # 写出 keyword_topics_action_hints.csv（无数据也输出表头）
        if isinstance(hints_out, pd.DataFrame) and not hints_out.empty:
            # 只保留稳定列顺序（方便 Excel 透视/筛选）
            cols2 = [c for c in keyword_hints_cols if c in hints_out.columns]
            hints_out[cols2].to_csv(keyword_topics_hints_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=keyword_hints_cols).to_csv(keyword_topics_hints_path, index=False, encoding="utf-8-sig")
        if isinstance(asin_ctx, pd.DataFrame) and not asin_ctx.empty:
            asin_ctx.to_csv(keyword_asin_context_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=keyword_asin_context_cols).to_csv(keyword_asin_context_path, index=False, encoding="utf-8-sig")

        # 4.10) keyword_topics_category_phase_summary.csv（主题→类目/生命周期汇总）
        keyword_cat_phase_cols = [
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
        keyword_cat_phase_path = dashboard_dir / "keyword_topics_category_phase_summary.csv"
        try:
            cat_phase = build_keyword_topic_category_phase_summary(
                asin_top_search_terms=asin_top_search_terms,
                asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                topic_hints=hints_out if isinstance(hints_out, pd.DataFrame) else None,
                stage=stage,
                policy=ktp,
            )
        except Exception:
            cat_phase = pd.DataFrame()
        if isinstance(cat_phase, pd.DataFrame) and not cat_phase.empty:
            cat_phase.to_csv(keyword_cat_phase_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=keyword_cat_phase_cols).to_csv(keyword_cat_phase_path, index=False, encoding="utf-8-sig")

        # 4.11) keyword_topics_segment_top.csv（类目×生命周期 → Top 主题概览）
        keyword_segment_top_cols = [
            "product_category",
            "current_phase",
            "reduce_topic_count",
            "reduce_waste_spend_sum",
            "reduce_top_topics",
            "scale_topic_count",
            "scale_sales_sum",
            "scale_top_topics",
        ]
        keyword_segment_top_path = dashboard_dir / "keyword_topics_segment_top.csv"
        try:
            seg_top = build_keyword_topic_segment_top(
                category_phase_summary=cat_phase if isinstance(cat_phase, pd.DataFrame) else None,
                policy=ktp,
            )
        except Exception:
            seg_top = pd.DataFrame()
        if isinstance(seg_top, pd.DataFrame) and not seg_top.empty:
            seg_top.to_csv(keyword_segment_top_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=keyword_segment_top_cols).to_csv(keyword_segment_top_path, index=False, encoding="utf-8-sig")

        # 4.99) shop_scorecard.json：补齐“抓重点”计数（动作数 + Watchlists 数）
        # - 不影响任何算数口径，只用于入口汇总与快速扫重点
        try:
            sc2 = sc_json.copy() if isinstance(sc_json, dict) else {}
            sc_score = sc2.get("scorecard") if isinstance(sc2.get("scorecard"), dict) else {}
            sc_score2 = sc_score.copy() if isinstance(sc_score, dict) else {}
            sc_score2["actions"] = build_actions_summary(action_board if isinstance(action_board, pd.DataFrame) else None)
            sc_score2["watchlists"] = build_watchlists_summary(
                profit_reduce_watchlist=profit_reduce_watchlist if isinstance(profit_reduce_watchlist, pd.DataFrame) else None,
                inventory_risk_watchlist=inventory_risk_watchlist if isinstance(inventory_risk_watchlist, pd.DataFrame) else None,
                inventory_sigmoid_watchlist=inventory_sigmoid_watchlist if isinstance(inventory_sigmoid_watchlist, pd.DataFrame) else None,
                profit_guard_watchlist=profit_guard_watchlist if isinstance(profit_guard_watchlist, pd.DataFrame) else None,
                oos_with_ad_spend_watchlist=oos_watchlist if isinstance(oos_watchlist, pd.DataFrame) else None,
                spend_up_no_sales_watchlist=spend_up_no_sales_watchlist if isinstance(spend_up_no_sales_watchlist, pd.DataFrame) else None,
                phase_down_recent_watchlist=phase_down_recent_watchlist if isinstance(phase_down_recent_watchlist, pd.DataFrame) else None,
                scale_opportunity_watchlist=scale_opportunity_watchlist if isinstance(scale_opportunity_watchlist, pd.DataFrame) else None,
            )
            sc2["scorecard"] = sc_score2
            if isinstance(scorecard_path, Path):
                scorecard_path.write_text(json_dumps(sc2), encoding="utf-8")
        except Exception:
            pass

        # 5) reports/dashboard.md（严格聚焦）
        if render_md:
            try:
                dash_md_path = shop_dir / "reports" / "dashboard.md"
                # drilldown：用于从 dashboard 的 ASIN 链接跳转
                try:
                    drilldown_path = shop_dir / "reports" / "asin_drilldown.md"
                    write_asin_drilldown_md(
                        out_path=drilldown_path,
                        shop=shop,
                        stage=stage,
                        date_start=date_start,
                        date_end=date_end,
                        asin_focus_all=asin_focus_all,
                        drivers_top_asins=drivers_df if isinstance(drivers_df, pd.DataFrame) else None,
                        action_board_full=action_board_full,
                        max_asins=30,
                    )
                except Exception:
                    pass
                # category drilldown：用于从 dashboard 的类目跳转
                try:
                    cat_path = shop_dir / "reports" / "category_drilldown.md"
                    write_category_drilldown_md(
                        out_path=cat_path,
                        shop=shop,
                        stage=stage,
                        date_start=date_start,
                        date_end=date_end,
                        category_cockpit=category_cockpit if isinstance(category_cockpit, pd.DataFrame) else None,
                        asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                        keyword_segment_top=seg_top if isinstance(seg_top, pd.DataFrame) else None,
                        max_categories=20,
                        asins_per_category=10,
                    )
                except Exception:
                    pass
                # phase drilldown：用于从 dashboard 的生命周期阶段跳转
                try:
                    ph_path = shop_dir / "reports" / "phase_drilldown.md"
                    write_phase_drilldown_md(
                        out_path=ph_path,
                        shop=shop,
                        stage=stage,
                        date_start=date_start,
                        date_end=date_end,
                        phase_cockpit=phase_cockpit if isinstance(phase_cockpit, pd.DataFrame) else None,
                        asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                        max_phases=20,
                        categories_per_phase=8,
                        asins_per_phase=12,
                    )
                except Exception:
                    pass
                # lifecycle overview：按「类目→ASIN」展示生命周期时间轴（更直观）
                try:
                    lc_path = shop_dir / "reports" / "lifecycle_overview.md"
                    write_lifecycle_overview_md(
                        out_path=lc_path,
                        shop=shop,
                        stage=stage,
                        date_start=date_start,
                        date_end=date_end,
                        lifecycle_segments=lifecycle_segments if isinstance(lifecycle_segments, pd.DataFrame) else None,
                        lifecycle_board=lifecycle_board if isinstance(lifecycle_board, pd.DataFrame) else None,
                        asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                        max_categories=30,
                        asins_per_category=60,
                        max_total_asins=800,
                    )
                except Exception:
                    pass
                # keyword_topics drilldown：用于从 dashboard 的关键词主题区块下钻
                try:
                    kw_path = shop_dir / "reports" / "keyword_topics.md"
                    write_keyword_topics_drilldown_md(
                        out_path=kw_path,
                        shop=shop,
                        stage=stage,
                        date_start=date_start,
                        date_end=date_end,
                        keyword_segment_top=seg_top if isinstance(seg_top, pd.DataFrame) else None,
                        keyword_action_hints=hints_out if isinstance(hints_out, pd.DataFrame) else None,
                        keyword_asin_context=asin_ctx if isinstance(asin_ctx, pd.DataFrame) else None,
                        policy=policy,
                        max_segments=12,
                    )
                except Exception:
                    pass
                write_dashboard_md(
                    out_path=dash_md_path,
                    shop=shop,
                    stage=stage,
                    date_start=date_start,
                    date_end=date_end,
                    scorecard=scorecard if isinstance(scorecard, dict) else {},
                    category_summary=category_summary,
                    category_cockpit=category_cockpit if isinstance(category_cockpit, pd.DataFrame) else None,
                    phase_cockpit=phase_cockpit if isinstance(phase_cockpit, pd.DataFrame) else None,
                    asin_focus=asin_focus,
                    action_board=action_board,
                    campaign_action_view=campaign_action_view if isinstance(campaign_action_view, pd.DataFrame) else None,
                    drivers_top_asins=drivers_df if isinstance(drivers_df, pd.DataFrame) else None,
                    keyword_topics=keyword_topics if isinstance(keyword_topics, pd.DataFrame) else None,
                    asin_cockpit=asin_cockpit if isinstance(asin_cockpit, pd.DataFrame) else None,
                    policy=policy,
                    budget_transfer_plan=budget_transfer_plan_effective,
                    unlock_scale_tasks=unlock_scale_tasks_table if isinstance(unlock_scale_tasks_table, pd.DataFrame) else None,
                    data_quality_hints=data_quality_hints,
                    action_review=action_review if isinstance(action_review, pd.DataFrame) else None,
                )

                # 6) reports/*.html（展示层：从 Markdown 自动转换，不改变口径）
                try:
                    reports_dir = shop_dir / "reports"
                    for name in (
                        "dashboard.md",
                        "asin_drilldown.md",
                        "category_drilldown.md",
                        "phase_drilldown.md",
                        "lifecycle_overview.md",
                        "keyword_topics.md",
                    ):
                        md_p = reports_dir / name
                        html_p = reports_dir / (Path(name).stem + ".html")
                        write_report_html_from_md(md_path=md_p, out_path=html_p)
                except Exception:
                    pass
            except Exception:
                dash_md_path = None
    except Exception:
        pass

    # fallback：确保 dashboard.md/html 产物存在（避免异常路径导致 reports 缺失）
    try:
        if render_md:
            reports_dir = shop_dir / "reports"
            dash_md = reports_dir / "dashboard.md"
            dash_html = reports_dir / "dashboard.html"
            if not dash_md.exists():
                reports_dir.mkdir(parents=True, exist_ok=True)

                def _read_csv(p: Path) -> pd.DataFrame:
                    try:
                        return pd.read_csv(p, encoding="utf-8-sig")
                    except Exception:
                        return pd.DataFrame()

                # 读回 dashboard 产物作为兜底输入
                sc_path = dashboard_dir / "shop_scorecard.json" if "dashboard_dir" in locals() else None
                scorecard = {}
                try:
                    if sc_path is not None and sc_path.exists():
                        sc_json = json.loads(sc_path.read_text(encoding="utf-8"))
                        if isinstance(sc_json, dict):
                            scorecard = sc_json.get("scorecard", {}) if isinstance(sc_json.get("scorecard"), dict) else {}
                except Exception:
                    scorecard = {}

                category_summary = _read_csv(dashboard_dir / "category_summary.csv") if "dashboard_dir" in locals() else pd.DataFrame()
                category_cockpit = _read_csv(dashboard_dir / "category_cockpit.csv") if "dashboard_dir" in locals() else pd.DataFrame()
                phase_cockpit = _read_csv(dashboard_dir / "phase_cockpit.csv") if "dashboard_dir" in locals() else pd.DataFrame()
                asin_focus = _read_csv(dashboard_dir / "asin_focus.csv") if "dashboard_dir" in locals() else pd.DataFrame()
                action_board = _read_csv(dashboard_dir / "action_board.csv") if "dashboard_dir" in locals() else pd.DataFrame()
                campaign_action_view = _read_csv(dashboard_dir / "campaign_action_view.csv") if "dashboard_dir" in locals() else pd.DataFrame()
                drivers_top_asins = _read_csv(dashboard_dir / "drivers_top_asins.csv") if "dashboard_dir" in locals() else pd.DataFrame()
                keyword_topics = _read_csv(dashboard_dir / "keyword_topics.csv") if "dashboard_dir" in locals() else pd.DataFrame()
                asin_cockpit = _read_csv(dashboard_dir / "asin_cockpit.csv") if "dashboard_dir" in locals() else pd.DataFrame()
                unlock_scale_tasks = _read_csv(dashboard_dir / "unlock_scale_tasks.csv") if "dashboard_dir" in locals() else pd.DataFrame()

                write_dashboard_md(
                    out_path=dash_md,
                    shop=shop,
                    stage=stage,
                    date_start=date_start,
                    date_end=date_end,
                    scorecard=scorecard,
                    category_summary=category_summary,
                    category_cockpit=category_cockpit if not category_cockpit.empty else None,
                    phase_cockpit=phase_cockpit if not phase_cockpit.empty else None,
                    asin_focus=asin_focus if isinstance(asin_focus, pd.DataFrame) else pd.DataFrame(),
                    action_board=action_board if isinstance(action_board, pd.DataFrame) else pd.DataFrame(),
                    campaign_action_view=campaign_action_view if not campaign_action_view.empty else None,
                    drivers_top_asins=drivers_top_asins if not drivers_top_asins.empty else None,
                    keyword_topics=keyword_topics if not keyword_topics.empty else None,
                    asin_cockpit=asin_cockpit if not asin_cockpit.empty else None,
                    policy=policy,
                    budget_transfer_plan={},
                    unlock_scale_tasks=unlock_scale_tasks if not unlock_scale_tasks.empty else None,
                    data_quality_hints=None,
                    action_review=None,
                )

            if dash_md.exists() and not dash_html.exists():
                write_report_html_from_md(md_path=dash_md, out_path=dash_html)
    except Exception:
        pass

    # SSOT（口径/字段）一致性审计底座：记录 dashboard/ 下各文件的列结构
    try:
        write_dashboard_schema_manifest(
            dashboard_dir=dashboard_dir,
            shop=shop,
            stage=stage,
            date_start=date_start,
            date_end=date_end,
        )
    except Exception:
        pass

    return (scorecard_path, asin_focus_path, action_board_path, category_summary_path, dash_md_path)
