# -*- coding: utf-8 -*-
"""
检查 output/<run>/ 下 HTML 的本地内链是否断链。

用途：
- 解决“点进去回不来/点到不存在文件”的问题
- 在你改了输出逻辑或目录结构后，快速自检

用法示例：
  python scripts/check_output_links.py --run-dir output/20260113_174937
  python scripts/check_output_links.py  # 自动选择 output/ 下最新一次 run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


def _pick_latest_run(output_root: Path) -> Path | None:
    try:
        runs = [p for p in output_root.iterdir() if p.is_dir()]
        runs = sorted(runs, key=lambda p: p.name)
        return runs[-1] if runs else None
    except Exception:
        return None


def _extract_hrefs(html_text: str) -> list[str]:
    # 极简提取：够用即可（我们只做离线链接自检，不需要完整 HTML parser）
    try:
        if not html_text:
            return []
        # 支持 href="..." / href='...'
        return re.findall(r"""href\s*=\s*["']([^"']+)["']""", html_text, flags=re.IGNORECASE)
    except Exception:
        return []


def _is_external_href(href: str) -> bool:
    try:
        p = urlsplit(href)
        if p.scheme in {"http", "https", "mailto", "tel", "javascript", "data"}:
            return True
        if p.netloc:
            return True
        return False
    except Exception:
        return False


def check_run(run_dir: Path) -> int:
    broken: list[tuple[str, str]] = []
    try:
        html_files = sorted(run_dir.rglob("*.html"))
    except Exception:
        html_files = []

    for html_path in html_files:
        try:
            text = html_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for href in _extract_hrefs(text):
            try:
                href_s = str(href or "").strip()
                if not href_s:
                    continue
                if href_s.startswith("#"):
                    continue
                if _is_external_href(href_s):
                    continue

                parts = urlsplit(href_s)
                rel_path = parts.path or ""
                if not rel_path:
                    continue
                # 只检查本地文件链接（不检查纯 fragment）
                target = (html_path.parent / rel_path).resolve()
                if not target.exists():
                    broken.append((str(html_path.relative_to(run_dir)), href_s))
            except Exception:
                continue

    if broken:
        print(f"[X] 发现断链: {len(broken)}")
        for src, href in broken[:200]:
            print(f"- {src} -> {href}")
        if len(broken) > 200:
            print(f"... 省略 {len(broken) - 200} 条")
        return 1

    print("[OK] 未发现断链")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="", help="output/<run>/ 目录路径")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
    if run_dir is None:
        run_dir = _pick_latest_run(Path("output"))
    if run_dir is None or (not run_dir.exists()):
        print("[X] 未找到 run 目录：请使用 --run-dir 指定 output/<run>/")
        return 2

    return int(check_run(run_dir))


if __name__ == "__main__":
    raise SystemExit(main())

