# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src() -> None:
    try:
        repo_root = Path(__file__).resolve().parent
        src_dir = repo_root / "src"
        if src_dir.is_dir():
            s = str(src_dir)
            if s not in sys.path:
                sys.path.insert(0, s)
    except Exception:
        pass


def main() -> int:
    _bootstrap_src()
    # src/cli.py 里实现了完整参数解析
    from cli import main as real_main  # type: ignore

    return int(real_main())


if __name__ == "__main__":
    raise SystemExit(main())
