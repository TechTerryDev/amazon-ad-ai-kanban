# -*- coding: utf-8 -*-

from __future__ import annotations

def main() -> int:
    # src/cli.py 里实现了完整参数解析
    from src.cli import main as real_main

    return int(real_main())


if __name__ == "__main__":
    raise SystemExit(main())
