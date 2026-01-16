# -*- coding: utf-8 -*-
"""
生命周期参数的读取与合并（JSON + CLI 覆盖）。

设计目标：
- 让“生命周期阈值”可配置、可版本化（JSON 文件可随仓库一起管理）
- CLI 可针对某次分析临时覆盖，避免频繁改代码
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from lifecycle.lifecycle import LifecycleConfig


def load_lifecycle_config(path: Path) -> LifecycleConfig:
    """
    从 JSON 文件读取 LifecycleConfig。

    - 读取失败时不抛出异常，返回默认 LifecycleConfig（避免主流程崩溃）
    """
    try:
        if path is None or not Path(path).exists():
            return LifecycleConfig()
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return LifecycleConfig()
        # 只接收 dataclass 里存在的字段
        base = asdict(LifecycleConfig())
        merged: Dict[str, Any] = {**base}
        for k, v in data.items():
            if k in base:
                merged[k] = v
        return LifecycleConfig(**merged)  # type: ignore[arg-type]
    except Exception:
        return LifecycleConfig()


def merge_lifecycle_overrides(cfg: LifecycleConfig, overrides: Dict[str, Optional[float]]) -> LifecycleConfig:
    """
    用 CLI 覆盖 LifecycleConfig（只覆盖传入且非 None 的值）。
    """
    try:
        base = asdict(cfg)
        for k, v in overrides.items():
            if v is None:
                continue
            if k in base:
                base[k] = v
        return LifecycleConfig(**base)  # type: ignore[arg-type]
    except Exception:
        return cfg
