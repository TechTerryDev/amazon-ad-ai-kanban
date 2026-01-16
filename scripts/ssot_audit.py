# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _pick_latest_run(output_root: Path) -> Path | None:
    try:
        runs = [p for p in output_root.iterdir() if p.is_dir()]
        runs = sorted(runs, key=lambda p: p.name)
        return runs[-1] if runs else None
    except Exception:
        return None


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_ssot_spec_from_md(path: Path) -> dict:
    """
    从 ssot.md 中提取 ```json ... ``` code block。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}

    # 这里不要用“匹配大括号”的方式（JSON 内部有嵌套对象/数组，容易只截到第一层）
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return {}
    try:
        return json.loads(str(m.group(1) or "").strip())
    except Exception:
        return {}


def _find_shop_manifests(run_dir: Path, only_shop: str | None) -> list[Path]:
    manifests: list[Path] = []
    try:
        for p in run_dir.iterdir():
            if not p.is_dir():
                continue
            if only_shop and p.name != only_shop:
                continue
            m = p / "dashboard" / "schema_manifest.json"
            if m.exists():
                manifests.append(m)
    except Exception:
        manifests = []
    return sorted(manifests, key=lambda p: p.as_posix())


def _index_manifest(manifest: dict) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """
    返回：
    - csv_columns: filename -> set(columns)
    - json_keys: filename -> set(keys)
    """
    csv_columns: dict[str, set[str]] = {}
    json_keys: dict[str, set[str]] = {}
    items = manifest.get("items")
    if not isinstance(items, list):
        return csv_columns, json_keys
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("path") or "").strip()
        typ = str(it.get("type") or "").strip().lower()
        if not name:
            continue
        if typ == "csv":
            cols = it.get("columns")
            if isinstance(cols, list):
                csv_columns[name] = {str(c or "") for c in cols}
        elif typ == "json":
            keys = it.get("keys")
            if isinstance(keys, list):
                json_keys[name] = {str(k or "") for k in keys}
    return csv_columns, json_keys


def audit_one_shop(manifest_path: Path, ssot_spec: dict) -> tuple[int, str]:
    manifest_obj = _load_json(manifest_path)
    manifest = manifest_obj if isinstance(manifest_obj, dict) else {}
    shop = str(manifest.get("shop") or manifest_path.parents[2].name)

    artifacts = ssot_spec.get("artifacts") if isinstance(ssot_spec.get("artifacts"), dict) else {}
    if not isinstance(artifacts, dict) or not artifacts:
        return 2, f"[X] SSOT 规范为空：请检查 helloagents/wiki/ssot.md 的 JSON code block"

    csv_cols, json_keys = _index_manifest(manifest)
    fail = 0
    lines: list[str] = []
    lines.append(f"== {shop} ==")
    lines.append(f"- manifest: {manifest_path}")

    for key, spec in artifacts.items():
        if not isinstance(spec, dict):
            continue
        name = Path(str(key)).name
        typ = str(spec.get("type") or "").strip().lower()
        if typ == "csv":
            required = spec.get("required_columns") if isinstance(spec.get("required_columns"), list) else []
            required_set = {str(x or "") for x in required}
            actual = csv_cols.get(name, set())
            missing = sorted([x for x in required_set if x not in actual])
            if missing:
                fail += 1
                lines.append(f"[X] {name}: 缺失 {len(missing)} 列 -> {', '.join(missing[:30])}")
            else:
                lines.append(f"[OK] {name}: 关键列齐全 ({len(required_set)})")
        elif typ == "json":
            required = spec.get("required_keys") if isinstance(spec.get("required_keys"), list) else []
            required_set = {str(x or "") for x in required}
            actual = json_keys.get(name, set())
            missing = sorted([x for x in required_set if x not in actual])
            if missing:
                fail += 1
                lines.append(f"[X] {name}: 缺失 {len(missing)} keys -> {', '.join(missing[:30])}")
            else:
                lines.append(f"[OK] {name}: 关键 keys 齐全 ({len(required_set)})")

    return (1 if fail else 0), "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="", help="output/<run>/ 目录路径（默认自动选择最新）")
    parser.add_argument("--shop", type=str, default="", help="只审计单店铺")
    parser.add_argument("--ssot-md", type=str, default="helloagents/wiki/ssot.md", help="SSOT 规范文档路径")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
    if run_dir is None:
        run_dir = _pick_latest_run(Path("output"))
    if run_dir is None or (not run_dir.exists()):
        print("[X] 未找到 run 目录：请使用 --run-dir 指定 output/<run>/")
        return 2

    ssot_md = Path(args.ssot_md).expanduser()
    if not ssot_md.exists():
        print(f"[X] 未找到 SSOT 文档：{ssot_md}")
        return 2

    ssot_spec = _extract_ssot_spec_from_md(ssot_md)
    manifests = _find_shop_manifests(run_dir, only_shop=(args.shop.strip() or None))
    if not manifests:
        print("[X] 未找到 schema_manifest.json：请先运行 python main.py 生成输出")
        return 2

    any_fail = 0
    for m in manifests:
        code, out = audit_one_shop(m, ssot_spec)
        print(out)
        print("")
        any_fail = max(any_fail, code)

    return int(any_fail)


if __name__ == "__main__":
    raise SystemExit(main())
