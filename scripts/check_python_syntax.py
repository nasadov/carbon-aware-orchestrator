#!/usr/bin/env python3
"""Parse project Python files without importing them."""

from __future__ import annotations

import ast
from pathlib import Path

ROOTS = (
    Path("analysis"),
    Path("scripts"),
    Path("tests"),
    Path("pkg/carbon-aware/server-python"),
)


def main() -> int:
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print("Python syntax check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
