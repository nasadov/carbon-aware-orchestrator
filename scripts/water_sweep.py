#!/usr/bin/env python3
"""Compatibility entry point for carbon-water experiment sweeps."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from water_sweep_impl import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
