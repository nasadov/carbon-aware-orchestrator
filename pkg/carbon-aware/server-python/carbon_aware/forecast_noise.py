"""
Utilities for injecting controlled noise into carbon-intensity forecasts.

This module lets experiments perturb the Electricity Maps traces used by the
scheduler so we can test robustness to forecast error (e.g., 5–20% MAPE).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_FORECAST_PATH = str(Path(__file__).resolve().parents[1] / "all_forecasts.json")


def load_forecasts(path: str = DEFAULT_FORECAST_PATH) -> Dict[str, dict]:
    """Load the full Electricity Maps forecast JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_forecasts(data: Dict[str, dict], path: str) -> None:
    """Persist forecast data as JSON (pretty-printed for diffs)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def compute_mape(ground_truth: List[float], estimates: List[float]) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE) in percent.

    MAPE = (1/n) * Σ |(y_i - ẏ_i) / y_i| * 100
    """
    if len(ground_truth) != len(estimates):
        raise ValueError("ground_truth and estimates must have equal length")
    total = 0.0
    count = 0
    for actual, predicted in zip(ground_truth, estimates):
        if actual == 0:
            continue
        total += abs(actual - predicted) / actual
        count += 1
    if count == 0:
        return 0.0
    return (total / count) * 100.0


@dataclass
class NoiseResult:
    noisy_data: Dict[str, dict]
    achieved_mape: float
    scale: float


def generate_noisy_forecasts(
    base_data: Dict[str, dict],
    target_mape: float,
    seed: int,
    min_intensity: float = 1.0,
    tolerance_pct: float = 0.25,
    max_iters: int = 10,
) -> NoiseResult:
    """
    Create a noisy copy of the forecasts that approximately matches target MAPE.

    Noise model: multiplicative gaussian (Normal(0, scale)) applied per time-step.
    We pre-sample standard normal deviates with the provided seed so each scale
    update is deterministic, then adjust the scale via multiplicative updates
    until the achieved MAPE is within `tolerance_pct` percentage points.
    """
    if target_mape < 0:
        raise ValueError("target_mape must be non-negative")
    clone = copy.deepcopy(base_data)
    if target_mape == 0:
        # No noise requested
        flattened = [entry["carbonIntensity"] for region in clone.values() for entry in region.get("forecast", [])]
        return NoiseResult(clone, achieved_mape=0.0, scale=0.0)

    entries: List[Tuple[str, int, float]] = []
    base_values: List[float] = []
    for region, payload in base_data.items():
        for idx, entry in enumerate(payload.get("forecast", [])):
            value = float(entry["carbonIntensity"])
            entries.append((region, idx, value))
            base_values.append(value)

    if not entries:
        raise ValueError("No forecast entries found; cannot apply noise")

    rng = random.Random(seed)
    z_values = [rng.gauss(0, 1) for _ in entries]

    # Start with multiplicative scale roughly equal to target MAPE (as decimal).
    scale = target_mape / 100.0
    achieved = None

    for _ in range(max_iters):
        noisy_values: List[float] = []
        for (region, idx, baseline), z in zip(entries, z_values):
            noisy = baseline * (1.0 + z * scale)
            if noisy < min_intensity:
                noisy = min_intensity
            clone[region]["forecast"][idx]["carbonIntensity"] = noisy
            noisy_values.append(noisy)
        achieved = compute_mape(base_values, noisy_values)
        if math.isclose(achieved, target_mape, abs_tol=tolerance_pct):
            break
        if achieved == 0:
            scale *= 1.5
            continue
        scale *= target_mape / achieved

    return NoiseResult(clone, achieved_mape=achieved or 0.0, scale=scale)


def generate_and_save(
    target_mape: float,
    seed: int,
    output_path: str,
    base_path: str = DEFAULT_FORECAST_PATH,
    min_intensity: float = 1.0,
    tolerance_pct: float = 0.25,
    max_iters: int = 10,
) -> NoiseResult:
    """High-level helper: load -> perturb -> save -> return metadata."""
    base = load_forecasts(base_path)
    result = generate_noisy_forecasts(
        base,
        target_mape=target_mape,
        seed=seed,
        min_intensity=min_intensity,
        tolerance_pct=tolerance_pct,
        max_iters=max_iters,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_forecasts(result.noisy_data, output_path)
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate noisy carbon forecasts at a desired MAPE level.")
    parser.add_argument("--target-mape", type=float, required=True, help="Desired MAPE percentage (e.g., 10 for 10%%).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible noise.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the noisy forecast JSON.")
    parser.add_argument(
        "--base",
        type=str,
        default=DEFAULT_FORECAST_PATH,
        help=f"Ground-truth forecast file (default: {DEFAULT_FORECAST_PATH}).",
    )
    parser.add_argument("--min-intensity", type=float, default=1.0, help="Minimum gCO2/kWh after noise (default: 1).")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="Acceptable absolute error (percentage points) between achieved and target MAPE.",
    )
    parser.add_argument("--max-iters", type=int, default=10, help="Max iterations for scale adjustment.")
    return parser.parse_args()


def main():
    args = _parse_args()
    result = generate_and_save(
        target_mape=args.target_mape,
        seed=args.seed,
        output_path=args.output,
        base_path=args.base,
        min_intensity=args.min_intensity,
        tolerance_pct=args.tolerance,
        max_iters=args.max_iters,
    )
    print(
        f"✅ Generated noisy forecasts at {result.achieved_mape:.2f}% MAPE (target {args.target_mape}%) "
        f"using scale={result.scale:.4f} -> {args.output}"
    )


if __name__ == "__main__":
    main()
