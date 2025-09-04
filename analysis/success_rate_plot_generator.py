#!/usr/bin/env python3
"""
Success Rate Plot Generator

Generates success rate comparison plots by parsing experiment results.

Usage:
    python analysis/success_rate_plot_generator.py
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import re
import csv
from collections import defaultdict

def _parse_success_rates_from_experiments(experiments_root: str):
    """Scan experiments_root for runs and parse success rates per algorithm and pod count.

    Returns:
        results: dict[str, dict[int, list[float]]] mapping algo -> pods -> list of success rates
    """
    results = defaultdict(lambda: defaultdict(list))

    if not os.path.isdir(experiments_root):
        print(f"⚠️ Experiments directory not found: {experiments_root}")
        return results

    # Expect folders like: "global-optimal_181pods_20250822_100133" or "heuristic_86pods_..."
    for entry in os.listdir(experiments_root):
        entry_path = os.path.join(experiments_root, entry)
        if not os.path.isdir(entry_path):
            continue

        # Extract algorithm and pod count from the directory name
        # Pattern: ^<algo>(?:_op)?_<pods>pods_
        m = re.match(r"^(?P<algo>[^_]+?)(?:_op)?_(?P<pods>\d+)pods_", entry)
        if not m:
            # Skip directories that don't encode pod count (e.g., experiment_session)
            continue

        algo_key = m.group('algo')
        try:
            pods_count = int(m.group('pods'))
        except Exception:
            continue

        summary_text = None
        summary_path = os.path.join(entry_path, "placement_summary.log")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    summary_text = f.read()
            except Exception:
                summary_text = None

        # Parse success from a line like: "📈 Placed: 156/181 pods (86.2%)"
        if summary_text:
            placed_match = re.search(r"Placed:\s*(\d+)\s*/\s*(\d+)\s*pods\s*\(([-\d\.]+)%\)", summary_text)
            if placed_match:
                placed = int(placed_match.group(1))
                total = int(placed_match.group(2))
                success_rate = 100.0 * placed / total if total > 0 else 0.0
                results[algo_key][pods_count].append(success_rate)
                continue

        # Alt format in placement_summary.log: "Success rate: 156/181 = 86.2%"
        if summary_text:
            sr_match = re.search(r"Success\s*rate:\s*(\d+)\s*/\s*(\d+)", summary_text, flags=re.IGNORECASE)
            if sr_match:
                placed = int(sr_match.group(1))
                total = int(sr_match.group(2))
                success_rate = 100.0 * placed / total if total > 0 else 0.0
                results[algo_key][pods_count].append(success_rate)
                continue

        # Fallback: compute from counts if present separately
        placed_num = None
        total_num = None
        m_placed = re.search(r"(Unique\s+pods\s+successfully\s+placed|Total\s+unique\s+pods\s+placed):\s*(\d+)", summary_text or "", flags=re.IGNORECASE)
        m_total = re.search(r"Total\s+pods\s+in\s+workloads?:\s*(\d+)", summary_text or "", flags=re.IGNORECASE)
        if m_placed and m_total:
            placed_num = int(m_placed.group(2))
            total_num = int(m_total.group(1))
        if placed_num is not None and total_num is not None and total_num > 0:
            results[algo_key][pods_count].append(100.0 * placed_num / total_num)
            continue

        # Last resort: if no summary present or parse failed, try placement CSV directly
        # Compute success rate as (unique pods placed) / (target pods from folder name)
        placement_csv = None
        candidate_names = []
        if algo_key == 'global-optimal':
            candidate_names = ['global_optimal_placements_session.csv']
        elif algo_key == 'heuristic':
            candidate_names = ['heuristic_placements_session.csv']
        elif algo_key == 'vanilla':
            # Prefer bind-based placement CSVs by default (our bind generator writes vanilla_placement_session.csv)
            candidate_names = [
                'vanilla_placement_session.csv',
                'vanilla_placement_session_bind.csv',
                'vanilla_placement_session_presence.csv',
                'vanilla_placement_session_fixed.csv',
                'vanilla_placements.csv',
            ]
        else:
            candidate_names = [
                'global_optimal_placements_session.csv',
                'heuristic_placements_session.csv',
                'vanilla_placement_session_fixed.csv',
                'vanilla_placement_session.csv',
                'vanilla_placements.csv',
            ]

        for name in candidate_names:
            p = os.path.join(entry_path, name)
            if os.path.exists(p):
                placement_csv = p
                break

        if placement_csv:
            try:
                with open(placement_csv, 'r') as fh:
                    reader = csv.DictReader(fh)
                    unique_pods = set()
                    for row in reader:
                        pid = row.get('pod_id')
                        if pid:
                            unique_pods.add(pid)
                placed = len(unique_pods)
                total = pods_count
                if total > 0:
                    results[algo_key][pods_count].append(100.0 * placed / total)
            except Exception:
                # If CSV parsing fails, skip
                pass

    return results


def _aggregate_results(results):
    """Aggregate list of success rates into mean and standard error per algo/pods.

    Returns:
        algos_order: list[str]
        pod_counts_sorted: list[int]
        mean_rates: dict[str, dict[int, float]]
        std_errs: dict[str, dict[int, float]]
    """
    mean_rates = defaultdict(dict)
    std_errs = defaultdict(dict)

    # Collect the union of pod counts across all algorithms
    pod_counts = set()
    for algo, by_pods in results.items():
        for pods in by_pods.keys():
            pod_counts.add(pods)
    pod_counts_sorted = sorted(pod_counts)

    # Stable order for known algos; unknown algos appended at end
    known_order = ["vanilla", "heuristic", "global-optimal"]
    algos_present = list(results.keys())
    algos_order = [a for a in known_order if a in algos_present] + [a for a in algos_present if a not in known_order]

    for algo, by_pods in results.items():
        for pods, rates in by_pods.items():
            if not rates:
                continue
            arr = np.array(rates, dtype=float)
            mean_rates[algo][pods] = float(np.mean(arr))
            if arr.size > 1:
                std_errs[algo][pods] = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
            else:
                std_errs[algo][pods] = 0.0

    return algos_order, pod_counts_sorted, mean_rates, std_errs


def create_success_rate_plot():
    """Generate success rate comparison plot by parsing experiment outputs."""
    # Parse dynamic results
    experiments_root = "/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python/experiments"
    parsed = _parse_success_rates_from_experiments(experiments_root)
    if not parsed:
        print(f"❌ No experiment results found in: {experiments_root}")
        return

    algos_order, pod_counts_sorted, mean_rates, std_errs = _aggregate_results(parsed)

    # Create publication-quality figure
    plt.figure(figsize=(12, 8))

    # Define colors and markers for consistency
    colors = {
        'vanilla': '#2ca02c',
        'heuristic': '#ff7f0e', 
        'global-optimal': '#d62728'
    }

    markers = {
        'vanilla': '^',
        'heuristic': 's',
        'global-optimal': 'o'
    }

    labels = {
        'vanilla': 'Vanilla (Baseline)',
        'heuristic': 'Heuristic (Carbon-Aware)',
        'global-optimal': 'Global-Optimal (MILP Oracle)'
    }

    # Plot each algorithm
    for algo in algos_order:
        algo_label = labels.get(algo, algo.replace('-', ' ').title())
        color = colors.get(algo, '#1f77b4')
        marker = markers.get(algo, 'o')

        x_vals = []
        y_vals = []
        y_errs = []
        for pods in pod_counts_sorted:
            if pods in mean_rates[algo]:
                x_vals.append(pods)
                y_vals.append(mean_rates[algo][pods])
                y_errs.append(std_errs[algo][pods])

        if not x_vals:
            continue

        if any(e > 0 for e in y_errs):
            plt.errorbar(
                x_vals, y_vals, yerr=y_errs,
                marker=marker, color=color, label=algo_label,
                linewidth=3, markersize=10, alpha=0.9, capsize=4
            )
        else:
            plt.plot(
                x_vals, y_vals,
                marker=marker, color=color, label=algo_label,
                linewidth=3, markersize=10, alpha=0.9
            )

    plt.xlabel('Number of Pods to Schedule', fontsize=14, fontweight='bold')
    plt.ylabel('Scheduling Success Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Scheduling Success Rate vs. Pod Count\nComparison of Three Algorithms', 
              fontsize=16, fontweight='bold', pad=20)

    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    if pod_counts_sorted:
        plt.xlim(min(pod_counts_sorted) - 5, max(pod_counts_sorted) + 10)
    plt.ylim(0, 105)
    plt.xticks(pod_counts_sorted, fontsize=12)
    plt.yticks(range(0, 101, 10), fontsize=12)
    plt.legend(fontsize=12, loc='lower left', framealpha=0.9, shadow=True, fancybox=True)
    plt.tight_layout()

    # Save PDF only
    output_dir = "/root/carbon-aware-orchestrator/figures/SuccessRate"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"{output_dir}/success_rate_comparison_{timestamp}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    print("✅ SUCCESS RATE PLOT GENERATED!")
    print(f"📊 PDF saved: {pdf_path}")
    
    # Display summary of experimental results
    print(f"\n📈 EXPERIMENTAL RESULTS SUMMARY:")
    for algo in algos_order:
        rates_list = [mean_rates[algo][p] for p in pod_counts_sorted if p in mean_rates[algo]]
        if not rates_list:
            continue
        print(f"   {algo}: avg={np.mean(rates_list):.1f}% range={min(rates_list):.1f}%-{max(rates_list):.1f}%")
    
    plt.show()

if __name__ == "__main__":
    print("🚀 SUCCESS RATE PLOT GENERATOR")
    print("=" * 50)
    create_success_rate_plot() 