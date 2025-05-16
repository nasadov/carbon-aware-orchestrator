#!/usr/bin/env python3
# filepath: /root/carbon-aware-orchestrator/analysis/visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
import os

# Set the style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['figure.titlesize'] = 18

# Parse command line arguments
import argparse

parser = argparse.ArgumentParser(description='Generate visualizations from performance log data')
parser.add_argument('--log-file', required=True, help='Path to the performance log file(s)', action='append')
args = parser.parse_args()

# Create base output directory
base_output_dir = "/root/carbon-aware-orchestrator/figures"
os.makedirs(base_output_dir, exist_ok=True)

# Process each provided log file
for data_path in args.log_file:
    # Extract experiment name from file name (e.g., "heuristic_20250512_081713.csv" → "heuristic_20250512_081713")
    experiment_name = os.path.basename(data_path).split('.')[0]
    
    # Create experiment-specific output directory
    output_dir = f"{base_output_dir}/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing: {data_path}")
    print(f"Saving outputs to: {output_dir}")
    
    # Load the data
    df = pd.read_csv(data_path)

# Extract pod counts from call_id and pods_processed
df['pod_count'] = df['pods_processed']
df['cumulative_pods'] = df['pods_total']

# For outlier detection and handling
def is_outlier(s, threshold=3):
    """Identify outliers using IQR method"""
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    return (s < lower_bound) | (s > upper_bound)

# Flag outliers in execution time
df['is_execution_outlier'] = is_outlier(df['execution_time_ms'])

# Figure 1: Execution Time vs. Pods Processed
plt.figure(figsize=(12, 7))
g = sns.scatterplot(
    data=df, 
    x='pods_processed', 
    y='execution_time_ms',
    hue='is_execution_outlier',
    palette={True: 'red', False: 'blue'},
    s=100,
    alpha=0.7
)

# Add regression line excluding outliers
sns.regplot(
    data=df[~df['is_execution_outlier']], 
    x='pods_processed',
    y='execution_time_ms',
    scatter=False, 
    color='darkblue',
    line_kws={'linewidth': 2}
)

# Log scale for y-axis to handle the wide range
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel("Number of Pods Processed per Run")
plt.ylabel("Execution Time (ms, log scale)")
plt.title("Carbon-Aware Orchestration: Algorithm Execution Time vs. Workload Size")

# Annotate outliers
for idx, row in df[df['is_execution_outlier']].iterrows():
    plt.annotate(
        f"Call {row['call_id']}: {row['pods_processed']} pods",
        xy=(row['pods_processed'], row['execution_time_ms']),
        xytext=(10, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
    )

plt.tight_layout()
plt.savefig(f"{output_dir}/execution_time_vs_pods.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/execution_time_vs_pods.png", dpi=300, bbox_inches='tight')

# Figure 2: Per-Pod Execution Time Analysis
plt.figure(figsize=(12, 7))
df['per_pod_time'] = df['execution_time_ms'] / df['pods_processed']
df['is_per_pod_outlier'] = is_outlier(df['per_pod_time'])

sns.scatterplot(
    data=df, 
    x='pods_processed', 
    y='per_pod_time',
    hue='is_per_pod_outlier',
    palette={True: 'red', False: 'blue'},
    s=100,
    alpha=0.7
)

# Add regression line excluding outliers
sns.regplot(
    data=df[~df['is_per_pod_outlier']], 
    x='pods_processed',
    y='per_pod_time',
    scatter=False, 
    color='darkblue',
    line_kws={'linewidth': 2}
)

plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel("Number of Pods Processed per Run")
plt.ylabel("Per-Pod Execution Time (ms/pod, log scale)")
plt.title("Carbon-Aware Orchestration: Per-Pod Processing Efficiency")

plt.tight_layout()
plt.savefig(f"{output_dir}/per_pod_execution_time.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/per_pod_execution_time.png", dpi=300, bbox_inches='tight')

# Figure 3: Carbon Emissions Analysis
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df, 
    x='pods_placed', 
    y='per_pod_emissions_kg',
    s=100,
    hue='call_id',
    palette='viridis',
    alpha=0.7
)

plt.xlabel("Number of Pods Successfully Placed")
plt.ylabel("Carbon Emissions per Pod (kg CO₂e)")
plt.title("Carbon Efficiency of Orchestration Decisions")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/carbon_efficiency.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/carbon_efficiency.png", dpi=300, bbox_inches='tight')

# Figure 4: Combined Analysis - Execution Time, Carbon, and Resource Utilization
fig, ax1 = plt.subplots(figsize=(14, 8))

color = 'tab:blue'
ax1.set_xlabel('Call ID (Sequential Runs)')
ax1.set_ylabel('Execution Time (ms)', color=color)
line1 = ax1.plot(df['call_id'], df['execution_time_ms'], color=color, marker='o', linestyle='-', linewidth=2, label='Execution Time')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Per-Pod Carbon Emissions (kg CO₂e)', color=color)
line2 = ax2.plot(df['call_id'], df['per_pod_emissions_kg'], color=color, marker='s', linestyle='--', linewidth=2, label='Carbon per Pod')
ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.1))
color = 'tab:green'
ax3.set_ylabel('CPU Utilization (%)', color=color)
line3 = ax3.plot(df['call_id'], df['cpu_utilization_pct'], color=color, marker='^', linestyle='-.', linewidth=2, label='CPU Utilization')
ax3.tick_params(axis='y', labelcolor=color)

# Combine legends
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.title('Multi-Dimensional Analysis of Carbon-Aware Orchestration Performance')
plt.grid(True, alpha=0.3)

# Annotate number of pods on each point
for i, row in df.iterrows():
    ax1.annotate(
        f"{row['pods_processed']}",
        xy=(row['call_id'], row['execution_time_ms']),
        xytext=(0, 10),
        textcoords="offset points",
        ha='center',
        fontsize=9
    )

plt.tight_layout()
plt.savefig(f"{output_dir}/multi_dimension_analysis.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/multi_dimension_analysis.png", dpi=300, bbox_inches='tight')

# Figure 5: Comparing Algorithm Scalability
plt.figure(figsize=(12, 7))

# Create theoretical curves for comparison
x = np.linspace(1, 10, 100)
linear = x
quadratic = x**2 / 10
cubic = x**3 / 100

# Add our actual data
df_clean = df[~df['is_execution_outlier']]
plt.scatter(df_clean['pods_processed'], df_clean['execution_time_ms'] / df_clean['execution_time_ms'].min(), 
           s=100, color='blue', label='Our Algorithm (Actual)', alpha=0.7)

# Add theoretical curves
plt.plot(x, linear, 'k--', label='O(n) - Linear', alpha=0.7)
plt.plot(x, quadratic, 'r--', label='O(n²) - Quadratic', alpha=0.7)
plt.plot(x, cubic, 'm--', label='O(n³) - Cubic', alpha=0.7)

plt.xlabel('Problem Size (Number of Pods)')
plt.ylabel('Normalized Execution Time')
plt.title('Algorithmic Scalability Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(f"{output_dir}/algorithm_scalability.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/algorithm_scalability.png", dpi=300, bbox_inches='tight')

# Figure 6: Resource Utilization vs Carbon Emissions
plt.figure(figsize=(12, 8))

# Create a grid of 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: CPU Utilization vs Carbon Emissions
sns.scatterplot(
    data=df, 
    x='cpu_utilization_pct', 
    y='total_emissions_kg', 
    size='pods_processed',
    sizes=(50, 300),
    hue='call_id',
    palette='viridis',
    ax=axes[0, 0]
)
axes[0, 0].set_title('CPU Utilization vs Total Carbon Emissions')
axes[0, 0].set_xlabel('CPU Utilization (%)')
axes[0, 0].set_ylabel('Total Carbon Emissions (kg CO₂e)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Memory Utilization vs Carbon Emissions
sns.scatterplot(
    data=df, 
    x='memory_utilization_pct', 
    y='total_emissions_kg', 
    size='pods_processed',
    sizes=(50, 300),
    hue='call_id',
    palette='viridis',
    ax=axes[0, 1]
)
axes[0, 1].set_title('Memory Utilization vs Total Carbon Emissions')
axes[0, 1].set_xlabel('Memory Utilization (%)')
axes[0, 1].set_ylabel('Total Carbon Emissions (kg CO₂e)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: CPU Utilization vs Per-Pod Carbon Emissions
sns.scatterplot(
    data=df, 
    x='cpu_utilization_pct', 
    y='per_pod_emissions_kg', 
    size='pods_processed',
    sizes=(50, 300),
    hue='call_id',
    palette='viridis',
    ax=axes[1, 0]
)
axes[1, 0].set_title('CPU Utilization vs Per-Pod Carbon Emissions')
axes[1, 0].set_xlabel('CPU Utilization (%)')
axes[1, 0].set_ylabel('Per-Pod Carbon Emissions (kg CO₂e)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Memory Utilization vs Per-Pod Carbon Emissions
sns.scatterplot(
    data=df, 
    x='memory_utilization_pct', 
    y='per_pod_emissions_kg', 
    size='pods_processed',
    sizes=(50, 300),
    hue='call_id',
    palette='viridis',
    ax=axes[1, 1]
)
axes[1, 1].set_title('Memory Utilization vs Per-Pod Carbon Emissions')
axes[1, 1].set_xlabel('Memory Utilization (%)')
axes[1, 1].set_ylabel('Per-Pod Carbon Emissions (kg CO₂e)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/resource_utilization_carbon_analysis.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/resource_utilization_carbon_analysis.png", dpi=300, bbox_inches='tight')

print(f"Visualization complete! All figures saved to '{output_dir}'")
