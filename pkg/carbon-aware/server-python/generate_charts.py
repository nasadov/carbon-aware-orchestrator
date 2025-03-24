#!/usr/bin/env python3
"""
Generate publication-quality comparison charts from experiment data.
"""
import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

def load_experiment_data(base_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all experiment data from the given directory"""
    # Find all experiment result directories
    result_dirs = glob.glob(os.path.join(base_dir, "*"))
    result_dirs = [d for d in result_dirs if os.path.isdir(d)]
    
    if not result_dirs:
        print(f"No experiment directories found in {base_dir}")
        return {}
    
    data = {}
    
    # Load CSV data from each directory
    for result_dir in result_dirs:
        algo_name = os.path.basename(result_dir).split('_')[0]  # Extract algorithm name
        csv_files = glob.glob(os.path.join(result_dir, "detailed_results_*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {result_dir}")
            continue
            
        # Load and concatenate all CSV files for this algorithm
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        
        if dfs:
            data[algo_name] = pd.concat(dfs, ignore_index=True)
            print(f"Loaded {len(data[algo_name])} records for {algo_name}")
    
    return data

def generate_academic_charts(data: Dict[str, pd.DataFrame], output_dir: str):
    """Generate publication-quality charts for algorithm comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style for all plots
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 12,
        'figure.figsize': (10, 6),
        'figure.dpi': 300
    })
    
    # 1. Execution Time Analysis
    plt.figure(figsize=(10, 6))
    exec_data = []
    
    for algo, df in data.items():
        exec_data.append({
            'Algorithm': algo,
            'Mean Time (s)': df['execution_time'].mean(),
            'Std Dev': df['execution_time'].std()
        })
    
    exec_df = pd.DataFrame(exec_data)
    sns.barplot(x='Algorithm', y='Mean Time (s)', data=exec_df, palette='viridis')
    plt.errorbar(x=range(len(exec_df)), y=exec_df['Mean Time (s)'], 
                 yerr=exec_df['Std Dev'], fmt='none', color='black', capsize=5)
    plt.title('Algorithm Execution Time Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_comparison.pdf'))
    plt.savefig(os.path.join(output_dir, 'execution_time_comparison.png'), dpi=300)
    
    # 2. Success Rate Analysis
    plt.figure(figsize=(10, 6))
    success_data = []
    
    for algo, df in data.items():
        total = len(df)
        successful = df['success'].sum()
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        success_data.append({
            'Algorithm': algo,
            'Success Rate (%)': success_rate,
            'Successful': successful,
            'Total': total
        })
    
    success_df = pd.DataFrame(success_data)
    ax = sns.barplot(x='Algorithm', y='Success Rate (%)', data=success_df, palette='viridis')
    
    # Add value labels on bars
    for i, row in enumerate(success_data):
        ax.text(i, row['Success Rate (%)']/2, 
                f"{row['Successful']}/{row['Total']}", 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.title('Placement Success Rate Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_comparison.pdf'))
    plt.savefig(os.path.join(output_dir, 'success_rate_comparison.png'), dpi=300)
    
    # 3. Carbon Emissions Analysis
    plt.figure(figsize=(10, 6))
    carbon_data = []
    
    for algo, df in data.items():
        successful_df = df[df['success'] == True]
        if not successful_df.empty:
            carbon_data.append({
                'Algorithm': algo,
                'Mean Emissions (kgCO2e)': successful_df['emissions'].mean(),
                'Total Emissions (kgCO2e)': successful_df['emissions'].sum(),
                'Std Dev': successful_df['emissions'].std()
            })
    
    if carbon_data:
        carbon_df = pd.DataFrame(carbon_data)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Mean emissions plot
        sns.barplot(x='Algorithm', y='Mean Emissions (kgCO2e)', data=carbon_df, ax=ax1, palette='viridis')
        ax1.errorbar(x=range(len(carbon_df)), y=carbon_df['Mean Emissions (kgCO2e)'], 
                    yerr=carbon_df['Std Dev'], fmt='none', color='black', capsize=5)
        ax1.set_title('Average Carbon Emissions per Pod', fontsize=14)
        
        # Total emissions plot
        sns.barplot(x='Algorithm', y='Total Emissions (kgCO2e)', data=carbon_df, ax=ax2, palette='viridis')
        ax2.set_title('Total Carbon Emissions', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'carbon_emissions_comparison.pdf'))
        plt.savefig(os.path.join(output_dir, 'carbon_emissions_comparison.png'), dpi=300)
    
    # 4. Resource Utilization Analysis (if data available)
    for algo, df in data.items():
        if 'node' in df.columns and df['node'].notna().any():
            plt.figure(figsize=(10, 6))
            node_usage = df['node'].value_counts().sort_index()
            node_usage.plot(kind='bar')
            plt.title(f'Node Selection Distribution - {algo}', fontsize=14)
            plt.xlabel('Node')
            plt.ylabel('Number of Placements')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'node_distribution_{algo}.pdf'))
            plt.savefig(os.path.join(output_dir, f'node_distribution_{algo}.png'), dpi=300)
    
    # 5. Combined Metrics Table
    table_data = []
    
    for algo, df in data.items():
        successful_df = df[df['success'] == True]
        total = len(df)
        successful = len(successful_df)
        
        row = {
            'Algorithm': algo,
            'Total Pods': total,
            'Success Rate (%)': (successful / total) * 100 if total > 0 else 0,
            'Mean Execution Time (s)': df['execution_time'].mean(),
            'Max Execution Time (s)': df['execution_time'].max()
        }
        
        if not successful_df.empty:
            row.update({
                'Mean Carbon (kgCO2e)': successful_df['emissions'].mean(),
                'Total Carbon (kgCO2e)': successful_df['emissions'].sum()
            })
            
        table_data.append(row)
    
    if table_data:
        metrics_df = pd.DataFrame(table_data)
        metrics_df.to_csv(os.path.join(output_dir, 'algorithm_metrics_summary.csv'), index=False)
    
    print(f"All charts saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate academic comparison charts from experiment data')
    parser.add_argument('--data-dir', default='./experiment_results',
                        help='Directory containing experiment result folders')
    parser.add_argument('--output-dir', default='./comparison_charts',
                        help='Directory to save comparison charts')
    
    args = parser.parse_args()
    
    # Load data and generate charts
    data = load_experiment_data(args.data_dir)
    if data:
        generate_academic_charts(data, args.output_dir)
    else:
        print("No data found to analyze")