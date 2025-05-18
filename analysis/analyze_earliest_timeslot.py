#!/usr/bin/env python3

import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the placement data
csv_path = '/root/carbon-aware-orchestrator/pkg/carbon-aware/server-python/experiments/global-optimal_perf_log_session_20250517_144729/global_optimal_placements_session.csv'
placements_df = pd.read_csv(csv_path)

# Extract pod information
def extract_pod_file_timeslot(pod_id):
    """
    Extract the timeslot number from the file name pattern timeslot_X.yaml.
    
    Each pod comes from a specific timeslot_X.yaml file, and that X value is
    the earliest_timeslot for that pod. The mapping is as follows:
    
    - m000-m004: from timeslot_0.yaml (earliest_timeslot = 0)
    - m005-m009: from timeslot_1.yaml (earliest_timeslot = 1)
    - m010-m014: from timeslot_2.yaml (earliest_timeslot = 2)
    - m015-m018: from timeslot_3.yaml (earliest_timeslot = 3)
    - m019-m023: from timeslot_5.yaml (earliest_timeslot = 5)
    - m024-m025: from timeslot_6.yaml (earliest_timeslot = 6) 
    - m026-m030: from timeslot_7.yaml (earliest_timeslot = 7)
    - m031-m033: from timeslot_8.yaml (earliest_timeslot = 8)
    - m034-m038: from timeslot_9.yaml (earliest_timeslot = 9)
    - m039-m043: from timeslot_10.yaml (earliest_timeslot = 10)
    - m044-m046: from timeslot_11.yaml (earliest_timeslot = 11)
    
    This ensures we respect that pods can only be scheduled at or after their source file's timeslot number.
    """
    # Extract the pod number from the ID (e.g., 019 from m019-duration-3h-deadline-9h)
    match = re.match(r'([a-zA-Z]+)(\d+)[-_]?', pod_id)
    if match:
        pod_num = int(match.group(2))
        # Map pods to their source file's timeslot number
        if 0 <= pod_num <= 4:
            return 0  # timeslot_0.yaml
        elif 5 <= pod_num <= 9:
            return 1  # timeslot_1.yaml
        elif 10 <= pod_num <= 14:
            return 2  # timeslot_2.yaml
        elif 15 <= pod_num <= 18:
            return 3  # timeslot_3.yaml
        elif 19 <= pod_num <= 23:
            return 5  # timeslot_5.yaml
        elif 24 <= pod_num <= 25:
            return 6  # timeslot_6.yaml
        elif 26 <= pod_num <= 30:
            return 7  # timeslot_7.yaml
        elif 31 <= pod_num <= 33:
            return 8  # timeslot_8.yaml
        elif 34 <= pod_num <= 38:
            return 9  # timeslot_9.yaml
        elif 39 <= pod_num <= 43:
            return 10  # timeslot_10.yaml
        elif 44 <= pod_num <= 46:
            return 11  # timeslot_11.yaml
        else:
            # If we can't determine, ensure it's within the valid range (0-23)
            return min(pod_num, 23)
    return 0  # Default to timeslot 0 if we can't extract

# Add earliest_timeslot column based on pod ID mapping to file-based timeslots
placements_df['earliest_timeslot'] = placements_df['pod_id'].apply(extract_pod_file_timeslot)

# Check constraint compliance
placements_df['constraint_satisfied'] = placements_df['start_slot'] >= placements_df['earliest_timeslot']

# Summary statistics
total_pods = len(placements_df)
compliant_pods = placements_df['constraint_satisfied'].sum()
non_compliant_pods = total_pods - compliant_pods

print(f"Total placements: {total_pods}")
print(f"Earliest timeslot constraint satisfied: {compliant_pods} ({compliant_pods/total_pods*100:.2f}%)")
print(f"Earliest timeslot constraint violated: {non_compliant_pods} ({non_compliant_pods/total_pods*100:.2f}%)")

# Print details of non-compliant placements if any exist
if non_compliant_pods > 0:
    print("\nNon-compliant placements:")
    non_compliant = placements_df[~placements_df['constraint_satisfied']]
    print(non_compliant[['pod_id', 'node_id', 'earliest_timeslot', 'start_slot']])

# Plot the start_slot vs earliest_timeslot with a diagonal line showing the constraint boundary
plt.figure(figsize=(10, 8))
sns.scatterplot(data=placements_df, x='earliest_timeslot', y='start_slot', 
                hue='constraint_satisfied', s=100, alpha=0.7)

# Add diagonal line representing earliest_timeslot == start_slot (constraint boundary)
# We know that timeslots should be within 0-23 range
max_val = min(23, max(placements_df['earliest_timeslot'].max(), placements_df['start_slot'].max()))
plt.plot([0, max_val], [0, max_val], 'k--', label='Constraint Boundary')

# Points above the line satisfy the constraint
plt.fill_between([0, max_val], [0, max_val], [max_val, max_val], color='lightgreen', alpha=0.2, label='Valid Region')

plt.xlabel('Earliest Timeslot (from file timeslot_X.yaml)')
plt.ylabel('Actual Start Slot')
plt.title('Earliest Timeslot Constraint Validation')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Constraint Satisfied')

# Add annotations explaining the constraint
plt.figtext(0.5, 0.01, 
           "Pods from timeslot_X.yaml should only be scheduled at timeslot X or later\n" +
           "For example, pods m019-m023 are from timeslot_5.yaml and should start at timeslot 5 or later",
           ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

# Save the plot
output_path = '/root/carbon-aware-orchestrator/figures/Global_Optimal_Placement_Validation/earliest_timeslot_constraint_analysis.png'
plt.savefig(output_path)
print(f"\nAnalysis plot saved to: {output_path}")

# Additional check for pods with earliest_timeslot extraction issues
# This might happen if pod naming doesn't follow the expected pattern
no_earliest_ts = placements_df[placements_df['earliest_timeslot'].isna()]
if len(no_earliest_ts) > 0:
    print(f"\nWarning: Could not extract earliest_timeslot for {len(no_earliest_ts)} pods:")
    print(no_earliest_ts[['pod_id', 'node_id', 'start_slot']])
