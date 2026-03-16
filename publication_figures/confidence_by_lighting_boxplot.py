#!/usr/bin/env python3
"""
Grouped boxplot showing detection confidence by lighting condition.
Compares disruption vs impersonation attacks across different lighting conditions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('full_results_largedet.csv')

# Filter to all conditions including control, only valid detections
df_filtered = df[df['condition'].isin(['control', 'disruption', 'impersonation'])].copy()

# Only include valid detections (confidence > 0)
df_filtered = df_filtered[df_filtered['detected_plate_confidence'] > 0].copy()

# Define lighting conditions in order (brightest to darkest, flash last)
lighting_order = ['full sun', 'dusk', 'dark no flash', 'dark flash']

# Filter to only include lighting conditions in our data
available_lighting = [light for light in lighting_order if light in df_filtered['time_of_day'].unique()]

# Prepare data for boxplot
positions = []
data_control = []
data_disruption = []
data_impersonation = []
labels = []

for i, lighting in enumerate(available_lighting):
    # Get data for this lighting condition
    control_data = df_filtered[
        (df_filtered['time_of_day'] == lighting) &
        (df_filtered['condition'] == 'control')
    ]['detected_plate_confidence'].values

    disruption_data = df_filtered[
        (df_filtered['time_of_day'] == lighting) &
        (df_filtered['condition'] == 'disruption')
    ]['detected_plate_confidence'].values

    impersonation_data = df_filtered[
        (df_filtered['time_of_day'] == lighting) &
        (df_filtered['condition'] == 'impersonation')
    ]['detected_plate_confidence'].values

    data_control.append(control_data)
    data_disruption.append(disruption_data)
    data_impersonation.append(impersonation_data)
    labels.append(lighting.title())

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Define positions for grouped boxplots (3 boxes per group)
num_groups = len(available_lighting)
group_width = 0.8
box_width = group_width / 4
positions_control = np.arange(num_groups) - box_width
positions_disruption = np.arange(num_groups)
positions_impersonation = np.arange(num_groups) + box_width

# Create boxplots
bp_control = ax.boxplot(
    data_control,
    positions=positions_control,
    widths=box_width,
    patch_artist=True,
    showfliers=True,
    boxprops=dict(facecolor='#51cf66', alpha=0.7, edgecolor='black', linewidth=1.5),
    medianprops=dict(color='darkgreen', linewidth=2),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    flierprops=dict(marker='o', markersize=4, alpha=0.5, markerfacecolor='#51cf66', markeredgecolor='black')
)

bp_disruption = ax.boxplot(
    data_disruption,
    positions=positions_disruption,
    widths=box_width,
    patch_artist=True,
    showfliers=True,
    boxprops=dict(facecolor='#ff6b6b', alpha=0.7, edgecolor='black', linewidth=1.5),
    medianprops=dict(color='darkred', linewidth=2),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    flierprops=dict(marker='o', markersize=4, alpha=0.5, markerfacecolor='#ff6b6b', markeredgecolor='black')
)

bp_impersonation = ax.boxplot(
    data_impersonation,
    positions=positions_impersonation,
    widths=box_width,
    patch_artist=True,
    showfliers=True,
    boxprops=dict(facecolor='#9775fa', alpha=0.7, edgecolor='black', linewidth=1.5),
    medianprops=dict(color='darkviolet', linewidth=2),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    flierprops=dict(marker='o', markersize=4, alpha=0.5, markerfacecolor='#9775fa', markeredgecolor='black')
)

# Customize plot
ax.set_ylabel('Detection Confidence', fontsize=13, fontweight='bold')
ax.set_xlabel('Lighting Condition', fontsize=13, fontweight='bold')
ax.set_title('Detection Confidence by Lighting Condition',
             fontsize=14, fontweight='bold', pad=20)

# Set x-axis labels
ax.set_xticks(np.arange(num_groups))
ax.set_xticklabels(labels, fontsize=11)

# Set y-axis limits
ax.set_ylim(-0.05, 1.05)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#51cf66', edgecolor='black', label='Control', alpha=0.7),
    Patch(facecolor='#ff6b6b', edgecolor='black', label='Disruption', alpha=0.7),
    Patch(facecolor='#9775fa', edgecolor='black', label='Impersonation', alpha=0.7)
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=11, framealpha=0.9)


plt.tight_layout()
plt.savefig('confidence_by_lighting_boxplot.png', dpi=300, bbox_inches='tight')
print("Boxplot saved to confidence_by_lighting_boxplot.png")

# Print summary statistics
print("\n" + "="*80)
print("DETECTION CONFIDENCE SUMMARY BY LIGHTING CONDITION")
print("="*80)

for i, lighting in enumerate(available_lighting):
    print(f"\n{lighting.upper()}:")
    print("-" * 80)

    control_vals = data_control[i]
    disruption_vals = data_disruption[i]
    impersonation_vals = data_impersonation[i]

    print(f"\n  Control (n={len(control_vals)}):")
    if len(control_vals) > 0:
        print(f"    Mean:   {control_vals.mean():.3f}")
        print(f"    Median: {np.median(control_vals):.3f}")
        print(f"    Std:    {control_vals.std():.3f}")
        print(f"    Min:    {control_vals.min():.3f}")
        print(f"    Max:    {control_vals.max():.3f}")
    else:
        print("    No data")

    print(f"\n  Disruption (n={len(disruption_vals)}):")
    if len(disruption_vals) > 0:
        print(f"    Mean:   {disruption_vals.mean():.3f}")
        print(f"    Median: {np.median(disruption_vals):.3f}")
        print(f"    Std:    {disruption_vals.std():.3f}")
        print(f"    Min:    {disruption_vals.min():.3f}")
        print(f"    Max:    {disruption_vals.max():.3f}")
    else:
        print("    No data")

    print(f"\n  Impersonation (n={len(impersonation_vals)}):")
    if len(impersonation_vals) > 0:
        print(f"    Mean:   {impersonation_vals.mean():.3f}")
        print(f"    Median: {np.median(impersonation_vals):.3f}")
        print(f"    Std:    {impersonation_vals.std():.3f}")
        print(f"    Min:    {impersonation_vals.min():.3f}")
        print(f"    Max:    {impersonation_vals.max():.3f}")
    else:
        print("    No data")

print("\n" + "="*80)
