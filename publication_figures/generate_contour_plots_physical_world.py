#!/usr/bin/env python3
"""
Generate contour plots for physical world test showing different metrics (rows)
for control, disruption, and impersonation conditions (columns).
Averages over all lighting conditions at each coordinate point.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein edit distance between two strings"""
    if pd.isna(s1) or s1 is None:
        s1 = ""
    if pd.isna(s2) or s2 is None:
        s2 = ""

    s1, s2 = str(s1), str(s2)

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

print("Loading physical world test results...")
df = pd.read_csv('full_results_largedet.csv')

print(f"Total records: {len(df)}")
print(f"Conditions: {df['condition'].unique()}")
print(f"Lighting conditions: {df['time_of_day'].unique()}")
print(f"X coordinates: {sorted(df['x'].unique())}")
print(f"Y coordinates: {sorted(df['y'].unique())}")

# Calculate metrics
print("\nCalculating metrics...")

# Edit distance to true plate (VRJ7774)
df['edit_dist_true'] = df['detected_plate_text'].apply(
    lambda x: levenshtein_distance(x, "VRJ7774") if pd.notna(x) else np.nan
)

# Edit distance to impersonation target (VJJ7744)
df['edit_dist_target'] = df['detected_plate_text'].apply(
    lambda x: levenshtein_distance(x, "VJJ7744") if pd.notna(x) else np.nan
)

# Detection confidence (handle missing detections)
df['confidence'] = df['detected_plate_confidence'].fillna(0.0)

# Define conditions and metrics
conditions = ['control', 'disruption', 'impersonation']
condition_titles = ['Control', 'Disruption (SHX8459)', 'Impersonation (VJJ7744)']

# Metrics to plot (metric_name, column, title, cmap, vmin, vmax)
metrics = [
    ('confidence', 'confidence', 'Detection Confidence', 'RdYlGn', 0, 1),
    ('edit_dist_true', 'edit_dist_true', 'Edit Distance from True Plate', 'RdYlGn_r', 0, 7),
    ('edit_dist_target', 'edit_dist_target', 'Edit Distance from Impersonation Target', 'RdYlGn_r', 0, 7),
]

# Create figure
fig, axes = plt.subplots(len(metrics), len(conditions), figsize=(16, 10.5))

print("\nGenerating contour plots...")

# Process each metric and condition
for i, (metric_name, metric_col, metric_title, cmap, vmin, vmax) in enumerate(metrics):
    for j, condition in enumerate(conditions):
        ax = axes[i, j]

        # Filter data for this condition and aggregate over all lighting conditions
        subset = df[df['condition'] == condition].copy()

        print(f"  {condition} - {metric_name}: {len(subset)} records")

        if len(subset) > 0:
            # Group by x,y coordinates and average over lighting conditions
            grouped = subset.groupby(['x', 'y']).agg({
                metric_col: 'mean'
            }).reset_index()

            print(f"    After grouping: {len(grouped)} coordinate points")

            # Extract coordinates and metric values
            # NOTE: Swapping x and y to match camera view
            # y = distance from camera (depth)
            # x = horizontal offset (left/right)
            x_data = grouped['y'].values  # Distance from camera (horizontal axis in plot)
            y_data = grouped['x'].values  # Horizontal offset (vertical axis in plot)
            z_data = grouped[metric_col].values

            # Filter out NaN values
            valid_mask = ~np.isnan(z_data)
            x_valid = x_data[valid_mask]
            y_valid = y_data[valid_mask]
            z_valid = z_data[valid_mask]

            print(f"    Valid points: {len(x_valid)}")

            if len(x_valid) > 3:  # Need at least 3 points for interpolation
                # Create grid for interpolation
                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()

                x_padding = (x_max - x_min) * 0.1 or 1
                y_padding = (y_max - y_min) * 0.1 or 1

                xi = np.linspace(x_min - x_padding, x_max + x_padding, 100)
                yi = np.linspace(y_min - y_padding, y_max + y_padding, 100)
                xi_grid, yi_grid = np.meshgrid(xi, yi)

                # Interpolate
                zi = griddata((x_valid, y_valid), z_valid, (xi_grid, yi_grid), method='cubic')

                # Create contour plot
                contour = ax.contourf(xi_grid, yi_grid, zi, levels=15, cmap=cmap,
                                     alpha=0.8, vmin=vmin, vmax=vmax)

                # Add colorbar
                cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(metric_title, fontsize=9)

                # Plot actual data points
                ax.scatter(x_valid, y_valid, c='black', s=10, alpha=0.3, marker='.')

            else:
                ax.text(0.5, 0.5, f'Insufficient data\n({len(x_valid)} points)',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=10, color='red')

        else:
            ax.text(0.5, 0.5, 'No data available',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, color='red')

        # Set title and labels
        if i == 0:
            ax.set_title(condition_titles[j], fontsize=12, weight='bold')
        if j == 0:
            ax.set_ylabel(f'{metric_title}\n\nHorizontal Offset (ft)', fontsize=10, weight='bold')
        else:
            ax.set_ylabel('Horizontal Offset (ft)', fontsize=10)

        ax.set_xlabel('Distance from Camera (ft)', fontsize=10)
        ax.grid(True, alpha=0.3)

plt.suptitle('Horizontal Viewing-Angle Test Results\n(Averaged across all lighting conditions)',
             fontsize=14, weight='bold', y=0.995)
plt.tight_layout(h_pad=3.0)
plt.savefig('contour_plots_physical_world.png', dpi=300, bbox_inches='tight')
print("\n✓ Contour plots saved to: contour_plots_physical_world.png")
