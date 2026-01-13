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

# Read detection_results.csv
df = pd.read_csv('detection_results.csv')

# Relabel trial groups
df['condition'] = df['trial_group'].map({
    'control': 'control',
    'plate1': 'impersonation',
    'plate2': 'disruption'
})

# Calculate edit distances
df['edit_dist_true'] = df['plate'].apply(
    lambda x: levenshtein_distance(x, "VRJ7774") if pd.notna(x) else np.nan
)
df['edit_dist_impersonation'] = df['plate'].apply(
    lambda x: levenshtein_distance(x, "VJJ7744") if pd.notna(x) else np.nan
)

# Fill NaN confidence values with 0.0 for failed detections
df['confidence'] = df['confidence'].fillna(0.0)

# Define conditions and metrics
conditions = ['control', 'disruption', 'impersonation']
condition_labels = ['Control', 'Disruption', 'Impersonation']
metrics = [
    ('confidence', 'Detection Confidence', 0, 1, 'RdYlGn'),
    ('edit_dist_true', 'Edit Distance from True Plate', 0, 7, 'RdYlGn_r'),
    ('edit_dist_impersonation', 'Edit Distance from Impersonation Target', 0, 7, 'RdYlGn_r')
]

# Create 3x3 grid (3 metrics x 3 conditions)
fig, axes = plt.subplots(3, 3, figsize=(16, 14))

for row_idx, (metric_col, metric_label, vmin, vmax, cmap) in enumerate(metrics):
    for col_idx, (condition, cond_label) in enumerate(zip(conditions, condition_labels)):
        ax = axes[row_idx, col_idx]

        # Filter data for this condition
        subset = df[df['condition'] == condition].copy()

        if len(subset) > 0:
            # Extract coordinates and metric values
            x_data = subset['distance'].values
            y_data = subset['altitude'].values
            z_data = subset[metric_col].values

            # Separate detected vs no-detection points
            detected_mask = ~np.isnan(z_data)
            x_detected = x_data[detected_mask]
            y_detected = y_data[detected_mask]
            z_detected = z_data[detected_mask]

            x_no_detect = x_data[~detected_mask]
            y_no_detect = y_data[~detected_mask]

            # Only create contour if we have detected points
            if len(x_detected) > 0:
                # Create grid for interpolation
                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()

                # Add padding
                x_padding = (x_max - x_min) * 0.1 or 0.5
                y_padding = (y_max - y_min) * 0.1 or 0.1

                xi = np.linspace(x_min - x_padding, x_max + x_padding, 100)
                yi = np.linspace(y_min - y_padding, y_max + y_padding, 100)
                xi_grid, yi_grid = np.meshgrid(xi, yi)

                # Interpolate data onto grid
                zi = griddata((x_detected, y_detected), z_detected, (xi_grid, yi_grid), method='cubic')

                # Create contour plot
                contour = ax.contourf(xi_grid, yi_grid, zi, levels=15, cmap=cmap,
                                     alpha=0.8, vmin=vmin, vmax=vmax)

                # Add colorbar with formatted ticks
                cbar = plt.colorbar(contour, ax=ax, format='%.2f')

                # Add scatter points for detected values
                ax.scatter(x_detected, y_detected, c=z_detected, cmap=cmap,
                          edgecolors='black', linewidths=0.5, s=30, zorder=5,
                          vmin=vmin, vmax=vmax)

            # Add grey scatter points for no-detection cases
            if len(x_no_detect) > 0:
                ax.scatter(x_no_detect, y_no_detect, c='grey', marker='x',
                          linewidths=0.5, s=30, zorder=5)

            ax.set_xlabel('Camera Distance (m)', fontsize=9)
            ax.set_ylabel('Camera Altitude (m)', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)

        # Set title (condition label on top row only)
        if row_idx == 0:
            ax.set_title(f'{cond_label}', fontsize=11, fontweight='bold', pad=10)

        # Add metric label on left column
        if col_idx == 0:
            ax.text(-0.4, 0.5, metric_label, ha='right', va='center',
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   rotation=90)

plt.suptitle('Vertical Viewing-Angle Test Results', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0.05, 0, 1, 0.99])
plt.savefig('detection_results_contour_grid.png', dpi=300, bbox_inches='tight')
print("✅ Contour grid saved to detection_results_contour_grid.png")
