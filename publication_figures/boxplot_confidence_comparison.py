import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read both CSV files
df_standard = pd.read_csv('full_results.csv')
df_large = pd.read_csv('full_results_largedet.csv')

# Add model identifier
df_standard['model'] = 'Standard'
df_large['model'] = 'Large'

# Combine datasets
df_combined = pd.concat([df_standard, df_large], ignore_index=True)

# Filter out rows with no detection (NaN confidence)
df_combined = df_combined[df_combined['detected_plate_confidence'].notna()]

# Get unique values
conditions = ['control', 'disruption', 'impersonation']
times_of_day = ['full sun', 'dusk', 'dark no flash', 'dark flash']

# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Create boxplot for each lighting condition
for idx, time in enumerate(times_of_day):
    ax = axes[idx]

    # Filter data for this lighting condition
    data_time = df_combined[df_combined['time_of_day'] == time]

    # Prepare data for grouped boxplot
    positions = []
    data_to_plot = []
    labels = []
    colors = []

    for i, condition in enumerate(conditions):
        # Standard model data
        standard_data = data_time[
            (data_time['condition'] == condition) &
            (data_time['model'] == 'Standard')
        ]['detected_plate_confidence'].values

        # Large model data
        large_data = data_time[
            (data_time['condition'] == condition) &
            (data_time['model'] == 'Large')
        ]['detected_plate_confidence'].values

        # Position for grouped boxplots
        # Groups of 2, with spacing between attack types
        base_pos = i * 3  # spacing between attack types
        positions.extend([base_pos, base_pos + 1])
        data_to_plot.extend([standard_data, large_data])
        colors.extend(['lightblue', 'lightcoral'])

    # Create boxplot
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Make median lines more visible (black and thicker)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    # Set x-axis labels
    tick_positions = [i * 3 + 0.5 for i in range(len(conditions))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([c.title() for c in conditions])

    # Labels and title
    ax.set_ylabel('Detection Confidence', fontsize=10)
    ax.set_title(f'{time.title()}', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend to all subplots
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Standard Detection Model'),
        Patch(facecolor='lightcoral', label='Large Detection Model')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.suptitle('Detection Confidence Comparison by Detection Model and Lighting Condition',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('confidence_boxplot_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Boxplot comparison saved to confidence_boxplot_comparison.png")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
for time in times_of_day:
    print(f"\n{time.upper()}:")
    print("-" * 80)
    for condition in conditions:
        std_data = df_combined[
            (df_combined['time_of_day'] == time) &
            (df_combined['condition'] == condition) &
            (df_combined['model'] == 'Standard')
        ]['detected_plate_confidence']

        large_data = df_combined[
            (df_combined['time_of_day'] == time) &
            (df_combined['condition'] == condition) &
            (df_combined['model'] == 'Large')
        ]['detected_plate_confidence']

        print(f"  {condition.title()}:")
        print(f"    Standard: n={len(std_data):3d}, mean={std_data.mean():.4f}, median={std_data.median():.4f}")
        print(f"    Large:    n={len(large_data):3d}, mean={large_data.mean():.4f}, median={large_data.median():.4f}")
