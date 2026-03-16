import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the largedet CSV file
df = pd.read_csv('full_results_largedet.csv')

def categorize_detection(row):
    """Categorize each detection result"""
    if pd.isna(row['any_plate_detected']) or row['any_plate_detected'] == False:
        return 'No plate detected'
    elif row['detected_plate_text'] == 'VRJ7774':
        return 'Correct plate'
    elif row['detected_plate_text'] == 'VJJ7744':
        return 'Impersonation target'
    else:
        return 'Other plate (misread)'

# Categorize detections
df['category'] = df.apply(categorize_detection, axis=1)

# Get unique values for conditions and time of day
conditions = ['control', 'disruption', 'impersonation']  # Ordered logically
times_of_day = ['full sun', 'dusk', 'dark no flash', 'dark flash']  # Ordered by lighting (inverted)

# Define colors for consistency
colors = {
    'No plate detected': '#ff6b6b',
    'Correct plate': '#51cf66',
    'Impersonation target': '#ffd43b',
    'Other plate (misread)': '#ff922b'
}

category_order = ['No plate detected', 'Correct plate',
                  'Impersonation target', 'Other plate (misread)']

# Create figure with subplots (rows=conditions, cols=times_of_day)
fig, axes = plt.subplots(len(conditions), len(times_of_day),
                         figsize=(20, 12))

# Create each pie chart
for i, condition in enumerate(conditions):
    for j, time in enumerate(times_of_day):
        ax = axes[i, j]

        # Filter data for this combination
        subset = df[(df['time_of_day'] == time) & (df['condition'] == condition)]

        if len(subset) > 0:
            # Count categories
            counts = subset['category'].value_counts()

            # Prepare data for pie chart
            values = [counts.get(cat, 0) for cat in category_order if counts.get(cat, 0) > 0]
            labels = [cat for cat in category_order if counts.get(cat, 0) > 0]
            chart_colors = [colors[cat] for cat in category_order if counts.get(cat, 0) > 0]

            if len(values) > 0:
                # Create pie chart
                wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                                    colors=chart_colors, startangle=90,
                                                    textprops={'fontsize': 8})

                # Make percentage text bold
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(7)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)

        # Set title for each subplot
        title = f"{time.title()}\n{condition.title()}\n(n={len(subset)})"
        ax.set_title(title, fontsize=9, fontweight='bold', pad=10)

# Add column labels at the top (lighting conditions)
for j, time in enumerate(times_of_day):
    axes[0, j].text(0.5, 1.35, time.upper(), ha='center', va='bottom',
                   transform=axes[0, j].transAxes, fontsize=12, fontweight='bold')

# Add row labels on the left (patch types)
for i, condition in enumerate(conditions):
    axes[i, 0].text(-0.35, 0.5, condition.upper(), ha='right', va='center',
                   transform=axes[i, 0].transAxes, fontsize=11, fontweight='bold',
                   rotation=0)

plt.suptitle('Fast-alpr Output on Horizontal Viewing Angle Test',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0.03, 0, 1, 0.99])
plt.savefig('largedet_detailed_breakdown.png', dpi=300, bbox_inches='tight')
print("Detailed breakdown saved to largedet_detailed_breakdown.png")

# Print detailed statistics
print("\n" + "="*80)
print("DETAILED BREAKDOWN - LARGE DETECTOR MODEL")
print("="*80)

for time in times_of_day:
    print(f"\n{'='*80}")
    print(f"TIME OF DAY: {time.upper()}")
    print(f"{'='*80}")

    for condition in conditions:
        subset = df[(df['time_of_day'] == time) & (df['condition'] == condition)]
        print(f"\n  {condition.upper()} (n={len(subset)}):")
        print(f"  {'-'*76}")

        if len(subset) > 0:
            counts = subset['category'].value_counts()
            for cat in category_order:
                count = counts.get(cat, 0)
                pct = (count / len(subset)) * 100 if len(subset) > 0 else 0
                print(f"    {cat:32s}: {count:4d} ({pct:5.1f}%)")
        else:
            print("    No data")

print("\n" + "="*80)
