#!/usr/bin/env python3
"""
Generate pie chart showing detection outcomes for a specific patch condition.
Aggregates over all lighting conditions and viewing angles from physical world test data.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def categorize_result(detected_text, true_plate, impersonation_target):
    """
    Categorize detection result into one of four categories.

    Args:
        detected_text: Text detected by the model (or None/NaN if no detection)
        true_plate: The actual plate number (e.g., "VRJ7774")
        impersonation_target: The impersonation target plate (or None if not applicable)

    Returns:
        Category string
    """
    # Failed detection
    if pd.isna(detected_text) or detected_text is None or detected_text == "":
        return 'Failed detection'

    detected_text = str(detected_text).strip()

    # Correct read
    if detected_text == true_plate:
        return 'Correct read'

    # Successful impersonation
    if impersonation_target and detected_text == impersonation_target:
        return 'Successful impersonation'

    # Misread
    return 'Misread'


def main():
    parser = argparse.ArgumentParser(
        description='Generate pie chart of detection outcomes for a patch condition'
    )
    parser.add_argument(
        '--csv',
        default='full_results_largedet.csv',
        help='Path to results CSV (default: full_results_largedet.csv)'
    )
    parser.add_argument(
        '--condition',
        required=True,
        choices=['control', 'disruption', 'impersonation'],
        help='Patch condition to analyze'
    )
    parser.add_argument(
        '--impersonation-target',
        default='VJJ7744',
        help='Impersonation target plate text (default: VJJ7744)'
    )
    parser.add_argument(
        '--true-plate',
        default='VRJ7774',
        help='True plate text (default: VRJ7774)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output filename (default: {condition}_pie_chart.png)'
    )

    args = parser.parse_args()

    # Read data
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)

    # Filter to specified condition
    df_filtered = df[df['condition'] == args.condition].copy()

    if len(df_filtered) == 0:
        print(f"Error: No data found for condition '{args.condition}'")
        return

    print(f"Found {len(df_filtered)} samples for condition '{args.condition}'")

    # Categorize results
    df_filtered['category'] = df_filtered['detected_plate_text'].apply(
        lambda x: categorize_result(x, args.true_plate, args.impersonation_target)
    )

    # Count categories
    counts = df_filtered['category'].value_counts()

    # Define colors (consistent with digital_world_pie.py)
    colors = {
        'Correct read': '#5cb85c',              # Green
        'Failed detection': '#e57373',          # Red/pink
        'Misread': '#ff9800',                   # Orange
        'Successful impersonation': '#ffd54f'   # Yellow
    }

    # Ensure all categories are present (even if zero)
    all_categories = ['Correct read', 'Successful impersonation', 'Misread', 'Failed detection']
    for cat in all_categories:
        if cat not in counts.index:
            counts[cat] = 0

    # Reorder to standard order
    counts = counts.reindex(all_categories, fill_value=0)

    # Filter out zero counts for cleaner pie chart
    counts_nonzero = counts[counts > 0]
    color_list = [colors[cat] for cat in counts_nonzero.index]

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        counts_nonzero.values,
        labels=counts_nonzero.index,
        autopct='%1.1f%%',
        colors=color_list,
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    # Style percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_weight('bold')

    # Title
    condition_title = args.condition.upper()
    ax.set_title(f'{condition_title} Patch - Detection Outcomes',
                fontsize=16, weight='bold', pad=20)

    # Add subtitle with sample count
    plt.text(0.5, 0.95, f'(n={len(df_filtered)} samples, aggregated over all conditions)',
            ha='center', transform=fig.transFigure, fontsize=11, style='italic')

    # Save
    output_filename = args.output or f'{args.condition}_pie_chart.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Pie chart saved to: {output_filename}")

    # Print summary
    print("\n" + "="*70)
    print(f"DETECTION OUTCOME SUMMARY: {condition_title} PATCH")
    print("="*70)
    print(f"True plate: {args.true_plate}")
    if args.impersonation_target:
        print(f"Impersonation target: {args.impersonation_target}")
    print(f"Total samples: {len(df_filtered)}")
    print("-" * 70)

    for cat in all_categories:
        count = counts[cat]
        pct = count / len(df_filtered) * 100
        print(f"  {cat:30s}: {count:4d} ({pct:5.1f}%)")

    print("="*70)


if __name__ == '__main__':
    main()
