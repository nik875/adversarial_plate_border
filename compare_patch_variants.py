#!/usr/bin/env python3
"""
Compare patch variant performance across all ablation conditions.
Creates pie charts showing detection/OCR results for each variant.
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def categorize_result(row, target_plate=None):
    """Categorize detection result"""
    TRUE_PLATE = 'VRJ7774'  # Ground truth plate number for control images

    # Check if detection was eliminated (IoU = 0)
    if pd.isna(row.get('best_iou')) or row.get('best_iou', 0) == 0:
        return 'Detection Eliminated'

    # Get OCR text from detection
    ocr_text = str(row.get('detection_text', '')).strip()

    if pd.isna(row.get('detection_text')) or ocr_text == '':
        return 'Detected (No OCR)'
    elif ocr_text == TRUE_PLATE:
        # Attack failed - correct plate read
        return 'Correct Read (Attack Failed)'
    elif target_plate and ocr_text == target_plate:
        # Impersonation success
        return f'Impersonation Success ({target_plate})'
    else:
        # Misread to some other text
        return 'Misread (Other Text)'


def load_variant_results(eval_results_dir):
    """Load all variant results from evaluation directory"""
    results_dir = Path(eval_results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Evaluation results directory not found: {eval_results_dir}")

    variants = {}

    # Find all variant directories
    for variant_dir in sorted(results_dir.glob('patches_*')):
        if not variant_dir.is_dir():
            continue

        results_file = variant_dir / 'patch_evaluation_results.csv'
        if not results_file.exists():
            print(f"Warning: No patch_evaluation_results.csv found in {variant_dir.name}, skipping...")
            continue

        # Extract variant info from directory name
        variant_name = variant_dir.name.replace('patches_', '')

        # Determine target plate
        target_plate = None
        if 'VJJ7744' in variant_name:
            target_plate = 'VJJ7744'
        elif 'SHX8459' in variant_name:
            target_plate = 'SHX8459'

        # Load results
        df = pd.read_csv(results_file)
        df['category'] = df.apply(lambda row: categorize_result(row, target_plate), axis=1)

        variants[variant_name] = {
            'data': df,
            'target': target_plate,
            'path': variant_dir
        }

        print(f"Loaded {len(df)} results from {variant_name}")

    if len(variants) == 0:
        raise ValueError("No variant results found!")

    return variants


def create_comparison_plots(variants, output_dir):
    """Create comparison pie charts for all variants"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Define consistent colors
    colors = {
        'Detection Eliminated': '#ff6b6b',  # Red
        'Impersonation Success (VJJ7744)': '#9775fa',  # Purple
        'Impersonation Success (SHX8459)': '#845ef7',  # Purple (darker)
        'Misread (Other Text)': '#ff922b',  # Orange
        'Detected (No OCR)': '#ffd43b',  # Light Orange
        'Correct Read (Attack Failed)': '#51cf66'  # Green
    }

    # Create SINGLE figure with ALL 8 variants (2 rows x 4 cols)
    # Organize: VJJ7744 variants in top row, SHX8459 in bottom row
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    variant_positions = [
        ('VJJ7744', 0, 0),
        ('VJJ7744_notv', 0, 1),
        ('VJJ7744_nohomo', 0, 2),
        ('VJJ7744_notv_nohomo', 0, 3),
        ('SHX8459', 1, 0),
        ('SHX8459_notv', 1, 1),
        ('SHX8459_nohomo', 1, 2),
        ('SHX8459_notv_nohomo', 1, 3)
    ]

    for variant_name, row, col in variant_positions:
        ax = axes[row, col]

        if variant_name not in variants:
            ax.axis('off')
            ax.text(0.5, 0.5, f'{variant_name}\nNo data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            continue

        data = variants[variant_name]['data']
        target = variants[variant_name]['target']

        # Count categories
        category_counts = data['category'].value_counts()

        # Prepare data for pie chart
        sizes = []
        plot_colors = []
        labels = []

        for category in category_counts.index:
            count = category_counts[category]
            pct = (count / len(data)) * 100
            # Simpler labels for cleaner look
            labels.append(f'{category.split("(")[0].strip()}\n{pct:.1f}%')
            sizes.append(count)
            plot_colors.append(colors.get(category, '#cccccc'))

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=plot_colors,
            autopct='%1.0f',
            startangle=90,
            textprops={'fontsize': 8}
        )

        # Style percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)

        # Compact title with variant configuration
        title_parts = []
        if target:
            title_parts.append(f'Target: {target}')

        if '_notv' in variant_name and '_nohomo' in variant_name:
            title_parts.append('No TV Loss + No Homography')
        elif '_notv' in variant_name:
            title_parts.append('No TV Loss')
        elif '_nohomo' in variant_name:
            title_parts.append('No Homography')
        else:
            title_parts.append('Full (TV + Homography)')

        title_parts.append(f'n={len(data)}')

        ax.set_title('\n'.join(title_parts), fontsize=11, fontweight='bold', pad=10)

    # Add overall title
    plt.suptitle('Patch Variant Comparison - All Ablation Conditions',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save combined plot
    output_path = output_dir / 'all_variants_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined comparison plot: {output_path}")

    # Also create separate plots for each target (keep original functionality)
    create_separate_target_plots(variants, output_dir, colors)


def create_separate_target_plots(variants, output_dir, colors):
    """Create separate comparison plots for each target (original functionality)"""
    vjj7744_variants = {k: v for k, v in variants.items() if 'VJJ7744' in k}
    shx8459_variants = {k: v for k, v in variants.items() if 'SHX8459' in k}

    for target_name, target_variants in [('VJJ7744', vjj7744_variants), ('SHX8459', shx8459_variants)]:
        if len(target_variants) == 0:
            continue

        variant_order = []
        for suffix in ['', '_notv', '_nohomo', '_notv_nohomo']:
            key = f'{target_name}{suffix}'
            if key in target_variants:
                variant_order.append(key)

        n_variants = len(variant_order)
        fig, axes = plt.subplots(1, n_variants, figsize=(6*n_variants, 6))

        if n_variants == 1:
            axes = [axes]

        for idx, variant_name in enumerate(variant_order):
            ax = axes[idx]
            data = target_variants[variant_name]['data']
            target = target_variants[variant_name]['target']

            category_counts = data['category'].value_counts()

            labels = []
            sizes = []
            plot_colors = []

            for category in category_counts.index:
                count = category_counts[category]
                pct = (count / len(data)) * 100
                labels.append(f'{category}\n({count}, {pct:.1f}%)')
                sizes.append(count)
                plot_colors.append(colors.get(category, '#cccccc'))

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=plot_colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 9}
            )

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)

            title_parts = [f'Target: {target}']
            if '_notv' in variant_name:
                title_parts.append('TV Loss: OFF')
            else:
                title_parts.append('TV Loss: ON')

            if '_nohomo' in variant_name:
                title_parts.append('Homography: OFF')
            else:
                title_parts.append('Homography: ON')

            ax.set_title('\n'.join(title_parts), fontsize=12, fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / f'patch_comparison_{target_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {target_name} comparison plot: {output_path}")

    # Create combined summary statistics
    create_summary_table(variants, output_dir)

    # Create effectiveness comparison bar chart
    create_effectiveness_comparison(variants, output_dir)


def create_summary_table(variants, output_dir):
    """Create summary statistics table"""
    summary_data = []

    for variant_name, variant_info in sorted(variants.items()):
        data = variant_info['data']
        target = variant_info['target']

        total = len(data)
        eliminated = len(data[data['category'] == 'Detection Eliminated'])
        impersonated = len(data[data['category'].str.contains('Impersonation Success', na=False)])
        misread = len(data[data['category'] == 'Misread (Other Text)'])
        correct = len(data[data['category'] == 'Correct Read (Attack Failed)'])
        no_ocr = len(data[data['category'] == 'Detected (No OCR)'])

        # Calculate effectiveness metrics
        total_disruption = eliminated + impersonated + misread

        summary_data.append({
            'Variant': variant_name,
            'Target': target or 'N/A',
            'Total Images': total,
            'Eliminated (%)': f'{eliminated} ({eliminated/total*100:.1f}%)',
            'Impersonation (%)': f'{impersonated} ({impersonated/total*100:.1f}%)' if target else 'N/A',
            'Misread (%)': f'{misread} ({misread/total*100:.1f}%)',
            'No OCR (%)': f'{no_ocr} ({no_ocr/total*100:.1f}%)',
            'Correct (%)': f'{correct} ({correct/total*100:.1f}%)',
            'Total Disruption (%)': f'{total_disruption} ({total_disruption/total*100:.1f}%)'
        })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    output_path = output_dir / 'variant_comparison_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"Saved summary table: {output_path}")

    return summary_df


def create_effectiveness_comparison(variants, output_dir):
    """Create bar chart comparing effectiveness across variants"""
    # Separate by target
    for target_name in ['VJJ7744', 'SHX8459']:
        target_variants = {k: v for k, v in variants.items() if target_name in k}

        if len(target_variants) == 0:
            continue

        # Sort variants
        variant_order = []
        for suffix in ['', '_notv', '_nohomo', '_notv_nohomo']:
            key = f'{target_name}{suffix}'
            if key in target_variants:
                variant_order.append(key)

        # Calculate metrics
        metrics = {
            'Eliminated': [],
            'Impersonation': [],
            'Misread': [],
            'Failed': []
        }

        labels = []
        for variant_name in variant_order:
            data = target_variants[variant_name]['data']
            total = len(data)

            eliminated = len(data[data['category'] == 'Detection Eliminated']) / total * 100
            impersonated = len(data[data['category'].str.contains('Impersonation Success', na=False)]) / total * 100
            misread = len(data[data['category'] == 'Misread (Other Text)']) / total * 100
            failed = len(data[data['category'] == 'Correct Read (Attack Failed)']) / total * 100

            metrics['Eliminated'].append(eliminated)
            metrics['Impersonation'].append(impersonated)
            metrics['Misread'].append(misread)
            metrics['Failed'].append(failed)

            # Create short label
            label = 'Full'
            if '_notv_nohomo' in variant_name:
                label = 'No TV+Homo'
            elif '_notv' in variant_name:
                label = 'No TV'
            elif '_nohomo' in variant_name:
                label = 'No Homo'
            labels.append(label)

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(labels))
        width = 0.6

        # Stack bars
        p1 = ax.bar(x, metrics['Eliminated'], width, label='Detection Eliminated', color='#ff6b6b')
        p2 = ax.bar(x, metrics['Impersonation'], width, bottom=metrics['Eliminated'],
                   label='Impersonation Success', color='#9775fa')
        p3 = ax.bar(x, metrics['Misread'], width,
                   bottom=np.array(metrics['Eliminated']) + np.array(metrics['Impersonation']),
                   label='Misread (Other)', color='#ff922b')
        p4 = ax.bar(x, metrics['Failed'], width,
                   bottom=np.array(metrics['Eliminated']) + np.array(metrics['Impersonation']) + np.array(metrics['Misread']),
                   label='Attack Failed (Correct)', color='#51cf66')

        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Patch Effectiveness Comparison - {target_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i in x:
            total_disruption = metrics['Eliminated'][i] + metrics['Impersonation'][i] + metrics['Misread'][i]
            ax.text(i, total_disruption + 2, f'{total_disruption:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()

        output_path = output_dir / f'effectiveness_comparison_{target_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved effectiveness comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare patch variant performance across ablation conditions'
    )
    parser.add_argument('--eval-dir', required=True,
                       help='Evaluation results directory (e.g., evaluation_results_20260101_150754)')
    parser.add_argument('--output', default='variant_comparison',
                       help='Output directory for comparison plots')

    args = parser.parse_args()

    print("=" * 60)
    print("Patch Variant Comparison Analysis")
    print("=" * 60)

    # Load all variant results
    print("\nLoading variant results...")
    variants = load_variant_results(args.eval_dir)

    print(f"\nFound {len(variants)} variants:")
    for name in sorted(variants.keys()):
        target = variants[name]['target']
        print(f"  - {name} (Target: {target or 'None'})")

    # Create comparison plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(variants, args.output)

    print("\n" + "=" * 60)
    print(f"Analysis complete! Results saved to: {args.output}/")
    print("=" * 60)
    print("\nGenerated files:")
    output_dir = Path(args.output)
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
