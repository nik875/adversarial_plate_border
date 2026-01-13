#!/usr/bin/env python3
"""
Compare patch variant performance using bar graphs.
Shows ASR for impersonation and correct read reduction for disruption.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def categorize_result(row, target_plate=None):
    """Categorize detection result"""
    TRUE_PLATE = 'VRJ7774'  # Ground truth plate number

    # Check if detection was eliminated (IoU = 0)
    if pd.isna(row.get('best_iou')) or row.get('best_iou', 0) == 0:
        return 'No plate detected'

    # Get OCR text from detection
    ocr_text = str(row.get('detection_text', '')).strip()

    if pd.isna(row.get('detection_text')) or ocr_text == '':
        return 'Detected (no OCR)'
    elif ocr_text == TRUE_PLATE:
        return 'Correct plate'
    elif target_plate and ocr_text == target_plate:
        return 'Impersonation target'
    else:
        return 'Other plate (misread)'

def load_variant_results(eval_results_dir, final_patches_dir='patch_evaluation_results'):
    """Load all variant results from evaluation directory"""
    results_dir = Path(eval_results_dir)
    final_dir = Path(final_patches_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Evaluation results directory not found: {eval_results_dir}")

    variants = {}

    # Load full variants from final_patches directory first
    full_variants = {
        'VJJ7744': (final_dir / 'VJJ7744_impersonation_corrected' / 'patch_evaluation_results.csv', 'VJJ7744'),
        'SHX8459': (final_dir / 'SHX8459_disruption_corrected' / 'patch_evaluation_results.csv', 'SHX8459')
    }

    for variant_name, (results_file, target_plate) in full_variants.items():
        if results_file.exists():
            df = pd.read_csv(results_file)
            df['category'] = df.apply(lambda row: categorize_result(row, target_plate), axis=1)
            variants[variant_name] = {
                'data': df,
                'target': target_plate
            }
            print(f"Loaded {len(df)} results from {variant_name} (final patches)")
        else:
            print(f"Warning: Final patch results not found for {variant_name}: {results_file}")

    # Load ablation variants from evaluation directory
    for variant_dir in sorted(results_dir.glob('patches_*')):
        if not variant_dir.is_dir():
            continue

        # Extract variant info from directory name
        variant_name = variant_dir.name.replace('patches_', '')

        # Skip full variants (already loaded from final_patches)
        if variant_name in ['VJJ7744', 'SHX8459']:
            continue

        results_file = variant_dir / 'patch_evaluation_results.csv'
        if not results_file.exists():
            print(f"Warning: No patch_evaluation_results.csv found in {variant_dir.name}, skipping...")
            continue

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
            'target': target_plate
        }

        print(f"Loaded {len(df)} results from {variant_name} (ablation)")

    if len(variants) == 0:
        raise ValueError("No variant results found!")

    return variants

def load_control_baseline(final_patches_dir='patch_evaluation_results'):
    """Load control baseline to calculate reduction"""
    control_file = Path(final_patches_dir) / 'control_corrected' / 'patch_evaluation_results.csv'

    if not control_file.exists():
        print(f"Warning: Control baseline not found at {control_file}")
        return None

    df = pd.read_csv(control_file)
    df['category'] = df.apply(lambda row: categorize_result(row, None), axis=1)

    # Calculate baseline correct read rate
    correct_count = (df['category'] == 'Correct plate').sum()
    baseline_rate = (correct_count / len(df)) * 100 if len(df) > 0 else 0

    print(f"Loaded control baseline: {correct_count}/{len(df)} correct reads ({baseline_rate:.1f}%)")
    return baseline_rate

def create_variant_label(variant_name):
    """Create clean variant label"""
    if '_notv' in variant_name and '_nohomo' in variant_name:
        return 'No TV +\nNo Homography'
    elif '_notv' in variant_name:
        return 'No TV Loss'
    elif '_nohomo' in variant_name:
        return 'No Homography'
    else:
        return 'Full\n(TV + Homography)'

def main():
    parser = argparse.ArgumentParser(description='Compare patch variant performance with bar graphs')
    parser.add_argument('--eval-dir', required=True,
                       help='Evaluation results directory')
    parser.add_argument('--output', default='variant_comparison_bargraph.png',
                       help='Output filename for comparison plot')
    parser.add_argument('--final-patches-dir', default='patch_evaluation_results',
                       help='Directory containing final patch results (default: patch_evaluation_results)')

    args = parser.parse_args()

    # Load control baseline
    print("Loading control baseline...")
    baseline_correct_rate = load_control_baseline(args.final_patches_dir)

    # Load all variant results
    print("\nLoading variant results...")
    variants = load_variant_results(args.eval_dir, args.final_patches_dir)

    # Define variant configurations
    configs = [
        ('Full\n(TV + Homography)', 'VJJ7744', 'SHX8459'),
        ('No TV Loss', 'VJJ7744_notv', 'SHX8459_notv'),
        ('No Homography', 'VJJ7744_nohomo', 'SHX8459_nohomo'),
        ('No TV +\nNo Homography', 'VJJ7744_notv_nohomo', 'SHX8459_notv_nohomo')
    ]

    # Calculate metrics for each configuration
    config_labels = []
    impersonation_asr = []
    disruption_reduction = []

    print("\nCalculating metrics...")
    print("="*80)

    for config_label, imp_variant, dis_variant in configs:
        config_labels.append(config_label)

        # Impersonation ASR
        if imp_variant in variants:
            imp_data = variants[imp_variant]['data']
            imp_count = (imp_data['category'] == 'Impersonation target').sum()
            asr = (imp_count / len(imp_data)) * 100 if len(imp_data) > 0 else 0
            impersonation_asr.append(asr)
            print(f"{config_label.replace(chr(10), ' ')}:")
            print(f"  Impersonation ASR: {imp_count}/{len(imp_data)} = {asr:.1f}%")
        else:
            impersonation_asr.append(0)
            print(f"{config_label.replace(chr(10), ' ')}:")
            print(f"  Impersonation ASR: No data")

        # Disruption correct read reduction (as percentage of baseline)
        if dis_variant in variants and baseline_correct_rate is not None:
            dis_data = variants[dis_variant]['data']
            correct_count = (dis_data['category'] == 'Correct plate').sum()
            correct_rate = (correct_count / len(dis_data)) * 100 if len(dis_data) > 0 else 0
            absolute_reduction = baseline_correct_rate - correct_rate
            relative_reduction = (absolute_reduction / baseline_correct_rate) * 100 if baseline_correct_rate > 0 else 0
            disruption_reduction.append(relative_reduction)
            print(f"  Disruption reduction: ({baseline_correct_rate:.1f}% - {correct_rate:.1f}%) / {baseline_correct_rate:.1f}% = {relative_reduction:.1f}%")
        else:
            disruption_reduction.append(0)
            print(f"  Disruption reduction: No data")

    print("="*80)

    # Create bar graph
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(config_labels))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, impersonation_asr, width,
                   label='Impersonation: ASR (%)', color='#9467bd', alpha=0.8)
    bars2 = ax.bar(x + width/2, disruption_reduction, width,
                   label='Disruption: % Reduction in Correct Reads', color='#ff7f0e', alpha=0.8)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Ablation Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study Performance Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(impersonation_asr), max(disruption_reduction)) * 1.15)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison bar graph saved to {args.output}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Configuration':<30} {'Impersonation ASR':<20} {'Disruption Reduction':<20}")
    print("-"*80)
    for i, label in enumerate(config_labels):
        clean_label = label.replace('\n', ' ')
        print(f"{clean_label:<30} {impersonation_asr[i]:>6.1f}%{' '*13} {disruption_reduction[i]:>6.1f}%")
    print("="*80)

if __name__ == "__main__":
    main()
