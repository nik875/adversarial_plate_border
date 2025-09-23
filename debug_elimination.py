#!/usr/bin/env python3

"""
Debug script to isolate the elimination discrepancy bug between:
- Main visualization: 0% eliminated
- Focused plots: 45.6% eliminated

Usage: python debug_elimination.py
"""

from analyze_patch import PatchEvaluator
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import the original script's classes
sys.path.append('.')


class DebugPatchEvaluator(PatchEvaluator):
    """Extended evaluator with debugging for elimination discrepancy"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_data = {}

    def debug_elimination_calculations(self, results_df: pd.DataFrame):
        """Debug the exact point where elimination calculations diverge"""

        print("=" * 60)
        print("DEBUGGING ELIMINATION DISCREPANCY")
        print("=" * 60)

        # Step 1: Replicate exact filtering from create_visualizations
        print(f"\nSTEP 1: Data Filtering")
        print(f"Original results_df shape: {results_df.shape}")
        print(f"Original results_df columns: {list(results_df.columns)}")
        print(f"patch_applied column unique values: {results_df['patch_applied'].unique()}")

        valid_results = results_df[results_df['patch_applied'] == True].copy()
        print(f"After patch_applied filter: {valid_results.shape}")

        if len(valid_results) == 0:
            raise RuntimeError("NO VALID RESULTS FOUND - All images failed processing!")

        # Step 2: Analyze best_iou column in detail
        print(f"\nSTEP 2: best_iou Column Analysis")
        print(f"best_iou dtype: {valid_results['best_iou'].dtype}")
        print(f"best_iou shape: {valid_results['best_iou'].shape}")
        print(f"best_iou NaN count: {valid_results['best_iou'].isna().sum()}")
        print(f"best_iou infinite count: {np.isinf(valid_results['best_iou']).sum()}")

        # Show actual values
        unique_ious = valid_results['best_iou'].unique()
        print(f"Unique best_iou values ({len(unique_ious)}): {sorted(unique_ious)}")

        # Count by value type
        for val in sorted(unique_ious):
            count = (valid_results['best_iou'] == val).sum()
            print(f"  best_iou == {val}: {count} rows (type: {type(val)})")

        # Step 3: Method 1 - Main visualization calculation (direct boolean)
        print(f"\nSTEP 3: Method 1 - Main Visualization Calculation")
        print("Code: (valid_results['best_iou'] == 0).sum()")

        # Test different zero comparisons
        eq_zero = (valid_results['best_iou'] == 0)
        eq_zero_float = (valid_results['best_iou'] == 0.0)
        eq_zero_int = (valid_results['best_iou'] == int(0))

        print(f"best_iou == 0: {eq_zero.sum()} True values")
        print(f"best_iou == 0.0: {eq_zero_float.sum()} True values")
        print(f"best_iou == int(0): {eq_zero_int.sum()} True values")

        detections_eliminated_method1 = eq_zero.sum()
        print(f"METHOD 1 RESULT: {detections_eliminated_method1} eliminated")

        # Show which rows are eliminated by method 1
        eliminated_indices_method1 = valid_results[eq_zero].index.tolist()
        print(f"Method 1 eliminated row indices: {eliminated_indices_method1}")

        # Step 4: Method 2 - Focused plots calculation (apply function)
        print(f"\nSTEP 4: Method 2 - Focused Plots Calculation")
        print("Code: results_df.apply(categorize_effectiveness, axis=1)")

        def categorize_effectiveness_debug(row):
            """Debug version of categorize_effectiveness with detailed logging"""
            iou_val = row['best_iou']
            conf_change = row['confidence_change']

            # Debug each condition
            is_zero = (iou_val == 0)

            if is_zero:
                result = 'Eliminated'
            elif conf_change < -0.1:
                result = 'Strong Reduction'
            elif conf_change < -0.05:
                result = 'Moderate Reduction'
            elif abs(conf_change) <= 0.05:
                result = 'Minimal Impact'
            else:
                result = 'Increased'

            return result

        # Apply categorization
        debug_df = valid_results.copy()
        debug_df['effectiveness'] = debug_df.apply(categorize_effectiveness_debug, axis=1)
        effectiveness_counts = debug_df['effectiveness'].value_counts()

        print(f"Effectiveness counts: {dict(effectiveness_counts)}")
        detections_eliminated_method2 = effectiveness_counts.get('Eliminated', 0)
        print(f"METHOD 2 RESULT: {detections_eliminated_method2} eliminated")

        # Show which rows are eliminated by method 2
        eliminated_indices_method2 = debug_df[debug_df['effectiveness']
                                              == 'Eliminated'].index.tolist()
        print(f"Method 2 eliminated row indices: {eliminated_indices_method2}")

        # Step 5: Compare the two methods row by row
        print(f"\nSTEP 5: Row-by-Row Comparison")
        print(f"Method 1 eliminated: {len(eliminated_indices_method1)} rows")
        print(f"Method 2 eliminated: {len(eliminated_indices_method2)} rows")

        set1 = set(eliminated_indices_method1)
        set2 = set(eliminated_indices_method2)

        only_method1 = set1 - set2
        only_method2 = set2 - set1
        both_methods = set1 & set2

        print(f"Eliminated by both methods: {len(both_methods)}")
        print(f"Eliminated by method 1 ONLY: {len(only_method1)}")
        print(f"Eliminated by method 2 ONLY: {len(only_method2)}")

        if only_method1:
            print(f"  Method 1 only indices: {sorted(list(only_method1))[:10]}...")
        if only_method2:
            print(f"  Method 2 only indices: {sorted(list(only_method2))[:10]}...")

        # Step 6: Detailed analysis of discrepant rows
        print(f"\nSTEP 6: Detailed Analysis of Discrepant Rows")

        # Check first few rows that differ
        discrepant_indices = list(only_method1.union(only_method2))[:5]

        for idx in discrepant_indices:
            row = valid_results.loc[idx]
            iou_val = row['best_iou']

            # Test the comparison manually
            direct_zero_check = (iou_val == 0)
            apply_zero_check = categorize_effectiveness_debug(row) == 'Eliminated'

            print(f"\nRow {idx}:")
            print(f"  best_iou value: {repr(iou_val)} (type: {type(iou_val)})")
            print(f"  iou_val == 0: {direct_zero_check}")
            print(f"  apply() result: {apply_zero_check}")
            print(f"  confidence_change: {row['confidence_change']}")

            # Test different zero types
            print(f"  iou_val == 0: {iou_val == 0}")
            print(f"  iou_val == 0.0: {iou_val == 0.0}")
            print(f"  iou_val is 0: {iou_val is 0}")
            print(f"  iou_val is 0.0: {iou_val is 0.0}")
            print(f"  np.isclose(iou_val, 0): {np.isclose(iou_val, 0)}")

            if direct_zero_check != apply_zero_check:
                print(f"  *** DISCREPANCY FOUND IN ROW {idx} ***")
                # This is the smoking gun - save all details
                self.debug_data['discrepant_row'] = {
                    'index': idx,
                    'best_iou': iou_val,
                    'best_iou_type': str(type(iou_val)),
                    'best_iou_repr': repr(iou_val),
                    'direct_check': direct_zero_check,
                    'apply_check': apply_zero_check,
                    'confidence_change': row['confidence_change']
                }

                # Fail loudly as requested
                raise AssertionError(
                    f"ELIMINATION DISCREPANCY DETECTED!\n"
                    f"Row {idx}: best_iou={repr(iou_val)} (type: {type(iou_val)})\n"
                    f"Direct boolean check (iou_val == 0): {direct_zero_check}\n"
                    f"Apply function check: {apply_zero_check}\n"
                    f"This explains the 0% vs 45.6% discrepancy!"
                )

        # Step 7: Summary
        print(f"\nSTEP 7: Summary")
        method1_pct = (
            detections_eliminated_method1 /
            len(valid_results) *
            100) if len(valid_results) > 0 else 0
        method2_pct = (
            detections_eliminated_method2 /
            len(valid_results) *
            100) if len(valid_results) > 0 else 0

        print(
            f"Method 1 (main viz): {detections_eliminated_method1}/{len(valid_results)} = {method1_pct:.1f}%")
        print(
            f"Method 2 (focused): {detections_eliminated_method2}/{len(valid_results)} = {method2_pct:.1f}%")
        print(f"Discrepancy: {abs(method1_pct - method2_pct):.1f} percentage points")

        if abs(method1_pct - method2_pct) > 1.0:
            print("SIGNIFICANT DISCREPANCY DETECTED!")
        else:
            print("No significant discrepancy found - may be a different issue.")

        return {
            'method1_eliminated': detections_eliminated_method1,
            'method2_eliminated': detections_eliminated_method2,
            'method1_pct': method1_pct,
            'method2_pct': method2_pct,
            'total_valid': len(valid_results)
        }


def main():
    """Run the debug analysis"""

    # Configuration matching the user's command
    csv_path = "test_set_labels.csv"
    patch_file = "patch_epoch_0038.png"
    device = "mps"

    print("Debug Script for Elimination Discrepancy")
    print("=" * 50)
    print(f"CSV: {csv_path}")
    print(f"Patch: {patch_file}")
    print(f"Device: {device}")

    # Check files exist
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not Path(patch_file).exists():
        raise FileNotFoundError(f"Patch file not found: {patch_file}")

    try:
        # Initialize debug evaluator
        print(f"\nInitializing PatchEvaluator...")
        evaluator = DebugPatchEvaluator(
            csv_path=csv_path,
            patch_file=patch_file,
            device=device
        )

        # Run evaluation to get results
        print(f"\nRunning patch evaluation...")
        results_df = evaluator.evaluate_patch(output_dir="debug_output")

        # Run debug analysis
        print(f"\nRunning debug analysis...")
        debug_results = evaluator.debug_elimination_calculations(results_df)

        print(f"\nDEBUG COMPLETE")
        print(f"Debug results: {debug_results}")

        # Save debug data
        import json
        with open("debug_elimination_results.json", "w") as f:
            json.dump(debug_results, f, indent=2)
        print(f"Debug results saved to: debug_elimination_results.json")

    except Exception as e:
        print(f"\nDEBUG ANALYSIS FAILED:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

        # Print full traceback for debugging as requested
        import traceback
        traceback.print_exc()

        # Re-raise to fail loudly
        raise


if __name__ == "__main__":
    main()
