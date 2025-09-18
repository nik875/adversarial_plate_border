#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path


def validate_csv_data(csv_path: str, fix_issues: bool = False):
    """
    Validate CSV data and identify problematic rows that will cause perspective transform failures.

    Args:
        csv_path: Path to the CSV file
        fix_issues: If True, attempt to fix minor issues and save cleaned CSV
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows")

    # Required columns
    required_cols = ['processed_filename', 'alpr_x1', 'alpr_y1', 'alpr_x2', 'alpr_y2']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for duplicate filenames
    duplicate_filenames = df['processed_filename'].duplicated(keep=False)
    duplicate_groups = df[duplicate_filenames].groupby(
        'processed_filename').apply(lambda x: x.index.tolist()).to_dict()

    if duplicate_groups:
        print(
            f"\nFound {len(duplicate_groups)} duplicate filenames affecting {duplicate_filenames.sum()} rows:")
        for filename, row_indices in list(duplicate_groups.items())[:5]:  # Show first 5
            print(f"  '{filename}': rows {row_indices}")
        if len(duplicate_groups) > 5:
            print(f"  ... and {len(duplicate_groups) - 5} more duplicate filename groups")

    issues = []
    valid_rows = []

    for idx, row in df.iterrows():
        row_issues = []

        # Check if image file exists
        if not Path(row['processed_filename']).exists():
            row_issues.append(f"Image file not found: {row['processed_filename']}")

        # Extract bounding box
        try:
            x1, y1, x2, y2 = row['alpr_x1'], row['alpr_y1'], row['alpr_x2'], row['alpr_y2']

            # Check for NaN/infinite values
            coords = [x1, y1, x2, y2]
            if any(pd.isna(coord) or np.isinf(coord) for coord in coords):
                row_issues.append(f"Invalid coordinates: [{x1}, {y1}, {x2}, {y2}]")
                continue

            # Check bounding box validity
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            if bbox_width <= 0:
                row_issues.append(f"Invalid width: {bbox_width} (x1={x1}, x2={x2})")
            if bbox_height <= 0:
                row_issues.append(f"Invalid height: {bbox_height} (y1={y1}, y2={y2})")

            # Check minimum size
            if bbox_width < 5:
                row_issues.append(f"Width too small: {bbox_width}")
            if bbox_height < 5:
                row_issues.append(f"Height too small: {bbox_height}")

            # Check bounds (assuming 384x384 images)
            if x1 < 0 or y1 < 0 or x2 >= 384 or y2 >= 384:
                row_issues.append(f"Out of bounds: [{x1}, {y1}, {x2}, {y2}] (image: 384x384)")

            # If fixing issues is enabled, try to repair minor problems
            if fix_issues and not row_issues:
                # Clamp to valid bounds
                x1 = max(0, min(x1, 383))
                y1 = max(0, min(y1, 383))
                x2 = max(x1 + 5, min(x2, 384))  # Ensure minimum width
                y2 = max(y1 + 5, min(y2, 384))  # Ensure minimum height

                # Update the row
                row['alpr_x1'] = x1
                row['alpr_y1'] = y1
                row['alpr_x2'] = x2
                row['alpr_y2'] = y2

        except (KeyError, TypeError, ValueError) as e:
            row_issues.append(f"Error processing coordinates: {e}")

        # Check if this row has a duplicate filename
        is_duplicate = duplicate_filenames[idx] if idx in duplicate_filenames.index else False
        if is_duplicate:
            duplicate_info = f"Duplicate filename (appears in rows: {duplicate_groups.get(row['processed_filename'], [])})"
            row_issues.append(duplicate_info)

        if row_issues:
            issues.append({
                'row': idx,
                'filename': row.get('processed_filename', 'UNKNOWN'),
                'issues': row_issues,
                'is_duplicate': is_duplicate,
                'data': row[required_cols].to_dict() if all(col in row for col in required_cols) else {}
            })
        else:
            valid_rows.append(idx)

    # Count different types of issues
    duplicate_issues = sum(1 for issue in issues if issue.get('is_duplicate', False))
    coordinate_issues = len(issues) - duplicate_issues

    # Report results
    print(f"\nValidation Results:")
    print(f"  Valid rows: {len(valid_rows)}")
    print(f"  Problematic rows: {len(issues)}")
    print(f"    - Duplicate filenames: {duplicate_issues}")
    print(f"    - Coordinate/file issues: {coordinate_issues}")

    if issues:
        print(f"\nFirst 10 problematic rows:")
        for i, issue in enumerate(issues[:10]):
            print(f"  Row {issue['row']}: {issue['filename']}")
            for problem in issue['issues']:
                print(f"    - {problem}")
            if issue['data'] and not issue.get(
                    'is_duplicate', False):  # Don't show data for duplicate-only issues
                print(f"    - Data: {issue['data']}")
            print()

        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more problematic rows")

    # Save cleaned data if requested
    if fix_issues and len(valid_rows) > 0:
        # For duplicates, keep only the first occurrence unless there are coordinate issues
        rows_to_keep = valid_rows.copy()

        # Add back first occurrence of each duplicate filename if they're otherwise valid
        for filename, row_indices in duplicate_groups.items():
            # Find which of these rows have only duplicate issues (no coordinate problems)
            duplicate_only_rows = []
            for row_idx in row_indices:
                matching_issue = next((issue for issue in issues if issue['row'] == row_idx), None)
                if matching_issue:
                    non_duplicate_issues = [issue for issue in matching_issue['issues']
                                            if not issue.startswith('Duplicate filename')]
                    if not non_duplicate_issues:  # Only duplicate issue
                        duplicate_only_rows.append(row_idx)

            # Keep the first duplicate-only row if any exist
            if duplicate_only_rows:
                first_duplicate = min(duplicate_only_rows)
                if first_duplicate not in rows_to_keep:
                    rows_to_keep.append(first_duplicate)
                    print(
                        f"  Keeping first occurrence of duplicate '{filename}' (row {first_duplicate})")

        clean_df = df.loc[sorted(rows_to_keep)].copy()
        clean_path = csv_path.replace('.csv', '_cleaned.csv')
        clean_df.to_csv(clean_path, index=False)
        print(f"\nSaved {len(clean_df)} clean rows to: {clean_path}")

        # Report on what was cleaned
        removed_duplicates = duplicate_filenames.sum(
        ) - (clean_df['processed_filename'].duplicated().sum())
        if removed_duplicates > 0:
            print(f"  Removed {removed_duplicates} duplicate entries")

        return clean_path

    # Save problematic rows for inspection
    if issues:
        problem_df = pd.DataFrame([{
            'row_index': issue['row'],
            'filename': issue['filename'],
            'is_duplicate': issue.get('is_duplicate', False),
            'issues': '; '.join(issue['issues']),
            **issue['data']
        } for issue in issues])

        problem_path = csv_path.replace('.csv', '_problems.csv')
        problem_df.to_csv(problem_path, index=False)
        print(f"Saved {len(problem_df)} problematic rows to: {problem_path}")

        # Also save a summary of duplicate groups for easier inspection
        if duplicate_groups:
            duplicate_summary = []
            for filename, row_indices in duplicate_groups.items():
                for i, row_idx in enumerate(row_indices):
                    duplicate_summary.append({
                        'filename': filename,
                        'occurrence': i + 1,
                        'row_index': row_idx,
                        'total_occurrences': len(row_indices),
                        'keep_this_one': i == 0  # Mark which one would be kept during fix
                    })

            dup_df = pd.DataFrame(duplicate_summary)
            dup_path = csv_path.replace('.csv', '_duplicates.csv')
            dup_df.to_csv(dup_path, index=False)
            print(f"Saved duplicate filename analysis to: {dup_path}")

    return csv_path if len(issues) == 0 else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Validate CSV data for adversarial patch training')
    parser.add_argument('csv_path', help='Path to CSV file')
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Attempt to fix issues and save cleaned CSV')
    args = parser.parse_args()

    try:
        result_path = validate_csv_data(args.csv_path, fix_issues=args.fix)
        if result_path:
            print(f"\nValidation passed! Use: {result_path}")
        else:
            print(f"\nValidation failed! Check the problems CSV for details.")
            exit(1)
    except Exception as e:
        print(f"Validation error: {e}")
        raise
