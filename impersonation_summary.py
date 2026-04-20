#!/usr/bin/env python3
"""
impersonation_summary.py

Read evaluate_finetuned.py output directories and report impersonation
success rate for each patch × pipeline combination.

Usage:
    python impersonation_summary.py --results results/holdout_eval/ --target SHX8459
    python impersonation_summary.py --results results/holdout_eval/ --target SHX8459 --csv out.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()


def imp_rate(df: pd.DataFrame, target: str) -> tuple[float, int, int]:
    """Returns (rate, n_matches, n_total) where n_total excludes rows with no detection."""
    total = len(df)
    if total == 0:
        return float("nan"), 0, 0
    norm_target = normalize(target)
    matches = df["pred_text"].apply(lambda t: normalize(t) == norm_target).sum()
    return matches / total, int(matches), total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, metavar="DIR",
                        help="Base results directory (contains one subdir per patch/holdout)")
    parser.add_argument("--target", required=True, metavar="PLATE",
                        help="Impersonation target plate string (e.g. SHX8459)")
    parser.add_argument("--csv", default=None, metavar="FILE",
                        help="Also save the table to this CSV file")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"[error] Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover subdirectories (one per patch run)
    subdirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    if not subdirs:
        print(f"[error] No subdirectories found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all pipeline names across all subdirs
    all_pipelines: set[str] = set()
    for subdir in subdirs:
        for f in subdir.glob("raw_*_pipeline.csv"):
            pipeline = f.stem[len("raw_"):-len("_pipeline")]
            all_pipelines.add(pipeline)
    pipelines = sorted(all_pipelines)

    if not pipelines:
        print(f"[error] No raw_*_pipeline.csv files found under {results_dir}", file=sys.stderr)
        print("       Make sure evaluate_finetuned.py has been run first.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for subdir in subdirs:
        row = {"patch": subdir.name}
        for pipeline in pipelines:
            csv_path = subdir / f"raw_{pipeline}_pipeline.csv"
            if not csv_path.exists():
                row[pipeline] = float("nan")
                row[f"{pipeline}_n"] = ""
                continue
            df = pd.read_csv(csv_path)
            if "pred_text" not in df.columns:
                row[pipeline] = float("nan")
                row[f"{pipeline}_n"] = ""
                continue
            rate, n_match, n_total = imp_rate(df, args.target)
            row[pipeline] = rate
            row[f"{pipeline}_n"] = f"{n_match}/{n_total}"
        rows.append(row)

    # ── Print table ──────────────────────────────────────────────────────────
    col_w = max(20, *(len(p) + 2 for p in pipelines))
    patch_w = max(len(r["patch"]) for r in rows) + 2

    header = f"{'Patch':<{patch_w}}" + "".join(f"{p:^{col_w}}" for p in pipelines)
    print(f"\nImpersonation success rate  (target: {args.target})\n")
    print(header)
    print("─" * len(header))

    for row in rows:
        line = f"{row['patch']:<{patch_w}}"
        for p in pipelines:
            val = row[p]
            n   = row.get(f"{p}_n", "")
            if val != val:  # nan
                cell = "—"
            else:
                cell = f"{val:.1%} ({n})"
            line += f"{cell:^{col_w}}"
        print(line)

    print()

    # ── Optionally save CSV ──────────────────────────────────────────────────
    if args.csv:
        out_rows = []
        for row in rows:
            out_row = {"patch": row["patch"]}
            for p in pipelines:
                out_row[f"{p}_imp_rate"] = row[p]
                out_row[f"{p}_imp_count"] = row.get(f"{p}_n", "")
            out_rows.append(out_row)
        pd.DataFrame(out_rows).to_csv(args.csv, index=False)
        print(f"Saved to {args.csv}")


if __name__ == "__main__":
    main()
