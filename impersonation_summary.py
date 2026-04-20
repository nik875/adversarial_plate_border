#!/usr/bin/env python3
"""
impersonation_summary.py

Read evaluate_finetuned.py output directories and report impersonation
success rate and correct read rate for each patch × pipeline combination.

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


def match_rate(df: pd.DataFrame, target: str) -> tuple[float, int, int]:
    """Fraction of rows where pred_text normalizes to a fixed target string."""
    total = len(df)
    if total == 0:
        return float("nan"), 0, 0
    norm_target = normalize(target)
    matches = df["pred_text"].apply(lambda t: normalize(t) == norm_target).sum()
    return matches / total, int(matches), total


def correct_read_rate(df: pd.DataFrame) -> tuple[float, int, int]:
    """Fraction of rows where pred_text matches the row's own gt_text."""
    total = len(df)
    if total == 0:
        return float("nan"), 0, 0
    matches = df.apply(
        lambda r: normalize(r["pred_text"]) == normalize(r["gt_text"]), axis=1
    ).sum()
    return matches / total, int(matches), total


def print_table(title: str, rows: list[dict], pipelines: list[str],
                rate_key: str, count_key: str) -> None:
    col_w   = max(20, *(len(p) + 2 for p in pipelines))
    patch_w = max(len(r["patch"]) for r in rows) + 2

    header = f"{'Patch':<{patch_w}}" + "".join(f"{p:^{col_w}}" for p in pipelines)
    print(title)
    print(header)
    print("─" * len(header))
    for row in rows:
        line = f"{row['patch']:<{patch_w}}"
        for p in pipelines:
            val = row[f"{p}_{rate_key}"]
            n   = row.get(f"{p}_{count_key}", "")
            cell = "—" if val != val else f"{val:.1%} ({n})"
            line += f"{cell:^{col_w}}"
        print(line)
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, metavar="DIR",
                        help="Base results directory (contains one subdir per patch/holdout)")
    parser.add_argument("--target", required=True, metavar="PLATE",
                        help="Impersonation target plate string (e.g. SHX8459)")
    parser.add_argument("--csv", default=None, metavar="FILE",
                        help="Also save the combined table to this CSV file")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"[error] Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    subdirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    if not subdirs:
        print(f"[error] No subdirectories found in {results_dir}", file=sys.stderr)
        sys.exit(1)

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
                for key in ("imp_rate", "imp_n", "correct_rate", "correct_n"):
                    row[f"{pipeline}_{key}"] = float("nan") if key.endswith("rate") else ""
                continue
            df = pd.read_csv(csv_path)
            if "pred_text" not in df.columns or "gt_text" not in df.columns:
                for key in ("imp_rate", "imp_n", "correct_rate", "correct_n"):
                    row[f"{pipeline}_{key}"] = float("nan") if key.endswith("rate") else ""
                continue

            imp_r, imp_m, total      = match_rate(df, args.target)
            correct_r, correct_m, _ = correct_read_rate(df)

            row[f"{pipeline}_imp_rate"]     = imp_r
            row[f"{pipeline}_imp_n"]        = f"{imp_m}/{total}"
            row[f"{pipeline}_correct_rate"] = correct_r
            row[f"{pipeline}_correct_n"]    = f"{correct_m}/{total}"
        rows.append(row)

    print(f"\nImpersonation success rate  (target: {args.target})\n")
    print_table("", rows, pipelines, "imp_rate", "imp_n")

    print(f"Correct read rate  (attack failed — model reads real plate)\n")
    print_table("", rows, pipelines, "correct_rate", "correct_n")

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"Saved to {args.csv}")


if __name__ == "__main__":
    main()
