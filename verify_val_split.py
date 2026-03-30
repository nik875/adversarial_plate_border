#!/usr/bin/env python3
"""
verify_val_split.py

Cross-checks the labels stored in val_split.csv against the ground truth
derived directly from the source data:
  - CCPD files  : label is re-decoded from the filename
  - Texas files : label is re-read from CCPD2019_TEXAS/metadata.csv

Usage
-----
    python verify_val_split.py --val-csv weights/finetuned/val_split.csv \
                               --data-root /path/to/data
"""

import argparse
import csv
import sys
from pathlib import Path

# ── CCPD decoding (mirrors finetune_all_models.py) ───────────────────────────

_ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]
_ADS       = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + list("0123456789") + ["O"]
CHARS      = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _decode_ccpd_plate(plate_code: str) -> str:
    idx = list(map(int, plate_code.split("_")))
    chars = [_ALPHABETS[idx[1]]] + [_ADS[idx[i]] for i in range(2, 7)]
    return "".join(chars)


def _ccpd_label_from_path(path: Path) -> str | None:
    """Return the plate label encoded in a CCPD filename, or None on parse failure."""
    parts = path.stem.split("-")
    if len(parts) < 7:
        return None
    try:
        raw = _decode_ccpd_plate(parts[4])
        return "".join(c for c in raw if c in CHARS) or None
    except (ValueError, IndexError):
        return None


def _load_texas_labels(texas_root: Path) -> dict[str, str]:
    """Return {image_path_str: label} from CCPD2019_TEXAS/metadata.csv."""
    meta = texas_root / "metadata.csv"
    if not meta.exists():
        return {}
    labels: dict[str, str] = {}
    with open(meta, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_path = Path(row["output_image"])
            if not img_path.is_absolute():
                img_path = (texas_root.parent / img_path).resolve()
            label = "".join(c for c in row["generated_plate"].strip().upper()
                            if c in CHARS)
            labels[str(img_path)] = label
    return labels


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Verify val_split.csv labels against source data")
    p.add_argument("--val-csv",    default="weights/finetuned/val_split.csv",
                   help="Path to val_split.csv (default: weights/finetuned/val_split.csv)")
    p.add_argument("--data-root",  required=True,
                   help="Directory containing CCPD2019/ and optionally CCPD2019_TEXAS/")
    args = p.parse_args()

    val_csv   = Path(args.val_csv)
    data_root = Path(args.data_root)

    if not val_csv.exists():
        sys.exit(f"ERROR: val CSV not found: {val_csv}")

    texas_labels = _load_texas_labels(data_root / "CCPD2019_TEXAS")
    print(f"Loaded {len(texas_labels):,} Texas labels from metadata.csv")

    total = ok = mismatch = skipped = 0
    mismatches: list[dict] = []

    with open(val_csv, newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            img_path = Path(row["image_path"])
            csv_label = row["label"]

            # Determine ground-truth label
            gt_label = _ccpd_label_from_path(img_path)

            if gt_label is None:
                # Not a CCPD-encoded filename — try Texas lookup
                gt_label = texas_labels.get(str(img_path))

            if gt_label is None:
                skipped += 1
                continue

            if csv_label == gt_label:
                ok += 1
            else:
                mismatch += 1
                mismatches.append({
                    "image":     str(img_path),
                    "csv_label": csv_label,
                    "gt_label":  gt_label,
                })

    print(f"\nResults  ({total:,} rows in val_split.csv)")
    print(f"  ✓  Match    : {ok:,}")
    print(f"  ✗  Mismatch : {mismatch:,}")
    print(f"  ?  Skipped  : {skipped:,}  (couldn't derive GT label)")

    if mismatches:
        print(f"\nFirst {min(20, len(mismatches))} mismatches:")
        print(f"  {'CSV label':<12}  {'GT label':<12}  Image")
        for m in mismatches[:20]:
            print(f"  {m['csv_label']:<12}  {m['gt_label']:<12}  {m['image']}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
        sys.exit(1)
    else:
        print("\nAll labels match.")


if __name__ == "__main__":
    main()
