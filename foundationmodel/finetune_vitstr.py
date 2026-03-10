#!/usr/bin/env python3
"""
finetune_vitstr.py

Fine-tunes doctr's vitstr_small on the CCT-S-labeled license plate crops
produced by foundationmodel/dataset/generate_cct_labels.py.

Input CSV: foundationmodel/dataset/cct_labels.csv  (image_path, label)

Output:    weights/vitstr_small_finetuned.pt  (state_dict)

Usage:
    python foundationmodel/finetune_vitstr.py
    python foundationmodel/finetune_vitstr.py \
        --csv foundationmodel/dataset/cct_labels.csv \
        --output weights/vitstr_small_finetuned.pt \
        --epochs 10 --batch-size 64 --lr 1e-4 --device cuda

Requires:
    pip install python-doctr[torch] torch pillow numpy pandas tqdm
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

INPUT_H = 32
INPUT_W = 128


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: str):
        # Keep only rows whose label contains only vocab characters
        valid = df["label"].apply(
            lambda s: isinstance(s, str) and len(s) > 0
                      and all(c in vocab for c in s)
        )
        self.df   = df[valid].reset_index(drop=True)
        self.vocab = vocab
        dropped = len(df) - len(self.df)
        if dropped:
            print(f"  Dropped {dropped} rows with out-of-vocab characters")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = row["label"]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img = img.resize((INPUT_W, INPUT_H), resample=Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0   # [32, 128, 3]
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, 32, 128]
        except Exception:
            # Return a blank image with label on failure
            tensor = torch.zeros(3, INPUT_H, INPUT_W)
        return tensor, label


def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), list(labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune vitstr_small on CCT-labeled plate crops"
    )
    parser.add_argument("--csv",    default="foundationmodel/dataset/cct_labels.csv")
    parser.add_argument("--output", default="weights/vitstr_small_finetuned.pt")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=25,
                        help="Val sample images to save after training (default: 25)")
    parser.add_argument("--samples-dir", default="weights/vitstr_samples",
                        help="Directory for val sample images (default: weights/vitstr_samples)")
    args = parser.parse_args()

    try:
        from doctr.models import vitstr_small
    except ImportError:
        print("ERROR: pip install python-doctr[torch]")
        raise SystemExit(1)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("[Loading pretrained vitstr_small]")
    model = vitstr_small(pretrained=True).to(args.device)
    vocab = model.vocab
    print(f"  Vocab : {len(vocab)} chars")
    print(f"  Device: {args.device}")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    print(f"\n[Loading dataset: {args.csv}]")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} rows in CSV")

    dataset = PlateDataset(df, vocab)
    print(f"  {len(dataset)} usable samples")

    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train: {n_train}  Val: {n_val}")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )

    # -----------------------------------------------------------------------
    # Optimizer + cosine LR schedule
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(loader)  # based on train batches only
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\n[Training — {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}]")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch",
                    dynamic_ncols=True)
        for imgs, labels in pbar:
            imgs = imgs.to(args.device)

            optimizer.zero_grad()
            out  = model(imgs, target=labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix({"loss": f"{epoch_loss / n_batches:.4f}",
                              "lr":   f"{scheduler.get_last_lr()[0]:.2e}"})

        train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss  = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(args.device)
                out  = model(imgs, target=labels)
                val_loss += out["loss"].item()
                n_val_batches += 1
        val_loss /= max(n_val_batches, 1)

        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"  Saved → {output_path}  (best val so far)")

    print(f"\nDone. Best val loss: {best_loss:.4f}")
    print(f"Weights saved to: {output_path}")

    # -----------------------------------------------------------------------
    # Sample outputs from val set
    # -----------------------------------------------------------------------
    print(f"\n[Saving {args.samples} val sample images → {args.samples_dir}/]")
    samples_dir = Path(args.samples_dir)
    if samples_dir.exists():
        for f in samples_dir.iterdir():
            f.unlink()
    samples_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    model.eval()
    saved = 0
    with torch.no_grad():
        for imgs, labels in DataLoader(val_ds, batch_size=args.batch_size,
                                       shuffle=True, collate_fn=collate_fn):
            imgs_dev = imgs.to(args.device)
            out = model(imgs_dev, return_preds=True)
            preds = [p[0].upper() for p in out["preds"]]

            for img_t, gt, pred in zip(imgs, labels, preds):
                if saved >= args.samples:
                    break
                # img_t is [3, 32, 128] float32 [0,1] — convert back to PIL
                arr = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                vis = Image.fromarray(arr).resize(
                    (INPUT_W * 4, INPUT_H * 4), resample=Image.NEAREST
                )
                draw = ImageDraw.Draw(vis)
                text = f"GT:{gt}  PR:{pred}"
                draw.text((3, 3), text, fill=(0, 0, 0),       font=font)
                draw.text((2, 2), text, fill=(255, 255, 255), font=font)
                match = "ok" if pred == gt.upper() else "xx"
                vis.save(samples_dir / f"{saved:03d}_{match}.png")
                saved += 1

            if saved >= args.samples:
                break

    print(f"  Saved {saved} samples.")


if __name__ == "__main__":
    main()
