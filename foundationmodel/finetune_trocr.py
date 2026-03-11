#!/usr/bin/env python3
"""
finetune_trocr.py

Fine-tunes microsoft/trocr-small-printed on the CCT-S-labeled license plate
crops produced by foundationmodel/dataset/generate_cct_labels.py.

Input CSV: foundationmodel/dataset/cct_labels.csv  (image_path, label)

Output:    weights/trocr_small_finetuned.pt  (state_dict)

Usage:
    python foundationmodel/finetune_trocr.py
    python foundationmodel/finetune_trocr.py \
        --csv foundationmodel/dataset/cct_labels.csv \
        --output weights/trocr_small_finetuned.pt \
        --epochs 10 --batch-size 32 --lr 1e-4 --device cuda

Requires:
    pip install transformers torch pillow numpy pandas tqdm
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor):
        self.processor = processor
        # Drop rows with empty / non-string labels
        valid = df["label"].apply(lambda s: isinstance(s, str) and len(s) > 0)
        self.df = df[valid].reset_index(drop=True)
        dropped = len(df) - len(self.df)
        if dropped:
            print(f"  Dropped {dropped} rows with missing labels")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 384))
        return img, row["label"]


def collate_fn(batch, processor, max_length: int, device: str = "cpu"):
    """
    Collate PIL images + label strings into pixel_values + label token ids.
    Padding tokens are replaced with -100 so they're ignored in the CE loss.
    """
    imgs, labels = zip(*batch)

    pixel_values = processor(images=list(imgs), return_tensors="pt").pixel_values

    tokenized = processor.tokenizer(
        list(labels),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    label_ids = tokenized.input_ids
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        label_ids = label_ids.masked_fill(label_ids == pad_id, -100)

    return pixel_values, label_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune trocr-small-printed on CCT-labeled plate crops"
    )
    parser.add_argument("--csv",    default="foundationmodel/dataset/cct_labels.csv")
    parser.add_argument("--output", default="weights/trocr_small_finetuned.pt")
    parser.add_argument("--model-id", default="microsoft/trocr-small-printed",
                        help="HuggingFace model ID or local checkpoint dir")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--max-length", type=int,   default=16,
                        help="Max token length for labels (default: 16)")
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=25,
                        help="Val sample images to save after training (default: 25)")
    parser.add_argument("--samples-dir", default="weights/trocr_samples",
                        help="Directory for val sample images (default: weights/trocr_samples)")
    args = parser.parse_args()

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        print("ERROR: pip install transformers")
        raise SystemExit(1)

    from PIL import ImageDraw, ImageFont

    # -----------------------------------------------------------------------
    # Load model + processor
    # -----------------------------------------------------------------------
    print(f"[Loading {args.model_id}]")
    processor = TrOCRProcessor.from_pretrained(args.model_id)
    model     = VisionEncoderDecoderModel.from_pretrained(args.model_id)
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.to(args.device)
    print(f"  Device: {args.device}")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    print(f"\n[Loading dataset: {args.csv}]")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} rows in CSV")

    dataset = PlateDataset(df, processor)
    print(f"  {len(dataset)} usable samples")

    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train: {n_train}  Val: {n_val}")

    def make_collate(proc, max_len):
        def fn(batch):
            return collate_fn(batch, proc, max_len)
        return fn

    cf = make_collate(processor, args.max_length)

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=cf,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=cf,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )

    # -----------------------------------------------------------------------
    # Optimizer + cosine LR schedule
    # -----------------------------------------------------------------------
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(loader)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
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
        for pixel_values, label_ids in pbar:
            pixel_values = pixel_values.to(args.device)
            label_ids    = label_ids.to(args.device)

            optimizer.zero_grad()
            loss = model(pixel_values=pixel_values, labels=label_ids).loss
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
        val_loss      = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for pixel_values, label_ids in val_loader:
                pixel_values = pixel_values.to(args.device)
                label_ids    = label_ids.to(args.device)
                val_loss += model(pixel_values=pixel_values, labels=label_ids).loss.item()
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

    # Sample directly from val_ds (returns PIL image + string label, no collate)
    import random as _random
    val_indices = list(range(len(val_ds)))
    _random.shuffle(val_indices)
    sample_indices = val_indices[:args.samples]

    model.eval()
    saved = 0
    with torch.no_grad():
        for idx in sample_indices:
            pil_img, gt = val_ds[idx]
            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(args.device)
            generated    = model.generate(pixel_values, max_new_tokens=args.max_length)
            pred         = processor.batch_decode(generated, skip_special_tokens=True)[0].upper().strip()

            vis  = pil_img.convert("RGB")
            draw = ImageDraw.Draw(vis)
            text = f"GT:{gt}  PR:{pred}"
            draw.text((3, 3), text, fill=(0, 0, 0),       font=font)
            draw.text((2, 2), text, fill=(255, 255, 255), font=font)
            match = "ok" if pred == gt.upper() else "xx"
            vis.save(samples_dir / f"{saved:03d}_{match}.png")
            saved += 1

    print(f"  Saved {saved} samples.")


if __name__ == "__main__":
    main()
