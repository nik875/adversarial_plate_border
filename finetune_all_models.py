#!/usr/bin/env python3
"""
finetune_all_models.py

Fine-tunes all detector and OCR backends on CCPD2019 + CCPD2019_TEXAS data.

OCR head replacement
--------------------
CRNN:       rnn[1].embedding  Linear(256, old) → Linear(256, 37)
LPRNet:     dense             Linear(512, 36)  → Linear(512, 37)
TrOCR:      fine-tuned as-is (existing tokenizer already covers A-Z / 0-9)
doctr-vitstr: head replaced with Linear(embed_dim, 37), manual CE loop

Standard alphabet (37 classes):
    blank = 0   (CTC blank / padding)
    '0'-'9' → indices 1-10
    'A'-'Z' → indices 11-36

CCT (fast-plate-ocr ONNX) is not fine-tunable via this script.

Usage
-----
    python finetune_all_models.py \\
        --data-root /path/to/dir \\        # must contain CCPD2019/ and optionally CCPD2019_TEXAS/
        [--output-dir  weights/finetuned] \\
        [--weights-dir weights] \\
        [--models crnn lprnet trocr vitstr yolov8 yolov10 rtdetr fasterrcnn] \\
        [--epochs 10] [--batch-size 32] [--lr 1e-4] [--device cuda] \\
        [--limit 50000]
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Standard alphabet
# ─────────────────────────────────────────────────────────────────────────────

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 36 printable chars
BLANK_IDX = 0        # CTC blank / sequence padding
NUM_CLASSES = 37     # len(CHARS) + 1


def char_to_idx(c: str) -> int:
    """1-based index (blank occupies 0)."""
    return CHARS.index(c) + 1


def pin_memory(device: str) -> bool:
    return device.startswith("cuda")


def find_batch_size(
    probe_fn,
    device: str,
    max_bs: int | None = None,
    safety: float = 0.70,
) -> int:
    """
    Estimate the optimal batch size for a model by probing with batch_size=1
    and measuring peak GPU memory, then scaling to available free memory.

    probe_fn(bs) must run a full forward+backward pass with the given batch
    size using synthetic tensors (no real data loading). It should raise
    RuntimeError on OOM.

    Args:
        probe_fn:  callable(bs) — runs forward+backward, may raise on OOM
        device:    torch device string
        max_bs:    hard upper cap; None means no cap (use all available memory)
        safety:    fraction of estimated headroom to actually use (default 0.70)
                   — leaves room for optimizer states and memory fragmentation

    Returns:
        Chosen batch size (≥ 1).
    """
    if not device.startswith("cuda"):
        return max_bs if max_bs is not None else 32

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_mem, total_mem = torch.cuda.mem_get_info()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    try:
        probe_fn(1)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        per_sample_mem = max(1, peak - baseline)
    except Exception:
        print(f"  [auto-batch] probe failed — using batch_size=1")
        return 1
    finally:
        torch.cuda.empty_cache()

    free_after = torch.cuda.mem_get_info()[0]
    uncapped = max(1, int(free_after * safety / per_sample_mem))
    bs = uncapped if max_bs is None else min(uncapped, max_bs)
    cap_str = "none" if max_bs is None else str(max_bs)
    print(
        f"  [auto-batch] {free_after/1024**3:.1f}/{total_mem/1024**3:.1f} GB free"
        f"  |  {per_sample_mem/1024**2:.0f} MB/sample"
        f"  →  batch_size={bs}  (cap={cap_str})"
    )
    return bs


def time_probe(probe_fn, bs: int, device: str) -> float:
    """
    Time one forward+backward pass of probe_fn(bs) in seconds.
    Does one warmup pass first to ensure GPU kernels are loaded.
    """
    probe_fn(bs)  # warmup
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    t0 = time.perf_counter()
    probe_fn(bs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return elapsed


def fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {int(seconds % 60):02d}s"


def encode_ctc(labels: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of label strings for CTC loss (blank=0, chars 1-based)."""
    targets, lengths = [], []
    for lbl in labels:
        ids = [char_to_idx(c) for c in lbl if c in CHARS]
        targets.extend(ids)
        lengths.append(len(ids))
    return (torch.tensor(targets, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# CCPD parsing
# ─────────────────────────────────────────────────────────────────────────────

_ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]   # 25 entries
_ADS       = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + list("0123456789") + ["O"]  # 35 entries
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _decode_ccpd_plate(plate_code: str) -> str:
    """
    Decode "0_0_22_27_27_33_16" → 6-char alphanumeric string (province skipped).
    'O' is preserved as the letter O (last entry in each array = no-char sentinel,
    but valid CCPD plates never hit that index in practice).
    """
    idx = list(map(int, plate_code.split("_")))
    chars = [_ALPHABETS[idx[1]]] + [_ADS[idx[i]] for i in range(2, 7)]
    return "".join(chars)


def _parse_ccpd_file(path: Path) -> Tuple[Tuple[int, int, int, int], str]:
    parts = path.stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Bad CCPD name: {path.name}")
    tl, br = parts[2].split("_")
    x1, y1 = map(int, tl.split("&"))
    x2, y2 = map(int, br.split("&"))
    label = _decode_ccpd_plate(parts[4])
    return (x1, y1, x2, y2), label


def load_ccpd_records(ccpd_root: Path, limit: Optional[int] = None) -> List[dict]:
    records = []
    for p in sorted(ccpd_root.rglob("*")):
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        try:
            bbox, label = _parse_ccpd_file(p)
        except (ValueError, IndexError):
            continue
        # Keep only labels composed of our standard chars
        label = "".join(c for c in label if c in CHARS)
        if not label:
            continue
        records.append({"image": p, "bbox": bbox, "label": label})
        if limit and len(records) >= limit:
            break
    return records


def load_texas_records(texas_root: Path, limit: Optional[int] = None) -> List[dict]:
    csv_path = texas_root / "metadata.csv"
    if not csv_path.exists():
        print(f"[texas] {csv_path} not found — skipping Texas data.")
        return []
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_path = Path(row["output_image"])
            if not img_path.is_absolute():
                img_path = (texas_root.parent / img_path).resolve()
            if not img_path.exists():
                continue
            bbox = (int(row["bbox_x1"]), int(row["bbox_y1"]),
                    int(row["bbox_x2"]), int(row["bbox_y2"]))
            label = "".join(c for c in row["generated_plate"].strip().upper() if c in CHARS)
            if not label:
                continue
            records.append({"image": img_path, "bbox": bbox, "label": label})
            if limit and len(records) >= limit:
                break
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class OCRCropDataset(Dataset):
    """Crops bbox region, resizes to (H, W), returns (tensor, label_str)."""

    def __init__(self, records: List[dict], crop_hw: Tuple[int, int],
                 grayscale: bool = False):
        self.records   = records
        self.crop_hw   = crop_hw
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        x1, y1, x2, y2 = rec["bbox"]
        H, W = self.crop_hw
        try:
            img = cv2.imread(str(rec["image"]))
            if img is None:
                raise OSError("imread returned None")
            crop = img[max(0, y1):max(y1+1, y2), max(0, x1):max(x1+1, x2)]
            crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
            if self.grayscale:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                t = torch.from_numpy(crop.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)
        except Exception:
            C = 1 if self.grayscale else 3
            t = torch.zeros(C, H, W)
        return t, rec["label"]


class DetectionDataset(Dataset):
    """Returns (image_tensor [3,H,W], target_dict) for torchvision-style detectors."""

    def __init__(self, records: List[dict], img_size: int = 640):
        self.records  = records
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        x1, y1, x2, y2 = rec["bbox"]
        S = self.img_size
        try:
            img = cv2.imread(str(rec["image"]))
            if img is None:
                raise OSError()
            oh, ow = img.shape[:2]
            img = cv2.resize(img, (S, S))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t   = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
            sx, sy = S / ow, S / oh
            box = torch.tensor([[x1*sx, y1*sy, x2*sx, y2*sy]], dtype=torch.float32)
            box = box.clamp(0, S)
        except Exception:
            t   = torch.zeros(3, S, S)
            box = torch.zeros(1, 4)
        target = {"boxes": box, "labels": torch.ones(1, dtype=torch.int64)}
        return t, target


def _collate_det(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


# ─────────────────────────────────────────────────────────────────────────────
# YOLO dataset writer
# ─────────────────────────────────────────────────────────────────────────────

def write_yolo_dataset(records: List[dict], out_dir: Path,
                       val_frac: float = 0.1) -> Path:
    """
    Writes YOLO-format dataset under out_dir and returns path to dataset.yaml.
    Images are hard-linked (or copied) into images/{train,val}/.
    Labels (single box per image, class 0) are written to labels/{train,val}/.
    """
    rng = random.Random(42)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    splits = {"val": shuffled[:n_val], "train": shuffled[n_val:]}

    for split, recs in splits.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for rec in tqdm(recs, desc=f"YOLO {split}", leave=False):
            src = Path(rec["image"])
            dst = img_dir / src.name
            lbl = lbl_dir / (src.stem + ".txt")
            if lbl.exists():
                continue

            try:
                img = cv2.imread(str(src))
                if img is None:
                    continue
            except Exception:
                continue
            oh, ow = img.shape[:2]
            x1, y1, x2, y2 = rec["bbox"]
            cx = ((x1 + x2) / 2) / ow
            cy = ((y1 + y2) / 2) / oh
            w  = (x2 - x1) / ow
            h  = (y2 - y1) / oh
            cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
            w,  h  = max(1e-4, min(1.0, w)), max(1e-4, min(1.0, h))
            lbl.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: 1\n"
        f"names: ['license_plate']\n"
    )
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# LPRNet
# ─────────────────────────────────────────────────────────────────────────────

def build_lprnet(onnx_path: Path, device: str) -> nn.Module:
    from lprnet_torch import LPRNetTorch, load_weights_from_onnx
    model = LPRNetTorch()
    load_weights_from_onnx(model, str(onnx_path))

    old = model.dense
    new = nn.Linear(old.in_features, NUM_CLASSES)
    nn.init.xavier_uniform_(new.weight)
    nn.init.zeros_(new.bias)
    model.dense = new
    n = sum(p.numel() for p in model.parameters())
    print(f"[lprnet] head: Linear({old.in_features},{old.out_features}) → Linear({old.in_features},{NUM_CLASSES})  |  {n:,} params")
    return model.to(device).train()


def train_lprnet(model: nn.Module, train_records: List[dict], val_records: List[dict], args, batch_size: int) -> None:
    device = args.device
    bs = batch_size

    # LPRNet input: [B, 3, 48, 96] RGB; model sums channels internally
    train_ds = OCRCropDataset(train_records, crop_hw=(48, 96), grayscale=False)
    val_ds   = OCRCropDataset(val_records,   crop_hw=(48, 96), grayscale=False)

    pm = pin_memory(device)
    kw = dict(num_workers=args.workers, pin_memory=pm)
    train_dl = DataLoader(train_ds, bs, shuffle=True,  drop_last=True,  **kw)
    val_dl   = DataLoader(val_ds,   bs, shuffle=False, **kw)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    best  = float("inf"); no_improve = 0

    def _lp(imgs):
        # forward() applies softmax; convert to log-probs for CTC
        out = model(imgs)                     # [B, T=24, NUM_CLASSES]
        return torch.log(out.clamp(min=1e-9)) # [B, T, C] log-probs

    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0; window = collections.deque(maxlen=100)
        with tqdm(train_dl, desc=f"LPRNet {ep}", leave=False) as pbar:
            for imgs, labels in pbar:
                try:
                    imgs = imgs.to(device)
                    lp   = _lp(imgs).permute(1, 0, 2)   # [T, B, C]
                    T, B, _ = lp.shape
                    ilen = torch.full((B,), T, dtype=torch.long)
                    tgt, tlen = encode_ctc(labels)
                    loss = F.ctc_loss(lp, tgt, ilen, tlen, blank=BLANK_IDX, zero_infinity=True)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step(); sched.step(); tl += loss.item()
                    window.append(loss.item())
                    pbar.set_postfix(loss=f"{sum(window)/len(window):.4f}",
                                     lr=f"{sched.get_last_lr()[0]:.2e}")
                except Exception:
                    print("\n[WARNING] Skipping batch due to error:")
                    traceback.print_exc()
                    opt.zero_grad()

        if args.epochs == 1:
            print(f"  LPRNet ep{ep}: train={tl/len(train_dl):.4f}  (val skipped, epochs=1)")
            out = Path(args.output_dir) / "lprnet_finetuned.pt"
            torch.save(model.state_dict(), out)
            print(f"    → {out}")
            continue
        model.eval(); vl = 0.0
        with torch.no_grad():
            for imgs, labels in val_dl:
                lp = _lp(imgs.to(device)).permute(1, 0, 2)
                T, B, _ = lp.shape
                ilen = torch.full((B,), T, dtype=torch.long)
                tgt, tlen = encode_ctc(labels)
                vl += F.ctc_loss(lp, tgt, ilen, tlen, blank=BLANK_IDX,
                                 zero_infinity=True).item()
        vl /= max(1, len(val_dl))
        print(f"  LPRNet ep{ep}: train={tl/len(train_dl):.4f}  val={vl:.4f}")
        if vl < best:
            best = vl; no_improve = 0
            out  = Path(args.output_dir) / "lprnet_finetuned.pt"
            torch.save(model.state_dict(), out)
            print(f"    → {out}")
        else:
            no_improve += 1
            if no_improve > 1:
                print(f"  Early stopping (patience=1) at ep{ep}")
                break


# ─────────────────────────────────────────────────────────────────────────────
# TrOCR  (no head replacement — existing tokenizer covers A-Z/0-9)
# ─────────────────────────────────────────────────────────────────────────────

def build_trocr(device: str):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
    model     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[trocr] Loaded microsoft/trocr-small-printed  |  {n:,} params")
    return model, processor


def train_trocr(model, processor, train_records: List[dict], val_records: List[dict], args, batch_size: int) -> None:
    device = args.device
    pad_id = processor.tokenizer.pad_token_id

    class _DS(Dataset):
        def __init__(self, recs):
            self.recs = recs
        def __len__(self):
            return len(self.recs)
        def __getitem__(self, i):
            rec = self.recs[i]
            x1, y1, x2, y2 = rec["bbox"]
            try:
                img = Image.open(rec["image"]).convert("RGB")
                img = img.crop((x1, y1, x2, y2)).resize((384, 384), Image.BILINEAR)
            except Exception:
                img = Image.new("RGB", (384, 384))
            return img, rec["label"]

    def _collate(batch):
        imgs, labels = zip(*batch)
        pv  = processor(images=list(imgs), return_tensors="pt").pixel_values
        tok = processor.tokenizer(list(labels), padding=True, truncation=True,
                                  max_length=16, return_tensors="pt")
        ids = tok.input_ids
        if pad_id is not None:
            ids = ids.masked_fill(ids == pad_id, -100)
        return pv, ids

    bs = batch_size

    kw = dict(collate_fn=_collate, num_workers=args.workers, pin_memory=pin_memory(device))
    train_dl = DataLoader(_DS(train_records), bs, shuffle=True,  **kw)
    val_dl   = DataLoader(_DS(val_records),   bs, shuffle=False, **kw)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    best  = float("inf"); no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0; window = collections.deque(maxlen=100)
        with tqdm(train_dl, desc=f"TrOCR {ep}", leave=False) as pbar:
            for pv, ids in pbar:
                try:
                    pv, ids = pv.to(device), ids.to(device)
                    loss = model(pixel_values=pv, labels=ids).loss
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step(); tl += loss.item()
                    window.append(loss.item())
                    pbar.set_postfix(loss=f"{sum(window)/len(window):.4f}",
                                     lr=f"{sched.get_last_lr()[0]:.2e}")
                except Exception:
                    print("\n[WARNING] Skipping batch due to error:")
                    traceback.print_exc()
                    opt.zero_grad()

        if args.epochs == 1:
            print(f"  TrOCR ep{ep}: train={tl/len(train_dl):.4f}  (val skipped, epochs=1)")
            out = Path(args.output_dir) / "trocr_small_finetuned.pt"
            torch.save(model.state_dict(), out)
            print(f"    → {out}")
            continue
        model.eval(); vl = 0.0
        with torch.no_grad():
            for pv, ids in val_dl:
                vl += model(pixel_values=pv.to(device), labels=ids.to(device)).loss.item()
        vl /= max(1, len(val_dl))
        print(f"  TrOCR ep{ep}: train={tl/len(train_dl):.4f}  val={vl:.4f}")
        if vl < best:
            best = vl; no_improve = 0
            out  = Path(args.output_dir) / "trocr_small_finetuned.pt"
            torch.save(model.state_dict(), out)
            print(f"    → {out}")
        else:
            no_improve += 1
            if no_improve > 1:
                print(f"  Early stopping (patience=1) at ep{ep}")
                break


# ─────────────────────────────────────────────────────────────────────────────
# doctr-vitstr
# ─────────────────────────────────────────────────────────────────────────────

def build_vitstr(device: str) -> nn.Module:
    from doctr.models import vitstr_small
    model = vitstr_small(pretrained=True)
    n = sum(p.numel() for p in model.parameters())
    print(f"[vitstr] pretrained vitstr_small  |  {n:,} params")
    return model.to(device).train()


def train_vitstr(model: nn.Module, train_records: List[dict], val_records: List[dict], args, batch_size: int) -> None:
    """
    Fine-tune vitstr using doctr's built-in training loss.
    No head replacement — doctr's vocab already covers A-Z and 0-9.
    Labels are lowercased to match doctr's expected format.
    """
    device = args.device
    tr = OCRCropDataset(train_records, crop_hw=(32, 128), grayscale=False)
    va = OCRCropDataset(val_records,   crop_hw=(32, 128), grayscale=False)

    def _col(batch):
        imgs, labels = zip(*batch)
        return torch.stack(imgs), list(labels)

    bs = batch_size

    kw = dict(collate_fn=_col, num_workers=args.workers, pin_memory=pin_memory(device))
    train_dl = DataLoader(tr, bs, shuffle=True,  drop_last=True, **kw)
    val_dl   = DataLoader(va, bs, shuffle=False, **kw)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    best  = float("inf"); no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0; n = 0; window = collections.deque(maxlen=100)
        with tqdm(train_dl, desc=f"ViTSTR {ep}", leave=False) as pbar:
            for imgs, labels in pbar:
                try:
                    imgs   = imgs.to(device)
                    target = [l.lower() for l in labels]
                    out    = model(imgs, target=target)
                    loss   = out["loss"]
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step(); sched.step(); tl += loss.item(); n += 1
                    window.append(loss.item())
                    pbar.set_postfix(loss=f"{sum(window)/len(window):.4f}",
                                     lr=f"{sched.get_last_lr()[0]:.2e}")
                except Exception:
                    print("\n[WARNING] Skipping batch due to error:")
                    traceback.print_exc()
                    opt.zero_grad()

        if args.epochs == 1:
            print(f"  ViTSTR ep{ep}: train={tl/max(1,n):.4f}  (val skipped, epochs=1)")
            out_path = Path(args.output_dir) / "vitstr_small_finetuned.pt"
            torch.save(model.state_dict(), out_path)
            print(f"    → {out_path}")
            continue
        model.eval(); vl = 0; m = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                out = model(imgs.to(device), target=[l.lower() for l in labels])
                vl += out["loss"].item(); m += 1
        vl /= max(1, m)
        print(f"  ViTSTR ep{ep}: train={tl/max(1,n):.4f}  val={vl:.4f}")
        if vl < best:
            best = vl; no_improve = 0
            out_path = Path(args.output_dir) / "vitstr_small_finetuned.pt"
            torch.save(model.state_dict(), out_path)
            print(f"    → {out_path}")
        else:
            no_improve += 1
            if no_improve > 1:
                print(f"  Early stopping (patience=1) at ep{ep}")
                break


# ─────────────────────────────────────────────────────────────────────────────
# RT-DETR  (HuggingFace transformers)
# ─────────────────────────────────────────────────────────────────────────────

def train_rtdetr(train_records: List[dict], val_records: List[dict], args, batch_size: int) -> None:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    model_id  = "justjuu/rtdetr-v2-license-plate-detection"
    device    = args.device
    processor = AutoImageProcessor.from_pretrained(model_id)
    model     = AutoModelForObjectDetection.from_pretrained(model_id)
    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[rtdetr] Loaded {model_id}  |  {n:,} params")

    class _DS(Dataset):
        def __init__(self, recs):
            self.recs = recs
        def __len__(self):
            return len(self.recs)
        def __getitem__(self, i):
            rec = self.recs[i]
            x1, y1, x2, y2 = rec["bbox"]
            try:
                img = Image.open(rec["image"]).convert("RGB")
                w, h = img.size
            except Exception:
                img = Image.new("RGB", (640, 480)); w, h = 640, 480
                x1, y1, x2, y2 = 0, 0, 100, 50
            ann = {"image_id": i, "annotations": [{"id": i, "image_id": i,
                   "category_id": 0, "bbox": [x1, y1, x2-x1, y2-y1],
                   "area": max(1, (x2-x1)*(y2-y1)), "iscrowd": 0}]}
            return img, ann

    def _col(batch):
        imgs, anns = zip(*batch)
        return processor(images=list(imgs), annotations=list(anns), return_tensors="pt")

    bs = batch_size

    train_dl = DataLoader(_DS(train_records), bs, shuffle=True,  collate_fn=_col,
                          num_workers=args.workers)
    val_dl   = DataLoader(_DS(val_records),   bs, shuffle=False, collate_fn=_col,
                          num_workers=args.workers)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    best  = float("inf"); no_improve = 0
    out_dir = Path(args.output_dir) / "rtdetr_finetuned"

    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0; window = collections.deque(maxlen=100)
        with tqdm(train_dl, desc=f"RT-DETR {ep}", leave=False) as pbar:
            for enc in pbar:
                try:
                    labels = enc.pop("labels", None)
                    enc    = {k: v.to(device) for k, v in enc.items()}
                    if labels is not None:
                        labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in labels]
                    out  = model(**enc, labels=labels)
                    loss = out.loss
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    opt.step(); sched.step(); tl += loss.item()
                    window.append(loss.item())
                    postfix: dict = {"loss": f"{sum(window)/len(window):.4f}",
                                     "lr": f"{sched.get_last_lr()[0]:.2e}"}
                    if hasattr(out, "loss_dict") and out.loss_dict:
                        ld = {k: f"{v.item():.4f}" for k, v in out.loss_dict.items()}
                        postfix.update(ld)
                    pbar.set_postfix(**postfix)
                except Exception:
                    print("\n[WARNING] Skipping batch due to error:")
                    traceback.print_exc()
                    opt.zero_grad()

        if args.epochs == 1:
            print(f"  RT-DETR ep{ep}: train={tl/max(1,len(train_dl)):.4f}  (val skipped, epochs=1)")
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir))
            processor.save_pretrained(str(out_dir))
            print(f"    → {out_dir}")
            continue
        model.eval(); vl = 0.0
        with torch.no_grad():
            for enc in val_dl:
                labels = enc.pop("labels", None)
                enc    = {k: v.to(device) for k, v in enc.items()}
                if labels is not None:
                    labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in labels]
                vl += model(**enc, labels=labels).loss.item()
        vl /= max(1, len(val_dl))
        print(f"  RT-DETR ep{ep}: train={tl/max(1,len(train_dl)):.4f}  val={vl:.4f}")
        if vl < best:
            best = vl; no_improve = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir))
            processor.save_pretrained(str(out_dir))
            print(f"    → {out_dir}")
        else:
            no_improve += 1
            if no_improve > 1:
                print(f"  Early stopping (patience=1) at ep{ep}")
                break


# ─────────────────────────────────────────────────────────────────────────────
# OWL-ViT  (CLIP backbone, open-vocab detector fine-tuned for LP detection)
# ─────────────────────────────────────────────────────────────────────────────

_OWLVIT_MODEL_ID = "google/owlvit-base-patch32"
_OWLVIT_QUERY    = ["a license plate"]


def train_owlvit(train_records: List[dict], val_records: List[dict], args, batch_size: int) -> None:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    device    = args.device
    processor = OwlViTProcessor.from_pretrained(_OWLVIT_MODEL_ID)
    model     = OwlViTForObjectDetection.from_pretrained(_OWLVIT_MODEL_ID).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"[owlvit] {_OWLVIT_MODEL_ID} — {n_params:,} params")

    # Pre-tokenise the fixed text query once.
    text_inputs = processor(text=_OWLVIT_QUERY, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    class _DS(Dataset):
        def __init__(self, recs): self.recs = recs
        def __len__(self):        return len(self.recs)
        def __getitem__(self, i):
            rec = self.recs[i]
            x1, y1, x2, y2 = rec["bbox"]
            try:
                img = Image.open(rec["image"]).convert("RGB")
                W, H = img.size
            except Exception:
                img = Image.new("RGB", (640, 480)); W, H = 640, 480
                x1, y1, x2, y2 = 0, 0, 100, 50
            # OWL-ViT expects normalized cx,cy,w,h
            cx = (x1 + x2) / 2 / W; cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W;     bh = (y2 - y1) / H
            return img, torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

    def _col(batch):
        imgs, boxes = zip(*batch)
        pv = processor(images=list(imgs), return_tensors="pt")["pixel_values"]
        return pv, torch.stack(boxes)

    bs = batch_size

    train_dl = DataLoader(_DS(train_records), bs, shuffle=True,  collate_fn=_col,
                          num_workers=args.workers)
    val_dl   = DataLoader(_DS(val_records),   bs, shuffle=False, collate_fn=_col,
                          num_workers=args.workers)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    best  = float("inf"); no_improve = 0
    out_dir = Path(args.output_dir) / "owlvit_finetuned"

    from torchvision.ops import box_iou

    def _owlvit_loss(pixel_values, gt_boxes):
        batch_text = {k: v.expand(pixel_values.shape[0], -1) for k, v in text_inputs.items()}
        out = model(pixel_values=pixel_values, **batch_text)
        pred_boxes  = out.pred_boxes
        pred_logits = out.logits.squeeze(-1)
        B, N, _ = pred_boxes.shape
        box_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        for b in range(B):
            pb  = pred_boxes[b]
            gb  = gt_boxes[b].unsqueeze(0)
            pb_xyxy = torch.stack([pb[:,0]-pb[:,2]/2, pb[:,1]-pb[:,3]/2,
                                   pb[:,0]+pb[:,2]/2, pb[:,1]+pb[:,3]/2], dim=1)
            gb_xyxy = torch.stack([gb[:,0]-gb[:,2]/2, gb[:,1]-gb[:,3]/2,
                                   gb[:,0]+gb[:,2]/2, gb[:,1]+gb[:,3]/2], dim=1)
            iou    = box_iou(pb_xyxy, gb_xyxy).squeeze(1)
            best_i = iou.argmax()
            box_loss += F.l1_loss(pb[best_i], gt_boxes[b])
            lbl = torch.zeros(N, device=device); lbl[best_i] = 1.0
            cls_loss += F.binary_cross_entropy_with_logits(
                pred_logits[b], lbl, pos_weight=torch.tensor(N - 1.0, device=device))
        return (box_loss + cls_loss) / B, box_loss, cls_loss

    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0; window = collections.deque(maxlen=100)
        with tqdm(train_dl, desc=f"OWL-ViT {ep}", leave=False) as pbar:
            for pixel_values, gt_boxes in pbar:
                try:
                    pixel_values = pixel_values.to(device)
                    gt_boxes     = gt_boxes.to(device)
                    loss, box_loss, cls_loss = _owlvit_loss(pixel_values, gt_boxes)
                    B = pixel_values.shape[0]
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    opt.step(); sched.step(); tl += loss.item()
                    window.append(loss.item())
                    pbar.set_postfix(
                        loss=f"{sum(window)/len(window):.4f}",
                        box=f"{box_loss.item()/B:.4f}",
                        cls=f"{cls_loss.item()/B:.4f}",
                        lr=f"{sched.get_last_lr()[0]:.2e}",
                    )
                except Exception:
                    print("\n[WARNING] Skipping batch due to error:")
                    traceback.print_exc()
                    opt.zero_grad()

        if args.epochs == 1:
            print(f"  OWL-ViT ep{ep}: train={tl/max(1,len(train_dl)):.4f}  (val skipped, epochs=1)")
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir))
            processor.save_pretrained(str(out_dir))
            print(f"    → {out_dir}")
            continue
        model.eval(); vl = 0.0
        with torch.no_grad():
            for pixel_values, gt_boxes in val_dl:
                pixel_values = pixel_values.to(device)
                gt_boxes     = gt_boxes.to(device)
                vl += _owlvit_loss(pixel_values, gt_boxes)[0].item()
        vl /= max(1, len(val_dl))
        print(f"  OWL-ViT ep{ep}: train={tl/max(1,len(train_dl)):.4f}  val={vl:.4f}")
        if vl < best:
            best = vl; no_improve = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir))
            processor.save_pretrained(str(out_dir))
            print(f"    → {out_dir}")
        else:
            no_improve += 1
            if no_improve > 1:
                print(f"  Early stopping (patience=1) at ep{ep}")
                break


# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN  (torchvision, COCO pretrained → 2-class LP head)
# ─────────────────────────────────────────────────────────────────────────────

def build_fasterrcnn(device: str) -> nn.Module:
    from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                               FasterRCNN_ResNet50_FPN_Weights)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes=2)
    n = sum(p.numel() for p in model.parameters())
    print(f"[fasterrcnn] COCO pretrained, head replaced for 2 classes  |  {n:,} params")
    return model.to(device).train()


def train_fasterrcnn(model: nn.Module, train_records: List[dict], val_records: List[dict], args, batch_size: int) -> None:
    device = args.device
    bs     = batch_size
    tr = DetectionDataset(train_records)
    va = DetectionDataset(val_records)

    kw = dict(collate_fn=_collate_det, num_workers=args.workers)
    train_dl = DataLoader(tr, bs, shuffle=True,  **kw)
    val_dl   = DataLoader(va, bs, shuffle=False, **kw)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    best  = float("inf"); no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0; window = collections.deque(maxlen=100)
        with tqdm(train_dl, desc=f"FasterRCNN {ep}", leave=False) as pbar:
            for imgs, targets in pbar:
                try:
                    imgs    = [i.to(device) for i in imgs]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    losses  = model(imgs, targets)
                    loss    = sum(losses.values())
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                    tl += loss.item()
                    window.append(loss.item())
                    pbar.set_postfix(
                        loss=f"{sum(window)/len(window):.4f}",
                        cls=f"{losses.get('loss_classifier', 0.):.4f}",
                        box=f"{losses.get('loss_box_reg', 0.):.4f}",
                        obj=f"{losses.get('loss_objectness', 0.):.4f}",
                        rpn=f"{losses.get('loss_rpn_box_reg', 0.):.4f}",
                        lr=f"{sched.get_last_lr()[0]:.2e}",
                    )
                except Exception:
                    print("\n[WARNING] Skipping batch due to error:")
                    traceback.print_exc()
                    opt.zero_grad()

        if args.epochs == 1:
            print(f"  FasterRCNN ep{ep}: train={tl/max(1,len(train_dl)):.4f}  (val skipped, epochs=1)")
            out = Path(args.output_dir) / "fasterrcnn_finetuned.pt"
            torch.save(model.state_dict(), out)
            print(f"    → {out}")
            continue
        model.eval(); vl = 0.0
        with torch.no_grad():
            for imgs, targets in val_dl:
                imgs    = [i.to(device) for i in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # Faster R-CNN only returns loss dict in train mode; switch temporarily
                model.train()
                vl += sum(model(imgs, targets).values()).item()
                model.eval()
        vl /= max(1, len(val_dl))
        print(f"  FasterRCNN ep{ep}: train={tl/max(1,len(train_dl)):.4f}  val={vl:.4f}")
        if vl < best:
            best = vl; no_improve = 0
            out  = Path(args.output_dir) / "fasterrcnn_finetuned.pt"
            torch.save(model.state_dict(), out)
            print(f"    → {out}")
        else:
            no_improve += 1
            if no_improve > 1:
                print(f"  Early stopping (patience=1) at ep{ep}")
                break


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks — one forward+backward pass per model before full training
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_mini_batch(records: List[dict], crop_hw, grayscale: bool, n: int = 4):
    """Return a (imgs, labels) micro-batch from the real dataset."""
    ds = OCRCropDataset(records[:max(n, 16)], crop_hw=crop_hw, grayscale=grayscale)
    dl = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0)
    return next(iter(dl))


def _det_mini_batch(records: List[dict], n: int = 2):
    ds = DetectionDataset(records[:max(n, 8)])
    dl = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0,
                    collate_fn=_collate_det)
    return next(iter(dl))


def _check(name: str, fn) -> bool:
    """Run fn(), print pass/fail, return True on success."""
    try:
        fn()
        print(f"  [OK]  {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def run_sanity_checks(args, records: List[dict],
                      weights_dir: Path, todo: set):
    """
    Attempts one forward+backward step for every model in `todo` and determines
    the optimal batch size for each.  Returns a dict mapping model name → batch
    size on success, or None if any check fails.
    """
    print("\n=== Sanity checks (1 batch + batch-size probe + time estimate each) ===")
    results  = {}   # name → True/False
    bsizes   = {}   # name → int
    etimes   = {}   # name → estimated total training seconds
    n_total  = len(records)

    def _record_estimate(name: str, probe_fn, bs: int, label: str) -> None:
        """Time probe_fn(bs), compute epoch/total estimates, store in etimes."""
        t_batch = time_probe(probe_fn, bs, args.device)
        n_train = int(n_total * 0.9)
        batches_ep = math.ceil(n_train / bs)
        est = args.epochs * batches_ep * t_batch
        etimes[name] = est
        print(f"  [{label}] bs={bs}  {t_batch:.2f}s/batch  "
              f"→ ~{batches_ep} batches/ep × {args.epochs} ep "
              f"= {fmt_duration(est)}")

    # ── LPRNet ────────────────────────────────────────────────────────────────
    if "lprnet" in todo:
        p = weights_dir / "lprnet_deployable_onnx_v1.1" / "us_lprnet_baseline18_deployable.onnx"
        if not p.exists():
            print(f"  [SKIP] lprnet: {p} not found")
        else:
            m_lprnet = build_lprnet(p, args.device)
            def _lprnet():
                imgs, labels = _ocr_mini_batch(records, (48, 96), grayscale=False)
                imgs = imgs.to(args.device)
                lp = torch.log(m_lprnet(imgs).clamp(min=1e-9)).permute(1, 0, 2)
                T, B, _ = lp.shape
                ilen = torch.full((B,), T, dtype=torch.long)
                tgt, tlen = encode_ctc(labels)
                F.ctc_loss(lp, tgt, ilen, tlen, blank=BLANK_IDX,
                           zero_infinity=True).backward()
            ok = _check("lprnet", _lprnet)
            results["lprnet"] = ok
            if ok:
                def _lprnet_probe(bs):
                    x = torch.randn(bs, 3, 48, 96, device=args.device)
                    lp = torch.log(m_lprnet(x).clamp(min=1e-9)).permute(1, 0, 2)
                    T, B2, _ = lp.shape
                    ilen = torch.full((B2,), T, dtype=torch.long)
                    tgt  = torch.ones(B2 * 4, dtype=torch.long)
                    tlen = torch.full((B2,), 4, dtype=torch.long)
                    F.ctc_loss(lp, tgt, ilen, tlen, blank=BLANK_IDX, zero_infinity=True).backward()
                bsizes["lprnet"] = find_batch_size(_lprnet_probe, args.device, max_bs=args.batch_size)
                _record_estimate("lprnet", _lprnet_probe, bsizes["lprnet"], "lprnet")

    # ── TrOCR ─────────────────────────────────────────────────────────────────
    if "trocr" in todo:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        m_trocr_proc  = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        m_trocr = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-small-printed").to(args.device)
        m_trocr.config.decoder_start_token_id = m_trocr_proc.tokenizer.bos_token_id
        m_trocr.config.pad_token_id           = m_trocr_proc.tokenizer.pad_token_id
        n = sum(p.numel() for p in m_trocr.parameters())
        print(f"  [trocr] {n:,} params")
        def _trocr():
            recs   = records[:4]
            imgs   = []
            labels = []
            for rec in recs:
                x1, y1, x2, y2 = rec["bbox"]
                try:
                    img = Image.open(rec["image"]).convert("RGB")
                    img = img.crop((x1, y1, x2, y2)).resize((384, 384), Image.BILINEAR)
                except Exception:
                    img = Image.new("RGB", (384, 384))
                imgs.append(img)
                labels.append(rec["label"])
            pv  = m_trocr_proc(images=imgs, return_tensors="pt").pixel_values.to(args.device)
            tok = m_trocr_proc.tokenizer(labels, padding=True, truncation=True,
                                         max_length=16, return_tensors="pt")
            ids = tok.input_ids
            pad_id = m_trocr_proc.tokenizer.pad_token_id
            if pad_id is not None:
                ids = ids.masked_fill(ids == pad_id, -100)
            m_trocr(pixel_values=pv, labels=ids.to(args.device)).loss.backward()
        ok = _check("trocr", _trocr)
        results["trocr"] = ok
        if ok:
            def _trocr_probe(bs):
                pv  = torch.randn(bs, 3, 384, 384, device=args.device)
                lbl = torch.ones(bs, 8, dtype=torch.long, device=args.device)
                m_trocr(pixel_values=pv, labels=lbl).loss.backward()
            bsizes["trocr"] = find_batch_size(_trocr_probe, args.device, max_bs=args.batch_size)
            _record_estimate("trocr", _trocr_probe, bsizes["trocr"], "trocr")

    # ── doctr-vitstr ──────────────────────────────────────────────────────────
    if "vitstr" in todo:
        m_vitstr = build_vitstr(args.device)
        def _vitstr():
            imgs, labels = _ocr_mini_batch(records, (32, 128), grayscale=False)
            imgs   = imgs.to(args.device)
            target = [l.lower() for l in labels]
            m_vitstr(imgs, target=target)["loss"].backward()
        ok = _check("vitstr", _vitstr)
        results["vitstr"] = ok
        if ok:
            def _vitstr_probe(bs):
                x = torch.randn(bs, 3, 32, 128, device=args.device)
                m_vitstr(x, target=["abc"] * bs)["loss"].backward()
            bsizes["vitstr"] = find_batch_size(_vitstr_probe, args.device, max_bs=args.batch_size)
            _record_estimate("vitstr", _vitstr_probe, bsizes["vitstr"], "vitstr")

    # ── OWL-ViT ───────────────────────────────────────────────────────────────
    if "owlvit" in todo:
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        from torchvision.ops import box_iou
        m_owlvit_proc = OwlViTProcessor.from_pretrained(_OWLVIT_MODEL_ID)
        m_owlvit = OwlViTForObjectDetection.from_pretrained(_OWLVIT_MODEL_ID).to(args.device)
        n = sum(p.numel() for p in m_owlvit.parameters())
        print(f"  [owlvit] {n:,} params")
        m_owlvit.train()
        _owlvit_text = {k: v.expand(1, -1).to(args.device)
                        for k, v in m_owlvit_proc(text=_OWLVIT_QUERY, return_tensors="pt",
                                                   padding=True).items()}
        def _owlvit():
            recs  = records[:2]
            imgs_pil = []
            gt_boxes = []
            for rec in recs:
                x1, y1, x2, y2 = rec["bbox"]
                try:
                    img = Image.open(rec["image"]).convert("RGB")
                    W, H = img.size
                except Exception:
                    img = Image.new("RGB", (640, 480)); W, H = 640, 480
                    x1, y1, x2, y2 = 0, 0, 100, 50
                imgs_pil.append(img)
                cx = (x1+x2)/2/W; cy = (y1+y2)/2/H
                bw = (x2-x1)/W;   bh = (y2-y1)/H
                gt_boxes.append(torch.tensor([cx, cy, bw, bh]))
            pv   = m_owlvit_proc(images=imgs_pil, return_tensors="pt")["pixel_values"].to(args.device)
            text = {k: v.expand(pv.shape[0], -1) for k, v in _owlvit_text.items()}
            out  = m_owlvit(pixel_values=pv, **text)
            pred_boxes  = out.pred_boxes
            pred_logits = out.logits.squeeze(-1)
            B, N, _ = pred_boxes.shape
            gt = torch.stack(gt_boxes).to(args.device)
            loss = torch.tensor(0.0, device=args.device)
            for b in range(B):
                pb = pred_boxes[b]
                pb_xyxy = torch.stack([pb[:,0]-pb[:,2]/2, pb[:,1]-pb[:,3]/2,
                                       pb[:,0]+pb[:,2]/2, pb[:,1]+pb[:,3]/2], dim=1)
                gb = gt[b].unsqueeze(0)
                gb_xyxy = torch.stack([gb[:,0]-gb[:,2]/2, gb[:,1]-gb[:,3]/2,
                                       gb[:,0]+gb[:,2]/2, gb[:,1]+gb[:,3]/2], dim=1)
                best_i = box_iou(pb_xyxy, gb_xyxy).squeeze(1).argmax()
                lbl = torch.zeros(N, device=args.device); lbl[best_i] = 1.0
                loss += F.l1_loss(pb[best_i], gt[b])
                loss += F.binary_cross_entropy_with_logits(pred_logits[b], lbl)
            (loss / B).backward()
        ok = _check("owlvit", _owlvit)
        results["owlvit"] = ok
        if ok:
            def _owlvit_probe(bs):
                pv   = torch.randn(bs, 3, 768, 768, device=args.device)
                txt  = {k: v.expand(bs, -1) for k, v in _owlvit_text.items()}
                out  = m_owlvit(pixel_values=pv, **txt)
                pred_boxes  = out.pred_boxes
                pred_logits = out.logits.squeeze(-1)
                N = pred_boxes.shape[1]
                gt = torch.tensor([0.5, 0.5, 0.3, 0.1], device=args.device).unsqueeze(0).expand(bs, -1)
                loss = torch.tensor(0.0, device=args.device)
                for b in range(bs):
                    pb = pred_boxes[b]
                    pb_x = torch.stack([pb[:,0]-pb[:,2]/2, pb[:,1]-pb[:,3]/2,
                                        pb[:,0]+pb[:,2]/2, pb[:,1]+pb[:,3]/2], dim=1)
                    gb   = gt[b:b+1]
                    gb_x = torch.stack([gb[:,0]-gb[:,2]/2, gb[:,1]-gb[:,3]/2,
                                        gb[:,0]+gb[:,2]/2, gb[:,1]+gb[:,3]/2], dim=1)
                    best_i = box_iou(pb_x, gb_x).squeeze(1).argmax()
                    lbl = torch.zeros(N, device=args.device); lbl[best_i] = 1.0
                    loss += F.l1_loss(pb[best_i], gt[b])
                    loss += F.binary_cross_entropy_with_logits(pred_logits[b], lbl)
                (loss / bs).backward()
            bsizes["owlvit"] = find_batch_size(_owlvit_probe, args.device, max_bs=args.batch_size)
            _record_estimate("owlvit", _owlvit_probe, bsizes["owlvit"], "owlvit")

    # ── RT-DETR ───────────────────────────────────────────────────────────────
    if "rtdetr" in todo:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        _rtdetr_model_id = "justjuu/rtdetr-v2-license-plate-detection"
        m_rtdetr_proc = AutoImageProcessor.from_pretrained(_rtdetr_model_id)
        m_rtdetr = AutoModelForObjectDetection.from_pretrained(_rtdetr_model_id).to(args.device)
        n = sum(p.numel() for p in m_rtdetr.parameters())
        print(f"  [rtdetr] {n:,} params")
        def _rtdetr():
            recs  = records[:2]
            imgs_pil, anns = [], []
            for i, rec in enumerate(recs):
                x1, y1, x2, y2 = rec["bbox"]
                try:
                    img = Image.open(rec["image"]).convert("RGB")
                    w, h = img.size
                except Exception:
                    img = Image.new("RGB", (640, 480)); w, h = 640, 480
                    x1, y1, x2, y2 = 0, 0, 100, 50
                imgs_pil.append(img)
                anns.append({"image_id": i, "annotations": [
                    {"id": i, "image_id": i, "category_id": 0,
                     "bbox": [x1, y1, x2-x1, y2-y1],
                     "area": max(1,(x2-x1)*(y2-y1)), "iscrowd": 0}]})
            enc    = m_rtdetr_proc(images=imgs_pil, annotations=anns, return_tensors="pt")
            labels = enc.pop("labels")
            enc    = {k: v.to(args.device) for k, v in enc.items()}
            labels = [{k: v.to(args.device) for k, v in lbl.items()} for lbl in labels]
            m_rtdetr(**enc, labels=labels).loss.backward()
        ok = _check("rtdetr", _rtdetr)
        results["rtdetr"] = ok
        if ok:
            def _rtdetr_probe(bs):
                pv  = torch.randn(bs, 3, 640, 640, device=args.device)
                lbl = [{"class_labels": torch.zeros(1, dtype=torch.long, device=args.device),
                        "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.2]], device=args.device)}
                       for _ in range(bs)]
                m_rtdetr(pixel_values=pv, labels=lbl).loss.backward()
            bsizes["rtdetr"] = find_batch_size(_rtdetr_probe, args.device, max_bs=args.batch_size)
            _record_estimate("rtdetr", _rtdetr_probe, bsizes["rtdetr"], "rtdetr")

    # ── Faster R-CNN ──────────────────────────────────────────────────────────
    if "fasterrcnn" in todo:
        m_fasterrcnn = build_fasterrcnn(args.device)
        def _frcnn():
            m_fasterrcnn.train()
            imgs, targets = _det_mini_batch(records, n=2)
            imgs    = [i.to(args.device) for i in imgs]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            sum(m_fasterrcnn(imgs, targets).values()).backward()
        ok = _check("fasterrcnn", _frcnn)
        results["fasterrcnn"] = ok
        if ok:
            def _frcnn_probe(bs):
                imgs    = [torch.randn(3, 640, 640, device=args.device) for _ in range(bs)]
                targets = [{"boxes":  torch.tensor([[10., 10., 200., 100.]], device=args.device),
                            "labels": torch.zeros(1, dtype=torch.long, device=args.device)}
                           for _ in range(bs)]
                sum(m_fasterrcnn(imgs, targets).values()).backward()
            bsizes["fasterrcnn"] = find_batch_size(_frcnn_probe, args.device, max_bs=args.batch_size)
            _record_estimate("fasterrcnn", _frcnn_probe, bsizes["fasterrcnn"], "fasterrcnn")

    # ── Summary ───────────────────────────────────────────────────────────────
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n  ✗ Sanity check FAILED for: {failed}")
        print("  Fix the issues above before running full training.")
        return None

    total_est = sum(etimes.values())
    print(f"\n  Per-model estimates:")
    for name, secs in etimes.items():
        print(f"    {name:12s}  {fmt_duration(secs)}")
    print(f"  Total estimated training time: {fmt_duration(total_est)}")

    limit = 72 * 3600
    if total_est > limit:
        print(f"\n  ✗ Estimated time ({fmt_duration(total_est)}) exceeds 72h limit.")
        print("  Reduce --epochs, use --models to select a subset, or use --batch-size to raise throughput.")
        return None

    print(f"  ✓ All {len(results)} checks passed.  Batch sizes: {bsizes}\n")
    return bsizes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODELS = ["lprnet", "trocr", "vitstr",
              "owlvit", "rtdetr", "fasterrcnn"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune all LP models on CCPD2019 + CCPD2019_TEXAS")
    p.add_argument("--data-root",   required=True,  type=Path,
                   help="Dir containing CCPD2019/ and (optionally) CCPD2019_TEXAS/")
    p.add_argument("--output-dir",  default="weights/finetuned")
    p.add_argument("--weights-dir", default="weights",
                   help="Dir with original model files (default: weights/)")
    p.add_argument("--models",      nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS, metavar="MODEL",
                   help=f"Models to train (default: all). Choices: {ALL_MODELS}")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=None,
                   help="Hard cap on batch size (default: no cap, use GPU memory optimally)")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit",       type=int,   default=None,
                   help="Max records per dataset (for quick smoke-tests)")
    return p.parse_args()


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    _seed_everything(42)
    data_root   = Path(args.data_root)
    weights_dir = Path(args.weights_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Ensure repo root is importable (for ocr_backends, lprnet_torch, etc.)
    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # ── Load records ──────────────────────────────────────────────────────────
    print("Loading CCPD2019...")
    ccpd = load_ccpd_records(data_root / "CCPD2019", limit=args.limit)
    print(f"  {len(ccpd):,} records")

    print("Loading CCPD2019_TEXAS...")
    texas = load_texas_records(data_root / "CCPD2019_TEXAS", limit=args.limit)
    print(f"  {len(texas):,} records")

    all_records = ccpd + texas
    print(f"  Total: {len(all_records):,} records\n")
    if not all_records:
        sys.exit("ERROR: no records found — check --data-root")

    # ── Single shared 90/10 split (deterministic) ─────────────────────────────
    rng = random.Random(42)
    shuffled = list(all_records)
    rng.shuffle(shuffled)
    n_val         = max(1, int(len(shuffled) * 0.1))
    val_records   = shuffled[:n_val]
    train_records = shuffled[n_val:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_split(path: Path, records: list) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "x1", "y1", "x2", "y2", "label"])
            for r in records:
                x1, y1, x2, y2 = r["bbox"]
                w.writerow([str(r["image"]), x1, y1, x2, y2, r["label"]])

    val_csv   = out_dir / "val_split.csv"
    train_csv = out_dir / "train_split.csv"
    _write_split(val_csv,   val_records)
    _write_split(train_csv, train_records)
    print(f"Val split:   {len(val_records):,} records → {val_csv}")
    print(f"Train split: {len(train_records):,} records → {train_csv}\n")

    todo = set(args.models)

    # ── Sanity checks + batch-size probe ──────────────────────────────────────
    checked = run_sanity_checks(args, train_records, weights_dir, todo)
    if checked is None:
        sys.exit(1)

    # ── Detectors (longest first) ─────────────────────────────────────────────
    if "fasterrcnn" in checked:
        print("\n=== Faster R-CNN ===")
        train_fasterrcnn(build_fasterrcnn(args.device), train_records, val_records, args,
                         batch_size=checked["fasterrcnn"])

    if "owlvit" in checked:
        print("\n=== OWL-ViT ===")
        train_owlvit(train_records, val_records, args, batch_size=checked["owlvit"])

    if "rtdetr" in checked:
        print("\n=== RT-DETR ===")
        train_rtdetr(train_records, val_records, args, batch_size=checked["rtdetr"])

    # ── OCR ───────────────────────────────────────────────────────────────────
    if "lprnet" in checked:
        print("\n=== LPRNet ===")
        p = weights_dir / "lprnet_deployable_onnx_v1.1" / "us_lprnet_baseline18_deployable.onnx"
        train_lprnet(build_lprnet(p, args.device), train_records, val_records, args,
                     batch_size=checked["lprnet"])

    if "trocr" in checked:
        print("\n=== TrOCR ===")
        model, proc = build_trocr(args.device)
        train_trocr(model, proc, train_records, val_records, args, batch_size=checked["trocr"])

    if "vitstr" in checked:
        print("\n=== doctr-vitstr ===")
        train_vitstr(build_vitstr(args.device), train_records, val_records, args,
                     batch_size=checked["vitstr"])

    print("\nAll done.")


if __name__ == "__main__":
    main()
