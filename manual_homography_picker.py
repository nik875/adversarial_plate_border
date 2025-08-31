#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch HEIC Homography Picker with Plate-Aware Cropping → CSV

- Scans a directory for images (HEIC/HEIF/JPG/PNG).
- Runs fast_alpr to detect license plates.
- If --target-plate is provided and found, crops to that plate; otherwise uses the highest-confidence plate.
- Applies 15% padding of the chosen bbox width/height on all sides (clamped to image bounds).
- Opens a small, efficient corner picker over the CROPPED region (smaller markers/lines).
- Maps clicked points back to ORIGINAL image coordinates (accounts for crop offset + scaling).
- Computes homography to a (out_w x out_h) canonical rectangle and appends to CSV.

Controls:
  - Left click: add a point (max 4)
  - 'u' or Backspace: undo last point
  - Enter or Space: SUBMIT (save to CSV and advance)
  - 'q' or ESC: quit

CSV columns:
  filename,
  p1_x,p1_y,p2_x,p2_y,p3_x,p3_y,p4_x,p4_y,    # ordered TL,TR,BR,BL (original image space)
  H00,H01,H02,H10,H11,H12,H20,H21,H22,        # rect->image homography
  out_w,out_h

Requires:
  pip install opencv-python numpy pillow pillow-heif fast-alpr
"""

import os
import cv2
import csv
import math
import argparse
import numpy as np
from dataclasses import dataclass

from PIL import Image

# HEIC support (Pillow opener)
HEIC_OK = False
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

# fast_alpr
try:
    from fast_alpr import ALPR
except ImportError as e:
    raise SystemExit("fast_alpr is required. Install with: pip install fast-alpr") from e


# --------------------- Image IO ---------------------

def load_image_any(path):
    """Load with Pillow (handles HEIC if opener registered), return RGB np.array and BGR np.array."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".heic", ".heif") and not HEIC_OK:
        raise RuntimeError("HEIC file but pillow-heif isn't installed. pip install pillow-heif")

    pil_img = Image.open(path).convert("RGB")
    rgb = np.array(pil_img)  # HxWx3 RGB
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb, bgr


# --------------------- ALPR Helpers ---------------------

@dataclass
class PlateBox:
    text: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int


def detect_plates_bgr(alpr, bgr_img):
    """Return list[PlateBox] using fast_alpr on BGR image."""
    preds = alpr.predict(bgr_img)
    results = []
    for p in preds or []:
        bb = p.detection.bounding_box
        ocr = p.ocr
        results.append(PlateBox(
            text=(ocr.text or "").strip().upper(),
            conf=float(ocr.confidence),
            x1=int(bb.x1), y1=int(bb.y1), x2=int(bb.x2), y2=int(bb.y2)
        ))
    return results


def choose_plate(plates, target_plate=None):
    """Pick plate matching target (case-insensitive) else highest confidence."""
    if not plates:
        return None
    if target_plate:
        tp = target_plate.strip().upper()
        matches = [pl for pl in plates if pl.text == tp]
        if matches:
            # choose highest confidence among matches
            return max(matches, key=lambda p: p.conf)
    return None


def crop_with_padding(bgr_img, box: PlateBox, pad_frac=0.15):
    """Crop around bbox with symmetric padding as a fraction of bbox size; clamp to image bounds.
    Returns: cropped_bgr, (x_off, y_off)
    """
    H, W = bgr_img.shape[:2]
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    px = int(round(pad_frac * w))
    py = int(round(pad_frac * h))

    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(W, x2 + px)
    cy2 = min(H, y2 + py)

    cropped = bgr_img[cy1:cy2, cx1:cx2].copy()
    return cropped, (cx1, cy1)


# --------------------- Geometry ---------------------

def order_corners_clockwise(pts):
    """Given 4 (x,y), order clockwise starting TL-ish (min x+y)."""
    if len(pts) != 4:
        return pts
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    with_ang = [(x, y, math.atan2(y - cy, x - cx)) for (x, y) in pts]
    with_ang.sort(key=lambda t: t[2])
    idx = min(range(4), key=lambda i: with_ang[i][0] + with_ang[i][1])
    rot = with_ang[idx:] + with_ang[:idx]
    return [(x, y) for (x, y, _) in rot]


def compute_homography(ordered_pts, out_w, out_h):
    """ordered_pts in image coords (TL,TR,BR,BL). Return H (rect->image), H_inv."""
    if len(ordered_pts) != 4:
        return None, None
    src = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    dst = np.float32(ordered_pts)
    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    return H, H_inv


# --------------------- Corner Picker GUI ---------------------

class CornerPicker:
    def __init__(self, img_bgr, window_name="Pick corners", origin_offset=(0, 0)):
        """
        img_bgr: cropped image shown to the user
        origin_offset: (x_off, y_off) where this crop starts in the ORIGINAL image
        """
        self.img_bgr = img_bgr
        self.h, self.w = img_bgr.shape[:2]
        self.window = window_name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.origin_offset = origin_offset  # add to map back into original

        self.display_bgr, self.scale = self._fit_to_screen(self.img_bgr)
        self.points_disp = []

        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _fit_to_screen(self, img):
        # modest bounding so UI is snappy
        max_w, max_h = 1200, 800
        h, w = img.shape[:2]
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        if scale < 1.0:
            disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            disp = img.copy()
        return disp, (scale if scale > 0 else 1.0)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points_disp) < 4:
                self.points_disp.append((x, y))

    def undo(self):
        if self.points_disp:
            self.points_disp.pop()

    def draw_overlay(self, filename=None, index=None, total=None):
        c = self.display_bgr.copy()
        # Smaller UI text and thinner lines
        header = "Click 4 corners. [u]/Backspace=Undo  [Enter/Space]=Submit  [q]/ESC=Quit"
        cv2.putText(c, header, (10, 22), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if filename is not None:
            line2 = f"{index}/{total}  {os.path.basename(filename)}"
            cv2.putText(c, line2, (10, 44), self.font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # draw points and helper polyline (smaller markers/lines)
        for i, (x, y) in enumerate(self.points_disp):
            cv2.circle(c, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)  # radius 3
            cv2.putText(c, str(i + 1), (x + 6, y - 6), self.font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        if len(self.points_disp) >= 2:
            cv2.polylines(c, [np.array(self.points_disp, dtype=np.int32)],
                          False, (255, 0, 255), 1, cv2.LINE_AA)
        if len(self.points_disp) == 4:
            cv2.polylines(
                c,
                [np.array(self.points_disp + [self.points_disp[0]], dtype=np.int32)],
                False,
                (0, 128, 255),
                1,
                cv2.LINE_AA,
            )
        return c

    def get_points_in_original(self):
        """Map display->crop->original coords."""
        inv = 1.0 / self.scale
        x_off, y_off = self.origin_offset
        pts = []
        for (px, py) in self.points_disp:
            cx = int(round(px * inv))
            cy = int(round(py * inv))
            ox = cx + x_off
            oy = cy + y_off
            pts.append((ox, oy))
        return pts


# --------------------- Batch Runner ---------------------

def list_images(root_dir):
    exts = (".heic", ".heif", ".png")
    return [os.path.join(root_dir, n)
            for n in sorted(os.listdir(root_dir)) if n.lower().endswith(exts)]


def ensure_parent(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description="Plate-aware homography picker over cropped plate regions.")
    ap.add_argument("--dir", required=True, help="Directory of images (HEIC/HEIF/JPG/PNG)")
    ap.add_argument("--csv", required=True, help="Output CSV path")
    ap.add_argument("--out-w", type=int, default=600, help="Output rect width (for H computation)")
    ap.add_argument("--out-h", type=int, default=400, help="Output rect height (for H computation)")
    ap.add_argument("--target-plate", type=str, default=None,
                    help="Preferred plate text to crop to (e.g., VRJ7774)")
    ap.add_argument("--pad-frac", type=float, default=0.15,
                    help="Padding as fraction of bbox size (default 0.15)")
    ap.add_argument("--save-crops", type=str, default=None,
                    help="Optional directory to save the cropped images")
    args = ap.parse_args()

    images = list_images(args.dir)
    if not images:
        raise SystemExit(f"No supported images found in: {args.dir}")

    # Prepare CSV
    ensure_parent(args.csv)
    write_header = not os.path.exists(args.csv)

    # Initialize ALPR once
    print("Initializing ALPR...")
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v1-global-model",
    )
    print("✓ ALPR ready")

    # Prepare crops dir
#    if args.save - crops:
#        os.makedirs(args.save - crops, exist_ok=True)

    with open(args.csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = [
                "filename",
                "p1_x", "p1_y", "p2_x", "p2_y", "p3_x", "p3_y", "p4_x", "p4_y",
                "H00", "H01", "H02", "H10", "H11", "H12", "H20", "H21", "H22",
                "out_w", "out_h",
            ]
            writer.writerow(header)

        total = len(images)
        for idx, path in enumerate(images, start=1):
            base = os.path.basename(path)
            print(f"\n[{idx}/{total}] {base}")

            try:
                rgb, bgr = load_image_any(path)
            except Exception as e:
                print(f"Skipping (load error): {e}")
                continue

            # Detect plates
            plates = detect_plates_bgr(alpr, bgr)
            if not plates:
                print("No plates detected; skipping.")
                continue
            chosen = choose_plate(plates, target_plate=args.target_plate)
            if not chosen:
                print("Target plate not detected, skipping.")
                continue
            print(f"Using plate: {chosen.text} (conf={chosen.conf:.3f})")
            crop_bgr, offset = crop_with_padding(bgr, chosen, pad_frac=float(args.pad_frac))

            # optional save crop
#            if args.save - crops:
#                crop_name = os.path.splitext(base)[0] + "_crop.png"
#                cv2.imwrite(os.path.join(args.save - crops, crop_name), crop_bgr)

            # Corner picking on the CROP (small markers/lines)
            picker = CornerPicker(
                crop_bgr,
                window_name="Homography Picker (Cropped)",
                origin_offset=offset)

            while True:
                canvas = picker.draw_overlay(filename=path, index=idx, total=total)
                cv2.imshow(picker.window, canvas)
                key = cv2.waitKey(20) & 0xFF

                if key in (27, ord('q')):  # ESC or q
                    cv2.destroyAllWindows()
                    print("Aborted by user.")
                    return
                elif key in (ord('u'), 8):  # 'u' or Backspace
                    picker.undo()
                elif key in (13, 32):  # Enter or Space → SUBMIT
                    pts_img = picker.get_points_in_original()
                    if len(pts_img) != 4:
                        print("Need exactly 4 points before submit.")
                        continue

                    ordered = order_corners_clockwise(pts_img)
                    H, _ = compute_homography(ordered, args.out_w, args.out_h)
                    if H is None:
                        print("Failed to compute homography.")
                        continue

                    flat_pts = [v for p in ordered for v in p]
                    flat_H = [float(x) for x in np.array(H).reshape(-1).tolist()]
                    row = [os.path.abspath(path)] + flat_pts + flat_H + [args.out_w, args.out_h]
                    writer.writerow(row)
                    f.flush()
                    print(f"✓ Saved row for {base}")
                    cv2.destroyWindow(picker.window)
                    break

    cv2.destroyAllWindows()
    print(f"\nDone. CSV written to: {args.csv}")


if __name__ == "__main__":
    main()
