#!/usr/bin/env python3
"""
Create a Texas-style synthetic license plate dataset from CCPD.

For each CCPD image, this script:
1. Parses the original plate annotations embedded in the filename.
2. Generates a plausible Texas-style plate string, or uses a fixed one if provided.
3. Renders a Texas-style plate image.
4. Warps the rendered plate into the CCPD plate quadrilateral.
5. Writes the composited image to a new dataset directory.
6. Writes metadata (original bbox + generated string) to JSONL and CSV.

Example (quick test on 50 images):
    python ccpd_to_texas_dataset.py \
        --input /path/CCPD2019/subset \
        --output /path/output_test \
        --limit 50

Example (all images recursively):
    python ccpd_to_texas_dataset.py \
        --input /path/CCPD2019/subset \
        --output /path/output_texas \
        --recursive

Example (all images with a fixed plate string):
    python ccpd_to_texas_dataset.py \
        --input /path/CCPD2019/subset \
        --output /path/output_texas \
        --recursive \
        --fixed-plate ABC1234
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Ambiguous letters often omitted in plate-like synthetic data.
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
DIGITS = "0123456789"

# Default: a plausible Texas-style 7-character passenger plate string.
# Rendered display inserts a dash after 3 characters: ABC-1234.
DEFAULT_PATTERN = "LLLDDDD"


@dataclass
class CCPDAnnotation:
    area_ratio: str
    tilt: str
    bbox_tl: Tuple[int, int]
    bbox_br: Tuple[int, int]
    vertices_raw: List[Tuple[int, int]]
    plate_code: str
    brightness: str
    blurriness: str
    stem: str
    suffix: str

    @property
    def bbox_xyxy(self) -> Tuple[int, int, int, int]:
        x1, y1 = self.bbox_tl
        x2, y2 = self.bbox_br
        return x1, y1, x2, y2


class PlateGenerationError(RuntimeError):
    pass


# -----------------------------
# CCPD parsing
# -----------------------------

def parse_point(text: str) -> Tuple[int, int]:
    x_str, y_str = text.split("&")
    return int(x_str), int(y_str)


def parse_ccpd_filename(path: Path) -> CCPDAnnotation:
    """
    CCPD filename format (7 fields separated by '-'):
      area-tilt-bbox-vertices-plate-brightness-blurriness.jpg

    Example:
      025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    """
    stem = path.stem
    parts = stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Unexpected CCPD filename format: {path.name}")

    area_ratio = parts[0]
    tilt = parts[1]

    bbox_left_top_str, bbox_right_bottom_str = parts[2].split("_")
    bbox_tl = parse_point(bbox_left_top_str)
    bbox_br = parse_point(bbox_right_bottom_str)

    vertices_raw = [parse_point(p) for p in parts[3].split("_")]
    if len(vertices_raw) != 4:
        raise ValueError(f"Expected 4 plate vertices in {path.name}, got {len(vertices_raw)}")

    plate_code = parts[4]
    brightness = parts[5]
    blurriness = parts[6]

    return CCPDAnnotation(
        area_ratio=area_ratio,
        tilt=tilt,
        bbox_tl=bbox_tl,
        bbox_br=bbox_br,
        vertices_raw=vertices_raw,
        plate_code=plate_code,
        brightness=brightness,
        blurriness=blurriness,
        stem=stem,
        suffix=path.suffix,
    )


# -----------------------------
# Geometry helpers
# -----------------------------

def order_points_clockwise(points: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected shape (4,2), got {pts.shape}")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return ordered


def quadrilateral_size(pts_tl_tr_br_bl: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = pts_tl_tr_br_bl
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    width = max(1, int(round(max(width_top, width_bottom))))
    height = max(1, int(round(max(height_left, height_right))))
    return width, height


# -----------------------------
# Texas-style plate string generation
# -----------------------------

def generate_plate_string(rng: random.Random, pattern: str = DEFAULT_PATTERN) -> str:
    chars: List[str] = []
    for token in pattern:
        if token == "L":
            chars.append(rng.choice(LETTERS))
        elif token == "D":
            chars.append(rng.choice(DIGITS))
        else:
            raise ValueError(f"Unsupported pattern token: {token!r}")
    return "".join(chars)


def display_plate_string(raw_string: str) -> str:
    # Display as ABC-1234 when length is 7.
    if len(raw_string) == 7:
        return raw_string[:3] + "-" + raw_string[3:]
    return raw_string


def validate_plate_string(raw_string: str, pattern: str = DEFAULT_PATTERN) -> str:
    raw_string = raw_string.strip().upper().replace("-", "").replace(" ", "")
    if len(raw_string) != len(pattern):
        raise ValueError(
            f"Fixed plate string length {len(raw_string)} does not match pattern length {len(pattern)}"
        )

    for idx, (char, token) in enumerate(zip(raw_string, pattern), start=1):
        if token == "L" and char not in LETTERS:
            raise ValueError(
                f"Character {idx} in fixed plate must be one of allowed letters {LETTERS!r}; got {char!r}"
            )
        if token == "D" and char not in DIGITS:
            raise ValueError(
                f"Character {idx} in fixed plate must be a digit; got {char!r}"
            )
    return raw_string


# -----------------------------
# Plate rendering
# -----------------------------

def find_font(font_path: Optional[str], target_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: List[Path] = []
    if font_path:
        candidates.append(Path(font_path))

    # Bundled font (relative to this script).
    _here = Path(__file__).parent
    candidates.append(_here / "Zurich Extra Condensed Regular" / "Zurich Extra Condensed Regular.otf")

    # Common Linux/macOS locations.
    candidates.extend([
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        Path("/usr/share/fonts/open-sans/OpenSans-Bold.ttf"),
        Path("/usr/share/fonts/google-droid-sans-fonts/DroidSans-Bold.ttf"),
        Path("/usr/share/fonts/google-carlito-fonts/Carlito-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
        Path("/System/Library/Fonts/SFNS.ttf"),
    ])

    for candidate in candidates:
        try:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=target_px)
        except Exception:
            pass

    return ImageFont.load_default()



def draw_round_rect(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], radius: int, outline, width: int, fill):
    draw.rounded_rectangle(box, radius=radius, outline=outline, width=width, fill=fill)


def _spaced_text_width(draw: ImageDraw.ImageDraw, font, text: str, spacing: int) -> int:
    """Total pixel width of text with custom inter-character spacing."""
    total = 0
    for i, ch in enumerate(text):
        bb = draw.textbbox((0, 0), ch, font=font)
        total += bb[2] - bb[0]
        if i < len(text) - 1:
            total += spacing
    return total


def _draw_spaced_text(
    draw: ImageDraw.ImageDraw, font, text: str,
    x: int, y: int, spacing: int, fill,
) -> None:
    """Draw text with custom inter-character spacing."""
    cx = x
    for ch in text:
        draw.text((cx, y), ch, font=font, fill=fill)
        bb = draw.textbbox((0, 0), ch, font=font)
        cx += (bb[2] - bb[0]) + spacing


def _draw_char_in_cell(
    draw: ImageDraw.ImageDraw, font, ch: str,
    cell_x: int, cell_y: int, cell_w: int, cell_h: int, fill,
) -> None:
    """Draw a single character centered within a fixed-width cell."""
    bb = draw.textbbox((0, 0), ch, font=font)
    ch_w = bb[2] - bb[0]
    ch_h = bb[3] - bb[1]
    x = cell_x + (cell_w - ch_w) // 2 - bb[0]
    y = cell_y + (cell_h - ch_h) // 2 - bb[1]
    draw.text((x, y), ch, font=font, fill=fill)


def _draw_star(draw: ImageDraw.ImageDraw, cx: float, cy: float, r_outer: float, fill) -> None:
    """Draw a 5-pointed star centered at (cx, cy)."""
    r_inner = r_outer * 0.382  # classic pentagram inner radius ratio
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = r_outer if i % 2 == 0 else r_inner
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=fill)


def render_texas_plate(
    plate_text: str,
    width: int,
    height: int,
    font_path: Optional[str] = None,
) -> np.ndarray:
    """
    Render a Texas general-issue passenger plate per TxDMV spec (August 2016).

    Plate dimensions: 12" × 6" (aspect ratio 2:1, enforced internally).
    All proportions derived from the official spec:
      - TEXAS header:  max 1"×1" chars, 0.125" inter-char spacing
      - Primary ROI:   1"×2.5625" chars, 0.375" inter-char spacing,
                       Texas silhouette between char 3 and char 4
      - Legend:        "THE LONE STAR STATE", max 5/16"×5/16" chars
      - Bolt holes:    7" horizontal × 4.75" vertical spacing, centered
    """
    # Enforce 2:1 aspect ratio (12" × 6") using width as the authority.
    width  = max(200, width)
    height = width // 2          # always 2:1
    W, H   = width, height

    # ── Background & border ───────────────────────────────────────────────────
    img      = Image.new("RGBA", (W, H), (252, 252, 250, 255))
    draw     = ImageDraw.Draw(img)
    border_w = max(2, W // 120)
    radius   = max(4, min(W, H) // 15)
    draw_round_rect(draw, (1, 1, W - 2, H - 2),
                    radius=radius, outline=(30, 30, 30, 255),
                    width=border_w, fill=(252, 252, 250, 255))

    # ── Bolt-hole reference positions ─────────────────────────────────────────
    # Horizontal: 7" apart on 12" plate → left=2.5", right=9.5" from left edge
    # Vertical:   4.75" apart on 6" plate → top=0.625", bottom=5.375" from top
    bx_l = int(W * 2.5  / 12)
    bx_r = int(W * 9.5  / 12)
    by_t = int(H * 0.625 / 6)
    by_b = int(H * 5.375 / 6)
    # Elongated hole: ½" wide × 0.28" tall
    bh_w = max(3, int(W * 0.5  / 12))
    bh_h = max(2, int(H * 0.28 / 6))
    for bx, by in [(bx_l, by_t), (bx_r, by_t), (bx_l, by_b), (bx_r, by_b)]:
        draw.ellipse(
            (bx - bh_w // 2, by - bh_h // 2, bx + bh_w // 2, by + bh_h // 2),
            fill=(180, 180, 180, 255), outline=(120, 120, 120, 255),
        )

    # ── "TEXAS" jurisdiction header ───────────────────────────────────────────
    # Max char height: 1" on 6" plate → H/6; inter-char spacing: 0.125" → W*0.125/12
    texas_h  = max(8, int(H / 6))
    texas_sp = max(1, int(W * 0.125 / 12))
    title_font = find_font(font_path, texas_h)
    tx_w = _spaced_text_width(draw, title_font, "TEXAS", texas_sp)
    tx_x = (W - tx_w) // 2
    tx_y = max(border_w + 2, (by_t - texas_h) // 2)
    _draw_spaced_text(draw, title_font, "TEXAS", tx_x, tx_y, texas_sp,
                      fill=(10, 40, 100, 255))

    # ── Primary ROI — fixed-width character cells with silhouette separator ───
    # Char cell: 1" wide × 2.5625" tall; inter-cell gap: 0.375"
    # Layout: [C][g][C][g][C] [g][star][g] [C][g][C][g][C][g][C]
    #           ←── group 1 ──→              ←──── group 2 ───────→
    cell_w = max(8,  int(W / 12))
    char_h = max(20, int(H * 2.5625 / 6))
    gap    = max(2,  int(W * 0.375 / 12))
    # Silhouette occupies one cell_w slot between the two groups
    sil_w  = cell_w
    # Total ROI width: 7 cells + 6 inter-char gaps + 2 flanking gaps around silhouette + silhouette
    roi_w  = 7 * cell_w + (6 + 2) * gap + sil_w
    roi_x  = (W - roi_w) // 2
    # Vertical center between the two bolt-hole rows
    char_y = (by_t + by_b - char_h) // 2

    serial_font = find_font(font_path, char_h)
    x = roi_x
    for i, ch in enumerate(plate_text[:7]):
        if i == 3:
            # Gap + Texas silhouette (star) + gap between char groups
            x += gap
            _draw_star(draw, x + sil_w / 2, char_y + char_h / 2,
                       min(sil_w, char_h) * 0.38, fill=(10, 40, 100, 255))
            x += sil_w + gap
        elif i > 0:
            x += gap
        _draw_char_in_cell(draw, serial_font, ch, x, char_y, cell_w, char_h,
                           fill=(20, 20, 20, 255))
        x += cell_w

    # ── Legend: "THE LONE STAR STATE" ────────────────────────────────────────
    # Max char height: 5/16" on 6" plate → H * 5/96; centered between bottom bolt row and edge
    leg_h  = max(6, int(H * 5 / 96))
    leg_sp = max(1, int(W / 150))
    legend_font = find_font(font_path, leg_h)
    lg_w = _spaced_text_width(draw, legend_font, "THE LONE STAR STATE", leg_sp)
    lg_x = (W - lg_w) // 2
    lg_y = by_b + (H - by_b - leg_h) // 2
    _draw_spaced_text(draw, legend_font, "THE LONE STAR STATE", lg_x, lg_y, leg_sp,
                      fill=(60, 60, 60, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


# -----------------------------
# Overlay / compositing
# -----------------------------

def warp_plate_into_image(
    base_bgr: np.ndarray,
    plate_bgra: np.ndarray,
    dst_quad_tl_tr_br_bl: np.ndarray,
    alpha_blend: float = 0.98,
) -> np.ndarray:
    out = base_bgr.copy()
    h, w = plate_bgra.shape[:2]
    src_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_quad, dst_quad_tl_tr_br_bl.astype(np.float32))

    warped_rgba = cv2.warpPerspective(
        plate_bgra,
        H,
        (base_bgr.shape[1], base_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    warped_rgb = warped_rgba[:, :, :3].astype(np.float32)
    alpha = (warped_rgba[:, :, 3:4].astype(np.float32) / 255.0) * float(alpha_blend)

    base_float = out.astype(np.float32)
    composited = warped_rgb * alpha + base_float * (1.0 - alpha)
    return np.clip(composited, 0, 255).astype(np.uint8)


# -----------------------------
# Filesystem / pipeline
# -----------------------------

def iter_images(input_root: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path



def make_output_path(output_root: Path, input_root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(input_root)
    out_name = rel.stem + "_texas" + rel.suffix
    return output_root / rel.parent / out_name



def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)



def process_one_image(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    rng: random.Random,
    font_path: Optional[str],
    pattern: str,
    fixed_plate: Optional[str],
    alpha_blend: float,
    plate_scale: float,
) -> dict:
    ann = parse_ccpd_filename(image_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise PlateGenerationError(f"Could not read image: {image_path}")

    quad = order_points_clockwise(ann.vertices_raw)
    quad_w, quad_h = quadrilateral_size(quad)

    render_w = max(200, int(round(quad_w * plate_scale)))
    render_h = max(70, int(round(quad_h * plate_scale)))

    texas_plate_raw = fixed_plate if fixed_plate is not None else generate_plate_string(rng, pattern=pattern)
    texas_plate_img = render_texas_plate(texas_plate_raw, render_w, render_h, font_path=font_path)
    composited = warp_plate_into_image(image, texas_plate_img, quad, alpha_blend=alpha_blend)

    out_path = make_output_path(output_root, input_root, image_path)
    ensure_parent(out_path)
    ok = cv2.imwrite(str(out_path), composited)
    if not ok:
        raise PlateGenerationError(f"Failed to write image: {out_path}")

    x1, y1, x2, y2 = ann.bbox_xyxy
    record = {
        "source_image": str(image_path),
        "output_image": str(out_path),
        "generated_plate": texas_plate_raw,
        "generated_plate_display": display_plate_string(texas_plate_raw),
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "vertices_tl_tr_br_bl": quad.astype(int).tolist(),
        "vertices_raw_ccpd_order": [list(p) for p in ann.vertices_raw],
        "area_ratio": ann.area_ratio,
        "tilt": ann.tilt,
        "brightness": ann.brightness,
        "blurriness": ann.blurriness,
        "original_ccpd_plate_code": ann.plate_code,
    }
    return record



def write_metadata(records: List[dict], output_root: Path) -> Tuple[Path, Path]:
    jsonl_path = output_root / "metadata.jsonl"
    csv_path = output_root / "metadata.csv"

    output_root.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    fieldnames = [
        "source_image",
        "output_image",
        "generated_plate",
        "generated_plate_display",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "vertices_tl_tr_br_bl",
        "vertices_raw_ccpd_order",
        "area_ratio",
        "tilt",
        "brightness",
        "blurriness",
        "original_ccpd_plate_code",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "source_image": r["source_image"],
                    "output_image": r["output_image"],
                    "generated_plate": r["generated_plate"],
                    "generated_plate_display": r["generated_plate_display"],
                    "bbox_x1": r["bbox"]["x1"],
                    "bbox_y1": r["bbox"]["y1"],
                    "bbox_x2": r["bbox"]["x2"],
                    "bbox_y2": r["bbox"]["y2"],
                    "vertices_tl_tr_br_bl": json.dumps(r["vertices_tl_tr_br_bl"]),
                    "vertices_raw_ccpd_order": json.dumps(r["vertices_raw_ccpd_order"]),
                    "area_ratio": r["area_ratio"],
                    "tilt": r["tilt"],
                    "brightness": r["brightness"],
                    "blurriness": r["blurriness"],
                    "original_ccpd_plate_code": r["original_ccpd_plate_code"],
                }
            )

    return jsonl_path, csv_path



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CCPD images into a Texas-style synthetic plate dataset.")
    parser.add_argument("--input", required=True, type=Path, help="Input CCPD root directory or a CCPD subset directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for composited images and metadata.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for images under --input.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images (useful for tests).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducible plate strings.")
    parser.add_argument("--font-path", type=str, default=None, help="Optional path to a .ttf font for plate text rendering.")
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="Plate string pattern using L=letter and D=digit. Default: LLLDDDD -> ABC1234",
    )
    parser.add_argument(
        "--fixed-plate",
        type=str,
        default=None,
        help="Optional fixed plate string to use for every image, e.g. ABC1234 or ABC-1234.",
    )
    parser.add_argument(
        "--alpha-blend",
        type=float,
        default=0.98,
        help="Overlay opacity in [0, 1]. Default: 0.98",
    )
    parser.add_argument(
        "--plate-scale",
        type=float,
        default=1.1,
        help="Scale factor for rendered plate size relative to plate quad. Default: 1.1",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip malformed/unreadable images instead of stopping.",
    )
    return parser



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_root: Path = args.input
    output_root: Path = args.output

    if not input_root.exists():
        print(f"Input path does not exist: {input_root}", file=sys.stderr)
        return 2

    if args.limit is not None and args.limit <= 0:
        print("--limit must be positive", file=sys.stderr)
        return 2

    if not (0.0 <= args.alpha_blend <= 1.0):
        print("--alpha-blend must be between 0 and 1", file=sys.stderr)
        return 2

    if args.plate_scale <= 0:
        print("--plate-scale must be > 0", file=sys.stderr)
        return 2

    if not re.fullmatch(r"[LD]+", args.pattern):
        print("--pattern must contain only L and D characters", file=sys.stderr)
        return 2

    fixed_plate: Optional[str] = None
    if args.fixed_plate is not None:
        try:
            fixed_plate = validate_plate_string(args.fixed_plate, pattern=args.pattern)
        except ValueError as exc:
            print(f"--fixed-plate is invalid: {exc}", file=sys.stderr)
            return 2

    rng = random.Random(args.seed)

    image_paths = list(iter_images(input_root, recursive=args.recursive))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        print("No images found.", file=sys.stderr)
        return 1

    records: List[dict] = []
    failures = 0

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            record = process_one_image(
                image_path=image_path,
                input_root=input_root,
                output_root=output_root,
                rng=rng,
                font_path=args.font_path,
                pattern=args.pattern,
                fixed_plate=fixed_plate,
                alpha_blend=args.alpha_blend,
                plate_scale=args.plate_scale,
            )
            records.append(record)
        except Exception as exc:
            failures += 1
            msg = f"[{idx}/{len(image_paths)}] FAILED: {image_path} -> {exc}"
            if args.skip_errors:
                print(msg, file=sys.stderr)
                continue
            print(msg, file=sys.stderr)
            return 1

        if idx % 100 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)} images")

    jsonl_path, csv_path = write_metadata(records, output_root)

    print("Done.")
    print(f"Processed images: {len(records)}")
    print(f"Failures: {failures}")
    print(f"Metadata JSONL: {jsonl_path}")
    print(f"Metadata CSV:   {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

