#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linear License Plate Analysis Pipeline

Flow (strictly builds step-to-step):
1) Detect license plates (ALPR)
2) For each plate crop: grayscale → blur → Canny edges
3) Connected components on the edge map
4) Filter components that encircle an area (have holes)
5) Among those, keep components whose hole centers are near image center
6) For each centered encircling component, keep only the border pixels of its largest hole
7) From those, select the biggest component by area and run parallelogram detection
8) Compute homography and produce perspective-corrected plate
9) Save a 3×3 composite panel and a master image with boxes

All parameters live in Config below.
"""

from fast_alpr import ALPR
import os
import cv2
import json
import math
import argparse
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image, ImageDraw, ImageFont

# Optional HEIC/HEIF support
HEIC_SUPPORT = False
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
    print("✓ HEIC support enabled")
except ImportError:
    print("ℹ️ HEIC support not available (optional). Install: pip install pillow-heif")

# ALPR backend


# ------------------------- Configuration -------------------------

@dataclass
class Config:
    # Target plate text (upper-case)
    target_plate: str = "VRJ7774"

    # Cropping: if padding_px > 0 it wins; otherwise padding_pct of max(img_h, img_w)
    padding_px: int = 0
    padding_pct: float = 2.0

    # Preprocessing
    blur_ksize: int = 5              # must be odd; auto-corrected to odd
    canny_low: int = 15
    canny_high: int = 50

    # Edge thickening (morphological dilation)
    dilate_ksize: int = 5            # must be odd; auto-corrected to odd
    dilate_iters: int = 2

    # “Encircling area” heuristics
    min_hole_abs_area: int = 20      # min pixels for a hole
    min_hole_rel_area: float = 0.05  # min % of outer contour area
    center_thresh_frac_x: float = 0.10  # hole center must be within 10% of image width from center
    center_thresh_frac_y: float = 0.10  # and within 10% of image height from center

    # Hough / parallelogram detection
    hough_threshold: int = 25
    # angle split for two orientation groups (roughly horizontal vs vertical in Hesse form)
    angle_split_deg: float = 45.0

    # Homography canonical rectangle (W×H)
    plate_out_w: int = 200
    plate_out_h: int = 100

    # Composite layout
    separator_w: int = 8
    row_label_h: int = 40
    font_scale: float = 0.4
    label_thickness: int = 1

    # Output control
    save_homography_json: Optional[str] = None


# ------------------------- Utilities -------------------------

def ensure_odd(v: int) -> int:
    return v if v % 2 == 1 else max(1, v - 1)


def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def load_image_any(path: str) -> Image.Image:
    # Single place for HEIC vs standard image loading
    ext = os.path.splitext(path)[1].lower()
    if ext in (".heic", ".heif") and HEIC_SUPPORT:
        return Image.open(path).convert("RGB")
    return Image.open(path).convert("RGB")


def color_from_label(label: int, sat: int = 255, val: int = 255) -> Tuple[int, int, int]:
    # golden-angle hue stepping for separation
    hue = (label * 137.5) % 360
    hsv = np.uint8([[[hue / 2, sat, val]]])   # OpenCV hue range 0..179
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def order_corners_clockwise(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(pts) != 4:
        return pts
    cx = sum(x for x, _ in pts) / 4.0
    cy = sum(y for _, y in pts) / 4.0
    pts_with_ang = [(x, y, math.atan2(y - cy, x - cx)) for x, y in pts]
    pts_with_ang.sort(key=lambda t: t[2])
    # rotate so that top-left-ish (smallest x+y) is first
    xy_sum = [(i, p[0] + p[1]) for i, p in enumerate(pts_with_ang)]
    start = min(xy_sum, key=lambda t: t[1])[0]
    rot = pts_with_ang[start:] + pts_with_ang[:start]
    return [(x, y) for (x, y, _) in rot]


# ------------------------- ALPR Wrapper -------------------------

class PlateDetector:
    def __init__(self):
        self.alpr = None

    def init(self):
        if self.alpr is None:
            print("Initializing ALPR models...")
            self.alpr = ALPR(
                detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="cct-xs-v1-global-model",
            )
            print("✓ ALPR initialized")

    def detect(self, cv_image: np.ndarray) -> List[Dict[str, Any]]:
        self.init()
        preds = self.alpr.predict(cv_image)
        plates = []
        for p in preds or []:
            box = p.detection.bounding_box
            plate = {
                "text": p.ocr.text,
                "confidence": float(p.ocr.confidence),
                "x1": int(box.x1), "y1": int(box.y1),
                "x2": int(box.x2), "y2": int(box.y2),
            }
            plate["width"] = plate["x2"] - plate["x1"]
            plate["height"] = plate["y2"] - plate["y1"]
            plates.append(plate)
        return plates


# ------------------------- Core Image Logic -------------------------

class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detector = PlateDetector()
        self.last_homography: Optional[np.ndarray] = None
        self.last_corners: Optional[List[Tuple[float, float]]] = None

    # ---- Step 1: Plate detection ----
    def detect_plates(self, cv_img: np.ndarray) -> Tuple[List[dict], List[dict]]:
        plates = self.detector.detect(cv_img)
        targets = [p for p in plates if p["text"].upper() == self.cfg.target_plate.upper()]
        return plates, targets

    # ---- Crop with padding ----
    def crop_with_padding(self, cv_img: np.ndarray,
                          box: dict) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = cv_img.shape[:2]
        pad = self.cfg.padding_px if self.cfg.padding_px > 0 else int(
            max(h, w) * (self.cfg.padding_pct / 100.0))
        x1 = max(0, box["x1"] - pad)
        y1 = max(0, box["y1"] - pad)
        x2 = min(w, box["x2"] + pad)
        y2 = min(h, box["y2"] + pad)
        return cv_img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    # ---- Step 2: Edges ----
    def edges_from_crop(self, crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        blur_k = ensure_odd(self.cfg.blur_ksize)
        if blur_k > 1:
            gray_blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        else:
            gray_blur = gray
        edges = cv2.Canny(gray_blur, self.cfg.canny_low, self.cfg.canny_high)

        dil_k = ensure_odd(self.cfg.dilate_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k))
        edges_thick = cv2.dilate(edges, kernel, iterations=self.cfg.dilate_iters)
        return gray, edges, edges_thick

    # ---- Step 3–6: Component analysis (single pass) ----
    def analyze_components(
        self, gray: np.ndarray, edges_thick: np.ndarray
    ) -> Dict[str, Any]:
        """
        Returns colored visualizations and lists of component labels:
        - components_img: all components colored
        - encircling_img: only components with holes
        - centered_img: only encircling components whose hole centers are near image center
        - hole_border_img: only border pixels of the LARGEST hole for each centered component
        - biggest_component_img: hole border pixels for the biggest centered component by area
        - labels_data: per-label stats (area, has_holes, is_centered, largest_hole_mask, etc.)
        """
        h, w = gray.shape
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            edges_thick, connectivity=8)

        # canvases
        components_img = np.zeros((h, w, 3), np.uint8)
        encircling_img = np.zeros_like(components_img)
        centered_img = np.zeros_like(components_img)
        hole_border_img = np.zeros_like(components_img)
        biggest_component_img = np.zeros_like(components_img)

        # all components visualization
        rng = np.random.default_rng(42)
        palette = rng.integers(0, 255, size=(num_labels, 3), dtype=np.uint8)
        palette[0] = [0, 0, 0]  # background
        components_img[:] = palette[labels]

        # helpers
        center_x, center_y = w / 2.0, h / 2.0
        th_x = w * self.cfg.center_thresh_frac_x
        th_y = h * self.cfg.center_thresh_frac_y

        labels_data = {}
        encircling_labels = []
        centered_labels = []
        hole_border_labels = []

        def largest_hole_for_mask(
                component_mask_bin: np.ndarray) -> Tuple[Optional[np.ndarray], List[Tuple[int, int]]]:
            """
            Return (largest_hole_mask, hole_centers) in component, if any.
            Two strategies:
              1) Contour hierarchy child detection
              2) Flood-fill unreachable regions (backup)
            """
            contours, hierarchy = cv2.findContours(
                component_mask_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            hole_candidates = []
            centers = []

            if hierarchy is not None:
                for i in range(len(contours)):
                    child = hierarchy[0][i][2]
                    if child != -1:
                        hole_cnt = contours[child]
                        hole_area = cv2.contourArea(hole_cnt)
                        outer_area = cv2.contourArea(contours[i])
                        if hole_area >= max(self.cfg.min_hole_abs_area,
                                            outer_area * self.cfg.min_hole_rel_area):
                            mask = np.zeros_like(component_mask_bin)
                            cv2.fillPoly(mask, [hole_cnt], 255)
                            hole_candidates.append((hole_area, mask))
                            M = cv2.moments(hole_cnt)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                centers.append((cx, cy))

            # backup flood-fill if no candidates
            if not hole_candidates:
                hgt, wdt = component_mask_bin.shape
                flood_mask = np.zeros((hgt + 2, wdt + 2), np.uint8)
                flood_src = component_mask_bin.copy()
                filled = flood_src.copy()
                cv2.floodFill(filled, None, (0, 0), 255)
                inv_filled = cv2.bitwise_not(filled)
                unreachable = cv2.bitwise_and(inv_filled, cv2.bitwise_not(component_mask_bin))

                if cv2.countNonZero(unreachable) >= self.cfg.min_hole_abs_area:
                    conts, _ = cv2.findContours(
                        unreachable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in conts:
                        area = cv2.contourArea(cnt)
                        if area >= self.cfg.min_hole_abs_area:
                            mask = np.zeros_like(component_mask_bin)
                            cv2.fillPoly(mask, [cnt], 255)
                            hole_candidates.append((area, mask))
                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                centers.append((cx, cy))

            if not hole_candidates:
                return None, []

            largest = max(hole_candidates, key=lambda t: t[0])[1]
            return largest, centers

        # iterate labels (skip 0 background)
        biggest_label = None
        biggest_area = 0

        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])

            component_mask = (labels == lbl).astype(np.uint8) * 255
            largest_hole_mask, hole_centers = largest_hole_for_mask(component_mask)

            has_holes = largest_hole_mask is not None
            is_centered = False

            if has_holes:
                encircling_labels.append(lbl)
                # draw in encircling image
                encircling_img[labels == lbl] = color_from_label(lbl, 255, 255)

                # check if any hole center is near image center
                for (hx, hy) in hole_centers:
                    if abs(hx - center_x) <= th_x and abs(hy - center_y) <= th_y:
                        is_centered = True
                        break

                if is_centered:
                    centered_labels.append(lbl)
                    centered_img[labels == lbl] = color_from_label(lbl, 255, 255)

                    # keep only border pixels of largest hole that coincide with the component
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    hole_border = cv2.morphologyEx(largest_hole_mask, cv2.MORPH_GRADIENT, kernel)
                    hole_border_on_component = cv2.bitwise_and(hole_border, component_mask)
                    mask = hole_border_on_component > 0
                    hole_border_img[mask] = color_from_label(lbl, 255, 255)
                    hole_border_labels.append(lbl)

                    # track biggest component by area among centered encircling components
                    if area > biggest_area:
                        biggest_area = area
                        biggest_label = lbl

            labels_data[lbl] = dict(
                area=area,
                has_holes=has_holes,
                is_centered=is_centered,
            )

        if biggest_label is not None:
            # paint biggest component’s hole-border pixels only
            component_mask = (labels == biggest_label).astype(np.uint8) * 255
            # recompute largest hole border for that component
            largest_hole_mask, _ = largest_hole_for_mask(component_mask)
            if largest_hole_mask is not None:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                hole_border = cv2.morphologyEx(largest_hole_mask, cv2.MORPH_GRADIENT, kernel)
                hole_border_on_component = cv2.bitwise_and(hole_border, component_mask)
                mask = hole_border_on_component > 0
                biggest_component_img[mask] = color_from_label(biggest_label, 255, 255)

        return dict(
            components_img=components_img,
            encircling_img=encircling_img,
            centered_img=centered_img,
            hole_border_img=hole_border_img,
            biggest_component_img=biggest_component_img,
            labels_data=labels_data,
            biggest_label=biggest_label,
        )

    # ---- Step 7–8: Parallelogram + Homography + Perspective ----
    def parallelogram_and_perspective(
        self, gray_bg: np.ndarray, biggest_component_img: np.ndarray, panel_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], Optional[np.ndarray], np.ndarray]:
        """
        - Detect two line orientations from biggest_component_img via HoughLines
        - Compute 4 corners from pairwise intersections
        - Draw overlay on gray background
        - Compute homography from canonical plate rect → detected quad
        - Warp inverse to produce perspective-corrected view on a square panel
        """
        h, w = gray_bg.shape
        # edge mask from biggest component image (non-black pixels)
        if biggest_component_img.ndim == 3:
            gray_edge = cv2.cvtColor(biggest_component_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_edge = biggest_component_img
        edge_bin = (gray_edge > 0).astype(np.uint8) * 255

        # Hough lines
        lines = cv2.HoughLines(edge_bin, 1, np.pi / 180, self.cfg.hough_threshold)
        overlay = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR)

        if lines is None or len(lines) < 4:
            # Not enough structure; produce defaults
            homography_matrix = None
            warped_square = self._empty_perspective_panel(panel_size, text="No Homography")
            return overlay, [], homography_matrix, warped_square

        # Normalize & split into two angle groups (roughly orthogonal)
        vals = []
        for l in lines[:100]:
            rho, theta = l[0]
            if rho < 0:
                rho = -rho
                theta = (theta + np.pi) % np.pi
            else:
                theta = theta % np.pi
            vals.append((rho, theta))
        angles_deg = np.degrees([t for _, t in vals])

        group_a, group_b = [], []
        for i, (rho, theta) in enumerate(vals):
            ang = angles_deg[i] % 180.0
            if (ang < self.cfg.angle_split_deg) or (ang > 180.0 - self.cfg.angle_split_deg):
                group_a.append((rho, theta))
            else:
                group_b.append((rho, theta))

        # Need at least two lines per group
        if len(group_a) < 2 or len(group_b) < 2:
            homography_matrix = None
            warped_square = self._empty_perspective_panel(panel_size, text="No Homography")
            return overlay, [], homography_matrix, warped_square

        # From each group: pick the two lines furthest apart in rho (maximize separation)
        def two_most_separated(lines_group):
            best = None
            best_sep = -1
            n = len(lines_group)
            for i in range(n):
                for j in range(i + 1, n):
                    sep = abs(lines_group[i][0] - lines_group[j][0])
                    if sep > best_sep:
                        best_sep = sep
                        best = (lines_group[i], lines_group[j])
            return best

        a1, a2 = two_most_separated(group_a)
        b1, b2 = two_most_separated(group_b)

        selected = [a1, a2, b1, b2]
        # draw all lines lightly
        for (rho, theta) in vals:
            a, b = math.cos(theta), math.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * a))
            pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * a))
            cv2.line(overlay, pt1, pt2, (255, 200, 100), 1)

        # draw selected in red & compute intersections
        line_params = []
        for (rho, theta) in selected:
            a, b = math.cos(theta), math.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * a))
            pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * a))
            cv2.line(overlay, pt1, pt2, (0, 0, 255), 3)
            line_params.append((rho, theta))

        # intersections between A and B groups: (2×2)=4 corners
        def intersect(r1, t1, r2, t2) -> Optional[Tuple[float, float]]:
            a1, b1 = math.cos(t1), math.sin(t1)
            a2, b2 = math.cos(t2), math.sin(t2)
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-10:
                return None
            x = (r1 * b2 - r2 * b1) / det
            y = (r2 * a1 - r1 * a2) / det
            return (x, y)

        corners: List[Tuple[float, float]] = []
        for (rhoA, thA) in [a1, a2]:
            for (rhoB, thB) in [b1, b2]:
                pt = intersect(rhoA, thA, rhoB, thB)
                if pt is not None:
                    corners.append(pt)

        # keep only those near image bounds (with slack)
        valid = []
        for (x, y) in corners:
            if -w * 0.5 <= x <= w * 1.5 and -h * 0.5 <= y <= h * 1.5:
                valid.append((x, y))

        if len(valid) < 4:
            homography_matrix = None
            warped_square = self._empty_perspective_panel(panel_size, text="No Homography")
            return overlay, [], homography_matrix, warped_square

        # pick 4 and order
        corners4 = order_corners_clockwise(valid[:4])

        # paint corners and outline
        for i, (x, y) in enumerate(corners4):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(overlay, (int(x), int(y)), 6, (0, 255, 0), -1)
                cv2.putText(overlay, str(i + 1), (int(x) + 8, int(y) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        pts = np.array([[int(x), int(y)] for (x, y) in corners4], dtype=np.int32)
        cv2.polylines(overlay, [pts], True, (255, 0, 255), 2)

        # homography: src canonical rect → dst corners
        src = np.float32([[0, 0], [self.cfg.plate_out_w, 0], [self.cfg.plate_out_w,
                         self.cfg.plate_out_h], [0, self.cfg.plate_out_h]])
        dst = np.float32(corners4)
        try:
            H = cv2.getPerspectiveTransform(src, dst)
            self.last_homography = H
            self.last_corners = [(float(x), float(y)) for (x, y) in corners4]
        except cv2.error:
            H = None

        # inverse warp into canonical rect, then paste centered into square panel
        warped_square = self.perspective_square_from(
            crop_bgr=None, homography=H, panel_size=panel_size, background=(
                0, 0, 0), source_gray_bg=gray_bg)

        return overlay, corners4, H, warped_square

    def _empty_perspective_panel(
            self, panel_size: Tuple[int, int], text: str = "No Homography") -> np.ndarray:
        ph, pw = panel_size
        img = np.zeros((ph, pw, 3), np.uint8)
        cv2.putText(img, text, (10, ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    def perspective_square_from(
        self,
        crop_bgr: Optional[np.ndarray],
        homography: Optional[np.ndarray],
        panel_size: Tuple[int, int],
        background=(0, 0, 0),
        source_gray_bg: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ph, pw = panel_size
        if homography is None:
            return self._empty_perspective_panel(panel_size)

        # If we weren't passed the original crop_bgr, synthesize from bg gray
        if crop_bgr is None and source_gray_bg is not None:
            crop_bgr = cv2.cvtColor(source_gray_bg, cv2.COLOR_GRAY2BGR)

        if crop_bgr is None:
            return self._empty_perspective_panel(panel_size)

        try:
            H_inv = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return self._empty_perspective_panel(panel_size, text="Singular Matrix")

        plate_w, plate_h = self.cfg.plate_out_w, self.cfg.plate_out_h
        unwarped = cv2.warpPerspective(
            crop_bgr, H_inv, (plate_w, plate_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background
        )

        # center-fit unwarped into square panel
        canvas = np.zeros((ph, pw, 3), np.uint8)
        max_side = int(min(pw * 0.8, ph * 0.8))
        scale = min(max_side / plate_w, max_side / plate_h)
        sw, sh = max(1, int(plate_w * scale)), max(1, int(plate_h * scale))
        unwarped_resized = cv2.resize(unwarped, (sw, sh))
        sx = (pw - sw) // 2
        sy = (ph - sh) // 2
        canvas[sy:sy + sh, sx:sx + sw] = unwarped_resized
        return canvas

    # ---- Composite (3×3) ----
    def make_composite(
        self,
        crop_bgr: np.ndarray,
        edges_thick: np.ndarray,
        components_img: np.ndarray,
        encircling_img: np.ndarray,
        centered_img: np.ndarray,
        hole_border_img: np.ndarray,
        biggest_component_img: np.ndarray,
        parallelogram_img: np.ndarray,
        perspective_img: np.ndarray,
        plate_text: str,
        plate_conf: float,
    ) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        panel_w = w
        panel_h = h
        sep = self.cfg.separator_w
        row_gap = self.cfg.row_label_h

        row_w = panel_w * 3 + sep * 2
        comp_w = row_w
        comp_h = panel_h * 3 + row_gap * 2
        composite = np.zeros((comp_h, comp_w, 3), np.uint8)

        # row1: original | edges | all components
        # convert edges to BGR
        edges_bgr = cv2.cvtColor(edges_thick, cv2.COLOR_GRAY2BGR)
        x1 = 0
        x2 = panel_w + sep
        x3 = panel_w * 2 + sep * 2
        y1 = 0
        composite[y1:y1 + panel_h, 0:panel_w] = crop_bgr
        composite[y1:y1 + panel_h, x2:x2 + panel_w] = edges_bgr
        composite[y1:y1 + panel_h, x3:x3 + panel_w] = components_img

        # separators row 1
        composite[y1:y1 + panel_h, panel_w:panel_w + sep] = 255
        composite[y1:y1 + panel_h, x2 - sep:x2] = 255

        # row2: encircling | centered | hole borders
        y2 = panel_h + row_gap
        composite[y2:y2 + panel_h, 0:panel_w] = encircling_img
        composite[y2:y2 + panel_h, x2:x2 + panel_w] = centered_img
        composite[y2:y2 + panel_h, x3:x3 + panel_w] = hole_border_img

        # separators row 2
        composite[y2:y2 + panel_h, panel_w:panel_w + sep] = 255
        composite[y2:y2 + panel_h, x2 - sep:x2] = 255

        # row3: biggest | parallelogram | perspective
        y3 = panel_h * 2 + row_gap * 2
        composite[y3:y3 + panel_h, 0:panel_w] = biggest_component_img
        composite[y3:y3 + panel_h, x2:x2 + panel_w] = parallelogram_img
        composite[y3:y3 + panel_h, x3:x3 + panel_w] = perspective_img

        # separators row 3
        composite[y3:y3 + panel_h, panel_w:panel_w + sep] = 255
        composite[y3:y3 + panel_h, x2 - sep:x2] = 255

        # labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = self.cfg.font_scale
        th = self.cfg.label_thickness
        white = (255, 255, 255)

        cv2.putText(composite, "Original", (10, 20), font, fs, white, th)
        cv2.putText(composite, "Edges", (x2 + 10, 20), font, fs, white, th)
        cv2.putText(composite, "All Components", (x3 + 10, 20), font, fs, white, th)

        cv2.putText(composite, "Encircling Only", (10, y2 + 20), font, fs, white, th)
        cv2.putText(composite, "Centered Encircling", (x2 + 10, y2 + 20), font, fs, white, th)
        cv2.putText(composite, "Largest Hole Borders", (x3 + 10, y2 + 20), font, fs, white, th)

        cv2.putText(composite, "Biggest Component", (10, y3 + 20), font, fs, white, th)
        cv2.putText(composite, "Parallelogram Analysis", (x2 + 10, y3 + 20), font, fs, white, th)
        cv2.putText(composite, "Perspective Corrected", (x3 + 10, y3 + 20), font, fs, white, th)

        # crosshair on centered panel
        cx = x2 + panel_w // 2
        cy = y2 + panel_h // 2
        cv2.drawMarker(composite, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

        # plate info
        cv2.putText(composite, f"{plate_text} (conf: {plate_conf:.3f})",
                    (10, comp_h - 10), font, fs, white, th)

        return composite

    # ---- Drawing boxes on original ----
    @staticmethod
    def draw_boxes(pil_img: Image.Image,
                   all_plates: List[dict], targets: List[dict]) -> Image.Image:
        out = pil_img.copy()
        draw = ImageDraw.Draw(out)
        # try a truetype font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", max(16, min(out.size) // 50))
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        # all plates (light gray)
        for p in all_plates:
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            draw.rectangle([x1, y1, x2, y2], outline="lightgray", width=2)
            label = f"{p['text']} ({p['confidence']:.3f})"
            tw, th = (draw.textbbox((0, 0), label, font=font)[2:] if font else (len(label) * 8, 12))
            tx, ty = x1, y1 - th - 5
            if ty < 0:
                ty = y2 + 5
            if tx + tw > out.size[0]:
                tx = out.size[0] - tw
            draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2],
                           fill="white", outline="lightgray")
            draw.text((tx, ty), label, fill="black", font=font)

        # targets (red)
        for p in targets:
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            label = f"🎯 {p['text']} ({p['confidence']:.3f})"
            tw, th = (draw.textbbox((0, 0), label, font=font)[2:] if font else (len(label) * 8, 12))
            tx, ty = x1, y1 - th - 10
            if ty < 0:
                ty = y2 + 10
            if tx + tw > out.size[0]:
                tx = out.size[0] - tw
            draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2],
                           fill="red", outline="darkred")
            draw.text((tx, ty), label, fill="white", font=font)

        return out

    # ---- Main per-image process ----
    def process_image(self, image_path: str,
                      output_path: Optional[str] = None) -> Tuple[Optional[str], List[dict], List[dict], List[str]]:
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"Looking for license plate: {self.cfg.target_plate}")

        pil_img = load_image_any(image_path)
        cv_img = pil_to_cv(pil_img)

        print("🔍 Detecting license plates...")
        all_plates, targets = self.detect_plates(cv_img)
        if not all_plates:
            print("❌ No license plates detected!")
            return None, [], [], []

        print(f"✓ Found {len(all_plates)} plate(s) total")
        if targets:
            for i, p in enumerate(targets):
                print(f" 🎯 Target {i+1}: {p['text']}  conf={p['confidence']:.3f}")
        else:
            print(f"ℹ️ Target '{self.cfg.target_plate}' not found in this image")

        # draw master
        if output_path is None:
            base_dir = os.path.dirname(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_detected.png")

        master = self.draw_boxes(pil_img, all_plates, targets)
        master.save(output_path, "PNG", quality=95)
        print(f"✓ Saved main result: {output_path}")

        # crops + analysis panels
        print("🔍 Creating cropped images with linear 9-panel analysis...")
        out_files: List[str] = []

        # Order crops: non-targets first, then targets (target panels labeled)
        plate_queue = [(False, p) for p in all_plates if p not in targets] + \
            [(True, p) for p in targets]

        for is_target, p in plate_queue:
            crop, (cx1, cy1, cx2, cy2) = self.crop_with_padding(cv_img, p)
            if crop.size == 0:
                continue

            # Step 2: edges
            gray, edges, edges_thick = self.edges_from_crop(crop)

            # Steps 3–6: components
            comp = self.analyze_components(gray, edges_thick)

            # Step 7–8: parallelogram & perspective
            paral_img, corners, H, persp_img = self.parallelogram_and_perspective(
                gray_bg=gray,
                biggest_component_img=comp["biggest_component_img"],
                panel_size=(crop.shape[0], crop.shape[1])
            )

            # Save optional homography info
            if self.cfg.save_homography_json is not None:
                entry = {
                    "original_image": image_path,
                    "crop_bbox_xyxy": [cx1, cy1, cx2, cy2],
                    "plate_text": p["text"],
                    "corners_in_crop": [(float(x), float(y)) for (x, y) in self.last_corners or []],
                    "homography_matrix": (self.last_homography.tolist() if self.last_homography is not None else None),
                }
                self._append_json(self.cfg.save_homography_json, entry)

            # Composite
            composite = self.make_composite(
                crop_bgr=crop,
                edges_thick=edges_thick,
                components_img=comp["components_img"],
                encircling_img=comp["encircling_img"],
                centered_img=comp["centered_img"],
                hole_border_img=comp["hole_border_img"],
                biggest_component_img=comp["biggest_component_img"],
                parallelogram_img=paral_img,
                perspective_img=persp_img,
                plate_text=p["text"],
                plate_conf=p["confidence"],
            )

            base_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            if base_name.endswith("_detected"):
                base_name = base_name[:-9]
            tag = "TARGET_" if is_target else ""
            crop_file = os.path.join(base_dir, f"{base_name}_{tag}{p['text']}_analysis.png")
            cv2.imwrite(crop_file, composite)
            out_files.append(crop_file)
            print(f"✓ Saved analysis: {os.path.basename(crop_file)}")

        if out_files:
            print(f"✓ Created {len(out_files)} cropped analysis image(s)")
        else:
            print("⚠ No cropped images created")

        return output_path, all_plates, targets, out_files

    @staticmethod
    def _append_json(path: str, item: dict):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = []
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
            except Exception:
                data = []
        data.append(item)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(description="Linear License Plate Analysis Pipeline")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--output", type=str, help="Output path for master result (optional)")
    parser.add_argument("--target", type=str, help="Target plate string (e.g. VRJ7774)")

    # Cropping
    parser.add_argument(
        "--padding-px",
        type=int,
        help="Absolute padding (pixels) around detected bbox (wins if >0)")
    parser.add_argument(
        "--padding-pct",
        type=float,
        help="Padding %% of max(image_dim) around bbox (used if padding-px <= 0)")

    # Preprocessing
    parser.add_argument("--blur-ksize", type=int, help="Gaussian blur kernel (odd)")
    parser.add_argument("--canny-low", type=int, help="Canny low threshold")
    parser.add_argument("--canny-high", type=int, help="Canny high threshold")

    # Dilation
    parser.add_argument("--dilate-ksize", type=int, help="Dilation kernel (odd)")
    parser.add_argument("--dilate-iters", type=int, help="Dilation iterations")

    # Encircling
    parser.add_argument("--min-hole-abs", type=int, help="Min hole area (px)")
    parser.add_argument("--min-hole-rel", type=float, help="Min hole area ratio (0..1)")
    parser.add_argument(
        "--center-frac-x",
        type=float,
        help="Centered threshold as fraction of width (e.g. 0.1)")
    parser.add_argument(
        "--center-frac-y",
        type=float,
        help="Centered threshold as fraction of height (e.g. 0.1)")

    # Hough / Parallelogram
    parser.add_argument("--hough-thresh", type=int, help="Hough threshold")
    parser.add_argument(
        "--angle-split",
        type=float,
        help="Angle degrees for group split (default ~45)")

    # Homography rect
    parser.add_argument("--plate-out-w", type=int, help="Canonical plate width")
    parser.add_argument("--plate-out-h", type=int, help="Canonical plate height")

    # Composite layout
    parser.add_argument("--sep", type=int, help="Separator width between panels")
    parser.add_argument("--row-label-h", type=int, help="Gap between rows (for labels)")

    # Homography dump
    parser.add_argument(
        "--homography-json",
        type=str,
        help="Append homography/corners entries to this JSON file")

    args = parser.parse_args()
    cfg = Config()

    # Apply overrides
    if args.target:
        cfg.target_plate = args.target
    if args.padding_px is not None:
        cfg.padding_px = max(0, args.padding_px)
    if args.padding_pct is not None:
        cfg.padding_pct = max(0.0, args.padding_pct)

    if args.blur_ksize is not None:
        cfg.blur_ksize = ensure_odd(max(1, args.blur_ksize))
    if args.canny_low is not None:
        cfg.canny_low = max(0, args.canny_low)
    if args.canny_high is not None:
        cfg.canny_high = max(0, args.canny_high)

    if args.dilate_ksize is not None:
        cfg.dilate_ksize = ensure_odd(max(1, args.dilate_ksize))
    if args.dilate_iters is not None:
        cfg.dilate_iters = max(0, args.dilate_iters)

    if args.min_hole_abs is not None:
        cfg.min_hole_abs_area = max(1, args.min_hole_abs)
    if args.min_hole_rel is not None:
        cfg.min_hole_rel_area = max(0.0, float(args.min_hole_rel))
    if args.center_frac_x is not None:
        cfg.center_thresh_frac_x = max(0.0, float(args.center_frac_x))
    if args.center_frac_y is not None:
        cfg.center_thresh_frac_y = max(0.0, float(args.center_frac_y))

    if args.hough_thresh is not None:
        cfg.hough_threshold = max(1, args.hough_thresh)
    if args.angle_split is not None:
        cfg.angle_split_deg = float(args.angle_split)

    if args.plate_out_w is not None:
        cfg.plate_out_w = max(10, args.plate_out_w)
    if args.plate_out_h is not None:
        cfg.plate_out_h = max(10, args.plate_out_h)

    if args.sep is not None:
        cfg.separator_w = max(0, args.sep)
    if args.row_label_h is not None:
        cfg.row_label_h = max(0, args.row_label_h)

    if args.homography_json:
        cfg.save_homography_json = args.homography_json

    # Print active config (optional, comment out if noisy)
    print("\nActive Config:")
    print(json.dumps(asdict(cfg), indent=2))
    print()

    pipe = Pipeline(cfg)

    if args.image:
        # CLI mode
        res = pipe.process_image(args.image, args.output)
        if not res[0]:
            return
        result_path, all_plates, targets, panels = res
        print("\n🎉 Processing complete!")
        print(f"📁 Main result saved to: {result_path}")
        if panels:
            print("📱 Cropped analysis images:")
            for f in panels:
                print(f"  - {os.path.basename(f)}")
        if targets:
            print(f"🎯 SUCCESS: found {len(targets)} instance(s) of '{cfg.target_plate}'")
        else:
            print(
                f"ℹ️ Target '{cfg.target_plate}' not found; detected {len(all_plates)} plate(s) total.")
    else:
        # Minimal interactive fallback (Tk), kept very simple
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            ft = [("All supported", "*.jpg *.jpeg *.png" + (" *.heic *.heif" if HEIC_SUPPORT else ""))]
            path = filedialog.askopenfilename(title="Select image", filetypes=ft)
            if not path:
                print("No image selected.")
                return
            pipe.process_image(path, None)
        except Exception as e:
            print(f"Interactive selection failed: {e}")
            print("Please re-run with --image PATH")


if __name__ == "__main__":
    main()
