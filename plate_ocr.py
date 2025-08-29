import cv2
import numpy as np
from fast_alpr import ALPR
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog
import os
import math

# Try to import HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
    print("✓ HEIC support enabled")
except ImportError:
    HEIC_SUPPORT = False
    print("⚠ HEIC support not available. Install pillow-heif for HEIC support: pip install pillow-heif")


class LicensePlatePerspectiveCorrector:
    def __init__(self):
        self.alpr = None
        self.target_plate = "VRJ7774"

    def initialize_alpr(self):
        """Initialize ALPR models"""
        print("Initializing ALPR models...")
        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v1-global-model",
        )
        print("✓ ALPR initialized")

    def load_image(self, image_path):
        """Load image with HEIC support"""
        file_ext = os.path.splitext(image_path)[1].lower()

        if file_ext in ['.heic', '.heif'] and HEIC_SUPPORT:
            print("Loading HEIC image...")
            pil_image = Image.open(image_path)
            return pil_image.convert('RGB')
        else:
            return Image.open(image_path).convert('RGB')

    def detect_license_plates(self, image_path):
        """Detect license plate regions using ALPR"""
        if self.alpr is None:
            self.initialize_alpr()

        # Load image for ALPR (OpenCV format)
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.heic', '.heif'] and HEIC_SUPPORT:
            pil_image = Image.open(image_path).convert('RGB')
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = cv2.imread(image_path)

        if cv_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Run ALPR detection
        predictions = self.alpr.predict(cv_image)
        if not predictions:
            return [], []

        # Process predictions
        all_plates = []
        target_plates = []

        for pred in predictions:
            detection = pred.detection
            ocr = pred.ocr
            bbox = detection.bounding_box

            plate_info = {
                'text': ocr.text,
                'confidence': ocr.confidence,
                'x1': int(bbox.x1),
                'y1': int(bbox.y1),
                'x2': int(bbox.x2),
                'y2': int(bbox.y2),
                'bbox': [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                'cv_image': cv_image
            }

            all_plates.append(plate_info)

            if ocr.text == self.target_plate:
                target_plates.append(plate_info)

        return all_plates, target_plates

    def find_plate_corners(self, cv_image, plate_bbox):
        """Find the actual corners of the license plate using edge detection"""
        x1, y1, x2, y2 = [int(coord) for coord in plate_bbox]

        # Add significant padding to capture the full plate
        padding = 500
        h, w = cv_image.shape[:2]
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        # Crop the region with padding
        plate_region = cv_image[y1_pad:y2_pad, x1_pad:x2_pad]

        if plate_region.size == 0:
            return None, None, None

        # Convert to grayscale
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive threshold to get clean edges
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Apply edge detection
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest rectangular contour (likely the license plate)
        plate_contour = None
        max_area = 0

        for contour in contours:
            # Approximate the contour to reduce points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the approximation has 4 points (quadrilateral)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                # Check if this contour is large enough and has reasonable aspect ratio
                if area > max_area and area > 1000:  # Minimum area threshold
                    # Check aspect ratio (license plates are roughly 2:1)
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        if 1.5 < aspect_ratio < 4.0:  # Reasonable aspect ratio for license plate
                            max_area = area
                            plate_contour = approx

        if plate_contour is None:
            # Fallback: use Hough lines to find plate edges
            return self.find_plate_corners_hough(plate_region, x1_pad, y1_pad)

        # Convert contour points back to original image coordinates
        corners = []
        for point in plate_contour:
            x_abs = point[0][0] + x1_pad
            y_abs = point[0][1] + y1_pad
            corners.append([x_abs, y_abs])

        # Sort corners to get them in the right order (top-left, top-right,
        # bottom-right, bottom-left)
        corners = np.array(corners, dtype=np.float32)
        corners = self.order_points(corners)

        return corners, plate_region, (x1_pad, y1_pad)

    def find_plate_corners_hough(self, plate_region, offset_x, offset_y):
        """Fallback method using Hough lines to find plate boundaries"""
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Find lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) < 2:
            return None, plate_region, (offset_x, offset_y)

        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi

            # Classify lines as horizontal or vertical based on angle
            if 80 < angle < 100 or angle < 10 or angle > 170:  # Horizontal-ish lines
                horizontal_lines.append((rho, theta))
            elif 35 < angle < 55 or 125 < angle < 145:  # Vertical-ish lines
                vertical_lines.append((rho, theta))

        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            # Find the extreme lines to form the bounding rectangle
            h, w = plate_region.shape[:2]

            # Use the original bounding box but apply a slight perspective based on detected lines
            # This is a simplified approach - in a real scenario you'd intersect the detected lines
            corners = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype=np.float32)

            # Apply small adjustments based on detected line angles if available
            if horizontal_lines:
                avg_angle = np.mean([theta for _, theta in horizontal_lines])
                # Apply slight rotation to corners based on average horizontal line angle
                rotation_adjustment = (avg_angle - np.pi / 2) * 0.1  # Small adjustment
                corners = self.apply_slight_rotation(corners, rotation_adjustment, w / 2, h / 2)

            # Convert back to original image coordinates
            for corner in corners:
                corner[0] += offset_x
                corner[1] += offset_y

            return corners, plate_region, (offset_x, offset_y)

        return None, plate_region, (offset_x, offset_y)

    def apply_slight_rotation(self, points, angle, cx, cy):
        """Apply a slight rotation to points around a center"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for point in points:
            x, y = point[0] - cx, point[1] - cy
            point[0] = x * cos_a - y * sin_a + cx
            point[1] = x * sin_a + y * cos_a + cy
        return points

    def order_points(self, pts):
        """Order points in the order: top-left, top-right, bottom-right, bottom-left"""
        # Sort by y-coordinate
        y_sorted = pts[np.argsort(pts[:, 1]), :]

        # Get the top and bottom pairs
        top_pair = y_sorted[:2, :]
        bottom_pair = y_sorted[2:, :]

        # Sort top pair by x-coordinate (left to right)
        top_pair = top_pair[np.argsort(top_pair[:, 0]), :]
        tl, tr = top_pair[0], top_pair[1]

        # Sort bottom pair by x-coordinate (left to right)
        bottom_pair = bottom_pair[np.argsort(bottom_pair[:, 0]), :]
        bl, br = bottom_pair[0], bottom_pair[1]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def perspective_transform_plate(self, corners):
        """Apply perspective transformation to create a rectified plate view"""
        if corners is None:
            return None

        # Calculate the width and height of the rectified plate
        # Use the maximum distance between corresponding points
        widthA = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) +
                         ((corners[2][1] - corners[3][1]) ** 2))
        widthB = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) +
                         ((corners[1][1] - corners[0][1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((corners[1][0] - corners[2][0]) ** 2) +
                          ((corners[1][1] - corners[2][1]) ** 2))
        heightB = np.sqrt(((corners[0][0] - corners[3][0]) ** 2) +
                          ((corners[0][1] - corners[3][1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Define destination points for the rectified plate
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(corners, dst)

        return transform_matrix, (maxWidth, maxHeight)

    def draw_detection_results(self, image, all_plates, target_plates, transformations):
        """Draw the original detection and perspective-corrected results"""
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)

        # Try to load a font
        try:
            font_size = max(12, min(image.size) // 80)
            font = ImageFont.truetype("arial.ttf", font_size)
        except BaseException:
            try:
                font = ImageFont.load_default()
            except BaseException:
                font = None

        # Process all detected plates
        for i, plate in enumerate(all_plates):
            is_target = plate in target_plates
            plate_color = 'red' if is_target else 'lightblue'

            # Draw original ALPR bounding box
            x1, y1, x2, y2 = plate['x1'], plate['y1'], plate['x2'], plate['y2']
            draw.rectangle([x1, y1, x2, y2], outline=plate_color, width=2)

            # Draw plate label
            label = f"{'🎯 ' if is_target else ''}{plate['text']} ({plate['confidence']:.3f})"

            # Position label above the plate
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(label) * 8
                text_height = 12

            text_x = x1
            text_y = y1 - text_height - 5

            if text_y < 0:
                text_y = y2 + 5
            if text_x + text_width > image.size[0]:
                text_x = image.size[0] - text_width

            # Draw text background
            bg_color = 'red' if is_target else 'white'
            text_color = 'white' if is_target else 'black'

            draw.rectangle(
                [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
                fill=bg_color,
                outline=plate_color
            )
            draw.text((text_x, text_y), label, fill=text_color, font=font)

            # Draw detected corners if available
            if i < len(transformations) and transformations[i]['corners'] is not None:
                corners = transformations[i]['corners']
                corner_color = 'red' if is_target else 'blue'

                # Draw the detected plate boundary
                corner_points = [(int(corner[0]), int(corner[1])) for corner in corners]
                draw.polygon(corner_points, outline=corner_color, width=3, fill=None)

                # Draw corner points
                for j, corner in enumerate(corners):
                    cx, cy = int(corner[0]), int(corner[1])
                    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5],
                                 fill=corner_color, outline='white', width=1)
                    # Label corners
                    corner_labels = ['TL', 'TR', 'BR', 'BL']
                    draw.text((cx + 8, cy - 8), corner_labels[j], fill=corner_color, font=font)

        return result_image

    def process_image(self, image_path, output_path=None):
        """Process an image and apply perspective transformation to license plates"""
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"Looking for license plate: {self.target_plate}")

        # Load the image
        pil_image = self.load_image(image_path)

        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Detect license plates
        print("🔍 Detecting license plates...")
        all_plates, target_plates = self.detect_license_plates(image_path)

        if not all_plates:
            print("❌ No license plates detected!")
            return None

        print(f"✓ Found {len(all_plates)} license plate(s) total")

        if target_plates:
            print(f"🎯 Found {len(target_plates)} instance(s) of target plate '{self.target_plate}'!")
        else:
            print(f"❌ Target plate '{self.target_plate}' not found")

        # Process each plate for perspective correction
        print("📐 Applying perspective transformation...")
        transformations = []
        corrected_plates = []

        for i, plate in enumerate(all_plates):
            print(f"Processing plate {i+1}: {plate['text']}")

            # Find the actual plate corners
            corners, plate_region, offset = self.find_plate_corners(cv_image, plate['bbox'])

            transformation_info = {
                'corners': corners,
                'plate_region': plate_region,
                'offset': offset
            }

            if corners is not None:
                # Apply perspective transformation
                transform_matrix, (width, height) = self.perspective_transform_plate(corners)

                # Apply the transformation to the original image
                rectified = cv2.warpPerspective(cv_image, transform_matrix, (width, height))

                transformation_info['rectified'] = rectified
                transformation_info['transform_matrix'] = transform_matrix
                transformation_info['dimensions'] = (width, height)

                corrected_plates.append(rectified)
                print(f"  ✓ Successfully rectified plate: {width}x{height}")
            else:
                print(f"  ❌ Could not find plate corners for transformation")

            transformations.append(transformation_info)

        # Draw results on the original image
        result_image = self.draw_detection_results(
            pil_image, all_plates, target_plates, transformations)

        # Generate output paths
        if output_path is None:
            base_dir = os.path.dirname(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_perspective_corrected.png")

        # Save the main result image
        result_image.save(output_path, 'PNG', quality=95)
        print(f"✓ Saved main result: {output_path}")

        # Save individual rectified plates
        for i, rectified in enumerate(corrected_plates):
            if rectified is not None:
                plate_output_path = os.path.join(base_dir, f"{base_name}_plate_{i+1}_rectified.png")
                rectified_rgb = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
                rectified_pil = Image.fromarray(rectified_rgb)
                rectified_pil.save(plate_output_path, 'PNG', quality=95)
                print(f"✓ Saved rectified plate {i+1}: {plate_output_path}")

        return output_path, all_plates, target_plates, transformations

    def select_and_process_image(self):
        """GUI file selection and processing"""
        root = tk.Tk()
        root.withdraw()

        # Set up file types based on HEIC support
        if HEIC_SUPPORT:
            filetypes = [
                ("All supported", "*.jpg *.jpeg *.png *.heic *.heif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("HEIC files", "*.heic *.heif")
            ]
        else:
            filetypes = [("Image files", "*.jpg *.jpeg *.png")]

        image_path = filedialog.askopenfilename(
            title="Select Image to Search for License Plates",
            filetypes=filetypes
        )

        if not image_path:
            print("No image selected.")
            return

        try:
            result = self.process_image(image_path)
            if result:
                result_path, all_plates, target_plates, transformations = result
                print(f"\n🎉 Processing complete!")
                print(f"📁 Main result saved to: {result_path}")

                corrected_count = sum(1 for t in transformations if 'rectified' in t)
                print(f"📐 Successfully corrected {corrected_count} out of {len(all_plates)} plates")

                if target_plates:
                    print(
                        f"🎯 SUCCESS: Found {len(target_plates)} instance(s) of '{self.target_plate}'")
                else:
                    print(f"ℹ️  Target plate '{self.target_plate}' was not found in this image")

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    print("🔍 License Plate Perspective Correction Tool")
    print("============================================")
    print("📱 Supports JPEG, PNG" + (", and HEIC" if HEIC_SUPPORT else ""))
    print("📐 Applies perspective transformation to rectify license plates")
    print("🎯 Specifically searches for license plate: VRJ7774")
    print("💾 Saves both annotated original and rectified plate images")
    print()

    detector = LicensePlatePerspectiveCorrector()

    # Option to change target plate
    change_target = input(f"Current target plate: {detector.target_plate}\n"
                          "Change target plate? (y/n) [n]: ").lower().strip()

    if change_target == 'y':
        new_target = input("Enter new target plate number: ").strip().upper()
        if new_target:
            detector.target_plate = new_target
            print(f"✓ Target plate changed to: {detector.target_plate}")

    # Process image
    detector.select_and_process_image()


if __name__ == "__main__":
    main()
