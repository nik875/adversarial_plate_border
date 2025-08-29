import cv2
import numpy as np
from fast_alpr import ALPR
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog
import os

# Try to import HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
    print("✓ HEIC support enabled")
except ImportError:
    HEIC_SUPPORT = False
    print("⚠ HEIC support not available. Install pillow-heif for HEIC support: pip install pillow-heif")


class LicensePlateDetector:
    def __init__(self):
        self.alpr = None
        self.target_plate = "VRJ7774"
        self.crop_padding = 2

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

    def detect_target_plates(self, image_path):
        """Detect all license plates and filter for target plate"""
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

        # Separate all plates and target plates
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
                'width': int(bbox.x2 - bbox.x1),
                'height': int(bbox.y2 - bbox.y1)
            }

            all_plates.append(plate_info)

            # Check if this matches our target plate
            if ocr.text == self.target_plate:
                target_plates.append(plate_info)

        return all_plates, target_plates

    def create_connected_components_image(self, gray_image, edges):
        """Create connected component analysis visualization"""
        # Apply connected component analysis to the edges
        num_labels, labels = cv2.connectedComponents(edges)

        # Create a colored image for the connected components
        # Generate different colors for each component
        np.random.seed(42)  # For consistent colors
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Set background (label 0) to black

        # Create the colored connected components image
        colored_components = colors[labels]

        # Alternative approach: use connectedComponentsWithStats for more info
        num_labels_stats, labels_stats, stats, centroids = cv2.connectedComponentsWithStats(edges)

        # Create a more sophisticated coloring based on component size
        component_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

        # Color each component based on its area (larger components get brighter colors)
        for label in range(1, num_labels_stats):  # Skip background (label 0)
            # Get component area
            area = stats[label, cv2.CC_STAT_AREA]

            # Create color based on label and area
            hue = (label * 137.5) % 360  # Golden angle distribution for good color separation
            saturation = min(255, 100 + (area * 0.5))  # Larger components more saturated
            value = min(255, 150 + (area * 0.1))  # Larger components brighter

            # Convert HSV to BGR
            hsv_color = np.uint8([[[hue / 2, saturation, value]]])  # OpenCV hue is 0-179
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

            # Apply color to this component
            component_mask = (labels_stats == label)
            component_image[component_mask] = bgr_color

        return component_image

    def create_encircling_components_image(self, gray_image, edges):
        """Create image showing only components that encircle/enclose areas"""
        # Find components that have holes (encircle areas)
        num_labels, labels_stats, stats, centroids = cv2.connectedComponentsWithStats(edges)

        # Create output image
        encircling_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

        encircling_components = []

        # Check each component for encircled areas
        for label in range(1, num_labels):  # Skip background (label 0)
            # Create mask for this component
            component_mask = (labels_stats == label).astype(np.uint8) * 255

            # Find contours in this component
            contours, hierarchy = cv2.findContours(
                component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Check if there are holes (hierarchy[0][i][2] != -1 means contour i has a child hole)
            has_holes = False
            if hierarchy is not None:
                for i in range(len(contours)):
                    if hierarchy[0][i][2] != -1:  # Has child (hole)
                        # Check if the hole is significant (not just noise)
                        hole_idx = hierarchy[0][i][2]
                        hole_area = cv2.contourArea(contours[hole_idx])
                        outer_area = cv2.contourArea(contours[i])

                        # Filter: hole should be at least 5% of outer contour area and minimum 20
                        # pixels
                        if hole_area > max(20, outer_area * 0.05):
                            has_holes = True
                            break

            # Alternative method: flood fill to detect enclosed regions
            if not has_holes:
                # Create a slightly larger canvas for flood fill
                h, w = component_mask.shape
                flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

                # Copy component mask to center of flood mask
                flood_mask[1:h + 1, 1:w + 1] = component_mask

                # Flood fill from border - this will fill everything reachable from edges
                filled = flood_mask.copy()
                cv2.floodFill(filled, None, (0, 0), 255)

                # Extract the filled region (remove the padding)
                filled_center = filled[1:h + 1, 1:w + 1]

                # Find areas that weren't reached by flood fill (enclosed regions)
                unreachable = cv2.bitwise_and(
                    cv2.bitwise_not(filled_center),
                    cv2.bitwise_not(component_mask)
                )

                # Check if there are significant unreachable areas
                unreachable_area = cv2.countNonZero(unreachable)
                if unreachable_area > 20:  # Minimum threshold for enclosed area
                    has_holes = True

            # If this component encircles an area, color it
            if has_holes:
                encircling_components.append(label)

                # Create vibrant color for encircling components
                hue = (label * 137.5) % 360
                saturation = 255  # Full saturation for encircling components
                value = 255      # Full brightness

                # Convert HSV to BGR
                hsv_color = np.uint8([[[hue / 2, saturation, value]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

                # Apply color to this component
                component_mask_bool = (labels_stats == label)
                encircling_image[component_mask_bool] = bgr_color

        return encircling_image

    def create_centered_encircling_components_image(self, gray_image, edges):
        """Create image showing only encircling components whose enclosed area centers are near image center"""
        # Find components that have holes (encircle areas)
        num_labels, labels_stats, stats, centroids = cv2.connectedComponentsWithStats(edges)

        # Calculate image center and 10% threshold
        img_height, img_width = gray_image.shape
        img_center_x, img_center_y = img_width / 2, img_height / 2
        threshold_x = img_width * 0.10  # 10% of width
        threshold_y = img_height * 0.10  # 10% of height

        # Create output image
        centered_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        centered_components = []

        # Check each component for encircled areas and their centers
        for label in range(1, num_labels):  # Skip background (label 0)
            # Create mask for this component
            component_mask = (labels_stats == label).astype(np.uint8) * 255

            # Find contours in this component
            contours, hierarchy = cv2.findContours(
                component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Check if there are holes and find their centers
            hole_centers = []
            has_significant_holes = False

            if hierarchy is not None:
                for i in range(len(contours)):
                    if hierarchy[0][i][2] != -1:  # Has child (hole)
                        # Check if the hole is significant
                        hole_idx = hierarchy[0][i][2]
                        hole_area = cv2.contourArea(contours[hole_idx])
                        outer_area = cv2.contourArea(contours[i])

                        # Filter: hole should be at least 5% of outer contour area and minimum 20
                        # pixels
                        if hole_area > max(20, outer_area * 0.05):
                            has_significant_holes = True
                            # Calculate center of this hole
                            M = cv2.moments(contours[hole_idx])
                            if M["m00"] != 0:
                                hole_center_x = int(M["m10"] / M["m00"])
                                hole_center_y = int(M["m01"] / M["m00"])
                                hole_centers.append((hole_center_x, hole_center_y))

            # Alternative method: flood fill to detect enclosed regions and find their centers
            if not has_significant_holes:
                # Create a slightly larger canvas for flood fill
                h, w = component_mask.shape
                flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

                # Copy component mask to center of flood mask
                flood_mask[1:h + 1, 1:w + 1] = component_mask

                # Flood fill from border
                filled = flood_mask.copy()
                cv2.floodFill(filled, None, (0, 0), 255)

                # Extract the filled region (remove the padding)
                filled_center = filled[1:h + 1, 1:w + 1]

                # Find areas that weren't reached by flood fill (enclosed regions)
                unreachable = cv2.bitwise_and(
                    cv2.bitwise_not(filled_center),
                    cv2.bitwise_not(component_mask)
                )

                # Check if there are significant unreachable areas
                unreachable_area = cv2.countNonZero(unreachable)
                if unreachable_area > 20:
                    has_significant_holes = True
                    # Find center of unreachable regions
                    unreachable_contours, _ = cv2.findContours(
                        unreachable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in unreachable_contours:
                        if cv2.contourArea(contour) > 20:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                hole_center_x = int(M["m10"] / M["m00"])
                                hole_center_y = int(M["m01"] / M["m00"])
                                hole_centers.append((hole_center_x, hole_center_y))

            # Check if any hole centers are near the image center
            if has_significant_holes and hole_centers:
                component_is_centered = False

                for hole_center_x, hole_center_y in hole_centers:
                    # Check if this hole center is within 10% of image center
                    distance_x = abs(hole_center_x - img_center_x)
                    distance_y = abs(hole_center_y - img_center_y)

                    if distance_x <= threshold_x and distance_y <= threshold_y:
                        component_is_centered = True
                        break

                # If this component has a centered encircled area, color it
                if component_is_centered:
                    centered_components.append(label)

                    # Create vibrant color for centered encircling components
                    hue = (label * 137.5) % 360
                    saturation = 255  # Full saturation
                    value = 255      # Full brightness

                    # Convert HSV to BGR
                    hsv_color = np.uint8([[[hue / 2, saturation, value]]])
                    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

                    # Apply color to this component
                    component_mask_bool = (labels_stats == label)
                    centered_image[component_mask_bool] = bgr_color

                    # Draw a small circle at the hole center(s) for visualization
                    for hole_center_x, hole_center_y in hole_centers:
                        distance_x = abs(hole_center_x - img_center_x)
                        distance_y = abs(hole_center_y - img_center_y)

                        if distance_x <= threshold_x and distance_y <= threshold_y:
                            cv2.circle(
                                centered_image, (hole_center_x, hole_center_y), 3, (255, 255, 255), -1)

        print(f"🎯 Found {len(centered_components)} centered encircling components")
        return centered_image

    def create_cropped_edge_images(self, image_path, all_plates, target_plates, base_output_path):
        """Create cropped images with edge detection and connected components for all detected plates"""
        # Load original image as OpenCV format for edge detection
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.heic', '.heif'] and HEIC_SUPPORT:
            pil_image = Image.open(image_path).convert('RGB')
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = cv2.imread(image_path)

        if cv_image is None:
            return []

        image_height, image_width = cv_image.shape[:2]
        cropped_files = []

        # Process all plates (both target and non-target)
        plates_to_process = []

        # Add all non-target plates
        for plate in all_plates:
            if plate not in target_plates:
                plates_to_process.append(('plate', plate))

        # Add target plates (these will be processed last and marked as target)
        for plate in target_plates:
            plates_to_process.append(('target', plate))

        for plate_type, plate in plates_to_process:
            x1, y1, x2, y2 = plate['x1'], plate['y1'], plate['x2'], plate['y2']

            # Calculate padded crop area
            padding = int(max(image_height, image_width) * self.crop_padding / 100)
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(image_width, x2 + padding)
            crop_y2 = min(image_height, y2 + padding)

            # Crop the image
            cropped = cv_image[crop_y1:crop_y2, crop_x1:crop_x2]

            if cropped.size == 0:
                continue

            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 15, 50)

            kernel_size = round(max(image_height, image_width) / 3000)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            edges_thick = cv2.dilate(edges, kernel, iterations=2)

            import pickle
            with open("edges.pkl", "wb") as f:
                pickle.dump(edges_thick, f)

            # Create connected components analysis
            components_image = self.create_connected_components_image(gray, edges_thick)

            # Create encircling components analysis (only components that encircle areas)
            encircling_image = self.create_encircling_components_image(gray, edges_thick)

            # Create centered encircling components analysis
            centered_image = self.create_centered_encircling_components_image(gray, edges_thick)

            # Convert edges to 3-channel
            edges_colored = cv2.cvtColor(edges_thick, cv2.COLOR_GRAY2BGR)

            # Create two-row composite image
            # Row 1: original | edges | all components | encircling components
            # Row 2: centered encircling (full width)
            panel_width = cropped.shape[1]
            panel_height = cropped.shape[0]
            separator_width = 8
            row_separator_height = 40  # Space between rows for labels

            # Row 1: 4 panels + 3 separators
            row1_width = panel_width * 4 + separator_width * 3
            # Row 2: 1 panel (centered)
            row2_width = panel_width

            # Composite dimensions
            composite_width = max(row1_width, row2_width)
            composite_height = panel_height * 2 + row_separator_height
            composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)

            # === ROW 1 ===
            # Place original (panel 1)
            composite[:panel_height, :panel_width] = cropped

            # Add separator 1
            sep1_start = panel_width
            composite[:panel_height, sep1_start:sep1_start + separator_width] = [255, 255, 255]

            # Place edges (panel 2)
            panel2_start = sep1_start + separator_width
            composite[:panel_height, panel2_start:panel2_start + panel_width] = edges_colored

            # Add separator 2
            sep2_start = panel2_start + panel_width
            composite[:panel_height, sep2_start:sep2_start + separator_width] = [255, 255, 255]

            # Place all connected components (panel 3)
            panel3_start = sep2_start + separator_width
            composite[:panel_height, panel3_start:panel3_start + panel_width] = components_image

            # Add separator 3
            sep3_start = panel3_start + panel_width
            composite[:panel_height, sep3_start:sep3_start + separator_width] = [255, 255, 255]

            # Place encircling components (panel 4)
            panel4_start = sep3_start + separator_width
            composite[:panel_height, panel4_start:panel4_start + panel_width] = encircling_image

            # === ROW 2 ===
            # Center the panel in row 2
            row2_start_y = panel_height + row_separator_height
            row2_panel_x = (composite_width - panel_width) // 2  # Center horizontally
            composite[row2_start_y:row2_start_y + panel_height,
                      row2_panel_x:row2_panel_x + panel_width] = centered_image

            # Add text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            color = (255, 255, 255)
            thickness = 1

            # Row 1 labels
            cv2.putText(composite, "Original", (10, 20), font, font_scale, color, thickness)
            cv2.putText(composite, "Edges", (panel2_start + 10, 20),
                        font, font_scale, color, thickness)
            cv2.putText(composite, "All Components", (panel3_start + 10, 20),
                        font, font_scale, color, thickness)
            cv2.putText(
                composite,
                "Encircling Only",
                (panel4_start + 10,
                 20),
                font,
                font_scale,
                color,
                thickness)

            # Row 2 label
            cv2.putText(composite, "Centered Encircling (<10% from center)",
                        (row2_panel_x + 10, row2_start_y + 20), font, font_scale, color, thickness)

            # Add image center crosshair on row 2 for reference
            img_center_x = row2_panel_x + panel_width // 2
            img_center_y = row2_start_y + panel_height // 2
            cv2.drawMarker(composite, (img_center_x, img_center_y), (255, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)

            # Add plate info at bottom
            plate_info = f"{plate['text']} (conf: {plate['confidence']:.3f})"
            cv2.putText(composite, plate_info, (10, composite_height - 10),
                        font, font_scale, color, thickness)

            # Generate output filename
            base_dir = os.path.dirname(base_output_path)
            base_name = os.path.splitext(os.path.basename(base_output_path))[0]

            # Remove "_detected" if it exists
            if base_name.endswith("_detected"):
                base_name = base_name[:-9]

            if plate_type == 'target':
                crop_filename = f"{base_name}_TARGET_{plate['text']}_analysis.png"
            else:
                crop_filename = f"{base_name}_{plate['text']}_analysis.png"

            crop_path = os.path.join(base_dir, crop_filename)

            # Save the three-panel composite image
            cv2.imwrite(crop_path, composite)
            cropped_files.append(crop_path)

            if plate_type == 'target':
                print(f"🎯 Saved TARGET plate analysis: {crop_filename}")
            else:
                print(f"📱 Saved plate analysis: {crop_filename}")

        return cropped_files

    def draw_bounding_boxes(self, image, all_plates, target_plates):
        """Draw bounding boxes on the image"""
        # Create a copy for drawing
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)

        # Try to load a font, fallback to default if not available
        try:
            # Try to use a larger font if available
            font_size = max(16, min(image.size) // 50)  # Scale font with image size
            font = ImageFont.truetype("arial.ttf", font_size)
        except BaseException:
            try:
                font = ImageFont.load_default()
            except BaseException:
                font = None

        # Draw all detected plates in light gray
        for plate in all_plates:
            x1, y1, x2, y2 = plate['x1'], plate['y1'], plate['x2'], plate['y2']

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline='lightgray', width=2)

            # Draw label
            label = f"{plate['text']} ({plate['confidence']:.3f})"

            # Calculate text position (above the box)
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # Rough estimation if no font available
                text_width = len(label) * 8
                text_height = 12

            text_x = x1
            text_y = y1 - text_height - 5

            # Ensure text stays within image bounds
            if text_y < 0:
                text_y = y2 + 5
            if text_x + text_width > image.size[0]:
                text_x = image.size[0] - text_width

            # Draw text background for better visibility
            draw.rectangle(
                [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
                fill='white',
                outline='lightgray'
            )

            # Draw text
            draw.text((text_x, text_y), label, fill='black', font=font)

        # Draw target plates in bright red (on top)
        for plate in target_plates:
            x1, y1, x2, y2 = plate['x1'], plate['y1'], plate['x2'], plate['y2']

            # Draw thick red rectangle
            draw.rectangle([x1, y1, x2, y2], outline='red', width=4)

            # Draw label with red background
            label = f"🎯 {plate['text']} ({plate['confidence']:.3f})"

            # Calculate text position
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(label) * 8
                text_height = 12

            text_x = x1
            text_y = y1 - text_height - 10  # A bit higher for target plates

            # Ensure text stays within image bounds
            if text_y < 0:
                text_y = y2 + 10
            if text_x + text_width > image.size[0]:
                text_x = image.size[0] - text_width

            # Draw red background for target plate
            draw.rectangle(
                [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
                fill='red',
                outline='darkred'
            )

            # Draw white text
            draw.text((text_x, text_y), label, fill='white', font=font)

        return result_image

    def process_image(self, image_path, output_path=None):
        """Process an image and save result with bounding boxes and cropped analysis images"""
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"Looking for license plate: {self.target_plate}")

        # Load the image
        image = self.load_image(image_path)

        # Detect license plates
        print("🔍 Detecting license plates...")
        all_plates, target_plates = self.detect_target_plates(image_path)

        if not all_plates:
            print("❌ No license plates detected!")
            return None

        print(f"✓ Found {len(all_plates)} license plate(s) total")

        if target_plates:
            print(f"🎯 Found {len(target_plates)} instance(s) of target plate '{self.target_plate}'!")
            for i, plate in enumerate(target_plates):
                print(f"   Target {i+1}: confidence = {plate['confidence']:.3f}")
        else:
            print(f"❌ Target plate '{self.target_plate}' not found")

        # Draw bounding boxes
        result_image = self.draw_bounding_boxes(image, all_plates, target_plates)

        # Generate output path if not provided
        if output_path is None:
            base_dir = os.path.dirname(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_detected.png")

        # Save main result
        result_image.save(output_path, 'PNG', quality=95)
        print(f"✓ Saved main result: {output_path}")

        # Create cropped analysis images (5-panel analysis)
        print("🔍 Creating cropped images with comprehensive 5-panel analysis...")
        cropped_files = self.create_cropped_edge_images(
            image_path, all_plates, target_plates, output_path)

        if cropped_files:
            print(f"✓ Created {len(cropped_files)} cropped analysis image(s)")
        else:
            print("⚠ No cropped images created")

        return output_path, all_plates, target_plates, cropped_files

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
                result_path, all_plates, target_plates, cropped_files = result
                print(f"\n🎉 Processing complete!")
                print(f"📁 Main result saved to: {result_path}")

                if cropped_files:
                    print(f"📱 {len(cropped_files)} cropped analysis images created:")
                    for crop_file in cropped_files:
                        print(f"   - {os.path.basename(crop_file)}")

                # Summary
                if target_plates:
                    print(
                        f"🎯 SUCCESS: Found {len(target_plates)} instance(s) of '{self.target_plate}'")
                else:
                    print(f"ℹ️  Target plate '{self.target_plate}' was not found in this image")
                    print(f"   However, {len(all_plates)} other plate(s) were detected")

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise


def main():
    print("🔍 Enhanced License Plate Detector - VRJ7774 Finder")
    print("===================================================")
    print("📱 Supports JPEG, PNG" + (", and HEIC" if HEIC_SUPPORT else ""))
    print("🎯 Specifically searches for license plate: VRJ7774")
    print("📦 Detects all plates and highlights the target plate in red")
    print("✂️  Creates cropped images with 5-panel analysis:")
    print("   Row 1:")
    print("     1️⃣ Original image")
    print("     2️⃣ Edge detection")
    print("     3️⃣ All connected components (different colors)")
    print("     4️⃣ Encircling components only (components that form closed loops)")
    print("   Row 2:")
    print("     5️⃣ Centered encircling components (holes <10% from image center)")
    print()

    detector = LicensePlateDetector()

    # Option to change target plate
    change_target = input(f"Current target plate: {detector.target_plate}\n"
                          "Change target plate? (y/n) [n]: ").lower().strip()

    if change_target == 'y':
        new_target = input("Enter new target plate number: ").strip().upper()
        if new_target:
            detector.target_plate = new_target
            print(f"✓ Target plate changed to: {detector.target_plate}")

    # Option to change crop padding
    change_padding = input(f"Current crop padding: {detector.crop_padding} pixels\n"
                           "Change crop padding? (y/n) [n]: ").lower().strip()

    if change_padding == 'y':
        try:
            new_padding = int(input("Enter new padding in pixels [100]: ") or "100")
            if new_padding >= 0:
                detector.crop_padding = new_padding
                print(f"✓ Crop padding changed to: {detector.crop_padding} pixels")
        except ValueError:
            print("Invalid padding value, keeping default (100)")

    # Process image
    detector.select_and_process_image()


if __name__ == "__main__":
    main()
