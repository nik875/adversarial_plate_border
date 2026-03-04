import cv2
import numpy as np
import asyncio
from fastanpr import FastANPR
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
        self.anpr = None
        self.target_plate = "VRJ7774"

    def initialize_alpr(self):
        """Initialize FastANPR model"""
        print("Initializing FastANPR models...")
        self.anpr = FastANPR()
        print("✓ FastANPR initialized")

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
        if self.anpr is None:
            self.initialize_alpr()

        # Load image for FastANPR (OpenCV format first)
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.heic', '.heif'] and HEIC_SUPPORT:
            pil_image = Image.open(image_path).convert('RGB')
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = cv2.imread(image_path)

        if cv_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB for fastanpr
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Run FastANPR detection (async call wrapped in asyncio.run)
        plates = asyncio.run(self.anpr.run([rgb_image]))
        
        if not plates or not plates[0]:
            return [], []

        # Separate all plates and target plates
        all_plates = []
        target_plates = []

        for plate in plates[0]:
            plate_info = {
                'text': plate.rec_text,
                'confidence': plate.rec_conf,
                'x1': int(plate.det_box[0]),
                'y1': int(plate.det_box[1]),
                'x2': int(plate.det_box[2]),
                'y2': int(plate.det_box[3]),
                'width': int(plate.det_box[2] - plate.det_box[0]),
                'height': int(plate.det_box[3] - plate.det_box[1]),
                'det_conf': plate.det_conf
            }

            all_plates.append(plate_info)

            # Check if this matches our target plate
            if plate.rec_text == self.target_plate:
                target_plates.append(plate_info)

        return all_plates, target_plates

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
        """Process an image and save result with bounding boxes"""
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

        # Save result
        result_image.save(output_path, 'PNG', quality=95)
        print(f"✓ Saved result: {output_path}")

        return output_path, all_plates, target_plates

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
            result_path, all_plates, target_plates = self.process_image(image_path)
            if result_path:
                print(f"\n🎉 Processing complete!")
                print(f"📁 Result saved to: {result_path}")

                # Summary
                if target_plates:
                    print(
                        f"🎯 SUCCESS: Found {len(target_plates)} instance(s) of '{self.target_plate}'")
                else:
                    print(f"ℹ️  Target plate '{self.target_plate}' was not found in this image")
                    print(f"   However, {len(all_plates)} other plate(s) were detected")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


def main():
    print("🔍 License Plate Detector - VRJ7774 Finder")
    print("==========================================")
    print("📱 Supports JPEG, PNG" + (", and HEIC" if HEIC_SUPPORT else ""))
    print("🎯 Specifically searches for license plate: VRJ7774")
    print("📦 Detects all plates but highlights the target plate in red")
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

    # Process image
    detector.select_and_process_image()


if __name__ == "__main__":
    main()
