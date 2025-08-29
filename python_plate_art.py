import cv2
import numpy as np
from fast_alpr import ALPR
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import filedialog
import os
import random
from datetime import datetime
import multiprocessing as mp
from functools import partial

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("📊 Install tqdm for progress bars: pip install tqdm")

# Try to import HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("⚠ HEIC support not available. Install pillow-heif for HEIC support: pip install pillow-heif")


def generate_single_mask_worker(args):
    """Worker function for parallel mask generation"""
    mask_idx, image_path, output_dir, plate_boxes, num_shapes, random_seed, downsample_factor = args

    # Set random seed for this worker to ensure different results
    random.seed(random_seed + mask_idx)
    np.random.seed(random_seed + mask_idx)

    try:
        # Load and downsample image
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.heic', '.heif'] and HEIC_SUPPORT:
            pil_image = Image.open(image_path)
            background = pil_image.convert('RGB')
        else:
            background = Image.open(image_path).convert('RGB')

        # Apply downsampling if specified
        if downsample_factor != 1.0:
            new_width = int(background.size[0] * downsample_factor)
            new_height = int(background.size[1] * downsample_factor)
            background = background.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a copy for full overlay
        full_overlay_image = background.copy()
        draw = ImageDraw.Draw(full_overlay_image)

        # Generate noise shapes and collect shape data
        all_shapes = []
        for box in plate_boxes:
            shapes = generate_scaled_noise_shapes_worker(draw, box, num_shapes)
            all_shapes.append(shapes)

        # Save full overlay image
        full_overlay_path = os.path.join(output_dir, f"full_overlay{mask_idx:04d}.png")
        full_overlay_image.save(full_overlay_path, 'PNG')

        # Create and save mask-only image (transparent background)
        mask_only_image = create_mask_only_image_worker(background.size, plate_boxes, all_shapes)
        mask_only_path = os.path.join(output_dir, f"mask{mask_idx:04d}.png")
        mask_only_image.save(mask_only_path, 'PNG')

        # Create and save plate overlay images (cropped to each plate)
        for plate_idx, box in enumerate(plate_boxes):
            # Calculate crop area with some padding
            padding = 10
            crop_x1 = max(0, box['x1'] - padding)
            crop_y1 = max(0, box['y1'] - padding)
            crop_x2 = min(background.size[0], box['x2'] + padding)
            crop_y2 = min(background.size[1], box['y2'] + padding)

            # Crop the full overlay image
            plate_crop = full_overlay_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Save cropped plate overlay
            if len(plate_boxes) > 1:
                plate_overlay_path = os.path.join(
                    output_dir, f"plate_overlay{mask_idx:04d}_plate{plate_idx}.png")
            else:
                plate_overlay_path = os.path.join(output_dir, f"plate_overlay{mask_idx:04d}.png")

            plate_crop.save(plate_overlay_path, 'PNG')

        return True, mask_idx

    except Exception as e:
        return False, f"Mask {mask_idx}: {str(e)}"


def generate_scaled_noise_shapes_worker(draw, box, num_shapes=15):
    """Worker version of generate_scaled_noise_shapes for multiprocessing"""
    x1, y1 = box['x1'], box['y1']
    width, height = box['width'], box['height']

    # Calculate scale factor based on license plate size
    original_plate_width = 40
    scale_factor = max(1, width / original_plate_width)
    scale_factor /= 3

    # Scale the shape sizes
    min_size = max(1, int(2 * scale_factor))
    max_size = max(min_size + 1, int(8 * scale_factor))

    shapes_drawn = []

    for i in range(num_shapes):
        # Random position within the license plate area
        x = x1 + random.randint(0, max(1, width - max_size))
        y = y1 + random.randint(0, max(1, height - max_size))

        # Random shape type: 0=rectangle, 1=ellipse, 2=triangle
        shape_type = random.randint(0, 2)

        if shape_type == 0:  # Rectangle
            rect_w = random.randint(min_size, max_size)
            rect_h = random.randint(min_size, max_size)
            coords = [x, y, x + rect_w, y + rect_h]
            draw.rectangle(coords, fill='black')
            shapes_drawn.append(('rectangle', coords))

        elif shape_type == 1:  # Ellipse
            ellipse_w = random.randint(min_size, max_size)
            ellipse_h = random.randint(min_size, max_size)
            coords = [x, y, x + ellipse_w, y + ellipse_h]
            draw.ellipse(coords, fill='black')
            shapes_drawn.append(('ellipse', coords))

        elif shape_type == 2:  # Triangle
            triangle_size = max_size
            x1_tri = x
            y1_tri = y
            x2_tri = x + random.randint(-triangle_size, triangle_size)
            y2_tri = y + random.randint(-triangle_size, triangle_size)
            x3_tri = x + random.randint(-triangle_size, triangle_size)
            y3_tri = y + random.randint(-triangle_size, triangle_size)

            coords = [(x1_tri, y1_tri), (x2_tri, y2_tri), (x3_tri, y3_tri)]
            draw.polygon(coords, fill='black')
            shapes_drawn.append(('triangle', coords))

    return shapes_drawn


def create_mask_only_image_worker(image_size, plate_boxes, shapes_list):
    """Worker version of create_mask_only_image for multiprocessing"""
    # Create transparent image
    mask_image = Image.new('RGBA', image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_image)

    # Draw all the shapes from shapes_list
    for shapes in shapes_list:
        for shape_type, coords in shapes:
            if shape_type == 'rectangle':
                draw.rectangle(coords, fill=(0, 0, 0, 255))
            elif shape_type == 'ellipse':
                draw.ellipse(coords, fill=(0, 0, 0, 255))
            elif shape_type == 'triangle':
                draw.polygon(coords, fill=(0, 0, 0, 255))

    return mask_image


class SimpleLicensePlateNoiseGenerator:
    def __init__(self):
        self.alpr = None

    def load_and_downsample_image(self, image_path, downsample_factor=1.0):
        """Load image with HEIC support and apply downsampling"""
        file_ext = os.path.splitext(image_path)[1].lower()

        if file_ext in ['.heic', '.heif'] and HEIC_SUPPORT:
            print("    Loading HEIC image...")
            pil_image = Image.open(image_path)
            image = pil_image.convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')

        # Apply downsampling if specified
        if downsample_factor != 1.0:
            original_size = image.size
            new_width = int(image.size[0] * downsample_factor)
            new_height = int(image.size[1] * downsample_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"    Downsampled from {original_size[0]}x{original_size[1]} to {new_width}x{new_height} "
                  f"({downsample_factor:.2f}x factor)")

        return image

    def get_downsampling_input(self):
        """Get downsampling factor from user input"""
        downsample_input = input(
            "Downsample factor [1.0 = no downsampling, 0.5 = half size, etc.]: ").strip()
        if downsample_input:
            try:
                downsample_factor = float(downsample_input)
                if downsample_factor <= 0 or downsample_factor > 1:
                    print("⚠ Warning: Using factor outside typical range (0, 1]. Proceeding anyway.")
                return downsample_factor
            except ValueError:
                print("Invalid input, using no downsampling (1.0)")
                return 1.0
        else:
            return 1.0

    def initialize_alpr(self):
        """Initialize ALPR models"""
        print("Initializing ALPR models...")
        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v1-global-model",
        )
        print("✓ ALPR initialized")

    def detect_license_plates(self, image_path, downsample_factor=1.0):
        """Automatically detect license plates and return bounding boxes"""
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

        # Apply downsampling if specified
        if downsample_factor != 1.0:
            new_height = int(cv_image.shape[0] * downsample_factor)
            new_width = int(cv_image.shape[1] * downsample_factor)
            cv_image = cv2.resize(cv_image, (new_width, new_height),
                                  interpolation=cv2.INTER_LANCZOS4)

        # Run ALPR detection
        predictions = self.alpr.predict(cv_image)
        if not predictions:
            return []

        # Extract bounding boxes
        boxes = []
        for pred in predictions:
            detection = pred.detection
            ocr = pred.ocr
            bbox = detection.bounding_box

            boxes.append({
                'text': ocr.text,
                'confidence': ocr.confidence,
                'x1': int(bbox.x1),
                'y1': int(bbox.y1),
                'x2': int(bbox.x2),
                'y2': int(bbox.y2),
                'width': int(bbox.x2 - bbox.x1),
                'height': int(bbox.y2 - bbox.y1)
            })

        return boxes

    def generate_scaled_noise_shapes(self, draw, box, num_shapes=15):
        """
        Generate noise shapes exactly like the reference Processing code,
        but scaled appropriately for the detected license plate size
        """
        x1, y1 = box['x1'], box['y1']
        width, height = box['width'], box['height']

        # Calculate scale factor based on license plate size
        # Original Processing code was designed for ~40x20 pixel plates
        original_plate_width = 40
        scale_factor = max(1, width / original_plate_width)
        scale_factor /= 3

        # Scale the shape sizes (original was 2-8 pixels)
        min_size = max(1, int(2 * scale_factor))
        max_size = max(min_size + 1, int(8 * scale_factor))

        print(f"    Plate: {width}x{height} pixels, scale factor: {scale_factor:.1f}")
        print(f"    Shape sizes: {min_size}-{max_size} pixels")

        shapes_drawn = []

        for i in range(num_shapes):
            # Random position within the license plate area
            x = x1 + random.randint(0, max(1, width - max_size))
            y = y1 + random.randint(0, max(1, height - max_size))

            # Random shape type: 0=rectangle, 1=ellipse, 2=triangle
            shape_type = random.randint(0, 2)

            if shape_type == 0:  # Rectangle
                rect_w = random.randint(min_size, max_size)
                rect_h = random.randint(min_size, max_size)
                coords = [x, y, x + rect_w, y + rect_h]
                draw.rectangle(coords, fill='black')
                shapes_drawn.append(('rectangle', coords))

            elif shape_type == 1:  # Ellipse
                ellipse_w = random.randint(min_size, max_size)
                ellipse_h = random.randint(min_size, max_size)
                coords = [x, y, x + ellipse_w, y + ellipse_h]
                draw.ellipse(coords, fill='black')
                shapes_drawn.append(('ellipse', coords))

            elif shape_type == 2:  # Triangle
                # Triangle with scaled size
                triangle_size = max_size
                x1_tri = x
                y1_tri = y
                x2_tri = x + random.randint(-triangle_size, triangle_size)
                y2_tri = y + random.randint(-triangle_size, triangle_size)
                x3_tri = x + random.randint(-triangle_size, triangle_size)
                y3_tri = y + random.randint(-triangle_size, triangle_size)

                # Draw triangle
                coords = [(x1_tri, y1_tri), (x2_tri, y2_tri), (x3_tri, y3_tri)]
                draw.polygon(coords, fill='black')
                shapes_drawn.append(('triangle', coords))

        return shapes_drawn

    def create_mask_only_image(self, image_size, plate_boxes, shapes_list):
        """Create a transparent image with only the mask shapes"""
        # Create transparent image
        mask_image = Image.new('RGBA', image_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask_image)

        # Draw all the shapes from shapes_list
        for shapes in shapes_list:
            for shape_type, coords in shapes:
                if shape_type == 'rectangle':
                    draw.rectangle(coords, fill=(0, 0, 0, 255))
                elif shape_type == 'ellipse':
                    draw.ellipse(coords, fill=(0, 0, 0, 255))
                elif shape_type == 'triangle':
                    draw.polygon(coords, fill=(0, 0, 0, 255))

        return mask_image

    def generate_multiple_masks(self, image_path, output_dir, num_masks=10, num_shapes=15,
                                num_jobs=1, downsample_factor=1.0):
        """
        Generate multiple mask variations and save them in different formats
        Supports parallel processing with num_jobs parameter
        """
        print(f"Generating {num_masks} masks for: {os.path.basename(image_path)}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Load image with HEIC support and downsampling
        background = self.load_and_downsample_image(image_path, downsample_factor)

        # Automatically detect license plates
        print("  🔍 Auto-detecting license plates...")
        plate_boxes = self.detect_license_plates(image_path, downsample_factor)

        if not plate_boxes:
            print("  ❌ No license plates detected!")
            return None

        print(f"  ✓ Found {len(plate_boxes)} license plate(s)")

        if num_jobs == 1:
            # Sequential processing with progress tracking
            success_count = 0

            if TQDM_AVAILABLE:
                # Use tqdm progress bar for sequential processing
                mask_iterator = tqdm(
                    range(num_masks),
                    desc="  🎨 Generating masks",
                    unit="mask",
                    ncols=80)
            else:
                mask_iterator = range(num_masks)

            for mask_idx in mask_iterator:
                if not TQDM_AVAILABLE:
                    print(f"  📸 Generating mask {mask_idx + 1}/{num_masks}")

                try:
                    # Create a copy for full overlay
                    full_overlay_image = background.copy()
                    draw = ImageDraw.Draw(full_overlay_image)

                    # Generate noise shapes and collect shape data
                    all_shapes = []
                    for i, box in enumerate(plate_boxes):
                        shapes = self.generate_scaled_noise_shapes(draw, box, num_shapes)
                        all_shapes.append(shapes)

                    # Save full overlay image
                    full_overlay_path = os.path.join(output_dir, f"full_overlay{mask_idx:04d}.png")
                    full_overlay_image.save(full_overlay_path, 'PNG')

                    # Create and save mask-only image (transparent background)
                    mask_only_image = self.create_mask_only_image(
                        background.size, plate_boxes, all_shapes)
                    mask_only_path = os.path.join(output_dir, f"mask{mask_idx:04d}.png")
                    mask_only_image.save(mask_only_path, 'PNG')

                    # Create and save plate overlay images (cropped to each plate)
                    for plate_idx, box in enumerate(plate_boxes):
                        # Calculate crop area with some padding
                        padding = 10
                        crop_x1 = max(0, box['x1'] - padding)
                        crop_y1 = max(0, box['y1'] - padding)
                        crop_x2 = min(background.size[0], box['x2'] + padding)
                        crop_y2 = min(background.size[1], box['y2'] + padding)

                        # Crop the full overlay image
                        plate_crop = full_overlay_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                        # Save cropped plate overlay
                        if len(plate_boxes) > 1:
                            plate_overlay_path = os.path.join(
                                output_dir, f"plate_overlay{mask_idx:04d}_plate{plate_idx}.png")
                        else:
                            plate_overlay_path = os.path.join(
                                output_dir, f"plate_overlay{mask_idx:04d}.png")

                        plate_crop.save(plate_overlay_path, 'PNG')

                    success_count += 1

                except Exception as e:
                    if TQDM_AVAILABLE:
                        tqdm.write(f"    ❌ Failed to generate mask {mask_idx + 1}: {str(e)}")
                    else:
                        print(f"    ❌ Failed to generate mask {mask_idx + 1}: {str(e)}")

        else:
            # Parallel processing
            print(f"  🚀 Using {num_jobs} parallel workers")

            # Prepare arguments for worker processes
            random_seed = random.randint(0, 1000000)
            worker_args = [
                (mask_idx, image_path, output_dir, plate_boxes,
                 num_shapes, random_seed, downsample_factor)
                for mask_idx in range(num_masks)
            ]

            # Use multiprocessing pool with progress tracking
            success_count = 0
            errors = []

            with mp.Pool(processes=num_jobs) as pool:
                if TQDM_AVAILABLE:
                    # Use tqdm for real-time progress bar
                    print(f"  📸 Processing {num_masks} masks in parallel...")

                    # Use imap for real-time progress updates
                    results = list(tqdm(
                        pool.imap(generate_single_mask_worker, worker_args),
                        total=num_masks,
                        desc="  🎨 Generating masks",
                        unit="mask",
                        ncols=80,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                    ))
                else:
                    # Fallback progress indicator without tqdm
                    print(f"  📸 Processing {num_masks} masks in parallel...")

                    # Use imap_unordered for better performance without progress bar
                    results = []
                    completed = 0

                    for result in pool.imap_unordered(generate_single_mask_worker, worker_args):
                        results.append(result)
                        completed += 1

                        # Simple text-based progress indicator
                        if completed % max(1, num_masks // 20) == 0 or completed == num_masks:
                            progress_percent = (completed / num_masks) * 100
                            progress_bar = "█" * int(progress_percent // 5) + \
                                "░" * (20 - int(progress_percent // 5))
                            print(
                                f"\r    Progress: [{progress_bar}] {completed}/{num_masks} ({progress_percent:.1f}%)",
                                end="",
                                flush=True)

                    print()  # New line after progress

                # Count successes and collect errors
                for success, result in results:
                    if success:
                        success_count += 1
                    else:
                        errors.append(result)

                # Report any errors
                if errors:
                    print(f"  ⚠ Errors encountered:")
                    for error in errors[:5]:  # Show first 5 errors
                        print(f"    {error}")
                    if len(errors) > 5:
                        print(f"    ... and {len(errors) - 5} more errors")

        print(f"  ✓ Successfully generated {success_count}/{num_masks} mask sets")
        return success_count

    def generate_noise_on_image(self, image_path, num_shapes=15, downsample_factor=1.0):
        """
        Automatically detect license plates and generate scaled noise on them
        """
        print(f"Processing: {os.path.basename(image_path)}")

        # Load image with HEIC support and downsampling
        background = self.load_and_downsample_image(image_path, downsample_factor)

        # Automatically detect license plates
        print("  🔍 Auto-detecting license plates...")
        plate_boxes = self.detect_license_plates(image_path, downsample_factor)

        if not plate_boxes:
            print("  ❌ No license plates detected!")
            return None

        print(f"  ✓ Found {len(plate_boxes)} license plate(s)")

        # Save cropped original plates (without noise) first
        output_dir = os.path.dirname(image_path)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        downsample_suffix = f"_ds{downsample_factor:.2f}" if downsample_factor != 1.0 else ""

        print("  📸 Saving original plate crops...")
        for i, box in enumerate(plate_boxes):
            # Calculate crop area with some padding
            padding = 10
            crop_x1 = max(0, box['x1'] - padding)
            crop_y1 = max(0, box['y1'] - padding)
            crop_x2 = min(background.size[0], box['x2'] + padding)
            crop_y2 = min(background.size[1], box['y2'] + padding)

            # Crop the original image
            plate_crop = background.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Save cropped original plate
            if len(plate_boxes) > 1:
                plate_original_filename = f"{base_name}_plate{i}_original{downsample_suffix}_{timestamp}.png"
            else:
                plate_original_filename = f"{base_name}_plate_original{downsample_suffix}_{timestamp}.png"

            plate_original_path = os.path.join(output_dir, plate_original_filename)
            plate_crop.save(plate_original_path, 'PNG', quality=95)
            print(f"    ✓ Saved original plate {i+1}: {plate_original_filename}")

        # Create a copy for editing
        result_image = background.copy()
        draw = ImageDraw.Draw(result_image)

        # Add noise to each detected plate
        for i, box in enumerate(plate_boxes):
            print(
                f"  🎨 Adding noise to plate {i+1}: '{box['text']}' (confidence: {box['confidence']:.3f})")
            self.generate_scaled_noise_shapes(draw, box, num_shapes)

        # Save result with noise
        output_filename = f"{base_name}_noise{downsample_suffix}_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)

        result_image.save(output_path, 'PNG', quality=95)
        print(f"  ✓ Saved with noise: {output_filename}")

        # Also save cropped plates with noise for comparison
        print("  📸 Saving noisy plate crops...")
        for i, box in enumerate(plate_boxes):
            # Calculate crop area with same padding
            padding = 10
            crop_x1 = max(0, box['x1'] - padding)
            crop_y1 = max(0, box['y1'] - padding)
            crop_x2 = min(background.size[0], box['x2'] + padding)
            crop_y2 = min(background.size[1], box['y2'] + padding)

            # Crop the noisy image
            plate_crop_noisy = result_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Save cropped noisy plate
            if len(plate_boxes) > 1:
                plate_noisy_filename = f"{base_name}_plate{i}_noisy{downsample_suffix}_{timestamp}.png"
            else:
                plate_noisy_filename = f"{base_name}_plate_noisy{downsample_suffix}_{timestamp}.png"

            plate_noisy_path = os.path.join(output_dir, plate_noisy_filename)
            plate_crop_noisy.save(plate_noisy_path, 'PNG', quality=95)
            print(f"    ✓ Saved noisy plate {i+1}: {plate_noisy_filename}")

        return output_path, plate_boxes

    def process_single_image(self):
        """Process a single image with GUI file selection (supports HEIC)"""
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
            title="Select Image with License Plate",
            filetypes=filetypes
        )

        if not image_path:
            print("No image selected.")
            return

        num_shapes = int(input("Number of shapes per plate [15]: ") or "15")
        downsample_factor = self.get_downsampling_input()

        try:
            result_path, plates = self.generate_noise_on_image(
                image_path, num_shapes, downsample_factor)
            if result_path:
                print(f"\n✓ Success! Generated noise on {len(plates)} plate(s)")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    def process_batch_masks(self):
        """Generate multiple masks for a single image"""
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
            title="Select Image with License Plate for Batch Mask Generation",
            filetypes=filetypes
        )

        if not image_path:
            print("No image selected.")
            return

        output_dir = input("Enter output directory path: ").strip()
        if not output_dir:
            # Default to a subdirectory next to the image
            base_dir = os.path.dirname(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.join(base_dir, f"{base_name}_masks")

        num_masks = int(input("Number of masks to generate [10]: ") or "10")
        num_shapes = int(input("Number of shapes per plate per mask [15]: ") or "15")
        downsample_factor = self.get_downsampling_input()

        # Get number of parallel jobs
        max_jobs = mp.cpu_count()
        num_jobs_input = input(f"Number of parallel jobs [1, max {max_jobs}]: ").strip()
        if num_jobs_input:
            try:
                num_jobs = int(num_jobs_input)
                num_jobs = max(1, min(num_jobs, max_jobs))  # Clamp between 1 and max_jobs
            except ValueError:
                print(f"Invalid input, using 1 job")
                num_jobs = 1
        else:
            num_jobs = 1

        try:
            success_count = self.generate_multiple_masks(
                image_path, output_dir, num_masks, num_shapes, num_jobs, downsample_factor)
            if success_count and success_count > 0:
                print(f"\n✓ Success! Generated {success_count} mask sets in: {output_dir}")
                print(f"   Each mask set includes:")
                print(f"   - full_overlayXXXX.png (full image with mask)")
                print(f"   - maskXXXX.png (mask only, transparent background)")
                print(f"   - plate_overlayXXXX.png (cropped to plate area)")
                if downsample_factor != 1.0:
                    print(f"   Images processed with {downsample_factor:.2f}x downsampling")
                if num_jobs > 1:
                    print(f"   Used {num_jobs} parallel workers for faster processing")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            # Handle potential multiprocessing errors
            if "spawn" in str(e) or "pickle" in str(e):
                print("   Try reducing the number of jobs or using sequential processing (num_jobs=1)")
            elif "BrokenProcessPool" in str(e):
                print("   Process pool error - try reducing number of jobs or restarting")

    def process_directory(self, directory_path, num_shapes=15, downsample_factor=1.0):
        """Process all images in a directory (includes HEIC support)"""
        if not os.path.isdir(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return

        # Find image files (including HEIC if supported)
        image_extensions = ['.jpg', '.jpeg', '.png']
        if HEIC_SUPPORT:
            image_extensions.extend(['.heic', '.heif'])

        image_files = []
        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(directory_path, file))

        if not image_files:
            supported_formats = ", ".join(image_extensions)
            print(f"❌ No image files found in directory")
            print(f"   Supported formats: {supported_formats}")
            return

        print(f"Processing {len(image_files)} images with auto license plate detection...")
        if downsample_factor != 1.0:
            print(f"Using {downsample_factor:.2f}x downsampling factor")

        success_count = 0

        if TQDM_AVAILABLE:
            # Use tqdm progress bar for directory processing
            file_iterator = tqdm(
                enumerate(image_files),
                total=len(image_files),
                desc="🖼️  Processing images",
                unit="img",
                ncols=80)
        else:
            file_iterator = enumerate(image_files)

        for i, image_path in file_iterator:
            if not TQDM_AVAILABLE:
                print(f"\n[{i+1}/{len(image_files)}]", end=" ")

            try:
                result_path, plates = self.generate_noise_on_image(
                    image_path, num_shapes, downsample_factor)
                if result_path:
                    success_count += 1
            except Exception as e:
                error_msg = f"❌ Failed: {str(e)}"
                if TQDM_AVAILABLE:
                    tqdm.write(
                        f"[{i+1}/{len(image_files)}] {os.path.basename(image_path)}: {error_msg}")
                else:
                    print(error_msg)

        print(f"\n✓ Complete! Successfully processed {success_count}/{len(image_files)} images")


def main():
    generator = SimpleLicensePlateNoiseGenerator()

    print("🎨 Auto License Plate Noise Generator")
    print("====================================")
    print("✨ Automatically detects license plates and generates scaled noise")
    print("📱 Supports JPEG, PNG" + (", and HEIC" if HEIC_SUPPORT else ""))
    print("🔍 No manual rectangle selection needed - fully automatic!")
    print("🚀 Supports parallel processing for faster batch generation")
    print("📏 Supports image downsampling to reduce processing time and memory usage")
    if TQDM_AVAILABLE:
        print("📊 Real-time progress bars enabled")
    else:
        print("📊 Install tqdm for enhanced progress bars: pip install tqdm")
    print()

    mode = input("Choose mode:\n"
                 "  (s) Single image - select one image for auto-processing\n"
                 "  (b) Batch masks - generate multiple masks for one image (supports parallel)\n"
                 "  (d) Directory - auto-process folder of images\n"
                 "  (q) Quit\n"
                 "Enter choice [s/b/d/q]: ").lower().strip()

    if mode == 's':
        generator.process_single_image()
    elif mode == 'b':
        generator.process_batch_masks()
    elif mode == 'd':
        directory_path = input("Enter directory path: ").strip()
        num_shapes = int(input("Number of shapes per plate [15]: ") or "15")
        downsample_factor = generator.get_downsampling_input()
        generator.process_directory(directory_path, num_shapes, downsample_factor)
    elif mode == 'q':
        print("Goodbye!")
    else:
        print("Invalid choice. Please run again.")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set
        pass
    main()
