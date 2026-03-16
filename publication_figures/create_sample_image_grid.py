import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import messagebox
from pillow_heif import register_heif_opener

# Register HEIF opener for PIL
register_heif_opener()

# Configuration
BASE_PATH = "physical_world_test/full test/organized"
X_COORD = 5   # Slight angle
Y_COORD = -10  # Close to plate
LIGHTING_CONDITIONS = ['full sun', 'dusk', 'dark no flash', 'dark flash']

# Store corner points
corner_points = []
current_image = None
current_ax = None

def click_event(event, image_path):
    """Handle mouse clicks to mark corners"""
    global corner_points, current_image, current_ax

    if event.inaxes != current_ax:
        return

    if len(corner_points) < 4:
        corner_points.append((int(event.xdata), int(event.ydata)))
        current_ax.plot(event.xdata, event.ydata, 'ro', markersize=3)  # Smaller dots
        plt.draw()

        if len(corner_points) == 4:
            print(f"Four corners marked for {image_path}")
            plt.close()

def load_and_convert_image(image_path):
    """Load image (handles HEIC conversion if needed)"""
    try:
        # Try to load with PIL (works for most formats)
        img = Image.open(image_path)
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def find_image_for_condition(lighting, x, y, condition='plate2'):
    """Find image file for given parameters"""
    # Try different possible names
    possible_names = [
        f"x_{x:+03d}_y_{y:02d}.heic",
        f"x_{x:+03d}_y_{y:02d}.jpg",
        f"x_{x:+03d}_y_{y:02d}.png",
    ]

    base_dir = os.path.join(BASE_PATH, lighting, condition, f"y_{y:02d}")

    for name in possible_names:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path

    print(f"Warning: Could not find image for {lighting}, x={x}, y={y}")
    return None

def blur_region(image, corners):
    """Blur the region defined by four corners"""
    if len(corners) != 4:
        return image

    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(corners, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # Apply strong blur
    blurred = cv2.GaussianBlur(image, (99, 99), 30)

    # Combine using mask
    result = image.copy()
    result[mask == 255] = blurred[mask == 255]

    return result

# Step 1: Find and load images
print("Loading images from each lighting condition...")
images = {}
image_paths = {}

for lighting in LIGHTING_CONDITIONS:
    img_path = find_image_for_condition(lighting, X_COORD, Y_COORD)
    if img_path:
        img = load_and_convert_image(img_path)
        if img is not None:
            images[lighting] = img
            image_paths[lighting] = img_path
            print(f"✓ Loaded: {lighting}")

if len(images) == 0:
    print("Error: No images could be loaded!")
    exit(1)

# Step 2: Create initial plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, lighting in enumerate(LIGHTING_CONDITIONS):
    ax = axes[idx]

    if lighting in images:
        ax.imshow(images[lighting])
        ax.set_title(f'{lighting.title()}\n(x={X_COORD:+d}, y={Y_COORD})',
                    fontsize=11, fontweight='bold')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, f'{lighting.title()}\n(Not available)',
               ha='center', va='center', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

plt.suptitle(f'Sample Images from Different Lighting Conditions (Position: x={X_COORD:+d}, y={Y_COORD})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_images_grid_unblurred.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved unblurred version: sample_images_grid_unblurred.png")

# Step 3: GUI for marking corners on each image
print("\n" + "="*60)
print("Now please mark the four corners of the license plate")
print("Click on: TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT")
print("="*60)

blurred_images = {}

for lighting in images.keys():
    corner_points = []

    # Show image for corner marking (larger for better precision)
    fig_mark, ax_mark = plt.subplots(figsize=(20, 14))  # Much larger
    current_ax = ax_mark
    current_image = images[lighting]

    ax_mark.imshow(current_image)
    ax_mark.set_title(f'Mark 4 corners for: {lighting.title()}\n(Click: TL, TR, BR, BL)',
                     fontsize=14, fontweight='bold')
    ax_mark.axis('off')

    # Zoom in to center area
    height, width = current_image.shape[:2]
    # Focus on center area both horizontally and vertically
    ax_mark.set_xlim(width * 0.3, width * 0.7)  # Center 40% horizontally
    ax_mark.set_ylim(height * 0.7, height * 0.3)  # Center 40% vertically (inverted y-axis)

    # Connect click event
    cid = fig_mark.canvas.mpl_connect('button_press_event',
                                      lambda event: click_event(event, lighting))

    plt.show()

    # Apply blur
    if len(corner_points) == 4:
        blurred = blur_region(current_image.copy(), corner_points)
        blurred_images[lighting] = blurred
        print(f"✓ Blurred license plate in {lighting}")
    else:
        print(f"⚠ Skipped {lighting} (not enough corners marked)")
        blurred_images[lighting] = current_image

# Step 4: Create final blurred plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, lighting in enumerate(LIGHTING_CONDITIONS):
    ax = axes[idx]

    if lighting in blurred_images:
        ax.imshow(blurred_images[lighting])
        ax.set_title(f'{lighting.title()}\n(x={X_COORD:+d}, y={Y_COORD})',
                    fontsize=11, fontweight='bold')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, f'{lighting.title()}\n(Not available)',
               ha='center', va='center', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

plt.suptitle(f'Sample Images from Different Lighting Conditions (Position: x={X_COORD:+d}, y={Y_COORD})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_images_grid.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved final blurred version: sample_images_grid.png")
print("Done!")
