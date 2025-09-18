import os
import argparse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pillow_heif
from tqdm.auto import tqdm

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()


def load_image(filepath):
    """Load image with support for HEIC files"""
    file_ext = os.path.splitext(filepath)[1].lower()

    if file_ext in ['.heic', '.heif']:
        # Use PIL for HEIC files
        pil_image = Image.open(filepath)
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        # Convert PIL image to numpy array and BGR format (to match cv2)
        img_array = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_bgr
    else:
        # Use cv2 for other formats
        img = cv2.imread(filepath)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {filepath}")
        return img


def preprocess(img: np.ndarray, img_size: tuple[int, int]
               | int, corners: np.ndarray, homography: np.ndarray):
    """
    Preprocess the input image for model inference.
    :param img: Input image in BGR format.
    :param img_size: Desired size to resize the image.
    :param corners: Corner coordinates as [x1, y1, x2, y2, x3, y3, x4, y4]
    :param homography: 3x3 homography matrix
    :return: Preprocessed image tensor, updated corners, updated homography.
    """
    # Resize the input image to match training format
    im, corners, homography, transform = letterbox(
        img, corners, homography, new_shape=img_size)
    # HWC to CHW, BGR to RGB
    im = im.transpose((2, 0, 1))[::-1]
    # 0 - 255 to 0.0 - 1.0
    im = im / 255.0
    # Model precision is FP32
    im = im.astype(np.float32)
    # Add batch dimension
    im = np.expand_dims(im, 0)
    return im, corners, homography, transform


def letterbox(
    im: np.ndarray,
    corners: np.ndarray,
    homography: np.ndarray,
    new_shape: tuple[int, int] | int = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    """
    Simplified letterbox function with fixed behavior for YOLOv9 preprocessing.
    Resizes and pads the input image to the desired size while maintaining aspect ratio.
    """
    shape = im.shape[:2]  # current shape [height, width]
    # Convert integer new_shape to a tuple (new_shape, new_shape)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Calculate the scaling ratio and resize the image
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    # Calculate new unpadded dimensions and padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2  # divide padding into 2 sides
    dh = (new_shape[0] - new_unpad[1]) / 2

    # Transform all corner coordinates
    corners = corners.copy()  # Avoid modifying original
    corners *= r  # Scale all coordinates
    corners[::2] += dw   # Add horizontal padding to all x coordinates
    corners[1::2] += dh  # Add vertical padding to all y coordinates

    # Create transformation matrix
    T = np.array([
        [r, 0, dw],
        [0, r, dh],
        [0, 0, 1]
    ])
    homography = T @ homography

    # Resize the image to the new unpadded dimensions
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # Add padding to maintain the new shape with the specified color
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color)  # add border
    return im, corners, homography, (r, dw, dh)


def reorder_corners(corners):
    """Ensure corners are in order: top-left, top-right, bottom-left, bottom-right"""
    # corners is [x1, y1, x2, y2, x3, y3, x4, y4]
    points = corners.reshape(4, 2)  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    # Sort by y coordinate first (top vs bottom)
    sorted_by_y = points[np.argsort(points[:, 1])]

    # Top two points (smaller y)
    top_points = sorted_by_y[:2]
    # Bottom two points (larger y)
    bottom_points = sorted_by_y[2:]

    # Sort top points by x coordinate (left vs right)
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    # Sort bottom points by x coordinate (left vs right)
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

    # Return in desired order: top-left, top-right, bottom-left, bottom-right
    reordered = np.array([top_left, top_right, bottom_right, bottom_left])
    return reordered.flatten()  # back to [x1, y1, x2, y2, x3, y3, x4, y4]


def calculate_bounding_box(corners):
    """Calculate rectangular bounding box enclosing the quadrilateral"""
    # corners is [x1, y1, x2, y2, x3, y3, x4, y4]
    x_coords = corners[::2]  # [x1, x2, x3, x4]
    y_coords = corners[1::2]  # [y1, y2, y3, y4]

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    return min_x, min_y, max_x, max_y  # left, top, right, bottom


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', default='out.csv', help='Path to input CSV file')
    parser.add_argument(
        '--output_csv',
        default='preproc_labels.csv',
        help='Path to output CSV file')
    parser.add_argument(
        '--output_dir',
        default='preprocessed_images',
        help='Directory for preprocessed images')
    parser.add_argument('--img_size', type=int, default=384, help='Image size for preprocessing')
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.input_csv)

    # Drop duplicate filename rows, keep first version only
    df = df.drop_duplicates(subset=['filename'], keep='first')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each row
    preprocessed_filenames = []
    bbox_left_tops = []
    bbox_right_tops = []
    new_corners_list = []
    new_homographies = []
    transforms = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Load image (with HEIC support)
        img = load_image(row['filename'])

        # Extract corners as [x1, y1, x2, y2, x3, y3, x4, y4]
        corners = np.array([row['p1_x'], row['p1_y'], row['p2_x'], row['p2_y'],
                           row['p3_x'], row['p3_y'], row['p4_x'], row['p4_y']], dtype=np.float32)

        # Ensure p1 is top left, p2 is top right, p3 is bottom left, p4 is bottom right
        corners = reorder_corners(corners)

        # Calculate rectangular bounding box enclosing the full quadrilateral
        min_x, min_y, max_x, max_y = calculate_bounding_box(corners)
        bbox_left_tops.append((min_x, min_y))
        bbox_right_tops.append((max_x, min_y))

        # Extract homography matrix
        H = np.array([[row['H00'], row['H01'], row['H02']],
                      [row['H10'], row['H11'], row['H12']],
                      [row['H20'], row['H21'], row['H22']]], dtype=np.float32)

        # Preprocess image
        im_preprocessed, corners_new, H_new, transform = preprocess(img, args.img_size, corners, H)
        transforms.append(transform)

        # Save preprocessed image
        base_filename = os.path.splitext(os.path.basename(row['filename']))[0]
        preprocessed_filename = f"{base_filename}_preprocessed.jpg"
        preprocessed_path = os.path.join(args.output_dir, preprocessed_filename)

        # Convert back to uint8 BGR for saving (reverse the preprocessing)
        im_save = (im_preprocessed[0].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        cv2.imwrite(preprocessed_path, im_save)

        preprocessed_filenames.append(preprocessed_path)
        new_corners_list.append(corners_new)
        new_homographies.append(H_new)

    # Add new columns to dataframe
    df['bbox_left_top_x'] = [pt[0] for pt in bbox_left_tops]
    df['bbox_left_top_y'] = [pt[1] for pt in bbox_left_tops]
    df['bbox_right_top_x'] = [pt[0] for pt in bbox_right_tops]
    df['bbox_right_top_y'] = [pt[1] for pt in bbox_right_tops]
    df['preprocessed_filename'] = preprocessed_filenames
    df['scale_factor'] = [i[0] for i in transforms]
    df['dw'] = [i[1] for i in transforms]
    df['dh'] = [i[2] for i in transforms]

    # Add new corner coordinates (after preprocessing)
    for i in range(4):
        df[f'new_p{i+1}_x'] = [corners[i * 2] for corners in new_corners_list]
        df[f'new_p{i+1}_y'] = [corners[i * 2 + 1] for corners in new_corners_list]

    # Add new homography matrix elements
    for i in range(3):
        for j in range(3):
            df[f'new_H{i}{j}'] = [H[i, j] for H in new_homographies]

    # Save updated dataframe
    df.to_csv(args.output_csv, index=False)
    print(f"Processed {len(df)} images. Results saved to {args.output_csv}")
    print(f"Preprocessed images saved to {args.output_dir}/")
