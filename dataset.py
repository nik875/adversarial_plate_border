import os
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import pillow_heif
import getpass


# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()


def transform_path_for_user(filepath):
    """Transform file paths based on current user"""
    current_user = getpass.getuser()

    if current_user == "ubuntu":
        # Strip the old base path and replace with new one
        old_base = "/Users/NikhilKalidasu/Documents/Adversarial Plate"
        new_base = "/home/ubuntu/adversarial_plate_border"

        if filepath.startswith(old_base):
            relative_path = filepath[len(old_base):].lstrip('/')
            transformed_path = os.path.join(new_base, relative_path)
            return transformed_path
        else:
            # If path doesn't start with old_base, assume it's already relative to new_base
            return os.path.join(new_base, filepath.lstrip('/'))

    # For non-ubuntu users, return path as-is
    return filepath


def load_image(filepath):
    """Load image with support for HEIC files"""
    # Transform path based on current user
    filepath = transform_path_for_user(filepath)
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
    # Use cv2 for other formats
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    return img


# ---------------------------------------------------------------------------
# Detector-specific preprocessing factories
# (kept for debug comparison only — not used in training pipeline)
# ---------------------------------------------------------------------------

def make_letterbox_prep(target_size: int):
    """Return a prep_fn that letterboxes to a square with grey padding, ÷255."""
    def fn(img_hwc: np.ndarray, corners: np.ndarray):
        shape = img_hwc.shape[:2]
        r = min(target_size / shape[0], target_size / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (target_size - new_unpad[0]) / 2
        dh = (target_size - new_unpad[1]) / 2
        img = cv2.resize(img_hwc, new_unpad, interpolation=cv2.INTER_LINEAR)
        top    = int(round(dh - 0.1)); bottom = int(round(dh + 0.1))
        left   = int(round(dw - 0.1)); right  = int(round(dw + 0.1))
        img    = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        new_corners = corners.astype(np.float32).copy()
        new_corners[:, 0] = new_corners[:, 0] * r + dw
        new_corners[:, 1] = new_corners[:, 1] * r + dh
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0, new_corners
    return fn


def make_resize_prep(width: int, height: int):
    """Return a prep_fn that hard-resizes to (width, height), ÷255."""
    def fn(img_hwc: np.ndarray, corners: np.ndarray):
        orig_h, orig_w = img_hwc.shape[:2]
        img = cv2.resize(img_hwc, (width, height), interpolation=cv2.INTER_LINEAR)
        new_corners = corners.astype(np.float32).copy()
        new_corners[:, 0] *= width  / orig_w
        new_corners[:, 1] *= height / orig_h
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0, new_corners
    return fn


def make_passthrough_prep():
    """Return a prep_fn that only divides by 255, leaving corners unchanged."""
    def fn(img_hwc: np.ndarray, corners: np.ndarray):
        return (torch.from_numpy(img_hwc).permute(2, 0, 1).float() / 255.0,
                corners.astype(np.float32).copy())
    return fn


def letterbox_preprocess(img: np.ndarray, corners: np.ndarray, homography: np.ndarray,
                        target_size: int = 384) -> tuple:
    """Preprocess image with letterbox resizing (YOLO-style).
    
    Args:
        img: Input image in BGR format (H, W, C)
        corners: Plate corners as numpy array shape (4, 2) or (8,)
        homography: 3x3 homography matrix
        target_size: Target size for square output
    
    Returns:
        Tuple of (preprocessed_img_bgr, new_corners, new_homography, scale_factor, dw, dh)
    """
    shape = img.shape[:2]  # current shape [height, width]
    
    # Reshape corners if flat
    if corners.ndim == 1:
        corners = corners.reshape(4, 2)
    corners = corners.copy().astype(np.float32)
    
    # Calculate scaling ratio
    r = min(target_size / shape[0], target_size / shape[1])
    
    # Calculate new unpadded dimensions
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (target_size - new_unpad[0]) / 2
    dh = (target_size - new_unpad[1]) / 2
    
    # Transform corners
    new_corners = corners * r
    new_corners[:, 0] += dw  # x coordinates
    new_corners[:, 1] += dh  # y coordinates
    
    # Transform homography
    T = np.array([
        [r, 0, dw],
        [0, r, dh],
        [0, 0, 1]
    ], dtype=np.float32)
    new_homography = homography @ T
    
    # Resize image
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return img, new_corners, new_homography, r, dw, dh


class AdversarialPatchDataset(Dataset):
    def __init__(self, df, transform=None, preload=False, target_size=384,
                 use_original: bool = False, gpu_device=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or T.ToTensor()
        self.preload = preload and (gpu_device is None)
        self.target_size = target_size
        self.use_original = use_original
        self.gpu_device = gpu_device
        self._gpu_cache = None

        # When use_original=True, always ignore preprocessed_filename
        if use_original:
            self.has_preprocessed = False
            print("use_original=True: returning full-res images without letterbox preprocessing")
        else:
            # Check if CSV has preprocessed images or needs on-the-fly preprocessing
            self.has_preprocessed = 'preprocessed_filename' in df.columns

            if self.has_preprocessed:
                print(f"Using preprocessed images from CSV")
            else:
                print(f"No preprocessed images found - will preprocess on-the-fly to {target_size}x{target_size}")

        if self.preload:
            desc = f"Preloading {len(self.df)} images"
            self.preloaded_images = []
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=desc):
                # PIL will raise IOError if files don't exist - letting it fail loudly
                orig_img = load_image(row['filename'])
                
                if self.has_preprocessed:
                    prep_img = load_image(row['preprocessed_filename'])
                    self.preloaded_images.append((orig_img, prep_img, None))
                else:
                    # Preprocess on-the-fly during preload
                    corners = np.array([
                        [row['p1_x'], row['p1_y']],
                        [row['p2_x'], row['p2_y']],
                        [row['p3_x'], row['p3_y']],
                        [row['p4_x'], row['p4_y']]
                    ], dtype=np.float32)
                    
                    H = np.array([
                        [row['H00'], row['H01'], row['H02']],
                        [row['H10'], row['H11'], row['H12']],
                        [row['H20'], row['H21'], row['H22']]
                    ], dtype=np.float32)
                    
                    prep_img, new_corners, new_H, r, dw, dh = letterbox_preprocess(
                        orig_img, corners, H, self.target_size)
                    
                    prep_data = {
                        'new_corners': new_corners,
                        'new_H': new_H,
                        'scale_factor': r,
                        'dw': dw,
                        'dh': dh
                    }
                    self.preloaded_images.append((orig_img, prep_img, prep_data))
        else:
            self.preloaded_images = None

        if gpu_device is not None:
            gpu_cache = []
            for idx in tqdm(range(len(self.df)), desc=f"Loading dataset to {gpu_device}"):
                item = self[idx]  # loads from disk, no RAM cache
                gpu_cache.append({
                    k: (v.to(gpu_device) if isinstance(v, torch.Tensor) else v)
                    for k, v in item.items()
                })
            self._gpu_cache = gpu_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self._gpu_cache is not None:
            return self._gpu_cache[idx]

        row = self.df.iloc[idx]

        # Load images - from memory if preloaded, from disk otherwise
        if self.preload:
            orig_img, prep_img, prep_data = self.preloaded_images[idx]
        else:
            # Load from disk
            orig_img = load_image(row['filename'])

            if self.use_original:
                prep_img = None
                prep_data = None
            elif self.has_preprocessed:
                prep_img = load_image(row['preprocessed_filename'])
                prep_data = None
            else:
                # Preprocess on-the-fly
                corners = np.array([
                    [row['p1_x'], row['p1_y']],
                    [row['p2_x'], row['p2_y']],
                    [row['p3_x'], row['p3_y']],
                    [row['p4_x'], row['p4_y']]
                ], dtype=np.float32)

                H = np.array([
                    [row['H00'], row['H01'], row['H02']],
                    [row['H10'], row['H11'], row['H12']],
                    [row['H20'], row['H21'], row['H22']]
                ], dtype=np.float32)

                prep_img, new_corners, new_H, r, dw, dh = letterbox_preprocess(
                    orig_img, corners, H, self.target_size)

                prep_data = {
                    'new_corners': new_corners,
                    'new_H': new_H,
                    'scale_factor': r,
                    'dw': dw,
                    'dh': dh
                }

        # Original corners (4x2 tensor: [p1, p2, p3, p4])
        orig_corners = torch.tensor([
            [row['p1_x'], row['p1_y']],
            [row['p2_x'], row['p2_y']],
            [row['p3_x'], row['p3_y']],
            [row['p4_x'], row['p4_y']]
        ], dtype=torch.float32)

        # Original homography (3x3 tensor)
        orig_H = torch.tensor([
            [row['H00'], row['H01'], row['H02']],
            [row['H10'], row['H11'], row['H12']],
            [row['H20'], row['H21'], row['H22']]
        ], dtype=torch.float32)

        # When use_original=True, return only the full-res image + original corners
        if self.use_original:
            if self.transform:
                orig_img = self.transform(orig_img)
            return {
                'orig_image':      orig_img,
                'orig_corners':    orig_corners,
                'orig_homography': orig_H,
                'filename':        row['filename'],
            }

        if self.transform:
            orig_img = self.transform(orig_img)
            prep_img = self.transform(prep_img)

        # Get new corners and homography (from CSV or preprocessing)
        if self.has_preprocessed:
            new_corners = torch.tensor([
                [row['new_p1_x'], row['new_p1_y']],
                [row['new_p2_x'], row['new_p2_y']],
                [row['new_p3_x'], row['new_p3_y']],
                [row['new_p4_x'], row['new_p4_y']]
            ], dtype=torch.float32)

            new_H = torch.tensor([
                [row['new_H00'], row['new_H01'], row['new_H02']],
                [row['new_H10'], row['new_H11'], row['new_H12']],
                [row['new_H20'], row['new_H21'], row['new_H22']]
            ], dtype=torch.float32)

            transform = torch.tensor([row['scale_factor'], row['dw'], row['dh']])
        else:
            # Use preprocessed data from on-the-fly processing
            new_corners = torch.from_numpy(prep_data['new_corners'])
            new_H = torch.from_numpy(prep_data['new_H'])
            transform = torch.tensor([prep_data['scale_factor'], prep_data['dw'], prep_data['dh']])

        return {
            'orig_image': orig_img,
            'prep_image': prep_img,
            'orig_corners': orig_corners,
            'new_corners': new_corners,
            'orig_homography': orig_H,
            'new_homography': new_H,
            'transform': transform,
            'filename': row['filename']
        }


def create_dataloaders(csv_path="preproc_labels.csv", batch_size=8, train_split=0.8,
                       n_jobs=1, limit=0, use_all_for_train=False,
                       use_original: bool = False,
                       pin_memory=True, gpu_device=None, **kwargs):
    """Create train and validation DataLoaders

    Args:
        csv_path: Path to CSV file with image paths and labels
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training (ignored if use_all_for_train=True)
        n_jobs: Number of worker processes
        limit: Limit number of samples (0 = no limit)
        use_all_for_train: If True, use all data for training (val_loader will be empty)
        pin_memory: Enable DataLoader pinned memory (useful for CUDA, but can increase RAM pressure)
        gpu_device: If set, preload entire dataset as GPU tensors (implies preload=True, n_jobs=0)
        **kwargs: Additional arguments passed to AdversarialPatchDataset
    """
    if gpu_device is not None:
        if n_jobs > 0:
            raise ValueError("gpu_device requires n_jobs=0 (tensors cannot be shared across workers)")
        kwargs["preload"] = True
        pin_memory = False  # data is already on GPU

    preload = kwargs.get("preload", False)
    if preload and n_jobs > 0:
        raise ValueError(
            "preload=True and num_workers>0 are incompatible: each DataLoader worker "
            "would hold a full in-memory copy of the dataset. "
            "Use preload=True with num_workers=0, or preload=False with num_workers>0."
        )

    df = pd.read_csv(csv_path)
    if limit:
        df = df.iloc[-limit:]
    print(f"Loaded {len(df)} samples")

    # Shuffle
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if use_all_for_train:
        train_df = df_shuffled
        val_df = df_shuffled.iloc[0:0]  # Empty dataframe with same columns
        print(f"Train: {len(train_df)}, Val: 0 (using all data for training)")
    else:
        train_size = int(train_split * len(df_shuffled))
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:]
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Create datasets
    train_dataset = AdversarialPatchDataset(train_df, use_original=use_original, gpu_device=gpu_device, **kwargs)
    val_dataset   = AdversarialPatchDataset(val_df,   use_original=use_original, gpu_device=gpu_device, **kwargs)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_jobs,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Usage examples:
    # Default (load from disk each time):
    transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_loader, val_loader = create_dataloaders('preproc_labels.csv', transform=transform)

    # Maximum speed (preload all images into memory):
    train_loader, val_loader = create_dataloaders(
        'preproc_labels.csv', transform=transform, preload=True)
