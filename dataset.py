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
    # Use cv2 for other formats
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    return img


class AdversarialPatchDataset(Dataset):
    def __init__(self, df, transform=None, preload=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform or T.ToTensor()
        self.preload = preload

        if self.preload:
            desc = f"Preloading {len(self.df)} images"
            self.preloaded_images = []
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=desc):
                # PIL will raise IOError if files don't exist - letting it fail loudly
                orig_img = load_image(row['filename'])
                prep_img = load_image(row['preprocessed_filename'])
                self.preloaded_images.append((orig_img, prep_img))
        else:
            self.preloaded_images = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load images - from memory if preloaded, from disk otherwise
        if self.preload:
            orig_img, prep_img = self.preloaded_images[idx]
        else:
            # PIL will raise IOError if files don't exist - letting it fail loudly
            orig_img = load_image(row['filename'])
            prep_img = load_image(row['preprocessed_filename'])

        if self.transform:
            orig_img = self.transform(orig_img)
            prep_img = self.transform(prep_img)

        # Original corners (4x2 tensor: [p1, p2, p3, p4])
        orig_corners = torch.tensor([
            [row['p1_x'], row['p1_y']],
            [row['p2_x'], row['p2_y']],
            [row['p3_x'], row['p3_y']],
            [row['p4_x'], row['p4_y']]
        ], dtype=torch.float32)

        # New corners (4x2 tensor)
        new_corners = torch.tensor([
            [row['new_p1_x'], row['new_p1_y']],
            [row['new_p2_x'], row['new_p2_y']],
            [row['new_p3_x'], row['new_p3_y']],
            [row['new_p4_x'], row['new_p4_y']]
        ], dtype=torch.float32)

        # Original homography (3x3 tensor)
        orig_H = torch.tensor([
            [row['H00'], row['H01'], row['H02']],
            [row['H10'], row['H11'], row['H12']],
            [row['H20'], row['H21'], row['H22']]
        ], dtype=torch.float32)

        # New homography (3x3 tensor)
        new_H = torch.tensor([
            [row['new_H00'], row['new_H01'], row['new_H02']],
            [row['new_H10'], row['new_H11'], row['new_H12']],
            [row['new_H20'], row['new_H21'], row['new_H22']]
        ], dtype=torch.float32)

        transform = torch.tensor([row['scale_factor'], row['dw'], row['dh']])

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


def create_dataloaders(csv_path, batch_size=8, train_split=0.8, n_jobs=1, limit=0, **kwargs):
    """Create train and validation DataLoaders"""
    df = pd.read_csv(csv_path)
    if limit:
        df = df.iloc[-limit:]
    print(f"Loaded {len(df)} samples")

    # Shuffle and split
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(train_split * len(df_shuffled))

    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size:]

    print(f"Train: {train_size}, Val: {len(val_df)}")

    # Create datasets
    train_dataset = AdversarialPatchDataset(train_df, **kwargs)
    val_dataset = AdversarialPatchDataset(val_df, **kwargs)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_jobs,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs,
        pin_memory=True
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
