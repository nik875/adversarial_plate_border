#!/usr/bin/env python3
"""
Realistic License Plate Lighting Augmentation - UPDATED WITH RANDOM MODE

This script applies realistic lighting transformations to license plates.

TWO MODES:
  1. SYSTEMATIC: Apply each augmentation type once (23 types, no duplicates)
  2. RANDOM: Random combinations of filters with random parameters per variant

This script:
1. Detects license plate corners using SAM
2. Extracts and flattens the license plate via homography
3. Applies realistic lighting transformations
4. Re-projects the augmented plate back onto the original image

Lighting transformations include:
- Brightness/darkness variations
- Shadow casting (directional)
- Specular reflections/glare
- Color temperature shifts (warm/cool)
- Weathering effects (dirt, rain)
- HDR-style local contrast
- Time-of-day lighting

Usage - RANDOM MODE (NEW!):
    # Single image with 10 random variants
    python augment_plate_lighting.py \\
        --image car.jpg \\
        --corners-csv corners.csv \\
        --output-dir augmented/ \\
        --random \\
        --num-random-variants 10
    
    # Control randomness parameters
    python augment_plate_lighting.py \\
        --image car.jpg \\
        --corners-csv corners.csv \\
        --output-dir augmented/ \\
        --random \\
        --num-random-variants 20 \\
        --filter-prob 0.4 \\
        --min-filters 2 \\
        --max-filters 5
    
    # Batch with reproducible random seed
    python augment_plate_lighting.py \\
        --input-csv dataset.csv \\
        --output-dir augmented/ \\
        --random \\
        --num-random-variants 5 \\
        --seed 42

Usage - SYSTEMATIC MODE (original):
    # Single image with all 23 augmentation types
    python augment_plate_lighting.py \\
        --image car.jpg \\
        --corners-csv corners.csv \\
        --output-dir augmented/
    
    # Batch process entire dataset
    python augment_plate_lighting.py \\
        --input-csv dataset.csv \\
        --output-dir augmented_dataset/
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
import random


class LicensePlateAugmenter:
    """
    Apply realistic lighting transformations to license plates
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize augmenter
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def extract_plate(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        output_size: Tuple[int, int] = (512, 256)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and flatten license plate using homography
        
        Args:
            image: Source image (BGR format)
            corners: 4 corners in order [TL, TR, BR, BL] as (4, 2) array
            output_size: Output size (width, height)
            
        Returns:
            (flattened_plate, homography_matrix)
        """
        # Define destination rectangle (flattened plate)
        width, height = output_size
        dst_corners = np.array([
            [0, 0],           # Top-left
            [width, 0],       # Top-right
            [width, height],  # Bottom-right
            [0, height]       # Bottom-left
        ], dtype=np.float32)
        
        # Compute homography matrix
        H, _ = cv2.findHomography(corners.astype(np.float32), dst_corners)
        
        # Warp perspective to flatten plate
        flattened = cv2.warpPerspective(image, H, output_size)
        
        return flattened, H
    
    def reproject_plate(
        self,
        background: np.ndarray,
        augmented_plate: np.ndarray,
        H_inverse: np.ndarray,
        corners: np.ndarray,
        blend_mode: str = 'alpha'
    ) -> np.ndarray:
        """
        Re-project augmented plate back onto original image
        
        Args:
            background: Original image
            augmented_plate: Transformed plate
            H_inverse: Inverse homography matrix
            corners: Original corner positions
            blend_mode: Blending mode ('alpha', 'poisson', 'feather')
            
        Returns:
            Image with re-projected plate
        """
        h, w = background.shape[:2]
        
        # Warp augmented plate back to original perspective
        warped = cv2.warpPerspective(
            augmented_plate,
            H_inverse,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Create mask for blending
        mask = np.zeros((h, w), dtype=np.uint8)
        corners_int = corners.astype(np.int32)
        cv2.fillConvexPoly(mask, corners_int, 255)
        
        if blend_mode == 'alpha':
            # Simple alpha blending
            result = background.copy()
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = (background * (1 - mask_3ch) + warped * mask_3ch).astype(np.uint8)
            
        elif blend_mode == 'feather':
            # Feathered edge blending
            # Erode mask slightly and blur for soft edges
            kernel = np.ones((5, 5), np.uint8)
            mask_eroded = cv2.erode(mask, kernel, iterations=2)
            mask_feathered = cv2.GaussianBlur(mask_eroded, (15, 15), 5)
            
            result = background.copy()
            mask_3ch = cv2.cvtColor(mask_feathered, cv2.COLOR_GRAY2BGR) / 255.0
            result = (background * (1 - mask_3ch) + warped * mask_3ch).astype(np.uint8)
            
        elif blend_mode == 'poisson':
            # Poisson blending (seamless cloning)
            try:
                # Get center of plate region
                center = tuple(corners.mean(axis=0).astype(np.int32))
                result = cv2.seamlessClone(warped, background, mask, center, cv2.NORMAL_CLONE)
            except:
                # Fallback to alpha blending
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                result = (background * (1 - mask_3ch) + warped * mask_3ch).astype(np.uint8)
        else:
            raise ValueError(f"Unknown blend_mode: {blend_mode}")
        
        return result
    
    # =========================================================================
    # Lighting Transformations
    # =========================================================================
    
    def apply_brightness(self, plate: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust brightness
        
        Args:
            plate: Input plate image
            factor: Brightness factor (0.5 = darker, 1.5 = brighter)
        """
        hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def apply_directional_shadow(
        self,
        plate: np.ndarray,
        angle: float = 45,
        intensity: float = 0.3,
        softness: float = 0.5
    ) -> np.ndarray:
        """
        Apply directional shadow (simulates overhead object casting shadow)
        
        Args:
            plate: Input plate image
            angle: Shadow angle in degrees (0 = right, 90 = down)
            intensity: Shadow darkness (0-1)
            softness: Shadow edge softness (0-1)
        """
        h, w = plate.shape[:2]
        
        # Create gradient shadow mask
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Rotate gradient based on angle
        angle_rad = np.radians(angle)
        rotated = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        
        # Create shadow gradient
        shadow = np.clip(rotated, 0, 1)
        shadow = 1 - (shadow * intensity)
        
        # Apply softness (blur)
        if softness > 0:
            kernel_size = int(15 * softness)
            if kernel_size % 2 == 0:
                kernel_size += 1
            shadow = cv2.GaussianBlur(shadow, (kernel_size, kernel_size), 0)
        
        # Apply shadow
        result = plate.copy().astype(np.float32)
        result *= shadow[:, :, np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_specular_reflection(
        self,
        plate: np.ndarray,
        position: Tuple[float, float] = (0.5, 0.3),
        size: float = 0.3,
        intensity: float = 0.6
    ) -> np.ndarray:
        """
        Apply specular reflection/glare (simulates light source reflection)
        
        Args:
            plate: Input plate image
            position: Reflection center (normalized 0-1)
            size: Reflection size (0-1)
            intensity: Reflection brightness (0-1)
        """
        h, w = plate.shape[:2]
        
        # Create reflection mask (radial gradient)
        center_x = int(w * position[0])
        center_y = int(h * position[1])
        
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize and create falloff
        max_dist = max(w, h) * size
        reflection = np.clip(1 - (dist / max_dist), 0, 1)
        reflection = reflection ** 2  # Sharper falloff
        
        # Add some noise for realism
        noise = np.random.rand(h, w) * 0.1
        reflection = np.clip(reflection + noise, 0, 1)
        
        # Apply reflection
        result = plate.copy().astype(np.float32)
        white = np.array([255, 255, 255], dtype=np.float32)
        
        for i in range(3):
            result[:, :, i] = result[:, :, i] * (1 - reflection * intensity) + \
                            white[i] * reflection * intensity
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_color_temperature(
        self,
        plate: np.ndarray,
        temperature: str = 'warm'
    ) -> np.ndarray:
        """
        Adjust color temperature
        
        Args:
            plate: Input plate image
            temperature: 'warm' (orange/yellow), 'cool' (blue), or 'neutral'
        """
        result = plate.copy().astype(np.float32)
        
        if temperature == 'warm':
            # Increase red/yellow (sunset, tungsten light)
            result[:, :, 2] *= 1.2  # More red
            result[:, :, 1] *= 1.1  # Slight green boost
            result[:, :, 0] *= 0.9  # Less blue
        elif temperature == 'cool':
            # Increase blue (overcast, LED light)
            result[:, :, 0] *= 1.2  # More blue
            result[:, :, 1] *= 1.0  # Keep green
            result[:, :, 2] *= 0.9  # Less red
        elif temperature == 'neutral':
            # Slight desaturation
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            result = cv2.addWeighted(result, 0.8, 
                                    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32), 
                                    0.2, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_dirt_weathering(
        self,
        plate: np.ndarray,
        intensity: float = 0.3
    ) -> np.ndarray:
        """
        Apply dirt/weathering effects
        
        Args:
            plate: Input plate image
            intensity: Dirt intensity (0-1)
        """
        h, w = plate.shape[:2]
        
        # Create dirt texture using Perlin-like noise
        # Simple approach: multiple scales of random noise
        dirt = np.zeros((h, w), dtype=np.float32)
        
        for scale in [2, 4, 8, 16]:
            h_scaled = h // scale
            w_scaled = w // scale
            noise = np.random.rand(h_scaled, w_scaled)
            noise_resized = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
            dirt += noise_resized / scale
        
        # Normalize
        dirt = (dirt - dirt.min()) / (dirt.max() - dirt.min())
        
        # Add some structure (edges get more dirt)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_blurred = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0) / 255.0
        dirt = np.clip(dirt + edges_blurred * 0.3, 0, 1)
        
        # Apply dirt (darkening)
        result = plate.copy().astype(np.float32)
        dirt_effect = 1 - (dirt * intensity)
        result *= dirt_effect[:, :, np.newaxis]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_rain_drops(
        self,
        plate: np.ndarray,
        num_drops: int = 10,
        intensity: float = 0.5
    ) -> np.ndarray:
        """
        Apply rain drop effects
        
        Args:
            plate: Input plate image
            num_drops: Number of rain drops
            intensity: Drop visibility (0-1)
        """
        h, w = plate.shape[:2]
        result = plate.copy()
        
        for _ in range(num_drops):
            # Random drop position and size
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            radius = random.randint(3, 8)
            
            # Create drop mask (ellipse with slight blur)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (x, y), (radius, int(radius * 1.2)), 
                       random.randint(0, 180), 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Apply water effect (slight darkening and blur)
            drop_region = cv2.GaussianBlur(result, (3, 3), 0)
            drop_region = (drop_region * 0.9).astype(np.uint8)
            
            # Blend
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0 * intensity
            result = (result * (1 - mask_3ch) + drop_region * mask_3ch).astype(np.uint8)
        
        return result
    
    def apply_saturation(
        self,
        plate: np.ndarray,
        factor: float = 1.5
    ) -> np.ndarray:
        """
        Adjust color saturation
        
        Args:
            plate: Input plate image
            factor: Saturation factor (0 = grayscale, 1 = original, >1 = more saturated)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation channel
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def apply_contrast(
        self,
        plate: np.ndarray,
        factor: float = 1.5
    ) -> np.ndarray:
        """
        Adjust image contrast
        
        Args:
            plate: Input plate image
            factor: Contrast factor (0.5 = low contrast, 1 = original, 2 = high contrast)
        """
        # Convert to float
        result = plate.astype(np.float32)
        
        # Calculate mean for each channel
        mean = result.mean(axis=(0, 1))
        
        # Apply contrast adjustment: (pixel - mean) * factor + mean
        result = (result - mean) * factor + mean
        
        # Clip to valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def apply_tree_shadow(
        self,
        plate: np.ndarray,
        density: float = 0.5,
        complexity: int = 3
    ) -> np.ndarray:
        """
        Apply organic tree/foliage shadow pattern
        
        Args:
            plate: Input plate image
            density: Shadow coverage (0-1)
            complexity: Number of shadow layers (more = more realistic)
        """
        h, w = plate.shape[:2]
        
        # Create organic shadow pattern using multiple overlapping shapes
        shadow_mask = np.ones((h, w), dtype=np.float32)
        
        for layer in range(complexity):
            # Create random organic shapes (simulating leaves/branches)
            num_shapes = int(20 * density * (layer + 1))
            
            for _ in range(num_shapes):
                # Random position
                center_x = random.randint(-w//4, w + w//4)
                center_y = random.randint(-h//4, h + h//4)
                
                # Random organic shape (ellipse with random rotation)
                size_x = random.randint(w//10, w//4)
                size_y = random.randint(h//15, h//6)
                angle = random.randint(0, 180)
                
                # Create temporary mask for this shape
                temp_mask = np.ones((h, w), dtype=np.float32)
                cv2.ellipse(temp_mask, (center_x, center_y), (size_x, size_y),
                           angle, 0, 360, 0.0, -1)
                
                # Blend with existing shadow
                shadow_mask = np.minimum(shadow_mask, temp_mask)
        
        # Add some noise for leaf texture
        noise = np.random.rand(h, w) * 0.15
        shadow_mask = np.clip(shadow_mask + noise, 0, 1)
        
        # Blur for soft edges (sunlight diffusion through leaves)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
        
        # Ensure shadows don't completely black out (sunlight still penetrates)
        shadow_intensity = 0.3 + (1 - density) * 0.3  # Varies from 0.3 to 0.6
        shadow_mask = shadow_mask * (1 - shadow_intensity) + shadow_intensity
        
        # Apply shadow
        result = plate.copy().astype(np.float32)
        result *= shadow_mask[:, :, np.newaxis]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_hdr_local_contrast(
        self,
        plate: np.ndarray,
        strength: float = 1.5
    ) -> np.ndarray:
        """
        Apply HDR-style local contrast enhancement
        
        Args:
            plate: Input plate image
            strength: Contrast strength (1.0 = no change, 2.0 = strong)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=strength, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l.astype(np.uint8)).astype(np.float32)
        
        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result
    
    def apply_time_of_day(
        self,
        plate: np.ndarray,
        time: str = 'noon'
    ) -> np.ndarray:
        """
        Apply time-of-day lighting
        
        Args:
            plate: Input plate image
            time: 'sunrise', 'noon', 'sunset', 'night', 'overcast'
        """
        if time == 'sunrise':
            # Warm, low brightness
            plate = self.apply_color_temperature(plate, 'warm')
            plate = self.apply_brightness(plate, 0.8)
            plate = self.apply_directional_shadow(plate, angle=70, intensity=0.2)
            
        elif time == 'noon':
            # Bright, neutral, harsh shadows
            plate = self.apply_brightness(plate, 1.2)
            plate = self.apply_directional_shadow(plate, angle=90, intensity=0.4, softness=0.2)
            
        elif time == 'sunset':
            # Very warm, medium brightness
            plate = self.apply_color_temperature(plate, 'warm')
            plate = self.apply_brightness(plate, 0.9)
            plate = self.apply_directional_shadow(plate, angle=110, intensity=0.3)
            
        elif time == 'night':
            # Dark, cool, possible artificial light
            plate = self.apply_brightness(plate, 0.5)
            plate = self.apply_color_temperature(plate, 'cool')
            # Add slight specular (street light)
            if random.random() > 0.5:
                plate = self.apply_specular_reflection(
                    plate, 
                    position=(random.uniform(0.2, 0.8), random.uniform(0.1, 0.4)),
                    size=0.2,
                    intensity=0.4
                )
        
        elif time == 'overcast':
            # Diffuse light, no harsh shadows, slightly cool
            plate = self.apply_color_temperature(plate, 'cool')
            plate = self.apply_brightness(plate, 0.9)
        
        return plate
    
    # =========================================================================
    # Preset Augmentation Pipelines
    # =========================================================================
    
    def get_all_augmentations(self, randomize_params: bool = False) -> List[Tuple[str, callable]]:
        """
        Get all available augmentations as a list
        
        Args:
            randomize_params: If True, use random parameters for each augmentation
        
        Returns:
            List of (name, function) tuples
        """
        if randomize_params:
            # Random parameters for variety
            return [
                ('bright', lambda p: self.apply_brightness(p, random.uniform(1.2, 1.6))),
                ('dark', lambda p: self.apply_brightness(p, random.uniform(0.4, 0.8))),
                ('shadow_light', lambda p: self.apply_directional_shadow(
                    p, random.uniform(0, 180), random.uniform(0.2, 0.4), random.uniform(0.4, 0.6)
                )),
                ('shadow_heavy', lambda p: self.apply_directional_shadow(
                    p, random.uniform(0, 180), random.uniform(0.4, 0.6), random.uniform(0.2, 0.4)
                )),
                ('glare_center', lambda p: self.apply_specular_reflection(
                    p, (random.uniform(0.3, 0.7), random.uniform(0.2, 0.4)), 
                    random.uniform(0.2, 0.4), random.uniform(0.5, 0.7)
                )),
                ('glare_side', lambda p: self.apply_specular_reflection(
                    p, (random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)), 
                    random.uniform(0.2, 0.35), random.uniform(0.4, 0.6)
                )),
                ('warm', lambda p: self.apply_color_temperature(p, 'warm')),
                ('cool', lambda p: self.apply_color_temperature(p, 'cool')),
                ('neutral', lambda p: self.apply_color_temperature(p, 'neutral')),
                ('dirt_light', lambda p: self.apply_dirt_weathering(p, random.uniform(0.2, 0.35))),
                ('dirt_heavy', lambda p: self.apply_dirt_weathering(p, random.uniform(0.4, 0.6))),
                ('rain_light', lambda p: self.apply_rain_drops(p, random.randint(3, 7), random.uniform(0.3, 0.5))),
                ('rain_heavy', lambda p: self.apply_rain_drops(p, random.randint(12, 20), random.uniform(0.5, 0.7))),
                ('tree_shadow_light', lambda p: self.apply_tree_shadow(p, random.uniform(0.2, 0.4), random.randint(2, 3))),
                ('tree_shadow_medium', lambda p: self.apply_tree_shadow(p, random.uniform(0.4, 0.6), random.randint(3, 4))),
                ('tree_shadow_heavy', lambda p: self.apply_tree_shadow(p, random.uniform(0.6, 0.8), random.randint(4, 5))),
                ('hdr_light', lambda p: self.apply_hdr_local_contrast(p, random.uniform(1.2, 1.5))),
                ('hdr_strong', lambda p: self.apply_hdr_local_contrast(p, random.uniform(1.8, 2.2))),
                ('saturation_low', lambda p: self.apply_saturation(p, random.uniform(0.3, 0.6))),
                ('saturation_high', lambda p: self.apply_saturation(p, random.uniform(1.4, 1.8))),
                ('contrast_low', lambda p: self.apply_contrast(p, random.uniform(0.5, 0.8))),
                ('contrast_high', lambda p: self.apply_contrast(p, random.uniform(1.4, 1.8))),
                ('sunrise', lambda p: self.apply_time_of_day(p, 'sunrise')),
                ('noon', lambda p: self.apply_time_of_day(p, 'noon')),
                ('sunset', lambda p: self.apply_time_of_day(p, 'sunset')),
                ('night', lambda p: self.apply_time_of_day(p, 'night')),
                ('overcast', lambda p: self.apply_time_of_day(p, 'overcast')),
            ]
        else:
            # Fixed parameters for consistency
            return [
                ('bright', lambda p: self.apply_brightness(p, 1.4)),
                ('dark', lambda p: self.apply_brightness(p, 0.6)),
                ('shadow_light', lambda p: self.apply_directional_shadow(p, 45, 0.3, 0.5)),
                ('shadow_heavy', lambda p: self.apply_directional_shadow(p, 90, 0.5, 0.3)),
                ('glare_center', lambda p: self.apply_specular_reflection(p, (0.5, 0.3), 0.3, 0.6)),
                ('glare_side', lambda p: self.apply_specular_reflection(p, (0.2, 0.2), 0.25, 0.5)),
                ('warm', lambda p: self.apply_color_temperature(p, 'warm')),
                ('cool', lambda p: self.apply_color_temperature(p, 'cool')),
                ('neutral', lambda p: self.apply_color_temperature(p, 'neutral')),
                ('dirt_light', lambda p: self.apply_dirt_weathering(p, 0.25)),
                ('dirt_heavy', lambda p: self.apply_dirt_weathering(p, 0.5)),
                ('rain_light', lambda p: self.apply_rain_drops(p, 5, 0.4)),
                ('rain_heavy', lambda p: self.apply_rain_drops(p, 15, 0.6)),
                ('tree_shadow_light', lambda p: self.apply_tree_shadow(p, 0.3, 2)),
                ('tree_shadow_medium', lambda p: self.apply_tree_shadow(p, 0.5, 3)),
                ('tree_shadow_heavy', lambda p: self.apply_tree_shadow(p, 0.7, 4)),
                ('hdr_light', lambda p: self.apply_hdr_local_contrast(p, 1.3)),
                ('hdr_strong', lambda p: self.apply_hdr_local_contrast(p, 2.0)),
                ('saturation_low', lambda p: self.apply_saturation(p, 0.5)),
                ('saturation_high', lambda p: self.apply_saturation(p, 1.6)),
                ('contrast_low', lambda p: self.apply_contrast(p, 0.7)),
                ('contrast_high', lambda p: self.apply_contrast(p, 1.6)),
                ('sunrise', lambda p: self.apply_time_of_day(p, 'sunrise')),
                ('noon', lambda p: self.apply_time_of_day(p, 'noon')),
                ('sunset', lambda p: self.apply_time_of_day(p, 'sunset')),
                ('night', lambda p: self.apply_time_of_day(p, 'night')),
                ('overcast', lambda p: self.apply_time_of_day(p, 'overcast')),
            ]
    
    def augment_all(self, plate: np.ndarray, randomize_params: bool = False) -> List[Tuple[np.ndarray, str]]:
        """
        Apply all augmentations systematically (no duplicates)
        
        Args:
            randomize_params: If True, use random parameters for each augmentation
        
        Returns:
            List of (augmented_plate, description) tuples
        """
        augmentations = self.get_all_augmentations(randomize_params)
        results = []
        
        for name, transform in augmentations:
            augmented = transform(plate)
            results.append((augmented, name))
        
        return results
    
    def augment_random_combination(
        self,
        plate: np.ndarray,
        prob_per_filter: float = 0.3,
        min_filters: int = 1,
        max_filters: int = 5
    ) -> Tuple[np.ndarray, str]:
        """
        Apply random combination of filters with random parameters
        
        Args:
            plate: Input plate image
            prob_per_filter: Probability of applying each filter (0-1)
            min_filters: Minimum number of filters to apply
            max_filters: Maximum number of filters to apply
        
        Returns:
            (augmented_plate, description of applied filters)
        """
        # Get all augmentations with random parameters
        all_augmentations = self.get_all_augmentations(randomize_params=True)
        
        # Randomly select which filters to apply
        applied_filters = []
        result = plate.copy()
        
        for name, transform in all_augmentations:
            # Apply with probability
            if random.random() < prob_per_filter:
                result = transform(result)
                applied_filters.append(name)
        
        # Ensure minimum number of filters
        while len(applied_filters) < min_filters:
            name, transform = random.choice(all_augmentations)
            if name not in applied_filters:
                result = transform(result)
                applied_filters.append(name)
        
        # Limit to maximum number of filters
        if len(applied_filters) > max_filters:
            # If we have too many, start over with just max_filters
            applied_filters = []
            result = plate.copy()
            
            # Randomly select max_filters
            selected = random.sample(all_augmentations, max_filters)
            for name, transform in selected:
                result = transform(result)
                applied_filters.append(name)
        
        # Create description
        if len(applied_filters) == 0:
            description = "original"
        else:
            description = "+".join(applied_filters)
        
        return result, description
    
    def augment_random(self, plate: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply random augmentation pipeline (DEPRECATED - use augment_all instead)
        
        Returns:
            (augmented_plate, description)
        """
        augmentations = self.get_all_augmentations()
        name, transform = random.choice(augmentations)
        augmented = transform(plate)
        
        return augmented, name
    
    def augment_preset(self, plate: np.ndarray, preset: str) -> np.ndarray:
        """
        Apply specific preset augmentation
        
        Args:
            plate: Input plate
            preset: Preset name
        """
        augmentations = dict(self.get_all_augmentations())
        
        if preset not in augmentations:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(augmentations.keys())}")
        
        return augmentations[preset](plate)


def process_single_image(
    image_path: str,
    corners_csv: str,
    output_dir: str,
    num_variants: int = None,  # Now optional - defaults to all
    blend_mode: str = 'feather',
    save_extracted: bool = False
) -> List[str]:
    """
    Process single image with systematic lighting variants (one of each type)
    
    Args:
        image_path: Path to input image
        corners_csv: CSV with corner coordinates
        output_dir: Output directory
        num_variants: Number of variants (None = all, or specific count)
        blend_mode: Blending mode for re-projection
        save_extracted: Save extracted plates
        
    Returns:
        List of output file paths
    """
    # Load corners CSV
    df = pd.read_csv(corners_csv)
    
    # Find matching row
    image_name = Path(image_path).name
    matching = df[df['image_path'].str.contains(image_name)]
    
    if len(matching) == 0:
        raise ValueError(f"No corners found for {image_name} in CSV")
    
    # Use first match
    row = matching.iloc[0]
    
    # Extract corners
    corners = np.array([
        [row['top_left_x'], row['top_left_y']],
        [row['top_right_x'], row['top_right_y']],
        [row['bottom_right_x'], row['bottom_right_y']],
        [row['bottom_left_x'], row['bottom_left_y']]
    ])
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Initialize augmenter
    augmenter = LicensePlateAugmenter()
    
    # Extract plate
    plate, H = augmenter.extract_plate(image, corners)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save extracted plate if requested
    if save_extracted:
        extracted_path = output_path / f"{Path(image_path).stem}_extracted.png"
        cv2.imwrite(str(extracted_path), plate)
    
    # Generate ALL variants systematically
    H_inv = np.linalg.inv(H)
    all_augmentations = augmenter.augment_all(plate)
    
    # Limit if num_variants specified
    if num_variants is not None:
        all_augmentations = all_augmentations[:num_variants]
    
    output_files = []
    base_name = Path(image_path).stem
    
    for i, (augmented_plate, aug_name) in enumerate(all_augmentations):
        # Re-project onto original image
        result = augmenter.reproject_plate(image, augmented_plate, H_inv, corners, blend_mode)
        
        # Save
        output_file = output_path / f"{base_name}_aug_{i:03d}_{aug_name}.jpg"
        cv2.imwrite(str(output_file), result)
        output_files.append(str(output_file))
    
    print(f"Generated {len(output_files)} variants (one of each type)")
    
    return output_files


def process_single_image_random(
    image_path: str,
    corners_csv: str,
    output_dir: str,
    num_variants: int = 10,
    prob_per_filter: float = 0.3,
    min_filters: int = 1,
    max_filters: int = 5,
    blend_mode: str = 'feather',
    save_extracted: bool = False
) -> List[str]:
    """
    Process single image with random filter combinations
    
    Args:
        image_path: Path to input image
        corners_csv: CSV with corner coordinates
        output_dir: Output directory
        num_variants: Number of random variants to generate
        prob_per_filter: Probability of applying each filter (0-1)
        min_filters: Minimum filters per variant
        max_filters: Maximum filters per variant
        blend_mode: Blending mode for re-projection
        save_extracted: Save extracted plates
        
    Returns:
        List of output file paths
    """
    # Load corners CSV
    df = pd.read_csv(corners_csv)
    
    # Find matching row
    image_name = Path(image_path).name
    matching = df[df['image_path'].str.contains(image_name)]
    
    if len(matching) == 0:
        raise ValueError(f"No corners found for {image_name} in CSV")
    
    # Use first match
    row = matching.iloc[0]
    
    # Extract corners
    corners = np.array([
        [row['top_left_x'], row['top_left_y']],
        [row['top_right_x'], row['top_right_y']],
        [row['bottom_right_x'], row['bottom_right_y']],
        [row['bottom_left_x'], row['bottom_left_y']]
    ])
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Initialize augmenter
    augmenter = LicensePlateAugmenter()
    
    # Extract plate
    plate, H = augmenter.extract_plate(image, corners)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save extracted plate if requested
    if save_extracted:
        extracted_path = output_path / f"{Path(image_path).stem}_extracted.png"
        cv2.imwrite(str(extracted_path), plate)
    
    # Generate random variants
    H_inv = np.linalg.inv(H)
    output_files = []
    base_name = Path(image_path).stem
    
    for i in range(num_variants):
        # Generate random combination
        augmented_plate, aug_description = augmenter.augment_random_combination(
            plate,
            prob_per_filter=prob_per_filter,
            min_filters=min_filters,
            max_filters=max_filters
        )
        
        # Re-project onto original image
        result = augmenter.reproject_plate(image, augmented_plate, H_inv, corners, blend_mode)
        
        # Save with truncated description
        desc_short = aug_description.replace("+", "_")[:80]
        output_file = output_path / f"{base_name}_rnd{i:03d}_{desc_short}.jpg"
        cv2.imwrite(str(output_file), result)
        output_files.append(str(output_file))
    
    print(f"Generated {len(output_files)} random variants with filter combinations")
    
    return output_files


def process_batch(
    input_csv: str,
    output_dir: str,
    variants_per_image: int = None,  # None = all variants
    blend_mode: str = 'feather',
    max_images: Optional[int] = None
) -> pd.DataFrame:
    """
    Batch process images from CSV with systematic augmentations (one of each type)
    
    Args:
        input_csv: CSV with corner coordinates
        output_dir: Output directory
        variants_per_image: Number of variants per image (None = all)
        blend_mode: Blending mode
        max_images: Maximum images to process
        
    Returns:
        DataFrame with augmentation results
    """
    # Load CSV
    df = pd.read_csv(input_csv)
    
    # Get unique images
    unique_images = df['image_path'].unique()
    
    if max_images is not None:
        unique_images = unique_images[:max_images]
    
    augmenter = LicensePlateAugmenter()
    
    # Get total number of augmentations available
    total_augmentations = len(augmenter.get_all_augmentations())
    
    if variants_per_image is None:
        variants_per_image = total_augmentations
        print(f"Processing {len(unique_images)} images with ALL {total_augmentations} augmentation types...")
    else:
        print(f"Processing {len(unique_images)} images with {variants_per_image} augmentation types...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for img_path in tqdm(unique_images, desc="Processing images"):
        try:
            # Get all detections for this image
            img_detections = df[df['image_path'] == img_path]
            
            # Load image once
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not load {img_path}")
                continue
            
            # Process each detection (plate) in the image
            for idx, row in img_detections.iterrows():
                # Extract corners
                corners = np.array([
                    [row['top_left_x'], row['top_left_y']],
                    [row['top_right_x'], row['top_right_y']],
                    [row['bottom_right_x'], row['bottom_right_y']],
                    [row['bottom_left_x'], row['bottom_left_y']]
                ])
                
                # Extract plate
                plate, H = augmenter.extract_plate(image, corners)
                H_inv = np.linalg.inv(H)
                
                base_name = Path(img_path).stem
                det_idx = row['detection_idx']
                
                # Generate ALL variants systematically
                all_augmentations = augmenter.augment_all(plate)
                
                # Limit if specified
                if variants_per_image is not None:
                    all_augmentations = all_augmentations[:variants_per_image]
                
                # Process each augmentation
                for var_idx, (augmented_plate, aug_name) in enumerate(all_augmentations):
                    # Re-project
                    result = augmenter.reproject_plate(
                        image, augmented_plate, H_inv, corners, blend_mode
                    )
                    
                    # Save
                    output_file = output_path / f"{base_name}_det{det_idx}_var{var_idx:03d}_{aug_name}.jpg"
                    cv2.imwrite(str(output_file), result)
                    
                    # Record result
                    results.append({
                        'original_image': img_path,
                        'augmented_image': str(output_file),
                        'detection_idx': det_idx,
                        'variant_idx': var_idx,
                        'augmentation': aug_name,
                    })
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results CSV
    results_csv = output_path / "augmentation_results.csv"
    results_df.to_csv(results_csv, index=False)
    
    print(f"\n✓ Processed {len(unique_images)} images")
    print(f"✓ Generated {len(results)} augmented images")
    print(f"✓ Augmentation types used: {variants_per_image if variants_per_image else total_augmentations}")
    print(f"✓ Results saved to: {results_csv}")
    
    return results_df


def process_batch_random_combinations(
    input_csv: str,
    output_dir: str,
    num_variants: int = 20,
    prob_per_filter: float = 0.3,
    min_filters: int = 1,
    max_filters: int = 5,
    blend_mode: str = 'feather',
    max_images: Optional[int] = None
) -> pd.DataFrame:
    """
    Batch process images with RANDOM COMBINATIONS of filters
    
    Each variant gets a random combination of filters with random parameters.
    This generates highly diverse augmentations.
    
    Args:
        input_csv: CSV with corner coordinates
        output_dir: Output directory
        num_variants: Number of random variants per image
        prob_per_filter: Probability of applying each filter (0-1)
        min_filters: Minimum filters per variant
        max_filters: Maximum filters per variant
        blend_mode: Blending mode
        max_images: Maximum images to process
        
    Returns:
        DataFrame with augmentation results
    """
    # Load CSV
    df = pd.read_csv(input_csv)
    
    # Get unique images
    unique_images = df['image_path'].unique()
    
    if max_images is not None:
        unique_images = unique_images[:max_images]
    
    print(f"Processing {len(unique_images)} images with {num_variants} random combinations each...")
    print(f"Filter probability: {prob_per_filter} | Min filters: {min_filters} | Max filters: {max_filters}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    augmenter = LicensePlateAugmenter()
    
    results = []
    
    for img_path in tqdm(unique_images, desc="Processing images"):
        try:
            # Get all detections for this image
            img_detections = df[df['image_path'] == img_path]
            
            # Load image once
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not load {img_path}")
                continue
            
            # Process each detection (plate) in the image
            for idx, row in img_detections.iterrows():
                # Extract corners
                corners = np.array([
                    [row['top_left_x'], row['top_left_y']],
                    [row['top_right_x'], row['top_right_y']],
                    [row['bottom_right_x'], row['bottom_right_y']],
                    [row['bottom_left_x'], row['bottom_left_y']]
                ])
                
                # Extract plate
                plate, H = augmenter.extract_plate(image, corners)
                H_inv = np.linalg.inv(H)
                
                base_name = Path(img_path).stem
                det_idx = row['detection_idx']
                
                # Generate random combination variants
                for var_idx in range(num_variants):
                    augmented_plate, aug_description = augmenter.augment_random_combination(
                        plate,
                        prob_per_filter=prob_per_filter,
                        min_filters=min_filters,
                        max_filters=max_filters
                    )
                    
                    # Re-project
                    result = augmenter.reproject_plate(
                        image, augmented_plate, H_inv, corners, blend_mode
                    )
                    
                    # Save with truncated description (filenames can get long)
                    desc_short = aug_description.replace("+", "_")[:100]  # Limit length
                    output_file = output_path / f"{base_name}_det{det_idx}_rnd{var_idx:03d}_{desc_short}.jpg"
                    cv2.imwrite(str(output_file), result)
                    
                    # Record result
                    results.append({
                        'original_image': img_path,
                        'augmented_image': str(output_file),
                        'detection_idx': det_idx,
                        'variant_idx': var_idx,
                        'augmentation': aug_description,  # Full description
                    })
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results CSV
    results_csv = output_path / "augmentation_results.csv"
    results_df.to_csv(results_csv, index=False)
    
    print(f"\n✓ Processed {len(unique_images)} images")
    print(f"✓ Generated {len(results)} augmented images with random filter combinations")
    print(f"✓ Results saved to: {results_csv}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Realistic license plate lighting augmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START EXAMPLES:

RANDOM MODE (new!):
  # 10 random variants for a single image
  python augment_plate_lighting.py \\
      --image car.jpg \\
      --corners-csv corners.csv \\
      --output-dir augmented/ \\
      --random \\
      --num-random-variants 10
  
  # Control filter diversity
  python augment_plate_lighting.py \\
      --image car.jpg \\
      --corners-csv corners.csv \\
      --output-dir augmented/ \\
      --random \\
      --num-random-variants 20 \\
      --filter-prob 0.4 \\
      --min-filters 2 \\
      --max-filters 4
  
  # Batch processing with reproducible results
  python augment_plate_lighting.py \\
      --input-csv dataset.csv \\
      --output-dir augmented/ \\
      --random \\
      --num-random-variants 5 \\
      --seed 42

SYSTEMATIC MODE (original, comprehensive):
  # All 23 augmentation types - one of each
  python augment_plate_lighting.py \\
      --image car.jpg \\
      --corners-csv corners.csv \\
      --output-dir augmented/
  
  # Batch process entire dataset
  python augment_plate_lighting.py \\
      --input-csv dataset.csv \\
      --output-dir augmented_dataset/
  
  # Limit to first 10 augmentation types
  python augment_plate_lighting.py \\
      --input-csv dataset.csv \\
      --output-dir augmented/ \\
      --variants-per-image 10

MODES EXPLAINED:

  SYSTEMATIC (default):
    • Applies each of 23 augmentation types exactly once
    • Reproducible results
    • Comprehensive coverage of lighting scenarios
    • Good for: Systematic dataset augmentation, ensuring variety
    • ~23 images per input image

  RANDOM (--random flag):
    • Each variant gets a random combination of filters
    • Filters applied with random parameters
    • Highly diverse, unpredictable results
    • Good for: Data augmentation, robustness training
    • Customizable via --filter-prob, --min-filters, --max-filters
    • Number of variants: --num-random-variants

RANDOM MODE PARAMETERS:
  --num-random-variants N      How many random variants per image (default: 10)
  --filter-prob P              Probability each filter is applied (0-1, default: 0.3)
  --min-filters N              Minimum filters per variant (default: 1)
  --max-filters N              Maximum filters per variant (default: 5)
  --seed N                     Reproducible randomness (optional)

EXAMPLE PARAMETER COMBINATIONS:
  • Light variations: --filter-prob 0.2 --min-filters 1 --max-filters 2
  • Diverse mix:     --filter-prob 0.35 --min-filters 2 --max-filters 4
  • Heavy effects:   --filter-prob 0.5 --min-filters 3 --max-filters 6
        """
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Single image to process')
    group.add_argument('--input-csv', type=str, help='CSV with corner coordinates for batch processing')
    
    # Corners CSV (required for single image mode)
    parser.add_argument('--corners-csv', type=str, help='CSV with corner coordinates (required for --image)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    # Mode selection
    parser.add_argument('--random', action='store_true',
                       help='Use RANDOM mode (random filter combinations) instead of SYSTEMATIC')
    
    # Systematic mode options
    parser.add_argument('--num-variants', type=int, default=None,
                       help='[SYSTEMATIC] Number of augmentation types to use (default: ALL)')
    parser.add_argument('--variants-per-image', type=int, default=None,
                       help='[SYSTEMATIC] Number of augmentation types per image (batch, default: ALL)')
    
    # Random mode options
    parser.add_argument('--num-random-variants', type=int, default=10,
                       help='[RANDOM] Number of random variants to generate (default: 10)')
    parser.add_argument('--filter-prob', type=float, default=0.3,
                       help='[RANDOM] Probability of applying each filter (0-1, default: 0.3)')
    parser.add_argument('--min-filters', type=int, default=1,
                       help='[RANDOM] Minimum filters per variant (default: 1)')
    parser.add_argument('--max-filters', type=int, default=5,
                       help='[RANDOM] Maximum filters per variant (default: 5)')
    
    # Blending and other options
    parser.add_argument('--blend-mode', type=str, default='feather',
                       choices=['alpha', 'feather', 'poisson'],
                       help='Blending mode for re-projection (default: feather)')
    parser.add_argument('--save-extracted', action='store_true',
                       help='Save extracted plates')
    parser.add_argument('--max-images', type=int, default=None,
                       help='[BATCH] Maximum images to process')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--list-augmentations', action='store_true',
                       help='List all available augmentation types and exit')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # List augmentations and exit
    if args.list_augmentations:
        augmenter = LicensePlateAugmenter()
        augmentations = augmenter.get_all_augmentations()
        
        print("\nAvailable Augmentation Types (SYSTEMATIC mode):")
        print("="*60)
        for i, (name, _) in enumerate(augmentations, 1):
            print(f"  {i:2d}. {name}")
        print(f"\nTotal: {len(augmentations)} augmentation types")
        return
    
    # Validate arguments
    if args.image and not args.corners_csv:
        parser.error("--corners-csv is required when using --image")
    
    # Validate random mode parameters
    if args.random:
        if args.filter_prob < 0 or args.filter_prob > 1:
            parser.error("--filter-prob must be between 0 and 1")
        if args.min_filters < 0:
            parser.error("--min-filters must be >= 0")
        if args.max_filters < args.min_filters:
            parser.error("--max-filters must be >= --min-filters")
    
    mode = "RANDOM" if args.random else "SYSTEMATIC"
    
    print("\n" + "="*70)
    print("LICENSE PLATE LIGHTING AUGMENTATION")
    print(f"MODE: {mode}")
    print("="*70)
    
    if args.image:
        # Single image mode
        augmenter = LicensePlateAugmenter()
        total_augs = len(augmenter.get_all_augmentations())
        
        print(f"\nInput: Single image")
        print(f"Image: {args.image}")
        print(f"Corners CSV: {args.corners_csv}")
        print(f"Output directory: {args.output_dir}")
        print(f"Blend mode: {args.blend_mode}")
        
        if args.random:
            print(f"\nRANDOM MODE Parameters:")
            print(f"  Variants: {args.num_random_variants}")
            print(f"  Filter probability: {args.filter_prob}")
            print(f"  Min filters per variant: {args.min_filters}")
            print(f"  Max filters per variant: {args.max_filters}")
            if args.seed:
                print(f"  Seed: {args.seed}")
            print("="*70)
            
            output_files = process_single_image_random(
                args.image,
                args.corners_csv,
                args.output_dir,
                num_variants=args.num_random_variants,
                prob_per_filter=args.filter_prob,
                min_filters=args.min_filters,
                max_filters=args.max_filters,
                blend_mode=args.blend_mode,
                save_extracted=args.save_extracted
            )
        else:
            print(f"\nSYSTEMATIC MODE Parameters:")
            print(f"  Augmentation types: {args.num_variants if args.num_variants else f'ALL ({total_augs})'}")
            print("="*70)
            
            output_files = process_single_image(
                args.image,
                args.corners_csv,
                args.output_dir,
                num_variants=args.num_variants,
                blend_mode=args.blend_mode,
                save_extracted=args.save_extracted
            )
        
        print(f"\n✓ Generated {len(output_files)} variants")
        print(f"✓ Saved to: {args.output_dir}")
        
    else:
        # Batch mode
        augmenter = LicensePlateAugmenter()
        total_augs = len(augmenter.get_all_augmentations())
        
        print(f"\nInput: Batch processing")
        print(f"Input CSV: {args.input_csv}")
        print(f"Output directory: {args.output_dir}")
        print(f"Blend mode: {args.blend_mode}")
        if args.max_images:
            print(f"Max images: {args.max_images}")
        
        if args.random:
            print(f"\nRANDOM MODE Parameters:")
            print(f"  Variants per image: {args.num_random_variants}")
            print(f"  Filter probability: {args.filter_prob}")
            print(f"  Min filters per variant: {args.min_filters}")
            print(f"  Max filters per variant: {args.max_filters}")
            if args.seed:
                print(f"  Seed: {args.seed}")
            print("="*70 + "\n")
            
            results_df = process_batch_random_combinations(
                args.input_csv,
                args.output_dir,
                num_variants=args.num_random_variants,
                prob_per_filter=args.filter_prob,
                min_filters=args.min_filters,
                max_filters=args.max_filters,
                blend_mode=args.blend_mode,
                max_images=args.max_images
            )
        else:
            print(f"\nSYSTEMATIC MODE Parameters:")
            print(f"  Augmentation types per image: {args.variants_per_image if args.variants_per_image else f'ALL ({total_augs})'}")
            print("="*70 + "\n")
            
            results_df = process_batch(
                args.input_csv,
                args.output_dir,
                variants_per_image=args.variants_per_image,
                blend_mode=args.blend_mode,
                max_images=args.max_images
            )
        
        print("\n" + "="*70)
        print("COMPLETE!")
        print("="*70)


if __name__ == "__main__":
    main()