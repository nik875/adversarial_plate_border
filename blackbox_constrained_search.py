#!/usr/bin/env python3
"""
Black-box Constrained Search: Use CMA-ES to optimize generator latent codes
for black-box adversarial patch generation.

Users extend the BaseBlackBoxOracle class and implement query(image) to return
detected license plate text (or None if no detection). CMA-ES searches the
generator's latent space to find patches that fool the black-box system.
"""
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cma
from tqdm import tqdm
import kornia
import kornia.geometry as K

from progressive_patch import SimplePatchGenerator, FoundationPatchGenerator
from refine_generator import RefinementNetwork


class BaseBlackBoxOracle(ABC):
    """
    Abstract base class for black-box license plate detection oracles.

    Users extend this class and implement the query() method to interface
    with their target detection system.
    """

    @abstractmethod
    def query(self, image: np.ndarray, corners: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Query the black-box detection system with an image.

        Args:
            image: RGB image as numpy array [H, W, 3] in range [0, 255], uint8
            corners: Optional [4, 2] array of ground truth plate corners for IoU-based selection

        Returns:
            Detected license plate text as string, or None if no plate detected
        """
        pass


class BlackBoxPatchOptimizer:
    """
    CMA-ES optimizer for finding adversarial patches in generator latent space.

    Searches over the generator's latent space to find patches that fool a
    black-box detection system. For each candidate z, generates a base patch
    and optionally refines it before testing.
    """

    def __init__(self,
                 generator_checkpoint: str,
                 refinement_checkpoint: Optional[str] = None,
                 generator_type: str = 'simple',
                 device: str = None,
                 test_images_dir: Optional[str] = None,
                 csv_path: Optional[str] = None,
                 target_plate: Optional[str] = None,
                 disruption_mode: bool = True,
                 test_image_subset: Optional[int] = None):
        """
        Args:
            generator_checkpoint: Path to generator .pt file
            refinement_checkpoint: Path to refinement .pt file (optional)
            generator_type: 'simple' or 'foundation'
            device: 'cuda', 'mps', or 'cpu'
            test_images_dir: Directory containing test images with license plates (alternative to csv_path)
            csv_path: CSV file with image paths and corners (alternative to test_images_dir)
            target_plate: Target plate text for impersonation (None for disruption)
            disruption_mode: If True, optimize for detection failure. If False, impersonation.
        """

        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.target_plate = target_plate
        self.disruption_mode = disruption_mode
        self.test_image_subset = test_image_subset

        # Load generator
        print(f"Loading generator from: {generator_checkpoint}")
        self.generator, self.latent_dim = self._load_generator(
            generator_checkpoint, generator_type
        )
        # Use training mode - BatchNorm needs batch statistics for diverse outputs
        self.generator.train()
        print(f"Generator loaded (latent_dim={self.latent_dim})")

        # Load refinement network if provided
        self.refiner = None
        if refinement_checkpoint:
            print(f"Loading refinement network from: {refinement_checkpoint}")
            self.refiner = self._load_refiner(refinement_checkpoint)
            self.refiner.eval()
            print("Refinement network loaded")

        # Load test images
        self.test_images = []
        self.test_corners = []
        if test_images_dir:
            self._load_test_images(test_images_dir)
            print(f"Loaded {len(self.test_images)} test images")
        elif csv_path:
            self._load_test_images_from_csv(csv_path)
            print(f"Loaded {len(self.test_images)} test images from CSV")

    def _load_generator(self, checkpoint_path: str, generator_type: str):
        """Load frozen generator from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['generator_state_dict']

        # Auto-detect generator type from checkpoint if not explicitly specified
        if generator_type == 'simple' and 'vae.encoder.conv_in.weight' in state_dict:
            print("Warning: Detected Foundation generator in checkpoint, auto-switching to 'foundation' type")
            generator_type = 'foundation'
        elif generator_type == 'foundation' and 'vae.encoder.conv_in.weight' not in state_dict:
            print("Warning: Expected Foundation generator but checkpoint appears to be Simple, auto-switching to 'simple' type")
            generator_type = 'simple'

        # Extract latent_dim
        if 'basis_dim' in checkpoint:
            latent_dim = checkpoint['basis_dim']
        else:
            # Infer from state dict
            if generator_type == 'simple':
                first_layer_key = 'network.0.weight'
                latent_dim = state_dict[first_layer_key].shape[1]
            else:  # foundation
                first_layer_key = 'adapter.0.weight'
                latent_dim = state_dict[first_layer_key].shape[1]

        print(f"Detected generator type: {generator_type}, latent_dim={latent_dim}")

        # Initialize generator
        if generator_type == 'simple':
            generator = SimplePatchGenerator(
                latent_dim=latent_dim,
                patch_height=256,
                patch_width=512
            )
        elif generator_type == 'foundation':
            generator = FoundationPatchGenerator(
                latent_dim=latent_dim,
                patch_height=256,
                patch_width=512
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

        generator.load_state_dict(state_dict)
        generator.to(self.device)

        return generator, latent_dim

    def _load_refiner(self, checkpoint_path: str):
        """Load refinement network from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        latent_dim = checkpoint['basis_dim']
        use_latent_context = checkpoint.get('use_latent_context', False)
        patch_height, patch_width = checkpoint['patch_size']

        refiner = RefinementNetwork(
            patch_height=patch_height,
            patch_width=patch_width,
            use_latent_context=use_latent_context,
            latent_dim=latent_dim
        )

        refiner.load_state_dict(checkpoint['refinement_state_dict'])
        refiner.to(self.device)

        return refiner

    def _load_test_images(self, test_images_dir: str):
        """
        Load test images and their plate corners.

        Expected format:
        - test_images_dir/
          - image1.jpg
          - image1_corners.txt  (x1,y1 x2,y2 x3,y3 x4,y4)
          - image2.jpg
          - image2_corners.txt
        """
        test_dir = Path(test_images_dir)

        for img_path in sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png")):
            corners_path = img_path.with_suffix('').with_suffix('.txt').with_name(
                img_path.stem + '_corners.txt'
            )

            if not corners_path.exists():
                print(f"Warning: No corners file for {img_path.name}, skipping")
                continue

            # Load image
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)

            # Load corners
            with open(corners_path, 'r') as f:
                corners_text = f.read().strip()
                coords = [float(x) for x in corners_text.replace(',', ' ').split()]
                corners = np.array(coords).reshape(4, 2)

            self.test_images.append(image_np)
            self.test_corners.append(corners)

    def _load_test_images_from_csv(self, csv_path: str):
        """
        Load test images and corners from CSV file (preproc_labels.csv format).

        Expected CSV columns (from AdversarialPatchDataset):
        - 'preprocessed_filename': Path to preprocessed image
        - 'new_p1_x', 'new_p1_y', 'new_p2_x', 'new_p2_y', etc: Corner coordinates
        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            img_path = row['preprocessed_filename']

            if not Path(img_path).exists():
                print(f"Warning: Image not found: {img_path}, skipping")
                continue

            # Load image
            try:
                image = Image.open(img_path).convert('RGB')
                image_np = np.array(image)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}, skipping")
                continue

            # Parse corners from CSV columns
            try:
                corners = np.array([
                    [row['new_p1_x'], row['new_p1_y']],
                    [row['new_p2_x'], row['new_p2_y']],
                    [row['new_p3_x'], row['new_p3_y']],
                    [row['new_p4_x'], row['new_p4_y']]
                ], dtype=np.float32)
            except Exception as e:
                print(f"Warning: Failed to parse corners for {img_path}: {e}, skipping")
                continue

            self.test_images.append(image_np)
            self.test_corners.append(corners)

    def generate_patch(self, z: np.ndarray) -> torch.Tensor:
        """
        Generate patch from latent code z.

        Args:
            z: Latent code [latent_dim] in range [0, 1]

        Returns:
            patch: [3, H, W] tensor in range [0, 1]
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # Generate base patch
            base_patch = self.generator(z_tensor).squeeze(0)

            # Refine if refiner available
            if self.refiner is not None:
                refined_patch = self.refiner(
                    base_patch.unsqueeze(0),
                    z_tensor if hasattr(self.refiner, 'use_latent_context') and self.refiner.use_latent_context else None
                ).squeeze(0)
                return refined_patch
            else:
                return base_patch

    def apply_patch_to_image(self, image: np.ndarray, corners: np.ndarray,
                            patch: torch.Tensor) -> np.ndarray:
        """
        Apply adversarial patch to image.

        Args:
            image: [H, W, 3] numpy array, uint8, range [0, 255]
            corners: [4, 2] plate corner coordinates
            patch: [3, H_patch, W_patch] tensor, range [0, 1]

        Returns:
            patched_image: [H, W, 3] numpy array, uint8, range [0, 255]
        """
        H, W = image.shape[:2]

        # Convert image to tensor
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Convert corners to tensor
        corners_tensor = torch.from_numpy(corners).float().to(self.device)

        # Compute border corners (1.4x scale around plate)
        center = corners_tensor.mean(dim=0)
        border_corners = center.unsqueeze(0) + (corners_tensor - center.unsqueeze(0)) * 1.4
        border_corners = border_corners.unsqueeze(0)

        # Get patch corners
        patch_h, patch_w = patch.shape[1], patch.shape[2]
        src_corners = torch.tensor([
            [0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        # Compute homography
        M_border = K.get_perspective_transform(src_corners, border_corners)
        M_plate = K.get_perspective_transform(src_corners, corners_tensor.unsqueeze(0))

        # Warp patch
        patch_batch = patch.unsqueeze(0)
        warped_patch = K.warp_perspective(
            patch_batch, M_border, dsize=(H, W),
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        # Create mask (border - plate)
        patch_mask = torch.ones(1, 1, patch_h, patch_w, dtype=torch.float32, device=self.device)

        warped_border_mask = K.warp_perspective(
            patch_mask, M_border, dsize=(H, W),
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        warped_plate_mask = K.warp_perspective(
            patch_mask, M_plate, dsize=(H, W),
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        final_mask = torch.clamp(warped_border_mask - warped_plate_mask, 0, 1)
        final_mask = final_mask.expand(-1, 3, -1, -1)

        # Apply patch
        result = image_tensor * (1 - final_mask) + warped_patch * final_mask
        result = torch.clamp(result, 0, 1)

        # Convert back to numpy
        result_np = (result.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        return result_np

    def evaluate_fitness(self, z: np.ndarray, oracle: BaseBlackBoxOracle) -> float:
        """
        Evaluate fitness of latent code z using black-box oracle.

        Args:
            z: Latent code [latent_dim] in range [0, 1]
            oracle: Black-box oracle for querying detection system

        Returns:
            fitness: Scalar fitness value (lower is better for CMA-ES)
        """
        # Generate patch
        patch = self.generate_patch(z)

        if len(self.test_images) == 0:
            raise ValueError("No test images loaded. Provide test_images_dir.")

        # Subsample test images if specified
        if self.test_image_subset is not None and self.test_image_subset < len(self.test_images):
            indices = np.random.choice(len(self.test_images), size=self.test_image_subset, replace=False)
            test_images = [self.test_images[i] for i in indices]
            test_corners = [self.test_corners[i] for i in indices]
        else:
            test_images = self.test_images
            test_corners = self.test_corners

        results = []

        for image, corners in zip(test_images, test_corners):
            # Apply patch
            patched_image = self.apply_patch_to_image(image, corners, patch)

            # Query oracle (pass corners for IoU-based detection selection)
            detected_text = oracle.query(patched_image, corners)

            if self.disruption_mode:
                # Disruption: want no detection (None)
                if detected_text is None:
                    results.append(0.0)  # Perfect
                else:
                    # Partial credit for wrong text
                    if self.target_plate and detected_text != self.target_plate:
                        results.append(0.5)
                    else:
                        results.append(1.0)  # Detected correctly
            else:
                # Impersonation: want target_plate detected
                if detected_text == self.target_plate:
                    results.append(0.0)  # Perfect
                elif detected_text is None:
                    results.append(1.0)  # No detection
                else:
                    # Compute edit distance
                    edit_dist = self._levenshtein_distance(detected_text, self.target_plate)
                    max_len = max(len(detected_text), len(self.target_plate))
                    results.append(edit_dist / max_len if max_len > 0 else 1.0)

        # Return mean fitness across all test images
        return np.mean(results)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def optimize(self, oracle: BaseBlackBoxOracle,
                initial_z: Optional[np.ndarray] = None,
                sigma0: float = 0.3,
                max_iterations: int = 100,
                population_size: Optional[int] = None,
                seed: Optional[int] = None,
                checkpoint_dir: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """
        Optimize latent code using CMA-ES.

        Args:
            oracle: Black-box oracle for querying detection system
            initial_z: Initial latent code (default: random in [0, 1])
            sigma0: Initial standard deviation for CMA-ES
            max_iterations: Maximum number of CMA-ES generations
            population_size: Population size (default: 4 + 3*log(latent_dim))
            seed: Random seed for reproducibility (default: None)
            checkpoint_dir: Directory to save checkpoints at each best fitness (default: None)

        Returns:
            best_z: Best latent code found
            best_fitness: Fitness of best solution
        """
        # Initialize starting point
        if initial_z is None:
            initial_z = np.random.uniform(0, 1, size=self.latent_dim)
        else:
            initial_z = np.clip(initial_z, 0, 1)

        # Configure CMA-ES options
        opts = {
            'bounds': [0, 1],  # Constrain z to [0, 1]
            'verbose': -1,  # Suppress CMA-ES output (we'll use tqdm)
        }

        if population_size is not None:
            opts['popsize'] = population_size

        if seed is not None:
            opts['seed'] = seed

        # Create checkpoint directory if specified
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(initial_z, sigma0, opts)

        print(f"\nStarting CMA-ES optimization:")
        print(f"  Latent dimension: {self.latent_dim}")
        print(f"  Population size: {es.popsize}")
        print(f"  Initial sigma: {sigma0}")
        if seed is not None:
            print(f"  Seed: {seed}")
        print(f"  Mode: {'Disruption' if self.disruption_mode else 'Impersonation'}")
        if not self.disruption_mode:
            print(f"  Target plate: {self.target_plate}")
        print(f"  Test images: {len(self.test_images)}")
        print(f"  Max iterations: {max_iterations}")
        if checkpoint_dir is not None:
            print(f"  Checkpoints: {checkpoint_dir}")
        print()

        iteration = 0
        best_fitness_ever = float('inf')
        best_z_ever = initial_z.copy()

        with tqdm(total=max_iterations, desc="CMA-ES") as pbar:
            while not es.stop() and iteration < max_iterations:
                # Ask for new candidate solutions
                solutions = es.ask()

                # Evaluate fitness for each solution
                fitness_values = []
                for z in solutions:
                    fitness = self.evaluate_fitness(z, oracle)
                    fitness_values.append(fitness)

                # Tell CMA-ES the results
                es.tell(solutions, fitness_values)

                # Track best
                best_idx = np.argmin(fitness_values)
                if fitness_values[best_idx] < best_fitness_ever:
                    best_fitness_ever = fitness_values[best_idx]
                    best_z_ever = solutions[best_idx].copy()

                    # Save checkpoint if checkpoint directory specified
                    if checkpoint_dir is not None:
                        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_iter{iteration:04d}_fitness{best_fitness_ever:.6f}"
                        checkpoint_path.mkdir(exist_ok=True)

                        # Save patch
                        patch_path = checkpoint_path / "patch.png"
                        self.save_patch(best_z_ever, str(patch_path))

                        # Save latent code
                        latent_path = checkpoint_path / "latent.npy"
                        np.save(latent_path, best_z_ever)

                        # Save metadata
                        metadata_path = checkpoint_path / "metadata.txt"
                        with open(metadata_path, 'w') as f:
                            f.write(f"Iteration: {iteration}\n")
                            f.write(f"Fitness: {best_fitness_ever:.6f}\n")
                            f.write(f"Mode: {'Disruption' if self.disruption_mode else 'Impersonation'}\n")
                            if not self.disruption_mode:
                                f.write(f"Target plate: {self.target_plate}\n")

                # Update progress
                iteration += 1
                pbar.update(1)
                pbar.set_postfix({
                    'best_fitness': f'{best_fitness_ever:.4f}',
                    'current_best': f'{fitness_values[best_idx]:.4f}',
                    'sigma': f'{es.sigma:.4f}'
                })

        print(f"\nOptimization complete!")
        print(f"  Best fitness: {best_fitness_ever:.4f}")
        print(f"  Iterations: {iteration}")

        return best_z_ever, best_fitness_ever

    def save_patch(self, z: np.ndarray, output_path: str):
        """
        Generate and save patch for given latent code.

        Args:
            z: Latent code [latent_dim]
            output_path: Path to save patch image
        """
        patch = self.generate_patch(z)

        # Convert to numpy
        patch_np = (patch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Save
        Image.fromarray(patch_np).save(output_path)
        print(f"Patch saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Black-box adversarial patch optimization using CMA-ES'
    )
    parser.add_argument('--generator-checkpoint', required=True,
                       help='Path to generator checkpoint (.pt file)')
    parser.add_argument('--refinement-checkpoint', default=None,
                       help='Path to refinement checkpoint (.pt file, optional)')
    parser.add_argument('--disable-refiner', action='store_true',
                       help='Disable refinement network even if checkpoint provided')
    parser.add_argument('--generator-type', choices=['simple', 'foundation'],
                       default='simple', help='Generator architecture type')
    parser.add_argument('--test-images-dir', default=None,
                       help='Directory with test images and corner annotations (alternative to --csv)')
    parser.add_argument('--csv', default=None,
                       help='CSV file with image paths and corners (alternative to --test-images-dir)')
    parser.add_argument('--test-image-subset', type=int, default=None,
                       help='Sample this many test images per iteration (default: use all)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'mps', 'cpu'],
                       help='Device to use')
    parser.add_argument('--target-plate', default=None,
                       help='Target plate for impersonation (None for disruption mode)')
    parser.add_argument('--sigma0', type=float, default=0.3,
                       help='Initial CMA-ES standard deviation (default: 0.3)')
    parser.add_argument('--max-iterations', type=int, default=100,
                       help='Maximum CMA-ES iterations (default: 100)')
    parser.add_argument('--population-size', type=int, default=None,
                       help='CMA-ES population size (default: 4 + 3*log(latent_dim))')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for CMA-ES reproducibility (default: None)')
    parser.add_argument('--output-patch', default='optimized_patch.png',
                       help='Output path for optimized patch')
    parser.add_argument('--output-latent', default='optimized_latent.npy',
                       help='Output path for optimized latent code')

    args = parser.parse_args()

    # Validate that at least one test image source is provided
    if not args.test_images_dir and not args.csv:
        print("Error: Must provide either --test-images-dir or --csv")
        return

    print("=" * 70)
    print("BLACK-BOX ADVERSARIAL PATCH OPTIMIZATION")
    print("=" * 70)
    print("\nIMPORTANT: You must extend BaseBlackBoxOracle and implement query()")
    print("This is a template - modify the oracle implementation below.\n")

    # Example oracle implementation (users should replace this)
    class ExampleOracle(BaseBlackBoxOracle):
        def query(self, image: np.ndarray) -> Optional[str]:
            """
            Replace this with your actual detection system query.

            Args:
                image: RGB image [H, W, 3], uint8, range [0, 255]

            Returns:
                Detected plate text or None
            """
            # TODO: Implement your black-box query here
            # Example:
            # - Save image to temp file
            # - Call external API / subprocess
            # - Parse response
            # - Return detected text or None

            raise NotImplementedError(
                "You must implement query() method in your oracle class!"
            )

    # Initialize optimizer
    refinement_checkpoint = None if args.disable_refiner else args.refinement_checkpoint
    optimizer = BlackBoxPatchOptimizer(
        generator_checkpoint=args.generator_checkpoint,
        refinement_checkpoint=refinement_checkpoint,
        generator_type=args.generator_type,
        device=args.device,
        test_images_dir=args.test_images_dir,
        csv_path=args.csv,
        target_plate=args.target_plate,
        disruption_mode=(args.target_plate is None),
        test_image_subset=args.test_image_subset
    )

    # Create oracle instance
    oracle = ExampleOracle()

    # Run optimization
    best_z, best_fitness = optimizer.optimize(
        oracle,
        sigma0=args.sigma0,
        max_iterations=args.max_iterations
    )

    # Save results
    optimizer.save_patch(best_z, args.output_patch)
    np.save(args.output_latent, best_z)
    print(f"Latent code saved to: {args.output_latent}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
