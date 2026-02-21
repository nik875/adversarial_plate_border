#!/usr/bin/env python3
"""
cmaes_domain.py — CMA-ES black-box optimisation on any domain config.

Loads a trained generator from a run directory, loads the domain config,
and runs CMA-ES optimising the metric defined in the config.

Usage:
    python framework/scripts/cmaes_domain.py <run_dir> \
        --config framework/configs/classification_resnet50.yaml \
        --maxiter 100

Example:
    python framework/scripts/cmaes_domain.py framework_output/classification_resnet50 \
        --config framework/configs/classification_resnet50.yaml \
        --maxiter 50 --popsize 15 --sigma0 0.3
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description='CMA-ES black-box patch optimisation on any domain.'
    )
    parser.add_argument('run_dir', help='Path to training run directory')
    parser.add_argument('--config', required=True,
                        help='Path to YAML domain config file')
    parser.add_argument('--popsize', type=int, default=20)
    parser.add_argument('--maxiter', type=int, default=100)
    parser.add_argument('--sigma0', type=float, default=0.5)
    parser.add_argument('--n-eval-samples', type=int, default=50,
                        help='Validation samples per CMA-ES iteration')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', default='cmaes_domain_output',
                        help='Output directory')
    parser.add_argument('--device', default=None,
                        help='Device: cpu, cuda, mps')
    args = parser.parse_args()

    try:
        import cma
    except ImportError:
        print("Error: cma not installed. Install with: pip install cma", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load generator ----
    from framework.generator_loader import load_generator, generate_patch_from_z

    device = torch.device(args.device) if args.device else None
    generator, latent_dim, device = load_generator(args.run_dir, device=device)
    generator.eval()

    # ---- Load domain config ----
    from framework.config_loader import load_domain_config

    print(f"\nLoading domain config: {args.config}")
    cfg = load_domain_config(args.config)
    raw = cfg.raw

    # ---- Build evaluation dataset ----
    print("\nBuilding evaluation dataset...")
    cmaes_cfg = raw.get('cmaes', {})
    dataset = cfg.domain_adapter.build_dataset(split='val')
    n_eval = min(args.n_eval_samples, len(dataset))

    print(f"Dataset size: {len(dataset)}, using {n_eval} samples per iteration")

    # ---- Precompute control outputs ----
    print("\nPrecomputing control outputs...")
    # Sample evaluation subset
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    eval_indices = all_indices[:min(cmaes_cfg.get('n_eval_samples', 200), len(dataset))]

    eval_images = []
    for idx in eval_indices:
        item = dataset[idx]
        img = item['image']
        if img.dim() == 3:
            img = img.unsqueeze(0)  # [1, 3, H, W]
        eval_images.append(img.to(device))

    # Generate neutral composites for control
    neutral_composites = []
    for img in eval_images:
        neutral = cfg.strategy.apply_neutral(img)
        neutral_composites.append(neutral)

    # Precompute control metric outputs
    model = cfg.domain_adapter.model
    control_outputs = cfg.metric.precompute_control(
        [n.squeeze(0) for n in neutral_composites], model)

    print(f"Control outputs precomputed for {len(control_outputs)} images")

    # ---- State for CMA-ES ----
    best_metric = [0.0]
    best_z = [None]
    eval_count = [0]
    all_metrics = []
    current_iteration = [0]

    def objective(z: np.ndarray) -> float:
        """CMA-ES fitness function (returns negative metric; CMA-ES minimises)."""
        eval_count[0] += 1

        # Generate patch
        patch = generate_patch_from_z(generator, z, device)  # [3, H, W]
        patch_clamped = torch.clamp(patch, 0.0, 1.0)

        # Sample random subset of evaluation images
        num_use = min(args.n_eval_samples, len(eval_images))
        sampled_idx = random.sample(range(len(eval_images)), num_use)
        sampled_images = [eval_images[i] for i in sampled_idx]
        sampled_control = [control_outputs[i] for i in sampled_idx]

        # Apply patch to each image
        composited = []
        for img in sampled_images:
            comp, _ = cfg.strategy.apply(img, patch_clamped)
            composited.append(comp.squeeze(0))

        # Evaluate
        results = cfg.metric.compute(composited, sampled_control, model)
        primary = results.get('primary', 0.0)

        all_metrics.append(primary)

        if primary > best_metric[0]:
            best_metric[0] = primary
            best_z[0] = z.copy()
            tqdm.write(
                f"  [iter {current_iteration[0]}] new best: primary={primary:.4f} "
                f"success_rate={results.get('success_rate', 0.0):.1%}"
            )

        return -primary  # CMA-ES minimises

    # ---- CMA-ES loop ----
    x0 = np.zeros(latent_dim)
    es = cma.CMAEvolutionStrategy(
        x0, args.sigma0,
        {
            'popsize': args.popsize,
            'maxiter': args.maxiter,
            'verb_disp': 1,
            'verb_log': 0,
            'tolstagnation': np.inf,
            'tolfun': -np.inf,
            'tolflatfitness': np.inf,
            'tolxstagnation': np.inf,
            'tolx': -np.inf,
        }
    )

    print(f"\nStarting CMA-ES optimisation:")
    print(f"  latent_dim={latent_dim}  popsize={args.popsize}  maxiter={args.maxiter}")
    print(f"  Metric: {type(cfg.metric).__name__}")
    print(f"  Strategy: {type(cfg.strategy).__name__}")
    print("=" * 80)

    pbar = tqdm(total=args.maxiter, desc="CMA-ES", unit="iter")

    for iteration in range(args.maxiter):
        current_iteration[0] = iteration
        solutions = es.ask()

        cand_pbar = tqdm(total=len(solutions), desc=f"iter {iteration}",
                         unit="cand", position=1, leave=False)
        fitness_values = []
        for z in solutions:
            fitness_values.append(objective(z))
            cand_pbar.update(1)
        cand_pbar.close()

        es.tell(solutions, fitness_values)

        avg_m = np.mean(all_metrics[-len(solutions):]) if all_metrics else 0.0
        pbar.set_postfix({'best': f'{best_metric[0]:.4f}', 'avg': f'{avg_m:.4f}'})
        pbar.update(1)

    pbar.close()

    # ---- Save results ----
    print(f"\n{'='*80}")
    print(f"DONE.  Best primary metric: {best_metric[0]:.4f}")
    print(f"Total evaluations: {eval_count[0]}")

    if best_z[0] is not None:
        best_patch = generate_patch_from_z(generator, best_z[0], device)
        best_patch_c = torch.clamp(best_patch, 0, 1)

        # Save patch PNG
        patch_np = (best_patch_c.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / "best_patch.png"), patch_bgr)

        # Save latent code
        np.save(str(output_dir / "best_z.npy"), best_z[0])

        print(f"Saved best patch to: {output_dir / 'best_patch.png'}")
        print(f"Saved latent code to: {output_dir / 'best_z.npy'}")

    print(f"All results in: {output_dir}")


if __name__ == '__main__':
    main()
