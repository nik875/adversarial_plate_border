#!/usr/bin/env python3
"""
cmaes_domain.py — CMA-ES black-box optimisation on any domain config.

Loads a trained generator from a run directory, loads the domain config,
and runs CMA-ES maximising the metric defined in the config.

Features (matching the quality of optimize_patch_cmaes.py):
  - Full validation pool loaded once; random subset sampled per iteration
    (all candidates in a given iteration see the same subset)
  - save_best_patch: saves patch.png + composites + results.csv on every new best
  - First-iteration debug output: candidate patches + composites saved to debug/
  - Final full-pool evaluation after optimisation, with results CSV + images
  - optimization_results.txt metadata file

Usage:
    python framework/scripts/cmaes_domain.py <run_dir> \\
        --config framework/configs/classification_resnet50.yaml \\
        --maxiter 100

Example:
    python framework/scripts/cmaes_domain.py framework_output/resnet50 \\
        --config framework/configs/classification_resnet50.yaml \\
        --maxiter 50 --popsize 15 --sigma0 0.3 --n-eval-samples 64
"""
import argparse
import csv
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is on sys.path when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tensor_to_bgr(tensor):
    """Convert [3,H,W] float [0,1] tensor to BGR uint8 numpy array."""
    rgb = (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _composite_to_bgr(composite_tensor):
    """Squeeze optional batch dim then convert to BGR uint8."""
    t = composite_tensor
    if t.dim() == 4:
        t = t.squeeze(0)
    return _tensor_to_bgr(t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CMA-ES black-box patch optimisation on any domain.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('run_dir',
                        help='Path to training run directory (contains generator checkpoint)')
    parser.add_argument('--config', required=True,
                        help='Path to YAML domain config file')

    # CMA-ES
    parser.add_argument('--popsize', type=int, default=20,
                        help='CMA-ES population size')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='CMA-ES maximum iterations')
    parser.add_argument('--sigma0', type=float, default=0.5,
                        help='CMA-ES initial standard deviation')

    # Evaluation
    parser.add_argument('--n-eval-samples', type=int, default=50,
                        help='Validation samples evaluated per CMA-ES iteration '
                             '(subset randomly drawn each iteration)')
    parser.add_argument('--max-val-samples', type=int, default=None,
                        help='Cap on total validation pool loaded from disk '
                             '(default: load everything the dataset provides)')

    # Misc
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--outdir', default='cmaes_domain_output',
                        help='Output directory (cleared on each run)')
    parser.add_argument('--device', default=None,
                        help='Torch device: cpu, cuda, cuda:0, mps …')

    args = parser.parse_args()

    # ---- Dependencies ----
    try:
        import cma
    except ImportError:
        print("Error: cma not installed.  pip install cma", file=sys.stderr)
        sys.exit(1)

    # ---- Reproducibility ----
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # ---- Output directory (fresh each run, matching OCR script behaviour) ----
    output_dir = Path(args.outdir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    best_patches_dir = output_dir / "best_patches"
    best_patches_dir.mkdir()
    debug_dir = output_dir / "debug_output"
    debug_dir.mkdir()

    # ---- Load generator ----
    from framework.generator_loader import load_generator, generate_patch_from_z

    device = torch.device(args.device) if args.device else None
    print(f"\nLoading generator from: {args.run_dir}")
    generator, latent_dim, device = load_generator(args.run_dir, device=device)
    generator.eval()

    # ---- Load domain config ----
    from framework.config_loader import load_domain_config

    print(f"\nLoading domain config: {args.config}")
    cfg = load_domain_config(args.config)

    domain   = cfg.domain_adapter
    strategy = cfg.strategy
    metric   = cfg.metric
    model    = cfg.target_model

    print(f"  Domain:   {type(domain).__name__}")
    print(f"  Strategy: {type(strategy).__name__}")
    print(f"  Metric:   {type(metric).__name__}")

    # ---- Build full validation pool ----
    print("\nBuilding validation pool...")
    dataset = domain.build_dataset(split='val')
    pool_size = len(dataset)
    if args.max_val_samples is not None:
        pool_size = min(pool_size, args.max_val_samples)

    pool_indices = random.sample(range(len(dataset)), pool_size)

    val_images = []          # list of [1, 3, H, W] float tensors on device
    val_images_cpu = []      # same tensors kept on CPU for saving composites later
    for idx in tqdm(pool_indices, desc="Loading val pool"):
        item = dataset[idx]
        img = item['image']
        if img.dim() == 3:
            img = img.unsqueeze(0)   # [1, 3, H, W]
        val_images.append(img.to(device))
        val_images_cpu.append(img.cpu())

    print(f"Validation pool: {len(val_images)} images")

    # ---- Precompute control outputs (neutral composites on full pool) ----
    print("\nPrecomputing control outputs on full validation pool...")
    neutral_composites_cpu = []
    for img in tqdm(val_images_cpu, desc="Neutral composites"):
        neutral = strategy.apply_neutral(img)          # [1, 3, H, W]
        neutral_composites_cpu.append(neutral.squeeze(0).cpu())   # [3, H, W]

    control_outputs = metric.precompute_control(neutral_composites_cpu, model)
    print(f"Control outputs precomputed for {len(control_outputs)} images")

    # ---- CMA-ES state ----
    eval_count       = [0]
    best_fitness     = [0.0]   # best primary metric seen
    best_success_pct = [0.0]   # success_rate at best fitness
    best_z           = [None]
    best_count       = [0]
    current_iter     = [0]
    iter_metrics     = []      # fitness values for current iteration's candidates

    # Shared per-iteration sample (set once per iteration, used by all candidates)
    iter_images   = []
    iter_controls = []

    # ---- save_best_patch ----
    def save_best_patch(z, iteration, fitness, per_image_data, composites_np):
        """Save patch, composites, and per-image CSV whenever a new best is found."""
        tag = f"best{best_count[0]:04d}_iter{iteration:04d}"
        best_count[0] += 1
        save_dir = best_patches_dir / tag
        save_dir.mkdir(parents=True, exist_ok=True)

        # Patch image
        patch = generate_patch_from_z(generator, z, device)
        cv2.imwrite(str(save_dir / "patch.png"), _tensor_to_bgr(patch.cpu()))

        # Composite images from this evaluation
        for j, comp_bgr in enumerate(composites_np):
            cv2.imwrite(str(save_dir / f"composite_{j:04d}.jpg"), comp_bgr)

        # Per-image results CSV
        if per_image_data:
            fieldnames = list(per_image_data[0].keys())
            # Remove composite_np key if accidentally present
            fieldnames = [f for f in fieldnames if f != 'composite_np']
            with open(save_dir / "results.csv", 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, escapechar='\\',
                                   extrasaction='ignore')
                w.writeheader()
                w.writerows(per_image_data)

        tqdm.write(
            f"  [new best] iter={iteration} fitness={fitness:.4f} "
            f"success={best_success_pct[0]:.1f}% -> {save_dir}"
        )

    # ---- objective ----
    def objective(z: np.ndarray, candidate_idx=None) -> float:
        """Returns negative fitness (CMA-ES minimises)."""
        eval_count[0] += 1

        patch = generate_patch_from_z(generator, z, device)  # [3, H, W]
        patch_c = torch.clamp(patch, 0.0, 1.0)

        # Save debug output for iteration 0
        save_debug = (current_iter[0] == 0 and candidate_idx is not None)
        if save_debug:
            cv2.imwrite(
                str(debug_dir / f"iter0_cand{candidate_idx:03d}_patch.png"),
                _tensor_to_bgr(patch_c.cpu())
            )

        # Build composited images for this candidate
        composited_tensors = []
        composites_bgr = []
        for img in iter_images:
            comp, _ = strategy.apply(img, patch_c)   # [1, 3, H, W]
            comp_cpu = comp.squeeze(0).cpu()
            composited_tensors.append(comp_cpu)
            bgr = _composite_to_bgr(comp_cpu)
            composites_bgr.append(bgr)

            if save_debug:
                j = len(composited_tensors) - 1
                cv2.imwrite(
                    str(debug_dir / f"iter0_cand{candidate_idx:03d}_img{j:04d}_composite.jpg"),
                    bgr
                )

        # Evaluate via metric
        aggregate, per_image = metric.compute_detailed(
            composited_tensors, iter_controls, model
        )
        fitness   = aggregate.get('primary', 0.0)
        success_r = aggregate.get('success_rate', 0.0)

        iter_metrics.append(fitness)

        # Track best
        if fitness > best_fitness[0]:
            best_fitness[0]     = fitness
            best_success_pct[0] = success_r * 100.0
            best_z[0]           = z.copy()
            save_best_patch(z, current_iter[0], fitness, per_image, composites_bgr)

        return -fitness   # CMA-ES minimises

    # ---- CMA-ES setup ----
    x0 = np.zeros(latent_dim)
    es = cma.CMAEvolutionStrategy(
        x0, args.sigma0,
        {
            'popsize':        args.popsize,
            'maxiter':        args.maxiter,
            'verb_disp':      1,
            'verb_log':       0,
            'tolstagnation':  np.inf,
            'tolfun':        -np.inf,
            'tolflatfitness': np.inf,
            'tolxstagnation': np.inf,
            'tolx':          -np.inf,
        }
    )

    num_use = min(args.n_eval_samples, len(val_images))
    print(f"\nStarting CMA-ES optimisation:")
    print(f"  latent_dim={latent_dim}  popsize={args.popsize}  maxiter={args.maxiter}")
    print(f"  sigma0={args.sigma0}")
    print(f"  Full pool: {len(val_images)} images  |  Per-iteration sample: {num_use}")
    print(f"  Metric:   {type(metric).__name__}")
    print(f"  Strategy: {type(strategy).__name__}")
    print("=" * 80)

    pbar = tqdm(total=args.maxiter, desc="CMA-ES", unit="iter",
                position=0, leave=True, dynamic_ncols=True)

    # ---- Main loop ----
    for iteration in range(args.maxiter):
        current_iter[0] = iteration

        # Sample a fresh validation subset for this iteration (shared by all candidates)
        num_use = min(args.n_eval_samples, len(val_images))
        sampled_idx = random.sample(range(len(val_images)), num_use)
        iter_images.clear()
        iter_controls.clear()
        for si in sampled_idx:
            iter_images.append(val_images[si])
            iter_controls.append(control_outputs[si])

        # Reset per-iteration metrics
        iter_metrics.clear()

        if iteration == 0:
            tqdm.write(f"[iter 0] saving debug output to {debug_dir}")

        # Ask / evaluate / tell
        solutions = es.ask()
        cand_pbar = tqdm(total=len(solutions), desc=f"iter {iteration}",
                         unit="cand", position=1, leave=False, dynamic_ncols=True)
        fitness_values = []
        for i, z in enumerate(solutions):
            f = objective(z, candidate_idx=i if iteration == 0 else None)
            fitness_values.append(f)
            cand_pbar.update(1)
        cand_pbar.close()

        es.tell(solutions, fitness_values)

        avg_fit = np.mean(iter_metrics) if iter_metrics else 0.0
        pbar.set_postfix({
            'best_fit':  f'{best_fitness[0]:.4f}',
            'avg_fit':   f'{avg_fit:.4f}',
            'success%':  f'{best_success_pct[0]:.1f}%',
        })
        pbar.update(1)

    pbar.close()

    # ---- Final summary ----
    print("\n" + "=" * 80)
    print("OPTIMISATION COMPLETE")
    print("=" * 80)
    print(f"Iterations:        {args.maxiter}")
    print(f"Total evaluations: {eval_count[0]}")
    print(f"Best fitness:      {best_fitness[0]:.4f}")
    print(f"Best success rate: {best_success_pct[0]:.1f}%")

    # ---- Final evaluation on full pool ----
    if best_z[0] is not None:
        best_patch = generate_patch_from_z(generator, best_z[0], device)
        best_patch_c = torch.clamp(best_patch, 0, 1)

        # Save final best patch PNG + latent code
        cv2.imwrite(str(output_dir / "best_patch.png"), _tensor_to_bgr(best_patch_c.cpu()))
        np.save(str(output_dir / "best_z.npy"), best_z[0])
        print(f"\nSaved best patch  → {output_dir / 'best_patch.png'}")
        print(f"Saved latent code → {output_dir / 'best_z.npy'}")

        # Full-pool evaluation
        print(f"\nEvaluating best patch on full pool ({len(val_images)} images)...")
        final_results_dir = output_dir / "best_patch_results"
        final_results_dir.mkdir()

        final_composited = []
        final_composites_bgr = []
        for img_cpu in tqdm(val_images_cpu, desc="Building composites"):
            comp, _ = strategy.apply(img_cpu.to(device), best_patch_c)
            comp_cpu = comp.squeeze(0).cpu()
            final_composited.append(comp_cpu)
            final_composites_bgr.append(_composite_to_bgr(comp_cpu))

        final_agg, final_per_image = metric.compute_detailed(
            final_composited, control_outputs, model
        )

        # Save control + composite image pairs
        print(f"Saving composite images to {final_results_dir}...")
        for j, (comp_bgr, ctrl_bgr) in enumerate(
                tqdm(zip(final_composites_bgr, neutral_composites_cpu),
                     total=len(final_composites_bgr), desc="Saving images")):
            cv2.imwrite(str(final_results_dir / f"img{j:04d}_composite.jpg"), comp_bgr)
            ctrl_np = (ctrl_bgr.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                str(final_results_dir / f"img{j:04d}_control.jpg"),
                cv2.cvtColor(ctrl_np, cv2.COLOR_RGB2BGR)
            )

        # Save per-image results CSV
        if final_per_image:
            fieldnames = [k for k in final_per_image[0].keys() if k != 'composite_np']
            with open(final_results_dir / "results.csv", 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, escapechar='\\',
                                   extrasaction='ignore')
                w.writeheader()
                w.writerows(final_per_image)

        print(f"Final evaluation results:")
        for k, v in sorted(final_agg.items()):
            print(f"  {k}: {v:.4f}")

        # ---- Metadata file ----
        meta_path = output_dir / "optimization_results.txt"
        with open(meta_path, 'w') as f:
            f.write("CMA-ES Domain Optimisation Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp:         {datetime.now().isoformat()}\n\n")
            f.write(f"Run directory:     {args.run_dir}\n")
            f.write(f"Domain config:     {args.config}\n\n")
            f.write(f"Domain:            {type(domain).__name__}\n")
            f.write(f"Strategy:          {type(strategy).__name__}\n")
            f.write(f"Metric:            {type(metric).__name__}\n\n")
            f.write(f"CMA-ES Parameters:\n")
            f.write(f"  popsize:         {args.popsize}\n")
            f.write(f"  maxiter:         {args.maxiter}\n")
            f.write(f"  sigma0:          {args.sigma0}\n")
            f.write(f"  latent_dim:      {latent_dim}\n\n")
            f.write(f"Evaluation:\n")
            f.write(f"  Val pool size:   {len(val_images)}\n")
            f.write(f"  Per-iter sample: {min(args.n_eval_samples, len(val_images))}\n")
            if args.seed is not None:
                f.write(f"  Seed:            {args.seed}\n")
            f.write(f"  Device:          {device}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total evals:     {eval_count[0]}\n")
            f.write(f"  Best fitness:    {best_fitness[0]:.4f}\n")
            f.write(f"  Best success%:   {best_success_pct[0]:.1f}%\n\n")
            f.write(f"Full-pool final metrics:\n")
            for k, v in sorted(final_agg.items()):
                f.write(f"  {k}: {v:.4f}\n")
        print(f"\nMetadata → {meta_path}")

    print(f"\nAll results in: {output_dir}")


if __name__ == '__main__':
    main()
