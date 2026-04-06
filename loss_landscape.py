#!/usr/bin/env python3
"""
loss_landscape.py — linear interpolation loss survey between two checkpoints.

Finds the latest .pt in each run directory, linearly interpolates the seed
and decoder weights at N evenly-spaced alpha values between 0 and 1, and
estimates the average loss over a fixed set of data batches for all four
training pipelines.  Prints a table of per-pipeline and mean losses at each
alpha, giving a picture of the loss landscape between the two solutions.

Usage:
    python loss_landscape.py runs/run_a runs/run_b \\
        --finetuned-models finetuned_models \\
        --csv finetuned_models/train_split.csv

    python loss_landscape.py runs/run_a runs/run_b --n-steps 21 --n-batches 20
"""

import argparse
import contextlib
import random as _stdlib_random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from trainer import AdversarialPatchTrainer, PatchDecoder, PIPELINE_PAIRINGS
from detector_backends import build_backend
from ocr_backends import build_ocr_backend

FINETUNED_CHECKPOINT_MAP = {
    "fasterrcnn":   "fasterrcnn_finetuned.pt",
    "rtdetr":       "rtdetr_finetuned",
    "owlvit":       "owlvit_finetuned",
    "yolo-v9-608":  "yolo608_finetuned.pt",
    "lprnet":       "lprnet_finetuned.pt",
    "trocr":        "trocr_small_finetuned.pt",
    "doctr-vitstr": "vitstr_small_finetuned.pt",
    "cct":          "cct_s_finetuned.pt",
}


def latest_checkpoint(run_dir: Path) -> Path:
    patches_dir = run_dir / "patches"
    pts = sorted(patches_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoints found in {patches_dir}")
    return pts[-1]


def lerp_state_dict(sd_a: dict, sd_b: dict, alpha: float) -> dict:
    """(1 - alpha) * A + alpha * B, cast to float32 for precision."""
    return {
        k: (1.0 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()
        for k in sd_a
    }


def apply_interpolated_weights(
    trainer: AdversarialPatchTrainer,
    seed_a: torch.Tensor,
    seed_b: torch.Tensor,
    sd_a: dict,
    sd_b: dict,
    alpha: float,
) -> None:
    """Interpolate A and B and write the result into trainer in-place."""
    with torch.no_grad():
        interp_seed = (1.0 - alpha) * seed_a.float() + alpha * seed_b.float()
        trainer.seed.copy_(interp_seed.to(trainer.device))
    trainer.decoder.load_state_dict(lerp_state_dict(sd_a, sd_b, alpha))
    trainer.decoder.to(trainer.device)


@torch.no_grad()
def eval_pipeline_loss(
    trainer: AdversarialPatchTrainer,
    raw_items: list,
    batch_size: int,
) -> float:
    """
    Average total loss for the currently active pipeline over raw_items.

    raw_items: list of single-item dicts (loader output with leading dim stripped).
    Returns the mean loss across all sub-batches.
    """
    patch_norm = trainer.generate_patch(training_aug=False)
    total_loss = 0.0
    n_chunks = 0

    # Prepare all items once for this pipeline.
    prepared = []
    for raw in raw_items:
        item = trainer._prepare_one(raw, patch_norm, augment=False)
        item["_patch_norm"] = patch_norm
        prepared.append(item)

    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        loss, *_ = trainer.compute_loss_batch(chunk)
        total_loss += loss.item()
        n_chunks += 1

    return total_loss / max(n_chunks, 1)


@torch.no_grad()
def generate_patch_image(trainer: AdversarialPatchTrainer) -> Image.Image:
    """Render the current decoder weights as a PIL image."""
    patch = trainer.generate_patch(training_aug=False)  # [3, H, W] in [0, 1]
    arr = (patch.cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def make_patch_grid(
    patches: list,          # list of (label: str, pil_image: Image)
    gap: int = 8,
    label_h: int = 24,
    bg: tuple = (30, 30, 30),
) -> Image.Image:
    """Concatenate labelled patch images side by side."""
    pw, ph = patches[0][1].size
    n = len(patches)
    canvas_w = n * pw + (n - 1) * gap
    canvas_h = ph + label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, (label, img) in enumerate(patches):
        x = i * (pw + gap)
        canvas.paste(img, (x, label_h))
        draw.text((x + 4, 4), label, fill=(220, 220, 220), font=font)

    return canvas


def _params_to_vec(seed: torch.Tensor, sd: dict) -> torch.Tensor:
    """Flatten seed + all decoder tensors into a single float32 CPU vector."""
    parts = [seed.float().cpu().flatten()]
    for v in sd.values():
        parts.append(v.float().cpu().flatten())
    return torch.cat(parts)


def _vec_to_params(vec: torch.Tensor, seed_ref: torch.Tensor, sd_ref: dict):
    """Split a flat vector back into (seed, state_dict) matching the reference shapes."""
    offset = 0
    seed_n = seed_ref.numel()
    seed = vec[offset : offset + seed_n].reshape(seed_ref.shape)
    offset += seed_n
    sd = {}
    for k, v in sd_ref.items():
        n = v.numel()
        sd[k] = vec[offset : offset + n].reshape(v.shape)
        offset += n
    return seed, sd


def run_basin_profile(
    trainer: AdversarialPatchTrainer,
    seed_orig: torch.Tensor,
    sd_orig: dict,
    pipeline_backends: list,
    raw_items: list,
    batch_size: int,
    n_epsilons: int = 11,
    seed: int = 0,
    direction: torch.Tensor = None,   # pre-computed unit-norm direction, or None for random
) -> None:
    """
    Perturb checkpoint weights along a unit-norm direction by exponentially
    increasing magnitudes (1e-8 → 1e-3) and evaluate loss at each step.

    direction: if None, a random unit-norm vector is sampled (reproducible via seed).
               If provided (e.g. normalised A→B vector), it is used as-is.
    """
    base_vec = _params_to_vec(seed_orig, sd_orig)

    if direction is None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        direction = torch.randn(base_vec.numel(), generator=rng)
        direction /= direction.norm()
        dir_label = f"random direction, seed={seed}"
    else:
        dir_label = "direction A→B (normalised)"

    epsilons = np.geomspace(1e-8, 1e-3, n_epsilons).tolist()

    pipeline_names = [f"{d}/{o}" for d, o in PIPELINE_PAIRINGS]
    col_w = 20
    header = (
        f"{'epsilon':>10}  "
        + "  ".join(f"{n:>{col_w}}" for n in pipeline_names)
        + f"  {'mean':>{col_w}}"
    )
    sep = "-" * len(header)

    print(f"\nBasin sharpness profile  ({dir_label})")
    print(f"Total parameters in direction vector: {base_vec.numel():,}")
    print(header)
    print(sep)

    # Baseline at epsilon=0
    apply_interpolated_weights(trainer, seed_orig, seed_orig, sd_orig, sd_orig, 0.0)
    base_losses = []
    for det, ocr in pipeline_backends:
        trainer._activate_pipeline(det, ocr)
        base_losses.append(eval_pipeline_loss(trainer, raw_items, batch_size))
    base_mean = sum(base_losses) / len(base_losses)
    print(
        f"{'0 (base)':>10}  "
        + "  ".join(f"{l:>{col_w}.4f}" for l in base_losses)
        + f"  {base_mean:>{col_w}.4f}"
    )

    all_rows = []
    for eps in epsilons:
        perturbed_vec = base_vec + eps * direction
        p_seed, p_sd = _vec_to_params(perturbed_vec, seed_orig, sd_orig)
        apply_interpolated_weights(trainer, p_seed, p_seed, p_sd, p_sd, 0.0)

        row_losses = []
        for det, ocr in pipeline_backends:
            trainer._activate_pipeline(det, ocr)
            row_losses.append(eval_pipeline_loss(trainer, raw_items, batch_size))
        mean_loss = sum(row_losses) / len(row_losses)
        all_rows.append((eps, row_losses, mean_loss))

        print(
            f"{eps:>10.2e}  "
            + "  ".join(f"{l:>{col_w}.4f}" for l in row_losses)
            + f"  {mean_loss:>{col_w}.4f}"
        )

    print(sep)

    # Find first epsilon where mean loss exceeds base by >1%
    threshold = base_mean * 1.01
    first_broken = next(
        (eps for eps, _, m in all_rows if m > threshold), None
    )
    if first_broken:
        print(f"\nLoss rises >1% above baseline at epsilon ≈ {first_broken:.2e}  "
              f"(basin radius estimate)")
    else:
        print(f"\nLoss stays within 1% of baseline across all epsilons — very flat basin.")


# ═══════════════════════════════════════════════════════════════════════════════
# Q-score (quality-loss gradient) analysis
# ═══════════════════════════════════════════════════════════════════════════════

class _EncoderOnlyWrapper(torch.nn.Module):
    """
    Wraps a HuggingFace VisionEncoderDecoderModel so only the vision encoder
    runs.  The decoder requires autoregressive generation and cannot be driven
    by a plain tensor forward, so we discard it for the neuron-capture pass.
    """
    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder

    def named_modules(self):          # expose encoder leaf modules to hooks
        return self.enc.named_modules()

    def forward(self, x):
        return self.enc(pixel_values=x).last_hidden_state


def _qs_discover_leaf_shapes(model, sample_input, device):
    """
    One dummy forward with hooks on every leaf module.
    Returns {layer_name: output_shape_no_batch}.
    Swallows exceptions so partial execution still yields shapes.
    """
    shapes = {}
    hooks  = []

    def _make(name):
        def hook(mod, _inp, out):
            if isinstance(out, torch.Tensor) and out.dim() >= 1:
                shapes[name] = tuple(out.shape[1:])
        return hook

    for name, mod in model.named_modules():
        if not list(mod.children()):
            hooks.append(mod.register_forward_hook(_make(name)))
    try:
        with torch.no_grad():
            try:   model(sample_input)
            except Exception: pass
    finally:
        for h in hooks: h.remove()
    return shapes


def _qs_sample_neurons(shapes, k, rng):
    """
    Uniform-layer-first sampling of k (layer_name, flat_idx) pairs.
    Avoids large-layer bias by choosing the layer first, then a neuron within it.
    """
    names = list(shapes.keys())
    if not names:
        return []
    result = []
    for _ in range(k):
        name = rng.choice(names)
        size = max(1, int(np.prod(shapes[name])))
        result.append((name, rng.randint(0, size - 1)))
    return result


def _qs_capture(model, inp, neurons, no_grad, device):
    """
    Forward pass; return [k] float32 scalar activations.
    When no_grad=False the returned tensor retains its grad_fn so
    loss.backward() flows back through the model into the decoder.
    """
    layer_neurons = {}
    for pos, (name, idx) in enumerate(neurons):
        layer_neurons.setdefault(name, []).append((pos, idx))

    name_to_mod = dict(model.named_modules())
    captured    = {}
    hooks       = []

    def _make(name, detach):
        def hook(mod, _inp, out):
            if isinstance(out, torch.Tensor):
                captured[name] = out.detach().float() if detach else out.float()
        return hook

    for name in layer_neurons:
        if name in name_to_mod:
            hooks.append(name_to_mod[name].register_forward_hook(
                _make(name, detach=no_grad)
            ))

    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    try:
        with ctx:
            try:   model(inp)
            except Exception: pass

        # Build list of k scalar tensors, then stack (preserves grad_fn)
        elems = [torch.zeros((), device=device, dtype=torch.float32)
                 for _ in range(len(neurons))]

        for name, nlist in layer_neurons.items():
            if name not in captured:
                continue
            act = captured[name]
            try:
                flat = act[0].reshape(-1) if act.dim() > 1 else act.reshape(-1)
            except Exception:
                continue
            L = flat.numel()
            for pos, idx in nlist:
                elems[pos] = flat[idx % L]
            del captured[name]

        return torch.stack(elems)   # [k]
    finally:
        for h in hooks: h.remove()


def _qs_loss(adv_acts, ctrl_acts):
    """L_q = -log(rms(delta) + ε),  delta = adv - ctrl (ctrl detached)."""
    delta = adv_acts - ctrl_acts.detach()
    return -torch.log(delta.pow(2).mean().sqrt() + 1e-8)


def _collect_grad(trainer):
    """Flatten seed + decoder gradients into a single CPU float32 vector."""
    parts = []
    for p in [trainer.seed] + list(trainer.decoder.parameters()):
        g = p.grad
        parts.append(
            g.detach().float().cpu().flatten() if g is not None
            else torch.zeros(p.numel(), dtype=torch.float32)
        )
    return torch.cat(parts)


def _cosim(a, b):
    na, nb = a.norm().item(), b.norm().item()
    if na < 1e-12 or nb < 1e-12:
        return float('nan')
    return (a.dot(b) / (na * nb)).item()


def _qs_compute_one(
    trainer, nn_model, neurons, is_ocr, backend, pipeline_backends, pi, raw_items, device
):
    """
    Compute quality loss and its gradient w.r.t. decoder+seed params for one
    (model, neuron_set) pair.

    Returns (ql_value: float, grad_flat: Tensor[N]).
    """
    det_b, ocr_b = pipeline_backends[pi]
    trainer._activate_pipeline(det_b, ocr_b)

    # Zero decoder + seed gradients
    trainer.decoder.zero_grad()
    if trainer.seed.grad is not None:
        trainer.seed.grad.zero_()

    with torch.enable_grad():
        patch_norm = trainer.generate_patch(training_aug=False)
    gray = torch.full_like(patch_norm.detach(), 0.5)

    total_ql = torch.tensor(0.0, device=device)
    n_valid  = 0

    for raw_item in raw_items:
        try:
            if not is_ocr:
                # ── Detector: full preprocessed image ──────────────────────
                adv_item  = trainer._prepare_one(raw_item, patch_norm, augment=False)
                ctrl_item = trainer._prepare_one(raw_item, gray,       augment=False)
                adv_inp   = adv_item["patched_prep"].unsqueeze(0).to(device)
                ctrl_inp  = ctrl_item["patched_prep"].unsqueeze(0).to(device).detach()
            else:
                # ── OCR: GT plate crop + additive patch overlay ─────────────
                ctrl_item = trainer._prepare_one(raw_item, gray, augment=False)
                ctrl_crop = ctrl_item.get("ocr_crop")
                if ctrl_crop is None:
                    continue
                ctrl_crop = ctrl_crop.to(device)           # [3, H, W] float [0,1]
                oh, ow    = backend.ocr_crop_size
                # Resize patch to OCR crop size; gradient flows through this op
                p_small   = F.interpolate(
                    patch_norm.unsqueeze(0), size=(oh, ow),
                    mode='bilinear', align_corners=False
                ).squeeze(0)                               # [3, oh, ow]
                adv_crop  = ctrl_crop.detach() + 0.1 * p_small   # synthetic overlay
                # _preprocess is differentiable (F.interpolate + permute/scale)
                ctrl_inp  = backend._preprocess(ctrl_crop.unsqueeze(0)).detach()
                adv_inp   = backend._preprocess(adv_crop.unsqueeze(0))

            ctrl_acts = _qs_capture(nn_model, ctrl_inp, neurons, no_grad=True,  device=device)
            adv_acts  = _qs_capture(nn_model, adv_inp,  neurons, no_grad=False, device=device)
            total_ql  = total_ql + _qs_loss(adv_acts, ctrl_acts)
            n_valid  += 1
        except Exception:
            continue

    if n_valid == 0:
        n_params = sum(p.numel() for p in [trainer.seed] + list(trainer.decoder.parameters()))
        return float('nan'), torch.zeros(n_params, dtype=torch.float32)

    (total_ql / n_valid).backward()
    return (total_ql / n_valid).item(), _collect_grad(trainer)


def run_qscore_basin_profile(
    trainer, seed_orig, sd_orig, pipeline_backends,
    raw_items, batch_size, n_epsilons, direction_seed, k_neurons,
):
    """
    Basin sharpness profile augmented with quality-loss gradients.

    Design decisions (printed at runtime):
      1. Quality loss: L_q = -log(rms(adv_acts - ctrl_acts) + ε), unnormalised
         — no per-neuron std calibration pass required.
      2. Ctrl (detectors): full preprocessed image with constant gray (0.5) patch
         at plate boundary, same _prepare_one pipeline as training.
      3. Adv  (detectors): full preprocessed image with learned patch (standard).
      4. Ctrl (OCR): GT plate crop (item['ocr_crop']), backend._preprocess applied,
         detached. GT crop is pipeline-independent.
      5. Adv  (OCR): ctrl_crop + 0.1 × F.interpolate(patch, ocr_size), then
         _preprocess. Scale 0.1 injects a differentiable signal; the plate crop
         doesn't naturally contain the border patch.
      6. TrOCR: encoder-only forward via _EncoderOnlyWrapper — decoder requires
         autoregressive input tokens incompatible with a plain forward pass.
      7. Neuron sampling: uniform-layer-first (layer chosen uniformly, then neuron
         within it uniformly). Two sets/model, seeded at direction_seed and +1.
      8. Gradient: ∂L_q/∂(seed ‖ decoder_params), one independent backward per
         (model, set) pair.  Same layout as _params_to_vec.
      9. cos(grad, -direction): positive means quality gradient points back toward
         the trained checkpoint; negative means it diverges.
     10. Cross-model test uses set_0 gradients at the base checkpoint only.
     11. Cross-model pairs split into: within-pipeline (det↔ocr same pipeline),
         cross-pipeline det↔det, cross-pipeline ocr↔ocr, cross-pipeline det↔ocr.
    """
    device = trainer.device

    print("\n" + "═" * 72)
    print("Q-score basin profile")
    print("═" * 72)
    for i, line in enumerate([
        "Quality loss: -log(rms(adv-ctrl) + ε), unnormalised",
        "Ctrl detectors: gray-patch full image  Adv: learned-patch full image",
        "Ctrl OCR: GT plate crop via _preprocess (detached)",
        "Adv  OCR: ctrl_crop + 0.1 × resized_patch → _preprocess (grad flows)",
        "TrOCR: encoder-only wrapper used",
        f"Neurons: {k_neurons}/set, 2 sets/model, seeds={direction_seed}&{direction_seed+1}",
        "Gradient: ∂L_q/∂(seed ‖ decoder_params), independent backward/pair",
        "cos(g, -dir): +1 = quality grad aligns with 'return to checkpoint'",
    ], 1):
        print(f"  {i}. {line}")
    print()

    # ── Restore base weights ──────────────────────────────────────────────────
    apply_interpolated_weights(trainer, seed_orig, seed_orig, sd_orig, sd_orig, 0.0)

    # ── Build model registry ──────────────────────────────────────────────────
    # (label, nn_module, is_ocr, backend, pipeline_idx)
    registry = []
    for pi, (det, ocr) in enumerate(pipeline_backends):
        nn_det = getattr(det, '_model', None)
        if nn_det is not None:
            registry.append((det.name, nn_det, False, det, pi))
        nn_ocr = getattr(ocr, '_model', None)
        if nn_ocr is not None:
            # TrOCR: VisionEncoderDecoderModel → wrap encoder only
            if hasattr(nn_ocr, 'encoder') and hasattr(nn_ocr, 'decoder'):
                nn_ocr = _EncoderOnlyWrapper(nn_ocr.encoder)
            registry.append((ocr.name, nn_ocr, True, ocr, pi))

    # ── Discover leaf shapes + sample 2 neuron sets per model ────────────────
    rng_s0 = _stdlib_random.Random(direction_seed)
    rng_s1 = _stdlib_random.Random(direction_seed + 1)
    model_data = {}   # label → (ns0, ns1, nn_model, is_ocr, backend, pi)

    print("Discovering layers and sampling neurons:")
    for label, nn_model, is_ocr, backend, pi in registry:
        det_b, ocr_b = pipeline_backends[pi]
        trainer._activate_pipeline(det_b, ocr_b)
        try:
            with torch.no_grad():
                gray        = torch.full_like(trainer.generate_patch(training_aug=False), 0.5)
            sample_item = trainer._prepare_one(raw_items[0], gray, augment=False)
            if not is_ocr:
                sample_inp = sample_item["patched_prep"].unsqueeze(0).to(device)
            else:
                cc = sample_item.get("ocr_crop")
                if cc is None: raise ValueError("no ocr_crop in item")
                sample_inp = backend._preprocess(cc.unsqueeze(0).to(device))
            shapes = _qs_discover_leaf_shapes(nn_model, sample_inp, device)
            if not shapes: raise ValueError("no leaf shapes discovered")
            ns0 = _qs_sample_neurons(shapes, k_neurons, rng_s0)
            ns1 = _qs_sample_neurons(shapes, k_neurons, rng_s1)
            model_data[label] = (ns0, ns1, nn_model, is_ocr, backend, pi)
            print(f"  ✓  {label:20s}  {len(shapes):4d} layers")
        except Exception as e:
            print(f"  ✗  {label}: skip — {e}")

    if not model_data:
        print("[qscore] No usable models.")
        return

    labels = list(model_data.keys())

    # ── Build perturbation direction (random, same seed as basin profile) ─────
    base_vec  = _params_to_vec(seed_orig, sd_orig)
    trng      = torch.Generator()
    trng.manual_seed(direction_seed)
    direction = torch.randn(base_vec.numel(), generator=trng)
    direction = direction / direction.norm()

    epsilons = [0.0] + np.geomspace(1e-8, 1e-3, n_epsilons).tolist()

    # ── Sweep ─────────────────────────────────────────────────────────────────
    ql_store   = {}   # (label, set_idx, eps_i) → float
    grad_store = {}   # (label, set_idx, eps_i) → Tensor[N]

    n_total = len(epsilons) * len(labels) * 2
    done    = 0
    print(f"\nSweeping {len(epsilons)} epsilon values × {len(labels)} models × 2 sets "
          f"= {n_total} backward passes …")

    for eps_i, eps in enumerate(epsilons):
        if eps == 0.0:
            apply_interpolated_weights(trainer, seed_orig, seed_orig, sd_orig, sd_orig, 0.0)
        else:
            pv      = base_vec + eps * direction
            ps, pd  = _vec_to_params(pv, seed_orig, sd_orig)
            apply_interpolated_weights(trainer, ps, ps, pd, pd, 0.0)

        for label, (ns0, ns1, nn_model, is_ocr, backend, pi) in model_data.items():
            for si, neurons in enumerate([ns0, ns1]):
                ql, gv = _qs_compute_one(
                    trainer, nn_model, neurons, is_ocr, backend,
                    pipeline_backends, pi, raw_items, device,
                )
                ql_store[(label, si, eps_i)]  = ql
                grad_store[(label, si, eps_i)] = gv
                done += 1
                if done % 8 == 0:
                    print(f"  {done}/{n_total}", end="\r", flush=True)

    print(f"  {n_total}/{n_total}  done.         ")

    # ── Print basin profile table ─────────────────────────────────────────────
    cw = 8
    hdr = (f"{'epsilon':>10}  {'model':>20}  {'ql_s0':>{cw}}  {'ql_s1':>{cw}}"
           f"  {'cos(-d)_s0':>11}  {'cos(-d)_s1':>11}  {'cos(s0,s1)':>11}")
    sep = "─" * len(hdr)
    print(f"\n{hdr}")
    print(sep)

    for eps_i, eps in enumerate(epsilons):
        eps_s = "0 (base)" if eps == 0.0 else f"{eps:.2e}"
        for label in labels:
            g0 = grad_store.get((label, 0, eps_i))
            g1 = grad_store.get((label, 1, eps_i))
            q0 = ql_store.get((label, 0, eps_i), float('nan'))
            q1 = ql_store.get((label, 1, eps_i), float('nan'))
            cd0 = _cosim(g0, -direction) if g0 is not None else float('nan')
            cd1 = _cosim(g1, -direction) if g1 is not None else float('nan')
            ci  = _cosim(g0, g1) if g0 is not None and g1 is not None else float('nan')
            print(f"{eps_s:>10}  {label:>20}  {q0:>{cw}.4f}  {q1:>{cw}.4f}"
                  f"  {cd0:>11.4f}  {cd1:>11.4f}  {ci:>11.4f}")
        print()

    # ── Cross-model gradient similarity (base checkpoint, set_0) ─────────────
    print("═" * 72)
    print("Cross-model gradient cosine similarities  (base, set_0)")
    print("═" * 72)

    # Full pairwise matrix
    lw = max(len(l) for l in labels)
    print(f"\n{' ' * lw}", end="")
    for lb in labels:
        print(f"  {lb[:10]:>10}", end="")
    print()
    for la in labels:
        ga = grad_store.get((la, 0, 0))
        print(f"{la:{lw}}", end="")
        for lb in labels:
            gb = grad_store.get((lb, 0, 0))
            if ga is not None and gb is not None:
                print(f"  {_cosim(ga, gb):>10.4f}", end="")
            else:
                print(f"  {'---':>10}", end="")
        print()

    # Categorised pair summaries
    pipe_map  = {lbl: model_data[lbl][5] for lbl in labels}
    is_ocr_map = {lbl: model_data[lbl][3] for lbl in labels}

    same_pipe  = []
    cross_det  = []
    cross_ocr  = []
    cross_mix  = []

    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if j <= i:
                continue
            ga = grad_store.get((la, 0, 0))
            gb = grad_store.get((lb, 0, 0))
            if ga is None or gb is None:
                continue
            cs = _cosim(ga, gb)
            oa, ob = is_ocr_map[la], is_ocr_map[lb]
            if pipe_map[la] == pipe_map[lb]:
                same_pipe.append((la, lb, cs))
            elif not oa and not ob:
                cross_det.append((la, lb, cs))
            elif oa and ob:
                cross_ocr.append((la, lb, cs))
            else:
                cross_mix.append((la, lb, cs))

    def _print_group(pairs, title):
        if not pairs:
            return
        vals = [cs for _, _, cs in pairs]
        print(f"\n{title}  (n={len(pairs)}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f})")
        for la, lb, cs in sorted(pairs, key=lambda x: -x[2]):
            print(f"  {la} ↔ {lb}: {cs:+.4f}")

    _print_group(same_pipe, "Within-pipeline   (det ↔ ocr, same pipeline)")
    _print_group(cross_det, "Cross-pipeline    (det ↔ det)")
    _print_group(cross_ocr, "Cross-pipeline    (ocr ↔ ocr)")
    _print_group(cross_mix, "Cross-pipeline    (det ↔ ocr, different pipeline)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir_a", help="Run directory A  (alpha = 0, or single dir for basin mode)")
    parser.add_argument("run_dir_b", nargs="?", default=None,
                        help="Run directory B  (alpha = 1); omit to run basin sharpness profile")
    parser.add_argument("--finetuned-models", default="finetuned_models", metavar="DIR",
                        help="Directory containing finetuned model weights")
    parser.add_argument("--csv", default="finetuned_models/train_split.csv",
                        help="CCPD training CSV (same as --ccpd-train-csv in train_segmented.py)")
    parser.add_argument("--n-batches", type=int, default=10,
                        help="Number of data items to evaluate per alpha step "
                             "(same items reused for every pipeline and alpha)")
    parser.add_argument("--n-steps", type=int, default=11,
                        help="Number of interpolation points  "
                             "(default 11 → alpha = 0.0, 0.1, …, 1.0)")
    parser.add_argument("--n-epsilons", type=int, default=11,
                        help="Number of epsilon values in basin profile  "
                             "(default 11, log-spaced from 1e-8 to 1e-3)")
    parser.add_argument("--direction-seed", type=int, default=0,
                        help="RNG seed for the random perturbation direction")
    parser.add_argument("--local-direction", action="store_true",
                        help="With two run dirs: profile basin sharpness in the A→B direction "
                             "instead of running the interpolation sweep")
    parser.add_argument("--qscore", action="store_true",
                        help="Single run dir only: profile quality-loss gradient alignment "
                             "across all 8 pipeline models at each perturbation epsilon. "
                             "Reports per-model quality loss, cosine sim with -direction, "
                             "intra-model set cosine sim, and cross-model gradient similarity.")
    parser.add_argument("--k-neurons", type=int, default=1000,
                        help="Neurons per set per model for --qscore (default 1000)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Sub-batch size passed to compute_loss_batch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--impersonation-target", default="SHX8459")
    parser.add_argument("--tv-weight", type=float, default=100.0)
    parser.add_argument("--det-loss-weight", type=float, default=3.0)
    args = parser.parse_args()

    basin_mode = args.run_dir_b is None
    run_a = Path(args.run_dir_a)
    run_b = Path(args.run_dir_b) if not basin_mode else None
    fdir  = Path(args.finetuned_models)

    # ── Checkpoints ───────────────────────────────────────────────────────────
    ckpt_path_a = latest_checkpoint(run_a)
    print(f"Checkpoint A : {ckpt_path_a}")
    ckpt_a = torch.load(ckpt_path_a, map_location="cpu")

    if not basin_mode:
        ckpt_path_b = latest_checkpoint(run_b)
        print(f"Checkpoint B : {ckpt_path_b}")
        ckpt_b = torch.load(ckpt_path_b, map_location="cpu")
    else:
        ckpt_b = ckpt_a   # unused in basin mode

    seed_channels = int(ckpt_a.get("seed_channels", 128))
    seed_a = ckpt_a["seed"]   # [1, C, seed_h, 8]  cpu
    seed_b = ckpt_b["seed"]
    sd_a   = ckpt_a["decoder"]
    sd_b   = ckpt_b["decoder"]
    top_extend = seed_a.shape[2] == 8   # seed_h=8 → top_extend, seed_h=4 → normal

    print(f"Seed channels : {seed_channels}  top_extend={top_extend}")
    print(f"  A global_update = {ckpt_a.get('global_update', '?'):>8}   "
          f"backend = {ckpt_a.get('backend', '?')}")
    if not basin_mode:
        print(f"  B global_update = {ckpt_b.get('global_update', '?'):>8}   "
              f"backend = {ckpt_b.get('backend', '?')}")
    print()

    # ── Load all four pipeline backends ───────────────────────────────────────
    print("Loading pipeline backends…")
    pipeline_backends = []
    for det_name, ocr_name in PIPELINE_PAIRINGS:
        det_path = str(fdir / FINETUNED_CHECKPOINT_MAP[det_name])
        ocr_path = str(fdir / FINETUNED_CHECKPOINT_MAP[ocr_name])
        det = build_backend(det_name, det_path, device=args.device)
        det.load()
        det.eval()
        det.freeze()
        ocr = build_ocr_backend(ocr_name, ocr_path, device=args.device)
        ocr.load()
        ocr.eval()
        pipeline_backends.append((det, ocr))
        print(f"  ✓  {det_name} / {ocr_name}")
    print()

    # ── Build trainer (pipeline 0 used for data loading) ──────────────────────
    det0, ocr0 = pipeline_backends[0]
    trainer = AdversarialPatchTrainer(
        detector             = det0,
        ocr                  = ocr0,
        ccpd_csv             = args.csv,
        seed_channels        = seed_channels,
        impersonation_target = args.impersonation_target,
        tv_weight            = args.tv_weight,
        det_loss_weight      = args.det_loss_weight,
        eval_batch_size      = args.batch_size,
        skip_sanity          = True,
        training             = False,
        augment              = False,
        top_extend           = top_extend,
        run_name             = "_landscape_survey",
    )

    # ── Collect a fixed set of raw items (no pipeline-specific prep yet) ───────
    print(f"Collecting {args.n_batches} data items…")
    loader = trainer._make_pipeline_loader(seed=42)
    raw_items = []
    for batch in loader:
        raw_items.append({k: v[0] for k, v in batch.items()})
        if len(raw_items) >= args.n_batches:
            break
    print(f"Collected {len(raw_items)} items.\n")

    # ── Dispatch ─────────────────────────────────────────────────────────────
    if basin_mode and args.qscore:
        run_qscore_basin_profile(
            trainer, seed_a, sd_a, pipeline_backends,
            raw_items, args.batch_size, args.n_epsilons,
            args.direction_seed, args.k_neurons,
        )
        return

    if basin_mode or args.local_direction:
        if args.local_direction:
            vec_a = _params_to_vec(seed_a, sd_a)
            vec_b = _params_to_vec(seed_b, sd_b)
            diff  = vec_b - vec_a
            local_dir = diff / diff.norm()
            print(f"A→B distance (L2): {diff.norm().item():.4e}")
        else:
            local_dir = None
        run_basin_profile(
            trainer, seed_a, sd_a, pipeline_backends, raw_items,
            args.batch_size, args.n_epsilons, args.direction_seed,
            direction=local_dir,
        )
        return

    # ── Interpolation sweep ───────────────────────────────────────────────────
    alphas = [i / (args.n_steps - 1) for i in range(args.n_steps)]
    pipeline_names = [f"{d}/{o}" for d, o in PIPELINE_PAIRINGS]

    col_w = 20
    header = (
        f"{'alpha':>6}  "
        + "  ".join(f"{n:>{col_w}}" for n in pipeline_names)
        + f"  {'mean':>{col_w}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    all_results = []
    patch_images = {}   # alpha → PIL image for selected alphas
    mid_alpha = alphas[args.n_steps // 2]
    capture_alphas = {alphas[0], mid_alpha, alphas[-1]}

    for alpha in alphas:
        apply_interpolated_weights(trainer, seed_a, seed_b, sd_a, sd_b, alpha)

        # Capture patch image before switching pipelines (pipeline doesn't affect decoder)
        if alpha in capture_alphas:
            patch_images[alpha] = generate_patch_image(trainer)

        row_losses = []
        for det, ocr in pipeline_backends:
            trainer._activate_pipeline(det, ocr)
            loss = eval_pipeline_loss(trainer, raw_items, args.batch_size)
            row_losses.append(loss)

        mean_loss = sum(row_losses) / len(row_losses)
        all_results.append((alpha, row_losses, mean_loss))

        print(
            f"{alpha:>6.2f}  "
            + "  ".join(f"{l:>{col_w}.4f}" for l in row_losses)
            + f"  {mean_loss:>{col_w}.4f}"
        )

    print(sep)

    # ── Summary ───────────────────────────────────────────────────────────────
    best = min(all_results, key=lambda r: r[2])
    print(f"\nLowest mean loss: alpha={best[0]:.2f}  mean={best[2]:.4f}")

    mid_idx = args.n_steps // 2
    mid = all_results[mid_idx]
    end_a, end_b = all_results[0], all_results[-1]
    barrier = mid[2] - max(end_a[2], end_b[2])
    print(f"Midpoint (alpha={mid[0]:.2f}) mean loss: {mid[2]:.4f}")
    print(f"Loss barrier at midpoint vs worse endpoint: {barrier:+.4f}  "
          f"({'barrier present' if barrier > 0 else 'no barrier / valley'})")

    # ── Patch grid image ──────────────────────────────────────────────────────
    a0, a_mid, a1 = alphas[0], mid_alpha, alphas[-1]
    panels = [
        (f"A  (alpha={a0:.2f}  update={ckpt_a.get('global_update','?')})",  patch_images[a0]),
        (f"mid (alpha={a_mid:.2f})",                                          patch_images[a_mid]),
        (f"B  (alpha={a1:.2f}  update={ckpt_b.get('global_update','?')})",  patch_images[a1]),
    ]
    grid = make_patch_grid(panels)
    out_path = Path("loss_landscape_patches.png")
    grid.save(str(out_path))
    print(f"\nPatch grid saved → {out_path}")


if __name__ == "__main__":
    main()
