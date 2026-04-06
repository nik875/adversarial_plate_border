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

from tqdm import tqdm

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

    def named_modules(self, **kwargs):
        return self.enc.named_modules(**kwargs)

    def forward(self, x):
        return self.enc(pixel_values=x).last_hidden_state


class _VisionOnlyWrapper(torch.nn.Module):
    """
    Wraps an OWLViT (or any model with a .owlvit.vision_model sub-module)
    so only the visual backbone runs.  OWLViT's full forward requires text
    input tokens; the vision model accepts only pixel_values.
    """
    def __init__(self, vision_model):
        super().__init__()
        self.vm = vision_model

    def named_modules(self, **kwargs):
        return self.vm.named_modules(**kwargs)

    def forward(self, x):
        out = self.vm(pixel_values=x)
        # Return last_hidden_state if present, else the raw output
        return getattr(out, 'last_hidden_state', out)


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


def _qs_ocr_preprocess(backend, crop_tensor):
    """
    Apply backend-specific preprocessing to an OCR crop tensor [1, 3, H, W].

    For backends with a tensor _preprocess (LPRNet, CCT, DoctrViTSTR):
        delegates to backend._preprocess — matches trainer.py exactly.

    For TrOCR (no tensor _preprocess):
        trainer.py's differentiable_loss does: pixel_values = (crop - 0.5) / 0.5
        The crop is already [1, 3, 384, 384] from _bbox_ocr_crop (ocr_crop_size).
        We apply the same normalisation: resize to ocr_crop_size then (x-0.5)/0.5.
    """
    if hasattr(backend, '_preprocess'):
        return backend._preprocess(crop_tensor)
    oh, ow = backend.ocr_crop_size
    x = F.interpolate(crop_tensor, size=(oh, ow), mode='bilinear', align_corners=False)
    return (x - 0.5) / 0.5   # [0,1] → [-1,1], matches TrOCR differentiable_loss


def _qs_loss(adv_acts, ctrl_acts, neuron_stds):
    """
    L_q = -log(rms(delta / std) + ε),  delta = adv - ctrl (ctrl detached).
    neuron_stds: [k] CPU tensor of per-neuron activation std (from ctrl profiling).
    Normalising by std prevents high-variance neurons from dominating the gradient.
    """
    delta = adv_acts - ctrl_acts.detach()
    norm_delta = delta / neuron_stds.to(delta.device).clamp(min=1e-6)
    return -torch.log(norm_delta.pow(2).mean().sqrt() + 1e-8)


@torch.no_grad()
def _qs_profile_stds(nn_model, neurons, ctrl_inputs, device):
    """
    Estimate per-neuron activation std from a list of ctrl inputs.

    ctrl_inputs: list of model-ready tensors (already preprocessed, on device).
    Returns [k] float32 CPU tensor; clamped to ≥ 1e-6 so no neuron is silenced.
    Falls back to ones if fewer than 2 valid captures.
    """
    all_acts = []
    for inp in ctrl_inputs:
        try:
            acts = _qs_capture(nn_model, inp, neurons, no_grad=True, device=device)
            all_acts.append(acts.cpu())
        except Exception:
            continue
    if len(all_acts) < 2:
        return torch.ones(len(neurons), dtype=torch.float32)
    stacked = torch.stack(all_acts, dim=0)   # [N, k]
    return stacked.std(dim=0).clamp(min=1e-6)


@torch.no_grad()
def _qs_precompute_neuron_acts(nn_model, inputs, needed_neurons, device):
    """
    Run one no-grad forward pass per input, recording only the specific
    (layer_name, flat_idx) pairs in needed_neurons.

    needed_neurons: iterable of (layer_name, flat_idx) pairs
    inputs:         list of preprocessed model-input tensors

    Returns dict[(layer_name, flat_idx)] -> float32 Tensor[N_inputs].
    Entries are NaN for inputs whose forward pass raised or whose layer was absent.

    Memory cost: O(|needed_neurons| × N_inputs × 4 bytes) — typically a few MB
    rather than storing full activation tensors for every leaf layer.
    """
    layer_indices = {}
    for name, idx in needed_neurons:
        layer_indices.setdefault(name, set()).add(idx)

    name_to_mod = {n: m for n, m in nn_model.named_modules()}

    # accumulate scalar values: results[(name, idx)] = [float, ...]
    results = {(name, idx): [] for name, idxs in layer_indices.items() for idx in idxs}

    for inp in inputs:
        captured = {}
        hooks    = []

        def _make(name):
            def hook(mod, _inp, out):
                if isinstance(out, torch.Tensor) and out.dim() >= 1:
                    captured[name] = out.detach().float().cpu().flatten()
            return hook

        for name in layer_indices:
            if name in name_to_mod:
                hooks.append(name_to_mod[name].register_forward_hook(_make(name)))
        try:
            try:   nn_model(inp)
            except Exception: pass
        finally:
            for h in hooks: h.remove()

        for name, idxs in layer_indices.items():
            flat = captured.get(name)
            L    = len(flat) if flat is not None else 0
            for idx in idxs:
                val = flat[idx % L].item() if L > 0 else float('nan')
                results[(name, idx)].append(val)

    return {k: torch.tensor(v, dtype=torch.float32) for k, v in results.items()}


def _qs_stds_from_lookup(lookup, neurons):
    """
    Compute per-neuron activation stds from a compact lookup — no forward passes.

    lookup:  {(layer_name, flat_idx): Tensor[N_inputs]} from _qs_precompute_neuron_acts
    neurons: list of (layer_name, flat_idx)
    Returns [k] float32 CPU tensor, clamped to ≥ 1e-6.
    Falls back to 1.0 for neurons absent from the lookup or with < 2 valid values.
    """
    k   = len(neurons)
    out = torch.ones(k, dtype=torch.float32)
    for pos, key in enumerate(neurons):
        vals = lookup.get(key)
        if vals is not None:
            valid = vals[~torch.isnan(vals)]
            if len(valid) >= 2:
                out[pos] = valid.std().clamp(min=1e-6)
    return out


def _qs_ctrl_acts_from_lookup(lookup, neurons, lookup_idx, device):
    """
    Extract ctrl activations for one input from a compact lookup.

    lookup_idx: which column of each lookup tensor corresponds to this input.
    Returns [k] float32 tensor on device (detached). Falls back to 0.
    """
    k     = len(neurons)
    elems = torch.zeros(k, dtype=torch.float32)
    for pos, key in enumerate(neurons):
        vals = lookup.get(key)
        if vals is not None and lookup_idx < len(vals):
            elems[pos] = vals[lookup_idx]
    return elems.to(device)


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


def _compute_training_grad(trainer, pipeline_backends, raw_items, batch_size, device):
    """
    Compute the average training-loss gradient across all pipelines.

    One backward pass per pipeline (avoids holding all 4 graphs simultaneously).
    Gradients are accumulated into decoder+seed .grad attributes, then collected.

    Returns (mean_loss: float, grad_flat: Tensor[N]).
    """
    trainer.decoder.zero_grad()
    if trainer.seed.grad is not None:
        trainer.seed.grad.zero_()

    total_loss = 0.0
    n_pipes    = len(pipeline_backends)

    for det, ocr in pipeline_backends:
        trainer._activate_pipeline(det, ocr)

        # CuDNN LSTM (e.g. LPRNet) saves the training-mode flag during the
        # forward pass; backward fails if that flag was eval.  Switch before
        # the forward, restore after backward.
        ocr_nn = getattr(ocr, '_model', None)
        if ocr_nn is not None:
            ocr_nn.train()

        with torch.enable_grad():
            patch_norm = trainer.generate_patch(training_aug=False)

        items = []
        for raw in raw_items:
            try:
                item = trainer._prepare_one(raw, patch_norm, augment=False)
                item["_patch_norm"] = patch_norm
                items.append(item)
            except Exception:
                continue

        if not items:
            if ocr_nn is not None:
                ocr_nn.eval()
            continue

        pipe_loss = torch.tensor(0.0, device=device)
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            loss, *_ = trainer.compute_loss_batch(chunk)
            pipe_loss = pipe_loss + loss / len(pipeline_backends)

        pipe_loss.backward()   # accumulates into .grad; graph freed immediately
        if ocr_nn is not None:
            ocr_nn.eval()
        total_loss += pipe_loss.item()

    return total_loss, _collect_grad(trainer)


def _qs_compute_one(
    trainer, nn_model, neurons, neuron_stds, is_ocr, backend,
    pipeline_backends, pi, raw_items, device,
    item_ctrl_inputs=None,
    ctrl_lookup=None,
):
    """
    Compute quality loss and its gradient w.r.t. decoder+seed params for one
    (model, neuron_set) pair.

    neuron_stds: [k] CPU tensor from _qs_profile_stds — used to normalise delta.

    item_ctrl_inputs: optional list (one entry per raw_item) of either
        None  — item failed preparation; skip it
        (ctrl_inp_cpu, ctrl_crop_cpu_or_None, lookup_idx)
            ctrl_inp_cpu  : preprocessed ctrl tensor [1, C, H, W] on CPU
            ctrl_crop_cpu : original OCR crop [1, 3, H, W] on CPU, or None for det
            lookup_idx    : int index into ctrl_lookup for this item's activations
        When provided, _prepare_one is skipped.

    ctrl_lookup: optional {(layer_name, flat_idx): Tensor[N]} from
        _qs_precompute_neuron_acts — when provided, the no-grad ctrl forward
        pass is replaced by a lookup into this table.

    Returns (ql_value: float, grad_flat: Tensor[N]).
    """
    det_b, ocr_b = pipeline_backends[pi]
    trainer._activate_pipeline(det_b, ocr_b)

    # CuDNN LSTM (e.g. LPRNet) requires training mode for the entire
    # forward+backward cycle.  Only switch models that actually have RNN layers
    # to avoid unnecessary BatchNorm behaviour change in other models.
    has_rnn = any(isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN))
                  for m in nn_model.modules())
    if has_rnn:
        nn_model.train()

    # Zero decoder + seed gradients
    trainer.decoder.zero_grad()
    if trainer.seed.grad is not None:
        trainer.seed.grad.zero_()

    with torch.enable_grad():
        patch_norm = trainer.generate_patch(training_aug=False)
    gray = torch.full_like(patch_norm.detach(), 0.5)

    total_ql = torch.tensor(0.0, device=device)
    n_valid  = 0

    for item_idx, raw_item in enumerate(raw_items):
        try:
            cached = item_ctrl_inputs[item_idx] if item_ctrl_inputs is not None else None

            if not is_ocr:
                # ── Detector: additive patch overlay on ctrl background ─────
                # _prepare_one brightness-normalises patch_batch inside
                # torch.no_grad(), severing the gradient chain from patch_norm
                # to patched_prep.  Instead: get the gray-patch background once
                # (ctrl), then add the learned patch as a scaled additive
                # perturbation so gradients always flow from patch_norm.
                if cached is not None:
                    ctrl_inp_cpu, _ctrl_crop_cpu, lookup_idx = cached
                    ctrl_inp = ctrl_inp_cpu.to(device)
                else:
                    ctrl_item = trainer._prepare_one(raw_item, gray, augment=False)
                    ctrl_inp  = ctrl_item["patched_prep"].unsqueeze(0).to(device).detach()
                    lookup_idx = None
                h, w      = ctrl_inp.shape[-2], ctrl_inp.shape[-1]
                p_scaled  = F.interpolate(
                    patch_norm.unsqueeze(0), size=(h, w),
                    mode='bilinear', align_corners=False,
                )                                              # [1, 3, h, w]
                adv_inp   = ctrl_inp + 0.1 * p_scaled         # gradient flows here
            else:
                # ── OCR: GT plate crop + additive patch overlay ─────────────
                if cached is not None:
                    ctrl_inp_cpu, ctrl_crop_cpu, lookup_idx = cached
                    if ctrl_crop_cpu is None:
                        continue
                    ctrl_inp  = ctrl_inp_cpu.to(device)
                    ctrl_crop = ctrl_crop_cpu.to(device)
                else:
                    ctrl_item = trainer._prepare_one(raw_item, gray, augment=False)
                    ctrl_crop = ctrl_item.get("ocr_crop")
                    if ctrl_crop is None:
                        continue
                    # ocr_crop is [1, 3, H, W] (already batched by _prepare_batch)
                    ctrl_crop = ctrl_crop.to(device)           # [1, 3, H, W]
                    ctrl_inp  = _qs_ocr_preprocess(backend, ctrl_crop).detach()
                    lookup_idx = None
                oh, ow    = backend.ocr_crop_size
                # Keep batch dim: [1, 3, oh, ow]; gradient flows through interpolate
                p_batch   = F.interpolate(
                    patch_norm.unsqueeze(0), size=(oh, ow),
                    mode='bilinear', align_corners=False
                )                                          # [1, 3, oh, ow]
                adv_crop  = ctrl_crop.detach() + 0.1 * p_batch   # [1, 3, oh, ow]
                # Pass [1, 3, H, W] directly; _qs_ocr_preprocess falls back to
                # F.interpolate for backends without a tensor _preprocess (e.g. TrOCR)
                adv_inp   = _qs_ocr_preprocess(backend, adv_crop)

            if ctrl_lookup is not None and lookup_idx is not None:
                ctrl_acts = _qs_ctrl_acts_from_lookup(ctrl_lookup, neurons, lookup_idx, device)
            else:
                ctrl_acts = _qs_capture(nn_model, ctrl_inp, neurons, no_grad=True, device=device)
            adv_acts  = _qs_capture(nn_model, adv_inp,  neurons, no_grad=False, device=device)
            total_ql  = total_ql + _qs_loss(adv_acts, ctrl_acts, neuron_stds)
            n_valid  += 1
        except Exception:
            continue

    if n_valid == 0:
        n_params = sum(p.numel() for p in [trainer.seed] + list(trainer.decoder.parameters()))
        return float('nan'), torch.zeros(n_params, dtype=torch.float32)

    (total_ql / n_valid).backward()
    if has_rnn:
        nn_model.eval()
    return (total_ql / n_valid).item(), _collect_grad(trainer)


def run_qscore_basin_profile(
    trainer, seed_orig, sd_orig, pipeline_backends,
    raw_items, batch_size, direction_seed, k_values, beta,
):
    """
    Sweep over neuron counts k (exponential from k_values[0] to k_values[-1]).

    At each k, two fresh neuron sets are independently sampled for every model.
    Reports how gradient consistency (cos(s0,s1)) and alignment with the training
    gradient evolve with k, showing the minimum k needed for a stable signal.
    Training gradient is computed once at the base checkpoint and reused.
    Results are printed as two tables and also written to qscore_k_sweep.csv.

    Design decisions:
      1. Quality loss: L_q = -log(rms(delta/std) + ε), normalised by ctrl std.
      2. Ctrl (detectors): gray-patch image (additive, see note 3).
      3. Adv  (detectors): ctrl + 0.1 × F.interpolate(patch, det_size).
         _prepare_one brightness-normalises inside no_grad, severing grad chain.
      4. Ctrl (OCR): GT plate crop → _qs_ocr_preprocess (detached).
      5. Adv  (OCR): ctrl_crop + 0.1 × resized_patch → _qs_ocr_preprocess.
      6. TrOCR: _EncoderOnlyWrapper (encoder only).
      7. Neurons: uniform-layer-first, 2 independent sets per (model, k).
         Seeds: direction_seed + k_idx for s0, direction_seed + k_idx + N for s1.
      8. Training gradient: average over 4 pipeline losses, once at base.
      9. ctrl_inputs cached once per model; reused for std profiling at each k.
     10. Output table 1 — per pipeline per epsilon:
           train_loss | cos(train,-d) | mean_ql | cos(qs,-d) | cos(qs,train)
           cos(comb,-d) | magnitude_ratio (beta*|qs|/|train|)
     11. Output table 2 — per model per epsilon (set_0 and set_1):
           ql_s0 | ql_s1 | cos(s0,train) | cos(s1,train) | cos(s0,s1)
     12. Cross-model: pairwise cos of qscore set_0 grads at base, same grouping.
    """
    device = trainer.device
    import csv as _csv

    print("\n" + "═" * 72)
    print(f"Q-score k-sweep  (beta={beta},  k={k_values[0]}…{k_values[-1]})")
    print("═" * 72)
    for i, line in enumerate([
        f"k values: {k_values}",
        "Neurons resampled independently at each k (seeds shift per k-step)",
        "Training gradient computed once at base checkpoint, reused across k",
        "ctrl_inputs cached per model; reused for std profiling at each k",
        "cos(s0,s1): gradient consistency — higher = more stable signal",
        "cos(qs,t):  alignment with training gradient",
        f"Combined: train_grad + {beta} × mean_qs_grad",
    ], 1):
        print(f"  {i}. {line}")
    print()

    # ── Restore base weights ──────────────────────────────────────────────────
    apply_interpolated_weights(trainer, seed_orig, seed_orig, sd_orig, sd_orig, 0.0)

    # ── Build model registry ──────────────────────────────────────────────────
    registry = []
    for pi, (det, ocr) in enumerate(pipeline_backends):
        nn_det = getattr(det, '_model', None)
        if nn_det is not None:
            if hasattr(nn_det, 'owlvit') and hasattr(nn_det.owlvit, 'vision_model'):
                nn_det = _VisionOnlyWrapper(nn_det.owlvit.vision_model)
            registry.append((det.name, nn_det, False, det, pi))
        nn_ocr = getattr(ocr, '_model', None)
        if nn_ocr is not None:
            if hasattr(nn_ocr, 'encoder') and hasattr(nn_ocr, 'decoder'):
                nn_ocr = _EncoderOnlyWrapper(nn_ocr.encoder)
            registry.append((ocr.name, nn_ocr, True, ocr, pi))

    # ── Setup: discover shapes + collect ctrl_inputs (reused at every k) ─────
    # label → (ctrl_inputs, shapes, nn_model, is_ocr, backend, pi)
    model_data = {}

    print("Discovering layers and collecting ctrl inputs:")
    for label, nn_model, is_ocr, backend, pi in registry:
        det_b, ocr_b = pipeline_backends[pi]
        trainer._activate_pipeline(det_b, ocr_b)
        try:
            with torch.no_grad():
                gray = torch.full_like(trainer.generate_patch(training_aug=False), 0.5)

            ctrl_inputs = []
            for raw in raw_items:
                try:
                    ci = trainer._prepare_one(raw, gray, augment=False)
                    if not is_ocr:
                        ctrl_inputs.append(ci["patched_prep"].unsqueeze(0).to(device))
                    else:
                        cc = ci.get("ocr_crop")
                        if cc is not None:
                            ctrl_inputs.append(_qs_ocr_preprocess(backend, cc.to(device)))
                except Exception:
                    continue

            if not ctrl_inputs:
                raise ValueError("no ctrl inputs prepared")

            shapes = _qs_discover_leaf_shapes(nn_model, ctrl_inputs[0], device)
            if not shapes:
                raise ValueError("no leaf shapes discovered")

            model_data[label] = (ctrl_inputs, shapes, nn_model, is_ocr, backend, pi)
            print(f"  ✓  {label:20s}  {len(shapes):4d} layers  "
                  f"{len(ctrl_inputs)} ctrl inputs")
        except Exception as e:
            print(f"  ✗  {label}: skip — {e}")

    if not model_data:
        print("[qscore] No usable models.")
        return

    labels = list(model_data.keys())

    # ── Pre-generate all neuron sets (deterministic, seeds fully known) ─────────
    # Mirrors the sweep loop order exactly so rng state advances identically.
    n_k = len(k_values)
    print("\nPre-generating neuron sets for all k-steps …")
    all_neuron_sets   = {}   # (label, k_idx, si) → list of (name, idx)
    needed_per_model  = {}   # label → set of (name, idx)

    for k_idx, k in enumerate(k_values):
        rng_s0 = _stdlib_random.Random(direction_seed + k_idx)
        rng_s1 = _stdlib_random.Random(direction_seed + k_idx + n_k)
        for label, (ctrl_inputs, shapes, nn_model, is_ocr, backend, pi) in model_data.items():
            for si, rng in enumerate([rng_s0, rng_s1]):
                neurons = _qs_sample_neurons(shapes, k, rng)
                all_neuron_sets[(label, k_idx, si)] = neurons
                needed_per_model.setdefault(label, set()).update(neurons)

    for label, needed in needed_per_model.items():
        print(f"  {label:20s}  {len(needed):>7,} unique (layer, idx) pairs needed")

    # ── Precompute ctrl activation lookups (once per model, reused at every k) ─
    # ctrl_lookups[label]      : {(layer, idx): Tensor[N_valid]} — scalar activations
    #   only the specific neurons actually used across all k-steps are stored,
    #   so memory is O(unique_neurons × N_inputs × 4 bytes) instead of O(all_leaf_outputs)
    # item_ctrl_inputs[label]  : List[None | (ctrl_inp_cpu, ctrl_crop_cpu, lookup_idx)]
    #   lookup_idx maps each raw_item to its column in ctrl_lookups[label]
    print("\nPrecomputing ctrl activation lookups (once per model) …")
    ctrl_lookups      = {}
    item_ctrl_inputs  = {}

    for label, (ctrl_inputs, shapes, nn_model, is_ocr, backend, pi) in model_data.items():
        det_b, ocr_b = pipeline_backends[pi]
        trainer._activate_pipeline(det_b, ocr_b)

        with torch.no_grad():
            gray = torch.full_like(trainer.generate_patch(training_aug=False), 0.5)

        # Prepare ctrl tensors for every raw_item; valid ones are passed to precompute
        item_cache   = []
        valid_inputs = []   # model-ready tensors in lookup-index order
        for raw in raw_items:
            try:
                ctrl_item = trainer._prepare_one(raw, gray, augment=False)
                if not is_ocr:
                    ctrl_inp  = ctrl_item["patched_prep"].unsqueeze(0).to(device).detach()
                    ctrl_crop = None
                else:
                    ctrl_crop_t = ctrl_item.get("ocr_crop")
                    if ctrl_crop_t is None:
                        item_cache.append(None)
                        continue
                    ctrl_crop = ctrl_crop_t.to(device)
                    ctrl_inp  = _qs_ocr_preprocess(backend, ctrl_crop).detach()
                lookup_idx = len(valid_inputs)
                valid_inputs.append(ctrl_inp)
                item_cache.append((
                    ctrl_inp.cpu(),
                    ctrl_crop.cpu() if ctrl_crop is not None else None,
                    lookup_idx,
                ))
            except Exception:
                item_cache.append(None)

        item_ctrl_inputs[label] = item_cache
        ctrl_lookups[label] = _qs_precompute_neuron_acts(
            nn_model, valid_inputs, needed_per_model[label], device
        )
        n_valid = sum(1 for x in item_cache if x is not None)
        n_bytes = sum(v.numel() * 4 for v in ctrl_lookups[label].values())
        print(f"  ✓  {label:20s}  {n_valid}/{len(raw_items)} items  "
              f"lookup={len(ctrl_lookups[label]):,} entries  "
              f"{n_bytes / 1e6:.1f} MB")

    # ── Training gradient — computed once at base ─────────────────────────────
    print("\nComputing training gradient (base checkpoint) …")
    train_loss, train_grad = _compute_training_grad(
        trainer, pipeline_backends, raw_items, batch_size, device
    )
    print(f"  training loss = {train_loss:.4f}  |train_grad| = {train_grad.norm():.4e}")

    # ── k sweep ───────────────────────────────────────────────────────────────
    # (label, si, k_idx) → (ql: float, grad: Tensor)
    ql_store   = {}
    grad_store = {}

    n_total = n_k * len(labels) * 2
    print(f"\nSweeping {n_k} k values × {len(labels)} models × 2 sets = {n_total} backward passes")

    with tqdm(total=n_total, unit="bwd", ncols=80) as pbar:
        for k_idx, k in enumerate(k_values):
            for label, (ctrl_inputs, shapes, nn_model, is_ocr, backend, pi) in model_data.items():
                for si in range(2):
                    pbar.set_description(f"k={k:>6d} {label[:12]} s{si}")
                    neurons = all_neuron_sets[(label, k_idx, si)]
                    stds    = _qs_stds_from_lookup(ctrl_lookups[label], neurons)
                    ql, gv  = _qs_compute_one(
                        trainer, nn_model, neurons, stds, is_ocr, backend,
                        pipeline_backends, pi, raw_items, device,
                        item_ctrl_inputs=item_ctrl_inputs[label],
                        ctrl_lookup=ctrl_lookups[label],
                    )
                    ql_store[(label, si, k_idx)]   = ql
                    grad_store[(label, si, k_idx)]  = gv
                    pbar.update(1)

    # ── Build result rows ─────────────────────────────────────────────────────
    pipe_map   = {lbl: model_data[lbl][5] for lbl in labels}
    is_ocr_map = {lbl: model_data[lbl][3] for lbl in labels}

    rows = []   # list of dicts for CSV + printing
    for k_idx, k in enumerate(k_values):
        for label in labels:
            g0 = grad_store.get((label, 0, k_idx))
            g1 = grad_store.get((label, 1, k_idx))
            q0 = ql_store.get((label, 0, k_idx), float('nan'))
            q1 = ql_store.get((label, 1, k_idx), float('nan'))

            cs01  = _cosim(g0, g1)
            cs0t  = _cosim(g0, train_grad)
            cs1t  = _cosim(g1, train_grad)
            mean_g = torch.stack([g0, g1]).mean(dim=0) if (g0 is not None and g1 is not None) else None
            cmt   = _cosim(mean_g, train_grad)
            mag_r = float('nan')
            if train_grad is not None and mean_g is not None:
                tn = train_grad.norm().item()
                if tn > 1e-12:
                    mag_r = beta * mean_g.norm().item() / tn

            rows.append(dict(
                k=k, model=label,
                is_ocr=int(is_ocr_map[label]), pipeline=pipe_map[label],
                ql_s0=q0, ql_s1=q1,
                cos_s0s1=cs01, cos_s0t=cs0t, cos_s1t=cs1t,
                cos_meant=cmt, mag_ratio=mag_r,
            ))

    # ── Print Table 1: summary per k ─────────────────────────────────────────
    cw = 9
    h1 = (f"{'k':>8}  {'model':>20}  {'ql_s0':>{cw}}  {'ql_s1':>{cw}}"
          f"  {'cos(s0,s1)':>11}  {'cos(s0,t)':>{cw}}  {'cos(s1,t)':>{cw}}"
          f"  {'cos(mn,t)':>{cw}}  {'β|qs|/|t|':>{cw}}")
    sep = "─" * len(h1)
    print(f"\n{'━'*len(h1)}")
    print("k-sweep results — gradient stability vs neuron count")
    print(f"{'━'*len(h1)}")
    print(h1)
    prev_k = None
    for r in rows:
        if r['k'] != prev_k:
            print(sep)
            prev_k = r['k']
        print(f"{r['k']:>8d}  {r['model']:>20}"
              f"  {r['ql_s0']:>{cw}.4f}  {r['ql_s1']:>{cw}.4f}"
              f"  {r['cos_s0s1']:>11.4f}  {r['cos_s0t']:>{cw}.4f}"
              f"  {r['cos_s1t']:>{cw}.4f}  {r['cos_meant']:>{cw}.4f}"
              f"  {r['mag_ratio']:>{cw}.4f}")

    # ── Print summary: mean cos(s0,s1) per k across all models ───────────────
    print(f"\n{'━'*50}")
    print("cos(s0,s1) summary — gradient stability across k")
    print(f"{'━'*50}")
    print(f"{'k':>8}  {'mean':>8}  {'std':>8}  per-model")
    print("─" * 50)
    for k_idx, k in enumerate(k_values):
        vals = [r['cos_s0s1'] for r in rows if r['k'] == k and not np.isnan(r['cos_s0s1'])]
        per  = "  ".join(f"{r['model'][:6]}={r['cos_s0s1']:+.3f}"
                         for r in rows if r['k'] == k)
        if vals:
            print(f"{k:>8d}  {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}  {per}")

    # ── Cross-model matrix at largest k ──────────────────────────────────────
    last_k_idx = n_k - 1
    print(f"\n{'═'*72}")
    print(f"Cross-model qscore gradient cosine similarities  (k={k_values[-1]}, set_0)")
    print(f"{'═'*72}")
    lw = max(len(l) for l in labels)
    print(f"\n{' '*lw}", end="")
    for lb in labels:
        print(f"  {lb[:10]:>10}", end="")
    print()
    for la in labels:
        ga = grad_store.get((la, 0, last_k_idx))
        print(f"{la:{lw}}", end="")
        for lb in labels:
            gb = grad_store.get((lb, 0, last_k_idx))
            print(f"  {_cosim(ga, gb):>10.4f}" if (ga is not None and gb is not None)
                  else f"  {'---':>10}", end="")
        print()

    same_pipe, cross_det, cross_ocr, cross_mix = [], [], [], []
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if j <= i: continue
            ga = grad_store.get((la, 0, last_k_idx))
            gb = grad_store.get((lb, 0, last_k_idx))
            if ga is None or gb is None: continue
            cs = _cosim(ga, gb)
            oa, ob = is_ocr_map[la], is_ocr_map[lb]
            if pipe_map[la] == pipe_map[lb]:   same_pipe.append((la, lb, cs))
            elif not oa and not ob:             cross_det.append((la, lb, cs))
            elif oa and ob:                     cross_ocr.append((la, lb, cs))
            else:                               cross_mix.append((la, lb, cs))

    def _print_group(pairs, title):
        if not pairs: return
        vals = [cs for _, _, cs in pairs]
        print(f"\n{title}  (n={len(pairs)}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f})")
        for la, lb, cs in sorted(pairs, key=lambda x: -x[2]):
            print(f"  {la} ↔ {lb}: {cs:+.4f}")

    _print_group(same_pipe, "Within-pipeline   (det ↔ ocr, same pipeline)")
    _print_group(cross_det, "Cross-pipeline    (det ↔ det)")
    _print_group(cross_ocr, "Cross-pipeline    (ocr ↔ ocr)")
    _print_group(cross_mix, "Cross-pipeline    (det ↔ ocr, different pipeline)")

    # ── CSV dump ──────────────────────────────────────────────────────────────
    csv_path = "qscore_k_sweep.csv"
    fieldnames = ["k", "model", "is_ocr", "pipeline",
                  "ql_s0", "ql_s1", "cos_s0s1", "cos_s0t", "cos_s1t",
                  "cos_meant", "mag_ratio"]
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV saved → {csv_path}")


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
    parser.add_argument("--k-steps", type=int, default=11,
                        help="Number of k values for --qscore k-sweep  "
                             "(default 11, log-spaced from 100 to 100 000)")
    parser.add_argument("--direction-seed", type=int, default=0,
                        help="RNG seed for the random perturbation direction")
    parser.add_argument("--local-direction", action="store_true",
                        help="With two run dirs: profile basin sharpness in the A→B direction "
                             "instead of running the interpolation sweep")
    parser.add_argument("--qscore", action="store_true",
                        help="Single run dir only: sweep neuron counts from 100 to 100k "
                             "(log-spaced, --k-steps points) to measure how gradient "
                             "stability and training alignment evolve with k.")
    parser.add_argument("--k-neurons", type=int, default=1000,
                        help="(unused in --qscore k-sweep; kept for compatibility)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Weight of quality loss in combined gradient for --qscore "
                             "(default 0.1; quality loss is in log-space so this makes "
                             "it ~10%% of typical training loss magnitude)")
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
        k_vals_raw = np.geomspace(100, 100_000, args.k_steps).round().astype(int).tolist()
        # deduplicate while preserving order
        seen, k_values = set(), []
        for k in k_vals_raw:
            if k not in seen:
                seen.add(k); k_values.append(k)
        run_qscore_basin_profile(
            trainer, seed_a, sd_a, pipeline_backends,
            raw_items, args.batch_size, args.direction_seed,
            k_values, args.beta,
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
