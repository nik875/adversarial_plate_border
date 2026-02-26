"""
NeuronSampler — uniform random sampling of individual neurons from any nn.Module.

Two-phase design:
    1. discover_layers():  one dummy forward with hooks on all leaf modules to record
                           output shapes; result cached by id(model).
    2. sample_neurons():   for each of k neurons, pick a layer uniformly at random,
                           then pick a neuron within that layer uniformly at random.
    3. capture_sampled_activations(): run the real forward with hooks grouped by
                           unique layer; extract k scalar values; release all buffers.

Memory note:
    - When no_grad=True: captured tensors are detached immediately after extraction,
      so no large activation tensors are held across call boundaries.
    - When no_grad=False: each extracted scalar val retains its grad_fn chain back
      to the model input (needed for the adversarial training gradient). The large
      per-layer activation tensors are released as soon as all neurons for that layer
      are extracted (del captured[layer_name] after the inner loop).
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class NeuronSampler:
    """
    Samples random neurons from any nn.Module and captures their activations.

    Usage::

        sampler = NeuronSampler(device=torch.device('cpu'))
        neurons  = sampler.sample_neurons(model, k=10_000, (3, 224, 224))
        ctrl     = sampler.capture_sampled_activations(model, inp, neurons, no_grad=True)
        adv      = sampler.capture_sampled_activations(model, inp, neurons, no_grad=False)
    """

    def __init__(self, device: torch.device):
        self.device = device
        # Cache: id(model) → {layer_name: output_shape_no_batch}
        self._shape_cache: Dict[int, Dict[str, Tuple[int, ...]]] = {}
        # Precomputed per-neuron std profiles: model_id → {layer_name → std Tensor (CPU)}
        self._neuron_stds: Dict[int, Dict[str, Tensor]] = {}
        # Per-model β floor: 10% of median live-neuron std (CPU scalar)
        self._neuron_betas: Dict[int, float] = {}
        # Final (last forward-executing) layer name per model_id
        self._final_layer_names: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Layer discovery
    # ------------------------------------------------------------------

    def discover_layers(
        self,
        model: nn.Module,
        sample_input_shape: Tuple[int, int, int],
    ) -> Dict[str, Tuple[int, ...]]:
        """
        Run a dummy forward pass with hooks on ALL leaf modules simultaneously.

        Only leaf modules (those with no child modules) are registered, which
        avoids double-counting activations from wrapper modules.

        Args:
            model:              the (frozen) model to probe
            sample_input_shape: (C, H, W) — single-image shape (no batch dim)

        Returns:
            {layer_name: output_shape_no_batch}  — cached by id(model)
        """
        model_id = id(model)
        if model_id in self._shape_cache:
            return self._shape_cache[model_id]

        shapes: Dict[str, Tuple[int, ...]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, inp, out):
                if isinstance(out, Tensor) and out.dim() >= 1:
                    # Remove batch dimension
                    shapes[name] = tuple(out.shape[1:])
            return hook

        for name, module in model.named_modules():
            # Leaf modules only (no children)
            if len(list(module.children())) == 0 and len(list(module.parameters())) >= 0:
                hooks.append(module.register_forward_hook(make_hook(name)))

        dummy = torch.zeros(1, *sample_input_shape, device=self.device)
        try:
            with torch.no_grad():
                model(dummy)
        finally:
            for h in hooks:
                h.remove()

        # Keep only layers that produced output
        self._shape_cache[model_id] = shapes
        return shapes

    # ------------------------------------------------------------------
    # Neuron sampling
    # ------------------------------------------------------------------

    def sample_neurons(
        self,
        model: nn.Module,
        k: int,
        sample_input_shape: Tuple[int, int, int],
    ) -> List[Tuple[str, int]]:
        """
        Sample k (layer_name, flat_idx) pairs, uniform-layer-first.

        For each neuron:
            layer_name = random.choice(all_layer_names)   (uniform over layers)
            flat_idx   = random.randint(0, prod(shape)-1) (uniform within layer)

        Args:
            model:              target model
            k:                  number of neurons to sample
            sample_input_shape: (C, H, W) for layer discovery

        Returns:
            List of (layer_name, flat_idx) of length k
        """
        shapes = self.discover_layers(model, sample_input_shape)
        layer_names = list(shapes.keys())
        if not layer_names:
            raise ValueError(
                "No leaf modules produced Tensor outputs — cannot sample neurons. "
                "Check that the model performs a forward pass with Tensor outputs."
            )

        neurons: List[Tuple[str, int]] = []
        for _ in range(k):
            name = random.choice(layer_names)
            shape = shapes[name]
            num_neurons = 1
            for s in shape:
                num_neurons *= s
            flat_idx = random.randint(0, num_neurons - 1)
            neurons.append((name, flat_idx))

        return neurons

    # ------------------------------------------------------------------
    # Activation capture
    # ------------------------------------------------------------------

    def capture_sampled_activations(
        self,
        model: nn.Module,
        model_input: Tensor,
        sampled_neurons: List[Tuple[str, int]],
        no_grad: bool = True,
    ) -> Tensor:
        """
        Run a forward pass and extract the k sampled neuron activations.

        Groups neurons by unique layer (1 hook per unique layer).
        All hooks are removed in a finally block, regardless of exceptions.
        Activation buffers are released as soon as all neurons for a layer
        have been extracted.

        Args:
            model:           target model (all params should have requires_grad=False)
            model_input:     [B, C, H, W] preprocessed batch of inputs
            sampled_neurons: list of (layer_name, flat_idx) from sample_neurons()
            no_grad:         if True, run under torch.no_grad() and detach activations;
                             if False, activations retain grad_fn for backprop

        Returns:
            Tensor [B, k] float32 — one row per batch item, one scalar per sampled neuron
        """
        B = model_input.shape[0]

        # Group: {layer_name → [(position_in_output, flat_idx), ...]}
        layer_to_neurons: Dict[str, List[Tuple[int, int]]] = {}
        for pos, (name, flat_idx) in enumerate(sampled_neurons):
            if name not in layer_to_neurons:
                layer_to_neurons[name] = []
            layer_to_neurons[name].append((pos, flat_idx))

        # Build name → module map
        name_to_module: Dict[str, nn.Module] = {
            n: m for n, m in model.named_modules()
        }

        captured: Dict[str, Tensor] = {}
        hooks = []

        def make_hook(layer_name: str, detach: bool):
            def hook(module, inp, out):
                if isinstance(out, Tensor):
                    captured[layer_name] = out.detach() if detach else out
            return hook

        try:
            for layer_name in layer_to_neurons:
                if layer_name in name_to_module:
                    h = name_to_module[layer_name].register_forward_hook(
                        make_hook(layer_name, detach=no_grad)
                    )
                    hooks.append(h)

            if no_grad:
                with torch.no_grad():
                    model(model_input)
            else:
                model(model_input)

            # Extract values — vectorised per layer (one index op per layer,
            # not one per neuron) so PyTorch dispatch overhead is O(layers)
            # rather than O(neurons).
            # Collect positions + values across all layers, then assemble in
            # one differentiable scatter into a [B, k] result tensor.
            k = len(sampled_neurons)
            all_positions: List[Tensor] = []
            all_vals:      List[Tensor] = []  # each [B, n_layer]

            for layer_name, neuron_list in layer_to_neurons.items():
                if layer_name not in captured:
                    continue  # hook didn't fire — stays zero in result

                act = captured[layer_name]     # [B, *layer_shape] or folded

                # Use the cached per-sample shape to decompose output,
                # even if batch dim is folded with spatial/head dims.
                model_id = id(model)
                per_sample_size = None
                if model_id in self._shape_cache and layer_name in self._shape_cache[model_id]:
                    # Compute expected per-sample size from B=1 shape cache
                    for d in self._shape_cache[model_id][layer_name]:
                        per_sample_size = d if per_sample_size is None else per_sample_size * d

                if (per_sample_size is not None
                    and act.numel() % per_sample_size == 0
                    and act.shape[0] % B == 0):  # First dim accounts for batch
                    # Output is divisible by per-sample size and first dim accounts for batch.
                    # Check if we got exactly B sample units.
                    num_sample_units = act.numel() // per_sample_size
                    if num_sample_units != B:
                        raise RuntimeError(
                            f"Layer {layer_name}: expected {B} sample units but got {num_sample_units}. "
                            f"act.shape={act.shape}, per_sample_size={per_sample_size}, "
                            f"act.numel()={act.numel()}"
                        )
                    # Reshape [B, per_sample] and extract
                    flat = act.reshape(B, per_sample_size)
                    L = per_sample_size

                    idx_t = torch.tensor(
                        [idx % L   for _, idx   in neuron_list],
                        dtype=torch.long, device=flat.device,
                    )
                    vals = flat[:, idx_t]  # [B, n_layer] — retains grad_fn

                elif act.shape[0] == B:
                    # First dim is batch: extract per-sample independently
                    vals_per_sample = []
                    for b in range(B):
                        act_b = act[b].reshape(-1)
                        L = act_b.shape[0]
                        idx_t = torch.tensor(
                            [idx % L   for _, idx   in neuron_list],
                            dtype=torch.long, device=act_b.device,
                        )
                        vals_b = act_b[idx_t]  # [n_layer]
                        vals_per_sample.append(vals_b)
                    vals = torch.stack(vals_per_sample, dim=0)  # [B, n_layer]

                else:
                    # Skip non-batching layers (e.g., position embeddings)
                    if act.shape[0] == 1 and B > 1:
                        del captured[layer_name]
                        continue

                    # Unexpected shape: raise error
                    raise RuntimeError(
                        f"Layer {layer_name}: unable to extract activations for B={B} samples. "
                        f"act.shape={act.shape}, act.numel()={act.numel()}, "
                        f"per_sample_size={per_sample_size}, "
                        f"shape[0]={act.shape[0]}, shape[0]%B={act.shape[0] % B if B > 0 else 'N/A'}"
                    )

                pos_t = torch.tensor(
                    [pos for pos, _ in neuron_list],
                    dtype=torch.long, device=vals.device,
                )
                all_positions.append(pos_t)
                all_vals.append(vals)

                # Release the large activation buffer immediately.
                del captured[layer_name]

            if all_positions:
                all_pos_t  = torch.cat(all_positions)              # [total_fired]
                all_vals_t = torch.cat(all_vals, dim=1).float()    # [B, total_fired]
                n_fired    = all_pos_t.shape[0]
                # Differentiable scatter into [B, k]: expand indices across batch dim
                batch_idx  = torch.arange(B, device=self.device).unsqueeze(1).expand(B, n_fired).reshape(-1)
                pos_exp    = all_pos_t.unsqueeze(0).expand(B, -1).reshape(-1)
                result_tensor = torch.zeros(
                    B, k, device=self.device, dtype=torch.float32,
                ).index_put((batch_idx, pos_exp), all_vals_t.reshape(-1))
            else:
                result_tensor = torch.zeros(B, k, device=self.device, dtype=torch.float32)

            return result_tensor

        finally:
            for h in hooks:
                h.remove()

    # ------------------------------------------------------------------
    # Neuron profiling (precomputed per-neuron std)
    # ------------------------------------------------------------------

    def init_profile(
        self,
        model: nn.Module,
        sample_input_shape: Tuple[int, int, int],
    ) -> Dict:
        """
        Initialise fresh Welford accumulators for profiling.

        Returns a state dict to be passed to update_profile() and finish_profile().
        """
        shapes = self.discover_layers(model, sample_input_shape)
        return {
            'shapes': shapes,
            'counts': {n: 0 for n in shapes},
            'means':  {n: torch.zeros(s, dtype=torch.float32) for n, s in shapes.items()},
            'M2s':    {n: torch.zeros(s, dtype=torch.float32) for n, s in shapes.items()},
        }

    def update_profile(
        self,
        model: nn.Module,
        batch_images: List[Tensor],
        state: Dict,
    ) -> None:
        """
        Run one batch of images through the model and update Welford accumulators.

        Args:
            model:        frozen model (already on compute device)
            batch_images: list of preprocessed [1, C, H, W] tensors (one batch)
            state:        accumulator dict returned by init_profile()
        """
        counts, means, M2s, shapes = state['counts'], state['means'], state['M2s'], state['shapes']

        # Stack batch: list of [1, C, H, W] → [B, C, H, W]
        batch = torch.cat(batch_images, dim=0)

        captured: Dict[str, Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, inp, out):
                if isinstance(out, Tensor):
                    captured[name] = out.detach().cpu().float()
            return hook

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            with torch.no_grad():
                model(batch)
        finally:
            for h in hooks:
                h.remove()

        # Welford update per layer, per image in batch
        for layer_name, act in captured.items():
            if layer_name not in counts:
                continue
            for b in range(act.shape[0]):
                x = act[b]
                if x.shape != means[layer_name].shape:
                    continue
                counts[layer_name] += 1
                n = counts[layer_name]
                delta  = x - means[layer_name]
                means[layer_name] += delta / n
                delta2 = x - means[layer_name]
                M2s[layer_name]   += delta * delta2

    def finish_profile(
        self,
        model_id: int,
        state: Dict,
    ) -> None:
        """
        Finalise accumulators: compute per-neuron std and beta, store results.

        Args:
            model_id: integer id to key the stored statistics
            state:    accumulator dict returned by init_profile()
        """
        shapes, counts, M2s = state['shapes'], state['counts'], state['M2s']

        stds: Dict[str, Tensor] = {}
        for layer_name in shapes:
            n = counts[layer_name]
            if n > 1:
                variance = M2s[layer_name] / (n - 1)
                stds[layer_name] = variance.sqrt_().clamp_(min=1e-8)
            else:
                stds[layer_name] = torch.full(
                    shapes[layer_name], 1e-8, dtype=torch.float32
                )

        self._neuron_stds[model_id] = stds

        layer_names = list(shapes.keys())
        if layer_names:
            self._final_layer_names[model_id] = layer_names[-1]

        # β = 10% of median live-neuron std
        all_stds = torch.cat([s.flatten() for s in stds.values()])
        live_stds = all_stds[all_stds > 1e-7]
        beta = torch.median(live_stds).item() * 0.1 if live_stds.numel() > 0 else 1e-3
        self._neuron_betas[model_id] = beta

    def profile_model(
        self,
        model: nn.Module,
        model_id: int,
        images: List[Tensor],
        sample_input_shape: Tuple[int, int, int],
    ) -> None:
        """
        Convenience wrapper: profile a model from a pre-loaded list of images.

        For incremental (memory-efficient) profiling, use init_profile() /
        update_profile() / finish_profile() directly.
        """
        state = self.init_profile(model, sample_input_shape)
        batch_size = 10
        for batch_start in range(0, len(images), batch_size):
            self.update_profile(model, images[batch_start:batch_start + batch_size], state)
        self.finish_profile(model_id, state)

    def lookup_neuron_stds(
        self,
        model_id: int,
        sampled_neurons: List[Tuple[str, int]],
    ) -> Tensor:
        """
        Return per-neuron std values for the given sampled neurons (CPU tensor).

        Args:
            model_id:        model whose stats to look up
            sampled_neurons: list of (layer_name, flat_idx) from sample_neurons()

        Returns:
            [k] float32 CPU tensor — one std value per neuron, clamped to ≥ 1e-8.
            Falls back to 1e-8 for any neuron whose layer wasn't profiled.
        """
        if model_id not in self._neuron_stds:
            return torch.full((len(sampled_neurons),), 1e-8, dtype=torch.float32)

        stds = self._neuron_stds[model_id]
        result: List[Tensor] = []
        for layer_name, flat_idx in sampled_neurons:
            if layer_name not in stds:
                result.append(torch.tensor(1e-8, dtype=torch.float32))
            else:
                flat = stds[layer_name].flatten()
                result.append(flat[flat_idx % flat.numel()])
        return torch.stack(result)   # [k] on CPU

    def lookup_beta(self, model_id: int) -> float:
        """
        Return the per-model β floor (10% of median live-neuron std).

        Falls back to 1e-3 if the model hasn't been profiled.
        Used as clamp(min=β) on neuron_stds before quality normalization.
        """
        return self._neuron_betas.get(model_id, 1e-3)

    def get_final_layer_neurons(self, model_id: int) -> List[Tuple[str, int]]:
        """
        Return ALL (layer_name, flat_idx) tuples from the final layer.

        The final layer is the last leaf module to fire during the profiling
        forward pass.  All neurons are returned because fql is computed as a
        simple average over neurons (no gram matrix), so using every neuron
        gives a more accurate estimate with no added cost.
        Returns [] if the model hasn't been profiled.
        """
        layer_name = self._final_layer_names.get(model_id, '')
        if not layer_name:
            return []
        stds = self._neuron_stds.get(model_id, {})
        if layer_name not in stds:
            return []
        n = stds[layer_name].numel()
        return [(layer_name, i) for i in range(n)]

    # ------------------------------------------------------------------
    # Profile persistence
    # ------------------------------------------------------------------

    def save_profile(self, model_id: int, path) -> None:
        """
        Persist the profiling data for model_id to a .pt file.

        Saves neuron_stds dict and the computed β scalar.
        """
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'neuron_stds':      self._neuron_stds[model_id],
            'beta':             self._neuron_betas[model_id],
            'final_layer_name': self._final_layer_names.get(model_id, ''),
        }, path)

    def load_profile(self, model_id: int, path) -> bool:
        """
        Load profiling data from a .pt file into model_id's slot.

        Returns True on success, False if the file doesn't exist or is corrupt.
        """
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = torch.load(p, map_location='cpu', weights_only=True)
            self._neuron_stds[model_id]  = data['neuron_stds']
            self._neuron_betas[model_id] = float(data['beta'])
            final = data.get('final_layer_name', '')
            if final:
                self._final_layer_names[model_id] = final
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self, model: nn.Module) -> None:
        """Clear the shape cache for a specific model instance."""
        model_id = id(model)
        if model_id in self._shape_cache:
            del self._shape_cache[model_id]
