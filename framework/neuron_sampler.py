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
            model:          target model (all params should have requires_grad=False)
            model_input:    [1, C, H, W] preprocessed input tensor
            sampled_neurons: list of (layer_name, flat_idx) from sample_neurons()
            no_grad:        if True, run under torch.no_grad() and detach activations;
                            if False, activations retain grad_fn for backprop

        Returns:
            Tensor [k] float32 — one scalar per sampled neuron, in input order
        """
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

            # Extract scalar values
            results: List[Tensor | None] = [None] * len(sampled_neurons)

            for layer_name, neuron_list in layer_to_neurons.items():
                if layer_name not in captured:
                    # Hook didn't fire (layer not reached); fill with zeros
                    for pos, _ in neuron_list:
                        results[pos] = torch.tensor(
                            0.0, device=self.device, dtype=torch.float32
                        )
                    continue

                act = captured[layer_name]     # [1, *shape] (batch size = 1)
                flat = act.reshape(1, -1)      # [1, N]
                N = flat.shape[1]

                for pos, flat_idx in neuron_list:
                    idx = flat_idx % N          # guard against shape mismatch
                    val = flat[0, idx]          # scalar (retains grad_fn if no_grad=False)
                    results[pos] = val

                # Release the large activation buffer immediately after all
                # neurons for this layer have been extracted.
                del captured[layer_name]

            # Stack into [k] — each element is a scalar (possibly with grad_fn)
            result_tensor = torch.stack([
                r if r is not None
                else torch.tensor(0.0, device=self.device, dtype=torch.float32)
                for r in results
            ])
            return result_tensor.float()

        finally:
            for h in hooks:
                h.remove()

    # ------------------------------------------------------------------
    # Neuron profiling (precomputed per-neuron std)
    # ------------------------------------------------------------------

    def profile_model(
        self,
        model: nn.Module,
        model_id: int,
        images: List[Tensor],
        sample_input_shape: Tuple[int, int, int],
    ) -> None:
        """
        Profile per-neuron activation statistics using Welford's online algorithm.

        Runs each image through the model with hooks on all leaf layers,
        accumulating a running mean and variance per neuron. Stores the
        resulting per-neuron std tensors on CPU keyed by model_id.

        Args:
            model:              frozen model (already on compute device)
            model_id:           integer id to key the stored statistics
            images:             list of preprocessed [1, C, H, W] tensors
            sample_input_shape: (C, H, W) for layer discovery
        """
        shapes = self.discover_layers(model, sample_input_shape)

        # Welford running state on CPU
        counts: Dict[str, int] = {n: 0 for n in shapes}
        means:  Dict[str, Tensor] = {
            n: torch.zeros(s, dtype=torch.float32) for n, s in shapes.items()
        }
        M2s:    Dict[str, Tensor] = {
            n: torch.zeros(s, dtype=torch.float32) for n, s in shapes.items()
        }

        for image in images:
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
                    model(image)
            finally:
                for h in hooks:
                    h.remove()

            # Welford update per layer
            for layer_name, act in captured.items():
                if layer_name not in counts:
                    continue
                x = act.squeeze(0)  # remove batch dim → shape matches stored shape
                if x.shape != means[layer_name].shape:
                    continue        # shape mismatch (e.g. dynamic layers) — skip
                counts[layer_name] += 1
                n = counts[layer_name]
                delta  = x - means[layer_name]
                means[layer_name] += delta / n
                delta2 = x - means[layer_name]
                M2s[layer_name]   += delta * delta2

        # Finalise: compute std from M2
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

        # β = 10% of median live-neuron std.
        # Using the median (not a low percentile) ensures β stays well above the
        # dead-neuron tail even when ReLU networks have 20-40% near-zero neurons.
        all_stds = torch.cat([s.flatten() for s in stds.values()])
        live_stds = all_stds[all_stds > 1e-7]
        if live_stds.numel() > 0:
            beta = torch.median(live_stds).item() * 0.1
        else:
            beta = 1e-3  # fallback for pathological models
        self._neuron_betas[model_id] = beta

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
            'neuron_stds': self._neuron_stds[model_id],
            'beta':        self._neuron_betas[model_id],
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
