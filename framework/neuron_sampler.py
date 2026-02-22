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
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self, model: nn.Module) -> None:
        """Clear the shape cache for a specific model instance."""
        model_id = id(model)
        if model_id in self._shape_cache:
            del self._shape_cache[model_id]
