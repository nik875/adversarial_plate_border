"""
ocr_backends.py

Swappable OCR backend abstraction for the adversarial patch trainer.

The OCR backend sits in the second half of partial_loss: it receives a
cropped licence-plate image and returns character probabilities that the
trainer uses to compute impersonation or disruption loss.

Trainable backends (gradient flows into the patch)
---------------------------------------------------
  crnn          GitYCC/crnn-pytorch — lightweight CRNN, CTC-decoded
                ~5 M params, pure PyTorch, fully differentiable

Evaluation-only backends (no autograd)
---------------------------------------
    trocr         microsoft/trocr-small-printed (Hugging Face)
    dtrb          deep-text-recognition-benchmark style model checkpoint
  fastanpr      fastanpr.recognition.Recogniser — used in original trainer
  mock          Returns fixed predictions — for unit testing

Adding a new backend
--------------------
1. Subclass OCRBackend.
2. Implement load(), predict(), and parameters().
3. Register in REGISTRY at the bottom.
"""

from __future__ import annotations

import abc
from types import SimpleNamespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    """
    Output of one OCR forward pass.

    logits : torch.Tensor or None
        Shape [T, num_classes] — raw CTC log-probabilities.
        Present for trainable backends; None for external pipelines.
        Gradient flows through this tensor into the patch.
    text : str or None
        Decoded plate string, e.g. "ABC123".  None if recognition failed.
    confidence : float
        Scalar confidence in [0, 1].
    """
    logits: Optional[torch.Tensor]
    text: Optional[str]
    confidence: float = 0.0

    def char_accuracy(self, target: str) -> float:
        """Fraction of characters matching target (left-padded with spaces)."""
        if self.text is None:
            return 0.0
        pred = self.text.ljust(len(target))[:len(target)]
        return sum(a == b for a, b in zip(pred, target)) / max(len(target), 1)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class OCRBackend(abc.ABC):
    """
    Interface every OCR backend must satisfy.

    Subclasses must implement:
        load()        — initialise model / download weights
        predict()     — run OCR on a single CHW float32 image tensor
        parameters()  — yield nn.Parameters (so trainer can freeze them)

    The is_trainable property signals whether gradients flow through predict().
    Only trainable backends may be used during adversarial patch training.

    ocr_crop_size : (H, W)
        Size to which the trainer crops the plate region before passing to this
        backend.  Must match the model's expected input resolution.
    """

    name: str = "base"
    is_trainable: bool = False       # override to True in differentiable backends
    ocr_crop_size: tuple = (32, 128) # (H, W) — override in each subclass

    def __init__(self, model_path: str = "none", device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self._loaded = False

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights.  Called once before the first predict()."""

    @abc.abstractmethod
    def predict(self, image: torch.Tensor) -> OCRResult:
        """
        Run OCR on a single cropped plate image.

        Parameters
        ----------
        image : torch.Tensor
            Shape [C, H, W], float32, values in [0, 1].

        Returns
        -------
        OCRResult
            logits carries the gradient graph for trainable backends.
        """

    @abc.abstractmethod
    def parameters(self) -> Iterator[nn.Parameter]:
        """Yield all learnable parameters (for freezing in the trainer)."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def differentiable_loss_batch(
        self,
        crops: list,           # B × [1, 3, H_c, W_c]
        target_text: str,
        impersonation: bool = False,
    ) -> list:
        """Default: sequential loop. Override for batch GPU inference."""
        losses = []
        for crop in crops:
            if self.is_trainable and hasattr(self, "differentiable_loss"):
                losses.append(self.differentiable_loss(
                    crop, target_text, impersonation=impersonation))
            else:
                with torch.no_grad():
                    result = self.predict(crop.squeeze(0))
                acc = result.char_accuracy(target_text)
                losses.append(torch.tensor(
                    (1.0 - acc) if impersonation else acc,
                    device=self.device,
                ))
        return losses

    def eval(self) -> "OCRBackend":
        return self

    def train(self) -> "OCRBackend":
        return self

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"name={self.name!r}, trainable={self.is_trainable})")


# ---------------------------------------------------------------------------
# CRNN architecture (inlined from GitYCC/crnn-pytorch)
# ---------------------------------------------------------------------------
# Defined here so the weights file is the ONLY thing you need — no repo clone.
# Architecture exactly matches GitYCC so crnn_synth90k.pt loads without errors.

class _BidirectionalLSTM(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.rnn       = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.embedding = nn.Linear(n_hidden * 2, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        output  = self.embedding(recurrent.view(T * b, h))
        return output.view(T, b, -1)


class _CRNN(nn.Module):
    """
    Exact replica of GitYCC/crnn-pytorch CRNN class.
    State-dict keys are identical so any GitYCC checkpoint loads cleanly.
    """
    def __init__(self, img_h: int, nc: int, n_class: int,
                 n_hidden: int = 256, leaky_relu: bool = False):
        super().__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()
        activation = (lambda: nn.LeakyReLU(0.2, inplace=True)
                      if leaky_relu else lambda: nn.ReLU(inplace=True))

        def conv_relu(i: int, batch_norm: bool = False):
            n_in  = nc if i == 0 else nm[i - 1]
            n_out = nm[i]
            cnn.add_module(f"conv{i}",      nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            if batch_norm:
                cnn.add_module(f"batchnorm{i}", nn.BatchNorm2d(n_out))
            cnn.add_module(f"relu{i}",      activation()())

        conv_relu(0)
        cnn.add_module("pooling0", nn.MaxPool2d(2, 2))
        conv_relu(1)
        cnn.add_module("pooling1", nn.MaxPool2d(2, 2))
        conv_relu(2, batch_norm=True)
        conv_relu(3)
        cnn.add_module("pooling2", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv_relu(4, batch_norm=True)
        conv_relu(5)
        cnn.add_module("pooling3", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv_relu(6, batch_norm=True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            _BidirectionalLSTM(512, n_hidden, n_hidden),
            _BidirectionalLSTM(n_hidden, n_hidden, n_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)                    # [B, C, 1, W]
        b, c, h, w = features.size()
        assert h == 1, f"CRNN expects height=1 after CNN, got {h}"
        features = features.squeeze(2)            # [B, C, W]
        features = features.permute(2, 0, 1)      # [W, B, C]
        return self.rnn(features)                 # [T, B, n_class]


class _CRNNMapToSeq(nn.Module):
    """
    Alternate CRNN variant used by some public checkpoints.

    Expected key layout includes:
        map_to_seq.*, rnn1.*, rnn2.*, dense.*, cnn.batchnorm5.*
    """

    def __init__(self, img_h: int, nc: int, n_class: int,
                 map_dim: int = 64, n_hidden: int = 256):
        super().__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv_relu(i: int, batch_norm: bool = False):
            n_in = nc if i == 0 else nm[i - 1]
            n_out = nm[i]
            cnn.add_module(f"conv{i}", nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            if batch_norm:
                cnn.add_module(f"batchnorm{i}", nn.BatchNorm2d(n_out))
            cnn.add_module(f"relu{i}", nn.ReLU(inplace=True))

        conv_relu(0)
        cnn.add_module("pooling0", nn.MaxPool2d(2, 2))
        conv_relu(1)
        cnn.add_module("pooling1", nn.MaxPool2d(2, 2))
        conv_relu(2, batch_norm=False)
        conv_relu(3)
        cnn.add_module("pooling2", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module("pooling3", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv_relu(6, batch_norm=False)

        self.cnn = cnn
        self.map_to_seq = nn.Linear(512, map_dim)
        self.rnn1 = nn.LSTM(map_dim, n_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(n_hidden * 2, n_hidden, bidirectional=True)
        self.dense = nn.Linear(n_hidden * 2, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)                    # [B, C, 1, W]
        b, c, h, w = features.size()
        assert h == 1, f"CRNN expects height=1 after CNN, got {h}"
        features = features.squeeze(2)            # [B, C, W]
        features = features.permute(2, 0, 1)      # [W, B, C]

        T, B, C = features.size()
        seq = self.map_to_seq(features.reshape(T * B, C)).reshape(T, B, -1)
        seq, _ = self.rnn1(seq)
        seq, _ = self.rnn2(seq)

        T, B, H = seq.size()
        logits = self.dense(seq.reshape(T * B, H)).reshape(T, B, -1)
        return logits


# ---------------------------------------------------------------------------
# CRNN backend  — uses inlined architecture, loads GitYCC weights directly
# ---------------------------------------------------------------------------
#
# Weights
# -------
#   Download crnn_synth90k.pt (or crnn.pth) from GitYCC/crnn-pytorch releases:
#     https://github.com/GitYCC/crnn-pytorch/releases
#   No repo clone required — just pass the file path.
#
# Alphabet
# --------
#   crnn_synth90k.pt was trained on digits + lowercase a-z (36 chars).
#   The DEFAULT_ALPHABET below matches this exactly.
#   If your checkpoint uses a different charset, pass it as the alphabet kwarg.

class CRNNBackend(OCRBackend):
    """
    Trainable CRNN OCR backend.

    Architecture is inlined — no external repo needed.  Just point model_path
    at your downloaded crnn_synth90k.pt (or any GitYCC-format checkpoint).

    Gradients flow from CTC loss through LSTM and CNN back to the patch tensor.

    Parameters
    ----------
    model_path : str
        Path to a GitYCC CRNN checkpoint (.pt or .pth).
    device : str
        Torch device.
    alphabet : str
        Characters in index order, matching the checkpoint's training charset.
        Default: '0123456789abcdefghijklmnopqrstuvwxyz' (synth90k checkpoint).
    img_height : int
        Input height the model expects (default 32).
    n_hidden : int
        LSTM hidden size (default 256, matches GitYCC default).
    """

    name         = "crnn"
    is_trainable  = True
    ocr_crop_size = (32, None)  # (H, W=None) — resize to h=32, preserve aspect ratio

    # synth90k uses lowercase — matches crnn_synth90k.pt exactly
    DEFAULT_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

    def __init__(self, model_path: str, device: str = "cpu",
                 alphabet: str = DEFAULT_ALPHABET,
                 img_height: int = 32, n_hidden: int = 256):
        super().__init__(model_path, device)
        self.alphabet    = alphabet
        self.img_height  = img_height
        self.n_hidden    = n_hidden
        self.num_classes = len(alphabet) + 1   # +1 for CTC blank
        self._model: Optional[nn.Module] = None

    def _normalise_state_dict_keys(self, state: dict) -> dict:
        """Strip common wrappers (DataParallel/module prefixes)."""
        cleaned = {}
        for k, v in state.items():
            key = k
            if key.startswith("module."):
                key = key[len("module."):]
            cleaned[key] = v
        return cleaned

    def _remap_alt_crnn_keys(self, state: dict) -> tuple[dict, list[str]]:
        """
        Remap common alternate CRNN key names to this inlined GitYCC layout.

        Some public checkpoints use keys like rnn1/rnn2/dense/map_to_seq and
        cnn.batchnorm5 instead of rnn.*/embedding and cnn.batchnorm6.
        """
        mapped = dict(state)
        notes: list[str] = []

        # CNN naming shift seen in some CRNN variants.
        if "cnn.batchnorm5.weight" in state and "cnn.batchnorm6.weight" not in mapped:
            for suffix in [
                "weight", "bias", "running_mean", "running_var", "num_batches_tracked"
            ]:
                src = f"cnn.batchnorm5.{suffix}"
                dst = f"cnn.batchnorm6.{suffix}"
                if src in state and dst not in mapped:
                    mapped[dst] = state[src]
            notes.append("mapped cnn.batchnorm5 -> cnn.batchnorm6")

        # LSTM blocks often use rnn1/rnn2 naming.
        for src_prefix, dst_prefix in [("rnn1", "rnn.0.rnn"), ("rnn2", "rnn.1.rnn")]:
            for suffix in [
                "weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
                "weight_ih_l0_reverse", "weight_hh_l0_reverse",
                "bias_ih_l0_reverse", "bias_hh_l0_reverse",
            ]:
                src = f"{src_prefix}.{suffix}"
                dst = f"{dst_prefix}.{suffix}"
                if src in state and dst not in mapped:
                    mapped[dst] = state[src]
            if any(k.startswith(src_prefix + ".") for k in state):
                notes.append(f"mapped {src_prefix} -> {dst_prefix}")

        # Final classifier in some checkpoints is named dense.*
        if "dense.weight" in state and "rnn.1.embedding.weight" not in mapped:
            mapped["rnn.1.embedding.weight"] = state["dense.weight"]
            notes.append("mapped dense.weight -> rnn.1.embedding.weight")
        if "dense.bias" in state and "rnn.1.embedding.bias" not in mapped:
            mapped["rnn.1.embedding.bias"] = state["dense.bias"]
            notes.append("mapped dense.bias -> rnn.1.embedding.bias")

        # Some variants use map_to_seq as the projection before first BiLSTM.
        if "map_to_seq.weight" in state and "rnn.0.embedding.weight" not in mapped:
            cand = state["map_to_seq.weight"]
            target = self._model.state_dict().get("rnn.0.embedding.weight")
            if target is not None and tuple(cand.shape) == tuple(target.shape):
                mapped["rnn.0.embedding.weight"] = cand
                notes.append("mapped map_to_seq.weight -> rnn.0.embedding.weight")
        if "map_to_seq.bias" in state and "rnn.0.embedding.bias" not in mapped:
            cand = state["map_to_seq.bias"]
            target = self._model.state_dict().get("rnn.0.embedding.bias")
            if target is not None and tuple(cand.shape) == tuple(target.shape):
                mapped["rnn.0.embedding.bias"] = cand
                notes.append("mapped map_to_seq.bias -> rnn.0.embedding.bias")

        return mapped, notes

    def load(self) -> None:
        if str(self.model_path) != "none":
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"CRNN checkpoint not found: {self.model_path}\n"
                    f"Download crnn_synth90k.pt from "
                    f"https://github.com/GitYCC/crnn-pytorch/releases"
                )
            ckpt  = torch.load(str(self.model_path), map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            state = self._normalise_state_dict_keys(state)

            # Some checkpoints use an alternate CRNN layout with map_to_seq +
            # rnn1/rnn2/dense names. Detect and build the matching model.
            is_map_to_seq_variant = "map_to_seq.weight" in state
            if is_map_to_seq_variant:
                map_dim = int(state["map_to_seq.weight"].shape[0])
                hidden = int(state["rnn1.weight_hh_l0"].shape[1]) if "rnn1.weight_hh_l0" in state else self.n_hidden
                self._model = _CRNNMapToSeq(
                    img_h=self.img_height,
                    nc=1,
                    n_class=self.num_classes,
                    map_dim=map_dim,
                    n_hidden=hidden,
                )
                print(f"[{self.name}] Detected map_to_seq CRNN checkpoint format")
            else:
                self._model = _CRNN(
                    img_h=self.img_height,
                    nc=1,
                    n_class=self.num_classes,
                    n_hidden=self.n_hidden,
                )

            try:
                self._model.load_state_dict(state, strict=True)
            except RuntimeError:
                compat_state, notes = self._remap_alt_crnn_keys(state)
                load_info = self._model.load_state_dict(compat_state, strict=False)
                if notes:
                    print(f"[{self.name}] Compatibility remap: {', '.join(notes)}")
                if load_info.missing_keys:
                    print(
                        f"[{self.name}] Warning: {len(load_info.missing_keys)} missing keys "
                        "after compatibility remap; unmatched layers keep random init."
                    )
                if load_info.unexpected_keys:
                    print(
                        f"[{self.name}] Warning: {len(load_info.unexpected_keys)} unexpected "
                        "checkpoint keys were ignored."
                    )
            print(f"[{self.name}] Loaded weights from {self.model_path}")
        else:
            self._model = _CRNN(
                img_h=self.img_height,
                nc=1,
                n_class=self.num_classes,
                n_hidden=self.n_hidden,
            )
            print(f"[{self.name}] No checkpoint — using random weights")

        self._model.to(self.device)
        self._model.eval()

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convert CHW float32 RGB [0,1] → 1HW float32 greyscale, resized to
        img_height, normalised to [-1, 1] as GitYCC expects.
        """
        # RGB → greyscale via luminance weights
        grey = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).unsqueeze(0)
        # Resize height, keep aspect ratio
        _, H, W = grey.shape
        new_w   = max(1, int(W * self.img_height / H))
        grey    = F.interpolate(
            grey.unsqueeze(0), size=(self.img_height, new_w),
            mode="bilinear", align_corners=False,
        )  # [1, 1, H, W]
        # Normalise to [-1, 1]
        grey = (grey - 0.5) / 0.5
        return grey  # [1, 1, img_height, W]

    def predict(self, image: torch.Tensor) -> OCRResult:
        """
        Forward pass.  Returns OCRResult with logits attached to autograd graph.
        """
        self.ensure_loaded()
        inp     = self._preprocess(image.to(self.device))   # [1, 1, H, W]
        logits  = self._model(inp)                          # [T, 1, num_classes]

        # Greedy CTC decode (no beam search needed for plate recognition)
        log_probs = F.log_softmax(logits, dim=2)            # [T, 1, C]
        pred_ids  = log_probs.argmax(dim=2).squeeze(1)      # [T]

        text, confidence = self._ctc_decode(
            pred_ids.detach().cpu(), log_probs.detach().cpu()
        )

        return OCRResult(
            logits=logits.squeeze(1),    # [T, num_classes]  grad-connected
            text=text,
            confidence=confidence,
        )

    def _ctc_decode(self, pred_ids: torch.Tensor,
                    log_probs: torch.Tensor) -> tuple[Optional[str], float]:
        """Greedy CTC decode: collapse repeats, remove blanks."""
        # crnn_synth90k.pt uses blank=0 (Baek/DTRB convention):
        #   index 0 → CTC blank, indices 1..N → alphabet[0..N-1]
        blank_id = 0
        chars    = []
        prev     = blank_id
        confs    = []

        for t, idx in enumerate(pred_ids.tolist()):
            if idx != prev and idx != blank_id:
                chars.append(self.alphabet[idx - 1])   # -1: blank at 0 shifts chars up by 1
                confs.append(log_probs[t, 0, idx].exp().item())
            prev = idx

        if not chars:
            return None, 0.0

        return "".join(chars), float(sum(confs) / len(confs))

    def ctc_loss(self, logits: torch.Tensor, target_text: str) -> torch.Tensor:
        """
        Convenience method: compute CTC loss between model logits and a
        target string.  Useful for impersonation training.

        Parameters
        ----------
        logits : torch.Tensor
            Shape [T, num_classes] — from OCRResult.logits.
        target_text : str
            Desired plate string, e.g. "ABC123".

        Returns
        -------
        torch.Tensor
            Scalar CTC loss (lower = closer to target_text).
        """
        log_probs   = F.log_softmax(logits.unsqueeze(1), dim=2)  # [T, 1, C]
        # blank=0 convention: character indices are 1-based
        target_ids  = torch.tensor(
            [self.alphabet.index(c) + 1 for c in target_text if c in self.alphabet],
            dtype=torch.long,
        )
        input_len   = torch.tensor([log_probs.shape[0]], dtype=torch.long)
        target_len  = torch.tensor([len(target_ids)],    dtype=torch.long)

        return F.ctc_loss(
            log_probs, target_ids.unsqueeze(0),
            input_len, target_len,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )

    def differentiable_loss(self, crop: torch.Tensor, target_text: str,
                             impersonation: bool = False) -> torch.Tensor:
        """
        Fully differentiable CTC loss on a crop tensor.

        Parameters
        ----------
        crop : torch.Tensor
            Shape [1, 3, H, W] from kornia.geometry.crop_and_resize.
        target_text : str
            Plate string to move toward (impersonation) or away from (disruption).
        impersonation : bool
            True → minimise CTC loss toward target_text.
            False → maximise CTC loss (negate).
        """
        self.ensure_loaded()
        preprocessed = self._preprocess(crop)          # [1, 1, 32, W]
        logits = self._model(preprocessed).squeeze(1)  # [T, num_classes]
        base = self.ctc_loss(logits, target_text)
        return base if impersonation else -base

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "CRNNBackend":
        self.ensure_loaded()
        if self._model is not None:
            self._model.eval()
        return self

    def train(self) -> "CRNNBackend":
        self.ensure_loaded()
        if self._model is not None:
            self._model.train()
        return self

    def to(self, device: str) -> "CRNNBackend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# ---------------------------------------------------------------------------
# LPRNet backend  — NVIDIA TAO US LPRNet (lprnet_torch.py), differentiable
# ---------------------------------------------------------------------------
#
# Wraps the native PyTorch reconstruction of the NVIDIA TAO US LPRNet ONNX
# model.  Weights are loaded directly from the .onnx file via
# load_weights_from_onnx(); the full ResNet+LSTM graph is differentiable.
#
# Architecture: ResNet backbone (res2–res5) + forward LSTM(3600→512) + Dense
# Input:  [B, 3, 48, 96]  RGB, values in [0, 1]
# Output: [B, 24, 36]     softmax probabilities (CTC blank = index 35)
#
# Weights
# -------
#   Pass the path to the .onnx file (e.g. us_lprnet_patched.onnx).
#   Pass "none" to use random weights (for architecture testing only).


class LPRNetBackend(OCRBackend):
    """
    Trainable NVIDIA TAO US LPRNet OCR backend.

    Wraps LPRNetTorch (lprnet_torch.py) — a native PyTorch reconstruction of
    the NVIDIA TAO us_lprnet_baseline18_deployable ONNX model.  The full
    ResNet+LSTM graph is differentiable, so CTC loss flows back into the patch.

    Parameters
    ----------
    model_path : str
        Path to the ONNX file (e.g. us_lprnet_patched.onnx).
        Pass "none" to skip weight loading (random weights — for testing only).
    device : str
        Torch device string ("cpu", "cuda", "cuda:0", …).
    """

    name          = "lprnet"
    is_trainable  = True
    ocr_crop_size = (48, 96)   # (H, W) — NVIDIA TAO LPRNet input size

    ALPHABET  = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
    BLANK_IDX = 35   # last index; matches NVIDIA TAO convention

    def __init__(self, model_path: str = "none", device: str = "cpu"):
        super().__init__(model_path, device)
        self._model: Optional[nn.Module] = None

    def load(self) -> None:
        from lprnet_torch import LPRNetTorch, load_weights_from_onnx
        self._model = LPRNetTorch()
        if str(self.model_path) != "none":
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"LPRNet ONNX file not found: {self.model_path}"
                )
            load_weights_from_onnx(self._model, str(self.model_path))
            print(f"[{self.name}] Loaded weights from {self.model_path}")
        else:
            print(f"[{self.name}] No checkpoint — using random weights")
        self._model.to(self.device)
        self._model.eval()

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """CHW or NCHW float32 [0,1] → [1, 3, 48, 96]."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return F.interpolate(
            image, size=(48, 96),
            mode="bilinear", align_corners=False,
        ).to(self.device)

    def predict(self, image: torch.Tensor) -> OCRResult:
        self.ensure_loaded()
        inp   = self._preprocess(image)          # [1, 3, 48, 96]
        probs = self._model(inp)                 # [1, 24, 36]  softmax probs
        text, confidence = self._greedy_decode(probs[0].detach().cpu())
        # expose log-probs as [T, C] for any external CTC use
        log_probs = torch.log(probs[0].clamp(min=1e-8))   # [24, 36]
        return OCRResult(
            logits=log_probs,
            text=text,
            confidence=confidence,
        )

    def _greedy_decode(self, probs: torch.Tensor) -> tuple:
        """Greedy CTC decode on [T, C] softmax probabilities."""
        pred = probs.argmax(dim=-1)   # [T]
        chars, confs, prev = [], [], self.BLANK_IDX
        for t, idx in enumerate(pred.tolist()):
            if idx != prev and idx != self.BLANK_IDX:
                chars.append(self.ALPHABET[idx])
                confs.append(probs[t, idx].item())
            prev = idx
        if not chars:
            return None, 0.0
        return "".join(chars), float(sum(confs) / len(confs))

    def ctc_loss(self, log_probs: torch.Tensor, target_text: str) -> torch.Tensor:
        """CTC loss on [T, C] log-probabilities."""
        lp = log_probs.unsqueeze(1)   # [T, 1, C]
        target_ids = torch.tensor(
            [self.ALPHABET.index(c) for c in target_text if c in self.ALPHABET],
            dtype=torch.long,
        )
        input_len  = torch.tensor([lp.shape[0]], dtype=torch.long)
        target_len = torch.tensor([len(target_ids)], dtype=torch.long)
        return F.ctc_loss(
            lp, target_ids.unsqueeze(0),
            input_len, target_len,
            blank=self.BLANK_IDX,
            reduction="mean", zero_infinity=True,
        )

    def differentiable_loss(self, crop: torch.Tensor, target_text: str,
                             impersonation: bool = False) -> torch.Tensor:
        """Differentiable CTC loss on a [1, 3, H, W] crop tensor."""
        self.ensure_loaded()
        probs     = self._model(self._preprocess(crop))       # [1, 24, 36]
        log_probs = torch.log(probs[0].clamp(min=1e-8))      # [24, 36]
        base = self.ctc_loss(log_probs, target_text)
        return base if impersonation else -base

    def differentiable_loss_batch(self, crops: list, target_text: str,
                                   impersonation: bool = False) -> list:
        """True batch forward: preprocess all crops, one model call."""
        self.ensure_loaded()
        inp = torch.cat([self._preprocess(c) for c in crops], dim=0)  # [B, 3, 48, 96]
        probs = self._model(inp)   # [B, 24, 36]
        losses = []
        for i in range(len(crops)):
            log_probs = torch.log(probs[i].clamp(min=1e-8))  # [24, 36]
            base = self.ctc_loss(log_probs, target_text)
            losses.append(base if impersonation else -base)
        return losses

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "LPRNetBackend":
        self.ensure_loaded()
        if self._model is not None:
            self._model.eval()
        return self

    def train(self) -> "LPRNetBackend":
        self.ensure_loaded()
        if self._model is not None:
            self._model.train()
        return self

    def to(self, device: str) -> "LPRNetBackend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# ---------------------------------------------------------------------------
# fastanpr OCR backend  — wraps the existing recogniser (evaluation only)
# ---------------------------------------------------------------------------

class FastANPROCRBackend(OCRBackend):
    """
    Thin wrapper around fastanpr.recognition.Recogniser.

    Matches the interface the original trainer used, but now pluggable.
    Not differentiable — use for baseline evaluation only.
    """

    name         = "fastanpr-ocr"
    is_trainable  = False
    ocr_crop_size = (64, 128)   # (H, W)

    def __init__(self, model_path: str = "none", device: str = "cpu"):
        super().__init__(model_path, device)
        self._recogniser = None

    def load(self) -> None:
        import fastanpr
        self._recogniser = fastanpr.recognition.Recogniser(device=self.device)
        print(f"[{self.name}] fastanpr Recogniser loaded")

    def predict(self, image: torch.Tensor) -> OCRResult:
        self.ensure_loaded()
        img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        result = self._recogniser.run(img_np)

        if result is None:
            return OCRResult(logits=None, text=None, confidence=0.0)

        return OCRResult(
            logits=None,
            text=result.text,
            confidence=float(result.conf) if hasattr(result, "conf") else 0.0,
        )

    def parameters(self) -> Iterator[nn.Parameter]:
        return iter([])


# ---------------------------------------------------------------------------
# TrOCR backend  (Hugging Face microsoft/trocr-small-printed)
# ---------------------------------------------------------------------------

class TrOCROCRBackend(OCRBackend):
    """
    Wrapper around Hugging Face TrOCR.

    Default model id is ``microsoft/trocr-small-printed``. Pass another HF id
    (or a local checkpoint dir) via model_path to override.

    This backend supports differentiable loss for training via
    differentiable_loss() which bypasses PIL entirely.
    """

    name         = "trocr"
    is_trainable  = True
    ocr_crop_size = (384, 384)   # (H, W) — TrOCR expects 384×384
    DEFAULT_MODEL_ID = "microsoft/trocr-small-printed"

    DEFAULT_WEIGHTS = "weights/trocr_small_finetuned.pt"

    def __init__(self, model_path: str = "none", device: str = "cpu",
                 max_new_tokens: int = 16):
        if model_path == "none":
            model_path = self.DEFAULT_WEIGHTS
        super().__init__(model_path, device)
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None

    def load(self) -> None:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for trocr backend. "
                "Install with: pip install transformers"
            ) from exc

        self._processor = TrOCRProcessor.from_pretrained(self.DEFAULT_MODEL_ID)
        self._model     = VisionEncoderDecoderModel.from_pretrained(self.DEFAULT_MODEL_ID)

        # Configure model for training with labels
        # VisionEncoderDecoderConfig needs pad_token_id and decoder_start_token_id
        self._model.config.decoder_start_token_id = self._processor.tokenizer.bos_token_id
        self._model.config.pad_token_id           = self._processor.tokenizer.pad_token_id

        weights_path = Path(str(self.model_path))
        if str(self.model_path) not in {"", "none"} and weights_path.exists():
            import torch as _torch
            self._model.load_state_dict(
                _torch.load(str(weights_path), map_location=self.device)
            )
            print(f"[{self.name}] Loaded fine-tuned weights: {weights_path}")
        else:
            if str(self.model_path) not in {"", "none"}:
                print(f"[{self.name}] WARNING: {self.model_path} not found — using pretrained weights")
            print(f"[{self.name}] Loaded pretrained model from {self.DEFAULT_MODEL_ID}")

        self._model.to(self.device)
        self._model.eval()

    def _to_pil(self, image: torch.Tensor):
        from PIL import Image

        img_np = (image.permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
        return Image.fromarray(img_np)

    def predict(self, image: torch.Tensor) -> OCRResult:
        self.ensure_loaded()

        pil_img = self._to_pil(image)
        pixel_values = self._processor(images=pil_img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            generated = self._model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

        text = self._processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
        confidence = 0.0
        if generated.scores:
            step_conf = [
                torch.softmax(scores, dim=-1).max(dim=-1).values.mean().item()
                for scores in generated.scores
            ]
            confidence = float(sum(step_conf) / max(len(step_conf), 1))

        return OCRResult(logits=None, text=(text if text else None), confidence=confidence)

    def compute_target_loss(self, image: torch.Tensor, target_text: str) -> torch.Tensor:
        """
        Differentiable sequence loss toward target_text.

        Uses teacher forcing on TrOCR with labels and returns model loss.
        Gradients flow to image input (and therefore the adversarial patch).
        """
        self.ensure_loaded()

        pil_img = self._to_pil(image)
        pixel_values = self._processor(images=pil_img, return_tensors="pt").pixel_values.to(self.device)

        labels = self._processor.tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_new_tokens,
        ).input_ids.to(self.device)

        pad_id = self._processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels = labels.masked_fill(labels == pad_id, -100)

        outputs = self._model(pixel_values=pixel_values, labels=labels)
        return outputs.loss

    def _sequence_log_prob(
        self, pixel_values: torch.Tensor, text: str
    ) -> torch.Tensor:
        """
        Log P(text | image) via a single teacher-forced forward pass.

        pixel_values : [1, 3, H, W], already normalised to [-1, 1].
        Returns a scalar log-probability (≤ 0).  Fully differentiable.
        """
        tok = self._processor.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_new_tokens,
        ).input_ids.to(self.device)   # [1, L]

        if tok.shape[1] < 2:
            return torch.tensor(float("-inf"), device=self.device,
                                dtype=pixel_values.dtype)

        decoder_input_ids = tok[:, :-1]   # [1, L-1]: BOS + all-but-last
        label_ids         = tok[:, 1:]    # [1, L-1]: all-but-BOS

        outputs   = self._model(pixel_values=pixel_values,
                                decoder_input_ids=decoder_input_ids)
        log_probs = F.log_softmax(outputs.logits, dim=-1)   # [1, L-1, vocab]
        token_lp  = log_probs.gather(2, label_ids.unsqueeze(-1)).squeeze(-1)
        return token_lp.mean()   # mean per-token log P (normalised by length)

    def differentiable_loss(self, crop: torch.Tensor, target_text: str,
                             impersonation: bool = False,
                             variants: list = []) -> torch.Tensor:
        """
        Differentiable loss based on the combined log-probability of the target
        text and any spelling variants (e.g. "VRJ-7774", "VRJ 7774").

        Computes log P(any variant | image) via log-sum-exp of individual
        teacher-forced log-probabilities, which is numerically stable and
        avoids vanishing products.

        Disruption  (impersonation=False): minimise log P → probability falls.
        Impersonation (impersonation=True): maximise log P → probability rises.

        crop     : [1, 3, 384, 384] in [0, 1].
        variants : extra acceptable spellings of target_text (base NOT included).
        """
        self.ensure_loaded()
        pixel_values = (crop - 0.5) / 0.5   # [1, 3, H, W], differentiable

        all_texts  = [target_text] + list(variants)
        log_probs  = torch.stack([
            self._sequence_log_prob(pixel_values, t) for t in all_texts
        ])
        total_log_prob = torch.logsumexp(log_probs, dim=0)   # log P(any variant)

        # Disruption : return log P  (optimiser minimises → P decreases)
        # Impersonation: return -log P (optimiser minimises → P increases)
        return -total_log_prob if impersonation else total_log_prob

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "TrOCROCRBackend":
        self.ensure_loaded()
        if self._model is not None:
            self._model.eval()
        return self

    def train(self) -> "TrOCROCRBackend":
        self.ensure_loaded()
        if self._model is not None:
            self._model.train()
        return self


# ---------------------------------------------------------------------------
# deep-text-recognition-benchmark backend  (roatienza fork compatible)
# ---------------------------------------------------------------------------

class DeepTextRecognitionBenchmarkOCRBackend(OCRBackend):
    """
    Wrapper for deep-text-recognition-benchmark style models.

    Expected checkpoint is a .pth exported from a DTRB-compatible ``Model``.
    By default this backend tries to import ``model.py`` and ``utils.py`` from
    the current Python path. If needed, pass ``dtrb_root=...`` (repo directory)
    when constructing the backend.
    """

    name         = "dtrb"
    # CTC-based DTRB models can provide differentiable logits for training.
    # Attention-mode checkpoints remain eval-only in this backend.
    is_trainable  = True
    ocr_crop_size = (224, 224)   # (H, W) — ViTSTR/DTRB expects 224×224
    DEFAULT_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtrb_root: Optional[str] = None,
        character: str = DEFAULT_CHARS,
        img_h: int = 32,
        img_w: int = 100,
        max_label_length: int = 25,
        prediction: str = "CTC",
        transformation: str = "None",
        feature_extraction: str = "ResNet",
        sequence_modeling: str = "BiLSTM",
    ):
        super().__init__(model_path, device)

        # Auto-detect ViTSTR: ViTSTR uses attention decoding, not CTC.
        # Silently override prediction="CTC" → "Attn" for ViTSTR models so that
        # the architecture built from opt matches the checkpoint.
        is_vitstr = "vitstr" in feature_extraction.lower() or "vit" in feature_extraction.lower()
        if is_vitstr and prediction.upper() == "CTC":
            print(f"[dtrb] ViTSTR detected — overriding prediction CTC → Attn")
            prediction = "Attn"

        requested_prediction = prediction.upper()
        if requested_prediction not in {"CTC", "ATTN"}:
            raise ValueError("dtrb prediction must be one of: CTC, Attn")

        # Instance-level flag so trainer can route loss correctly.
        self.is_trainable = requested_prediction == "CTC"
        self.dtrb_root = dtrb_root
        self.character = character

        # Adjust image size for ViTSTR
        if is_vitstr and img_h == 32 and img_w == 100:
            img_h = 224
            img_w = 224
        
        self.img_h = img_h
        self.img_w = img_w
        self.max_label_length = max_label_length
        self.prediction = requested_prediction
        self.transformation = transformation
        self.feature_extraction = feature_extraction
        self.sequence_modeling = sequence_modeling
        self._model = None
        self._converter = None
        self._prediction = prediction

    def load(self) -> None:
        import sys

        if self.dtrb_root:
            root = str(self.dtrb_root)
            if root not in sys.path:
                sys.path.insert(0, root)

        try:
            from model import Model as DTRBModel
            from utils import CTCLabelConverter, AttnLabelConverter
        except ImportError as exc:
            hint = (
                f"Could not import DTRB modules from sys.path.\n"
                f"  dtrb_root provided: {self.dtrb_root or 'None'}\n"
                f"  sys.path[0]: {sys.path[0] if sys.path else 'empty'}\n\n"
                f"Solutions:\n"
                f"  1. Clone the repo: git clone https://github.com/clovaai/deep-text-recognition-benchmark\n"
                f"  2. Pass --ocr-repo-root /path/to/deep-text-recognition-benchmark\n"
                f"  3. Add the repo to PYTHONPATH before running\n"
            )
            raise ImportError(hint) from exc

        if str(self.model_path) in {"", "none"}:
            raise ValueError("dtrb backend requires --ocr-model-path to a .pth checkpoint")
        if not self.model_path.exists():
            raise FileNotFoundError(f"DTRB checkpoint not found: {self.model_path}")

        # Load checkpoint to inspect architecture
        state = torch.load(str(self.model_path), map_location="cpu")
        state = state.get("state_dict", state)

        # Strip DataParallel 'module.' prefix if present (checkpoint saved with nn.DataParallel)
        if any(k.startswith("module.") for k in state):
            state = {(k[len("module."):] if k.startswith("module.") else k): v
                     for k, v in state.items()}
            print(f"[dtrb] Stripped 'module.' prefix from checkpoint keys")

        pred = self.prediction
        self._prediction = "CTC" if pred == "CTC" else "Attn"

        # Auto-detect ViTSTR from feature extraction name
        is_vitstr = "vitstr" in self.feature_extraction.lower() or "vit" in self.feature_extraction.lower()

        # Detect actual input channels from checkpoint
        input_channel = 1  # default
        for k, v in state.items():
            if "patch_embed" in k and "proj.weight" in k:
                input_channel = int(v.shape[1])  # [out_ch, in_ch, h, w]
                break
        self.input_channel = input_channel

        # Auto-detect character set from head shape.
        # AttnLabelConverter adds 2 special tokens ([GO], [s]), so:
        #   num_class = 2 + len(character)
        # DTRB "sensitive" mode uses string.printable[:-6] (94 chars) → 96 classes.
        if is_vitstr:
            import string as _string
            for k, v in state.items():
                if k.endswith("head.weight") and v.dim() == 2:
                    detected_num_class = v.shape[0]
                    expected_char_len = detected_num_class - 2  # subtract GO + EOS tokens
                    if expected_char_len != len(self.character):
                        if expected_char_len == 94:
                            self.character = _string.printable[:-6]
                        else:
                            # Best-effort: truncate or warn
                            print(f"[dtrb] WARNING: head has {detected_num_class} classes but "
                                  f"character set gives {len(self.character)+2}. "
                                  f"Override --dtrb-character if results are wrong.")
                        print(f"[dtrb] Auto-detected character set: {detected_num_class} classes "
                              f"→ {len(self.character)} chars")
                    break

        converter_cls = CTCLabelConverter if self._prediction == "CTC" else AttnLabelConverter
        self._converter = converter_cls(self.character)

        opt = SimpleNamespace(
            Transformation=self.transformation,
            FeatureExtraction=self.feature_extraction,
            SequenceModeling=self.sequence_modeling,
            Prediction=self._prediction,
            Transformer=is_vitstr,  # Enable ViTSTR mode
            TransformerModel=self.feature_extraction if is_vitstr else None,
            num_fiducial=20,
            imgH=self.img_h,
            imgW=self.img_w,
            input_channel=input_channel,
            output_channel=512,
            hidden_size=256,
            num_class=len(self._converter.character),
            batch_max_length=self.max_label_length,
            character=self.character,
            rgb=(input_channel == 3),  # True if RGB, False if grayscale
        )

        self._model = DTRBModel(opt)
        self._model.load_state_dict(state, strict=False)
        self._model.to(self.device)
        self._model.eval()
        print(f"[{self.name}] Loaded weights from {self.model_path}")

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        # Use the input_channel detected from checkpoint
        if hasattr(self, 'input_channel') and self.input_channel == 3:
            # RGB input (standard ViTSTR)
            resized = F.interpolate(
                image.unsqueeze(0),
                size=(self.img_h, self.img_w),
                mode="bilinear",
                align_corners=False,
            )
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            resized = (resized - mean) / std
        else:
            # Grayscale input (traditional DTRB or grayscale ViTSTR)
            grey = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).unsqueeze(0)
            resized = F.interpolate(
                grey.unsqueeze(0),
                size=(self.img_h, self.img_w),
                mode="bilinear",
                align_corners=False,
            )
            resized = (resized - 0.5) / 0.5
        return resized.to(self.device)

    def predict(self, image: torch.Tensor) -> OCRResult:
        self.ensure_loaded()
        inp = self._preprocess(image)

        batch_size = inp.size(0)
        if self._prediction == "CTC":
            # Keep graph when trainable so CTC loss can backprop into patch.
            if self.is_trainable:
                text_for_pred = torch.LongTensor(batch_size, self.max_label_length + 1).fill_(0).to(self.device)
                preds = self._model(inp, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                decoded = self._converter.decode(preds_index, preds_size)
                probs = torch.softmax(preds, dim=2)
                confidence = float(probs.max(2).values.mean().item())
                text = decoded[0] if decoded else None
            else:
                with torch.no_grad():
                    text_for_pred = torch.LongTensor(batch_size, self.max_label_length + 1).fill_(0).to(self.device)
                    preds = self._model(inp, text_for_pred)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    decoded = self._converter.decode(preds_index, preds_size)
                    probs = torch.softmax(preds, dim=2)
                    confidence = float(probs.max(2).values.mean().item())
                    text = decoded[0] if decoded else None
            logits = preds.squeeze(0) if preds.dim() == 3 and preds.size(0) == 1 else None
        else:
            with torch.no_grad():
                text_for_pred = torch.LongTensor(batch_size, self.max_label_length + 1).fill_(0).to(self.device)
                preds = self._model(inp, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                length_for_pred = torch.IntTensor([self.max_label_length] * batch_size)
                decoded = self._converter.decode(preds_index, length_for_pred)
                probs = torch.softmax(preds, dim=2)
                confidence = float(probs.max(2).values.mean().item())
                text = decoded[0] if decoded else None
                if text is not None and "[s]" in text:
                    text = text.split("[s]")[0]
            logits = None

        text = text.strip() if isinstance(text, str) else text
        return OCRResult(logits=logits, text=(text if text else None), confidence=confidence)

    def ctc_loss(self, logits: torch.Tensor, target_text: str) -> torch.Tensor:
        """
        CTC loss for DTRB CTC models.

        Expects logits shape [T, C] from OCRResult.logits.
        """
        if self._prediction != "CTC":
            raise RuntimeError("ctc_loss is only valid when DTRB prediction=CTC")

        if logits.dim() != 2:
            raise ValueError(f"Expected logits [T, C], got shape {tuple(logits.shape)}")

        # Most DTRB plate checkpoints use lowercase alpha chars.
        normalized = target_text.lower()
        if hasattr(self._converter, "dict"):
            indices = [self._converter.dict[c] for c in normalized if c in self._converter.dict]
        else:
            indices = [self.character.index(c) + 1 for c in normalized if c in self.character]

        if len(indices) == 0:
            # Keep training stable when target text has no overlap with charset.
            return logits.new_tensor(0.0)

        targets = torch.tensor(indices, dtype=torch.long, device=logits.device)
        log_probs = F.log_softmax(logits.unsqueeze(1), dim=2)  # [T, 1, C]
        input_len = torch.tensor([log_probs.shape[0]], dtype=torch.long, device=logits.device)
        target_len = torch.tensor([targets.numel()], dtype=torch.long, device=logits.device)

        return F.ctc_loss(
            log_probs,
            targets,
            input_len,
            target_len,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )

    def differentiable_loss(self, crop: torch.Tensor, target_text: str,
                             impersonation: bool = False) -> torch.Tensor:
        """Differentiable CTC loss on a [1, 3, H, W] crop tensor (CTC mode only)."""
        if self._prediction != "CTC" or not self.is_trainable:
            return torch.tensor(0.0, device=self.device)
        self.ensure_loaded()
        preprocessed = self._preprocess(crop)
        text_for_pred = torch.LongTensor(1, self.max_label_length + 1).fill_(0).to(self.device)
        preds = self._model(preprocessed, text_for_pred)  # [1, T, C]
        logits = preds.squeeze(0)                          # [T, C]
        base = self.ctc_loss(logits, target_text)
        return base if impersonation else -base

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()


# ---------------------------------------------------------------------------
# Mock OCR backend  — for unit testing
# ---------------------------------------------------------------------------

class MockOCRBackend(OCRBackend):
    name         = "mock-ocr"
    is_trainable  = False
    ocr_crop_size = (32, 128)

    def __init__(self, fixed_text: str = "ABC123", device: str = "cpu"):
        super().__init__("none", device)
        self._text = fixed_text

    def load(self) -> None:
        print(f"[{self.name}] Mock OCR loaded (always returns '{self._text}')")

    def predict(self, image: torch.Tensor) -> OCRResult:
        self.ensure_loaded()
        return OCRResult(logits=None, text=self._text, confidence=0.9)

    def parameters(self) -> Iterator[nn.Parameter]:
        return iter([])


# ---------------------------------------------------------------------------
# CCT (Compact Character Transcription) backend  — native PyTorch via cct_ocr_torch.py
# ---------------------------------------------------------------------------
#
# Model path (auto-downloaded by fast-plate-ocr on first use):
#   ~/.cache/fast-plate-ocr/cct-s-v1-global-model/cct_s_v1_global.onnx
#
# Input convention:
#   NHWC float32 [0, 255], shape [batch, 64, 128, 3]
#
# Output convention:
#   [batch, seq_len=9, vocab_size=37]  — one softmax probability per plate slot
#
# Alphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'  (_ = pad/blank)

class CCTOCRBackend(OCRBackend):
    """
    Differentiable CCT (fast-plate-ocr) backend via native PyTorch.

    Uses CCTOCRTorch from cct_ocr_torch.py — a pure PyTorch reconstruction
    of the cct-s-v1-global ONNX model with weights loaded directly from the
    ONNX file.  No onnx2torch required.

    Unlike CTC-based backends, CCT classifies each plate character position
    directly (fixed-length output), so cross-entropy (not CTC) is used.
    """

    name         = "cct"
    is_trainable  = True
    ocr_crop_size = (64, 128)   # (H, W)

    ONNX_PATH = ("~/.cache/fast-plate-ocr/"
                 "cct-s-v1-global-model/cct_s_v1_global.onnx")
    ALPHABET  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"

    def __init__(self, model_path: str = "none", device: str = "cpu"):
        super().__init__(model_path, device)
        self._model: Optional[nn.Module] = None

    def load(self) -> None:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from cct_ocr_torch import CCTOCRTorch

        onnx_path = Path(self.ONNX_PATH).expanduser()

        if not onnx_path.exists():
            print(f"[{self.name}] ONNX model not found — downloading via fast-plate-ocr…")
            try:
                from fast_plate_ocr import ONNXPlateRecognizer
                ONNXPlateRecognizer("global-plates-mobile-vit-v2-model")
            except Exception:
                pass  # model may still be downloaded into the cache dir

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"[{self.name}] CCT ONNX model not found: {onnx_path}\n"
                "Install fast-plate-ocr and run `python -c \"from fast_plate_ocr import "
                "ONNXPlateRecognizer; ONNXPlateRecognizer('global-plates-mobile-vit-v2-model')\"`"
            )

        self._model = CCTOCRTorch.from_onnx(str(onnx_path))
        self._model.to(self.device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        print(f"[{self.name}] Loaded from {onnx_path}")

    def _preprocess(self, crop: torch.Tensor) -> torch.Tensor:
        """
        Convert [1, 3, H, W] CHW float [0,1] → NHWC [1, 64, 128, 3] float [0,255].
        Differentiable (only F.interpolate + permute + scale).
        """
        x = F.interpolate(crop, size=(64, 128), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1)   # NHWC: [1, 64, 128, 3]
        return x * 255.0

    def predict(self, image: torch.Tensor) -> "OCRResult":
        """
        Non-differentiable eval path.

        Parameters
        ----------
        image : torch.Tensor
            CHW float32 [0, 1], any spatial size.
        """
        self.ensure_loaded()
        # Preprocess: resize to (64,128), convert to NHWC [0,255]
        with torch.no_grad():
            x = self._preprocess(image.unsqueeze(0).to(self.device))  # [1,64,128,3]
            output = self._model(x)                                     # [1,9,37]

        logits = output[0]                         # [9, 37]
        pred_ids = logits.argmax(dim=1).tolist()   # [9]
        chars = [self.ALPHABET[i] for i in pred_ids if i < len(self.ALPHABET)]
        text = "".join(c for c in chars if c != "_").strip() or None

        # logits are already softmax probabilities from the model
        confidence = float(logits.max(dim=1).values.mean().item())

        return OCRResult(logits=logits.detach(), text=text, confidence=confidence)

    def ce_loss(self, probs: torch.Tensor, target_text: str) -> torch.Tensor:
        """
        Per-slot NLL loss.  Expects softmax *probabilities* [1/T, vocab] as
        output by CCTOCRTorch (which returns softmax, not raw logits).

        Parameters
        ----------
        probs : torch.Tensor
            Shape [1, T, vocab] or [T, vocab] — softmax probabilities.
        target_text : str
            Plate string, e.g. "VRJ7774".  Padded/trimmed to T slots with '_'.
        """
        if probs.dim() == 3:
            probs = probs.squeeze(0)   # [T, vocab]
        T = probs.shape[0]
        padded = (target_text + "_" * T)[:T]
        target_ids = torch.tensor(
            [self.ALPHABET.index(c) if c in self.ALPHABET else len(self.ALPHABET) - 1
             for c in padded],
            dtype=torch.long, device=probs.device,
        )
        # CCTOCRTorch returns softmax probs — use NLL loss directly to avoid double-softmax
        return F.nll_loss(torch.log(probs.clamp(min=1e-8)), target_ids)

    def differentiable_loss(self, crop: torch.Tensor, target_text: str,
                             impersonation: bool = False) -> torch.Tensor:
        """
        Differentiable NLL loss on a [1, 3, H, W] crop tensor.
        Gradients flow through CCTOCRTorch model → crop → patch.
        """
        self.ensure_loaded()
        preprocessed = self._preprocess(crop.to(self.device))   # [1, 64, 128, 3]
        output = self._model(preprocessed)                       # [1, 9, 37]
        base = self.ce_loss(output, target_text)
        return base if impersonation else -base

    def differentiable_loss_batch(self, crops: list, target_text: str,
                                   impersonation: bool = False) -> list:
        """
        True batch forward: one model call for all crops.

        crops : list of [1, 3, H, W] tensors (may vary in H/W; resized in preprocess)
        """
        self.ensure_loaded()
        # Stack + preprocess all crops in one F.interpolate call
        batched = torch.cat([c.to(self.device) for c in crops], dim=0)  # [B, 3, H, W]
        preprocessed = self._preprocess(batched)                         # [B, 64, 128, 3]
        output = self._model(preprocessed)                               # [B, 9, 37]

        T = output.shape[1]
        padded = (target_text + "_" * T)[:T]
        target_ids = torch.tensor(
            [self.ALPHABET.index(c) if c in self.ALPHABET else len(self.ALPHABET) - 1
             for c in padded],
            dtype=torch.long, device=output.device,
        )
        log_probs = torch.log(output.clamp(min=1e-8))   # [B, T, V]
        losses = [F.nll_loss(log_probs[i], target_ids) for i in range(output.shape[0])]
        return [l if impersonation else -l for l in losses]

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "CCTOCRBackend":
        if self._model is not None:
            self._model.eval()
        return self

    def train(self) -> "CCTOCRBackend":
        if self._model is not None:
            self._model.train()
        return self

    def to(self, device: str) -> "CCTOCRBackend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# ---------------------------------------------------------------------------
# DoctrViTSTR backend
# ---------------------------------------------------------------------------

class DoctrViTSTRBackend(OCRBackend):
    """
    Differentiable doctr vitstr_small backend.

    Loads fine-tuned weights from weights/vitstr_small_finetuned.pt by default.
    Gradients flow through the attention decoder into the adversarial patch via
    doctr's built-in cross-entropy loss (model called in train mode with target).

    Input crop: [C, H, W] float32 [0, 1] — resized internally to [3, 32, 128].
    """

    name          = "doctr-vitstr"
    is_trainable  = True
    ocr_crop_size = (32, 128)   # (H, W)

    DEFAULT_WEIGHTS = "weights/vitstr_small_finetuned.pt"

    def __init__(self, model_path: str = "none", device: str = "cpu"):
        # If no explicit path given, use the fine-tuned checkpoint
        if model_path == "none":
            model_path = self.DEFAULT_WEIGHTS
        super().__init__(model_path, device)
        self._model = None

    def load(self) -> None:
        try:
            from doctr.models import vitstr_small
        except ImportError:
            raise ImportError("[doctr-vitstr] pip install python-doctr[torch]")

        weights = Path(self.model_path)
        if weights.exists():
            self._model = vitstr_small(pretrained=False)
            self._model.load_state_dict(
                torch.load(str(weights), map_location=self.device)
            )
            print(f"[{self.name}] Loaded fine-tuned weights: {weights}")
        else:
            print(f"[{self.name}] WARNING: {weights} not found — using pretrained weights")
            self._model = vitstr_small(pretrained=True)

        self._model.to(self.device).eval()

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """[C, H, W] or [1, C, H, W] float32 [0,1] → [1, C, 32, 128] on device."""
        x = image.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return F.interpolate(x, size=(32, 128), mode="bilinear", align_corners=False)

    def predict(self, image: torch.Tensor) -> OCRResult:
        self.ensure_loaded()
        with torch.no_grad():
            inp = self._preprocess(image)
            out = self._model(inp, return_preds=True)

        text, conf = out["preds"][0]
        text = text.upper().replace("-", "") or None
        return OCRResult(logits=None, text=text, confidence=conf)

    def differentiable_loss(self, crop: torch.Tensor, target_text: str,
                             impersonation: bool = False) -> torch.Tensor:
        """
        Cross-entropy loss via doctr's built-in compute_loss.
        Gradients flow through the ViT into the patch.
        target_text is lowercased to match the doctr vocab.
        """
        self.ensure_loaded()
        inp = self._preprocess(crop)
        self._model.train()
        # doctr vocab is lowercase; ViTSTR was trained on lowercase text
        target = [target_text.lower()]
        out = self._model(inp, target=target)
        self._model.eval()
        loss = out["loss"]
        return loss if impersonation else -loss

    def differentiable_loss_batch(self, crops: list, target_text: str,
                                   impersonation: bool = False) -> list:
        """Batch preprocessing + sequential model calls (doctr targets per-sample)."""
        self.ensure_loaded()
        self._model.train()
        target = [target_text.lower()]
        losses = []
        for crop in crops:
            inp = self._preprocess(crop)
            out = self._model(inp, target=target)
            loss = out["loss"]
            losses.append(loss if impersonation else -loss)
        self._model.eval()
        return losses

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "DoctrViTSTRBackend":
        if self._model is not None:
            self._model.eval()
        return self

    def train(self) -> "DoctrViTSTRBackend":
        if self._model is not None:
            self._model.train()
        return self

    def to(self, device: str) -> "DoctrViTSTRBackend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRAINABLE_OCR_REGISTRY: dict[str, type[OCRBackend]] = {
    "crnn":         CRNNBackend,
    "trocr":        TrOCROCRBackend,
    "dtrb":         DeepTextRecognitionBenchmarkOCRBackend,
    "cct":          CCTOCRBackend,
    "doctr-vitstr": DoctrViTSTRBackend,
    "lprnet":       LPRNetBackend,
}

EVAL_ONLY_OCR_REGISTRY: dict[str, type[OCRBackend]] = {
    "fastanpr-ocr": FastANPROCRBackend,
    "mock-ocr":     MockOCRBackend,
}

OCR_REGISTRY: dict[str, type[OCRBackend]] = {
    **TRAINABLE_OCR_REGISTRY,
    **EVAL_ONLY_OCR_REGISTRY,
}

NON_DIFFERENTIABLE_OCR_BACKENDS = set(EVAL_ONLY_OCR_REGISTRY.keys())


def build_ocr_backend(name: str, model_path: str = "none",
                      device: str = "cpu", **kwargs) -> OCRBackend:
    """
    Factory — create an OCR backend by name.

    Example
    -------
    >>> ocr = build_ocr_backend("crnn", "weights/crnn.pth", device="cuda")
    >>> ocr.load()
    >>> result = ocr.predict(plate_tensor)   # OCRResult with .logits, .text
    """
    if name not in OCR_REGISTRY:
        raise ValueError(
            f"Unknown OCR backend {name!r}. Available: {list(OCR_REGISTRY)}"
        )
    return OCR_REGISTRY[name](model_path=model_path, device=device, **kwargs)
