"""
detector_backends.py

Pluggable detector backend abstraction for license plate detection.
Each backend wraps a model and exposes a uniform interface so the trainer
and evaluator never need to know which underlying model is running.

Usage
-----
    backend = YOLOv8Backend("license_plate_detector.pt", device="cuda")
    backend.load()
    detections = backend.predict(image_tensor)   # [N, 7]  xyxy conf cls extra

Adding a new model
------------------
1. Subclass DetectorBackend.
2. Implement load(), predict(), and parameters().
3. Register it in REGISTRY at the bottom of this file.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Detection result container
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Normalised detection result – coordinates in the *input* image space."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    # raw 7-element tensor kept for gradient-flow in adversarial training
    raw: Optional[torch.Tensor] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def box(self) -> torch.Tensor:
        """Return [x1, y1, x2, y2] as a float32 tensor (from raw if available)."""
        if self.raw is not None:
            return self.raw[1:5]
        return torch.tensor([self.x1, self.y1, self.x2, self.y2], dtype=torch.float32)

    @property
    def conf(self) -> torch.Tensor:
        """Return confidence as a scalar tensor (grad-compatible when raw is set)."""
        if self.raw is not None:
            return self.raw[6]
        return torch.tensor(self.confidence)

    def to_dict(self) -> dict:
        return dict(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2,
                    confidence=self.confidence, class_id=self.class_id)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DetectorBackend(abc.ABC):
    """
    Interface every detector backend must satisfy.

    Subclasses *must* implement:
        load()       – initialise / restore the model
        predict()    – run inference on a single CHW float32 image tensor
        parameters() – yield nn.Parameters so the trainer can freeze them

    Subclasses *may* override:
        eval() / train()  – set the model's train/eval mode
        to()              – move model to a device
    """

    #: Human-readable name used in evaluation reports
    name: str = "base"

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self._loaded = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights.  Called once before the first predict()."""

    @abc.abstractmethod
    def predict(self, image: torch.Tensor) -> List[Detection]:
        """
        Run detection on a single image.

        Parameters
        ----------
        image : torch.Tensor
            Shape ``[C, H, W]``, dtype float32, values in ``[0, 1]``.
            The tensor is on *any* device; move it inside this method if needed.

        Returns
        -------
        List[Detection]
            One entry per detected bounding box, sorted by confidence (desc).
        """

    @abc.abstractmethod
    def parameters(self) -> Iterator[nn.Parameter]:
        """Yield all learnable parameters (so the trainer can freeze them)."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def eval(self) -> "DetectorBackend":
        return self

    def train_mode(self) -> "DetectorBackend":
        return self

    def to(self, device: str) -> "DetectorBackend":
        self.device = device
        return self

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    def freeze(self) -> None:
        """Disable gradients for all parameters (call after load())."""
        for p in self.parameters():
            p.requires_grad_(False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, path={self.model_path})"


# ---------------------------------------------------------------------------
# YOLOv8 backend  (ultralytics)
# ---------------------------------------------------------------------------

def _results_to_detections(results, conf_threshold: float,
                            source_image: torch.Tensor) -> List[Detection]:
    """
    Convert an ultralytics ``Results`` object to a list of ``Detection``.

    Works for any architecture (YOLOv8, RT-DETR, …) because all of them
    surface detections through the same ``Results.boxes`` interface after
    the high-level ``.predict()`` call.

    A synthetic 7-element raw tensor is built so downstream gradient-using
    code (adversarial loss) still has something to differentiate through.
    The tensor layout matches the original trainer convention::

        [batch_idx=0, x1, y1, x2, y2, class_id, conf]
    """
    detections: List[Detection] = []
    if results is None or len(results) == 0:
        return detections

    boxes = results[0].boxes          # ultralytics Boxes object
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy   = boxes.xyxy               # [N, 4]  absolute pixel coords
    confs  = boxes.conf               # [N]
    cls_ids = boxes.cls               # [N]

    for i in range(len(xyxy)):
        conf = confs[i].item()
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = (xyxy[i, 0].item(), xyxy[i, 1].item(),
                           xyxy[i, 2].item(), xyxy[i, 3].item())

        # Build a synthetic raw tensor that carries gradients when the
        # source_image is part of the computation graph.
        # For adversarial training we re-run the *internal* model in
        # trainer.partial_loss; here we just need a plain float tensor.
        synthetic = torch.tensor(
            [0.0, x1, y1, x2, y2, cls_ids[i].item(), conf],
            dtype=torch.float32,
        )

        detections.append(Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=conf,
            class_id=int(cls_ids[i].item()),
            raw=synthetic,
        ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


class YOLOv8Backend(DetectorBackend):
    """
    Wraps an Ultralytics YOLOv8 model.

    Uses the high-level ``.predict()`` API so output parsing is identical
    across all Ultralytics architectures (YOLOv8, RT-DETR, …).

    For gradient-aware adversarial training the trainer calls
    ``self._yolo.model(batch)`` directly via ``raw_forward()``; that
    internal path still returns the flat 7-column tensor YOLOv8 produces.
    """

    name = "yolov8"

    def __init__(self, model_path: str, device: str = "cpu",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        super().__init__(model_path, device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._yolo = None             # ultralytics YOLO wrapper
        self._model: Optional[nn.Module] = None   # raw nn.Module for grad training

    def load(self) -> None:
        from ultralytics import YOLO

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLOv8 weights not found: {self.model_path}")

        self._yolo  = YOLO(str(self.model_path))
        self._model = self._yolo.model
        self._model.to(self.device)
        self._model.eval()
        print(f"[{self.name}] Loaded from {self.model_path}")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        """Run inference via the high-level ultralytics API."""
        self.ensure_loaded()
        # Convert CHW float32 [0,1] tensor → HWC uint8 numpy for ultralytics
        img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        results = self._yolo.predict(img_np, conf=self.conf_threshold,
                                     iou=self.iou_threshold, verbose=False)
        return _results_to_detections(results, self.conf_threshold, image)

    def raw_forward(self, batch: torch.Tensor):
        """
        Direct nn.Module forward for use inside adversarial training loops
        where gradients must flow through the detection scores.
        Returns the raw model output (architecture-dependent format).
        """
        self.ensure_loaded()
        return self._model(batch)

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "YOLOv8Backend":
        if self._model is not None:
            self._model.eval()
        return self

    def train_mode(self) -> "YOLOv8Backend":
        if self._model is not None:
            self._model.train()
        return self

    def to(self, device: str) -> "YOLOv8Backend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# ---------------------------------------------------------------------------
# YOLOv5 backend  (torch.hub)
# ---------------------------------------------------------------------------

class YOLOv5Backend(DetectorBackend):
    """
    Loads a YOLOv5 model via ``torch.hub`` or from a local ``.pt`` file.

    Output normalisation mirrors the YOLOv8 format so the trainer sees the
    same ``[batch_idx, x1, y1, x2, y2, class_id, conf]`` rows.
    """

    name = "yolov5"

    def __init__(self, model_path: str, device: str = "cpu",
                 conf_threshold: float = 0.25, repo: str = "ultralytics/yolov5"):
        super().__init__(model_path, device)
        self.conf_threshold = conf_threshold
        self.repo = repo
        self._model = None

    def load(self) -> None:
        import torch

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLOv5 weights not found: {self.model_path}")

        # torch.hub loads the full model; custom=True allows local .pt files
        self._model = torch.hub.load(
            self.repo, "custom",
            path=str(self.model_path),
            force_reload=False,
            verbose=False,
        )
        self._model.conf = self.conf_threshold
        self._model.to(self.device)
        self._model.eval()
        print(f"[{self.name}] Loaded from {self.model_path}")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        self.ensure_loaded()
        # YOLOv5's hub model accepts a tensor or numpy array
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        results = self._model(img_np)
        raw_boxes = results.xyxy[0]  # [N, 6] → x1 y1 x2 y2 conf cls

        detections: List[Detection] = []
        for row in raw_boxes:
            conf = row[4].item()
            # Synthesise a 7-element tensor matching the YOLOv8 convention
            # [batch_idx=0, x1, y1, x2, y2, class_id, conf]
            synthetic = torch.zeros(7, device=row.device)
            synthetic[1:5] = row[:4]
            synthetic[5] = row[5]
            synthetic[6] = conf
            detections.append(Detection(
                x1=row[0].item(), y1=row[1].item(),
                x2=row[2].item(), y2=row[3].item(),
                confidence=conf,
                class_id=int(row[5].item()),
                raw=synthetic,
            ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def to(self, device: str) -> "YOLOv5Backend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# ---------------------------------------------------------------------------
# RT-DETR backend  (ultralytics)
# ---------------------------------------------------------------------------

class RTDETRBackend(DetectorBackend):
    """
    Wraps an Ultralytics RT-DETR model.

    Output normalisation is identical to YOLOv8Backend because RT-DETR is
    accessed through the same Ultralytics API.
    """

    name = "rtdetr"

    def __init__(self, model_path: str, device: str = "cpu",
                 conf_threshold: float = 0.25):
        super().__init__(model_path, device)
        self.conf_threshold = conf_threshold
        self._rtdetr = None
        self._model: Optional[nn.Module] = None

    def load(self) -> None:
        from ultralytics import RTDETR

        if not self.model_path.exists():
            raise FileNotFoundError(f"RT-DETR weights not found: {self.model_path}")

        self._rtdetr = RTDETR(str(self.model_path))
        self._model  = self._rtdetr.model
        self._model.to(self.device)
        self._model.eval()
        print(f"[{self.name}] Loaded from {self.model_path}")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        """
        Use the high-level ultralytics API.

        Calling ``self._model(batch)`` directly returns a raw
        ``(tensor, dict)`` tuple whose layout differs from YOLOv8's flat
        rows — hence the IndexError you'd see when trying ``row[6]``.
        The high-level ``.predict()`` normalises everything into Results.
        """
        self.ensure_loaded()
        img_np  = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        results = self._rtdetr.predict(img_np, conf=self.conf_threshold, verbose=False)
        return _results_to_detections(results, self.conf_threshold, image)

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "RTDETRBackend":
        if self._model is not None:
            self._model.eval()
        return self

    def to(self, device: str) -> "RTDETRBackend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# ---------------------------------------------------------------------------
# Mock / stub backend  (useful for unit-testing without GPU/weights)
# ---------------------------------------------------------------------------

class MockBackend(DetectorBackend):
    """
    Returns a fixed set of detections.  Useful for testing the trainer
    and evaluator pipelines without real model weights.
    """

    name = "mock"

    def __init__(self, fixed_detections: Optional[List[Detection]] = None,
                 device: str = "cpu"):
        super().__init__("mock", device)
        self._fixed = fixed_detections or [
            Detection(x1=50, y1=50, x2=200, y2=100,
                      confidence=0.85, class_id=0)
        ]

    def load(self) -> None:
        print(f"[{self.name}] Mock backend loaded (no weights required)")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        self.ensure_loaded()
        return list(self._fixed)  # return a copy

    def parameters(self) -> Iterator[nn.Parameter]:
        return iter([])  # no parameters


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type[DetectorBackend]] = {
    "yolov8":  YOLOv8Backend,
    "yolov5":  YOLOv5Backend,
    "rtdetr":  RTDETRBackend,
    "mock":    MockBackend,
}


def build_backend(name: str, model_path: str, device: str = "cpu",
                  **kwargs) -> DetectorBackend:
    """
    Factory function – create a backend by name.

    Parameters
    ----------
    name : str
        Key into REGISTRY (e.g. ``"yolov8"``, ``"yolov5"``).
    model_path : str
        Path to model weights file.
    device : str
        Torch device string.
    **kwargs
        Forwarded to the backend constructor.

    Example
    -------
    >>> backend = build_backend("yolov8", "weights/lp.pt", device="cuda")
    >>> backend.load()
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {list(REGISTRY)}"
        )
    return REGISTRY[name](model_path=model_path, device=device, **kwargs)