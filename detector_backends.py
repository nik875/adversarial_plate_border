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


def _box_iou_scalar(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """IoU between two [1,4] xyxy boxes; returns a plain float."""
    x1 = torch.max(box_a[:, 0], box_b[:, 0])
    y1 = torch.max(box_a[:, 1], box_b[:, 1])
    x2 = torch.min(box_a[:, 2], box_b[:, 2])
    y2 = torch.min(box_a[:, 3], box_b[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    return (inter / (area_a + area_b - inter + 1e-6)).item()


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

    def differentiable_det_loss(self, image: torch.Tensor,
                                 target_box: torch.Tensor) -> torch.Tensor:
        """
        Return a scalar confidence score for the best-matching detection,
        with gradients flowing back to ``image``.

        The default falls back to ``predict()`` — gradients only flow if the
        backend's ``predict()`` already keeps the autograd graph (e.g. Yolov9).
        Override in backends where ``predict()`` runs under ``no_grad``.

        Parameters
        ----------
        image      : [C, H, W] float32 on any device
        target_box : [x1, y1, x2, y2] float32 — selects the best-IoU detection
        """
        dets = self.predict(image)
        if not dets:
            return torch.tensor(0.0, device=self.device)
        box_t = target_box.to(self.device).unsqueeze(0)
        best = max(dets, key=lambda d: (
            _box_iou_scalar(d.box.to(self.device).unsqueeze(0), box_t)
            * d.confidence
        ))
        return best.conf.to(self.device)

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
    Wraps a Hugging Face Transformers RT-DETR model.
    
    Uses the fine-tuned license plate detection model from:
    justjuu/rtdetr-v2-license-plate-detection
    """

    name = "rtdetr"

    def __init__(self, model_path: str, device: str = "cpu",
                 conf_threshold: float = 0.25, model_id: str = "justjuu/rtdetr-v2-license-plate-detection"):
        super().__init__(model_path, device)
        self.conf_threshold = conf_threshold
        self.model_id = model_id
        self._model: Optional[nn.Module] = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        
        # Use model_id if path is "none" or doesn't exist
        if str(self.model_path) in {"", "none"} or not self.model_path.exists():
            source = self.model_id
            print(f"[{self.name}] Loading from Hugging Face: {source}")
        else:
            source = str(self.model_path)
            print(f"[{self.name}] Loading from local path: {source}")
        
        self._processor = AutoImageProcessor.from_pretrained(source)
        self._model = AutoModelForObjectDetection.from_pretrained(source)
        self._model.to(self.device)
        self._model.eval()
        print(f"[{self.name}] Model loaded successfully")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        """
        Run RT-DETR detection on image.
        
        Parameters
        ----------
        image : torch.Tensor
            Shape [C, H, W], dtype float32, values in [0, 1]
        
        Returns
        -------
        List[Detection]
            Detected bounding boxes sorted by confidence
        """
        self.ensure_loaded()
        
        # Convert to PIL Image format expected by processor
        # image is [C, H, W] in [0, 1]
        img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        
        # Use processor to prepare inputs
        from PIL import Image
        pil_img = Image.fromarray(img_np)
        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get original image size for denormalization
        orig_h, orig_w = image.shape[1], image.shape[2]
        
        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        # Post-process outputs
        # RT-DETR returns logits [batch, num_queries, num_classes] and boxes [batch, num_queries, 4]
        # boxes are in cxcywh format normalized to [0, 1]
        target_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)
        results = self._processor.post_process_object_detection(
            outputs, 
            threshold=self.conf_threshold,
            target_sizes=target_sizes
        )[0]  # Get first (and only) image results
        
        # Convert to Detection objects
        detections: List[Detection] = []
        
        scores = results["scores"].cpu()
        labels = results["labels"].cpu()
        boxes = results["boxes"].cpu()  # Already in xyxy format from post_process
        
        for i in range(len(scores)):
            score = scores[i].item()
            if score < self.conf_threshold:
                continue
            
            x1, y1, x2, y2 = boxes[i].tolist()
            class_id = labels[i].item()
            
            # Build synthetic raw tensor for gradient compatibility
            synthetic = torch.tensor(
                [0.0, x1, y1, x2, y2, class_id, score],
                dtype=torch.float32,
            )
            
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=score,
                class_id=int(class_id),
                raw=synthetic,
            ))
        
        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

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
# FastANPR backend  (detection-only wrapper around fastanpr.FastANPR)
# ---------------------------------------------------------------------------

class FastANPRBackend(DetectorBackend):
    """
    Wraps the ``fastanpr.FastANPR`` pipeline as a DetectorBackend.

    FastANPR bundles its own internal YOLOv8 detector, so no ``model_path``
    is needed — pass ``"none"`` on the CLI.  The library is async-native;
    this wrapper calls ``asyncio.run()`` to keep the synchronous predict()
    contract.

    Important limitations
    ---------------------
    * **Not differentiable.**  FastANPR runs through numpy and PaddleOCR
      with no PyTorch gradient graph, so ``Detection.raw`` carries only
      plain float tensors.  Use this backend for *evaluation* only;
      adversarial *training* requires a backend whose forward pass stays
      inside the PyTorch autograd graph.
    * Coordinates returned by ``plate.det_box`` are in the original image
      pixel space (no letterboxing correction needed).
    """

    name = "fastanpr"

    def __init__(self, model_path: str = "none", device: str = "cpu",
                 conf_threshold: float = 0.25):
        super().__init__(model_path, device)
        self.conf_threshold = conf_threshold
        self._anpr = None

    def load(self) -> None:
        import asyncio
        from fastanpr import FastANPR  # deferred import

        self._anpr = FastANPR()
        print(f"[{self.name}] FastANPR initialised (built-in YOLOv8 + PaddleOCR)")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        """
        Run FastANPR on a single CHW float32 tensor.

        The tensor is converted to a HWC uint8 numpy array (RGB) before
        being passed to FastANPR, which expects that format.
        """
        import asyncio
        self.ensure_loaded()

        # CHW float32 [0,1]  →  HWC uint8
        img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")

        # FastANPR is async; run it synchronously
        results = asyncio.run(self._anpr.run([img_np]))

        plates = results[0] if results else []
        detections: List[Detection] = []

        for plate in plates:
            conf = float(plate.det_conf)
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = (float(plate.det_box[0]), float(plate.det_box[1]),
                               float(plate.det_box[2]), float(plate.det_box[3]))

            # Build a plain (non-grad) synthetic raw tensor so downstream
            # code that reads det.raw[1:5] / det.raw[6] still works.
            synthetic = torch.tensor(
                [0.0, x1, y1, x2, y2, 0.0, conf], dtype=torch.float32
            )

            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=0,
                raw=synthetic,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def parameters(self) -> Iterator[nn.Parameter]:
        return iter([])  # no exposed PyTorch parameters

    def to(self, device: str) -> "FastANPRBackend":
        # FastANPR manages its own device internally; we track it for
        # consistency but can't force it from here.
        self.device = device
        return self


# ---------------------------------------------------------------------------
# open-image-models backend  (ONNX YOLOv9 — pip install open-image-models[onnx-cpu])
# ---------------------------------------------------------------------------

class OpenImageModelsBackend(DetectorBackend):
    """
    Wraps ``open-image-models`` LicensePlateDetector.

    Uses pre-trained YOLOv9 ONNX models — no custom weights file required,
    models are downloaded automatically on first use.

    Install
    -------
    CPU:  pip install open-image-models[onnx-cpu]
    GPU:  pip install open-image-models[onnx-gpu]

    Available detector_model strings (pass via model_path):
        yolo-v9-t-256-license-plate-end2end  (default, fastest)
        yolo-v9-t-384-license-plate-end2end  (balanced)
        yolo-v9-s-384-license-plate-end2end  (more accurate)
        yolo-v9-c-384-license-plate-end2end  (most accurate)

    Pass the model string as model_path:
        OpenImageModelsBackend("yolo-v9-t-384-license-plate-end2end")
    """

    name = "open-image-models"

    def __init__(self, model_path: str = "yolo-v9-t-384-license-plate-end2end",
                 device: str = "cpu", conf_threshold: float = 0.25):
        super().__init__(model_path, device)
        self.conf_threshold = conf_threshold
        self.detector_model_name = str(model_path)  # model_path holds the model string
        self._detector = None

    def load(self) -> None:
        from open_image_models import LicensePlateDetector  # deferred import

        self._detector = LicensePlateDetector(
            detection_model=self.detector_model_name,
            conf_thresh=self.conf_threshold,
        )
        print(f"[{self.name}] Loaded model '{self.detector_model_name}'")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        self.ensure_loaded()
        # CHW float32 [0,1] → HWC uint8 BGR  (open-image-models uses OpenCV convention)
        img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        # open-image-models expects BGR
        import cv2
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        raw_dets = self._detector.predict(img_bgr)  # list of detection objects

        detections: List[Detection] = []
        for det in (raw_dets or []):
            # open-image-models returns objects with .bounding_box [x1,y1,x2,y2]
            # and .confidence — handle both list and array formats
            try:
                bb   = det.bounding_box
                conf = float(det.confidence)
            except AttributeError:
                # Fallback: some versions return namedtuples or plain arrays
                bb   = det[:4]
                conf = float(det[4])

            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            synthetic = torch.tensor([0.0, x1, y1, x2, y2, 0.0, conf],
                                     dtype=torch.float32)
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf, class_id=0, raw=synthetic,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def parameters(self) -> Iterator[nn.Parameter]:
        return iter([])


# ---------------------------------------------------------------------------
# Faster R-CNN backend  (torchvision / fine-tuned .pt)
# ---------------------------------------------------------------------------

class FasterRCNNBackend(DetectorBackend):
    """
    Wraps a fine-tuned Faster R-CNN checkpoint saved as ``.pt``.

    Supports either:
    - a serialized nn.Module, or
    - a checkpoint dict containing ``model_state_dict`` / ``state_dict``.
    """

    name = "fasterrcnn"

    def __init__(self, model_path: str, device: str = "cpu",
                 conf_threshold: float = 0.25):
        super().__init__(model_path, device)
        self.conf_threshold = conf_threshold
        self._model: Optional[nn.Module] = None

    def _build_default_model(self, num_classes: int = 2) -> nn.Module:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        return fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Faster R-CNN weights not found: {self.model_path}")

        checkpoint = torch.load(str(self.model_path), map_location="cpu")

        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict):
            if isinstance(checkpoint.get("model"), nn.Module):
                model = checkpoint["model"]
            else:
                state = checkpoint.get("model_state_dict")
                if state is None:
                    state = checkpoint.get("state_dict", checkpoint)
                num_classes = int(checkpoint.get("num_classes", 2))
                model = self._build_default_model(num_classes=num_classes)
                model.load_state_dict(state, strict=False)
        else:
            raise RuntimeError(
                f"Unsupported Faster R-CNN checkpoint format at {self.model_path}. "
                "Expected nn.Module or checkpoint dict."
            )

        self._model = model.to(self.device)
        self._model.eval()
        print(f"[{self.name}] Loaded from {self.model_path}")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        self.ensure_loaded()
        inp = image.to(self.device)
        if inp.dim() == 4:
            inp = inp.squeeze(0)   # GeneralizedRCNN expects List[Tensor], not batched

        with torch.no_grad():
            outputs = self._model([inp])

        if isinstance(outputs, dict):
            outputs = [outputs]
        if not outputs:
            return []

        out = outputs[0]
        boxes = out.get("boxes")
        scores = out.get("scores")
        labels = out.get("labels")
        if boxes is None or scores is None or labels is None:
            return []

        detections: List[Detection] = []
        for i in range(len(scores)):
            conf = float(scores[i].item())
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
            class_id = int(labels[i].item())
            synthetic = torch.tensor([0.0, x1, y1, x2, y2, float(class_id), conf],
                                     dtype=torch.float32)
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf, class_id=class_id, raw=synthetic,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def differentiable_det_loss(self, image: torch.Tensor,
                                 target_box: torch.Tensor) -> torch.Tensor:
        """Run forward without no_grad so the returned score has a gradient."""
        self.ensure_loaded()
        inp = image.to(self.device)
        if inp.dim() == 4:
            inp = inp.squeeze(0)

        # eval-mode forward without no_grad — scores stay in the autograd graph
        outputs = self._model([inp])
        if isinstance(outputs, dict):
            outputs = [outputs]
        if not outputs:
            return torch.tensor(0.0, device=self.device)

        out = outputs[0]
        scores = out.get("scores")
        boxes  = out.get("boxes")

        if scores is None or len(scores) == 0:
            return torch.tensor(0.0, device=self.device)

        # Select best box by IoU × conf (non-differentiable selection only)
        with torch.no_grad():
            box_t   = target_box.unsqueeze(0).to(self.device)
            weights = torch.tensor([
                _box_iou_scalar(boxes[i].unsqueeze(0), box_t) * scores[i].item()
                for i in range(len(scores))
            ])
            best_idx = int(weights.argmax().item())

        return scores[best_idx]

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "FasterRCNNBackend":
        if self._model is not None:
            self._model.eval()
        return self

    def train_mode(self) -> "FasterRCNNBackend":
        if self._model is not None:
            self._model.train()
        return self

    def to(self, device: str) -> "FasterRCNNBackend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# ---------------------------------------------------------------------------
# YOLO-NAS backend  (Deci AI super-gradients)
# pip install super-gradients
# ---------------------------------------------------------------------------

class YOLONASBackend(DetectorBackend):
    """
    Wraps Deci AI's YOLO-NAS via the ``super-gradients`` library.

    YOLO-NAS models come in three sizes: yolo_nas_s, yolo_nas_m, yolo_nas_l.
    Pass the size string as model_path (no .pt extension).

    Install
    -------
    pip install super-gradients

    The first call to load() downloads COCO pre-trained weights automatically.
    For license-plate fine-tuned weights, pass a local checkpoint path instead
    and set ``pretrained_weights=None`` in the constructor.

    model_path examples:
        "yolo_nas_s"   – small  (fastest)
        "yolo_nas_m"   – medium
        "yolo_nas_l"   – large  (most accurate)
        "path/to/custom_ckpt.pth"  – custom fine-tuned checkpoint

    Note: YOLO-NAS uses its own non-PyTorch-differentiable post-processing,
    so this backend is evaluation-only.
    """

    name = "yolo-nas"

    _KNOWN_ARCHS = {"yolo_nas_s", "yolo_nas_m", "yolo_nas_l"}

    def __init__(self, model_path: str = "yolo_nas_s", device: str = "cpu",
                 conf_threshold: float = 0.25,
                 num_classes: int = 1,
                 pretrained_weights: Optional[str] = "coco"):
        super().__init__(model_path, device)
        self.conf_threshold  = conf_threshold
        self.num_classes     = num_classes
        self.pretrained_weights = pretrained_weights
        self._model          = None
        self._arch           = str(model_path)

    def load(self) -> None:
        from super_gradients.training import models as sg_models  # deferred import

        is_known_arch = self._arch in self._KNOWN_ARCHS

        if is_known_arch:
            # Load from super-gradients model zoo (downloads weights automatically)
            self._model = sg_models.get(
                self._arch,
                num_classes=self.num_classes,
                pretrained_weights=self.pretrained_weights,
            )
        else:
            # Custom checkpoint path: load architecture then restore weights
            import torch as _torch
            arch = "yolo_nas_s"  # default arch for custom ckpt; override if needed
            self._model = sg_models.get(arch, num_classes=self.num_classes)
            ckpt = _torch.load(str(self.model_path), map_location="cpu")
            state = ckpt.get("net", ckpt)
            self._model.load_state_dict(state, strict=False)

        self._model = self._model.to(self.device)
        self._model.eval()
        print(f"[{self.name}] Loaded '{self._arch}' "
              f"({'pretrained: ' + self.pretrained_weights if self.pretrained_weights else 'custom ckpt'})")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        self.ensure_loaded()
        import cv2
        img_np  = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # super-gradients predict returns an ImageDetectionPrediction
        result = self._model.predict(img_bgr, conf=self.conf_threshold)
        pred   = result.prediction                   # DetectionPrediction
        bboxes = pred.bboxes_xyxy                   # numpy [N, 4]
        confs  = pred.confidence                    # numpy [N]
        labels = pred.labels.astype(int)            # numpy [N]

        detections: List[Detection] = []
        for i in range(len(bboxes)):
            conf = float(confs[i])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = (float(bboxes[i, 0]), float(bboxes[i, 1]),
                               float(bboxes[i, 2]), float(bboxes[i, 3]))
            synthetic = torch.tensor([0.0, x1, y1, x2, y2, float(labels[i]), conf],
                                     dtype=torch.float32)
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf, class_id=int(labels[i]), raw=synthetic,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "YOLONASBackend":
        if self._model is not None:
            self._model.eval()
        return self

    def to(self, device: str) -> "YOLONASBackend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self




# ---------------------------------------------------------------------------
# YOLOv11 backend  (ultralytics — fine-tuned LP weights available on HuggingFace)
# pip install ultralytics huggingface_hub
# ---------------------------------------------------------------------------

class YOLOv11Backend(DetectorBackend):
    """
    Wraps Ultralytics YOLOv11.

    Pre-trained LP weights are available directly from HuggingFace:
      morsetechlab/yolov11-license-plate-detection
    Sizes: n, s, m, l, x — pass the filename as model_path.

    Downloading weights
    -------------------
    Option A — HuggingFace Hub (automatic):
        backend = YOLOv11Backend("yolov11s-license-plate.pt", download_hf=True)

    Option B — manual wget then pass local path:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("morsetechlab/yolov11-license-plate-detection",
                               "yolov11s-license-plate.pt")
        backend = YOLOv11Backend(path)

    Install: pip install ultralytics huggingface_hub
    """

    name = "yolov11"
    _HF_REPO = "morsetechlab/yolov11-license-plate-detection"
    _VALID_SIZES = {"n", "s", "m", "l", "x"}

    def __init__(self, model_path: str, device: str = "cpu",
                 conf_threshold: float = 0.25, download_hf: bool = True):
        super().__init__(model_path, device)
        self.conf_threshold = conf_threshold
        self.download_hf    = download_hf
        self._yolo          = None
        self._model: Optional[nn.Module] = None

    def load(self) -> None:
        from ultralytics import YOLO

        local_path = str(self.model_path)

        # Auto-download from HuggingFace if requested or if file doesn't exist
        if self.download_hf or not self.model_path.exists():
            from huggingface_hub import hf_hub_download
            filename = self.model_path.name  # e.g. "yolov11s-license-plate.pt"
            print(f"[{self.name}] Downloading '{filename}' from HF Hub "
                  f"({self._HF_REPO})...")
            local_path = hf_hub_download(self._HF_REPO, filename)
            print(f"[{self.name}] Saved to {local_path}")
        elif not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLOv11 weights not found: {self.model_path}\n"
                f"Pass download_hf=True or download manually from:\n"
                f"  https://huggingface.co/{self._HF_REPO}"
            )

        self._yolo  = YOLO(local_path)
        self._model = self._yolo.model
        self._model.to(self.device)
        self._model.eval()
        print(f"[{self.name}] Loaded from {local_path}")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        self.ensure_loaded()
        img_np  = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        results = self._yolo.predict(img_np, conf=self.conf_threshold, verbose=False)
        return _results_to_detections(results, self.conf_threshold, image)

    def raw_forward(self, batch: torch.Tensor):
        self.ensure_loaded()
        return self._model(batch)

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "YOLOv11Backend":
        if self._model is not None:
            self._model.eval()
        return self

    def train_mode(self) -> "YOLOv11Backend":
        if self._model is not None:
            self._model.train()
        return self

    def to(self, device: str) -> "YOLOv11Backend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# ---------------------------------------------------------------------------
# YOLOv9 384 onnx2torch backend  — differentiable via onnx2torch
# ---------------------------------------------------------------------------

class Yolov9Onnx2TorchBackend(DetectorBackend):
    """
    Wraps the YOLOv9-t-384 licence-plate ONNX model via onnx2torch.

    The ONNX model is downloaded automatically on first use by instantiating
    LicensePlateDetector from open-image-models.  After that, onnx2torch
    converts it to a native PyTorch nn.Module whose forward pass is fully
    differentiable — gradients flow from the detection outputs back through
    the input image into the adversarial patch.

    Model path
    ----------
    ~/.cache/open-image-models/yolo-v9-t-384-license-plate-end2end/
        yolo-v9-t-384-license-plates-end2end.onnx

    Input convention
    ----------------
    CHW float32 [0, 1], letterboxed to 384×384, BGR channel order.

    Output convention (after onnx2torch)
    -------------------------------------
    The model returns a list of 7-element tensors, one per detection:
        [batch_idx, x1, y1, x2, y2, class_id, confidence]
    NMS is embedded ("end2end"), so no post-processing is needed.
    """

    name = "yolo-v9-384"

    ONNX_PATH = ("~/.cache/open-image-models/"
                 "yolo-v9-t-384-license-plate-end2end/"
                 "yolo-v9-t-384-license-plates-end2end.onnx")

    def __init__(self, model_path: str = "none", device: str = "cpu",
                 conf_threshold: float = 0.25):
        super().__init__(model_path, device)
        self.conf_threshold = conf_threshold
        self._model: Optional[nn.Module] = None

    def load(self) -> None:
        import onnx
        import onnx2torch

        onnx_path = Path(self.ONNX_PATH).expanduser()

        if not onnx_path.exists():
            # Trigger download via open-image-models cache
            print(f"[{self.name}] ONNX model not found — downloading via open-image-models…")
            from open_image_models import LicensePlateDetector
            LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"[{self.name}] ONNX model still not found after download attempt: {onnx_path}"
            )

        onnx_model = onnx.load(str(onnx_path))
        self._model = onnx2torch.convert(onnx_model)
        self._model.to(self.device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        print(f"[{self.name}] Loaded from {onnx_path}")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        """
        Run detection.  Gradients flow through the output tensors because the
        model parameters are frozen but the computation graph is intact.
        """
        self.ensure_loaded()
        # image: CHW float32 [0,1] at 384×384
        batch = image.unsqueeze(0).to(self.device)   # [1, 3, 384, 384]
        # Do NOT wrap in no_grad — we need the autograd graph for adversarial training.
        output = self._model(batch)

        detections: List[Detection] = []
        for det in output:
            if det.numel() < 7:
                continue
            conf = det[6].item()
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = det[1].item(), det[2].item(), det[3].item(), det[4].item()
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=int(det[5].item()),
                raw=det,   # keep the live tensor so gradients can flow
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "Yolov9Onnx2TorchBackend":
        if self._model is not None:
            self._model.eval()
        return self

    def to(self, device: str) -> "Yolov9Onnx2TorchBackend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# Backends where gradients flow through the detector — usable for adversarial training.
TRAINABLE_REGISTRY: dict[str, type[DetectorBackend]] = {
    "yolov8":      YOLOv8Backend,            # pip install ultralytics  |  weights: your fine-tuned .pt
    "fasterrcnn":  FasterRCNNBackend,        # pip install torchvision  |  weights: weights/model.pt
    "yolov11":     YOLOv11Backend,           # pip install ultralytics  |  weights: morsetechlab/yolov11-license-plate-detection (HF)
    "rtdetr":      RTDETRBackend,            # pip install transformers  |  weights: justjuu/rtdetr-v2-license-plate-detection (HF)
    "yolo-v9-384": Yolov9Onnx2TorchBackend, # pip install onnx onnx2torch open-image-models  |  auto-downloaded
}

# Backends that run through ONNX / external C++ / numpy — no autograd.
# Useful for evaluation / baseline comparison only.
EVAL_ONLY_REGISTRY: dict[str, type[DetectorBackend]] = {
    "open-image-models": OpenImageModelsBackend,  # pip install open-image-models[onnx-cpu]
    "fastanpr":          FastANPRBackend,          # pip install fastanpr
    "yolo-nas":          YOLONASBackend,           # pip install super-gradients
    "yolov5":            YOLOv5Backend,            # torch.hub (numpy pipeline)
    "mock":              MockBackend,
}

REGISTRY: dict[str, type[DetectorBackend]] = {
    **TRAINABLE_REGISTRY,
    **EVAL_ONLY_REGISTRY,
}

NON_DIFFERENTIABLE_BACKENDS = set(EVAL_ONLY_REGISTRY.keys())


def build_backend(name: str, model_path: str, device: str = "cpu",
                  **kwargs) -> DetectorBackend:
    """
    Factory function – create a backend by name.

    Parameters
    ----------
    name : str
        Key into REGISTRY (e.g. ``"yolov8"``, ``"open-image-models"``).
    model_path : str
        Path to model weights file, or a model-name string for backends that
        don't need a local file (open-image-models, yolo-nas, fastanpr → "none").
    device : str
        Torch device string.
    **kwargs
        Forwarded to the backend constructor.

    Example
    -------
    >>> backend = build_backend("yolov8", "weights/lp.pt", device="cuda")
    >>> backend.load()
    >>> backend = build_backend("open-image-models", "yolo-v9-t-384-license-plate-end2end")
    >>> backend.load()
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {list(REGISTRY)}"
        )
    return REGISTRY[name](model_path=model_path, device=device, **kwargs)
