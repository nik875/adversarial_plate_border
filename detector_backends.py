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
from typing import Iterator, List, Optional, Tuple

import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_to_stride(img_np: "np.ndarray", stride: int = 32) -> "np.ndarray":
    """Pad HWC uint8 image so H and W are multiples of stride (right/bottom pad)."""
    h, w = img_np.shape[:2]
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    if pad_h == 0 and pad_w == 0:
        return img_np
    return np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)


def _nms(boxes: torch.Tensor, scores: torch.Tensor,
         iou_threshold: float) -> torch.Tensor:
    """
    Pure-PyTorch greedy NMS.  Returns indices of kept boxes (sorted by score).

    boxes  : [N, 4]  xyxy, detached float32
    scores : [N]     detached float32
    """
    order = scores.argsort(descending=True)
    keep  = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou  = _box_iou_vectorized(boxes[i], boxes[rest])
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


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


def _box_iou_vectorized(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """IoU between a single [4] xyxy box and N [N, 4] xyxy boxes; returns [N]."""
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_box  = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_box + area_boxes - inter + 1e-6)


def _select_best_from_scores_boxes(
    scores: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    target_box: torch.Tensor,
    conf_threshold: float,
    device: str,
    mode: str = "suppress",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Select (conf_loss, pred_box) for target_box from pre-computed scores/boxes.

    mode='suppress': caller minimizes the returned score (real plate, suppress flow).
      No IoU overlap → already succeeded; return constant 0.0 with no gradient.
    mode='attract':  caller negates and minimizes the returned score (top-region flow).
      No IoU overlap → return proximity-weighted sum of scores so gradient pulls
      the spatially nearest anchors toward the target (weights fixed, scores carry grad).
      The weights are computed under no_grad and used outside it so gradient flows.
    """
    tb = target_box.to(device)
    proximity_weights = None
    best_idx = None
    with torch.no_grad():
        ious = _box_iou_vectorized(tb, boxes_xyxy.detach())
        if ious.max().item() < 1e-6:
            if mode == "suppress":
                return torch.tensor(0.0, device=device), None
            # attract — no overlap yet: compute fixed proximity weights.
            # scores * proximity_weights is evaluated outside no_grad below.
            box_centers  = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2
            target_center = torch.stack([(tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2])
            dists        = ((box_centers - target_center) ** 2).sum(-1).sqrt()
            target_size  = max((tb[2] - tb[0]).item(), (tb[3] - tb[1]).item(), 1.0)
            proximity_weights = torch.softmax(-dists / target_size, dim=0)
        else:
            best_idx = int((ious * scores.detach()).argmax().item())

    # Outside no_grad: gradient flows through scores.
    if proximity_weights is not None:
        return (scores * proximity_weights).sum(), None
    if scores[best_idx].detach().item() < conf_threshold:
        return scores[best_idx], None
    return scores[best_idx], boxes_xyxy[best_idx]


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

    def differentiable_det_loss_batch(
        self,
        images: torch.Tensor,           # [B, C, H, W]
        target_boxes: list,             # B × [x1, y1, x2, y2]
    ) -> list:
        """Default: sequential loop. Override for true batch GPU inference."""
        return [
            self.differentiable_det_loss(images[i], target_boxes[i])
            for i in range(images.shape[0])
        ]

    def differentiable_predict_box(
        self,
        image: torch.Tensor,
        target_box: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Like differentiable_det_loss but also returns the best predicted box.

        Returns (conf_loss, pred_box) where pred_box is [x1, y1, x2, y2] in
        image pixel coords, or (conf_loss, None) if no plate is detected above
        conf_threshold.  Gradients flow through conf_loss; box coords carry
        gradients only in backends that override this method.
        """
        dets = self.predict(image)
        if not dets:
            return torch.tensor(0.0, device=self.device), None
        box_t = target_box.to(self.device).unsqueeze(0)
        best = max(dets, key=lambda d: (
            _box_iou_scalar(d.box.to(self.device).unsqueeze(0), box_t)
            * d.confidence
        ))
        if _box_iou_scalar(best.box.to(self.device).unsqueeze(0), box_t) < 1e-6:
            return torch.tensor(0.0, device=self.device), None
        thresh = getattr(self, "conf_threshold", 0.25)
        if best.confidence < thresh:
            return best.conf.to(self.device), None
        return best.conf.to(self.device), best.box.to(self.device)

    def differentiable_predict_box_batch(
        self,
        images: torch.Tensor,
        target_boxes: list,
    ) -> list:
        """Default: sequential loop. Override for true batch GPU inference."""
        return [
            self.differentiable_predict_box(images[i], target_boxes[i])
            for i in range(images.shape[0])
        ]

    def differentiable_predict_box_batch_two_targets(
        self,
        images: torch.Tensor,
        target_boxes1: list,
        target_boxes2: list,
    ) -> list:
        """One forward pass, two independent selections per image.

        Returns list of ((conf1, box1), (conf2, box2)) per image.
        Default: sequential, calls predict() once per image and selects twice.
        Override in batched backends to avoid sequential inference.
        """
        thresh = getattr(self, "conf_threshold", 0.25)
        results = []
        for i in range(images.shape[0]):
            dets = self.predict(images[i])
            if not dets:
                zero = torch.tensor(0.0, device=self.device)
                results.append(((zero, None), (zero, None)))
                continue
            sc = torch.stack([d.conf.to(self.device) for d in dets])
            bx = torch.stack([d.box.to(self.device) for d in dets])
            r1 = _select_best_from_scores_boxes(sc, bx, target_boxes1[i], thresh, self.device, mode="suppress")
            r2 = _select_best_from_scores_boxes(sc, bx, target_boxes2[i], thresh, self.device, mode="attract")
            results.append((r1, r2))
        return results

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
        img_np = _pad_to_stride(img_np)   # ensure H,W divisible by 32
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

    def differentiable_det_loss_batch(self, images: torch.Tensor,
                                       target_boxes: list) -> list:
        """True batch forward: one GPU call for B images."""
        self.ensure_loaded()
        raw = self._model(images)           # tuple; raw[0] shape [B, 4+C, A]
        preds = raw[0].permute(0, 2, 1)    # [B, A, 4+C]
        boxes_cxcywh = preds[..., :4]
        boxes_xyxy = torch.cat([
            boxes_cxcywh[..., :2] - boxes_cxcywh[..., 2:] / 2,
            boxes_cxcywh[..., :2] + boxes_cxcywh[..., 2:] / 2,
        ], dim=-1)                          # [B, A, 4]
        scores = preds[..., 4:].max(-1).values  # [B, A]

        losses = []
        for i in range(images.shape[0]):
            tb = target_boxes[i].to(self.device)
            with torch.no_grad():
                ious = _box_iou_vectorized(tb, boxes_xyxy[i].detach())
                best_idx = int((ious * scores[i].detach()).argmax().item())
            losses.append(scores[i][best_idx])
        return losses

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

    def _diff_preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Replicate the HF processor's resize + normalize in a differentiable way.
        image: [C, H, W] float32 [0, 1]
        Returns: [1, C, target_h, target_w] normalized tensor with grad.
        """
        size = self._processor.size  # e.g. {"height": 640, "width": 640}
        th, tw = size["height"], size["width"]
        x = F.interpolate(image.unsqueeze(0), size=(th, tw),
                          mode="bilinear", align_corners=False)
        mean = torch.tensor(self._processor.image_mean,
                            dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor(self._processor.image_std,
                            dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std  # [1, 3, th, tw]

    def differentiable_det_loss(self, image: torch.Tensor,
                                target_box: torch.Tensor) -> torch.Tensor:
        """
        Differentiable forward: bypass the PIL/numpy processor path so
        gradients flow from the confidence score back to the input image.
        """
        self.ensure_loaded()
        inp = self._diff_preprocess(image.to(self.device))   # [1, 3, H, W]
        outputs = self._model(pixel_values=inp)

        # outputs.logits:    [1, num_queries, num_classes]
        # outputs.pred_boxes:[1, num_queries, 4]  cxcywh  normalised [0,1]
        logits    = outputs.logits[0]      # [Q, C]
        pred_boxes = outputs.pred_boxes[0] # [Q, 4]

        # Confidence = max sigmoid score over classes
        scores = logits.sigmoid().max(dim=-1).values  # [Q]

        # Convert pred_boxes cxcywh→xyxy in pixel coords
        orig_h, orig_w = image.shape[1], image.shape[2]
        cx, cy, bw, bh = pred_boxes.unbind(-1)
        boxes_xyxy = torch.stack([
            (cx - bw / 2) * orig_w,
            (cy - bh / 2) * orig_h,
            (cx + bw / 2) * orig_w,
            (cy + bh / 2) * orig_h,
        ], dim=-1)  # [Q, 4]

        if len(scores) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Select best anchor by IoU × confidence (detached for selection only)
        box_t = target_box.to(self.device).unsqueeze(0)
        with torch.no_grad():
            weights = torch.stack([
                _box_iou_scalar(boxes_xyxy[i].detach().unsqueeze(0), box_t)
                * scores[i].detach()
                for i in range(len(scores))
            ])
        best_idx = int(weights.argmax().item())
        return scores[best_idx]

    def differentiable_det_loss_batch(self, images: torch.Tensor,
                                      target_boxes: list) -> list:
        """True batch forward: one model call for all B images."""
        self.ensure_loaded()
        B = images.shape[0]
        # Stack preprocessed images into one batch
        preprocessed = torch.cat(
            [self._diff_preprocess(images[i].to(self.device)) for i in range(B)],
            dim=0,
        )  # [B, 3, H, W]
        outputs = self._model(pixel_values=preprocessed)

        losses = []
        for i in range(B):
            logits     = outputs.logits[i]      # [Q, C]
            pred_boxes = outputs.pred_boxes[i]  # [Q, 4]

            scores = logits.sigmoid().max(dim=-1).values  # [Q]

            orig_h, orig_w = images.shape[2], images.shape[3]
            cx, cy, bw, bh = pred_boxes.unbind(-1)
            boxes_xyxy = torch.stack([
                (cx - bw / 2) * orig_w,
                (cy - bh / 2) * orig_h,
                (cx + bw / 2) * orig_w,
                (cy + bh / 2) * orig_h,
            ], dim=-1)  # [Q, 4]

            if len(scores) == 0:
                losses.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                continue

            box_t = target_boxes[i].to(self.device).unsqueeze(0)
            with torch.no_grad():
                weights = torch.stack([
                    _box_iou_scalar(boxes_xyxy[j].detach().unsqueeze(0), box_t)
                    * scores[j].detach()
                    for j in range(len(scores))
                ])
            best_idx = int(weights.argmax().item())
            losses.append(scores[best_idx])

        return losses

    def differentiable_predict_box_batch(self, images: torch.Tensor,
                                          target_boxes: list) -> list:
        """Batch forward returning (conf_loss, pred_box_or_None) per image."""
        self.ensure_loaded()
        B = images.shape[0]
        preprocessed = torch.cat(
            [self._diff_preprocess(images[i].to(self.device)) for i in range(B)],
            dim=0,
        )
        outputs = self._model(pixel_values=preprocessed)

        results = []
        for i in range(B):
            logits     = outputs.logits[i]
            pred_boxes = outputs.pred_boxes[i]
            scores = logits.sigmoid().max(dim=-1).values

            orig_h, orig_w = images.shape[2], images.shape[3]
            cx, cy, bw, bh = pred_boxes.unbind(-1)
            boxes_xyxy = torch.stack([
                (cx - bw / 2) * orig_w,
                (cy - bh / 2) * orig_h,
                (cx + bw / 2) * orig_w,
                (cy + bh / 2) * orig_h,
            ], dim=-1)

            if len(scores) == 0:
                results.append((torch.tensor(0.0, device=self.device,
                                             requires_grad=True), None))
                continue

            box_t = target_boxes[i].to(self.device).unsqueeze(0)
            with torch.no_grad():
                weights = torch.stack([
                    _box_iou_scalar(boxes_xyxy[j].detach().unsqueeze(0), box_t)
                    * scores[j].detach()
                    for j in range(len(scores))
                ])
            if weights.max().item() < 1e-6:
                results.append((torch.tensor(0.0, device=self.device), None))
                continue
            best_idx = int(weights.argmax().item())
            if scores[best_idx].detach().item() < self.conf_threshold:
                results.append((scores[best_idx], None))
            else:
                results.append((scores[best_idx], boxes_xyxy[best_idx]))

        return results

    def differentiable_predict_box_batch_two_targets(self, images, target_boxes1, target_boxes2):
        self.ensure_loaded()
        B = images.shape[0]
        preprocessed = torch.cat(
            [self._diff_preprocess(images[i].to(self.device)) for i in range(B)], dim=0)
        outputs = self._model(pixel_values=preprocessed)
        results = []
        for i in range(B):
            logits     = outputs.logits[i]
            pred_boxes = outputs.pred_boxes[i]
            scores_i   = logits.sigmoid().max(dim=-1).values
            orig_h, orig_w = images.shape[2], images.shape[3]
            cx, cy, bw, bh = pred_boxes.unbind(-1)
            boxes_xyxy_i = torch.stack([
                (cx - bw / 2) * orig_w, (cy - bh / 2) * orig_h,
                (cx + bw / 2) * orig_w, (cy + bh / 2) * orig_h,
            ], dim=-1)
            if len(scores_i) == 0:
                zero = torch.tensor(0.0, device=self.device, requires_grad=True)
                results.append(((zero, None), (zero, None)))
                continue
            r1 = _select_best_from_scores_boxes(scores_i, boxes_xyxy_i, target_boxes1[i], self.conf_threshold, self.device, mode="suppress")
            r2 = _select_best_from_scores_boxes(scores_i, boxes_xyxy_i, target_boxes2[i], self.conf_threshold, self.device, mode="attract")
            results.append((r1, r2))
        return results


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
        yolo-v9-t-256-license-plate-end2end  (fastest)
        yolo-v9-t-384-license-plate-end2end  (default, balanced)
        yolo-v9-t-416-license-plate-end2end
        yolo-v9-t-512-license-plate-end2end
        yolo-v9-t-640-license-plate-end2end
        yolo-v9-s-608-license-plate-end2end  (most accurate — small model)

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

            # BoundingBox is a dataclass with .x1/.y1/.x2/.y2 attributes;
            # fall back to index-based access for older versions.
            try:
                x1, y1, x2, y2 = float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)
            except AttributeError:
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

    def differentiable_det_loss_batch(self, images: torch.Tensor,
                                       target_boxes: list) -> list:
        """True batch forward: pass all B images in one model call."""
        self.ensure_loaded()
        images_list = [images[i] for i in range(images.shape[0])]
        outputs = self._model(images_list)
        if isinstance(outputs, dict):
            outputs = [outputs]

        losses = []
        for i, out in enumerate(outputs):
            scores = out.get("scores")
            boxes  = out.get("boxes")
            tb = target_boxes[i].to(self.device)
            if scores is None or len(scores) == 0:
                losses.append(torch.tensor(0.0, device=self.device))
                continue
            with torch.no_grad():
                weights = torch.tensor([
                    _box_iou_scalar(boxes[j].unsqueeze(0), tb.unsqueeze(0))
                    * scores[j].item()
                    for j in range(len(scores))
                ])
                best_idx = int(weights.argmax().item())
            losses.append(scores[best_idx])
        return losses

    def differentiable_predict_box_batch(self, images: torch.Tensor,
                                          target_boxes: list) -> list:
        """Batch forward returning (conf_loss, pred_box_or_None) per image."""
        self.ensure_loaded()
        images_list = [images[i] for i in range(images.shape[0])]
        outputs = self._model(images_list)
        if isinstance(outputs, dict):
            outputs = [outputs]

        results = []
        for i, out in enumerate(outputs):
            scores = out.get("scores")
            boxes  = out.get("boxes")
            tb = target_boxes[i].to(self.device)
            if scores is None or len(scores) == 0:
                results.append((torch.tensor(0.0, device=self.device), None))
                continue
            with torch.no_grad():
                weights = torch.tensor([
                    _box_iou_scalar(boxes[j].unsqueeze(0), tb.unsqueeze(0))
                    * scores[j].item()
                    for j in range(len(scores))
                ])
            if weights.max().item() < 1e-6:
                results.append((torch.tensor(0.0, device=self.device), None))
                continue
            with torch.no_grad():
                best_idx = int(weights.argmax().item())
            if scores[best_idx].detach().item() < self.conf_threshold:
                results.append((scores[best_idx], None))
            else:
                results.append((scores[best_idx], boxes[best_idx]))
        return results

    def differentiable_predict_box_batch_two_targets(self, images, target_boxes1, target_boxes2):
        self.ensure_loaded()
        images_list = [images[i] for i in range(images.shape[0])]
        outputs = self._model(images_list)
        if isinstance(outputs, dict):
            outputs = [outputs]
        results = []
        for i, out in enumerate(outputs):
            scores_i = out.get("scores")
            boxes_i  = out.get("boxes")
            if scores_i is None or len(scores_i) == 0:
                zero = torch.tensor(0.0, device=self.device)
                results.append(((zero, None), (zero, None)))
                continue
            r1 = _select_best_from_scores_boxes(scores_i, boxes_i, target_boxes1[i], self.conf_threshold, self.device, mode="suppress")
            r2 = _select_best_from_scores_boxes(scores_i, boxes_i, target_boxes2[i], self.conf_threshold, self.device, mode="attract")
            results.append((r1, r2))
        return results

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
# SAM backend  (segment-anything, differentiable prompt-to-mask path)
# ---------------------------------------------------------------------------

class SAMBackend(DetectorBackend):
    """
    Wraps Meta's Segment Anything Model as a detector-like backend.

    This backend keeps the forward path differentiable by avoiding numpy,
    argmax-based mask selection, and OpenCV contour extraction. Instead it:

    1. encodes the image with SAM,
    2. queries a small bank of fixed box prompts,
    3. converts mask logits to soft probabilities,
    4. extracts a soft bounding box from mask moments.

    The resulting boxes are heuristic rather than a purpose-built detection
    head, but gradients flow from the returned confidence tensor back to the
    input image through SAM.
    """

    name = "sam"

    def __init__(self, model_path: str, device: str = "cpu",
                 conf_threshold: float = 0.20, model_type: Optional[str] = None,
                 multimask_output: bool = True):
        super().__init__(model_path, device)
        self.conf_threshold = conf_threshold
        self.multimask_output = multimask_output
        self.model_type = model_type or self._infer_model_type(model_path)
        self._model: Optional[nn.Module] = None

    @staticmethod
    def _infer_model_type(model_path: str) -> str:
        name = Path(model_path).name.lower()
        if "vit_h" in name:
            return "vit_h"
        if "vit_l" in name:
            return "vit_l"
        if "vit_b" in name:
            return "vit_b"
        return "vit_b"

    @staticmethod
    def _resize_longest_side(image: torch.Tensor, target_length: int) -> torch.Tensor:
        _, _, h, w = image.shape
        scale = float(target_length) / float(max(h, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)

    @staticmethod
    def _scale_boxes(boxes: torch.Tensor, original_size: tuple[int, int], resized_size: tuple[int, int]) -> torch.Tensor:
        orig_h, orig_w = original_size
        resized_h, resized_w = resized_size
        scaled = boxes.clone()
        scaled[:, [0, 2]] = scaled[:, [0, 2]] * (float(resized_w) / float(orig_w))
        scaled[:, [1, 3]] = scaled[:, [1, 3]] * (float(resized_h) / float(orig_h))
        return scaled

    @staticmethod
    def _prompt_bank(width: int, height: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        w = float(width)
        h = float(height)
        boxes = [
            [0.00 * w, 0.00 * h, 1.00 * w, 1.00 * h],
            [0.05 * w, 0.25 * h, 0.95 * w, 0.80 * h],
            [0.10 * w, 0.35 * h, 0.90 * w, 0.75 * h],
            [0.15 * w, 0.42 * h, 0.85 * w, 0.70 * h],
            [0.20 * w, 0.45 * h, 0.80 * w, 0.68 * h],
            [0.00 * w, 0.40 * h, 1.00 * w, 0.72 * h],
        ]
        return torch.tensor(boxes, device=device, dtype=dtype)

    @staticmethod
    def _soft_box_from_mask(mask_probs: torch.Tensor) -> torch.Tensor:
        h, w = mask_probs.shape
        rows = mask_probs.sum(dim=1)
        cols = mask_probs.sum(dim=0)
        row_total = rows.sum().clamp_min(1e-6)
        col_total = cols.sum().clamp_min(1e-6)

        y_coords = torch.arange(h, device=mask_probs.device, dtype=mask_probs.dtype)
        x_coords = torch.arange(w, device=mask_probs.device, dtype=mask_probs.dtype)

        y_mean = (rows * y_coords).sum() / row_total
        x_mean = (cols * x_coords).sum() / col_total
        y_var = (rows * (y_coords - y_mean).square()).sum() / row_total
        x_var = (cols * (x_coords - x_mean).square()).sum() / col_total

        height = (4.0 * torch.sqrt(y_var + 1e-6)).clamp(8.0, float(h))
        width = (4.0 * torch.sqrt(x_var + 1e-6)).clamp(16.0, float(w))

        x1 = (x_mean - width / 2.0).clamp(0.0, float(w - 1))
        y1 = (y_mean - height / 2.0).clamp(0.0, float(h - 1))
        x2 = (x_mean + width / 2.0).clamp(0.0, float(w - 1))
        y2 = (y_mean + height / 2.0).clamp(0.0, float(h - 1))
        return torch.stack([x1, y1, x2, y2])

    @staticmethod
    def _confidence_from_mask(mask_probs: torch.Tensor, iou_prediction: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        h, w = mask_probs.shape
        mask_mass = mask_probs.mean().clamp(0.0, 1.0)
        box_w = (box[2] - box[0]).clamp_min(1.0)
        box_h = (box[3] - box[1]).clamp_min(1.0)
        aspect = box_w / box_h
        aspect_score = torch.exp(-torch.abs(aspect - 3.0) / 2.5)
        rectangularity = (mask_probs.sum() / (box_w * box_h).clamp_min(1.0)).clamp(0.0, 1.0)
        spatial_prior = (box_h / float(h)).clamp(0.0, 1.0) * (box_w / float(w)).clamp(0.0, 1.0)
        iou_score = iou_prediction.clamp(0.0, 1.0)
        return (0.50 * iou_score + 0.20 * mask_mass + 0.20 * aspect_score + 0.10 * rectangularity * spatial_prior).clamp(0.0, 1.0)

    def load(self) -> None:
        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "segment-anything is required for the SAM backend. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {self.model_path}")

        self._model = sam_model_registry[self.model_type](checkpoint=str(self.model_path))
        self._model.to(self.device)
        self._model.eval()
        print(f"[{self.name}] Loaded {self.model_type} from {self.model_path}")

    def predict(self, image: torch.Tensor) -> List[Detection]:
        self.ensure_loaded()
        assert self._model is not None

        image = image.to(self.device).clamp(0.0, 1.0)
        if image.dim() != 3:
            raise ValueError(f"SAM backend expects CHW tensor, got shape {tuple(image.shape)}")

        _, orig_h, orig_w = image.shape
        rgb = image.unsqueeze(0) * 255.0
        resized = self._resize_longest_side(rgb, self._model.image_encoder.img_size)
        input_image = self._model.preprocess(resized)
        resized_h, resized_w = resized.shape[-2:]

        image_embeddings = self._model.image_encoder(input_image)

        prompt_boxes = self._prompt_bank(orig_w, orig_h, image.device, image.dtype)
        prompt_boxes = self._scale_boxes(prompt_boxes, (orig_h, orig_w), (resized_h, resized_w))

        sparse_embeddings, dense_embeddings = self._model.prompt_encoder(
            points=None,
            boxes=prompt_boxes,
            masks=None,
        )

        n_prompts = prompt_boxes.shape[0]
        low_res_masks, iou_predictions = self._model.mask_decoder(
            image_embeddings=image_embeddings.repeat_interleave(n_prompts, dim=0),
            image_pe=self._model.prompt_encoder.get_dense_pe().repeat_interleave(n_prompts, dim=0),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
        )

        masks = self._model.postprocess_masks(
            low_res_masks,
            input_size=(resized_h, resized_w),
            original_size=(orig_h, orig_w),
        ).sigmoid()

        detections: List[Detection] = []
        zero = image.new_tensor(0.0)

        for prompt_idx in range(masks.shape[0]):
            for mask_idx in range(masks.shape[1]):
                mask_probs = masks[prompt_idx, mask_idx]
                box = self._soft_box_from_mask(mask_probs)
                conf = self._confidence_from_mask(mask_probs, iou_predictions[prompt_idx, mask_idx], box)
                conf_value = float(conf.detach().item())
                if conf_value < self.conf_threshold:
                    continue

                raw = torch.stack([
                    zero,
                    box[0], box[1], box[2], box[3],
                    zero,
                    conf,
                ])
                detections.append(Detection(
                    x1=float(box[0].detach().item()),
                    y1=float(box[1].detach().item()),
                    x2=float(box[2].detach().item()),
                    y2=float(box[3].detach().item()),
                    confidence=conf_value,
                    class_id=0,
                    raw=raw,
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "SAMBackend":
        if self._model is not None:
            self._model.eval()
        return self

    def train_mode(self) -> "SAMBackend":
        if self._model is not None:
            self._model.train()
        return self

    def to(self, device: str) -> "SAMBackend":
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
        img_np  = _pad_to_stride(img_np)   # ensure H,W divisible by 32
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

    def differentiable_det_loss_batch(self, images: torch.Tensor,
                                       target_boxes: list) -> list:
        """True batch forward: one GPU call for B images."""
        self.ensure_loaded()
        raw = self._model(images)           # tuple; raw[0] shape [B, 4+C, A]
        preds = raw[0].permute(0, 2, 1)    # [B, A, 4+C]
        boxes_cxcywh = preds[..., :4]
        boxes_xyxy = torch.cat([
            boxes_cxcywh[..., :2] - boxes_cxcywh[..., 2:] / 2,
            boxes_cxcywh[..., :2] + boxes_cxcywh[..., 2:] / 2,
        ], dim=-1)                          # [B, A, 4]
        scores = preds[..., 4:].max(-1).values  # [B, A]

        losses = []
        for i in range(images.shape[0]):
            tb = target_boxes[i].to(self.device)
            with torch.no_grad():
                ious = _box_iou_vectorized(tb, boxes_xyxy[i].detach())
                best_idx = int((ious * scores[i].detach()).argmax().item())
            losses.append(scores[i][best_idx])
        return losses

    def differentiable_predict_box_batch(self, images: torch.Tensor,
                                          target_boxes: list) -> list:
        """Batch forward returning (conf_loss, pred_box_or_None) per image."""
        self.ensure_loaded()
        raw = self._model(images)
        preds = raw[0].permute(0, 2, 1)    # [B, A, 4+C]
        boxes_cxcywh = preds[..., :4]
        boxes_xyxy = torch.cat([
            boxes_cxcywh[..., :2] - boxes_cxcywh[..., 2:] / 2,
            boxes_cxcywh[..., :2] + boxes_cxcywh[..., 2:] / 2,
        ], dim=-1)
        scores = preds[..., 4:].max(-1).values  # [B, A]

        results = []
        for i in range(images.shape[0]):
            tb = target_boxes[i].to(self.device)
            with torch.no_grad():
                ious = _box_iou_vectorized(tb, boxes_xyxy[i].detach())
                if ious.max().item() < 1e-6:
                    results.append((torch.tensor(0.0, device=self.device), None))
                    continue
                best_idx = int((ious * scores[i].detach()).argmax().item())
            if scores[i][best_idx].detach().item() < self.conf_threshold:
                results.append((scores[i][best_idx], None))
            else:
                results.append((scores[i][best_idx], boxes_xyxy[i][best_idx]))
        return results

    def differentiable_predict_box_batch_two_targets(self, images, target_boxes1, target_boxes2):
        self.ensure_loaded()
        raw = self._model(images)
        preds = raw[0].permute(0, 2, 1)
        boxes_cxcywh = preds[..., :4]
        boxes_xyxy = torch.cat([
            boxes_cxcywh[..., :2] - boxes_cxcywh[..., 2:] / 2,
            boxes_cxcywh[..., :2] + boxes_cxcywh[..., 2:] / 2,
        ], dim=-1)
        scores = preds[..., 4:].max(-1).values
        results = []
        for i in range(images.shape[0]):
            if scores[i].numel() == 0:
                zero = torch.tensor(0.0, device=self.device)
                results.append(((zero, None), (zero, None)))
                continue
            r1 = _select_best_from_scores_boxes(scores[i], boxes_xyxy[i], target_boxes1[i], self.conf_threshold, self.device, mode="suppress")
            r2 = _select_best_from_scores_boxes(scores[i], boxes_xyxy[i], target_boxes2[i], self.conf_threshold, self.device, mode="attract")
            results.append((r1, r2))
        return results

    def to(self, device: str) -> "YOLOv11Backend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# ---------------------------------------------------------------------------
# YOLOv9 384 native-PyTorch backend  — differentiable via yolov9_torch.py
# ---------------------------------------------------------------------------

class Yolov9TorchBackend(DetectorBackend):
    """
    Wraps the YOLOv9-t-384 licence-plate model via the native PyTorch
    reconstruction in yolov9_torch.py (ultralytics DetectionModel + ONNX
    weights).  No onnx2torch required.

    The model is built from the ultralytics YOLOv9-t YAML with nc=1, BN
    fused, and all Conv weights loaded directly from the ONNX file.

    Model path (auto-downloaded by open-image-models on first use)
    --------------------------------------------------------------
    ~/.cache/open-image-models/yolo-v9-t-384-license-plate-end2end/
        yolo-v9-t-384-license-plates-end2end.onnx

    Input convention
    ----------------
    CHW float32 [0, 1], letterboxed to 384×384.

    Output convention (after NMS)
    ------------------------------
    Returns a list of Detection objects whose ``raw`` tensor carries a
    grad-tracked confidence score — gradients flow from conf back through
    the model into the input image / adversarial patch.
    """

    name = "yolo-v9-384"

    ONNX_PATH = ("~/.cache/open-image-models/"
                 "yolo-v9-t-384-license-plate-end2end/"
                 "yolo-v9-t-384-license-plates-end2end.onnx")

    def __init__(self, model_path: str = "none", device: str = "cpu",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        super().__init__(model_path, device)
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self._model: Optional[nn.Module] = None

    def load(self) -> None:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from yolov9_torch import load_yolov9t_from_onnx

        onnx_path = Path(self.ONNX_PATH).expanduser()

        if not onnx_path.exists():
            print(f"[{self.name}] ONNX model not found — downloading via open-image-models…")
            from open_image_models import LicensePlateDetector
            LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"[{self.name}] ONNX model still not found after download attempt: {onnx_path}"
            )

        self._model = load_yolov9t_from_onnx(str(onnx_path), nc=1)
        self._model.to(self.device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        print(f"[{self.name}] Loaded from {onnx_path}")

    # Fixed input size this model was trained on.
    INPUT_SIZE: int = 384

    @staticmethod
    def _letterbox(image: torch.Tensor, size: int) -> Tuple[torch.Tensor, float, int, int]:
        """
        Letterbox a CHW float32 [0,1] tensor to (size × size).
        Returns (padded_tensor, scale, pad_left, pad_top).
        Detections produced on the padded tensor can be mapped back to
        original coordinates with:  x_orig = (x_padded - pad_left) / scale
        """
        c, h, w = image.shape
        scale   = size / max(h, w)
        new_h   = round(h * scale)
        new_w   = round(w * scale)
        resized = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                                mode="bilinear", align_corners=False).squeeze(0)
        pad_top  = (size - new_h) // 2
        pad_left = (size - new_w) // 2
        padded   = F.pad(resized,
                         (pad_left, size - new_w - pad_left,
                          pad_top,  size - new_h - pad_top),
                         value=114 / 255)   # standard YOLO letterbox gray
        return padded, scale, pad_left, pad_top

    def predict(self, image: torch.Tensor) -> List[Detection]:
        """
        Run detection.  Input is letterboxed to INPUT_SIZE×INPUT_SIZE so the
        model always receives a stride-aligned tensor regardless of the original
        image dimensions.  Detections are returned in original image coordinates.

        Gradients flow through the raw confidence scores because the model
        parameters are frozen but the computation graph (image → conv → scores)
        is intact.
        """
        self.ensure_loaded()
        lb, scale, pad_left, pad_top = self._letterbox(image, self.INPUT_SIZE)
        batch = lb.unsqueeze(0).to(self.device)   # [1, 3, INPUT_SIZE, INPUT_SIZE]

        # Run model — do NOT use no_grad so the autograd graph is intact.
        output = self._model(batch)
        if isinstance(output, (list, tuple)):
            output = output[0]   # [1, 5, num_anchors]

        pred = output[0]         # [5, num_anchors]

        # Boxes are in letterboxed pixel space; unscale to original image space.
        bx = (pred[0] - pad_left) / scale
        by = (pred[1] - pad_top)  / scale
        bw = pred[2] / scale
        bh = pred[3] / scale
        scores = pred[4]         # [num_anchors]

        # Detach coordinates for NMS (non-differentiable op).
        x1 = (bx - bw / 2).detach()
        y1 = (by - bh / 2).detach()
        x2 = (bx + bw / 2).detach()
        y2 = (by + bh / 2).detach()
        boxes_xyxy  = torch.stack([x1, y1, x2, y2], dim=1)   # [A, 4]
        scores_det  = scores.detach()                          # [A]

        # Confidence filter
        mask = scores_det > self.conf_threshold
        if not mask.any():
            return []

        boxes_f  = boxes_xyxy[mask]
        scores_f = scores_det[mask]
        scores_g = scores[mask]     # grad-tracked conf for kept anchors

        kept = _nms(boxes_f, scores_f, self.iou_threshold)

        detections: List[Detection] = []
        for idx in kept:
            b    = boxes_f[idx]
            conf = scores_f[idx].item()
            x1v, y1v, x2v, y2v = b[0].item(), b[1].item(), b[2].item(), b[3].item()
            raw = torch.cat([
                torch.tensor([0.0, x1v, y1v, x2v, y2v, 0.0],
                             dtype=torch.float32, device=self.device),
                scores_g[idx].unsqueeze(0),
            ])
            detections.append(Detection(
                x1=x1v, y1=y1v, x2=x2v, y2=y2v,
                confidence=conf, class_id=0, raw=raw,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def differentiable_det_loss_batch(self, images: torch.Tensor,
                                       target_boxes: list) -> list:
        """True batch forward: one model call for all B images."""
        self.ensure_loaded()
        batch  = images.to(self.device)   # [B, 3, 384, 384]
        output = self._model(batch)
        if isinstance(output, (list, tuple)):
            output = output[0]            # [B, 5, A]

        bx, by, bw, bh = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        scores = output[:, 4]             # [B, A]
        boxes  = torch.stack([bx - bw / 2, by - bh / 2,
                               bx + bw / 2, by + bh / 2], dim=-1)  # [B, A, 4]

        losses = []
        for i in range(images.shape[0]):
            tb = target_boxes[i].to(self.device)
            with torch.no_grad():
                ious     = _box_iou_vectorized(tb, boxes[i].detach())
                best_idx = int((ious * scores[i].detach()).argmax().item())
            losses.append(scores[i][best_idx])
        return losses

    def differentiable_predict_box_batch(self, images: torch.Tensor,
                                          target_boxes: list) -> list:
        """Batch forward returning (conf_loss, pred_box_or_None) per image."""
        self.ensure_loaded()
        batch  = images.to(self.device)
        output = self._model(batch)
        if isinstance(output, (list, tuple)):
            output = output[0]            # [B, 5, A]

        bx, by, bw, bh = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        scores = output[:, 4]             # [B, A]
        boxes  = torch.stack([bx - bw / 2, by - bh / 2,
                               bx + bw / 2, by + bh / 2], dim=-1)  # [B, A, 4]

        results = []
        for i in range(images.shape[0]):
            tb = target_boxes[i].to(self.device)
            with torch.no_grad():
                ious     = _box_iou_vectorized(tb, boxes[i].detach())
                if ious.max().item() < 1e-6:
                    results.append((torch.tensor(0.0, device=self.device), None))
                    continue
                best_idx = int((ious * scores[i].detach()).argmax().item())
            if scores[i][best_idx].detach().item() < self.conf_threshold:
                results.append((scores[i][best_idx], None))
            else:
                results.append((scores[i][best_idx], boxes[i][best_idx]))
        return results

    def differentiable_predict_box_batch_two_targets(self, images, target_boxes1, target_boxes2):
        self.ensure_loaded()
        batch = images.to(self.device)
        output = self._model(batch)
        if isinstance(output, (list, tuple)):
            output = output[0]
        bx, by, bw, bh = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        scores = output[:, 4]
        boxes = torch.stack([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2], dim=-1)
        results = []
        for i in range(images.shape[0]):
            if scores[i].numel() == 0:
                zero = torch.tensor(0.0, device=self.device)
                results.append(((zero, None), (zero, None)))
                continue
            r1 = _select_best_from_scores_boxes(scores[i], boxes[i], target_boxes1[i], self.conf_threshold, self.device, mode="suppress")
            r2 = _select_best_from_scores_boxes(scores[i], boxes[i], target_boxes2[i], self.conf_threshold, self.device, mode="attract")
            results.append((r1, r2))
        return results

    def parameters(self) -> Iterator[nn.Parameter]:
        if self._model is not None:
            yield from self._model.parameters()

    def eval(self) -> "Yolov9TorchBackend":
        if self._model is not None:
            self._model.eval()
        return self

    def to(self, device: str) -> "Yolov9TorchBackend":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self


# Backends where gradients flow through the detector — usable for adversarial training.
TRAINABLE_REGISTRY: dict[str, type[DetectorBackend]] = {
    "sam":         SAMBackend,               # pip install segment-anything | weights: sam_vit_*.pth
    "yolov8":      YOLOv8Backend,            # pip install ultralytics  |  weights: your fine-tuned .pt
    "fasterrcnn":  FasterRCNNBackend,        # pip install torchvision  |  weights: weights/model.pt
    "yolov11":     YOLOv11Backend,           # pip install ultralytics  |  weights: morsetechlab/yolov11-license-plate-detection (HF)
    "rtdetr":      RTDETRBackend,            # pip install transformers  |  weights: justjuu/rtdetr-v2-license-plate-detection (HF)
    "yolo-v9-384": Yolov9TorchBackend,       # pip install onnx open-image-models  |  auto-downloaded
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
