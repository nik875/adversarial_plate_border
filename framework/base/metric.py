"""
EvalMetric ABC and concrete metrics.

An EvalMetric encapsulates how to measure the effectiveness of a patch:
  - TopKAccuracyDrop       : for image classification (Top-1 / Top-5)
  - EditDistanceMetric     : for OCR (Levenshtein; reference implementation)
  - DetectionDisruptionMetric : for object detection (mAP drop / miss-rate rise)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class EvalMetric(ABC):
    """
    Abstract base class for evaluation metrics.

    Usage pattern:
        control_outputs = metric.precompute_control(control_images, model)
        results = metric.compute(composited, control_outputs, model)
        # results['primary'] is the main scalar (higher = more adversarial)
    """

    @abstractmethod
    def precompute_control(
        self,
        control_images: List[Any],
        model: torch.nn.Module,
        **kwargs,
    ) -> List[Any]:
        """
        Run the model on clean/neutral images and cache output for comparison.

        Args:
            control_images: list of control images (numpy HWC uint8 or tensors)
            model: frozen target model
            **kwargs: extra context

        Returns:
            list of per-image control outputs (logits, texts, boxes, etc.)
        """

    @abstractmethod
    def compute(
        self,
        composited: List[Any],
        control_outputs: List[Any],
        model: torch.nn.Module,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate the patch effectiveness.

        Args:
            composited: list of composited images (same format as control_images)
            control_outputs: output from precompute_control
            model: frozen target model
            **kwargs: extra context

        Returns:
            dict with at least:
              'primary'      : main scalar (larger = more adversarial)
              'success_rate' : fraction of images where attack succeeded
        """


# ---------------------------------------------------------------------------
# Top-K accuracy drop (image classification)
# ---------------------------------------------------------------------------

class TopKAccuracyDrop(EvalMetric):
    """
    Measures how much the patch reduces Top-1 (or Top-K) accuracy.

    precompute_control stores the predicted class for each clean image.
    compute counts how many predictions changed (or left the top-k set).
    """

    def __init__(self, k: int = 1, target_class: Optional[int] = None):
        """
        Args:
            k: top-k accuracy to measure (default 1).
            target_class: if set, success = model predicts this class
                          (targeted attack); otherwise success = any misclassification.
        """
        self.k = k
        self.target_class = target_class

    @torch.no_grad()
    def precompute_control(
        self,
        control_images: List[Tensor],
        model: torch.nn.Module,
        **kwargs,
    ) -> List[int]:
        """Return list of predicted class indices for clean images."""
        model.eval()
        control_preds = []
        for img in control_images:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            logits = model(img)
            pred = logits.argmax(dim=1).item()
            control_preds.append(pred)
        return control_preds

    @torch.no_grad()
    def compute(
        self,
        composited: List[Tensor],
        control_outputs: List[int],
        model: torch.nn.Module,
        **kwargs,
    ) -> Dict[str, float]:
        model.eval()
        successes = 0
        top1_drop = 0.0

        for img, ctrl_pred in zip(composited, control_outputs):
            if img.dim() == 3:
                img = img.unsqueeze(0)
            logits = model(img)
            top_k = logits.topk(self.k, dim=1).indices.squeeze(0).tolist()

            if self.target_class is not None:
                success = (self.target_class in top_k)
            else:
                success = (ctrl_pred not in top_k)

            if success:
                successes += 1
            if logits.argmax(dim=1).item() != ctrl_pred:
                top1_drop += 1.0

        n = len(composited) if composited else 1
        return {
            'primary': top1_drop / n,          # fraction of Top-1 flips
            'success_rate': successes / n,      # targeted or untargeted success
            'top1_drop': top1_drop / n,
        }


# ---------------------------------------------------------------------------
# Edit distance (OCR) — reference; OCR domain keeps its own path in cmaes
# ---------------------------------------------------------------------------

class EditDistanceMetric(EvalMetric):
    """
    Levenshtein edit-distance metric for OCR attacks.

    precompute_control runs OCR on neutral composites and stores predicted texts.
    compute runs OCR on adversarial composites and returns mean edit distance.
    """

    def __init__(self, correct_text: Optional[str] = None):
        """
        Args:
            correct_text: if provided, images where clean OCR ≠ correct_text are skipped.
        """
        try:
            import Levenshtein as _lev
            self._lev = _lev
        except ImportError:
            raise ImportError("Install python-Levenshtein: pip install python-Levenshtein")

        self.correct_text = correct_text

    def _run_ocr(self, images, model):
        results = []
        for img in images:
            result = model.predict(img)
            results.append(result.text if result is not None else "")
        return results

    def precompute_control(
        self,
        control_images: List[Any],
        model,
        **kwargs,
    ) -> List[str]:
        return self._run_ocr(control_images, model)

    def compute(
        self,
        composited: List[Any],
        control_outputs: List[str],
        model,
        **kwargs,
    ) -> Dict[str, float]:
        composite_texts = self._run_ocr(composited, model)
        total_edit = 0.0
        misreads = 0
        for ctrl_text, comp_text in zip(control_outputs, composite_texts):
            dist = min(self._lev.distance(ctrl_text, comp_text), max(len(ctrl_text), 1))
            total_edit += dist
            if comp_text != ctrl_text:
                misreads += 1
        n = len(composited) if composited else 1
        return {
            'primary': total_edit / n,
            'success_rate': misreads / n,
            'mean_edit_distance': total_edit / n,
            'misread_rate': misreads / n,
        }


# ---------------------------------------------------------------------------
# Detection disruption (object detection)
# ---------------------------------------------------------------------------

class DetectionDisruptionMetric(EvalMetric):
    """
    Measures how much the patch disrupts object detection.

    Stub implementation — override precompute_control and compute for a
    specific detector (YOLO, Faster-RCNN, etc.).

    primary metric: fraction of images where the target object is no longer detected.
    """

    def __init__(self, iou_threshold: float = 0.5, conf_threshold: float = 0.25):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def precompute_control(
        self,
        control_images: List[Any],
        model,
        **kwargs,
    ) -> List[List[Dict]]:
        """
        Run detector on clean images.

        Returns list of per-image detection lists, each detection being a dict
        with keys: 'bbox' (x1,y1,x2,y2), 'score', 'class'.
        """
        raise NotImplementedError(
            "DetectionDisruptionMetric.precompute_control must be implemented "
            "for the specific detector being used."
        )

    def compute(
        self,
        composited: List[Any],
        control_outputs: List[List[Dict]],
        model,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compare detections on adversarial composites vs control.

        primary = fraction of images where target object is no longer detected.
        """
        raise NotImplementedError(
            "DetectionDisruptionMetric.compute must be implemented "
            "for the specific detector being used."
        )
