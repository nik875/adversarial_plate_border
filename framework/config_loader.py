"""
YAML config loader → domain + strategy + metric instances.

Reads a YAML file and instantiates the right DomainAdapter, AttackStrategy,
EvalMetric, and supporting objects.  Used by both training and CMA-ES scripts.

Example YAML:
    domain: classification
    model:
      type: resnet50
      pretrained: true
    dataset:
      root: /data/imagenet
      split: val
      max_samples: 500
    strategy:
      type: border
      center_ratio: 0.5
    metric:
      type: top1_accuracy_drop
      target_class: 207
    patch:
      height: 256
      width: 512
    trainer:
      basis_dim: 16
      max_epochs: 100
      learning_rate: 0.01
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


@dataclass
class DomainConfig:
    """Instantiated objects from a YAML config."""
    domain_adapter: Any           # DomainAdapter instance
    strategy: Any                 # AttackStrategy instance
    metric: Any                   # EvalMetric instance
    target_model: Any             # The frozen target model (same as domain_adapter.model)
    raw: Dict                     # Original YAML dict for extra fields


def load_domain_config(config_path: str | Path) -> DomainConfig:
    """
    Parse a YAML config file and instantiate framework objects.

    Args:
        config_path: path to .yaml config

    Returns:
        DomainConfig with domain_adapter, strategy, metric, target_model, raw
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    domain_type = cfg.get('domain', '').lower()
    domain_adapter = _build_domain(domain_type, cfg)
    strategy = _build_strategy(cfg.get('strategy', {}))
    metric = _build_metric(cfg.get('metric', {}))

    return DomainConfig(
        domain_adapter=domain_adapter,
        strategy=strategy,
        metric=metric,
        target_model=domain_adapter.model,
        raw=cfg,
    )


# ---------------------------------------------------------------------------
# Internal builder functions
# ---------------------------------------------------------------------------

def _build_domain(domain_type: str, cfg: dict):
    if domain_type == 'classification':
        from framework.domains.classification import ImageClassificationDomain
        model_cfg = cfg.get('model', {})
        dataset_cfg = cfg.get('dataset', {})
        patch_cfg = cfg.get('patch', {})
        return ImageClassificationDomain(
            model_name=model_cfg.get('type', 'resnet50'),
            pretrained=model_cfg.get('pretrained', True),
            dataset_root=dataset_cfg.get('root', '/data/imagenet'),
            max_samples=dataset_cfg.get('max_samples', None),
            patch_height=patch_cfg.get('height', 224),
            patch_width=patch_cfg.get('width', 224),
        )
    elif domain_type == 'detection':
        from framework.domains.detection import ObjectDetectionDomain
        model_cfg = cfg.get('model', {})
        dataset_cfg = cfg.get('dataset', {})
        return ObjectDetectionDomain(
            model_name=model_cfg.get('type', 'yolov5s'),
            dataset_root=dataset_cfg.get('root', '/data/coco'),
        )
    elif domain_type == 'face':
        from framework.domains.face import FaceRecognitionDomain
        model_cfg = cfg.get('model', {})
        dataset_cfg = cfg.get('dataset', {})
        return FaceRecognitionDomain(
            model_name=model_cfg.get('type', 'arcface'),
            dataset_root=dataset_cfg.get('root', '/data/faces'),
        )
    else:
        raise ValueError(
            f"Unknown domain type: '{domain_type}'. "
            f"Supported: classification, detection, face"
        )


def _build_strategy(strategy_cfg: dict):
    from framework.base.attack_strategy import BorderStrategy, StickerStrategy, PerturbationStrategy

    strategy_type = strategy_cfg.get('type', 'border').lower()

    if strategy_type == 'border':
        return BorderStrategy(
            center_ratio=strategy_cfg.get('center_ratio', 0.6),
            neutral_color=strategy_cfg.get('neutral_color', 0.5),
        )
    elif strategy_type == 'sticker':
        bbox = strategy_cfg.get('bbox', None)
        if bbox is not None:
            bbox = tuple(bbox)
        return StickerStrategy(
            bbox=bbox,
            sticker_h=strategy_cfg.get('sticker_h', None),
            sticker_w=strategy_cfg.get('sticker_w', None),
            neutral_color=strategy_cfg.get('neutral_color', 0.5),
        )
    elif strategy_type in ('perturbation', 'additive'):
        return PerturbationStrategy(
            budget=strategy_cfg.get('budget', 0.05),
            norm=strategy_cfg.get('norm', 'linf'),
        )
    else:
        raise ValueError(
            f"Unknown strategy type: '{strategy_type}'. "
            f"Supported: border, sticker, perturbation"
        )


def _build_metric(metric_cfg: dict):
    from framework.base.metric import (
        TopKAccuracyDrop,
        EditDistanceMetric,
        DetectionDisruptionMetric,
    )

    metric_type = metric_cfg.get('type', '').lower()

    if metric_type in ('top1_accuracy_drop', 'top1', 'classification'):
        return TopKAccuracyDrop(
            k=metric_cfg.get('k', 1),
            target_class=metric_cfg.get('target_class', None),
        )
    elif metric_type in ('topk_accuracy_drop', 'topk'):
        return TopKAccuracyDrop(
            k=metric_cfg.get('k', 5),
            target_class=metric_cfg.get('target_class', None),
        )
    elif metric_type in ('edit_distance', 'ocr', 'levenshtein'):
        return EditDistanceMetric(
            correct_text=metric_cfg.get('correct_text', None),
        )
    elif metric_type in ('detection_disruption', 'detection', 'map_drop'):
        return DetectionDisruptionMetric(
            iou_threshold=metric_cfg.get('iou_threshold', 0.5),
            conf_threshold=metric_cfg.get('conf_threshold', 0.25),
        )
    else:
        raise ValueError(
            f"Unknown metric type: '{metric_type}'. "
            f"Supported: top1_accuracy_drop, topk_accuracy_drop, edit_distance, detection_disruption"
        )
