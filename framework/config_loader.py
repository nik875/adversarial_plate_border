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

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class EnsembleConfig:
    """
    Instantiated objects from an ensemble YAML config.

    The ensemble_pool and dataset_pool are returned pre-configured
    from the YAML; models are registered by the calling script.
    """
    ensemble_pool: Any            # EnsembleModelPool (models registered by caller)
    dataset_pool: Any             # LazyDatasetPool (paths registered from YAML)
    prior_registry: Any           # PriorRegistry or None
    task_encoder: Any             # TaskEncoder
    generator_cfg: Dict           # dict of generator hyperparameters
    trainer_cfg: Dict             # dict of EnsembleTrainer hyperparameters
    models_cfg: List[Dict]        # raw list of model configs from YAML
    strategies_cfg: List[Dict]    # raw list of strategy configs from YAML
    raw: Dict                     # full original YAML dict


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


def load_ensemble_config(config_path: str | Path) -> EnsembleConfig:
    """
    Parse an ensemble YAML config and instantiate framework objects.

    Objects built:
        - EnsembleModelPool  (empty — caller registers models from models_cfg)
        - LazyDatasetPool    (datasets registered from YAML)
        - TaskEncoder        (dimensions inferred from YAML)
        - prior_registry     always None (PriorRegistry removed)

    Args:
        config_path: path to ensemble .yaml config

    Returns:
        EnsembleConfig

    Example YAML structure::

        compute_device: cpu
        models:
          - name: resnet50
            domain_type: classification
            strategy: border
            strategy_id: 0
            input_shape: [224, 224]
        datasets:
          - name: imagenet
            root: /data/imagenet/val
            max_samples: 1000
        priors: []
        task_encoder:
          hidden_dim: 128
          num_layers: 3
          alpha: 0.5
        generator:
          latent_dim: 16
          patch_height: 256
          patch_width: 512
        trainer:
          k_neurons: 10000
          patches_per_batch: 4
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    from framework.ensemble import EnsembleModelPool
    from framework.dataset_pool import LazyDatasetPool
    from framework.task_encoder import TaskEncoder

    # Compute device
    device_str = cfg.get('compute_device', 'cpu')
    compute_device = torch.device(device_str)

    # Build EnsembleModelPool (empty — models registered by caller)
    ensemble_pool = EnsembleModelPool(compute_device=compute_device)

    # Dimensions for TaskEncoder
    models_cfg: List[Dict] = cfg.get('models', [])
    strategies_cfg: List[Dict] = cfg.get('strategies', [])
    num_models = max(len(models_cfg), 1)
    num_strategies = max(len(strategies_cfg), 1)
    datasets_cfg: List[Dict] = cfg.get('datasets', [])
    num_datasets = max(len(datasets_cfg), 1)

    # Load manifest if specified at top level of YAML
    manifest_path = cfg.get('manifest')
    manifest: Dict[Tuple[str, str], List[str]] = {}
    if manifest_path:
        mp = Path(manifest_path).expanduser()
        if mp.exists():
            with open(mp, newline='') as _f:
                for row in csv.DictReader(_f):
                    key = (row['dataset'], row['split'])
                    manifest.setdefault(key, []).append(row['path'])
        else:
            warnings.warn(
                f"Manifest file not found: {mp}. "
                "Falling back to glob-based dataset loading."
            )

    # PriorRegistry is no longer used — always None
    prior_registry = None
    gen_cfg = cfg.get('generator', {})
    latent_dim = gen_cfg.get('latent_dim', 16)

    # Build LazyDatasetPool — resize raw images to generator patch dimensions so
    # compositing happens at the right resolution; each model's preprocess_fn
    # then scales down to its own input size.
    import torchvision.transforms as _T
    gen_h = gen_cfg.get('patch_height', 512)
    gen_w = gen_cfg.get('patch_width', 512)
    raw_transform = _T.Compose([
        _T.Resize((gen_h, gen_w)),
        _T.ToTensor(),
    ])
    dataset_pool = LazyDatasetPool(transform=raw_transform)
    for ds in datasets_cfg:
        name = ds.get('name', 'unnamed')
        domain_type = ds.get('domain_type', 'generic')
        split = ds.get('split')

        if manifest and split is not None:
            paths = manifest.get((name, split), [])
            if paths:
                dataset_pool.register_paths(name, paths, domain_type)
            else:
                warnings.warn(
                    f"No paths found in manifest for dataset='{name}' "
                    f"split='{split}'. Dataset will be skipped."
                )
        else:
            root = ds.get('root', '/tmp/no_data')
            dataset_pool.register(
                name=name,
                root=root,
                domain_type=domain_type,
                max_samples=ds.get('max_samples', None),
            )

    # Build TaskEncoder
    te_cfg = cfg.get('task_encoder', {})
    task_encoder = TaskEncoder(
        num_models=num_models,
        num_strategies=num_strategies,
        num_datasets=num_datasets,
        latent_dim=latent_dim,
        hidden_dim=te_cfg.get('hidden_dim', 128),
        num_layers=te_cfg.get('num_layers', 3),
        alpha=te_cfg.get('alpha', 0.5),
    )

    return EnsembleConfig(
        ensemble_pool=ensemble_pool,
        dataset_pool=dataset_pool,
        prior_registry=prior_registry,
        task_encoder=task_encoder,
        generator_cfg=gen_cfg,
        trainer_cfg=cfg.get('trainer', {}),
        models_cfg=models_cfg,
        strategies_cfg=strategies_cfg,
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
