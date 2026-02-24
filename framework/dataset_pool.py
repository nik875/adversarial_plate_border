"""
LazyDatasetPool — memory-efficient multi-dataset image pool.

Design:
    - Only file paths are stored at registration time.
    - Actual image loading is deferred to each sample() call (no pixel data retained).
    - Path discovery (glob) is deferred to the first sample_from() call for each dataset.

Usage::

    pool = LazyDatasetPool()
    did0 = pool.register('imagenet', '/data/imagenet/val', max_samples=1000)
    did1 = pool.register('coco',     '/data/coco/images',  max_samples=500)

    item = pool.sample()             # uniform over datasets, then within dataset
    item = pool.sample_from(did0)    # specific dataset

    item.image        # [3, H, W] float in [0, 1]
    item.dataset_id   # int
    item.original_index  # int (index within this dataset's path list)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms


# Default transform used when no custom transform is provided
_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPEG', 'JPG', 'PNG', 'webp'}


@dataclass
class SampledItem:
    """A single image drawn from the pool, with provenance metadata."""
    image: Tensor          # [3, H, W] float in [0, 1]
    dataset_id: int
    original_index: int    # index within this dataset's shuffled path list


class LazyDatasetPool:
    """
    A pool of multiple image datasets, loaded lazily by file path.

    Thread-safety: not thread-safe (single-threaded training assumed).
    """

    def __init__(self, transform: Optional[Callable] = None):
        """
        Args:
            transform: optional torchvision-style transform applied to each loaded PIL image.
                       Defaults to Resize(224×224) + ToTensor().
        """
        self._transform: Callable = transform or _DEFAULT_TRANSFORM
        self._entries: Dict[int, dict] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        root: str,
        domain_type: str = 'generic',
        max_samples: Optional[int] = None,
    ) -> int:
        """
        Register a dataset directory.

        Path discovery (glob) is deferred to the first sample_from() call.

        Args:
            name:        human-readable identifier
            root:        root directory; images are discovered recursively
            domain_type: arbitrary tag (e.g. 'imagenet', 'coco', 'generic')
            max_samples: cap on number of images; None = use all found

        Returns:
            dataset_id (int) — use this to sample from a specific dataset
        """
        dataset_id = self._next_id
        self._next_id += 1
        self._entries[dataset_id] = {
            'name': name,
            'root': Path(root).expanduser(),
            'domain_type': domain_type,
            'max_samples': max_samples,
            'paths': None,   # populated lazily
        }
        return dataset_id

    def register_paths(
        self,
        name: str,
        paths: List[str],
        domain_type: str = 'generic',
    ) -> int:
        """
        Register with an explicit list of paths (no filesystem discovery).

        Useful for synthetic datasets or pre-filtered path lists.
        """
        dataset_id = self._next_id
        self._next_id += 1
        self._entries[dataset_id] = {
            'name': name,
            'root': None,
            'domain_type': domain_type,
            'max_samples': None,
            'paths': list(paths),
        }
        return dataset_id

    # ------------------------------------------------------------------
    # Path discovery (lazy)
    # ------------------------------------------------------------------

    def _ensure_paths(self, dataset_id: int) -> List[str]:
        """Discover and cache paths for a dataset if not yet done."""
        entry = self._entries[dataset_id]
        if entry['paths'] is not None:
            return entry['paths']

        root: Path = entry['root']
        if root is None or not root.exists():
            raise RuntimeError(
                f"Dataset '{entry['name']}' (id={dataset_id}): "
                f"root directory '{root}' does not exist."
            )

        paths: List[str] = []
        for ext in _IMAGE_EXTENSIONS:
            paths.extend(str(p) for p in root.glob(f'**/*.{ext}'))

        random.shuffle(paths)

        cap = entry['max_samples']
        if cap is not None:
            paths = paths[:cap]

        if not paths:
            raise RuntimeError(
                f"Dataset '{entry['name']}' (id={dataset_id}): "
                f"no images found in '{root}'."
            )

        entry['paths'] = paths
        return paths

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self) -> SampledItem:
        """Sample uniformly over registered datasets, then uniformly within."""
        if not self._entries:
            raise RuntimeError("No datasets registered. Call register() first.")
        dataset_id = random.choice(list(self._entries.keys()))
        return self.sample_from(dataset_id)

    def sample_from(self, dataset_id: int) -> SampledItem:
        """Sample a uniformly random image from the specified dataset."""
        if dataset_id not in self._entries:
            raise KeyError(f"dataset_id={dataset_id} not registered.")

        paths = self._ensure_paths(dataset_id)
        idx = random.randint(0, len(paths) - 1)
        path = paths[idx]

        img = Image.open(path).convert('RGB')
        tensor: Tensor = self._transform(img)   # [3, H, W] in [0, 1]

        return SampledItem(image=tensor, dataset_id=dataset_id, original_index=idx)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def num_datasets(self) -> int:
        return len(self._entries)

    def total_images(self) -> int:
        """Return total number of images across all registered datasets (triggers path discovery)."""
        return sum(len(self._ensure_paths(did)) for did in self._entries)

    def dataset_name(self, dataset_id: int) -> str:
        return self._entries[dataset_id]['name']

    def __repr__(self) -> str:
        names = [e['name'] for e in self._entries.values()]
        return f"LazyDatasetPool(datasets={names})"
