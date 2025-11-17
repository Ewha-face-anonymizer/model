"""
기준 이미지에서 한번 계산한 임베딩을 캐시해 반복 계산을 줄입니다.
Caches and manages reference embeddings derived from the target image.
"""
from pathlib import Path
from typing import Dict

import numpy as np

from components.embedding.arcface_embedder import FaceEmbedder


class ReferenceManager:
    """Loads/caches reference embeddings to avoid redundant computation."""

    def __init__(self, embedder: FaceEmbedder) -> None:
        self.embedder = embedder
        self._cache: Dict[Path, np.ndarray] = {}

    def get_embedding(self, reference_image: Path) -> np.ndarray:
        if reference_image not in self._cache:
            # Placeholder: load pre-cropped face until detector integration
            image = np.zeros((112, 112, 3), dtype=np.uint8)
            self._cache[reference_image] = self.embedder.embed(image)
        return self._cache[reference_image]
