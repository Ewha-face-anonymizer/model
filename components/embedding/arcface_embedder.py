"""
ArcFace/FaceNet을 사용해 얼굴을 벡터로 만드는 임베더 스텁입니다.
Face embedding utilities built around ArcFace or FaceNet.
"""
from pathlib import Path
from typing import Iterable, List

import numpy as np


class FaceEmbedder:
    """Loads the embedding network and produces normalized embeddings."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model = None  # Placeholder for ArcFace/FaceNet model

    def embed(self, face: np.ndarray) -> np.ndarray:
        """
        Run embedding inference on a single face crop.
        Replace with actual model inference code.
        """
        return np.zeros(512, dtype=np.float32)

    def embed_all(self, faces: Iterable[np.ndarray]) -> List[np.ndarray]:
        return [self.embed(face) for face in faces]
