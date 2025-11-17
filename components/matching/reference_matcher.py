"""
 기준 임베딩과의 코사인 거리로 같은 사람인지 판단합니다.
Logic for comparing embeddings against a reference template.
"""
from typing import Iterable, List

import numpy as np


class ReferenceMatcher:
    """Computes cosine distance to determine if a face matches the reference."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def match(self, embedding: np.ndarray, reference: np.ndarray) -> bool:
        if embedding.size == 0 or reference.size == 0:
            return False
        cosine_similarity = float(
            np.dot(embedding, reference)
            / (np.linalg.norm(embedding) * np.linalg.norm(reference) + 1e-8)
        )
        distance = 1.0 - cosine_similarity
        return distance <= self.threshold

    def match_batch(self, embeddings: Iterable[np.ndarray], reference: np.ndarray) -> List[bool]:
        return [self.match(embedding, reference) for embedding in embeddings]
