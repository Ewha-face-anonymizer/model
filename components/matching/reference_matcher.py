# components/matching/reference_matcher.py
"""
기준 임베딩과의 코사인 거리로 같은 사람인지 판단합니다.
Logic for comparing embeddings against a reference template.
"""
from typing import Iterable, List

import numpy as np


class ReferenceMatcher:
    """
    기준 임베딩 기반 매칭
    - 저장된 기준 임베딩들과 코사인 거리 계산
    - 최소 거리로 동일 인물 여부 판단
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def match(self, embedding: np.ndarray, reference: np.ndarray) -> bool:
        """단일 얼굴 임베딩을 기준 임베딩과 비교."""
        if embedding.size == 0 or reference.size == 0:
            return False

        cosine_similarity = float(
            np.dot(embedding, reference)
            / (np.linalg.norm(embedding) * np.linalg.norm(reference) + 1e-8)
        )
        distance = 1.0 - cosine_similarity
        return distance <= self.threshold

    def match_batch(
        self,
        embeddings: Iterable[np.ndarray],
        reference: np.ndarray,
    ) -> List[bool]:
        """여러 얼굴 임베딩을 한 번에 비교."""
        return [self.match(embedding, reference) for embedding in embeddings]
