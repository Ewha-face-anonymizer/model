# components/matching/reference_matcher.py
"""
기준 임베딩과의 코사인 거리로 같은 사람인지 판단합니다.
Logic for comparing embeddings against a reference template.
"""
from typing import Iterable, List
import numpy as np
from ..reference.reference_manager import ReferenceManager


class ReferenceMatcher:
    """
    기준 임베딩 기반 매칭
    - 저장된 기준 임베딩들과 코사인 거리 계산
    - 최소 거리로 동일 인물 여부 판단
    """

    def __init__(self, threshold: float, ref_manager: ReferenceManager) -> None:
        self.threshold = threshold
        self.ref_manager = ref_manager  # reference_manager 연결

    def match(self, embedding: np.ndarray) -> bool:
        """
        manager에 저장된 모든 기준 임베딩과 비교. (기준 임베딩 개수 상관X)
        """
        references = self.ref_manager.get_all()   # manager에서 baseline 받아오기

        if len(references) == 0:
            return False  # 기준 없음

        # cosine distance 계산
        distances = []
        for reference in references:
            if embedding.size == 0 or reference.size == 0:
                continue
            cosine_similarity = float(
                np.dot(embedding, reference)
                / (np.linalg.norm(embedding) * np.linalg.norm(reference) + 1e-8)
            )
            distance = 1.0 - cosine_similarity
            distances.append(distance)

        if len(distances) == 0:
            return False # 기준 있음, 하지만 비어있음
        
        # 여러 기준 embedding 중 최소 거리만 비교
        min_dist = min(distances)
        
        return min_dist <= self.threshold


    # reference 인자를 없애고 manager 기반으로 처리하도록 수정.
    def match_batch(self, embeddings: Iterable[np.ndarray]) -> List[bool]:
        return [self.match(embedding) for embedding in embeddings]
