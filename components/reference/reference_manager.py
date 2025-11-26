# components/reference/reference_manager.py
"""
기준 이미지에서 한번 계산한 임베딩을 캐시해 반복 계산을 줄입니다.
Caches and manages reference embeddings derived from the target image.
"""
from typing import List
import numpy as np
from ..embedding.arcface_embedder import FaceEmbedder


class ReferenceManager:
    """
    기준 임베딩 관리자
    - 기준 인물 얼굴 임베딩 저장
    - 중복 계산 방지
    - matcher에서 사용하기 위한 임베딩 목록 제공
    """

    def __init__(self, embedder: FaceEmbedder) -> None:
        self.embedder = embedder
        self.references: List[np.ndarray] = []  # 기준 임베딩 리스트

    def get_embedding(self, aligned_face):
        return self.embedder.embed(aligned_face)
    
    # 기준 임베딩 추가 (내부에서 get_embedding 사용)
    def add(self, aligned_face: np.ndarray) -> None:
        emb = self.get_embedding(aligned_face)
        if emb is not None and emb.size != 0:
            self.references.append(emb)

    # matcher가 사용하기 위한 리스트
    def get_all(self) -> List[np.ndarray]:
        return self.references
