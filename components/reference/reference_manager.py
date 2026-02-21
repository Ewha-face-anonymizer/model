# components/reference/reference_manager.py
"""
기준 이미지에서 한번 계산한 임베딩을 캐시해 반복 계산을 줄입니다.
Caches and manages reference embeddings derived from the target image.
"""
from pathlib import Path
from typing import Dict

import numpy as np

from ..detection.mtcnn_detector import FaceDetector
from ..embedding.arcface_embedder import FaceEmbedder


class ReferenceManager:
    """
    기준 임베딩 관리자
    - 기준 인물 이미지에서 얼굴을 감지하고 임베딩을 계산
    - 동일 경로의 기준 이미지는 캐시해 반복 계산을 방지
    """

    def __init__(self, embedder: FaceEmbedder, detector: FaceDetector) -> None:
        self.embedder = embedder
        self.detector = detector
        self.cache: Dict[Path, np.ndarray] = {}

    def get_embedding(self, image_path: Path) -> np.ndarray:
        """
        기준 인물 이미지 경로를 받아 얼굴을 탐지하고 임베딩을 반환합니다.
        한 번 계산한 결과는 경로 기준으로 캐시됩니다.
        """
        image_path = image_path.resolve()
        if image_path in self.cache:
            return self.cache[image_path]

        image = self.detector.load_image(image_path)
        bboxes, faces = self.detector.detect_faces(image)
        if not faces:
            raise ValueError(f"기준 이미지에서 얼굴을 찾지 못했습니다: {image_path}")

        # 가장 첫 얼굴을 기준으로 사용
        embedding = self.embedder.embed(faces[0])
        self.cache[image_path] = embedding
        return embedding
