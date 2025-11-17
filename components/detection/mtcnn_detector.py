"""
MTCNN 모델을 감싸는 감지기 스텁입니다. 이미지 로딩/저장과 향후 얼굴 탐지 로직을 담당합니다.

Face detection utilities built around MTCNN.
"""
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


class FaceDetector:
    """
    Thin wrapper intended to handle MTCNN model loading, inference,
    and conversion into bounding boxes + aligned face crops.
    """

    def __init__(self, model_path: Path, min_size: int) -> None:
        self.model_path = model_path
        self.min_size = min_size
        # Placeholder: load actual model framework (e.g., facenet-pytorch)
        self.model = None

    def load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to load image at {image_path}")
        return image

    def save_image(self, image: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

    def detect_faces(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """
        Run the detector and return bounding boxes plus aligned crops.
        Replace the placeholder implementation with actual MTCNN inference.
        """
        # Placeholder: return no detections
        return [], []
