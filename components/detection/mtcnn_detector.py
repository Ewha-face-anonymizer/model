"""
MTCNN 모델을 감싸는 감지기 스텁입니다. 이미지 로딩/저장과 향후 얼굴 탐지 로직을 담당합니다.

Face detection utilities built around MTCNN.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from facenet_pytorch import MTCNN

ARC_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # mouth left
        [70.7299, 92.2041],   # mouth right
    ],
    dtype=np.float32,
)
ARC_OUTPUT_SIZE = (112, 112)


class FaceDetector:
    """
    Thin wrapper intended to handle MTCNN model loading, inference,
    and conversion into bounding boxes + aligned face crops.
    """

    def __init__(self, model_path: Path = None, min_size: int = 40) -> None:
        self.model_path = model_path  # MTCNN은 facenet_pytorch 라이브러리 사용, 실제 모델 파일 불필요
        self.min_size = min_size
        self.model = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=min_size,
            thresholds=[0.6, 0.7, 0.7],
            keep_all=True,
            post_process=False,
        )

    def load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to load image at {image_path}")
        return image  # BGR

    def save_image(self, image: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

    def detect_faces(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """
        Input: BGR frame
        Return: bbox, ArcFace
        -------
        bboxes: List[(x1, y1, x2, y2)]
        faces:  List[np.ndarray]  # RGB, (112,112,3)
        """
        if frame is None or frame.size == 0:
            return [], []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        boxes, probs, landmarks = self.model.detect(rgb, landmarks=True)

        if boxes is None or len(boxes) == 0:
            return [], []

        bboxes: List[Tuple[int, int, int, int]] = []
        faces: List[np.ndarray] = []

        for box, prob, lms in zip(boxes, probs, landmarks):
            if prob is None:
                continue

            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            if lms is not None and lms.shape == (5, 2):
                aligned = self._align_arcface(rgb, lms.astype(np.float32))
            else:
                crop = rgb[y1:y2, x1:x2]
                aligned = cv2.resize(crop, ARC_OUTPUT_SIZE)

            bboxes.append((x1, y1, x2, y2))
            faces.append(aligned)

        return bboxes, faces

    def _align_arcface(self, rgb_image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        src = landmarks.astype(np.float32)
        dst = ARC_TEMPLATE

        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        h, w = rgb_image.shape[:2]

        if M is None:
            x_min = int(np.clip(np.min(src[:, 0]), 0, w - 1))
            y_min = int(np.clip(np.min(src[:, 1]), 0, h - 1))
            x_max = int(np.clip(np.max(src[:, 0]), 0, w))
            y_max = int(np.clip(np.max(src[:, 1]), 0, h))

            if x_max <= x_min or y_max <= y_min:
                side = min(h, w)
                sx = (w - side) // 2
                sy = (h - side) // 2
                crop = rgb_image[sy:sy+side, sx:sx+side]
            else:
                crop = rgb_image[y_min:y_max, x_min:x_max]

            aligned = cv2.resize(crop, ARC_OUTPUT_SIZE)
        else:
            aligned = cv2.warpAffine(
                rgb_image,
                M,
                ARC_OUTPUT_SIZE,
                borderMode=cv2.BORDER_REPLICATE,
            )

        return aligned.astype(np.uint8)
