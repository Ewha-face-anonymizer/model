"""
특정 박스 영역에 모자이크(픽셀화)를 적용합니다.
Image post-processing helpers responsible for applying mosaic effects.
"""
from typing import Iterable, Tuple

import cv2
import numpy as np


class MosaicProcessor:
    """Applies a simple pixelation effect to specified bounding boxes."""

    def __init__(self, kernel_size: int) -> None:
        self.kernel_size = kernel_size

    def apply(
        self,
        frame: np.ndarray,
        bboxes: Iterable[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        output = frame.copy()
        for x1, y1, x2, y2 in bboxes:
            roi = output[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi = cv2.resize(
                roi,
                (self.kernel_size, self.kernel_size),
                interpolation=cv2.INTER_LINEAR,
            )
            roi = cv2.resize(
                roi,
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST,
            )
            output[y1:y2, x1:x2] = roi
        return output
