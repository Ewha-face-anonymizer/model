"""
OpenCV VideoCapture를 감싸 프레임을 yield하고, 창 표시 및 종료 키 처리까지 해줍니다.
Abstraction over OpenCV VideoCapture for webcam streaming.
"""
from typing import Iterator

import cv2
import numpy as np


class VideoStream:
    """Context manager yielding frames and handling window display."""

    def __init__(self, webcam_id: int) -> None:
        self.webcam_id = webcam_id
        self.capture = None

    def __enter__(self) -> "VideoStream":
        self.capture = cv2.VideoCapture(self.webcam_id)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open webcam {self.webcam_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        if self.capture is None:
            raise StopIteration
        success, frame = self.capture.read()
        if not success:
            raise StopIteration
        return frame

    def display_frame(self, frame: np.ndarray) -> None:
        cv2.imshow("Selective Mosaic", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise StopIteration
