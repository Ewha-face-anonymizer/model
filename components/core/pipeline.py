"""
Shared orchestration logic for the selective mosaic processing pipeline.
전체 “탐지 → 임베딩 → 기준 비교 → 모자이크”를 조합하는 파이프라인 클래스입니다. 
이미지 모드와 웹캠 모드 모두 여기서 처리합니다.

"""
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from config.loader import AppConfig
from components.detection.mtcnn_detector import FaceDetector
from components.embedding.arcface_embedder import FaceEmbedder
from components.matching.reference_matcher import ReferenceMatcher
from components.processing.mosaic_processor import MosaicProcessor
from components.reference.reference_manager import ReferenceManager
from components.streaming.video_stream import VideoStream


class MosaicPipeline:
    """Coordinates detection, embedding, matching, and mosaic operations."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.detector = FaceDetector(
            model_path=Path(config.models.detector),
            min_size=config.processing.detection_min_size,
        )
        self.embedder = FaceEmbedder(model_path=Path(config.models.embedder))
        self.matcher = ReferenceMatcher(threshold=config.mosaic_threshold)
        self.mosaicer = MosaicProcessor(kernel_size=config.processing.mosaic_kernel)
        self.reference_manager = ReferenceManager(
            embedder=self.embedder,
        )

    def process_image(
        self,
        input_image: Path,
        reference_image: Path,
        output_dir: Path,
    ) -> None:
        """Run the pipeline on a static image path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        image = self.detector.load_image(input_image)
        ref_embedding = self.reference_manager.get_embedding(reference_image)
        bboxes, faces = self.detector.detect_faces(image)
        embeddings = self.embedder.embed_all(faces)
        keep_mask = self.matcher.match_batch(embeddings, ref_embedding)
        mosaicked = self._mosaic_non_matches(image, bboxes, keep_mask)
        self.detector.save_image(mosaicked, output_dir / f"{input_image.stem}_mosaic.jpg")

    def process_stream(
        self,
        reference_image: Path,
        webcam_id: int,
    ) -> None:
        """Process frames from a webcam indefinitely."""
        ref_embedding = self.reference_manager.get_embedding(reference_image)
        with VideoStream(webcam_id) as stream:
            for frame in stream:
                bboxes, faces = self.detector.detect_faces(frame)
                embeddings = self.embedder.embed_all(faces)
                keep_mask = self.matcher.match_batch(embeddings, ref_embedding)
                mosaicked = self._mosaic_non_matches(frame, bboxes, keep_mask)
                stream.display_frame(mosaicked)

    def _mosaic_non_matches(
        self,
        frame: np.ndarray,
        bboxes: Iterable[Tuple[int, int, int, int]],
        keep_mask: List[bool],
    ) -> np.ndarray:
        """Apply mosaic to bounding boxes whose mask entry is False."""
        to_mosaic = [bbox for bbox, keep in zip(bboxes, keep_mask) if not keep]
        return self.mosaicer.apply(frame, to_mosaic)
