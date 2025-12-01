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
            model_path=None,  # MTCNN은 facenet_pytorch에서 자동 로드
            min_size=config.processing.detection_min_size,
        )
        self.embedder = FaceEmbedder(model_path=Path(config.models.embedder))
        self.reference_manager = ReferenceManager(embedder=self.embedder)
        self.matcher = ReferenceMatcher(
            threshold=config.mosaic_threshold,
            ref_manager=self.reference_manager
        )
        self.mosaicer = MosaicProcessor(kernel_size=config.processing.mosaic_kernel)

    def process_image(
        self,
        input_image: Path,
        reference_images: List[Path],
        output_dir: Path,
        person_name: str = None,
    ) -> None:
        """Run the pipeline on a static image path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 여러 기준 이미지에서 얼굴 임베딩 생성
        self.reference_manager.references = []  # 초기화
        for ref_img_path in reference_images:
            ref_img = self.detector.load_image(ref_img_path)
            _, ref_faces = self.detector.detect_faces(ref_img)
            if ref_faces:
                self.reference_manager.add(ref_faces[0])
            else:
                print(f"경고: 얼굴 검출 실패 (건너뜀): {ref_img_path}")
        
        if not self.reference_manager.get_all():
            print("오류: 모든 기준 이미지에서 얼굴 검출 실패")
            return
        
        # 테스트 이미지 처리
        image = self.detector.load_image(input_image)
        bboxes, faces = self.detector.detect_faces(image)
        embeddings = self.embedder.embed_all(faces)
        keep_mask = self.matcher.match_batch(embeddings)
        mosaicked = self._mosaic_non_matches(image, bboxes, keep_mask)
        
        # 파일명 생성
        if person_name:
            output_filename = f"{input_image.stem}_{person_name}_mosaic.jpg"
        else:
            output_filename = f"{input_image.stem}_mosaic.jpg"
        
        self.detector.save_image(mosaicked, output_dir / output_filename)

    def process_stream(
        self,
        reference_images: List[Path],
        webcam_id: int,
    ) -> None:
        """Process frames from a webcam indefinitely."""
        # 여러 기준 이미지에서 얼굴 임베딩 생성
        self.reference_manager.references = []  # 초기화
        for ref_img_path in reference_images:
            ref_img = self.detector.load_image(ref_img_path)
            _, ref_faces = self.detector.detect_faces(ref_img)
            if ref_faces:
                self.reference_manager.add(ref_faces[0])
            else:
                print(f"경고: 얼굴 검출 실패 (건너뜀): {ref_img_path}")
        
        if not self.reference_manager.get_all():
            print("오류: 모든 기준 이미지에서 얼굴 검출 실패")
            return
        
        with VideoStream(webcam_id) as stream:
            for frame in stream:
                bboxes, faces = self.detector.detect_faces(frame)
                embeddings = self.embedder.embed_all(faces)
                keep_mask = self.matcher.match_batch(embeddings)
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
