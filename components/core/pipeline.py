"""
Shared orchestration logic for the selective mosaic processing pipeline.
전체 "탐지 → 임베딩 → 기준 비교 → 모자이크"를 조합하는 파이프라인 클래스입니다. 
이미지 모드와 웹캠 모드 모두 여기서 처리합니다.

"""
from pathlib import Path
from typing import Iterable, List, Tuple
from datetime import datetime
import json

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
        
        start_time = datetime.now()
        
        # 로그 데이터 초기화
        log_data = {
            "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_image": str(input_image),
            "person_name": person_name,
            "threshold": self.config.mosaic_threshold,
            "reference_images": [str(img) for img in reference_images],
            "faces": []
        }
        
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
        
        log_data["reference_embeddings_count"] = len(self.reference_manager.get_all())
        
        # 테스트 이미지 처리
        image = self.detector.load_image(input_image)
        bboxes, faces = self.detector.detect_faces(image)
        
        log_data["total_faces_detected"] = len(faces)
        
        # 각 얼굴 임베딩 및 거리 계산
        embeddings = self.embedder.embed_all(faces)
        
        # 각 얼굴별 상세 정보 수집
        for i, (bbox, emb) in enumerate(zip(bboxes, embeddings)):
            # 모든 기준 임베딩과의 거리 계산
            distances = []
            for ref_emb in self.reference_manager.get_all():
                cosine_similarity = float(
                    np.dot(emb, ref_emb) / 
                    (np.linalg.norm(emb) * np.linalg.norm(ref_emb) + 1e-8)
                )
                distance = 1.0 - cosine_similarity
                distances.append(distance)
            
            min_distance = min(distances) if distances else float('inf')
            is_same_person = min_distance <= self.config.mosaic_threshold
            
            face_info = {
                "face_index": i,
                "bbox": [int(x) for x in bbox],
                "min_distance": float(min_distance),
                "all_distances": [float(d) for d in distances],
                "is_same_person": is_same_person,
                "will_mosaic": not is_same_person
            }
            log_data["faces"].append(face_info)
        
        keep_mask = self.matcher.match_batch(embeddings)
        mosaicked = self._mosaic_non_matches(image, bboxes, keep_mask)
        
        # 처리 시간 계산
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        log_data["processing_time_seconds"] = processing_time
        log_data["same_person_count"] = sum(1 for f in log_data["faces"] if f["is_same_person"])
        log_data["mosaicked_count"] = sum(1 for f in log_data["faces"] if f["will_mosaic"])
        
        # 파일명 생성
        if person_name:
            output_filename = f"{input_image.stem}_{person_name}_mosaic.jpg"
            log_filename = f"{input_image.stem}_{person_name}_log.json"
        else:
            output_filename = f"{input_image.stem}_mosaic.jpg"
            log_filename = f"{input_image.stem}_log.json"
        
        # 이미지 저장
        self.detector.save_image(mosaicked, output_dir / output_filename)
        
        # 로그 저장
        log_path = output_dir / log_filename
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"로그 저장: {log_path}")

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
