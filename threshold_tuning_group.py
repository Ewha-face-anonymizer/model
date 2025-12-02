#!/usr/bin/env python3
"""
단체사진 기반 Threshold 튜닝 스크립트
단체사진 + 기준인물 사진으로 여러 threshold 값을 테스트하고 결과를 비교합니다.
포스터 세션용 결과 분석 및 시각화
"""
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import cv2
from datetime import datetime
import argparse

from components.detection.mtcnn_detector import FaceDetector
from components.embedding.arcface_embedder import FaceEmbedder
from components.reference.reference_manager import ReferenceManager
from components.matching.reference_matcher import ReferenceMatcher
from components.processing.mosaic_processor import MosaicProcessor
from config.loader import load_config


class GroupPhotoThresholdTuner:
    """단체사진 기반 Threshold 튜닝"""
    
    def __init__(self, config_path: Path):
        self.config = load_config(config_path)
        self.detector = FaceDetector(
            model_path=None,  # MTCNN은 facenet_pytorch에서 자동 로드
            min_size=self.config.processing.detection_min_size,
        )
        self.embedder = FaceEmbedder(model_path=Path(self.config.models.embedder))
        self.mosaicer = MosaicProcessor(kernel_size=self.config.processing.mosaic_kernel)
        
        # 결과 저장용
        self.results: List[Dict] = []
        
    def analyze_group_photo(
        self, 
        group_photo_path: Path, 
        reference_photo_paths: List[Path],
        threshold: float
    ) -> Dict:
        """단체사진 분석 및 각 얼굴과 기준인물 간 거리 측정"""
        
        # 기준 인물 임베딩 생성 (여러 장의 사진 사용)
        ref_manager = ReferenceManager(embedder=self.embedder)
        ref_embeddings = []
        
        for ref_path in reference_photo_paths:
            ref_img = self.detector.load_image(ref_path)
            _, ref_faces = self.detector.detect_faces(ref_img)
            if not ref_faces:
                print(f"경고: 얼굴 검출 실패 (건너뜀): {ref_path}")
                continue
            ref_embedding = self.embedder.embed(ref_faces[0])
            ref_embeddings.append(ref_embedding)
        
        if not ref_embeddings:
            raise ValueError("모든 기준 이미지에서 얼굴 검출에 실패했습니다.")
        
        ref_manager.references = ref_embeddings
        
        # 단체사진에서 얼굴 검출 및 임베딩
        group_img = self.detector.load_image(group_photo_path)
        bboxes, faces = self.detector.detect_faces(group_img)
        
        if not faces:
            raise ValueError(f"단체사진에서 얼굴 검출 실패: {group_photo_path}")
        
        # 각 얼굴과 기준인물 간 거리 계산
        face_distances = []
        for i, face in enumerate(faces):
            emb = self.embedder.embed(face)
            
            # 모든 기준 임베딩과 비교하여 최소 거리 사용
            min_distance = float('inf')
            for ref_embedding in ref_embeddings:
                # 코사인 거리 계산
                cosine_similarity = float(
                    np.dot(emb, ref_embedding) / 
                    (np.linalg.norm(emb) * np.linalg.norm(ref_embedding) + 1e-8)
                )
                distance = 1.0 - cosine_similarity
                min_distance = min(min_distance, distance)
            
            # threshold 기준 판단
            is_same_person = min_distance <= threshold
            
            face_distances.append({
                'face_index': i,
                'bbox': bboxes[i],
                'distance': min_distance,
                'is_same_person': is_same_person,
                'will_mosaic': not is_same_person
            })
        
        result = {
            'threshold': threshold,
            'total_faces': len(faces),
            'same_person_count': sum(1 for f in face_distances if f['is_same_person']),
            'diff_person_count': sum(1 for f in face_distances if not f['is_same_person']),
            'face_details': face_distances,
            'avg_distance': np.mean([f['distance'] for f in face_distances]),
            'min_distance': min([f['distance'] for f in face_distances]),
            'max_distance': max([f['distance'] for f in face_distances])
        }
        
        return result, group_img, bboxes
    
    def generate_result_image(
        self,
        group_img: np.ndarray,
        bboxes: List,
        face_details: List[Dict],
        threshold: float,
        output_path: Path
    ):
        """threshold 적용 결과 이미지 생성 (모자이크 + 거리값 표시)"""
        result_img = group_img.copy()
        
        # 모자이크 적용할 얼굴 리스트
        mosaic_bboxes = [
            face['bbox'] for face in face_details if face['will_mosaic']
        ]
        
        # 모자이크 적용
        if mosaic_bboxes:
            result_img = self.mosaicer.apply(result_img, mosaic_bboxes)
        
        # 각 얼굴에 바운딩 박스 및 거리값 표시
        for face in face_details:
            x1, y1, x2, y2 = face['bbox']
            distance = face['distance']
            is_same = face['is_same_person']
            
            # 바운딩 박스 색상 (같은 사람: 초록, 다른 사람: 빨강)
            color = (0, 255, 0) if is_same else (0, 0, 255)
            
            # 바운딩 박스 그리기
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            # 거리값 표시
            text = f"{distance:.3f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1
            text_y = y1 - 5 if y1 - 5 > 10 else y1 + 15
            
            # 텍스트 배경
            cv2.rectangle(result_img, 
                         (text_x, text_y - text_size[1] - 2),
                         (text_x + text_size[0], text_y + 2),
                         color, -1)
            cv2.putText(result_img, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # threshold 정보 표시
        info_text = f"Threshold: {threshold:.2f} | Faces: {len(face_details)} | Mosaicked: {sum(1 for f in face_details if f['will_mosaic'])}"
        cv2.putText(result_img, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # 저장
        cv2.imwrite(str(output_path), result_img)
        
    def tune(
        self,
        group_photo_path: Path,
        reference_photo_paths: List[Path],
        threshold_range: np.ndarray = None,
        output_dir: Path = None,
        person_name: str = None
    ):
        """여러 threshold 값에 대해 단체사진 분석 및 결과 이미지 생성"""
        
        if threshold_range is None:
            threshold_range = np.arange(0.2, 0.85, 0.05)
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if person_name:
                output_dir = Path(f"data/output/threshold_tuning_group_{person_name}_{timestamp}")
            else:
                output_dir = Path(f"data/output/threshold_tuning_group_{timestamp}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print(f"단체사진 Threshold 튜닝 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"단체사진: {group_photo_path}")
        print(f"기준인물 사진: {len(reference_photo_paths)}장")
        for i, ref_path in enumerate(reference_photo_paths, 1):
            print(f"  {i}. {ref_path}")
        print(f"Threshold 범위: {threshold_range[0]:.2f} ~ {threshold_range[-1]:.2f}")
        print("=" * 70)
        
        self.results = []
        
        for threshold in threshold_range:
            print(f"\n{'='*70}")
            print(f"Threshold: {threshold:.2f}")
            print(f"{'='*70}")
            
            # 분석 수행
            result, group_img, bboxes = self.analyze_group_photo(
                group_photo_path, reference_photo_paths, threshold
            )
            
            # 결과 출력
            print(f"전체 얼굴: {result['total_faces']}개")
            print(f"기준인물과 동일: {result['same_person_count']}개 (모자이크 안함)")
            print(f"기준인물과 다름: {result['diff_person_count']}개 (모자이크 함)")
            print(f"거리 범위: {result['min_distance']:.4f} ~ {result['max_distance']:.4f}")
            print(f"\n각 얼굴 상세:")
            for face in result['face_details']:
                status = "SAME (보존)" if face['is_same_person'] else "DIFF (모자이크)"
                print(f"  얼굴 #{face['face_index']}: 거리={face['distance']:.4f} -> {status}")
            
            # 결과 이미지 생성
            output_image_path = output_dir / f"result_th{threshold:.2f}.jpg"
            self.generate_result_image(
                group_img, bboxes, result['face_details'], threshold, output_image_path
            )
            print(f"결과 이미지 저장: {output_image_path}")
            
            self.results.append(result)
        
        return self.results
    
    def save_summary(self, output_path: Path):
        """전체 결과 요약을 CSV로 저장"""
        summary_data = []
        
        for result in self.results:
            summary_data.append({
                'threshold': result['threshold'],
                'total_faces': result['total_faces'],
                'same_person_count': result['same_person_count'],
                'diff_person_count': result['diff_person_count'],
                'avg_distance': result['avg_distance'],
                'min_distance': result['min_distance'],
                'max_distance': result['max_distance']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        print(f"\n요약 결과 저장: {output_path}")
        return df
    
    def save_detailed_results(self, output_path: Path):
        """각 얼굴별 상세 결과를 CSV로 저장"""
        detailed_data = []
        
        for result in self.results:
            threshold = result['threshold']
            for face in result['face_details']:
                detailed_data.append({
                    'threshold': threshold,
                    'face_index': face['face_index'],
                    'distance': face['distance'],
                    'is_same_person': face['is_same_person'],
                    'will_mosaic': face['will_mosaic']
                })
        
        df = pd.DataFrame(detailed_data)
        df.to_csv(output_path, index=False)
        print(f"상세 결과 저장: {output_path}")
        return df
    
    def print_recommendation(self):
        """최적 threshold 추천"""
        print("\n" + "=" * 70)
        print("Threshold 추천")
        print("=" * 70)
        
        # 각 threshold에서 모자이크된 얼굴 수 확인
        for result in self.results:
            print(f"Threshold {result['threshold']:.2f}: "
                  f"모자이크 {result['diff_person_count']}개 / "
                  f"보존 {result['same_person_count']}개")
        
        print("\n권장사항:")
        print("- 결과 이미지들을 확인하여 원하는 결과에 가장 가까운 threshold를 선택하세요.")
        print("- 거리값이 낮을수록 기준인물과 유사합니다.")
        print("- 초록 박스: 기준인물(보존), 빨강 박스: 다른 사람(모자이크)")
        print("=" * 70)


def main():
    """메인 실행 함수"""
    
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description='단체사진 기반 Threshold 튜닝')
    parser.add_argument('--group', '-g', type=str, required=True,
                       help='단체사진 경로 (예: data/input/단체사진.jpg)')
    parser.add_argument('--reference', '-r', type=str, required=True,
                       help='기준인물 사진 폴더 또는 파일 (예: data/input/베일리 또는 data/input/베일리/베일리_1.jpg)')
    parser.add_argument('--min-threshold', type=float, default=0.2,
                       help='최소 threshold 값 (기본: 0.2)')
    parser.add_argument('--max-threshold', type=float, default=0.85,
                       help='최대 threshold 값 (기본: 0.85)')
    parser.add_argument('--step', type=float, default=0.05,
                       help='threshold 증가 간격 (기본: 0.05)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='출력 디렉토리 (기본: data/output/threshold_tuning_TIMESTAMP)')
    
    args = parser.parse_args()
    
    # 설정
    config_path = Path("config/pipeline.yaml")
    tuner = GroupPhotoThresholdTuner(config_path)
    
    # 입력 이미지 설정
    group_photo = Path(args.group)
    reference_path = Path(args.reference)
    
    if not group_photo.exists():
        print(f"오류: 단체사진 파일을 찾을 수 없습니다: {group_photo}")
        return
    
    # 기준인물 사진 찾기 (폴더 또는 단일 파일)
    reference_photos = []
    if reference_path.is_dir():
        # 폴더인 경우: 모든 이미지 파일 찾기
        reference_photos = sorted(list(reference_path.glob("*.jpg")) + 
                                 list(reference_path.glob("*.png")) +
                                 list(reference_path.glob("*.jpeg")))
        if not reference_photos:
            print(f"오류: 폴더에서 이미지를 찾을 수 없습니다: {reference_path}")
            return
    elif reference_path.exists():
        # 단일 파일인 경우
        reference_photos = [reference_path]
    else:
        print(f"오류: 기준인물 사진 파일/폴더를 찾을 수 없습니다: {reference_path}")
        return
    
    # Threshold 범위 설정
    threshold_range = np.arange(args.min_threshold, args.max_threshold, args.step)
    
    # 출력 디렉토리 및 기준인물 이름 추출
    if args.output:
        output_dir = Path(args.output)
        person_name = None
    else:
        output_dir = None
        # 기준인물 이름 추출 (폴더명 또는 파일명에서)
        if reference_path.is_dir():
            person_name = reference_path.name
        else:
            person_name = reference_path.parent.name
    
    # 튜닝 실행
    threshold_range = np.arange(args.min_threshold, args.max_threshold + args.step, args.step)
    results = tuner.tune(group_photo, reference_photos, threshold_range, output_dir, person_name)
    
    # 결과 저장
    tuner.save_summary(output_dir / "summary.csv")
    tuner.save_detailed_results(output_dir / "details.csv")
    tuner.print_recommendation()
    
    print(f"\n모든 결과는 {output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()
