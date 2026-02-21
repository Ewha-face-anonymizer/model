#!/usr/bin/env python3
"""
Threshold 튜닝 스크립트
다양한 threshold 값에 대한 성능을 비교하고 기록합니다.
포스터 세션용 결과 분석 및 시각화
"""
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from components.core.pipeline import MosaicPipeline
from components.detection.mtcnn_detector import FaceDetector
from components.embedding.arcface_embedder import FaceEmbedder
from components.reference.reference_manager import ReferenceManager
from components.matching.reference_matcher import ReferenceMatcher
from config.loader import load_config


class ThresholdTuner:
    """Threshold 값에 따른 성능 비교 및 분석"""
    
    def __init__(self, config_path: Path):
        self.config = load_config(config_path)
        self.detector = FaceDetector(
            model_path=None,  # MTCNN은 facenet_pytorch에서 자동 로드
            min_size=self.config.processing.detection_min_size,
        )
        self.embedder = FaceEmbedder(model_path=Path(self.config.models.embedder))
        self.ref_manager = ReferenceManager(embedder=self.embedder)
        
        # 결과 저장용
        self.results: List[Dict] = []
        
    def compute_similarity(self, img1_path: Path, img2_path: Path) -> float:
        """두 이미지 간 코사인 거리 계산"""
        # 이미지 1 처리
        img1 = self.detector.load_image(img1_path)
        _, faces1 = self.detector.detect_faces(img1)
        if not faces1:
            raise ValueError(f"얼굴 검출 실패: {img1_path}")
        emb1 = self.embedder.embed(faces1[0])
        
        # 이미지 2 처리
        img2 = self.detector.load_image(img2_path)
        _, faces2 = self.detector.detect_faces(img2)
        if not faces2:
            raise ValueError(f"얼굴 검출 실패: {img2_path}")
        emb2 = self.embedder.embed(faces2[0])
        
        # 코사인 거리 계산
        cosine_similarity = float(
            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        )
        distance = 1.0 - cosine_similarity
        
        return distance
    
    def evaluate_threshold(
        self, 
        threshold: float,
        same_person_pairs: List[Tuple[Path, Path]],
        diff_person_pairs: List[Tuple[Path, Path]]
    ) -> Dict:
        """특정 threshold 값에 대한 성능 평가"""
        
        # 같은 사람 쌍 평가
        same_distances = []
        same_correct = 0
        for img1, img2 in same_person_pairs:
            dist = self.compute_similarity(img1, img2)
            same_distances.append(dist)
            if dist <= threshold:  # 같은 사람으로 판단 (모자이크 안 함)
                same_correct += 1
        
        # 다른 사람 쌍 평가
        diff_distances = []
        diff_correct = 0
        for img1, img2 in diff_person_pairs:
            dist = self.compute_similarity(img1, img2)
            diff_distances.append(dist)
            if dist > threshold:  # 다른 사람으로 판단 (모자이크 함)
                diff_correct += 1
        
        # 성능 지표 계산
        total_pairs = len(same_person_pairs) + len(diff_person_pairs)
        total_correct = same_correct + diff_correct
        accuracy = total_correct / total_pairs if total_pairs > 0 else 0
        
        same_accuracy = same_correct / len(same_person_pairs) if same_person_pairs else 0
        diff_accuracy = diff_correct / len(diff_person_pairs) if diff_person_pairs else 0
        
        result = {
            'threshold': threshold,
            'accuracy': accuracy,
            'same_person_accuracy': same_accuracy,
            'diff_person_accuracy': diff_accuracy,
            'same_person_avg_dist': np.mean(same_distances) if same_distances else 0,
            'diff_person_avg_dist': np.mean(diff_distances) if diff_distances else 0,
            'same_person_std': np.std(same_distances) if same_distances else 0,
            'diff_person_std': np.std(diff_distances) if diff_distances else 0,
            'same_correct': same_correct,
            'same_total': len(same_person_pairs),
            'diff_correct': diff_correct,
            'diff_total': len(diff_person_pairs)
        }
        
        return result
    
    def tune(
        self,
        same_person_pairs: List[Tuple[Path, Path]],
        diff_person_pairs: List[Tuple[Path, Path]],
        threshold_range: np.ndarray = None
    ):
        """여러 threshold 값에 대해 성능 비교"""
        
        if threshold_range is None:
            # 기본 범위: 0.2 ~ 0.8, 0.05 간격
            threshold_range = np.arange(0.2, 0.85, 0.05)
        
        print("=" * 70)
        print(f"Threshold 튜닝 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"테스트 쌍: 같은 사람 {len(same_person_pairs)}쌍, 다른 사람 {len(diff_person_pairs)}쌍")
        print("=" * 70)
        
        self.results = []
        for threshold in threshold_range:
            result = self.evaluate_threshold(threshold, same_person_pairs, diff_person_pairs)
            self.results.append(result)
            
            print(f"\nThreshold: {threshold:.2f}")
            print(f"  전체 정확도: {result['accuracy']:.2%}")
            print(f"  같은 사람 정확도: {result['same_person_accuracy']:.2%} "
                  f"({result['same_correct']}/{result['same_total']})")
            print(f"  다른 사람 정확도: {result['diff_person_accuracy']:.2%} "
                  f"({result['diff_correct']}/{result['diff_total']})")
            print(f"  평균 거리 - 같은 사람: {result['same_person_avg_dist']:.4f}, "
                  f"다른 사람: {result['diff_person_avg_dist']:.4f}")
        
        return self.results
    
    def save_results(self, output_path: Path):
        """결과를 CSV 파일로 저장"""
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"\n결과 저장 완료: {output_path}")
        return df
    
    def plot_results(self, output_path: Path):
        """결과 시각화 및 저장"""
        df = pd.DataFrame(self.results)
        
        # 1개의 그래프만 생성 (Overall Accuracy)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Threshold Tuning Results', fontsize=16, fontweight='bold')
        
        # 전체 정확도
        ax.plot(df['threshold'], df['accuracy'], marker='o', linewidth=2, color='#2E86AB')
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Overall Accuracy vs Threshold', fontsize=14)
        ax.grid(True, alpha=0.3)
        best_idx = df['accuracy'].idxmax()
        ax.axvline(df.loc[best_idx, 'threshold'], color='r', 
                  linestyle='--', alpha=0.5, 
                  label=f'Best: {df.loc[best_idx, "threshold"]:.2f}')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"그래프 저장 완료: {output_path}")
        plt.close()
    
    def print_summary(self):
        """최적 threshold 및 요약 출력"""
        df = pd.DataFrame(self.results)
        best_idx = df['accuracy'].idxmax()
        best = df.loc[best_idx]
        
        print("\n" + "=" * 70)
        print("최적 Threshold 분석 결과")
        print("=" * 70)
        print(f"최적 Threshold: {best['threshold']:.2f}")
        print(f"전체 정확도: {best['accuracy']:.2%}")
        print(f"같은 사람 정확도: {best['same_person_accuracy']:.2%}")
        print(f"다른 사람 정확도: {best['diff_person_accuracy']:.2%}")
        print(f"\n평균 코사인 거리:")
        print(f"  같은 사람: {best['same_person_avg_dist']:.4f} (±{best['same_person_std']:.4f})")
        print(f"  다른 사람: {best['diff_person_avg_dist']:.4f} (±{best['diff_person_std']:.4f})")
        print("=" * 70)


def main():
    """메인 실행 함수"""
    
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description='이미지 쌍 기반 Threshold 튜닝')
    parser.add_argument('--person-dir', '-p', type=str, required=True,
                       help='한 사람의 사진들이 있는 폴더 (예: data/input/Dataset/person00)')
    parser.add_argument('--min-threshold', type=float, default=0.2,
                       help='최소 threshold 값 (기본: 0.2)')
    parser.add_argument('--max-threshold', type=float, default=0.85,
                       help='최대 threshold 값 (기본: 0.85)')
    parser.add_argument('--step', type=float, default=0.05,
                       help='threshold 증가 간격 (기본: 0.05)')
    
    args = parser.parse_args()
    
    # 설정 로드
    config_path = Path("config/pipeline.yaml")
    tuner = ThresholdTuner(config_path)
    
    # 폴더에서 이미지 파일 찾기 (ref_XX.jpg 패턴)
    person_dir = Path(args.person_dir)
    if not person_dir.exists():
        print(f"오류: 폴더를 찾을 수 없습니다: {person_dir}")
        return
    
    # ref_XX.jpg 패턴의 파일만 선택
    image_files = sorted(list(person_dir.glob("ref_*.jpg")))
    if len(image_files) < 2:
        print(f"오류: 최소 2개 이상의 ref_XX.jpg 이미지가 필요합니다. 현재: {len(image_files)}개")
        return
    
    # 같은 사람 이미지 쌍 생성 (모든 조합)
    same_person_pairs = []
    for i in range(len(image_files)):
        for j in range(i+1, len(image_files)):
            same_person_pairs.append((image_files[i], image_files[j]))
    
    print(f"같은 사람 이미지 쌍: {len(same_person_pairs)}개 생성")
    
    # 다른 사람 쌍은 비어있음 (선택사항)
    diff_person_pairs = []
    
    # Threshold 범위 설정
    threshold_range = np.arange(args.min_threshold, args.max_threshold, args.step)
    
    # 튜닝 실행
    results = tuner.tune(same_person_pairs, diff_person_pairs, threshold_range)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    person_name = person_dir.name
    output_dir = Path(f"data/output/threshold_tuning_{person_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tuner.save_results(output_dir / f"results_{timestamp}.csv")
    tuner.plot_results(output_dir / f"plot_{timestamp}.png")
    tuner.print_summary()


if __name__ == "__main__":
    main()
