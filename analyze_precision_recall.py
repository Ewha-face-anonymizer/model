#!/usr/bin/env python3
"""
전체 person에 대한 동일인물 거리 + 단체사진 결과 + 정밀도/재현율 통합 분석
"""
import pandas as pd
from pathlib import Path
import numpy as np
import json

def load_same_person_distances():
    """threshold_tuning 결과에서 동일인물 간 거리 로드"""
    output_dir = Path('data/output')
    results = {}
    
    for person_dir in sorted(output_dir.glob('threshold_tuning_person*')):
        person_name = person_dir.name.replace('threshold_tuning_', '')
        
        # 가장 최근 results CSV 파일 찾기
        csv_files = list(person_dir.glob('results_*.csv'))
        if csv_files:
            latest_csv = sorted(csv_files)[-1]
            df = pd.read_csv(latest_csv)
            
            # 동일인물 평균 거리와 표준편차
            same_avg = df['same_person_avg_dist'].iloc[0]
            same_std = df['same_person_std'].iloc[0]
            
            results[person_name] = {
                'same_avg': same_avg,
                'same_std': same_std
            }
    
    return results

def load_group_results():
    """threshold_tuning_group 결과에서 단체사진 분석 로드"""
    output_dir = Path('data/output')
    results = {}
    
    for person_dir in sorted(output_dir.glob('threshold_tuning_group_person*')):
        person_name = person_dir.name.replace('threshold_tuning_group_', '')
        
        # summary.csv 로드
        summary_file = person_dir / 'summary.csv'
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            
            # threshold 0.45 결과 찾기
            th_045 = df[df['threshold'] == 0.45]
            if len(th_045) > 0:
                row = th_045.iloc[0]
                results[person_name] = {
                    'threshold': 0.45,
                    'total_faces': int(row['total_faces']),
                    'same_person_count': int(row['same_person_count']),
                    'min_distance': float(row['min_distance']),
                    'detected': int(row['same_person_count']) > 0
                }
    
    return results

def calculate_precision_recall(all_results, threshold=0.45):
    """
    정밀도/재현율 계산
    
    TP (True Positive): 기준인물을 기준인물로 인식 (same_person_count > 0)
    FP (False Positive): 타인을 기준인물로 인식 (매우 드물지만 발생 가능)
    FN (False Negative): 기준인물을 놓침 (same_person_count = 0)
    TN (True Negative): 타인을 타인으로 인식 (정확히 측정하려면 모든 얼굴 검증 필요)
    
    단체사진 실험에서:
    - TP: 기준인물 탐지 성공 (same_person_count > 0)
    - FN: 기준인물 탐지 실패 (same_person_count = 0)
    - FP: 타인을 기준인물로 오인 (total_faces - same_person_count 중 실제 타인이 보존된 경우)
    
    정밀도 = TP / (TP + FP) ≈ 탐지된 얼굴 중 진짜 본인 비율
    재현율 = TP / (TP + FN) = 전체 기준인물 중 탐지 성공 비율
    """
    
    tp = 0  # 기준인물 탐지 성공
    fn = 0  # 기준인물 탐지 실패
    fp_estimate = 0  # 타인 오인 (추정)
    
    detected_persons = []
    failed_persons = []
    
    for person, data in all_results.items():
        if 'detected' in data:
            if data['detected']:
                tp += 1
                detected_persons.append(person)
                
                # FP 추정: same_person_count가 2 이상이면 타인 오인 가능성
                # (단, 실제로는 각도가 다른 같은 사람일 수도 있음)
                if data['same_person_count'] > 1:
                    fp_estimate += (data['same_person_count'] - 1)
            else:
                fn += 1
                failed_persons.append(person)
    
    # 정밀도: 보존한 얼굴 중 실제 본인 비율 (FP가 거의 없다고 가정하면 높음)
    precision = tp / (tp + fp_estimate) if (tp + fp_estimate) > 0 else 0
    
    # 재현율: 전체 본인 중 찾은 비율
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fn': fn,
        'fp_estimate': fp_estimate,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'detected_persons': detected_persons,
        'failed_persons': failed_persons
    }

def main():
    print("=" * 80)
    print("전체 Person 통합 분석: 동일인물 거리 + 단체사진 결과 + 정밀도/재현율")
    print("=" * 80)
    print()
    
    # 1. 동일인물 거리 로드
    print("📊 동일인물 간 거리 분석 로드 중...")
    same_distances = load_same_person_distances()
    print(f"   로드 완료: {len(same_distances)}명")
    print()
    
    # 2. 단체사진 결과 로드
    print("📸 단체사진 분석 결과 로드 중...")
    group_results = load_group_results()
    print(f"   로드 완료: {len(group_results)}명")
    print()
    
    # 3. 데이터 통합
    all_results = {}
    for person in sorted(set(same_distances.keys()) | set(group_results.keys())):
        all_results[person] = {}
        
        if person in same_distances:
            all_results[person].update(same_distances[person])
        
        if person in group_results:
            all_results[person].update(group_results[person])
    
    print("=" * 80)
    print(f"통합 분석 결과 (총 {len(all_results)}명, Threshold = 0.45)")
    print("=" * 80)
    print()
    print(f"{'Person':<12} {'동일인물 평균':>12} {'동일인물 std':>12} {'그룹 최소거리':>12} "
          f"{'탐지 얼굴수':>10} {'전체 얼굴':>10} {'탐지 성공':>10}")
    print("-" * 80)
    
    # 통계 수집
    same_avgs = []
    group_mins = []
    success_count = 0
    fail_count = 0
    
    for person, data in all_results.items():
        same_avg = data.get('same_avg', 0)
        same_std = data.get('same_std', 0)
        group_min = data.get('min_distance', 0)
        same_count = data.get('same_person_count', 0)
        total_faces = data.get('total_faces', 0)
        detected = data.get('detected', False)
        
        if same_avg > 0:
            same_avgs.append(same_avg)
        if group_min > 0:
            group_mins.append(group_min)
        
        if detected:
            success_count += 1
            status = "✅ 성공"
        else:
            fail_count += 1
            status = "❌ 실패"
        
        print(f"{person:<12} {same_avg:>12.4f} {same_std:>12.4f} {group_min:>12.4f} "
              f"{same_count:>10} {total_faces:>10} {status:>10}")
    
    print("=" * 80)
    print()
    
    # 4. 통계 요약
    print("📈 통계 요약")
    print("=" * 80)
    print()
    
    print("🔹 동일인물 간 거리 (레퍼런스 3장 조합)")
    if same_avgs:
        print(f"   평균: {np.mean(same_avgs):.4f}")
        print(f"   중앙값: {np.median(same_avgs):.4f}")
        print(f"   표준편차: {np.std(same_avgs):.4f}")
        print(f"   최소: {np.min(same_avgs):.4f}")
        print(f"   최대: {np.max(same_avgs):.4f}")
    print()
    
    print("🔹 단체사진 최소 거리 (레퍼런스와 가장 가까운 얼굴)")
    if group_mins:
        print(f"   평균: {np.mean(group_mins):.4f}")
        print(f"   중앙값: {np.median(group_mins):.4f}")
        print(f"   표준편차: {np.std(group_mins):.4f}")
        print(f"   최소: {np.min(group_mins):.4f}")
        print(f"   최대: {np.max(group_mins):.4f}")
    print()
    
    print("🔹 Threshold 0.45 성능")
    total = success_count + fail_count
    print(f"   탐지 성공: {success_count}명 ({success_count/total*100:.1f}%)")
    print(f"   탐지 실패: {fail_count}명 ({fail_count/total*100:.1f}%)")
    print()
    
    # 5. 정밀도/재현율 계산
    print("=" * 80)
    print("🎯 정밀도/재현율 분석 (Threshold = 0.45)")
    print("=" * 80)
    print()
    
    metrics = calculate_precision_recall(all_results, threshold=0.45)
    
    print("📊 혼동 행렬 (Confusion Matrix)")
    print(f"   TP (True Positive - 기준인물 탐지 성공): {metrics['tp']}명")
    print(f"   FN (False Negative - 기준인물 탐지 실패): {metrics['fn']}명")
    print(f"   FP (False Positive - 타인 오인 추정): {metrics['fp_estimate']}건")
    print()
    
    print("📈 성능 지표")
    print(f"   정밀도 (Precision): {metrics['precision']:.2%}")
    print(f"      → 보존한 얼굴 중 실제 본인 비율")
    print(f"   재현율 (Recall): {metrics['recall']:.2%}")
    print(f"      → 전체 본인 중 찾은 비율")
    print(f"   F1 Score: {metrics['f1_score']:.2%}")
    print(f"      → 정밀도와 재현율의 조화 평균")
    print()
    
    print("✅ 탐지 성공 목록:")
    for i, person in enumerate(metrics['detected_persons'], 1):
        print(f"   {i}. {person}")
    print()
    
    print("❌ 탐지 실패 목록:")
    for i, person in enumerate(metrics['failed_persons'], 1):
        print(f"   {i}. {person}")
    print()
    
    # 6. CSV/JSON 저장
    output_dir = Path('data/output')
    
    # CSV 저장
    df = pd.DataFrame([
        {
            'person': person,
            'same_person_avg_distance': data.get('same_avg', 0),
            'same_person_std': data.get('same_std', 0),
            'group_min_distance': data.get('min_distance', 0),
            'detected_faces_count': data.get('same_person_count', 0),
            'total_faces': data.get('total_faces', 0),
            'detected_success': data.get('detected', False)
        }
        for person, data in all_results.items()
    ])
    csv_path = output_dir / 'precision_recall_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"💾 CSV 저장: {csv_path}")
    
    # JSON 저장 (통계 포함)
    summary = {
        'threshold': 0.45,
        'total_persons': len(all_results),
        'statistics': {
            'same_person_distance': {
                'mean': float(np.mean(same_avgs)) if same_avgs else 0,
                'median': float(np.median(same_avgs)) if same_avgs else 0,
                'std': float(np.std(same_avgs)) if same_avgs else 0,
                'min': float(np.min(same_avgs)) if same_avgs else 0,
                'max': float(np.max(same_avgs)) if same_avgs else 0
            },
            'group_min_distance': {
                'mean': float(np.mean(group_mins)) if group_mins else 0,
                'median': float(np.median(group_mins)) if group_mins else 0,
                'std': float(np.std(group_mins)) if group_mins else 0,
                'min': float(np.min(group_mins)) if group_mins else 0,
                'max': float(np.max(group_mins)) if group_mins else 0
            }
        },
        'performance': {
            'detection_success': success_count,
            'detection_failure': fail_count,
            'success_rate': success_count / total if total > 0 else 0
        },
        'precision_recall': {
            'true_positive': metrics['tp'],
            'false_negative': metrics['fn'],
            'false_positive_estimate': metrics['fp_estimate'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        },
        'detected_persons': metrics['detected_persons'],
        'failed_persons': metrics['failed_persons']
    }
    
    json_path = output_dir / 'precision_recall_analysis.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"💾 JSON 저장: {json_path}")
    print()
    
    print("=" * 80)
    print("✅ 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()
