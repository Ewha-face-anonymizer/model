#!/usr/bin/env python3
"""
전체 50명의 threshold별 성능 분포 분석
0.45 ~ 0.57 범위 (0.02 간격)
"""
import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict

def main():
    output_dir = Path('data/output')
    
    # Threshold별 최초 탐지 인원 추적
    first_detection = {}
    threshold_success = defaultdict(int)  # 각 threshold에서 성공한 총 인원
    
    thresholds = [0.45, 0.47, 0.49, 0.51, 0.53, 0.55, 0.57]
    
    for person_dir in sorted(output_dir.glob('threshold_tuning_group_person*')):
        person_name = person_dir.name.replace('threshold_tuning_group_', '')
        summary_file = person_dir / 'summary.csv'
        
        if not summary_file.exists():
            continue
            
        df = pd.read_csv(summary_file)
        
        # 이 person이 처음으로 탐지된 threshold 찾기
        for th in thresholds:
            th_row = df[np.isclose(df['threshold'], th, atol=0.001)]
            
            if len(th_row) > 0:
                same_count = int(th_row.iloc[0]['same_person_count'])
                
                if same_count > 0:
                    threshold_success[th] += 1
                    
                    if person_name not in first_detection:
                        first_detection[person_name] = th
                    break
    
    print("=" * 80)
    print("전체 50명 Threshold별 성능 분포 분석")
    print("=" * 80)
    print()
    
    print(f"분석 대상: {len(first_detection)}명")
    print(f"탐지 실패: {50 - len(first_detection)}명")
    print()
    
    # 최초 탐지 threshold 분포
    first_detection_count = defaultdict(int)
    for th in first_detection.values():
        first_detection_count[th] += 1
    
    print("📊 각 Threshold에서 최초 탐지된 인원")
    print("-" * 80)
    print(f"{'Threshold':<12} {'최초 탐지':<12} {'비율':<12} {'누적 성공률':<12}")
    print("-" * 80)
    
    total_detected = len(first_detection)
    cumulative = 0
    
    for th in thresholds:
        count = first_detection_count.get(th, 0)
        cumulative += count
        ratio = (count / total_detected * 100) if total_detected > 0 else 0
        cumulative_rate = (cumulative / 50 * 100)
        
        print(f"{th:<12.2f} {count:<12}명 {ratio:<11.1f}% {cumulative_rate:<11.1f}%")
    
    print("-" * 80)
    print()
    
    # Threshold별 실제 탐지 성공률
    print("📈 Threshold별 실제 탐지 성공률")
    print("-" * 80)
    print(f"{'Threshold':<12} {'탐지 성공':<12} {'성공률':<12}")
    print("-" * 80)
    
    for th in thresholds:
        success = threshold_success[th]
        rate = (success / 50 * 100)
        print(f"{th:<12.2f} {success:<12}명 {rate:<11.1f}%")
    
    print("-" * 80)
    print()
    
    # 통계 요약
    if first_detection:
        threshold_values = list(first_detection.values())
        print("📋 통계 요약")
        print("-" * 80)
        print(f"평균 최적 threshold: {np.mean(threshold_values):.3f}")
        print(f"중앙값 최적 threshold: {np.median(threshold_values):.3f}")
        
        # 최빈값
        mode_th = max(first_detection_count.items(), key=lambda x: x[1])
        print(f"최빈값: {mode_th[0]:.2f} ({mode_th[1]}명, {mode_th[1]/total_detected*100:.1f}%)")
        print(f"표준편차: {np.std(threshold_values):.3f}")
        print()
    
    # 탐지 실패한 인원 리스트
    all_persons = set(f"person{i:02d}" for i in range(1, 51))
    detected_persons = set(first_detection.keys())
    failed_persons = sorted(all_persons - detected_persons)
    
    if failed_persons:
        print(f"❌ 모든 threshold에서 탐지 실패: {len(failed_persons)}명")
        print("-" * 80)
        print(", ".join(failed_persons))
        print()

if __name__ == "__main__":
    main()
