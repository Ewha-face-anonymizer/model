#!/bin/bash

# threshold_tuning.py를 person00~50에 대해 실행
# 같은 사람 이미지 간 거리 분석 (0.45~0.55, 0.02 간격)

for i in $(seq -f "%02g" 0 50); do
    person_dir="data/input/Dataset/person$i"
    
    # 폴더 존재 확인
    if [ ! -d "$person_dir" ]; then
        echo "⏭️  person$i: 폴더 없음"
        continue
    fi
    
    # ref_*.jpg 파일 개수 확인
    ref_count=$(ls -1 "$person_dir"/ref_*.jpg 2>/dev/null | wc -l)
    if [ "$ref_count" -lt 2 ]; then
        echo "⏭️  person$i: ref 이미지 부족 ($ref_count개)"
        continue
    fi
    
    echo "=========================================="
    echo "처리 중: person$i ($ref_count개 ref 이미지)"
    echo "=========================================="
    
    python3 threshold_tuning.py \
        --person-dir "$person_dir" \
        --min-threshold 0.45 \
        --max-threshold 0.55 \
        --step 0.02
    
    if [ $? -eq 0 ]; then
        echo "✅ person$i 완료"
    else
        echo "❌ person$i 실패"
    fi
    echo ""
done

echo "=========================================="
echo "전체 작업 완료!"
echo "=========================================="
