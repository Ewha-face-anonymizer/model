#!/bin/bash
# person01~person50까지 threshold_tuning_group 실행
# Threshold 범위: 0.35~0.60, 0.05 간격

echo "======================================================================"
echo "Threshold Tuning 대규모 실행 시작"
echo "대상: person01 ~ person50 (person06 제외)"
echo "Threshold 범위: 0.45 ~ 0.55 (0.02 간격)"
echo "======================================================================"

DATASET_DIR="data/input/Dataset"

# person00~person50 반복
for i in {0..50}; do
    PERSON=$(printf "person%02d" $i)
    PERSON_DIR="${DATASET_DIR}/${PERSON}"
    
    # 폴더가 없으면 건너뛰기
    if [ ! -d "$PERSON_DIR" ]; then
        echo "⏭️  ${PERSON}: 폴더 없음 (건너뜀)"
        continue
    fi
    
    # 단체사진 찾기 (ref_가 아닌 jpg 파일)
    GROUP_PHOTO=$(find "$PERSON_DIR" -maxdepth 1 -type f -name "*.jpg" ! -name "ref_*" | head -n 1)
    
    if [ -z "$GROUP_PHOTO" ]; then
        echo "⚠️  ${PERSON}: 단체사진 없음 (건너뜀)"
        continue
    fi
    
    echo ""
    echo "======================================================================"
    echo "🔄 처리 중: ${PERSON}"
    echo "======================================================================"
    
    # threshold_tuning_group 실행
    python3 threshold_tuning_group.py \
        --group "$GROUP_PHOTO" \
        --reference "$PERSON_DIR" \
        --min-threshold 0.45 \
        --max-threshold 0.55 \
        --step 0.02
    
    if [ $? -eq 0 ]; then
        echo "✅ ${PERSON}: 완료"
    else
        echo "❌ ${PERSON}: 실패"
    fi
done

echo ""
echo "======================================================================"
echo "모든 작업 완료!"
echo "======================================================================"
