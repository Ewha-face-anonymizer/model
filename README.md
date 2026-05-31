# 우리아이만 (Our Kid Only) — 선택적 얼굴 익명화 시스템

<div align="center">

*등록한 얼굴만 남기고, 나머지는 자동으로 블러 처리합니다*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square)

</div>

---

## Why

어린이집 단체 사진, 운동회 영상, SNS 공유 콘텐츠 — 내 아이 사진을 올리고 싶지만 다른 아이들의 얼굴이 함께 찍혀 있으면 업로드하기 망설여집니다.

기존 전체 블러 처리는 사진의 의미를 훼손합니다. **내 아이 얼굴은 선명하게, 다른 얼굴은 자동으로 블러** — 그게 이 시스템의 목표입니다.

---

## How It Works

```
입력 이미지/영상
      ↓
MTCNN 얼굴 감지        ← 얼굴 위치 + 랜드마크 탐지
      ↓
ArcFace 임베딩         ← 각 얼굴을 512차원 벡터로 변환
      ↓
코사인 거리 매칭        ← 등록한 기준 얼굴과 비교
      ↓
기준 얼굴? → 원본 유지
아니오?   → OpenCV 모자이크 블러
      ↓
결과 이미지/영상 출력
```

---

## 핵심 기술

| 컴포넌트 | 기술 | 선택 이유 |
|----------|------|----------|
| **얼굴 감지** | MTCNN (facenet-pytorch) | 소형 얼굴·측면에 강인한 멀티스테이지 CNN |
| **얼굴 임베딩** | ArcFace (ONNX) | Angular Margin Loss 기반, FaceNet 대비 높은 정확도 |
| **유사도 측정** | 코사인 거리 | 조명·각도 변화에 강건한 임베딩 비교 |
| **블러 처리** | OpenCV | 픽셀화 또는 가우시안 블러 선택 적용 |

---

## 실험 및 검증

단순 구현에 그치지 않고 **실제 성능을 측정**했습니다.

- **Precision-Recall 분석**: threshold 값에 따른 정밀도·재현율 변화 측정
- **Threshold Tuning**: 인물별 최적 임계값 탐색 (동일 인물 / 그룹 시나리오 분리)
- **A/B 성능 실험**: 프레임 스킵, 리사이즈, 트래킹 도입 효과 비교
- **아동 특화 평가셋**: 소형 얼굴(원거리), 가림(모자/손), 빠른 움직임 포함

```bash
# Threshold 튜닝 실행
bash run_threshold_tuning.sh

# Precision-Recall 분석
python analyze_precision_recall.py
```

---

## 시작하기

### 환경 설정

```bash
conda env create -f environment.yml
conda activate face-embed
python -m pip install --upgrade pip
```

> Apple Silicon(M1/M2)에서 onnxruntime 오류 시: `pip install onnxruntime-silicon`

### 설치 확인

```bash
python test_environment.py
```

### 기본 실행

1. `config/pipeline.yaml`에 기준 인물 이미지 경로 설정
2. 파이프라인 실행

```bash
python main.py
```

결과는 `data/output/`에 저장됩니다.

---

## 프로젝트 구조

```
├── components/
│   ├── detection/      # MTCNN 얼굴 감지
│   ├── embedding/      # ArcFace 임베딩 추출
│   ├── matching/       # 코사인 거리 매칭
│   ├── processing/     # 모자이크 블러 적용
│   └── reference/      # 기준 얼굴 임베딩 캐시
├── config/             # YAML 설정 파일
├── models/             # ArcFace, MTCNN ONNX 모델
├── data/
│   ├── input/          # 입력 이미지 및 기준 인물 이미지
│   └── output/         # 처리 결과
├── runner/             # 실험 실행 스크립트
└── tests/              # 단위 테스트
```

---

## 문제 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| `onnxruntime` ImportError | Apple Silicon 미지원 | `pip install onnxruntime-silicon` |
| 얼굴 검출 안 됨 | BGR→RGB 변환 누락 | 이미지 로딩 시 RGB 변환 확인 |
| GPU 미동작 | CUDA 없는 환경 | CPU 모드 자동 사용, 별도 설정 불필요 |

---

## License

[LICENSE](LICENSE) 참고
