# Face Recognition 환경 설정 완료

## 설치된 환경
- **Conda 환경명**: face-embed
- **Python 버전**: 3.10
- **주요 패키지들**:
  - OpenCV (opencv)
  - MTCNN (facenet-pytorch)
  - ArcFace (insightface)
  - PyTorch 2.2.2
  - ONNX Runtime
  - FastAPI
  - NumPy, PIL

## 환경 활성화 방법
```bash
conda activate face-embed
```

## 테스트 이미지
`/Users/yxpjseo/ML/model/data/input/lfw_sample/` 디렉토리에 5개의 얼굴 이미지 샘플이 준비되어 있습니다:
- Brad_Pitt_0001.jpg
- Obama_0001.jpg  
- Biden_0001.jpg
- Tom_Hanks_0001.jpg
- Angelina_Jolie_0001.jpg

## 환경 검증
```bash
cd /Users/yxpjseo/ML/model
conda activate face-embed
python simple_test.py
```

## 다음 단계
1. `conda activate face-embed` - 환경 활성화
2. `python main.py` - 메인 애플리케이션 실행
3. 또는 `python api/fastapi_app.py` - FastAPI 서버 실행

## 패키지 업데이트
환경을 업데이트하려면:
```bash
conda env update -f environment.yml --prune
```

---
환경 설정이 완료되었습니다! 🎉