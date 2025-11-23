# ArcFace ONNX 모델 설치 및 테스트 가이드

## 1. 모델 다운로드
아래 명령어 중 하나로 모델을 다운로드하세요:
```bash
# macOS/Linux (wget)
wget "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true" -O models/arcface.onnx
# macOS/Linux (curl)
curl -L "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true" -o models/arcface.onnx
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true" -OutFile "models/arcface.onnx"
```

## 2. 필수 패키지 설치
```bash
pip install onnxruntime numpy opencv-python
# (M1/M2 Mac: pip install onnxruntime-silicon)
```

## 3. 모델 로딩 및 임베딩 테스트
```python
import cv2, numpy as np, onnxruntime as ort
img = cv2.imread("face.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (112, 112))
img = (img.astype(np.float32) - 127.5) / 128.0
img = img[np.newaxis, ...]
sess = ort.InferenceSession("models/arcface.onnx")
emb = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: img})[0][0]
emb = emb / np.linalg.norm(emb)
print("Embedding shape:", emb.shape)  # (512,)
```
출력이 (512,)이면 설치 성공입니다.

## 4. 모델 정보
- 입력: (1, 112, 112, 3)
- 출력: (1, 512)
- 임베딩 크기: 512-dim vector
- 아키텍처: ArcFace (ONNX)
- 출처: https://huggingface.co/garavv/arcface-onnx

## 5. 자주 발생하는 오류
- onnxruntime ImportError: M1/M2 Mac은 onnxruntime-silicon 설치
- 이미지 shape mismatch: 입력은 반드시 (1,112,112,3)
- BGR 색상 오류: 반드시 RGB 변환 필요
- 모델 파일 경로 오류: models/arcface.onnx 경로 확인

## 6. 참고
- ArcFace 논문: https://arxiv.org/abs/1801.07698
- HuggingFace 모델 페이지: https://huggingface.co/garavv/arcface-onnx