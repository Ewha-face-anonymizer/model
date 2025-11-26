# components/embedding/arcface_embedder.py
"""
ArcFace/FaceNet을 사용해 얼굴을 벡터로 만드는 임베더 스텁입니다.
Face embedding utilities built around ArcFace or FaceNet.
"""
from pathlib import Path
from typing import Iterable, List
import numpy as np
import onnxruntime as ort #ONNX 모델 실행용
import cv2 #OpenCV


class FaceEmbedder:
    """
    ArcFace ONNX 임베더
    - 모델 로드
    - 얼굴 임베딩 생성 (512차원)
    - L2 정규화 적용
    """

    def __init__(self, model_path: str):
        self.model_path = str(model_path)

        # ArcFace ONNX 모델 로딩 (모델 객체 생성)
        self.session = ort.InferenceSession( # arcFace.onnx 모델 메모리에 로딩
            self.model_path,
            providers=["CPUExecutionProvider"] # CPU에서 실행
        )

        # 입력/출력 정보 획득 (ONNX 모델은 입력 이름 필요)
        self.input_name = self.session.get_inputs()[0].name # 입력 텐서 이름
        self.output_name = self.session.get_outputs()[0].name # 출력 텐서 이름(임베딩 512차원)


    # 1. 이미지 전처리 함수 (얼굴 이미지 -> ArcFace 모델 요구 형식으로 변환)
    def preprocess(self, face: np.ndarray) -> np.ndarray:
        # 112x112로 resize
        face = cv2.resize(face, (112, 112))

        # BGR → RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize: (img - 127.5) / 128.0
        face = (face.astype(np.float32) - 127.5) / 128.0

        # HWC 그대로, batch dimension 추가
        face = np.expand_dims(face, axis=0)  # (1,112,112,3) (배치크기, 높이, 너비, 채널)

        return face


    # 2. L2 정규화 함수
    def l2_normalize(self, x: np.ndarray, eps=1e-10) -> np.ndarray:
        return x / (np.linalg.norm(x) + eps) # eps: 0으로 나누기 방지용 작은 숫자


    # 3. ArcFace 임베딩 추출
    def embed(self, face: np.ndarray) -> np.ndarray:
        """
        얼굴 임베딩 생성 함수 (512차원)
        """
        # 1) 전처리
        inp = self.preprocess(face)

        # 2) ArcFace ONNX 모델 추론
        emb = self.session.run(
            [self.output_name],      # ArcFace 모델의 출력 이름
            {self.input_name: inp},  # ArcFace 모델의 입력 이름에 inp 전달
        )[0]

        emb = emb.squeeze()  # 배치 차원 제거 (1,512) → (512,)

        # 3) L2 normalize
        emb = self.l2_normalize(emb)

        # 512차원 벡터 반환
        return emb.astype(np.float32)

    # 4. 여러 얼굴 임베딩 생성
    def embed_all(self, faces: Iterable[np.ndarray]) -> List[np.ndarray]:
        return [self.embed(face) for face in faces]
