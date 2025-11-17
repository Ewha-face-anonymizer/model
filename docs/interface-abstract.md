# Interface Abstraction – Selective Mosaic Pipeline

팀원들이 스켈레톤 구조를 한눈에 파악할 수 있도록, 주요 모듈 간 데이터/제어 흐름, I/O 타입, 확장 포인트를 정리했다.

---

## 1. Top-Level Flow

```
main.py / runner / API
        |
        v
[MosaicPipeline]
    |  (config.AppConfig)
    v
Detection  ->  Embedding  ->  Matching  ->  Mosaic
    |            |              |             |
    v            v              v             v
 bboxes       embeddings    keep_mask      output frame
```

- `main.py`는 FastAPI 서버를 띄우거나 runner에서 직접 `MosaicPipeline` 메서드를 호출한다.
- `config/pipeline.yaml`은 경로·threshold·모델 파일 정보를 담고 `config.loader.load_config`가 `AppConfig`로 변환한다.

---

## 2. Module Responsibilities

| Layer | Module | Core Methods | Input | Output | Notes |
| --- | --- | --- | --- | --- | --- |
| Config | `config/loader.py` | `load_config` | YAML path | `AppConfig` | 전역 설정 dataclass |
| API | `api/fastapi_app.py` | `process_image` | 업로드 파일 | 저장된 경로, 파이프라인 트리거 | HTTP 엔드포인트 |
| Pipeline | `components/core/pipeline.py` | `process_image`, `process_stream` | 이미지/웹캠 ID, 참조 이미지 | 모자이크된 이미지/프레임 | 오케스트레이션 |
| Detection | `components/detection/mtcnn_detector.py` | `load_image`, `detect_faces`, `save_image` | `Path` or frame | `List[bbox]`, `List[face_crop]` | 추후 MTCNN 연결 |
| Embedding | `components/embedding/arcface_embedder.py` | `embed`, `embed_all` | face crop | 512-dim embedding | ArcFace/FaceNet 로더 |
| Reference | `components/reference/reference_manager.py` | `get_embedding` | 참조 이미지 | cached embedding | 다중 기준 지원 예정 |
| Matching | `components/matching/reference_matcher.py` | `match`, `match_batch` | embedding(s), reference | bool mask | 거리 함수/threshold 튜닝 |
| Processing | `components/processing/mosaic_processor.py` | `apply` | frame, target bboxes | mosaiced frame | ROI 픽셀화 |
| Streaming | `components/streaming/video_stream.py` | context manager, `display_frame` | webcam ID | frame iterator | RTSP/녹화 확장 포인트 |

---

## 3. Data Interfaces

| Interface | Producer | Consumer | Type |
| --- | --- | --- | --- |
| `AppConfig` | `load_config` | pipeline/API | dataclass |
| `frame` | detector loader / VideoStream | detector, mosaic | `np.ndarray (H,W,3)` BGR |
| `bboxes` | `FaceDetector.detect_faces` | pipeline/mosaicer | `List[Tuple[int,int,int,int]]` |
| `faces` | `detect_faces` | `FaceEmbedder.embed_all` | `List[np.ndarray]` |
| `embeddings` | `embed_all` | `ReferenceMatcher` | `List[np.ndarray]` (float32) |
| `keep_mask` | `match_batch` | `_mosaic_non_matches` | `List[bool]` |

각 타입/형태가 바뀌면 이 표와 관련 모듈 docstring을 함께 갱신한다.

---

## 4. Execution Modes

### 4.1 HTTP 이미지 처리

1. 클라이언트가 `/process-image` 엔드포인트로 파일 업로드.
2. FastAPI가 `config.input_dir`에 저장 후 `MosaicPipeline.process_image` 호출.
3. 파이프라인이 결과를 `config.output_dir`에 저장하고 경로 메타데이터를 응답.

### 4.2 로컬 웹캠 스트림

1. Runner/CLI가 `MosaicPipeline.process_stream(reference_image, webcam_id)` 호출.
2. `VideoStream` 컨텍스트 매니저가 OpenCV 캡처를 제공.
3. 각 프레임마다 동일한 감지→임베딩→매칭→모자이크 과정을 수행하고 창에 표시.

---

## 5. 확장 가이드

1. **모델 연결**: `FaceDetector`와 `FaceEmbedder` 스텁에 실제 MTCNN/ArcFace 로더 및 전처리를 구현.
2. **다중 기준 인물**: ReferenceManager 캐시를 ID 기반으로 확장하고 API에서 대상 ID를 입력받도록 조정.
3. **성능 로깅**: 각 단계별 latency를 측정해 로깅/모니터링 훅 추가.
4. **배포**: FastAPI 서버 실행 스크립트와 접근 제어 정책 문서화.

---

## 6. 유지보수 지침

- 새 모듈이 추가되거나 I/O 규약이 바뀌면 이 문서를 동시에 업데이트한다.
- 다이어그램/테이블이 실제 구현과 다르면 코드 리뷰 시 반드시 동기화.
- README의 “인터페이스 추상화” 섹션은 본 문서의 요약본이므로, 변경 시 README도 함께 조정한다.
