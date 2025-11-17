# Tests

Pytest를 사용해 감지기, 매칭 로직, 모자이크 후처리의 최소 동작을 검증합니다.

```bash
pytest
```

주요 커버리지
- `FaceDetector.load_image/save_image` 예외 처리 및 파일 입출력
- `ReferenceMatcher` 임계값 및 0-벡터 대응
- `MosaicProcessor` ROI 픽셀화가 대상 영역에만 적용되는지 확인
