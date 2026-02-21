#!/usr/bin/env python3
"""
Entry point for running mosaic processing on a static image.
Loads configuration and executes the shared MosaicPipeline once.

정적 이미지 테스트용. config/pipeline.yaml을 읽고 
MosaicPipeline.process_image 한 번 실행 후 data/output/에 결과를 저장합니다.
"""
#!/usr/bin/env python3
from pathlib import Path
import sys

print("[A] script start")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("[B] before imports from project")

from components.core.pipeline import MosaicPipeline
from config.loader import load_config

print("[C] after imports")


def main() -> None:
    print("[D] before load_config")
    config = load_config(Path("config/pipeline.yaml"))
    print("[E] after load_config")

    print("[F] before MosaicPipeline init")
    pipeline = MosaicPipeline(config)
    print("[G] after MosaicPipeline init")

    print("[H] before process_image")
    pipeline.process_image(
        input_image=Path(config.test_image),
        reference_image=Path(config.reference_image),
        output_dir=Path(config.output_dir),
    )
    print("[I] after process_image")


if __name__ == "__main__":
    print("[J] before main()")
    main()
    print("[K] after main()")
