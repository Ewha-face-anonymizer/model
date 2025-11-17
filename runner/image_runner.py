#!/usr/bin/env python3
"""
Entry point for running mosaic processing on a static image.
Loads configuration and executes the shared MosaicPipeline once.

정적 이미지 테스트용. config/pipeline.yaml을 읽고 
MosaicPipeline.process_image 한 번 실행 후 data/output/에 결과를 저장합니다.
"""
from pathlib import Path

from components.core.pipeline import MosaicPipeline
from config.loader import load_config


def main() -> None:
    config = load_config(Path("config/pipeline.yaml"))
    pipeline = MosaicPipeline(config)
    pipeline.process_image(
        input_image=Path(config.test_image),
        reference_image=Path(config.reference_image),
        output_dir=Path(config.output_dir),
    )


if __name__ == "__main__":
    main()
