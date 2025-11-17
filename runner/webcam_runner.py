#!/usr/bin/env python3
"""
Entry point for running the selective mosaic demo using a webcam feed.
Initializes shared MosaicPipeline and continuously processes incoming frames.

실시간 데모용. 동일한 설정으로 process_stream을 돌려 웹캠 영상을 연속 처리합니다.

"""
from pathlib import Path

from components.core.pipeline import MosaicPipeline
from config.loader import load_config


def main() -> None:
    config = load_config(Path("config/pipeline.yaml"))
    pipeline = MosaicPipeline(config)
    pipeline.process_stream(
        reference_image=Path(config.reference_image),
        webcam_id=config.webcam_id,
    )


if __name__ == "__main__":
    main()
