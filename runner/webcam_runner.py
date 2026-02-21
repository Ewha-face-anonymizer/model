#!/usr/bin/env python3
"""
Entry point for running the selective mosaic demo using a webcam feed.
Initializes shared MosaicPipeline and continuously processes incoming frames.

실시간 데모용. 동일한 설정으로 process_stream을 돌려 웹캠 영상을 연속 처리합니다.

"""
import sys
from pathlib import Path
import argparse

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.core.pipeline import MosaicPipeline
from config.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description='웹캠 모자이크 처리')
    parser.add_argument('--reference', '-r', type=str, required=False,
                       help='기준인물 사진 폴더 또는 파일 (예: data/input/베일리)')
    parser.add_argument('--webcam', '-w', type=int, required=False,
                       help='웹캠 ID (기본: 0)')
    
    args = parser.parse_args()
    
    config = load_config(Path("config/pipeline.yaml"))
    
    # 커맨드 라인 인자로 덮어쓰기
    reference_path = Path(args.reference) if args.reference else Path(config.reference_image)
    webcam_id = args.webcam if args.webcam is not None else config.webcam_id
    
    # 기준인물 사진 경로 처리 (폴더 또는 파일)
    if reference_path.is_dir():
        reference_images = sorted(list(reference_path.glob("*.jpg")) + 
                                 list(reference_path.glob("*.png")) +
                                 list(reference_path.glob("*.jpeg")))
        if not reference_images:
            print(f"오류: 폴더에서 이미지를 찾을 수 없습니다: {reference_path}")
            return
    elif reference_path.exists():
        reference_images = [reference_path]
    else:
        print(f"오류: 기준인물 사진 파일/폴더를 찾을 수 없습니다: {reference_path}")
        return
    
    pipeline = MosaicPipeline(config)
    pipeline.process_stream(
        reference_images=reference_images,
        webcam_id=webcam_id,
    )


if __name__ == "__main__":
    main()
