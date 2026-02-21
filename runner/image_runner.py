#!/usr/bin/env python3
"""
Entry point for running mosaic processing on a static image.
Loads configuration and executes the shared MosaicPipeline once.

정적 이미지 테스트용. config/pipeline.yaml을 읽고 
MosaicPipeline.process_image 한 번 실행 후 data/output/에 결과를 저장합니다.
"""
import sys
from pathlib import Path
import argparse

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.core.pipeline import MosaicPipeline
from config.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description='이미지 모자이크 처리')
    parser.add_argument('--reference', '-r', type=str, required=False,
                       help='기준인물 사진 폴더 또는 파일 (예: data/input/베일리)')
    parser.add_argument('--test', '-t', type=str, required=False,
                       help='테스트 이미지 (단체사진) 경로 (예: data/input/단체사진.jpg)')
    parser.add_argument('--output', '-o', type=str, required=False,
                       help='출력 디렉토리 (기본: data/output)')
    
    args = parser.parse_args()
    
    config = load_config(Path("config/pipeline.yaml"))
    
    # 커맨드 라인 인자로 덮어쓰기
    reference_path = Path(args.reference) if args.reference else Path(config.reference_image)
    test_image = Path(args.test) if args.test else Path(config.test_image)
    output_dir = Path(args.output) if args.output else Path(config.output_dir)
    
    # 기준인물 사진 경로 처리 (폴더 또는 파일)
    if reference_path.is_dir():
        # 폴더인 경우: 모든 이미지 파일 찾기
        reference_images = sorted(list(reference_path.glob("*.jpg")) + 
                                 list(reference_path.glob("*.png")) +
                                 list(reference_path.glob("*.jpeg")))
        if not reference_images:
            print(f"오류: 폴더에서 이미지를 찾을 수 없습니다: {reference_path}")
            return
    elif reference_path.exists():
        # 단일 파일인 경우
        reference_images = [reference_path]
    else:
        print(f"오류: 기준인물 사진 파일/폴더를 찾을 수 없습니다: {reference_path}")
        return
    
    if not test_image.exists():
        print(f"오류: 테스트 이미지를 찾을 수 없습니다: {test_image}")
        return
    
    # 기준인물 이름 추출 (폴더명 또는 파일의 부모 폴더명)
    if reference_path.is_dir():
        person_name = reference_path.name
    else:
        person_name = reference_path.parent.name
    
    pipeline = MosaicPipeline(config)
    pipeline.process_image(
        input_image=test_image,
        reference_images=reference_images,
        output_dir=output_dir,
        person_name=person_name,
    )
    
    print(f"\n처리 완료! 결과: {output_dir / f'{test_image.stem}_{person_name}_mosaic.jpg'}")


if __name__ == "__main__":
    main()
