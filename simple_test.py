#!/usr/bin/env python3
"""
간단한 패키지 import 검증 스크립트
"""

def test_imports():
    print("=== 패키지 Import 테스트 ===")
    
    # OpenCV
    try:
        import cv2
        print("✓ OpenCV import 성공")
    except ImportError:
        print("✗ OpenCV import 실패")
    
    # NumPy
    try:
        import numpy as np
        print("✓ NumPy import 성공")
    except ImportError:
        print("✗ NumPy import 실패")
        
    # PIL/Pillow
    try:
        from PIL import Image
        print("✓ PIL/Pillow import 성공")
    except ImportError:
        print("✗ PIL/Pillow import 실패")
        
    # PyTorch
    try:
        import torch
        print(f"✓ PyTorch import 성공 (version: {torch.__version__})")
    except ImportError:
        print("✗ PyTorch import 실패")
        
    # FaceNet PyTorch (MTCNN 포함)
    try:
        from facenet_pytorch import MTCNN
        print("✓ FaceNet-PyTorch (MTCNN) import 성공")
    except ImportError:
        print("✗ FaceNet-PyTorch import 실패")
        
    # InsightFace
    try:
        import insightface
        print("✓ InsightFace import 성공")
    except ImportError:
        print("✗ InsightFace import 실패")
        
    # ONNX Runtime
    try:
        import onnxruntime
        print("✓ ONNX Runtime import 성공")
    except ImportError:
        print("✗ ONNX Runtime import 실패")

def test_sample_images():
    print("\n=== 샘플 이미지 확인 ===")
    import os
    from pathlib import Path
    
    sample_dir = Path("/Users/yxpjseo/ML/model/data/input/lfw_sample")
    image_files = list(sample_dir.glob("*.jpg"))
    
    print(f"다운로드된 이미지 파일 수: {len(image_files)}")
    
    for img_file in image_files:
        size = img_file.stat().st_size
        if size > 1000:  # 1KB 이상인 파일만 유효한 이미지로 판단
            print(f"✓ {img_file.name} ({size:,} bytes)")
        else:
            print(f"? {img_file.name} ({size} bytes) - 크기가 작음")

if __name__ == "__main__":
    test_imports()
    test_sample_images()
    
    print("\n=== 환경 설정 완료 ===")
    print("conda activate face-embed 를 실행한 후 개발을 시작하세요!")