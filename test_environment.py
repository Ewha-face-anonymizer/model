#!/usr/bin/env python3
"""
설치된 패키지들과 다운로드된 테스트 이미지 검증 스크립트
OpenCV, MTCNN, ArcFace 패키지 import 테스트 및 이미지 로딩 테스트
"""

import sys
from pathlib import Path

def test_package_imports():
    """필수 패키지들 import 테스트"""
    print("=== 패키지 Import 테스트 ===")
    
    packages = [
        ("opencv-cv2", "cv2"),
        ("numpy", "numpy"),
        ("PIL", "PIL"),
        ("torch", "torch"),
        ("facenet-pytorch", "facenet_pytorch"),
        ("mtcnn", "mtcnn"),
        ("insightface", "insightface"),
        ("onnxruntime", "onnxruntime")
    ]
    
    success_count = 0
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name}: 정상")
            success_count += 1
        except ImportError as e:
            print(f"✗ {package_name}: 실패 - {e}")
        except Exception as e:
            print(f"? {package_name}: 경고 - {e}")
    
    print(f"\n{success_count}/{len(packages)} 패키지 import 성공")
    return success_count == len(packages)

def test_image_loading():
    """다운로드된 이미지들 로딩 테스트"""
    print("\n=== 이미지 로딩 테스트 ===")
    
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"이미지 처리 라이브러리 import 실패: {e}")
        return False
    
    image_dir = Path("/Users/yxpjseo/ML/model/data/input/lfw_sample")
    image_files = list(image_dir.glob("*.jpg"))
    
    if not image_files:
        print("테스트할 이미지가 없습니다.")
        return False
    
    success_count = 0
    for img_path in image_files[:5]:  # 처음 5개만 테스트
        try:
            # OpenCV로 로딩
            img_cv = cv2.imread(str(img_path))
            if img_cv is not None:
                height, width = img_cv.shape[:2]
                
                # PIL로도 로딩
                img_pil = Image.open(img_path)
                
                print(f"✓ {img_path.name}: {width}x{height}, OpenCV+PIL 로딩 성공")
                success_count += 1
            else:
                print(f"✗ {img_path.name}: OpenCV 로딩 실패")
                
        except Exception as e:
            print(f"✗ {img_path.name}: 에러 - {e}")
    
    print(f"\n{success_count}/{min(5, len(image_files))} 이미지 로딩 성공")
    return success_count > 0

def test_mtcnn_basic():
    """MTCNN 기본 초기화 테스트"""
    print("\n=== MTCNN 초기화 테스트 ===")
    
    try:
        from facenet_pytorch import MTCNN
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], 
            factor=0.709, 
            post_process=False,
            device=device
        )
        
        print(f"✓ MTCNN 초기화 성공 (device: {device})")
        return True
        
    except Exception as e:
        print(f"✗ MTCNN 초기화 실패: {e}")
        return False

def main():
    print("Face Recognition 환경 설정 검증을 시작합니다...\n")
    
    # 테스트 수행
    import_ok = test_package_imports()
    image_ok = test_image_loading()
    mtcnn_ok = test_mtcnn_basic()
    
    # 결과 요약
    print("\n" + "="*50)
    print("검증 결과 요약:")
    print(f"  - 패키지 Import: {'✓' if import_ok else '✗'}")
    print(f"  - 이미지 로딩: {'✓' if image_ok else '✗'}")
    print(f"  - MTCNN 초기화: {'✓' if mtcnn_ok else '✗'}")
    
    if import_ok and image_ok and mtcnn_ok:
        print("\n🎉 모든 테스트 통과! 환경 설정이 완료되었습니다.")
        print("\n다음 단계:")
        print("  conda activate face-embed")
        print("  python main.py  # 메인 애플리케이션 실행")
    else:
        print("\n⚠️  일부 테스트 실패. 환경 설정을 확인해주세요.")
    
    return import_ok and image_ok and mtcnn_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)