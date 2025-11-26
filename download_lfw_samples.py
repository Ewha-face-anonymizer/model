#!/usr/bin/env python3
"""
LFW 데이터셋 샘플 다운로드 스크립트
실제 LFW 데이터셋에서 몇 개의 인물 이미지를 다운로드합니다.
"""

import os
import urllib.request
import ssl
from pathlib import Path

# SSL 컨텍스트 설정 (일부 사이트에서 필요할 수 있음)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# 다운로드할 샘플 이미지 URL들 (LFW 형식의 이미지들)
sample_images = [
    {
        "name": "Aaron_Eckhart_0001.jpg",
        "url": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/aaron_eckhart.jpg"
    },
    {
        "name": "Biden_0001.jpg", 
        "url": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg"
    },
    {
        "name": "Obama_0001.jpg",
        "url": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg"
    },
    {
        "name": "Tom_Hanks_0001.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Tom_Hanks_TIFF_2019.jpg/256px-Tom_Hanks_TIFF_2019.jpg"
    },
    {
        "name": "Emma_Stone_0001.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Emma_Stone_at_the_39th_Mill_Valley_Film_Festival_%28cropped%29.jpg/256px-Emma_Stone_at_the_39th_Mill_Valley_Film_Festival_%28cropped%29.jpg"
    }
]

def download_image(url, filename):
    """이미지 다운로드"""
    try:
        print(f"다운로드 중: {filename}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, context=ssl_context) as response:
            data = response.read()
            
        with open(filename, 'wb') as f:
            f.write(data)
            
        print(f"✓ 성공: {filename} ({len(data)} bytes)")
        return True
        
    except Exception as e:
        print(f"✗ 실패: {filename} - {str(e)}")
        return False

def main():
    # 현재 디렉토리 확인
    output_dir = Path("/Users/yxpjseo/ML/model/data/input/lfw_sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"LFW 샘플 이미지들을 {output_dir}에 다운로드합니다...")
    
    success_count = 0
    for img_info in sample_images:
        filepath = output_dir / img_info["name"]
        if download_image(img_info["url"], str(filepath)):
            success_count += 1
    
    print(f"\n완료: {success_count}/{len(sample_images)} 이미지 다운로드됨")
    
    # 다운로드된 파일 목록 출력
    print("\n다운로드된 파일들:")
    for file in output_dir.glob("*.jpg"):
        size = file.stat().st_size
        print(f"  - {file.name} ({size:,} bytes)")

if __name__ == "__main__":
    main()