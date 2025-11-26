FROM python:3.10-slim

WORKDIR /app

# 1. 필수 시스템 패키지 설치 및 wget 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. Torch CPU 버전 설치
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 4. 나머지 모든 패키지 설치: requirements.txt 파일 사용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 🚨 YOLOv8s-World 모델 파일 다운로드 (Grounding DINO Lite 대체)
# 정확도를 높이고자 YOLOv8s-World 체크포인트를 다운로드합니다.
RUN wget -O /app/yolov8s-world.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt

# 6. MobileSAM 모델 파일 다운로드
RUN wget -O /app/mobile_sam.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt

# 앱 코드 복사
COPY server.py .

# FastAPI 실행
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}