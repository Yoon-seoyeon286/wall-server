FROM python:3.10-slim

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 2. PyTorch CPU 버전 설치
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# 3. 나머지 모든 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. MobileSAM 및 YOLOv8n 모델 파일 다운로드
RUN wget -O /app/mobile_sam.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt
RUN wget -O /app/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

# 앱 코드 복사
COPY server.py .

# FastAPI 실행
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}