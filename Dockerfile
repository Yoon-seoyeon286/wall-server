FROM python:3.10-slim

WORKDIR /app

# 1. 필수 시스템 패키지 설치
# OpenGL 관련 패키지 (libgl1, libglib2.0-0)는 OpenCV-Python의 필수 의존성입니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# 2. pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. Torch CPU 버전 설치
# PyTorch와 그 의존성(torchvision, torchaudio)을 먼저 설치합니다.
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 4. 나머지 모든 패키지 설치: requirements.txt 파일 사용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사 (server.py는 이전 답변의 최종 버전으로 가정합니다.)
COPY server.py .

# FastAPI 실행
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}