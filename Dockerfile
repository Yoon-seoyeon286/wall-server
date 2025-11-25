FROM python:3.10-slim

WORKDIR /app

# 필수 시스템 패키지 (OpenCV + git + build deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 1) Numpy 먼저 (torch 의존 대비)
RUN pip install --no-cache-dir "numpy<2"

# 2) Torch CPU only (버전 고정)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    torchaudio==2.2.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 3) 나머지 requirements 설치 (torch 제외)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) 앱 코드 복사
COPY server.py .

# 5) FastAPI 런
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
