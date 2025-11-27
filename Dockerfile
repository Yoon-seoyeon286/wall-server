Python 3.10 slim 이미지를 사용합니다.

FROM python:3.10-slim

WORKDIR /app

필요한 시스템 패키지 설치 (OpenCV, wget 및 GL 라이브러리)

RUN apt-get update && apt-get install -y --no-install-recommends 

wget 

libgl1 

libglib2.0-0 

libjpeg62-turbo 

libpng16-16 

libsm6 

libxrender1 

libfontconfig1 

libxext6 

&& rm -rf /var/lib/apt/lists/*

pip 업그레이드

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

모든 PyPI 종속성 설치 (PyTorch CPU 버전 포함)

COPY requirements.txt .
RUN pip install --no-cache-dir 

torch==2.2.2 

torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu && 

pip install --no-cache-dir -r requirements.txt

MobileSAM 및 YOLOv8s 모델 파일 다운로드

RUN wget -O /app/mobile_sam.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt
RUN wget -O /app/yolov8s.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt

앱 코드 복사

COPY server.py .

FastAPI 실행

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]