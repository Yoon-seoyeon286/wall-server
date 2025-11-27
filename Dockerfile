FROM python:3.10-slim

WORKDIR /app


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



RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip install --no-cache-dir 

torch==2.2.2 

torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu && 

pip install --no-cache-dir -r requirements.txt



RUN wget -O /app/mobile_sam.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt
RUN wget -O /app/yolov8s.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt

COPY server.py .



CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]