import io
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld, SAM
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Lazy loading
det_model = None
sam_model = None
device = None

def load_models():
    global det_model, sam_model, device
    
    if det_model is None:
        print("Loading lightweight models...")
        device = "cpu"  # CPU 명시
        
        # Nano 모델 (가장 작음)
        det_model = YOLOWorld("yolov8n-worldv2.pt")  # s → n
        det_model.to(device)
        det_model.set_classes(["wall"])
        
        # Mobile SAM (이미 최소)
        sam_model = SAM("mobile_sam.pt")
        sam_model.to(device)
        
        print("Models loaded!")
        
        # 메모리 정리
        import gc
        gc.collect()

def np_from_upload(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def filter_small_boxes(boxes, img_shape, min_ratio=0.03):
    H, W = img_shape
    area_img = H * W
    filtered = []
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area / area_img > min_ratio:
            filtered.append([float(x1), float(y1), float(x2), float(y2)])
    return filtered

def post_refine(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    
    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 1, thickness=cv2.FILLED)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean

@app.get("/")
async def root():
    return {"status": "ok", "message": "Wall Segmentation Server (Lightweight)"}

@app.get("/health")
async def health():
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return {
        "status": "healthy",
        "models_loaded": det_model is not None,
        "memory_mb": round(memory_mb, 2)
    }

@app.post("/segment_wall_mask")
async def segment_wall_mask(file: UploadFile = File(...)):
    try:
        load_models()
        
        img = np_from_upload(await file.read())
        
        # 이미지 크기 축소 (메모리 절약)
        max_size = 640  # 1024 → 640
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
        
        pil_img = img.copy()

        # YOLO 예측 (이미지 크기 축소)
        results = det_model.predict(
            pil_img, 
            conf=0.20, 
            imgsz=640,  # 1024 → 640
            device=device, 
            verbose=False
        )[0]
        
        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        boxes = filter_small_boxes(xyxy, pil_img.size[::-1])

        if not boxes and len(xyxy) > 0:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy]
            biggest = xyxy[np.argmax(areas)].tolist()
            boxes = [biggest]

        if not boxes:
            return Response(content=b'', status_code=422)

        # SAM 예측
        res = sam_model.predict(
            pil_img, 
            bboxes=boxes, 
            device=device, 
            retina_masks=False,  # True → False (메모리 절약)
            verbose=False
        )[0]
        
        if res.masks is None:
            return Response(content=b'', status_code=422)

        mask = res.masks.data.cpu().numpy()
        union = (mask.sum(axis=0) > 0).astype(np.uint8)
        refined = post_refine(union)

        mask_img = (refined * 255).astype(np.uint8)
        _, png = cv2.imencode(".png", mask_img)

        # 메모리 정리
        del img, pil_img, results, mask, union, refined, mask_img
        import gc
        gc.collect()

        return Response(
            content=png.tobytes(),
            media_type="image/png",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            content=str(e).encode(),
            status_code=500
        )

@app.options("/segment_wall_mask")
async def options_segment_wall_mask():
    return Response(
        content=b'',
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )