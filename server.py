import os
import io
import cv2
import torch
import numpy as np
import gc
import logging
from PIL import Image
from ultralytics import YOLO, SAM
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 전역 변수
det_model = None  # YOLOv8n (COCO general detection)
sam_model = None  # MobileSAM
device = "cpu"


@app.on_event("startup")
def load_models_on_startup():
    """서버 시작 시 YOLOv8n + MobileSAM 로드"""
    global det_model, sam_model, device
    
    logger.info("[🔥] Starting model loading for YOLOv8n + MobileSAM...")
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[⚙️] Device: {device}")
    
    # Dockerfile에서 다운로드하는 파일명과 일치
    yolo_checkpoint_path = "yolov8n.pt" 
    sam_checkpoint_path = "mobile_sam.pt"

    try:
        # 1. YOLOv8n 모델 로드 (COCO trained)
        if not os.path.exists(yolo_checkpoint_path):
             logger.error(f"[❌] YOLOv8n checkpoint not found at: {yolo_checkpoint_path}")
        else:
            det_model = YOLO(yolo_checkpoint_path)
            det_model.to(device)
            logger.info("[✅] YOLOv8n loaded.")
        
        # 2. MobileSAM 로드
        if not os.path.exists(sam_checkpoint_path):
             logger.error(f"[❌] MobileSAM checkpoint not found at: {sam_checkpoint_path}")
        else:
            sam_model = SAM(sam_checkpoint_path)
            sam_model.to(device)
            logger.info("[✅] MobileSAM loaded.")
        
    except Exception as e:
        logger.error(f"[❌] FATAL Model loading failed: {e}", exc_info=True)


def np_from_upload(file_bytes: bytes) -> Image.Image:
    """바이트를 PIL Image로 변환"""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def filter_small_boxes(boxes, img_shape, min_ratio=0.03):
    """너무 작은 박스 필터링 (노이즈 제거)"""
    H, W = img_shape
    area_img = H * W
    filtered = []
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        # 면적이 전체 이미지의 3% 미만이면 필터링
        if area / area_img > min_ratio:
            filtered.append([float(x1), float(y1), float(x2), float(y2)])
    return filtered


def post_refine(mask: np.ndarray):
    """마스크 후처리: 노이즈 제거, 확대, 가장 큰 연결 영역만 남기기 (벽 영역 추정)"""
    mask = mask.astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)

    # 노이즈 제거 (Opening) + 경계 채우기 (Dilate)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 가장 큰 연결 영역만 남기기 (가장 큰 배경 또는 객체 영역을 찾으려는 의도)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask

    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 1, thickness=cv2.FILLED)
    
    # 영역을 부드럽게 닫기 (Closing)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean


@app.get("/")
async def root():
    return {"status": "ok", "message": "YOLOv8n + MobileSAM Wall Segmentation Server (Reverted)"}


@app.get("/health")
async def health():
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    gc.collect()
    
    return {
        "status": "healthy",
        "models_loaded": det_model is not None and sam_model is not None,
        "device": device,
        "memory_mb": round(memory_mb, 2)
    }


@app.post("/segment_wall_mask")
async def segment_wall_mask(file: UploadFile = File(...)):
    """YOLOv8n으로 객체 감지 → MobileSAM으로 분할 → 후처리로 벽 영역 추출"""
    
    # 모델 로딩 여부 확인
    if det_model is None or sam_model is None:
        logger.error("Segmentation services are unavailable due to model loading failure.")
        return Response(content="Model load failed. Check server startup logs.", status_code=503)

    try:
        file_bytes = await file.read()
        if not file_bytes:
            return Response(content="File is empty.", status_code=400)
        
        img = np_from_upload(file_bytes)
        original_size = img.size
        
        # 이미지 크기 축소 (메모리 절약)
        max_size = 640
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)

        pil_img = img.copy()
        w, h = pil_img.size
        logger.info(f"[📸] 이미지: {w}x{h}")

        # 1. YOLOv8n 예측 (COCO 모든 객체 감지)
        logger.info("[🔍] YOLOv8n: 객체 감지 중...")
        results = det_model.predict(
            pil_img,
            conf=0.20, # 충분히 낮은 confidence
            imgsz=640,
            device=device,
            verbose=False,
            # classes 필터링 없이 모든 COCO 클래스 사용
        )[0]

        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        boxes = filter_small_boxes(xyxy, pil_img.size[::-1])
        
        logger.info(f"[✅] {len(boxes)}개의 유효 객체 박스 발견")

        # 2. 예외 처리: 박스가 없거나 너무 작으면, 전체 이미지를 박스로 사용
        if not boxes:
            logger.warning("[⚠️] 객체 박스가 없어 전체 이미지 박스 사용.")
            boxes = [[0.0, 0.0, float(w), float(h)]]
        
        # 3. MobileSAM 예측
        logger.info("[🎨] MobileSAM: 객체 분할 중...")
        sam_boxes = boxes
        
        res = sam_model.predict(
            pil_img,
            bboxes=sam_boxes,
            device=device,
            retina_masks=False,
            verbose=False
        )[0]

        if res.masks is None:
            logger.warning("[⚠️] MobileSAM 분할 실패. 전체 화면 반환.")
            mask_img = np.ones((h, w), dtype=np.uint8) * 255
        else:
            # 4. 마스크 통합 및 후처리
            mask_data = res.masks.data.cpu().numpy()
            union = (mask_data.sum(axis=0) > 0).astype(np.uint8)
            
            # 후처리 (가장 큰 연결 영역만 남김)
            refined = post_refine(union)
            mask_img = (refined * 255).astype(np.uint8)
            
            # 💡 경계면 부드럽게 처리 (Smoothing) - 커널 크기 증가 (9, 9)
            # 마스크 경계를 더욱 부드럽게 만들기 위해 Gaussian Blur 커널 크기 증가
            mask_img = cv2.GaussianBlur(mask_img, (9, 9), 0)
            
            del mask_data, union, refined
        
        # 5. 원본 크기로 복원
        if img.size != original_size:
            mask_img = cv2.resize(
                mask_img, 
                original_size, 
                interpolation=cv2.INTER_LINEAR
            )
        
        # PNG 인코딩
        _, png = cv2.imencode(".png", mask_img)

        # 🚨 메모리 정리 강화 
        del img, pil_img, results, boxes, sam_boxes
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        
        gc.collect() 

        final_png_bytes = png.tobytes()
        del png, _
        gc.collect()

        return Response(
            content=final_png_bytes,
            media_type="image/png",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache"
            }
        )

    except Exception as e:
        logger.error(f"❌ ERROR in segmentation processing: {e}", exc_info=True)
        gc.collect()
        return Response(
            content=f"Internal Server Error: {e}".encode(),
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