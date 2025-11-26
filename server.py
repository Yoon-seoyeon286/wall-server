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

# ==============================================================================
# 💡 [조정 가능한 설정] - Wall/Object Estimation Parameters (객체 제외 심화)
# ==============================================================================
# 1. YOLOv8 객체 감지 민감도: 낮출수록 더 많은 객체를 감지하여 벽 영역에서 제외 (0.10)
YOLO_CONF_THRESHOLD = 0.10 
# 2. 너무 작은 객체 박스 필터링 기준: 낮출수록 작은 객체까지 포함하여 제외 (0.01)
MIN_BOX_RATIO = 0.01
# 3. 마스크 후처리 시 사용할 모폴로지 커널 크기: 클수록 정제 효과가 강함
MORPHOLOGY_KERNEL_SIZE = 9
# 4. 최종 마스크 경계의 Gaussian Blur 크기: 클수록 경계가 더 부드러움 
GAUSSIAN_BLUR_SIZE = 13

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


def filter_small_boxes(boxes, img_shape, min_ratio=MIN_BOX_RATIO):
    """너무 작은 박스 필터링 (노이즈 제거). 조정 가능한 MIN_BOX_RATIO 사용"""
    H, W = img_shape
    area_img = H * W
    filtered = []
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        # 면적이 전체 이미지의 min_ratio 미만이면 필터링
        if area / area_img > min_ratio:
            filtered.append([float(x1), float(y1), float(x2), float(y2)])
    return filtered


def post_refine(mask: np.ndarray):
    """마스크 후처리: 노이즈 제거, 확대, 가장 큰 연결 영역만 남기기 (벽 영역 추정). MORPHOLOGY_KERNEL_SIZE 사용"""
    mask = mask.astype(np.uint8)
    # 💡 조정 가능한 커널 크기 적용
    kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)

    # 노이즈 제거 (Opening) + 경계 채우기 (Dilate)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 가장 큰 연결 영역만 남기기 (가장 큰 영역을 선택하여 벽 영역을 명확히 함)
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
    return {"status": "ok", "message": "YOLOv8n + MobileSAM Wall Segmentation Server (Tuning Ready)"}


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
    """YOLOv8n으로 객체 감지 → MobileSAM으로 분할 → 객체 마스크 반전 및 후처리로 벽 영역 추출"""
    
    # 모델 로딩 여부 확인
    if det_model is None or sam_model is None:
        logger.error("Segmentation services are unavailable due to model loading failure.")
        return Response(content="Model load failed. Check server startup logs.", status_code=503)

    # 메모리 정리를 위해 변수들을 None으로 초기화합니다.
    img = pil_img = results = boxes = sam_boxes = None 

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

        # 1. YOLOv8n 예측 (객체 감지)
        logger.info("[🔍] YOLOv8n: 객체 감지 중...")
        results = det_model.predict(
            pil_img,
            conf=YOLO_CONF_THRESHOLD, 
            imgsz=640,
            device=device,
            verbose=False,
        )[0]

        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        boxes = filter_small_boxes(xyxy, pil_img.size[::-1])
        
        logger.info(f"[✅] {len(boxes)}개의 유효 객체 박스 발견")

        # 2. 예외 처리: 박스가 없거나 너무 작으면, 벽은 전체 화면 (마스크 100%)
        if not boxes:
            logger.warning("[⚠️] 객체 박스가 없어 전체 이미지(벽) 박스 사용.")
            mask_img = np.ones((h, w), dtype=np.uint8) * 255
        else:
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
                # 4. 마스크 통합 및 **반전** (벽 영역 추출)
                mask_data = res.masks.data.cpu().numpy()
                # 모든 객체들의 통합 마스크 (객체 = 1, 배경 = 0)
                union_objects = (mask_data.sum(axis=0) > 0).astype(np.uint8)
                
                # 💡 객체 마스크를 반전하여 벽(배경) 마스크를 얻습니다. (핵심)
                background_mask = 1 - union_objects
                
                # 5. 후처리 (가장 큰 배경 영역만 남김)
                refined = post_refine(background_mask) 
                mask_img = (refined * 255).astype(np.uint8)
                
                # 6. 경계면 부드럽게 처리 (Smoothing)
                mask_img = cv2.GaussianBlur(mask_img, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
                
                # 🚨 메모리 정리 강화
                del mask_data, union_objects, background_mask, refined
        
        # 7. 원본 크기로 복원
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