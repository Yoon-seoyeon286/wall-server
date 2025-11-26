import os
import io
import cv2
import torch
import numpy as np
import gc
import logging
from PIL import Image
from ultralytics import SAM # YOLOv8 import 제거
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 전역 변수
sam_model = None   # MobileSAM 모델만 사용
device = "cpu"


@app.on_event("startup")
def load_models_on_startup():
    """서버 시작 시 MobileSAM만 로드"""
    global sam_model, device
    
    logger.info("[🔥] Starting model loading for MobileSAM (Standalone Mode)...")
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[⚙️] Device: {device}")
    
    sam_checkpoint_path = "mobile_sam.pt"

    try:
        # 1. MobileSAM 로드
        if not os.path.exists(sam_checkpoint_path):
             logger.error(f"[❌] MobileSAM checkpoint not found at: {sam_checkpoint_path}")
        else:
            # SAM 모델 로드 시 YOLO를 참조하지 않도록 SAM만 로드
            sam_model = SAM(sam_checkpoint_path)
            sam_model.to(device)
            logger.info("[✅] MobileSAM loaded.")
        
    except Exception as e:
        logger.error(f"[❌] FATAL Model loading failed: {e}", exc_info=True)


def np_from_upload(file_bytes: bytes) -> Image.Image:
    """바이트를 PIL Image로 변환"""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def expand_mask(mask, iterations=25):
    """마스크 확장: 마스크 경계를 부드럽게 만들고 누락된 틈을 메웁니다."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, kernel, iterations=iterations)


# ----------------------------------------------------------------------
# FastAPI 엔드포인트
# ----------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "message": "MobileSAM Standalone Server"}


@app.get("/health")
async def health():
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    gc.collect()
    
    return {
        "status": "healthy",
        "models_loaded": sam_model is not None,
        "device": device,
        "memory_mb": round(memory_mb, 2)
    }


@app.post("/segment_wall_mask")
async def segment_wall_mask(file: UploadFile = File(...)):
    """MobileSAM으로 전체 이미지에 대한 마스크 분할"""
    
    # 모델 로딩 여부 확인
    if sam_model is None:
        logger.error("Segmentation services are unavailable due to model loading failure.")
        return Response(content="Model load failed. Check server startup logs.", status_code=503)

    try:
        file_bytes = await file.read()
        if not file_bytes:
            return Response(content="File is empty.", status_code=400)
        
        img = np_from_upload(file_bytes)
        original_size = img.size
        
        # 리사이즈 (속도 향상 및 연산량 감소)
        max_size = 480 # <-- 이미지 최대 크기를 480으로 제한
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
        
        w, h = img.size
        logger.info(f"[📸] 이미지: {w}x{h}")
        
        # 1️⃣ MobileSAM 입력 박스: 이미지 전체 영역을 사용
        boxes = np.array([[0, 0, w, h]])
        sam_boxes = boxes.tolist()
        
        logger.info("[🎨] MobileSAM: 전체 영역 분할 중...")
        
        # ultralytics SAM predict (전체 영역 박스 입력)
        results = sam_model.predict(
            img,
            bboxes=sam_boxes,
            device=device,
            verbose=False,
            retina_masks=False 
        )[0]
        
        if results.masks is None or len(results.masks.data) == 0:
            logger.warning("[⚠️] MobileSAM 실패. 전체 화면 사용.")
            mask = np.ones((h, w), dtype=np.uint8)
        else:
            # 모든 마스크 합치기 (전체 영역을 분할할 때 SAM은 여러 개의 마스크를 반환할 수 있음)
            masks_tensor = results.masks.data.cpu()
            masks = masks_tensor.numpy()
            
            # 가장 큰 마스크만 선택하거나, 모든 마스크 합치기 (여기서는 모든 마스크 합치기 유지)
            mask = (masks.sum(axis=0) > 0).astype(np.uint8)
            
            # 확장
            mask = expand_mask(mask)

            # 명시적으로 텐서 삭제 (메모리 정리)
            del masks_tensor, masks
        
        # 원본 크기로 복원
        if img.size != original_size:
            mask_img = (mask * 255).astype(np.uint8)
            mask_img = cv2.resize(
                mask_img, 
                original_size, 
                interpolation=cv2.INTER_LINEAR
            )
        else:
            mask_img = (mask * 255).astype(np.uint8)
        
        # 통계
        wall_pixels = np.sum(mask_img > 0)
        total_pixels = mask_img.shape[0] * mask_img.shape[1]
        coverage = (wall_pixels / total_pixels) * 100
        
        logger.info(f"[✅] Coverage: {coverage:.1f}% ({wall_pixels}/{total_pixels} pixels)")
        
        # 마스크 커버리지가 너무 낮으면 전체 화면을 마스크로 간주
        if coverage < 5.0:
            logger.warning(f"[⚠️] Coverage 너무 낮음. 전체 화면 사용.")
            mask_img = np.ones_like(mask_img) * 255
        
        # PNG 인코딩
        _, png = cv2.imencode(".png", mask_img)
        
        # 🚨 메모리 정리 강화 (필수)
        del img, results, mask, mask_img, file_bytes, boxes
        
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
        return Response(content=f"Internal Server Error: {e}".encode(), status_code=500)


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