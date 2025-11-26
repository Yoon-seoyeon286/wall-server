import os
import io
import cv2
import torch
import numpy as np
import gc
import logging
from PIL import Image
from ultralytics import SAM
# Grounding DINO Lite는 transformers를 통해 IDEA-Research/grounding-dino-tiny 모델 사용
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
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
grounding_dino_processor = None
grounding_dino_model = None
sam_model = None
device = "cpu"


@app.on_event("startup")
def load_models_on_startup():
    """서버 시작 시 Grounding DINO Lite + MobileSAM 로드"""
    global grounding_dino_processor, grounding_dino_model, sam_model, device
    
    logger.info("[🔥] Starting model loading for Grounding DINO Lite + MobileSAM...")
    
    # Dockerfile이 CPU 전용이므로, 명시적으로 'cpu' 사용
    device = "cpu" 
    logger.info(f"[⚙️] Device: {device}")
    
    try:
        # 1. Grounding DINO Lite 로드
        model_id = "IDEA-Research/grounding-dino-tiny"
        
        # 모델 로드 시 cache_dir 명시 (권한 문제 방지)
        grounding_dino_processor = AutoProcessor.from_pretrained(model_id, cache_dir="./cache")
        grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir="./cache")
        grounding_dino_model.to(device)
        
        logger.info("[✅] Grounding DINO Lite loaded.")
        
        # 2. MobileSAM 로드
        sam_checkpoint_path = "mobile_sam.pt"
        if not os.path.exists(sam_checkpoint_path):
             logger.error(f"[❌] MobileSAM checkpoint not found at: {sam_checkpoint_path}")
        else:
            # ultralytics의 SAM 래퍼 사용
            sam_model = SAM(sam_checkpoint_path)
            sam_model.to(device)
            logger.info("[✅] MobileSAM loaded.")
        
    except Exception as e:
        logger.error(f"[❌] FATAL Model loading failed: {e}", exc_info=True)
        # 로딩 실패 시 전역 변수는 None으로 유지됩니다.


def np_from_upload(file_bytes: bytes) -> Image.Image:
    """바이트를 PIL Image로 변환"""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def detect_walls_grounding_dino(image: Image.Image, text_prompt: str = "wall"):
    """Grounding DINO로 벽 감지"""
    # 이미지 크기 정규화 (Grounding DINO 입력 요구사항)
    inputs = grounding_dino_processor(
        images=image,
        text=text_prompt,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = grounding_dino_model(**inputs)
    
    # 결과 후처리
    # box_threshold를 0.3으로 낮춰서 감도를 높입니다.
    results = grounding_dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,  # 낮은 threshold (더 많이 감지)
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]  # (height, width)
    )[0]
    
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    
    return boxes, scores


def expand_mask(mask, iterations=25):
    """마스크 확장"""
    # AR 환경에서 마스크를 벽에 완전히 밀착시키기 위해 확장(dilate) 사용
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, kernel, iterations=iterations)


# ----------------------------------------------------------------------
# FastAPI 엔드포인트
# ----------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "message": "Grounding DINO Lite + MobileSAM Server"}


@app.get("/health")
async def health():
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return {
        "status": "healthy",
        "models_loaded": grounding_dino_model is not None and sam_model is not None,
        "device": device,
        "memory_mb": round(memory_mb, 2)
    }


@app.post("/segment_wall_mask")
async def segment_wall_mask(file: UploadFile = File(...)):
    """Grounding DINO로 벽 찾고 → MobileSAM으로 정밀 분할"""
    
    # 모델 로딩 여부 확인 (startup 이벤트에서 실패했을 경우)
    if grounding_dino_model is None or sam_model is None:
        logger.error("Segmentation services are unavailable due to model loading failure.")
        return Response(content="Model load failed. Check server startup logs.", status_code=503)

    try:
        file_bytes = await file.read()
        if not file_bytes:
            return Response(content="File is empty.", status_code=400)
        
        img = np_from_upload(file_bytes)
        original_size = img.size
        
        # 리사이즈 (속도 향상 및 DINO Lite 입력 크기 맞추기)
        max_size = 640
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
        
        w, h = img.size
        logger.info(f"[📸] 이미지: {w}x{h}")
        
        # 1️⃣ Grounding DINO로 벽 감지
        logger.info("[🔍] Grounding DINO: 벽 감지 중...")
        boxes, scores = detect_walls_grounding_dino(img, text_prompt="wall")
        
        if len(boxes) == 0:
            logger.warning("[⚠️] 벽을 찾지 못했습니다. 전체 이미지를 박스로 사용.")
            boxes = np.array([[0, 0, w, h]])
        else:
            logger.info(f"[✅] {len(boxes)}개의 벽 후보 발견 (최고 confidence: {scores[0]:.2f})")
        
        # 2️⃣ MobileSAM으로 정밀 분할
        logger.info("[🎨] MobileSAM: 정밀 분할 중...")
        
        sam_boxes = boxes.tolist()
        
        # ultralytics SAM predict
        results = sam_model.predict(
            img,
            bboxes=sam_boxes,
            device=device,
            verbose=False,
            retina_masks=False # 일반 마스크 출력
        )[0]
        
        if results.masks is None or len(results.masks.data) == 0:
            logger.warning("[⚠️] MobileSAM 실패. 전체 화면 사용.")
            mask = np.ones((h, w), dtype=np.uint8)
        else:
            # 모든 마스크 합치기 (여러 벽이 있을 수 있음)
            masks = results.masks.data.cpu().numpy()
            mask = (masks.sum(axis=0) > 0).astype(np.uint8)
            
            # 확장
            mask = expand_mask(mask)
        
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
        
        # 메모리 정리 (매우 중요)
        del img, results, mask, mask_img, file_bytes, boxes, scores
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        
        return Response(
            content=png.tobytes(),
            media_type="image/png",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache"
            }
        )
    
    except Exception as e:
        logger.error(f"❌ ERROR in segmentation processing: {e}", exc_info=True)
        # 에러 발생 시 500 오류와 함께 상세 메시지 반환
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