import os
import io
import cv2
import torch
import numpy as np
import gc
import logging
from PIL import Image
from ultralytics import SAM, YOLO # YOLO import 추가
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
yolo_model = None  # YOLOv8 World 모델
sam_model = None   # MobileSAM 모델
device = "cpu"


@app.on_event("startup")
def load_models_on_startup():
    """서버 시작 시 YOLOv8s-World + MobileSAM 로드"""
    global yolo_model, sam_model, device
    
    logger.info("[🔥] Starting model loading for YOLOv8 World + MobileSAM...")
    
    # Dockerfile 설정에 따라 'cpu' 또는 'cuda' 자동 감지 및 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[⚙️] Device: {device}")
    
    # 메모리 안정성을 위해, CPU 환경에서는 s-World 대신 n-World 사용을 고려해야 합니다.
    # 여기서는 Dockerfile에서 다운로드된 모델 이름을 사용합니다.
    yolo_checkpoint_path = "yolov8s-world.pt" # Dockerfile에서 다운로드하는 파일명과 일치시켜야 함
    sam_checkpoint_path = "mobile_sam.pt"

    try:
        # 1. YOLOv8 World 모델 로드
        if not os.path.exists(yolo_checkpoint_path):
             logger.error(f"[❌] YOLOv8 World checkpoint not found at: {yolo_checkpoint_path}")
        else:
            yolo_model = YOLO(yolo_checkpoint_path)
            yolo_model.to(device)
            logger.info(f"[✅] YOLOv8 World ({yolo_checkpoint_path}) loaded.")
        
        # 2. MobileSAM 로드
        if not os.path.exists(sam_checkpoint_path):
             logger.error(f"[❌] MobileSAM checkpoint not found at: {sam_checkpoint_path}")
        else:
            sam_model = SAM(sam_checkpoint_path)
            sam_model.to(device)
            logger.info("[✅] MobileSAM loaded.")
        
    except Exception as e:
        logger.error(f"[❌] FATAL Model loading failed: {e}", exc_info=True)
        # 로딩 실패 시 전역 변수는 None으로 유지됩니다.


def np_from_upload(file_bytes: bytes) -> Image.Image:
    """바이트를 PIL Image로 변환"""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def detect_walls_yolo(image: Image.Image, text_prompt: str = "wall"):
    """YOLOv8 World로 벽 감지 및 수동 필터링"""
    
    # YOLOv8의 예측. classes 인자 사용 시 라이브러리 내부 오류(ValueError)가 발생하므로 사용하지 않습니다.
    results = yolo_model.predict(
        source=image,
        conf=0.25, # 낮은 confidence로 감도 증가
        iou=0.7,
        verbose=False,
        device=yolo_model.device # 모델이 로드된 device 사용
    )[0]
    
    # 1. 'wall' 클래스 인덱스 찾기
    wall_class_index = yolo_model.names.get(text_prompt)
    if wall_class_index is None:
        logger.warning(f"[⚠️] YOLO model does not have class '{text_prompt}'. Returning empty boxes.")
        return np.array([]), np.array([])
        
    # 2. 결과에서 'wall'에 해당하는 박스만 수동 필터링
    wall_mask = (results.boxes.cls.cpu().numpy() == wall_class_index)
    
    # 3. 박스와 점수 추출
    boxes = results.boxes.xyxy.cpu().numpy()[wall_mask]
    scores = results.boxes.conf.cpu().numpy()[wall_mask]
    
    # 예측 결과를 사용한 후 Torch 텐서 객체를 명시적으로 삭제 (메모리 정리)
    del results, wall_mask
    
    return boxes, scores


def expand_mask(mask, iterations=25):
    """마스크 확장: 마스크 경계를 부드럽게 만들고 누락된 틈을 메웁니다."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, kernel, iterations=iterations)


# ----------------------------------------------------------------------
# FastAPI 엔드포인트
# ----------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "message": "YOLOv8 World + MobileSAM Wall Segmentation Server"}


@app.get("/health")
async def health():
    import psutil
    process = psutil.Process()
    # RSS (상주 메모리) 확인
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    # GC 강제 실행 (Health Check 시 메모리 정리)
    gc.collect()
    
    return {
        "status": "healthy",
        "models_loaded": yolo_model is not None and sam_model is not None,
        "device": device,
        "memory_mb": round(memory_mb, 2)
    }


@app.post("/segment_wall_mask")
async def segment_wall_mask(file: UploadFile = File(...)):
    """YOLOv8 World로 벽 찾고 → MobileSAM으로 정밀 분할"""
    
    # 모델 로딩 여부 확인 (startup 이벤트에서 실패했을 경우)
    if yolo_model is None or sam_model is None:
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
        
        # 1️⃣ YOLOv8 World로 벽 감지
        logger.info("[🔍] YOLOv8 World: 벽 감지 중...")
        boxes, scores = detect_walls_yolo(img, text_prompt="wall")
        
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
            retina_masks=False 
        )[0]
        
        if results.masks is None or len(results.masks.data) == 0:
            logger.warning("[⚠️] MobileSAM 실패. 전체 화면 사용.")
            mask = np.ones((h, w), dtype=np.uint8)
        else:
            # 모든 마스크 합치기 
            masks_tensor = results.masks.data.cpu()
            masks = masks_tensor.numpy()
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
        
        # 🚨 메모리 정리 강화 (이 부분이 핵심)
        # 모든 큰 변수 명시적 삭제
        del img, results, mask, mask_img, file_bytes, boxes, scores
        
        # 파이토치 캐시 정리 (GPU가 없더라도 안정성 확보를 위해 포함)
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        
        # 파이썬 가비지 컬렉터 강제 실행
        gc.collect() 
        
        # 응답을 위해 최종 PNG 바이트를 얻은 후, 임시 변수도 삭제
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
        # 에러 발생 시에도 메모리 정리 후 500 오류 반환
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