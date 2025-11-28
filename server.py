import os
import io
import cv2
import torch
import numpy as np
import gc
import logging
from PIL import Image
from ultralytics import YOLO, SAM, __version__ as ultralytics_version
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
import psutil

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
)

# 📌 상수 설정
YOLO_CONF_THRESHOLD = 0.001
MIN_BOX_RATIO = 0.003
MORPHOLOGY_KERNEL_SIZE = 11
GAUSSIAN_BLUR_SIZE = 21
DEPTH_DIFF_THRESHOLD = 8
TOP_BOTTOM_REMOVE_RATIO = 0.15
MAX_IMAGE_SIZE_PIXELS = 640

# 📌 전역 모델
det_model = None
sam_model = None
device = "cpu"

# 사용할 모델 파일 이름
YOLO_MODEL_NAME = "yolov8n.pt"
SAM_MODEL_NAME = "mobile_sam.pt"


# ⭐ 모델 로드 및 다운로드
@app.on_event("startup")
def load_models_on_startup():
    global det_model, sam_model, device
    logger.info("[🔥] Loading Models...")
    # NOTE: 실제 환경에서는 GPU(cuda)를 사용해야 성능이 나옵니다.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu" 

    # 1. YOLO 모델 로드 및 다운로드
    try:
        if not os.path.exists(YOLO_MODEL_NAME):
            logger.info(f"[⬇️] {YOLO_MODEL_NAME} 파일을 다운로드 중...")
            det_model = YOLO(YOLO_MODEL_NAME)
            # 다운로드 확인을 위한 export (필수 아님)
            det_model.export(format='torchscript', dynamic=True) 
        else:
            det_model = YOLO(YOLO_MODEL_NAME)
        
        det_model.to(device)
        logger.info(f"[✔️] {YOLO_MODEL_NAME} Loaded on {device}")
    except Exception as e:
        logger.error(f"[💥] YOLO Model Load/Download Error: {e}. 경로 및 메모리 확인.")
        det_model = None 

    # 2. SAM 모델 로드 및 다운로드
    try:
        if not os.path.exists(SAM_MODEL_NAME):
            logger.info(f"[⬇️] {SAM_MODEL_NAME} 파일을 다운로드 중...")
            sam_model = SAM(SAM_MODEL_NAME)
        else:
            sam_model = SAM(SAM_MODEL_NAME)

        sam_model.to(device)
        logger.info(f"[✔️] {SAM_MODEL_NAME} Loaded on {device}")
    except Exception as e:
        logger.error(f"[💥] SAM Model Load/Download Error: {e}. 경로 및 메모리 확인.")
        sam_model = None 

    if det_model is None or sam_model is None:
        logger.warning("[⚠️] 모델 로드 실패: '/health' 엔드포인트에서 상태 확인 필요.")


# 🧰 이미지 로드 및 리사이징
def pil_from_bytes(file_bytes: bytes, mode="RGB") -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert(mode)
        w, h = img.size
        
        # 깊이 맵(Luminance)은 스케일링을 건너뛰는 것이 좋지만,
        # 여기서는 통일성을 위해 MAX_IMAGE_SIZE_PIXELS에 맞춰 리사이징합니다.
        if max(w, h) > MAX_IMAGE_SIZE_PIXELS:
            ratio = MAX_IMAGE_SIZE_PIXELS / max(w, h)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
            logger.warning(f"[⚠️] 이미지 크기를 {w}x{h}에서 {new_size[0]}x{new_size[1]}로 축소했습니다.")
            
        return img

    except Exception as e:
        logger.error(f"Image Load Error: {e}")
        return None

# 🧱 마스크 후처리
def post_refine(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask

    # 가장 큰 영역만 남기기
    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 1, thickness=cv2.FILLED)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean


# 🪓 천장 + 바닥 제거 (AR 수직면을 가정)
def remove_top_bottom(mask, ratio=TOP_BOTTOM_REMOVE_RATIO):
    h = mask.shape[0]
    cut = int(h * ratio)
    mask[:cut, :] = 0
    mask[h-cut:, :] = 0
    return mask


# 🧱 수직면(벽)만 남기기 (깊이 변화가 적은 영역)
def filter_vertical_surfaces(depth_map, threshold=DEPTH_DIFF_THRESHOLD):
    # 깊이 맵의 그래디언트(Sobel) 계산
    depth_map = depth_map.astype(np.float32)
    dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(dx, dy)
    
    # 그래디언트 크기가 임계값보다 작은 영역 = 깊이 변화가 적어 수직면일 확률 높음
    vertical_strong_mask = (magnitude < threshold).astype(np.uint8) 
    
    return vertical_strong_mask

# 🛑 전경 객체 제거 (Sobel 기반) (현재 사용하지 않음)
def create_depth_occlusion_mask(depth_map: np.ndarray, threshold=DEPTH_DIFF_THRESHOLD) -> np.ndarray:
    if depth_map is None:
        return None
    depth_map = depth_map.astype(np.float32)
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    occl = (magnitude > threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(occl, kernel, iterations=2)


# 🚪 Wall Mask API
@app.post("/segment_wall_mask")
async def generate_mask(
        image: UploadFile = File(...),
        depth: UploadFile = File(...)
):
    global det_model, sam_model

    if det_model is None or sam_model is None:
        logger.error("Models not loaded.")
        return Response(content="Model load failed. Check server startup logs for file/memory issues.", status_code=503)
    
    # 📌 이미지 가져오기
    img_pil = pil_from_bytes(await image.read())
    depth_bytes = await depth.read()
    depth_pil = pil_from_bytes(depth_bytes, mode="L") # Grayscale (L)로 로드
    
    if img_pil is None or depth_pil is None:
        return Response(content="Invalid Image or Depth File.", status_code=400)

    img = np.array(img_pil)
    depth_map = np.array(depth_pil)
    h, w, _ = img.shape
    
    # 🧱 YOLO 감지 (모든 객체 bbox)
    logger.info("[🔍] YOLOv8n: 객체 감지 중...")
    try:
        det = det_model(img, conf=YOLO_CONF_THRESHOLD, device=device, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else []
    except Exception as e:
        logger.error(f"[💥] YOLO inference failed: {e}")
        boxes = []
        
    del det
    gc.collect()

    if len(boxes) == 0:
        logger.warning("[⚠️] 객체 박스가 없어 전체 화면(벽) 마스크 반환.")
        final_mask = np.ones((h, w), dtype=np.uint8)
    else:
        # 🎯 MobileSAM predict (모든 객체 분할하여 합집합 계산)
        logger.info(f"[🎨] MobileSAM: {len(boxes)}개 객체 분할 중...")
        try:
            sam_results = sam_model.predict(img, bboxes=boxes, device=device, verbose=False)[0]

            if sam_results.masks is None or sam_results.masks.data is None:
                logger.warning("[⚠️] MobileSAM 분할 실패. 전체 화면(벽) 마스크 반환.")
                initial_wall_mask = np.ones((h, w), dtype=np.uint8)
            else:
                # 모든 객체 마스크의 합집합 (Union) 계산
                mask_data = sam_results.masks.data.cpu().numpy()
                union_objects_mask = (mask_data.sum(axis=0) > 0).astype(np.uint8)
                del mask_data 
                
                # 벽 마스크 = 1 - 객체 합집합 마스크
                initial_wall_mask = 1 - union_objects_mask
            
            del sam_results
            gc.collect()
                
            # 🧼 후처리 (SAM 결과 정제 및 천장/바닥 제거)
            wall_mask_after_sam = post_refine(initial_wall_mask.copy())
            wall_mask_after_sam = remove_top_bottom(wall_mask_after_sam, ratio=TOP_BOTTOM_REMOVE_RATIO)
                
            # 📌 깊이 정보 활용: 수직면 필터링
            logger.info("[📏] 깊이 맵으로 수직면 필터링 중...")
            vertical_mask = filter_vertical_surfaces(depth_map, threshold=DEPTH_DIFF_THRESHOLD)
            
            # 3. SAM 기반 벽 마스크와 깊이 기반 수직면 마스크의 교집합
            final_mask = cv2.bitwise_and(wall_mask_after_sam, vertical_mask)
            
            del initial_wall_mask, wall_mask_after_sam, vertical_mask
            gc.collect()

        except Exception as e:
            logger.error(f"[💥] SAM inference failed: {e}")
            final_mask = np.ones((h, w), dtype=np.uint8)

    # ----------------------------------------------------
    # 💧 경계 부드럽게 (블러)
    final_mask = cv2.GaussianBlur(final_mask.astype(np.float32), (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)

    # 🎁 PNG 반환 (0-1 float 마스크를 0-255 8비트 이미지로 인코딩)
    _, png = cv2.imencode(".png", (final_mask * 255).astype(np.uint8))
    
    gc.collect()
    
    return Response(png.tobytes(), media_type="image/png")


# 서버 상태
@app.get("/")
async def root():
    return {"status": "ok", "message": "Wall Detection Server Ready"}


@app.get("/health")
async def health():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    gc.collect()
    return {
        "status": "healthy",
        "models_loaded": det_model is not None and sam_model is not None,
        "memory_mb": round(memory_mb, 2)
    }