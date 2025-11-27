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
import psutil  # 메모리 사용량 추적을 위해 추가

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

# ==============================================================================
# 💡 [조정 가능한 설정] - Wall/Object Estimation Parameters
# ==============================================================================
YOLO_CONF_THRESHOLD = 0.001  # YOLO 객체 감지 민감도
MIN_BOX_RATIO = 0.003  # 너무 작은 박스 제외
MORPHOLOGY_KERNEL_SIZE = 11  # 노이즈 제거 & 연결부 강화
GAUSSIAN_BLUR_SIZE = 21  # 경계 연화
DEPTH_DIFF_THRESHOLD = 8  # 깊이 경계 민감도
MAX_IMAGE_SIZE_PIXELS = 640
TOP_BOTTOM_REMOVE_RATIO = 0.15  # 천장/바닥 제거 비율

# 전역 변수
det_model = None
sam_model = None
device = "cpu"


# ==============================================================================
# 📌 모델 로드
# ==============================================================================
@app.on_event("startup")
def load_models_on_startup():
    global det_model, sam_model, device

    logger.info("[🔥] Loading YOLOv8s + MobileSAM (no MiDaS)...")
    device = "cpu"

    yolo_checkpoint_path = "yolov8s.pt"
    sam_checkpoint_path = "mobile_sam.pt"

    try:
        if os.path.exists(yolo_checkpoint_path):
            det_model = YOLO(yolo_checkpoint_path)
            det_model.to(device)
            logger.info("[✅] YOLOv8s Loaded.")
        else:
            logger.error(f"[❌] Not found: {yolo_checkpoint_path}")

        if os.path.exists(sam_checkpoint_path):
            sam_model = SAM(sam_checkpoint_path)
            sam_model.to(device)
            logger.info("[✅] MobileSAM Loaded.")
        else:
            logger.error(f"[❌] Not found: {sam_checkpoint_path}")

        logger.info("[ℹ️] MiDaS removed (using Unity Depth only).")

    except Exception as e:
        logger.error(f"[❌] Model loading failed: {e}", exc_info=True)


# ==============================================================================
# 📌 Utility : 이미지 로드
# ==============================================================================
def np_from_upload(file_bytes: bytes, mode="RGB") -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert(mode)
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        return None


# ==============================================================================
# 🧹 작은 박스 제거
# ==============================================================================
def filter_small_boxes(boxes, img_shape, min_ratio=MIN_BOX_RATIO):
    H, W = img_shape
    area_img = H * W
    filtered = []
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area / area_img > min_ratio:
            filtered.append([float(x1), float(y1), float(x2), float(y2)])
    return filtered


# ==============================================================================
# 🧱 마스크 후처리 (노이즈 제거 + 가장 큰 영역 유지)
# ==============================================================================
def post_refine(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask

    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 1, thickness=cv2.FILLED)

    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean


# ==============================================================================
# ⛓ 깊이 기반 전경 제거 (Occlusion)
# ==============================================================================
def create_depth_occlusion_mask(depth_map: np.ndarray, threshold=DEPTH_DIFF_THRESHOLD) -> np.ndarray:
    if depth_map is None:
        return None

    depth_map = depth_map.astype(np.float32)

    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(grad_x, grad_y)
    occlusion_mask = (magnitude > threshold).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=2)

    logger.info("[🧱] Occlusion Mask Created.")
    return occlusion_mask


# ==============================================================================
# 🪓 천장 + 바닥 제거
# ==============================================================================
def remove_top_bottom(mask, ratio=TOP_BOTTOM_REMOVE_RATIO):
    h = mask.shape[0]
    cut = int(h * ratio)

    mask[:cut, :] = 0   # 천장 제거
    mask[h-cut:, :] = 0  # 바닥 제거
    return mask


# ==============================================================================
# 🧱 수직 면만 남기기
# ==============================================================================
def filter_vertical_surfaces(depth_map, threshold=DEPTH_DIFF_THRESHOLD):
    depth_map = depth_map.astype(np.float32)
    dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(dx, dy)
    direction_mask = (dy > dx).astype(np.uint8)
    strong_edges = (magnitude > threshold).astype(np.uint8)
    vertical_mask = strong_edges * direction_mask

    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(vertical_mask, kernel, iterations=2)


# ==============================================================================
# 🧾 서버 상태
# ==============================================================================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Wall Detection Server"}


@app.get("/health")
async def health():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    gc.collect()

    return {
        "status": "healthy",
        "models_loaded": det_model is not None and sam_model is not None,
        "device": device,
        "memory_mb": round(memory_mb, 2)
    }
