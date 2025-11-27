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
YOLO_CONF_THRESHOLD = 0.001  # YOLOv8 객체 감지 민감도
MIN_BOX_RATIO = 0.003  # 너무 작은 객체 박스 필터링 기준
MORPHOLOGY_KERNEL_SIZE = 11  # 마스크 후처리 시 사용할 모폴로지 커널 크기
GAUSSIAN_BLUR_SIZE = 21  # 최종 마스크 경계의 Gaussian Blur 크기
DEPTH_DIFF_THRESHOLD = 8  # 깊이 맵 기반 객체 제거 민감도
MAX_IMAGE_SIZE_PIXELS = 640  # 메모리 보호를 위한 최대 이미지 크기 제한

# 전역 변수
det_model = None  # YOLOv8s
sam_model = None  # MobileSAM
device = "cpu"


@app.on_event("startup")
def load_models_on_startup():
    """서버 시작 시 YOLOv8s + MobileSAM 로드 (MiDaS 제거)"""
    global det_model, sam_model, device

    logger.info("[🔥] Starting model loading for YOLOv8s + MobileSAM (MiDaS Removed)...")

    device = "cpu"
    logger.info(f"[⚙️] Device: {device}")

    yolo_checkpoint_path = "yolov8s.pt"
    sam_checkpoint_path = "mobile_sam.pt"

    try:
        if not os.path.exists(yolo_checkpoint_path):
            logger.error(f"[❌] YOLOv8s checkpoint not found at: {yolo_checkpoint_path}")
        else:
            det_model = YOLO(yolo_checkpoint_path)
            det_model.to(device)
            logger.info("[✅] YOLOv8s loaded.")

        if not os.path.exists(sam_checkpoint_path):
            logger.error(f"[❌] MobileSAM checkpoint not found at: {sam_checkpoint_path}")
        else:
            sam_model = SAM(sam_checkpoint_path)
            sam_model.to(device)
            logger.info("[✅] MobileSAM loaded.")

        logger.info("[ℹ️] MiDaS 깊이 모델 제거됨 (Unity 깊이 데이터만 사용).")

    except Exception as e:
        logger.error(f"[❌] FATAL Model loading failed: {e}", exc_info=True)


def np_from_upload(file_bytes: bytes, mode="RGB") -> Image.Image:
    """바이트를 PIL Image로 변환"""
    try:
        return Image.open(io.BytesIO(file_bytes)).convert(mode)
    except Exception as e:
        logger.error(f"Failed to open image from bytes: {e}")
        return None


def filter_small_boxes(boxes, img_shape, min_ratio=MIN_BOX_RATIO):
    """너무 작은 박스 필터링 (노이즈 제거)."""
    H, W = img_shape
    area_img = H * W
    filtered = []
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area / area_img > min_ratio:
            filtered.append([float(x1), float(y1), float(x2), float(y2)])
    return filtered


def post_refine(mask: np.ndarray):
    """마스크 후처리: 노이즈 제거, 확대, 가장 큰 연결 영역만 남기기 (벽 영역 추정)."""
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


def create_depth_occlusion_mask(depth_map: np.ndarray, threshold=DEPTH_DIFF_THRESHOLD) -> np.ndarray:
    """Unity 깊이 지도를 사용하여 전경 객체(Occlusion) 마스크 생성."""
    if depth_map is None:
        return None

    depth_map = depth_map.astype(np.float32)

    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(grad_x, grad_y)
    del grad_x, grad_y

    occlusion_mask = (magnitude > threshold).astype(np.uint8)
    del magnitude

    kernel = np.ones((5, 5), np.uint8)
    occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=2)

    logger.info(f"[✅] Unity 깊이 데이터로 전경 객체 마스크 생성 완료 (Threshold: {threshold}).")
    return occlusion_mask


@app.get("/")
async def root():
    return {"status": "ok", "message": "YOLOv8s + MobileSAM + Unity Depth Integration Server"}


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
