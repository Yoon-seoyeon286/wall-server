import os
import io
import cv2
import torch
import numpy as np
import gc
import logging
from PIL import Image
from ultralytics import YOLO, SAM
from fastapi import FastAPI, File, UploadFile, Response, Form
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

# 🌟 설정값
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


# ⭐ 모델 로드
@app.on_event("startup")
def load_models_on_startup():
    global det_model, sam_model, device
    logger.info("[🔥] Loading Models...")
    device = "cpu"

    try:
        if os.path.exists("yolov8s.pt"):
            det_model = YOLO("yolov8s.pt")
            det_model.to(device)
            logger.info("[✔️] YOLOv8s Loaded")
        else:
            logger.error("[❌] yolov8s.pt Not Found")

        if os.path.exists("mobile_sam.pt"):
            sam_model = SAM("mobile_sam.pt")
            sam_model.to(device)
            logger.info("[✔️] MobileSAM Loaded")
        else:
            logger.error("[❌] mobile_sam.pt Not Found")

    except Exception as e:
        logger.error(f"[💥] Model Load Error: {e}")


# 🧰 이미지 로드
def pil_from_bytes(file_bytes: bytes, mode="RGB") -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert(mode)
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

    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 1, thickness=cv2.FILLED)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean


# 🪓 천장 + 바닥 제거
def remove_top_bottom(mask, ratio=TOP_BOTTOM_REMOVE_RATIO):
    h = mask.shape[0]
    cut = int(h * ratio)
    mask[:cut, :] = 0
    mask[h-cut:, :] = 0
    return mask


# 🧱 수직면(벽)만 남기기
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


# 🛑 전경 객체 제거
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

    # 📌 이미지 가져오기
    img_pil = pil_from_bytes(await image.read())
    depth_pil = pil_from_bytes(await depth.read(), mode="L")

    if img_pil is None or depth_pil is None:
        return {"error": "Invalid Image or Depth"}

    img = np.array(img_pil)
    depth_map = np.array(depth_pil)

    # 🧱 YOLO 감지 (벽 후보 bbox)
    det = det_model(img, conf=YOLO_CONF_THRESHOLD)[0]
    boxes = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else []

    if len(boxes) == 0:
        return {"error": "No Wall Detected"}

    # 🎯 SAM 마스크 선택
    x1, y1, x2, y2 = boxes[0].astype(int)
    mask = sam_model.predict(img, [x1, y1, x2, y2])[0]
    mask = post_refine(mask)

    # ⛓ 깊이 기반 제거
    occl = create_depth_occlusion_mask(depth_map)
    if occl is not None:
        mask = mask * (1 - occl)

    # 🔨 천장/바닥 제거
    mask = remove_top_bottom(mask)

    # 🎯 수직 벽만 유지
    mask = mask * filter_vertical_surfaces(depth_map)

    # 💧 경계 부드럽게
    mask = cv2.GaussianBlur(mask.astype(np.float32), (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)

    # 🎁 PNG 반환
    _, png = cv2.imencode(".png", (mask * 255).astype(np.uint8))
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
