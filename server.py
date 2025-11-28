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


# ⭐ 모델 로드 (메모리 최적화를 위해 YOLOv8n 권장)
@app.on_event("startup")
def load_models_on_startup():
    global det_model, sam_model, device
    logger.info("[🔥] Loading Models...")
    device = "cpu"

    # 🚨 메모리 안정성 확보를 위해 YOLOv8n.pt 사용을 강력히 권장합니다.
    YOLO_MODEL_NAME = "yolov8n.pt" 
    SAM_MODEL_NAME = "mobile_sam.pt"

    try:
        if os.path.exists(YOLO_MODEL_NAME):
            det_model = YOLO(YOLO_MODEL_NAME)
            det_model.to(device)
            logger.info(f"[✔️] {YOLO_MODEL_NAME} Loaded")
        else:
            logger.error(f"[❌] {YOLO_MODEL_NAME} Not Found")

        if os.path.exists(SAM_MODEL_NAME):
            sam_model = SAM(SAM_MODEL_NAME)
            sam_model.to(device)
            logger.info(f"[✔️] {SAM_MODEL_NAME} Loaded")
        else:
            logger.error(f"[❌] {SAM_MODEL_NAME} Not Found")

    except Exception as e:
        logger.error(f"[💥] Model Load Error: {e}")


# 🧰 이미지 로드
def pil_from_bytes(file_bytes: bytes, mode="RGB") -> Image.Image:
    try:
        # 이미지를 로드할 때, 리사이징 로직을 추가하여 메모리 초과 방지
        img = Image.open(io.BytesIO(file_bytes)).convert(mode)
        w, h = img.size
        
        if max(w, h) > MAX_IMAGE_SIZE_PIXELS:
            ratio = MAX_IMAGE_SIZE_PIXELS / max(w, h)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
            logger.warning(f"[⚠️] 이미지 크기를 {w}x{h}에서 {new_size[0]}x{new_size[1]}로 축소했습니다.")
            
        return img

    except Exception as e:
        logger.error(f"Image Load Error: {e}")
        return None

# (나머지 필터링 함수는 동일하게 유지)
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
    
    # 수평 기울기가 수직 기울기보다 약할 때 (즉, 평평한 수직면일 가능성)를 벽으로 가정.
    # 이 로직은 깊이 맵이 매우 깨끗하지 않으면 노이즈가 심할 수 있음.
    vertical_strong_mask = (magnitude < threshold * 2).astype(np.uint8) # 큰 깊이 변화는 객체로 간주
    
    # 깊이 맵의 노이즈가 심할 경우 이 필터는 최종 결과에 악영향을 줌
    return vertical_strong_mask


# 🛑 전경 객체 제거 (Sobel 기반)
def create_depth_occlusion_mask(depth_map: np.ndarray, threshold=DEPTH_DIFF_THRESHOLD) -> np.ndarray:
    if depth_map is None:
        return None
    depth_map = depth_map.astype(np.float32)
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    # 깊이 차이가 임계값 이상인 곳을 객체의 경계(Occlusion)로 간주
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
        return Response(content="Model load failed. Check server startup logs.", status_code=503)
    
    # 📌 이미지 가져오기
    # 이 함수에서 리사이징을 처리합니다.
    img_pil = pil_from_bytes(await image.read())
    # 깊이 맵도 동일한 크기로 리사이징 및 흑백 변환
    depth_bytes = await depth.read()
    depth_pil = pil_from_bytes(depth_bytes, mode="L")
    
    if img_pil is None or depth_pil is None:
        return Response(content="Invalid Image or Depth File.", status_code=400)

    img = np.array(img_pil)
    depth_map = np.array(depth_pil)
    h, w, _ = img.shape
    
    # 🧱 YOLO 감지 (모든 객체 bbox)
    logger.info("[🔍] YOLOv8n: 객체 감지 중...")
    det = det_model(img, conf=YOLO_CONF_THRESHOLD, device=device, verbose=False)[0]
    boxes = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else []
    
    del det
    gc.collect()

    if len(boxes) == 0:
        logger.warning("[⚠️] 객체 박스가 없어 전체 화면(벽) 마스크 반환.")
        final_mask = np.ones((h, w), dtype=np.uint8)
    else:
        # 🎯 MobileSAM predict (모든 객체 분할하여 합집합 계산)
        logger.info(f"[🎨] MobileSAM: {len(boxes)}개 객체 분할 중...")
        sam_results = sam_model.predict(img, bboxes=boxes, device=device, verbose=False)[0]

        if sam_results.masks is None or sam_results.masks.data is None:
            logger.warning("[⚠️] MobileSAM 분할 실패. 전체 화면(벽) 마스크 반환.")
            final_mask = np.ones((h, w), dtype=np.uint8)
        else:
            # 모든 객체 마스크의 합집합 (Union) 계산
            mask_data = sam_results.masks.data.cpu().numpy()
            union_objects_mask = (mask_data.sum(axis=0) > 0).astype(np.uint8)
            del mask_data, sam_results
            
            # 벽 마스크 = 1 - 객체 합집합 마스크
            initial_wall_mask = 1 - union_objects_mask
            
            # 🧼 후처리
            initial_wall_mask = post_refine(initial_wall_mask)
            
            final_mask = initial_wall_mask
            del initial_wall_mask

    # ----------------------------------------------------
    # 🔨 깊이 및 위치 기반 필터링 (디버깅을 위해 주석 해제하여 테스트하세요)
    # ----------------------------------------------------
    
    # 1. 🛑 깊이 기반 전경 객체 제거 (occlusion)
    occl = create_depth_occlusion_mask(depth_map)
    if occl is not None:
        # final_mask *= (1 - occl) 
        pass # 현재는 주석 처리하여 안정성 우선

    # 2. 🪓 천장/바닥 제거
    # final_mask = remove_top_bottom(final_mask)
    
    # 3. 🎯 수직 벽만 유지 (가장 강력한 필터)
    # vertical_surface_mask = filter_vertical_surfaces(depth_map)
    # final_mask *= vertical_surface_mask
    
    # ----------------------------------------------------
    
    # 💧 경계 부드럽게
    final_mask = cv2.GaussianBlur(final_mask.astype(np.float32), (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)

    # 🎁 PNG 반환
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