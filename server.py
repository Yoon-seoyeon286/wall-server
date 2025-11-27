import os
import io
import cv2
import torch
import numpy as np
import gc
import logging
import math
from PIL import Image
from ultralytics import YOLO, SAM
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
import psutil # 메모리 사용량 추적을 위해 추가

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
# 1. YOLOv8 객체 감지 민감도
YOLO_CONF_THRESHOLD = 0.001 
# 2. 너무 작은 객체 박스 필터링 기준
MIN_BOX_RATIO = 0.003
# 3. 마스크 후처리 시 사용할 모폴로지 커널 크기
MORPHOLOGY_KERNEL_SIZE = 11
# 4. 최종 마스크 경계의 Gaussian Blur 크기
GAUSSIAN_BLUR_SIZE = 21
# 5. 깊이 맵 기반 객체 제거 민감도
DEPTH_DIFF_THRESHOLD = 8 
# 6. 메모리 보호를 위한 최대 이미지 크기 제한
MAX_IMAGE_SIZE_PIXELS = 640 
# 7. 천장 영역으로 간주하고 제거할 상단 비율 (0.0 - 1.0)
CEILING_EXCLUSION_RATIO = 0.15 
# 8. 바닥 영역으로 간주하고 제거할 하단 비율 (0.0 - 1.0)
FLOOR_EXCLUSION_RATIO = 0.20
# 9. 문/창문 등 좁은 영역을 제거할 최소 수직 너비 비율
MIN_WALL_WIDTH_RATIO = 0.40


# 전역 변수
det_model = None  # YOLOv8s
sam_model = None  # MobileSAM
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
            
        logger.info("[✅] MiDaS 깊이 모델은 메모리 문제로 인해 제거되었으며, Unity 깊이 데이터만 사용합니다.")

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


def post_filter_vertical_plane(
    mask: np.ndarray, 
    ceiling_ratio: float, 
    floor_ratio: float, 
    min_wall_width_ratio: float
) -> np.ndarray:
    """
    천장/바닥을 제거하고 벽 영역만 남기는 수직 필터링 및 복원 로직.
    """
    H, W = mask.shape
    
    # 1. 휴리스틱 필터링: 천장과 바닥을 강제로 제거
    cut_mask = mask.copy()
    ceiling_pixels = int(H * ceiling_ratio)
    floor_pixels = int(H * floor_ratio)
    
    # 상단(천장)과 하단(바닥) 영역을 0으로 설정
    if ceiling_pixels > 0:
        cut_mask[:ceiling_pixels, :] = 0
    if floor_pixels > 0:
        cut_mask[H - floor_pixels:, :] = 0
    
    logger.info(f"[✂️] 상단 {ceiling_pixels}px, 하단 {floor_pixels}px 제거 완료.")

    # 2. 제거 후 남은 마스크에서 가장 큰 수직 영역만 추출 (문/창문 제거)
    cnts, _ = cv2.findContours(cut_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return np.zeros_like(mask)

    # 가장 큰 영역 찾기
    largest_cnt = max(cnts, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(largest_cnt)
    
    # 가로 길이가 너무 좁은 영역(문, 창문 등)은 제거
    if w_box / W < min_wall_width_ratio:
        logger.warning(f"[⚠️] 가장 큰 영역 너비 비율 ({w_box/W:.2f})이 너무 작아 벽으로 인식하지 않습니다.")
        return np.zeros_like(mask)
    
    # 3. 벽 마스크 복원: 잘린 상하단 픽셀 채우기
    # 벽의 바운딩 박스를 기준으로 원래 이미지 크기까지 확장하여 채움
    restored_mask = np.zeros_like(mask)
    
    # 벽의 좌우 경계만 유지하고, 상하 경계를 원래 이미지 상단/하단까지 확장
    # y_start는 0으로, y_end는 H로 설정하여 벽을 이미지 끝까지 확장
    restored_mask[0:H, x:x + w_box] = 255 
    
    # 확장된 마스크에 원래의 마스크를 AND 연산하여 벽의 형태만 남김
    # 이 연산을 통해 확장된 직사각형 안에서만 픽셀이 채워집니다.
    final_wall_mask = cv2.bitwise_and(mask, restored_mask)
    
    del cut_mask, cnts, largest_cnt, restored_mask
    logger.info("[✅] 수직 필터링 및 벽 영역 복원 완료.")
    return final_wall_mask


def create_depth_occlusion_mask(depth_map: np.ndarray, threshold=DEPTH_DIFF_THRESHOLD) -> np.ndarray:
    """
    Unity 깊이 지도를 사용하여 전경 객체(Occlusion) 마스크 생성.
    """
    if depth_map is None:
        return None
        
    depth_map = depth_map.astype(np.float32)
    
    # Sobel 필터를 사용하여 깊이 맵의 경계(깊이 변화가 큰 부분)를 검출
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


@app.post("/segment_wall_mask")
async def segment_wall_mask(
    rgb_file: UploadFile = File(..., alias="rgb_file"),
    depth_file: UploadFile = File(..., alias="depth_file")
):
    """YOLOv8s+SAM으로 객체 감지/분할 후, Unity 깊이 지도로 최종 가려짐 마스크를 적용하여 벽 영역 추출"""
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    logger.info(f"[🧠] 요청 시작 메모리: {initial_memory:.2f} MB")
    
    if det_model is None or sam_model is None:
        logger.error("Segmentation services are unavailable due to model loading failure.")
        return Response(content="Model load failed. Check server startup logs.", status_code=503)

    pil_img = depth_img_np = depth_occlusion_mask = None 

    try:
        # 1. RGB 이미지 로드 및 메모리 보호를 위한 크기 조정
        rgb_bytes = await rgb_file.read()
        pil_img = np_from_upload(rgb_bytes, mode="RGB")
        del rgb_bytes
        
        if pil_img is None:
            logger.error("RGB file could not be loaded.")
            return Response(content="Invalid RGB image file.", status_code=400)
            
        original_size = pil_img.size
        w, h = pil_img.size
        
        # 최대 이미지 크기 제한 적용
        if max(w, h) > MAX_IMAGE_SIZE_PIXELS:
            ratio = MAX_IMAGE_SIZE_PIXELS / max(w, h)
            new_size = tuple(int(dim * ratio) for dim in pil_img.size)
            pil_img = pil_img.resize(new_size, Image.LANCZOS) 
            w, h = pil_img.size
            logger.warning(f"[⚠️] 원본 이미지 {original_size[0]}x{original_size[1]}를 메모리 보호를 위해 {w}x{h}로 축소했습니다.")

        logger.info(f"[📸] 처리 이미지: {w}x{h}")
        
        
        # 2. Unity 깊이 지도 로드 및 전처리
        depth_bytes = await depth_file.read()
        
        if not depth_bytes or len(depth_bytes) < 100: 
            logger.warning("[❌] 클라이언트 깊이 파일이 비어 있거나 유효하지 않습니다. 2D AI 마스크만 사용합니다.")
            depth_img_np = None 
        else:
            # L (흑백 8비트) 대신 16비트 깊이 데이터도 처리할 수 있도록 모드를 조정합니다.
            try:
                # 클라이언트가 16비트 흑백 이미지를 보낼 수 있으므로 'I;16'을 시도합니다.
                depth_img_pil = Image.open(io.BytesIO(depth_bytes)) 
                
                # 8비트로 변환 및 리사이즈
                depth_img_np = np.array(depth_img_pil.convert('L').resize((w, h), Image.NEAREST))
                del depth_img_pil
                logger.info("[✅] 클라이언트 깊이 지도 로드 및 8비트 변환 완료.")
            except Exception as depth_e:
                logger.error(f"[❌] 클라이언트 깊이 데이터 파싱 실패: {depth_e}")
                depth_img_np = None
        
        del depth_bytes


        # 3. YOLOv8s + MobileSAM으로 초기 벽 마스크 생성
        # ... (YOLO/SAM 로직은 이전과 동일)
        logger.info("[🔍] YOLOv8s: 객체 감지 중...")
        results = det_model.predict(
            pil_img, conf=YOLO_CONF_THRESHOLD, imgsz=640, device=device, verbose=False,
        )[0]
        
        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        del results 
        
        boxes = filter_small_boxes(xyxy, pil_img.size[::-1])
        del xyxy
        logger.info(f"[✅] {len(boxes)}개의 유효 객체 박스 발견")

        if not boxes:
            logger.warning("[⚠️] 객체 박스가 없어 전체 이미지(벽) 박스 사용.")
            initial_wall_mask = np.ones((h, w), dtype=np.uint8) * 255
        else:
            logger.info("[🎨] MobileSAM: 객체 분할 중...")
            res = sam_model.predict(
                pil_img, bboxes=boxes, device=device, retina_masks=False, verbose=False
            )[0]
            del boxes 
            
            if res.masks is None:
                logger.warning("[⚠️] MobileSAM 분할 실패. 전체 화면 반환.")
                initial_wall_mask = np.ones((h, w), dtype=np.uint8) * 255
                del res
            else:
                mask_data = res.masks.data.cpu().numpy()
                del res 
                
                union_objects = (mask_data.sum(axis=0) > 0).astype(np.uint8)
                del mask_data
                
                background_mask = 1 - union_objects 
                del union_objects
                
                refined_background = post_refine(background_mask) 
                del background_mask
                
                initial_wall_mask = (refined_background * 255).astype(np.uint8)
                del refined_background


        # 4. 깊이 지도를 이용한 최종 객체 제외 마스킹 (Depth Occlusion)
        final_mask_img = initial_wall_mask.copy()
        del initial_wall_mask
        
        if depth_img_np is not None:
            depth_occlusion_mask = create_depth_occlusion_mask(depth_img_np)
            del depth_img_np 
            
            if depth_occlusion_mask is not None:
                wall_from_depth = 1 - depth_occlusion_mask 
                del depth_occlusion_mask
                
                combined_mask = cv2.bitwise_and(final_mask_img, wall_from_depth * 255)
                final_mask_img = combined_mask
                del wall_from_depth, combined_mask
                
                logger.info("[✅] Unity 클라이언트 깊이 데이터로 최종 가려짐 보정 완료.")
            else:
                logger.warning("[⚠️] 깊이 마스크 생성 실패. 2D AI 마스크만 사용합니다.")
        else:
            logger.warning("[⚠️] 깊이 데이터가 유효하지 않아 2D AI 마스크만 사용합니다.")

        # 5. 💡 [복구된 로직] 천장/바닥/문 분리를 위한 수직 필터링 적용
        final_mask_img = post_filter_vertical_plane(
            final_mask_img,
            CEILING_EXCLUSION_RATIO,
            FLOOR_EXCLUSION_RATIO,
            MIN_WALL_WIDTH_RATIO
        )
        # 마스크를 0 또는 255 값으로 정규화 (post_filter_vertical_plane의 출력이 0/255가 아닐 수 있으므로)
        final_mask_img = (final_mask_img > 0).astype(np.uint8) * 255

        # 6. 최종 마스크 정리 및 인코딩
        final_mask_img = cv2.GaussianBlur(final_mask_img, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
        
        if pil_img.size != original_size:
            final_mask_img = cv2.resize(
                final_mask_img, 
                original_size, 
                interpolation=cv2.INTER_LINEAR
            )
        
        del pil_img
        
        _, png = cv2.imencode(".png", final_mask_img)
        del final_mask_img, _ 

        final_png_bytes = png.tobytes()
        del png
        
        gc.collect() 
        final_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"[🧠] 요청 완료 메모리: {final_memory:.2f} MB (변동: {final_memory - initial_memory:.2f} MB)")


        return Response(
            content=final_png_bytes,
            media_type="image/png",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache"
            }
        )

    except Exception as e:
        # 오류 발생 시 메모리 상태 로깅
        error_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logger.critical(f"❌ CRITICAL ERROR (Possible OOM) during segmentation. Current Memory: {error_memory:.2f} MB. Error: {e}", exc_info=True)
        gc.collect()
        # 클라이언트에게 500 Internal Server Error 반환
        return Response(
            content=f"Internal Server Error: Segmentation processing failed.".encode(),
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