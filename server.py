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
import torch.hub

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
# 💡 [조정 가능한 설정] - Wall/Object Estimation Parameters
# ==============================================================================
# 1. YOLOv8 객체 감지 민감도: 낮출수록 더 많은 객체를 감지하여 벽 영역에서 제외 
YOLO_CONF_THRESHOLD = 0.05 
# 2. 너무 작은 객체 박스 필터링 기준: 낮출수록 작은 객체까지 포함하여 제외
MIN_BOX_RATIO = 0.01
# 3. 마스크 후처리 시 사용할 모폴로지 커널 크기: 클수록 정제 효과가 강함
MORPHOLOGY_KERNEL_SIZE = 9
# 4. 최종 마스크 경계의 Gaussian Blur 크기: 클수록 경계가 더 부드러움 
GAUSSIAN_BLUR_SIZE = 13
# 5. 깊이 맵 기반 객체 제거 민감도: 이 값보다 깊이 차이가 크면 객체로 간주 (낮출수록 민감)
DEPTH_DIFF_THRESHOLD = 15 # 0-255 스케일의 깊이 맵에서 경계 차이 기준

# 전역 변수
det_model = None  # YOLOv8n
sam_model = None  # MobileSAM
midas_model = None # MiDaS for Monocular Depth Estimation
midas_transform = None # MiDaS input transformation
device = "cpu"


@app.on_event("startup")
def load_models_on_startup():
    """서버 시작 시 YOLOv8n + MobileSAM + MiDaS 로드"""
    global det_model, sam_model, midas_model, midas_transform, device
    
    logger.info("[🔥] Starting model loading for YOLOv8n + MobileSAM + MiDaS...")
    
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[⚙️] Device: {device}")
    
    yolo_checkpoint_path = "yolov8n.pt"  
    sam_checkpoint_path = "mobile_sam.pt"

    try:
        # 1. YOLOv8n 모델 로드
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
            
        # 3. MiDaS 모델 로드 (MiDaS_small 사용)
        midas_type = "MiDaS_small"
        midas_model = torch.hub.load("intel-isl/MiDaS", midas_type, trust_repo=True)
        midas_model.to(device)
        midas_model.eval()
        
        # MiDaS 모델에 맞는 입력 변환(Transform) 함수 로드
        midas_transforms_module = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if midas_type == "MiDaS_small":
            midas_transform = midas_transforms_module.small_transform
        else:
            # DPT-Hybrid 등 다른 모델을 사용할 경우:
            midas_transform = midas_transforms_module.dpt_transform
            
        logger.info(f"[✅] MiDaS ({midas_type}) loaded.")

    except Exception as e:
        logger.error(f"[❌] FATAL Model loading failed: {e}", exc_info=True)


def np_from_upload(file_bytes: bytes, mode="RGB") -> Image.Image:
    """바이트를 PIL Image로 변환"""
    try:
        return Image.open(io.BytesIO(file_bytes)).convert(mode)
    except Exception as e:
        logger.error(f"Failed to open image from bytes: {e}")
        return None

# ==============================================================================
# --- MiDaS 깊이 맵 생성 함수 ---
# ==============================================================================
def generate_depth_map_midas(pil_img: Image.Image, output_size: tuple) -> np.ndarray:
    """
    MiDaS 모델을 사용하여 RGB 이미지로부터 깊이 맵을 추정합니다.
    """
    if midas_model is None or midas_transform is None:
        logger.error("MiDaS model or transform not initialized.")
        return None

    try:
        # 1. MiDaS 입력 변환 적용
        input_batch = midas_transform(pil_img).to(device)
        
        with torch.no_grad():
            # 2. MiDaS 모델 실행
            prediction = midas_model(input_batch)
            
            # 3. 출력 크기를 원본 이미지 크기에 맞게 조정
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=pil_img.size[::-1], # (H, W)
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # 4. NumPy로 변환 및 정규화
        depth_map = prediction.cpu().numpy()
        
        # 5. 깊이 맵을 0-255 스케일로 정규화 (Occlusion Mask 생성에 활용하기 위함)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max - depth_min > 0:
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = np.zeros_like(depth_map)

        # 0-255 범위의 8비트 정수형으로 변환
        normalized_depth_uint8 = (normalized_depth * 255).astype(np.uint8)
        
        logger.info("[✅] MiDaS 깊이 맵 생성 완료.")
        return normalized_depth_uint8

    except Exception as e:
        logger.error(f"MiDaS depth generation failed: {e}", exc_info=True)
        return None


def filter_small_boxes(boxes, img_shape, min_ratio=MIN_BOX_RATIO):
    """너무 작은 박스 필터링 (노이즈 제거)."""
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
    """마스크 후처리: 노이즈 제거, 확대, 가장 큰 연결 영역만 남기기 (벽 영역 추정)."""
    mask = mask.astype(np.uint8)
    kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)

    # 노이즈 제거 (Opening) + 경계 채우기 (Dilate)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 가장 큰 연결 영역만 남기기
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask

    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 1, thickness=cv2.FILLED)
    
    # 영역을 부드럽게 닫기 (Closing)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean


def create_depth_occlusion_mask(depth_map: np.ndarray, threshold=DEPTH_DIFF_THRESHOLD) -> np.ndarray:
    """
    깊이 지도를 사용하여 전경 객체(Occlusion) 마스크 생성.
    인접 픽셀 간의 급격한 깊이 변화(경계)를 찾아 객체를 분리합니다.
    """
    if depth_map is None:
        return None
        
    depth_map = depth_map.astype(np.float32)
    
    # Sobel 필터를 사용하여 깊이 맵의 경계(깊이 변화가 큰 부분)를 검출
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    
    # 경계 강도 계산 (Magnitude)
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # 임계값 이상의 경계만 마스킹 (객체 = 1, 배경 = 0)
    occlusion_mask = (magnitude > threshold).astype(np.uint8)
    
    # 마스크 확장 (dilate)하여 객체 영역을 확실하게 덮습니다.
    kernel = np.ones((5, 5), np.uint8)
    occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=2)
    
    return occlusion_mask


@app.get("/")
async def root():
    return {"status": "ok", "message": "YOLOv8n + MobileSAM + MiDaS Integrated Server"}


@app.get("/health")
async def health():
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    gc.collect()
    
    return {
        "status": "healthy",
        "models_loaded": det_model is not None and sam_model is not None and midas_model is not None,
        "device": device,
        "memory_mb": round(memory_mb, 2)
    }


@app.post("/segment_wall_mask")
async def segment_wall_mask(
    rgb_file: UploadFile = File(..., alias="rgb_file"), # 유니티 카메라 이미지
    depth_file: UploadFile = File(..., alias="depth_file") # 유니티 깊이 지도 (흑백 PNG 가정)
):
    """YOLOv8n+SAM으로 객체 감지/분할 후, MiDaS 또는 실제 깊이 지도로 최종 가려짐 마스크를 적용하여 벽 영역 추출"""
    
    # 모델 로딩 여부 확인
    if det_model is None or sam_model is None or midas_model is None:
        logger.error("Segmentation services are unavailable due to model loading failure.")
        return Response(content="Model load failed. Check server startup logs.", status_code=503)

    img = pil_img = results = boxes = sam_boxes = depth_img_np = depth_occlusion_mask = None 

    try:
        # 1. RGB 이미지 로드 및 전처리
        rgb_bytes = await rgb_file.read()
        pil_img = np_from_upload(rgb_bytes, mode="RGB")
        if pil_img is None:
            logger.error("RGB file could not be loaded.")
            return Response(content="Invalid RGB image file.", status_code=400)
            
        original_size = pil_img.size
        
        max_size = 640
        if max(pil_img.size) > max_size:
            ratio = max_size / max(pil_img.size)
            new_size = tuple(int(dim * ratio) for dim in pil_img.size)
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

        w, h = pil_img.size
        logger.info(f"[📸] RGB 이미지: {w}x{h}")
        
        # 2. 깊이 지도 로드 및 MiDaS 폴백 적용
        depth_bytes = await depth_file.read()
        
        # 클라이언트에서 보낸 깊이 데이터가 유효한지 확인 (빈 PNG는 100바이트 미만일 수 있음)
        if depth_bytes and len(depth_bytes) > 100: 
            # 2-1. 클라이언트의 실제 깊이 데이터 사용
            depth_img = np_from_upload(depth_bytes, mode="L")
            if depth_img is not None:
                depth_img = depth_img.resize((w, h), Image.NEAREST) 
                depth_img_np = np.array(depth_img)
                logger.info("[✅] 클라이언트 깊이 지도 로드 완료.")
            else:
                 # 깊이 데이터 로드 실패 시 MiDaS 폴백
                logger.warning("[⚠️] 클라이언트 깊이 데이터 로드 실패. MiDaS로 대체합니다.")
                depth_img_np = generate_depth_map_midas(pil_img, (w, h))
        else:
            # 2-2. 클라이언트 깊이 데이터가 없을 경우 MiDaS 사용 (폴백)
            logger.warning("[⚠️] 클라이언트 깊이 파일이 비어 있습니다. MiDaS로 깊이 맵을 생성합니다.")
            depth_img_np = generate_depth_map_midas(pil_img, (w, h))


        # 3. YOLOv8n + MobileSAM으로 초기 벽 마스크 생성
        logger.info("[🔍] YOLOv8n: 객체 감지 중...")
        results = det_model.predict(
            pil_img, conf=YOLO_CONF_THRESHOLD, imgsz=640, device=device, verbose=False,
        )[0]
        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        boxes = filter_small_boxes(xyxy, pil_img.size[::-1])
        logger.info(f"[✅] {len(boxes)}개의 유효 객체 박스 발견 (Threshold: {YOLO_CONF_THRESHOLD})")

        if not boxes:
            logger.warning("[⚠️] 객체 박스가 없어 전체 이미지(벽) 박스 사용.")
            initial_wall_mask = np.ones((h, w), dtype=np.uint8) * 255
        else:
            logger.info("[🎨] MobileSAM: 객체 분할 중...")
            sam_boxes = boxes
            res = sam_model.predict(
                pil_img, bboxes=sam_boxes, device=device, retina_masks=False, verbose=False
            )[0]

            if res.masks is None:
                logger.warning("[⚠️] MobileSAM 분할 실패. 전체 화면 반환.")
                initial_wall_mask = np.ones((h, w), dtype=np.uint8) * 255
            else:
                # 마스크 통합 및 반전 (벽 영역 추출)
                mask_data = res.masks.data.cpu().numpy()
                union_objects = (mask_data.sum(axis=0) > 0).astype(np.uint8)
                background_mask = 1 - union_objects # 객체 마스크 반전
                
                # 후처리 (가장 큰 배경 영역만 남김)
                refined_background = post_refine(background_mask) 
                initial_wall_mask = (refined_background * 255).astype(np.uint8)
                
                del mask_data, union_objects, background_mask, refined_background


        # 4. 깊이 지도를 이용한 최종 객체 제외 마스킹 (Depth Occlusion)
        final_mask_img = initial_wall_mask.copy()
        
        if depth_img_np is not None:
            depth_occlusion_mask = create_depth_occlusion_mask(depth_img_np)
            
            # 깊이 마스크를 반전하여 벽 마스크(벽=1, 객체=0)를 얻고 기존 마스크와 AND 연산
            wall_from_depth = 1 - depth_occlusion_mask 
            
            # 2D AI 마스크와 3D 깊이 마스크를 결합 (두 마스크 모두 1인 영역만 남김)
            combined_mask = cv2.bitwise_and(final_mask_img, wall_from_depth * 255)
            final_mask_img = combined_mask
            logger.info("[✅] 깊이 데이터(클라이언트 or MiDaS)로 최종 가려짐 보정 완료.")
            
            del wall_from_depth, combined_mask
        else:
            logger.warning("[⚠️] 깊이 데이터가 없어 2D AI 마스크만 사용합니다.")


        # 5. 최종 마스크 정리 및 인코딩
        
        # 경계면 부드럽게 처리 (Smoothing)
        final_mask_img = cv2.GaussianBlur(final_mask_img, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
        
        # 원본 크기로 복원
        if pil_img.size != original_size:
            final_mask_img = cv2.resize(
                final_mask_img, 
                original_size, 
                interpolation=cv2.INTER_LINEAR
            )
        
        # PNG 인코딩
        _, png = cv2.imencode(".png", final_mask_img)

        # 🚨 메모리 정리 강화 
        del pil_img, results, boxes, sam_boxes, depth_img_np, depth_occlusion_mask
        
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