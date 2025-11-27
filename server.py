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
# 1. YOLOv8 객체 감지 민감도: 낮출수록 더 많은 객체를 감지하여 벽 영역에서 제외 
YOLO_CONF_THRESHOLD = 0.01 
# 2. 너무 작은 객체 박스 필터링 기준: 낮출수록 작은 객체까지 포함하여 제외
MIN_BOX_RATIO = 0.005
# 3. 마스크 후처리 시 사용할 모폴로지 커널 크기: 클수록 정제 효과가 강함
MORPHOLOGY_KERNEL_SIZE = 9
# 4. 최종 마스크 경계의 Gaussian Blur 크기: 클수록 경계가 더 부드러움 
GAUSSIAN_BLUR_SIZE = 13
# 5. 깊이 맵 기반 객체 제거 민감도: 이 값보다 깊이 차이가 크면 객체로 간주 (낮출수록 민감)
DEPTH_DIFF_THRESHOLD = 10 

# 전역 변수
det_model = None  # YOLOv8s
sam_model = None  # MobileSAM
midas_model = None # MiDaS for Monocular Depth Estimation
device = "cpu"

# MiDaS DPT_Hybrid_Small 모델의 표준 전처리 값 (MiDaS v2.1 Small과 동일)
MIDAS_MEAN = torch.tensor([0.5, 0.5, 0.5]).float()
MIDAS_STD = torch.tensor([0.5, 0.5, 0.5]).float()

@app.on_event("startup")
def load_models_on_startup():
    """서버 시작 시 YOLOv8s + MobileSAM + MiDaS 로드"""
    global det_model, sam_model, midas_model, device
    
    logger.info("[🔥] Starting model loading for YOLOv8s + MobileSAM + MiDaS...")
    
    # CPU 환경 설정
    device = "cpu"
    logger.info(f"[⚙️] Device: {device}")
    
    yolo_checkpoint_path = "yolov8s.pt"
    sam_checkpoint_path = "mobile_sam.pt"

    try:
        # 1. YOLOv8s 모델 로드
        if not os.path.exists(yolo_checkpoint_path):
             logger.error(f"[❌] YOLOv8s checkpoint not found at: {yolo_checkpoint_path}")
        else:
            det_model = YOLO(yolo_checkpoint_path)
            det_model.to(device)
            logger.info("[✅] YOLOv8s loaded.")
        
        # 2. MobileSAM 로드
        if not os.path.exists(sam_checkpoint_path):
             logger.error(f"[❌] MobileSAM checkpoint not found at: {sam_checkpoint_path}")
        else:
            sam_model = SAM(sam_checkpoint_path)
            sam_model.to(device)
            logger.info("[✅] MobileSAM loaded.")
            
        # 3. MiDaS 모델 로드 (최소형 모델 DPT_Hybrid_Small로 변경)
        midas_type = "DPT_Hybrid_Small" 
        midas_model = torch.hub.load("intel-isl/MiDaS", midas_type, trust_repo=True, map_location=device)
        midas_model.to(device)
        midas_model.eval()
        
        logger.info(f"[✅] MiDaS ({midas_type}) loaded on CPU. (최소 메모리 모델)")

    except Exception as e:
        logger.error(f"[❌] FATAL Model loading failed: {e}", exc_info=True)
        midas_model = None


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
    [수동 전처리]: 오류를 회피하기 위해 transform 대신 수동으로 전처리합니다.
    """
    if midas_model is None:
        logger.error("MiDaS model not initialized.")
        return None

    try:
        # 1. NumPy 배열로 변환 및 정규화
        img_np = np.array(pil_img) # H, W, 3 (uint8)
        img_float = img_np.astype(np.float32) / 255.0 # H, W, 3 (float 0-1)
        
        # 2. PyTorch 텐서로 변환 및 차원 변경 (H, W, C -> C, H, W)
        tensor = torch.from_numpy(img_float).permute(2, 0, 1) # 3, H, W
        
        # 3. MiDaS 표준 정규화 적용 (Mean and Std)
        for i in range(3):
            tensor[i].sub_(MIDAS_MEAN[i]).div_(MIDAS_STD[i])

        
        # 4. 배치 차원 추가 및 디바이스 이동
        input_batch = tensor.unsqueeze(0).to(device) # 1, 3, H, W
        del tensor # 🚨 메모리 해제
        
        with torch.no_grad():
            # 5. MiDaS 모델 실행
            prediction = midas_model(input_batch)
            del input_batch # 🚨 메모리 해제
            
            # 6. 출력 크기를 원본 이미지 크기에 맞게 조정
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=pil_img.size[::-1], # (H, W)
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # 7. NumPy로 변환 및 정규화
        depth_map = prediction.cpu().numpy()
        del prediction # 🚨 메모리 해제
        
        # 8. 깊이 맵을 0-255 스케일로 정규화
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_range = depth_max - depth_min
        
        if depth_range > 0:
            normalized_depth = (depth_map - depth_min) / depth_range
        else:
            normalized_depth = np.zeros_like(depth_map, dtype=np.float32)

        # 0-255 범위의 8비트 정수형으로 변환
        normalized_depth_uint8 = (normalized_depth * 255).astype(np.uint8)
        
        logger.info("[✅] MiDaS (DPT_Hybrid_Small) 깊이 맵 생성 완료.")
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
    del grad_x, grad_y # 🚨 메모리 해제
    
    # 임계값 이상의 경계만 마스킹 (객체 = 1, 배경 = 0)
    occlusion_mask = (magnitude > threshold).astype(np.uint8)
    del magnitude # 🚨 메모리 해제
    
    # 마스크 확장 (dilate)하여 객체 영역을 확실하게 덮습니다.
    kernel = np.ones((5, 5), np.uint8)
    occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=2)
    
    return occlusion_mask


@app.get("/")
async def root():
    return {"status": "ok", "message": "YOLOv8s + MobileSAM + DPT_Hybrid_Small Integrated Server"}


@app.get("/health")
async def health():
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
    """YOLOv8s+SAM으로 객체 감지/분할 후, MiDaS 또는 실제 깊이 지도로 최종 가려짐 마스크를 적용하여 벽 영역 추출"""
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    logger.info(f"[🧠] 요청 시작 메모리: {initial_memory:.2f} MB")
    
    # 모델 로딩 여부 확인
    if det_model is None or sam_model is None or midas_model is None:
        logger.error("Segmentation services are unavailable due to model loading failure or MiDaS initialization failure.")
        return Response(content="Model load failed. Check server startup logs.", status_code=503)

    pil_img = depth_img_np = depth_occlusion_mask = None 

    try:
        # 1. RGB 이미지 로드 및 전처리
        rgb_bytes = await rgb_file.read()
        pil_img = np_from_upload(rgb_bytes, mode="RGB")
        del rgb_bytes # 🚨 메모리 해제
        
        if pil_img is None:
            logger.error("RGB file could not be loaded.")
            return Response(content="Invalid RGB image file.", status_code=400)
            
        original_size = pil_img.size
        
        max_size = 640
        if max(pil_img.size) > max_size:
            ratio = max_size / max(pil_img.size)
            new_size = tuple(int(dim * ratio) for dim in pil_img.size)
            # PIL Image.resize는 새 이미지를 반환하므로 이전 pil_img는 GC 대상이 됨
            pil_img = pil_img.resize(new_size, Image.LANCZOS) 

        w, h = pil_img.size
        logger.info(f"[📸] RGB 이미지: {w}x{h}")
        
        # 2. 깊이 지도 로드 및 MiDaS 폴백 적용
        depth_bytes = await depth_file.read()
        
        # 클라이언트 깊이 데이터 유효성 검사
        if depth_bytes and len(depth_bytes) > 100: 
            depth_img = np_from_upload(depth_bytes, mode="L")
            if depth_img is not None:
                depth_img_np = np.array(depth_img.resize((w, h), Image.NEAREST))
                del depth_img
                logger.info("[✅] 클라이언트 깊이 지도 로드 완료.")
            else:
                logger.warning("[⚠️] 클라이언트 깊이 데이터 로드 실패. MiDaS로 대체합니다.")
                depth_img_np = generate_depth_map_midas(pil_img, (w, h))
        else:
            logger.warning("[⚠️] 클라이언트 깊이 파일이 비어 있습니다. MiDaS로 깊이 맵을 생성합니다.")
            depth_img_np = generate_depth_map_midas(pil_img, (w, h))
        
        del depth_bytes # 🚨 메모리 해제

        # 3. YOLOv8s + MobileSAM으로 초기 벽 마스크 생성
        logger.info("[🔍] YOLOv8s: 객체 감지 중...")
        results = det_model.predict(
            pil_img, conf=YOLO_CONF_THRESHOLD, imgsz=640, device=device, verbose=False,
        )[0]
        
        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        del results # 🚨 YOLO 결과 객체 즉시 메모리 해제
        
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
            del boxes # 🚨 메모리 해제
            
            if res.masks is None:
                logger.warning("[⚠️] MobileSAM 분할 실패. 전체 화면 반환.")
                initial_wall_mask = np.ones((h, w), dtype=np.uint8) * 255
                del res
            else:
                # 마스크 통합 및 반전 (벽 영역 추출)
                mask_data = res.masks.data.cpu().numpy()
                del res # 🚨 SAM 결과 객체 즉시 메모리 해제
                
                union_objects = (mask_data.sum(axis=0) > 0).astype(np.uint8)
                del mask_data
                
                background_mask = 1 - union_objects # 객체 마스크 반전
                del union_objects
                
                # 후처리 (가장 큰 배경 영역만 남김)
                refined_background = post_refine(background_mask) 
                del background_mask
                
                initial_wall_mask = (refined_background * 255).astype(np.uint8)
                del refined_background


        # 4. 깊이 지도를 이용한 최종 객체 제외 마스킹 (Depth Occlusion)
        final_mask_img = initial_wall_mask.copy()
        del initial_wall_mask
        
        if depth_img_np is not None:
            depth_occlusion_mask = create_depth_occlusion_mask(depth_img_np)
            del depth_img_np # 🚨 메모리 해제
            
            if depth_occlusion_mask is not None:
                # 깊이 마스크를 반전하여 벽 마스크(벽=1, 객체=0)를 얻고 기존 마스크와 AND 연산
                wall_from_depth = 1 - depth_occlusion_mask 
                del depth_occlusion_mask
                
                # 2D AI 마스크와 3D 깊이 마스크를 결합
                combined_mask = cv2.bitwise_and(final_mask_img, wall_from_depth * 255)
                final_mask_img = combined_mask
                del wall_from_depth, combined_mask
                
                logger.info("[✅] 깊이 데이터(클라이언트 or MiDaS)로 최종 가려짐 보정 완료.")
            else:
                logger.warning("[⚠️] 깊이 마스크 생성 실패. 2D AI 마스크만 사용합니다.")
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
        
        del pil_img
        
        # PNG 인코딩
        _, png = cv2.imencode(".png", final_mask_img)
        del final_mask_img, _ # 🚨 메모리 해제

        final_png_bytes = png.tobytes()
        del png
        
        # 🚨 최종 메모리 정리 및 로깅
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