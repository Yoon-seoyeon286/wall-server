import io
import cv2
import torch
import numpy as np
import gc
from PIL import Image
from ultralytics import YOLO, SAM
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 앱 초기화
app = FastAPI()

# CORS 설정 (모든 출처 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Lazy loading을 위한 전역 변수
det_model = None
sam_model = None
device = "cpu"


def load_models():
    """모델을 로드하거나 이미 로드된 경우 건너뜁니다. (자동 다운로드)"""
    global det_model, sam_model, device

    if det_model is not None and sam_model is not None:
        return

    # RT-DETR-L과 표준 SAM-B는 Apache 2.0 라이선스로 상업적 사용에 제한이 없습니다.
    print("[🔥] Loading heavyweight models (RT-DETR-L + SAM-B)... This may take time on first run.")
    
    # GPU 사용 가능 시 CUDA, 아니면 CPU를 사용합니다.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[⚙️] Device set to: {device}")

    try:
        # ✅ RT-DETR 로드: Ultralytics가 자동으로 rtdetr-l.pt 파일을 다운로드합니다.
        det_model_local = YOLO("rtdetr-l.pt") 
        det_model_local.to(device)

        # ✅ SAM-B 로드: Ultralytics가 자동으로 sam_b.pt 파일을 다운로드합니다.
        sam_model_local = SAM("sam_b.pt") 
        sam_model_local.to(device)

        # 할당 완료 후 전역에 넣기
        globals()["det_model"] = det_model_local
        globals()["sam_model"] = sam_model_local
        
        print("[✔] Models loaded!")
        
    except Exception as e:
        print(f"[❌] Model loading failed: {e}")
        # 모델 로드 실패 시 None으로 설정
        globals()["det_model"] = None
        globals()["sam_model"] = None


def np_from_upload(file_bytes: bytes) -> Image.Image:
    """업로드된 바이트를 PIL Image 객체로 변환합니다."""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def filter_small_boxes(boxes, img_shape, min_ratio=0.03):
    """이미지 전체 면적 대비 작은 박스를 필터링합니다."""
    H, W = img_shape
    area_img = H * W
    filtered = []
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area / area_img > min_ratio:
            filtered.append([float(x1), float(y1), float(x2), float(y2)])
    return filtered


def post_refine(mask: np.ndarray):
    """마스크 후처리: 노이즈 제거, 확대, 가장 큰 영역만 남기기."""
    mask = mask.astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)

    # 노이즈 제거 + 살짝 확대 (Open -> Dilate)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 가장 큰 연결 영역만 남기기
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask

    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 1, thickness=cv2.FILLED)
    
    # 마지막으로 영역 채우기 및 매끄럽게 처리 (Close)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean


# ----------------------------------------------------------------------
# FastAPI 엔드포인트
# ----------------------------------------------------------------------

@app.get("/")
async def root():
    """서버 상태 확인"""
    return {"status": "ok", "message": "Wall Segmentation Server (RT-DETR + SAM-B)"}


@app.get("/health")
async def health():
    """서버 상태 및 메모리 정보 확인"""
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return {
        "status": "healthy",
        "models_loaded": det_model is not None,
        "device": device,
        "memory_mb": round(memory_mb, 2)
    }


@app.post("/segment_wall_mask")
async def segment_wall_mask(file: UploadFile = File(...)):
    """업로드된 이미지에서 벽 분할 마스크를 PNG 파일로 반환합니다."""
    try:
        # 필요할 때만 모델 로딩 (첫 요청)
        load_models()

        if det_model is None or sam_model is None:
             # 모델 로드 실패 시 503 오류 반환
             return Response(content="Model load failed. Check server logs.", status_code=503)

        # 업로드 이미지 → PIL
        file_bytes = await file.read()
        if not file_bytes:
             return Response(content="File is empty.", status_code=400)
             
        img = np_from_upload(file_bytes)

        # 이미지 크기 축소 (메모리 절약 및 추론 속도 개선)
        max_size = 640
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS) 

        pil_img = img.copy()

        # 1. RT-DETR 예측 (벽 감지)
        results = det_model.predict(
            pil_img,
            conf=0.20,
            imgsz=640,
            device=device,
            verbose=False
        )[0]

        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        boxes = filter_small_boxes(xyxy, pil_img.size[::-1])

        # 박스가 너무 작아 다 걸러지면, 가장 큰 거 하나라도 선택
        if not boxes and len(xyxy) > 0:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy]
            biggest = xyxy[np.argmax(areas)].tolist()
            boxes = [biggest]

        # 진짜로 아무것도 못 찾으면 전체 이미지를 박스로 (안전 장치)
        if not boxes:
            w, h = pil_img.size
            boxes = [[0.0, 0.0, float(w), float(h)]]

        # 2. SAM-B 예측 (분할)
        res = sam_model.predict(
            pil_img,
            bboxes=boxes,
            device=device,
            retina_masks=False,
            verbose=False
        )[0]

        if res.masks is None:
            # SAM이 어떤 마스크도 생성하지 못한 경우
            return Response(content=b'', status_code=422)

        # 마스크들을 합치고 후처리
        mask = res.masks.data.cpu().numpy()
        union = (mask.sum(axis=0) > 0).astype(np.uint8)
        refined = post_refine(union)

        # 마스크 이미지를 PNG 바이트로 변환
        mask_img = (refined * 255).astype(np.uint8)
        _, png = cv2.imencode(".png", mask_img)

        # 메모리 정리 
        del img, pil_img, results, mask, union, refined, mask_img, xyxy, boxes, res, file_bytes
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
        print("🔥 /segment_wall_mask ERROR:", e)
        import traceback
        traceback.print_exc()
        return Response(
            content=str(e).encode(),
            status_code=500
        )


@app.options("/segment_wall_mask")
async def options_segment_wall_mask():
    """CORS Pre-flight 요청 처리"""
    return Response(
        content=b'',
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )