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
    """모델을 로드하거나 이미 로드된 경우 건너뛰며, 자동 다운로드합니다."""
    global det_model, sam_model, device

    if det_model is not None and sam_model is not None:
        return

    print("[🔥] Loading heavyweight models (RT-DETR-L + SAM-B)... This may take time on first run.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[⚙️] Device set to: {device}")

    try:
        # ✅ RT-DETR 로드 (자동 다운로드)
        det_model_local = YOLO("rtdetr-l.pt") 
        det_model_local.to(device)

        # ✅ SAM-B 로드 (자동 다운로드)
        sam_model_local = SAM("sam_b.pt") 
        sam_model_local.to(device)

        globals()["det_model"] = det_model_local
        globals()["sam_model"] = sam_model_local
        
        print("[✔] Models loaded!")
        
    except Exception as e:
        print(f"[❌] Model loading failed: {e}")
        globals()["det_model"] = None
        globals()["sam_model"] = None


def np_from_upload(file_bytes: bytes) -> Image.Image:
    """업로드된 바이트를 PIL Image 객체로 변환합니다."""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def filter_small_boxes(boxes, img_shape, min_ratio=0.03):
    """(현재 디버깅을 위해 사용되지 않음) 이미지 전체 면적 대비 작은 박스를 필터링합니다."""
    H, W = img_shape
    area_img = H * W
    filtered = []
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area / area_img > min_ratio:
            filtered.append([float(x1), float(y1), float(x2), float(y2)])
    return filtered


def post_refine(mask: np.ndarray):
    """(현재 디버깅을 위해 사용되지 않음) 마스크 후처리 함수."""
    mask = mask.astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask

    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 1, thickness=cv2.FILLED)
    
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
    """업로드된 이미지에서 벽 분할 마스크를 PNG 파일로 반환합니다. (탐지 필터링 완화)"""
    try:
        load_models()

        if det_model is None or sam_model is None:
             return Response(content="Model load failed. Check server logs.", status_code=503)

        file_bytes = await file.read()
        if not file_bytes:
             return Response(content="File is empty.", status_code=400)
             
        img = np_from_upload(file_bytes)

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
        
        # 🚨 디버깅 수정 지점: 작은 박스 필터링 (filter_small_boxes)을 건너뛰고 모든 박스를 사용 🚨
        boxes = xyxy.tolist() if xyxy.size > 0 else [] 

        # 박스가 하나도 없으면 전체 이미지를 박스로 (강제)
        if not boxes:
            w, h = pil_img.size
            boxes = [[0.0, 0.0, float(w), float(h)]]
            print("[🔍] RT-DETR이 박스를 찾지 못해 전체 이미지 박스를 SAM에 강제 전달합니다.")
        else:
            print(f"[🔍] RT-DETR이 {len(boxes)}개의 박스를 찾았습니다.")


        # 2. SAM-B 예측 (분할)
        res = sam_model.predict(
            pil_img,
            bboxes=boxes,
            device=device,
            retina_masks=False,
            verbose=False
        )[0]

        if res.masks is None:
            # 422 상태 코드 반환 (마스크가 생성되지 않음)
            return Response(content=b'', status_code=422) 

        # 마스크들을 합치고 후처리 (post_refine은 계속 건너뛴 상태)
        mask = res.masks.data.cpu().numpy()
        union = (mask.sum(axis=0) > 0).astype(np.uint8)
        
        # 💡 디버깅 상태 유지: post_refine을 호출하지 않고 union 마스크를 바로 사용
        refined = union 

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