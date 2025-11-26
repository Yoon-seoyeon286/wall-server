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

# CORS 설정
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

    print("[🔥] Loading heavyweight models (RT-DETR-L + SAM-B)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[⚙️] Device set to: {device}")

    try:
        det_model_local = YOLO("rtdetr-l.pt") 
        det_model_local.to(device)

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


# 🔥 마스크 대폭 확장 함수
def expand_mask_massive(mask, iterations=50):
    """마스크를 매우 크게 확장시킵니다."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    expanded = cv2.dilate(mask, kernel, iterations=iterations)
    
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    expanded = cv2.dilate(expanded, kernel_large, iterations=10)
    
    return expanded


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
            conf=0.10,  # 🔥 더욱 낮춤 (0.15 → 0.10)
            imgsz=640,
            device=device,
            verbose=False
        )[0]

        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        boxes = xyxy.tolist() if xyxy.size > 0 else [] 

        # 박스가 없으면 전체 이미지를 박스로
        if not boxes:
            w, h = pil_img.size
            boxes = [[0.0, 0.0, float(w), float(h)]]
            print("[🔍] RT-DETR이 박스를 찾지 못해 전체 이미지 박스를 강제 전달합니다.")
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
            print("[⚠️] SAM이 마스크를 생성하지 못했습니다. 전체 이미지를 흰색으로 반환합니다.")
            # 🔥 마스크 생성 실패 시 전체를 흰색으로
            h, w = pil_img.size[1], pil_img.size[0]
            refined = np.ones((h, w), dtype=np.uint8)
        else:
            # 마스크 합치기
            mask = res.masks.data.cpu().numpy()
            union = (mask.sum(axis=0) > 0).astype(np.uint8)
            
            # 🔥 마스크 대폭 확장
            refined = expand_mask_massive(union, iterations=80)  # 80으로 증가
        
        wall_pixels = np.sum(refined)
        total_pixels = refined.shape[0] * refined.shape[1]
        coverage_percent = (wall_pixels / total_pixels) * 100
        
        print(f"[🔍] Mask pixels: {wall_pixels} / {total_pixels} ({coverage_percent:.1f}% coverage)")
        
        # 🔥 픽셀이 너무 적으면 전체를 흰색으로 강제 변환
        if wall_pixels < 10000:  # 10,000 픽셀 미만이면
            print(f"[⚠️] 마스크가 너무 작습니다 ({wall_pixels} pixels). 전체 화면을 마스크로 사용합니다.")
            refined = np.ones_like(refined, dtype=np.uint8)
            wall_pixels = np.sum(refined)
            print(f"[✔️] 강제 전체 마스크 생성: {wall_pixels} pixels")

        # 🔥🔥🔥 마스크를 255로 변환 (완전 흰색)
        mask_img = (refined * 255).astype(np.uint8)
        
        # 🔥 추가: 밝기 확인
        avg_brightness = np.mean(mask_img)
        print(f"[🔍] 마스크 평균 밝기: {avg_brightness:.1f} / 255")
        
        _, png = cv2.imencode(".png", mask_img)

        # 메모리 정리 
        del img, pil_img, results, mask_img, xyxy, boxes, res, file_bytes, refined
        if 'mask' in locals():
            del mask, union
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