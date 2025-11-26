import io
import cv2
import torch
import numpy as np
import gc
from PIL import Image
from ultralytics import SAM
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

# 전역 변수
sam_model = None
device = "cpu"


def load_model():
    """MobileSAM 모델 로드 (가볍고 빠름)"""
    global sam_model, device
    
    if sam_model is not None:
        return
    
    print("[🔥] Loading MobileSAM (lightweight & fast)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[⚙️] Device: {device}")
    
    try:
        # 🔥 MobileSAM 사용 (sam_b.pt 대신 mobile_sam.pt)
        sam_model_local = SAM("mobile_sam.pt")
        sam_model_local.to(device)
        
        globals()["sam_model"] = sam_model_local
        print("[✅] MobileSAM loaded!")
        
    except Exception as e:
        print(f"[❌] Model loading failed: {e}")
        globals()["sam_model"] = None


def np_from_upload(file_bytes: bytes) -> Image.Image:
    """업로드된 바이트를 PIL Image로 변환"""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def get_center_point(img_shape):
    """이미지 중앙점 반환 (벽이 화면 중앙에 있다고 가정)"""
    h, w = img_shape[:2]
    return [[w // 2, h // 2]]


def expand_mask(mask, iterations=20):
    """마스크 확장"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    expanded = cv2.dilate(mask, kernel, iterations=iterations)
    return expanded


# ----------------------------------------------------------------------
# FastAPI 엔드포인트
# ----------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "message": "MobileSAM Wall Detection Server"}


@app.get("/health")
async def health():
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return {
        "status": "healthy",
        "model_loaded": sam_model is not None,
        "device": device,
        "memory_mb": round(memory_mb, 2)
    }


@app.post("/segment_wall_mask")
async def segment_wall_mask(file: UploadFile = File(...)):
    """MobileSAM으로 벽 감지 (빠르고 정확)"""
    try:
        load_model()
        
        if sam_model is None:
            return Response(content="Model load failed.", status_code=503)
        
        file_bytes = await file.read()
        if not file_bytes:
            return Response(content="File is empty.", status_code=400)
        
        img = np_from_upload(file_bytes)
        
        # 리사이즈 (속도 향상)
        max_size = 640
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
        
        pil_img = img.copy()
        w, h = pil_img.size
        
        print(f"[📸] 이미지 크기: {w}x{h}")
        
        # 🔥 전략 1: 중앙점 클릭 (벽이 화면 중앙에 있다고 가정)
        center_points = get_center_point((h, w))
        
        # MobileSAM 예측 (포인트 프롬프트 사용)
        results = sam_model.predict(
            pil_img,
            points=center_points,
            labels=[1],  # 1 = foreground (벽)
            device=device,
            verbose=False
        )[0]
        
        if results.masks is None or len(results.masks.data) == 0:
            print("[⚠️] 중앙점 감지 실패. 전체 이미지 박스 사용.")
            # 🔥 전략 2: 전체 이미지를 박스로
            results = sam_model.predict(
                pil_img,
                bboxes=[[0, 0, w, h]],
                device=device,
                verbose=False
            )[0]
        
        if results.masks is None:
            print("[❌] SAM 감지 완전 실패. 전체 화면 반환.")
            mask = np.ones((h, w), dtype=np.uint8)
        else:
            # 마스크 추출
            mask_data = results.masks.data.cpu().numpy()
            mask = (mask_data[0] > 0.5).astype(np.uint8)  # 첫 번째 마스크 사용
            
            # 🔥 마스크 확장 (적당히)
            mask = expand_mask(mask, iterations=25)
        
        # 통계
        wall_pixels = np.sum(mask)
        total_pixels = h * w
        coverage = (wall_pixels / total_pixels) * 100
        
        print(f"[✅] Coverage: {coverage:.1f}% ({wall_pixels}/{total_pixels} pixels)")
        
        # 너무 작으면 전체 사용
        if coverage < 10.0:
            print(f"[⚠️] Coverage 너무 낮음. 전체 화면 사용.")
            mask = np.ones((h, w), dtype=np.uint8)
        
        # PNG 변환
        mask_img = (mask * 255).astype(np.uint8)
        _, png = cv2.imencode(".png", mask_img)
        
        # 메모리 정리
        del img, pil_img, results, mask, mask_data, mask_img, file_bytes
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
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return Response(content=str(e).encode(), status_code=500)


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