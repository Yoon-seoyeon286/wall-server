import io
import cv2
import torch
import numpy as np
import gc
from PIL import Image
from ultralytics import SAM
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
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
grounding_dino_processor = None
grounding_dino_model = None
sam_model = None
device = "cpu"


def load_models():
    """Grounding DINO Lite + MobileSAM 로드"""
    global grounding_dino_processor, grounding_dino_model, sam_model, device
    
    if grounding_dino_model is not None and sam_model is not None:
        return
    
    print("[🔥] Loading Grounding DINO Lite + MobileSAM...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[⚙️] Device: {device}")
    
    try:
        # 1. Grounding DINO Lite 로드
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor_local = AutoProcessor.from_pretrained(model_id)
        model_local = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        model_local.to(device)
        
        # 2. MobileSAM 로드
        sam_local = SAM("mobile_sam.pt")
        sam_local.to(device)
        
        globals()["grounding_dino_processor"] = processor_local
        globals()["grounding_dino_model"] = model_local
        globals()["sam_model"] = sam_local
        
        print("[✅] Models loaded!")
        
    except Exception as e:
        print(f"[❌] Model loading failed: {e}")
        globals()["grounding_dino_processor"] = None
        globals()["grounding_dino_model"] = None
        globals()["sam_model"] = None


def np_from_upload(file_bytes: bytes) -> Image.Image:
    """바이트를 PIL Image로 변환"""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def detect_walls_grounding_dino(image: Image.Image, text_prompt: str = "wall"):
    """Grounding DINO로 벽 감지"""
    inputs = grounding_dino_processor(
        images=image,
        text=text_prompt,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = grounding_dino_model(**inputs)
    
    # 결과 후처리
    results = grounding_dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,  # 낮은 threshold (더 많이 감지)
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]  # (height, width)
    )[0]
    
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]
    
    return boxes, scores, labels


def expand_mask(mask, iterations=20):
    """마스크 확장"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, kernel, iterations=iterations)


# ----------------------------------------------------------------------
# FastAPI 엔드포인트
# ----------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "message": "Grounding DINO Lite + MobileSAM Server"}


@app.get("/health")
async def health():
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return {
        "status": "healthy",
        "models_loaded": grounding_dino_model is not None and sam_model is not None,
        "device": device,
        "memory_mb": round(memory_mb, 2)
    }


@app.post("/segment_wall_mask")
async def segment_wall_mask(file: UploadFile = File(...)):
    """Grounding DINO로 벽 찾고 → MobileSAM으로 정밀 분할"""
    try:
        load_models()
        
        if grounding_dino_model is None or sam_model is None:
            return Response(content="Model load failed.", status_code=503)
        
        file_bytes = await file.read()
        if not file_bytes:
            return Response(content="File is empty.", status_code=400)
        
        img = np_from_upload(file_bytes)
        original_size = img.size
        
        # 리사이즈 (속도 향상)
        max_size = 640
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
        
        w, h = img.size
        print(f"[📸] 이미지: {w}x{h}")
        
        # 1️⃣ Grounding DINO로 벽 감지
        print("[🔍] Grounding DINO: 벽 감지 중...")
        boxes, scores, labels = detect_walls_grounding_dino(img, text_prompt="wall")
        
        if len(boxes) == 0:
            print("[⚠️] 벽을 찾지 못했습니다. 전체 이미지를 박스로 사용.")
            boxes = np.array([[0, 0, w, h]])
        else:
            print(f"[✅] {len(boxes)}개의 벽 후보 발견 (confidence: {scores[0]:.2f})")
        
        # 2️⃣ MobileSAM으로 정밀 분할
        print("[🎨] MobileSAM: 정밀 분할 중...")
        
        # 박스 형식 변환: [x1, y1, x2, y2] → [[x1, y1, x2, y2]]
        sam_boxes = boxes.tolist()
        
        results = sam_model.predict(
            img,
            bboxes=sam_boxes,
            device=device,
            verbose=False
        )[0]
        
        if results.masks is None or len(results.masks.data) == 0:
            print("[⚠️] MobileSAM 실패. 전체 화면 사용.")
            mask = np.ones((h, w), dtype=np.uint8)
        else:
            # 모든 마스크 합치기 (여러 벽이 있을 수 있음)
            masks = results.masks.data.cpu().numpy()
            mask = (masks.sum(axis=0) > 0).astype(np.uint8)
            
            # 확장
            mask = expand_mask(mask, iterations=25)
        
        # 원본 크기로 복원
        if img.size != original_size:
            mask_img = (mask * 255).astype(np.uint8)
            mask_img = cv2.resize(
                mask_img, 
                original_size, 
                interpolation=cv2.INTER_LINEAR
            )
        else:
            mask_img = (mask * 255).astype(np.uint8)
        
        # 통계
        wall_pixels = np.sum(mask_img > 0)
        total_pixels = mask_img.shape[0] * mask_img.shape[1]
        coverage = (wall_pixels / total_pixels) * 100
        
        print(f"[✅] Coverage: {coverage:.1f}% ({wall_pixels}/{total_pixels} pixels)")
        
        # 너무 작으면 전체 사용
        if coverage < 5.0:
            print(f"[⚠️] Coverage 너무 낮음. 전체 화면 사용.")
            mask_img = np.ones_like(mask_img) * 255
        
        # PNG 인코딩
        _, png = cv2.imencode(".png", mask_img)
        
        # 메모리 정리
        del img, results, mask, mask_img, file_bytes, boxes, scores, labels
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