import io
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO, SAM
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Lazy loading
det_model = None
sam_model = None
device = "cpu"


def load_models():
    global det_model, sam_model, device

    if det_model is not None and sam_model is not None:
        return

    print("[🔥] Loading lightweight models (YOLO + MobileSAM)...")
    device = "cpu"

    # ✅ 일반 YOLO 모델 사용 (YOLO-World 대신)
    # yolov8n.pt 는 가장 가벼운 모델
    det_model_local = YOLO("yolov8n.pt")
    det_model_local.to(device)

    # ✅ Mobile SAM 공식 이름
    sam_model_local = SAM("mobile_sam.pt")
    sam_model_local.to(device)

    # 할당 완료 후 전역에 넣기
    globals()["det_model"] = det_model_local
    globals()["sam_model"] = sam_model_local

    print("[✔] Models loaded!")


def np_from_upload(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def filter_small_boxes(boxes, img_shape, min_ratio=0.03):
    H, W = img_shape
    area_img = H * W
    filtered = []
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area / area_img > min_ratio:
            filtered.append([float(x1), float(y1), float(x2), float(y2)])
    return filtered


def post_refine(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)

    # 노이즈 제거 + 살짝 확대
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 가장 큰 연결 영역만 남기기
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask

    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 1, thickness=cv2.FILLED)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean


@app.get("/")
async def root():
    return {"status": "ok", "message": "Wall Segmentation Server (Lightweight)"}


@app.get("/health")
async def health():
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return {
        "status": "healthy",
        "models_loaded": det_model is not None,
        "memory_mb": round(memory_mb, 2)
    }


@app.post("/segment_wall_mask")
async def segment_wall_mask(file: UploadFile = File(...)):
    try:
        # 필요할 때만 모델 로딩 (첫 요청)
        load_models()

        # 업로드 이미지 → PIL
        img = np_from_upload(await file.read())

        # 이미지 크기 축소 (메모리 절약)
        max_size = 640
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)

        pil_img = img.copy()

        # YOLO 예측 (벽 감지 - class 0은 person이지만 임시로 사용)
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

        # 진짜로 아무것도 못 찾으면 전체 이미지를 박스로
        if not boxes:
            w, h = pil_img.size
            boxes = [[0.0, 0.0, float(w), float(h)]]

        # SAM 예측
        res = sam_model.predict(
            pil_img,
            bboxes=boxes,
            device=device,
            retina_masks=False,
            verbose=False
        )[0]

        if res.masks is None:
            return Response(content=b'', status_code=422)

        mask = res.masks.data.cpu().numpy()
        union = (mask.sum(axis=0) > 0).astype(np.uint8)
        refined = post_refine(union)

        mask_img = (refined * 255).astype(np.uint8)
        _, png = cv2.imencode(".png", mask_img)

        # 메모리 정리
        del img, pil_img, results, mask, union, refined, mask_img
        import gc
        gc.collect()

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
    return Response(
        content=b'',
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )