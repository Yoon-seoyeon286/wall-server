import io
import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLOWorld, SAM

# -------------------------------------------------------
# ⚠️ 서버 시작 시 딱 1회만 로드 (가장 중요)
# -------------------------------------------------------
print("\n[🔥 WALL SERVER BOOT] Loading lightweight models...")

device = "cpu"

det_model = YOLOWorld("yolov8n-worldv2.pt", verbose=False)
det_model.to(device)
det_model.set_classes(["wall"])

sam_model = SAM("mobile_sam.pt", verbose=False)
sam_model.to(device)

# Warmup: CPU 모델 준비 시간 단축
dummy = Image.new("RGB", (64, 64), (0, 0, 0))
_ = det_model.predict(dummy, conf=0.2, imgsz=64, device=device, verbose=False)

print("[✔] Models loaded & warmed up!\n")

# -------------------------------------------------------
# FastAPI 초기화
# -------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    expose_headers=["*"]
)


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
def np_from_upload(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def filter_small_boxes(boxes, img_shape, min_ratio=0.02):
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
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 1, thickness=cv2.FILLED)
    return clean


# -------------------------------------------------------
# Endpoints
# -------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "msg": "Wall Segmentation Server (Stable CPU Mode)"}


@app.get("/health")
async def health():
    import psutil
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024
    return {
        "status": "alive",
        "memory_mb": round(mem, 2),
        "models_loaded": True
    }


@app.post("/segment_wall_mask")
async def segment_wall_mask(file: UploadFile = File(...)):
    try:
        img = np_from_upload(await file.read())

        # Resize safely
        max_size = 640
        ratio = min(max_size / img.width, max_size / img.height)
        if ratio < 1:
            new_w, new_h = int(img.width * ratio), int(img.height * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # YOLO
        result = det_model.predict(
            img, conf=0.25, imgsz=640, device=device,
            retina=False, verbose=False
        )[0]

        xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        boxes = filter_small_boxes(xyxy, img.size[::-1])

        if not boxes and len(xyxy) > 0:
            largest = xyxy[np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy])]
            boxes = [largest.tolist()]

        if not boxes:
            return Response(content=b'', status_code=422)

        # SAM
        pred = sam_model.predict(
            img, bboxes=boxes, device=device,
            retina_masks=False, verbose=False
        )[0]

        if pred.masks is None:
            return Response(content=b'', status_code=422)

        mask = pred.masks.data.cpu().numpy()
        union = (mask.sum(axis=0) > 0).astype(np.uint8)
        refined = post_refine(union)

        out = (refined * 255).astype(np.uint8)
        _, png = cv2.imencode(".png", out)

        del mask, union, refined, result, pred
        torch.cuda.empty_cache()  # CPU에서도 안전

        return Response(
            content=png.tobytes(),
            media_type="image/png",
            headers={"Access-Control-Allow-Origin": "*"}
        )

    except Exception as e:
        print("🔥 SERVER ERROR:", e)
        import traceback
        traceback.print_exc()
        return Response(content=str(e).encode(), status_code=500)
