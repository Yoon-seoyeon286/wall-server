import io
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld, SAM
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# LOAD MODELS
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

det_model = YOLOWorld("yolov8s-worldv2.pt")
det_model.to(device)
det_model.set_classes(["wall"])

sam_model = SAM("mobile_sam.pt")
sam_model.to(device)

app = FastAPI()

# ----------------------------
# CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ----------------------------
# HELPERS
# ----------------------------
def np_from_upload(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def filter_small_boxes(boxes, img_shape, min_ratio=0.03):
    """Remove too-small wall boxes"""
    H, W = img_shape
    area_img = H * W
    filtered = []
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area / area_img > min_ratio:
            filtered.append([float(x1), float(y1), float(x2), float(y2)])
    return filtered

def post_refine(mask: np.ndarray):
    """Morphological clean + fill holes + keep largest area"""
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

# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Wall Segmentation Server"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ----------------------------
# SEGMENT WALL
# ----------------------------
@app.post("/segment_wall_mask")
async def segment_wall_mask(file: UploadFile = File(...)):
    try:
        img = np_from_upload(await file.read())
        pil_img = img.copy()

        results = det_model.predict(pil_img, conf=0.20, imgsz=1024, device=device, verbose=False)[0]
        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        boxes = filter_small_boxes(xyxy, pil_img.size[::-1])

        if not boxes and len(xyxy) > 0:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy]
            biggest = xyxy[np.argmax(areas)].tolist()
            boxes = [biggest]

        if not boxes:
            return Response(
                content=b'',
                status_code=422,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*"
                }
            )

        res = sam_model.predict(pil_img, bboxes=boxes, device=device, retina_masks=True, verbose=False)[0]
        if res.masks is None:
            return Response(
                content=b'',
                status_code=422,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*"
                }
            )

        mask = res.masks.data.cpu().numpy()
        union = (mask.sum(axis=0) > 0).astype(np.uint8)
        refined = post_refine(union)

        mask_img = (refined * 255).astype(np.uint8)
        _, png = cv2.imencode(".png", mask_img)

        return Response(
            content=png.tobytes(),
            media_type="image/png",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return Response(
            content=str(e).encode(),
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*"
            }
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