import os, io, cv2, torch, numpy as np, gc, logging, psutil
from PIL import Image
from ultralytics import YOLO, SAM
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

YOLO_CONF_THRESHOLD = 0.001
MIN_BOX_RATIO = 0.003
MORPHOLOGY_KERNEL_SIZE = 11
GAUSSIAN_BLUR_SIZE = 21
DEPTH_DIFF_THRESHOLD = 8
MAX_IMAGE_SIZE_PIXELS = 640
CEILING_EXCLUSION_RATIO = 0.15
FLOOR_EXCLUSION_RATIO = 0.20
MIN_WALL_WIDTH_RATIO = 0.40

det_model = None
sam_model = None
device = "cpu"

@app.on_event("startup")
def load_models_on_startup():
    global det_model, sam_model, device
    device = "cpu"
    try:
        if os.path.exists("yolov8s.pt"):
            det_model = YOLO("yolov8s.pt"); det_model.to(device)
            logger.info("[YOLOv8s Loaded]")
        else: logger.error("YOLOv8s checkpoint missing.")
        if os.path.exists("mobile_sam.pt"):
            sam_model = SAM("mobile_sam.pt"); sam_model.to(device)
            logger.info("[MobileSAM Loaded]")
        else: logger.error("MobileSAM checkpoint missing.")
    except Exception as e:
        logger.error(f"Model load failed: {e}", exc_info=True)

def np_from_upload(file_bytes: bytes, mode="RGB"):
    try: return Image.open(io.BytesIO(file_bytes)).convert(mode)
    except: return None

def filter_small_boxes(boxes, img_shape, min_ratio=MIN_BOX_RATIO):
    H, W = img_shape; area = H * W
    return [[float(x1), float(y1), float(x2), float(y2)]
            for x1, y1, x2, y2 in boxes
            if (x2-x1)*(y2-y1)/area > min_ratio]

def post_refine(mask):
    mask = mask.astype(np.uint8)
    k = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.dilate(mask, k)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return mask
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [max(cnts, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    return cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=2)

def post_filter_vertical_plane(mask, c_ratio, f_ratio, width_ratio):
    H, W = mask.shape; cut = mask.copy()
    c,h = int(H*c_ratio), int(H*f_ratio)
    if c>0: cut[:c] = 0
    if h>0: cut[H-h:] = 0
    cnts,_ = cv2.findContours(cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return np.zeros_like(mask)
    x,y,w_box,h_box = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if w_box/W < width_ratio: return np.zeros_like(mask)
    restored = np.zeros_like(mask); restored[:, x:x+w_box] = 255
    return cv2.bitwise_and(mask, restored)

def create_depth_occlusion_mask(depth_map, threshold=DEPTH_DIFF_THRESHOLD):
    if depth_map is None: return None
    depth_map = depth_map.astype(np.float32)
    gx = cv2.Sobel(depth_map, cv2.CV_32F, 1,0)
    gy = cv2.Sobel(depth_map, cv2.CV_32F, 0,1)
    mag = cv2.magnitude(gx, gy)
    m = (mag>threshold).astype(np.uint8)
    return cv2.dilate(m, np.ones((5,5), np.uint8), iterations=2)

@app.get("/")
async def root(): return {"status":"ok"}

@app.get("/health")
async def health():
    memory = psutil.Process().memory_info().rss/1024/1024; gc.collect()
    return {"status":"healthy", "device":device, "memory_mb":round(memory,2),
            "models_loaded": det_model is not None and sam_model is not None}

@app.post("/segment_wall_mask")
async def segment_wall_mask(rgb_file:UploadFile=File(...), depth_file:UploadFile=File(...)):
    try:
        process = psutil.Process()
        rgb_bytes = await rgb_file.read()
        pil_img = np_from_upload(rgb_bytes)
        if pil_img is None: return Response(b"Invalid RGB", status_code=400)
        w,h = pil_img.size; ori = pil_img.size
        if max(w,h) > MAX_IMAGE_SIZE_PIXELS:
            r = MAX_IMAGE_SIZE_PIXELS/max(w,h)
            pil_img = pil_img.resize((int(w*r),int(h*r)), Image.LANCZOS)
            w,h = pil_img.size

        depth_bytes = await depth_file.read()
        if depth_bytes and len(depth_bytes)>100:
            try:
                d = Image.open(io.BytesIO(depth_bytes))
                depth = np.array(d.convert("L").resize((w,h), Image.NEAREST))
            except: depth=None
        else: depth=None

        results = det_model.predict(pil_img, conf=YOLO_CONF_THRESHOLD, imgsz=640, device=device, verbose=False)[0]
        boxes = filter_small_boxes(results.boxes.xyxy.cpu().numpy(), pil_img.size[::-1]) if results.boxes is not None else []
        if not boxes:
            wall = np.ones((h,w),np.uint8)*255
        else:
            res = sam_model.predict(pil_img, bboxes=boxes, device=device, retina_masks=False, verbose=False)[0]
            if res.masks is None: wall=np.ones((h,w),np.uint8)*255
            else:
                union=(res.masks.data.cpu().numpy().sum(0)>0).astype(np.uint8)
                wall = (post_refine(1-union)*255).astype(np.uint8)

        if depth is not None:
            occ = create_depth_occlusion_mask(depth)
            if occ is not None: wall = cv2.bitwise_and(wall, (1-occ)*255)

        wall = post_filter_vertical_plane(wall, CEILING_EXCLUSION_RATIO, FLOOR_EXCLUSION_RATIO, MIN_WALL_WIDTH_RATIO)
        wall = (wall>0).astype(np.uint8)*255
        wall = cv2.GaussianBlur(wall,(GAUSSIAN_BLUR_SIZE,GAUSSIAN_BLUR_SIZE),0)
        if pil_img.size!=ori: wall = cv2.resize(wall, ori, cv2.INTER_LINEAR)
        _,png = cv2.imencode(".png", wall)

        return Response(content=png.tobytes(), media_type="image/png", headers={"Access-Control-Allow-Origin":"*","Cache-Control":"no-cache"})
    except Exception as e:
        gc.collect()
        return Response(content=b"Segmentation Failed", status_code=500)

@app.options("/segment_wall_mask")
async def opt():
    return Response(b"",200,{"Access-Control-Allow-Origin":"*","Access-Control-Allow-Methods":"POST, OPTIONS","Access-Control-Allow-Headers":"*"})
