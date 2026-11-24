import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import numpy as np
import argparse
import time
import threading

# ==========================
# CONFIG
# ==========================

GDINO_TEXT_QUERY = "whiteboard . pizarron . blackboard"

GDINO_BOX_TH = 0.30
GDINO_TEXT_TH = 0.30

YOLO_CONF_TH = 0.50
YOLO_MODEL = "yolov8n-seg.pt"

PROJECTION_CLASSES = {"tv", "monitor", "laptop"}

GDINO_EVERY_N_FRAMES = 2

COLOR_PERSON     = (0, 0, 255)
COLOR_PROJECTION = (255, 0, 0)
COLOR_WHITEBOARD = (0, 255, 80)

# ==========================
# DEVICE
# ==========================

def resolve_device():
    try:
        import intel_extension_for_pytorch  as ipex
        if torch.xpu.is_available():
            print(f"✅ Intel XPU: {torch.xpu.get_device_name(0)}")
            return "xpu", ipex
        return "cpu", ipex
    except ImportError:
        return "cpu", None

DEVICE, ipex = resolve_device()
print(f"⚙️  Dispositivo: {DEVICE}\n")

# ==========================
# MODELOS
# ==========================

print("📦 Cargando YOLO...")
from ultralytics import YOLO

yolo_device = 0 if DEVICE == "cuda" else "cpu"
yolo_model  = YOLO(YOLO_MODEL)

print("📦 Cargando Grounding DINO...")

import groundingdino
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from PIL import Image as PILImage

GDINO_CONFIG = os.path.join(os.path.dirname(groundingdino.__file__), "config")
GDINO_WEIGHTS = "weights/groundingdino_swint_ogc.pth"

gdino_model = load_model(GDINO_CONFIG, GDINO_WEIGHTS, device=DEVICE)
gdino_model.eval()

print("✅ Modelos listos\n")

# ==========================
# THREAD STATE
# ==========================

whiteboard_cache = []
gdino_running = False

# ==========================
# GDINO
# ==========================

def frame_to_gdino_tensor(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = PILImage.fromarray(frame_rgb)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    img_tensor,_ = transform(pil_img,None)
    return img_tensor


def run_gdino(frame):

    h,w = frame.shape[:2]
    img_tensor = frame_to_gdino_tensor(frame)