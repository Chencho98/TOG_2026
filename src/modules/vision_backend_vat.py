"""
Shared imports for the classroom video processor modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import time
from ultralytics import YOLO

import cv2
import numpy as np

try:
    from src.modules.project_config_vat import *
except ImportError:
    try:
        from project_config_vat import *
    except ImportError:
        from .project_config_vat import *

# ============================================================================
# INFERENCE BACKEND
# ============================================================================


class VisionBackend:
    def __init__(
        self,
        *,
        yolo_model_name: str = YOLO_MODEL,
        gdino_weights: str = GDINO_WEIGHTS,
    ):
        self.yolo_model_name = yolo_model_name
        self.gdino_weights = gdino_weights

        self.device = "cpu"  # torch device para GDINO
        self.yolo_device: object = "cpu"
        self.yolo_model_path = self.yolo_model_name
        self._yolo_backends: List[tuple[str, str, object]] = []
        self._yolo_backend_index = 0
        self._loaded = False

        # Importante: aquí debe ir el modelo exportado a OpenVINO
        self._yolo_model = None

        self._gdino_model = None
        self._gdino_predict = None
        self._gdino_transforms = None
        self._pil_image_cls = None
        self._torch = None

    def resolve_device_candidates(self) -> List[str]:
        import torch

        candidates: List[str] = []
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
            candidates.append("cuda")

        try:
            import intel_extension_for_pytorch  # noqa: F401

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                print(f"✅ Intel XPU: {torch.xpu.get_device_name(0)}")
                candidates.append("xpu")
        except ImportError:
            pass

        candidates.append("cpu")
        return candidates

    def resolve_device(self) -> str:
        return self.resolve_device_candidates()[0]

    def _build_yolo_backends(self) -> List[tuple[str, str, object]]:
        backends: List[tuple[str, str, object]] = []
        for device in self.resolve_device_candidates():
            if device == "cuda":
                backends.append(("cuda", self.yolo_model_name, 0))
            elif device == "xpu":
                backends.append(("xpu", "yolov8n-seg_openvino_model/", "intel:gpu"))
            else:
                backends.append(("cpu", "yolov8n-seg_openvino_model/", "intel:cpu"))
                backends.append(("cpu", self.yolo_model_name, "cpu"))
        return backends

    def _activate_yolo_backend(self, index: int) -> None:
        backend_name, model_path, yolo_device = self._yolo_backends[index]
        self._yolo_model = YOLO(model_path)
        self._yolo_backend_index = index
        self.yolo_model_path = model_path
        self.yolo_device = yolo_device
        print(f"⚙️ YOLO backend: {backend_name} ({model_path}, {yolo_device})")


    def load(self) -> None:
        if self._loaded:
            return

        import torch
        import groundingdino
        from groundingdino.util.inference import load_model, predict
        import groundingdino.datasets.transforms as T
        from PIL import Image as PILImage

        self._torch = torch

        print("📦 Cargando modelos...")

        gdino_config = os.path.join(
            os.path.dirname(groundingdino.__file__),
            "config",
            "GroundingDINO_SwinT_OGC.py",
        )
        gdino_error: Optional[Exception] = None
        for device in self.resolve_device_candidates():
            try:
                self._gdino_model = load_model(gdino_config, self.gdino_weights, device=device)
                self._gdino_model.eval()
                self.device = device
                break
            except Exception as exc:
                gdino_error = exc
                print(f"⚠️ GroundingDINO no pudo usar {device}: {exc}")
        if self._gdino_model is None:
            raise RuntimeError("No se pudo cargar GroundingDINO en CUDA, XPU ni CPU") from gdino_error

        self._yolo_backends = self._build_yolo_backends()
        yolo_error: Optional[Exception] = None
        for index in range(len(self._yolo_backends)):
            try:
                self._activate_yolo_backend(index)
                break
            except Exception as exc:
                yolo_error = exc
                print(f"⚠️ YOLO no pudo cargar {self._yolo_backends[index][0]}: {exc}")
        if self._yolo_model is None:
            raise RuntimeError("No se pudo cargar YOLO en CUDA, XPU ni CPU") from yolo_error

        self._gdino_predict = predict
        self._gdino_transforms = T
        self._pil_image_cls = PILImage

        self._loaded = True
        print(f"⚙️ Dispositivo GDINO: {self.device}")
        print(f"⚙️ Dispositivo YOLO: {self.yolo_device}")
        print("✅ Modelos listos\n")


    def run_yolo(self, frame: np.ndarray) -> List[Dict]:
        self.load()

        last_error: Optional[Exception] = None
        while self._yolo_backend_index < len(self._yolo_backends):
            try:
                results = self._yolo_model.predict(
                    source=frame,
                    device=self.yolo_device,
                    imgsz=640,
                    conf=YOLO_CONF_TH,
                    verbose=False,
                )
                break
            except Exception as exc:
                last_error = exc
                next_index = self._yolo_backend_index + 1
                if next_index >= len(self._yolo_backends):
                    raise RuntimeError("YOLO falló en todos los backends disponibles") from exc
                print(f"⚠️ YOLO fallback desde {self._yolo_backends[self._yolo_backend_index][0]}: {exc}")
                self._activate_yolo_backend(next_index)
        else:
            raise RuntimeError("YOLO no tiene un backend disponible") from last_error

        persons = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self._yolo_model.names[cls_id]

                if cls_name != "person":
                    continue

                persons.append(
                    {
                        "bbox": list(map(int, box.xyxy[0].tolist())),
                        "conf": float(box.conf[0]),
                    }
                )

        return persons

    # Convierte un frame BGR al tensor esperado por GroundingDINO.
    def frame_to_gdino_tensor(self, frame_bgr: np.ndarray):
        self.load()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = self._pil_image_cls.fromarray(frame_rgb)
        transform = self._gdino_transforms.Compose(
            [
                self._gdino_transforms.RandomResize([800], max_size=1333),
                self._gdino_transforms.ToTensor(),
                self._gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img_tensor, _ = transform(pil_img, None)
        return img_tensor

    # Ejecuta GroundingDINO con una consulta textual concreta.
    def run_gdino_query(
        self,
        frame: np.ndarray,
        caption: str,
        box_threshold: float,
        text_threshold: float,
    ) -> List[Dict]:
        self.load()
        h, w = frame.shape[:2]
        img_tensor = self.frame_to_gdino_tensor(frame)

        boxes, scores, labels = self._gdino_predict(
            model=self._gdino_model,
            image=img_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            aspect_ratio = (x2 - x1) / float(max(1, y2 - y1))
            if aspect_ratio < 1.9:
                continue

            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": float(score),
                    "label": label,
                }
            )
        return detections

    # Detecta el pizarron con GroundingDINO.
    def run_gdino(self, frame: np.ndarray) -> List[Dict]:
        detections = self.run_gdino_query(frame, GDINO_TEXT_QUERY, GDINO_BOX_TH, GDINO_TEXT_TH)
        if not detections:
            return []
        return [max(detections, key=lambda item: item["conf"])]
