"""
Sistema de clases y caché multinivel
Integrado con Grounding DINO + YOLO
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from collections import deque
import json
from pathlib import Path
import time

# ==============================================================================
# ENUMS
# ==============================================================================

class InfoType(Enum):
    TEXTO_ALFANUMERICO = "texto"
    IMAGEN = "imagen"
    CODIGO = "codigo"
    DIAGRAMA = "diagrama"
    ECUACION = "ecuacion"
    GRAFICA = "grafica"

class SystemState(Enum):
    CALIBRATING = "calibrating"
    MONITORING = "monitoring"
    PROCESSING_CONTENT = "processing"
    RECALIBRATING = "recalibrating"

class ProcessingMethod(Enum):
    LOCAL_OCR = "local"
    CLOUD_API = "cloud"
    HYBRID = "hybrid"

# ==============================================================================
# CLASES DE DATOS
# ==============================================================================

@dataclass
class DetectedInfo:
    """Información detectada en un frame"""
    type: InfoType
    content: str
    raw_roi: np.ndarray
    confidence: float
    timestamp: float
    bbox_relative: List[int]
    processing_method: ProcessingMethod
    frame_number: int

    def to_dict(self):
        return {
            'type': self.type.value,
            'content': self.content,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'bbox': self.bbox_relative,
            'method': self.processing_method.value,
            'frame': self.frame_number
        }

@dataclass
class Persona:
    """Persona detectada por foto"""
    bbox: List[int]
    confidence: float
    frame_number: int
    tracking_id: Optional[int] = None

    def overlaps_with(self, region_bbox: List(int), threshold=0.1) -> bool:
            """Verifica si se superpone con una región"""
            x1, y1, x2, y2 = self.bbox
            rx1, ry1, rx2, ry2 = region_bbox
            
            inter_x1 = max(x1, rx1)
            inter_y1 = max(y1, ry1)
            inter_x2 = min(x2, rx2)
            inter_y2 = min(y2, ry2)
            
            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                return False

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            person_area = (x2 - x1) * (y2 - y1)
            
            overlap_ratio = inter_area / person_area if person_area > 0 else 0
            
            return overlap_ratio > threshold
        
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

# ==============================================================================
# REGIONES DE INTERÉS
# ==============================================================================

class Pizarron:
    """Pizarrón detectado con Grounding DINO"""