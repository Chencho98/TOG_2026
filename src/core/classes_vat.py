"""
Core data structures and cache helpers for the Visual Assistance Tool pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence
import json
import threading
import time

import cv2
import numpy as np


# ============================================================================
# STATES AND PROTOCOLS
# ============================================================================

# Estado usado cuando la proyeccion corresponde a diapositivas del PDF.
PRESENTATION_STATE = "PRESENTACION"
# Estado usado cuando la proyeccion corresponde a un editor o codigo.
CODE_STATE = "CODIGO"
# Estado usado cuando la proyeccion no es presentacion ni codigo.
OTHER_STATE = "OTRO"

# Dimension maxima usada al guardar mascaras de persona dentro del ROI.
ROI_PERSON_MASK_SAVE_MAX_DIM = 192


class SystemState(Enum):
    # Estado inicial mientras se detectan y calibran las regiones principales.
    CALIBRATING = "calibrating"
    # Estado normal cuando el sistema ya procesa el video en seguimiento.
    MONITORING = "monitoring"


class WriterProtocol(Protocol):
    # Encola una imagen para escritura asincrona en disco.
    def submit_image(
        self,
        path: Path,
        image: np.ndarray,
        params: Optional[Sequence[int]] = None,
    ) -> None:
        ...

    # Encola datos JSON para escritura asincrona en disco.
    def submit_json(
        self,
        path: Path,
        data: Any,
        *,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> None:
        ...


# ============================================================================
# CLASSIFICATION AND DETECTION DATA
# ============================================================================

@dataclass
class ProjectionClassification:
    frame_number: int
    state: str
    confidence: float
    page_index: int = -1
    page_number: Optional[int] = None
    logo_score: float = 0.0
    ocr_confidence: float = 0.0
    language: Optional[str] = None
    code_text: str = ""
    code_ocr_method: Optional[str] = None
    blue_ratio: float = 0.0
    code_active_file: Optional[str] = None
    code_known_files: int = 0
    code_structure_changed: bool = False
    debug: Optional[Dict[str, Any]] = field(default=None, repr=False)

    # Convierte la instancia a un diccionario serializable.
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": int(self.frame_number),
            "state": self.state,
            "confidence": float(self.confidence),
            "page_index": int(self.page_index),
            "page_number": self.page_number,
            "logo_score": float(self.logo_score),
            "ocr_confidence": float(self.ocr_confidence),
            "language": self.language,
            "code_text": self.code_text,
            "code_ocr_method": self.code_ocr_method,
            "blue_ratio": float(self.blue_ratio),
            "code_active_file": self.code_active_file,
            "code_known_files": int(self.code_known_files),
            "code_structure_changed": bool(self.code_structure_changed),
        }


@dataclass
class PersonDetection:
    bbox: List[int]
    confidence: float
    frame_number: int
    tracking_id: Optional[int] = None


# ============================================================================
# REGION MODELS
# ============================================================================

@dataclass
class WhiteboardRegion:
    bbox: List[int]
    confidence: float
    mask: Optional[np.ndarray] = None
    polygon: Optional[List[List[int]]] = None
    last_roi: Optional[np.ndarray] = None
    last_person_mask: Optional[np.ndarray] = None

    # Extrae el recorte de la region desde el frame recibido.
    def extract_roi(self, frame: np.ndarray, use_mask: bool = True) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if use_mask and self.mask is not None:
            masked = cv2.bitwise_and(frame, frame, mask=self.mask)
            return masked[y1:y2, x1:x2].copy()

        return frame[y1:y2, x1:x2].copy()


@dataclass
class ProjectionRegion:
    bbox: List[int]
    mask: Optional[np.ndarray]
    confidence: float
    projection_type: str
    polygon: Optional[List[List[int]]] = None
    last_roi: Optional[np.ndarray] = None
    last_person_mask: Optional[np.ndarray] = None
    is_active: bool = True
    is_calibrated: bool = False
    last_seen_frame: int = 0
    context_state: str = OTHER_STATE
    context_confidence: float = 0.0
    matched_page_index: Optional[int] = None
    matched_page_number: Optional[int] = None
    matched_page_confidence: float = 0.0
    code_language: Optional[str] = None
    code_text: str = ""
    code_ocr_method: Optional[str] = None
    code_active_file: Optional[str] = None
    code_known_files: int = 0
    code_structure_changed: bool = False

    # Extrae el recorte de la region desde el frame recibido.
    def extract_roi(self, frame: np.ndarray, use_mask: bool = True) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if use_mask and self.mask is not None:
            masked = cv2.bitwise_and(frame, frame, mask=self.mask)
            return masked[y1:y2, x1:x2].copy()

        return frame[y1:y2, x1:x2].copy()

    # Limpia el contexto clasificado asociado a la region.
    def clear_context(self) -> None:
        self.context_state = OTHER_STATE
        self.context_confidence = 0.0
        self.matched_page_index = None
        self.matched_page_number = None
        self.matched_page_confidence = 0.0
        self.code_language = None
        self.code_text = ""
        self.code_ocr_method = None
        self.code_active_file = None
        self.code_known_files = 0
        self.code_structure_changed = False

    # Copia una clasificacion de contenido al estado de la proyeccion.
    def apply_classification(self, classification: ProjectionClassification) -> None:
        self.context_state = classification.state
        self.context_confidence = float(classification.confidence)
        self.code_language = classification.language
        self.code_text = classification.code_text
        self.code_ocr_method = classification.code_ocr_method
        self.code_active_file = classification.code_active_file
        self.code_known_files = int(classification.code_known_files)
        self.code_structure_changed = bool(classification.code_structure_changed)

        if classification.state == PRESENTATION_STATE:
            self.matched_page_index = classification.page_index
            self.matched_page_number = classification.page_number
            self.matched_page_confidence = float(classification.confidence)
        else:
            self.matched_page_index = None
            self.matched_page_number = None
            self.matched_page_confidence = 0.0


# ============================================================================
# SESSION CACHE
# ============================================================================

class RuntimeCache:
    """Fast in-memory cache shared only from the coordinator thread."""

    # Inicializa la instancia y configura su estado interno.
    def __init__(self):
        self._lock = threading.Lock()
        self.current_whiteboard_roi: Optional[np.ndarray] = None
        self.current_whiteboard_person_mask: Optional[np.ndarray] = None
        self.current_projection_roi: Optional[np.ndarray] = None
        self.current_projection_person_mask: Optional[np.ndarray] = None

    # Actualiza en memoria el ROI y mascara del pizarron.
    def update_whiteboard_roi(
        self,
        roi: np.ndarray,
        person_mask: Optional[np.ndarray] = None,
    ) -> None:
        with self._lock:
            self.current_whiteboard_roi = roi.copy()
            self.current_whiteboard_person_mask = None if person_mask is None else person_mask.copy()

    # Actualiza en memoria el ROI y mascara de la proyeccion.
    def update_projection_roi(
        self,
        roi: np.ndarray,
        person_mask: Optional[np.ndarray] = None,
    ) -> None:
        with self._lock:
            self.current_projection_roi = roi.copy()
            self.current_projection_person_mask = None if person_mask is None else person_mask.copy()

    # Limpia el estado almacenado en memoria.
    def clear(self) -> None:
        with self._lock:
            self.current_whiteboard_roi = None
            self.current_whiteboard_person_mask = None
            self.current_projection_roi = None
            self.current_projection_person_mask = None


class SessionCache:
    """
    Session cache with asynchronous disk persistence support.

    L1 stays in memory. L2 writes are optionally delegated to an async writer.
    """

    # Inicializa la instancia y configura su estado interno.
    def __init__(
        self,
        session_id: str,
        output_dir: str = "output/sessions",
        writer: Optional[WriterProtocol] = None,
    ):
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.session_dir = self.output_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.runtime = RuntimeCache()
        self.writer = writer

        self._classification_buffer: List[Dict[str, Any]] = []
        self._classification_flush_threshold = 10
        self._flush_index = 0
        self._lock = threading.Lock()

    # Actualiza y persiste el ROI actual del pizarron.
    def update_whiteboard(
        self,
        roi: np.ndarray,
        frame_number: int,
        person_mask: Optional[np.ndarray] = None,
    ) -> None:
        if person_mask is None:
            person_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        self.runtime.update_whiteboard_roi(roi, person_mask)
        self._write_roi("whiteboard", roi, frame_number)
        self._write_person_mask("whiteboard", person_mask, frame_number)

    # Actualiza y persiste el ROI actual de la proyeccion.
    def update_projection(
        self,
        roi: np.ndarray,
        frame_number: int,
        person_mask: Optional[np.ndarray] = None,
    ) -> None:
        if person_mask is None:
            person_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        self.runtime.update_projection_roi(roi, person_mask)
        self._write_roi("projection", roi, frame_number)
        self._write_person_mask("projection", person_mask, frame_number)

    # Agrega una clasificacion al buffer persistente de la sesion.
    def record_classification(self, classification: ProjectionClassification) -> None:
        with self._lock:
            self._classification_buffer.append(classification.to_dict())
            should_flush = len(self._classification_buffer) >= self._classification_flush_threshold

        if should_flush:
            self.flush()

    # Fuerza la escritura de los datos acumulados pendientes.
    def flush(self) -> None:
        with self._lock:
            if not self._classification_buffer:
                return
            payload = list(self._classification_buffer)
            self._classification_buffer.clear()
            self._flush_index += 1

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.session_dir / f"classifications_{timestamp}_{self._flush_index:03d}.json"
        self._write_json(path, payload)

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> None:
        self.flush()

    # Escribe roi.
    def _write_roi(self, roi_type: str, roi: np.ndarray, frame_number: int) -> None:
        path = self.session_dir / "rois" / roi_type / f"frame_{frame_number:06d}.jpg"
        self._write_image(path, roi, params=[cv2.IMWRITE_JPEG_QUALITY, 92])

    # Escribe person mask.
    def _write_person_mask(self, roi_type: str, person_mask: np.ndarray, frame_number: int) -> None:
        path = self.session_dir / "roi_person_masks" / roi_type / f"frame_{frame_number:06d}.png"
        mask_to_save = self._resize_mask_for_disk(person_mask)
        self._write_image(path, mask_to_save)

    # Maneja resize mask for disk dentro de SessionCache.
    def _resize_mask_for_disk(self, person_mask: np.ndarray) -> np.ndarray:
        mask_to_save = person_mask
        h, w = person_mask.shape[:2]
        max_dim = max(h, w)
        if max_dim > ROI_PERSON_MASK_SAVE_MAX_DIM:
            scale = ROI_PERSON_MASK_SAVE_MAX_DIM / float(max_dim)
            target_w = max(1, int(round(w * scale)))
            target_h = max(1, int(round(h * scale)))
            resized = cv2.resize(person_mask, (target_w, target_h), interpolation=cv2.INTER_AREA)
            mask_to_save = np.where(resized > 0, 255, 0).astype(np.uint8)
        return mask_to_save

    # Escribe image.
    def _write_image(
        self,
        path: Path,
        image: np.ndarray,
        params: Optional[Sequence[int]] = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.writer is not None:
            self.writer.submit_image(path, image, params=params)
            return

        if params:
            cv2.imwrite(str(path), image, list(params))
        else:
            cv2.imwrite(str(path), image)

    # Escribe JSON.
    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.writer is not None:
            self.writer.submit_json(path, data, indent=2, ensure_ascii=False)
            return

        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, ensure_ascii=False)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CODE_STATE",
    "OTHER_STATE",
    "PRESENTATION_STATE",
    "PersonDetection",
    "ProjectionClassification",
    "ProjectionRegion",
    "RuntimeCache",
    "SessionCache",
    "SystemState",
    "WhiteboardRegion",
]
