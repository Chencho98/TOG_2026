"""
Shared imports for the code reconstruction pipeline modules.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import base64
import difflib
import gc
import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from src.core.classes_vat import (
        CODE_STATE,
        OTHER_STATE,
        PRESENTATION_STATE,
        PersonDetection,
        ProjectionClassification,
        ProjectionRegion,
        SessionCache,
        SystemState,
        WhiteboardRegion,
    )
except ImportError:
    try:
        from classes_vat import (
            CODE_STATE,
            OTHER_STATE,
            PRESENTATION_STATE,
            PersonDetection,
            ProjectionClassification,
            ProjectionRegion,
            SessionCache,
            SystemState,
            WhiteboardRegion,
        )
    except ImportError:
        from .classes_vat import (
            CODE_STATE,
            OTHER_STATE,
            PRESENTATION_STATE,
            PersonDetection,
            ProjectionClassification,
            ProjectionRegion,
            SessionCache,
            SystemState,
            WhiteboardRegion,
        )

try:
    from src.modules.code_config_vat import *
except ImportError:
    try:
        from code_config_vat import *
    except ImportError:
        from .code_config_vat import *

try:
    from src.modules.code_utils_vat import *
except ImportError:
    try:
        from code_utils_vat import *
    except ImportError:
        from .code_utils_vat import *

try:
    from src.modules.code_models_vat import *
except ImportError:
    try:
        from code_models_vat import *
    except ImportError:
        from .code_models_vat import *

try:
    from src.modules.code_vision_vat import *
except ImportError:
    try:
        from code_vision_vat import *
    except ImportError:
        from .code_vision_vat import *

try:
    from src.modules.code_pipeline_vat import *
except ImportError:
    try:
        from code_pipeline_vat import *
    except ImportError:
        from .code_pipeline_vat import *

# ============================================================================
# PROJECTION CODE CLASSIFIER
# ============================================================================

class CodeClassifier:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self, debug_visualization: bool = False):
        self.debug_visualization = bool(debug_visualization)
        self.last_code_frame = -1
        self.last_status_llm_frame = -1
        self.last_input_frame = -1
        self.observed_input_gap_frames = 0
        self.last_code_result: Optional[Dict[str, Any]] = None
        self.session_dir: Optional[Path] = None
        self._frames_original_dir: Optional[Path] = None
        self.code_pipeline = CodeIntelligencePipeline(debug_visualization=debug_visualization)
        self.status_llm_client: Optional[OpenAICodeVisionClient] = None
        self.status_llm_warning_emitted = False
        self._finalized = False
        self._final_summary: Dict[str, Any] = {}

    # Configura los directorios de salida de la sesion.
    def set_session_dir(self, session_dir: str) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._frames_original_dir = self.session_dir / "analyzed_frames" / "original"
        self._frames_original_dir.mkdir(parents=True, exist_ok=True)
        self.code_pipeline.set_session_dir(self.session_dir)

    # Finaliza el procesamiento pendiente y devuelve el resumen producido.
    def finalize(self) -> Dict[str, Any]:
        if self._finalized:
            return dict(self._final_summary)
        self._final_summary = self.code_pipeline.finalize()
        self._finalized = True
        return dict(self._final_summary)

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> None:
        self.finalize()

    # Classification result helpers.
    # Aplica pipeline summary.
    def _apply_pipeline_summary(
        self,
        result: ProjectionClassification,
        pipeline_summary: Optional[Dict[str, Any]],
    ) -> ProjectionClassification:
        if not pipeline_summary:
            return result
        result.code_active_file = pipeline_summary.get("active_file")
        result.code_known_files = int(pipeline_summary.get("known_files", 0) or 0)
        result.code_structure_changed = bool(pipeline_summary.get("structure_changed", False))
        return result

    # Construye local classification roi.
    def _build_local_classification_roi(self, roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        return downscale_projection_roi(roi)

    # Maneja submit code pipeline dentro de CodeClassifier.
    def _submit_code_pipeline(
        self,
        roi: np.ndarray,
        frame_number: int,
        fps: float,
        language_hint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if roi is None or roi.size == 0:
            return None
        effective_language = language_hint
        if not effective_language and self.last_code_result is not None:
            effective_language = self.last_code_result.get("language")
        return self.code_pipeline.submit_sample(
            roi,
            frame_number=frame_number,
            fps=fps,
            language_hint=effective_language,
        )

    # Guarda analyzed frames.
    def _save_analyzed_frames(
        self,
        frame_number: int,
        original_roi: np.ndarray,
    ) -> None:
        if (
            self.session_dir is None
            or self._frames_original_dir is None
            or frame_number < 0
        ):
            return

        original_path = self._frames_original_dir / f"frame_{frame_number:06d}.jpg"
        if isinstance(original_roi, np.ndarray) and original_roi.size > 0:
            cv2.imwrite(str(original_path), original_roi, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Maneja finalize projection result dentro de CodeClassifier.
    def _finalize_projection_result(
        self,
        result: ProjectionClassification,
        debug_payload: Optional[Dict[str, Any]],
        pipeline_summary: Optional[Dict[str, Any]],
        low_roi: np.ndarray,
        original_roi: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProjectionClassification:
        if debug_payload is not None:
            debug_payload = self._merge_debug_payload(debug_payload, pipeline_summary)
        result = self._apply_pipeline_summary(result, pipeline_summary)
        if result.state == CODE_STATE and result.frame_number >= 0:
            self.last_code_frame = int(result.frame_number)

        if result.state == CODE_STATE:
            self._save_analyzed_frames(
                frame_number=int(result.frame_number),
                original_roi=original_roi,
            )
        return self._attach_debug(result, debug_payload)

    # Clasifica el ROI de proyeccion como presentacion, codigo u otro.
    def classify_projection(
        self,
        roi: np.ndarray,
        frame_number: int = -1,
        fps: float = 0.0,
        original_roi: Optional[np.ndarray] = None,
        allow_status_llm: bool = True,
    ) -> ProjectionClassification:
        base = ProjectionClassification(
            frame_number=frame_number,
            state=OTHER_STATE,
            confidence=0.0,
        )
        self._record_input_frame(frame_number)
        if roi is None or roi.size == 0:
            return self._finalize_projection_result(
                base,
                self._empty_debug_payload() if self.debug_visualization else None,
                None,
                np.zeros((0, 0, 3), dtype=np.uint8),
                np.zeros((0, 0, 3), dtype=np.uint8),
                metadata={"classification_source": "empty"},
            )

        original_code_roi = original_roi if isinstance(original_roi, np.ndarray) and original_roi.size > 0 else roi
        local_roi = self._build_local_classification_roi(roi)
        status_bar = self._extract_status_bar(local_roi)
        original_status_bar = self._extract_status_bar(original_code_roi)
        text_region = self._extract_status_text_region(original_status_bar)
        if text_region is None or text_region.size == 0:
            text_region = self._extract_status_text_region(status_bar)
        blue_ratio, blue_mask = self._compute_blue_features(status_bar)

        pipeline_summary = None
        debug_payload = None
        if self.debug_visualization:
            debug_payload = {
                "status_bar": status_bar.copy() if isinstance(status_bar, np.ndarray) else np.zeros((0, 0), dtype=np.uint8),
                "text_region": text_region.copy() if isinstance(text_region, np.ndarray) else np.zeros((0, 0), dtype=np.uint8),
                "blue_mask": blue_mask.copy() if isinstance(blue_mask, np.ndarray) else np.zeros((0, 0), dtype=np.uint8),
                "text_view": np.zeros((0, 0), dtype=np.uint8),
                "ocr_image": np.zeros((0, 0), dtype=np.uint8),
                "layout_overlay": np.zeros((0, 0, 3), dtype=np.uint8),
                "layout_sheet": np.zeros((0, 0, 3), dtype=np.uint8),
                "sidebar_crop": np.zeros((0, 0, 3), dtype=np.uint8),
                "editor_crop": np.zeros((0, 0, 3), dtype=np.uint8),
                "sidebar_edges": np.zeros((0, 0, 3), dtype=np.uint8),
            }

        if blue_ratio < CODE_MIN_BLUE_RATIO:
            cached = self._build_cached_code_result(frame_number, blue_ratio, fps)
            if cached is not None:
                pipeline_summary = self._submit_code_pipeline(
                    original_code_roi,
                    frame_number=frame_number,
                    fps=fps,
                    language_hint=cached.language,
                )
                return self._finalize_projection_result(
                    cached,
                    debug_payload,
                    pipeline_summary,
                    local_roi,
                    original_code_roi,
                    metadata={
                        "classification_source": "cached_low_res",
                        "used_original_code_roi": bool(pipeline_summary),
                    },
                )
            if blue_ratio < CODE_KEEP_BLUE_RATIO:
                self._clear_cached_code()
            base.confidence = float(min(0.25, blue_ratio / max(CODE_MIN_BLUE_RATIO, 1e-6)))
            base.blue_ratio = float(blue_ratio)
            return self._finalize_projection_result(
                base,
                debug_payload,
                None,
                local_roi,
                original_code_roi,
                metadata={"classification_source": "low_res_blue_reject"},
            )

        prepared = self._prepare_text_for_ocr(text_region)
        llm_image = self._prepare_text_for_status_llm(text_region, original_status_bar)
        if debug_payload is not None:
            if prepared["view"].size > 0:
                debug_payload["text_view"] = prepared["view"].copy()
            if llm_image.size > 0:
                debug_payload["ocr_image"] = llm_image.copy()
            elif prepared["ocr_image"].size > 0:
                debug_payload["ocr_image"] = prepared["ocr_image"].copy()

        if not allow_status_llm:
            cached = self._build_cached_code_result(frame_number, blue_ratio, fps)
            if cached is not None:
                return self._finalize_projection_result(
                    cached,
                    debug_payload,
                    None,
                    local_roi,
                    original_code_roi,
                    metadata={
                        "classification_source": "cached_status_llm_suppressed",
                        "used_original_code_roi": False,
                    },
                )
            base.confidence = float(min(0.45, blue_ratio / max(CODE_STRONG_BLUE_RATIO, 1e-6)))
            base.blue_ratio = float(blue_ratio)
            return self._finalize_projection_result(
                base,
                debug_payload,
                None,
                local_roi,
                original_code_roi,
                metadata={"classification_source": "low_res_status_llm_suppressed"},
            )

        should_run_status_llm = True
        if frame_number >= 0 and self.last_status_llm_frame >= 0:
            interval_frames = self._seconds_to_frames(fps, CODE_STATUS_LLM_INTERVAL_SECONDS)
            should_run_status_llm = (frame_number - self.last_status_llm_frame) >= interval_frames

        if not should_run_status_llm:
            cached = self._build_cached_code_result(frame_number, blue_ratio, fps)
            if cached is not None:
                pipeline_summary = self._submit_code_pipeline(
                    original_code_roi,
                    frame_number=frame_number,
                    fps=fps,
                    language_hint=cached.language,
                )
                return self._finalize_projection_result(
                    cached,
                    debug_payload,
                    pipeline_summary,
                    local_roi,
                    original_code_roi,
                    metadata={
                        "classification_source": "cached_interval_skip",
                        "used_original_code_roi": bool(pipeline_summary),
                    },
                )
            base.confidence = float(min(0.45, blue_ratio / max(CODE_STRONG_BLUE_RATIO, 1e-6)))
            base.blue_ratio = float(blue_ratio)
            return self._finalize_projection_result(
                base,
                debug_payload,
                None,
                local_roi,
                original_code_roi,
                metadata={"classification_source": "low_res_status_llm_interval_skip"},
            )

        if not self._status_llm_available():
            cached = self._build_cached_code_result(frame_number, blue_ratio, fps)
            if cached is not None:
                pipeline_summary = self._submit_code_pipeline(
                    original_code_roi,
                    frame_number=frame_number,
                    fps=fps,
                    language_hint=cached.language,
                )
                return self._finalize_projection_result(
                    cached,
                    debug_payload,
                    pipeline_summary,
                    local_roi,
                    original_code_roi,
                    metadata={
                        "classification_source": "cached_no_status_llm",
                        "used_original_code_roi": bool(pipeline_summary),
                    },
                )
            base.confidence = float(min(0.45, blue_ratio / max(CODE_STRONG_BLUE_RATIO, 1e-6)))
            base.blue_ratio = float(blue_ratio)
            return self._finalize_projection_result(
                base,
                debug_payload,
                None,
                local_roi,
                original_code_roi,
                metadata={"classification_source": "low_res_no_status_llm"},
            )

        try:
            if frame_number >= 0:
                self.last_status_llm_frame = int(frame_number)
            llm_result = self._detect_code_with_status_llm(llm_image)
            self._log_code_status_llm_result(frame_number, blue_ratio, llm_result)
        except Exception:
            cached = self._build_cached_code_result(frame_number, blue_ratio, fps)
            if cached is not None:
                pipeline_summary = self._submit_code_pipeline(
                    original_code_roi,
                    frame_number=frame_number,
                    fps=fps,
                    language_hint=cached.language,
                )
                return self._finalize_projection_result(
                    cached,
                    debug_payload,
                    pipeline_summary,
                    local_roi,
                    original_code_roi,
                    metadata={
                        "classification_source": "cached_status_llm_error",
                        "used_original_code_roi": bool(pipeline_summary),
                    },
                )
            base.confidence = float(min(0.45, blue_ratio / max(CODE_STRONG_BLUE_RATIO, 1e-6)))
            base.blue_ratio = float(blue_ratio)
            return self._finalize_projection_result(
                base,
                debug_payload,
                None,
                local_roi,
                original_code_roi,
                metadata={"classification_source": "low_res_status_llm_error"},
            )

        language = llm_result.get("language")
        if (
            debug_payload is not None
            and isinstance(llm_result.get("llm_image"), np.ndarray)
            and llm_result["llm_image"].size > 0
        ):
            debug_payload["ocr_image"] = llm_result["llm_image"].copy()
        if language is None:
            cached = self._build_cached_code_result(frame_number, blue_ratio, fps)
            if cached is not None:
                pipeline_summary = self._submit_code_pipeline(
                    original_code_roi,
                    frame_number=frame_number,
                    fps=fps,
                    language_hint=cached.language,
                )
                return self._finalize_projection_result(
                    cached,
                    debug_payload,
                    pipeline_summary,
                    local_roi,
                    original_code_roi,
                    metadata={
                        "classification_source": "cached_low_res_status_llm",
                        "used_original_code_roi": bool(pipeline_summary),
                        "status_text": normalize_whitespace(llm_result.get("status_text", "")),
                        "status_llm_error": llm_result.get("error"),
                    },
                )
            base.confidence = float(min(0.45, blue_ratio / max(CODE_STRONG_BLUE_RATIO, 1e-6)))
            base.blue_ratio = float(blue_ratio)
            return self._finalize_projection_result(
                base,
                debug_payload,
                None,
                local_roi,
                original_code_roi,
                metadata={
                    "classification_source": "low_res_status_llm_unknown",
                    "status_text": normalize_whitespace(llm_result.get("status_text", "")),
                    "status_llm_error": llm_result.get("error"),
                },
            )

        blue_score = min(1.0, blue_ratio / CODE_STRONG_BLUE_RATIO)
        llm_confidence = safe_float(llm_result.get("confidence"), 0.0)
        confidence = max(0.55 + (0.35 * blue_score), llm_confidence)
        pipeline_summary = self._submit_code_pipeline(
            original_code_roi,
            frame_number=frame_number,
            fps=fps,
            language_hint=language,
        )
        result = ProjectionClassification(
            frame_number=frame_number,
            state=CODE_STATE,
            confidence=float(min(1.0, confidence)),
            language=language,
            code_text=llm_result.get("status_text", "") or "",
            code_ocr_method=llm_result.get("ocr_method"),
            blue_ratio=float(blue_ratio),
        )
        if frame_number >= 0:
            self.last_code_frame = int(frame_number)
        self.last_code_result = {
            "language": result.language,
            "ocr_text": result.code_text,
            "ocr_method": result.code_ocr_method,
            "confidence": result.confidence,
        }
        return self._finalize_projection_result(
            result,
            debug_payload,
            pipeline_summary,
            local_roi,
            original_code_roi,
            metadata={
                "classification_source": "low_res_status_llm_code",
                "used_original_code_roi": bool(pipeline_summary),
                "status_text": normalize_whitespace(llm_result.get("status_text", "")),
                "says_python": llm_result.get("says_python"),
            },
        )

    # Extrae relative region.
    def _extract_relative_region(
        self,
        image: np.ndarray,
        rel_box: Tuple[float, float, float, float],
    ) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        h, w = image.shape[:2]
        rx1, ry1, rx2, ry2 = rel_box
        x1 = int(round(w * rx1))
        y1 = int(round(h * ry1))
        x2 = int(round(w * rx2))
        y2 = int(round(h * ry2))

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        return image[y1:y2, x1:x2]

    # Maneja resize to target height dentro de CodeClassifier.
    def _resize_to_target_height(self, gray: np.ndarray, target_h: int) -> np.ndarray:
        if gray is None or gray.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        h, w = gray.shape[:2]
        if h <= 0:
            return np.zeros((0, 0), dtype=np.uint8)
        target_h = max(1, int(target_h))
        target_w = max(1, int(round(w * target_h / float(h))))
        interpolation = cv2.INTER_CUBIC if target_h >= h else cv2.INTER_AREA
        return cv2.resize(gray, (target_w, target_h), interpolation=interpolation)

    # Maneja seconds to frames dentro de CodeClassifier.
    def _seconds_to_frames(self, fps: float, seconds: float) -> int:
        fps = max(1.0, float(fps or 1.0))
        return max(1, int(round(fps * seconds)))

    # Registra input frame.
    def _record_input_frame(self, frame_number: int) -> None:
        if frame_number < 0:
            return
        if self.last_input_frame >= 0:
            gap = max(1, int(frame_number - self.last_input_frame))
            if self.observed_input_gap_frames <= 0:
                self.observed_input_gap_frames = gap
            else:
                blended_gap = (0.7 * float(self.observed_input_gap_frames)) + (0.3 * float(gap))
                self.observed_input_gap_frames = max(1, int(round(blended_gap)))
        self.last_input_frame = int(frame_number)

    # Maneja effective hold frames dentro de CodeClassifier.
    def _effective_hold_frames(self, fps: float) -> int:
        hold_frames = self._seconds_to_frames(fps, CODE_HOLD_SECONDS)
        if self.observed_input_gap_frames > 0:
            hold_frames = max(
                hold_frames,
                int(round(float(self.observed_input_gap_frames) * CODE_HOLD_MIN_SAMPLE_GAPS)),
            )
        return max(1, int(hold_frames))

    # Maneja clear cached code dentro de CodeClassifier.
    def _clear_cached_code(self) -> None:
        self.last_code_frame = -1
        self.last_code_result = None

    # Construye cached code result.
    def _build_cached_code_result(
        self,
        frame_number: int,
        blue_ratio: float,
        fps: float,
    ) -> Optional[ProjectionClassification]:
        if self.last_code_result is None:
            return None

        if blue_ratio < CODE_KEEP_BLUE_RATIO:
            return None

        if frame_number >= 0 and self.last_code_frame >= 0:
            age_frames = max(0, int(frame_number - self.last_code_frame))
            if age_frames > self._effective_hold_frames(fps):
                return None
        else:
            age_frames = 0

        base_confidence = float(self.last_code_result.get("confidence", 0.0))
        hold_frames = self._effective_hold_frames(fps)
        decay = 1.0 - (0.25 * min(1.0, age_frames / float(max(1, hold_frames))))
        confidence = float(max(0.55, min(1.0, base_confidence * decay)))
        return ProjectionClassification(
            frame_number=frame_number,
            state=CODE_STATE,
            confidence=confidence,
            language=self.last_code_result.get("language"),
            code_text=self.last_code_result.get("ocr_text", ""),
            code_ocr_method=None
            if not self.last_code_result.get("ocr_method")
            else f"cached/{self.last_code_result['ocr_method']}",
            blue_ratio=float(blue_ratio),
        )

    # Extrae status bar.
    def _extract_status_bar(self, roi: np.ndarray) -> np.ndarray:
        return self._extract_relative_region(roi, CODE_STATUS_BAR_ROI_REL)

    # Extrae status text region.
    def _extract_status_text_region(self, status_bar: np.ndarray) -> np.ndarray:
        return self._extract_relative_region(status_bar, CODE_STATUS_TEXT_ROI_REL)

    # Calcula blue features.
    def _compute_blue_features(self, status_bar: np.ndarray) -> Tuple[float, np.ndarray]:
        if status_bar is None or status_bar.size == 0 or status_bar.ndim != 3:
            return 0.0, np.zeros((0, 0), dtype=np.uint8)
        hsv = cv2.cvtColor(status_bar, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, CODE_BLUE_HSV_LOWER, CODE_BLUE_HSV_UPPER)
        blue_ratio = float(cv2.countNonZero(blue_mask)) / float(blue_mask.size)
        return blue_ratio, blue_mask

    # Prepara text for OCR.
    def _prepare_text_for_ocr(self, text_region: np.ndarray) -> Dict[str, np.ndarray]:
        empty = {
            "view": np.zeros((0, 0), dtype=np.uint8),
            "ocr_image": np.zeros((0, 0), dtype=np.uint8),
        }
        if text_region is None or text_region.size == 0:
            return empty

        if text_region.ndim == 3:
            gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = text_region.copy()

        gray = self._resize_to_target_height(gray, max(CODE_TEXT_TARGET_HEIGHT, gray.shape[0] * 2))
        if gray.size == 0:
            return empty

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        view = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)
        if view.size == 0:
            return empty

        ocr_image = cv2.copyMakeBorder(
            view,
            CODE_TEXT_BORDER,
            CODE_TEXT_BORDER,
            CODE_TEXT_BORDER,
            CODE_TEXT_BORDER,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        return {
            "view": view,
            "ocr_image": ocr_image,
        }

    # Prepara text for status LLM.
    def _prepare_text_for_status_llm(
        self,
        text_region: np.ndarray,
        status_bar: np.ndarray,
    ) -> np.ndarray:
        source = (
            text_region
            if isinstance(text_region, np.ndarray) and text_region.size > 0
            else status_bar
        )
        if source is None or source.size == 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        if source.ndim == 2:
            image = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        else:
            image = source.copy()

        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        target_h = max(CODE_STATUS_LLM_TARGET_HEIGHT, height)
        target_w = max(1, int(round(width * target_h / float(height))))
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        return cv2.copyMakeBorder(
            resized,
            CODE_TEXT_BORDER,
            CODE_TEXT_BORDER,
            CODE_TEXT_BORDER,
            CODE_TEXT_BORDER,
            cv2.BORDER_REPLICATE,
        )

    # Maneja status LLM prompt dentro de CodeClassifier.
    def _status_llm_prompt(self) -> str:
        return (
            "Analiza solo esta imagen de la barra de estado de un editor. "
            "Puede ser un recorte ampliado del texto de lenguaje, por ejemplo Python.\n"
            "Devuelve solo JSON valido con este esquema exacto:\n"
            "{\n"
            '  "text": "texto visible exacto o la mejor lectura",\n'
            '  "says_python": true,\n'
            '  "language": "PYTHON",\n'
            '  "is_code_editor": true,\n'
            '  "confidence": 0.0\n'
            "}\n"
            "Usa says_python=true si la palabra parece decir Python aunque no sea perfecta. "
            "Si no dice Python, pon says_python=false y devuelve en text lo que si parezca decir. "
            "Si no puedes leer nada, usa text=\"\", language=null, is_code_editor=false y baja confianza."
        )

    # Maneja status LLM api enabled dentro de CodeClassifier.
    def _status_llm_api_enabled(self) -> bool:
        explicit = os.environ.get(CODE_STATUS_LLM_ENABLE_API_ENV_VAR)
        if explicit is not None and str(explicit).strip() != "":
            return str(explicit).strip().lower() not in {"0", "false", "no", "off", "local"}

        provider = str(os.environ.get(CODE_VISION_PROVIDER_ENV_VAR, "") or "").strip().lower()
        enable_api = str(os.environ.get(CODE_VISION_ENABLE_API_ENV_VAR, "") or "").strip().lower()
        return provider != "local" and enable_api not in {"0", "false", "no", "off"}

    # Maneja warn status LLM unavailable once dentro de CodeClassifier.
    def _warn_status_llm_unavailable_once(self, reason: str) -> None:
        if self.status_llm_warning_emitted:
            return
        print(f"⚠️ LLM de estado CODIGO no disponible: {reason}")
        self.status_llm_warning_emitted = True

    # Obtiene status LLM client.
    def _get_status_llm_client(self) -> Optional[OpenAICodeVisionClient]:
        if not self._status_llm_api_enabled():
            self._warn_status_llm_unavailable_once(
                f"habilita {CODE_STATUS_LLM_ENABLE_API_ENV_VAR} o {CODE_VISION_ENABLE_API_ENV_VAR}"
            )
            return None
        if self.status_llm_client is not None:
            return self.status_llm_client

        status_profile = dict(CODE_STATUS_LLM_PROFILE_GEMINI)
        if not profile_has_remote_api_key(status_profile):
            self._warn_status_llm_unavailable_once(
                f"define {GEMINI_API_KEY_ENV_VAR} o {GOOGLE_API_KEY_ENV_VAR}"
            )
            return None

        self.status_llm_client = OpenAICodeVisionClient(profile=status_profile, role="status")
        return self.status_llm_client

    # Maneja status LLM available dentro de CodeClassifier.
    def _status_llm_available(self) -> bool:
        client = self._get_status_llm_client()
        return client is not None and bool(client.api_key)

    # Parsea LLM bool.
    def _parse_llm_bool(self, value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "si", "sí", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
        return None

    # Detecta code with status LLM.
    def _detect_code_with_status_llm(self, image: np.ndarray) -> Dict[str, Any]:
        result = {
            "language": None,
            "status_text": "",
            "ocr_text": "",
            "ocr_method": "llm/status_bar",
            "llm_image": (
                image.copy()
                if isinstance(image, np.ndarray) and image.size > 0
                else np.zeros((0, 0), dtype=np.uint8)
            ),
            "says_python": None,
            "confidence": 0.0,
        }
        client = self._get_status_llm_client()
        if client is None:
            result["error"] = "status_llm_unavailable"
            return result
        if image is None or image.size == 0:
            result["error"] = "empty_status_image"
            return result

        payload = client._run_json_prompt(
            self._status_llm_prompt(),
            image,
            image_format="jpg",
            jpeg_quality=CODE_STATUS_LLM_JPEG_QUALITY,
        )
        if not isinstance(payload, dict) or not payload:
            result["error"] = "empty_status_llm_response"
            return result
        if payload.get("error"):
            result["error"] = payload.get("error")
            result["message"] = payload.get("message", "")
            return result

        status_text = normalize_whitespace(
            payload.get("text")
            or payload.get("status_text")
            or payload.get("visible_text")
            or ""
        )
        says_python = self._parse_llm_bool(payload.get("says_python"))
        language_value = str(payload.get("language") or "").strip().upper()
        language = None
        if (
            says_python is True
            or language_value == "PYTHON"
            or extract_language(status_text) == "PYTHON"
        ):
            language = "PYTHON"

        result.update(
            {
                "language": language,
                "status_text": status_text,
                "ocr_text": status_text,
                "says_python": says_python,
                "is_code_editor": self._parse_llm_bool(payload.get("is_code_editor")),
                "confidence": safe_float(payload.get("confidence"), 0.0),
            }
        )
        if isinstance(payload.get("__usage__"), dict):
            result["__usage__"] = dict(payload["__usage__"])
        return result

    # Debug helpers.
    # Maneja empty debug payload dentro de CodeClassifier.
    def _empty_debug_payload(self) -> Dict[str, Any]:
        return {
            "status_bar": np.zeros((0, 0), dtype=np.uint8),
            "text_region": np.zeros((0, 0), dtype=np.uint8),
            "blue_mask": np.zeros((0, 0), dtype=np.uint8),
            "text_view": np.zeros((0, 0), dtype=np.uint8),
            "ocr_image": np.zeros((0, 0), dtype=np.uint8),
            "layout_overlay": np.zeros((0, 0, 3), dtype=np.uint8),
            "layout_sheet": np.zeros((0, 0, 3), dtype=np.uint8),
            "sidebar_crop": np.zeros((0, 0, 3), dtype=np.uint8),
            "editor_crop": np.zeros((0, 0, 3), dtype=np.uint8),
            "sidebar_edges": np.zeros((0, 0, 3), dtype=np.uint8),
        }

    # Maneja attach debug dentro de CodeClassifier.
    def _attach_debug(
        self,
        result: ProjectionClassification,
        debug_payload: Optional[Dict[str, Any]],
    ) -> ProjectionClassification:
        if self.debug_visualization:
            result.debug = debug_payload
        return result

    # Fusiona debug payload.
    def _merge_debug_payload(
        self,
        debug_payload: Optional[Dict[str, Any]],
        pipeline_summary: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if debug_payload is None or not pipeline_summary:
            return debug_payload
        pipeline_debug = pipeline_summary.get("debug")
        if not isinstance(pipeline_debug, dict):
            return debug_payload
        for key in ("layout_overlay", "layout_sheet", "sidebar_crop", "editor_crop", "sidebar_edges"):
            value = pipeline_debug.get(key)
            if isinstance(value, np.ndarray) and value.size > 0:
                debug_payload[key] = value.copy()
        return debug_payload

    # Maneja log code status LLM result dentro de CodeClassifier.
    def _log_code_status_llm_result(self, frame_number: int, blue_ratio: float, llm_result: Dict[str, Any]) -> None:
        raw_text = normalize_whitespace(llm_result.get("status_text", "") or llm_result.get("ocr_text", ""))
        if len(raw_text) > 80:
            raw_text = raw_text[:77] + "..."
        text_repr = raw_text if raw_text else "<sin texto>"
        error = llm_result.get("error")
        status = f" error={error}" if error else ""
        message = normalize_whitespace(llm_result.get("message", ""))
        if len(message) > 180:
            message = message[:177] + "..."
        if error and message:
            status += f" msg='{message}'"
        print(
            f"🧾 LLM codigo frame={frame_number if frame_number >= 0 else '-'} "
            f"blue={blue_ratio:.2f} "
            f"text='{text_repr}' "
            f"says_python={llm_result.get('says_python')} "
            f"lang={llm_result.get('language')}"
            f"{status}"
        )
