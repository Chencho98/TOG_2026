"""
Projection classifier backed by an optional PDF reference plus code heuristics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re

import cv2
import numpy as np

try:
    from src.core.classes_vat import (
        CODE_STATE,
        OTHER_STATE,
        PRESENTATION_STATE,
        ProjectionClassification,
    )
except ImportError:
    from classes_vat import (
        CODE_STATE,
        OTHER_STATE,
        PRESENTATION_STATE,
        ProjectionClassification,
    )

try:
    from src.modules.code_vat import (
        CODE_BLUE_HSV_LOWER,
        CODE_BLUE_HSV_UPPER,
        CODE_STATUS_BAR_ROI_REL,
        CODE_STATUS_TEXT_ROI_REL,
        CodeClassifier,
        configure_tesseract_command,
        downscale_projection_roi,
    )
except ImportError:
    from code_vat import (
        CODE_BLUE_HSV_LOWER,
        CODE_BLUE_HSV_UPPER,
        CODE_STATUS_BAR_ROI_REL,
        CODE_STATUS_TEXT_ROI_REL,
        CodeClassifier,
        configure_tesseract_command,
        downscale_projection_roi,
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

# Umbral minimo para aceptar que el ROI coincide con una diapositiva.
PRESENTATION_MATCH_THRESHOLD = 0.35
# Mejora minima de confianza para registrar otra coincidencia de slide.
REGISTER_CONFIDENCE_DELTA = 0.03
# Resolucion usada al renderizar paginas del PDF para comparar.
RENDER_DPI = 150
# Ancho maximo al normalizar imagenes antes del matching.
MATCH_WIDTH_CAP = 320
# Ancho maximo del logo/template usado en comparaciones locales.
LOGO_MATCH_MAX_WIDTH = 96
# Confianza minima para aceptar numeros de diapositiva detectados por OCR.
OCR_MIN_CONFIDENCE = 0.20
# Alto objetivo para escalar el ROI del numero antes de OCR.
OCR_NUMBER_TARGET_HEIGHT = 128
# Variable de entorno que permite fijar el binario de Tesseract.
OCR_TESSERACT_ENV_VAR = "TESSERACT_CMD"
# Borde agregado al ROI del numero para mejorar el OCR.
OCR_NUMBER_BORDER = 10
# Modos PSM probados por Tesseract al leer numeros de slide.
OCR_NUMBER_PSMS = (7, 8, 13)
# Ancho del panel de debug visual del clasificador de PDF.
PDF_DEBUG_PANEL_W = 240
# Alto del panel de debug visual del clasificador de PDF.
PDF_DEBUG_PANEL_H = 140

# Region relativa donde se busca el logo dentro de la proyeccion.
PROJECTION_LOGO_ROI_REL = (0.85, 0.05, 1.00, 0.20)
# Region relativa equivalente del logo dentro de paginas renderizadas del PDF.
PDF_LOGO_ROI_REL = (0.85, 0.00, 1.00, 0.20)
# Region relativa donde se espera encontrar el numero de diapositiva.
SLIDE_NUMBER_ROI_REL = (0.90, 0.85, 0.95, 0.92)


try:
    import fitz  # PyMuPDF

    # Indica que PyMuPDF esta disponible para renderizar el PDF de referencia.
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    # Indica que PyMuPDF no esta disponible y se omite el matching con PDF.
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError

    configure_tesseract_command(pytesseract)
    # Indica que pytesseract esta disponible para leer numeros de slide.
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    TesseractNotFoundError = RuntimeError
    # Indica que pytesseract no esta disponible y se desactiva el OCR local.
    PYTESSERACT_AVAILABLE = False


# ============================================================================
# PROJECTION CLASSIFIER
# ============================================================================

class ProjectionClassifier:
    # Inicializa la instancia y configura su estado interno.
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        match_threshold: float = PRESENTATION_MATCH_THRESHOLD,
        debug_visualization: bool = False,
    ):
        self.pdf_path = Path(pdf_path) if pdf_path else None
        self.match_threshold = float(match_threshold)
        self.debug_visualization = bool(debug_visualization)

        self.code_classifier = CodeClassifier(debug_visualization=debug_visualization)

        self.doc = None
        self.num_pages = 0
        self.max_page_digits = 1
        self.pages_bgr: List[np.ndarray] = []
        self.page_logo_regions_gray: List[np.ndarray] = []
        self.logo_template: Optional[np.ndarray] = None
        self.session_dir: Optional[Path] = None

        self.current_page = -1
        self.match_confidence = 0.0
        self.detected_pages: set[int] = set()
        self.matched_frames: List[Dict] = []

        self.ocr_enabled = bool(PYTESSERACT_AVAILABLE)
        self.ocr_warning_emitted = False

        if self.pdf_path is not None:
            self._try_open_pdf()

    # Clasifica el ROI de proyeccion como presentacion, codigo u otro.
    def classify_projection(
        self,
        roi: np.ndarray,
        frame_number: int,
        fps: float,
    ) -> ProjectionClassification:
        processing_roi = downscale_projection_roi(roi)
        match = self.find_matching_page(processing_roi)
        presentation_detected = bool(match.get("presentation_detected", False))
        confidence = float(match.get("presentation_confidence", 0.0))
        page_index = int(match.get("page_index", -1))
        code_match = self.code_classifier.classify_projection(
            processing_roi,
            frame_number=frame_number,
            fps=fps,
            original_roi=roi,
            allow_status_llm=not presentation_detected,
        )
        if self.debug_visualization and match.get("debug") is not None:
            code_debug = code_match.debug or {}
            match["debug"]["code_status_bar"] = code_debug.get("status_bar")
            match["debug"]["code_text_region"] = code_debug.get("text_region")
            match["debug"]["code_blue_mask"] = code_debug.get("blue_mask")
            match["debug"]["code_ocr_image"] = code_debug.get("ocr_image")
            match["debug"]["code_layout_overlay"] = code_debug.get("layout_overlay")
            match["debug"]["code_layout_sheet"] = code_debug.get("layout_sheet")
            match["debug"]["code_sidebar_crop"] = code_debug.get("sidebar_crop")
            match["debug"]["code_editor_crop"] = code_debug.get("editor_crop")
            match["debug"]["code_sidebar_edges"] = code_debug.get("sidebar_edges")

        if presentation_detected:
            should_register = (
                page_index >= 0
                and (
                    page_index != self.current_page
                    or confidence > (self.match_confidence + REGISTER_CONFIDENCE_DELTA)
                )
            )

            self.current_page = page_index if page_index >= 0 else -1
            self.match_confidence = confidence

            if should_register:
                self.register_match(frame_number, page_index, confidence, fps=fps, roi=processing_roi)

            classification = ProjectionClassification(
                frame_number=frame_number,
                state=PRESENTATION_STATE,
                confidence=confidence,
                page_index=page_index,
                page_number=None if page_index < 0 else page_index + 1,
                logo_score=float(match.get("logo_score", 0.0)),
                ocr_confidence=float(match.get("ocr_confidence", 0.0)),
                language=code_match.language,
                code_text=code_match.code_text,
                code_ocr_method=code_match.code_ocr_method,
                blue_ratio=float(code_match.blue_ratio),
                code_active_file=code_match.code_active_file,
                code_known_files=int(code_match.code_known_files),
                code_structure_changed=bool(code_match.code_structure_changed),
            )
            classification.debug = self._build_classification_debug_payload(match, classification)
            return classification

        self.current_page = -1
        self.match_confidence = 0.0

        if code_match.state == CODE_STATE:
            classification = ProjectionClassification(
                frame_number=frame_number,
                state=CODE_STATE,
                confidence=float(code_match.confidence),
                logo_score=float(match.get("logo_score", 0.0)),
                ocr_confidence=float(match.get("ocr_confidence", 0.0)),
                language=code_match.language,
                code_text=code_match.code_text,
                code_ocr_method=code_match.code_ocr_method,
                blue_ratio=float(code_match.blue_ratio),
                code_active_file=code_match.code_active_file,
                code_known_files=int(code_match.code_known_files),
                code_structure_changed=bool(code_match.code_structure_changed),
            )
            classification.debug = self._build_classification_debug_payload(match, classification)
            return classification

        classification = ProjectionClassification(
            frame_number=frame_number,
            state=OTHER_STATE,
            confidence=max(confidence, float(code_match.confidence)),
            logo_score=float(match.get("logo_score", 0.0)),
            ocr_confidence=float(match.get("ocr_confidence", 0.0)),
            language=code_match.language,
            code_text=code_match.code_text,
            code_ocr_method=code_match.code_ocr_method,
            blue_ratio=float(code_match.blue_ratio),
            code_active_file=code_match.code_active_file,
            code_known_files=int(code_match.code_known_files),
            code_structure_changed=bool(code_match.code_structure_changed),
        )
        classification.debug = self._build_classification_debug_payload(match, classification)
        return classification

    # Busca la pagina del PDF que mejor coincide con la proyeccion.
    def find_matching_page(self, roi: np.ndarray) -> Dict:
        if roi is None or roi.size == 0:
            return {
                "page_index": -1,
                "presentation_confidence": 0.0,
                "logo_score": 0.0,
                "ocr_page_number": None,
                "ocr_confidence": 0.0,
                "ocr_method": None,
                "presentation_detected": False,
                "debug": None,
            }

        roi_gray = np.zeros((0, 0), dtype=np.uint8)
        code_status_bar = np.zeros((0, 0), dtype=np.uint8)
        code_text_region = np.zeros((0, 0), dtype=np.uint8)
        code_blue_mask = np.zeros((0, 0), dtype=np.uint8)
        if self.debug_visualization:
            roi_gray_full = self._preprocess_gray(roi)
            target_w = min(roi_gray_full.shape[1], MATCH_WIDTH_CAP) if roi_gray_full.size > 0 else MATCH_WIDTH_CAP
            roi_gray = self._resize_keep_aspect(roi_gray_full, target_w)
            code_status_bar = self._extract_relative_region(roi, CODE_STATUS_BAR_ROI_REL)
            code_text_region = self._extract_relative_region(code_status_bar, CODE_STATUS_TEXT_ROI_REL)
            code_blue_mask = self._compute_projection_code_blue_mask(code_status_bar)

        roi_logo_corner = self._preprocess_logo_gray(self._extract_projection_logo_corner(roi))
        logo_match = self._match_logo_in_corner(roi_logo_corner)
        logo_score = float(logo_match["score"])
        presentation_detected = bool(logo_score >= self.match_threshold)

        roi_number = self._extract_slide_number_region(roi)
        roi_number_view = roi_number.copy()
        page_index = -1
        ocr_result = {
            "page_number": None,
            "confidence": 0.0,
            "text": "",
            "method": None,
            "ocr_image": np.zeros((0, 0), dtype=np.uint8),
            "debug_number_image": np.zeros((0, 0), dtype=np.uint8),
        }
        if presentation_detected and self.logo_template is not None:
            ocr_result = self._detect_page_number_with_ocr(roi_number)
            if isinstance(ocr_result.get("debug_number_image"), np.ndarray) and ocr_result["debug_number_image"].size > 0:
                roi_number_view = ocr_result["debug_number_image"].copy()
            if (
                ocr_result["page_number"] is not None
                and float(ocr_result["confidence"]) >= OCR_MIN_CONFIDENCE
            ):
                page_index = int(ocr_result["page_number"]) - 1

        match = {
            "page_index": page_index,
            "presentation_confidence": float(logo_score),
            "logo_score": float(logo_score),
            "ocr_page_number": ocr_result["page_number"],
            "ocr_confidence": float(ocr_result["confidence"]),
            "ocr_method": ocr_result["method"],
            "presentation_detected": presentation_detected,
        }
        if self.debug_visualization:
            match["debug"] = {
                "roi_gray": roi_gray.copy(),
                "roi_logo_corner": roi_logo_corner.copy(),
                "logo_template_used": None if logo_match["template_used"] is None else logo_match["template_used"].copy(),
                "logo_match_patch": None if logo_match["match_patch"] is None else logo_match["match_patch"].copy(),
                "logo_absdiff": None if logo_match["absdiff"] is None else logo_match["absdiff"].copy(),
                "logo_match_map": None if logo_match["match_map"] is None else logo_match["match_map"].copy(),
                "logo_match_location": logo_match["match_location"],
                "roi_number_view": roi_number_view.copy(),
                "ocr_number_image": ocr_result["ocr_image"].copy() if isinstance(ocr_result.get("ocr_image"), np.ndarray) else None,
                "code_status_bar": code_status_bar.copy(),
                "code_text_region": code_text_region.copy(),
                "code_blue_mask": code_blue_mask.copy(),
                "code_ocr_image": code_text_region.copy(),
            }
        return match

    # Registra una coincidencia de diapositiva para persistencia y resumen.
    def register_match(
        self,
        frame_number: int,
        page_index: int,
        confidence: float,
        fps: float,
        roi: Optional[np.ndarray] = None,
    ) -> None:
        if page_index < 0:
            return

        entry = {
            "frame": int(frame_number),
            "page_index": int(page_index),
            "page_number": int(page_index + 1),
            "confidence": round(float(confidence), 4),
            "timestamp_s": round(float(frame_number) / max(1.0, float(fps or 1.0)), 2),
        }
        self.matched_frames.append(entry)
        self.detected_pages.add(entry["page_number"])

        print(
            f"📑 Diapositiva detectada: {entry['page_number']}/{self.num_pages} "
            f"(conf: {confidence:.2f}, frame: {frame_number})"
        )

        if roi is not None and self.session_dir is not None:
            path = self.session_dir / "roi_captures" / f"frame_{frame_number:06d}_slide_{page_index + 1:03d}.jpg"
            cv2.imwrite(str(path), roi, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.flush_json()

    # Configura los directorios de salida de la sesion.
    def set_session_dir(self, session_dir: str) -> None:
        self.session_dir = Path(session_dir) / "pdf"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "logo").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "roi_captures").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "slide_thumbnails").mkdir(parents=True, exist_ok=True)
        self.code_classifier.set_session_dir(str(Path(session_dir) / "code_intelligence"))

        if self.logo_template is not None:
            self._save_logo()
        if self.pages_bgr:
            self._save_thumbnails()

    # Devuelve la lista ordenada de paginas detectadas.
    def get_detected_pages(self) -> List[int]:
        return sorted(self.detected_pages)

    # Escribe el historial de coincidencias de diapositivas en JSON.
    def flush_json(self) -> Optional[Path]:
        if self.session_dir is None or not self.matched_frames:
            return None

        best_per_page: Dict[int, Dict] = {}
        for entry in self.matched_frames:
            page_index = entry["page_index"]
            if page_index not in best_per_page or entry["confidence"] > best_per_page[page_index]["confidence"]:
                best_per_page[page_index] = entry

        data = {
            "pdf_file": None if self.pdf_path is None else self.pdf_path.name,
            "total_pages": self.num_pages,
            "total_matches": len(self.matched_frames),
            "pages_detected": self.get_detected_pages(),
            "best_match_per_page": list(best_per_page.values()),
            "full_history": self.matched_frames,
        }

        path = self.session_dir / "matchings.json"
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, ensure_ascii=False)
        return path

    # Genera una copia del PDF con paginas detectadas anotadas.
    def generate_annotated_pdf(self, output_path: str) -> Optional[Path]:
        if self.doc is None or not self.matched_frames:
            return None

        out_doc = fitz.open()
        best_per_page: Dict[int, Dict] = {}
        for entry in self.matched_frames:
            page_index = entry["page_index"]
            if page_index not in best_per_page or entry["confidence"] > best_per_page[page_index]["confidence"]:
                best_per_page[page_index] = entry

        for page_index in range(self.num_pages):
            out_doc.insert_pdf(self.doc, from_page=page_index, to_page=page_index)
            out_page = out_doc[-1]

            if page_index in best_per_page:
                entry = best_per_page[page_index]
                out_page.draw_rect(out_page.rect, color=(0.0, 0.7, 0.2), width=4)
                rect = fitz.Rect(10, 10, 240, 38)
                out_page.draw_rect(rect, color=(0.0, 0.7, 0.2), fill=(0.0, 0.7, 0.2))
                out_page.insert_text(
                    fitz.Point(15, 29),
                    f"Proyectada  frame {entry['frame']}  conf {entry['confidence']:.2f}",
                    fontsize=10,
                    color=(1, 1, 1),
                )
            else:
                rect = fitz.Rect(10, 10, 120, 28)
                out_page.draw_rect(rect, color=(0.6, 0.6, 0.6), fill=(0.6, 0.6, 0.6))
                out_page.insert_text(
                    fitz.Point(14, 22),
                    "No proyectada",
                    fontsize=8,
                    color=(1, 1, 1),
                )

        output = Path(output_path)
        out_doc.save(str(output))
        out_doc.close()
        return output

    # Finaliza el procesamiento pendiente y devuelve el resumen producido.
    def finalize(self) -> Dict:
        json_path = self.flush_json()
        annotated_pdf_path = None
        detected_pages = self.get_detected_pages()
        if detected_pages and self.session_dir is not None and self.doc is not None:
            annotated_pdf_path = self.generate_annotated_pdf(str(self.session_dir / "slides_detected.pdf"))
        summary = {
            "detected_pages": detected_pages,
            "matchings_json": None if json_path is None else str(json_path),
            "annotated_pdf": None if annotated_pdf_path is None else str(annotated_pdf_path),
        }
        summary.update(self.code_classifier.finalize())
        return summary

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> None:
        try:
            self.code_classifier.close()
        except Exception:
            pass
        if self.doc is not None:
            self.doc.close()
            self.doc = None

    # Calcula projection code blue mask.
    def _compute_projection_code_blue_mask(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0 or image.ndim != 3:
            return np.zeros((0, 0), dtype=np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, CODE_BLUE_HSV_LOWER, CODE_BLUE_HSV_UPPER)

    # Maneja try open pdf dentro de ProjectionClassifier.
    def _try_open_pdf(self) -> None:
        if not PYMUPDF_AVAILABLE:
            print("⚠️ PyMuPDF no disponible. Se usará solo el detector de código.")
            self.pdf_path = None
            return

        try:
            self.doc = fitz.open(str(self.pdf_path))
        except Exception as exc:
            print(f"⚠️ No se pudo abrir el PDF '{self.pdf_path}': {exc}")
            self.doc = None
            self.pdf_path = None
            return

        self.num_pages = len(self.doc)
        self.max_page_digits = max(1, len(str(self.num_pages)))
        self._render_all_pages()
        self._extract_logo_from_pdf()
        print(f"📄 PDF cargado: {self.pdf_path.name} — {self.num_pages} páginas")

    # Maneja preprocess gray dentro de ProjectionClassifier.
    def _preprocess_gray(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    # Maneja preprocess logo gray dentro de ProjectionClassifier.
    def _preprocess_logo_gray(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.size == 0:
            return gray
        target_w = min(gray.shape[1], LOGO_MATCH_MAX_WIDTH)
        return self._resize_keep_aspect(gray, target_w)

    # Maneja resize keep aspect dentro de ProjectionClassifier.
    def _resize_keep_aspect(self, gray: np.ndarray, target_w: int) -> np.ndarray:
        if gray.size == 0:
            return gray
        h, w = gray.shape[:2]
        if w <= 0:
            return gray
        target_w = max(1, int(target_w))
        target_h = max(1, int(round(h * target_w / float(w))))
        return cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Maneja resize to target height dentro de ProjectionClassifier.
    def _resize_to_target_height(self, gray: np.ndarray, target_h: int) -> np.ndarray:
        if gray.size == 0:
            return gray
        h, w = gray.shape[:2]
        if h <= 0:
            return gray
        target_h = max(1, int(target_h))
        target_w = max(1, int(round(w * target_h / float(h))))
        interpolation = cv2.INTER_CUBIC if target_h >= h else cv2.INTER_AREA
        return cv2.resize(gray, (target_w, target_h), interpolation=interpolation)

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

    # Extrae projection logo corner.
    def _extract_projection_logo_corner(self, image: np.ndarray) -> np.ndarray:
        return self._extract_relative_region(image, PROJECTION_LOGO_ROI_REL)

    # Extrae pdf logo corner.
    def _extract_pdf_logo_corner(self, image: np.ndarray) -> np.ndarray:
        return self._extract_relative_region(image, PDF_LOGO_ROI_REL)

    # Extrae slide number region.
    def _extract_slide_number_region(self, image: np.ndarray) -> np.ndarray:
        return self._extract_relative_region(image, SLIDE_NUMBER_ROI_REL)

    # Prepara number for OCR.
    def _prepare_number_for_ocr(self, image: np.ndarray) -> Dict:
        empty = {
            "view": np.zeros((0, 0), dtype=np.uint8),
            "ocr_image": np.zeros((0, 0), dtype=np.uint8),
            "ocr_image_inv": np.zeros((0, 0), dtype=np.uint8),
        }
        if image is None or image.size == 0:
            return empty

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.size == 0:
            return empty

        resized = self._resize_to_target_height(gray, max(OCR_NUMBER_TARGET_HEIGHT, gray.shape[0] * 2))
        if resized.size == 0:
            return empty

        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        view = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)
        _, binary = cv2.threshold(view, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ocr_image = cv2.copyMakeBorder(
            binary,
            OCR_NUMBER_BORDER,
            OCR_NUMBER_BORDER,
            OCR_NUMBER_BORDER,
            OCR_NUMBER_BORDER,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        ocr_image_inv = 255 - ocr_image
        return {
            "view": view,
            "ocr_image": ocr_image,
            "ocr_image_inv": ocr_image_inv,
        }

    # Maneja warn OCR unavailable once dentro de ProjectionClassifier.
    def _warn_ocr_unavailable_once(self, message: str) -> None:
        if self.ocr_warning_emitted:
            return
        print(message)
        self.ocr_warning_emitted = True

    # Extrae OCR page candidates.
    def _extract_ocr_page_candidates(self, raw_text: str) -> List[int]:
        if not raw_text:
            return []

        translated = str(raw_text).translate(
            str.maketrans(
                {
                    "O": "0",
                    "o": "0",
                    "Q": "0",
                    "D": "0",
                    "I": "1",
                    "l": "1",
                    "|": "1",
                    "!": "1",
                    "S": "5",
                    "s": "5",
                    "B": "8",
                    "Z": "2",
                    "z": "2",
                }
            )
        )

        candidates: List[int] = []
        seen = set()
        for digits_text in re.findall(r"\d+", translated):
            if not digits_text:
                continue

            options = [digits_text]
            if len(digits_text) > self.max_page_digits:
                options = [digits_text[-self.max_page_digits :]]

            for option in options:
                if not option:
                    continue
                page_number = int(option)
                if page_number < 1 or page_number > self.num_pages:
                    continue
                if page_number in seen:
                    continue
                seen.add(page_number)
                candidates.append(page_number)

        return candidates

    # Ejecuta tesseract on number image.
    def _run_tesseract_on_number_image(self, image: np.ndarray, psm: int) -> Optional[Dict]:
        if not self.ocr_enabled or not PYTESSERACT_AVAILABLE or image.size == 0:
            return None

        config = (
            f"--oem 3 --psm {psm} "
            "-c tessedit_char_whitelist=0123456789 "
            "-c classify_bln_numeric_mode=1 "
            "-c load_system_dawg=0 "
            "-c load_freq_dawg=0"
        )
        raw_text = pytesseract.image_to_string(image, config=config)
        page_candidates = self._extract_ocr_page_candidates(raw_text)
        if not page_candidates:
            return None

        page_number = int(page_candidates[0])
        return {
            "page_number": page_number,
            "confidence": 1.0,
            "text": str(raw_text).strip(),
            "psm": psm,
        }

    # Detecta page number with OCR.
    def _detect_page_number_with_ocr(self, roi_number: np.ndarray) -> Dict:
        prepared = self._prepare_number_for_ocr(roi_number)
        display_image = prepared.get("view")
        ocr_image = prepared.get("ocr_image")
        ocr_image_inv = prepared.get("ocr_image_inv")
        result = {
            "page_number": None,
            "confidence": 0.0,
            "text": "",
            "method": None,
            "ocr_image": ocr_image.copy() if isinstance(ocr_image, np.ndarray) and ocr_image.size > 0 else np.zeros((0, 0), dtype=np.uint8),
            "debug_number_image": display_image.copy() if isinstance(display_image, np.ndarray) and display_image.size > 0 else np.zeros((0, 0), dtype=np.uint8),
        }

        if not isinstance(ocr_image, np.ndarray) or ocr_image.size == 0:
            return result

        if not self.ocr_enabled or not PYTESSERACT_AVAILABLE:
            self._warn_ocr_unavailable_once(
                f"⚠️ OCR deshabilitado: instala pytesseract y configura {OCR_TESSERACT_ENV_VAR}"
            )
            return result

        attempts = [
            ("gray", display_image),
            ("otsu", ocr_image),
            ("otsu_inv", ocr_image_inv),
        ]
        for attempt_name, attempt_image in attempts:
            if not isinstance(attempt_image, np.ndarray) or attempt_image.size == 0:
                continue

            for psm in OCR_NUMBER_PSMS:
                try:
                    ocr_result = self._run_tesseract_on_number_image(attempt_image, psm)
                except TesseractNotFoundError:
                    self.ocr_enabled = False
                    self._warn_ocr_unavailable_once(
                        f"⚠️ No se encontró el ejecutable de tesseract. Ajusta {OCR_TESSERACT_ENV_VAR} o el PATH."
                    )
                    return result
                except Exception:
                    continue

                if ocr_result is None:
                    continue

                return {
                    "page_number": ocr_result["page_number"],
                    "confidence": ocr_result["confidence"],
                    "text": ocr_result["text"],
                    "method": f"{attempt_name}/psm{psm}",
                    "ocr_image": attempt_image.copy(),
                    "debug_number_image": display_image.copy() if isinstance(display_image, np.ndarray) and display_image.size > 0 else np.zeros((0, 0), dtype=np.uint8),
                }

        return result

    # Maneja render all pages dentro de ProjectionClassifier.
    def _render_all_pages(self) -> None:
        if self.doc is None:
            return
        mat_render = fitz.Matrix(RENDER_DPI / 72.0, RENDER_DPI / 72.0)
        for page in self.doc:
            pix = page.get_pixmap(matrix=mat_render, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            page_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.pages_bgr.append(page_bgr)

            page_logo_crop = self._extract_pdf_logo_corner(page_bgr)
            self.page_logo_regions_gray.append(self._preprocess_logo_gray(page_logo_crop))

    # Extrae logo from pdf.
    def _extract_logo_from_pdf(self) -> None:
        if self.num_pages < 2:
            return

        logo_crops = []
        for crop in self.page_logo_regions_gray[1 : min(6, self.num_pages)]:
            if crop.size == 0 or float(np.mean(crop)) < 100.0:
                continue
            logo_crops.append(crop.astype(np.float32))

        if not logo_crops:
            return

        ref_h, ref_w = logo_crops[0].shape[:2]
        aligned = [logo_crops[0]]
        for crop in logo_crops[1:]:
            aligned.append(cv2.resize(crop, (ref_w, ref_h), interpolation=cv2.INTER_AREA))

        self.logo_template = np.mean(aligned, axis=0).astype(np.uint8)

    # Maneja match logo in corner dentro de ProjectionClassifier.
    def _match_logo_in_corner(self, corner_gray: np.ndarray) -> Dict:
        if self.logo_template is None or corner_gray.size == 0:
            return {
                "score": 0.0,
                "template_used": None,
                "match_patch": None,
                "absdiff": None,
                "match_map": None,
                "match_location": None,
            }

        template = self.logo_template
        th, tw = template.shape[:2]
        ch, cw = corner_gray.shape[:2]
        if ch < 2 or cw < 2:
            return {
                "score": 0.0,
                "template_used": None,
                "match_patch": None,
                "absdiff": None,
                "match_map": None,
                "match_location": None,
            }

        if th >= ch or tw >= cw:
            scale = min((ch - 1) / float(th), (cw - 1) / float(tw)) * 0.9
            if scale <= 0:
                return {
                    "score": 0.0,
                    "template_used": None,
                    "match_patch": None,
                    "absdiff": None,
                    "match_map": None,
                    "match_location": None,
                }
            target_w = max(1, int(round(tw * scale)))
            target_h = max(1, int(round(th * scale)))
            template = cv2.resize(template, (target_w, target_h), interpolation=cv2.INTER_AREA)

        result = cv2.matchTemplate(corner_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        score = float(max(0.0, min(1.0, max_val)))
        if not self.debug_visualization:
            return {
                "score": score,
                "template_used": None,
                "match_patch": None,
                "absdiff": None,
                "match_map": None,
                "match_location": None,
            }

        used_template = template.copy()
        uh, uw = used_template.shape[:2]
        x, y = int(max_loc[0]), int(max_loc[1])
        match_patch = corner_gray[y : y + uh, x : x + uw].copy()
        if match_patch.shape[:2] != used_template.shape[:2]:
            match_patch = cv2.resize(match_patch, (uw, uh), interpolation=cv2.INTER_AREA)
        absdiff = cv2.absdiff(match_patch, used_template) if match_patch.size > 0 else None
        return {
            "score": score,
            "template_used": used_template,
            "match_patch": match_patch,
            "absdiff": absdiff,
            "match_map": self._normalize_debug_float_image(result),
            "match_location": (x, y),
        }

    # Guarda logo.
    def _save_logo(self) -> None:
        if self.session_dir is None or self.logo_template is None:
            return
        path = self.session_dir / "logo" / "logo_escuela.jpg"
        cv2.imwrite(str(path), self.logo_template, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Guarda thumbnails.
    def _save_thumbnails(self) -> None:
        if self.session_dir is None:
            return
        thumb_dir = self.session_dir / "slide_thumbnails"
        for index, page_bgr in enumerate(self.pages_bgr):
            h, w = page_bgr.shape[:2]
            thumb_w = 320
            thumb_h = max(1, int(round(h * thumb_w / float(max(1, w)))))
            thumb = cv2.resize(page_bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            path = thumb_dir / f"slide_{index + 1:03d}.jpg"
            cv2.imwrite(str(path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Debug visualization helpers.
    # Maneja as bgr dentro de ProjectionClassifier.
    def _as_bgr(self, image: Optional[np.ndarray]) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((PDF_DEBUG_PANEL_H, PDF_DEBUG_PANEL_W, 3), dtype=np.uint8)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()

    # Construye debug panel.
    def _build_debug_panel(
        self,
        image: Optional[np.ndarray],
        title: str,
        score_text: Optional[str] = None,
    ) -> np.ndarray:
        panel = np.zeros((PDF_DEBUG_PANEL_H, PDF_DEBUG_PANEL_W, 3), dtype=np.uint8)
        content = self._as_bgr(image)
        h, w = content.shape[:2]
        header_h = 28
        max_w = PDF_DEBUG_PANEL_W - 8
        max_h = PDF_DEBUG_PANEL_H - header_h - 8
        scale = min(max_w / float(max(1, w)), max_h / float(max(1, h)))
        target_w = max(1, int(round(w * scale)))
        target_h = max(1, int(round(h * scale)))
        resized = cv2.resize(
            content,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        x = (PDF_DEBUG_PANEL_W - target_w) // 2
        y = header_h + ((max_h - target_h) // 2)
        panel[y : y + target_h, x : x + target_w] = resized
        cv2.putText(panel, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if score_text:
            cv2.putText(
                panel,
                score_text,
                (6, PDF_DEBUG_PANEL_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.rectangle(panel, (0, 0), (PDF_DEBUG_PANEL_W - 1, PDF_DEBUG_PANEL_H - 1), (70, 70, 70), 1)
        return panel

    # Construye debug text panel.
    def _build_debug_text_panel(self, summary: Dict, debug: Optional[Dict] = None) -> np.ndarray:
        panel = np.zeros((PDF_DEBUG_PANEL_H, PDF_DEBUG_PANEL_W, 3), dtype=np.uint8)
        state = str(summary.get("state", OTHER_STATE))
        presentation_confidence = float(summary.get("presentation_confidence", 0.0))
        logo_score = float(summary.get("logo_score", 0.0))
        logo_match_location = None if debug is None else debug.get("logo_match_location")
        ocr_page_number = summary.get("ocr_page_number")
        ocr_confidence = float(summary.get("ocr_confidence", 0.0))
        ocr_method = summary.get("ocr_method")
        language = summary.get("language")
        blue_ratio = float(summary.get("blue_ratio", 0.0))
        code_text = str(summary.get("code_text", "") or "")
        code_ocr_method = summary.get("code_ocr_method")
        code_active_file = summary.get("code_active_file")
        code_known_files = int(summary.get("code_known_files", 0))
        code_structure_changed = bool(summary.get("code_structure_changed", False))

        lines = [
            "PDF matching",
            f"estado: {state}",
            f"pres: {presentation_confidence:.2f}",
            f"logo: {logo_score:.2f}",
            f"ocr: {ocr_page_number if ocr_page_number is not None else '-'} {ocr_confidence:.2f}",
        ]
        if logo_match_location is not None:
            lines.append(f"logo xy: {logo_match_location[0]}, {logo_match_location[1]}")
        if ocr_method:
            lines.append(f"ocr cfg: {ocr_method}")
        if blue_ratio > 0.0 or language or code_text:
            lines.append(f"blue: {blue_ratio:.2f}")
            if language:
                lines.append(f"lang: {language}")
            if code_ocr_method:
                lines.append(f"code cfg: {code_ocr_method}")
            if code_active_file:
                lines.append(f"file: {str(code_active_file)[:22]}")
            if code_known_files:
                lines.append(f"files: {code_known_files}")
            if code_structure_changed:
                lines.append("sidebar: cambio")
            if code_text:
                normalized_text = " ".join(code_text.split())
                lines.append(f"txt: {normalized_text[:26]}")

        y = 18
        for index, line in enumerate(lines):
            color = (255, 255, 255) if index == 0 else (180, 255, 180)
            cv2.putText(panel, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y += 15
        cv2.rectangle(panel, (0, 0), (PDF_DEBUG_PANEL_W - 1, PDF_DEBUG_PANEL_H - 1), (70, 70, 70), 1)
        return panel

    # Construye classification debug payload.
    def _build_classification_debug_payload(
        self,
        match: Dict,
        classification: ProjectionClassification,
    ) -> Optional[Dict]:
        if not self.debug_visualization:
            return None
        debug = match.get("debug")
        if not debug:
            return None

        summary = {
            "state": classification.state,
            "presentation_detected": bool(match.get("presentation_detected", False)),
            "presentation_confidence": float(match.get("presentation_confidence", 0.0)),
            "logo_score": float(match.get("logo_score", 0.0)),
            "ocr_page_number": match.get("ocr_page_number"),
            "ocr_confidence": float(match.get("ocr_confidence", 0.0)),
            "ocr_method": match.get("ocr_method"),
            "language": classification.language,
            "blue_ratio": float(classification.blue_ratio),
            "code_text": classification.code_text,
            "code_ocr_method": classification.code_ocr_method,
            "code_active_file": classification.code_active_file,
            "code_known_files": int(classification.code_known_files),
            "code_structure_changed": bool(classification.code_structure_changed),
        }
        panel = self._compose_debug_view(summary, debug)
        return {"panel": panel}

    # Maneja compose debug view dentro de ProjectionClassifier.
    def _compose_debug_view(self, summary: Dict, debug: Dict) -> np.ndarray:
        panels = [
            self._build_debug_panel(debug.get("roi_gray"), "ROI"),
            self._build_debug_text_panel(summary, debug),
            self._build_debug_panel(debug.get("logo_match_map"), "Logo heatmap"),
            self._build_debug_panel(debug.get("roi_logo_corner"), "ROI logo"),
            self._build_debug_panel(debug.get("logo_template_used"), "Template logo", f"logo {summary.get('logo_score', 0.0):.2f}"),
            self._build_debug_panel(debug.get("logo_match_patch"), "Logo patch"),
            self._build_debug_panel(debug.get("logo_absdiff"), "Logo diff"),
            self._build_debug_panel(debug.get("roi_number_view"), "ROI numero"),
            self._build_debug_panel(debug.get("ocr_number_image"), "OCR numero"),
            self._build_debug_panel(debug.get("code_status_bar"), "Code barra", f"blue {summary.get('blue_ratio', 0.0):.2f}"),
            self._build_debug_panel(debug.get("code_blue_mask"), "Code azul"),
            self._build_debug_panel(debug.get("code_ocr_image"), "Code OCR"),
            self._build_debug_panel(debug.get("code_layout_overlay"), "Code layout"),
            self._build_debug_panel(debug.get("code_sidebar_crop"), "Code sidebar"),
            self._build_debug_panel(debug.get("code_editor_crop"), "Code editor"),
        ]
        row1 = np.hstack(panels[0:3])
        row2 = np.hstack(panels[3:6])
        row3 = np.hstack(panels[6:9])
        row4 = np.hstack(panels[9:12])
        row5 = np.hstack(panels[12:15])
        return np.vstack([row1, row2, row3, row4, row5])

    # Normaliza debug float image.
    def _normalize_debug_float_image(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ProjectionClassifier",
    "CODE_BLUE_HSV_LOWER",
    "CODE_BLUE_HSV_UPPER",
    "CODE_STATUS_BAR_ROI_REL",
    "CODE_STATUS_TEXT_ROI_REL",
]
