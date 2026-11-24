"""
Visual Assistance Tool classroom video processor.

Background work is intentionally limited to:
- asynchronous session writes
- background OCR/PDF classification

Geometry, tracking and state transitions stay serial to preserve temporal
consistency across frames.
"""

from __future__ import annotations

import cv2
import numpy as np
import argparse
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import load_dotenv

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from src.core.classes_vat import (
        CODE_STATE,
        OTHER_STATE,
        PRESENTATION_STATE,
        PersonDetection,
        ProjectionRegion,
        SessionCache,
        SystemState,
        WhiteboardRegion,
    )
except ImportError:
    from classes_vat import (
        CODE_STATE,
        OTHER_STATE,
        PRESENTATION_STATE,
        PersonDetection,
        ProjectionRegion,
        SessionCache,
        SystemState,
        WhiteboardRegion,
    )

try:
    from workers_vat import AsyncDiskWriter, ProjectionContentWorker
except ImportError:
    from .workers_vat import AsyncDiskWriter, ProjectionContentWorker

try:
    from src.modules.whiteboard_vat import WhiteboardAnalyzer
except ImportError:
    try:
        from whiteboard_vat import WhiteboardAnalyzer
    except ImportError:
        from .whiteboard_vat import WhiteboardAnalyzer

try:
    from src.modules.project_config_vat import *
except ImportError:
    try:
        from project_config_vat import *
    except ImportError:
        from .project_config_vat import *

try:
    from src.modules.vision_backend_vat import VisionBackend
except ImportError:
    try:
        from vision_backend_vat import VisionBackend
    except ImportError:
        from .vision_backend_vat import VisionBackend

try:
    from src.modules.camera_motion_vat import CameraMotionDetector
except ImportError:
    try:
        from camera_motion_vat import CameraMotionDetector
    except ImportError:
        from .camera_motion_vat import CameraMotionDetector

try:
    from src.modules.geometry_vat import *
except ImportError:
    try:
        from geometry_vat import *
    except ImportError:
        from .geometry_vat import *

try:
    from src.modules.whiteboard_geometry_vat import refine_whiteboard_geometry_with_canny
except ImportError:
    try:
        from whiteboard_geometry_vat import refine_whiteboard_geometry_with_canny
    except ImportError:
        from .whiteboard_geometry_vat import refine_whiteboard_geometry_with_canny

try:
    from src.modules.curtain_vat import CurtainTracker, detect_cortina
except ImportError:
    try:
        from curtain_vat import CurtainTracker, detect_cortina
    except ImportError:
        from .curtain_vat import CurtainTracker, detect_cortina

try:
    from src.modules.visualization_vat import annotate_frame, resize_frame_to_fit
except ImportError:
    try:
        from visualization_vat import annotate_frame, resize_frame_to_fit
    except ImportError:
        from .visualization_vat import annotate_frame, resize_frame_to_fit

try:
    from src.modules.bitacora_vat import LiveBitacoraWriter
except ImportError:
    try:
        from bitacora_vat import LiveBitacoraWriter
    except ImportError:
        from .bitacora_vat import LiveBitacoraWriter

# ============================================================================
# VIDEO PROCESSOR
# ============================================================================


class VideoProcessor:
    # Inicializa la instancia y configura su estado interno.
    def __init__(
        self,
        session_id: str,
        models: VisionBackend,
        *,
        pdf_path: Optional[str] = None,
        output_dir: str = "output/sessions",
        debug_projection: bool = True,
    ):
        self.models = models
        self.state = SystemState.CALIBRATING
        self.debug_projection = bool(debug_projection)

        self.writer = AsyncDiskWriter()
        self.cache = SessionCache(session_id, output_dir=output_dir, writer=self.writer)
        self.content_worker = ProjectionContentWorker(
            pdf_path,
            str(self.cache.session_dir),
            debug_visualization=self.debug_projection,
        )
        self.whiteboard_analyzer = WhiteboardAnalyzer(
            str(self.cache.session_dir),
            writer=self.writer,
        )
        self.bitacora = LiveBitacoraWriter(
            self.cache.session_dir,
            output_name=BITACORA_NOTEBOOK_NAME,
            interval_seconds=BITACORA_UPDATE_INTERVAL_SECONDS,
            enabled=BITACORA_LIVE_ENABLED,
        )

        self.whiteboard: Optional[WhiteboardRegion] = None
        self.projection: Optional[ProjectionRegion] = None
        self.whiteboard_candidates: List[Dict] = []
        self.detected_pages: set[int] = set()
        self.latest_projection_debug: Optional[np.ndarray] = None
        self.latest_whiteboard_event: Optional[Dict] = None

        self.frame_count = 0
        self.fps_video = 16.0

        self.camera_detector = CameraMotionDetector(threshold=10.0)
        self.check_camera_motion_interval = 100

        self.calibration_delay = 2.0
        self.t_motion_detected = None
        self.buffer_active = False
        self.waiting_for_people = False

        self._projection_generation = 0
        self._last_classification_frame = -1

    # Procesa un frame y actualiza el estado serial del video.
    def process_frame(self, frame, yolo_persons, cv2_projection):
        self._apply_ready_classifications()

        personas_list = [
            PersonDetection(bbox=person["bbox"], confidence=person["conf"], frame_number=self.frame_count)
            for person in yolo_persons
        ]

        if self.state == SystemState.CALIBRATING:
            if self.buffer_active:
                if (time.perf_counter() - self.t_motion_detected) < self.calibration_delay:
                    self.frame_count += 1
                    return personas_list
                self.buffer_active = False

            self._handle_projection(cv2_projection)

            if personas_list:
                self.waiting_for_people = True
                self.frame_count += 1
                return personas_list
            self.waiting_for_people = False

            if not self.whiteboard_candidates and not self.whiteboard:
                detections = self.models.run_gdino(frame)
                if detections:
                    self.whiteboard_candidates = detections
                    self.calibrate_whiteboard(frame)
                else:
                    self.frame_count += 1
                    return personas_list

            if self.whiteboard_candidates and not self.whiteboard:
                self.calibrate_whiteboard(frame)

            if self.whiteboard:
                self.state = SystemState.MONITORING
                print("📊 Estado: MONITORING")

        elif self.state == SystemState.MONITORING:
            if self.frame_count % self.check_camera_motion_interval == 0:
                has_camera_motion, displacement = self.camera_detector.detect_motion(frame, personas_list)
                if has_camera_motion:
                    print(f"📷 Movimiento de cámara detectado ({displacement:.1f}px). Recalibrando.")
                    self.whiteboard = None
                    self.whiteboard_candidates = []
                    self._reset_projection_state()
                    self.whiteboard_analyzer.reset()
                    self.latest_whiteboard_event = None
                    self.state = SystemState.CALIBRATING
                    self.cache.runtime.clear()
                    self.camera_detector.motion_counter = 0
                    self.buffer_active = True
                    self.t_motion_detected = time.perf_counter()
                    self.waiting_for_people = False
                    self.frame_count += 1
                    return personas_list

            self._handle_projection(cv2_projection)

            if self.frame_count % self._roi_capture_interval_frames() == 0:
                if self.whiteboard is not None:
                    self.check_changes_whiteboard(frame, personas_list)
                if self.projection is not None and self.projection.is_active:
                    self.check_changes_projection(frame, personas_list)

        self._apply_ready_classifications()
        self.frame_count += 1
        return personas_list

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> Dict:
        self._apply_ready_classifications()
        summary = self.content_worker.finalize()
        self._apply_ready_classifications()
        summary["whiteboard_analysis"] = self.whiteboard_analyzer.finalize()
        self.cache.close()
        self.writer.close()
        bitacora_summary = self.bitacora.close()
        if bitacora_summary:
            summary["bitacora_notebook"] = bitacora_summary.get("notebook")
            summary["bitacora_cell_count"] = bitacora_summary.get("cell_count")
        if self.bitacora.last_error:
            summary["bitacora_error"] = self.bitacora.last_error
        if self.debug_projection:
            try:
                cv2.destroyWindow(PROJECTION_DEBUG_WINDOW_NAME)
            except Exception:
                pass
        return summary

    # Devuelve la lista ordenada de paginas detectadas.
    def get_detected_pages(self) -> List[int]:
        return sorted(self.detected_pages)

    # Inicializa o refina la region del pizarron detectada.
    def calibrate_whiteboard(self, frame) -> bool:
        if not self.whiteboard_candidates:
            return False

        detection = self.whiteboard_candidates[0]
        bbox = [detection["x1"], detection["y1"], detection["x2"], detection["y2"]]
        geometry = refine_whiteboard_geometry_with_canny(frame, bbox)
        bbox = geometry["bbox"]
        detection["x1"], detection["y1"], detection["x2"], detection["y2"] = bbox

        self.whiteboard = WhiteboardRegion(
            bbox=bbox,
            confidence=detection["conf"],
            mask=geometry.get("mask"),
            polygon=geometry.get("polygon"),
        )
        roi = self._extract_whiteboard_roi(frame)
        self.whiteboard.last_roi = roi
        projection_mask = self._build_whiteboard_projection_mask(roi.shape)
        self.whiteboard.last_person_mask = self._build_whiteboard_person_mask(roi.shape, [])
        self.cache.update_whiteboard(roi, self.frame_count, self.whiteboard.last_person_mask)
        initial_event = self.whiteboard_analyzer.initialize(
            self.frame_count,
            roi,
            blocked_mask=projection_mask,
        )
        if initial_event is not None:
            self.latest_whiteboard_event = initial_event.to_dict()
            print(
                "📝 Pizarrón: snapshot inicial enviado "
                f"(frame={self.frame_count})"
            )
        print(f"✅ Pizarrón calibrado: bbox={bbox}, conf={detection['conf']:.2f}")
        return True

    # Inicializa o actualiza la region de proyeccion/cortina.
    def calibrate_projection(self, projection_data) -> bool:
        if not projection_data:
            return False

        bbox = projection_data["bbox"]
        mask = projection_data["mask"]
        polygon = projection_data.get("polygon")
        confidence = projection_data["conf"]
        projection_type = projection_data["type"]
        is_stable = projection_data.get("is_stable", False)

        if self.projection is None:
            self._advance_projection_generation()
            self.projection = ProjectionRegion(
                bbox=bbox,
                mask=mask,
                confidence=confidence,
                projection_type=projection_type,
                polygon=polygon,
            )
            self.projection.clear_context()
            print(f"✅ Cortina calibrada: {self.projection.projection_type}")
        else:
            bbox_changed = self.projection.bbox != bbox
            if bbox_changed:
                self._advance_projection_generation()

            self.projection.bbox = bbox
            self.projection.mask = mask
            self.projection.polygon = polygon
            self.projection.confidence = confidence
            self.projection.projection_type = projection_type
            if bbox_changed or not is_stable:
                self.projection.last_roi = None
                self.projection.clear_context()

        self.projection.is_calibrated = is_stable
        return True

    # Actualiza el analisis del pizarron para el frame actual.
    def check_changes_whiteboard(self, frame, personas_list) -> None:
        if self.whiteboard is None:
            return

        roi = self._extract_whiteboard_roi(frame)
        if roi is None or roi.size == 0:
            return

        person_only_mask = self._build_roi_person_mask(
            roi.shape,
            self.whiteboard.bbox,
            personas_list,
            region_mask=self.whiteboard.mask,
        )
        projection_mask = self._build_whiteboard_projection_mask(roi.shape)
        blocked_mask = cv2.bitwise_or(person_only_mask, projection_mask)
        llm_blocked_mask = cv2.bitwise_and(projection_mask, cv2.bitwise_not(person_only_mask))
        self.whiteboard.last_roi = roi
        self.whiteboard.last_person_mask = blocked_mask
        event = self.whiteboard_analyzer.analyze(
            self.frame_count,
            roi,
            person_mask=person_only_mask,
            blocked_mask=llm_blocked_mask,
        )
        if event is not None:
            self.latest_whiteboard_event = event.to_dict()
            print(
                "📝 Pizarrón: "
                f"snapshot {event.event_type} reason={event.reason} "
                f"frame={self.frame_count}"
            )
        self.cache.update_whiteboard(roi, self.frame_count, blocked_mask)

    # Envia la proyeccion estable al worker de clasificacion.
    def check_changes_projection(self, frame, personas_list) -> None:
        if self.projection is None or not self.projection.is_active or not self.projection.is_calibrated:
            return

        roi = self.projection.extract_roi(frame, use_mask=True)
        if roi is None or roi.size == 0:
            return

        person_mask = self._build_roi_person_mask(
            roi.shape,
            self.projection.bbox,
            personas_list,
            region_mask=self.projection.mask,
        )
        self.projection.last_roi = roi
        self.projection.last_person_mask = person_mask
        self.cache.update_projection(roi, self.frame_count, person_mask)
        self.content_worker.submit(
            self._projection_generation,
            self.frame_count,
            self.fps_video,
            roi,
        )

    # Aplica ready classifications.
    def _apply_ready_classifications(self) -> None:
        for result in self.content_worker.poll_results():
            if result.generation != self._projection_generation:
                continue
            if result.frame_number <= self._last_classification_frame:
                continue
            if self.projection is None or not self.projection.is_active:
                continue

            previous_state = self.projection.context_state or OTHER_STATE
            previous_page = self.projection.matched_page_number

            self.projection.apply_classification(result.classification)
            self.cache.record_classification(result.classification)
            self._last_classification_frame = result.frame_number
            debug_payload = result.classification.debug or {}
            if isinstance(debug_payload.get("panel"), np.ndarray) and debug_payload["panel"].size > 0:
                self.latest_projection_debug = debug_payload["panel"].copy()

            if result.classification.state == PRESENTATION_STATE and result.classification.page_number is not None:
                page_number = int(result.classification.page_number)
                is_new_page = page_number not in self.detected_pages
                self.detected_pages.add(page_number)
                if is_new_page:
                    print(f"📚 Slides detectadas hasta ahora: {self.get_detected_pages()}")

            if (
                self.projection.context_state != previous_state
                or self.projection.matched_page_number != previous_page
            ):
                if self.projection.context_state == PRESENTATION_STATE and self.projection.matched_page_number is not None:
                    print(
                        f"🖥️ Proyección clasificada: {PRESENTATION_STATE} "
                        f"diapositiva {self.projection.matched_page_number} "
                        f"(conf: {self.projection.context_confidence:.2f})"
                    )
                elif self.projection.context_state == CODE_STATE:
                    details = []
                    if self.projection.code_language:
                        details.append(str(self.projection.code_language))
                    if self.projection.code_active_file:
                        details.append(str(self.projection.code_active_file))
                    if self.projection.code_known_files:
                        details.append(f"files={self.projection.code_known_files}")
                    if self.projection.code_structure_changed:
                        details.append("sidebar=cambio")
                    suffix = f" {' | '.join(details)}" if details else ""
                    print(
                        f"🖥️ Proyección clasificada: {CODE_STATE}{suffix} "
                        f"(conf: {self.projection.context_confidence:.2f})"
                    )
                else:
                    print(
                        f"🖥️ Proyección clasificada: {self.projection.context_state} "
                        f"(conf: {self.projection.context_confidence:.2f})"
                    )

    # Maneja handle projection dentro de VideoProcessor.
    def _handle_projection(self, cv2_projection) -> None:
        if cv2_projection:
            is_observed = cv2_projection.get("is_observed", True)
            ttl_frames = int(cv2_projection.get("ttl_frames", CORTINA_LOST_TTL_FRAMES))
            was_stable = self.projection.is_calibrated if self.projection is not None else False

            if is_observed and self.projection is None:
                print("🆕 Inicializando objeto cortina por primera vez...")

            if is_observed or self.projection is None:
                self.calibrate_projection(cv2_projection)

            if self.projection is not None:
                if is_observed:
                    self.projection.last_seen_frame = self.frame_count
                if is_observed and not self.projection.is_active:
                    self.projection.is_active = True
                    print("🎬 TRIGGER: Cortina activa en pantalla")
                if self.projection.is_calibrated and not was_stable:
                    print("🎯 Cortina estable. Habilitando monitoreo de contenido.")

                if not is_observed:
                    frames_sin_deteccion = self.frame_count - self.projection.last_seen_frame
                    if frames_sin_deteccion > ttl_frames:
                        self.projection.is_active = False
                        print("⬆️ TRIGGER: Cortina ya no se observa. Se oculta la detección.")
                        self._reset_projection_state()
        else:
            if self.projection and self.projection.is_active:
                frames_sin_deteccion = self.frame_count - self.projection.last_seen_frame
                if frames_sin_deteccion > CORTINA_LOST_TTL_FRAMES:
                    self.projection.is_active = False
                    print("⬆️ TRIGGER: Cortina perdida visualmente. Se asume que subió.")
                    self._reset_projection_state()

    # Maneja advance projection generation dentro de VideoProcessor.
    def _advance_projection_generation(self) -> None:
        self._projection_generation += 1
        self._last_classification_frame = -1

    # Maneja reset projection state dentro de VideoProcessor.
    def _reset_projection_state(self) -> None:
        self._advance_projection_generation()
        self.projection = None
        self.latest_projection_debug = None

    # Maneja roi capture interval frames dentro de VideoProcessor.
    def _roi_capture_interval_frames(self) -> int:
        return max(1, int(round(self.fps_video * ROI_CAPTURE_INTERVAL_SECONDS)))

    # Extrae whiteboard roi.
    def _extract_whiteboard_roi(self, frame) -> Optional[np.ndarray]:
        if self.whiteboard is None:
            return None
        return self.whiteboard.extract_roi(frame, use_mask=True)

    # Maneja crop region mask to roi dentro de VideoProcessor.
    def _crop_region_mask_to_roi(self, region_mask, roi_bbox, roi_shape):
        if region_mask is None or roi_bbox is None:
            return None

        rx1, ry1, rx2, ry2 = roi_bbox
        region_crop = region_mask[ry1:ry2, rx1:rx2]
        if region_crop.size == 0:
            return None

        if region_crop.shape[:2] != roi_shape[:2]:
            region_crop = cv2.resize(region_crop, (roi_shape[1], roi_shape[0]), interpolation=cv2.INTER_NEAREST)
        if len(region_crop.shape) == 3:
            region_crop = region_crop[:, :, 0]

        return np.where(region_crop > 0, 255, 0).astype(np.uint8)

    # Construye roi person mask.
    def _build_roi_person_mask(self, roi_shape, roi_bbox, personas_list, region_mask=None):
        mask = np.zeros(roi_shape[:2], dtype=np.uint8)
        if not personas_list or roi_bbox is None:
            return mask

        local_personas = extract_local_personas(personas_list, roi_bbox)
        for persona in local_personas:
            bbox = persona["bbox"] if isinstance(persona, dict) else persona.bbox
            clipped = clip_bbox_to_frame(
                [
                    bbox[0] - ROI_PERSON_MASK_PADDING_PX,
                    bbox[1] - ROI_PERSON_MASK_PADDING_PX,
                    bbox[2] + ROI_PERSON_MASK_PADDING_PX,
                    bbox[3] + ROI_PERSON_MASK_PADDING_PX,
                ],
                roi_shape,
            )
            if clipped is None:
                continue

            x1, y1, x2, y2 = clipped
            mask[y1:y2, x1:x2] = 255

        if region_mask is None:
            return mask

        region_crop = self._crop_region_mask_to_roi(region_mask, roi_bbox, roi_shape)
        if region_crop is None:
            return mask

        return cv2.bitwise_and(mask, region_crop)

    # Construye whiteboard projection mask.
    def _build_whiteboard_projection_mask(self, roi_shape):
        if self.whiteboard is None:
            return np.zeros(roi_shape[:2], dtype=np.uint8)

        if self.projection is None or not self.projection.is_active:
            return np.zeros(roi_shape[:2], dtype=np.uint8)

        overlap_mask = None
        if self.projection.mask is not None:
            overlap_mask = self._crop_region_mask_to_roi(self.projection.mask, self.whiteboard.bbox, roi_shape)
        elif self.projection.bbox is not None:
            px1, py1, px2, py2 = self.whiteboard.bbox
            cx1, cy1, cx2, cy2 = self.projection.bbox
            local_overlap_bbox = clip_bbox_to_frame(
                [
                    max(px1, cx1) - px1,
                    max(py1, cy1) - py1,
                    min(px2, cx2) - px1,
                    min(py2, cy2) - py1,
                ],
                roi_shape,
            )
            if local_overlap_bbox is not None:
                overlap_mask = np.zeros(roi_shape[:2], dtype=np.uint8)
                ox1, oy1, ox2, oy2 = local_overlap_bbox
                overlap_mask[oy1:oy2, ox1:ox2] = 255

        if overlap_mask is None:
            return np.zeros(roi_shape[:2], dtype=np.uint8)

        whiteboard_region_mask = self._crop_region_mask_to_roi(self.whiteboard.mask, self.whiteboard.bbox, roi_shape)
        if whiteboard_region_mask is not None:
            overlap_mask = cv2.bitwise_and(overlap_mask, whiteboard_region_mask)

        return overlap_mask

    # Construye whiteboard person mask.
    def _build_whiteboard_person_mask(self, roi_shape, personas_list):
        if self.whiteboard is None:
            return np.zeros(roi_shape[:2], dtype=np.uint8)

        exclusion_mask = self._build_roi_person_mask(
            roi_shape,
            self.whiteboard.bbox,
            personas_list,
            region_mask=self.whiteboard.mask,
        )
        projection_mask = self._build_whiteboard_projection_mask(roi_shape)
        return cv2.bitwise_or(exclusion_mask, projection_mask)

# ============================================================================
# MAIN
# ============================================================================


# Ejecuta el flujo principal desde los argumentos de entrada.
def main(video_path: str, pdf_path: Optional[str] = None, debug_projection: bool = True) -> None:
    session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    models = VisionBackend()
    processor = VideoProcessor(session_id, models, pdf_path=pdf_path, debug_projection=debug_projection)

    last_persons = []
    last_projection = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    processor.fps_video = fps_video if fps_video > 0 else 16.0
    processor.whiteboard_analyzer.set_fps(processor.fps_video)
    processor.bitacora.set_fps(processor.fps_video)

    paused = False
    t_prev = time.perf_counter()
    cortina_tracker = CurtainTracker()

    print(f"🎬 Procesando video: {video_path}")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                if processor.frame_count % YOLO_INTERVAL == 0:
                    last_persons = models.run_yolo(frame)
                    projection_for_filter = last_projection if last_projection is not None else processor.projection
                    last_persons = filter_projected_person_detections(last_persons, projection_for_filter)

                cortina_interval = CORTINA_CHECK_INTERVAL
                if (processor.projection and processor.projection.is_active) or cortina_tracker.active_projection is not None:
                    cortina_interval = CORTINA_CHECK_INTERVAL_ACTIVE

                if processor.frame_count % cortina_interval == 0:
                    whiteboard_bbox = None
                    if processor.whiteboard is not None:
                        whiteboard_bbox = processor.whiteboard.bbox
                    elif processor.whiteboard_candidates:
                        detection = processor.whiteboard_candidates[0]
                        whiteboard_bbox = [detection["x1"], detection["y1"], detection["x2"], detection["y2"]]

                    last_projection = detect_cortina(
                        frame,
                        last_persons,
                        cortina_tracker,
                        whiteboard_bbox=whiteboard_bbox,
                        frame_index=processor.frame_count,
                        fps=processor.fps_video,
                    )
                    projection_for_filter = last_projection if last_projection is not None else processor.projection
                    last_persons = filter_projected_person_detections(last_persons, projection_for_filter)

                if processor.frame_count == 0 and not processor.whiteboard_candidates:
                    detections = models.run_gdino(frame)
                    if detections:
                        processor.whiteboard_candidates = detections

                personas_list = processor.process_frame(frame, last_persons, last_projection)
                processor.bitacora.maybe_update()
                annotated = annotate_frame(
                    frame,
                    processor.whiteboard,
                    processor.projection,
                    personas_list,
                    display_target=(DISPLAY_TARGET_W, DISPLAY_TARGET_H),
                )

                t_now = time.perf_counter()
                fps = 1.0 / max(t_now - t_prev, 1e-6)
                t_prev = t_now

                cv2.putText(
                    annotated,
                    f"FPS {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    annotated,
                    f"Estado: {processor.state.value}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                overlay_y = 90
                if processor.projection and processor.projection.is_active:
                    projection_text = f"Proyeccion: {processor.projection.context_state or OTHER_STATE}"
                    if processor.projection.matched_page_number is not None:
                        projection_text += f" P{processor.projection.matched_page_number}"
                    cv2.putText(
                        annotated,
                        projection_text,
                        (10, overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )
                    overlay_y += 30

                if pdf_path is not None:
                    detected_pages = processor.get_detected_pages()
                    pages_text = "-" if not detected_pages else ",".join(str(page) for page in detected_pages[:SLIDES_OVERLAY_MAX])
                    if len(detected_pages) > SLIDES_OVERLAY_MAX:
                        pages_text += "..."
                    cv2.putText(
                        annotated,
                        f"Slides detectadas: {pages_text}",
                        (10, overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

                cv2.imshow("Procesador Paralelo", annotated)
                if processor.debug_projection and processor.latest_projection_debug is not None:
                    cv2.imshow(PROJECTION_DEBUG_WINDOW_NAME, processor.latest_projection_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()
        summary = processor.close()
        detected_pages = sorted(set(processor.get_detected_pages()) | set(summary.get("detected_pages", [])))
        if detected_pages:
            print(f"📚 Diapositivas proyectadas detectadas: {detected_pages}")
            if summary.get("matchings_json"):
                print(f"📋 Historial de matches: {summary['matchings_json']}")
            if summary.get("annotated_pdf"):
                print(f"📄 PDF anotado: {summary['annotated_pdf']}")
        else:
            print("📚 No se detectaron diapositivas del PDF durante la sesión")
        if summary.get("code_repository_json"):
            print(f"🗂️ Repositorio de código: {summary['code_repository_json']}")
        if summary.get("code_repository_root"):
            print(f"🧩 Archivos reconstruidos: {summary['code_repository_root']}")
        if summary.get("code_active_file"):
            print(f"📌 Último archivo activo: {summary['code_active_file']}")
        if summary.get("bitacora_notebook"):
            print(f"🧾 Bitácora accesible: {summary['bitacora_notebook']}")
        if summary.get("bitacora_error"):
            print(f"⚠️ Bitácora accesible falló: {summary['bitacora_error']}")
        print("✅ Sesión guardada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--pdf", default=None)
    parser.add_argument("--no-debug-projection", action="store_true")
    args = parser.parse_args()
    main(args.video, pdf_path=args.pdf, debug_projection=not args.no_debug_projection)
