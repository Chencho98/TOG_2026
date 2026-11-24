"""
Whiteboard snapshot analysis for the classroom pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from queue import Full, Queue
from typing import Any, Dict, List, Optional, Sequence, Tuple
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

# Variable de entorno que habilita o deshabilita el analisis LLM del pizarron.
WHITEBOARD_LLM_ENABLE_API_ENV_VAR = "WHITEBOARD_LLM_ENABLE_API"
# Variable de entorno desde donde se lee la API key del proveedor LLM.
WHITEBOARD_LLM_API_KEY_ENV_VAR = "OPENAI_API_KEY"
# Selecciona openai o gemini para experimentos reproducibles por sesión.
WHITEBOARD_LLM_PROVIDER_ENV_VAR = "WHITEBOARD_LLM_PROVIDER"
# Variable de entorno que permite cambiar el modelo usado para el pizarron.
WHITEBOARD_LLM_MODEL_ENV_VAR = "WHITEBOARD_LLM_MODEL"
# Variable de entorno que permite cambiar la URL base del proveedor LLM.
WHITEBOARD_LLM_BASE_URL_ENV_VAR = "OPENAI_BASE_URL"
# Variable de entorno que controla el timeout de requests del pizarron.
WHITEBOARD_LLM_TIMEOUT_ENV_VAR = "WHITEBOARD_LLM_TIMEOUT_SEC"
# Modelo por defecto para describir contenido visible del pizarron.
WHITEBOARD_LLM_DEFAULT_MODEL = "gemini-2.5-flash-lite"
# URL base por defecto para requests compatibles con OpenAI.
WHITEBOARD_LLM_DEFAULT_BASE_URL = "https://api.openai.com/v1"
# Timeout por defecto de cada request LLM del pizarron.
WHITEBOARD_LLM_DEFAULT_TIMEOUT_SEC = 45.0
# Limite maximo de tokens de salida para el analisis del pizarron.
WHITEBOARD_LLM_MAX_TOKENS = 1200
# Calidad JPEG usada al enviar snapshots del pizarron al LLM.
WHITEBOARD_LLM_JPEG_QUALITY = 92
# Tamano maximo de la cola asincrona de snapshots del pizarron.
WHITEBOARD_SNAPSHOT_QUEUE_SIZE = 4
# Valor gris neutro usado para cubrir regiones bloqueadas por cortina.
WHITEBOARD_BLOCKED_NEUTRAL_GRAY = 180
# Proporcion minima de area de persona para considerarla presente en el pizarron.
WHITEBOARD_PERSON_PRESENT_RATIO = 0.010
# Proporcion maxima de area de persona para considerar que ya salio del pizarron.
WHITEBOARD_PERSON_EXIT_RATIO = 0.004
# Caida de area que sugiere que la persona dejo de extender el brazo al escribir.
WHITEBOARD_PERSON_AREA_SHRINK_RATIO = 0.62
# Cambio minimo de centro de persona para considerar que se movio de region.
WHITEBOARD_PERSON_MOVE_RATIO = 0.10
# Distancia maxima para tratar una posicion como la misma zona del pizarron.
WHITEBOARD_PERSON_REGION_STABLE_RATIO = 0.045
# Tiempo que la persona debe quedarse en una zona antes de disparar snapshot.
WHITEBOARD_PERSON_REGION_DWELL_SECONDS = 2.0
# Separacion minima entre envios consecutivos para evitar requests redundantes.
WHITEBOARD_MIN_TRIGGER_GAP_SECONDS = 2.0


# ============================================================================
# SNAPSHOT DATA
# ============================================================================

@dataclass
class WhiteboardSnapshotEvent:
    frame_number: int
    event_type: str
    reason: str
    bbox: List[int]
    area_pixels: int
    person_area_ratio: float
    blocked_ratio: float
    previous_content: str
    snapshot_path: Optional[str] = None
    response_path: Optional[str] = None
    status: str = "queued"
    model_content: str = ""
    model_added: List[str] = field(default_factory=list)
    model_erased: List[str] = field(default_factory=list)
    model_changed: List[str] = field(default_factory=list)
    model_categories: List[str] = field(default_factory=list)
    model_visual_description: str = ""
    model_code_detected: Optional[bool] = None
    model_code_description: str = ""
    still_writing: Optional[bool] = None
    timestamp_s: float = field(default_factory=time.time)

    # Convierte la instancia a un diccionario serializable.
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": int(self.frame_number),
            "event_type": self.event_type,
            "reason": self.reason,
            "bbox": [int(value) for value in self.bbox],
            "area_pixels": int(self.area_pixels),
            "person_area_ratio": float(self.person_area_ratio),
            "blocked_ratio": float(self.blocked_ratio),
            "previous_content": self.previous_content,
            "snapshot_path": self.snapshot_path,
            "response_path": self.response_path,
            "status": self.status,
            "model_content": self.model_content,
            "model_added": list(self.model_added),
            "model_erased": list(self.model_erased),
            "model_changed": list(self.model_changed),
            "model_categories": list(self.model_categories),
            "model_visual_description": self.model_visual_description,
            "model_code_detected": self.model_code_detected,
            "model_code_description": self.model_code_description,
            "still_writing": self.still_writing,
            "timestamp_s": float(self.timestamp_s),
        }


@dataclass
class WhiteboardSnapshotTask:
    task_id: int
    frame_number: int
    event_type: str
    reason: str
    image: np.ndarray
    person_bbox: List[int]
    person_area_ratio: float
    blocked_ratio: float
    previous_content: str
    created_ts: float = field(default_factory=time.time)


# ============================================================================
# LLM CLIENT
# ============================================================================

class WhiteboardLLMClient:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self):
        self.enabled = self._api_enabled()
        self.provider = str(os.environ.get(WHITEBOARD_LLM_PROVIDER_ENV_VAR, "gemini") or "gemini").strip().lower()
        if self.provider == "gemini":
            self.api_key = str(
                os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
            ).strip()
        else:
            self.api_key = str(os.environ.get(WHITEBOARD_LLM_API_KEY_ENV_VAR, "") or "").strip()
        self.model = str(
            os.environ.get(WHITEBOARD_LLM_MODEL_ENV_VAR)
            or os.environ.get("CODE_VISION_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or WHITEBOARD_LLM_DEFAULT_MODEL
        ).strip()
        default_base_url = (
            "https://generativelanguage.googleapis.com/v1beta/openai"
            if self.provider == "gemini"
            else WHITEBOARD_LLM_DEFAULT_BASE_URL
        )
        base_url = str(
            os.environ.get(WHITEBOARD_LLM_BASE_URL_ENV_VAR)
            or default_base_url
        ).strip().rstrip("/")
        self.endpoint = (
            base_url
            if base_url.endswith("/chat/completions")
            else base_url + "/chat/completions"
        )
        self.timeout = self._safe_float(
            os.environ.get(WHITEBOARD_LLM_TIMEOUT_ENV_VAR),
            WHITEBOARD_LLM_DEFAULT_TIMEOUT_SEC,
        )

    # Analiza una tarea o ROI y devuelve el resultado correspondiente.
    def analyze(self, task: WhiteboardSnapshotTask) -> Dict[str, Any]:
        if not self.enabled:
            return {"error": "api_disabled", "message": "WHITEBOARD_LLM_ENABLE_API disabled"}
        if not self.api_key:
            key_hint = "GEMINI_API_KEY or GOOGLE_API_KEY" if self.provider == "gemini" else WHITEBOARD_LLM_API_KEY_ENV_VAR
            return {"error": "missing_api_key", "message": key_hint}
        image_url = self._image_to_data_url(task.image)
        if not image_url:
            return {"error": "empty_image", "message": "whiteboard snapshot empty"}

        payload = {
            "model": self.model,
            "max_tokens": WHITEBOARD_LLM_MAX_TOKENS,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You analyze classroom whiteboard screenshots. "
                        "Return only valid JSON and do not invent unreadable content."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": self._build_prompt(task)},
                    ],
                },
            ],
        }
        request_data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=request_data, method="POST")
        request.add_header("Authorization", f"Bearer {self.api_key}")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                error_body = ""
            return {"error": f"http_{exc.code}", "message": error_body}
        except Exception as exc:
            return {"error": "request_failed", "message": str(exc)}

        try:
            response_json = json.loads(raw)
        except Exception:
            return {"error": "invalid_response_json", "message": raw}
        text = self._extract_output_text(response_json)
        parsed = self._parse_jsonish_dict(text)
        if not parsed:
            return {"error": "invalid_output_json", "message": text}
        usage = response_json.get("usage") if isinstance(response_json, dict) else None
        if isinstance(usage, dict):
            parsed["__usage__"] = dict(usage)
        return parsed

    # Construye prompt.
    def _build_prompt(self, task: WhiteboardSnapshotTask) -> str:
        previous = task.previous_content.strip() or "<sin contenido previo conocido>"
        return (
            "Analiza el contenido visible del pizarrón.\n"
            f"Evento: {task.event_type}\n"
            f"Razón del envío: {task.reason}\n"
            f"Contenido previo conocido:\n{previous}\n\n"
            "Devuelve JSON con este esquema exacto:\n"
            "{\n"
            '  "content": "descripción completa y actual del pizarrón",\n'
            '  "added": ["elementos agregados desde el contenido previo"],\n'
            '  "erased": ["elementos borrados desde el contenido previo"],\n'
            '  "changed": ["elementos modificados"],\n'
            '  "content_categories": ["texto"],\n'
            '  "visual_description": "descripción de dibujos, diagramas o tablas si existen",\n'
            '  "code_detected": false,\n'
            '  "code_description": "lenguaje o estructura si parece código; vacío si no hay",\n'
            '  "still_writing": true,\n'
            '  "confidence": 0.0\n'
            "}\n"
            "En content_categories usa solo las categorías que apliquen: texto, "
            "codigo, dibujo, diagrama, tabla, formula u otro. Si hay dibujo, "
            "diagrama o tabla, descríbelo en visual_description aunque el texto "
            "sea mínimo. Si parece código o pseudocódigo, usa codigo en "
            "content_categories, code_detected=true y describe el lenguaje, "
            "estructura o fragmentos legibles en code_description.\n"
            "Las zonas grises/neutras uniformes fueron bloqueadas porque ahi "
            "hay cortina/proyeccion sobre el pizarron; ignoralas y no las "
            "interpretes como contenido escrito. La persona visible no fue "
            "tapada para que puedas decidir si sigue escribiendo.\n"
            "Si una persona bloquea parte del pizarrón o parece seguir escribiendo, "
            "usa still_writing=true. Si no puedes leer algo, dilo explícitamente."
        )

    # Maneja image to data url dentro de WhiteboardLLMClient.
    def _image_to_data_url(self, image: np.ndarray) -> Optional[str]:
        if image is None or image.size == 0:
            return None
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, WHITEBOARD_LLM_JPEG_QUALITY],
        )
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    # Extrae output text.
    def _extract_output_text(self, payload: Dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        return ""

    # Parsea JSONish dict.
    def _parse_jsonish_dict(self, text: str) -> Dict[str, Any]:
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    # Maneja api enabled dentro de WhiteboardLLMClient.
    def _api_enabled(self) -> bool:
        value = str(os.environ.get(WHITEBOARD_LLM_ENABLE_API_ENV_VAR, "1") or "1").strip().lower()
        return value not in {"0", "false", "no", "off", "local"}

    # Maneja safe float dentro de WhiteboardLLMClient.
    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)


# ============================================================================
# SNAPSHOT ANALYZER
# ============================================================================


class WhiteboardAnalyzer:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self, session_dir: str, writer: Optional[Any] = None):
        self.session_dir = Path(session_dir)
        self.writer = writer
        self.analysis_dir = self.session_dir / "whiteboard_analysis"
        self.snapshots_dir = self.analysis_dir / "snapshots"
        self.responses_dir = self.analysis_dir / "responses"
        self.client = WhiteboardLLMClient()
        self._queue: Queue = Queue(maxsize=WHITEBOARD_SNAPSHOT_QUEUE_SIZE)
        self._sentinel = object()
        self._thread = threading.Thread(
            target=self._run_worker,
            name="whiteboard-llm-worker",
            daemon=True,
        )
        self._thread.start()
        self._lock = threading.Lock()
        self._events: List[WhiteboardSnapshotEvent] = []
        self._latest_content = ""
        self._initialized = False
        self._next_task_id = 1
        self._last_submit_ts = 0.0
        self._last_frame = -1
        self._person_present = False
        self._person_peak_area_ratio = 0.0
        self._person_last_centroid: Optional[Tuple[float, float]] = None
        self._last_writing_check_frame = -1
        self._awaiting_completion_snapshot = False
        self._completion_anchor_centroid: Optional[Tuple[float, float]] = None
        self._completion_region_centroid: Optional[Tuple[float, float]] = None
        self._completion_region_start_frame = -1
        self._fps = 16.0
        self._closed = False

    # Actualiza el FPS usado para convertir umbrales temporales a frames.
    def set_fps(self, fps: float) -> None:
        try:
            parsed = float(fps)
        except Exception:
            parsed = 16.0
        self._fps = parsed if parsed > 0.0 else 16.0

    # Reinicia el estado interno para comenzar una nueva medicion.
    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._latest_content = ""
            self._initialized = False
            self._last_frame = -1
            self._person_present = False
            self._person_peak_area_ratio = 0.0
            self._person_last_centroid = None
            self._last_writing_check_frame = -1
            self._awaiting_completion_snapshot = False
            self._completion_anchor_centroid = None
            self._completion_region_centroid = None
            self._completion_region_start_frame = -1
        self._write_summary()

    # Envia el primer snapshot del pizarron y marca el analizador como inicializado.
    def initialize(
        self,
        frame_number: int,
        roi: np.ndarray,
        blocked_mask: Optional[np.ndarray] = None,
    ) -> Optional[WhiteboardSnapshotEvent]:
        if roi is None or roi.size == 0:
            return None
        with self._lock:
            if self._initialized:
                return None
            self._initialized = True
            self._last_frame = int(frame_number)
        return self._submit_snapshot(
            frame_number=frame_number,
            roi=roi,
            event_type="initial",
            reason="initial_whiteboard_roi",
            person_bbox=[0, 0, roi.shape[1], roi.shape[0]],
            person_area_ratio=0.0,
            blocked_ratio=self._mask_ratio(blocked_mask, roi.shape[:2]),
            blocked_mask=blocked_mask,
            force=True,
        )

    # Analiza una tarea o ROI y devuelve el resultado correspondiente.
    def analyze(
        self,
        frame_number: int,
        roi: np.ndarray,
        person_mask: Optional[np.ndarray] = None,
        blocked_mask: Optional[np.ndarray] = None,
    ) -> Optional[WhiteboardSnapshotEvent]:
        if roi is None or roi.size == 0:
            return None
        if not self._initialized:
            return self.initialize(frame_number, roi, blocked_mask)

        stats = self._person_stats(person_mask, roi.shape[:2])
        person_present = stats["area_ratio"] >= WHITEBOARD_PERSON_PRESENT_RATIO
        person_entered = (not self._person_present) and person_present
        person_exited = self._person_present and stats["area_ratio"] <= WHITEBOARD_PERSON_EXIT_RATIO
        if person_entered:
            self._person_peak_area_ratio = stats["area_ratio"]
            self._person_last_centroid = stats["centroid"]
        elif person_present:
            self._person_peak_area_ratio = max(self._person_peak_area_ratio, stats["area_ratio"])

        area_shrank = (
            person_present
            and self._person_peak_area_ratio > 0.0
            and stats["area_ratio"] <= self._person_peak_area_ratio * WHITEBOARD_PERSON_AREA_SHRINK_RATIO
            and int(frame_number) != self._last_writing_check_frame
        )
        moved_after_writing = (
            person_present
            and self._awaiting_completion_snapshot
            and self._centroid_moved_enough(
                self._completion_anchor_centroid,
                stats["centroid"],
                roi.shape,
            )
        )

        event = None
        if person_exited:
            event = self._submit_snapshot(
                frame_number,
                roi,
                "after_person_exit",
                "person_left_whiteboard_area",
                stats["bbox"],
                stats["area_ratio"],
                self._mask_ratio(blocked_mask, roi.shape[:2]),
                blocked_mask=blocked_mask,
                force=True,
            )
            self._awaiting_completion_snapshot = False
            self._completion_anchor_centroid = None
            self._completion_region_centroid = None
            self._completion_region_start_frame = -1
            self._person_peak_area_ratio = 0.0
        elif area_shrank:
            event = self._submit_snapshot(
                frame_number,
                roi,
                "writing_check",
                "person_roi_shrank_after_extension",
                stats["bbox"],
                stats["area_ratio"],
                self._mask_ratio(blocked_mask, roi.shape[:2]),
                blocked_mask=blocked_mask,
            )
            self._last_writing_check_frame = int(frame_number)
            self._completion_anchor_centroid = stats["centroid"]
            self._completion_region_centroid = None
            self._completion_region_start_frame = -1
        elif moved_after_writing:
            if not self._centroid_within_same_region(
                self._completion_region_centroid,
                stats["centroid"],
                roi.shape,
            ):
                self._completion_region_centroid = stats["centroid"]
                self._completion_region_start_frame = int(frame_number)
            elif self._completion_region_dwelled(int(frame_number)):
                event = self._submit_snapshot(
                    frame_number,
                    roi,
                    "completion_check",
                    "person_stayed_after_model_said_still_writing",
                    stats["bbox"],
                    stats["area_ratio"],
                    self._mask_ratio(blocked_mask, roi.shape[:2]),
                    blocked_mask=blocked_mask,
                )
                if event is not None:
                    self._awaiting_completion_snapshot = False
                    self._completion_anchor_centroid = None
                    self._completion_region_centroid = None
                    self._completion_region_start_frame = -1
        else:
            self._completion_region_centroid = None
            self._completion_region_start_frame = -1

        self._person_present = person_present
        if stats["centroid"] is not None:
            self._person_last_centroid = stats["centroid"]
        self._last_frame = int(frame_number)
        self._write_summary()
        return event

    # Finaliza el procesamiento pendiente y devuelve el resumen producido.
    def finalize(self) -> Dict[str, Any]:
        if not self._closed:
            self._queue.put(self._sentinel)
            self._queue.join()
            self._thread.join(timeout=10.0)
            self._closed = True
        self._write_summary()
        return self.summary()

    # Construye el resumen actual del estado procesado.
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            counts: Dict[str, int] = {}
            for event in self._events:
                counts[event.event_type] = counts.get(event.event_type, 0) + 1
            return {
                "initialized": bool(self._initialized),
                "event_count": len(self._events),
                "event_counts": counts,
                "last_frame": int(self._last_frame),
                "latest_content": self._latest_content,
                "events": [event.to_dict() for event in self._events],
                "summary_path": str(self.analysis_dir / "summary.json"),
            }

    # Maneja submit snapshot dentro de WhiteboardAnalyzer.
    def _submit_snapshot(
        self,
        frame_number: int,
        roi: np.ndarray,
        event_type: str,
        reason: str,
        person_bbox: List[int],
        person_area_ratio: float,
        blocked_ratio: float,
        blocked_mask: Optional[np.ndarray] = None,
        force: bool = False,
    ) -> Optional[WhiteboardSnapshotEvent]:
        now = time.time()
        if not force and now - self._last_submit_ts < WHITEBOARD_MIN_TRIGGER_GAP_SECONDS:
            return None
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            previous_content = self._latest_content
        snapshot_image = self._neutralize_blocked_region(roi, blocked_mask)
        snapshot_path = self._write_snapshot(frame_number, event_type, snapshot_image)
        task = WhiteboardSnapshotTask(
            task_id=task_id,
            frame_number=int(frame_number),
            event_type=event_type,
            reason=reason,
            image=snapshot_image.copy(),
            person_bbox=[int(value) for value in person_bbox],
            person_area_ratio=float(person_area_ratio),
            blocked_ratio=float(blocked_ratio),
            previous_content=previous_content,
        )
        event = WhiteboardSnapshotEvent(
            frame_number=int(frame_number),
            event_type=event_type,
            reason=reason,
            bbox=task.person_bbox,
            area_pixels=int(round(person_area_ratio * roi.shape[0] * roi.shape[1])),
            person_area_ratio=float(person_area_ratio),
            blocked_ratio=float(blocked_ratio),
            previous_content=previous_content,
            snapshot_path=snapshot_path,
        )
        try:
            self._queue.put_nowait(task)
            self._last_submit_ts = now
        except Full:
            event.status = "dropped_queue_full"
        with self._lock:
            self._events.append(event)
        self._write_summary()
        return event

    # Ejecuta worker.
    def _run_worker(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is self._sentinel:
                    return
                if not isinstance(task, WhiteboardSnapshotTask):
                    continue
                started = time.time()
                payload = self.client.analyze(task)
                self._apply_response(task, payload, time.time() - started)
            finally:
                self._queue.task_done()

    # Aplica response.
    def _apply_response(
        self,
        task: WhiteboardSnapshotTask,
        payload: Dict[str, Any],
        elapsed_sec: float,
    ) -> None:
        response_path = self._write_response(task, payload, elapsed_sec)
        content = str(payload.get("content", "") or "").strip()
        added = self._string_list(payload.get("added"))
        erased = self._string_list(payload.get("erased"))
        changed = self._string_list(payload.get("changed"))
        categories = self._string_list(payload.get("content_categories") or payload.get("categories"))
        visual_description = str(payload.get("visual_description", "") or "").strip()
        code_detected = self._parse_bool(payload.get("code_detected"))
        if code_detected is None and any(self._is_code_category(item) for item in categories):
            code_detected = True
        code_description = str(payload.get("code_description", "") or "").strip()
        still_writing = self._parse_bool(payload.get("still_writing"))
        with self._lock:
            if content:
                self._latest_content = content
            for event in reversed(self._events):
                if (
                    event.frame_number == task.frame_number
                    and event.event_type == task.event_type
                    and event.reason == task.reason
                ):
                    event.status = "error" if payload.get("error") else "ok"
                    event.response_path = response_path
                    event.model_content = content
                    event.model_added = added
                    event.model_erased = erased
                    event.model_changed = changed
                    event.model_categories = categories
                    event.model_visual_description = visual_description
                    event.model_code_detected = code_detected
                    event.model_code_description = code_description
                    event.still_writing = still_writing
                    break
            if still_writing is True:
                self._awaiting_completion_snapshot = True
                self._completion_anchor_centroid = self._person_last_centroid
                self._completion_region_centroid = None
                self._completion_region_start_frame = -1
            elif still_writing is False and task.event_type in {"writing_check", "completion_check"}:
                self._awaiting_completion_snapshot = False
                self._completion_anchor_centroid = None
                self._completion_region_centroid = None
                self._completion_region_start_frame = -1
        self._write_summary()

    # Maneja person stats dentro de WhiteboardAnalyzer.
    def _person_stats(self, person_mask: Optional[np.ndarray], roi_shape: Tuple[int, int]) -> Dict[str, Any]:
        if person_mask is None or person_mask.size == 0:
            return {"area_ratio": 0.0, "bbox": [0, 0, 0, 0], "centroid": None}
        mask = person_mask
        if mask.shape[:2] != roi_shape[:2]:
            mask = cv2.resize(mask, (roi_shape[1], roi_shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        area = int(cv2.countNonZero(binary))
        points = cv2.findNonZero(binary)
        if points is None:
            return {"area_ratio": 0.0, "bbox": [0, 0, 0, 0], "centroid": None}
        x, y, w, h = cv2.boundingRect(points)
        moments = cv2.moments(binary)
        centroid = None
        if moments["m00"] > 0:
            centroid = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
        return {
            "area_ratio": float(area) / float(max(1, binary.size)),
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "centroid": centroid,
        }

    # Maneja centroid moved enough dentro de WhiteboardAnalyzer.
    def _centroid_moved_enough(
        self,
        previous: Optional[Tuple[float, float]],
        current: Optional[Tuple[float, float]],
        roi_shape: Tuple[int, ...],
    ) -> bool:
        if previous is None or current is None:
            return False
        height, width = roi_shape[:2]
        distance = float(np.hypot(current[0] - previous[0], current[1] - previous[1]))
        return distance >= max(width, height) * WHITEBOARD_PERSON_MOVE_RATIO

    # Maneja centroid within same region dentro de WhiteboardAnalyzer.
    def _centroid_within_same_region(
        self,
        previous: Optional[Tuple[float, float]],
        current: Optional[Tuple[float, float]],
        roi_shape: Tuple[int, ...],
    ) -> bool:
        if previous is None or current is None:
            return False
        height, width = roi_shape[:2]
        distance = float(np.hypot(current[0] - previous[0], current[1] - previous[1]))
        return distance <= max(width, height) * WHITEBOARD_PERSON_REGION_STABLE_RATIO

    # Maneja completion region dwelled dentro de WhiteboardAnalyzer.
    def _completion_region_dwelled(self, frame_number: int) -> bool:
        if self._completion_region_start_frame < 0:
            return False
        elapsed_frames = max(0, int(frame_number) - int(self._completion_region_start_frame))
        elapsed_seconds = float(elapsed_frames) / max(1.0, float(self._fps))
        return elapsed_seconds >= WHITEBOARD_PERSON_REGION_DWELL_SECONDS

    # Maneja mask ratio dentro de WhiteboardAnalyzer.
    def _mask_ratio(self, mask: Optional[np.ndarray], roi_shape: Tuple[int, int]) -> float:
        if mask is None or mask.size == 0:
            return 0.0
        work = mask
        if work.shape[:2] != roi_shape[:2]:
            work = cv2.resize(work, (roi_shape[1], roi_shape[0]), interpolation=cv2.INTER_NEAREST)
        if work.ndim == 3:
            work = work[:, :, 0]
        return float(cv2.countNonZero(np.where(work > 0, 255, 0).astype(np.uint8))) / float(max(1, work.size))

    # Maneja neutralize blocked region dentro de WhiteboardAnalyzer.
    def _neutralize_blocked_region(
        self,
        image: np.ndarray,
        blocked_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        snapshot = image.copy()
        if blocked_mask is None or blocked_mask.size == 0:
            return snapshot

        mask = blocked_mask
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        blocked = np.where(mask > 0, 255, 0).astype(np.uint8)
        if cv2.countNonZero(blocked) == 0:
            return snapshot

        kernel = np.ones((7, 7), dtype=np.uint8)
        blocked = cv2.dilate(blocked, kernel, iterations=1)
        if snapshot.ndim == 2:
            snapshot[blocked > 0] = WHITEBOARD_BLOCKED_NEUTRAL_GRAY
        else:
            snapshot[blocked > 0] = (
                WHITEBOARD_BLOCKED_NEUTRAL_GRAY,
                WHITEBOARD_BLOCKED_NEUTRAL_GRAY,
                WHITEBOARD_BLOCKED_NEUTRAL_GRAY,
            )
        return snapshot

    # Escribe snapshot.
    def _write_snapshot(self, frame_number: int, event_type: str, roi: np.ndarray) -> str:
        path = self.snapshots_dir / f"frame_{int(frame_number):06d}_{event_type}.jpg"
        self._write_image(path, roi, [cv2.IMWRITE_JPEG_QUALITY, WHITEBOARD_LLM_JPEG_QUALITY])
        return str(path)

    # Escribe response.
    def _write_response(
        self,
        task: WhiteboardSnapshotTask,
        payload: Dict[str, Any],
        elapsed_sec: float,
    ) -> str:
        path = self.responses_dir / f"frame_{int(task.frame_number):06d}_{task.event_type}.json"
        self._write_json(
            path,
            {
                "task": {
                    "task_id": int(task.task_id),
                    "frame_number": int(task.frame_number),
                    "event_type": task.event_type,
                    "reason": task.reason,
                    "person_bbox": task.person_bbox,
                    "person_area_ratio": float(task.person_area_ratio),
                    "blocked_ratio": float(task.blocked_ratio),
                    "previous_content": task.previous_content,
                },
                "elapsed_sec": float(elapsed_sec),
                "payload": payload,
            },
        )
        return str(path)

    # Escribe summary.
    def _write_summary(self) -> None:
        self._write_json(self.analysis_dir / "summary.json", self.summary())

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

    # Maneja string list dentro de WhiteboardAnalyzer.
    def _string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    # Evalua la condicion is code category y devuelve el resultado booleano.
    def _is_code_category(self, value: str) -> bool:
        normalized = str(value).strip().lower()
        return normalized in {"codigo", "código", "code", "programacion", "programación"}

    # Parsea bool.
    def _parse_bool(self, value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "si", "sí"}
        return bool(value)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "WhiteboardAnalyzer",
    "WhiteboardLLMClient",
    "WhiteboardSnapshotEvent",
]
