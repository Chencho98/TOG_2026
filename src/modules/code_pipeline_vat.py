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
    from src.modules.code_segmentation_vat import *
except ImportError:
    try:
        from code_segmentation_vat import *
    except ImportError:
        from .code_segmentation_vat import *

try:
    from src.modules.code_vision_vat import *
except ImportError:
    try:
        from code_vision_vat import *
    except ImportError:
        from .code_vision_vat import *

try:
    from src.modules.code_store_vat import *
except ImportError:
    try:
        from code_store_vat import *
    except ImportError:
        from .code_store_vat import *

# ============================================================================
# ASYNC CODE INTELLIGENCE PIPELINE
# ============================================================================

class CodeIntelligencePipeline:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self, debug_visualization: bool = False):
        self.debug_visualization = bool(debug_visualization)
        self._debug_enabled = env_flag(CODE_PIPELINE_DEBUG_ENV_VAR, CODE_PIPELINE_DEBUG_DEFAULT)
        self.segmenter = CodeViewportSegmenter(debug_visualization=debug_visualization)
        self.sidebar_tracker = VisualChangeTracker(
            CODE_SIDEBAR_CHANGE_THRESHOLD,
            CODE_SIDEBAR_EDGE_RATIO_DELTA,
        )
        self.editor_tracker = VisualChangeTracker(
            CODE_EDITOR_CHANGE_THRESHOLD,
            CODE_EDITOR_EDGE_RATIO_DELTA,
        )
        self.store = CodeKnowledgeStore(debug_logger=self._debug)
        self.client = build_code_vision_client()
        self._sidebar_frame_candidates: List[BufferedVisionCandidate] = []
        self._editor_frame_candidates: List[BufferedVisionCandidate] = []

        self._editor_request_queue: Queue = Queue(maxsize=CODE_EDITOR_QUEUE_SIZE)
        self._sidebar_request_queue: Queue = Queue(maxsize=CODE_SIDEBAR_QUEUE_SIZE)
        self._response_queue: Queue = Queue()
        self._request_sentinel = object()
        self._response_sentinel = object()
        self._errors: List[BaseException] = []
        self._error_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._disabled = False
        self._disable_reason: Optional[str] = None
        self._next_task_id = 1
        self._queued_counts: Dict[str, int] = {"sidebar": 0, "editor": 0}
        self._in_flight_counts: Dict[str, int] = {"sidebar": 0, "editor": 0}
        self._stats: Dict[str, Dict[str, int]] = {
            "sidebar": {
                "enqueued": 0,
                "started": 0,
                "responses_ok": 0,
                "responses_error": 0,
                "skipped_error_payload": 0,
                "assembled": 0,
                "stale_applied": 0,
                "dropped": 0,
            },
            "editor": {
                "enqueued": 0,
                "started": 0,
                "responses_ok": 0,
                "responses_error": 0,
                "skipped_error_payload": 0,
                "assembled": 0,
                "stale_applied": 0,
                "dropped": 0,
            },
        }
        self._drop_history: List[Dict[str, Any]] = []
        self._last_editor_submit_frame = -1
        self._last_editor_selected_frame_number = -1
        self._last_editor_selected_score = 0.0
        self._last_editor_selected_quality_rank = 0
        self._warning_emitted = False
        self._final_summary: Optional[Dict[str, Any]] = None

        self._editor_request_threads = [
            threading.Thread(
                target=self._run_request_worker,
                args=("editor", self._editor_request_queue, worker_index + 1),
                name=f"code-editor-request-worker-{worker_index + 1}",
                daemon=True,
            )
            for worker_index in range(CODE_EDITOR_MAX_IN_FLIGHT)
        ]
        self._sidebar_request_threads = [
            threading.Thread(
                target=self._run_request_worker,
                args=("sidebar", self._sidebar_request_queue, worker_index + 1),
                name=f"code-sidebar-request-worker-{worker_index + 1}",
                daemon=True,
            )
            for worker_index in range(CODE_SIDEBAR_MAX_IN_FLIGHT)
        ]
        self._assemble_thread = threading.Thread(
            target=self._run_assembler,
            name="code-assembly-worker",
            daemon=True,
        )
        for thread in self._editor_request_threads + self._sidebar_request_threads:
            thread.start()
        self._assemble_thread.start()
        self._debug(
            "pipeline started "
            f"editor_workers={CODE_EDITOR_MAX_IN_FLIGHT} "
            f"sidebar_workers={CODE_SIDEBAR_MAX_IN_FLIGHT} "
            f"editor_queue={CODE_EDITOR_QUEUE_SIZE} "
            f"sidebar_queue={CODE_SIDEBAR_QUEUE_SIZE} "
            f"interval={CODE_EDITOR_SEND_INTERVAL_SECONDS:.2f}s "
            f"window={CODE_FRAME_SELECTION_WINDOW_SIZE} "
            f"editor_encoding={CODE_EDITOR_IMAGE_FORMAT}"
        )

    # Configura los directorios de salida de la sesion.
    def set_session_dir(self, session_dir: Path) -> None:
        self.store.set_session_dir(Path(session_dir))

    # Maneja submit sample dentro de CodeIntelligencePipeline.
    def submit_sample(
        self,
        roi: np.ndarray,
        frame_number: int,
        fps: float = 0.0,
        language_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        snapshot = self.store.snapshot()
        result = {
            "active_file": snapshot.get("active_file"),
            "known_files": int(snapshot.get("known_files", 0)),
            "structure_version": int(snapshot.get("structure_version", 0)),
            "structure_changed": False,
            "sidebar_signature": snapshot.get("sidebar_signature"),
            "pipeline_disabled": self._is_disabled(),
            "debug": None,
        }

        if self._is_disabled():
            return result

        segmentation = self.segmenter.segment(roi)

        if segmentation is None:
            return result

        self._remember_frame_candidate("sidebar", segmentation.sidebar_crop, frame_number)
        self._remember_frame_candidate("editor", segmentation.editor_crop, frame_number)

        sidebar_eval = self.sidebar_tracker.evaluate(segmentation.sidebar_crop, frame_number)
        result["structure_changed"] = bool(sidebar_eval.get("should_send", False))
        result["sidebar_signature"] = sidebar_eval.get("signature") or result["sidebar_signature"]
        editor_eval = self.editor_tracker.evaluate(segmentation.editor_crop, frame_number)
        known_active_file = snapshot.get("active_file")
        editor_prompt_hint = snapshot.get("editor_prompt_hint")
        editor_candidate = self._select_editor_stream_candidate()
        if editor_candidate is None and segmentation.editor_crop.size > 0:
            editor_candidate = BufferedVisionCandidate(
                kind="editor",
                frame_number=int(frame_number),
                image=segmentation.editor_crop.copy(),
                score=0.0,
                metrics={},
            )
        periodic_due = False
        if segmentation.editor_crop.size > 0:
            if fps > 0:
                interval_frames = max(int(round(float(fps) * CODE_EDITOR_SEND_INTERVAL_SECONDS)), 1)
                periodic_due = (
                    self._last_editor_submit_frame < 0
                    or (int(frame_number) - self._last_editor_submit_frame) >= interval_frames
                )
            else:
                periodic_due = self._last_editor_submit_frame < 0

        if sidebar_eval.get("should_send") and segmentation.sidebar_crop.size > 0:
            sidebar_candidate = self._select_best_frame_candidate("sidebar")
            if sidebar_candidate is None:
                sidebar_candidate = BufferedVisionCandidate(
                    kind="sidebar",
                    frame_number=int(frame_number),
                    image=segmentation.sidebar_crop.copy(),
                    score=0.0,
                    metrics={},
                )
            self._enqueue_task(
                CodeVisionTask(
                    kind="sidebar",
                    frame_number=int(frame_number),
                    image=sidebar_candidate.image.copy(),
                    active_file_hint=known_active_file,
                    language_hint=language_hint,
                    image_format=CODE_SIDEBAR_IMAGE_FORMAT,
                    jpeg_quality=CODE_SIDEBAR_JPEG_QUALITY,
                    meta={
                        "signature": sidebar_eval.get("signature"),
                        "change_score": sidebar_eval.get("change_score"),
                        "edge_ratio": sidebar_eval.get("edge_ratio"),
                        "reason": sidebar_eval.get("reason"),
                        "trigger_frame_number": int(frame_number),
                        "selected_frame_number": int(sidebar_candidate.frame_number),
                        "selected_score": float(sidebar_candidate.score),
                        "selected_metrics": dict(sidebar_candidate.metrics),
                        "send_reasons": [f"change:{sidebar_eval.get('reason') or 'unknown'}"],
                    },
                )
            )

        should_send_editor = bool(
            segmentation.editor_crop.size > 0
            and (
                editor_eval.get("should_send")
                or sidebar_eval.get("should_send")
                or not known_active_file
                or periodic_due
            )
        )
        if should_send_editor:
            send_reasons: List[str] = []
            if editor_eval.get("should_send"):
                send_reasons.append(f"change:{editor_eval.get('reason') or 'unknown'}")
            if sidebar_eval.get("should_send"):
                send_reasons.append("sidebar_changed")
            if not known_active_file:
                send_reasons.append("missing_active_file")
            if periodic_due:
                send_reasons.append("periodic")
            enqueued = self._enqueue_task(
                CodeVisionTask(
                    kind="editor",
                    frame_number=int(frame_number),
                    image=editor_candidate.image.copy(),
                    active_file_hint=editor_prompt_hint,
                    language_hint=language_hint,
                    image_format=CODE_EDITOR_IMAGE_FORMAT,
                    jpeg_quality=CODE_EDITOR_JPEG_QUALITY,
                    meta={
                        "sidebar_signature": sidebar_eval.get("signature"),
                        "editor_reason": editor_eval.get("reason"),
                        "editor_change_score": editor_eval.get("change_score"),
                        "editor_edge_ratio": editor_eval.get("edge_ratio"),
                        "trigger_frame_number": int(frame_number),
                        "selected_frame_number": int(editor_candidate.frame_number),
                        "selected_score": float(editor_candidate.score),
                        "frame_quality": str(editor_candidate.metrics.get("frame_quality") or "usable"),
                        "estimated_visible_rows": int(safe_float(editor_candidate.metrics.get("estimated_visible_rows"), 0.0)),
                        "selected_metrics": dict(editor_candidate.metrics),
                        "send_reasons": send_reasons,
                    },
                )
            )
            if enqueued:
                self._last_editor_submit_frame = int(frame_number)
                self._last_editor_selected_frame_number = int(editor_candidate.frame_number)
                self._last_editor_selected_score = float(editor_candidate.score)
                self._last_editor_selected_quality_rank = editor_frame_quality_rank(
                    editor_candidate.metrics.get("frame_quality")
                )

        if self.debug_visualization:
            edges = sidebar_eval.get("edges")
            edges_bgr = np.zeros((0, 0, 3), dtype=np.uint8)
            if isinstance(edges, np.ndarray) and edges.size > 0:
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            result["debug"] = {
                "layout_overlay": segmentation.overlay.copy(),
                "layout_sheet": segmentation.debug_sheet.copy(),
                "sidebar_crop": segmentation.sidebar_crop.copy(),
                "editor_crop": segmentation.editor_crop.copy(),
                "sidebar_edges": edges_bgr,
            }

        return result

    # Registra frame candidate.
    def _remember_frame_candidate(
        self,
        kind: str,
        image: np.ndarray,
        frame_number: int,
    ) -> None:
        if image is None or image.size == 0:
            return
        score, metrics = self._score_frame_candidate(kind, image)
        candidate = BufferedVisionCandidate(
            kind=kind,
            frame_number=int(frame_number),
            image=image.copy(),
            score=float(score),
            metrics=metrics,
        )
        if kind == "sidebar":
            self._sidebar_frame_candidates.append(candidate)
            if len(self._sidebar_frame_candidates) > CODE_FRAME_SELECTION_WINDOW_SIZE:
                self._sidebar_frame_candidates = self._sidebar_frame_candidates[-CODE_FRAME_SELECTION_WINDOW_SIZE:]
        else:
            self._editor_frame_candidates.append(candidate)
            if len(self._editor_frame_candidates) > CODE_FRAME_SELECTION_WINDOW_SIZE:
                self._editor_frame_candidates = self._editor_frame_candidates[-CODE_FRAME_SELECTION_WINDOW_SIZE:]

    # Selecciona best frame candidate.
    def _select_best_frame_candidate(self, kind: str) -> Optional[BufferedVisionCandidate]:
        candidates = self._sidebar_frame_candidates if kind == "sidebar" else self._editor_frame_candidates
        if not candidates:
            return None
        if kind == "editor":
            return max(
                candidates,
                key=lambda item: (
                    editor_frame_quality_rank(item.metrics.get("frame_quality")),
                    float(item.score),
                    int(item.frame_number),
                ),
            )
        return max(candidates, key=lambda item: (float(item.score), int(item.frame_number)))

    # Selecciona editor stream candidate.
    def _select_editor_stream_candidate(self) -> Optional[BufferedVisionCandidate]:
        candidates = list(self._editor_frame_candidates)
        if not candidates:
            return None

        usable_candidates = [
            item
            for item in candidates
            if str(item.metrics.get("frame_quality") or "usable") != "scroll_blur"
        ]
        if not usable_candidates:
            usable_candidates = candidates

        newer_candidates = [
            item
            for item in usable_candidates
            if int(item.frame_number) > int(self._last_editor_selected_frame_number)
        ]
        if newer_candidates:
            return max(
                newer_candidates,
                key=lambda item: (
                    int(item.frame_number),
                    editor_frame_quality_rank(item.metrics.get("frame_quality")),
                    float(item.score),
                ),
            )

        return self._select_best_frame_candidate("editor")

    # Maneja editor candidate is materially better dentro de CodeIntelligencePipeline.
    def _editor_candidate_is_materially_better(
        self,
        candidate: Optional[BufferedVisionCandidate],
    ) -> bool:
        if candidate is None:
            return False
        quality_rank = editor_frame_quality_rank(candidate.metrics.get("frame_quality"))
        if quality_rank > self._last_editor_selected_quality_rank:
            return True
        if quality_rank == self._last_editor_selected_quality_rank and (
            float(candidate.score) - float(self._last_editor_selected_score)
        ) >= CODE_EDITOR_QUALITY_IMPROVEMENT_DELTA:
            return True
        return False

    # Maneja score frame candidate dentro de CodeIntelligencePipeline.
    def _score_frame_candidate(
        self,
        kind: str,
        image: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        if image is None or image.size == 0:
            return 0.0, {}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        if gray.size == 0:
            return 0.0, {}

        metric_view = gray
        if kind == "editor":
            height, width = gray.shape[:2]
            x0 = clamp(int(round(width * CODE_EDITOR_SCORE_TRIM_LEFT_RATIO)), 0, max(0, width - 1))
            x1 = clamp(int(round(width * (1.0 - CODE_EDITOR_SCORE_TRIM_RIGHT_RATIO))), x0 + 1, width)
            y0 = clamp(int(round(height * CODE_EDITOR_SCORE_TRIM_TOP_RATIO)), 0, max(0, height - 1))
            y1 = clamp(int(round(height * (1.0 - CODE_EDITOR_SCORE_TRIM_BOTTOM_RATIO))), y0 + 1, height)
            cropped = gray[y0:y1, x0:x1]
            if cropped.size > 0:
                metric_view = cropped

        center_view = metric_view
        if kind == "editor":
            vh, vw = metric_view.shape[:2]
            cx0 = clamp(int(round(vw * 0.08)), 0, max(0, vw - 1))
            cx1 = clamp(int(round(vw * 0.92)), cx0 + 1, vw)
            cy0 = clamp(int(round(vh * 0.06)), 0, max(0, vh - 1))
            cy1 = clamp(int(round(vh * 0.94)), cy0 + 1, vh)
            cropped = metric_view[cy0:cy1, cx0:cx1]
            if cropped.size > 0:
                center_view = cropped

        sharpness = float(np.log1p(cv2.Laplacian(metric_view, cv2.CV_64F).var()))
        center_sharpness = float(np.log1p(cv2.Laplacian(center_view, cv2.CV_64F).var()))
        contrast = float(metric_view.std() / 64.0)
        edges = cv2.Canny(metric_view, 60, 160)
        edge_density = 0.0 if edges.size == 0 else float(cv2.countNonZero(edges)) / float(edges.size)
        _, binary = cv2.threshold(metric_view, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text_density = 0.0 if binary.size == 0 else float(cv2.countNonZero(binary)) / float(binary.size)
        diff_x = 0.0 if metric_view.shape[1] <= 1 else float(np.mean(np.abs(np.diff(metric_view.astype(np.float32), axis=1))))
        diff_y = 0.0 if metric_view.shape[0] <= 1 else float(np.mean(np.abs(np.diff(metric_view.astype(np.float32), axis=0))))
        dominant_diff = max(diff_x, diff_y, 1e-6)
        axis_balance = float(min(diff_x, diff_y) / dominant_diff)
        row_signal = smooth_signal(
            binary.mean(axis=1).astype(np.float32) / 255.0,
            max(5, metric_view.shape[0] // 120),
        )
        row_threshold = max(0.006, float(np.percentile(row_signal, 70) * 0.55))
        estimated_visible_rows = count_sustained_runs(
            row_signal,
            row_threshold,
            max(1, metric_view.shape[0] // 180),
        )
        row_variation = float(row_signal.std()) if row_signal.size > 0 else 0.0

        if kind == "editor":
            frame_quality = classify_editor_frame_quality(
                {
                    "sharpness": sharpness,
                    "edge_density": edge_density,
                    "axis_balance": axis_balance,
                    "row_variation": row_variation,
                }
            )
            score = (
                (1.8 * sharpness)
                + (1.5 * center_sharpness)
                + (1.1 * contrast)
                + (6.0 * edge_density)
                + (2.6 * text_density)
                + (1.8 * axis_balance)
                + (8.0 * row_variation)
                + (0.14 * min(estimated_visible_rows, 48))
            )
            if frame_quality == "partial_blur":
                score -= 2.5
            elif frame_quality == "scroll_blur":
                score -= 6.0
        else:
            frame_quality = "usable"
            score = (
                (1.6 * sharpness)
                + (1.2 * contrast)
                + (4.2 * edge_density)
                + (1.8 * text_density)
            )
        return float(score), {
            "sharpness": float(sharpness),
            "center_sharpness": float(center_sharpness),
            "contrast": float(contrast),
            "edge_density": float(edge_density),
            "text_density": float(text_density),
            "diff_x": float(diff_x),
            "diff_y": float(diff_y),
            "axis_balance": float(axis_balance),
            "row_variation": float(row_variation),
            "estimated_visible_rows": float(estimated_visible_rows),
            "frame_quality": frame_quality,
            "frame_quality_rank": float(editor_frame_quality_rank(frame_quality)),
        }

    # Finaliza el procesamiento pendiente y devuelve el resumen producido.
    def finalize(self) -> Dict[str, Any]:
        if self._closed:
            return dict(self._final_summary or {})

        self._closing = True
        alive_editor_threads = [thread for thread in self._editor_request_threads if thread.is_alive()]
        alive_sidebar_threads = [thread for thread in self._sidebar_request_threads if thread.is_alive()]

        if alive_editor_threads:
            for _ in alive_editor_threads:
                self._editor_request_queue.put(self._request_sentinel)
            for thread in alive_editor_threads:
                thread.join()
        else:
            self._drain_queue(self._editor_request_queue, kind="editor")

        if alive_sidebar_threads:
            for _ in alive_sidebar_threads:
                self._sidebar_request_queue.put(self._request_sentinel)
            for thread in alive_sidebar_threads:
                thread.join()
        else:
            self._drain_queue(self._sidebar_request_queue, kind="sidebar")

        self._drain_queue(self._editor_request_queue, kind="editor")
        self._drain_queue(self._sidebar_request_queue, kind="sidebar")

        if self._assemble_thread.is_alive():
            self._response_queue.put(self._response_sentinel)
            self._assemble_thread.join()
        else:
            self._drain_queue(self._response_queue)
        try:
            self.client.close()
        except Exception:
            pass
        self._closed = True
        self._final_summary = self.store.finalize()
        self._final_summary["code_pipeline_stats"] = self._build_runtime_stats()
        if self._disable_reason:
            self._final_summary["code_pipeline_disabled"] = True
            self._final_summary["code_pipeline_disable_reason"] = self._disable_reason
        runtime_stats = self._final_summary.get("code_pipeline_stats", {})
        self._debug(
            "finalize "
            f"editor={runtime_stats.get('editor')} "
            f"sidebar={runtime_stats.get('sidebar')} "
            f"drops={len(runtime_stats.get('recent_drops', []))}"
        )
        return dict(self._final_summary or {})

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> None:
        self.finalize()

    # Maneja enqueue task dentro de CodeIntelligencePipeline.
    def _enqueue_task(self, task: CodeVisionTask) -> bool:
        if self._closing or self._closed or self._is_disabled():
            return False
        if task.task_id <= 0:
            task.task_id = self._reserve_task_id()

        queue_obj = self._request_queue_for_kind(task.kind)
        dropped_task: Optional[CodeVisionTask] = None

        try:
            queue_obj.put_nowait(task)
        except Full:
            try:
                evicted = queue_obj.get_nowait()
                queue_obj.task_done()
                if isinstance(evicted, CodeVisionTask):
                    dropped_task = evicted
                    with self._state_lock:
                        self._queued_counts[task.kind] = max(0, self._queued_counts[task.kind] - 1)
                else:
                    evicted = None
            except Empty:
                evicted = None

            if dropped_task is not None:
                self._record_drop_event(
                    dropped_task,
                    "backpressure_evict_oldest",
                    replacement_frame=int(task.frame_number),
                )

            try:
                queue_obj.put_nowait(task)
            except Full:
                self._record_drop_event(task, "backpressure_drop_new")
                return False

        with self._state_lock:
            self._queued_counts[task.kind] += 1
            self._stats[task.kind]["enqueued"] += 1
            queued = self._queued_counts[task.kind]
            in_flight = self._in_flight_counts[task.kind]

        reasons = ",".join(str(reason) for reason in task.meta.get("send_reasons", [])) or "-"
        self._debug(
            f"enqueue kind={task.kind} task={task.task_id} frame={task.frame_number} "
            f"queued={queued} in_flight={in_flight} reasons={reasons}"
        )
        return True

    # Ejecuta request worker.
    def _run_request_worker(self, kind: str, queue_obj: Queue, worker_index: int) -> None:
        worker_name = f"{kind}-req-{worker_index}"
        try:
            while True:
                task = queue_obj.get()
                try:
                    if task is self._request_sentinel:
                        return
                    if not isinstance(task, CodeVisionTask):
                        continue

                    request_start_ts = time.time()
                    queued_for_request_sec = max(0.0, request_start_ts - float(task.created_ts))
                    with self._state_lock:
                        self._queued_counts[kind] = max(0, self._queued_counts[kind] - 1)
                        self._in_flight_counts[kind] += 1
                        self._stats[kind]["started"] += 1
                        queued = self._queued_counts[kind]
                        in_flight = self._in_flight_counts[kind]
                    self._debug(
                        f"send kind={kind} worker={worker_name} task={task.task_id} "
                        f"frame={task.frame_number} queue_wait={queued_for_request_sec:.3f}s "
                        f"queued={queued} in_flight={in_flight}"
                    )

                    try:
                        if kind == "sidebar":
                            payload = self.client.analyze_sidebar(task)
                        else:
                            payload = self.client.analyze_editor(task)
                    except BaseException as exc:
                        with self._state_lock:
                            self._in_flight_counts[kind] = max(0, self._in_flight_counts[kind] - 1)
                        self._debug(
                            f"fatal kind={kind} worker={worker_name} task={task.task_id} "
                            f"frame={task.frame_number} error={exc}"
                        )
                        self._disable_pipeline(exc)
                        return

                    completed_ts = time.time()
                    request_duration_sec = max(0.0, completed_ts - request_start_ts)
                    payload_error = payload.get("error") if isinstance(payload, dict) else None
                    usage_text = self._format_usage(self._extract_usage(payload))
                    with self._state_lock:
                        self._in_flight_counts[kind] = max(0, self._in_flight_counts[kind] - 1)
                        if payload_error:
                            self._stats[kind]["responses_error"] += 1
                        else:
                            self._stats[kind]["responses_ok"] += 1
                        in_flight = self._in_flight_counts[kind]
                        response_backlog = self._response_queue.qsize()
                    status = f"error={payload_error}" if payload_error else "ok"
                    self._debug(
                        f"recv kind={kind} worker={worker_name} task={task.task_id} "
                        f"frame={task.frame_number} status={status} "
                        f"request={request_duration_sec:.3f}s queue_wait={queued_for_request_sec:.3f}s "
                        f"response_backlog={response_backlog} in_flight={in_flight}{usage_text}"
                    )
                    if response_backlog >= CODE_RESPONSE_QUEUE_WARN_SIZE:
                        self._debug(
                            f"backlog response_queue={response_backlog} "
                            f"after task={task.task_id} frame={task.frame_number}"
                        )
                    self._response_queue.put(
                        CodeVisionResponse(
                            task=task,
                            payload=payload,
                            worker_name=worker_name,
                            queued_for_request_sec=queued_for_request_sec,
                            request_duration_sec=request_duration_sec,
                            completed_ts=completed_ts,
                        )
                    )
                finally:
                    queue_obj.task_done()
        except BaseException as exc:
            self._disable_pipeline(exc)

    # Ejecuta assembler.
    def _run_assembler(self) -> None:
        try:
            while True:
                item = self._response_queue.get()
                try:
                    if item is self._response_sentinel:
                        return
                    if not isinstance(item, CodeVisionResponse):
                        continue

                    item.assembled_delay_sec = max(
                        0.0,
                        time.time() - item.completed_ts,
                    ) if item.completed_ts else 0.0

                    payload_error = item.payload.get("error") if isinstance(item.payload, dict) else None
                    if payload_error:
                        usage_text = self._format_usage(self._extract_usage(item.payload))
                        with self._state_lock:
                            self._stats[item.task.kind]["skipped_error_payload"] += 1
                        self._debug(
                            f"skip kind={item.task.kind} task={item.task.task_id} "
                            f"frame={item.task.frame_number} reason={payload_error} "
                            f"assembled_delay={item.assembled_delay_sec:.3f}s{usage_text}"
                        )
                        continue

                    try:
                        if item.task.kind == "sidebar":
                            apply_info = self.store.apply_sidebar_result(item)
                        else:
                            apply_info = self.store.apply_editor_result(item)
                    except BaseException as exc:
                        self._warn_once(
                            f"⚠️ Code intelligence ignoró una respuesta inválida: {exc}"
                        )
                        self._debug(
                            f"assemble-error kind={item.task.kind} task={item.task.task_id} "
                            f"frame={item.task.frame_number} error={exc}"
                        )
                        continue

                    with self._state_lock:
                        self._stats[item.task.kind]["assembled"] += 1
                        if apply_info.get("stale_metadata"):
                            self._stats[item.task.kind]["stale_applied"] += 1

                    measure_label = "entries" if item.task.kind == "sidebar" else "lines"
                    measure_value = apply_info.get("entry_count") if item.task.kind == "sidebar" else apply_info.get("line_count")
                    total_age = (
                        float(item.queued_for_request_sec)
                        + float(item.request_duration_sec)
                        + float(item.assembled_delay_sec)
                    )
                    self._debug(
                        f"apply kind={item.task.kind} task={item.task.task_id} "
                        f"frame={item.task.frame_number} {measure_label}={measure_value} "
                        f"stale={apply_info.get('stale_metadata')} "
                        f"active={apply_info.get('active_file') or '-'} "
                        f"active_applied={apply_info.get('active_file_applied')} "
                        f"assembled_delay={item.assembled_delay_sec:.3f}s total={total_age:.3f}s"
                    )
                finally:
                    self._response_queue.task_done()
        except BaseException as exc:
            self._disable_pipeline(exc)

    # Registra error.
    def _record_error(self, exc: BaseException) -> None:
        with self._error_lock:
            self._errors.append(exc)

    # Maneja reserve task id dentro de CodeIntelligencePipeline.
    def _reserve_task_id(self) -> int:
        with self._state_lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            return int(task_id)

    # Maneja request queue for kind dentro de CodeIntelligencePipeline.
    def _request_queue_for_kind(self, kind: str) -> Queue:
        if kind == "sidebar":
            return self._sidebar_request_queue
        return self._editor_request_queue

    # Registra drop event.
    def _record_drop_event(
        self,
        task: CodeVisionTask,
        reason: str,
        replacement_frame: Optional[int] = None,
    ) -> None:
        event = {
            "kind": task.kind,
            "task_id": int(task.task_id),
            "frame_number": int(task.frame_number),
            "reason": reason,
            "replacement_frame": replacement_frame,
            "age_sec": max(0.0, time.time() - float(task.created_ts)),
            "send_reasons": list(task.meta.get("send_reasons", [])),
        }
        with self._state_lock:
            self._stats[task.kind]["dropped"] += 1
            self._drop_history.append(event)
            if len(self._drop_history) > 200:
                self._drop_history = self._drop_history[-200:]
            queued = self._queued_counts[task.kind]
            in_flight = self._in_flight_counts[task.kind]
        self._debug(
            f"drop kind={task.kind} task={task.task_id} frame={task.frame_number} "
            f"reason={reason} replacement_frame={replacement_frame if replacement_frame is not None else '-'} "
            f"queued={queued} in_flight={in_flight} "
            f"send_reasons={','.join(event['send_reasons']) or '-'}"
        )

    # Construye runtime stats.
    def _build_runtime_stats(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "editor": {
                    **self._stats["editor"],
                    "queued": int(self._queued_counts["editor"]),
                    "in_flight": int(self._in_flight_counts["editor"]),
                },
                "sidebar": {
                    **self._stats["sidebar"],
                    "queued": int(self._queued_counts["sidebar"]),
                    "in_flight": int(self._in_flight_counts["sidebar"]),
                },
                "recent_drops": list(self._drop_history[-50:]),
            }

    # Extrae usage.
    def _extract_usage(self, payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        usage = payload.get("__usage__")
        if not isinstance(usage, dict):
            return None
        return dict(usage)

    # Formatea usage.
    def _format_usage(self, usage: Optional[Dict[str, Any]]) -> str:
        if not usage:
            return ""
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        parts = []
        if total_tokens is not None:
            parts.append(f"total={total_tokens}")
        if prompt_tokens is not None:
            parts.append(f"prompt={prompt_tokens}")
        if completion_tokens is not None:
            parts.append(f"completion={completion_tokens}")
        if not parts:
            return ""
        return " usage[" + " ".join(parts) + "]"

    # Maneja drain queue dentro de CodeIntelligencePipeline.
    def _drain_queue(self, queue_obj: Queue, kind: Optional[str] = None) -> None:
        while True:
            try:
                item = queue_obj.get_nowait()
            except Empty:
                return
            else:
                if kind is not None and isinstance(item, CodeVisionTask):
                    with self._state_lock:
                        self._queued_counts[kind] = max(0, self._queued_counts[kind] - 1)
                queue_obj.task_done()

    # Evalua la condicion is disabled y devuelve el resultado booleano.
    def _is_disabled(self) -> bool:
        with self._state_lock:
            return bool(self._disabled)

    # Maneja disable pipeline dentro de CodeIntelligencePipeline.
    def _disable_pipeline(self, exc: BaseException) -> None:
        if exc is None:
            return
        self._record_error(exc)
        with self._state_lock:
            if self._disabled:
                return
            self._disabled = True
            self._disable_reason = str(exc)
        self._warn_once(f"⚠️ Code intelligence deshabilitada: {exc}")

    # Maneja warn once dentro de CodeIntelligencePipeline.
    def _warn_once(self, message: str) -> None:
        with self._state_lock:
            if self._warning_emitted:
                return
            self._warning_emitted = True
        print(message)

    # Maneja debug dentro de CodeIntelligencePipeline.
    def _debug(self, message: str) -> None:
        if not self._debug_enabled:
            return
        timestamp = time.time()
        seconds = time.strftime("%H:%M:%S", time.localtime(timestamp))
        millis = int((timestamp - int(timestamp)) * 1000)
        print(f"🧠 [{seconds}.{millis:03d}] {message}")
