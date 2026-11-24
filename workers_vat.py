"""
Background workers for disk writes and projection content classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, Sequence
import copy
import json
import threading

import cv2
import numpy as np

try:
    from src.core.classes_vat import ProjectionClassification
except ImportError:
    from classes_vat import ProjectionClassification

try:
    from src.modules.pdf_vat import ProjectionClassifier
except ImportError:
    from pdf_vat import ProjectionClassifier


# ============================================================================
# QUEUE PAYLOADS
# ============================================================================

@dataclass
class ClassificationTask:
    generation: int
    frame_number: int
    fps: float
    roi: np.ndarray


@dataclass
class ClassificationResult:
    generation: int
    frame_number: int
    classification: ProjectionClassification


# ============================================================================
# ASYNC DISK WRITER
# ============================================================================

class AsyncDiskWriter:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self, max_queue_size: int = 128):
        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._sentinel = object()
        self._thread = threading.Thread(target=self._run, name="async-disk-writer", daemon=True)
        self._errors: List[BaseException] = []
        self._error_lock = threading.Lock()
        self._closed = False
        self._thread.start()

    # Encola una imagen para escritura asincrona en disco.
    def submit_image(
        self,
        path: Path,
        image: np.ndarray,
        params: Optional[Sequence[int]] = None,
    ) -> None:
        self._raise_if_error()
        if self._closed:
            raise RuntimeError("AsyncDiskWriter ya está cerrado")
        payload = ("image", Path(path), image.copy(), None if params is None else list(params))
        self._queue.put(payload)

    # Encola datos JSON para escritura asincrona en disco.
    def submit_json(
        self,
        path: Path,
        data: Any,
        *,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> None:
        self._raise_if_error()
        if self._closed:
            raise RuntimeError("AsyncDiskWriter ya está cerrado")
        payload = (
            "json",
            Path(path),
            copy.deepcopy(data),
            {"indent": indent, "ensure_ascii": ensure_ascii},
        )
        self._queue.put(payload)

    # Fuerza la escritura de los datos acumulados pendientes.
    def flush(self) -> None:
        self._queue.join()
        self._raise_if_error()

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> None:
        if self._closed:
            self._raise_if_error()
            return

        self.flush()
        self._closed = True
        self._queue.put(self._sentinel)
        self._thread.join()
        self._raise_if_error()

    # Maneja run dentro de AsyncDiskWriter.
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._sentinel:
                    return

                kind = item[0]
                if kind == "image":
                    _, path, image, params = item
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if params:
                        cv2.imwrite(str(path), image, params)
                    else:
                        cv2.imwrite(str(path), image)
                elif kind == "json":
                    _, path, data, options = item
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as file_obj:
                        json.dump(
                            data,
                            file_obj,
                            indent=options["indent"],
                            ensure_ascii=options["ensure_ascii"],
                        )
            except BaseException as exc:
                self._record_error(exc)
            finally:
                self._queue.task_done()

    # Registra error.
    def _record_error(self, exc: BaseException) -> None:
        with self._error_lock:
            self._errors.append(exc)

    # Maneja raise if error dentro de AsyncDiskWriter.
    def _raise_if_error(self) -> None:
        with self._error_lock:
            if not self._errors:
                return
            exc = self._errors[0]
        raise RuntimeError(f"AsyncDiskWriter falló: {exc}") from exc


# ============================================================================
# PROJECTION CLASSIFICATION WORKER
# ============================================================================

class ProjectionContentWorker:
    """
    Runs projection classification in a dedicated thread.

    The coordinator keeps geometry/state updates serial while OCR and PDF matching
    run in the background.
    """

    # Inicializa la instancia y configura su estado interno.
    def __init__(
        self,
        pdf_path: Optional[str],
        session_dir: str,
        *,
        debug_visualization: bool = False,
        max_queue_size: int = 4,
    ):
        self.pdf_path = pdf_path
        self.session_dir = session_dir
        self.debug_visualization = bool(debug_visualization)

        self._request_queue: Queue = Queue(maxsize=max_queue_size)
        self._result_queue: Queue = Queue()
        self._shutdown_queue: Queue = Queue(maxsize=1)
        self._sentinel = object()
        self._errors: List[BaseException] = []
        self._error_lock = threading.Lock()
        self._closing = False
        self._closed = False

        self._thread = threading.Thread(
            target=self._run,
            name="projection-content-worker",
            daemon=True,
        )
        self._thread.start()

    # Encola una tarea de procesamiento si el worker sigue activo.
    def submit(
        self,
        generation: int,
        frame_number: int,
        fps: float,
        roi: np.ndarray,
    ) -> bool:
        self._raise_if_error()
        if self._closing or self._closed:
            return False
        if roi is None or roi.size == 0:
            return False

        task = ClassificationTask(
            generation=int(generation),
            frame_number=int(frame_number),
            fps=float(fps or 0.0),
            roi=roi.copy(),
        )

        while True:
            try:
                self._request_queue.put_nowait(task)
                return True
            except Full:
                try:
                    dropped = self._request_queue.get_nowait()
                except Empty:
                    continue
                else:
                    if dropped is self._sentinel:
                        self._request_queue.put_nowait(self._sentinel)
                        return False
                    self._request_queue.task_done()

    # Extrae resultados disponibles sin bloquear el hilo principal.
    def poll_results(self, max_items: int = 16) -> List[ClassificationResult]:
        self._raise_if_error()
        results: List[ClassificationResult] = []
        for _ in range(max_items):
            try:
                item = self._result_queue.get_nowait()
            except Empty:
                break
            results.append(item)
        return results

    # Finaliza el procesamiento pendiente y devuelve el resumen producido.
    def finalize(self) -> Dict:
        self._raise_if_error()
        if self._closed:
            try:
                return self._shutdown_queue.get_nowait()
            except Empty:
                return {}

        self._closing = True
        self._request_queue.put(self._sentinel)
        self._request_queue.join()
        self._thread.join()
        self._closed = True
        self._raise_if_error()

        try:
            return self._shutdown_queue.get_nowait()
        except Empty:
            return {}

    # Maneja run dentro de ProjectionContentWorker.
    def _run(self) -> None:
        classifier: Optional[ProjectionClassifier] = None
        try:
            classifier = ProjectionClassifier(
                self.pdf_path,
                debug_visualization=self.debug_visualization,
            )
            classifier.set_session_dir(self.session_dir)

            while True:
                task = self._request_queue.get()
                try:
                    if task is self._sentinel:
                        summary = classifier.finalize()
                        self._shutdown_queue.put(summary)
                        return

                    classification = classifier.classify_projection(
                        task.roi,
                        frame_number=task.frame_number,
                        fps=task.fps,
                    )
                    self._result_queue.put(
                        ClassificationResult(
                            generation=task.generation,
                            frame_number=task.frame_number,
                            classification=classification,
                        )
                    )
                finally:
                    self._request_queue.task_done()
        except BaseException as exc:
            self._record_error(exc)
        finally:
            if classifier is not None:
                try:
                    classifier.close()
                except BaseException as exc:
                    self._record_error(exc)

    # Registra error.
    def _record_error(self, exc: BaseException) -> None:
        with self._error_lock:
            self._errors.append(exc)

    # Maneja raise if error dentro de ProjectionContentWorker.
    def _raise_if_error(self) -> None:
        with self._error_lock:
            if not self._errors:
                return
            exc = self._errors[0]
        raise RuntimeError(f"ProjectionContentWorker falló: {exc}") from exc


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AsyncDiskWriter",
    "ClassificationResult",
    "ProjectionContentWorker",
]
