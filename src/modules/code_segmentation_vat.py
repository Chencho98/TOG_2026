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

# ============================================================================
# VIEWPORT SEGMENTATION
# ============================================================================

class CodeViewportSegmenter:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self, debug_visualization: bool = False):
        self.debug_visualization = bool(debug_visualization)

    # Maneja segment dentro de CodeViewportSegmenter.
    def segment(self, roi: np.ndarray) -> Optional[CodeViewportSegmentation]:
        if roi is None or roi.size == 0:
            return None

        text_mask, code_lines = self._build_masks(roi)
        sidebar, editor = self._detect_regions(text_mask, code_lines)
        editor = self._refine_editor_box(text_mask, code_lines, editor)

        sidebar_crop = self._crop_region(roi, sidebar)
        editor_crop = self._crop_region(roi, editor)
        overlay = np.zeros((0, 0, 3), dtype=np.uint8)
        debug_sheet = np.zeros((0, 0, 3), dtype=np.uint8)
        if self.debug_visualization:
            overlay = self._draw_overlay(roi.copy(), [sidebar, editor])
            debug_sheet = self._build_debug_sheet(roi, text_mask, code_lines, overlay)
        return CodeViewportSegmentation(
            sidebar=sidebar,
            editor=editor,
            sidebar_crop=sidebar_crop,
            editor_crop=editor_crop,
            overlay=overlay,
            debug_sheet=debug_sheet,
            text_mask=text_mask,
            code_lines=code_lines,
        )

    # Construye masks.
    def _build_masks(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, text_mask = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        text_mask = cv2.morphologyEx(
            text_mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
        )

        width = image.shape[1]
        code_kernel_width = clamp(width // 14, 40, max(56, width // 4))
        code_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (code_kernel_width, 1)
        )
        code_lines = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, code_kernel)

        bridge_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (clamp(width // 55, 10, 36), 1)
        )
        code_lines = cv2.morphologyEx(code_lines, cv2.MORPH_CLOSE, bridge_kernel)
        return text_mask, code_lines

    # Maneja find first sustained span dentro de CodeViewportSegmenter.
    def _find_first_sustained_span(
        self,
        signal: np.ndarray,
        threshold: float,
        min_run: int,
        start: int,
        end: int,
    ) -> Optional[Tuple[int, int]]:
        run_start = None
        for index in range(start, end):
            if signal[index] >= threshold:
                if run_start is None:
                    run_start = index
            else:
                if run_start is not None and (index - run_start) >= min_run:
                    return run_start, index
                run_start = None

        if run_start is not None and (end - run_start) >= min_run:
            return run_start, end
        return None

    # Maneja find last sustained span dentro de CodeViewportSegmenter.
    def _find_last_sustained_span(
        self,
        signal: np.ndarray,
        threshold: float,
        min_run: int,
        start: int,
        end: int,
    ) -> Optional[Tuple[int, int]]:
        run_end = None
        for index in range(end - 1, start - 1, -1):
            if signal[index] >= threshold:
                if run_end is None:
                    run_end = index + 1
            else:
                if run_end is not None and (run_end - (index + 1)) >= min_run:
                    return index + 1, run_end
                run_end = None

        if run_end is not None and (run_end - start) >= min_run:
            return start, run_end
        return None

    # Detecta editor start.
    def _detect_editor_start(self, text_mask: np.ndarray, code_lines: np.ndarray) -> int:
        _, width = text_mask.shape

        text_col = smooth_signal(
            text_mask.mean(axis=0).astype(np.float32) / 255.0,
            max(15, width // 70),
        )
        code_col = smooth_signal(
            code_lines.mean(axis=0).astype(np.float32) / 255.0,
            max(15, width // 70),
        )

        search_start = max(int(width * 0.28), 40)
        search_end = max(search_start + 1, int(width * 0.58))

        if float(code_col.max()) < 0.003:
            return int(width * 0.28)

        best_x = int(width * 0.25)
        best_score = float("-inf")
        left_window = max(36, width // 7)
        right_window = max(90, width // 3)
        valley_window = max(8, width // 110)

        for x in range(search_start, search_end):
            left0 = max(0, x - left_window)
            right1 = min(width, x + right_window)
            valley0 = max(0, x - valley_window)
            valley1 = min(width, x + valley_window)

            left_text = float(text_col[left0:x].mean()) if x > left0 else 0.0
            left_code = float(code_col[left0:x].mean()) if x > left0 else 0.0
            right_code = float(code_col[x:right1].mean()) if right1 > x else 0.0
            boundary_density = float(text_col[valley0:valley1].mean())

            score = (
                3.1 * right_code
                - 1.2 * left_code
                + 0.15 * left_text
                - 0.6 * boundary_density
            )

            if score > best_score:
                best_score = score
                best_x = x

        refine_radius = max(8, width // 80)
        refine_start = max(search_start, best_x - refine_radius)
        refine_end = min(search_end, best_x + refine_radius + 1)
        local_valley = text_col[refine_start:refine_end]
        if local_valley.size > 0:
            best_x = refine_start + int(np.argmin(local_valley))

        best_x += max(16, width // 90)
        return clamp(best_x, int(width * 0.14), int(width * 0.68))

    # Detecta regions.
    def _detect_regions(self, text_mask: np.ndarray, code_lines: np.ndarray) -> Tuple[Region, Region]:
        height, width = text_mask.shape
        editor_start = self._detect_editor_start(text_mask, code_lines)

        sidebar = Region(
            name="sidebar",
            x=0,
            y=0,
            w=editor_start,
            h=height,
            color=(70, 180, 255),
        )
        editor = Region(
            name="editor",
            x=editor_start,
            y=0,
            w=width - editor_start,
            h=height,
            color=(90, 220, 120),
        )
        return sidebar, editor

    # Maneja refine editor box dentro de CodeViewportSegmenter.
    def _refine_editor_box(
        self,
        text_mask: np.ndarray,
        code_lines: np.ndarray,
        editor: Region,
    ) -> Region:
        crop_text = text_mask[:, editor.x : editor.x + editor.w]
        crop_code = code_lines[:, editor.x : editor.x + editor.w]

        if crop_text.size == 0:
            return editor

        image_height = crop_text.shape[0]
        image_width = text_mask.shape[1]
        text_row = smooth_signal(
            crop_text.mean(axis=1).astype(np.float32) / 255.0,
            max(11, image_height // 70),
        )
        code_row = smooth_signal(
            crop_code.mean(axis=1).astype(np.float32) / 255.0,
            max(11, image_height // 70),
        )

        row_score = 0.75 * code_row + 0.25 * text_row
        threshold = max(0.003, float(np.percentile(row_score, 68) * 0.30))
        min_run = max(18, image_height // 20)

        top_span = self._find_first_sustained_span(
            row_score, threshold, min_run, 0, image_height
        )
        top = 0 if top_span is None else top_span[0]
        top = max(0, top - max(8, image_height // 90))
        if top > int(image_height * 0.25):
            top = 0

        cropped_rows_text = crop_text[top:, :]
        cropped_rows_code = crop_code[top:, :]
        if cropped_rows_text.size == 0 or cropped_rows_code.size == 0:
            return Region(
                name=editor.name,
                x=editor.x,
                y=top,
                w=editor.w,
                h=image_height - top,
                color=editor.color,
            )

        text_col = smooth_signal(
            cropped_rows_text.mean(axis=0).astype(np.float32) / 255.0,
            max(11, editor.w // 70),
        )
        code_col = smooth_signal(
            cropped_rows_code.mean(axis=0).astype(np.float32) / 255.0,
            max(11, editor.w // 70),
        )
        col_threshold = max(0.0025, float(np.percentile(code_col, 60) * 0.35))
        col_min_run = max(36, editor.w // 12)
        right_span = self._find_last_sustained_span(
            code_col, col_threshold, col_min_run, 0, editor.w
        )

        if right_span is None:
            right = image_width - max(80, image_width // 10)
        else:
            right = editor.x + right_span[1] + max(12, image_width // 120)

        edge_window = max(20, editor.w // 12)
        right_edge_activity = text_col[-edge_window:].mean()
        right_edge_code = code_col[-edge_window:].mean()
        if right_edge_activity > 0.006 and right_edge_code < col_threshold * 0.65:
            right = min(right, image_width - max(70, image_width // 11))

        right = clamp(right, editor.x + max(120, image_width // 5), image_width)

        return Region(
            name=editor.name,
            x=editor.x,
            y=top,
            w=right - editor.x,
            h=image_height - top,
            color=editor.color,
        )

    # Maneja crop region dentro de CodeViewportSegmenter.
    def _crop_region(self, image: np.ndarray, region: Region) -> np.ndarray:
        x0 = max(0, region.x)
        y0 = max(0, region.y)
        x1 = min(image.shape[1], region.x + region.w)
        y1 = min(image.shape[0], region.y + region.h)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        return image[y0:y1, x0:x1].copy()

    # Debug visualization helpers.
    # Maneja to bgr dentro de CodeViewportSegmenter.
    def _to_bgr(self, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.size == 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Maneja draw overlay dentro de CodeViewportSegmenter.
    def _draw_overlay(self, image: np.ndarray, regions: List[Region]) -> np.ndarray:
        overlay = image.copy()
        for region in regions:
            if region.w <= 0 or region.h <= 0:
                continue

            x, y, w, h = region.as_tuple()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), region.color, -1)
            cv2.rectangle(image, (x, y), (x + w, y + h), region.color, 2)
            cv2.putText(
                image,
                f"{region.name}: ({x}, {y}) {w}x{h}",
                (x + 8, max(24, y + 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                region.color,
                2,
                cv2.LINE_AA,
            )

        return cv2.addWeighted(overlay, 0.22, image, 0.78, 0.0)

    # Construye debug sheet.
    def _build_debug_sheet(
        self,
        image: np.ndarray,
        text_mask: np.ndarray,
        code_lines: np.ndarray,
        overlay: np.ndarray,
    ) -> np.ndarray:
        source = image.copy()
        panels = [source, self._to_bgr(text_mask), self._to_bgr(code_lines), overlay]
        labels = ["source", "text_mask", "code_lines", "overlay"]

        for label, panel in zip(labels, panels):
            if panel.size == 0:
                continue
            cv2.putText(
                panel,
                label,
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        top = np.hstack((panels[0], panels[1]))
        bottom = np.hstack((panels[2], panels[3]))
        return np.vstack((top, bottom))
# ============================================================================
# VISUAL CHANGE TRACKING
# ============================================================================

class VisualChangeTracker:
    # Inicializa la instancia y configura su estado interno.
    def __init__(
        self,
        change_threshold: float,
        edge_ratio_delta: float,
    ):
        self.change_threshold = float(change_threshold)
        self.edge_ratio_delta = float(edge_ratio_delta)
        self.last_sent_edges: Optional[np.ndarray] = None
        self.last_sent_hash: Optional[str] = None
        self.last_sent_edge_ratio: float = 0.0

    # Maneja evaluate dentro de VisualChangeTracker.
    def evaluate(self, sidebar_crop: np.ndarray, frame_number: int) -> Dict[str, Any]:
        edges = self._compute_edges(sidebar_crop)
        edge_ratio = 0.0
        if edges.size > 0:
            edge_ratio = float(cv2.countNonZero(edges)) / float(edges.size)
        digest = hashlib.sha1(edges.tobytes()).hexdigest()[:16] if edges.size > 0 else None

        if self.last_sent_edges is None or edges.size == 0:
            change_score = 1.0
            changed = bool(edges.size > 0)
            reason = "initial" if changed else "empty"
        else:
            diff = cv2.absdiff(edges, self.last_sent_edges)
            change_score = float(diff.mean()) / 255.0
            edge_delta = abs(edge_ratio - self.last_sent_edge_ratio)
            changed = (
                change_score >= self.change_threshold
                or edge_delta >= self.edge_ratio_delta
            )
            reason = "canny_delta" if changed else "stable"

        if changed and edges.size > 0:
            self.last_sent_edges = edges.copy()
            self.last_sent_hash = digest
            self.last_sent_edge_ratio = edge_ratio

        return {
            "frame_number": int(frame_number),
            "should_send": bool(changed),
            "reason": reason,
            "change_score": float(change_score),
            "edge_ratio": float(edge_ratio),
            "signature": digest,
            "edges": edges,
        }

    # Calcula edges.
    def _compute_edges(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        gray = cv2.resize(
            gray,
            (CODE_SIGNATURE_TARGET_W, CODE_SIGNATURE_TARGET_H),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 60, 160)
        return cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)
