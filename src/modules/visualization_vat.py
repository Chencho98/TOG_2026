"""
Shared imports for the classroom video processor modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import time

import cv2
import numpy as np

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
    from src.modules.project_config_vat import *
except ImportError:
    try:
        from project_config_vat import *
    except ImportError:
        from .project_config_vat import *

# ============================================================================
# VISUALIZATION / DEBUG DISPLAY HELPERS
# ============================================================================


# Dibuja una etiqueta legible sobre el frame mostrado.
def draw_label(frame: np.ndarray, text: str, x1: int, y1: int, color: tuple) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x1 + 2, ty), font, 0.6, color, 2)


# Redimensiona un frame para caber dentro del area objetivo.
def resize_frame_to_fit(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame

    scale_w = float(target_w) / float(width)
    scale_h = float(target_h) / float(height)
    scale = min(1.0, scale_w, scale_h)
    if scale >= 0.999:
        return frame.copy()

    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)


# Maneja scale bbox.
def _scale_bbox(bbox: List[int], scale_x: float, scale_y: float) -> List[int]:
    return [
        int(round(bbox[0] * scale_x)),
        int(round(bbox[1] * scale_y)),
        int(round(bbox[2] * scale_x)),
        int(round(bbox[3] * scale_y)),
    ]


# Maneja scale polygon.
def _scale_polygon(points: List[List[int]], scale_x: float, scale_y: float) -> np.ndarray:
    scaled_points = [
        [int(round(point[0] * scale_x)), int(round(point[1] * scale_y))]
        for point in points
    ]
    return np.array(scaled_points, dtype=np.int32)


# Dibuja regiones, personas y estado sobre el frame de debug.
def annotate_frame(
    frame: np.ndarray,
    whiteboard_obj: Optional[WhiteboardRegion],
    projection_obj: Optional[ProjectionRegion],
    personas_list: List[PersonDetection],
    display_target: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    if display_target is not None:
        target_w, target_h = display_target
        annotated = resize_frame_to_fit(frame, target_w, target_h)
    else:
        annotated = frame.copy()

    original_h, original_w = frame.shape[:2]
    annotated_h, annotated_w = annotated.shape[:2]
    scale_x = 1.0 if original_w <= 0 else float(annotated_w) / float(original_w)
    scale_y = 1.0 if original_h <= 0 else float(annotated_h) / float(original_h)

    if whiteboard_obj:
        x1, y1, x2, y2 = _scale_bbox(whiteboard_obj.bbox, scale_x, scale_y)
        if whiteboard_obj.polygon:
            cv2.polylines(
                annotated,
                [_scale_polygon(whiteboard_obj.polygon, scale_x, scale_y)],
                True,
                COLOR_WHITEBOARD,
                2,
            )
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_WHITEBOARD, 2)
        draw_label(annotated, f"PIZARRON {whiteboard_obj.confidence:.2f}", x1, y1, COLOR_WHITEBOARD)

    if projection_obj and projection_obj.is_active:
        x1, y1, x2, y2 = _scale_bbox(projection_obj.bbox, scale_x, scale_y)
        if projection_obj.polygon:
            cv2.polylines(
                annotated,
                [_scale_polygon(projection_obj.polygon, scale_x, scale_y)],
                True,
                COLOR_PROJECTION,
                2,
            )
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_PROJECTION, 2)

        label = f"{projection_obj.projection_type} {projection_obj.confidence:.2f}"
        if projection_obj.context_state:
            label = projection_obj.context_state
            if projection_obj.matched_page_number is not None:
                label += f" P{projection_obj.matched_page_number}"
            if projection_obj.code_language:
                label += f" {projection_obj.code_language}"
            if projection_obj.context_confidence > 0.0:
                label += f" {projection_obj.context_confidence:.2f}"
        draw_label(annotated, label, x1, y1, COLOR_PROJECTION)

    for index, persona in enumerate(personas_list):
        x1, y1, x2, y2 = _scale_bbox(persona.bbox, scale_x, scale_y)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_PERSON, 2)
        draw_label(annotated, f"PERSONA {index + 1}", x1, y1, COLOR_PERSON)

    return annotated
