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

try:
    from src.modules.geometry_vat import *
except ImportError:
    try:
        from geometry_vat import *
    except ImportError:
        from .geometry_vat import *

# ============================================================================
# CURTAIN DETECTION
# ============================================================================


@dataclass
class CurtainTracker:
    whiteboard_bbox: Optional[list] = None
    line_pair: Optional[dict] = None
    active_projection: Optional[dict] = None
    line_present_since: Optional[int] = None
    line_missing_since: Optional[int] = None
    line_occluded_by_person: bool = False

    # Reinicia el estado interno para comenzar una nueva medicion.
    def reset(self) -> None:
        self.whiteboard_bbox = None
        self.line_pair = None
        self.active_projection = None
        self.line_present_since = None
        self.line_missing_since = None
        self.line_occluded_by_person = False


# Detecta los dos bordes verticales compatibles con una cortina.
def detect_curtain_line_pair(frame, whiteboard_bbox, personas_list):
    roi_bbox = expand_bbox(whiteboard_bbox, frame.shape, pad_ratio=CORTINA_ROI_PAD_RATIO)
    if roi_bbox is None:
        return None

    x1, y1, x2, y2 = roi_bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    local_personas = extract_local_personas(personas_list, roi_bbox)
    valid_mask = build_person_exclusion_mask(roi.shape, local_personas)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CORTINA_CANNY_LOW, CORTINA_CANNY_HIGH)
    edges = cv2.bitwise_and(edges, valid_mask)

    roi_h, roi_w = roi.shape[:2]
    min_len = max(20, int(roi_h * CORTINA_LINE_MIN_LEN_RATIO))
    hough_threshold = max(18, int(roi_h * 0.12))
    max_gap = max(10, int(roi_h * 0.08))

    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    vertical_lines = []
    if raw_lines is not None:
        for raw in raw_lines[:, 0, :]:
            lx1, ly1, lx2, ly2 = map(int, raw)
            dx = lx2 - lx1
            dy = ly2 - ly1
            length = abs(dy)
            if length < min_len:
                continue
            if abs(dx) > max(6, int(length * CORTINA_LINE_VERTICAL_DX_RATIO)):
                continue

            top = min(ly1, ly2)
            bottom = max(ly1, ly2)
            if top > int(roi_h * CORTINA_LINE_TOP_MAX_RATIO):
                continue

            vertical_lines.append(
                {
                    "x": int(round((lx1 + lx2) * 0.5)),
                    "top": int(top),
                    "bottom": int(bottom),
                    "length": int(length),
                    "p1": [int(lx1), int(ly1)],
                    "p2": [int(lx2), int(ly2)],
                }
            )

    line_groups_local = merge_vertical_line_groups(vertical_lines)
    min_width = int(roi_w * CORTINA_LINE_PAIR_MIN_WIDTH_RATIO)
    max_width = int(roi_w * CORTINA_LINE_PAIR_MAX_WIDTH_RATIO)
    best_pair = None

    for idx, left in enumerate(line_groups_local):
        if left["length"] < min_len:
            continue

        for right in line_groups_local[idx + 1 :]:
            if right["length"] < min_len:
                continue

            overlap_top = max(left["top"], right["top"])
            overlap_bottom = min(left["bottom"], right["bottom"])
            overlap = overlap_bottom - overlap_top
            mid_y = 0.5 * (overlap_top + overlap_bottom)
            width = int(round(group_line_x_at_y(right, mid_y) - group_line_x_at_y(left, mid_y)))
            if width < min_width or width > max_width:
                continue

            if overlap < int(min(left["length"], right["length"]) * CORTINA_LINE_MIN_OVERLAP_RATIO):
                continue

            score = left["score"] + right["score"] + float(overlap)
            score -= 0.5 * abs(left["top"] - right["top"])

            if best_pair is None or score > best_pair["score"]:
                pair_top = min(left["top"], right["top"])
                pair_bottom = max(left["bottom"], right["bottom"])
                best_pair = {
                    "left_x": int(round(x1 + group_line_x_at_y(left, mid_y))),
                    "right_x": int(round(x1 + group_line_x_at_y(right, mid_y))),
                    "left_top_x": int(round(x1 + group_line_x_at_y(left, pair_top))),
                    "right_top_x": int(round(x1 + group_line_x_at_y(right, pair_top))),
                    "left_bottom_x": int(round(x1 + group_line_x_at_y(left, pair_bottom))),
                    "right_bottom_x": int(round(x1 + group_line_x_at_y(right, pair_bottom))),
                    "top_y": y1 + pair_top,
                    "bottom_y": y1 + pair_bottom,
                    "width": int(width),
                    "score": float(score),
                }
    return best_pair


# Construye la region de proyeccion a partir de bordes de cortina.
def build_curtain_projection_from_lines(frame_shape, line_pair):
    if line_pair is None:
        return None

    frame_h, frame_w = frame_shape[:2]
    top_y = 0
    bottom_y = max(1, min(frame_h - 1, int(round(line_pair["bottom_y"]))))
    left_top_x = int(round(line_pair_side_x_at_y(line_pair, "left", top_y)))
    right_top_x = int(round(line_pair_side_x_at_y(line_pair, "right", top_y)))
    left_bottom_x = int(round(line_pair_side_x_at_y(line_pair, "left", bottom_y)))
    right_bottom_x = int(round(line_pair_side_x_at_y(line_pair, "right", bottom_y)))

    polygon = [
        [max(0, min(frame_w - 1, left_top_x)), top_y],
        [max(0, min(frame_w - 1, right_top_x)), top_y],
        [max(0, min(frame_w - 1, right_bottom_x)), bottom_y],
        [max(0, min(frame_w - 1, left_bottom_x)), bottom_y],
    ]

    if polygon[0][0] >= polygon[1][0] or polygon[3][0] >= polygon[2][0]:
        return None

    curtain_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(curtain_mask, np.array(polygon, dtype=np.int32), 255)
    curtain_bbox = mask_to_bbox(curtain_mask)
    if curtain_bbox is None:
        return None

    return {
        "bbox": curtain_bbox,
        "mask": curtain_mask,
        "polygon": polygon,
        "conf": float(line_pair.get("score", 1.0)),
        "type": "cortina_canny_polygon",
        "is_stable": True,
        "moving_down": False,
        "is_observed": True,
    }


# Refina la mascara de cortina usando umbralizacion Otsu.
def refine_curtain_projection_with_otsu(frame, projection):
    if projection is None:
        return None

    coarse_mask = projection.get("mask")
    coarse_bbox = clip_bbox_to_frame(projection.get("bbox"), frame.shape)
    if coarse_mask is None or coarse_bbox is None:
        return projection

    x1, y1, x2, y2 = coarse_bbox
    roi = frame[y1:y2, x1:x2]
    local_mask = coarse_mask[y1:y2, x1:x2]
    if roi.size == 0 or local_mask.size == 0:
        return projection

    coarse_area = float(np.count_nonzero(local_mask))
    if coarse_area < 100.0:
        return projection

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    masked_pixels = gray[local_mask > 0]
    if masked_pixels.size < 64:
        return projection

    otsu_threshold, _ = cv2.threshold(
        masked_pixels.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    _, otsu = cv2.threshold(gray, otsu_threshold, 255, cv2.THRESH_BINARY)
    otsu = cv2.bitwise_and(otsu, local_mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, close_kernel)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return projection

    top_limit = int(otsu.shape[0] * CORTINA_OTSU_TOP_MAX_RATIO)
    best_contour = None
    best_score = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < (coarse_area * CORTINA_OTSU_MIN_AREA_RATIO):
            continue

        _, top_y, _, _ = cv2.boundingRect(contour)
        if top_y > top_limit:
            continue

        perimeter = cv2.arcLength(contour, True)
        score = float(area) - (0.20 * float(perimeter))
        if best_contour is None or score > best_score:
            best_contour = contour
            best_score = score

    if best_contour is None:
        return projection

    refined_local_mask = np.zeros_like(local_mask)
    cv2.drawContours(refined_local_mask, [best_contour], -1, 255, thickness=-1)
    refined_local_mask = cv2.bitwise_and(refined_local_mask, local_mask)

    refined_area = float(np.count_nonzero(refined_local_mask))
    if refined_area < (coarse_area * CORTINA_OTSU_MIN_AREA_RATIO):
        return projection

    refined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    refined_mask[y1:y2, x1:x2] = refined_local_mask
    refined_bbox = mask_to_bbox(refined_mask)
    if refined_bbox is None:
        return projection

    hull = cv2.convexHull(best_contour)
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, max(2.0, 0.015 * perimeter), True)
    polygon = projection.get("polygon")
    if len(approx) >= 4:
        polygon = [[int(x1 + pt[0][0]), int(y1 + pt[0][1])] for pt in approx]

    refined_projection = dict(projection)
    refined_projection["bbox"] = refined_bbox
    refined_projection["mask"] = refined_mask
    refined_projection["polygon"] = polygon
    refined_projection["type"] = "cortina_canny_otsu_polygon"
    return refined_projection


# Actualiza la deteccion y estado temporal de la cortina.
def detect_cortina(frame, personas_list, tracker, whiteboard_bbox=None, frame_index: int = 0, fps: float = 16.0):
    fps = max(1.0, float(fps or 16.0))
    confirm_frames = max(1, int(round(CORTINA_LINE_CONFIRM_SECONDS * fps)))
    clear_frames = max(1, int(round(CORTINA_LINE_CLEAR_SECONDS * fps)))

    if whiteboard_bbox is None:
        tracker.reset()
        return None

    whiteboard_bbox = clip_bbox_to_frame(whiteboard_bbox, frame.shape)
    if whiteboard_bbox is None:
        tracker.reset()
        return None

    prev_whiteboard_bbox = tracker.whiteboard_bbox
    if prev_whiteboard_bbox is not None and bbox_intersection_ratio(prev_whiteboard_bbox, whiteboard_bbox) < 0.70:
        tracker.reset()
    tracker.whiteboard_bbox = whiteboard_bbox

    detected_line_pair = detect_curtain_line_pair(frame, whiteboard_bbox, personas_list)
    previous_line_pair = tracker.line_pair
    line_pair = detected_line_pair
    line_occluded_by_person = False
    if line_pair is None and previous_line_pair is not None and person_occludes_curtain_lines(personas_list, previous_line_pair, frame.shape):
        line_pair = previous_line_pair
        line_occluded_by_person = True
        if not tracker.line_occluded_by_person:
            print("🚶 Persona ocluye bordes de cortina. Conservando la última geometría.")
    elif detected_line_pair is not None and tracker.line_occluded_by_person:
        print("👀 Bordes de cortina visibles de nuevo.")

    tracker.line_occluded_by_person = line_occluded_by_person
    active_projection = tracker.active_projection

    if line_pair is not None:
        tracker.line_pair = line_pair
        tracker.line_missing_since = None
        if tracker.line_present_since is None:
            tracker.line_present_since = frame_index
            print("👀 Líneas verticales compatibles con cortina detectadas.")

        line_age = frame_index - tracker.line_present_since
        if active_projection is None and line_age >= confirm_frames:
            frozen_projection = build_curtain_projection_from_lines(frame.shape, line_pair)
            if frozen_projection is not None:
                frozen_projection = refine_curtain_projection_with_otsu(frame, frozen_projection)
                print("✅ Cortina confirmada por persistencia de líneas.")
                tracker.active_projection = frozen_projection
                active_projection = frozen_projection
            else:
                print("⚠️ Líneas de cortina estables, pero no se pudo construir la región.")

        if active_projection is not None:
            projection = dict(active_projection)
            projection["is_observed"] = True
            projection["is_stable"] = True
            projection["moving_down"] = False
            return projection
    else:
        tracker.line_pair = None
        tracker.line_present_since = None

    if active_projection is not None and line_pair is None:
        if tracker.line_missing_since is None:
            tracker.line_missing_since = frame_index
            print("⌛ Líneas de cortina perdidas. Esperando antes de ocultar la región.")

        missing_age = frame_index - tracker.line_missing_since
        if missing_age >= clear_frames:
            print("⬆️ Cortina removida: desaparecieron las líneas laterales.")
            tracker.reset()
        else:
            projection = dict(active_projection)
            projection["is_observed"] = False
            projection["is_stable"] = True
            projection["moving_down"] = False
            projection["ttl_frames"] = clear_frames
            return projection
    return None
