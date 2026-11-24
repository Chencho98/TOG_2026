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
# WHITEBOARD GEOMETRY
# ============================================================================


# Refina la geometria del pizarron usando bordes Canny.
def refine_whiteboard_geometry_with_canny(frame, bbox):
    roi_bbox = clip_bbox_to_frame(bbox, frame.shape)
    if roi_bbox is None:
        return {"bbox": bbox, "mask": None, "polygon": None}

    x1, y1, x2, y2 = roi_bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {"bbox": roi_bbox, "mask": None, "polygon": None}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CORTINA_CANNY_LOW, CORTINA_CANNY_HIGH)

    roi_h, roi_w = roi.shape[:2]
    min_len = max(25, int(min(roi_h, roi_w) * WHITEBOARD_REFINE_MIN_LEN_RATIO))
    hough_threshold = max(20, int(min(roi_h, roi_w) * 0.10))
    max_gap = max(10, int(min(roi_h, roi_w) * 0.05))

    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    if raw_lines is None:
        return {"bbox": roi_bbox, "mask": None, "polygon": None}

    side_band = max(10, int(roi_w * WHITEBOARD_REFINE_MARGIN_RATIO))
    top_band = max(10, int(roi_h * WHITEBOARD_REFINE_MARGIN_RATIO))
    min_vertical_span = max(20, int(roi_h * WHITEBOARD_REFINE_MIN_VSPAN_RATIO))
    min_horizontal_span = max(20, int(roi_w * WHITEBOARD_REFINE_MIN_HSPAN_RATIO))
    top_cover_limit = int(roi_h * WHITEBOARD_REFINE_TOP_COVER_RATIO)
    bot_cover_limit = int(roi_h * WHITEBOARD_REFINE_BOT_COVER_RATIO)

    left_candidates = []
    right_candidates = []
    top_candidates = []
    bottom_candidates = []

    for raw in raw_lines[:, 0, :]:
        lx1, ly1, lx2, ly2 = map(int, raw)
        dx = lx2 - lx1
        dy = ly2 - ly1
        angle = abs(np.degrees(np.arctan2(float(dy), float(dx) if dx != 0 else 1e-6)))
        if angle > 90.0:
            angle = 180.0 - angle

        x_min = min(lx1, lx2)
        x_max = max(lx1, lx2)
        y_min = min(ly1, ly2)
        y_max = max(ly1, ly2)

        if angle >= (90.0 - WHITEBOARD_REFINE_ANGLE_TOL_DEG):
            span_y = y_max - y_min
            if span_y < min_vertical_span:
                continue
            if y_min > top_cover_limit or y_max < bot_cover_limit:
                continue

            if x_max <= side_band:
                left_candidates.append(
                    {
                        "inner_x": int(x_max),
                        "span": int(span_y),
                        "p1": [int(lx1), int(ly1)],
                        "p2": [int(lx2), int(ly2)],
                    }
                )
            if x_min >= (roi_w - side_band):
                right_candidates.append(
                    {
                        "inner_x": int(x_min),
                        "span": int(span_y),
                        "p1": [int(lx1), int(ly1)],
                        "p2": [int(lx2), int(ly2)],
                    }
                )
        elif angle <= WHITEBOARD_REFINE_ANGLE_TOL_DEG:
            span_x = x_max - x_min
            if span_x < min_horizontal_span:
                continue
            if x_min > int(roi_w * 0.35) or x_max < int(roi_w * 0.65):
                continue

            if y_max <= top_band:
                top_candidates.append(
                    {
                        "inner_y": int(y_max),
                        "span": int(span_x),
                        "p1": [int(lx1), int(ly1)],
                        "p2": [int(lx2), int(ly2)],
                    }
                )
            if y_min >= (roi_h - top_band):
                bottom_candidates.append(
                    {
                        "inner_y": int(y_min),
                        "span": int(span_x),
                        "p1": [int(lx1), int(ly1)],
                        "p2": [int(lx2), int(ly2)],
                    }
                )

    refined_x1 = 0
    refined_x2 = roi_w
    refined_y1 = 0
    refined_y2 = roi_h

    if left_candidates:
        left_line = max(left_candidates, key=lambda item: (item["inner_x"], item["span"]))
        refined_x1 = min(roi_w - 1, left_line["inner_x"])
    else:
        left_line = None

    if right_candidates:
        right_line = min(right_candidates, key=lambda item: (item["inner_x"], -item["span"]))
        refined_x2 = max(1, right_line["inner_x"])
    else:
        right_line = None

    if top_candidates:
        top_line = max(top_candidates, key=lambda item: (item["inner_y"], item["span"]))
        refined_y1 = min(roi_h - 1, top_line["inner_y"])
    else:
        top_line = None

    if bottom_candidates:
        bottom_line = min(bottom_candidates, key=lambda item: (item["inner_y"], -item["span"]))
        refined_y2 = max(1, bottom_line["inner_y"])
    else:
        bottom_line = None

    refined_bbox = clip_bbox_to_frame(
        [x1 + refined_x1, y1 + refined_y1, x1 + refined_x2, y1 + refined_y2],
        frame.shape,
    )
    if refined_bbox is None:
        return {"bbox": roi_bbox, "mask": None, "polygon": None}

    refined_w = refined_bbox[2] - refined_bbox[0]
    refined_h = refined_bbox[3] - refined_bbox[1]
    if refined_w < int((x2 - x1) * 0.45) or refined_h < int((y2 - y1) * 0.35):
        return {"bbox": roi_bbox, "mask": None, "polygon": None}

    left_line_coeffs = (
        line_coeffs_from_points(left_line["p1"], left_line["p2"])
        if left_line is not None
        else line_coeffs_from_points([refined_x1, 0], [refined_x1, roi_h - 1])
    )
    right_line_coeffs = (
        line_coeffs_from_points(right_line["p1"], right_line["p2"])
        if right_line is not None
        else line_coeffs_from_points([refined_x2 - 1, 0], [refined_x2 - 1, roi_h - 1])
    )
    top_line_coeffs = (
        line_coeffs_from_points(top_line["p1"], top_line["p2"])
        if top_line is not None
        else line_coeffs_from_points([0, refined_y1], [roi_w - 1, refined_y1])
    )
    bottom_line_coeffs = (
        line_coeffs_from_points(bottom_line["p1"], bottom_line["p2"])
        if bottom_line is not None
        else line_coeffs_from_points([0, refined_y2 - 1], [roi_w - 1, refined_y2 - 1])
    )

    local_points = [
        intersect_lines(left_line_coeffs, top_line_coeffs),
        intersect_lines(right_line_coeffs, top_line_coeffs),
        intersect_lines(right_line_coeffs, bottom_line_coeffs),
        intersect_lines(left_line_coeffs, bottom_line_coeffs),
    ]
    if any(point is None for point in local_points):
        return {"bbox": refined_bbox, "mask": None, "polygon": None}

    local_points = [clip_point_to_roi(point, roi_w, roi_h) for point in local_points]
    local_points = inset_polygon_towards_center(local_points, WHITEBOARD_REFINE_INSET_PX)
    polygon = [[int(round(x1 + pt[0])), int(round(y1 + pt[1]))] for pt in local_points]
    polygon_bbox = clip_bbox_to_frame(polygon_to_bbox(polygon), frame.shape)
    if polygon_bbox is None:
        return {"bbox": refined_bbox, "mask": None, "polygon": None}

    polygon_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(polygon_mask, np.array(polygon, dtype=np.int32), 255)

    return {"bbox": polygon_bbox, "mask": polygon_mask, "polygon": polygon}
