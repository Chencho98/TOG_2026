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

# ============================================================================
# GEOMETRY HELPERS
# ============================================================================


# Traslada una caja por un desplazamiento dado.
def translate_bbox(bbox: Optional[List[int]], dx: int, dy: int) -> Optional[List[int]]:
    if bbox is None:
        return None
    return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]


# Construye una mascara que excluye regiones ocupadas por personas.
def build_person_exclusion_mask(shape, personas_list, padding: int = CORTINA_PERSON_PADDING) -> np.ndarray:
    mask = np.ones(shape[:2], dtype=np.uint8) * 255
    if not personas_list:
        return mask

    h, w = shape[:2]
    for persona in personas_list:
        bbox = persona["bbox"] if isinstance(persona, dict) else persona.bbox
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        mask[y1:y2, x1:x2] = 0
    return mask


# Recorta una caja para que permanezca dentro del frame.
def clip_bbox_to_frame(bbox, frame_shape):
    if bbox is None:
        return None

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w, int(x1)))
    y1 = max(0, min(h, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


# Expande una caja con margen y la recorta al frame.
def expand_bbox(bbox, frame_shape, pad_ratio: float = CORTINA_ROI_PAD_RATIO):
    clipped = clip_bbox_to_frame(bbox, frame_shape)
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    return clip_bbox_to_frame([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], frame_shape)


# Calcula la proporcion de interseccion entre dos cajas.
def bbox_intersection_ratio(bbox_a, bbox_b) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter_area / min(area_a, area_b)


# Indica si dos cajas se intersectan.
def bboxes_intersect(bbox_a, bbox_b) -> bool:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


# Convierte una mascara binaria en su caja envolvente.
def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


# Obtiene caja y mascara asociadas a una proyeccion.
def projection_geometry(projection):
    if projection is None:
        return None, None
    if isinstance(projection, dict):
        return projection.get("bbox"), projection.get("mask")
    return getattr(projection, "bbox", None), getattr(projection, "mask", None)


# Calcula cuanto de una caja esta cubierto por una mascara.
def mask_coverage_ratio(mask: np.ndarray, bbox) -> float:
    if mask is None or bbox is None:
        return 0.0

    clipped = clip_bbox_to_frame(bbox, mask.shape)
    if clipped is None:
        return 0.0

    x1, y1, x2, y2 = clipped
    area = max(1, (x2 - x1) * (y2 - y1))
    return float(np.count_nonzero(mask[y1:y2, x1:x2])) / float(area)


# Detecta si una persona corresponde al contenido proyectado.
def is_projected_person_detection(person_bbox, projection_bbox, projection_mask=None) -> bool:
    if projection_bbox is None or person_bbox is None:
        return False

    bx1, by1, bx2, by2 = person_bbox
    px1, py1, px2, py2 = projection_bbox
    proj_w = max(1, px2 - px1)
    proj_h = max(1, py2 - py1)
    side_margin = max(8, int(proj_w * 0.02))
    top_margin = max(8, int(proj_h * 0.02))
    bottom_margin = max(10, int(proj_h * 0.04))

    inside_projection_bbox = (
        bx1 >= (px1 - side_margin)
        and bx2 <= (px2 + side_margin)
        and by1 >= (py1 - top_margin)
        and by2 <= (py2 + bottom_margin)
    )
    if not inside_projection_bbox:
        return False

    if projection_mask is None:
        return True

    return mask_coverage_ratio(projection_mask, person_bbox) >= 0.88


# Filtra detecciones de persona que realmente son parte de la proyeccion.
def filter_projected_person_detections(persons, projection):
    if not persons:
        return persons

    projection_bbox, projection_mask = projection_geometry(projection)
    if projection_bbox is None:
        return persons

    filtered = []
    for person in persons:
        bbox = person["bbox"] if isinstance(person, dict) else person.bbox
        if is_projected_person_detection(bbox, projection_bbox, projection_mask):
            continue
        filtered.append(person)
    return filtered


# Convierte detecciones globales a coordenadas locales de un ROI.
def extract_local_personas(personas_list, roi_bbox):
    x1, y1, x2, y2 = roi_bbox
    local_personas = []
    for persona in personas_list:
        bbox = persona["bbox"] if isinstance(persona, dict) else persona.bbox
        local_bbox = translate_bbox(bbox, -x1, -y1)
        if local_bbox[0] >= (x2 - x1) or local_bbox[2] <= 0 or local_bbox[1] >= (y2 - y1) or local_bbox[3] <= 0:
            continue
        local_personas.append(
            {
                "bbox": [
                    max(0, local_bbox[0]),
                    max(0, local_bbox[1]),
                    min(x2 - x1, local_bbox[2]),
                    min(y2 - y1, local_bbox[3]),
                ]
            }
        )
    return local_personas


# Agrupa lineas verticales cercanas en candidatos de borde.
def merge_vertical_line_groups(lines, x_tolerance: int = CORTINA_LINE_X_MERGE_PX):
    if not lines:
        return []

    groups = []
    for line in sorted(lines, key=lambda item: item["x"]):
        if not groups or abs(line["x"] - groups[-1]["x"]) > x_tolerance:
            groups.append(
                {
                    "x": int(line["x"]),
                    "top": int(line["top"]),
                    "bottom": int(line["bottom"]),
                    "count": 1,
                    "score": float(line["length"]),
                    "points": [tuple(line["p1"]), tuple(line["p2"])],
                }
            )
            continue

        group = groups[-1]
        new_count = group["count"] + 1
        group["x"] = int(round((group["x"] * group["count"] + line["x"]) / float(new_count)))
        group["top"] = min(group["top"], int(line["top"]))
        group["bottom"] = max(group["bottom"], int(line["bottom"]))
        group["count"] = new_count
        group["score"] += float(line["length"])
        group["points"].extend((tuple(line["p1"]), tuple(line["p2"])))

    for group in groups:
        group["length"] = int(group["bottom"] - group["top"])
        group["score"] += 10.0 * max(0, group["count"] - 1)
        line_model = fit_vertical_line_model(group.pop("points", []))
        group["line_model"] = line_model
        if line_model is not None:
            mid_y = 0.5 * (group["top"] + group["bottom"])
            group["top_x"] = int(round(line_model_x_at_y(line_model, group["top"])))
            group["bottom_x"] = int(round(line_model_x_at_y(line_model, group["bottom"])))
            group["x"] = int(round(line_model_x_at_y(line_model, mid_y)))
        else:
            group["top_x"] = int(group["x"])
            group["bottom_x"] = int(group["x"])
    return groups


# Calcula los coeficientes de una recta a partir de dos puntos.
def line_coeffs_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = float(y1 - y2)
    b = float(x2 - x1)
    c = float((x1 * y2) - (x2 * y1))
    norm = np.hypot(a, b)
    if norm < 1e-6:
        return None
    return (a / norm, b / norm, c / norm)


# Calcula el punto de interseccion entre dos rectas.
def intersect_lines(line_a, line_b):
    if line_a is None or line_b is None:
        return None
    a1, b1, c1 = line_a
    a2, b2, c2 = line_b
    det = (a1 * b2) - (a2 * b1)
    if abs(det) < 1e-6:
        return None

    x = ((b1 * c2) - (b2 * c1)) / det
    y = ((c1 * a2) - (c2 * a1)) / det
    return [float(x), float(y)]


# Ajusta un modelo vertical para un conjunto de puntos.
def fit_vertical_line_model(points):
    if points is None or len(points) < 2:
        return None

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    try:
        fitted = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    except cv2.error:
        return None

    vx, vy, x0, y0 = [float(value) for value in fitted.reshape(-1)]
    if abs(vy) < 1e-6:
        return None
    return {"vx": vx, "vy": vy, "x0": x0, "y0": y0}


# Evalua la posicion X de un modelo de linea para una coordenada Y.
def line_model_x_at_y(line_model, y_value):
    if line_model is None:
        return None
    return float(line_model["x0"] + (float(y_value) - line_model["y0"]) * (line_model["vx"] / line_model["vy"]))


# Evalua la posicion X representativa de un grupo de lineas.
def group_line_x_at_y(group, y_value) -> float:
    x_value = line_model_x_at_y(group.get("line_model"), y_value)
    if x_value is not None:
        return float(x_value)
    return float(group.get("x", 0.0))


# Evalua el borde izquierdo o derecho de una pareja de lineas.
def line_pair_side_x_at_y(line_pair, side, y_value) -> float:
    base_key = f"{side}_x"
    top_key = f"{side}_top_x"
    bottom_key = f"{side}_bottom_x"
    x_top = float(line_pair.get(top_key, line_pair.get(base_key, 0.0)))
    x_bottom = float(line_pair.get(bottom_key, line_pair.get(base_key, 0.0)))
    y_top = float(line_pair.get("top_y", 0.0))
    y_bottom = float(line_pair.get("bottom_y", y_top))
    if abs(y_bottom - y_top) < 1e-6:
        return x_bottom
    ratio = (float(y_value) - y_top) / (y_bottom - y_top)
    return x_top + ratio * (x_bottom - x_top)


# Construye un corredor alrededor de un borde de cortina.
def build_line_corridor_bbox(line_pair, side, frame_shape, padding: int = CORTINA_PERSON_PADDING):
    frame_h, frame_w = frame_shape[:2]
    top_y = max(0, min(frame_h, int(round(line_pair.get("top_y", 0)))))
    bottom_y = max(top_y + 1, min(frame_h, int(round(line_pair.get("bottom_y", top_y + 1)))))
    top_x = line_pair_side_x_at_y(line_pair, side, top_y)
    bottom_x = line_pair_side_x_at_y(line_pair, side, bottom_y)
    x1 = int(np.floor(min(top_x, bottom_x) - padding))
    x2 = int(np.ceil(max(top_x, bottom_x) + padding))
    return clip_bbox_to_frame([x1, top_y, x2, bottom_y], frame_shape)


# Detecta si una persona tapa los bordes esperados de la cortina.
def person_occludes_curtain_lines(personas_list, line_pair, frame_shape) -> bool:
    if line_pair is None or not personas_list:
        return False

    left_corridor = build_line_corridor_bbox(line_pair, "left", frame_shape)
    right_corridor = build_line_corridor_bbox(line_pair, "right", frame_shape)
    if left_corridor is None and right_corridor is None:
        return False

    for persona in personas_list:
        bbox = persona["bbox"] if isinstance(persona, dict) else persona.bbox
        if (left_corridor is not None and bboxes_intersect(bbox, left_corridor)) or (
            right_corridor is not None and bboxes_intersect(bbox, right_corridor)
        ):
            return True
    return False


# Ajusta un punto para que permanezca dentro del ROI.
def clip_point_to_roi(point, width, height):
    return [
        float(max(0.0, min(float(width - 1), point[0]))),
        float(max(0.0, min(float(height - 1), point[1]))),
    ]


# Contrae un poligono moviendo sus puntos hacia el centro.
def inset_polygon_towards_center(points, inset_px: int):
    if not points or inset_px <= 0:
        return points

    pts = np.array(points, dtype=np.float32)
    center = np.mean(pts, axis=0)
    adjusted = []
    for pt in pts:
        vec = center - pt
        dist = float(np.hypot(vec[0], vec[1]))
        if dist < 1e-6:
            adjusted.append(pt.tolist())
            continue
        step = min(float(inset_px), dist * 0.45)
        adjusted.append((pt + (vec / dist) * step).tolist())
    return adjusted


# Calcula la caja envolvente de un poligono.
def polygon_to_bbox(points) -> List[int]:
    pts = np.array(points, dtype=np.float32)
    x1 = int(np.floor(np.min(pts[:, 0])))
    y1 = int(np.floor(np.min(pts[:, 1])))
    x2 = int(np.ceil(np.max(pts[:, 0]))) + 1
    y2 = int(np.ceil(np.max(pts[:, 1]))) + 1
    return [x1, y1, x2, y2]
