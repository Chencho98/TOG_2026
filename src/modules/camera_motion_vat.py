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

# ============================================================================
# CAMERA MOTION
# ============================================================================


class CameraMotionDetector:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self, threshold: float = 15.0):
        self.prev_gray = None
        self.threshold = threshold
        self.motion_counter = 0
        self.required_consecutive_frames = 2

        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=15,
            blockSize=7,
        )
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    # Estima movimiento de camara comparando puntos entre frames.
    def detect_motion(
        self,
        frame: np.ndarray,
        personas_list: Optional[List[PersonDetection]] = None,
    ) -> tuple[bool, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return False, 0.0

        h, w = gray.shape
        mask = np.ones(gray.shape, dtype=np.uint8) * 255

        border_margin = int(min(h, w) * 0.15)
        mask[:border_margin, :] = 0
        mask[-border_margin:, :] = 0
        mask[:, :border_margin] = 0
        mask[:, -border_margin:] = 0

        if personas_list:
            for persona in personas_list:
                x1, y1, x2, y2 = persona.bbox
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                mask[y1:y2, x1:x2] = 0

        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
        if p0 is None or len(p0) < 30:
            self.prev_gray = gray.copy()
            self.motion_counter = 0
            return False, 0.0

        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
        if p1 is None or st is None:
            self.prev_gray = gray.copy()
            self.motion_counter = 0
            return False, 0.0

        good_new = p1[st == 1]
        good_old = p0[st == 1]
        if len(good_new) < 30:
            self.prev_gray = gray.copy()
            self.motion_counter = 0
            return False, 0.0

        displacements = np.sqrt(((good_new - good_old) ** 2).sum(axis=1))

        q25 = np.percentile(displacements, 25)
        q75 = np.percentile(displacements, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        valid_mask = (displacements >= max(0, lower_bound)) & (displacements <= upper_bound)
        valid_displacements = displacements[valid_mask]

        if len(valid_displacements) < 20:
            self.prev_gray = gray.copy()
            self.motion_counter = 0
            return False, 0.0

        median_disp = float(np.median(valid_displacements))
        vectors = good_new - good_old
        vectors_valid = vectors[valid_mask]
        avg_direction = np.mean(vectors_valid, axis=0)
        avg_direction_norm = np.linalg.norm(avg_direction)

        if avg_direction_norm > 0.1:
            dots = np.dot(vectors_valid, avg_direction) / (
                np.linalg.norm(vectors_valid, axis=1) * avg_direction_norm
            )
            consistent_ratio = float(np.sum(dots > 0.7) / len(dots))
        else:
            consistent_ratio = 0.0

        is_camera_motion = (median_disp > self.threshold) and (consistent_ratio > 0.6)
        self.prev_gray = gray.copy()

        if is_camera_motion:
            self.motion_counter += 1
        else:
            self.motion_counter = 0

        return self.motion_counter >= self.required_consecutive_frames, median_disp
