"""
Compatibility facade for the modular code reconstruction pipeline.
"""

from __future__ import annotations

try:
    from src.modules.code_config_vat import *
except ImportError:
    try:
        from code_config_vat import *
    except ImportError:
        from .code_config_vat import *

try:
    from src.modules.code_utils_vat import configure_tesseract_command, downscale_projection_roi
except ImportError:
    try:
        from code_utils_vat import configure_tesseract_command, downscale_projection_roi
    except ImportError:
        from .code_utils_vat import configure_tesseract_command, downscale_projection_roi

try:
    from src.modules.code_classifier_vat import CodeClassifier
except ImportError:
    try:
        from code_classifier_vat import CodeClassifier
    except ImportError:
        from .code_classifier_vat import CodeClassifier


__all__ = [
    "CODE_BLUE_HSV_LOWER",
    "CODE_BLUE_HSV_UPPER",
    "CODE_STATUS_BAR_ROI_REL",
    "CODE_STATUS_TEXT_ROI_REL",
    "CodeClassifier",
    "configure_tesseract_command",
    "downscale_projection_roi",
]
