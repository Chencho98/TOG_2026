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
# CODE VISION CLIENTS
# ============================================================================

class CodeVisionClient(Protocol):
    # Maneja analyze sidebar dentro de CodeVisionClient.
    def analyze_sidebar(self, task: CodeVisionTask) -> Dict[str, Any]:
        ...

    # Maneja analyze editor dentro de CodeVisionClient.
    def analyze_editor(self, task: CodeVisionTask) -> Dict[str, Any]:
        ...

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> None:
        ...


class LocalOcrCodeVisionClient:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self):
        self.ocr_enabled = bool(PYTESSERACT_AVAILABLE)
        self._warning_emitted = False

    # Maneja analyze sidebar dentro de LocalOcrCodeVisionClient.
    def analyze_sidebar(self, task: CodeVisionTask) -> Dict[str, Any]:
        result = {
            "active_file": task.active_file_hint,
            "entries": [],
            "notes": "local_ocr",
        }
        if not self.ocr_enabled or not PYTESSERACT_AVAILABLE:
            self._warn_once("⚠️ Visión de código sin API: OCR local deshabilitado.")
            return result

        prepared = self._prepare_sidebar_image(task.image)
        raw_text = self._run_tesseract_string(prepared, "--oem 3 --psm 6")
        entries = self._parse_sidebar_entries(raw_text)
        active_file = self._infer_active_file(entries) or task.active_file_hint
        return {
            "active_file": active_file,
            "entries": entries,
            "raw_text": raw_text,
            "notes": "local_ocr",
        }

    # Maneja analyze editor dentro de LocalOcrCodeVisionClient.
    def analyze_editor(self, task: CodeVisionTask) -> Dict[str, Any]:
        frame_quality = str(task.meta.get("frame_quality") or "usable")
        if frame_quality == "scroll_blur":
            return build_editor_skip_payload(task, "scroll_blur")
        result = {
            "active_file": task.active_file_hint,
            "language": task.language_hint,
            "start_line": None,
            "end_line": None,
            "sticky_context": [],
            "lines": [],
            "tab_title": None,
            "confidence": 0.0,
            "frame_quality": frame_quality,
            "notes": "local_ocr",
        }
        if not self.ocr_enabled or not PYTESSERACT_AVAILABLE:
            self._warn_once("⚠️ Visión de código sin API: OCR local deshabilitado.")
            return result

        prepared = self._prepare_editor_image(task.image)
        line_entries, avg_conf = self._extract_editor_lines_with_tesseract(prepared)
        raw_text = "\n".join(entry["text"] for entry in line_entries if entry.get("text"))
        language = task.language_hint or extract_language(raw_text)

        visible_numbers = [safe_int(entry.get("number")) for entry in line_entries]
        visible_numbers = [number for number in visible_numbers if number is not None]
        start_line = min(visible_numbers) if visible_numbers else None
        end_line = max(visible_numbers) if visible_numbers else None

        payload = {
            "active_file": task.active_file_hint,
            "language": language,
            "start_line": start_line,
            "end_line": end_line,
            "sticky_context": [],
            "lines": line_entries,
            "tab_title": None,
            "confidence": float(avg_conf),
            "raw_text": raw_text,
            "frame_quality": frame_quality,
            "notes": "local_ocr",
        }
        return sanitize_editor_payload(payload, task)

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> None:
        return

    # Prepara sidebar image.
    def _prepare_sidebar_image(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        gray = cv2.resize(
            gray,
            (max(1, int(gray.shape[1] * 2.0)), max(1, int(gray.shape[0] * 2.0))),
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)

    # Prepara editor image.
    def _prepare_editor_image(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        scale = max(1.5, 900.0 / float(max(1, gray.shape[0])))
        target_w = max(1, int(round(gray.shape[1] * scale)))
        target_h = max(1, int(round(gray.shape[0] * scale)))
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    # Ejecuta tesseract string.
    def _run_tesseract_string(self, image: np.ndarray, config: str) -> str:
        if image is None or image.size == 0 or not self.ocr_enabled or not PYTESSERACT_AVAILABLE:
            return ""
        try:
            text = pytesseract.image_to_string(image, config=config)
        except TesseractNotFoundError:
            self.ocr_enabled = False
            self._warn_once(
                f"⚠️ No se encontró tesseract para visión local. Ajusta {OCR_TESSERACT_ENV_VAR} o el PATH."
            )
            return ""
        except Exception:
            return ""
        return str(text or "").strip()

    # Parsea sidebar entries.
    def _parse_sidebar_entries(self, raw_text: str) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        seen = set()
        for raw_line in str(raw_text or "").splitlines():
            line = normalize_whitespace(raw_line)
            if not line or line in seen:
                continue
            seen.add(line)
            depth = max(0, (len(raw_line) - len(raw_line.lstrip())) // 2)
            path = sanitize_relative_code_path(line) if re.search(r"\.[A-Za-z0-9]{1,8}$", line) else None
            entry_type = "file" if path else "folder"
            entries.append(
                {
                    "label": line,
                    "path": path,
                    "depth": int(depth),
                    "type": entry_type,
                    "selected": False,
                }
            )
        return entries

    # Infiere active file.
    def _infer_active_file(self, entries: List[Dict[str, Any]]) -> Optional[str]:
        for entry in entries:
            path = sanitize_relative_code_path(entry.get("path"))
            if path and re.search(r"\.(py|js|ts|java|cpp|c|go|rs|json|ya?ml)$", path, re.IGNORECASE):
                return path
        return None

    # Extrae editor lines with tesseract.
    def _extract_editor_lines_with_tesseract(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        if image is None or image.size == 0 or not self.ocr_enabled or not PYTESSERACT_AVAILABLE:
            return [], 0.0

        config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
        try:
            data = pytesseract.image_to_data(image, output_type=Output.DICT, config=config)
        except Exception:
            return [], 0.0

        rows: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
        item_count = len(data.get("text", []))
        gutter_limit = max(40, int(image.shape[1] * 0.14))
        conf_values: List[float] = []

        for index in range(item_count):
            text = str(data["text"][index] or "").strip()
            conf = safe_float(data["conf"][index], -1.0)
            if conf < 0.0 or not text:
                continue
            x = int(data["left"][index])
            y = int(data["top"][index])
            w = int(data["width"][index])
            h = int(data["height"][index])
            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            rows.setdefault(key, []).append(
                {
                    "text": text,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": conf,
                    "is_gutter": (x + (w // 2)) < gutter_limit,
                }
            )
            conf_values.append(conf / 100.0)

        line_entries: List[Dict[str, Any]] = []
        for key in sorted(rows, key=lambda value: (value[0], value[1], value[2])):
            tokens = sorted(rows[key], key=lambda item: item["x"])
            gutter_tokens = [item for item in tokens if item["is_gutter"]]
            code_tokens = [item for item in tokens if not item["is_gutter"]]
            if not code_tokens:
                continue

            line_number = None
            gutter_numbers = [
                safe_int(re.sub(r"\D", "", token["text"]))
                for token in gutter_tokens
                if re.sub(r"\D", "", token["text"])
            ]
            gutter_numbers = [number for number in gutter_numbers if number is not None]
            if gutter_numbers:
                line_number = gutter_numbers[-1]

            text = " ".join(token["text"] for token in code_tokens).rstrip()
            if not text:
                continue

            line_entries.append(
                {
                    "number": line_number,
                    "text": text,
                    "confidence": float(np.mean([token["conf"] for token in code_tokens]) / 100.0),
                }
            )

        return self._normalize_editor_lines(line_entries), (float(np.mean(conf_values)) if conf_values else 0.0)

    # Normaliza editor lines.
    def _normalize_editor_lines(self, line_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not line_entries:
            return []

        known_positions = []
        for index, entry in enumerate(line_entries):
            number = safe_int(entry.get("number"))
            if number is not None:
                known_positions.append((index, number))

        if len(known_positions) == 1:
            anchor_index, anchor_number = known_positions[0]
            for index, entry in enumerate(line_entries):
                entry["number"] = anchor_number + (index - anchor_index)
        elif len(known_positions) >= 2:
            for index in range(len(known_positions) - 1):
                left_index, left_number = known_positions[index]
                right_index, right_number = known_positions[index + 1]
                gap = right_index - left_index
                if gap <= 1:
                    continue
                if right_number > left_number:
                    for middle in range(1, gap):
                        line_entries[left_index + middle]["number"] = left_number + middle

        return line_entries

    # Maneja warn once dentro de LocalOcrCodeVisionClient.
    def _warn_once(self, message: str) -> None:
        if self._warning_emitted:
            return
        print(message)
        self._warning_emitted = True


# Agrupa regiones de Florence en filas visuales.
def group_florence_regions_into_rows(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for region in regions:
        text = normalize_florence_label_text(region.get("text"))
        rect = florence_box_to_rect(region.get("box"))
        if not text or rect is None:
            continue
        x0, y0, x1, y1 = rect
        prepared.append(
            {
                "text": text,
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "cx": float((x0 + x1) / 2.0),
                "cy": float((y0 + y1) / 2.0),
                "h": float(max(1.0, y1 - y0)),
            }
        )

    if not prepared:
        return []

    prepared.sort(key=lambda item: (item["cy"], item["x0"]))
    row_gap = max(10.0, float(np.median([item["h"] for item in prepared])) * 0.65)
    rows: List[Dict[str, Any]] = []
    for item in prepared:
        if not rows or abs(item["cy"] - rows[-1]["cy"]) > row_gap:
            rows.append({"cy": item["cy"], "tokens": [item]})
            continue
        rows[-1]["tokens"].append(item)
        token_count = len(rows[-1]["tokens"])
        rows[-1]["cy"] = ((rows[-1]["cy"] * (token_count - 1)) + item["cy"]) / float(token_count)

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        tokens = sorted(row["tokens"], key=lambda item: item["x0"])
        normalized_rows.append(
            {
                "cy": float(row["cy"]),
                "tokens": tokens,
                "text": " ".join(token["text"] for token in tokens).strip(),
            }
        )
    return normalized_rows


# Convierte filas Florence en lineas de editor.
def parse_florence_rows_as_lines(
    rows: List[Dict[str, Any]],
    image_width: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not rows:
        return [], []

    code_start_index: Optional[int] = None
    gutter_limit = max(24.0, float(image_width) * CODE_FLORENCE_GUTTER_RATIO)
    for index, row in enumerate(rows):
        tokens = row.get("tokens", [])
        for token in tokens:
            if token.get("x0", 0.0) > gutter_limit:
                continue
            if looks_like_florence_line_number(token.get("text")):
                code_start_index = index
                break
        if code_start_index is not None:
            break

    tab_rows = rows[: min(code_start_index or 0, CODE_FLORENCE_MAX_TAB_ROWS)] if code_start_index is not None else rows[:CODE_FLORENCE_MAX_TAB_ROWS]
    tabs = []
    for row in tab_rows:
        candidate = coerce_florence_tab_candidate(row.get("text", ""))
        if candidate and candidate not in tabs:
            tabs.append(candidate)

    if code_start_index is None:
        return tabs, []

    lines: List[Dict[str, Any]] = []
    for row in rows[code_start_index:]:
        tokens = list(row.get("tokens", []))
        if not tokens:
            continue
        line_number: Optional[int] = None
        raw_number_token: Optional[str] = None
        text_tokens: List[str] = []
        for token in tokens:
            numeric = normalize_florence_numeric_token(token.get("text"))
            if (
                line_number is None
                and numeric is not None
                and float(token.get("x0", 0.0)) <= gutter_limit
            ):
                line_number = safe_int(numeric)
                raw_number_token = numeric
                continue
            text_tokens.append(normalize_florence_label_text(token.get("text")))
        text = " ".join(part for part in text_tokens if part).strip()
        if not text:
            continue
        lines.append(
            {
                "number": line_number,
                "raw_number_token": raw_number_token,
                "text": text,
                "confidence": 0.34 if line_number is not None else 0.22,
                "number_source": "explicit" if line_number is not None else "missing",
                "source": "florence",
            }
        )

    return tabs, repair_florence_number_sequence(lines)


# Convierte una secuencia Florence en lineas y pestañas.
def parse_florence_sequence_as_lines(labels: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    tabs: List[str] = []
    lines: List[Dict[str, Any]] = []
    current_number: Optional[int] = None
    current_raw_number: Optional[str] = None
    current_parts: List[str] = []
    in_code = False

    for label in labels:
        numeric = normalize_florence_numeric_token(label)
        if numeric is not None:
            if current_number is not None or current_parts:
                text = " ".join(current_parts).strip()
                if text:
                    lines.append(
                        {
                            "number": current_number,
                            "raw_number_token": current_raw_number,
                            "text": text,
                            "confidence": 0.30 if current_number is not None else 0.20,
                            "number_source": "explicit" if current_number is not None else "missing",
                            "source": "florence",
                        }
                    )
            current_number = safe_int(numeric)
            current_raw_number = numeric
            current_parts = []
            in_code = True
            continue

        if not in_code:
            candidate = coerce_florence_tab_candidate(label)
            if candidate and candidate not in tabs:
                tabs.append(candidate)
            continue

        current_parts.append(label)

    if current_number is not None or current_parts:
        text = " ".join(current_parts).strip()
        if text:
            lines.append(
                {
                    "number": current_number,
                    "raw_number_token": current_raw_number,
                    "text": text,
                    "confidence": 0.30 if current_number is not None else 0.20,
                    "number_source": "explicit" if current_number is not None else "missing",
                    "source": "florence",
                }
            )

    return tabs, repair_florence_number_sequence(lines)


# Selecciona el mejor candidato de pestaña activa.
def choose_florence_tab_candidate(
    tab_candidates: List[str],
    active_file_hint: Optional[str],
) -> Optional[str]:
    if not tab_candidates:
        return None
    hinted_basename = path_basename(active_file_hint).lower()
    if hinted_basename:
        best_candidate = None
        best_distance = None
        for candidate in tab_candidates:
            distance = levenshtein_distance(path_basename(candidate).lower(), hinted_basename)
            if best_distance is None or distance < best_distance:
                best_candidate = candidate
                best_distance = distance
        if best_candidate is not None and best_distance is not None and best_distance <= 2:
            return best_candidate
    return tab_candidates[-1]


# Parsea el payload Florence del editor.
def parse_florence_editor_payload(
    parsed_output: Any,
    image_shape: Tuple[int, ...],
    active_file_hint: Optional[str],
) -> Dict[str, Any]:
    labels = extract_florence_labels(parsed_output)
    regions = extract_florence_regions(parsed_output)
    rows = group_florence_regions_into_rows(regions)
    image_width = int(image_shape[1]) if len(image_shape) >= 2 else 0

    tab_candidates, row_lines = parse_florence_rows_as_lines(rows, image_width)
    sequence_tabs, sequence_lines = parse_florence_sequence_as_lines(labels)
    for candidate in sequence_tabs:
        if candidate not in tab_candidates:
            tab_candidates.append(candidate)

    lines = row_lines
    if len(sequence_lines) > len(lines):
        lines = sequence_lines
    elif sequence_lines:
        existing_keys = {
            (
                safe_int(entry.get("number")),
                normalize_code_line_for_match(entry.get("text", "")),
            )
            for entry in lines
        }
        for entry in sequence_lines:
            key = (
                safe_int(entry.get("number")),
                normalize_code_line_for_match(entry.get("text", "")),
            )
            if key in existing_keys:
                continue
            lines.append(entry)
            existing_keys.add(key)
        lines = repair_florence_number_sequence(
            sorted(
                lines,
                key=lambda entry: (
                    safe_int(entry.get("number")) is None,
                    safe_int(entry.get("number")) if safe_int(entry.get("number")) is not None else 10**9,
                    normalize_code_line_for_match(entry.get("text", "")),
                ),
            )
        )

    sticky_context, lines = split_sticky_context_prefix(lines)
    explicit_numbers = [safe_int(entry.get("number")) for entry in lines]
    explicit_numbers = [number for number in explicit_numbers if number is not None]
    tab_candidate = choose_florence_tab_candidate(tab_candidates, active_file_hint)
    active_file_candidate = tab_candidate if tab_candidate and "/" in tab_candidate else None
    tab_title_candidate = path_basename(tab_candidate) if tab_candidate else None
    language = None
    if tab_candidate and tab_candidate.lower().endswith(".py"):
        language = "PYTHON"
    elif lines:
        language = extract_language("\n".join(str(entry.get("text", "")) for entry in lines))

    return {
        "tab_candidates": tab_candidates[:CODE_FLORENCE_MAX_TAB_ROWS],
        "active_file_candidate": active_file_candidate,
        "tab_title_candidate": tab_title_candidate,
        "sticky_context": sticky_context,
        "lines": lines,
        "start_line": min(explicit_numbers) if explicit_numbers else None,
        "end_line": max(explicit_numbers) if explicit_numbers else None,
        "raw_labels": labels[:CODE_FLORENCE_MAX_LABELS],
        "region_count": len(regions),
        "row_count": len(rows),
        "language": language,
    }


# Normaliza entradas crudas de lineas del editor.
def prepare_editor_line_entries(
    payload: Dict[str, Any],
    default_confidence: float,
    source: str,
) -> List[Dict[str, Any]]:
    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list):
        return []
    prepared: List[Dict[str, Any]] = []
    for entry in raw_lines:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text", "") or "").rstrip()
        original_number = safe_int(entry.get("number"))
        if not should_keep_editor_line(text, original_number):
            continue
        confidence = safe_float(entry.get("confidence"), default_confidence)
        if source == "remote":
            confidence = max(confidence, max(default_confidence, 0.62))
        else:
            confidence = min(max(confidence, 0.18), 0.45)
        number_source = str(entry.get("number_source", "") or ("explicit" if original_number is not None else "missing"))
        segments = split_compound_editor_line(original_number, text)
        for segment_index, (number, segment_text) in enumerate(segments):
            segment_text = str(segment_text or "").rstrip()
            if not should_keep_editor_line(segment_text, number):
                continue
            prepared.append(
                {
                    "number": number,
                    "text": segment_text,
                    "confidence": float(confidence),
                    "number_source": number_source if segment_index == 0 else "embedded_split",
                    "source": source,
                    "line_kind": infer_editor_line_kind(segment_text),
                }
            )
    return prepared


# Separa contexto sticky del bloque principal.
def split_sticky_context_prefix(
    raw_entries: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(raw_entries, list):
        return [], []

    entries = [dict(entry) for entry in raw_entries if isinstance(entry, dict)]
    if len(entries) < (CODE_STICKY_BODY_MIN_NUMBERED_LINES + 1):
        return [], entries

    best_split = 0
    best_score = float("-inf")
    max_prefix = min(CODE_STICKY_MAX_PREFIX_LINES, len(entries) - CODE_STICKY_BODY_MIN_NUMBERED_LINES)
    for split_index in range(1, max_prefix + 1):
        prefix = entries[:split_index]
        suffix = entries[split_index:]

        prefix_numbers = [safe_int(entry.get("number")) for entry in prefix]
        if any(number is None for number in prefix_numbers):
            continue
        suffix_numbers = [safe_int(entry.get("number")) for entry in suffix if safe_int(entry.get("number")) is not None]
        if len(suffix_numbers) < CODE_STICKY_BODY_MIN_NUMBERED_LINES:
            continue
        if max(prefix_numbers) >= min(suffix_numbers):
            continue

        prefix_gaps = [
            int(right) - int(left)
            for left, right in zip(prefix_numbers, prefix_numbers[1:])
            if left is not None and right is not None and int(right) > int(left)
        ]
        suffix_dense_pairs = sum(
            1
            for left, right in zip(suffix_numbers, suffix_numbers[1:])
            if 0 < (int(right) - int(left)) <= 2
        )
        body_start_gap = int(suffix_numbers[0]) - int(prefix_numbers[-1])
        if body_start_gap < CODE_STICKY_BODY_START_GAP:
            continue
        if suffix_dense_pairs < CODE_STICKY_BODY_MIN_DENSE_PAIRS:
            continue
        if prefix_gaps and max(prefix_gaps) < CODE_STICKY_PREFIX_SPARSE_GAP and body_start_gap < (CODE_STICKY_BODY_START_GAP + 1):
            continue

        score = (
            (2.0 * float(suffix_dense_pairs))
            + (0.20 * float(body_start_gap))
            + (0.10 * float(max(prefix_gaps) if prefix_gaps else body_start_gap))
            - (0.25 * float(split_index))
        )
        if score > best_score:
            best_score = score
            best_split = split_index

    if best_split <= 0:
        return [], entries
    return entries[:best_split], entries[best_split:]


# Fusiona entradas de contexto del editor sin duplicarlas.
def merge_editor_context_entries(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[Tuple[Optional[int], str]] = set()
    for entry in list(primary) + list(secondary):
        if not isinstance(entry, dict):
            continue
        line_number = safe_int(entry.get("number"))
        text = str(entry.get("text", "") or "").rstrip()
        text_key = normalize_code_line_for_match(text)
        if not should_keep_editor_line(text, line_number) or not text_key or text_key == BLANK_EDITOR_LINE_SENTINEL:
            continue
        key = (line_number, text_key)
        if key in seen:
            continue
        merged.append(dict(entry))
        seen.add(key)
    return merged[:CODE_STICKY_MAX_PREFIX_LINES]


# Deduplica lineas usando numero y texto.
def deduplicate_editor_entries_by_number(
    entries: List[Dict[str, Any]],
    language_hint: Optional[str],
) -> List[Dict[str, Any]]:
    best_by_number: Dict[int, Tuple[float, Dict[str, Any]]] = {}
    for entry in entries:
        line_number = safe_int(entry.get("number"))
        if line_number is None:
            continue
        score = float(
            safe_float(entry.get("confidence"), 0.0)
            + code_like_text_score(entry.get("text", ""), language_hint)
        )
        existing = best_by_number.get(int(line_number))
        if existing is None or score >= existing[0]:
            best_by_number[int(line_number)] = (score, dict(entry))

    deduped: List[Dict[str, Any]] = []
    emitted_numbers: set[int] = set()
    for entry in entries:
        line_number = safe_int(entry.get("number"))
        if line_number is None:
            deduped.append(dict(entry))
            continue
        if int(line_number) in emitted_numbers:
            continue
        chosen = best_by_number.get(int(line_number))
        deduped.append(dict(chosen[1] if chosen is not None else entry))
        emitted_numbers.add(int(line_number))
    return deduped


# Limpia y valida lineas detectadas del editor.
def sanitize_editor_line_entries(
    entries: List[Dict[str, Any]],
    estimated_visible_rows: int,
    language_hint: Optional[str],
) -> List[Dict[str, Any]]:
    if not entries:
        return []

    sanitized: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text", "") or "").rstrip()
        line_number = safe_int(entry.get("number"))
        if line_number is not None and int(line_number) < 1:
            continue
        if not should_keep_editor_line(text, line_number):
            continue
        normalized_entry = dict(entry)
        normalized_entry["text"] = text
        normalized_entry["number"] = None if line_number is None else int(line_number)
        normalized_entry["line_kind"] = infer_editor_line_kind(text)
        sanitized.append(normalized_entry)

    if not sanitized:
        return []

    sanitized = deduplicate_editor_entries_by_number(sanitized, language_hint)
    cluster_indices = choose_densest_line_number_cluster(
        sanitized,
        max_gap=max(3, min(CODE_EDITOR_MAX_MONOTONIC_GAP, max(4, int(estimated_visible_rows) // 3 + 2))),
    )
    if cluster_indices is not None:
        explicit_count = sum(1 for entry in sanitized if safe_int(entry.get("number")) is not None)
        if explicit_count > len(cluster_indices):
            sanitized = [
                dict(entry)
                for index, entry in enumerate(sanitized)
                if safe_int(entry.get("number")) is None or index in cluster_indices
            ]

    monotonic: List[Dict[str, Any]] = []
    previous_number: Optional[int] = None
    max_gap = max(3, min(CODE_EDITOR_MAX_MONOTONIC_GAP, max(4, int(estimated_visible_rows) // 3 + 2)))
    for entry in sanitized:
        line_number = safe_int(entry.get("number"))
        if line_number is None:
            monotonic.append(dict(entry))
            continue
        if previous_number is not None:
            diff = int(line_number) - int(previous_number)
            if diff <= 0 or diff > max_gap:
                continue
        monotonic.append(dict(entry))
        previous_number = int(line_number)
    return monotonic


# Normaliza un payload completo del editor.
def sanitize_editor_payload(
    payload: Dict[str, Any],
    task: CodeVisionTask,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    sanitized = dict(payload)
    selected_metrics = task.meta.get("selected_metrics")
    estimated_visible_rows = safe_int(task.meta.get("estimated_visible_rows"))
    if estimated_visible_rows is None and isinstance(selected_metrics, dict):
        estimated_visible_rows = safe_int(selected_metrics.get("estimated_visible_rows"))
    estimated_visible_rows = max(CODE_EDITOR_MIN_VISIBLE_ROWS, int(estimated_visible_rows or CODE_EDITOR_MIN_VISIBLE_ROWS))
    language_hint = sanitized.get("language") or task.language_hint

    sticky_entries = sanitize_editor_line_entries(
        list(sanitized.get("sticky_context") or []),
        estimated_visible_rows=max(CODE_STICKY_MAX_PREFIX_LINES + 1, min(estimated_visible_rows, 8)),
        language_hint=language_hint,
    )
    body_entries = sanitize_editor_line_entries(
        list(sanitized.get("lines") or []),
        estimated_visible_rows=estimated_visible_rows,
        language_hint=language_hint,
    )

    if not sticky_entries and body_entries:
        inferred_sticky, inferred_body = split_sticky_context_prefix(body_entries)
        sticky_entries = sanitize_editor_line_entries(
            inferred_sticky,
            estimated_visible_rows=max(CODE_STICKY_MAX_PREFIX_LINES + 1, min(estimated_visible_rows, 8)),
            language_hint=language_hint,
        )
        body_entries = sanitize_editor_line_entries(
            inferred_body,
            estimated_visible_rows=estimated_visible_rows,
            language_hint=language_hint,
        )

    explicit_numbers = [safe_int(entry.get("number")) for entry in body_entries]
    explicit_numbers = [number for number in explicit_numbers if number is not None]
    visible_span = 0 if not explicit_numbers else (max(explicit_numbers) - min(explicit_numbers))
    max_visible_span = (
        max(CODE_EDITOR_MIN_VISIBLE_ROWS, int(estimated_visible_rows)) * CODE_EDITOR_MAX_VISIBLE_LINE_RANGE_MULTIPLIER
        + CODE_EDITOR_MAX_VISIBLE_LINE_RANGE_PADDING
    )
    suspicious_lines = sum(
        1
        for entry in body_entries
        if (
            not is_blank_editor_line_text(entry.get("text", ""))
            and code_like_text_score(entry.get("text", ""), language_hint) < -0.10
        )
    )
    reject_reason: Optional[str] = None
    if explicit_numbers and visible_span > max_visible_span:
        reject_reason = "implausible_visible_range"
    elif body_entries and suspicious_lines >= max(2, len(body_entries) // 2):
        reject_reason = "noisy_editor_payload"

    if reject_reason:
        body_entries = []
        sanitized["raw_text"] = ""

    sanitized["active_file"] = sanitize_relative_code_path(sanitized.get("active_file"))
    sanitized["tab_title"] = sanitize_relative_code_path(sanitized.get("tab_title"))
    sanitized["language"] = language_hint
    sanitized["sticky_context"] = sticky_entries
    sanitized["lines"] = body_entries
    sanitized["start_line"] = min(explicit_numbers) if explicit_numbers and not reject_reason else None
    sanitized["end_line"] = max(explicit_numbers) if explicit_numbers and not reject_reason else None
    sanitized["frame_quality"] = str(
        sanitized.get("frame_quality")
        or task.meta.get("frame_quality")
        or "usable"
    )
    sanitized["__quality__"] = {
        "estimated_visible_rows": int(estimated_visible_rows),
        "visible_span": int(visible_span),
        "max_visible_span": int(max_visible_span),
        "line_count": len(body_entries),
        "sticky_count": len(sticky_entries),
        "reject_reason": reject_reason,
    }
    if reject_reason:
        sanitized["skip_reason"] = reject_reason
    return sanitized


# Extrae contexto sticky y cuerpo desde un payload.
def extract_editor_line_groups(
    payload: Dict[str, Any],
    default_confidence: float,
    source: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return [], []

    raw_lines = payload.get("lines")
    raw_sticky = payload.get("sticky_context")
    sticky_entries = prepare_editor_line_entries(
        {"lines": raw_sticky},
        default_confidence,
        source=source,
    )
    body_source = raw_lines if isinstance(raw_lines, list) else []

    if not sticky_entries:
        inferred_sticky_raw, inferred_body_raw = split_sticky_context_prefix(body_source)
        sticky_entries = prepare_editor_line_entries(
            {"lines": inferred_sticky_raw},
            default_confidence,
            source=source,
        )
        body_source = inferred_body_raw

    body_entries = prepare_editor_line_entries(
        {"lines": body_source},
        default_confidence,
        source=source,
    )
    sticky_keys = {
        (
            safe_int(entry.get("number")),
            normalize_code_line_for_match(entry.get("text", "")),
        )
        for entry in sticky_entries
        if normalize_code_line_for_match(entry.get("text", ""))
    }
    if sticky_keys:
        body_entries = [
            entry
            for entry in body_entries
            if (
                safe_int(entry.get("number")),
                normalize_code_line_for_match(entry.get("text", "")),
            ) not in sticky_keys
        ]
    return sticky_entries, body_entries


# Alinea lineas Florence contra lineas remotas.
def align_florence_lines_to_remote(
    florence_lines: List[Dict[str, Any]],
    remote_lines: List[Dict[str, Any]],
    remote_payload: Dict[str, Any],
    florence_payload: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not florence_lines or not remote_lines:
        return florence_lines, {"offset": 0, "anchors": 0}

    remote_by_text: Dict[str, int] = {}
    for entry in remote_lines:
        line_number = safe_int(entry.get("number"))
        text_key = normalize_code_line_for_match(entry.get("text", ""))
        if (
            line_number is None
            or not text_key
            or text_key == BLANK_EDITOR_LINE_SENTINEL
            or not any(char.isalpha() for char in text_key)
            or text_key in remote_by_text
        ):
            continue
        remote_by_text[text_key] = int(line_number)

    offsets: Dict[int, int] = {}
    for entry in florence_lines:
        line_number = safe_int(entry.get("number"))
        text_key = normalize_code_line_for_match(entry.get("text", ""))
        if (
            line_number is None
            or not text_key
            or text_key == BLANK_EDITOR_LINE_SENTINEL
            or not any(char.isalpha() for char in text_key)
        ):
            continue
        remote_number = remote_by_text.get(text_key)
        if remote_number is None:
            continue
        offset = int(remote_number) - int(line_number)
        offsets[offset] = offsets.get(offset, 0) + 1

    best_offset = 0
    best_anchors = 0
    for offset, anchor_count in offsets.items():
        if anchor_count > best_anchors or (anchor_count == best_anchors and abs(offset) < abs(best_offset)):
            best_offset = int(offset)
            best_anchors = int(anchor_count)

    if best_anchors == 0:
        remote_start = safe_int(remote_payload.get("start_line"))
        florence_start = safe_int(florence_payload.get("start_line"))
        if remote_start is not None and florence_start is not None:
            best_offset = int(remote_start) - int(florence_start)

    if best_offset == 0:
        return florence_lines, {"offset": 0, "anchors": best_anchors}

    aligned: List[Dict[str, Any]] = []
    for entry in florence_lines:
        updated = dict(entry)
        line_number = safe_int(updated.get("number"))
        if line_number is not None:
            updated["number"] = int(line_number) + best_offset
            if str(updated.get("number_source", "")) == "explicit":
                updated["number_source"] = "interpolated"
        aligned.append(updated)
    return aligned, {"offset": best_offset, "anchors": best_anchors}


# Aplica hints de numeros de linea de Florence.
def apply_florence_line_number_hints(
    remote_lines: List[Dict[str, Any]],
    florence_lines: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    if not remote_lines or not florence_lines:
        return [dict(entry) for entry in remote_lines], 0

    florence_by_text: Dict[str, Dict[str, Any]] = {}
    for entry in florence_lines:
        line_number = safe_int(entry.get("number"))
        text_key = normalize_code_line_for_match(entry.get("text", ""))
        if (
            line_number is None
            or not text_key
            or text_key == BLANK_EDITOR_LINE_SENTINEL
            or not any(char.isalpha() for char in text_key)
        ):
            continue
        florence_by_text.setdefault(text_key, dict(entry))

    patched = 0
    updated_lines: List[Dict[str, Any]] = []
    for entry in remote_lines:
        updated = dict(entry)
        line_number = safe_int(updated.get("number"))
        text_key = normalize_code_line_for_match(updated.get("text", ""))
        florence_entry = florence_by_text.get(text_key)
        if florence_entry is not None:
            florence_number = safe_int(florence_entry.get("number"))
            number_source = str(updated.get("number_source", "unknown") or "unknown")
            if florence_number is not None and (
                line_number is None
                or number_source in {"missing", "raw_text", "start_line", "anchor_inferred"}
            ):
                updated["number"] = int(florence_number)
                updated["number_source"] = "florence_anchor"
                patched += 1
        updated_lines.append(updated)
    return updated_lines, patched


# Decide si un payload remoto necesita apoyo Florence.
def editor_payload_needs_florence(
    payload: Dict[str, Any],
    task: CodeVisionTask,
    last_run_age_sec: float,
) -> Optional[str]:
    if not isinstance(payload, dict):
        return "missing_primary_payload"
    if payload.get("error"):
        return "remote_error"

    quality_summary = payload.get("__quality__") if isinstance(payload.get("__quality__"), dict) else {}
    reject_reason = quality_summary.get("reject_reason")
    if reject_reason:
        return str(reject_reason)

    line_count = len(payload.get("lines") or [])
    confidence = safe_float(payload.get("confidence"), 0.0)
    active_file = sanitize_relative_code_path(payload.get("active_file"))
    tab_title = sanitize_relative_code_path(payload.get("tab_title"))
    start_line = safe_int(payload.get("start_line"))
    end_line = safe_int(payload.get("end_line"))

    if not active_file and not tab_title and not task.active_file_hint:
        return "missing_file_identity"
    if line_count <= 0:
        return "missing_lines"
    if line_count < CODE_FLORENCE_MIN_REMOTE_LINE_COUNT and confidence < CODE_FLORENCE_REMOTE_CONFIDENCE_MIN:
        return "thin_remote_block"
    if start_line is None or end_line is None:
        return "missing_visible_range"
    if last_run_age_sec >= CODE_FLORENCE_INTERVAL_SECONDS and confidence < 0.90:
        return "periodic_structural_refresh"
    return None


# Fusiona resultados remotos y Florence del editor.
def merge_remote_and_florence_editor_payload(
    remote_payload: Dict[str, Any],
    florence_payload: Dict[str, Any],
    task: CodeVisionTask,
) -> Dict[str, Any]:
    remote_error = remote_payload.get("error") if isinstance(remote_payload, dict) else None
    florence_info = florence_payload if isinstance(florence_payload, dict) else {}
    florence_parsed = florence_info.get("parsed") if isinstance(florence_info.get("parsed"), dict) else {}

    if remote_error and not florence_parsed:
        return dict(remote_payload)

    merged = dict(remote_payload) if isinstance(remote_payload, dict) and not remote_error else {}
    if isinstance(remote_payload, dict) and isinstance(remote_payload.get("__provider__"), dict):
        merged["__provider__"] = dict(remote_payload.get("__provider__"))
    if remote_error and isinstance(remote_payload, dict):
        raw_usage = remote_payload.get("__usage__")
        if isinstance(raw_usage, dict):
            merged["__usage__"] = dict(raw_usage)
    remote_confidence = safe_float(merged.get("confidence"), 0.0)
    florence_confidence = 0.38 if florence_parsed else 0.0
    remote_sticky, remote_lines = extract_editor_line_groups(merged, remote_confidence, source="remote")
    florence_sticky, florence_lines = extract_editor_line_groups(florence_parsed, florence_confidence, source="florence")
    florence_lines, florence_alignment = align_florence_lines_to_remote(
        florence_lines,
        remote_lines,
        merged,
        florence_parsed,
    )

    merged_lines: List[Dict[str, Any]]
    florence_added = 0
    florence_patched_numbers = 0
    florence_used_as_fallback = False
    if remote_lines:
        merged_lines, florence_patched_numbers = apply_florence_line_number_hints(
            remote_lines,
            florence_lines,
        )
    else:
        merged_lines = [dict(entry) for entry in florence_lines]
        florence_added = len(merged_lines)
        florence_used_as_fallback = bool(merged_lines)

    merged_lines.sort(
        key=lambda entry: (
            safe_int(entry.get("number")) is None,
            safe_int(entry.get("number")) if safe_int(entry.get("number")) is not None else 10**9,
            normalize_code_line_for_match(entry.get("text", "")),
        )
    )

    explicit_numbers = [safe_int(entry.get("number")) for entry in merged_lines]
    explicit_numbers = [number for number in explicit_numbers if number is not None]
    merged["active_file"] = (
        sanitize_relative_code_path(merged.get("active_file"))
        or sanitize_relative_code_path(florence_parsed.get("active_file_candidate"))
        or sanitize_relative_code_path(task.active_file_hint)
    )
    merged["tab_title"] = (
        sanitize_relative_code_path(merged.get("tab_title"))
        or sanitize_relative_code_path(florence_parsed.get("tab_title_candidate"))
    )
    merged["language"] = (
        merged.get("language")
        or florence_parsed.get("language")
        or task.language_hint
    )
    merged["frame_quality"] = str(task.meta.get("frame_quality") or "usable")
    merged["sticky_context"] = merge_editor_context_entries(remote_sticky, florence_sticky)
    merged["lines"] = merged_lines
    merged["start_line"] = (
        (min(explicit_numbers) if explicit_numbers else None)
        or safe_int(merged.get("start_line"))
        or safe_int(florence_parsed.get("start_line"))
    )
    merged["end_line"] = (
        (max(explicit_numbers) if explicit_numbers else None)
        or safe_int(merged.get("end_line"))
        or safe_int(florence_parsed.get("end_line"))
    )
    merged["confidence"] = max(remote_confidence, 0.0)
    if remote_error or florence_used_as_fallback:
        merged["confidence"] = max(merged["confidence"], florence_confidence)
    merged["__florence__"] = {
        "status": florence_info.get("status"),
        "ok": bool(florence_parsed),
        "model_id": florence_info.get("model_id"),
        "device": florence_info.get("device"),
        "runtime_backend": florence_info.get("runtime_backend"),
        "runtime_class": florence_info.get("runtime_class"),
        "elapsed_sec": safe_float(florence_info.get("elapsed_sec"), 0.0),
        "tab_candidates": list(florence_parsed.get("tab_candidates", [])),
        "active_file_candidate": florence_parsed.get("active_file_candidate"),
        "tab_title_candidate": florence_parsed.get("tab_title_candidate"),
        "line_count": len(florence_lines),
        "row_count": safe_int(florence_parsed.get("row_count")) or 0,
        "region_count": safe_int(florence_parsed.get("region_count")) or 0,
        "raw_labels": list(florence_parsed.get("raw_labels", [])),
        "alignment": dict(florence_alignment),
        "added_lines": int(florence_added),
        "patched_line_numbers": int(florence_patched_numbers),
        "used_as_fallback": bool(florence_used_as_fallback),
        "error": florence_info.get("error"),
    }
    if remote_error:
        merged["__remote_error__"] = str(remote_error)
        if remote_payload.get("message"):
            merged["__remote_error_message__"] = str(remote_payload.get("message"))
    return merged


class FlorenceCodeVisionHelper:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self):
        self.enabled = bool(CODE_FLORENCE_ENABLE)
        self._warning_emitted = False
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._disabled_reason: Optional[str] = None
        self._loaded = False
        self._torch = None
        self._processor = None
        self._model = None
        self._device: Optional[str] = None
        self._model_id: Optional[str] = None
        self._runtime_backend: str = "torch"
        self._runtime_class: str = ""

    # Maneja analyze editor dentro de FlorenceCodeVisionHelper.
    def analyze_editor(self, task: CodeVisionTask) -> Dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled"}
        if task.image is None or task.image.size == 0:
            return {"status": "empty"}
        if CODE_FLORENCE_BUSY_SKIP:
            acquired = self._infer_lock.acquire(blocking=False)
            if not acquired:
                return {"status": "busy"}
        else:
            self._infer_lock.acquire()

        started = time.perf_counter()
        try:
            self._ensure_loaded()
            if self._disabled_reason:
                return {"status": "unavailable", "error": self._disabled_reason}

            prepared_image = prepare_editor_image_for_florence(task.image)
            task_prompt = "<OCR_WITH_REGION>" if CODE_FLORENCE_TASK == "ocr_with_region" else "<OCR>"
            inputs = self._processor(text=task_prompt, images=prepared_image, return_tensors="pt")
            if self._runtime_backend == "torch":
                inputs = move_batch_to_device(inputs, self._device or CODE_FLORENCE_DEVICE)

            generate_kwargs = {
                "max_new_tokens": int(CODE_FLORENCE_MAX_NEW_TOKENS),
                "do_sample": False,
                "num_beams": 1,
            }
            if inputs.get("input_ids") is not None:
                generate_kwargs["input_ids"] = inputs["input_ids"]
            if inputs.get("attention_mask") is not None:
                generate_kwargs["attention_mask"] = inputs["attention_mask"]
            if inputs.get("pixel_values") is not None:
                generate_kwargs["pixel_values"] = inputs["pixel_values"]

            if self._runtime_backend == "torch" and self._torch is not None:
                with self._torch.inference_mode():
                    generated_ids = self._model.generate(**generate_kwargs)
            else:
                generated_ids = self._model.generate(**generate_kwargs)

            decoded_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]
            parsed_output = self._processor.post_process_generation(
                decoded_text,
                task=task_prompt,
                image_size=(prepared_image.width, prepared_image.height),
            )
            parsed_editor = parse_florence_editor_payload(
                parsed_output,
                task.image.shape,
                task.active_file_hint,
            )
            elapsed = time.perf_counter() - started
            return {
                "status": "ok",
                "model_id": self._model_id,
                "device": self._device,
                "runtime_backend": self._runtime_backend,
                "runtime_class": self._runtime_class,
                "elapsed_sec": elapsed,
                "parsed": parsed_editor,
                "decoded_preview": str(decoded_text or "")[:400],
            }
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return {
                "status": "error",
                "model_id": self._model_id,
                "device": self._device,
                "runtime_backend": self._runtime_backend,
                "runtime_class": self._runtime_class,
                "elapsed_sec": elapsed,
                "error": str(exc),
            }
        finally:
            self._infer_lock.release()

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> None:
        if self._torch is not None:
            cleanup_torch(self._torch, self._device)
        self._processor = None
        self._model = None
        self._torch = None
        self._runtime_backend = "torch"
        self._runtime_class = ""

    # Maneja ensure loaded dentro de FlorenceCodeVisionHelper.
    def _ensure_loaded(self) -> None:
        if self._loaded or self._disabled_reason:
            return
        with self._load_lock:
            if self._loaded or self._disabled_reason:
                return
            try:
                (
                    torch_module,
                    auto_config_cls,
                    auto_model_cls,
                    auto_processor_cls,
                ) = import_florence_dependencies()
                loaded_model = load_florence_model(
                    candidate_models(CODE_FLORENCE_MODEL, DEFAULT_FLORENCE_MODELS),
                    CODE_FLORENCE_BACKEND,
                    CODE_FLORENCE_DEVICE,
                    auto_config_cls,
                    auto_model_cls,
                    auto_processor_cls,
                    torch_module,
                )
                self._torch = torch_module
                self._processor = loaded_model.processor
                self._model = loaded_model.model
                self._device = loaded_model.runtime_device
                self._model_id = loaded_model.model_id
                self._runtime_backend = loaded_model.runtime_backend
                self._runtime_class = loaded_model.runtime_class
                self._loaded = True
            except Exception as exc:
                self._disabled_reason = str(exc)
                self._warn_once(f"⚠️ Florence-2 no está disponible: {exc}")

    # Maneja warn once dentro de FlorenceCodeVisionHelper.
    def _warn_once(self, message: str) -> None:
        if self._warning_emitted:
            return
        print(message)
        self._warning_emitted = True


class HybridCodeVisionClient:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self, primary: CodeVisionClient, florence: Optional[FlorenceCodeVisionHelper] = None):
        self.primary = primary
        self.florence = florence
        self._last_florence_ts = 0.0

    # Maneja analyze sidebar dentro de HybridCodeVisionClient.
    def analyze_sidebar(self, task: CodeVisionTask) -> Dict[str, Any]:
        return self.primary.analyze_sidebar(task)

    # Maneja analyze editor dentro de HybridCodeVisionClient.
    def analyze_editor(self, task: CodeVisionTask) -> Dict[str, Any]:
        if str(task.meta.get("frame_quality") or "usable") == "scroll_blur":
            return build_editor_skip_payload(task, "scroll_blur")
        primary_payload = sanitize_editor_payload(self.primary.analyze_editor(task), task)
        if self.florence is None:
            return primary_payload
        florence_reason = editor_payload_needs_florence(
            primary_payload,
            task,
            last_run_age_sec=max(0.0, time.time() - float(self._last_florence_ts)),
        )
        if not florence_reason:
            primary_payload["__florence__"] = {
                "status": "skipped",
                "ok": False,
                "reason": "not_needed",
                "elapsed_sec": 0.0,
            }
            return primary_payload

        florence_payload = self.florence.analyze_editor(task)
        if str(florence_payload.get("status") or "") not in {"busy", "empty"}:
            self._last_florence_ts = time.time()
        merged = merge_remote_and_florence_editor_payload(primary_payload, florence_payload, task)
        if isinstance(merged.get("__florence__"), dict):
            merged["__florence__"]["reason"] = florence_reason
        return sanitize_editor_payload(merged, task)

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            if self.florence is not None:
                self.florence.close()


class OpenAICodeVisionClient:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self, profile: Optional[Dict[str, Any]] = None, role: str = "primary"):
        self.profile = dict(profile or CODE_VISION_ACTIVE_PROFILE)
        self.role = str(role or "primary").strip().lower()
        self.provider_name = str(self.profile.get("provider_name", "openai") or "openai").strip().lower()
        self.provider_label = str(self.profile.get("label", "Remote API") or "Remote API").strip()
        self.prompt_order = str(self.profile.get("prompt_order", "image_first") or "image_first").strip().lower()
        api_key_envs = tuple(self.profile.get("api_key_envs", (CODE_VISION_API_KEY_ENV_VAR,)))
        self.api_key = resolve_remote_api_key(api_key_envs)
        model_override = ""
        base_url_override = ""
        if profile is None:
            model_override = str(
                os.environ.get(CODE_VISION_MODEL_ENV_VAR)
                or os.environ.get(OPENAI_MODEL_ENV_VAR)
                or ""
            ).strip()
            base_url_override = str(os.environ.get(CODE_VISION_BASE_URL_ENV_VAR) or "").strip()
        self.model = (model_override or str(self.profile.get("model", CODE_VISION_DEFAULT_MODEL))).strip()
        base_url = (
            base_url_override
            or str(self.profile.get("base_url", CODE_VISION_DEFAULT_BASE_URL))
            or CODE_VISION_DEFAULT_BASE_URL
        ).strip()
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.endpoint = self.base_url
        else:
            self.endpoint = self.base_url + "/chat/completions"
        self.timeout = safe_float(os.environ.get(CODE_VISION_TIMEOUT_ENV_VAR), CODE_VISION_DEFAULT_TIMEOUT_SEC)
        print(f"🌐 Code vision provider: {self.provider_label} ({self.model}) role={self.role}")

    # Maneja analyze sidebar dentro de OpenAICodeVisionClient.
    def analyze_sidebar(self, task: CodeVisionTask) -> Dict[str, Any]:
        payload = self._run_json_prompt(
            SIDEBAR_ANALYSIS_PROMPT,
            task.image,
            image_format=task.image_format,
            jpeg_quality=task.jpeg_quality,
        )
        if not isinstance(payload, dict):
            return {}
        return payload

    # Maneja analyze editor dentro de OpenAICodeVisionClient.
    def analyze_editor(self, task: CodeVisionTask) -> Dict[str, Any]:
        if str(task.meta.get("frame_quality") or "usable") == "scroll_blur":
            return build_editor_skip_payload(task, "scroll_blur")
        prompt = build_editor_analysis_prompt(task)
        payload = self._run_json_prompt(
            prompt,
            task.image,
            image_format=task.image_format,
            jpeg_quality=task.jpeg_quality,
        )
        if not isinstance(payload, dict):
            return {}
        payload.setdefault("sticky_context", [])
        payload.setdefault("frame_quality", str(task.meta.get("frame_quality") or "usable"))
        return sanitize_editor_payload(payload, task)

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> None:
        return

    # Ejecuta JSON prompt.
    def _run_json_prompt(
        self,
        prompt: str,
        image: np.ndarray,
        image_format: str = "jpg",
        jpeg_quality: int = 90,
    ) -> Dict[str, Any]:
        image_url = image_to_data_url(
            image,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )
        if not image_url:
            return {}

        request_payload = {
            "model": self.model,
            # "max_tokens": CODE_VISION_MAX_TOKENS, #4o
            "max_completion_tokens": CODE_VISION_MAX_TOKENS,#5.4mini
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an accessibility assistant that extracts code and file structure "
                        "from VS Code screenshots. Return only valid JSON matching the requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_user_content(prompt, image_url),
                },
            ],
        }
        # request_payload = {
        #     "model": self.model,
        #     "max_completion_tokens": CODE_VISION_MAX_TOKENS, # Correcto para modelos nuevos
        #     # ❌ ELIMINADO: "temperature": 0, (Prohibido en modelos de razonamiento)
        #     "messages": [
        #         {
        #             # ✅ CAMBIO: De "system" a "developer" (el nuevo estándar para estas IAs)
        #             "role": "developer", 
        #             "content": (
        #                 "You are an accessibility assistant that extracts code and file structure "
        #                 "from VS Code screenshots. Return only valid JSON matching the requested schema."
        #             ),
        #         },
        #         {
        #             "role": "user",
        #             "content": self._build_user_content(prompt, image_url),
        #         },
        #     ],
        # }
        request_data = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=request_data, method="POST")
        request.add_header("Authorization", f"Bearer {self.api_key}")
        request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="ignore")
                print(f"\n🚨 ERROR 400 EN FRAME: {error_body}\n")
            except Exception:
                error_body = ""
            return {"error": f"http_{exc.code}", "message": error_body}
        except Exception as exc:
            return {"error": "request_failed", "message": str(exc)}

        try:
            response_json = json.loads(raw)
        except Exception:
            return {"error": "invalid_response_json", "message": raw}

        usage = response_json.get("usage") if isinstance(response_json, dict) else None
        output_text = self._extract_output_text(response_json)
        parsed = parse_jsonish_dict(output_text)
        if parsed:
            if isinstance(usage, dict):
                parsed = dict(parsed)
                parsed["__usage__"] = dict(usage)
            return parsed
        error_payload = {"error": "invalid_output_json", "message": output_text}
        if isinstance(usage, dict):
            error_payload["__usage__"] = dict(usage)
        return error_payload

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

        collected: List[str] = []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    collected.append(text_value)
        return "\n".join(collected).strip()

    # Construye user content.
    def _build_user_content(self, prompt: str, image_url: str) -> List[Dict[str, Any]]:
        image_part = {"type": "image_url", "image_url": {"url": image_url}}
        text_part = {"type": "text", "text": prompt}
        if self.prompt_order == "text_first":
            return [text_part, image_part]
        return [image_part, text_part]


# Verifica si un perfil remoto tiene API key disponible.
def profile_has_remote_api_key(profile: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(profile, dict):
        return False
    api_key_envs = tuple(profile.get("api_key_envs", (CODE_VISION_API_KEY_ENV_VAR,)))
    return bool(resolve_remote_api_key(api_key_envs))


# Compara si dos perfiles remotos representan el mismo proveedor.
def profiles_are_equivalent(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return (
        str(left.get("provider_name", "") or "").strip().lower(),
        str(left.get("model", "") or "").strip().lower(),
        str(left.get("base_url", "") or "").strip().rstrip("/").lower(),
    ) == (
        str(right.get("provider_name", "") or "").strip().lower(),
        str(right.get("model", "") or "").strip().lower(),
        str(right.get("base_url", "") or "").strip().rstrip("/").lower(),
    )


class AlternatingRemoteCodeVisionClient:
    # Inicializa la instancia y configura su estado interno.
    def __init__(
        self,
        primary: OpenAICodeVisionClient,
        secondary: Optional[OpenAICodeVisionClient] = None,
    ):
        self.primary = primary
        self.secondary = secondary
        self._lock = threading.Lock()
        self._editor_turn = 0
        if self.secondary is not None:
            print(
                "🌐 Editor alternation enabled: "
                f"{self.primary.provider_label} -> {self.secondary.provider_label} "
                f"every {CODE_EDITOR_SEND_INTERVAL_SECONDS:.2f}s"
            )

    # Maneja analyze sidebar dentro de AlternatingRemoteCodeVisionClient.
    def analyze_sidebar(self, task: CodeVisionTask) -> Dict[str, Any]:
        payload = self.primary.analyze_sidebar(task)
        return self._tag_payload(payload, self.primary, role="primary")

    # Maneja analyze editor dentro de AlternatingRemoteCodeVisionClient.
    def analyze_editor(self, task: CodeVisionTask) -> Dict[str, Any]:
        client, role = self._choose_editor_client()
        payload = client.analyze_editor(task)
        return self._tag_payload(payload, client, role=role)

    # Cierra recursos abiertos y fuerza el cierre ordenado del componente.
    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            if self.secondary is not None and self.secondary is not self.primary:
                self.secondary.close()

    # Selecciona editor client.
    def _choose_editor_client(self) -> Tuple[OpenAICodeVisionClient, str]:
        if self.secondary is None:
            return self.primary, "primary"
        with self._lock:
            turn = self._editor_turn
            self._editor_turn += 1
        if turn % 2 == 0:
            return self.primary, "primary"
        return self.secondary, "secondary"

    # Etiqueta payload.
    def _tag_payload(
        self,
        payload: Dict[str, Any],
        client: OpenAICodeVisionClient,
        role: str,
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        tagged = dict(payload)
        if role == "secondary":
            tagged["confidence"] = min(safe_float(tagged.get("confidence"), 0.0), 0.68)
        tagged["__provider__"] = {
            "name": client.provider_name,
            "label": client.provider_label,
            "model": client.model,
            "role": role,
        }
        return tagged


# Construye el cliente de vision de codigo configurado.
def build_code_vision_client() -> CodeVisionClient:
    provider = str(os.environ.get(CODE_VISION_PROVIDER_ENV_VAR, "") or "").strip().lower()
    enable_api = str(os.environ.get(CODE_VISION_ENABLE_API_ENV_VAR, "") or "").strip().lower()
    api_disabled = provider == "local" or enable_api in {"0", "false", "no", "off"}

    base_client: CodeVisionClient
    # El proveedor se selecciona por entorno para permitir experimentos sin editar
    # el código entre sesiones. Un valor vacío conserva el perfil configurado.
    if provider == "gemini":
        primary_profile = dict(CODE_VISION_PROFILE_GEMINI)
    elif provider == "openai":
        primary_profile = dict(CODE_VISION_PROFILE_OPENAI)
    else:
        primary_profile = dict(CODE_VISION_ACTIVE_PROFILE)
    # El cliente recibe un perfil explícito; por ello aplicamos aquí los
    # overrides de experimento antes de construirlo.
    model_override = str(
        os.environ.get(CODE_VISION_MODEL_ENV_VAR)
        or os.environ.get(OPENAI_MODEL_ENV_VAR)
        or ""
    ).strip()
    if model_override:
        primary_profile["model"] = model_override
    base_url_override = str(os.environ.get(CODE_VISION_BASE_URL_ENV_VAR) or "").strip()
    if base_url_override:
        primary_profile["base_url"] = base_url_override
    secondary_profile = dict(CODE_VISION_SECONDARY_PROFILE) if isinstance(CODE_VISION_SECONDARY_PROFILE, dict) else None
    if not profile_has_remote_api_key(primary_profile) and profile_has_remote_api_key(secondary_profile):
        primary_profile = dict(secondary_profile)
        secondary_profile = None

    if not api_disabled and profile_has_remote_api_key(primary_profile):
        secondary_client: Optional[OpenAICodeVisionClient] = None
        if (
            secondary_profile
            and profile_has_remote_api_key(secondary_profile)
            and not profiles_are_equivalent(primary_profile, secondary_profile)
        ):
            secondary_client = OpenAICodeVisionClient(profile=secondary_profile, role="secondary")
        base_client = AlternatingRemoteCodeVisionClient(
            OpenAICodeVisionClient(profile=primary_profile, role="primary"),
            secondary=secondary_client,
        )
    else:
        print("🧾 Code vision provider: OCR local")
        base_client = LocalOcrCodeVisionClient()

    if CODE_FLORENCE_ENABLE:
        return HybridCodeVisionClient(base_client, FlorenceCodeVisionHelper())
    return base_client
