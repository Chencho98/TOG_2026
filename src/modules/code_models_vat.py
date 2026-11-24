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

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Region:
    name: str
    x: int
    y: int
    w: int
    h: int
    color: Tuple[int, int, int]

    # Devuelve la region como tupla de coordenadas y tamano.
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class CodeViewportSegmentation:
    sidebar: Region
    editor: Region
    sidebar_crop: np.ndarray
    editor_crop: np.ndarray
    overlay: np.ndarray
    debug_sheet: np.ndarray
    text_mask: np.ndarray
    code_lines: np.ndarray


@dataclass
class CodeVisionTask:
    kind: str
    frame_number: int
    image: np.ndarray
    active_file_hint: Optional[str] = None
    language_hint: Optional[str] = None
    image_format: str = "jpg"
    jpeg_quality: int = 90
    meta: Dict[str, Any] = field(default_factory=dict)
    task_id: int = 0
    created_ts: float = field(default_factory=time.time)


@dataclass
class CodeVisionResponse:
    task: CodeVisionTask
    payload: Dict[str, Any]
    worker_name: str = ""
    queued_for_request_sec: float = 0.0
    request_duration_sec: float = 0.0
    assembled_delay_sec: float = 0.0
    completed_ts: float = 0.0


# Construye una respuesta de editor omitido con motivo.
def build_editor_skip_payload(task: CodeVisionTask, reason: str) -> Dict[str, Any]:
    frame_quality = str(task.meta.get("frame_quality") or "unknown")
    return {
        "active_file": sanitize_relative_code_path(task.active_file_hint),
        "language": task.language_hint,
        "start_line": None,
        "end_line": None,
        "sticky_context": [],
        "selection": None,
        "lines": [],
        "tab_title": None,
        "confidence": 0.0,
        "raw_text": "",
        "frame_quality": frame_quality,
        "skip_reason": str(reason or "skipped"),
    }


# Construye el prompt de analisis para el editor.
def build_editor_analysis_prompt(task: CodeVisionTask) -> str:
    prompt = (
        EDITOR_ANALYSIS_PROMPT_TEMPLATE
        .replace(EDITOR_ACTIVE_FILE_HINT_TOKEN, str(task.active_file_hint or "null"))
        .replace(EDITOR_LANGUAGE_HINT_TOKEN, str(task.language_hint or "null"))
    )
    frame_quality = str(task.meta.get("frame_quality") or "usable").strip().lower()
    quality_notes = {
        "usable": (
            "- calidad del frame: usable. Extrae el bloque principal y cualquier sticky context por separado.\n"
            "- No inventes caracteres, comentarios, signos ni indentación que no sean visibles."
        ),
        "partial_blur": (
            "- calidad del frame: partial_blur. Hay barrido parcial por scroll.\n"
            "- Extrae únicamente el texto totalmente legible; omite segmentos dudosos.\n"
            "- Si una línea está cortada o borrosa, no la completes."
        ),
        "scroll_blur": (
            "- calidad del frame: scroll_blur. La imagen está severamente barrida por scroll.\n"
            "- No reconstruyas el código; devuelve \"lines\": [] y no adivines contenido."
        ),
    }
    quality_note = quality_notes.get(frame_quality, quality_notes["usable"])
    return prompt + "\nContexto adicional:\n" + quality_note


@dataclass
class CodeLineObservation:
    line_number: int
    text: str
    confidence: float
    observations: int
    last_frame: int
    line_kind: str = "code"
    number_source: str = "unknown"
    source: str = "api"

    # Convierte la instancia a un diccionario serializable.
    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_number": int(self.line_number),
            "text": self.text,
            "confidence": float(self.confidence),
            "observations": int(self.observations),
            "last_frame": int(self.last_frame),
            "line_kind": self.line_kind,
            "number_source": self.number_source,
            "source": self.source,
        }


@dataclass
class CodeBlockObservation:
    frame_number: int
    file_path: Optional[str]
    language: Optional[str]
    start_line: Optional[int]
    end_line: Optional[int]
    confidence: float
    raw_text: str
    sticky_context: List[Dict[str, Any]] = field(default_factory=list)
    frame_quality: str = "usable"
    signature: Optional[str] = None
    lines: List[Dict[str, Any]] = field(default_factory=list)
    indent_levels: List[int] = field(default_factory=list)
    bracket_balance: int = 0
    estimated_visible_rows: int = 0
    source_provider: str = ""
    source_role: str = ""

    # Convierte la instancia a un diccionario serializable.
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": int(self.frame_number),
            "file_path": self.file_path,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "confidence": float(self.confidence),
            "raw_text": self.raw_text,
            "sticky_context": self.sticky_context,
            "frame_quality": self.frame_quality,
            "signature": self.signature,
            "lines": self.lines,
            "indent_levels": list(self.indent_levels),
            "bracket_balance": int(self.bracket_balance),
            "estimated_visible_rows": int(self.estimated_visible_rows),
            "source_provider": self.source_provider,
            "source_role": self.source_role,
        }


@dataclass
class PendingLineCandidate:
    key: str
    line_number: int
    text: str
    confidence: float
    number_source: str
    first_frame: int
    last_frame: int
    observations: int = 0
    frames: List[int] = field(default_factory=list)
    block_signatures: List[str] = field(default_factory=list)

    # Convierte la instancia a un diccionario serializable.
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "line_number": int(self.line_number),
            "text": self.text,
            "confidence": float(self.confidence),
            "number_source": self.number_source,
            "first_frame": int(self.first_frame),
            "last_frame": int(self.last_frame),
            "observations": int(self.observations),
            "frames": list(self.frames),
            "block_signatures": list(self.block_signatures),
        }


@dataclass
class SidebarTreeNode:
    label: str
    path: Optional[str]
    depth: int
    type: str
    selected: bool = False
    path_source: str = "none"
    children: List["SidebarTreeNode"] = field(default_factory=list)

    # Convierte la instancia a un diccionario serializable.
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "depth": int(self.depth),
            "type": self.type,
            "selected": bool(self.selected),
            "path_source": self.path_source,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class BufferedVisionCandidate:
    kind: str
    frame_number: int
    image: np.ndarray
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CodeFileNode:
    path: str
    language: Optional[str] = None
    lines: Dict[int, CodeLineObservation] = field(default_factory=dict)
    blocks: List[CodeBlockObservation] = field(default_factory=list)
    observations: int = 0
    last_frame: int = -1
    pending_lines: Dict[str, PendingLineCandidate] = field(default_factory=dict)
    block_observations: Dict[str, List[int]] = field(default_factory=dict)

    # Fusiona block.
    def merge_block(self, block: CodeBlockObservation, metadata_stale: bool = False) -> Dict[str, Any]:
        aligned_lines, alignment_offset, anchor_matches = self._align_block_lines(block.lines)
        if alignment_offset:
            if block.start_line is not None:
                block.start_line += alignment_offset
            if block.end_line is not None:
                block.end_line += alignment_offset
        block.lines = aligned_lines
        block.signature = self._compute_block_signature(block.lines)
        block_observation_count, duplicate_block = self._observe_block_signature(
            block.signature,
            int(block.frame_number),
        )
        validation = self._block_validation(block)
        required_confirmations = (
            max(CODE_PENDING_LINE_CONFIRMATION_FRAMES, CODE_EDITOR_SUSPICIOUS_CONFIRMATION_FRAMES)
            if validation.get("requires_confirmation")
            else CODE_PENDING_LINE_CONFIRMATION_FRAMES
        )

        self.observations += 1
        self.last_frame = max(self.last_frame, int(block.frame_number))
        if block.language and not self.language:
            self.language = block.language

        numbered_lines = [
            entry
            for entry in block.lines
            if should_keep_editor_line(entry.get("text", ""), safe_int(entry.get("number")))
            and safe_int(entry.get("number")) is not None
        ]
        reliable_numbered_count = sum(
            1
            for entry in numbered_lines
            if not self._is_weak_number_source(str(entry.get("number_source", "unknown") or "unknown"))
        )
        if self.lines and anchor_matches < CODE_LINE_ALIGNMENT_MIN_ANCHORS and reliable_numbered_count == 0:
            return {
                "alignment_offset": int(alignment_offset),
                "anchor_matches": int(anchor_matches),
                "inserted": 0,
                "updated": 0,
                "deduped": 0,
                "skipped": int(len(numbered_lines)),
                "rejected_reason": "weak_unanchored_block",
                "estimated_rows": int(validation.get("estimated_rows", 0)),
                "visible_span": int(validation.get("visible_span", 0)),
                "python_soft_score": float(validation.get("python_soft_score", 0.0)),
            }
        if validation.get("reject_reason"):
            return {
                "alignment_offset": int(alignment_offset),
                "anchor_matches": int(anchor_matches),
                "inserted": 0,
                "updated": 0,
                "deduped": 0,
                "skipped": int(len(numbered_lines)),
                "rejected_reason": str(validation.get("reject_reason")),
                "estimated_rows": int(validation.get("estimated_rows", 0)),
                "visible_span": int(validation.get("visible_span", 0)),
                "python_soft_score": float(validation.get("python_soft_score", 0.0)),
            }

        if not duplicate_block:
            self.blocks.append(block)
            if len(self.blocks) > CODE_BLOCK_HISTORY_LIMIT:
                self.blocks = self.blocks[-CODE_BLOCK_HISTORY_LIMIT:]

        inserted = 0
        updated = 0
        deduped = 0
        skipped = 0
        pending = 0
        confirmed = 0
        block_source = self._block_source_name(block)
        for entry in block.lines:
            line_number = safe_int(entry.get("number"))
            text = str(entry.get("text", "") or "").rstrip()
            confidence = safe_float(entry.get("confidence"), block.confidence)
            number_source = str(entry.get("number_source", "unknown") or "unknown")
            line_kind = str(entry.get("line_kind") or infer_editor_line_kind(text))
            if line_number is None or not should_keep_editor_line(text, line_number):
                continue

            snapped_line = self._find_nearby_matching_line(line_number, text)
            if snapped_line is not None and snapped_line != line_number:
                line_number = snapped_line
                deduped += 1
            elif self._is_weak_number_source(number_source) and anchor_matches < CODE_LINE_ALIGNMENT_MIN_ANCHORS:
                global_line = self._find_global_matching_line(text)
                if global_line is not None and global_line != line_number:
                    line_number = global_line
                    deduped += 1
                elif self.lines and not self._touches_known_line(line_number):
                    skipped += 1
                    continue

            existing = self.lines.get(line_number)
            if existing is None:
                candidate = self._observe_pending_line(
                    line_number=line_number,
                    text=text,
                    confidence=confidence,
                    number_source=number_source,
                    frame_number=int(block.frame_number),
                    block_signature=block.signature,
                )
                if candidate.observations < required_confirmations:
                    pending += 1
                    continue
                self._discard_pending_line(line_number, text)
                self.lines[line_number] = CodeLineObservation(
                    line_number=int(line_number),
                    text=candidate.text,
                    confidence=float(max(confidence, candidate.confidence)),
                    observations=1,
                    last_frame=int(block.frame_number),
                    line_kind=infer_editor_line_kind(candidate.text),
                    number_source=candidate.number_source,
                    source=block_source,
                )
                inserted += 1
                confirmed += 1
                continue

            self._discard_pending_line(line_number, text)
            existing_text_key = normalize_code_line_for_match(existing.text)
            new_text_key = normalize_code_line_for_match(text)
            if metadata_stale and existing_text_key and new_text_key and existing_text_key != new_text_key:
                existing.observations += 1
                existing.last_frame = max(existing.last_frame, int(block.frame_number))
                skipped += 1
                continue
            if (
                existing_text_key
                and new_text_key
                and existing_text_key != new_text_key
                and anchor_matches < CODE_LINE_ALIGNMENT_MIN_ANCHORS
                and self._is_weak_number_source(number_source)
                and not self._is_weak_number_source(existing.number_source)
            ):
                existing.observations += 1
                existing.last_frame = max(existing.last_frame, int(block.frame_number))
                skipped += 1
                continue

            existing_score = self._line_score(
                existing.text,
                existing.confidence,
                existing.number_source,
                existing.source,
            )
            new_score = self._line_score(text, confidence, number_source, block_source)
            if new_text_key == existing_text_key:
                preferred_text = choose_preferred_line_text(
                    existing.text,
                    text,
                    prefer_new=new_score >= existing_score,
                )
                if preferred_text != existing.text:
                    existing.text = preferred_text
                    existing.line_kind = infer_editor_line_kind(preferred_text)
                existing.confidence = max(float(existing.confidence), float(confidence))
                if self._number_source_rank(number_source) >= self._number_source_rank(existing.number_source):
                    existing.number_source = number_source
                if self._source_rank(block_source) >= self._source_rank(existing.source):
                    existing.source = block_source
                updated += 1
            elif lines_are_symbolic_variants(existing.text, text):
                preferred_text = choose_preferred_line_text(
                    existing.text,
                    text,
                    prefer_new=new_score >= existing_score,
                )
                if preferred_text != existing.text:
                    existing.text = preferred_text
                    existing.line_kind = infer_editor_line_kind(preferred_text)
                existing.confidence = max(float(existing.confidence), float(confidence))
                if self._number_source_rank(number_source) >= self._number_source_rank(existing.number_source):
                    existing.number_source = number_source
                if self._source_rank(block_source) >= self._source_rank(existing.source):
                    existing.source = block_source
                updated += 1
            elif new_score >= existing_score:
                existing.text = choose_preferred_line_text(existing.text, text, prefer_new=True)
                existing.confidence = max(float(existing.confidence), float(confidence))
                existing.line_kind = infer_editor_line_kind(existing.text)
                if self._number_source_rank(number_source) >= self._number_source_rank(existing.number_source):
                    existing.number_source = number_source
                existing.source = block_source
                updated += 1
            else:
                skipped += 1
            existing.observations += 1
            existing.last_frame = max(existing.last_frame, int(block.frame_number))

        return {
            "alignment_offset": int(alignment_offset),
            "anchor_matches": int(anchor_matches),
            "inserted": int(inserted),
            "updated": int(updated),
            "deduped": int(deduped),
            "skipped": int(skipped),
            "pending": int(pending),
            "confirmed": int(confirmed),
            "block_observations": int(block_observation_count),
            "duplicate_block": bool(duplicate_block),
            "estimated_rows": int(validation.get("estimated_rows", 0)),
            "visible_span": int(validation.get("visible_span", 0)),
            "python_soft_score": float(validation.get("python_soft_score", 0.0)),
            "requires_confirmation": bool(validation.get("requires_confirmation")),
        }

    # Convierte la instancia a un diccionario serializable.
    def to_dict(self) -> Dict[str, Any]:
        ordered_lines = [self.lines[line_no].to_dict() for line_no in sorted(self.lines)]
        return {
            "path": self.path,
            "language": self.language,
            "observations": int(self.observations),
            "last_frame": int(self.last_frame),
            "known_line_count": len(self.lines),
            "pending_line_count": len(self.pending_lines),
            "lines": ordered_lines,
            "blocks": [block.to_dict() for block in self.blocks[-50:]],
        }

    # Maneja render sparse text dentro de CodeFileNode.
    def render_sparse_text(self) -> str:
        if not self.lines:
            return ""

        rendered: List[str] = []
        previous_line: Optional[int] = None
        previous_observation: Optional[CodeLineObservation] = None
        for line_number in sorted(self.lines):
            current_observation = self.lines[line_number]
            if previous_line is None and line_number > 1 and self._line_number_is_reliable(current_observation):
                rendered.extend([""] * (line_number - 1))
            if previous_line is not None:
                missing_count = line_number - previous_line - 1
                if (
                    missing_count >= CODE_RENDER_MISSING_MIN_LINE_COUNT
                    and previous_observation is not None
                    and self._gap_is_reliable(previous_observation, current_observation)
                ):
                    rendered.extend([""] * missing_count)
            rendered.append(current_observation.text)
            previous_line = line_number
            previous_observation = current_observation
        return "\n".join(rendered).rstrip() + "\n"

    # Alinea block lines.
    def _align_block_lines(
        self,
        entries: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        offsets: Dict[int, int] = {}
        for entry in entries:
            line_number = safe_int(entry.get("number"))
            text_key = normalize_code_line_for_match(entry.get("text", ""))
            if line_number is None or not self._is_alignment_anchor_text(text_key):
                continue

            start = max(1, int(line_number) - CODE_LINE_ALIGNMENT_WINDOW)
            end = int(line_number) + CODE_LINE_ALIGNMENT_WINDOW
            for candidate_line in range(start, end + 1):
                existing = self.lines.get(candidate_line)
                if existing is None:
                    continue
                if normalize_code_line_for_match(existing.text) != text_key:
                    continue
                offset = int(candidate_line) - int(line_number)
                offsets[offset] = offsets.get(offset, 0) + 1

        best_offset = 0
        best_matches = 0
        for offset, match_count in offsets.items():
            if match_count > best_matches or (
                match_count == best_matches and abs(offset) < abs(best_offset)
            ):
                best_offset = int(offset)
                best_matches = int(match_count)

        if best_matches < CODE_LINE_ALIGNMENT_MIN_ANCHORS:
            return [dict(entry) for entry in entries], 0, best_matches

        aligned_entries: List[Dict[str, Any]] = []
        for entry in entries:
            aligned_entry = dict(entry)
            line_number = safe_int(aligned_entry.get("number"))
            if line_number is not None:
                aligned_entry["number"] = int(line_number) + best_offset
            aligned_entries.append(aligned_entry)
        return aligned_entries, best_offset, best_matches

    # Maneja find nearby matching line dentro de CodeFileNode.
    def _find_nearby_matching_line(self, line_number: int, text: str) -> Optional[int]:
        text_key = normalize_code_line_for_match(text)
        if not text_key:
            return None
        if text_key == BLANK_EDITOR_LINE_SENTINEL:
            existing = self.lines.get(int(line_number))
            if existing is None:
                return None
            return int(line_number) if normalize_code_line_for_match(existing.text) == text_key else None

        best_line: Optional[int] = None
        best_distance = CODE_LINE_DEDUP_WINDOW + 1
        start = max(1, int(line_number) - CODE_LINE_DEDUP_WINDOW)
        end = int(line_number) + CODE_LINE_DEDUP_WINDOW
        for candidate_line in range(start, end + 1):
            existing = self.lines.get(candidate_line)
            if existing is None:
                continue
            if normalize_code_line_for_match(existing.text) != text_key:
                continue
            distance = abs(int(candidate_line) - int(line_number))
            if best_line is None or distance < best_distance:
                best_line = int(candidate_line)
                best_distance = distance
        return best_line

    # Maneja find global matching line dentro de CodeFileNode.
    def _find_global_matching_line(self, text: str) -> Optional[int]:
        text_key = normalize_code_line_for_match(text)
        if not self._is_distinctive_line_text(text_key):
            return None

        matches = [
            int(line_number)
            for line_number, observation in self.lines.items()
            if normalize_code_line_for_match(observation.text) == text_key
        ]
        if len(matches) != 1:
            return None
        return matches[0]

    # Maneja touches known line dentro de CodeFileNode.
    def _touches_known_line(self, line_number: int) -> bool:
        if not self.lines:
            return True
        start = max(1, int(line_number) - CODE_LINE_ATTACH_WINDOW)
        end = int(line_number) + CODE_LINE_ATTACH_WINDOW
        return any(candidate in self.lines for candidate in range(start, end + 1))

    # Maneja gap is reliable dentro de CodeFileNode.
    def _gap_is_reliable(
        self,
        left: CodeLineObservation,
        right: CodeLineObservation,
    ) -> bool:
        return (
            self._line_number_source_is_reliable(left.number_source)
            and self._line_number_source_is_reliable(right.number_source)
        )

    # Evalua si la posicion de una linea es confiable para renderizar huecos.
    def _line_number_is_reliable(self, observation: CodeLineObservation) -> bool:
        return self._line_number_source_is_reliable(observation.number_source)

    # Evalua si la fuente del numero de linea es suficientemente confiable.
    def _line_number_source_is_reliable(self, number_source: str) -> bool:
        return str(number_source or "") in {
            "explicit",
            "embedded_split",
            "interpolated",
            "florence_anchor",
            "start_line",
        }

    # Evalua la condicion is alignment anchor text y devuelve el resultado booleano.
    def _is_alignment_anchor_text(self, text_key: str) -> bool:
        return len(text_key) >= 5 and any(char.isalpha() for char in text_key)

    # Evalua la condicion is distinctive line text y devuelve el resultado booleano.
    def _is_distinctive_line_text(self, text_key: str) -> bool:
        return len(text_key) >= CODE_GLOBAL_DEDUP_MIN_TEXT_LENGTH and any(char.isalpha() for char in text_key)

    # Evalua la condicion is weak number source y devuelve el resultado booleano.
    def _is_weak_number_source(self, number_source: str) -> bool:
        return not self._line_number_source_is_reliable(number_source)

    # Maneja python block soft score dentro de CodeFileNode.
    def _python_block_soft_score(self, lines: List[Dict[str, Any]]) -> float:
        if not lines:
            return 0.0
        scores = [python_line_soft_score(entry.get("text", "")) for entry in lines if str(entry.get("text", "")).strip()]
        if not scores:
            return 0.0
        return float(sum(scores) / float(len(scores)))

    # Maneja block validation dentro de CodeFileNode.
    def _block_validation(self, block: CodeBlockObservation) -> Dict[str, Any]:
        explicit_numbers = [safe_int(entry.get("number")) for entry in block.lines]
        explicit_numbers = [number for number in explicit_numbers if number is not None]
        estimated_rows = max(
            CODE_EDITOR_MIN_VISIBLE_ROWS,
            int(block.estimated_visible_rows or len(explicit_numbers) or len(block.lines) or CODE_EDITOR_MIN_VISIBLE_ROWS),
        )
        max_visible_span = (
            estimated_rows * CODE_EDITOR_MAX_VISIBLE_LINE_RANGE_MULTIPLIER
            + CODE_EDITOR_MAX_VISIBLE_LINE_RANGE_PADDING
        )
        visible_span = 0 if not explicit_numbers else (max(explicit_numbers) - min(explicit_numbers))
        suspicious_lines = sum(
            1
            for entry in block.lines
            if (
                not is_blank_editor_line_text(entry.get("text", ""))
                and code_like_text_score(entry.get("text", ""), block.language) < -0.10
            )
        )
        python_soft_score = (
            self._python_block_soft_score(block.lines)
            if str(block.language or "").upper() == "PYTHON"
            else 0.0
        )

        reject_reason: Optional[str] = None
        if explicit_numbers and min(explicit_numbers) < 1:
            reject_reason = "invalid_line_number"
        elif explicit_numbers and visible_span > max_visible_span:
            reject_reason = "implausible_visible_range"

        requires_confirmation = bool(
            block.frame_quality != "usable"
            or suspicious_lines > 0
            or python_soft_score < -0.05
        )
        return {
            "estimated_rows": int(estimated_rows),
            "visible_span": int(visible_span),
            "max_visible_span": int(max_visible_span),
            "suspicious_lines": int(suspicious_lines),
            "python_soft_score": float(python_soft_score),
            "reject_reason": reject_reason,
            "requires_confirmation": requires_confirmation,
        }

    # Maneja line score dentro de CodeFileNode.
    def _line_score(self, text: str, confidence: float, number_source: str, source: str) -> float:
        language_hint = self.language
        if not language_hint and path_extension(self.path) == ".py":
            language_hint = "PYTHON"
        return (
            float(confidence)
            + self._number_source_rank(number_source)
            + self._source_rank(source)
            + code_like_text_score(text, language_hint)
        )

    # Maneja number source rank dentro de CodeFileNode.
    def _number_source_rank(self, number_source: str) -> float:
        ranking = {
            "explicit": 0.30,
            "embedded_split": 0.24,
            "interpolated": 0.22,
            "florence_anchor": 0.20,
            "start_line": 0.12,
            "anchor_inferred": 0.05,
            "raw_text": -0.05,
            "unknown": 0.0,
            "missing": -0.10,
        }
        return float(ranking.get(str(number_source or "unknown"), 0.0))

    # Maneja source rank dentro de CodeFileNode.
    def _source_rank(self, source: str) -> float:
        normalized = str(source or "unknown").strip().lower()
        ranking = {
            "openai_primary": 0.16,
            "openai": 0.14,
            "gemini_primary": 0.12,
            "gemini_secondary": -0.02,
            "gemini": 0.0,
            "florence": -0.08,
            "fallback": -0.10,
            "local_ocr": -0.12,
            "api": 0.06,
            "unknown": 0.0,
        }
        return float(ranking.get(normalized, ranking.get(normalized.split("_", 1)[0], 0.0)))

    # Maneja block source name dentro de CodeFileNode.
    def _block_source_name(self, block: CodeBlockObservation) -> str:
        provider = str(block.source_provider or "").strip().lower()
        role = str(block.source_role or "").strip().lower()
        if provider and role:
            return f"{provider}_{role}"
        if provider:
            return provider
        if role:
            return role
        return "api" if block.file_path else "fallback"

    # Calcula block signature.
    def _compute_block_signature(self, entries: List[Dict[str, Any]]) -> Optional[str]:
        parts: List[str] = []
        for index, entry in enumerate(entries):
            text_key = normalize_code_line_for_match(entry.get("text", ""))
            if not text_key:
                continue
            parts.append(f"{index}:{text_key}")
        if not parts:
            return None
        payload = "\n".join(parts).encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()[:20]

    # Maneja observe block signature dentro de CodeFileNode.
    def _observe_block_signature(self, signature: Optional[str], frame_number: int) -> Tuple[int, bool]:
        self._prune_block_observations(frame_number)
        if not signature:
            return 0, False
        frames = list(self.block_observations.get(signature, []))
        if not frames or frames[-1] != int(frame_number):
            frames.append(int(frame_number))
        self.block_observations[signature] = frames[-CODE_PENDING_LINE_CONFIRMATION_FRAMES - 2 :]
        return len(self.block_observations[signature]), len(self.block_observations[signature]) > 1

    # Descarta block observations.
    def _prune_block_observations(self, current_frame: int) -> None:
        if current_frame < 0:
            return
        stale_signatures = [
            signature
            for signature, frames in self.block_observations.items()
            if not frames or (int(current_frame) - int(frames[-1])) > CODE_BLOCK_DUPLICATE_WINDOW_FRAMES
        ]
        for signature in stale_signatures:
            self.block_observations.pop(signature, None)

    # Maneja pending line key dentro de CodeFileNode.
    def _pending_line_key(self, line_number: int, text: str) -> str:
        return f"{int(line_number)}::{normalize_code_line_for_match(text)}"

    # Resuelve pending line key.
    def _resolve_pending_line_key(self, line_number: int, text: str) -> str:
        direct_key = self._pending_line_key(line_number, text)
        if direct_key in self.pending_lines:
            return direct_key
        text_key = normalize_code_line_for_match(text)
        if not text_key or text_key == BLANK_EDITOR_LINE_SENTINEL:
            return direct_key
        for candidate_line in range(
            max(1, int(line_number) - CODE_LINE_DEDUP_WINDOW),
            int(line_number) + CODE_LINE_DEDUP_WINDOW + 1,
        ):
            nearby_key = f"{int(candidate_line)}::{text_key}"
            if nearby_key in self.pending_lines:
                return nearby_key
        return direct_key

    # Maneja observe pending line dentro de CodeFileNode.
    def _observe_pending_line(
        self,
        line_number: int,
        text: str,
        confidence: float,
        number_source: str,
        frame_number: int,
        block_signature: Optional[str],
    ) -> PendingLineCandidate:
        self._prune_pending_lines(frame_number)
        key = self._resolve_pending_line_key(line_number, text)
        candidate = self.pending_lines.get(key)
        if candidate is None:
            candidate = PendingLineCandidate(
                key=key,
                line_number=int(line_number),
                text=text,
                confidence=float(confidence),
                number_source=str(number_source or "unknown"),
                first_frame=int(frame_number),
                last_frame=int(frame_number),
                observations=0,
            )
            self.pending_lines[key] = candidate
        candidate.line_number = int(line_number)
        candidate.text = text
        candidate.confidence = max(float(candidate.confidence), float(confidence))
        if self._number_source_rank(number_source) >= self._number_source_rank(candidate.number_source):
            candidate.number_source = str(number_source or "unknown")
        if not candidate.frames or candidate.frames[-1] != int(frame_number):
            candidate.frames.append(int(frame_number))
            candidate.observations += 1
        if block_signature and block_signature not in candidate.block_signatures:
            candidate.block_signatures.append(block_signature)
        candidate.last_frame = int(frame_number)
        candidate.frames = candidate.frames[-CODE_PENDING_LINE_CONFIRMATION_FRAMES - 2 :]
        candidate.block_signatures = candidate.block_signatures[-6:]
        return candidate

    # Maneja discard pending line dentro de CodeFileNode.
    def _discard_pending_line(self, line_number: int, text: str) -> None:
        key = self._resolve_pending_line_key(line_number, text)
        self.pending_lines.pop(key, None)

    # Descarta pending lines.
    def _prune_pending_lines(self, current_frame: int) -> None:
        if current_frame < 0:
            return
        stale_keys = [
            key
            for key, candidate in self.pending_lines.items()
            if (int(current_frame) - int(candidate.last_frame)) > CODE_PENDING_LINE_MAX_AGE_FRAMES
        ]
        for key in stale_keys:
            self.pending_lines.pop(key, None)
