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

try:
    from src.modules.code_vision_vat import split_sticky_context_prefix
except ImportError:
    try:
        from code_vision_vat import split_sticky_context_prefix
    except ImportError:
        from .code_vision_vat import split_sticky_context_prefix

# ============================================================================
# CODE KNOWLEDGE STORE
# ============================================================================

class CodeKnowledgeStore:
    # Inicializa la instancia y configura su estado interno.
    def __init__(self, debug_logger: Optional[Callable[[str], None]] = None):
        self._lock = threading.Lock()
        self._debug_logger = debug_logger
        self._files: Dict[str, CodeFileNode] = {}
        self._path_aliases: Dict[str, str] = {}
        self._sidebar_history: List[Dict[str, Any]] = []
        self._editor_history: List[Dict[str, Any]] = []
        self._sidebar_tree: List[Dict[str, Any]] = []
        self._visible_file_paths: set[str] = set()
        self._visible_folder_paths: set[str] = set()
        self._visible_file_paths_by_basename: Dict[str, List[str]] = {}
        self._provisional_file_paths: set[str] = set()
        self._temporary_file_paths: set[str] = set()
        self._active_file: Optional[str] = None
        self._last_editor_file_path: Optional[str] = None
        self._last_editor_file_frame = -1
        self._temp_file_counter = 0
        self._structure_version = 0
        self._sidebar_signature: Optional[str] = None
        self._last_sidebar_metadata_frame = -1
        self._last_editor_metadata_frame = -1
        self._last_active_file_frame = -1
        self._last_change_count = 0
        self._last_flush_ts = time.time()
        self._session_dir: Optional[Path] = None
        self._summary_path: Optional[Path] = None
        self._render_root: Optional[Path] = None

    # Configura los directorios de salida de la sesion.
    def set_session_dir(self, session_dir: Path) -> None:
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "responses" / "sidebar").mkdir(parents=True, exist_ok=True)
        (session_dir / "responses" / "editor").mkdir(parents=True, exist_ok=True)
        (session_dir / "rendered").mkdir(parents=True, exist_ok=True)
        self._session_dir = session_dir
        self._summary_path = session_dir / "repository_summary.json"
        self._render_root = session_dir / "rendered"

    # Consulta active file sin modificar estado.
    def peek_active_file(self) -> Optional[str]:
        with self._lock:
            return self._active_file

    # Maneja snapshot dentro de CodeKnowledgeStore.
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            active_node = self._files.get(self._active_file) if self._active_file else None
            active_pending_lines = 0 if active_node is None else len(active_node.pending_lines)
            return {
                "active_file": self._active_file,
                "editor_prompt_hint": self._current_editor_fallback_hint_locked(),
                "known_files": len(self._files),
                "structure_version": int(self._structure_version),
                "sidebar_signature": self._sidebar_signature,
                "visible_files": len(self._visible_file_paths),
                "provisional_files": len(self._provisional_file_paths),
                "active_pending_lines": int(active_pending_lines),
                "needs_editor_refresh": bool(
                    self._active_file is None
                    or active_pending_lines > 0
                    or bool(self._provisional_file_paths)
                ),
            }

    # Aplica sidebar result.
    def apply_sidebar_result(self, response: CodeVisionResponse) -> Dict[str, Any]:
        payload = response.payload if isinstance(response.payload, dict) else {}
        frame_number = int(response.task.frame_number)
        raw_entries = self._normalize_sidebar_entries(payload.get("entries"))
        sidebar_tree_nodes, entries, sidebar_flat_nodes = self._build_sidebar_tree(raw_entries)
        candidate_active_file = sanitize_relative_code_path(payload.get("active_file")) or response.task.active_file_hint
        signature = response.task.meta.get("signature")
        debug_info = self._response_debug_info(response)
        stale_metadata = False
        active_file_applied = False
        active_file: Optional[str] = None
        materialized_outputs: Dict[str, str] = {}

        with self._lock:
            stale_metadata = frame_number < self._last_sidebar_metadata_frame
            if entries and not stale_metadata:
                self._set_sidebar_visibility_locked(sidebar_tree_nodes, entries)
            active_file = self._resolve_file_path_locked(
                candidate_active_file,
                hint=response.task.active_file_hint,
                stale=stale_metadata,
                allow_create=self._allow_sidebar_file_creation_locked(candidate_active_file),
            )
            active_file = self._maybe_promote_last_editor_temp_file_locked(
                active_file,
                stale=stale_metadata,
            )
            if active_file:
                self._files.setdefault(active_file, CodeFileNode(path=active_file))
                self._remember_editor_file_locked(active_file, frame_number)
                if frame_number >= self._last_active_file_frame:
                    self._active_file = active_file
                    self._last_active_file_frame = frame_number
                    active_file_applied = True

            if entries:
                for entry in entries:
                    path = self._resolve_file_path_locked(
                        entry.get("path"),
                        hint=active_file or response.task.active_file_hint,
                        stale=stale_metadata,
                        allow_create=True,
                    ) if entry.get("type") == "file" else None
                    entry["path"] = path
                    if path and entry.get("type") == "file":
                        self._files.setdefault(path, CodeFileNode(path=path))
                for entry, node in zip(entries, sidebar_flat_nodes):
                    node.path = entry.get("path")
                    node.path_source = str(entry.get("path_source", node.path_source) or node.path_source)
                if not stale_metadata:
                    self._structure_version += 1
                    self._set_sidebar_visibility_locked(sidebar_tree_nodes, entries)
                released_paths = self._release_provisional_visible_files_locked(
                    [
                        entry.get("path")
                        for entry in entries
                        if entry.get("type") == "file"
                    ]
                )
                for released_path in released_paths:
                    node = self._files.get(released_path)
                    if node is not None:
                        materialized_outputs[released_path] = node.render_sparse_text()
            if active_file:
                released_paths = self._release_provisional_visible_files_locked([active_file])
                for released_path in released_paths:
                    node = self._files.get(released_path)
                    if node is not None:
                        materialized_outputs[released_path] = node.render_sparse_text()

            if not stale_metadata:
                self._sidebar_signature = signature or self._sidebar_signature
                self._last_sidebar_metadata_frame = frame_number
            history_entry = {
                "frame_number": frame_number,
                "task_id": int(response.task.task_id),
                "active_file": self._active_file,
                "entries": entries,
                "signature": self._sidebar_signature,
                "meta": dict(response.task.meta),
                "debug": debug_info,
                "visible_files": len(self._visible_file_paths),
                "stale_metadata": bool(stale_metadata),
                "active_file_applied": bool(active_file_applied),
            }
            self._sidebar_history.append(history_entry)
            if len(self._sidebar_history) > 200:
                self._sidebar_history = self._sidebar_history[-200:]
            self._last_change_count += 1

        for path, rendered in materialized_outputs.items():
            if rendered:
                self._write_rendered_file(path, rendered)
            else:
                self._touch_render_path(path)
        self._persist_response("sidebar", frame_number, history_entry)
        self._flush_if_needed()
        self._debug(
            "store sidebar "
            f"task={response.task.task_id} frame={frame_number} entries={len(entries)} "
            f"stale={stale_metadata} active_applied={active_file_applied} "
            f"active={active_file or self._active_file or '-'}"
        )
        return {
            "frame_number": frame_number,
            "task_id": int(response.task.task_id),
            "entry_count": len(entries),
            "stale_metadata": bool(stale_metadata),
            "active_file": active_file or self.peek_active_file(),
            "active_file_applied": bool(active_file_applied),
        }

    # Aplica editor result.
    def apply_editor_result(self, response: CodeVisionResponse) -> Dict[str, Any]:
        payload = response.payload if isinstance(response.payload, dict) else {}
        frame_number = int(response.task.frame_number)
        raw_active_file = sanitize_relative_code_path(payload.get("active_file"))
        raw_tab_title = sanitize_relative_code_path(payload.get("tab_title"))
        candidate_active_file = (
            raw_active_file
            or raw_tab_title
            or response.task.active_file_hint
        )
        language = payload.get("language") or response.task.language_hint
        frame_quality = str(
            payload.get("frame_quality")
            or response.task.meta.get("frame_quality")
            or "usable"
        )
        debug_info = self._response_debug_info(response)
        payload_quality = payload.get("__quality__") if isinstance(payload.get("__quality__"), dict) else {}
        provider_info = payload.get("__provider__") if isinstance(payload.get("__provider__"), dict) else {}

        normalized_sticky_context = self._normalize_editor_line_group(payload, field_name="sticky_context")
        normalized_lines = self._normalize_editor_line_group(payload, field_name="lines")
        if not normalized_sticky_context and normalized_lines:
            normalized_sticky_context, normalized_lines = split_sticky_context_prefix(normalized_lines)
        start_line = safe_int(payload.get("start_line"))
        end_line = safe_int(payload.get("end_line"))
        raw_text = str(payload.get("raw_text", "") or "")
        skip_reason = str(payload.get("skip_reason", "") or "").strip()

        if not normalized_lines and raw_text and not skip_reason:
            normalized_lines = self._build_line_entries_from_raw_text(raw_text, start_line)

        explicit_numbers = [safe_int(entry.get("number")) for entry in normalized_lines]
        explicit_numbers = [number for number in explicit_numbers if number is not None]
        if explicit_numbers:
            body_start_line = min(explicit_numbers)
            body_end_line = max(explicit_numbers)
            if start_line is None or (normalized_sticky_context and safe_int(start_line) is not None and int(start_line) < int(body_start_line)):
                start_line = body_start_line
            if end_line is None or (safe_int(end_line) is not None and int(end_line) < int(body_end_line)):
                end_line = body_end_line

        if not normalized_lines and (not raw_text or skip_reason):
            active_file_applied = False
            stale_editor_metadata = False
            active_file: Optional[str] = None
            provisional_file = False
            with self._lock:
                editor_hint = self._current_editor_fallback_hint_locked() or response.task.active_file_hint
                stale_editor_metadata = frame_number < self._last_editor_metadata_frame
                active_file = self._resolve_editor_file_path_locked(
                    active_file=raw_active_file,
                    tab_title=raw_tab_title,
                    hint=editor_hint,
                    stale=stale_editor_metadata,
                )
                if active_file:
                    self._files.setdefault(active_file, CodeFileNode(path=active_file))
                    provisional_file = self._mark_editor_file_visibility_locked(active_file)
                    self._remember_editor_file_locked(active_file, frame_number)
                    if frame_number >= self._last_active_file_frame:
                        self._active_file = active_file
                        self._last_active_file_frame = frame_number
                        active_file_applied = True
                if frame_number >= self._last_editor_metadata_frame:
                    self._last_editor_metadata_frame = frame_number
            self._debug(
                "store editor-empty "
                f"task={response.task.task_id} frame={frame_number} active={active_file or '-'} "
                f"stale={stale_editor_metadata} quality={frame_quality} "
                f"sticky={len(normalized_sticky_context)} active_applied={active_file_applied} "
                f"reason={skip_reason or '-'}"
            )
            return {
                "frame_number": frame_number,
                "task_id": int(response.task.task_id),
                "active_file": active_file,
                "line_count": 0,
                "sticky_count": len(normalized_sticky_context),
                "frame_quality": frame_quality,
                "stale_metadata": bool(stale_editor_metadata),
                "active_file_applied": bool(active_file_applied),
                "provisional_file": bool(provisional_file),
                "rendered": False,
                "skip_reason": skip_reason or None,
            }

        indent_levels = self._compute_indent_levels(normalized_lines)
        bracket_balance = self._compute_bracket_balance(normalized_lines)
        rendered_text = ""
        stale_editor_metadata = False
        active_file_applied = False
        merge_info: Dict[str, Any] = {}
        history_entry: Optional[Dict[str, Any]] = None
        active_file: Optional[str] = None
        block: Optional[CodeBlockObservation] = None
        provisional_file = False
        with self._lock:
            editor_hint = self._current_editor_fallback_hint_locked() or response.task.active_file_hint
            stale_editor_metadata = frame_number < self._last_editor_metadata_frame
            active_file = self._resolve_editor_file_path_locked(
                active_file=raw_active_file,
                tab_title=raw_tab_title,
                hint=editor_hint,
                stale=stale_editor_metadata,
            )
            if active_file:
                active_file = self._maybe_promote_last_editor_temp_file_locked(
                    active_file,
                    stale=stale_editor_metadata,
                )
                self._files.setdefault(active_file, CodeFileNode(path=active_file))
                provisional_file = self._mark_editor_file_visibility_locked(active_file)
                self._remember_editor_file_locked(active_file, frame_number)
            if not active_file and stale_editor_metadata:
                merge_info = {"rejected_reason": "uncertain_file_identity"}
            else:
                if not active_file:
                    active_file = self._get_or_create_editor_temp_file_locked(language)
                    provisional_file = self._mark_editor_file_visibility_locked(active_file)
                    self._remember_editor_file_locked(active_file, frame_number)
                    merge_info = {"temporary_file": True}

                if active_file:
                    block = CodeBlockObservation(
                        frame_number=frame_number,
                        file_path=active_file,
                        language=language,
                        start_line=start_line,
                        end_line=end_line,
                        confidence=float(safe_float(payload.get("confidence"), 0.0)),
                        raw_text=raw_text or "\n".join(str(entry.get("text", "")) for entry in normalized_lines),
                        sticky_context=normalized_sticky_context,
                        frame_quality=frame_quality,
                        lines=normalized_lines,
                        indent_levels=indent_levels,
                        bracket_balance=bracket_balance,
                        estimated_visible_rows=int(
                            safe_int(payload_quality.get("estimated_visible_rows"))
                            or safe_int(response.task.meta.get("estimated_visible_rows"))
                            or 0
                        ),
                        source_provider=str(provider_info.get("name", "") or "").strip().lower(),
                        source_role=str(provider_info.get("role", "") or "").strip().lower(),
                    )

                    node = self._files.setdefault(active_file, CodeFileNode(path=active_file))
                    if language and not node.language:
                        node.language = language
                    merge_result = node.merge_block(block, metadata_stale=stale_editor_metadata)
                    merge_info.update(merge_result)
                    merge_info["provisional_file"] = bool(provisional_file)
                    rendered_text = node.render_sparse_text()
                    if not self._is_temporary_file_path_locked(active_file):
                        if frame_number >= self._last_active_file_frame:
                            self._active_file = active_file
                            self._last_active_file_frame = frame_number
                            active_file_applied = True
                    if frame_number >= self._last_editor_metadata_frame:
                        self._last_editor_metadata_frame = frame_number

            history_entry = {
                "frame_number": frame_number,
                "task_id": int(response.task.task_id),
                "active_file": active_file,
                "language": language,
                "start_line": None if block is None else block.start_line,
                "end_line": None if block is None else block.end_line,
                "confidence": float(safe_float(payload.get("confidence"), 0.0) if block is None else block.confidence),
                "line_count": len(normalized_lines),
                "sticky_count": len(normalized_sticky_context),
                "frame_quality": frame_quality,
                "debug": debug_info,
                "stale_metadata": bool(stale_editor_metadata),
                "active_file_applied": bool(active_file_applied),
                "provisional_file": bool(provisional_file),
                "merge": dict(merge_info),
            }
            self._editor_history.append(history_entry)
            if len(self._editor_history) > 500:
                self._editor_history = self._editor_history[-500:]
            self._last_change_count += 1

        if block is not None:
            block_payload = block.to_dict()
            block_payload["debug"] = debug_info
            block_payload["stale_metadata"] = bool(stale_editor_metadata)
            block_payload["active_file_applied"] = bool(active_file_applied)
            block_payload["merge"] = dict(merge_info)
            self._persist_response("editor", frame_number, block_payload)
            self._write_rendered_file(active_file, rendered_text)
            self.flush_incremental()
            self._last_change_count = 0
            self._last_flush_ts = time.time()
            self._debug(
                "store editor "
                f"task={response.task.task_id} frame={frame_number} file={active_file} "
                f"lines={len(normalized_lines)} stale={stale_editor_metadata} "
                f"quality={frame_quality} sticky={len(normalized_sticky_context)} "
                f"align={merge_info.get('alignment_offset', 0)} anchors={merge_info.get('anchor_matches', 0)} "
                f"inserted={merge_info.get('inserted', 0)} updated={merge_info.get('updated', 0)} "
                f"deduped={merge_info.get('deduped', 0)} skipped={merge_info.get('skipped', 0)} "
                f"active_applied={active_file_applied}"
            )
        else:
            self._persist_response("editor", frame_number, history_entry or {})
            self._flush_if_needed()
            self._debug(
                "store editor-skip "
                f"task={response.task.task_id} frame={frame_number} "
                f"file={candidate_active_file or '-'} stale={stale_editor_metadata} "
                f"quality={frame_quality} sticky={len(normalized_sticky_context)} "
                f"reason={merge_info.get('rejected_reason', 'unknown')}"
            )
        return {
            "frame_number": frame_number,
            "task_id": int(response.task.task_id),
            "active_file": active_file,
            "line_count": len(normalized_lines),
            "sticky_count": len(normalized_sticky_context),
            "frame_quality": frame_quality,
            "stale_metadata": bool(stale_editor_metadata),
            "active_file_applied": bool(active_file_applied),
            "rendered": bool(rendered_text),
            "merge": dict(merge_info),
        }

    # Finaliza el procesamiento pendiente y devuelve el resumen producido.
    def finalize(self) -> Dict[str, Any]:
        summary = self._build_summary()
        summary_path = None
        render_root = None
        reference_comparison: List[Dict[str, Any]] = []
        reference_comparison_path = None
        if self._summary_path is not None:
            self._write_summary_json(summary)
            summary_path = self._summary_path

        if self._render_root is not None:
            self._render_root.mkdir(parents=True, exist_ok=True)
            render_root = self._render_root
            self._materialize_sidebar_structure()
            rendered_outputs: Dict[str, str] = {}
            for path, node in sorted(self._files.items()):
                rendered = node.render_sparse_text()
                rendered_outputs[path] = rendered
                if rendered:
                    self._write_rendered_file(path, rendered)
                elif path in self._visible_file_paths:
                    self._touch_render_path(path)
            reference_comparison = self._build_reference_comparison_report(
                rendered_outputs
            )
            reference_comparison_path = self._write_reference_comparison_json(
                reference_comparison
            )

        if reference_comparison:
            summary["reference_comparison"] = reference_comparison
            summary["reference_comparison_json"] = reference_comparison_path
            if self._summary_path is not None:
                self._write_summary_json(summary)

        return {
            "code_repository_json": None if summary_path is None else str(summary_path),
            "code_repository_root": None if render_root is None else str(render_root),
            "code_known_files": len(summary.get("files", {})),
            "code_active_file": summary.get("active_file"),
            "code_reference_comparison_json": reference_comparison_path,
            "code_reference_comparison": reference_comparison,
        }

    # Maneja flush incremental dentro de CodeKnowledgeStore.
    def flush_incremental(self) -> None:
        if self._summary_path is None:
            return
        summary = self._build_summary()
        self._write_summary_json(summary)
        self._materialize_sidebar_structure()

    # Resuelve file path locked.
    def _resolve_file_path_locked(
        self,
        candidate: Optional[str],
        hint: Optional[str] = None,
        stale: bool = False,
        allow_create: bool = False,
    ) -> Optional[str]:
        candidate = sanitize_relative_code_path(candidate)
        hint = sanitize_relative_code_path(hint)

        if candidate:
            candidate = self._path_aliases.get(candidate, candidate)
        if hint:
            hint = self._path_aliases.get(hint, hint)

        if candidate and candidate in self._files:
            self._path_aliases[candidate] = candidate
            return candidate

        if hint and hint in self._files:
            if not candidate or candidate == hint:
                return hint
            if path_basename(candidate).lower() == path_basename(hint).lower():
                self._path_aliases[candidate] = hint
                return hint
            if self._looks_like_fuzzy_path_match(candidate, hint):
                self._path_aliases[candidate] = hint
                return hint
            if (
                candidate
                and not stale
                and self._is_temporary_file_path_locked(hint)
                and not self._is_temporary_file_path_locked(candidate)
            ):
                return self._promote_file_path_locked(hint, candidate)

        if not candidate:
            return hint

        visible_basename_matches = self._visible_file_paths_by_basename.get(
            path_basename(candidate).lower(),
            [],
        )
        if len(visible_basename_matches) == 1:
            known_path = visible_basename_matches[0]
            self._path_aliases[candidate] = known_path
            return known_path

        exact_basename_matches = [
            known_path
            for known_path in self._files
            if path_basename(known_path).lower() == path_basename(candidate).lower()
        ]
        if len(exact_basename_matches) == 1:
            known_path = exact_basename_matches[0]
            if candidate.count("/") > known_path.count("/") and not stale:
                return self._promote_file_path_locked(known_path, candidate)
            self._path_aliases[candidate] = known_path
            return known_path

        suffix_matches = [
            known_path
            for known_path in self._files
            if known_path.endswith("/" + candidate) or candidate.endswith("/" + known_path)
        ]
        if len(suffix_matches) == 1:
            known_path = suffix_matches[0]
            if candidate.endswith("/" + known_path) and candidate.count("/") > known_path.count("/") and not stale:
                return self._promote_file_path_locked(known_path, candidate)
            self._path_aliases[candidate] = known_path
            return known_path

        fuzzy_matches = [
            known_path
            for known_path in self._files
            if self._looks_like_fuzzy_path_match(candidate, known_path)
        ]
        if len(fuzzy_matches) == 1:
            known_path = fuzzy_matches[0]
            self._path_aliases[candidate] = known_path
            return known_path

        if stale:
            return hint if hint in self._files else None

        if not allow_create:
            return None

        self._path_aliases.setdefault(candidate, candidate)
        return candidate

    # Evalua si el valor parece fuzzy path match.
    def _looks_like_fuzzy_path_match(self, candidate: Optional[str], known_path: Optional[str]) -> bool:
        candidate = sanitize_relative_code_path(candidate)
        known_path = sanitize_relative_code_path(known_path)
        if not candidate or not known_path:
            return False
        if path_extension(candidate) != path_extension(known_path):
            return False
        candidate_stem = path_stem(candidate)
        known_stem = path_stem(known_path)
        if not candidate_stem or not known_stem:
            return False
        return levenshtein_distance(candidate_stem, known_stem) <= 1

    # Promueve file path locked.
    def _promote_file_path_locked(self, old_path: str, new_path: str) -> str:
        old_path = self._path_aliases.get(sanitize_relative_code_path(old_path) or "", sanitize_relative_code_path(old_path) or "")
        new_path = sanitize_relative_code_path(new_path) or old_path
        if not old_path or old_path == new_path:
            self._path_aliases[new_path] = new_path
            return new_path

        old_was_temporary = self._is_temporary_file_path_locked(old_path)

        source_node = self._files.pop(old_path, None)
        target_node = self._files.get(new_path)
        if source_node is None:
            self._path_aliases[old_path] = new_path
            self._path_aliases[new_path] = new_path
            if old_was_temporary:
                self._temporary_file_paths.discard(old_path)
            return new_path

        if target_node is None:
            source_node.path = new_path
            self._files[new_path] = source_node
        else:
            self._merge_file_nodes_locked(target_node, source_node)

        self._path_aliases[old_path] = new_path
        self._path_aliases[new_path] = new_path
        if old_path in self._provisional_file_paths:
            self._provisional_file_paths.discard(old_path)
            self._provisional_file_paths.add(new_path)
        if old_was_temporary:
            self._temporary_file_paths.discard(old_path)
        if self._active_file == old_path:
            self._active_file = new_path
        if self._last_editor_file_path == old_path:
            self._last_editor_file_path = new_path
        self._move_rendered_file(old_path, new_path)
        return new_path

    # Fusiona file nodes locked.
    def _merge_file_nodes_locked(self, target_node: CodeFileNode, source_node: CodeFileNode) -> None:
        if target_node is source_node:
            return
        if not target_node.language:
            target_node.language = source_node.language
        target_node.observations += int(source_node.observations)
        target_node.last_frame = max(int(target_node.last_frame), int(source_node.last_frame))

        for line_number, source_observation in source_node.lines.items():
            existing = target_node.lines.get(line_number)
            if existing is None:
                target_node.lines[line_number] = source_observation
                continue

            existing_score = target_node._line_score(
                existing.text,
                existing.confidence,
                existing.number_source,
            )
            source_score = target_node._line_score(
                source_observation.text,
                source_observation.confidence,
                source_observation.number_source,
            )
            if source_score > existing_score:
                target_node.lines[line_number] = source_observation
            else:
                existing.observations += int(source_observation.observations)
                existing.last_frame = max(int(existing.last_frame), int(source_observation.last_frame))
                existing.confidence = max(float(existing.confidence), float(source_observation.confidence))

        for source_candidate in source_node.pending_lines.values():
            target_key = target_node._resolve_pending_line_key(
                source_candidate.line_number,
                source_candidate.text,
            )
            existing_candidate = target_node.pending_lines.get(target_key)
            if existing_candidate is None:
                source_candidate.key = target_key
                target_node.pending_lines[target_key] = source_candidate
                continue

            if float(source_candidate.confidence) >= float(existing_candidate.confidence):
                existing_candidate.line_number = int(source_candidate.line_number)
                existing_candidate.text = source_candidate.text
            existing_candidate.confidence = max(
                float(existing_candidate.confidence),
                float(source_candidate.confidence),
            )
            if (
                target_node._number_source_rank(source_candidate.number_source)
                >= target_node._number_source_rank(existing_candidate.number_source)
            ):
                existing_candidate.number_source = source_candidate.number_source
            existing_candidate.first_frame = min(
                int(existing_candidate.first_frame),
                int(source_candidate.first_frame),
            )
            existing_candidate.last_frame = max(
                int(existing_candidate.last_frame),
                int(source_candidate.last_frame),
            )
            merged_frames = sorted(
                {
                    int(frame)
                    for frame in list(existing_candidate.frames) + list(source_candidate.frames)
                }
            )
            existing_candidate.frames = merged_frames[-CODE_PENDING_LINE_CONFIRMATION_FRAMES - 2 :]
            existing_candidate.observations = max(
                int(existing_candidate.observations),
                int(source_candidate.observations),
                len(existing_candidate.frames),
            )
            merged_signatures: List[str] = []
            for signature in list(existing_candidate.block_signatures) + list(source_candidate.block_signatures):
                if signature and signature not in merged_signatures:
                    merged_signatures.append(signature)
            existing_candidate.block_signatures = merged_signatures[-6:]

        for signature, source_frames in source_node.block_observations.items():
            if not signature:
                continue
            merged_frames = sorted(
                {
                    int(frame)
                    for frame in list(target_node.block_observations.get(signature, [])) + list(source_frames)
                }
            )
            target_node.block_observations[signature] = merged_frames[-CODE_PENDING_LINE_CONFIRMATION_FRAMES - 2 :]

        target_node.blocks.extend(source_node.blocks)
        if len(target_node.blocks) > 200:
            target_node.blocks = target_node.blocks[-200:]

    # Normaliza sidebar entries.
    def _normalize_sidebar_entries(self, entries: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not isinstance(entries, list):
            return normalized
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = normalize_whitespace(entry.get("label", "") or entry.get("path", ""))
            path = sanitize_relative_code_path(entry.get("path"))
            entry_type = str(entry.get("type", "file") or "file").strip().lower()
            if not label and not path:
                continue
            normalized.append(
                {
                    "label": label or path,
                    "path": path,
                    "depth": max(0, int(safe_float(entry.get("depth"), 0))),
                    "type": "folder" if entry_type == "folder" else "file",
                    "selected": bool(entry.get("selected", False)),
                }
            )
        return normalized

    # Construye sidebar tree.
    def _build_sidebar_tree(
        self,
        entries: List[Dict[str, Any]],
    ) -> Tuple[List[SidebarTreeNode], List[Dict[str, Any]], List[SidebarTreeNode]]:
        root_nodes: List[SidebarTreeNode] = []
        flat_nodes: List[SidebarTreeNode] = []
        stack: List[SidebarTreeNode] = []
        normalized_entries: List[Dict[str, Any]] = []
        for raw_entry in entries[:CODE_SIDEBAR_TREE_LIMIT]:
            depth = max(0, int(safe_float(raw_entry.get("depth"), 0)))
            while len(stack) > depth:
                stack.pop()

            parent_node = stack[-1] if stack else None
            entry_type = str(raw_entry.get("type", "file") or "file").strip().lower()
            label = normalize_whitespace(raw_entry.get("label", "") or raw_entry.get("path", ""))
            path = sanitize_relative_code_path(raw_entry.get("path"))
            path_source = "explicit" if path else "none"
            if not path:
                path = self._infer_sidebar_entry_path(
                    label=label,
                    entry_type=entry_type,
                    parent_path=None if parent_node is None else parent_node.path,
                )
                if path:
                    path_source = "inferred"

            node = SidebarTreeNode(
                label=label or (path or ""),
                path=path,
                depth=depth,
                type="folder" if entry_type == "folder" else "file",
                selected=bool(raw_entry.get("selected", False)),
                path_source=path_source,
            )
            if parent_node is None:
                root_nodes.append(node)
            else:
                parent_node.children.append(node)
            flat_nodes.append(node)
            if node.type == "folder":
                stack.append(node)

            normalized_entries.append(
                {
                    "label": node.label,
                    "path": node.path,
                    "depth": int(node.depth),
                    "type": node.type,
                    "selected": bool(node.selected),
                    "path_source": node.path_source,
                }
            )
        return root_nodes, normalized_entries, flat_nodes

    # Infiere sidebar entry path.
    def _infer_sidebar_entry_path(
        self,
        label: str,
        entry_type: str,
        parent_path: Optional[str],
    ) -> Optional[str]:
        label = normalize_whitespace(label)
        if not label:
            return None
        sanitized_label = sanitize_relative_code_path(label)
        if parent_path:
            combined = sanitize_relative_code_path(f"{parent_path}/{label}")
            if combined:
                return combined
        if not sanitized_label:
            return None
        if entry_type == "folder":
            return sanitized_label
        if path_extension(sanitized_label):
            return sanitized_label
        return None

    # Actualiza sidebar visibility locked.
    def _set_sidebar_visibility_locked(
        self,
        root_nodes: List[SidebarTreeNode],
        entries: List[Dict[str, Any]],
    ) -> None:
        self._sidebar_tree = [node.to_dict() for node in root_nodes]
        self._visible_file_paths = {
            str(entry.get("path"))
            for entry in entries
            if entry.get("type") == "file" and entry.get("path")
        }
        self._visible_folder_paths = {
            str(entry.get("path"))
            for entry in entries
            if entry.get("type") == "folder" and entry.get("path")
        }
        basename_map: Dict[str, List[str]] = defaultdict(list)
        for path in sorted(self._visible_file_paths):
            basename = path_basename(path).lower()
            if basename:
                basename_map[basename].append(path)
        self._visible_file_paths_by_basename = dict(basename_map)

    # Marca editor file visibility locked.
    def _mark_editor_file_visibility_locked(self, path: Optional[str]) -> bool:
        sanitized = sanitize_relative_code_path(path)
        if not sanitized:
            return False
        if sanitized in self._visible_file_paths:
            self._provisional_file_paths.discard(sanitized)
            return False
        self._provisional_file_paths.add(sanitized)
        return True

    # Libera provisional visible files locked.
    def _release_provisional_visible_files_locked(self, paths: List[Optional[str]]) -> List[str]:
        released: List[str] = []
        for path in paths:
            sanitized = sanitize_relative_code_path(path)
            if not sanitized or sanitized not in self._visible_file_paths:
                continue
            if sanitized in self._provisional_file_paths:
                self._provisional_file_paths.discard(sanitized)
                released.append(sanitized)
        return released

    # Evalua la condicion is temporary file path locked y devuelve el resultado booleano.
    def _is_temporary_file_path_locked(self, path: Optional[str]) -> bool:
        sanitized = sanitize_relative_code_path(path)
        if not sanitized:
            return False
        return sanitized in self._temporary_file_paths or sanitized.startswith("__tmp__/")

    # Maneja temporary extension for language dentro de CodeKnowledgeStore.
    def _temporary_extension_for_language(self, language: Optional[str]) -> str:
        key = str(language or "").strip().upper()
        mapping = {
            "PYTHON": ".py",
            "JAVASCRIPT": ".js",
            "TYPESCRIPT": ".ts",
            "TSX": ".tsx",
            "JSX": ".jsx",
            "JAVA": ".java",
            "C": ".c",
            "CPP": ".cpp",
            "C++": ".cpp",
            "C#": ".cs",
            "CSHARP": ".cs",
            "GO": ".go",
            "RUST": ".rs",
            "KOTLIN": ".kt",
            "SWIFT": ".swift",
            "PHP": ".php",
            "RUBY": ".rb",
            "JSON": ".json",
            "YAML": ".yml",
            "XML": ".xml",
            "HTML": ".html",
            "CSS": ".css",
            "SQL": ".sql",
            "SHELL": ".sh",
            "BASH": ".sh",
        }
        return mapping.get(key, ".txt")

    # Obtiene or create editor temp file locked.
    def _get_or_create_editor_temp_file_locked(self, language: Optional[str]) -> str:
        current = sanitize_relative_code_path(self._last_editor_file_path)
        if current and current in self._files and self._is_temporary_file_path_locked(current):
            return current

        self._temp_file_counter += 1
        extension = self._temporary_extension_for_language(language)
        temp_path = sanitize_relative_code_path(
            f"__tmp__/editor_buffer_{self._temp_file_counter:03d}{extension}"
        ) or f"__tmp__/editor_buffer_{self._temp_file_counter:03d}.txt"
        self._temporary_file_paths.add(temp_path)
        self._path_aliases.setdefault(temp_path, temp_path)
        self._files.setdefault(temp_path, CodeFileNode(path=temp_path))
        self._provisional_file_paths.add(temp_path)
        return temp_path

    # Maneja maybe promote last editor temp file locked dentro de CodeKnowledgeStore.
    def _maybe_promote_last_editor_temp_file_locked(self, target_path: Optional[str], stale: bool = False) -> Optional[str]:
        sanitized_target = sanitize_relative_code_path(target_path)
        if not sanitized_target or stale or self._is_temporary_file_path_locked(sanitized_target):
            return sanitized_target

        current = sanitize_relative_code_path(self._last_editor_file_path)
        if not current or current not in self._files or not self._is_temporary_file_path_locked(current):
            return sanitized_target
        return self._promote_file_path_locked(current, sanitized_target)

    # Maneja current editor fallback hint locked dentro de CodeKnowledgeStore.
    def _current_editor_fallback_hint_locked(self) -> Optional[str]:
        if (
            self._last_editor_file_path
            and self._last_editor_file_path in self._files
            and self._is_temporary_file_path_locked(self._last_editor_file_path)
        ):
            return self._last_editor_file_path

        recent_file: Optional[str] = None
        recent_key: Optional[Tuple[int, int, int, str]] = None
        for path, node in self._files.items():
            if int(node.last_frame) < 0:
                continue
            if not self._is_temporary_file_path_locked(path):
                continue
            candidate_key = (
                int(node.last_frame),
                int(node.observations),
                1 if path in self._provisional_file_paths else 0,
                str(path),
            )
            if recent_key is None or candidate_key > recent_key:
                recent_key = candidate_key
                recent_file = path
        if recent_file:
            return recent_file

        active_file = sanitize_relative_code_path(self._active_file)
        if active_file and active_file in self._files and self._is_temporary_file_path_locked(active_file):
            return active_file
        return None

    # Registra editor file locked.
    def _remember_editor_file_locked(self, path: Optional[str], frame_number: int) -> None:
        sanitized = sanitize_relative_code_path(path)
        if not sanitized or sanitized not in self._files:
            return
        if int(frame_number) < int(self._last_editor_file_frame):
            return
        self._last_editor_file_path = sanitized
        self._last_editor_file_frame = int(frame_number)

    # Resuelve editor file path locked.
    def _resolve_editor_file_path_locked(
        self,
        active_file: Optional[str],
        tab_title: Optional[str],
        hint: Optional[str],
        stale: bool,
    ) -> Optional[str]:
        sanitized_hint = sanitize_relative_code_path(hint)
        seen: set[str] = set()
        candidate_specs = [
            (
                sanitize_relative_code_path(active_file),
                self._allow_editor_file_creation_locked(active_file),
            ),
            (
                sanitize_relative_code_path(tab_title),
                bool(sanitize_relative_code_path(tab_title)),
            ),
            (
                sanitized_hint,
                False,
            ),
        ]
        for candidate, allow_create in candidate_specs:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            resolved = self._resolve_file_path_locked(
                candidate,
                hint=sanitized_hint,
                stale=stale,
                allow_create=allow_create,
            )
            if resolved:
                return resolved
        return None

    # Decide si se permite allow editor file creation locked.
    def _allow_editor_file_creation_locked(
        self,
        candidate: Optional[str],
    ) -> bool:
        sanitized = sanitize_relative_code_path(candidate)
        if not sanitized:
            return False
        if sanitized in self._files or sanitized in self._visible_file_paths:
            return True
        basename_matches = self._visible_file_paths_by_basename.get(path_basename(sanitized).lower(), [])
        if len(basename_matches) == 1:
            return True
        if self._last_sidebar_metadata_frame < 0 and path_extension(sanitized):
            return True
        return False

    # Decide si se permite allow sidebar file creation locked.
    def _allow_sidebar_file_creation_locked(self, candidate: Optional[str]) -> bool:
        sanitized = sanitize_relative_code_path(candidate)
        if not sanitized:
            return False
        if sanitized in self._visible_file_paths or sanitized in self._files:
            return True
        basename_matches = self._visible_file_paths_by_basename.get(path_basename(sanitized).lower(), [])
        return len(basename_matches) == 1

    # Normaliza editor line group.
    def _normalize_editor_line_group(
        self,
        payload: Dict[str, Any],
        field_name: str = "lines",
    ) -> List[Dict[str, Any]]:
        raw_lines = payload.get(field_name)
        if not isinstance(raw_lines, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for entry in raw_lines:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text", "") or "").rstrip()
            raw_number_source = str(entry.get("number_source", "") or "").strip()
            original_number = safe_int(entry.get("number"))
            if original_number is not None and int(original_number) < 1:
                continue
            if not should_keep_editor_line(text, original_number):
                continue
            segments = split_compound_editor_line(original_number, text)
            for segment_index, (line_number, segment_text) in enumerate(segments):
                segment_text = str(segment_text or "").rstrip()
                if line_number is not None and int(line_number) < 1:
                    continue
                if not should_keep_editor_line(segment_text, line_number):
                    continue
                normalized.append(
                    {
                        "number": line_number,
                        "text": segment_text,
                        "confidence": float(safe_float(entry.get("confidence"), payload.get("confidence", 0.0))),
                        "number_source": (
                            raw_number_source
                            or ("explicit" if line_number is not None else "missing")
                        ) if segment_index == 0 else "embedded_split",
                        "line_kind": infer_editor_line_kind(segment_text),
                    }
                )

        if not normalized:
            return normalized

        if field_name != "lines":
            return normalized

        start_line = safe_int(payload.get("start_line"))
        known_positions = [
            (index, safe_int(entry.get("number")))
            for index, entry in enumerate(normalized)
            if safe_int(entry.get("number")) is not None
        ]
        rebased_to_start = False
        if start_line is not None and known_positions:
            rebased_to_start = self._rebase_relative_line_numbers_to_start(
                normalized,
                int(start_line),
                known_positions,
            )
            if rebased_to_start:
                for index, entry in enumerate(normalized):
                    if safe_int(entry.get("number")) is None:
                        entry["number"] = int(start_line) + index
                        entry["number_source"] = "start_line"
                return normalized

        if not known_positions and start_line is not None:
            for index, entry in enumerate(normalized):
                entry["number"] = start_line + index
                entry["number_source"] = "start_line"
            return normalized

        if len(known_positions) == 1:
            anchor_index, anchor_number = known_positions[0]
            lower_bound = max(0, anchor_index - 2)
            upper_bound = min(len(normalized), anchor_index + 3)
            for index in range(lower_bound, upper_bound):
                if index == anchor_index:
                    continue
                normalized[index]["number"] = anchor_number + (index - anchor_index)
                normalized[index]["number_source"] = "anchor_inferred"
            return normalized

        for position in range(len(known_positions) - 1):
            left_index, left_number = known_positions[position]
            right_index, right_number = known_positions[position + 1]
            if left_number is None or right_number is None:
                continue
            gap = right_index - left_index
            if gap <= 1:
                continue
            expected_span = right_number - left_number
            if expected_span == gap:
                for middle in range(1, gap):
                    normalized[left_index + middle]["number"] = left_number + middle
                    normalized[left_index + middle]["number_source"] = "interpolated"

        return normalized

    # Reubica numeros relativos usando start_line como posicion absoluta.
    def _rebase_relative_line_numbers_to_start(
        self,
        normalized: List[Dict[str, Any]],
        start_line: int,
        known_positions: List[tuple],
    ) -> bool:
        if not normalized or start_line < 1 or not known_positions:
            return False

        first_index, first_number = known_positions[0]
        if first_number is None or int(first_number) >= int(start_line):
            return False

        offset = int(start_line) + int(first_index) - int(first_number)
        if offset <= 0:
            return False

        for entry in normalized:
            line_number = safe_int(entry.get("number"))
            if line_number is None:
                continue
            entry["number"] = int(line_number) + offset
            entry["number_source"] = "start_line"
        return True

    # Construye line entries from raw text.
    def _build_line_entries_from_raw_text(self, raw_text: str, start_line: Optional[int]) -> List[Dict[str, Any]]:
        lines = []
        current_line = start_line
        for raw_line in str(raw_text or "").splitlines():
            text = raw_line.rstrip()
            if not should_keep_editor_line(text, current_line):
                continue
            lines.append(
                {
                    "number": current_line,
                    "text": text,
                    "confidence": 0.0,
                    "number_source": "start_line" if current_line is not None else "missing",
                    "line_kind": infer_editor_line_kind(text),
                }
            )
            if current_line is not None:
                current_line += 1
        return lines

    # Calcula indent levels.
    def _compute_indent_levels(self, lines: List[Dict[str, Any]]) -> List[int]:
        levels = []
        for entry in lines:
            text = str(entry.get("text", "") or "")
            levels.append(len(text) - len(text.lstrip(" ")))
        return levels

    # Calcula bracket balance.
    def _compute_bracket_balance(self, lines: List[Dict[str, Any]]) -> int:
        balance = 0
        for entry in lines:
            text = str(entry.get("text", "") or "")
            balance += text.count("(") + text.count("[") + text.count("{")
            balance -= text.count(")") + text.count("]") + text.count("}")
        return balance

    # Maneja persist response dentro de CodeKnowledgeStore.
    def _persist_response(self, kind: str, frame_number: int, payload: Dict[str, Any]) -> None:
        if self._session_dir is None:
            return
        path = self._session_dir / "responses" / kind / f"frame_{frame_number:06d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)

    # Maneja flush if needed dentro de CodeKnowledgeStore.
    def _flush_if_needed(self) -> None:
        if self._last_change_count < CODE_STORE_FLUSH_CHANGES:
            return
        if (time.time() - self._last_flush_ts) < CODE_STORE_FLUSH_INTERVAL_SECONDS:
            return
        self._last_change_count = 0
        self._last_flush_ts = time.time()
        self.flush_incremental()

    # Construye reference comparison report.
    def _build_reference_comparison_report(
        self,
        rendered_outputs: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        if not CODE_REFERENCE_COMPARE_ENABLE:
            return []

        report: List[Dict[str, Any]] = []
        for path, rendered in sorted(rendered_outputs.items()):
            reference_path = self._reference_path_for_rendered_path(path)
            if not reference_path:
                continue
            comparison = self._compare_rendered_to_reference(
                path=path,
                rendered=rendered,
                reference_path=reference_path,
            )
            report.append(comparison)
            if comparison.get("status") == "ok":
                self._debug(
                    "reference compare "
                    f"file={path} relation={comparison.get('relation')} "
                    f"line_similarity={comparison.get('line_similarity')} "
                    f"text_similarity={comparison.get('text_similarity')} "
                    f"missing={comparison.get('missing_from_rendered')} "
                    f"extra={comparison.get('extra_in_rendered')} "
                    f"changed={comparison.get('changed_lines')}"
                )
            else:
                self._debug(
                    "reference compare "
                    f"file={path} status={comparison.get('status')} "
                    f"reference={reference_path}"
                )
        return report

    # Maneja reference path for rendered path dentro de CodeKnowledgeStore.
    def _reference_path_for_rendered_path(self, path: str) -> Optional[str]:
        basename = path_basename(path).lower()
        if not basename:
            return None
        for candidate, reference_path in CODE_REFERENCE_COMPARE_FILES.items():
            if path_basename(candidate).lower() == basename:
                return str(reference_path)
        return None

    # Maneja compare rendered to reference dentro de CodeKnowledgeStore.
    def _compare_rendered_to_reference(
        self,
        path: str,
        rendered: str,
        reference_path: str,
    ) -> Dict[str, Any]:
        reference_file = Path(reference_path)
        if not reference_file.exists() or not reference_file.is_file():
            return {
                "path": path,
                "reference_path": str(reference_file),
                "status": "missing_reference",
            }

        try:
            reference_text = reference_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            reference_text = reference_file.read_text(encoding="latin-1")

        rendered_text = str(rendered or "")
        rendered_lines = rendered_text.splitlines()
        reference_lines = reference_text.splitlines()
        line_matcher = difflib.SequenceMatcher(None, rendered_lines, reference_lines)
        text_matcher = difflib.SequenceMatcher(None, rendered_text, reference_text)

        matching_lines = 0
        missing_from_rendered = 0
        extra_in_rendered = 0
        changed_lines = 0
        for tag, i1, i2, j1, j2 in line_matcher.get_opcodes():
            if tag == "equal":
                matching_lines += int(i2 - i1)
            elif tag == "insert":
                missing_from_rendered += int(j2 - j1)
            elif tag == "delete":
                extra_in_rendered += int(i2 - i1)
            elif tag == "replace":
                changed_lines += int(max(i2 - i1, j2 - j1))

        line_similarity = float(line_matcher.ratio())
        text_similarity = float(text_matcher.ratio())
        return {
            "path": path,
            "reference_path": str(reference_file),
            "status": "ok",
            "relation": self._reference_relation_label(
                max(line_similarity, text_similarity)
            ),
            "line_similarity": round(line_similarity, 4),
            "text_similarity": round(text_similarity, 4),
            "rendered_lines": len(rendered_lines),
            "reference_lines": len(reference_lines),
            "matching_lines": int(matching_lines),
            "missing_from_rendered": int(missing_from_rendered),
            "extra_in_rendered": int(extra_in_rendered),
            "changed_lines": int(changed_lines),
        }

    # Maneja reference relation label dentro de CodeKnowledgeStore.
    def _reference_relation_label(self, similarity: float) -> str:
        if similarity >= 0.995:
            return "identico"
        if similarity >= 0.95:
            return "muy_alto"
        if similarity >= 0.85:
            return "alto"
        if similarity >= 0.70:
            return "medio"
        if similarity >= 0.50:
            return "bajo"
        return "muy_bajo"

    # Escribe reference comparison JSON.
    def _write_reference_comparison_json(
        self,
        comparison: List[Dict[str, Any]],
    ) -> Optional[str]:
        if self._session_dir is None or not comparison:
            return None
        path = self._session_dir / "reference_comparison.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(comparison, file_obj, indent=2, ensure_ascii=False)
        return str(path)

    # Construye summary.
    def _build_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active_file": self._active_file,
                "known_files": len(self._files),
                "structure_version": int(self._structure_version),
                "sidebar_signature": self._sidebar_signature,
                "sidebar_tree": list(self._sidebar_tree),
                "visible_files": sorted(self._visible_file_paths),
                "visible_folders": sorted(self._visible_folder_paths),
                "provisional_files": sorted(self._provisional_file_paths),
                "sidebar_history": list(self._sidebar_history),
                "editor_history": list(self._editor_history),
                "files": {path: node.to_dict() for path, node in sorted(self._files.items())},
            }

    # Escribe summary JSON.
    def _write_summary_json(self, summary: Dict[str, Any]) -> None:
        if self._summary_path is None:
            return
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._summary_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    # Rendered output helpers.
    # Escribe rendered file.
    def _write_rendered_file(self, path: str, rendered: str) -> None:
        if self._render_root is None or not rendered:
            return
        if not self._should_materialize_file(path):
            return
        self._render_root.mkdir(parents=True, exist_ok=True)
        safe_path = self._safe_render_path(path)
        target = self._render_root / safe_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as file_obj:
            file_obj.write(rendered)

    # Crea o actualiza render path.
    def _touch_render_path(self, path: str) -> None:
        if self._render_root is None:
            return
        if not self._should_materialize_file(path):
            return
        safe_path = self._safe_render_path(path)
        target = self._render_root / safe_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with open(target, "w", encoding="utf-8") as file_obj:
                file_obj.write("")

    # Mueve rendered file.
    def _move_rendered_file(self, old_path: Optional[str], new_path: Optional[str]) -> None:
        if self._render_root is None:
            return
        old_sanitized = sanitize_relative_code_path(old_path)
        new_sanitized = sanitize_relative_code_path(new_path)
        if not old_sanitized or not new_sanitized or old_sanitized == new_sanitized:
            return

        old_target = self._render_root / self._safe_render_path(old_sanitized)
        new_target = self._render_root / self._safe_render_path(new_sanitized)
        if not old_target.exists():
            return

        new_target.parent.mkdir(parents=True, exist_ok=True)
        if new_target.exists():
            old_target.unlink(missing_ok=True)
            return
        old_target.replace(new_target)

    # Maneja materialize sidebar structure dentro de CodeKnowledgeStore.
    def _materialize_sidebar_structure(self) -> None:
        if self._render_root is None:
            return
        for folder_path in sorted(self._visible_folder_paths):
            target = self._render_root / self._safe_render_path(folder_path)
            target.mkdir(parents=True, exist_ok=True)
        for file_path in sorted(self._visible_file_paths):
            self._touch_render_path(file_path)

    # Evalua la condicion should materialize file y devuelve el resultado booleano.
    def _should_materialize_file(self, path: Optional[str]) -> bool:
        sanitized = sanitize_relative_code_path(path)
        if not sanitized:
            return False
        with self._lock:
            return (
                sanitized not in self._provisional_file_paths
                or sanitized in self._visible_file_paths
                or sanitized in self._temporary_file_paths
                or sanitized == self._last_editor_file_path
            )

    # Maneja safe render path dentro de CodeKnowledgeStore.
    def _safe_render_path(self, path: str) -> Path:
        parts = [part for part in str(path).replace("\\", "/").split("/") if part and part not in {".", ".."}]
        if not parts:
            parts = ["unknown.txt"]
        return Path(*parts)

    # Debug helpers.
    # Maneja response debug info dentro de CodeKnowledgeStore.
    def _response_debug_info(self, response: CodeVisionResponse) -> Dict[str, Any]:
        usage = None
        florence = None
        provider = None
        remote_error = None
        if isinstance(response.payload, dict):
            raw_usage = response.payload.get("__usage__")
            if isinstance(raw_usage, dict):
                usage = dict(raw_usage)
            raw_florence = response.payload.get("__florence__")
            if isinstance(raw_florence, dict):
                florence = dict(raw_florence)
            raw_provider = response.payload.get("__provider__")
            if isinstance(raw_provider, dict):
                provider = dict(raw_provider)
            if response.payload.get("__remote_error__"):
                remote_error = {
                    "error": str(response.payload.get("__remote_error__")),
                    "message": str(response.payload.get("__remote_error_message__", "") or ""),
                }
        debug_info = {
            "task_id": int(response.task.task_id),
            "worker_name": response.worker_name,
            "queued_for_request_sec": float(response.queued_for_request_sec),
            "request_duration_sec": float(response.request_duration_sec),
            "assembled_delay_sec": float(response.assembled_delay_sec),
        }
        task_meta = {}
        for key in ("frame_quality", "selected_frame_number", "selected_score", "send_reasons"):
            if key in response.task.meta:
                task_meta[key] = response.task.meta.get(key)
        if task_meta:
            debug_info["task_meta"] = task_meta
        if usage:
            debug_info["usage"] = usage
        if provider:
            debug_info["provider"] = provider
        if florence:
            debug_info["florence"] = florence
        if remote_error:
            debug_info["remote_error"] = remote_error
        return debug_info

    # Maneja debug dentro de CodeKnowledgeStore.
    def _debug(self, message: str) -> None:
        if self._debug_logger is None:
            return
        self._debug_logger(message)
