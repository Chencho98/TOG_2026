#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


DEFAULT_SESSION_DIR = Path("output/sessions/session_20260507_200619")
DEFAULT_NOTEBOOK_NAME = "bitacora_accesible_links.ipynb"
DEFAULT_LIVE_NOTEBOOK_NAME = "bitacora_accesible_live.ipynb"
DEFAULT_FPS = 16.0
DEFAULT_MAX_CODE_EVENTS = 80
DEFAULT_MAX_RENDERED_FILES = 30
DEFAULT_MAX_CODE_LINES_PER_EVENT = 90
DEFAULT_MAX_FINAL_FILE_CHARS = 120_000
DEFAULT_MAX_TIMELINE_EVENTS = 220

TEXT_SUFFIXES = {
    "",
    ".bat",
    ".c",
    ".cfg",
    ".conf",
    ".cpp",
    ".csv",
    ".css",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".ps1",
    ".py",
    ".r",
    ".rst",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

IMAGE_OR_BINARY_SUFFIXES = {
    ".bin",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".onnx",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
    ".webp",
    ".zip",
}

SECRET_FILENAMES = {".env", ".env.local", ".env.production", ".env.dev"}


@dataclass
class TimelineItem:
    time_s: float | None
    frame: int | None
    kind: str
    summary: str
    detail: str = ""

    @property
    def sort_key(self) -> tuple[float, int, str]:
        if self.time_s is not None:
            primary = self.time_s
        elif self.frame is not None:
            primary = float(self.frame)
        else:
            primary = float("inf")
        return (primary, self.frame if self.frame is not None else 10**12, self.kind)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_frame_from_name(path: Path) -> int | None:
    match = re.search(r"frame[_-](\d+)", path.name)
    if not match:
        return None
    return safe_int(match.group(1))


def compact_text(value: Any, max_chars: int = 220) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def markdown_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", "<br>")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_Sin datos disponibles._"
    lines = [
        "| " + " | ".join(markdown_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = list(row)[: len(headers)]
        values.extend([""] * (len(headers) - len(values)))
        lines.append("| " + " | ".join(markdown_escape(value) for value in values) + " |")
    return "\n".join(lines)


def markdown_link(label: Any, href: str) -> str:
    text = str(label if label is not None else "")
    text = text.replace("[", "\\[").replace("]", "\\]")
    return f"[{text}]({href})"


def html_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def html_link(label: Any, href: str) -> str:
    return f'<a href="{html_escape(href)}">{html_escape(label)}</a>'


def html_heading_with_anchor(level: int, anchor_id: str, text: str) -> str:
    level = min(max(level, 1), 6)
    return f'<h{level} id="{html_escape(anchor_id)}">{text}</h{level}>'


def slugify_anchor_part(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "sin-dato"


def code_capture_anchor(event: dict[str, Any]) -> str:
    file_part = slugify_anchor_part(event.get("file_path") or "archivo-desconocido")
    frame = safe_int(event.get("frame_number"))
    start_line = safe_int(event.get("start_line"))
    frame_part = f"frame-{frame:06d}" if frame is not None else "frame-desconocido"
    if start_line is not None:
        return f"code-{file_part}-{frame_part}-l{start_line}"
    return f"code-{file_part}-{frame_part}"


def code_event_start_line(event: dict[str, Any]) -> int | None:
    start_line = safe_int(event.get("start_line"))
    if start_line is not None and start_line > 0:
        return start_line
    for line in as_list(event.get("lines")):
        if not isinstance(line, dict):
            continue
        for key in ("line_number", "number", "line"):
            line_number = safe_int(line.get(key))
            if line_number is not None and line_number > 0:
                return line_number
    return None


def resolve_session_artifact(session_dir: Path, artifact: Any) -> Path | None:
    raw = str(artifact or "").strip()
    if not raw:
        return None
    normalized = raw.replace("\\", "/")
    session_name = session_dir.name
    marker = "/" + session_name + "/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    output_prefix = f"output/sessions/{session_name}/"
    if normalized.startswith(output_prefix):
        normalized = normalized[len(output_prefix):]
    elif normalized.startswith(session_name + "/"):
        normalized = normalized[len(session_name) + 1:]

    path = Path(normalized)
    if path.is_absolute():
        return path
    return session_dir / path


def artifact_href(session_dir: Path, output_dir: Path, artifact: Any, fragment: str = "") -> str:
    path = resolve_session_artifact(session_dir, artifact)
    if path is None:
        return ""
    try:
        relative = os.path.relpath(str(path), str(output_dir))
    except Exception:
        relative = str(path)
    href = quote(Path(relative).as_posix(), safe="/:@%+.,_-=()")
    if fragment:
        href += "#" + quote(str(fragment), safe="=&")
    return href


def artifact_link(
    session_dir: Path,
    output_dir: Path,
    label: Any,
    artifact: Any,
    *,
    fragment: str = "",
    require_exists: bool = False,
) -> str:
    path = resolve_session_artifact(session_dir, artifact)
    if path is None:
        return ""
    if require_exists and not path.exists():
        return ""
    href = artifact_href(session_dir, output_dir, path, fragment=fragment)
    return markdown_link(label, href) if href else ""


def bullet_list(lines: list[str]) -> str:
    if not lines:
        return "_Sin eventos disponibles._"
    return "\n".join(f"- {line}" for line in lines)


def fenced(text: str, language: str = "") -> str:
    fence = "```" + language
    return f"{fence}\n{text.rstrip()}\n```"


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "tiempo desconocido"
    if seconds < 0:
        seconds = 0.0
    whole = int(seconds)
    tenth = int(round((seconds - whole) * 10))
    if tenth == 10:
        whole += 1
        tenth = 0
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{tenth:d}"
    return f"{minutes:02d}:{secs:02d}.{tenth:d}"


def frame_to_seconds(frame: int | None, fps: float) -> float | None:
    if frame is None or fps <= 0:
        return None
    return float(frame) / fps


def cell_markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.rstrip() + "\n"}


def cell_code(source: str, language: str = "python") -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": language, "tags": ["codigo-capturado"]},
        "outputs": [],
        "source": source.rstrip() + "\n",
    }


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def load_classifications(session_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(session_dir.glob("classifications_*.json")):
        payload = read_json(path, [])
        if isinstance(payload, dict):
            payload = payload.get("items") or payload.get("records") or payload.get("frames") or []
        for item in as_list(payload):
            if isinstance(item, dict):
                item = dict(item)
                item.setdefault("_source_path", str(path))
                item.setdefault("frame_number", parse_frame_from_name(path))
                records.append(item)
    records.sort(key=lambda item: safe_int(item.get("frame_number"), 10**12) or 10**12)
    return records


def load_pdf_matchings(session_dir: Path) -> dict[str, Any]:
    return read_json(session_dir / "pdf" / "matchings.json", {}) or {}


def load_whiteboard_summary(session_dir: Path) -> dict[str, Any]:
    return read_json(session_dir / "whiteboard_analysis" / "summary.json", {}) or {}


def load_whiteboard_events(session_dir: Path) -> list[dict[str, Any]]:
    summary = load_whiteboard_summary(session_dir)
    events = [event for event in as_list(summary.get("events")) if isinstance(event, dict)]
    if not events:
        response_dir = session_dir / "whiteboard_analysis" / "responses"
        for path in sorted(response_dir.glob("frame_*.json")):
            payload = read_json(path, {})
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("_source_path", str(path))
                payload.setdefault("frame_number", parse_frame_from_name(path))
                events.append(payload)
    events.sort(key=lambda item: safe_int(item.get("frame_number"), 10**12) or 10**12)
    return events


def load_code_editor_events(session_dir: Path) -> list[dict[str, Any]]:
    response_dir = session_dir / "code_intelligence" / "responses" / "editor"
    events: list[dict[str, Any]] = []
    for path in sorted(response_dir.glob("frame_*.json")):
        payload = read_json(path, {})
        if isinstance(payload, dict):
            payload = dict(payload)
            payload.setdefault("_source_path", str(path))
            payload.setdefault("frame_number", parse_frame_from_name(path))
            events.append(payload)
    events.sort(key=lambda item: safe_int(item.get("frame_number"), 10**12) or 10**12)
    return events


def load_sidebar_events(session_dir: Path) -> list[dict[str, Any]]:
    response_dir = session_dir / "code_intelligence" / "responses" / "sidebar"
    events: list[dict[str, Any]] = []
    for path in sorted(response_dir.glob("frame_*.json")):
        payload = read_json(path, {})
        if isinstance(payload, dict):
            payload = dict(payload)
            payload.setdefault("_source_path", str(path))
            payload.setdefault("frame_number", parse_frame_from_name(path))
            events.append(payload)
    events.sort(key=lambda item: safe_int(item.get("frame_number"), 10**12) or 10**12)
    return events


def match_entries(matchings: dict[str, Any], prefer_full_history: bool) -> list[dict[str, Any]]:
    if prefer_full_history:
        entries = as_list(matchings.get("full_history"))
        if entries:
            return [entry for entry in entries if isinstance(entry, dict)]
    entries = as_list(matchings.get("best_match_per_page"))
    return [entry for entry in entries if isinstance(entry, dict)]


def estimate_fps(matchings: dict[str, Any], explicit_fps: float | None) -> float:
    if explicit_fps and explicit_fps > 0:
        return explicit_fps
    samples: list[float] = []
    for entry in match_entries(matchings, prefer_full_history=True):
        frame = safe_float(entry.get("frame"))
        timestamp = safe_float(entry.get("timestamp_s"))
        if frame and timestamp and timestamp > 0:
            fps = frame / timestamp
            if 1.0 <= fps <= 240.0:
                samples.append(fps)
    if samples:
        return float(statistics.median(samples))
    return DEFAULT_FPS


def classification_state_events(classifications: list[dict[str, Any]], fps: float) -> list[TimelineItem]:
    events: list[TimelineItem] = []
    previous_key: tuple[Any, ...] | None = None
    for item in classifications:
        frame = safe_int(item.get("frame_number"))
        state = item.get("state") or "DESCONOCIDO"
        page = item.get("page_number")
        language = item.get("language")
        active_file = item.get("code_active_file")
        key = (state, page, language, active_file)
        if key == previous_key:
            continue
        previous_key = key
        parts = [f"estado {state}"]
        if page:
            parts.append(f"diapositiva {page}")
        if language:
            parts.append(f"lenguaje {language}")
        if active_file:
            parts.append(f"archivo activo {active_file}")
        confidence = safe_float(item.get("confidence"))
        if confidence is not None:
            parts.append(f"confianza {confidence:.2f}")
        events.append(
            TimelineItem(
                time_s=frame_to_seconds(frame, fps),
                frame=frame,
                kind="estado",
                summary=", ".join(parts),
            )
        )
    return events


def slide_timeline_events(matchings: dict[str, Any], fps: float) -> list[TimelineItem]:
    events: list[TimelineItem] = []
    previous_page: int | None = None
    entries = sorted(
        match_entries(matchings, prefer_full_history=True),
        key=lambda entry: safe_int(entry.get("frame"), 10**12) or 10**12,
    )
    for entry in entries:
        page = safe_int(entry.get("page_number"))
        frame = safe_int(entry.get("frame"))
        if page is None or page == previous_page:
            continue
        previous_page = page
        timestamp = safe_float(entry.get("timestamp_s"))
        confidence = safe_float(entry.get("confidence"))
        detail = f"confianza {confidence:.2f}" if confidence is not None else ""
        events.append(
            TimelineItem(
                time_s=timestamp if timestamp is not None else frame_to_seconds(frame, fps),
                frame=frame,
                kind="diapositiva",
                summary=f"se mostro la diapositiva {page}",
                detail=detail,
            )
        )
    return events


def whiteboard_timeline_events(events: list[dict[str, Any]], fps: float) -> list[TimelineItem]:
    timeline: list[TimelineItem] = []
    for event in events:
        frame = safe_int(event.get("frame_number"))
        event_type = event.get("event_type") or "evento"
        added = [str(item) for item in as_list(event.get("model_added")) if str(item).strip()]
        erased = [str(item) for item in as_list(event.get("model_erased")) if str(item).strip()]
        changed = [str(item) for item in as_list(event.get("model_changed")) if str(item).strip()]
        pieces: list[str] = [f"pizarron: {event_type}"]
        if added:
            pieces.append("agregado: " + "; ".join(compact_text(item, 90) for item in added))
        if erased:
            pieces.append("borrado: " + "; ".join(compact_text(item, 90) for item in erased))
        if changed:
            pieces.append("cambiado: " + "; ".join(compact_text(item, 90) for item in changed))
        if event.get("still_writing") is True:
            pieces.append("el modelo indica que aun se esta escribiendo")
        content = compact_text(event.get("model_content"), 160)
        detail = f"contenido observado: {content}" if content else ""
        timeline.append(
            TimelineItem(
                time_s=frame_to_seconds(frame, fps),
                frame=frame,
                kind="pizarron",
                summary="; ".join(pieces),
                detail=detail,
            )
        )
    return timeline


def code_timeline_events(events: list[dict[str, Any]], fps: float) -> list[TimelineItem]:
    timeline: list[TimelineItem] = []
    for event in events:
        frame = safe_int(event.get("frame_number"))
        file_path = event.get("file_path") or "archivo desconocido"
        language = event.get("language") or ""
        line_count = len(as_list(event.get("lines")))
        merge = event.get("merge") if isinstance(event.get("merge"), dict) else {}
        merge_parts = []
        for key in ("inserted", "updated", "deduped", "skipped"):
            value = merge.get(key)
            if value not in (None, "", 0):
                merge_parts.append(f"{key}={value}")
        summary = f"codigo capturado en {file_path}"
        if language:
            summary += f" ({language})"
        if line_count:
            summary += f", {line_count} lineas visibles"
        anchor = code_capture_anchor(event)
        merge_parts.append(f"ver {html_link('bloque', '#' + anchor)}")
        detail = ", ".join(merge_parts)
        timeline.append(
            TimelineItem(
                time_s=frame_to_seconds(frame, fps),
                frame=frame,
                kind="codigo",
                summary=summary,
                detail=detail,
            )
        )
    return timeline


def code_lines_to_text(event: dict[str, Any], max_lines: int) -> str:
    lines = [line for line in as_list(event.get("lines")) if isinstance(line, dict)]
    if lines:
        text_lines = [str(line.get("text", "")) for line in lines[:max_lines]]
        if len(lines) > max_lines:
            text_lines.append(f"# ... {len(lines) - max_lines} lineas omitidas por limite local")
        return "\n".join(text_lines)
    raw_text = str(event.get("raw_text") or "").rstrip()
    if raw_text:
        split_lines = raw_text.splitlines()
        if len(split_lines) > max_lines:
            split_lines = split_lines[:max_lines] + [
                f"# ... {len(raw_text.splitlines()) - max_lines} lineas omitidas por limite local"
            ]
        return "\n".join(split_lines)
    return "# Sin codigo visible en esta respuesta."


def line_range_text(event: dict[str, Any]) -> str:
    start = safe_int(event.get("start_line"))
    end = safe_int(event.get("end_line"))
    if start is None and end is None:
        return "rango desconocido"
    if start is None:
        return f"hasta linea {end}"
    if end is None:
        return f"desde linea {start}"
    return f"lineas {start}-{end}"


def merge_text(event: dict[str, Any]) -> str:
    merge = event.get("merge")
    if not isinstance(merge, dict):
        return "sin datos de merge"
    parts = []
    for key in ("inserted", "updated", "deduped", "skipped"):
        value = merge.get(key)
        if value is not None:
            parts.append(f"{key}: {value}")
    stale = event.get("stale_metadata")
    if stale is not None:
        parts.append(f"stale: {bool(stale)}")
    active_applied = event.get("active_file_applied")
    if active_applied is not None:
        parts.append(f"archivo activo aplicado: {bool(active_applied)}")
    return ", ".join(parts) if parts else "sin datos de merge"


def sidebar_tree_lines(entries: list[dict[str, Any]], depth: int = 0) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label") or entry.get("path") or "sin nombre"
        path = entry.get("path") or ""
        entry_type = entry.get("type") or "item"
        selected = " [activo]" if entry.get("selected") else ""
        indent = "  " * depth
        suffix = f" - {path}" if path and path != label else ""
        lines.append(f"{indent}- {label} ({entry_type}){selected}{suffix}")
        children = [child for child in as_list(entry.get("children")) if isinstance(child, dict)]
        if children:
            lines.extend(sidebar_tree_lines(children, depth + 1))
    return lines


def language_for_path(path: Path | str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".c": "c",
        ".cpp": "cpp",
        ".csv": "csv",
        ".css": "css",
        ".h": "c",
        ".hpp": "cpp",
        ".html": "html",
        ".java": "java",
        ".js": "javascript",
        ".json": "json",
        ".jsx": "jsx",
        ".md": "markdown",
        ".ps1": "powershell",
        ".py": "python",
        ".sh": "bash",
        ".sql": "sql",
        ".toml": "toml",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".txt": "text",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(suffix, "text")


def is_probably_text_file(path: Path, include_env: bool) -> bool:
    if path.name in SECRET_FILENAMES and not include_env:
        return False
    suffix = path.suffix.lower()
    if suffix in IMAGE_OR_BINARY_SUFFIXES:
        return False
    if suffix in TEXT_SUFFIXES:
        return True
    if path.stat().st_size > DEFAULT_MAX_FINAL_FILE_CHARS:
        return False
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
        if b"\x00" in chunk:
            return False
        chunk.decode("utf-8")
        return True
    except Exception:
        return False


def rendered_files(session_dir: Path, include_env: bool) -> list[Path]:
    root = session_dir / "code_intelligence" / "rendered"
    if not root.exists():
        return []
    files = [path for path in root.rglob("*") if path.is_file()]
    files = [path for path in files if is_probably_text_file(path, include_env)]
    return sorted(files, key=lambda path: str(path.relative_to(root)))


def read_text_file(path: Path, max_chars: int) -> tuple[str, bool]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text, False
    marker = f"\n\n# ... archivo truncado localmente: {len(text) - max_chars} caracteres omitidos\n"
    return text[:max_chars].rstrip() + marker, True


def slide_table(matchings: dict[str, Any], session_dir: Path, output_dir: Path) -> str:
    rows = []
    entries = sorted(
        match_entries(matchings, prefer_full_history=False),
        key=lambda entry: safe_int(entry.get("page_number"), 10**12) or 10**12,
    )
    for entry in entries:
        confidence = safe_float(entry.get("confidence"))
        timestamp = safe_float(entry.get("timestamp_s"))
        page = safe_int(entry.get("page_number"))
        frame = safe_int(entry.get("frame"))
        links = []
        if page is not None:
            pdf_link = artifact_link(
                session_dir,
                output_dir,
                "PDF",
                "pdf/slides_detected.pdf",
                fragment=f"page={page}",
                require_exists=True,
            )
            thumbnail_link = artifact_link(
                session_dir,
                output_dir,
                "miniatura",
                f"pdf/slide_thumbnails/slide_{page:03d}.jpg",
                require_exists=True,
            )
            links.extend(link for link in (pdf_link, thumbnail_link) if link)
        if frame is not None and page is not None:
            capture_link = artifact_link(
                session_dir,
                output_dir,
                "captura",
                f"pdf/roi_captures/frame_{frame:06d}_slide_{page:03d}.jpg",
                require_exists=True,
            )
            if capture_link:
                links.append(capture_link)
        rows.append(
            [
                page if page is not None else "",
                format_time(timestamp) if timestamp is not None else "",
                frame if frame is not None else "",
                f"{confidence:.3f}" if confidence is not None else "",
                ", ".join(links),
            ]
        )
    return markdown_table(["Diapositiva", "Tiempo", "Frame", "Confianza", "Links"], rows)


def whiteboard_section(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    fps: float,
    session_dir: Path,
    output_dir: Path,
) -> str:
    latest = summary.get("latest_content")
    if not latest and events:
        latest = events[-1].get("model_content")
    parts = ["## Evolucion del pizarron"]
    parts.append("")
    parts.append("### Contenido final conocido")
    parts.append(fenced(str(latest or "Sin contenido final registrado."), "text"))
    for index, event in enumerate(events, start=1):
        frame = safe_int(event.get("frame_number"))
        time_text = format_time(frame_to_seconds(frame, fps))
        title = event.get("event_type") or "evento"
        parts.append("")
        parts.append(f"### Evento {index}: {title} en {time_text}, frame {frame}")
        reason = event.get("reason")
        if reason:
            parts.append(f"- Motivo: `{reason}`")
        event_links = [
            artifact_link(
                session_dir,
                output_dir,
                "snapshot",
                event.get("snapshot_path"),
                require_exists=True,
            ),
            artifact_link(
                session_dir,
                output_dir,
                "respuesta JSON",
                event.get("response_path"),
                require_exists=True,
            ),
        ]
        event_links = [link for link in event_links if link]
        if event_links:
            parts.append(f"- Archivos: {', '.join(event_links)}")
        categories = ", ".join(str(item) for item in as_list(event.get("model_categories")) if item)
        if categories:
            parts.append(f"- Categorias detectadas: {categories}")
        if event.get("still_writing") is not None:
            parts.append(f"- Seguimiento de escritura: `{bool(event.get('still_writing'))}`")
        for label, key in (
            ("Agregado", "model_added"),
            ("Borrado", "model_erased"),
            ("Cambiado", "model_changed"),
        ):
            values = [str(item) for item in as_list(event.get(key)) if str(item).strip()]
            if values:
                parts.append(f"- {label}: " + "; ".join(values))
        visual = str(event.get("model_visual_description") or "").strip()
        if visual:
            parts.append(f"- Descripcion visual: {visual}")
        if event.get("model_code_detected"):
            description = str(event.get("model_code_description") or "").strip()
            parts.append("- Codigo detectado en pizarron: si")
            if description:
                parts.append(f"- Descripcion de codigo: {description}")
        content = str(event.get("model_content") or "").strip()
        if content:
            parts.append("")
            parts.append("Contenido observado:")
            parts.append(fenced(content, "text"))
    return "\n".join(parts)


def repository_summary_section(session_dir: Path, sidebar_events: list[dict[str, Any]]) -> str:
    summary = read_json(session_dir / "code_intelligence" / "repository_summary.json", {}) or {}
    rows = [
        ["Archivo activo final", summary.get("active_file", "")],
        ["Archivos conocidos", summary.get("known_files", "")],
        ["Version de estructura", summary.get("structure_version", "")],
        ["Firma de sidebar", summary.get("sidebar_signature", "")],
    ]
    parts = ["## Estructura de archivos observada", "", markdown_table(["Campo", "Valor"], rows)]
    tree = [entry for entry in as_list(summary.get("sidebar_tree")) if isinstance(entry, dict)]
    if not tree and sidebar_events:
        tree = [entry for entry in as_list(sidebar_events[-1].get("entries")) if isinstance(entry, dict)]
    if tree:
        parts.append("")
        parts.append("### Arbol observado")
        parts.append("\n".join(sidebar_tree_lines(tree)))
    return "\n".join(parts)


def reference_comparison_section(session_dir: Path, output_dir: Path) -> str:
    payload = read_json(session_dir / "code_intelligence" / "reference_comparison.json", []) or []
    rows = []
    for item in as_list(payload):
        if not isinstance(item, dict):
            continue
        rendered_path = item.get("path", "")
        links = []
        if rendered_path:
            rendered_link = artifact_link(
                session_dir,
                output_dir,
                "rendered",
                Path("code_intelligence") / "rendered" / str(rendered_path),
                require_exists=True,
            )
            if rendered_link:
                links.append(rendered_link)
        reference_path = str(item.get("reference_path", "") or "").strip()
        if reference_path:
            reference_candidates = [
                session_dir / reference_path,
                Path("/tmp") / reference_path,
                Path(reference_path),
            ]
            reference_link = ""
            for candidate in reference_candidates:
                if candidate.exists():
                    reference_link = markdown_link(
                        "referencia",
                        quote(os.path.relpath(str(candidate), str(output_dir)), safe="/:@%+.,_-=()"),
                    )
                    break
            if reference_link:
                links.append(reference_link)
        rows.append(
            [
                item.get("path", ""),
                item.get("reference_path", ""),
                item.get("relation", ""),
                item.get("line_similarity", ""),
                item.get("text_similarity", ""),
                item.get("missing_from_rendered", ""),
                item.get("extra_in_rendered", ""),
                item.get("changed_lines", ""),
                ", ".join(links),
            ]
        )
    return "\n".join(
        [
            "## Comparacion contra referencias",
            "",
            markdown_table(
                [
                    "Archivo",
                    "Referencia",
                    "Relacion",
                    "Similitud lineas",
                    "Similitud texto",
                    "Faltantes",
                    "Extras",
                    "Lineas cambiadas",
                    "Links",
                ],
                rows,
            ),
        ]
    )


def build_timeline_cell(items: list[TimelineItem], max_items: int) -> str:
    ordered = sorted(items, key=lambda item: item.sort_key)
    clipped = ordered[:max_items]
    lines: list[str] = []
    for item in clipped:
        frame = f"frame {item.frame}" if item.frame is not None else "frame desconocido"
        detail = f" Detalle: {item.detail}." if item.detail else ""
        lines.append(f"{format_time(item.time_s)} ({frame}) [{item.kind}] {item.summary}.{detail}")
    if len(ordered) > max_items:
        lines.append(f"... {len(ordered) - max_items} eventos omitidos por limite local.")
    return "\n".join(["## Linea de tiempo textual", "", bullet_list(lines)])


def build_code_capture_cells(
    editor_events: list[dict[str, Any]],
    fps: float,
    max_events: int,
    max_lines_per_event: int,
    session_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = [
        cell_markdown(
            "\n".join(
                [
                    "## Codigo capturado por evento",
                    "",
                    "Cada bloque corresponde a una respuesta del analizador de codigo.",
                    "El tiempo se calcula desde el numero de frame y el FPS estimado de la sesion.",
                ]
            )
        )
    ]
    for index, event in enumerate(editor_events[:max_events], start=1):
        frame = safe_int(event.get("frame_number"))
        file_path = event.get("file_path") or "archivo desconocido"
        language = str(event.get("language") or language_for_path(file_path)).lower()
        confidence = safe_float(event.get("confidence"))
        quality = event.get("frame_quality") or ""
        anchor = code_capture_anchor(event)
        links = [
            artifact_link(
                session_dir,
                output_dir,
                "respuesta JSON",
                event.get("_source_path"),
                require_exists=True,
            )
        ]
        rendered_link = ""
        rendered_artifact = None
        if file_path and file_path != "archivo desconocido":
            rendered_artifact = Path("code_intelligence") / "rendered" / str(file_path)
            rendered_link = artifact_link(
                session_dir,
                output_dir,
                "archivo final",
                rendered_artifact,
                require_exists=True,
            )
        if rendered_link:
            links.append(rendered_link)
        line_number = code_event_start_line(event)
        if rendered_artifact is not None and line_number is not None:
            rendered_line_link = artifact_link(
                session_dir,
                output_dir,
                f"archivo final linea {line_number}",
                rendered_artifact,
                fragment=f"L{line_number}",
                require_exists=True,
            )
            if rendered_line_link:
                links.append(rendered_line_link)
        links = [link for link in links if link]
        heading = [
            html_heading_with_anchor(
                3,
                anchor,
                f"Captura {index}: <code>{html_escape(file_path)}</code>",
            ),
            "",
            f"- Tiempo: {format_time(frame_to_seconds(frame, fps))}",
            f"- Frame: {frame}",
            f"- Rango: {line_range_text(event)}",
            f"- Calidad del frame: {quality or 'sin dato'}",
            f"- Confianza: {confidence:.2f}" if confidence is not None else "- Confianza: sin dato",
            f"- Merge: {merge_text(event)}",
        ]
        if links:
            heading.append(f"- Links: {', '.join(links)}")
        cells.append(cell_markdown("\n".join(heading)))
        cells.append(cell_code(code_lines_to_text(event, max_lines_per_event), language=language))
    if len(editor_events) > max_events:
        cells.append(
            cell_markdown(
                f"_Se omitieron {len(editor_events) - max_events} capturas de codigo por limite local._"
            )
        )
    return cells


def build_rendered_file_cells(
    session_dir: Path,
    output_dir: Path,
    include_env: bool,
    max_files: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    root = session_dir / "code_intelligence" / "rendered"
    files = rendered_files(session_dir, include_env)
    cells: list[dict[str, Any]] = [
        cell_markdown(
            "\n".join(
                [
                    "## Archivos reconstruidos finales",
                    "",
                    "Estos son los archivos de texto generados al final de la sesion.",
                    "Los archivos de imagen o binarios se omiten. `.env` se omite salvo que se use `--include-env`.",
                ]
            )
        )
    ]
    for path in files[:max_files]:
        rel = path.relative_to(root) if root.exists() else path.name
        text, truncated = read_text_file(path, max_chars)
        language = language_for_path(path)
        file_link = artifact_link(
            session_dir,
            output_dir,
            "abrir archivo",
            path,
            require_exists=True,
        )
        meta = [f"### `{rel}`", "", f"- Lenguaje/formato inferido: `{language}`"]
        if file_link:
            meta.append(f"- Link: {file_link}")
        if truncated:
            meta.append("- Archivo truncado en esta bitacora por limite local.")
        cells.append(cell_markdown("\n".join(meta)))
        if language == "python":
            cells.append(cell_code(text, language=language))
        else:
            cells.append(cell_markdown(fenced(text, language)))
    if len(files) > max_files:
        cells.append(cell_markdown(f"_Se omitieron {len(files) - max_files} archivos por limite local._"))
    return cells


def build_notebook(
    session_dir: Path,
    output_path: Path,
    explicit_fps: float | None,
    max_code_events: int,
    max_rendered_files: int,
    max_code_lines_per_event: int,
    max_final_file_chars: int,
    max_timeline_events: int,
    include_env: bool,
) -> dict[str, Any]:
    classifications = load_classifications(session_dir)
    matchings = load_pdf_matchings(session_dir)
    whiteboard_summary = load_whiteboard_summary(session_dir)
    whiteboard_events = load_whiteboard_events(session_dir)
    editor_events = load_code_editor_events(session_dir)
    sidebar_events = load_sidebar_events(session_dir)
    fps = estimate_fps(matchings, explicit_fps)
    rendered = rendered_files(session_dir, include_env)
    pages_detected = as_list(matchings.get("pages_detected"))
    output_dir = output_path.parent

    cells: list[dict[str, Any]] = []
    cells.append(
        cell_markdown(
            "\n".join(
                [
                    "# Bitacora accesible de la clase",
                    "",
                    f"- Sesion: `{session_dir}`",
                    f"- Notebook generado: `{output_path}`",
                    f"- Fecha de generacion: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"- FPS estimado para tiempos por frame: {fps:.3f}",
                    "",
                    "Este notebook usa la metadata textual de la sesion. No depende de imagenes ni de llamadas remotas.",
                ]
            )
        )
    )
    summary_rows = [
        ["Clasificaciones leidas", len(classifications)],
        ["Diapositivas detectadas", len(pages_detected)],
        ["Eventos de pizarron", len(whiteboard_events)],
        ["Capturas de codigo", len(editor_events)],
        ["Respuestas de sidebar", len(sidebar_events)],
        ["Archivos finales de texto", len(rendered)],
    ]
    cells.append(cell_markdown("\n".join(["## Resumen general", "", markdown_table(["Dato", "Valor"], summary_rows)])))

    pdf_name = matchings.get("pdf_file") or "PDF no registrado"
    total_pages = matchings.get("total_pages", "")
    slide_intro = [
        "## Diapositivas detectadas",
        "",
        f"- PDF: `{pdf_name}`",
        f"- Total de paginas del PDF: `{total_pages}`",
        f"- Paginas detectadas: {', '.join(str(page) for page in pages_detected) if pages_detected else 'sin datos'}",
        "",
        slide_table(matchings, session_dir, output_dir),
    ]
    cells.append(cell_markdown("\n".join(slide_intro)))

    timeline_items: list[TimelineItem] = []
    timeline_items.extend(classification_state_events(classifications, fps))
    timeline_items.extend(slide_timeline_events(matchings, fps))
    timeline_items.extend(whiteboard_timeline_events(whiteboard_events, fps))
    timeline_items.extend(code_timeline_events(editor_events, fps))
    cells.append(cell_markdown(build_timeline_cell(timeline_items, max_timeline_events)))
    cells.append(cell_markdown(whiteboard_section(whiteboard_summary, whiteboard_events, fps, session_dir, output_dir)))
    cells.append(cell_markdown(repository_summary_section(session_dir, sidebar_events)))
    cells.extend(
        build_code_capture_cells(
            editor_events,
            fps,
            max_code_events,
            max_code_lines_per_event,
            session_dir,
            output_dir,
        )
    )
    cells.append(cell_markdown(reference_comparison_section(session_dir, output_dir)))
    cells.extend(
        build_rendered_file_cells(
            session_dir=session_dir,
            output_dir=output_dir,
            include_env=include_env,
            max_files=max_rendered_files,
            max_chars=max_final_file_chars,
        )
    )
    cells.append(
        cell_markdown(
            "\n".join(
                [
                    "## Notas de accesibilidad",
                    "",
                    "- La bitacora evita depender de imagenes; el contenido principal esta en texto, tablas y bloques de codigo.",
                    "- Cuando el sistema solo conoce un frame, el tiempo se estima con el FPS calculado desde las coincidencias del PDF.",
                    "- Las capturas de codigo son evidencia intermedia; los archivos finales reconstruidos aparecen al cierre.",
                    "- Si una persona revisa esta bitacora con lector de pantalla, conviene leer primero el resumen, luego la linea de tiempo y finalmente los archivos finales.",
                ]
            )
        )
    )
    return notebook(cells)


def write_session_notebook(
    session_dir: Path,
    output_path: Path,
    *,
    explicit_fps: float | None = None,
    max_code_events: int = DEFAULT_MAX_CODE_EVENTS,
    max_rendered_files: int = DEFAULT_MAX_RENDERED_FILES,
    max_code_lines_per_event: int = DEFAULT_MAX_CODE_LINES_PER_EVENT,
    max_final_file_chars: int = DEFAULT_MAX_FINAL_FILE_CHARS,
    max_timeline_events: int = DEFAULT_MAX_TIMELINE_EVENTS,
    include_env: bool = False,
    atomic: bool = True,
) -> dict[str, Any]:
    session_dir = Path(session_dir).resolve()
    output_path = Path(output_path).resolve()
    nb = build_notebook(
        session_dir=session_dir,
        output_path=output_path,
        explicit_fps=explicit_fps,
        max_code_events=max(0, max_code_events),
        max_rendered_files=max(0, max_rendered_files),
        max_code_lines_per_event=max(1, max_code_lines_per_event),
        max_final_file_chars=max(1_000, max_final_file_chars),
        max_timeline_events=max(1, max_timeline_events),
        include_env=bool(include_env),
    )
    if atomic:
        write_json_atomic(output_path, nb)
    else:
        write_json(output_path, nb)
    return {
        "ok": True,
        "session_dir": str(session_dir),
        "notebook": str(output_path),
        "cell_count": len(nb.get("cells", [])),
    }


class LiveBitacoraWriter:
    # Regenera la bitacora periodicamente sin bloquear el loop principal.
    def __init__(
        self,
        session_dir: Path | str,
        *,
        output_name: str = DEFAULT_LIVE_NOTEBOOK_NAME,
        interval_seconds: float = 5.0,
        explicit_fps: float | None = None,
        enabled: bool = True,
    ):
        self.session_dir = Path(session_dir).resolve()
        self.output_path = self.session_dir / output_name
        self.interval_seconds = max(0.5, float(interval_seconds))
        self.explicit_fps = explicit_fps
        self.enabled = bool(enabled)
        self.last_result: dict[str, Any] | None = None
        self.last_error: str | None = None
        self._last_started = 0.0
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # Actualiza el FPS usado cuando el video ya reporto su valor real.
    def set_fps(self, fps: float | None) -> None:
        if fps is not None and fps > 0:
            self.explicit_fps = float(fps)

    # Lanza una regeneracion si ya paso el intervalo configurado.
    def maybe_update(self, *, force: bool = False, wait: bool = False) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        current_thread: threading.Thread | None = None
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                current_thread = self._thread
            elif not force and (time.perf_counter() - self._last_started) < self.interval_seconds:
                return self.last_result

        if current_thread is not None:
            if not force or not wait:
                return self.last_result
            current_thread.join()

        if wait:
            self._write_once()
            return self.last_result

        thread = threading.Thread(target=self._write_once, name="bitacora-vat-writer", daemon=True)
        with self._lock:
            self._thread = thread
            self._last_started = time.perf_counter()
        thread.start()
        return self.last_result

    # Espera cualquier actualizacion pendiente y escribe una version final.
    def close(self) -> dict[str, Any] | None:
        return self.maybe_update(force=True, wait=True)

    # Construye el notebook y lo reemplaza atomicamente.
    def _write_once(self) -> None:
        try:
            result = write_session_notebook(
                self.session_dir,
                self.output_path,
                explicit_fps=self.explicit_fps,
                atomic=True,
            )
            with self._lock:
                self.last_result = result
                self.last_error = None
                self._last_started = time.perf_counter()
        except Exception as exc:
            with self._lock:
                self.last_error = str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera una bitacora accesible en formato Jupyter Notebook a partir "
            "de una carpeta de sesion del pipeline VAT."
        )
    )
    parser.add_argument(
        "session_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_SESSION_DIR,
        help=f"Carpeta de sesion. Default: {DEFAULT_SESSION_DIR}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Ruta del .ipynb de salida. Default: <session_dir>/{DEFAULT_NOTEBOOK_NAME}",
    )
    parser.add_argument("--fps", type=float, default=None, help="FPS manual para convertir frame a tiempo.")
    parser.add_argument("--max-code-events", type=int, default=DEFAULT_MAX_CODE_EVENTS)
    parser.add_argument("--max-rendered-files", type=int, default=DEFAULT_MAX_RENDERED_FILES)
    parser.add_argument("--max-code-lines-per-event", type=int, default=DEFAULT_MAX_CODE_LINES_PER_EVENT)
    parser.add_argument("--max-final-file-chars", type=int, default=DEFAULT_MAX_FINAL_FILE_CHARS)
    parser.add_argument("--max-timeline-events", type=int, default=DEFAULT_MAX_TIMELINE_EVENTS)
    parser.add_argument(
        "--include-env",
        action="store_true",
        help="Incluye archivos .env reconstruidos. Por default se omiten.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = args.session_dir.resolve()
    if not session_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de sesion: {session_dir}")
    output_path = (args.output or (session_dir / DEFAULT_NOTEBOOK_NAME)).resolve()
    result = write_session_notebook(
        session_dir,
        output_path,
        explicit_fps=args.fps,
        max_code_events=max(0, args.max_code_events),
        max_rendered_files=max(0, args.max_rendered_files),
        max_code_lines_per_event=max(1, args.max_code_lines_per_event),
        max_final_file_chars=max(1_000, args.max_final_file_chars),
        max_timeline_events=max(1, args.max_timeline_events),
        include_env=bool(args.include_env),
        atomic=True,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
