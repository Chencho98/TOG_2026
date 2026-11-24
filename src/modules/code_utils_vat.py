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

# ============================================================================
# GENERAL HELPERS
# ============================================================================

# Configura la ruta de Tesseract si esta disponible.
def configure_tesseract_command(module=None) -> Optional[str]:
    candidates = []
    env_cmd = os.environ.get(OCR_TESSERACT_ENV_VAR)
    if env_cmd:
        candidates.append(env_cmd)

    which_cmd = shutil.which("tesseract")
    if which_cmd:
        candidates.append(which_cmd)

    candidates.extend(COMMON_TESSERACT_PATHS)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate) or candidate == which_cmd:
            if module is not None:
                module.pytesseract.tesseract_cmd = candidate
            return candidate
    return None


try:
    import pytesseract
    from pytesseract import Output, TesseractNotFoundError

    # Ruta final de Tesseract detectada para OCR local.
    DETECTED_TESSERACT_CMD = configure_tesseract_command(pytesseract)
    # Indica que pytesseract esta disponible para OCR local.
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    Output = None
    TesseractNotFoundError = RuntimeError
    # Indica que pytesseract no esta disponible y se desactiva el OCR local.
    PYTESSERACT_AVAILABLE = False
    # Ruta nula cuando no se pudo importar pytesseract.
    DETECTED_TESSERACT_CMD = None


# Limita un valor entero dentro del rango indicado.
def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


# Suaviza una senal numerica con una ventana movil.
def smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal, kernel, mode="same")


# Normaliza espacios y saltos de linea en texto.
def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


# Normaliza una linea para comparaciones tolerantes.
def normalize_code_line_for_match(text: str) -> str:
    raw = str(text or "").replace("\t", "    ")
    if not raw.strip():
        return BLANK_EDITOR_LINE_SENTINEL
    return re.sub(r"\s+", " ", raw.strip())


# Elimina fences Markdown de una respuesta JSON.
def strip_json_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# Convierte una respuesta parecida a JSON en diccionario.
def parse_jsonish_dict(text: str) -> Dict[str, Any]:
    text = strip_json_fence(text)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


# Normaliza texto usado para identificar lenguaje.
def normalize_language_text(text: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", str(text or "").upper())
    return normalized.translate(
        str.maketrans(
            {
                "0": "O",
                "1": "I",
                "5": "S",
                "8": "B",
            }
        )
    )


# Calcula la distancia de edicion entre dos cadenas.
def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion_cost = current_row[right_index - 1] + 1
            deletion_cost = previous_row[right_index] + 1
            substitution_cost = previous_row[right_index - 1] + (0 if left_char == right_char else 1)
            current_row.append(min(insertion_cost, deletion_cost, substitution_cost))
        previous_row = current_row
    return previous_row[-1]


# Evalua si un token parece decir Python.
def looks_like_python(token: str) -> bool:
    if not token:
        return False

    token = normalize_language_text(token)
    if not token:
        return False

    if "PYTHON" in token:
        return True

    distance = levenshtein_distance(token, "PYTHON")
    if "H" in token and distance <= 3:
        return True
    if 4 <= len(token) <= 8 and "HON" in token:
        return True
    if 4 <= len(token) <= 8 and token.endswith("ION") and token[0] in {"P", "R"}:
        return True
    if 4 <= len(token) <= 8 and token.endswith("TON") and token[0] in {"P", "R"} and distance <= 3:
        return True
    return False


# Extrae el lenguaje reconocido desde texto visible.
def extract_language(text: str) -> Optional[str]:
    raw_text = str(text or "")
    candidates = []

    normalized = normalize_language_text(raw_text)
    if normalized:
        candidates.append(normalized)

    for chunk in re.findall(r"[A-Za-z0-9_]+", raw_text):
        normalized_chunk = normalize_language_text(chunk)
        if normalized_chunk and normalized_chunk not in candidates:
            candidates.append(normalized_chunk)

    for candidate in candidates:
        if looks_like_python(candidate):
            return "PYTHON"
    return None


# Convierte un valor a float con fallback seguro.
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


# Convierte un valor a int cuando es posible.
def safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


# Lee una bandera booleana desde variables de entorno.
def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None or str(raw_value).strip() == "":
        return bool(default)
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


# Busca una API key remota en las variables configuradas.
def resolve_remote_api_key(env_names: Optional[Tuple[str, ...]] = None) -> str:
    if env_names is None:
        env_names = tuple(CODE_VISION_ACTIVE_PROFILE.get("api_key_envs", (CODE_VISION_API_KEY_ENV_VAR,)))
    for env_name in env_names:
        value = str(os.environ.get(str(env_name), "") or "").strip()
        if value:
            return value
    return ""


# Normaliza y valida rutas relativas de codigo.
def sanitize_relative_code_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    text = str(path).strip().replace("\\", "/")
    text = re.sub(r"^[A-Za-z]:/", "", text)
    parts = [part for part in text.split("/") if part and part not in {".", ".."}]
    if not parts:
        return None
    return "/".join(parts)


# Obtiene el nombre base de una ruta normalizada.
def path_basename(path: Optional[str]) -> str:
    sanitized = sanitize_relative_code_path(path)
    if not sanitized:
        return ""
    return sanitized.split("/")[-1]


# Obtiene el nombre base sin extension de una ruta.
def path_stem(path: Optional[str]) -> str:
    basename = path_basename(path)
    if not basename:
        return ""
    if "." not in basename:
        return basename.lower()
    return basename.rsplit(".", 1)[0].lower()


# Obtiene la extension de una ruta normalizada.
def path_extension(path: Optional[str]) -> str:
    basename = path_basename(path)
    if "." not in basename:
        return ""
    return "." + basename.rsplit(".", 1)[1].lower()


# Evalua si una linea parece ruido de separador.
def looks_like_separator_noise(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) < 5:
        return False
    if stripped.lstrip().startswith("#"):
        return False
    compact = stripped.replace(" ", "")
    if not compact:
        return False
    return len(set(compact)) <= 2 and all(char in "=*-_~:#/\\|." for char in compact)


# Indica si una linea del editor representa una linea vacia.
def is_blank_editor_line_text(text: Any) -> bool:
    return str(text or "").strip() == ""


# Infiere el tipo semantico de una linea del editor.
def infer_editor_line_kind(text: Any) -> str:
    if is_blank_editor_line_text(text):
        return "blank"
    if looks_like_separator_noise(text):
        return "separator"
    stripped = str(text or "").lstrip()
    if stripped.startswith("#"):
        compact_tail = stripped[1:].strip().replace(" ", "")
        if len(compact_tail) >= 3 and len(set(compact_tail)) <= 2 and all(
            char in "=*-_~:/\\|." for char in compact_tail
        ):
            return "separator"
    return "code"


# Decide si una linea detectada debe conservarse.
def should_keep_editor_line(text: Any, line_number: Optional[int]) -> bool:
    if looks_like_editor_hud_noise(text):
        return False
    if is_blank_editor_line_text(text):
        return line_number is not None
    if looks_like_separator_noise(text):
        return False
    return bool(str(text or "").rstrip())


# Genera una firma alfanumerica de una linea de codigo.
def code_line_alnum_signature(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(text or "")).lower()


# Genera una firma alfabetica de una linea de codigo.
def code_line_alpha_signature(text: Any) -> str:
    return re.sub(r"[^A-Za-z]+", "", str(text or "")).lower()


# Calcula el peso de simbolos en una linea de codigo.
def code_line_symbol_score(text: Any) -> int:
    raw = str(text or "")
    score = sum(1 for char in raw if char in "#=_:()[]{}.,'\"")
    if raw.lstrip().startswith("#"):
        score += 3
    if "=" in raw:
        score += 2
    return score


# Compara si dos lineas son variantes simbolicas.
def lines_are_symbolic_variants(left: Any, right: Any) -> bool:
    left_norm = normalize_code_line_for_match(left)
    right_norm = normalize_code_line_for_match(right)
    if left_norm == right_norm:
        return True
    if left_norm == BLANK_EDITOR_LINE_SENTINEL or right_norm == BLANK_EDITOR_LINE_SENTINEL:
        return False
    left_alnum = code_line_alnum_signature(left)
    right_alnum = code_line_alnum_signature(right)
    if left_alnum and left_alnum == right_alnum:
        return True
    left_alpha = code_line_alpha_signature(left)
    right_alpha = code_line_alpha_signature(right)
    return bool(
        left_alpha
        and left_alpha == right_alpha
        and max(len(left_alpha), len(right_alpha)) >= 4
        and abs(len(str(left or "")) - len(str(right or ""))) <= 32
    )


# Elige la version mas confiable de una linea.
def choose_preferred_line_text(existing_text: Any, new_text: Any, prefer_new: bool) -> str:
    existing = str(existing_text or "")
    new = str(new_text or "")
    if not existing:
        return new
    if not new:
        return existing

    existing_norm = normalize_code_line_for_match(existing)
    new_norm = normalize_code_line_for_match(new)
    if existing_norm == new_norm or lines_are_symbolic_variants(existing, new):
        existing_symbol_score = code_line_symbol_score(existing)
        new_symbol_score = code_line_symbol_score(new)
        if new_symbol_score > existing_symbol_score:
            return new
        if existing_symbol_score > new_symbol_score:
            return existing
        if len(new.rstrip()) > len(existing.rstrip()) + 1:
            return new
        if len(existing.rstrip()) > len(new.rstrip()) + 1:
            return existing

    return new if prefer_new else existing


# Detecta texto que pertenece al HUD del editor.
def looks_like_editor_hud_noise(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    compact = re.sub(r"\s+", "", raw).lower()
    lowered = raw.lower()
    if EDITOR_STATUS_LINE_RE.search(lowered):
        return True
    token_hits = len({match.group(0).lower() for match in EDITOR_STATUS_TOKEN_RE.finditer(lowered)})
    if token_hits >= 2:
        return True
    if compact in {
        "utf",
        "utf-",
        "utf8",
        "ut-",
        "crf",
        "crlf",
        "lf",
        "spaces",
        "selected",
        "informational",
    }:
        return True
    if compact.startswith("utf") and len(compact) <= 6:
        return True
    if compact.startswith("crf") and len(compact) <= 8:
        return True
    if compact.startswith("ln") and "col" in compact:
        return True
    if compact.endswith("spaces") and len(compact) <= 14:
        return True
    return False


# Calcula una puntuacion suave de linea Python.
def python_line_soft_score(text: str) -> float:
    stripped = str(text or "").rstrip()
    if not stripped:
        return -0.4
    if looks_like_editor_hud_noise(stripped):
        return -0.9

    lowered = stripped.lstrip()
    score = 0.0
    if lowered.startswith("#"):
        score += 0.18
    if re.fullmatch(r"import\s+[A-Za-z_][\w.]*(\s*,\s*[A-Za-z_][\w.]*)*", lowered):
        score += 0.24
    elif lowered.startswith("import "):
        score -= 0.20
    if re.fullmatch(r"from\s+[A-Za-z_][\w.]*\s+import\s+[\w*., ]+", lowered):
        score += 0.24
    elif lowered.startswith("from "):
        score -= 0.20
    if re.fullmatch(r"(class|def)\s+[A-Za-z_]\w*(\(.*\))?\s*:\s*", lowered):
        score += 0.22
    elif lowered.startswith(("class ", "def ")):
        score -= 0.25
    if re.fullmatch(r"[A-Za-z_]\w*\s*=\s*.+", lowered):
        score += 0.10
    if re.search(r"[=:_(){}\[\],.]", stripped):
        score += 0.04
    if re.search(r"\bImport\b", stripped):
        score -= 0.15
    if re.search(r"\b(def|class)\s+[A-Za-z_]+\s+[A-Za-z_]", lowered):
        score -= 0.12
    if " false" in f" {lowered}" and "False" not in stripped:
        score -= 0.12
    return score


# Puntua que tan parecido a codigo es un texto.
def code_like_text_score(text: str, language_hint: Optional[str] = None) -> float:
    stripped = str(text or "").rstrip()
    if not stripped:
        return -1.0
    if looks_like_separator_noise(stripped):
        return -0.7
    if looks_like_editor_hud_noise(stripped):
        return -1.0

    score = 0.0015 * len(stripped)
    if any(char in stripped for char in "=:_()[]{}.,"):
        score += 0.08
    if stripped.lstrip().startswith(("#", "import ", "from ", "class ", "def ")):
        score += 0.10
    if str(language_hint or "").upper() == "PYTHON":
        score += python_line_soft_score(stripped)
    return score


# Cuenta tramos sostenidos que superan un umbral.
def count_sustained_runs(signal: np.ndarray, threshold: float, min_run: int) -> int:
    if signal is None or signal.size == 0:
        return 0
    count = 0
    run_start: Optional[int] = None
    for index, value in enumerate(signal):
        if float(value) >= float(threshold):
            if run_start is None:
                run_start = index
            continue
        if run_start is not None and (index - run_start) >= int(min_run):
            count += 1
        run_start = None
    if run_start is not None and (len(signal) - run_start) >= int(min_run):
        count += 1
    return count


# Selecciona el cluster mas denso de numeros de linea.
def choose_densest_line_number_cluster(
    entries: List[Dict[str, Any]],
    max_gap: int,
) -> Optional[set[int]]:
    explicit_entries: List[Tuple[int, int, float]] = []
    for index, entry in enumerate(entries):
        line_number = safe_int(entry.get("number"))
        if line_number is None or line_number < 1:
            continue
        explicit_entries.append(
            (
                index,
                int(line_number),
                float(code_like_text_score(entry.get("text", ""), None) + safe_float(entry.get("confidence"), 0.0)),
            )
        )
    if len(explicit_entries) < 3:
        return None

    clusters: List[List[Tuple[int, int, float]]] = []
    current_cluster: List[Tuple[int, int, float]] = [explicit_entries[0]]
    for candidate in explicit_entries[1:]:
        previous = current_cluster[-1]
        diff = int(candidate[1]) - int(previous[1])
        if 0 < diff <= int(max_gap):
            current_cluster.append(candidate)
            continue
        clusters.append(current_cluster)
        current_cluster = [candidate]
    if current_cluster:
        clusters.append(current_cluster)

    if not clusters:
        return None

    best_cluster = max(
        clusters,
        key=lambda cluster: (
            len(cluster),
            round(sum(item[2] for item in cluster), 3),
            -((cluster[-1][1] - cluster[0][1]) if len(cluster) > 1 else 0),
        ),
    )
    if len(best_cluster) < 3:
        return None
    return {item[0] for item in best_cluster}


# Codifica una imagen como data URL para requests remotos.
def image_to_data_url(
    image: np.ndarray,
    image_format: str = "jpg",
    jpeg_quality: int = 90,
) -> Optional[str]:
    if image is None or image.size == 0:
        return None
    normalized_format = str(image_format or "jpg").strip().lower()
    if normalized_format == "png":
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        mime_type = "image/png"
    else:
        quality = max(40, min(100, int(jpeg_quality)))
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mime_type = "image/jpeg"
    if not ok:
        return None
    return f"data:{mime_type};base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


# Redimensiona un frame para caber dentro del area objetivo.
def resize_frame_to_fit(
    image: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    if image is None or image.size == 0 or image.ndim < 2:
        return image
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return image

    scale_w = float(target_w) / float(width)
    scale_h = float(target_h) / float(height)
    scale = min(1.0, scale_w, scale_h)
    if scale >= 0.999:
        return image

    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)


# Reduce el ROI de proyeccion para clasificacion general.
def downscale_projection_roi(image: np.ndarray) -> np.ndarray:
    return resize_frame_to_fit(
        image,
        CODE_CLASSIFICATION_TARGET_W,
        CODE_CLASSIFICATION_TARGET_H,
    )


# Convierte una etiqueta de calidad en rango comparable.
def editor_frame_quality_rank(quality: Optional[str]) -> int:
    normalized = str(quality or "usable").strip().lower()
    ranking = {
        "usable": 3,
        "partial_blur": 2,
        "scroll_blur": 1,
        "unknown": 0,
    }
    return int(ranking.get(normalized, 0))


# Clasifica la calidad visual de un frame del editor.
def classify_editor_frame_quality(metrics: Dict[str, float]) -> str:
    sharpness = safe_float(metrics.get("sharpness"), 0.0)
    edge_density = safe_float(metrics.get("edge_density"), 0.0)
    axis_balance = safe_float(metrics.get("axis_balance"), 1.0)
    row_variation = safe_float(metrics.get("row_variation"), 0.0)

    if (
        sharpness <= CODE_EDITOR_SCROLL_BLUR_SHARPNESS_MIN
        and edge_density <= CODE_EDITOR_SCROLL_BLUR_EDGE_MAX
        and row_variation <= CODE_EDITOR_SCROLL_BLUR_ROW_VARIATION_MAX
    ):
        return "scroll_blur"
    if (
        sharpness <= (CODE_EDITOR_SCROLL_BLUR_SHARPNESS_MIN + 0.25)
        and edge_density <= (CODE_EDITOR_SCROLL_BLUR_EDGE_MAX + 0.015)
        and axis_balance <= CODE_EDITOR_SCROLL_BLUR_AXIS_BALANCE_MAX
        and row_variation <= (CODE_EDITOR_SCROLL_BLUR_ROW_VARIATION_MAX + 0.006)
    ):
        return "scroll_blur"
    if (
        sharpness <= CODE_EDITOR_PARTIAL_BLUR_SHARPNESS_MIN
        or (
            edge_density <= CODE_EDITOR_PARTIAL_BLUR_EDGE_MAX
            and axis_balance <= CODE_EDITOR_PARTIAL_BLUR_AXIS_BALANCE_MAX
            and row_variation <= CODE_EDITOR_PARTIAL_BLUR_ROW_VARIATION_MAX
        )
    ):
        return "partial_blur"
    return "usable"


# Separa una linea OCR que contiene varios numeros de linea.
def split_compound_editor_line(
    line_number: Optional[int],
    text: str,
) -> List[Tuple[Optional[int], str]]:
    raw_text = str(text or "").rstrip()
    if not raw_text:
        if line_number is not None:
            return [(line_number, "")]
        return []

    normalized_text = raw_text
    if line_number is not None:
        normalized_text = re.sub(
            rf"^\s*{int(line_number)}\s+",
            "",
            normalized_text,
            count=1,
        ).rstrip()

    matches = list(COMPOUND_EDITOR_LINE_NUMBER_RE.finditer(normalized_text))
    if not matches:
        return [(line_number, normalized_text)]

    segments: List[Tuple[Optional[int], str]] = []
    current_number = line_number
    cursor = 0
    for match in matches:
        number = safe_int(match.group("number"))
        if current_number is None and match.start() == 0 and number is not None:
            current_number = number
            cursor = match.end()
            continue
        prefix = normalized_text[cursor:match.start()].strip()
        if prefix and current_number is not None:
            segments.append((current_number, prefix))
        current_number = number
        cursor = match.end()

    tail = normalized_text[cursor:].strip()
    if tail and current_number is not None:
        segments.append((current_number, tail))

    return segments or [(line_number, normalized_text)]


# Evalua si un texto parece nombre de archivo de codigo.
def looks_like_code_filename(text: Optional[str]) -> bool:
    sanitized = sanitize_relative_code_path(text)
    if not sanitized:
        return False
    extension = path_extension(sanitized).lstrip(".").lower()
    return extension in COMMON_CODE_EXTENSIONS


# Normaliza texto de etiquetas devueltas por Florence.
def normalize_florence_label_text(text: Any) -> str:
    normalized = str(text or "").replace("</s>", " ").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace("“", "\"").replace("”", "\"").replace("’", "'")
    return normalized.strip()


# Normaliza tokens numericos detectados por Florence.
def normalize_florence_numeric_token(text: Any) -> Optional[str]:
    raw = normalize_florence_label_text(text)
    if not raw:
        return None
    compact = re.sub(r"\s+", "", raw)
    compact = compact.translate(
        str.maketrans(
            {
                "O": "0",
                "o": "0",
                "Q": "0",
                "D": "0",
                "I": "1",
                "l": "1",
                "|": "1",
                "S": "5",
                "s": "5",
                "B": "8",
            }
        )
    )
    if not compact or not re.fullmatch(r"\d{1,4}", compact):
        return None
    return compact


# Evalua si una etiqueta Florence parece numero de linea.
def looks_like_florence_line_number(text: Any) -> bool:
    compact = normalize_florence_numeric_token(text)
    return bool(compact)


# Convierte una caja de Florence a rectangulo numerico.
def florence_box_to_rect(box: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(box, (list, tuple)):
        return None
    values = [float(value) for value in box if isinstance(value, (int, float))]
    if len(values) >= 8 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    if len(values) >= 4:
        return values[0], values[1], values[2], values[3]
    return None


# Extrae etiquetas textuales desde la salida de Florence.
def extract_florence_labels(parsed_output: Any) -> List[str]:
    labels: List[str] = []

    # Recorre estructuras anidadas para extraer elementos relevantes.
    def visit(node: Any) -> None:
        if isinstance(node, dict):
            raw_labels = node.get("labels")
            if isinstance(raw_labels, list):
                for label in raw_labels:
                    normalized = normalize_florence_label_text(label)
                    if normalized:
                        labels.append(normalized)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(parsed_output)
    deduped: List[str] = []
    for label in labels:
        if not deduped or deduped[-1] != label:
            deduped.append(label)
    return deduped[:CODE_FLORENCE_MAX_LABELS]

# Construye la lista ordenada de modelos Florence candidatos.
def candidate_models(explicit_model: str, defaults: List[str]) -> List[str]:
    explicit_model = str(explicit_model or "").strip()
    if explicit_model:
        return [explicit_model]
    return list(defaults)


@dataclass
class FlorenceLoadedModel:
    model_id: str
    processor: Any
    model: Any
    runtime_backend: str
    runtime_device: str
    runtime_class: str = ""


# Normaliza el backend Florence solicitado.
def normalize_florence_backend(requested: str) -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in {"ov", "openvino"}:
        return "openvino"
    return "torch"


# Maneja torch has xpu.
def _torch_has_xpu(torch_module: Any) -> bool:
    if not hasattr(torch_module, "xpu") or not hasattr(torch_module.xpu, "is_available"):
        return False
    try:
        return bool(torch_module.xpu.is_available())
    except Exception:
        return False


# Importa dependencias necesarias para Florence.
def import_florence_dependencies() -> Tuple[Any, Any, Any, Any]:
    if Image is None:
        raise RuntimeError(
            "Falta Pillow para Florence-2. Instala al menos: pip install pillow torch torchvision transformers timm"
        )
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    except Exception as exc:
        raise RuntimeError(
            "Faltan dependencias para Florence-2. "
            "Instala al menos: pip install torch torchvision transformers pillow timm"
        ) from exc
    return torch, AutoConfig, AutoModelForCausalLM, AutoProcessor


# Mueve un batch de tensores al dispositivo indicado.
def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


# Carga clases OpenVINO compatibles con Florence.
def import_openvino_florence_model_classes() -> List[Tuple[str, Any]]:
    imported: List[Tuple[str, Any]] = []
    seen: set[int] = set()
    candidates = (
        ("optimum.intel.openvino", "OVModelForVisualCausalLM"),
        ("optimum.intel", "OVModelForVisualCausalLM"),
        ("optimum.intel.openvino", "OVModelForVision2Seq"),
        ("optimum.intel", "OVModelForVision2Seq"),
        ("optimum.intel.openvino", "OVModelForCausalLM"),
        ("optimum.intel", "OVModelForCausalLM"),
    )
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name, None)
        except Exception:
            continue
        if model_class is None or id(model_class) in seen:
            continue
        imported.append((f"{module_name}.{class_name}", model_class))
        seen.add(id(model_class))
    return imported


# Normaliza el dispositivo OpenVINO solicitado.
def normalize_openvino_device(requested: str) -> str:
    normalized = str(requested or "auto").strip().lower()
    mapping = {
        "": "AUTO",
        "auto": "AUTO",
        "gpu": "GPU",
        "intel_gpu": "GPU",
        "arc": "GPU",
        "xpu": "GPU",
        "cpu": "CPU",
        "npu": "NPU",
    }
    if normalized in mapping:
        return mapping[normalized]
    return normalized.upper()


# Lista dispositivos OpenVINO a intentar.
def openvino_device_candidates(requested: str) -> List[str]:
    primary = normalize_openvino_device(requested)
    candidates: List[str] = [primary]
    if primary == "GPU":
        candidates.extend(["AUTO", "CPU"])
    elif primary == "AUTO":
        candidates.extend(["GPU", "CPU"])
    elif primary != "CPU":
        candidates.append("CPU")
    deduped: List[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


# Asegura que exista un atributo opcional en un objeto.
def patch_optional_attr(target: Any, attr_name: str, default: Any = None) -> None:
    if target is None:
        return
    try:
        getattr(target, attr_name)
    except Exception:
        try:
            setattr(target, attr_name, default)
        except Exception:
            pass


# Parchea configuracion Florence para atributos opcionales faltantes.
def patch_florence_config(target: Any, visited: Optional[set[int]] = None) -> None:
    if target is None:
        return
    if visited is None:
        visited = set()

    target_id = id(target)
    if target_id in visited:
        return
    visited.add(target_id)

    for attr_name in (
        "forced_bos_token_id",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "decoder_start_token_id",
    ):
        patch_optional_attr(target, attr_name, None)

    for child_name in ("text_config", "language_config", "decoder", "decoder_config"):
        child = getattr(target, child_name, None)
        if child is not None and child is not target:
            patch_florence_config(child, visited)


# Selecciona el dispositivo Torch disponible para Florence.
def choose_torch_device(torch_module: Any, requested: str) -> str:
    requested = str(requested or "auto").strip().lower()
    if requested in {"", "auto"}:
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            return "cuda"
        if _torch_has_xpu(torch_module):
            return "xpu"
        if (
            hasattr(torch_module, "backends")
            and hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"
    if requested in {"gpu", "intel_gpu", "arc", "xpu"}:
        if _torch_has_xpu(torch_module):
            return "xpu"
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch_module, "backends")
            and hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"
    return requested


# Libera memoria asociada al backend Torch.
def cleanup_torch(torch_module: Any, device: Optional[str]) -> None:
    gc.collect()
    if device == "cuda" and hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass
    if device == "xpu" and hasattr(torch_module, "xpu") and hasattr(torch_module.xpu, "empty_cache"):
        try:
            torch_module.xpu.empty_cache()
        except Exception:
            pass


# Carga Florence usando backend Torch.
def load_florence_torch_model(
    model_candidates: List[str],
    requested_device: str,
    auto_config_cls: Any,
    auto_model_cls: Any,
    auto_processor_cls: Any,
    torch_module: Any,
) -> FlorenceLoadedModel:
    device = choose_torch_device(torch_module, requested_device)
    errors: List[str] = []
    for model_id in model_candidates:
        try:
            config = auto_config_cls.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            patch_florence_config(config)

            processor = auto_processor_cls.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            model = auto_model_cls.from_pretrained(
                model_id,
                config=config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch_module.float32,
            )
            patch_florence_config(getattr(model, "config", None))
            patch_florence_config(getattr(model, "generation_config", None))
            if hasattr(model, "eval"):
                model = model.eval()
            if hasattr(model, "to"):
                model = model.to(device)
            return FlorenceLoadedModel(
                model_id=model_id,
                processor=processor,
                model=model,
                runtime_backend="torch",
                runtime_device=device,
                runtime_class=type(model).__name__,
            )
        except Exception as exc:
            errors.append(f"{model_id}: {exc}")
    raise RuntimeError("No se pudo cargar Florence-2 con torch.\n" + "\n".join(errors))


# Carga Florence usando backend OpenVINO.
def load_florence_openvino_model(
    model_candidates: List[str],
    requested_device: str,
    auto_processor_cls: Any,
) -> FlorenceLoadedModel:
    model_classes = import_openvino_florence_model_classes()
    if not model_classes:
        raise RuntimeError(
            "OpenVINO/Optimum Intel no está disponible. "
            "Instala al menos: pip install openvino optimum-intel"
        )

    errors: List[str] = []
    for model_id in model_candidates:
        try:
            processor = auto_processor_cls.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
        except Exception as exc:
            errors.append(f"{model_id} [processor]: {exc}")
            continue

        for runtime_device in openvino_device_candidates(requested_device):
            for class_label, model_class in model_classes:
                attempts = (
                    {
                        "export": True,
                        "trust_remote_code": True,
                        "compile": False,
                        "device": runtime_device,
                    },
                    {
                        "export": True,
                        "trust_remote_code": True,
                        "device": runtime_device,
                    },
                    {
                        "export": True,
                        "compile": False,
                        "device": runtime_device,
                    },
                    {
                        "export": True,
                        "device": runtime_device,
                    },
                    {
                        "export": True,
                        "trust_remote_code": True,
                        "compile": False,
                    },
                    {
                        "export": True,
                        "trust_remote_code": True,
                    },
                    {
                        "export": True,
                    },
                )
                for attempt_kwargs in attempts:
                    try:
                        model = model_class.from_pretrained(model_id, **attempt_kwargs)
                        patch_florence_config(getattr(model, "config", None))
                        patch_florence_config(getattr(model, "generation_config", None))
                        if hasattr(model, "to"):
                            try:
                                model.to(runtime_device)
                            except Exception:
                                pass
                        if hasattr(model, "compile"):
                            try:
                                model.compile()
                            except Exception:
                                pass
                        if hasattr(model, "eval"):
                            try:
                                model = model.eval()
                            except Exception:
                                pass
                        return FlorenceLoadedModel(
                            model_id=model_id,
                            processor=processor,
                            model=model,
                            runtime_backend="openvino",
                            runtime_device=runtime_device,
                            runtime_class=class_label,
                        )
                    except Exception as exc:
                        errors.append(f"{model_id} [{class_label} {runtime_device}]: {exc}")

    raise RuntimeError("No se pudo cargar Florence-2 con OpenVINO.\n" + "\n".join(errors))


# Carga Florence eligiendo backend y dispositivo apropiados.
def load_florence_model(
    model_candidates: List[str],
    backend: str,
    device: str,
    auto_config_cls: Any,
    auto_model_cls: Any,
    auto_processor_cls: Any,
    torch_module: Any,
) -> FlorenceLoadedModel:
    requested_backend = normalize_florence_backend(backend)
    errors: List[str] = []

    if requested_backend in {"auto", "openvino"}:
        try:
            return load_florence_openvino_model(
                model_candidates,
                device,
                auto_processor_cls,
            )
        except Exception as exc:
            errors.append(str(exc))

    try:
        return load_florence_torch_model(
            model_candidates,
            device,
            auto_config_cls,
            auto_model_cls,
            auto_processor_cls,
            torch_module,
        )
    except Exception as exc:
        errors.append(str(exc))

    raise RuntimeError("No se pudo cargar Florence-2.\n" + "\n\n".join(errors))


# Extrae regiones con cajas desde la salida de Florence.
def extract_florence_regions(parsed_output: Any) -> List[Dict[str, Any]]:
    regions: List[Dict[str, Any]] = []

    # Recorre estructuras anidadas para extraer elementos relevantes.
    def visit(node: Any) -> None:
        if isinstance(node, dict):
            labels = node.get("labels")
            boxes = (
                node.get("quad_boxes")
                or node.get("bboxes")
                or node.get("boxes")
                or node.get("polygons")
            )
            if isinstance(labels, list) and isinstance(boxes, list):
                for index, box in enumerate(boxes):
                    label = labels[index] if index < len(labels) else ""
                    normalized_label = normalize_florence_label_text(label)
                    rect = florence_box_to_rect(box)
                    if normalized_label and rect is not None:
                        regions.append(
                            {
                                "text": normalized_label,
                                "box": [float(value) for value in rect],
                            }
                        )
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(parsed_output)
    return regions


# Prepara el recorte del editor para OCR con Florence.
def prepare_editor_image_for_florence(image: np.ndarray) -> Any:
    if Image is None:
        raise RuntimeError("Pillow no está disponible para preparar imágenes de Florence-2.")
    if image is None or image.size == 0:
        raise RuntimeError("La imagen del editor está vacía.")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    upscaled = cv2.resize(rgb, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    cleaned = cv2.bilateralFilter(sharpened, d=5, sigmaColor=25, sigmaSpace=30)
    return Image.fromarray(cleaned)


# Normaliza una etiqueta como posible pestaña activa.
def coerce_florence_tab_candidate(text: str) -> Optional[str]:
    normalized = normalize_florence_label_text(text)
    if not normalized:
        return None
    normalized = re.sub(r"\s+[xX]$", "", normalized).strip()
    compact = normalized.replace("\\", "/").replace(" ", "")
    compact = re.sub(r"[|]+", "/", compact)
    compact = re.sub(r"/([A-Za-z]{1,5})$", r".\1", compact)
    for extension in sorted(COMMON_CODE_EXTENSIONS, key=len, reverse=True):
        if compact.lower().endswith("." + extension):
            break
        if compact.lower().endswith("/" + extension):
            compact = compact[: -(len(extension) + 1)] + "." + extension
            break
        if "." not in path_basename(compact) and compact.lower().endswith(extension) and len(compact) > len(extension):
            compact = compact[: -len(extension)] + "." + extension
            break
    sanitized = sanitize_relative_code_path(compact)
    if not looks_like_code_filename(sanitized):
        return None
    return sanitized


# Repara secuencias de numeros de linea de Florence.
def repair_florence_number_sequence(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    repaired: List[Dict[str, Any]] = []
    previous_number: Optional[int] = None
    for entry in lines:
        updated = dict(entry)
        raw_number = safe_int(updated.get("number"))
        raw_token = str(updated.get("raw_number_token", "") or "")
        candidates: List[int] = []
        if raw_number is not None:
            candidates.append(int(raw_number))
        normalized_token = normalize_florence_numeric_token(raw_token)
        if normalized_token:
            direct_value = safe_int(normalized_token)
            if direct_value is not None and direct_value not in candidates:
                candidates.append(direct_value)
            swapped = normalized_token.replace("8", "0")
            swapped_value = safe_int(swapped)
            if swapped_value is not None and swapped_value not in candidates:
                candidates.append(swapped_value)

        chosen_number = raw_number
        if previous_number is not None and candidates:
            scored_candidates = []
            for candidate in candidates:
                penalty = abs(candidate - (previous_number + 1))
                if candidate <= previous_number:
                    penalty += 8
                scored_candidates.append((penalty, abs(candidate - previous_number), candidate))
            scored_candidates.sort()
            chosen_number = scored_candidates[0][2]
        elif candidates:
            chosen_number = min(candidates)

        updated["number"] = chosen_number
        if chosen_number is not None and str(updated.get("number_source", "")) == "explicit":
            updated["number_source"] = "explicit"
        repaired.append(updated)
        if chosen_number is not None:
            previous_number = int(chosen_number)

    return repaired
