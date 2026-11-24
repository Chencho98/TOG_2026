"""
Configuration, constants and prompts for code reconstruction.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Variable de entorno que permite fijar el binario local de Tesseract.
OCR_TESSERACT_ENV_VAR = "TESSERACT_CMD"
# Rutas comunes donde se busca Tesseract cuando la variable no esta definida.
COMMON_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
)

# Region relativa de la barra inferior usada para detectar estado CODIGO.
CODE_STATUS_BAR_ROI_REL = (0.00, 0.95, 1.00, 1.00)
# Region relativa dentro de la barra donde suele aparecer el texto del lenguaje.
CODE_STATUS_TEXT_ROI_REL = (0.89, 0.00, 0.94, 1.00)

# Limite HSV inferior para detectar el azul tipico de la barra de VS Code.
CODE_BLUE_HSV_LOWER = np.array([95, 70, 45], dtype=np.uint8)
# Limite HSV superior para detectar el azul tipico de la barra de VS Code.
CODE_BLUE_HSV_UPPER = np.array([140, 255, 255], dtype=np.uint8)
# Proporcion minima de azul para considerar posible estado de codigo.
CODE_MIN_BLUE_RATIO = 0.18
# Proporcion de azul que da una senal fuerte de editor visible.
CODE_STRONG_BLUE_RATIO = 0.38
# Proporcion de azul que permite mantener estado CODIGO entre muestras.
CODE_KEEP_BLUE_RATIO = 0.10
# Ancho objetivo al reducir la proyeccion para clasificacion de codigo.
CODE_CLASSIFICATION_TARGET_W = 1202
# Alto objetivo al reducir la proyeccion para clasificacion de codigo.
CODE_CLASSIFICATION_TARGET_H = 684

# Alto objetivo del recorte de texto de estado antes de enviarlo al LLM.
CODE_TEXT_TARGET_HEIGHT = 128
# Borde aplicado a recortes de texto para mejorar lectura visual.
CODE_TEXT_BORDER = 10
# Tiempo que se mantiene una decision CODIGO antes de soltarla.
CODE_HOLD_SECONDS = 0.90
# Separacion minima entre muestras para aceptar cambios de estado sostenidos.
CODE_HOLD_MIN_SAMPLE_GAPS = 1.5
# Variable de entorno que habilita requests LLM para el texto de status.
CODE_STATUS_LLM_ENABLE_API_ENV_VAR = "CODE_STATUS_LLM_ENABLE_API"
# Intervalo minimo entre requests LLM del status text.
CODE_STATUS_LLM_INTERVAL_SECONDS = 0.35
# Alto objetivo del ROI de status enviado al LLM.
CODE_STATUS_LLM_TARGET_HEIGHT = 192
# Calidad JPEG usada para el ROI de status enviado al LLM.
CODE_STATUS_LLM_JPEG_QUALITY = 95

# Variable de entorno que permite seleccionar proveedor principal de vision.
CODE_VISION_PROVIDER_ENV_VAR = "CODE_VISION_PROVIDER"
# Variable de entorno que habilita/deshabilita la API remota de codigo.
CODE_VISION_ENABLE_API_ENV_VAR = "CODE_VISION_ENABLE_API"
# Variable de entorno de API key para OpenAI.
CODE_VISION_API_KEY_ENV_VAR = "OPENAI_API_KEY"
# Variable de entorno de API key especifica para Gemini.
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
# Variable de entorno alternativa de API key de Google/Gemini.
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"
# Variable de entorno que sobreescribe el modelo remoto de codigo.
CODE_VISION_MODEL_ENV_VAR = "CODE_VISION_MODEL"
# Variable de entorno alternativa para modelo OpenAI.
OPENAI_MODEL_ENV_VAR = "OPENAI_MODEL"
# Variable de entorno que sobreescribe la URL base del proveedor remoto.
CODE_VISION_BASE_URL_ENV_VAR = "OPENAI_BASE_URL"
# Variable de entorno que controla el timeout de requests remotos.
CODE_VISION_TIMEOUT_ENV_VAR = "CODE_VISION_TIMEOUT_SEC"

# Perfil de proveedor OpenAI usado para analizar editor y sidebar.
CODE_VISION_PROFILE_OPENAI = {
    "label": "OpenAI",
    "provider_name": "openai",
    "model": "gpt-4.1-mini",        #mejor
    # "model": "gpt-4o",            #mas rapido
    # "model": "gpt-5.4-mini",      #mejor respuesta
    # "model": "gpt-5.4-nano",      #medio
    # "model": "gpt-5-mini",        #lento y malo
    # "model": "gpt-5-nano",        #peor
    "base_url": "https://api.openai.com/v1",
    "api_key_envs": (CODE_VISION_API_KEY_ENV_VAR,),
    "prompt_order": "image_first",
}
# Perfil de proveedor Gemini compatible con endpoint tipo OpenAI.
CODE_VISION_PROFILE_GEMINI = {
    "label": "Gemini",
    "provider_name": "gemini",
    "model": "gemini-2.5-flash-lite",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    "api_key_envs": (
        GEMINI_API_KEY_ENV_VAR,
        GOOGLE_API_KEY_ENV_VAR,
        CODE_VISION_API_KEY_ENV_VAR,
    ),
    "prompt_order": "text_first",
}
# Perfil dedicado a consultas baratas del status text usando Gemini.
CODE_STATUS_LLM_PROFILE_GEMINI = {
    **CODE_VISION_PROFILE_GEMINI,
    "label": "Gemini status",
    "api_key_envs": (GEMINI_API_KEY_ENV_VAR, GOOGLE_API_KEY_ENV_VAR),
}

# Perfil remoto principal para solicitudes de vision del editor/sidebar.
CODE_VISION_ACTIVE_PROFILE = CODE_VISION_PROFILE_GEMINI
# CODE_VISION_ACTIVE_PROFILE = CODE_VISION_PROFILE_OPENAI

# Proveedor secundario para apoyar la extracción del editor.
# CODE_VISION_SECONDARY_PROFILE = CODE_VISION_PROFILE_GEMINI
# CODE_VISION_SECONDARY_PROFILE = CODE_VISION_PROFILE_OPENAI
# Perfil remoto secundario opcional; None evita doble envio.
CODE_VISION_SECONDARY_PROFILE = None

# Modelo efectivo por defecto derivado del perfil activo.
CODE_VISION_DEFAULT_MODEL = str(CODE_VISION_ACTIVE_PROFILE["model"])
# URL base efectiva por defecto derivada del perfil activo.
CODE_VISION_DEFAULT_BASE_URL = str(CODE_VISION_ACTIVE_PROFILE["base_url"])
# Timeout por defecto para requests remotos del pipeline de codigo.
CODE_VISION_DEFAULT_TIMEOUT_SEC = 45.0
# Limite maximo de tokens de salida para respuestas de codigo/sidebar.
CODE_VISION_MAX_TOKENS = 3000

# Maximo de requests de editor simultaneos.
CODE_EDITOR_MAX_IN_FLIGHT = 15

# Maximo de requests de sidebar simultaneos.
CODE_SIDEBAR_MAX_IN_FLIGHT = 2
# Tamano de cola para snapshots de editor pendientes.
CODE_EDITOR_QUEUE_SIZE = 4
# Tamano de cola para snapshots de sidebar pendientes.
CODE_SIDEBAR_QUEUE_SIZE = 2
# Diferencia visual minima para considerar que cambio el sidebar.
CODE_SIDEBAR_CHANGE_THRESHOLD = 0.12
# Cambio minimo de densidad de bordes para confirmar cambio del sidebar.
CODE_SIDEBAR_EDGE_RATIO_DELTA = 0.05
# Diferencia visual minima para considerar que cambio el editor.
CODE_EDITOR_CHANGE_THRESHOLD = 0.08
# Cambio minimo de densidad de bordes para confirmar cambio del editor.
CODE_EDITOR_EDGE_RATIO_DELTA = 0.03
# Ancho usado para firmas visuales compactas de editor/sidebar.
CODE_SIGNATURE_TARGET_W = 192
# Alto usado para firmas visuales compactas de editor/sidebar.
CODE_SIGNATURE_TARGET_H = 192
# Ventana de busqueda al alinear lineas reconstruidas.
CODE_LINE_ALIGNMENT_WINDOW = 24
# Minimo de anclas para confiar en una alineacion de lineas.
CODE_LINE_ALIGNMENT_MIN_ANCHORS = 2
# Ventana local para deduplicar lineas repetidas.
CODE_LINE_DEDUP_WINDOW = 2
# Ventana usada al anexar lineas cercanas al bloque conocido.
CODE_LINE_ATTACH_WINDOW = 2
# Longitud minima para aplicar deduplicacion global de texto.
CODE_GLOBAL_DEDUP_MIN_TEXT_LENGTH = 12
# Minimo de lineas faltantes para marcarlas en el render final.
CODE_RENDER_MISSING_MIN_LINE_COUNT = 1
# Habilita comparacion final contra archivos de referencia conocidos.
CODE_REFERENCE_COMPARE_ENABLE = True
# Mapa de archivos reconstruidos a sus referencias locales esperadas.
CODE_REFERENCE_COMPARE_FILES = {
    "detector_local.py": "detector_local_ref.py",
    "classes.py": "classes_ref.py",
}
# Intervalo maximo entre escrituras forzadas del store reconstruido.
CODE_STORE_FLUSH_INTERVAL_SECONDS = 8.0
# Numero de cambios acumulados antes de forzar flush del store.
CODE_STORE_FLUSH_CHANGES = 24
# Variable de entorno que controla logs/debug del pipeline de codigo.
CODE_PIPELINE_DEBUG_ENV_VAR = "CODE_PIPELINE_DEBUG"
# Valor por defecto del debug del pipeline cuando no hay variable de entorno.
CODE_PIPELINE_DEBUG_DEFAULT = True
# Tamano de backlog a partir del cual se emite advertencia de respuestas.
CODE_RESPONSE_QUEUE_WARN_SIZE = 12

# Intervalo minimo entre envios de frames de editor al modelo remoto.
CODE_EDITOR_SEND_INTERVAL_SECONDS = 0.6

# Tamano de ventana para seleccionar el mejor frame reciente del editor.
CODE_FRAME_SELECTION_WINDOW_SIZE = 4
# Formato de imagen usado al enviar/guardar recortes de editor.
CODE_EDITOR_IMAGE_FORMAT = "png"
# Calidad JPEG si el recorte de editor se guarda/envia como JPEG.
CODE_EDITOR_JPEG_QUALITY = 98
# Formato de imagen usado al enviar/guardar recortes de sidebar.
CODE_SIDEBAR_IMAGE_FORMAT = "jpg"
# Calidad JPEG usada para recortes de sidebar.
CODE_SIDEBAR_JPEG_QUALITY = 92
# Fraccion izquierda recortada al puntuar nitidez del editor.
CODE_EDITOR_SCORE_TRIM_LEFT_RATIO = 0.02
# Fraccion derecha recortada al puntuar nitidez del editor.
CODE_EDITOR_SCORE_TRIM_RIGHT_RATIO = 0.02
# Fraccion superior recortada al puntuar nitidez del editor.
CODE_EDITOR_SCORE_TRIM_TOP_RATIO = 0.03
# Fraccion inferior recortada al puntuar nitidez del editor.
CODE_EDITOR_SCORE_TRIM_BOTTOM_RATIO = 0.06
# Minimo de filas visibles para aceptar una respuesta de editor.
CODE_EDITOR_MIN_VISIBLE_ROWS = 4
# Salto monotono maximo permitido entre numeros de linea visibles.
CODE_EDITOR_MAX_MONOTONIC_GAP = 6
# Padding maximo agregado al rango visible de lineas.
CODE_EDITOR_MAX_VISIBLE_LINE_RANGE_PADDING = 8
# Multiplicador maximo del rango visible aceptado.
CODE_EDITOR_MAX_VISIBLE_LINE_RANGE_MULTIPLIER = 3
# Frames requeridos para confirmar una respuesta sospechosa.
CODE_EDITOR_SUSPICIOUS_CONFIRMATION_FRAMES = 2
# Mejora minima de calidad para reemplazar una version ya conocida.
CODE_EDITOR_QUALITY_IMPROVEMENT_DELTA = 0.75
# Intervalo para enviar editor aun sin cambios cuando el contenido esta estable.
CODE_EDITOR_PERIODIC_IF_SETTLED_SECONDS = 2.5
# Nitidez minima para tratar un frame como usable durante scroll.
CODE_EDITOR_SCROLL_BLUR_SHARPNESS_MIN = 1.80
# Densidad maxima de bordes para permitir frame de scroll.
CODE_EDITOR_SCROLL_BLUR_EDGE_MAX = 0.060
# Nitidez minima para aceptar frames parcialmente borrosos.
CODE_EDITOR_PARTIAL_BLUR_SHARPNESS_MIN = 2.60
# Densidad maxima de bordes para aceptar frames parcialmente borrosos.
CODE_EDITOR_PARTIAL_BLUR_EDGE_MAX = 0.095
# Balance maximo entre ejes para clasificar blur de scroll.
CODE_EDITOR_SCROLL_BLUR_AXIS_BALANCE_MAX = 0.42
# Balance maximo entre ejes para blur parcial aceptable.
CODE_EDITOR_PARTIAL_BLUR_AXIS_BALANCE_MAX = 0.30
# Variacion maxima por filas para frames de scroll.
CODE_EDITOR_SCROLL_BLUR_ROW_VARIATION_MAX = 0.020
# Variacion maxima por filas para frames parcialmente borrosos.
CODE_EDITOR_PARTIAL_BLUR_ROW_VARIATION_MAX = 0.030
# Maximo de lineas iniciales que pueden quedarse como prefijo sticky.
CODE_STICKY_MAX_PREFIX_LINES = 4
# Minimo de lineas numeradas para considerar que empezo el cuerpo real.
CODE_STICKY_BODY_MIN_NUMBERED_LINES = 2
# Minimo de pares densos para confirmar cuerpo de codigo.
CODE_STICKY_BODY_MIN_DENSE_PAIRS = 1
# Separacion que marca donde empieza el cuerpo despues de prefijos.
CODE_STICKY_BODY_START_GAP = 2
# Separacion que identifica prefijos dispersos a preservar aparte.
CODE_STICKY_PREFIX_SPARSE_GAP = 3
# Frames requeridos para confirmar lineas pendientes.
CODE_PENDING_LINE_CONFIRMATION_FRAMES = 1
# Edad maxima en frames de lineas pendientes antes de descartarlas.
CODE_PENDING_LINE_MAX_AGE_FRAMES = 180
# Ventana en frames para detectar bloques duplicados.
CODE_BLOCK_DUPLICATE_WINDOW_FRAMES = 180
# Maximo de bloques recientes conservados para deduplicacion.
CODE_BLOCK_HISTORY_LIMIT = 200
# Maximo de entradas del arbol de sidebar que se conservan.
CODE_SIDEBAR_TREE_LIMIT = 500
# Habilita uso de Florence como apoyo local de OCR/region.
CODE_FLORENCE_ENABLE = True
# Backend preferido para Florence: auto, cpu, gpu u otro soportado.
CODE_FLORENCE_BACKEND = "auto"
# Dispositivo preferido para Florence.
CODE_FLORENCE_DEVICE = "gpu"
# Modelo Florence explicito; vacio usa la lista por defecto.
CODE_FLORENCE_MODEL = ""
# Tarea Florence usada para OCR con regiones.
CODE_FLORENCE_TASK = "ocr_with_region"
# Limite de tokens generados por Florence.
CODE_FLORENCE_MAX_NEW_TOKENS = 1024
# Fraccion izquierda tratada como gutter/numero de linea en Florence.
CODE_FLORENCE_GUTTER_RATIO = 0.18
# Maximo de filas de tabs detectadas por Florence.
CODE_FLORENCE_MAX_TAB_ROWS = 4
# Maximo de etiquetas/regiones que se aceptan desde Florence.
CODE_FLORENCE_MAX_LABELS = 96
# Evita lanzar otra inferencia Florence si ya hay una en curso.
CODE_FLORENCE_BUSY_SKIP = True
# Intervalo minimo entre inferencias Florence.
CODE_FLORENCE_INTERVAL_SECONDS = 4.0
# Minimo de lineas remotas para comparar contra Florence.
CODE_FLORENCE_MIN_REMOTE_LINE_COUNT = 6
# Confianza remota minima para usarla frente al apoyo de Florence.
CODE_FLORENCE_REMOTE_CONFIDENCE_MIN = 0.72
# Modelos Florence probados cuando no se fija uno explicito.
DEFAULT_FLORENCE_MODELS = [
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large-ft",
]
# Extensiones reconocidas como archivos de codigo en sidebar/rutas.
COMMON_CODE_EXTENSIONS = (
    "py",
    "js",
    "ts",
    "tsx",
    "jsx",
    "java",
    "c",
    "cpp",
    "h",
    "hpp",
    "go",
    "rs",
    "json",
    "yml",
    "yaml",
    "xml",
    "html",
    "css",
    "sql",
    "php",
    "rb",
    "kt",
    "swift",
    "cs",
    "sh",
)
# Marcador textual usado para representar lineas vacias del editor.
BLANK_EDITOR_LINE_SENTINEL = "<blank>"
# Token especial que inyecta la pista de archivo activo en prompts.
EDITOR_ACTIVE_FILE_HINT_TOKEN = "__ACTIVE_FILE_HINT__"
# Token especial que inyecta la pista de lenguaje en prompts.
EDITOR_LANGUAGE_HINT_TOKEN = "__LANGUAGE_HINT__"
# Regex para separar numeros de linea pegados al texto del codigo.
COMPOUND_EDITOR_LINE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<number>\d{1,5})\s+(?=[A-Za-z_#'\"({\[])"
)
# Regex para reconocer lineas de status bar tipo Ln/Col.
EDITOR_STATUS_LINE_RE = re.compile(
    r"\bln\s*\d+\b.*\bcol\s*\d+\b",
    re.IGNORECASE,
)
# Regex para reconocer tokens comunes de status bar que no son codigo.
EDITOR_STATUS_TOKEN_RE = re.compile(
    r"\b(?:utf-?8|utf|crlf|crf|lf|spaces?|selected|informational|tabsize)\b",
    re.IGNORECASE,
)

# Prompt usado para pedir al LLM la estructura visible del sidebar.
SIDEBAR_ANALYSIS_PROMPT = """
Analiza esta imagen del panel lateral de VS Code (explorer).
Devuelve SOLO JSON válido con esta forma:
{
  "active_file": "ruta/relativa.py" | null,
  "entries": [
    {
      "label": "nombre visible",
      "path": "ruta/relativa.py" | null,
      "depth": 0,
      "type": "file" | "folder",
      "selected": false
    }
  ],
  "notes": "texto corto opcional"
}
Reglas:
- No inventes rutas invisibles.
- Si una entrada no tiene ruta completa visible, usa null en "path".
- Marca "selected" si el archivo parece el activo.
- Mantén el orden visual de arriba hacia abajo.
""".strip()

# Prompt usado para pedir al LLM lineas, archivo activo y metadatos del editor.
EDITOR_ANALYSIS_PROMPT_TEMPLATE = """
Analiza esta imagen del editor de VS Code.
Contexto:
- archivo activo sugerido: __ACTIVE_FILE_HINT__
- lenguaje sugerido: __LANGUAGE_HINT__

Devuelve SOLO JSON válido con esta forma:
{
  "active_file": "ruta/relativa.py" | null,
  "language": "PYTHON" | null,
  "start_line": 40 | null,
  "end_line": 55 | null,
  "sticky_context": [
    { "number": 52, "text": "class Foo:" }
  ],
  "selection": "text" | null,
  "lines": [
    { "number": 40, "text": "def foo():" },
    { "number": 41, "text": "    return 1" }
  ],
  "tab_title": "foo.py" | null,
  "confidence": 0.0
}
Reglas:
- Usa los números de línea visibles exactamente si existen.
- Puede haber varias pestañas activas, guiate por ello y antes del codigo puede que haya una ruta que identifique el archivo con formato ruta > ruta > archivo.py
- Si aparece Sticky Scroll de VS Code o contexto pegado arriba, colócalo en "sticky_context" y NO lo repitas en "lines".
- "lines" debe contener sólo el bloque principal visible del editor.
- Cada número de línea visible inicia una línea lógica nueva; nunca mezcles texto perteneciente a números distintos en una sola entrada.
- Conserva indentación inicial en cada línea.
- Si una línea numerada está vacía pero visible, devuélvela con su número y "text": "".
- Conserva exactamente las líneas de comentario y separadores visibles, por ejemplo "# =====", "# -----", "# CONFIG", incluyendo "#" y símbolos.
- Si hay huecos, no inventes líneas faltantes.
- Si el primer renglón visible es 100, usa 100.
- Si no ves número de línea para una línea concreta, puedes dejarla fuera.
- Si el archivo cambió, usa "active_file" o "tab_title" si es visible.
- Si hay texto subrayado o seleccionado indica cual es.
- Nunca devuelvas texto del status bar de VS Code como UTF-8, CRLF, Ln, Col, Spaces, selected o Informational.
- Nunca devuelvas números de línea negativos ni rangos que no correspondan al gutter visible.
- Si una línea está dudosa o barrida, omítela por completo en lugar de reconstruirla.
- Si la imagen está parcialmente barrida por scroll, devuelve sólo las líneas totalmente legibles.
- Si la imagen está severamente barrida por scroll, devuelve "lines": [] y no adivines contenido.
""".strip()
