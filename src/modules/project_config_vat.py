"""
Configuration constants for the classroom video processor.
"""

from __future__ import annotations

# ============================================================================
# CONFIGURATION
# ============================================================================

# Consulta textual que GroundingDINO usa para localizar pizarrones.
GDINO_TEXT_QUERY = "whiteboard . pizarron . blackboard"
# Umbral de confianza de caja para detecciones de GroundingDINO.
GDINO_BOX_TH = 0.38
# Umbral de confianza textual para detecciones de GroundingDINO.
GDINO_TEXT_TH = 0.40

# Confianza minima para aceptar detecciones de persona/proyeccion de YOLO.
YOLO_CONF_TH = 0.40
# Archivo o nombre del modelo YOLO usado por el backend de vision.
YOLO_MODEL = "yolov8n-seg.pt"
# Ruta de pesos de GroundingDINO usada para detectar el pizarron.
GDINO_WEIGHTS = "weights/groundingdino_swint_ogc.pth"

# Color BGR usado para dibujar personas en overlays de debug.
COLOR_PERSON = (0, 0, 255)
# Color BGR usado para dibujar la proyeccion en overlays de debug.
COLOR_PROJECTION = (255, 0, 0)
# Color BGR usado para dibujar el pizarron en overlays de debug.
COLOR_WHITEBOARD = (0, 255, 80)

# Intervalo minimo entre capturas persistidas de ROIs.
ROI_CAPTURE_INTERVAL_SECONDS = 1.0
# Padding aplicado a la mascara de persona dentro de ROIs guardados.
ROI_PERSON_MASK_PADDING_PX = 6

# Frecuencia en frames para ejecutar YOLO dentro del loop principal.
YOLO_INTERVAL = 8
# Frecuencia normal en frames para revisar si hay cortina.
CORTINA_CHECK_INTERVAL = 6
# Frecuencia reducida cuando la cortina ya esta activa/establecida.
CORTINA_CHECK_INTERVAL_ACTIVE = 24
# Frames que se conserva una deteccion de cortina antes de darla por perdida.
CORTINA_LOST_TTL_FRAMES = 18
# Umbral bajo de Canny para detectar bordes de la cortina.
CORTINA_CANNY_LOW = 60
# Umbral alto de Canny para detectar bordes de la cortina.
CORTINA_CANNY_HIGH = 140
# Padding alrededor de persona al ignorar bordes para buscar cortina.
CORTINA_PERSON_PADDING = 20
# Tiempo requerido para confirmar una linea candidata de cortina.
CORTINA_LINE_CONFIRM_SECONDS = 5.0
# Tiempo requerido sin linea para limpiar el estado de cortina.
CORTINA_LINE_CLEAR_SECONDS = 3.0
# Longitud relativa minima de una linea candidata de cortina.
CORTINA_LINE_MIN_LEN_RATIO = 0.28
# Traslape horizontal minimo para emparejar segmentos de linea de cortina.
CORTINA_LINE_MIN_OVERLAP_RATIO = 0.55
# Altura relativa maxima donde se espera la linea superior de cortina.
CORTINA_LINE_TOP_MAX_RATIO = 0.50
# Desplazamiento horizontal maximo permitido para tratar una linea como vertical.
CORTINA_LINE_VERTICAL_DX_RATIO = 0.12
# Ancho relativo minimo al unir pares de lineas de cortina.
CORTINA_LINE_PAIR_MIN_WIDTH_RATIO = 0.20
# Ancho relativo maximo al unir pares de lineas de cortina.
CORTINA_LINE_PAIR_MAX_WIDTH_RATIO = 0.88
# Distancia maxima en pixeles para fusionar lineas cercanas en X.
CORTINA_LINE_X_MERGE_PX = 18
# Margen relativo agregado al ROI usado para analizar la cortina.
CORTINA_ROI_PAD_RATIO = 0.04
# Area relativa minima que debe cubrir la mascara Otsu de cortina.
CORTINA_OTSU_MIN_AREA_RATIO = 0.18
# Altura relativa maxima donde puede iniciar la region Otsu de cortina.
CORTINA_OTSU_TOP_MAX_RATIO = 0.28
# Margen relativo usado al refinar el bbox del pizarron.
WHITEBOARD_REFINE_MARGIN_RATIO = 0.35
# Longitud relativa minima de bordes para aceptar refinamiento del pizarron.
WHITEBOARD_REFINE_MIN_LEN_RATIO = 0.30
# Tolerancia angular para clasificar bordes del pizarron como vertical/horizontal.
WHITEBOARD_REFINE_ANGLE_TOL_DEG = 25.0
# Cobertura vertical minima esperada de bordes del pizarron.
WHITEBOARD_REFINE_MIN_VSPAN_RATIO = 0.55
# Cobertura horizontal minima esperada de bordes del pizarron.
WHITEBOARD_REFINE_MIN_HSPAN_RATIO = 0.55
# Cobertura minima del borde superior al refinar el pizarron.
WHITEBOARD_REFINE_TOP_COVER_RATIO = 0.35
# Cobertura minima del borde inferior al refinar el pizarron.
WHITEBOARD_REFINE_BOT_COVER_RATIO = 0.65
# Inset final en pixeles aplicado al bbox refinado del pizarron.
WHITEBOARD_REFINE_INSET_PX = 2
# Nombre de la ventana OpenCV usada para debug de proyeccion/PDF.
PROJECTION_DEBUG_WINDOW_NAME = "PDF Debug"
# Maximo de miniaturas de diapositivas mostradas en el overlay.
SLIDES_OVERLAY_MAX = 12
# Ancho objetivo de visualizacion y procesamiento reducido.
DISPLAY_TARGET_W = 1202
# Alto objetivo de visualizacion y procesamiento reducido.
DISPLAY_TARGET_H = 684

# Activa la generacion incremental de la bitacora accesible.
BITACORA_LIVE_ENABLED = True
# Intervalo minimo entre regeneraciones de la bitacora en vivo.
BITACORA_UPDATE_INTERVAL_SECONDS = 5.0
# Nombre del notebook actualizado mientras corre la sesion.
BITACORA_NOTEBOOK_NAME = "bitacora_accesible_live.ipynb"
