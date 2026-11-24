# Descriptor automatico de sesiones academicas

Entrega final del procesador de video para generar una bitacora accesible de sesiones academicas. Detecta personas y pizarron, identifica proyecciones y diapositivas, reconstruye codigo visible y analiza cambios del pizarron.

## Requisitos

- Windows y PowerShell.
- Python 3.9 o posterior. El proyecto no fija una version exacta.
- Tesseract OCR instalado cuando se use OCR o clasificacion de PDF.
- Conexion a Internet para LLM remotos y para descargar modelos que no esten en disco.
- Clave de Gemini o Google para las funciones LLM remotas predeterminadas.

El proyecto incluye `weights/groundingdino_swint_ogc.pth` y el modelo OpenVINO `yolov8n-seg_openvino_model/`.

## Instalacion

Desde la raiz del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

El codigo tambien importa dependencias no enumeradas actualmente en `requirements.txt`. Para el flujo completo:

```powershell
py -m pip install PyMuPDF pytesseract transformers openvino
py -m pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

Instala Tesseract OCR para Windows. Si no se detecta automaticamente, define `TESSERACT_CMD` en `.env`.

### Instalacion de Tesseract OCR en Windows

https://www.softpedia.com/get/Programming/Other-Programming-Files/Tesseract-OCR.shtml

1. Instala Tesseract OCR y conserva la ruta de instalacion. En Windows puede quedar instalado para todos los usuarios o solo para el usuario actual. Las rutas habituales son:

   ```text
   C:\Program Files\Tesseract-OCR\tesseract.exe
   C:\Users\<usuario>\AppData\Local\Programs\Tesseract-OCR\tesseract.exe
   ```

2. Abre una terminal nueva y verifica la instalacion:

   ```powershell
   tesseract --version
   ```

3. Si el comando anterior funciona, no se requiere configuracion adicional. El proyecto busca automaticamente estas rutas:

   ```text
   C:\Program Files\Tesseract-OCR\tesseract.exe
   C:\Program Files (x86)\Tesseract-OCR\tesseract.exe
   C:\Users\<usuario>\AppData\Local\Programs\Tesseract-OCR\tesseract.exe
   ```

4. Si Tesseract se instalo en otra ubicacion o no esta en el `PATH`, agrega su ruta completa al archivo `.env`:

   ```env
   TESSERACT_CMD=C:\ruta\completa\a\tesseract.exe
   ```

5. Cierra y vuelve a abrir la terminal antes de ejecutar el proyecto si modificaste el `PATH`. Si solo modificaste `.env`, vuelve a iniciar el programa para que cargue la nueva variable.

### Aceleracion

- NVIDIA: instala PyTorch compatible con CUDA y un controlador NVIDIA compatible.
- Intel XPU: instala PyTorch e Intel Extension for PyTorch compatibles con XPU.
- CPU: no requiere acelerador.

El backend intenta CUDA/NVIDIA, luego Intel XPU y finalmente CPU. YOLO utiliza `yolov8n-seg.pt` en CUDA; si no existe, Ultralytics puede descargarlo en la primera ejecucion. En XPU y CPU utiliza primero el modelo OpenVINO incluido.

## Archivo .env

Crea `.env` en la raiz. No lo incluyas en entregas ni repositorios: ya esta en `.gitignore`.

```env
GEMINI_API_KEY=tu_clave_de_gemini

CODE_VISION_PROVIDER=gemini
CODE_VISION_MODEL=gemini-2.5-flash-lite
WHITEBOARD_LLM_PROVIDER=gemini
WHITEBOARD_LLM_MODEL=gemini-2.5-flash-lite

# Opcional si Tesseract no se detecta.
# TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
```

Tambien pueden definirse claves solo para la terminal actual:

```powershell
$env:GEMINI_API_KEY = "tu_clave_de_gemini"
```

Las variables `$env:` tienen prioridad sobre `.env`.

## Modelos predeterminados

| Funcion | Valor predeterminado |
|---|---|
| Vision de codigo y sidebar | Gemini, `gemini-2.5-flash-lite` |
| Consulta LLM de barra de estado | Gemini, `gemini-2.5-flash-lite` |
| Analisis de pizarron | Gemini, `gemini-2.5-flash-lite` |
| Apoyo local para codigo | Florence-2, habilitado |
| Timeout de solicitudes remotas | 45 segundos |

Para volver a usar OpenAI en codigo:

```env
OPENAI_API_KEY=tu_clave_de_openai
CODE_VISION_PROVIDER=openai
CODE_VISION_MODEL=gpt-4.1-mini
```

Para volver a usar OpenAI en pizarron:

```env
OPENAI_API_KEY=tu_clave_de_openai
WHITEBOARD_LLM_PROVIDER=openai
WHITEBOARD_LLM_MODEL=gpt-4o
```

| Variable | Efecto |
|---|---|
| `CODE_VISION_ENABLE_API=0` | Desactiva API remota de codigo y usa OCR local como cliente base. |
| `CODE_STATUS_LLM_ENABLE_API=0` | Desactiva la consulta Gemini de la barra de estado. |
| `WHITEBOARD_LLM_ENABLE_API=0` | Desactiva el analisis LLM del pizarron. |
| `CODE_FLORENCE_ENABLE=0` | No tiene efecto en esta copia: Florence-2 esta habilitado de forma fija. |
| `OPENAI_BASE_URL` | Cambia la URL base compatible con Chat Completions. |
| `CODE_VISION_TIMEOUT_SEC` | Cambia el timeout de codigo. |
| `WHITEBOARD_LLM_TIMEOUT_SEC` | Cambia el timeout del pizarron. |

## Ejecucion

Ejecuta siempre desde la raiz:

```powershell
py .\project_vat.py ruta\al\video.mp4
```

Con PDF de referencia:

```powershell
py .\project_vat.py data\videos\demo10.mp4 --pdf data\class1.pdf
```

Sin ventana de depuracion de proyeccion:

```powershell
py .\project_vat.py data\videos\demo10.mp4 --pdf data\class1.pdf --no-debug-projection
```

| Argumento | Obligatorio | Predeterminado | Descripcion |
|---|---:|---|---|
| `video` | Si | - | Ruta al video. |
| `--pdf RUTA` | No | Sin PDF | PDF de referencia para diapositivas. |
| `--no-debug-projection` | No | Depuracion activa | Desactiva la ventana de depuracion. |

## Resultados

Cada ejecucion crea `output/sessions/session_AAAAMMDD_HHMMSS/`. Puede incluir clasificaciones, coincidencias de PDF en `pdf/matchings.json`, analisis de pizarron en `whiteboard_analysis/`, codigo reconstruido en `code_intelligence/` y una bitacora accesible en formato `.ipynb`.

## Verificacion

Antes de procesar un video largo, confirma en consola:

```text
Dispositivo GDINO: cuda, xpu o cpu
YOLO backend: cuda, xpu o cpu
```

Si falta una clave API, el componente remoto registra el error o usa su alternativa local cuando esta disponible. La configuracion predeterminada requiere `GEMINI_API_KEY` o `GOOGLE_API_KEY` para los modulos remotos.
 
