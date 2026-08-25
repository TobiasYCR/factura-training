# Factura Training - Manual operativo

## 1. Objetivo del proyecto

Este proyecto convierte facturas, recibos e invoices en JSON normalizado.

El sistema completo no es solamente Qwen. El flujo real es:

```text
PDF o imagen
-> OCR local con Tesseract
-> parser/reglas para formatos conocidos
-> normalizacion
-> validacion
-> Qwen LoRA como fallback opcional
-> JSON final para la web/backend
```

La idea de produccion es usar el pipeline completo, porque es mas rapido, mas barato y mas controlable que depender solamente de un modelo generativo.

## 2. Componentes principales

- `api.py`: endpoint HTTP para subir PDFs/imagenes y recibir JSON.
- `ocr.py`: OCR local con Tesseract, renderizado de PDF y lectura de imagenes.
- `infer.py`: parsers, normalizadores, validadores y ejecucion del LoRA.
- `train.py`: entrenamiento LoRA con Unsloth/Qwen.
- `scripts/batch_extract_pdfs.py`: procesamiento masivo de carpetas con PDFs/imagenes.
- `scripts/build_real_dataset.py`: arma `real_train.jsonl` y `real_eval.jsonl`.
- `scripts/evaluate_real_dataset.py`: evalua el LoRA puro o el flujo production.
- `scripts/analyze_eval_results.py`: resume errores, campos flojos y casos prioritarios.
- `docs/production-readiness.md`: despliegue, API, seguridad y logs.
- `docs/arca-schema.md`: schema ARCA usado por el extractor.
- `docs/real-invoice-structure-notes.md`: formatos reales analizados.

## 3. Flujo operativo

1. Subir o copiar PDFs a la maquina donde se van a procesar.
2. Ejecutar batch para extraer OCR y JSON.
3. Revisar fallidos e incompletos.
4. Construir dataset real.
5. Entrenar el LoRA en la PC con GPU.
6. Evaluar `production` para medir el pipeline completo.
7. Evaluar `model` para medir Qwen LoRA puro.
8. Mejorar parsers o dataset segun los errores.
9. Levantar API.
10. Conectar la web a `POST /extract`.
11. Guardar logs y revisar casos dudosos.

## 4. Entrar a la PC con GPU

Los ejemplos usan placeholders para no dejar datos personales en el repo:

- `<GPU_USER>`: usuario SSH de la maquina con GPU.
- `<GPU_HOST>`: IP, hostname o dominio de la maquina con GPU.
- `<PROJECT_DIR>`: ruta donde esta clonado el proyecto.
- `<INPUT_DOCUMENTS_DIR>`: carpeta con PDFs/imagenes a procesar.
- `<API_KEY>`: clave privada de la API.
- `<REPOSITORY_URL>`: URL del repositorio.

Desde Windows:

```bash
ssh <GPU_USER>@<GPU_HOST>
```

Si al entrar quedas en Windows remoto, abrir WSL:

```bash
wsl
```

Entrar al proyecto:

```bash
cd <PROJECT_DIR>
```

Activar entorno:

```bash
source ~/miniconda3/bin/activate
conda activate factura-training
```

Traer los ultimos cambios:

```bash
git pull
```

## 5. Confirmar GPU y entorno

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Resultado esperado:

```text
True
NVIDIA GeForce RTX 4050 Laptop GPU
```

Si aparece `python: command not found`, el entorno conda no esta activo o la terminal esta fuera del entorno esperado.

## 6. Instalar OCR local

En Ubuntu/WSL:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng poppler-utils
```

Verificar:

```bash
tesseract --version
which tesseract
```

En Windows se puede instalar Tesseract y dejarlo en `PATH`, o definir:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 7. Procesar una carpeta de PDFs

Para procesar PDFs en subcarpetas:

```bash
python scripts/batch_extract_pdfs.py <INPUT_DOCUMENTS_DIR> \
  --pattern "**/*.pdf" \
  --output-dir data/real_invoices_analysis_pdf2_pdf \
  --write-ocr \
  --write-json
```

Para procesar imagenes:

```bash
python scripts/batch_extract_pdfs.py <INPUT_DOCUMENTS_DIR> \
  --pattern "**/*.jpg" \
  --output-dir data/real_invoices_analysis_pdf2_jpg \
  --write-ocr \
  --write-json
```

Para otros formatos, usar `jpeg` o `png` en el parametro `--pattern`.

Para reintentar solo los fallidos:

```bash
python scripts/batch_extract_pdfs.py <INPUT_DOCUMENTS_DIR> \
  --pattern "**/*.pdf" \
  --output-dir data/real_invoices_analysis_pdf2_pdf \
  --failed-from data/real_invoices_analysis_pdf2_pdf/batch_summary.jsonl \
  --write-ocr \
  --write-json
```

## 8. Construir dataset real

Cuando el batch ya genero `.txt` y `.json`, crear train/eval:

```bash
python scripts/build_real_dataset.py --input-dir data/real_invoices_analysis_pdf2_pdf
```

Salida esperada:

```text
data/real_train.jsonl
data/real_eval.jsonl
```

El split separa ejemplos de entrenamiento y evaluacion para medir con casos no vistos.

## 9. Entrenar el LoRA

Entrenamiento normal:

```bash
python train.py --max-steps 600
```

Entrenamiento reforzando externos:

```bash
python train.py \
  --data-files data/real_train.jsonl data/real_train_external.jsonl data/real_train_external.jsonl data/real_train_external.jsonl \
  --max-steps 800
```

El modelo queda guardado en:

```text
factura-qwen-lora/
```

Conviene guardar versiones buenas:

```bash
cp -r factura-qwen-lora factura-qwen-lora-external-weighted-800
```

## 10. Evaluar resultados

Evaluar el pipeline completo:

```bash
python scripts/evaluate_real_dataset.py \
  --eval-file data/real_eval.jsonl \
  --mode production \
  --out data/eval_results_production.jsonl
```

Evaluar Qwen LoRA puro:

```bash
python scripts/evaluate_real_dataset.py \
  --eval-file data/real_eval.jsonl \
  --model lora \
  --out data/eval_results_model.jsonl
```

Analizar errores:

```bash
python scripts/analyze_eval_results.py data/eval_results_production.jsonl --eval-file data/real_eval.jsonl
python scripts/analyze_eval_results.py data/eval_results_model.jsonl --eval-file data/real_eval.jsonl
```

Interpretacion:

- `JSON/schema OK`: salida valida y con estructura correcta.
- `Exactos`: todos los campos coinciden con el esperado.
- `Campos OK`: porcentaje campo por campo.
- `ARCA schema OK`: validez en facturas/notas ARCA.
- `Externos schema OK`: validez en recibos/invoices externos.

## 11. Probar una factura individual

Con OCR ya guardado:

```bash
python infer.py --model lora --ocr-file "data/real_invoices_analysis_pdf2_pdf/05 Mayo - Catalonia.txt"
```

Si la ruta tiene espacios, siempre va entre comillas.

## 12. Levantar API local

Sin API key:

```bash
python api.py --host 127.0.0.1 --port 8000
```

Con API key y logs:

```bash
FACTURA_API_KEY="<API_KEY>" \
FACTURA_LOG_FILE="$PWD/logs/extractions.jsonl" \
python api.py --host 0.0.0.0 --port 8000
```

Healthcheck:

```bash
curl http://127.0.0.1:8000/health
```

Extraer PDF:

```bash
curl -X POST "http://127.0.0.1:8000/extract?force_ocr=true" \
  -H "X-API-Key: <API_KEY>" \
  -F "file=@factura.pdf"
```

## 13. Usar Qwen como fallback desde la API

En una PC con GPU:

```bash
curl -X POST "http://127.0.0.1:8000/extract?force_ocr=true&use_model=true&model=lora&model_policy=low_confidence&min_confidence=0.82" \
  -H "X-API-Key: <API_KEY>" \
  -F "file=@factura.pdf"
```

Politicas:

- `fallback`: usa Qwen solo si el parser no resuelve.
- `low_confidence`: usa Qwen si el parser falla o devuelve baja confianza.
- `always`: prueba Qwen siempre y conserva la salida mas confiable.

## 14. Usar la API desde Windows

En `cmd.exe`:

```cmd
curl.exe -X POST "http://127.0.0.1:8000/extract?force_ocr=true" -F "file=@C:\RUTA\A\factura.pdf"
```

Con API key:

```cmd
curl.exe -X POST "http://127.0.0.1:8000/extract?force_ocr=true" -H "X-API-Key: <API_KEY>" -F "file=@C:\RUTA\A\factura.pdf"
```

Si el path tiene espacios, igual se deja todo el argumento `file=@...` entre comillas.

## 15. Desplegar en VPS

La VPS sin GPU corre OCR + parsers + validacion. Qwen LoRA se ejecuta cuando el servicio tiene entorno con GPU y dependencias de modelo.

Instalacion minima:

```bash
git clone <REPOSITORY_URL>
cd factura-training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng poppler-utils
```

Ejecutar:

```bash
FACTURA_API_KEY="<API_KEY>" \
FACTURA_LOG_FILE="$PWD/logs/extractions.jsonl" \
python api.py --host 0.0.0.0 --port 8000
```

Guia completa:

```text
docs/production-readiness.md
```

## 16. Desplegar con Docker

Construir:

```bash
docker build -t factura-training-api .
```

Ejecutar:

```bash
docker run --rm -p 8000:8000 \
  -e FACTURA_API_KEY="<API_KEY>" \
  -e FACTURA_LOG_FILE="/app/logs/extractions.jsonl" \
  -v "$PWD/logs:/app/logs" \
  factura-training-api
```

## 17. Logs y documentos de usuarios

El sistema actual registra logs de extraccion cuando `FACTURA_LOG_FILE` esta definido.

El log guarda:

- `request_id`;
- estado `ok`;
- fuente usada;
- confianza;
- errores;
- metadata del input;
- tipo de documento detectado.

El sistema actual no guarda PDFs completos ni OCR completo en el log. Tampoco reentrena automaticamente con documentos subidos.

## 18. Estado del sistema

Componentes disponibles:

- API HTTP.
- OCR local.
- Parsers de documentos conocidos.
- Validacion de JSON.
- Evaluacion de dataset real.
- Entrenamiento LoRA.
- Logs JSONL opcionales.
- Dockerfile para API liviana.

## 19. Problemas comunes

### 19.1 `python: command not found`

Activar conda:

```bash
source ~/miniconda3/bin/activate
conda activate factura-training
```

### 19.2 No encuentra PDFs

Usar pattern recursivo:

```bash
--pattern "**/*.pdf"
```

### 19.3 Rutas con espacios

Siempre usar comillas:

```bash
python infer.py --ocr-file "data/real_invoices_analysis_pdf2_pdf/05 Mayo - Catalonia.txt"
```

### 19.4 Tesseract no disponible

Instalar:

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng poppler-utils
```

### 19.5 API da 401

Falta header:

```text
X-API-Key: <API_KEY>
```

### 19.6 VPS no responde desde afuera

Revisar si el proceso esta escuchando en `0.0.0.0`, firewall, puerto abierto o usar tunel SSH.

## 20. Flujo Git

En local/Codex:

```bash
git add .
git commit -m "English commit message"
git push
```

En GPU/VPS:

```bash
cd <PROJECT_DIR>
git pull
```
