# Guia de API y despliegue

## 1. Que se despliega

En produccion se despliega el sistema completo de extraccion, no Qwen solo.

```text
PDF o imagen
-> OCR local
-> parser/reglas
-> normalizacion
-> validacion
-> fallback Qwen LoRA opcional
-> JSON final
```

La VPS sin GPU corre OCR + parsers + validacion. El fallback con Qwen LoRA requiere entorno con dependencias de modelo y GPU para tiempos bajos.

## 2. Componentes disponibles

El proyecto incluye:

1. API HTTP en `api.py`.
2. OCR local en `ocr.py`.
3. Parsers y validadores en `infer.py`.
4. Logs JSONL opcionales.
5. Dockerfile para API liviana.
6. Configuracion por variables de entorno.

## 3. Variables de entorno

Los ejemplos usan placeholders para evitar publicar datos sensibles:

- `<API_KEY>`: clave privada usada por la web/backend para llamar a la API.
- `<SERVICE_USER>`: usuario Linux que ejecuta el servicio.
- `<PROJECT_DIR>`: ruta absoluta del proyecto en el servidor.

```bash
FACTURA_API_KEY="<API_KEY>"
FACTURA_LOG_FILE="/var/log/factura-training/extractions.jsonl"
FACTURA_MAX_UPLOAD_MB="25"
OCR_LANG="spa+eng"
OCR_DPI="220"
```

Significado:

- `FACTURA_API_KEY`: protege `POST /extract`.
- `FACTURA_LOG_FILE`: guarda logs JSONL livianos.
- `FACTURA_MAX_UPLOAD_MB`: limite de subida.
- `OCR_LANG`: idiomas para Tesseract.
- `OCR_DPI`: resolucion para renderizar PDFs escaneados.

## 4. Seguridad de API

Si `FACTURA_API_KEY` esta definida, cada request a `/extract` usa:

```text
X-API-Key: <API_KEY>
```

`GET /health` queda abierto para monitoreo.

Ejemplo:

```bash
curl -X POST "http://127.0.0.1:8000/extract?force_ocr=true" \
  -H "X-API-Key: <API_KEY>" \
  -F "file=@factura.pdf"
```

## 5. Ejecutar en VPS con Python

Instalar sistema:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng poppler-utils
```

Preparar proyecto:

```bash
cd ~/factura-training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
```

Levantar API:

```bash
FACTURA_API_KEY="<API_KEY>" \
FACTURA_LOG_FILE="$PWD/logs/extractions.jsonl" \
python api.py --host 0.0.0.0 --port 8000
```

Probar:

```bash
curl http://127.0.0.1:8000/health
```

## 6. Ejecutar con Docker

La imagen Docker incluida es liviana. Instala OCR, parsers y validacion. No instala Unsloth ni carga el LoRA.

Construir:

```bash
docker build -t factura-training-api .
```

Ejecutar:

```bash
docker run --rm -p 8000:8000 \
  -e FACTURA_API_KEY="<API_KEY>" \
  -e FACTURA_LOG_FILE="/app/logs/extractions.jsonl" \
  -e FACTURA_MAX_UPLOAD_MB="25" \
  -v "$PWD/logs:/app/logs" \
  factura-training-api
```

## 7. Ejecutar como servicio systemd

Crear archivo:

```bash
sudo nano /etc/systemd/system/factura-api.service
```

Contenido:

```ini
[Unit]
Description=Factura Training API
After=network.target

[Service]
User=<SERVICE_USER>
WorkingDirectory=<PROJECT_DIR>
Environment="FACTURA_API_KEY=<API_KEY>"
Environment="FACTURA_LOG_FILE=<PROJECT_DIR>/logs/extractions.jsonl"
Environment="FACTURA_MAX_UPLOAD_MB=25"
ExecStart=<PROJECT_DIR>/.venv/bin/python <PROJECT_DIR>/api.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Activar:

```bash
sudo systemctl daemon-reload
sudo systemctl enable factura-api
sudo systemctl restart factura-api
sudo systemctl status factura-api
```

Ver logs:

```bash
sudo journalctl -u factura-api -n 100 --no-pager
```

## 8. Contrato del endpoint

Endpoint:

```text
POST /extract
```

Multipart:

```text
file=@factura.pdf
```

Query params utiles:

- `force_ocr=true`: fuerza OCR visual.
- `ocr_lang=spa+eng`: idiomas de OCR.
- `ocr_dpi=220`: DPI de renderizado.
- `use_model=true`: permite fallback con Qwen.
- `model=lora`: usa `factura-qwen-lora`.
- `model_policy=fallback`: usa Qwen solo si parser falla.
- `model_policy=low_confidence`: usa Qwen si parser falla o baja confianza.
- `model_policy=always`: prueba Qwen siempre.
- `min_confidence=0.82`: umbral para `low_confidence`.

## 9. Respuesta de la API

Campos principales:

- `ok`: si el resultado paso validacion.
- `source`: `parser`, `model` o `null`.
- `model`: modelo usado, si se uso Qwen.
- `confidence`: confianza estimada.
- `elapsed_ms`: tiempo de extraccion posterior al OCR/modelo.
- `errors`: errores de validacion.
- `data`: JSON normalizado.
- `raw_model_response`: salida cruda del modelo si se uso.
- `input`: metadata del archivo y OCR.
- `request_id`: identificador para cruzar logs.

## 10. Logs

Si `FACTURA_LOG_FILE` esta definido, se guarda un JSONL por request.

Incluye:

- `request_id`;
- timestamp;
- `ok`;
- `source`;
- `model`;
- `confidence`;
- `elapsed_ms`;
- errores;
- metadata del input;
- tipo de documento detectado.

No guarda PDF ni OCR completo.

## 11. Logs y documentos reales

El sistema actual no autoentrena con documentos subidos por usuarios.

Con `FACTURA_LOG_FILE` definido, el sistema guarda un log por request con metadata operativa.

El log no incluye PDF completo ni OCR completo.

Los datasets de entrenamiento se construyen con `scripts/build_real_dataset.py` a partir de archivos OCR/JSON ya procesados.

## 12. Medicion

Metricas disponibles:

- tiempo promedio por PDF;
- porcentaje `ok=true`;
- cantidad de documentos con baja `confidence`;
- errores mas frecuentes;
- proveedores/documentos que mas fallan;
- casos donde OCR no lee bien.

Comandos:

```bash
python scripts/evaluate_real_dataset.py --eval-file data/real_eval.jsonl --mode production --out data/eval_results_production.jsonl
python scripts/analyze_eval_results.py data/eval_results_production.jsonl --eval-file data/real_eval.jsonl
```

## 13. Evaluacion del pipeline

Comando para evaluar el flujo `production`:

```bash
python scripts/evaluate_real_dataset.py --eval-file data/real_eval.jsonl --mode production --out data/eval_results_production.jsonl
```

Comando para analizar resultados:

```bash
python scripts/analyze_eval_results.py data/eval_results_production.jsonl --eval-file data/real_eval.jsonl
```

## 14. Evaluacion del LoRA

Comando para evaluar Qwen LoRA puro:

```bash
python scripts/evaluate_real_dataset.py --eval-file data/real_eval.jsonl --model lora --out data/eval_results_model.jsonl
```

Comando para analizar resultados:

```bash
python scripts/analyze_eval_results.py data/eval_results_model.jsonl --eval-file data/real_eval.jsonl
```
