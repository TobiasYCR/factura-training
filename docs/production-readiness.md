# Preparacion para piloto/produccion

Este proyecto debe desplegarse como un sistema completo de extraccion:

1. PDF o imagen.
2. OCR local con Tesseract.
3. Parsers deterministicos para formatos conocidos.
4. Normalizacion al schema esperado.
5. Validacion.
6. Fallback opcional con Qwen LoRA cuando haya GPU disponible.
7. Logs para revisar errores y mejorar el dataset.

## Estado recomendado actual

Con los documentos reales actuales, el sistema esta listo para piloto interno si se usa el flujo completo, no Qwen solo.

Para produccion final todavia conviene:

- revisar los casos con baja coincidencia campo por campo;
- guardar logs de extracciones reales;
- conectar la web a `POST /extract`;
- medir tiempos con PDFs reales en el VPS;
- revisar legal/privacidad antes de guardar documentos de usuarios.

## Variables de entorno

```bash
FACTURA_API_KEY="cambiar-esta-clave"
FACTURA_LOG_FILE="/var/log/factura-training/extractions.jsonl"
FACTURA_MAX_UPLOAD_MB="25"
OCR_LANG="spa+eng"
OCR_DPI="220"
```

Si `FACTURA_API_KEY` esta definida, `POST /extract` exige:

```text
X-API-Key: cambiar-esta-clave
```

`GET /health` queda abierto para monitoreo.

## Ejecutar en VPS con Python

```bash
cd ~/factura-training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng poppler-utils
FACTURA_API_KEY="cambiar-esta-clave" \
FACTURA_LOG_FILE="$PWD/logs/extractions.jsonl" \
python api.py --host 0.0.0.0 --port 8000
```

## Ejecutar con Docker

La imagen Docker incluida es para API liviana: OCR, parsers y validacion. No instala Unsloth ni carga el LoRA.

```bash
docker build -t factura-training-api .
docker run --rm -p 8000:8000 \
  -e FACTURA_API_KEY="cambiar-esta-clave" \
  -e FACTURA_LOG_FILE="/app/logs/extractions.jsonl" \
  -v "$PWD/logs:/app/logs" \
  factura-training-api
```

Healthcheck:

```bash
curl http://127.0.0.1:8000/health
```

Extraccion:

```bash
curl -X POST "http://127.0.0.1:8000/extract?force_ocr=true" \
  -H "X-API-Key: cambiar-esta-clave" \
  -F "file=@factura.pdf"
```

En una maquina con GPU, para permitir que Qwen mejore respuestas de baja confianza:

```bash
curl -X POST "http://127.0.0.1:8000/extract?force_ocr=true&use_model=true&model=lora&model_policy=low_confidence&min_confidence=0.82" \
  -H "X-API-Key: cambiar-esta-clave" \
  -F "file=@factura.pdf"
```

## Respuesta de la API

La API devuelve:

- `ok`: si el JSON paso validacion.
- `source`: `parser`, `model` o `null`.
- `confidence`: confianza simple para priorizar revision.
- `elapsed_ms`: tiempo de extraccion posterior al OCR/modelo.
- `errors`: errores de validacion.
- `data`: JSON normalizado.
- `input`: archivo, metodo de texto/OCR y largo del OCR.
- `request_id`: identificador para cruzar logs.

`model_policy` controla cuando se usa Qwen:

- `fallback`: solo si el parser no resuelve.
- `low_confidence`: si el parser falla o devuelve baja confianza.
- `always`: prueba Qwen siempre y conserva la respuesta mas confiable.

## Logs

Si `FACTURA_LOG_FILE` esta definido, se guarda un JSONL por request con:

- `request_id`;
- estado `ok`;
- fuente usada;
- confianza;
- errores;
- tipo de documento;
- metadata del input.

No guarda el PDF ni el OCR completo. Si se decide guardar documentos de usuarios para mejorar el sistema, debe hacerse con consentimiento y revision humana.

## Analizar evaluaciones

Despues de correr:

```bash
python scripts/evaluate_real_dataset.py --eval-file data/real_eval.jsonl --mode production --out data/eval_results_production.jsonl
```

resumir errores con:

```bash
python scripts/analyze_eval_results.py data/eval_results_production.jsonl
```

Para el LoRA:

```bash
python scripts/evaluate_real_dataset.py --eval-file data/real_eval.jsonl --model lora --out data/eval_results_model.jsonl
python scripts/analyze_eval_results.py data/eval_results_model.jsonl
```

## Criterio de cierre

Para considerar la version lista para un piloto:

- `JSON/schema OK` cercano a 100% en production.
- tiempos reales por PDF por debajo de 10 segundos en promedio.
- API key activa.
- logs activos.
- web conectada a `/extract`.
- errores dudosos visibles para revision.

Para considerar produccion final:

- lote real amplio validado;
- politica de privacidad/retencion definida;
- monitoreo de errores;
- backups de modelos/datasets;
- proceso de reentrenamiento versionado.
