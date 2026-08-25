# Factura Training

Sistema para extraer datos estructurados desde facturas, recibos e invoices.

## 1. Que hace

Convierte documentos PDF o imagen en JSON normalizado.

Flujo principal:

```text
PDF / imagen
-> OCR local con Tesseract
-> parsers deterministicos
-> normalizacion
-> validacion
-> Qwen LoRA como fallback opcional
-> JSON final
```

## 2. Documentacion

1. `RUNBOOK.md`: manual operativo completo.
2. `docs/production-readiness.md`: despliegue, API, seguridad y logs.
3. `docs/arca-schema.md`: schema JSON para comprobantes ARCA.
4. `docs/real-invoice-structure-notes.md`: formatos reales soportados.

## 3. Uso rapido

Levantar API:

```bash
python api.py --host 127.0.0.1 --port 8000
```

Healthcheck:

```bash
curl http://127.0.0.1:8000/health
```

Extraer PDF:

```bash
curl -X POST "http://127.0.0.1:8000/extract?force_ocr=true" -F "file=@factura.pdf"
```

## 4. Entrenamiento

El entrenamiento LoRA se hace en la PC con GPU:

```bash
python train.py --max-steps 600
```

El modelo entrenado queda en:

```text
factura-qwen-lora/
```

## 5. Evaluacion

Pipeline completo:

```bash
python scripts/evaluate_real_dataset.py --eval-file data/real_eval.jsonl --mode production --out data/eval_results_production.jsonl
```

LoRA puro:

```bash
python scripts/evaluate_real_dataset.py --eval-file data/real_eval.jsonl --model lora --out data/eval_results_model.jsonl
```

Resumen de errores:

```bash
python scripts/analyze_eval_results.py data/eval_results_production.jsonl
```

## 6. API y despliegue

La guia de API, despliegue, variables de entorno y logs esta en:

```text
docs/production-readiness.md
```
