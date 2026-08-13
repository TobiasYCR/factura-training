# Factura OCR -> JSON fine-tuning

Flujo de prueba:

1. OCR local produce texto.
2. Qwen chico fine-tuneado recibe texto OCR.
3. El modelo devuelve JSON.
4. Scripts de validación revisan si el JSON es parseable y si tiene las claves esperadas.

## Entrar al entorno remoto

Desde tu PC:

```bash
ssh tobias@100.96.9.102
wsl
cd /mnt/c/Users/tobias/factura-training
source ~/miniconda3/bin/activate
conda activate factura-training
```

Si vuelve el error `No space left on device` durante instalaciones:

```bash
mkdir -p ~/tmp
export TMPDIR=$HOME/tmp
export TEMP=$HOME/tmp
export TMP=$HOME/tmp
```

## Confirmar GPU y dependencias

```bash
python -c "import torch; import unsloth; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Entrenar

```bash
python train.py
```

Por defecto, `train.py` usa:

- `data/train.jsonl`
- `data/synthetic_invoices/synthetic_train.jsonl`, si existe

Para entrenar solo con un archivo:

```bash
python train.py --data-files data/train.jsonl --max-steps 80
```

Resultado esperado:

- `outputs/checkpoint-20/`
- `factura-qwen-lora/`

El entrenamiento actual usa ejemplos ARCA manuales y puede sumar el dataset sintetico generado, con `max_seq_length=2048` y `max_steps=160` por defecto, porque el JSON final es mas largo que la prueba inicial.

## Probar una factura nueva con el LoRA

```bash
python infer.py --model lora --ocr-file data/test_ocr.txt
```

Para probar texto directo:

```bash
python infer.py --model lora --ocr-text "Factura A 0001-00000099 ..."
```

## Comparar base vs fine-tuneado

```bash
python compare_base_lora.py --ocr-file data/test_ocr.txt
```

Este script carga primero el modelo base, libera memoria CUDA y luego carga el LoRA. Es más amable con una RTX 4050 de 6 GB.

Si el modelo genera campos fuera del esquema o JSON roto, no es un problema de CUDA: es senal de que el dataset todavia es demasiado chico. Reentrena con el `train.py` actualizado y agrega mas ejemplos reales anonimizados.

## Evaluar campo por campo

```bash
python evaluate.py --model both --eval-file data/eval.jsonl
```

## Levantar endpoint local

El endpoint local permite subir un PDF y devolver el JSON normalizado usando el mismo parser de `infer.py`.

```bash
python api.py --host 127.0.0.1 --port 8000
```

Healthcheck:

```bash
curl http://127.0.0.1:8000/health
```

Extraer desde PDF:

```bash
curl -X POST http://127.0.0.1:8000/extract -F "file=@factura.pdf"
```

Extraer desde texto OCR:

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d "{\"ocr_text\":\"Factura A ...\"}"
```

Por defecto primero intenta resolver con parsers deterministicos para ARCA, GoDaddy y Teamwork/Wise. Si el documento no se reconoce y se quiere usar Qwen/LoRA como fallback:

```bash
curl -X POST "http://127.0.0.1:8000/extract?use_model=true&model=lora" -F "file=@factura.pdf"
```

### OCR local para PDFs escaneados

El endpoint no usa servicios externos pagos. Primero intenta leer texto embebido del PDF. Si el PDF es escaneado o se fuerza OCR, renderiza las paginas y usa Tesseract local.

En Ubuntu/WSL:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng poppler-utils
```

En Windows conviene instalar Tesseract y dejar `tesseract.exe` en el `PATH`, o definir:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Forzar OCR aunque el PDF tenga texto embebido:

```bash
curl -X POST "http://127.0.0.1:8000/extract?force_ocr=true&ocr_lang=spa+eng&ocr_dpi=220" -F "file=@factura-escaneada.pdf"
```

Tambien acepta imagenes:

```bash
curl -X POST "http://127.0.0.1:8000/extract?ocr_lang=spa+eng" -F "file=@factura.jpg"
```

El `GET /health` informa si Tesseract esta disponible y que comando detecto.

## Procesar una carpeta completa

Para probar muchos PDFs juntos:

```bash
python scripts/batch_extract_pdfs.py /mnt/c/Users/tobias/Desktop/PDF
```

El resumen queda en:

```text
data/real_invoices_analysis/batch_summary.jsonl
```

Ese directorio esta ignorado por Git porque puede contener datos reales. Si queres guardar el JSON de cada PDF:

```bash
python scripts/batch_extract_pdfs.py /mnt/c/Users/tobias/Desktop/PDF --write-json
```

En la PC con Tesseract instalado tambien se puede forzar OCR visual:

```bash
python scripts/batch_extract_pdfs.py /mnt/c/Users/tobias/Desktop/PDF --force-ocr
```

## Generar facturas sinteticas

```bash
python scripts/generate_synthetic_invoices.py --count 100 --seed 42
```

Salida:

- `data/synthetic_invoices/synthetic_train.jsonl`: dataset para entrenamiento.
- `data/synthetic_invoices/manifest.jsonl`: indice de ejemplos generados.
- `data/synthetic_invoices/pdfs/`: PDFs regenerables, ignorados por Git.
- `data/synthetic_invoices/ocr/`: textos OCR regenerables, ignorados por Git.

El archivo `data/eval.jsonl` debe contener ejemplos no usados en entrenamiento, con este formato:

```json
{"input":"texto OCR","output":"{\"tipo_comprobante\":\"Factura A\", ...}"}
```

Para que la medición tenga sentido, conviene separar:

- `data/train.jsonl`: ejemplos de entrenamiento.
- `data/eval.jsonl`: ejemplos nunca vistos, usados solo para medir.

## Próximo paso recomendado

El dataset ya fue adaptado a `docs/arca-schema.md` y `schemas/arca_invoice_schema.json`. El proximo salto es agregar 30-100 facturas reales anonimizadas de ARCA/OCR, manteniendo ese mismo esquema. Con pocos ejemplos el LoRA puede memorizar formato, pero todavia no demuestra generalizacion.
