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

Resultado esperado:

- `outputs/checkpoint-20/`
- `factura-qwen-lora/`

El entrenamiento actual usa ejemplos ARCA en `data/train.jsonl`, `max_seq_length=2048` y `max_steps=80`, porque el JSON final es mas largo que la prueba inicial.

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

El archivo `data/eval.jsonl` debe contener ejemplos no usados en entrenamiento, con este formato:

```json
{"input":"texto OCR","output":"{\"tipo_comprobante\":\"Factura A\", ...}"}
```

Para que la medición tenga sentido, conviene separar:

- `data/train.jsonl`: ejemplos de entrenamiento.
- `data/eval.jsonl`: ejemplos nunca vistos, usados solo para medir.

## Próximo paso recomendado

El dataset ya fue adaptado a `docs/arca-schema.md` y `schemas/arca_invoice_schema.json`. El proximo salto es agregar 30-100 facturas reales anonimizadas de ARCA/OCR, manteniendo ese mismo esquema. Con pocos ejemplos el LoRA puede memorizar formato, pero todavia no demuestra generalizacion.
