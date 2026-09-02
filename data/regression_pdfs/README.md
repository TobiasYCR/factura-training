# Regression PDFs

Carpeta para PDFs reales usados como set fijo de regresion OCR.

Uso recomendado:

1. Copiar aca documentos representativos y dificiles, sin subcarpetas.
2. Usar nombres descriptivos, por ejemplo:
   - `arca_factura_b_consumidor_final_001.pdf`
   - `arca_factura_b_mipyme_001.pdf`
   - `arca_factura_a_personalizada_cablevision_001.pdf`
   - `externo_godaddy_recibo_001.pdf`
3. No subir facturas con datos sensibles sin autorizacion.
4. Ejecutar:

```bash
python scripts/batch_extract_pdfs.py data/regression_pdfs --pattern "*.pdf" --output-dir tmp/regression_check --force-ocr --ocr-policy auto --write-json --write-ocr
```

Cuando el resultado este revisado, copiar el JSON correcto a `data/regression_expected/` con el mismo nombre base del PDF:

```text
data/regression_pdfs/arca_factura_b_mipyme_001.pdf
data/regression_expected/arca_factura_b_mipyme_001.json
```

Luego correr:

```bash
python scripts/run_regression_pdfs.py data/regression_pdfs --expected-dir data/regression_expected --force-ocr
```

El objetivo es mantener una muestra chica, de 20 a 40 PDFs, que cubra:

- Facturas A, B y C de ARCA.
- Facturas MiPyME.
- Notas de credito y debito.
- Proveedores personalizados.
- Recibos externos.
- Escaneos con baja calidad o layout raro.
