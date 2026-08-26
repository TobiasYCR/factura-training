# Schema ARCA normalizado

## 1. Objetivo

Este documento define el JSON que devuelve el sistema cuando detecta una factura, nota de credito o nota de debito tipo ARCA/ex AFIP.

No se intenta devolver XML/SOAP de ARCA. La salida es un JSON interno, estable y facil de validar.

`numero_factura` conserva el formato tradicional del comprobante. La API agrega
`numero_factura_completo` e `iva_porcentaje` como campos de integracion para la
pantalla administrativa, sin reemplazar los campos normalizados originales.

## 2. Fuente conceptual

El schema se basa en conceptos de comprobantes electronicos ARCA:

- tipo de comprobante;
- punto de venta;
- numero de comprobante;
- fecha;
- emisor;
- receptor;
- moneda;
- importes;
- IVA;
- tributos;
- CAE.

La documentacion revisada fue el manual de desarrollador ARCA/COMPG.

## 3. Flujo donde se usa

```text
PDF ARCA
-> OCR
-> parser o Qwen
-> normalizacion al schema
-> validacion
-> JSON final
```

## 4. Campos raiz

| Campo | Tipo | Descripcion |
| --- | --- | --- |
| `tipo_comprobante` | string/null | Ej: `Factura A`, `Factura B`, `Factura C`, `Nota de credito A`. |
| `codigo_comprobante` | integer/null | Codigo ARCA si se puede inferir. Ej: `1` para Factura A, `6` para Factura B, `11` para Factura C. |
| `punto_venta` | string/null | Punto de venta normalizado a 5 digitos. |
| `numero_comprobante` | string/null | Numero normalizado a 8 digitos. |
| `numero_factura` | string/null | Formato `00000-00000000`. |
| `numero_factura_completo` | string/null | Identificador para integraciones: `CUIT(11)_CODIGO(3)_PUNTO_VENTA(5)_NUMERO(8)`. |
| `fecha_emision` | string/null | Fecha ISO `YYYY-MM-DD`. |
| `emisor` | object | Datos del emisor. |
| `receptor` | object | Datos del receptor. |
| `moneda` | string/null | `PES`, `DOL` u otra moneda ARCA si aparece. |
| `tipo_cambio` | number/null | Para pesos normalmente `1`. |
| `subtotal` | number/null | Neto gravado o subtotal principal. |
| `importe_no_gravado` | number/null | Importe no gravado. |
| `importe_exento` | number/null | Importe exento. |
| `iva_total` | number/null | Total IVA. |
| `tributos_total` | number/null | Total tributos/percepciones. |
| `impuestos` | number/null | Suma general de impuestos cuando aplica. |
| `total` | number/null | Importe total final. |
| `cae` | string/null | Codigo de autorizacion electronico. |
| `fecha_vencimiento_cae` | string/null | Fecha ISO `YYYY-MM-DD`. |
| `iva_porcentaje` | number/array/null | Porcentaje calculado o informado; puede ser `0`, `21`, `10.5`, `27` o varios porcentajes. |
| `iva` | array | Detalle de IVA. |
| `tributos` | array | Detalle de tributos/percepciones. |
| `items` | array | Detalle de items si el OCR permite extraerlos. |

## 5. Persona: emisor y receptor

`emisor` y `receptor` tienen siempre estas claves:

| Campo | Tipo | Regla |
| --- | --- | --- |
| `nombre` | string/null | Razon social o nombre limpio, sin etiquetas como `Cliente:`. |
| `doc_tipo` | integer/null | `80` para CUIT cuando se puede inferir. |
| `doc_nro` | string/null | Documento sin guiones. |
| `cuit` | string/null | CUIT con formato `00-00000000-0`. |
| `condicion_iva` | string/null | Condicion frente al IVA si aparece. |

## 6. IVA

Cada item de `iva`:

```json
{
  "codigo": 5,
  "descripcion": "21%",
  "base_imponible": 100.0,
  "importe": 21.0
}
```

Codigos usados:

- `4`: IVA 10.5%.
- `5`: IVA 21%.
- `6`: IVA 27%.

Si el comprobante no discrimina IVA, usar `iva: []` y `iva_total: 0` o `null` segun corresponda.

## 7. Tributos

Cada item de `tributos`:

```json
{
  "codigo": 99,
  "descripcion": "Percepcion IIBB",
  "base_imponible": 100.0,
  "alicuota": 3.0,
  "importe": 3.0
}
```

Si el OCR no informa codigo especifico, usar `99`.

## 8. Items

Cada item de `items`:

```json
{
  "descripcion": "Servicio mensual",
  "cantidad": 1,
  "precio_unitario": 100.0,
  "importe": 100.0
}
```

Reglas:

- Extraer items solo cuando aparezcan lineas claras de productos/servicios.
- No inventar items.
- Si no se puede extraer con seguridad, usar `items: []`.
- En PDFs con ORIGINAL/DUPLICADO/TRIPLICADO, deduplicar items repetidos.

## 9. Moneda

Usar codigos compatibles con ARCA:

- `PES`: pesos argentinos.
- `DOL`: dolares.

No usar `ARS` para facturas ARCA si se apunta a compatibilidad con ARCA.

## 10. Fechas

Todas las fechas salen en formato ISO:

```text
YYYY-MM-DD
```

Ejemplos:

- `02/01/2023` -> `2023-01-02`.
- `31/12/2022` -> `2022-12-31`.

## 11. Numeracion

Reglas:

- `punto_venta`: 5 digitos.
- `numero_comprobante`: 8 digitos.
- `numero_factura`: `punto_venta-numero_comprobante`.

Ejemplo:

```json
{
  "punto_venta": "00002",
  "numero_comprobante": "00000045",
  "numero_factura": "00002-00000045"
}
```

## 12. Validaciones actuales

Validaciones implementadas o usadas por el pipeline:

- El JSON contiene todas las claves del schema.
- Fechas en formato ISO valido.
- CUIT en formato `00-00000000-0`.
- CAE numerico cuando aparece.
- `numero_factura` en formato `00000-00000000`.
- Importes como numeros o `null`.
- Arrays presentes aunque esten vacios.

## 13. Ejemplo completo

```json
{
  "tipo_comprobante": "Factura A",
  "codigo_comprobante": 1,
  "punto_venta": "00002",
  "numero_comprobante": "00000045",
  "numero_factura": "00002-00000045",
  "fecha_emision": "2019-01-02",
  "emisor": {
    "nombre": "MARRANO RICARDO ANDRES",
    "doc_tipo": 80,
    "doc_nro": "20234388518",
    "cuit": "20-23438851-8",
    "condicion_iva": "IVA Responsable Inscripto"
  },
  "receptor": {
    "nombre": "CS TECH CONSULTING S.A.",
    "doc_tipo": 80,
    "doc_nro": "30715444530",
    "cuit": "30-71544453-0",
    "condicion_iva": "Responsable Inscripto"
  },
  "moneda": "PES",
  "tipo_cambio": 1,
  "subtotal": 8880.0,
  "importe_no_gravado": 0.0,
  "importe_exento": 0.0,
  "iva_total": 0.0,
  "tributos_total": 0.0,
  "impuestos": 0.0,
  "total": 8880.0,
  "cae": "02690182985567",
  "fecha_vencimiento_cae": "2019-01-12",
  "iva": [],
  "tributos": [],
  "items": []
}
```

## 14. Formato usado en entrenamiento

Los ejemplos de entrenamiento usan valores limpios:

- Bien: `"numero_factura": "00008-00009123"`.
- Mal: `"numero_factura": "Comp. Nro: 0008-00009123"`.
- Bien: `"cuit": "30-87654321-0"`.
- Mal: `"cuit": "CUIT: 30-87654321-0"`.
- Bien: `"moneda": "PES"`.
- Mal: `"moneda": "ARS"` para ARCA.

El modelo recibe OCR con ruido y la salida esperada queda normalizada.
