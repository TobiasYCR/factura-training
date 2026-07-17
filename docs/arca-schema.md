# ARCA invoice extraction schema

Fuente revisada: `manual-desarrollador-ARCA-COMPG.pdf`, especialmente las secciones de `FECAESolicitar`, `FECompConsultar`, tipos de comprobante, tipos de documento, monedas, cotizacion, IVA y tributos.

## Decision

El modelo no debe intentar devolver el XML/SOAP de ARCA. Para nuestra app conviene devolver un JSON normalizado, estable y facil de validar. Los nombres internos quedan en `snake_case`, y cada campo mantiene una correspondencia clara con los conceptos de ARCA.

El pipeline queda:

```text
PDF factura ARCA
-> OCR
-> texto OCR
-> Qwen LoRA
-> JSON normalizado
-> validaciones por codigo
-> backend / base de datos
```

## Core fields

Campos minimos para facturas A, B y C:

| Campo JSON | Equivalente ARCA | Tipo | Reglas |
| --- | --- | --- | --- |
| `tipo_comprobante` | `CbteTipo` descripcion | string | Ej: `Factura A`, `Factura B`, `Factura C`. |
| `codigo_comprobante` | `CbteTipo` | integer/null | Ej: A suele mapear a `1`, B a `6`, C a `11` cuando se conozca. |
| `punto_venta` | `PtoVta` | string/null | Siempre normalizado a 5 digitos cuando aparezca. |
| `numero_comprobante` | `CbteDesde` / `CbteHasta` | string/null | Siempre normalizado a 8 digitos cuando aparezca. |
| `numero_factura` | `PtoVta-CbteDesde` | string/null | Formato `00000-00000000` para ARCA moderno. Aceptar `0000-00000000` en facturas viejas o muestras. |
| `fecha_emision` | `CbteFch` | string/null | Formato ISO `YYYY-MM-DD`. |
| `emisor.nombre` | texto factura | string/null | Razon social del emisor. |
| `emisor.cuit` | `Auth/Cuit` o emisor impreso | string/null | Formato `00-00000000-0`. |
| `receptor.nombre` | texto factura | string/null | Nombre o razon social del receptor. |
| `receptor.doc_tipo` | `DocTipo` | integer/null | Ej: `80` para CUIT si se puede inferir. |
| `receptor.doc_nro` | `DocNro` | string/null | Normalizado sin guiones para codigos ARCA; puede conservar CUIT con guiones en campo adicional si hace falta. |
| `moneda` | `MonId` | string | `PES`, `DOL`, etc. Usar `PES` para pesos ARCA, no `ARS`, si apuntamos a compatibilidad ARCA. |
| `tipo_cambio` | `MonCotiz` | number/null | Para `PES`, normalmente `1`. |
| `subtotal` | `ImpNeto` | number/null | Neto gravado. |
| `importe_no_gravado` | `ImpTotConc` | number/null | Neto no gravado. |
| `importe_exento` | `ImpOpEx` | number/null | Importe exento. |
| `impuestos` | `ImpIVA + ImpTrib` | number/null | Total de impuestos si solo queremos un resumen. |
| `iva_total` | `ImpIVA` | number/null | Suma del array IVA. |
| `tributos_total` | `ImpTrib` | number/null | Suma del array Tributos. |
| `total` | `ImpTotal` | number/null | Total final. |
| `cae` | `CAE` / `CodAutorizacion` | string/null | Codigo de autorizacion electronico. |
| `fecha_vencimiento_cae` | `CAEFchVto` / `FchVto` | string/null | Formato ISO `YYYY-MM-DD`. |

## Nested arrays

Cuando el OCR lo permita, extraer:

```json
{
  "iva": [
    {
      "codigo": 5,
      "descripcion": "21%",
      "base_imponible": 100.0,
      "importe": 21.0
    }
  ],
  "tributos": [
    {
      "codigo": 99,
      "descripcion": "Impuesto Municipal",
      "base_imponible": 150.0,
      "alicuota": 5.2,
      "importe": 7.8
    }
  ],
  "items": [
    {
      "descripcion": "Servicio mensual",
      "cantidad": 1,
      "precio_unitario": 100.0,
      "importe": 100.0
    }
  ]
}
```

## Validation rules

Validaciones que deben vivir en codigo, no en el modelo:

- `total` debe aproximar `subtotal + importe_no_gravado + importe_exento + iva_total + tributos_total`.
- Para comprobantes C, `iva_total` y array `iva` deberian ser `0` o vacios.
- `moneda = "PES"` implica `tipo_cambio = 1` salvo que la factura indique otra cosa rara.
- `fecha_emision` y `fecha_vencimiento_cae` deben ser fechas validas.
- `numero_factura` debe poder separarse en `punto_venta` y `numero_comprobante`.
- Si aparece CAE, debe ser numerico y usualmente de 14 digitos.

## Training guidance

Para fine-tuning, cada ejemplo debe enseñar valores limpios:

- Usar `numero_factura: "00008-00009123"`, no `"Comp. Nro: 0008-00009123"`.
- Usar `emisor.cuit: "30-87654321-0"`, no `"CUIT: 30-87654321-0"`.
- Usar `receptor.nombre: "Transporte Norte SA"`, no `"Cliente: Transporte Norte SA"`.
- Usar `moneda: "PES"` para pesos y `moneda: "DOL"` para dolares si queremos compatibilidad con ARCA.

El OCR puede traer etiquetas, saltos raros y ruido. Qwen debe normalizar; el validador debe detectar inconsistencias.
