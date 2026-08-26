# Estructura de documentos reales

## 1. Objetivo

Este documento resume las familias de documentos reales que aparecieron durante las pruebas y como las trata el sistema.

Sirve para decidir:

- que parser usar;
- que schema devolver;
- que schema devuelve cada familia.

## 2. Clasificacion general

El sistema clasifica el documento:

1. `arca_invoice`: factura/nota ARCA local.
2. `external_provider_receipt`: recibo de proveedor externo.
3. `external_provider_invoice`: invoice o comprobante externo.
4. `unknown`: documento no reconocido.

Despues de clasificar, aplica el extractor correspondiente.

## 3. Facturas ARCA locales

Incluye facturas A, B, C, notas de credito y notas de debito.

Estructura comun:

- tipo de comprobante;
- letra;
- codigo;
- punto de venta;
- numero de comprobante;
- fecha de emision;
- CUIT del emisor;
- razon social del emisor;
- condicion IVA;
- receptor;
- condicion de venta;
- items;
- subtotal;
- IVA;
- percepciones/tributos;
- total;
- CAE;
- vencimiento CAE.

La API tambien expone `descripcion` como campo de integracion para la pantalla
administrativa. En facturas con una tabla `Producto / Servicio` se toma la
descripcion del item. En documentos como OSDE, donde el concepto aparece bajo
`Referencia`, se usa ese valor.

Salida:

```text
schema ARCA normalizado
```

Documento de referencia:

```text
docs/arca-schema.md
```

## 4. ARCA con ORIGINAL/DUPLICADO/TRIPLICADO

Algunas facturas vienen en 3 paginas:

1. ORIGINAL.
2. DUPLICADO.
3. TRIPLICADO.

La informacion se repite. El extractor:

- tomar una sola version de los datos;
- deduplicar items repetidos;
- no sumar tres veces los totales.

## 5. GoDaddy

GoDaddy no usa formato ARCA en los comprobantes analizados.

Estructura observada:

- titulo tipo `Recibo` o pagina de facturacion;
- numero de recibo;
- fecha;
- numero de cliente;
- datos de facturacion;
- ID fiscal del cliente;
- metodo de pago;
- tarjeta y ultimos digitos;
- saldo anterior;
- pago recibido;
- saldo adeudado;
- productos/servicios;
- total;
- moneda.

Salida:

```text
external_provider_receipt
```

Tratamiento actual:

- No forzar GoDaddy al schema ARCA.
- Mantener schema externo.
- Extraer total, pago, saldo, cliente, numero y productos.
- Exponer `descripcion` usando el valor de `Producto`; si no existe, usar `Referencia`.

## 6. Teamwork / Wise

Documentos internacionales en ingles.

Estructura observada:

- `INVOICE`;
- referencia;
- fecha;
- numero de cuenta;
- proveedor extranjero;
- VAT o tax ID;
- cliente;
- CUIT local si aparece;
- metodo de pago;
- item;
- subtotal;
- total;
- paid/balance.

Salida:

```text
external_provider_invoice
```

Tratamiento actual:

- Mantener schema externo.
- No convertir moneda o impuestos si el documento no lo informa.
- Guardar notas legales o reverse charge en `notes`.

## 7. OSDE

OSDE puede aparecer como factura o nota de debito.

Estructura observada:

- proveedor OSDE;
- numero de factura/nota;
- fecha de emision;
- CUIT;
- cliente;
- periodo;
- neto;
- IVA;
- percepcion;
- total;
- CAE.

Salida:

```text
schema ARCA normalizado
```

Tratamiento actual:

- Tratarlo como ARCA si contiene numeracion, CUIT, CAE y totales compatibles.
- Soportar `Nota de debito A` cuando corresponda.

## 8. Despegar / viajes / aerolineas

Pueden aparecer como:

- factura ARCA;
- nota de credito;
- recibo de viaje;
- constancia de credito fiscal;
- detalle no fiscal.

Tratamiento actual:

- Si tiene CAE y datos ARCA, usar schema ARCA.
- Si es recibo o detalle sin CAE, usar schema externo o marcar como desconocido.
- No inventar CAE ni punto de venta cuando no aparece.

En facturas de Cablevision/Fibertel, cuando `Conceptos` contiene varias lineas,
la API conserva los items y genera una descripcion resumida que agrupa
television, packs, internet y descuentos, incluyendo el periodo si aparece.

## 9. FlyBondi / Aerolineas

En algunos casos aparecen comprobantes de credito fiscal o facturas de transporte.

Tratamiento actual:

- Si trae estructura fiscal, extraer como ARCA.
- Si es comprobante externo, usar schema externo.
- Mantener `items: []` si no hay tabla clara.

## 10. Catalonia / hoteles

Estructura observada:

- proveedor/hotel;
- numero de factura;
- fecha;
- cliente;
- servicios;
- impuestos;
- total;
- moneda.

Salida:

```text
external_provider_invoice
```

Tratamiento actual:

- No forzar a ARCA si no tiene CAE y estructura local.
- Extraer proveedor, comprador, documento, moneda, total e items si estan claros.

## 11. Lenovo / hardware

Algunas facturas de hardware tienen estructura ARCA pero layout distinto.

Tratamiento actual:

- Usar parser ARCA especializado si el OCR trae CUIT, CAE y totales.
- Validar numeracion y CAE.

## 12. iFastNet / WFWEF

Comprobantes externos de hosting/servicios.

Salida:

```text
external_provider_invoice
```

Tratamiento actual:

- Extraer numero, fecha, proveedor, comprador, moneda, total y pago.
- No convertir a ARCA si no hay CAE.

## 13. Schema externo

Los documentos no ARCA usan:

```text
schemas/external_provider_document_schema.json
```

Claves raiz:

- `document_type`;
- `provider`;
- `buyer`;
- `document`;
- `currency`;
- `subtotal`;
- `taxes`;
- `fees`;
- `total`;
- `paid`;
- `balance_due`;
- `payment`;
- `items`;
- `notes`.

## 14. Valores no inventados

No inventar:

- CAE;
- vencimiento CAE;
- CUIT;
- punto de venta;
- numero de comprobante;
- IVA discriminado;
- percepciones;
- items.

Si no aparecen o el OCR no los lee, usar `null` o array vacio segun el schema.
