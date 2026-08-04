# Notas de estructura - facturas reales

Analice 7 PDFs reales de junio para contrastar el flujo actual contra documentos fuera del dataset sintetico.

## Familias encontradas

### Facturas ARCA locales

Hay 2 comprobantes ARCA tipo C. Vienen como PDF de 3 paginas porque incluyen ORIGINAL, DUPLICADO y TRIPLICADO. La informacion de negocio se repite en cada copia, por lo que el extractor debe quedarse con una sola ocurrencia de cada item.

Estructura observada:

- Letra del comprobante en un bloque grande separado: `C`.
- Tipo: `FACTURA`.
- Codigo: `COD. 011`.
- Punto de venta y numero: `Punto de Venta: 00001 Comp. Nro: 00000001`.
- Fecha de emision en la misma linea que razon social.
- CUIT del emisor sin guiones.
- Condicion IVA real como `Responsable Monotributo`.
- Periodo facturado y fecha de vencimiento de pago.
- Receptor con CUIT, razon social, condicion IVA y domicilio.
- Condicion de venta.
- Tabla de items con codigo, descripcion, cantidad, unidad, precio unitario, bonificacion e importe.
- Totales: subtotal, otros tributos e importe total.
- CAE y fecha de vencimiento de CAE.

Decision tecnica aplicada:

- El parser estructurado ahora soporta este formato real de ARCA, ademas del formato sintetico.
- Se normaliza CUIT a `00-00000000-0`.
- Se deduplican items repetidos por ORIGINAL/DUPLICADO/TRIPLICADO.
- Se mantiene `doc_nro` como CUIT sin guiones.

### GoDaddy

Hay 4 comprobantes de GoDaddy. No tienen formato ARCA: son recibos de proveedor externo, en USD, con numero de recibo, fecha, numero de cliente, datos de facturacion, ID fiscal del cliente, metodo de pago y tabla de productos.

Estructura observada:

- Titulo `Recibo`.
- Numero de recibo.
- Fecha.
- Numero de cliente.
- Datos de facturacion.
- ID fiscal local del cliente sin formato ARCA.
- Pago con tarjeta.
- Saldo anterior, pago recibido, saldo adeudado.
- Items/productos con plazo, descripcion y monto.
- Total en USD.

Decision recomendada:

- No conviene forzar estos documentos al schema ARCA actual.
- Conviene agregar un schema separado para `recibo_proveedor_externo` o un schema general de comprobante de gasto.

### Teamwork / Wise

Hay 1 invoice internacional en ingles. Tampoco es ARCA. Tiene proveedor extranjero, cliente argentino con CUIT, referencia, fecha de emision, cuenta, metodo de pago, item, subtotal, total y estado pagado.

Estructura observada:

- Titulo `INVOICE`.
- Referencia del comprobante.
- Fecha de emision.
- Numero de cuenta.
- Proveedor con domicilio y VAT.
- Cliente con CUIT.
- Metodo de pago y ultimos digitos de tarjeta.
- Item con precio.
- Subtotal, total y paid en USD.
- Nota de reverse charge / VAT.

Decision recomendada:

- Tambien requiere schema separado de proveedor externo.
- Es util para una segunda etapa del extractor, pero no deberia mezclarse como factura ARCA A/B/C.

## Impacto sobre el entrenamiento

El dataset sintetico ayudo a estabilizar el JSON ARCA, pero estos documentos muestran que el sistema necesita clasificar primero el tipo de comprobante:

1. Si es ARCA, usar el schema ARCA actual.
2. Si es proveedor externo, usar otro schema.
3. Si el tipo no se reconoce, devolver un JSON de error/control con texto extraido y baja confianza.

## Proximo paso sugerido

Implementar una etapa previa de clasificacion:

- `arca_invoice`
- `external_provider_receipt`
- `external_provider_invoice`
- `unknown`

Despues de clasificar, aplicar el extractor correspondiente. Esto evita que Qwen invente campos ARCA para recibos de GoDaddy o invoices internacionales.
