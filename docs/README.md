# Documentacion del proyecto

## 1. Indice

1. `RUNBOOK.md`: manual operativo principal.
2. `docs/production-readiness.md`: despliegue, API, seguridad y logs.
3. `docs/arca-schema.md`: contrato JSON para comprobantes ARCA.
4. `docs/real-invoice-structure-notes.md`: familias de documentos reales.

## 2. Que documenta cada archivo

### 2.1 `RUNBOOK.md`

Manual de uso diario:

- entrar a la PC con GPU;
- activar entorno;
- procesar PDFs;
- construir dataset;
- entrenar;
- evaluar;
- levantar API;
- desplegar;
- resolver errores comunes.

### 2.2 `docs/production-readiness.md`

Guia de API y despliegue:

- variables de entorno;
- API key;
- logs;
- Docker;
- systemd;
- contrato de `/extract`;
- privacidad y logs.

### 2.3 `docs/arca-schema.md`

Define el JSON ARCA:

- campos raiz;
- emisor/receptor;
- IVA;
- tributos;
- items;
- fechas;
- moneda;
- validaciones.

### 2.4 `docs/real-invoice-structure-notes.md`

Resume formatos reales:

- ARCA;
- GoDaddy;
- Teamwork/Wise;
- OSDE;
- Despegar/viajes;
- FlyBondi/Aerolineas;
- Catalonia;
- Lenovo;
- iFastNet/WFWEF.

## 3. Informe Word

El informe Word se mantiene aparte.
