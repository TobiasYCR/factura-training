import unittest

from api import add_arca_integration_fields
from infer import (
    build_display_description,
    extract_arca_concept_items,
    parse_supported_document_ocr,
    validate_extracted_document_json,
)


def arca_text(letter, code, cuit, point_of_sale, receipt_number, cae, due_date, item, total, iva_line=""):
    code = int(code)
    point_of_sale = int(point_of_sale)
    receipt_number = int(receipt_number)
    return f"""Archivo: sample {cuit}_{code:03d}_{point_of_sale:05d}_{receipt_number:08d}.pdf
ORIGINAL
EMISOR DE PRUEBA
FACTURA {letter} COD. {code:03d}
Punto de Venta: {point_of_sale:05d} Comp. Nro: {receipt_number:08d}
Fecha de Emision: 12/04/2021
CUIT: {cuit}
Razon Social: EMISOR DE PRUEBA
Condicion frente al IVA: Responsable Monotributo
CUIT: 30715444530
Apellido y Nombre / Razon Social: CS TECH CONSULTING S.A.
Codigo Producto / Servicio Cantidad U. Medida Precio Unit. % Bonif Imp. Bonif. Subtotal
01 {item} 1,00 unidades {total:.2f} 0,00 0,00 {total:.2f}
Subtotal: $ {total:.2f}
Importe Otros Tributos: $ 0,00
{iva_line}
Importe Total: $ {total:.2f}
CAE Nro: {cae}
Fecha de Vto. de CAE: {due_date}
"""


class ArcaParserRegressionTests(unittest.TestCase):
    def test_factura_c_extracts_code_cae_emitter_and_item(self):
        text = arca_text(
            "C", 11, "27959140850", "00001", "00000003", "72500527231207", "23/12/2022", "Consultoria de procesos", 200000.0
        )
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text))

        self.assertEqual(parsed["codigo_comprobante"], 11)
        self.assertEqual(parsed["tipo_comprobante"], "Factura C")
        self.assertEqual(parsed["cae"], "72500527231207")
        self.assertEqual(parsed["emisor"]["nombre"], "EMISOR DE PRUEBA")
        self.assertEqual(parsed["items"][0]["descripcion"], "Consultoria de procesos")
        self.assertEqual(parsed["numero_factura_completo"], "27959140850_011_00001_00000003")
        self.assertEqual(validate_extracted_document_json(parsed), [])

    def test_visual_factura_c_letter_overrides_default_a(self):
        text = """Archivo: sample.pdf
ORIGINAL
GIANNI FLORENCIA SOLEDAD
C
FACTURA
COD. 011
Punto de Venta: 00002 Comp. Nro: 00000014
Razón Social: GIANNI FLORENCIA SOLEDAD Fecha de Emisión: 07/01/2021
CUIT: 27351860303
Condición frente al IVA: Responsable Monotributo
CUIT: 30715444530 Apellido y Nombre / Razón Social: CS TECH CONSULTING S.A.
Código Producto / Servicio Cantidad U. Medida Precio Unit. % Bonif Imp. Bonif. Subtotal
1 Servicio de consultoria 30,00 unidades 700,00 0,00 0,00 21000,00
Subtotal: $ 21000,00
Importe Total: $ 21000,00
CAE N°: 71023172497119
Fecha de Vto. de CAE: 17/01/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["tipo_comprobante"], "Factura C")
        self.assertEqual(parsed["codigo_comprobante"], 11)
        self.assertEqual(parsed["numero_factura_completo"], "27351860303_011_00002_00000014")

    def test_factura_a_derives_iva_percentage(self):
        text = arca_text(
            "A", 1, "30707186722", "00002", "00000071", "71152560834540", "22/04/2021", "Servicio tecnico", 436621.12,
            "IVA 21,00%: 91690,43",
        )
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text))

        self.assertEqual(parsed["codigo_comprobante"], 1)
        self.assertEqual(parsed["iva_porcentaje"], 21.0)
        self.assertEqual(parsed["numero_factura_completo"], "30707186722_001_00002_00000071")

    def test_duplicate_and_triplicate_copies_do_not_duplicate_items(self):
        original = arca_text(
            "C", 11, "27959140850", "00001", "00000003", "72500527231207", "23/12/2022", "Consultoria de procesos", 200000.0
        )
        duplicate = original.replace("ORIGINAL", "DUPLICADO", 1).replace("72500527231207", "99999999999999").replace("200000.00", "999999.00")
        triplicate = original.replace("ORIGINAL", "TRIPLICADO", 1).replace("72500527231207", "88888888888888").replace("200000.00", "888888.00")
        text = "--- OCR PAGE 1 ---\n" + original + "\n--- OCR PAGE 2 ---\n" + duplicate + "\n--- OCR PAGE 3 ---\n" + triplicate
        parsed = parse_supported_document_ocr(text)

        self.assertEqual(len(parsed["items"]), 1)
        self.assertEqual(parsed["cae"], "72500527231207")
        self.assertEqual(parsed["total"], 200000.0)

    def test_arca_product_service_becomes_display_description(self):
        text = arca_text(
            "C", 11, "27229910871", "00001", "00000005", "71011847336594", "14/01/2021",
            "Servicio de Consultoria SAP", 6750.0,
        )
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["descripcion"], "Servicio de Consultoria SAP")

    def test_split_arca_product_service_description_is_recovered(self):
        text = """Archivo: sample 27229910871_011_00001_00000005.pdf
ORIGINAL
VILLODAS ANDREA DANIELA
FACTURA C COD. 011
Punto de Venta: 00001 Comp. Nro: 00000005
Fecha de Emision: 04/01/2021
CUIT: 27229910871
Razon Social: VILLODAS ANDREA DANIELA
Condicion frente al IVA: Responsable Monotributo
CUIT: 30715444530 Apellido y Nombre / Razon Social: CS TECH CONSULTING S.A.
Codigo Producto / Servicio Cantidad U. Medida Precio Unit. % Bonif Imp. Bonif. Subtotal
Servicio de Consultoria SAP
1,00 unidades 6750,00 0,00 0,00 6750,00
Subtotal: $ 6750,00
Importe Total: $ 6750,00
CAE Nro: 71011847336594
Fecha de Vto. de CAE: 14/01/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["items"][0]["descripcion"], "Servicio de Consultoria SAP")
        self.assertEqual(parsed["descripcion"], "Servicio de Consultoria SAP")

    def test_arca_description_does_not_require_complete_item_amounts(self):
        text = """Archivo: sample 20307186722_011_00002_00000071.pdf
ORIGINAL
BARRIO GUSTAVO ARIEL
C
FACTURA
COD. 011
Punto de Venta: 00002 Comp. Nro: 00000071
Fecha de Emisión: 12/04/2021
CUIT: 20307186722
Razón Social: BARRIO GUSTAVO ARIEL
Condición frente al IVA: Responsable Monotributo
CUIT: 30715444530 Apellido y Nombre / Razón Social: CS TECH CONSULTING S.A.
Código Producto / Servicio
Consultoria Abap Marzo
Subtotal: $ 51150,00
Importe Total: $ 51150,00
CAE N°: 71152560834540
Fecha de Vto. de CAE: 22/04/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["tipo_comprobante"], "Factura C")
        self.assertEqual(parsed["descripcion"], "Consultoria Abap Marzo")

    def test_arca_description_survives_missing_product_header(self):
        text = """Archivo: sample 20247883933_011_00001_00000075.pdf
ORIGINAL
SALAS DIEGO RODRIGO
C
FACTURA
COD. 011
Punto de Venta: 00001 Comp. Nro: 00000075
Fecha de Emisión: 08/04/2021
CUIT: 20247883933
Razón Social: SALAS DIEGO RODRIGO
Condición frente al IVA: Responsable Monotributo
CUIT: 30715444530 Apellido y Nombre / Razón Social: CS TECH CONSULTING S.A.
Condición frente al IVA: IVA Responsable Inscripto
Condición de venta: Otra
Consultoria SAP - 2021/04 11,00 unidades 9500,00 0,00 0,00 104500,00
Subtotal: $ 104500,00
Importe Total: $ 104500,00
CAE N°: 71143416520094
Fecha de Vto. de CAE: 18/04/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["tipo_comprobante"], "Factura C")
        self.assertEqual(parsed["descripcion"], "Consultoria SAP - 2021/04")

    def test_arca_description_removes_ocr_noise_before_service_text(self):
        text = """Archivo: sample 20247883933_011_00001_00000075.pdf
ORIGINAL
SALAS DIEGO RODRIGO
C
FACTURA
COD. 011
Punto de Venta: 00001 Comp. Nro: 00000075
Fecha de Emisión: 08/04/2021
CUIT: 20247883933
Razón Social: SALAS DIEGO RODRIGO
Condición frente al IVA: Responsable Monotributo
CUIT: 30715444530 Apellido y Nombre / Razón Social: CS TECH CONSULTING S.A.
Condición frente al IVA: IVA Responsable Inscripto
Condición de venta: Otra
eago [Presos co Ju coins [te] o | stn y Consultoria SAP - 2021/04 11,00 unidades 9500,00 0,00 0,00 104500,00
Subtotal: $ 104500,00
Importe Total: $ 104500,00
CAE N°: 71143416520094
Fecha de Vto. de CAE: 18/04/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["descripcion"], "Consultoria SAP - 2021/04")

    def test_arca_description_removes_noisy_quantity_after_service_text(self):
        text = """Archivo: sample 20341566542_011_00001_00000009.pdf
ORIGINAL
TORRES MIGUEL EZEQUIEL
C
FACTURA
COD. 011
Punto de Venta: 00001 Comp. Nro: 00000009
Fecha de Emisión: 07/04/2021
CUIT: 20341566542
Razón Social: TORRES MIGUEL EZEQUIEL
Condición frente al IVA: Responsable Monotributo
CUIT: 30715444530 Apellido y Nombre / Razón Social: CS TECH CONSULTING S.A.
Condición frente al IVA: IVA Responsable Inscripto
Condición de venta: Contado
eago [Poco serten co Ju Posen [o] o | stan; Servicios profesionales 1,00 tras y unidades
Subtotal: $ 100000,00
Importe Total: $ 100000,00
CAE N°: 71142380280051
Fecha de Vto. de CAE: 17/04/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["descripcion"], "Servicios profesionales")

    def test_arca_description_removes_tax_tail_after_service_text(self):
        text = """Archivo: sample 20328965322_001_00002_00000122.pdf
ORIGINAL
WEST TECH INFORMATICA
A
FACTURA
COD. 001
Punto de Venta: 00002 Comp. Nro: 00000122
Fecha de Emisión: 07/04/2021
CUIT: 20328965322
Razón Social: WEST TECH INFORMATICA
Condición frente al IVA: IVA Responsable Inscripto
CUIT: 30715444530 Apellido y Nombre / Razón Social: CS TECH CONSULTING S.A.
Condición frente al IVA: IVA Responsable Inscripto
Condición de venta: Otra
ite |rta sence | cani [meo] ro Juan] as [e; Mantenimiento de Pc, Impresoras y Redes; Otros Tributos; Per./Ret. de Impuesto a las Ganancias; Per./Ret. de IVA; Per./Ret. Ingresos Brutos
Subtotal: $ 360000,00
IVA 21%: $ 75600,00
Importe Total: $ 435600,00
CAE N°: 71142372132304
Fecha de Vto. de CAE: 17/04/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["descripcion"], "Mantenimiento de Pc, Impresoras y Redes")

    def test_arca_description_removes_noise_before_honorarios(self):
        text = """Archivo: sample 20235490499_011_00001_00000029.pdf
ORIGINAL
BONATO JUAN CARLOS
C
FACTURA
COD. 011
Punto de Venta: 00001 Comp. Nro: 00000029
Fecha de Emisión: 13/04/2021
CUIT: 20235490499
Razón Social: BONATO JUAN CARLOS
Condición frente al IVA: Responsable Monotributo
CUIT: 30715444530 Apellido y Nombre / Razón Social: CS TECH CONSULTING S.A.
Condición frente al IVA: IVA Responsable Inscripto
Condición de venta: Contado
eago [Ipresos co Ju coins [te] o | stn y Honorarios Profesionales
Subtotal: $ 16250,00
Importe Total: $ 16250,00
CAE N°: 71159645126859
Fecha de Vto. de CAE: 23/04/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["descripcion"], "Honorarios Profesionales")

    def test_arca_description_removes_noise_before_consultoria_abap(self):
        text = """Archivo: sample 20307186722_011_00002_00000071.pdf
ORIGINAL
BARRIO GUSTAVO ARIEL
C
FACTURA
COD. 011
Punto de Venta: 00002 Comp. Nro: 00000071
Fecha de Emisión: 12/04/2021
CUIT: 20307186722
Razón Social: BARRIO GUSTAVO ARIEL
Condición frente al IVA: Responsable Monotributo
CUIT: 30715444530 Apellido y Nombre / Razón Social: CS TECH CONSULTING S.A.
Condición frente al IVA: IVA Responsable Inscripto
Condición de venta: Otra
eago Ipresos co Ju coins te o stn y Consultoria Abap Marzo
Subtotal: $ 51150,00
Importe Total: $ 51150,00
CAE N°: 71152560834540
Fecha de Vto. de CAE: 22/04/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["descripcion"], "Consultoria Abap Marzo")

    def test_osde_reference_becomes_display_description(self):
        text = """Archivo: 01 Enero - Osde 0070-00125470.pdf
OSDE
Nota de debito: 0070-00125470
Codigo: 02
Fecha de emision: 18/01/2021
CUIT: 30-54674125-3
CUIL/CUIT: 30-71544453-0
Período Referencia Nro. documento Importe
02/2020 Interés pago fuera de término 244920129328 $ 16.142,52
Neto Gravado $ 16.142,52
IVA Inscripto 10,50% $ 1.694,96
Total $ 19.048,17
CAE: 71033484715213
FECHA DE VENCIMIENTO: 28.01.2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["items"][0]["descripcion"], "Interés pago fuera de término")
        self.assertEqual(parsed["descripcion"], "Interés pago fuera de término")

    def test_osde_invoice_description_and_cae_due_date(self):
        text = """Archivo: 04 Abril - Osde 0082-00118966.pdf
OSDE
Factura: 0082-00118966
Fecha de emisión: 26/04/2021
CUIT: 30-54674125-3
CS TECH CONSULTING SA
CUIL/CUIT: 30-71544453-0
Descripción Importe
Total valor Plan de Servicio $ 40.866,07
Neto Gravado $ 40.866,07
IVA Inscripto 10,50% $ 4.290,94
Percepción $ 3.064,95
Total $ 48.221,96
CAE: 71173264375701
FECHA DE VENCIMIENTO: 06.05.2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["items"][0]["descripcion"], "Total valor Plan de Servicio")
        self.assertEqual(parsed["descripcion"], "Total valor Plan de Servicio")
        self.assertEqual(parsed["fecha_vencimiento_cae"], "2021-05-06")

    def test_telecom_concepts_are_summarized(self):
        text = """Archivo: 04 Abril - CV 6723-01768328.pdf
Cablevisión Fibertel
FACTURA N°: 6723-01768328
FECHA: 18-04-2021
C.U.I.T.: 30639453738
CONCEPTOS IMPORTE
Cablevision Flow Box 05-2021 2385,12
Adicional Cablevision Flow Box 05-2021 235,54
Servicios de Television Subtotal 2620,66
Pack Futbol 05-2021 686,78
Packs Premium Subtotal 686,78
Fibertel 100 Megas Wifi 05-2021 3244,63
Servicios Banda Ancha (SBA) Subtotal 3244,63
Promo Debito Automatico 6M -247,93
Otros Subtotal -247,93
Neto Gravado 3658,15
I.V.A. 21% 768,22
TOTAL 4682,45
CAE Nro: 71168883477372
"""
        concept_items = extract_arca_concept_items(text)
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)
        description = build_display_description({"items": [], "tipo_comprobante": "Factura A"}, text)

        self.assertGreaterEqual(len(concept_items), 5)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["descripcion"], description)
        self.assertEqual(
            description,
            "Servicios de televisión, packs premium, internet 100 megas y descuentos correspondientes al período 05-2021.",
        )

    def test_fibertel_visual_layout_is_parsed(self):
        text = """Archivo: 04 Abril - CV 8340-04053647.pdf
Cablevisión Fibertel
Telecom Argentina S.A.
La Pampa 2295 P.B
IVA Responsable Inscripto
A Codigo N° 01
FACTURA N°: 8340-04053647
FECHA: 19-04-2020
C.U.I.T.:30639453738
SR/A: TECH CONSULTING SA CS
CUITN°: 30-71544453-0
CONCEPTOS IMPORTE
Cablevisión Flow Box 05-2020 1987,61
Adicional Cablevision Flow Box 05-2020 196,70
Servicios de Television Subtotal 2184,31
Fibertel 100 Megas Wifi 05-2020 2704,14
Servicios Banda Ancha (SBA) Subtotal 2704,14
Promoción Combo Mes 6 de 12 12MX47% -2205,12
Promocion COMBO Subtotal -2205,12
Neto Gravado Subtotal 2683,33
I.V.A. 21% 563,50
PERCEP. IIBB BS. AS. 107,34
Percep. IVA-RG2408 80,50
TOTAL $3434,67
CAE Nro.: 70168206803905
Fecha Vto.: 29-04-2020
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["codigo_comprobante"], 1)
        self.assertEqual(parsed["numero_factura"], "08340-04053647")
        self.assertEqual(parsed["total"], 3434.67)
        self.assertEqual(parsed["cae"], "70168206803905")
        self.assertEqual(parsed["iva_porcentaje"], 21.0)
        self.assertEqual(
            parsed["descripcion"],
            "Servicios de televisión, internet 100 megas y descuentos correspondientes al período 05-2020.",
        )

    def test_external_product_precedes_reference(self):
        parsed = {
            "document_type": "external_provider_receipt",
            "items": [{"description": "Linux Hosting con cPanel Inicial - Renovación", "reference": "deeptics.com.ar"}],
        }

        self.assertEqual(
            build_display_description(parsed),
            "Linux Hosting con cPanel Inicial - Renovación",
        )

    def test_godaddy_keeps_wrapped_product_line(self):
        text = """Archivo: 01 Enero - GoDaddy 1805647812.pdf
Recibo
N° 1805647812
FECHA:
1/1/2021
NÚMERO DE CLIENTE: 203521924
FACTURAR A:
CS TECH CONSULTING
PAGO:
Plazo Producto Cantidad
1 mes Linux Hosting con cPanel Inicial - Renovación $ 1.199,99
deeptics.com.ar
Total (ARS) $ 1.199,99
Saldo adeudado (ARS) $ 0,00
"""
        parsed = parse_supported_document_ocr(text)

        self.assertIn("deeptics.com.ar", parsed["items"][0]["description"])
        self.assertIn("deeptics.com.ar", build_display_description(parsed))


if __name__ == "__main__":
    unittest.main()
