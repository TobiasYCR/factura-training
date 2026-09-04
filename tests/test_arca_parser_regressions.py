import unittest
from unittest.mock import patch

from api import add_arca_integration_fields, calculate_field_confidence, extract_document, extract_upload_text
from infer import (
    assess_document_quality,
    build_display_description,
    extract_arca_concept_items,
    parse_supported_document_ocr,
    validate_extracted_document_json,
)
from ocr import OcrUnavailableError


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

    def test_factura_b_extracts_code_number_and_total_without_iva_warning(self):
        text = arca_text(
            "B",
            6,
            "30715999999",
            "00003",
            "00000042",
            "71111111111111",
            "30/04/2021",
            "Servicio mensual de prueba",
            1210.0,
        )
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["tipo_comprobante"], "Factura B")
        self.assertEqual(parsed["codigo_comprobante"], 6)
        self.assertEqual(parsed["numero_factura"], "00003-00000042")
        self.assertEqual(parsed["total"], 1210.0)
        self.assertEqual(parsed["iva_total"], 0.0)
        self.assertEqual(parsed["iva_porcentaje"], 0)
        self.assertEqual(parsed["descripcion"], "Servicio mensual de prueba")
        self.assertEqual(validate_extracted_document_json(parsed), [])
        self.assertNotIn("Factura A sin IVA discriminado.", assess_document_quality(parsed, text))

    def test_factura_a_with_explicit_zero_iva_does_not_require_review(self):
        text = arca_text(
            "A",
            1,
            "27351860303",
            "00002",
            "00000016",
            "71183527410761",
            "21/05/2021",
            "Honorarios Abr21",
            67000.0,
            "Alicuota IVA 0% IVA 0,00",
        )

        result = extract_document(text, filename="factura-a-iva-0.pdf")

        self.assertTrue(result["ok"])
        self.assertFalse(result["requires_review"])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["data"]["iva_total"], 0.0)
        self.assertEqual(result["data"]["iva_porcentaje"], 0)

    def test_document_code_text_overrides_mismatched_filename_code(self):
        text = """Archivo: 05 Mayo - FG 27351860303_001_00002_00000016.pdf
ORIGINAL
GIANNI FLORENCIA SOLEDAD
C
FACTURA
COD. 011
Punto de Venta: 00002 Comp. Nro: 00000016
Fecha de Emision: 01/05/2021
CUIT: 27351860303
Condicion frente al IVA: Responsable Monotributo
CUIT: 30715444530
Apellido y Nombre / Razon Social: CS TECH CONSULTING S.A.
Codigo Producto / Servicio Cantidad Precio Unit. Subtotal
01 Honorarios Abr21 1,00 67000,00 67000,00
Subtotal: $ 67000,00
Importe Total: $ 67000,00
CAE Nro: 71183527410761
Fecha de Vto. de CAE: 21/05/2021
"""

        result = extract_document(text, filename="05 Mayo - FG 27351860303_001_00002_00000016.pdf")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["tipo_comprobante"], "Factura C")
        self.assertEqual(result["data"]["codigo_comprobante"], 11)
        self.assertEqual(result["data"]["numero_factura_completo"], "27351860303_011_00002_00000016")
        self.assertFalse(result["requires_review"])
        self.assertEqual(result["warnings"], [])

    def test_api_returns_field_confidence_and_review_flag(self):
        text = arca_text(
            "B",
            6,
            "30715999999",
            "00003",
            "00000042",
            "71111111111111",
            "30/04/2021",
            "Servicio mensual de prueba",
            1210.0,
        )

        result = extract_document(text, filename="factura-b.pdf")

        self.assertTrue(result["ok"])
        self.assertFalse(result["requires_review"])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["field_confidence"]["numero_factura"], 1.0)
        self.assertEqual(result["field_confidence"]["total"], 1.0)
        self.assertEqual(result["field_confidence"]["description"], 1.0)
        self.assertNotIn("fecha_vencimiento_pago", result["field_confidence"])

    def test_factura_c_confidence_does_not_flag_expected_empty_iva_or_items(self):
        data = {
            "tipo_comprobante": "Factura C",
            "codigo_comprobante": 11,
            "numero_factura": "00001-00000059",
            "fecha_emision": "2021-04-30",
            "fecha_vencimiento_cae": "2021-05-06",
            "emisor": {"nombre": "MARI DALINA ADRIANA", "cuit": "27-26493367-1"},
            "receptor": {"nombre": "CS TECH CONSULTING S.A.", "cuit": "30-71544453-0"},
            "subtotal": 27000.0,
            "iva_total": None,
            "tributos_total": 0.0,
            "total": 27000.0,
            "cae": "71181484605002",
            "items": [],
            "descripcion": "Servicios de consultoria Abril 2021",
        }

        field_confidence = calculate_field_confidence(data)

        self.assertNotIn("iva_total", field_confidence)
        self.assertNotIn("items", field_confidence)
        self.assertEqual(field_confidence["description"], 1.0)

    def test_forced_pdf_ocr_can_prefer_better_embedded_text(self):
        embedded_text = arca_text(
            "B",
            6,
            "30715999999",
            "00003",
            "00000042",
            "71111111111111",
            "30/04/2021",
            "Servicio mensual de prueba",
            1210.0,
        )
        with patch("api.ocr_pdf_bytes", return_value=("ruido sin datos fiscales", {"engine": "tesseract"})):
            with patch("api.extract_embedded_pdf_text", return_value=(embedded_text, {"method": "embedded_text"})):
                text, meta = extract_upload_text(b"pdf", "factura-b.pdf", force_ocr=True)

        self.assertEqual(text, embedded_text)
        self.assertEqual(meta["selected_text"], "embedded_text")
        self.assertTrue(meta["forced_ocr_requested"])

    def test_forced_pdf_ocr_falls_back_to_embedded_text_when_ocr_unavailable(self):
        embedded_text = arca_text(
            "B",
            6,
            "30715999999",
            "00003",
            "00000042",
            "71111111111111",
            "30/04/2021",
            "Servicio mensual de prueba",
            1210.0,
        )
        with patch("api.ocr_pdf_bytes", side_effect=OcrUnavailableError("sin tesseract")):
            with patch("api.extract_embedded_pdf_text", return_value=(embedded_text, {"method": "embedded_text"})):
                text, meta = extract_upload_text(b"pdf", "factura-b.pdf", force_ocr=True)

        self.assertEqual(text, embedded_text)
        self.assertEqual(meta["selected_text"], "embedded_text")
        self.assertEqual(meta["ocr_error"], "sin tesseract")

    def test_factura_b_labeled_item_rows_build_description(self):
        text = """FACTURA B
Cod. 006
Punto de Venta: 00018 Comp. Nro: 00000001
Fecha de Emision: 09/11/2026
Consultora Andina SRL
CUIT: 20-85521056-1
IVA Responsable Inscripto
Cliente: Hotel Plaza SRL
CUIT Cliente: 20-93267849-7
Condicion IVA: Consumidor Final
Moneda: PES
Tipo Cambio: 1
Item: Consultoria operativa Cant 3 P.Unit 60.157,08 Importe 180.471,24
Item: Capacitacion Cant 10 P.Unit 28.270,02 Importe 282.700,20
Subtotal: $ 463.171,44
Importe Total: $ 463.171,44
CAE: 10365667935643
Vto. CAE: 19/11/2026
"""

        result = extract_document(text, filename="factura-b.pdf")

        self.assertTrue(result["ok"])
        self.assertFalse(result["requires_review"])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(
            result["data"]["descripcion"],
            "Consultoria operativa y Capacitacion",
        )

    def test_mipyme_fce_extracts_payment_due_date_and_vat_totals(self):
        text = """FACTURA DE CRÉDITO ELECTRÓNICA MiPyMEs (FCE)
A CÓD. 201
Punto de Venta: 00002 Comp. Nro: 00000192
Fecha de Emisión: 14/01/2026
Razón Social: CS TECH CONSULTING S.A.
Domicilio Comercial: Maestra Rocha Montarce 1150 - El Palomar, Buenos Aires
CUIT: 30715444530
Ingresos Brutos: 902-3071544530
Fecha de Inicio de Actividades: 01/11/2016
Condición frente al IVA: IVA Responsable Inscripto
Fecha de Vto. para el pago: 28/01/2026 Período Facturado Desde: 01/12/2025 Hasta: 31/12/2025
CBU del Emisor: 0150523802000105524151 Alias CBU: CS-DEEPTICS-ARS
CUIT: 33699685459 Apellido y Nombre / Razón Social: CORREO ANDREANI SA
Condición frente al IVA: IVA Responsable Inscripto Domicilio Comercial: Vieytes 1228 - Capital Federal, Ciudad de Buenos Aires
Opción de Transferencia: Agente de Depósito Colectivo
Código Producto / Servicio Cantidad U. medida Precio Unit. % Bonif Subtotal Alícuota IVA Subtotal c/IVA
01 Analista de procesos Sr. - Dic 2025 160,00 unidades 33275,00 0,00 5324000,00 21% 6442040,00
Importe Otros Tributos: $ 0,00
Importe Neto Gravado: $ 5324000,00
IVA 27%: $ 0,00
IVA 21%: $ 1118040,00
IVA 10.5%: $ 0,00
IVA 5%: $ 0,00
IVA 2.5%: $ 0,00
IVA 0%: $ 0,00
Importe Otros Tributos: $ 0,00
Importe Total: $ 6442040,00
CAE Nro: 76012345678901
Fecha de Vto. de CAE: 24/01/2026
"""

        result = extract_document(text, filename="mipyme-fce.jpg")

        self.assertTrue(result["ok"])
        self.assertFalse(result["requires_review"])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["data"]["tipo_comprobante"], "Factura A")
        self.assertEqual(result["data"]["codigo_comprobante"], 201)
        self.assertEqual(result["data"]["numero_factura"], "00002-00000192")
        self.assertEqual(result["data"]["fecha_vencimiento_pago"], "2026-01-28")
        self.assertEqual(result["data"]["fecha_vencimiento"], "2026-01-28")
        self.assertEqual(result["data"]["fecha_vencimiento_cae"], "2026-01-24")
        self.assertEqual(result["data"]["subtotal"], 5324000.0)
        self.assertEqual(result["data"]["iva_total"], 1118040.0)
        self.assertEqual(result["data"]["total"], 6442040.0)
        self.assertEqual(result["data"]["iva"][0]["descripcion"], "21%")
        self.assertEqual(result["data"]["items"][0]["descripcion"], "Analista de procesos Sr. - Dic 2025")
        self.assertEqual(result["data"]["descripcion"], "Analista de procesos Sr. - Dic 2025")

    def test_fce_credit_note_code_builds_note_type(self):
        text = """NOTA DE CRÉDITO ELECTRÓNICA MiPyMEs (FCE)
A CÓD. 203
Punto de Venta: 00002 Comp. Nro: 00000193
Fecha de Emisión: 15/01/2026
Razón Social: CS TECH CONSULTING S.A.
CUIT: 30715444530
Condición frente al IVA: IVA Responsable Inscripto
Fecha de Vto. para el pago: 28/01/2026
CUIT: 33699685459 Apellido y Nombre / Razón Social: CORREO ANDREANI SA
01 Ajuste de servicio 1,00 unidades 1000,00 0,00 1000,00 21% 1210,00
Importe Neto Gravado: $ 1000,00
IVA 21%: $ 210,00
Importe Total: $ 1210,00
CAE Nro: 76012345678902
Fecha de Vto. de CAE: 24/01/2026
"""

        result = extract_document(text, filename="mipyme-fce-nc.jpg")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["tipo_comprobante"], "Nota de Credito A")
        self.assertEqual(result["data"]["codigo_comprobante"], 203)

    def test_payment_due_date_is_separate_from_cae_due_date_and_name_is_cleaned(self):
        text = """ORIGINAL
C
FACTURA
GONZALEZ THIAGO J AVIER
COD. 011
Punto de Venta: 00001 Comp. Nro: 00000013
Razón Social: GONZALEZ THIAGO J AVIER Fecha de Emisión: 01/09/2026
Domicilio Comercial: Miranda 343 - Monte Grande, Buenos Aires CUIT: 20468137985
Ingresos Brutos: 20468137985
Condición frente al IVA: Responsable Monotributo Fecha de Inicio de Actividades: 01/10/2025
Período Facturado Desde: 03/08/2026 Hasta:31/08/2026 Fecha de Vto. para el pago:15/09/2026
CUIT: 30715444530 Apellido y Nombre / Razón Social:CS TECH CONSULTING S.A.
Condición frente al IVA: IVA Responsable Inscripto Domicilio:Maestra Rocha Montarce 1150 - El Palomar, Buenos Aires
Condición de venta: Transferencia Bancaria
Código Producto / Servicio Cantidad U. Medida Precio Unit. % Bonif Imp. Bonif. Subtotal
01 Servicios consultorias - agosto 180,00 unidades 6000,00 0,00 0,00 1080000,00
Subtotal: $ 1080000,00
Importe Otros Tributos: $ 0,00
Importe Total: $ 1080000,00
Pág. 1/1 CAE Nº: 86350888636910
Fecha de Vto. de CAE: 11/09/2026
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["emisor"]["nombre"], "GONZALEZ THIAGO JAVIER")
        self.assertEqual(parsed["fecha_vencimiento_pago"], "2026-09-15")
        self.assertEqual(parsed["fecha_vencimiento"], "2026-09-15")
        self.assertEqual(parsed["fecha_vencimiento_cae"], "2026-09-11")
        self.assertEqual(validate_extracted_document_json(parsed), [])

    def test_monthly_consulting_description_recovers_service_prefix_and_year(self):
        text = """ORIGINAL
C
FACTURA
MARI DALINA ADRIANA
COD. 011
Punto de Venta: 00001 Comp. Nro: 00000059
Razón Social: MARI DALINA ADRIANA Fecha de Emisión: 30/04/2021
Domicilio Comercial: San Luis 3251 Piso:7 Dpto:A - Ciudad de CUIT: 27264933671
Condición frente al IVA: Responsable Monotributo Fecha de Inicio de Actividades: 01/10/2016
Período Facturado Desde: 01/04/2021 Hasta:30/04/2021 Fecha de Vto. para el pago:06/05/2021
CUIT: 30715444530 Apellido y Nombre / Razón Social:CS TECH CONSULTING S.A.
Códígo Producto / Servicio Cantidad U. Medida Precio Unit. % Bonif Imp. Bonif. Subtotal
1 consultoría Abril 1,00 unidades 27000,00 0,00 0,00 27000,00
Subtotal: $ 27000,00
Importe Total: $ 27000,00
CAE N°: 71181484605002
Fecha de Vto. de CAE: 10/05/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["items"][0]["descripcion"], "Servicios de consultoría Abril 2021")
        self.assertEqual(parsed["descripcion"], "Servicios de consultoría Abril 2021")

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

    def test_api_final_description_cleanup_covers_existing_dirty_value(self):
        parsed = {
            "tipo_comprobante": "Factura C",
            "codigo_comprobante": 11,
            "punto_venta": "00002",
            "numero_comprobante": "00000071",
            "numero_factura": "00002-00000071",
            "emisor": {"cuit": "20-30718672-2", "doc_nro": "20307186722"},
            "iva": [],
            "items": [],
            "descripcion": "eago [presos co Ju coins [te] o | stn y Consultoria Abap Marzo",
        }

        enriched = add_arca_integration_fields(parsed, "")

        self.assertEqual(enriched["descripcion"], "Consultoria Abap Marzo")

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

    def test_godaddy_removes_ocr_noise_before_product_name(self):
        text = """Archivo: 04 Abril - GoDaddy 1849643623.pdf
Recibo
N° 1849643623
FECHA:
05/04/2021
NÚMERO DE CLIENTE: 203521924
FACTURAR A:
Javier Nogues
CS TECH CONSULTING
PAGO:
Visa terminada en 7953 $ 4.799,88
Plazo Producto Cantidad
laño Correo Plus de Microsoft 365 de GoDaddy $ 4.799,88
Total (ARS) $ 4.799,88
Pago recibido $ 4.799,88
Saldo adeudado (ARS) $ 0,00
"""
        parsed = parse_supported_document_ocr(text)

        self.assertEqual(parsed["items"][0]["description"], "Correo Plus de Microsoft 365 de GoDaddy")
        self.assertEqual(build_display_description(parsed), "Correo Plus de Microsoft 365 de GoDaddy")

    def test_compact_industrial_usd_invoice_is_parsed(self):
        text = """ORIGINAL
FECHADEEMISIÓN:22.12.2025
FACTURA 0013-00000022
C.U.I.T.N°: 30-56976625-3
A
Codigo001 INICIOACTIVIDADES: 20.11.1978
SEÑOR(ES): NININABAKERYS.A.
IVARESPONSABLEINSCRIPTO CUITN° 30-71369120-4 IVAResponsableInscripto
ITEM CANT. DESCRIPCION PRECIO TOTAL
000100 3,000 EV910853-CG53-1012/C 95,01 95,01
000200 1,000 EV910031-CB20-124E1/C 287,57 287,57
SUBTOTALUSD 382,58
IVA21.0% 80,34
TOTALFACTURA USD462,92
CAE: 75513850400277
Vencimiento: 01.01.2026
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["tipo_comprobante"], "Factura A")
        self.assertEqual(parsed["numero_factura"], "00013-00000022")
        self.assertEqual(parsed["moneda"], "DOL")
        self.assertIsNone(parsed["emisor"]["nombre"])
        self.assertEqual(parsed["subtotal"], 382.58)
        self.assertEqual(parsed["total"], 462.92)
        self.assertEqual(parsed["cae"], "75513850400277")
        self.assertEqual(parsed["fecha_vencimiento_cae"], "2026-01-01")
        self.assertEqual(len(parsed["items"]), 2)
        self.assertEqual(parsed["descripcion"], "EV910853-CG53-1012/C y EV910031-CB20-124E1/C")

    def test_cianbox_detail_table_items_are_parsed(self):
        text = """A Factura
Nº 0003-00019315
Cod. 01
Fecha: 30/03/2021
C.U.I.T. 30-71450816-0
de Digital Store Tec SRL
Responsable Inscripto Inicio de Actividades: Junio de 2014
Sr./es CS TECH CONSULTING S.A Tel.: **
I.V.A. Responsable Inscripto CUIT 30-71544453-0
Cantidad Detalle Alicuota P.Unit. S.Total
1,00 Costo MercadoEnvios 21,00% 363,6281 363,63
1,00 Handy Baofeng BF-T3 Azul (par) 10,50% 1.447,0588 1.447,06
ORIGINAL :: pag. 1/1 No Gravado $ 0,00
Gravado $ 1.810,69
I.V.A. 21,00% $ 76,36
I.V.A. 10,50% $ 151,94
Total $ 2.038,99
C.A.E. Nº 71130959925098 Fecha Vto. C.A.E.: 09/04/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["numero_factura"], "00003-00019315")
        self.assertEqual(parsed["emisor"]["nombre"], "Digital Store Tec SRL")
        self.assertEqual(parsed["emisor"]["cuit"], "30-71450816-0")
        self.assertEqual(parsed["receptor"]["cuit"], "30-71544453-0")
        self.assertEqual(len(parsed["items"]), 2)
        self.assertEqual(parsed["descripcion"], "Costo MercadoEnvios y Handy Baofeng BF-T3 Azul (par)")

    def test_telecom_description_works_without_concept_header(self):
        text = """Archivo: 01 Enero - CV 8340-03607681.pdf
8340-03607681 Total Factura 3725,64
Telecom Argentina S.A.
La Pampa 2295 P.B A FECHA: 19-01-2020
C.U.I.T.:30639453738
Codigo N° 01
Cablevisión Flow Box 02-2020 1807,44
Adicional Cablevisión Flow Box 02-2020 179,34
Servicios de Television Subtotal 1986,78
Pack Fútbol 02-2020 549,59
Packs Premium Subtotal 549,59
Fibertel 100 Megas Wifi 02-2020 2458,68
Servicios Banda Ancha (SBA) Subtotal 2379,37
Promoción Combo Mes 3 de 12 12MX47% -2005,08
Promocion COMBO Subtotal -2005,08
Neto Gravado Subtotal 2910,66
I.V.A. 21% 611,24
TOTAL $3725,64
CAE Nro.: 70038951261058
Fecha Vto.: 29-01-2020
"""
        description = build_display_description({"items": [], "tipo_comprobante": "Factura A"}, text)

        self.assertEqual(
            description,
            "Servicios de televisión, packs premium, internet 100 megas y descuentos correspondientes al período 02-2020.",
        )

    def test_fibertel_ignores_phone_ad_in_concepts_and_completes_buyer(self):
        text = """Archivo: 03 Marzo - CV 6723-01613211.pdf
Cablevisión Fibertel
Telecom Argentina S.A. FACTURA N°: 6723-01613211
FECHA: 19-03-2021
A C.U.I.T.:30639453738
SR./A: TECH CONSULTING SA CS FORMA DE PAGO: Debito Automatico
PERIODO: 04-2021
CUIT No: 30-71544453-0
CONCEPTOS IMPORTE
Cablevisión Flow Box 04-2021 2.385,12
Adicional Cablevisión Flow Box 04-2021 235,54
Servicios de Television Subtotal 2620,66
Pack Futbol 04-2021 686,78
Packs Premium Subtotal 686,78
Fibertel 100 Megas Wifi 04-2021 3.244,63
Servicios Banda Ancha (SBA) Subtotal 3244,63
Promo Débito Automático 6M 04-2021 -247,93
0800 199 7771
Promoción Combo Mes 3 de 6 6MX47% -2.645,99
Promocion COMBO Subtotal -2645,99
Neto Gravado Subtotal 3.658,15
$4.682,45
CAE Nro: 71128471020836
Fecha Vto: 29/03/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["receptor"]["nombre"], "CS TECH CONSULTING SA")
        self.assertFalse(any(item["descripcion"].startswith("0800") for item in parsed["items"]))
        self.assertEqual(parsed["total"], 4682.45)
        self.assertEqual(parsed["iva_porcentaje"], 21.0)
        self.assertEqual(
            parsed["descripcion"],
            "Servicios de televisión, packs premium, internet 100 megas y descuentos correspondientes al período 04-2021.",
        )

    def test_fibertel_infers_tax_totals_when_ocr_omits_fiscal_lines(self):
        text = """Archivo: 03 Marzo - CV 6723-01613211.pdf
Cablevisión Fibertel
Telecom Argentina S.A. FACTURA N°: 6723-01613211
FECHA: 19-03-2021
A C.U.I.T.:30639453738
SR./A: TECH CONSULTING SA CS FORMA DE PAGO: Debito Automatico
PERIODO: 04-2021
CUIT No: 30-71544453-0
CONCEPTOS IMPORTE
Cablevisión Flow Box 04-2021 2.385,12
Adicional Cablevisión Flow Box 04-2021 235,54
Servicios de Television Subtotal 2620,66
Pack Futbol 04-2021 686,78
Packs Premium Subtotal 686,78
Fibertel 100 Megas Wifi 04-2021 3.244,63
Servicios Banda Ancha (SBA) Subtotal 3244,63
Promo Débito Automático 6M 04-2021 -247,93
0800 199 7771
Promoción Combo Mes 3 de 6 6MX47% -2.645,99
Promocion COMBO Subtotal -2645,99
Neto Gravado Subtotal 3.658,15
Total a pagar $4.682,45
CAE Nro: 71128471020836
Fecha Vto: 07/04/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["subtotal"], 3658.15)
        self.assertEqual(parsed["iva_total"], 768.21)
        self.assertEqual(parsed["tributos_total"], 256.09)
        self.assertEqual(parsed["impuestos"], 1024.3)
        self.assertEqual(parsed["total"], 4682.45)
        self.assertEqual(parsed["iva_porcentaje"], 21.0)

    def test_fibertel_recovers_missing_subtotal_and_partial_taxes_from_total(self):
        text = """Archivo: 04 Abril - CV 6723-01768328.pdf
Cablevisión Fibertel
Telecom Argentina S.A. FACTURA N°: 6723-01768328
FECHA: 18-04-2021
A C.U.I.T.:30639453738
SR./A: CS TECH CONSULTING SA
CUIT No: 30-71544453-0
PERIODO: 05-2021
CONCEPTOS IMPORTE
Cablevisión Flow Box 05-2021
Adicional Cablevisión Flow Box
Pack Futbol
Fibertel 100 Megas Wifi
Percep. IVA-RG2408 109,75
Total a pagar $4.682,45
CAE Nro: 71168888473772
Fecha Vto: 07/05/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["numero_factura"], "06723-01768328")
        self.assertEqual(parsed["subtotal"], 3658.16)
        self.assertEqual(parsed["iva_total"], 768.21)
        self.assertEqual(parsed["tributos_total"], 256.08)
        self.assertEqual(parsed["impuestos"], 1024.29)
        self.assertEqual(parsed["total"], 4682.45)
        self.assertEqual(parsed["iva_porcentaje"], 21.0)

    def test_osde_personalized_invoice_is_parsed(self):
        text = """Archivo: 04 Abril - Osde 0082-00118966.pdf
OSDE
A Codigo: 01
Factura: 0082-00118966
Fecha de emisión: 26/04/2021
CUIT: 30-54674125-3
CS TECH CONSULTING SA
CUIL/CUIT:30-71544453-0
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

        self.assertEqual(parsed["numero_factura"], "00082-00118966")
        self.assertEqual(parsed["emisor"]["nombre"], "OSDE")
        self.assertEqual(parsed["emisor"]["cuit"], "30-54674125-3")
        self.assertEqual(parsed["receptor"]["cuit"], "30-71544453-0")
        self.assertEqual(parsed["subtotal"], 40866.07)
        self.assertEqual(parsed["iva_total"], 4290.94)
        self.assertEqual(parsed["tributos_total"], 3064.95)
        self.assertEqual(parsed["total"], 48221.96)
        self.assertEqual(parsed["cae"], "71173264375701")
        self.assertEqual(parsed["fecha_vencimiento_cae"], "2021-05-06")
        self.assertEqual(parsed["descripcion"], "Total valor Plan de Servicio")

    def test_cetrogar_personalized_invoice_extracts_parties_and_items(self):
        text = """Archivo: 05 Mayo - TV 30592845748_001_00427_00030715.pdf
A
FACTURA COD. 001
Razón Social: CETROGAR S.A Nro. Factura: 0427-00030715
Domicilio: Fecha de Emisión: 15/05/2021 18:45
CUIT: 30-59284574-8
IVA RESPONSABLE INSCRIPTO: 30-59284574-8
Vendido a: Método de envío
CS Tech Consulting SA Silvia Renee Mateo
Documento: 30715444530 Documento: 30715444530
Productos SKU Precio Cantidad Imp. Int IVA Total
Smart Tv 43" LG 43LM6350PSB FHD TV2767 $35.536,36 1 $0,00 $7.462,64 $42.999,00
Envio a Domicilio FL0005 $742,98 1 $0,00 $156,02 $899,00
Subtotal: $36.279,34
IVA: $7.618,66
Total: $43.898,00
CAE Nº: 71208270852377
Fecha de Vto. de CAE: 2021-05-25
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["numero_factura"], "00427-00030715")
        self.assertEqual(parsed["emisor"]["nombre"], "CETROGAR S.A")
        self.assertEqual(parsed["emisor"]["cuit"], "30-59284574-8")
        self.assertEqual(parsed["receptor"]["cuit"], "30-71544453-0")
        self.assertEqual(parsed["subtotal"], 36279.34)
        self.assertEqual(parsed["iva_total"], 7618.66)
        self.assertEqual(parsed["total"], 43898.0)
        self.assertEqual(parsed["cae"], "71208270852377")
        self.assertEqual(len(parsed["items"]), 2)
        self.assertEqual(parsed["descripcion"], 'Smart Tv 43" LG 43LM6350PSB FHD y Envio a Domicilio')

    def test_cetrogar_personalized_invoice_defaults_buyer_name_from_known_cuit(self):
        text = """Archivo: 05 Mayo - TV 30592845748_001_00427_00030715.pdf
CETROGAR FACTURA
Nro. Factura: 0427-00030715
Fecha de Emisión: 15/05/2021 18:45
CUIT: 30-59284574-8
Documento: 30715444530
Subtotal: $36.279,34
IVA: $7.618,66
Total: $43.898,00
CAE Nº: 71208270852377
Fecha de Vto. de CAE: 2021-05-25
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["receptor"]["nombre"], "CS Tech Consulting SA")
        self.assertEqual(parsed["receptor"]["cuit"], "30-71544453-0")

    def test_hidroal_homecenter_invoice_is_parsed(self):
        text = """Archivo: 05 Mayo - Mesitas 0033-00016521.pdf
HIDROAL HOME CENTER
A Cod.01
Factura
N° 0033-00016521
Fecha: 14/05/2021
Razón Social: HIDROAL SA
CUIT: 30628724497
Nombre: CS TECH CONSULTING S.A.
IVA: Responsable Inscripto
CUIT: 30715444530
Condición de venta: Contado
Cantidad Detalle % IVA % Impuesto Interno Precio Unitario Descuento Importe
1,00 MESA LUZ CENTRO ESTANT MLBW BOTINERO WENGUE 21,00 0,00 3.069,45 0,00 $ 3.069,45
Garantía: 6 meses
Gravado: $ 3.069,45
Importe Iva: $ 644,58
Percepción Buenos Aires 4.00% $ 122,78
Total: $ 3.836,81
CAE: 71207195310091 - Vencimiento: 24/05/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["numero_factura"], "00033-00016521")
        self.assertEqual(parsed["emisor"]["nombre"], "HIDROAL SA")
        self.assertEqual(parsed["emisor"]["cuit"], "30-62872449-7")
        self.assertEqual(parsed["receptor"]["cuit"], "30-71544453-0")
        self.assertEqual(parsed["subtotal"], 3069.45)
        self.assertEqual(parsed["iva_total"], 644.58)
        self.assertEqual(parsed["tributos_total"], 122.78)
        self.assertEqual(parsed["total"], 3836.81)
        self.assertEqual(parsed["cae"], "71207195310091")
        self.assertEqual(parsed["fecha_vencimiento_cae"], "2021-05-24")
        self.assertEqual(parsed["descripcion"], "MESA LUZ CENTRO ESTANT MLBW BOTINERO WENGUE")

    def test_hidroal_homecenter_invoice_recovers_split_ocr_item(self):
        text = """Archivo: 05 Mayo - Mesitas 0033-00016521.pdf
HIDROAL HOME CENTER
A Cod.01
Factura N° 0033-00016521
Fecha: 14/05/2021
Razón Social: HIDROAL SA
CUIT: 30628724497
Nombre: CS TECH CONSULTING S.A.
IVA: Responsable Inscripto
CUIT: 30715444530
Cantidad Detalle % IVA % Impuesto
Interno Unitario
MESA LUZ CENTRO ESTANT MLBW BOTINERO WENGUE
Garantía: 6 meses
Gravado: $ 3.069,45
Importe Iva: $ 644,58
Percepción Buenos Aires 4.00% $ 122,78
Total: $ 3.836,81
CAE: 71207195310091 - Vencimiento 24/05/2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(len(parsed["items"]), 1)
        self.assertEqual(parsed["items"][0]["descripcion"], "MESA LUZ CENTRO ESTANT MLBW BOTINERO WENGUE")
        self.assertEqual(parsed["fecha_vencimiento_cae"], "2021-05-24")
        self.assertEqual(parsed["descripcion"], "MESA LUZ CENTRO ESTANT MLBW BOTINERO WENGUE")

    def test_hidroal_homecenter_invoice_handles_real_tesseract_header(self):
        text = """Archivo: 05 Mayo - Mesitas 0033-00016521.pdf
Factura
D HIDROAL A N° 0033-00016521
HOMECENTER Fecha: 14/05/2021
Cod.01
Razon Social: HIDROAL SA
Dirección: Venezuela 957, Caba CUIT: 30628724497
Nombre: CS TECH CONSULTING S.A. IVA: Responsable Inscripto
Localidad: El Palomar CUIT: 30715444530
Cantidad Detalle % IVA % Impuesto Pr eco Descuento Importe
Interno Unitario
1,00 MESA LUZ CENTRO ESTANT MLBW BOTINERO WENGUE 21,00 0,00 3.069,45 0,00 $ 3.069,45
Garantía: 6 meses
Gravado: $ 3.069,45 Exento: $ 0,00
Importe Iva: $ 644,58 Gravado:  $3.069,45
Percepción Buenos Aires 4.00% $ 122,78
Total: $ 3.836,81
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["numero_factura"], "00033-00016521")
        self.assertEqual(len(parsed["items"]), 1)
        self.assertEqual(parsed["descripcion"], "MESA LUZ CENTRO ESTANT MLBW BOTINERO WENGUE")
        self.assertEqual(parsed["fecha_vencimiento_cae"], "2021-05-24")

    def test_osde_invoice_keeps_provider_cuit_when_first_cuit_is_buyer(self):
        text = """Archivo: 02 Febrero - Osde 0078-00111613.pdf
OSDE Factura: 0078-00111613
Fecha de emisión: 24/02/2021
CS TECH CONSULTING SA
CUIL/CUIT:30-71544453-0
Descripción Importe
Total valor Plan de Servicio $ 37.785,51
Neto Gravado $ 37.785,51
IVA Inscripto 10,50% $ 3.967,48
Percepción $ 2.833,91
Total $ 44.586,90
CAE: 71083175046030
FECHA DE VENCIMIENTO: 06.03.2021
"""
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)

        self.assertEqual(parsed["emisor"]["cuit"], "30-54674125-3")
        self.assertEqual(parsed["receptor"]["cuit"], "30-71544453-0")

    def test_lenovo_invoice_uses_filename_when_ocr_header_is_noisy(self):
        text = """Archivo: 05 Mayo - Lenovo 30714731382_001_0007_00007635.pdf
A FACTURA N� 0007-00007635
Lenovo Argentina SRL
Av. del Libertador 7208 piso 6
C�digo N� 01
I.V.A.Responsable Inscripto Fecha de Emisi�n: 21 DE MAYO DE 2021
Se�or/es CS TECH CONSULTING S.A Fecha de Vencimiento: Ver leyenda al pie
C.U.I.T 30-71473138-2
IVA resp.inscripto sin percepci�n RG2408/08 C.U.I.T: 30715444530
D E S C R I P C I O N D E B E
Por la venta de las siguientes unidades Lenovo:
ZA3V0065AR 4340569272 Yoga Smart Tab - Iron G 1 36.198,19 * 36.198,19
SUBTOTAL....$ 36.198,19
I.V.A.INSC.10,50 %....$ 3.800,81
IIBB BS AS 4.00 %....$ 1.447,93
Percepcion IIBB CABA 3.50 %....$ 1.266,94
TOTAL....$ __ __ __ __ __ __ 4 __ 2 .__ 7 1__ 3 ,__ 8 7__
POR CONSULTAS SOBRE ESTE DOCUMENTO: C.A.E.: 71212538327742
T.E.: 5293-7800 AREA COBRANZAS FECHA VTO: 31/05/2021
"""
        result = extract_document(text, filename="05 Mayo - Lenovo 30714731382_001_0007_00007635.pdf")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["numero_factura"], "00007-00007635")
        self.assertEqual(result["data"]["emisor"]["cuit"], "30-71473138-2")
        self.assertEqual(result["data"]["receptor"]["cuit"], "30-71544453-0")
        self.assertEqual(result["data"]["subtotal"], 36198.19)
        self.assertEqual(result["data"]["iva_total"], 3800.81)
        self.assertEqual(result["data"]["tributos_total"], 2714.87)
        self.assertEqual(result["data"]["total"], 42713.87)
        self.assertEqual(result["data"]["cae"], "71212538327742")
        self.assertEqual(result["data"]["fecha_vencimiento_cae"], "2021-05-31")
        self.assertEqual(result["data"]["descripcion"], "Yoga Smart Tab - Iron G")

    def test_trentadue_mac_invoice_extracts_multiple_items(self):
        text = """Archivo: 05 Mayo - Mac 30715151061_001_00001_00000056.pdf
Fecha de Emisi�n:
ORIGINAL
TRENTADUE S.A.
31/05/2021
30715151061
30715444530 CS TECH CONSULTING S.A.
Punto de Venta: Comp. Nro:00001 00000056
RESPAWN
FACTURAACOD. 01
C�digo Producto / Servicio Cantidad U. medida Precio Unit. % Bonif Subtotal Alicuota
IVA Subtotal c/IVA
Apple iPad 8th Generation 10.2" Wi-Fi 128GB
SPACE GREY NEWEST MODEL. 1 Year
Warranty, ETA: 7 Days, Retail Box, New Factory
Sealed
1,00 unidades 57280,00 0,00 57280,00 10,5% 63294,40
Apple Macbook i3 8gb 256ssd 13.3" 3,00 unidades 114560,00 0,00 343680,00 10,5% 379766,40
Apple Macbook i3 8gb 256ssd 13.3" - REF 4,00 unidades 106326,00 0,00 425304,00 10,5% 469960,92
Apple Macbook i5 8gb 256ssd 13.3" 1,00 unidades 150360,00 0,00 150360,00 10,5% 166147,80
Descripci�n ImporteDetalle Al�c. %
Otros Tributos
CAE N�:
Fecha de Vto. de CAE:
10/06/2021
71221112841237
Importe Otros Tributos: $ 0,00
Importe Neto Gravado: $ 976624,00
IVA 10.5%: $ 102545,52
Importe Total: $ 1079169,52
"""

        result = extract_document(text, filename="05 Mayo - Mac 30715151061_001_00001_00000056.pdf")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["emisor"]["nombre"], "TRENTADUE S.A.")
        self.assertEqual(result["data"]["fecha_emision"], "2021-05-31")
        self.assertEqual(result["data"]["subtotal"], 976624.0)
        self.assertEqual(result["data"]["iva_total"], 102545.52)
        self.assertEqual(result["data"]["total"], 1079169.52)
        self.assertEqual(len(result["data"]["items"]), 4)
        self.assertEqual(result["data"]["items"][0]["descripcion"], 'Apple iPad 8th Generation 10.2" Wi-Fi 128GB SPACE GREY NEWEST MODEL. 1 Year Warranty, ETA: 7 Days, Retail Box, New Factory Sealed')
        self.assertEqual(result["data"]["items"][1]["descripcion"], 'Apple Macbook i3 8gb 256ssd 13.3"')
        self.assertEqual(result["data"]["items"][2]["cantidad"], 4)
        self.assertIn("Apple Macbook i5 8gb 256ssd 13.3", result["data"]["descripcion"])

    def test_quality_warnings_detect_inconsistent_totals(self):
        text = arca_text(
            "A",
            1,
            "30715999999",
            "00003",
            "00000043",
            "71111111111112",
            "30/04/2021",
            "Servicio mensual de prueba",
            1210.0,
            "IVA 21%: $ 210,00",
        )
        parsed = add_arca_integration_fields(parse_supported_document_ocr(text), text)
        parsed["iva_total"] = 210.0
        parsed["impuestos"] = 210.0

        warnings = assess_document_quality(parsed, text)

        self.assertIn("Importes inconsistentes: subtotal + IVA + tributos no coincide con total.", warnings)

    def test_external_quality_accepts_zero_balance(self):
        parsed = {
            "document_type": "external_provider_receipt",
            "provider": {
                "name": "GoDaddy.com, LLC",
                "business_name": "GoDaddy.com, LLC",
                "tax_id": None,
                "vat_number": None,
                "address": None,
                "country": "United States",
                "phone": None,
            },
            "buyer": {
                "name": "Javier Nogues",
                "business_name": "CS TECH CONSULTING",
                "tax_id": "30715444530",
                "vat_number": None,
                "address": None,
                "country": "Argentina",
                "phone": None,
            },
            "document": {
                "title": "Recibo",
                "number": "1848619872",
                "date": "2021-04-03",
                "account_number": None,
                "customer_number": "203521924",
                "status": "paid",
            },
            "currency": "ARS",
            "subtotal": 399.99,
            "taxes": 0.0,
            "fees": 0.0,
            "total": 399.99,
            "paid": 399.99,
            "balance_due": 0.0,
            "payment": {
                "method": "card",
                "card_brand": "Visa",
                "card_last4": "7953",
                "amount": 399.99,
            },
            "items": [
                {
                    "description": "Correo Plus de Microsoft 365 de GoDaddy",
                    "quantity": 1,
                    "unit_price": 399.99,
                    "amount": 399.99,
                    "term": "1 mes",
                    "reference": None,
                }
            ],
            "notes": None,
            "descripcion": "Correo Plus de Microsoft 365 de GoDaddy",
        }

        warnings = assess_document_quality(parsed)

        self.assertNotIn("Recibo externo con pagado, saldo y total inconsistentes.", warnings)
        self.assertNotIn("Recibo externo sin total numerico.", warnings)


if __name__ == "__main__":
    unittest.main()
