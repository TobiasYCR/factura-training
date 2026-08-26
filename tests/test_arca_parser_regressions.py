import unittest

from api import add_arca_integration_fields
from infer import parse_supported_document_ocr, validate_extracted_document_json


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


if __name__ == "__main__":
    unittest.main()
