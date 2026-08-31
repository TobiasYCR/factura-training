import unittest

from ocr import score_ocr_text


class OcrQualityTests(unittest.TestCase):
    def test_score_prefers_fiscal_fields_over_short_noise(self):
        good = """
Factura A
CUIT: 30-71544453-0
Neto Gravado 3.658,15
IVA 21% 768,21
Total a pagar $4.682,45
CAE 71128471020836
"""
        noisy = "|||| || eago [presos co Ju coins [te] o | stn y Consultoria Abap Marzo"

        self.assertGreater(score_ocr_text(good), score_ocr_text(noisy))

    def test_score_penalizes_empty_text(self):
        self.assertLess(score_ocr_text(""), score_ocr_text("Factura Total 1.000,00"))


if __name__ == "__main__":
    unittest.main()
