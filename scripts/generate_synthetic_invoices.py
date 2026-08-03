import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

try:
    from reportlab.graphics.barcode import qr
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
except ModuleNotFoundError:
    qr = None
    Drawing = None
    colors = None
    A4 = None
    mm = None
    canvas = None


INSTRUCTION = (
    "Converti este texto OCR de una factura ARCA en JSON valido usando el schema "
    "indicado. No inventes datos; si falta un dato usa null o array vacio."
)

INVOICE_TYPES = [
    ("A", 1, "IVA Responsable Inscripto", "Responsable Inscripto"),
    ("B", 6, "IVA Responsable Inscripto", "Consumidor Final"),
    ("C", 11, "Monotributo", "Responsable Inscripto"),
]

BUSINESS_NAMES = [
    "Servicios del Plata SRL",
    "Comercial Rio Sur SA",
    "Insumos Patagonia SRL",
    "Tecnica Norte SA",
    "Distribuidora Centro",
    "Consultora Andina SRL",
    "Mercado Federal",
    "Talleres Oeste SRL",
    "Soluciones Medicas SA",
    "Libreria Avenida",
]

CLIENT_NAMES = [
    "Transporte Norte SA",
    "Cliente Demo SRL",
    "Estudio Contable Sur",
    "Consumidor Final",
    "Industrias Belgrano SA",
    "Ferreteria Central",
    "Hotel Plaza SRL",
    "Agro Servicios Norte",
    "Clinica Modelo SA",
    "Tienda La Esquina",
]

ITEM_NAMES = [
    "Servicio tecnico mensual",
    "Abono de mantenimiento",
    "Venta de insumos",
    "Honorarios profesionales",
    "Repuestos varios",
    "Licencia de software",
    "Materiales de oficina",
    "Consultoria operativa",
    "Capacitacion",
    "Servicio logistico",
]


@dataclass
class Persona:
    nombre: str
    cuit: str | None
    doc_tipo: int | None
    doc_nro: str | None
    condicion_iva: str | None


def digits(value: str | None) -> str | None:
    return None if value is None else "".join(ch for ch in value if ch.isdigit())


def fake_cuit(rng: random.Random, prefix: str | None = None) -> str:
    p = prefix or rng.choice(["20", "23", "27", "30", "33"])
    body = rng.randint(10_000_000, 99_999_999)
    check = rng.randint(0, 9)
    return f"{p}-{body:08d}-{check}"


def money(value: float) -> float:
    return round(value + 0.00001, 2)


def money_ar(value: float) -> str:
    raw = f"{value:,.2f}"
    return raw.replace(",", "X").replace(".", ",").replace("X", ".")


def iso_to_ar(value: str) -> str:
    year, month, day = value.split("-")
    return f"{day}/{month}/{year}"


def build_invoice(idx: int, rng: random.Random) -> dict:
    letter, code, emitter_tax, default_receiver_tax = INVOICE_TYPES[idx % len(INVOICE_TYPES)]
    pv = f"{rng.randint(1, 35):05d}"
    nro = f"{idx + 1:08d}"
    emitted = date(2026, 1, 1) + timedelta(days=rng.randint(0, 364))
    due = emitted + timedelta(days=10)

    emitter = Persona(
        nombre=rng.choice(BUSINESS_NAMES),
        cuit=fake_cuit(rng, rng.choice(["20", "30", "33"])),
        doc_tipo=80,
        doc_nro=None,
        condicion_iva=emitter_tax,
    )
    emitter.doc_nro = digits(emitter.cuit)

    if letter == "B" and rng.random() < 0.65:
        receiver = Persona("Consumidor Final", None, 96, str(rng.randint(10_000_000, 45_000_000)), "Consumidor Final")
    else:
        receiver = Persona(
            nombre=rng.choice([name for name in CLIENT_NAMES if name != "Consumidor Final"]),
            cuit=fake_cuit(rng, rng.choice(["20", "27", "30"])),
            doc_tipo=80,
            doc_nro=None,
            condicion_iva=default_receiver_tax,
        )
        receiver.doc_nro = digits(receiver.cuit)

    item_count = rng.randint(1, 5)
    items = []
    subtotal = 0.0
    for _ in range(item_count):
        qty = rng.choice([1, 1, 2, 3, 4, 5, 10])
        unit = money(rng.uniform(2500, 85000))
        total = money(qty * unit)
        subtotal += total
        items.append(
            {
                "descripcion": rng.choice(ITEM_NAMES),
                "cantidad": qty,
                "precio_unitario": unit,
                "importe": total,
            }
        )

    subtotal = money(subtotal)
    importe_no_gravado = 0.0
    importe_exento = 0.0
    iva_items = []
    iva_total = 0.0
    if letter == "A":
        rate = rng.choice([10.5, 21.0, 21.0, 27.0])
        iva_total = money(subtotal * rate / 100)
        iva_items.append(
            {
                "codigo": {10.5: 4, 21.0: 5, 27.0: 6}[rate],
                "descripcion": f"{rate:g}%",
                "base_imponible": subtotal,
                "importe": iva_total,
            }
        )
    elif letter == "B":
        importe_no_gravado = 0.0

    tributos = []
    tributos_total = 0.0
    if letter == "A" and rng.random() < 0.2:
        tributos_total = money(subtotal * 0.01)
        tributos.append(
            {
                "codigo": 99,
                "descripcion": "Percepcion municipal",
                "base_imponible": subtotal,
                "alicuota": 1.0,
                "importe": tributos_total,
            }
        )

    total = money(subtotal + iva_total + tributos_total)
    cae = "".join(str(rng.randint(0, 9)) for _ in range(14))

    label = {
        "tipo_comprobante": f"Factura {letter}",
        "codigo_comprobante": code,
        "punto_venta": pv,
        "numero_comprobante": nro,
        "numero_factura": f"{pv}-{nro}",
        "fecha_emision": emitted.isoformat(),
        "emisor": emitter.__dict__,
        "receptor": receiver.__dict__,
        "moneda": "PES",
        "tipo_cambio": 1,
        "subtotal": subtotal,
        "importe_no_gravado": importe_no_gravado,
        "importe_exento": importe_exento,
        "iva_total": iva_total,
        "tributos_total": tributos_total,
        "impuestos": money(iva_total + tributos_total),
        "total": total,
        "cae": cae,
        "fecha_vencimiento_cae": due.isoformat(),
        "iva": iva_items,
        "tributos": tributos,
        "items": items,
    }
    return label


def ocr_text(label: dict, rng: random.Random) -> str:
    emisor = label["emisor"]
    receptor = label["receptor"]
    lines = [
        f"FACTURA {label['tipo_comprobante'][-1]}",
        f"Cod. {label['codigo_comprobante']:03d}",
        f"Punto de Venta: {label['punto_venta']} Comp. Nro: {label['numero_comprobante']}",
        f"Fecha de Emision: {iso_to_ar(label['fecha_emision'])}",
        emisor["nombre"],
        f"CUIT: {emisor['cuit']}",
        emisor["condicion_iva"] or "",
        f"Cliente: {receptor['nombre']}",
    ]
    if receptor["cuit"]:
        lines.append(f"CUIT Cliente: {receptor['cuit']}")
    elif receptor["doc_nro"]:
        lines.append(f"DNI: {receptor['doc_nro']}")
    if receptor["condicion_iva"]:
        lines.append(f"Condicion IVA: {receptor['condicion_iva']}")
    lines.extend(["Moneda: PES", "Tipo Cambio: 1"])
    if rng.random() < 0.7:
        for item in label["items"]:
            lines.append(
                f"{item['descripcion']} Cant {item['cantidad']} P.Unit {money_ar(item['precio_unitario'])} Importe {money_ar(item['importe'])}"
            )
    lines.append(f"Subtotal: $ {money_ar(label['subtotal'])}")
    if label["iva_total"]:
        for iva in label["iva"]:
            lines.append(f"IVA {iva['descripcion']}: $ {money_ar(iva['importe'])}")
    if label["tributos_total"]:
        lines.append(f"Tributos: $ {money_ar(label['tributos_total'])}")
    lines.extend(
        [
            f"Importe Total: $ {money_ar(label['total'])}",
            f"CAE: {label['cae']}",
            f"Vto. CAE: {iso_to_ar(label['fecha_vencimiento_cae'])}",
            "MUESTRA SIN VALIDEZ FISCAL",
        ]
    )
    return "\n".join(line for line in lines if line)


def draw_box(c, x: float, y: float, w: float, h: float, stroke=None, fill=None):
    stroke = stroke or colors.black
    c.setStrokeColor(stroke)
    if fill:
        c.setFillColor(fill)
        c.rect(x, y, w, h, stroke=1, fill=1)
        c.setFillColor(colors.black)
    else:
        c.rect(x, y, w, h, stroke=1, fill=0)


def draw_invoice_pdf(label: dict, output_path: Path, layout: int):
    if canvas is None:
        raise RuntimeError("reportlab no esta instalado; no se pueden generar PDFs.")

    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    margin = 14 * mm
    top = height - margin

    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.Color(0.92, 0.92, 0.92))
    c.saveState()
    c.translate(width / 2, height / 2)
    c.rotate(35)
    c.setFont("Helvetica-Bold", 44)
    c.setFillColor(colors.Color(0.82, 0.82, 0.82))
    c.drawCentredString(0, 0, "MUESTRA SIN VALIDEZ FISCAL")
    c.restoreState()

    left_w = (width - 2 * margin) * (0.57 if layout % 2 == 0 else 0.50)
    right_x = margin + left_w
    right_w = width - margin - right_x
    header_h = 39 * mm
    draw_box(c, margin, top - header_h, left_w, header_h)
    draw_box(c, right_x, top - header_h, right_w, header_h)

    letter = label["tipo_comprobante"][-1]
    code = label["codigo_comprobante"]
    badge_w = 22 * mm
    draw_box(c, width / 2 - badge_w / 2, top - 21 * mm, badge_w, 18 * mm, fill=colors.white)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(width / 2, top - 14 * mm, letter)
    c.setFont("Helvetica", 7)
    c.drawCentredString(width / 2, top - 18 * mm, f"Cod. {code:03d}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin + 5 * mm, top - 10 * mm, label["emisor"]["nombre"])
    c.setFont("Helvetica", 9)
    c.drawString(margin + 5 * mm, top - 17 * mm, f"CUIT: {label['emisor']['cuit']}")
    c.drawString(margin + 5 * mm, top - 23 * mm, label["emisor"]["condicion_iva"] or "")
    c.drawString(margin + 5 * mm, top - 29 * mm, "Domicilio Comercial: Calle Falsa 123, CABA")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(right_x + 5 * mm, top - 10 * mm, label["tipo_comprobante"].upper())
    c.setFont("Helvetica", 9)
    c.drawString(right_x + 5 * mm, top - 18 * mm, f"Punto de Venta: {label['punto_venta']}")
    c.drawString(right_x + 5 * mm, top - 24 * mm, f"Comp. Nro: {label['numero_comprobante']}")
    c.drawString(right_x + 5 * mm, top - 30 * mm, f"Fecha: {iso_to_ar(label['fecha_emision'])}")

    y = top - header_h - 9 * mm
    draw_box(c, margin, y - 25 * mm, width - 2 * margin, 25 * mm)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin + 4 * mm, y - 7 * mm, "Datos del receptor")
    c.setFont("Helvetica", 8.5)
    receptor = label["receptor"]
    c.drawString(margin + 4 * mm, y - 14 * mm, f"Cliente: {receptor['nombre']}")
    doc = receptor["cuit"] or receptor["doc_nro"] or ""
    c.drawString(margin + 4 * mm, y - 20 * mm, f"Documento/CUIT: {doc}")
    c.drawString(margin + 95 * mm, y - 20 * mm, f"Condicion IVA: {receptor['condicion_iva'] or ''}")

    y -= 35 * mm
    c.setFont("Helvetica-Bold", 8)
    headers = ["Descripcion", "Cant.", "P. Unitario", "Importe"]
    xs = [margin + 4 * mm, margin + 105 * mm, margin + 128 * mm, margin + 160 * mm]
    draw_box(c, margin, y - 8 * mm, width - 2 * margin, 8 * mm, fill=colors.Color(0.94, 0.94, 0.94))
    for x, header in zip(xs, headers):
        c.drawString(x, y - 5.5 * mm, header)
    c.setFont("Helvetica", 8)
    y -= 8 * mm
    for item in label["items"]:
        draw_box(c, margin, y - 8 * mm, width - 2 * margin, 8 * mm, stroke=colors.Color(0.75, 0.75, 0.75))
        c.drawString(xs[0], y - 5.5 * mm, item["descripcion"][:48])
        c.drawRightString(xs[1] + 12 * mm, y - 5.5 * mm, str(item["cantidad"]))
        c.drawRightString(xs[2] + 24 * mm, y - 5.5 * mm, money_ar(item["precio_unitario"]))
        c.drawRightString(xs[3] + 25 * mm, y - 5.5 * mm, money_ar(item["importe"]))
        y -= 8 * mm

    totals_x = width - margin - 70 * mm
    y = max(y - 10 * mm, 78 * mm)
    totals = [
        ("Subtotal", label["subtotal"]),
        ("IVA", label["iva_total"]),
        ("Tributos", label["tributos_total"]),
        ("Total", label["total"]),
    ]
    c.setFont("Helvetica", 9)
    for name, value in totals:
        c.drawString(totals_x, y, name)
        c.drawRightString(width - margin, y, f"$ {money_ar(value)}")
        y -= 6 * mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(totals_x, y, "Importe Total")
    c.drawRightString(width - margin, y, f"$ {money_ar(label['total'])}")

    qr_text = json.dumps(
        {
            "ver": 1,
            "fecha": label["fecha_emision"],
            "cuit": digits(label["emisor"]["cuit"]),
            "ptoVta": int(label["punto_venta"]),
            "tipoCmp": label["codigo_comprobante"],
            "nroCmp": int(label["numero_comprobante"]),
            "importe": label["total"],
            "moneda": "PES",
            "ctz": 1,
            "tipoDocRec": label["receptor"]["doc_tipo"],
            "nroDocRec": label["receptor"]["doc_nro"],
            "tipoCodAut": "E",
            "codAut": label["cae"],
        },
        separators=(",", ":"),
    )
    qrw = qr.QrCodeWidget(qr_text)
    bounds = qrw.getBounds()
    qr_size = 31 * mm
    drawing = Drawing(qr_size, qr_size, transform=[qr_size / (bounds[2] - bounds[0]), 0, 0, qr_size / (bounds[3] - bounds[1]), 0, 0])
    drawing.add(qrw)
    drawing.drawOn(c, margin, 24 * mm)

    c.setFont("Helvetica", 8)
    c.drawString(margin + 36 * mm, 46 * mm, f"CAE: {label['cae']}")
    c.drawString(margin + 36 * mm, 40 * mm, f"Fecha Vto. CAE: {iso_to_ar(label['fecha_vencimiento_cae'])}")
    c.setFont("Helvetica-Bold", 8)
    c.drawString(margin + 36 * mm, 31 * mm, "Documento generado automaticamente para entrenamiento OCR.")
    c.drawString(margin + 36 * mm, 26 * mm, "No usar como comprobante fiscal.")
    c.showPage()
    c.save()


def main():
    parser = argparse.ArgumentParser(description="Genera facturas ARCA sinteticas para entrenamiento OCR.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/synthetic_invoices")
    parser.add_argument("--no-pdf", action="store_true", help="Genera solo JSONL/OCR/manifest sin PDFs.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    pdf_dir = out / "pdfs"
    ocr_dir = out / "ocr"
    write_pdfs = not args.no_pdf and canvas is not None
    if not write_pdfs and not args.no_pdf:
        print("Aviso: reportlab no esta instalado; se generan solo JSONL/OCR/manifest sin PDFs.")
    if write_pdfs:
        pdf_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)

    train_path = out / "synthetic_train.jsonl"
    manifest_path = out / "manifest.jsonl"
    with train_path.open("w", encoding="utf-8") as train_file, manifest_path.open("w", encoding="utf-8") as manifest_file:
        for idx in range(args.count):
            label = build_invoice(idx, rng)
            stem = f"factura_{idx + 1:04d}_{label['tipo_comprobante'][-1]}"
            pdf_path = pdf_dir / f"{stem}.pdf"
            ocr_path = ocr_dir / f"{stem}.txt"
            text = ocr_text(label, rng)
            if write_pdfs:
                draw_invoice_pdf(label, pdf_path, idx % 4)
            ocr_path.write_text(text, encoding="utf-8")
            train_file.write(
                json.dumps(
                    {"instruction": INSTRUCTION, "input": text, "output": json.dumps(label, ensure_ascii=False, separators=(",", ":"))},
                    ensure_ascii=False,
                )
                + "\n"
            )
            manifest_file.write(
                json.dumps(
                    {
                        "pdf": str(pdf_path),
                        "pdf_generado": write_pdfs,
                        "ocr": str(ocr_path),
                        "tipo_comprobante": label["tipo_comprobante"],
                        "numero_factura": label["numero_factura"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Generadas {args.count} facturas sinteticas en {out}")
    print(f"Dataset JSONL: {train_path}")
    if write_pdfs:
        print(f"PDFs: {pdf_dir}")
    else:
        print("PDFs: omitidos")


if __name__ == "__main__":
    main()
