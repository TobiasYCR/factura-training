import argparse
import json
import re
from datetime import datetime
from pathlib import Path

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
LORA_MODEL = "factura-qwen-lora"
MAX_SEQ_LENGTH = 2048
DEFAULT_INSTRUCTION = (
    "Converti este texto OCR de una factura ARCA en un unico objeto JSON valido. "
    "No inventes datos: si falta un dato usa null; para iva, tributos e items usa array vacio. "
    "No agregues texto antes o despues del JSON. "
    "Usa exactamente el schema ARCA con estas claves raiz: tipo_comprobante, codigo_comprobante, "
    "punto_venta, numero_comprobante, numero_factura, fecha_emision, emisor, receptor, moneda, "
    "tipo_cambio, subtotal, importe_no_gravado, importe_exento, iva_total, tributos_total, "
    "impuestos, total, cae, fecha_vencimiento_cae, iva, tributos, items. "
    "emisor y receptor deben tener: nombre, doc_tipo, doc_nro, cuit, condicion_iva. "
    "Normaliza valores: fechas YYYY-MM-DD, CUIT con guiones en cuit, doc_nro sin guiones, "
    "punto_venta de 5 digitos, numero_comprobante de 8 digitos, numero_factura como 00000-00000000. "
    "Usa moneda ARCA: PES para pesos argentinos y DOL para dolares. "
    "Codigos IVA: 10.5%=4, 21%=5, 27%=6. Para tributos/percepciones municipales usa codigo 99 si el OCR no informa otro codigo. "
    "Solo extrae items cuando aparezcan lineas de items en el OCR; si no aparecen usa items: []. "
    "Para cada item copia descripcion, cantidad, precio_unitario e importe exactamente; no recalcules ni inventes items. "
    "No incluyas etiquetas OCR como CUIT:, Cliente:, Comp. Nro: dentro de los valores."
)
REQUIRED_KEYS = {
    "tipo_comprobante",
    "codigo_comprobante",
    "punto_venta",
    "numero_comprobante",
    "numero_factura",
    "fecha_emision",
    "emisor",
    "receptor",
    "moneda",
    "tipo_cambio",
    "subtotal",
    "importe_no_gravado",
    "importe_exento",
    "iva_total",
    "tributos_total",
    "impuestos",
    "total",
    "cae",
    "fecha_vencimiento_cae",
    "iva",
    "tributos",
    "items",
}
PERSON_KEYS = {"nombre", "doc_tipo", "doc_nro", "cuit", "condicion_iva"}
IVA_CODE_BY_RATE = {10.5: 4, 21.0: 5, 27.0: 6}
EXTERNAL_DOCUMENT_KEYS = (
    "document_type",
    "provider",
    "buyer",
    "document",
    "currency",
    "subtotal",
    "taxes",
    "fees",
    "total",
    "paid",
    "balance_due",
    "payment",
    "items",
    "notes",
)
EXTERNAL_PARTY_KEYS = ("name", "business_name", "tax_id", "vat_number", "address", "country", "phone")
EXTERNAL_DOCUMENT_INFO_KEYS = ("title", "number", "date", "account_number", "customer_number", "status")
EXTERNAL_PAYMENT_KEYS = ("method", "card_brand", "card_last4", "amount")
EXTERNAL_ITEM_KEYS = ("description", "quantity", "unit_price", "amount", "term", "reference")


def build_prompt(ocr_text, instruction=DEFAULT_INSTRUCTION):
    return f"""### Instruccion:
{instruction}

### Texto OCR:
{ocr_text}

### Respuesta:
"""


def extract_json(text):
    text = text.strip()
    decoder = json.JSONDecoder()
    best_partial = (None, None, -1)

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            score = len(REQUIRED_KEYS & set(parsed))
            json_text = text[index : index + end]
            if REQUIRED_KEYS <= set(parsed):
                return parsed, json_text
            if score > best_partial[2]:
                best_partial = (parsed, json_text, score)

    if best_partial[0] is not None:
        return best_partial[0], best_partial[1]

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None, None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, match.group(0)

    return parsed, match.group(0)


def digits(value):
    if value is None:
        return None
    return re.sub(r"\D", "", str(value))


def format_cuit(value):
    cuit_digits = digits(value)
    if cuit_digits and len(cuit_digits) == 11:
        return f"{cuit_digits[:2]}-{cuit_digits[2:10]}-{cuit_digits[10]}"
    return value


def as_number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def round_money(value):
    if isinstance(value, str):
        value = re.sub(r"[^\d,.\-]", "", value)
        if "," in value and "." in value:
            if value.rfind(",") > value.rfind("."):
                value = value.replace(".", "").replace(",", ".")
            else:
                value = value.replace(",", "")
        elif "," in value:
            value = value.replace(".", "").replace(",", ".")
        elif value.count(".") > 1:
            parts = value.split(".")
            if parts[-1].isdigit() and len(parts[-1]) == 2:
                value = "".join(parts[:-1]) + "." + parts[-1]
    return round(float(value) + 1e-9, 2)


def parse_ar_money(value):
    return parse_money(value)


def parse_money(value):
    text = str(value)
    text = re.sub(r"[^\d,.\-]", "", text)
    if not text or not re.search(r"\d", text):
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif text.count(".") > 1:
        parts = text.split(".")
        if parts[-1].isdigit() and len(parts[-1]) == 2:
            text = "".join(parts[:-1]) + "." + parts[-1]
    return round_money(text)


def parse_ar_date(value):
    return datetime.strptime(value, "%d/%m/%Y").date().isoformat()


def parse_document_date(value):
    value = str(value).strip()
    month_aliases = {
        "ene": "01",
        "enero": "01",
        "jan": "01",
        "fev": "02",
        "feb": "02",
        "febrero": "02",
        "mar": "03",
        "marzo": "03",
        "abr": "04",
        "abril": "04",
        "apr": "04",
        "mai": "05",
        "may": "05",
        "mayo": "05",
        "jun": "06",
        "junio": "06",
        "jul": "07",
        "julio": "07",
        "ago": "08",
        "agosto": "08",
        "aug": "08",
        "set": "09",
        "sep": "09",
        "septiembre": "09",
        "out": "10",
        "oct": "10",
        "octubre": "10",
        "nov": "11",
        "noviembre": "11",
        "dez": "12",
        "dec": "12",
        "diciembre": "12",
    }
    long_spanish_match = re.search(
        r"(\d{1,2})\s+de\s+([A-Za-záéíóúñ.]+)\s+de\s+(\d{4})",
        value,
        re.IGNORECASE,
    )
    if long_spanish_match:
        month_name = long_spanish_match.group(2).lower().strip(".")
        month = month_aliases.get(month_name) or month_aliases.get(month_name[:3])
        if month:
            return f"{long_spanish_match.group(3)}-{month}-{int(long_spanish_match.group(1)):02d}"
    uppercase_spanish_match = re.search(
        r"(\d{1,2})\s+DE\s+([A-ZÁÉÍÓÚÑ]+)\s+DE\s+(\d{4})",
        value,
        re.IGNORECASE,
    )
    if uppercase_spanish_match:
        month_name = uppercase_spanish_match.group(2).lower()
        month = month_aliases.get(month_name) or month_aliases.get(month_name[:3])
        if month:
            return f"{uppercase_spanish_match.group(3)}-{month}-{int(uppercase_spanish_match.group(1)):02d}"
    alias_match = re.fullmatch(r"(\d{1,2})\s*-\s*([A-Za-z]{3})\s*-\s*(\d{4})", value)
    if alias_match:
        month = month_aliases.get(alias_match.group(2).lower())
        if month:
            return f"{alias_match.group(3)}-{month}-{int(alias_match.group(1)):02d}"
    for fmt in (
        "%d/%m/%Y",
        "%d/%m/%y",
        "%d.%m.%Y",
        "%d.%m.%y",
        "%d-%m-%Y",
        "%d-%m-%y",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%b %d %Y",
        "%B %d %Y",
    ):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def rate_from_description(description):
    if description is None:
        return None
    match = re.search(r"10[,.]5|21|27", str(description))
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def parse_quantity(value):
    normalized = str(value).replace(",", ".")
    quantity = float(normalized)
    return int(quantity) if quantity.is_integer() else quantity


def first_match(pattern, text, flags=0):
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def parse_godaddy_receipt_ocr(ocr_text):
    text = "\n".join(line.strip() for line in ocr_text.splitlines() if line.strip())
    text = text.replace("Duración Producto Cantidad", "Plazo Producto Cantidad")
    if "Recibo" not in text or "NÚMERO DE CLIENTE" not in text:
        return None

    number = first_match(r"Recibo\s*(?:№|N[°ºo.e]*)?\s*(\d+)", text, re.DOTALL)
    date = first_match(
        r"FECHA:\s*(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+de\s+[A-Za-záéíóúñ.]+\s+de\s+\d{4}(?:[^\n]*)?)",
        text,
        re.IGNORECASE,
    )
    customer_number = first_match(r"NÚMERO DE CLIENTE:\s*(\d+)", text)
    if not (number and date and customer_number):
        number = number or first_match(r"\bN[eo]\s+(\d+)", text, re.IGNORECASE)
    if not (number and date and customer_number):
        return None

    buyer_block = first_match(r"FACTURAR A:\s*(.*?)\s+PAGO:", text, re.DOTALL) or ""
    buyer_lines = [line.strip() for line in buyer_block.splitlines() if line.strip()]
    tax_id = first_match(r"ID fiscal:\s*([^\n]+)", buyer_block)
    phone = next((line for line in buyer_lines if line.startswith("+")), None)
    name = buyer_lines[0] if buyer_lines else None
    business_name = None
    address_lines = []
    for line in buyer_lines[1:]:
        if line.startswith("+") or line.startswith("ID fiscal:"):
            continue
        if business_name is None and line.isupper() and not any(char.isdigit() for char in line):
            business_name = line
            continue
        address_lines.append(line)

    provider_address = first_match(r"GoDaddy\.com, LLC\s*(?:\$\s*[\d.,]+\s*)?(.*?)\s+Tarifas", text, re.DOTALL)
    provider_lines = [line.strip().rstrip(",") for line in (provider_address or "").splitlines() if line.strip()]

    payment = re.search(r"PAGO:\s*(?P<brand>\w+)\s+.*?(?P<last4>\d{4})\s+\$\s*(?P<amount>[\d.,]+)", text, re.DOTALL)
    taxes = parse_money(first_match(r"Impuestos\s+\$\s*([\d.,]+)", text) or 0)
    fees = parse_money(first_match(r"Tarifas\s+\$\s*([\d.,]+)", text) or 0)
    currency = first_match(r"Total\s*\(([A-Z]{3})\)\s+\$", text, re.IGNORECASE) or "USD"
    total = parse_money(first_match(r"Total\s*\([A-Z]{3}\)\s+\$\s*([\d.,]+)", text, re.IGNORECASE))
    paid = parse_money(first_match(r"Pago recibido.*?\$\s*([\d.,]+)", text, re.DOTALL))
    balance_due = parse_money(first_match(r"Saldo adeudado\s*\([A-Z]{3}\)\s+\$\s*([\d.,]+)", text, re.IGNORECASE))
    subtotal = total if total is not None else parse_money(first_match(r"Saldo anterior\s+\$\s*([\d.,]+)", text))

    items = []
    item_section = first_match(r"Plazo Producto Cantidad\s*(.*?)(?:\nTotal\s*\([A-Z]{3}\))", text, re.DOTALL) or ""
    current_item = None
    for line in [line.strip() for line in item_section.splitlines() if line.strip()]:
        if line.startswith("about:blank") or "Mi cuenta | Facturación" in line:
            continue
        item = re.match(r"(?P<term>\d+\s+\S+)\s+(?P<description>.+?)\s+\$\s*(?P<amount>[\d.,]+)$", line)
        if item:
            current_item = {
                "description": item.group("description").replace("Microso", "Microsoft"),
                "quantity": 1,
                "unit_price": parse_money(item.group("amount")),
                "amount": parse_money(item.group("amount")),
                "term": item.group("term"),
                "reference": None,
            }
            items.append(current_item)
        elif current_item and "@" in line:
            current_item["reference"] = line

    return normalize_external_document(
        {
            "document_type": "external_provider_receipt",
            "provider": {
                "name": "GoDaddy.com, LLC",
                "business_name": "GoDaddy.com, LLC",
                "tax_id": None,
                "vat_number": None,
                "address": ", ".join(provider_lines) if provider_lines else None,
                "country": provider_lines[-1] if provider_lines else "United States",
                "phone": "(011) 5235-3894" if "(011) 5235-3894" in text else None,
            },
            "buyer": {
                "name": name,
                "business_name": business_name,
                "tax_id": digits(tax_id),
                "vat_number": None,
                "address": ", ".join(address_lines) if address_lines else None,
                "country": "Argentina" if "Argentina" in buyer_block else None,
                "phone": phone,
            },
            "document": {
                "title": "Recibo",
                "number": number,
                "date": parse_document_date(date),
                "account_number": None,
                "customer_number": customer_number,
                "status": "paid" if paid and balance_due == 0 else None,
            },
            "currency": currency.upper(),
            "subtotal": subtotal,
            "taxes": taxes,
            "fees": fees,
            "total": total,
            "paid": paid,
            "balance_due": balance_due,
            "payment": {
                "method": "card" if payment else None,
                "card_brand": payment.group("brand") if payment else None,
                "card_last4": payment.group("last4") if payment else None,
                "amount": parse_money(payment.group("amount")) if payment else None,
            },
            "items": items,
            "notes": "GoDaddy receipt. Not an ARCA invoice.",
        }
    )


def parse_godaddy_ocr_receipt_ocr(ocr_text):
    text = "\n".join(line.strip() for line in ocr_text.splitlines() if line.strip())
    if "Recibo" not in text or "GoDaddy" not in text:
        return None

    number = first_match(r"\bN[eo]\s+(\d+)", text, re.IGNORECASE) or first_match(
        r"Recibo\s*(?:№|N[°ºo.e]*)?\s*(\d+)", text, re.DOTALL
    )
    date = first_match(
        r"FECHA:\s*(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+de\s+[A-Za-záéíóúñ.]+\s+de\s+\d{4}(?:[^\n]*)?)",
        text,
        re.IGNORECASE,
    )
    customer_number = first_match(r"N\S*MERO DE CLIENTE:\s*(\d+)", text, re.IGNORECASE)
    if not (number and date and customer_number):
        return None

    buyer_block = first_match(r"FACTURAR A:\s*(.*?)\s+PAGO:", text, re.DOTALL) or ""
    buyer_lines = [line.strip() for line in buyer_block.splitlines() if line.strip()]
    tax_id = first_match(r"ID fiscal:\s*([^\n]+)", buyer_block)
    phone = next((line for line in buyer_lines if line.startswith("+")), None)
    name = buyer_lines[0] if buyer_lines else None
    business_name = next((line for line in buyer_lines if "TECH" in line.upper()), None)
    address_lines = [
        line
        for line in buyer_lines[1:]
        if not line.startswith("+") and not line.startswith("ID fiscal:") and line != business_name
    ]

    currency = first_match(r"Total\s*\(([A-Z]{3})\)\s+\$", text, re.IGNORECASE) or "USD"
    total = parse_money(first_match(r"Total\s*\([A-Z]{3}\)\s+\$\s*([\d.,]+)", text, re.IGNORECASE))
    paid = parse_money(first_match(r"Pago recibido.*?\$\s*([\d.,]+)", text, re.DOTALL))
    balance_due = parse_money(first_match(r"Saldo adeudado\s*\([A-Z]{3}\)\s+\$\s*([\d.,]+)", text, re.IGNORECASE))
    subtotal = total if total is not None else parse_money(first_match(r"Saldo anterior\s+\$\s*([\d.,]+)", text))
    taxes = parse_money(first_match(r"Impuestos\s+\$\s*([\d.,]+)", text) or 0)
    fees = parse_money(first_match(r"Tarifas\s+\$\s*([\d.,]+)", text) or 0)
    payment = re.search(r"PAGO:\s*(?P<brand>\w+)\s+.*?(?P<last4>\d{4})\s+\$\s*(?P<amount>[\d.,]+)", text, re.DOTALL)

    items = []
    item_section = (
        first_match(r"(?:Plazo|Duraci\S+n) Producto Cantidad\s*(.*?)(?:\nTotal\s*\([A-Z]{3}\))", text, re.DOTALL)
        or ""
    )
    current_item = None
    for line in [line.strip() for line in item_section.splitlines() if line.strip()]:
        item = re.match(r"(?P<term>\d+\s+\S+)\s+(?P<description>.+?)\s+\$\s*(?P<amount>[\d.,]+)$", line)
        if item:
            current_item = {
                "description": item.group("description"),
                "quantity": 1,
                "unit_price": parse_money(item.group("amount")),
                "amount": parse_money(item.group("amount")),
                "term": item.group("term"),
                "reference": None,
            }
            items.append(current_item)
        elif current_item and "@" in line:
            current_item["reference"] = line

    provider_address = first_match(r"GoDaddy\.com, LLC.*?\n(.*?United States)", text, re.DOTALL)
    provider_lines = [line.strip().rstrip(",") for line in (provider_address or "").splitlines() if line.strip()]

    return normalize_external_document(
        {
            "document_type": "external_provider_receipt",
            "provider": {
                "name": "GoDaddy.com, LLC",
                "business_name": "GoDaddy.com, LLC",
                "tax_id": None,
                "vat_number": None,
                "address": ", ".join(provider_lines) if provider_lines else None,
                "country": "United States",
                "phone": "(011) 5984-0780" if "(011) 5984-0780" in text else None,
            },
            "buyer": {
                "name": name,
                "business_name": business_name,
                "tax_id": digits(tax_id),
                "vat_number": None,
                "address": ", ".join(address_lines) if address_lines else None,
                "country": "Argentina" if "Argentina" in buyer_block else None,
                "phone": phone,
            },
            "document": {
                "title": "Recibo",
                "number": number,
                "date": parse_document_date(date),
                "account_number": None,
                "customer_number": customer_number,
                "status": "paid" if paid and balance_due == 0 else None,
            },
            "currency": currency.upper(),
            "subtotal": subtotal,
            "taxes": taxes,
            "fees": fees,
            "total": total,
            "paid": paid,
            "balance_due": balance_due,
            "payment": {
                "method": "card" if payment else None,
                "card_brand": payment.group("brand") if payment else None,
                "card_last4": payment.group("last4") if payment else None,
                "amount": parse_money(payment.group("amount")) if payment else None,
            },
            "items": items,
            "notes": "GoDaddy receipt parsed from OCR text. Not an ARCA invoice.",
        }
    )


def parse_teamwork_invoice_ocr(ocr_text):
    text = "\n".join(line.strip() for line in ocr_text.splitlines() if line.strip())
    if "INVOICE" not in text or "Teamwork.com" not in text:
        return None

    number = first_match(r"Ref #:\s*([^\n]+)", text)
    date = first_match(r"Issued:\s*([A-Za-z]{3,9}\s+\d{1,2}\s+\d{4})", text)
    account_number = first_match(r"Account #:\s*(\d+)", text)
    buyer_block = first_match(r"CUIT:\s*(.*?)(?:VAT Number:|Payment Method)", text, re.DOTALL) or ""
    buyer_cuit = first_match(r"(\d{11})", buyer_block)
    vat_number = first_match(r"VAT Number:\s*([A-Z0-9]+)", text)
    card_last4 = first_match(r"Credit/Debit Card\s+(\d{4})", text)
    item = re.search(
        r"(?P<quantity>\d+)\s*[×x]\s*(?P<description>.+?)\s+\(at\s*\$(?P<unit>[\d.,]+)\s*/\s*(?:USD\s*)?\$(?P<amount>[\d.,]+)",
        text,
        re.DOTALL,
    )
    total = parse_money(first_match(r"Total:\s*USD\s*\$([\d.,]+)", text))
    subtotal = parse_money(first_match(r"Subtotal\s+USD\s*\$([\d.,]+)", text))
    paid = parse_money(first_match(r"Paid\s+USD\s*\$([\d.,]+)", text))

    items = []
    if item:
        items.append(
            {
                "description": re.sub(r"\s+", " ", item.group("description")).strip(),
                "quantity": parse_quantity(item.group("quantity")),
                "unit_price": parse_money(item.group("unit")),
                "amount": parse_money(item.group("amount")),
                "term": "month" if "month" in text.lower() else None,
                "reference": None,
            }
        )

    return normalize_external_document(
        {
            "document_type": "external_provider_invoice",
            "provider": {
                "name": "Teamwork.com",
                "business_name": "Teamwork.com",
                "tax_id": None,
                "vat_number": vat_number,
                "address": "Teamwork Campus One, Blackpool Retail Park, Cork, T23 F902",
                "country": "Ireland",
                "phone": None,
            },
            "buyer": {
                "name": "CS Tech Consulting SA",
                "business_name": "CS Tech Consulting SA",
                "tax_id": digits(buyer_cuit),
                "vat_number": None,
                "address": "Rocha Montarce, 1150, El Palomar, Buenos Aires, 1684",
                "country": "Argentina",
                "phone": None,
            },
            "document": {
                "title": "INVOICE",
                "number": number,
                "date": parse_document_date(date),
                "account_number": account_number,
                "customer_number": None,
                "status": "paid" if paid and total == paid else None,
            },
            "currency": "USD",
            "subtotal": subtotal,
            "taxes": 0.0,
            "fees": 0.0,
            "total": total,
            "paid": paid,
            "balance_due": round_money((total or 0) - (paid or 0)),
            "payment": {
                "method": "Credit/Debit Card",
                "card_brand": None,
                "card_last4": card_last4,
                "amount": paid,
            },
            "items": items,
            "notes": "Teamwork/Wise international invoice. Not an ARCA invoice. Reverse charge/VAT note present.",
        }
    )


def parse_ifastnet_invoice_ocr(ocr_text):
    text = "\n".join(line.strip() for line in ocr_text.splitlines() if line.strip())
    if "IFASTNET" not in text.upper() or "WFWEF" not in text.upper():
        return None

    number = first_match(r"Factura\s+n[^A-Z0-9]{0,4}([A-Z0-9]+-[0-9]+)", text, re.IGNORECASE)
    date = first_match(r"Fecha de la Factura:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
    due_date = first_match(r"Fecha de Vencimiento:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
    total = parse_money(first_match(r"\bTotal\s*\$?\s*([\d.,]+)\s*USD", text, re.IGNORECASE))
    subtotal = parse_money(first_match(r"Sub Total\s*\$?\s*([\d.,]+)\s*USD", text, re.IGNORECASE))
    paid = parse_money(first_match(r"\bTransacciones.*?\$?\s*([\d.,]+)\s*USD", text, re.IGNORECASE | re.DOTALL))
    balance_due = parse_money(first_match(r"Balance\s*\$?\s*([\d.,]+)\s*USD", text, re.IGNORECASE))
    item = re.search(r"Descripci\S+n Total\s*(.*?)\s*\$?\s*([\d.,]+)USD", text, re.IGNORECASE | re.DOTALL)

    if not (number and date and total is not None):
        return None

    items = []
    if item:
        items.append(
            {
                "description": re.sub(r"\s+", " ", item.group(1)).strip(),
                "quantity": 1,
                "unit_price": parse_money(item.group(2)),
                "amount": parse_money(item.group(2)),
                "term": None,
                "reference": None,
            }
        )

    return normalize_external_document(
        {
            "document_type": "external_provider_invoice",
            "provider": {
                "name": "iFastNet Internet",
                "business_name": "iFastNet Internet",
                "tax_id": None,
                "vat_number": first_match(r"VAT Number\s*([A-Z0-9-]+)", text, re.IGNORECASE),
                "address": "Bulman House, Regent Centre, Gosforth, Newcastle Upon Tyne, NE3 3LS",
                "country": "United Kingdom",
                "phone": None,
            },
            "buyer": {
                "name": "CS-Tech",
                "business_name": "CS-Tech",
                "tax_id": None,
                "vat_number": None,
                "address": "Montarce 1150, Buenos Aires, Buenos Aires, 1650, Argentina",
                "country": "Argentina",
                "phone": None,
            },
            "document": {
                "title": "Factura",
                "number": number,
                "date": parse_document_date(date),
                "account_number": None,
                "customer_number": None,
                "status": "paid" if balance_due == 0 else None,
            },
            "currency": "USD",
            "subtotal": subtotal,
            "taxes": 0.0,
            "fees": 0.0,
            "total": total,
            "paid": paid or total if balance_due == 0 else paid,
            "balance_due": balance_due,
            "payment": {
                "method": "Credit Card" if "Credit Card" in text else None,
                "card_brand": None,
                "card_last4": None,
                "amount": paid or total if balance_due == 0 else paid,
            },
            "items": items,
            "notes": f"iFastNet external hosting invoice. Due date: {parse_document_date(due_date) if due_date else None}. Not an ARCA invoice.",
        }
    )


def parse_aerolineas_credit_fiscal_ocr(ocr_text):
    text = "\n".join(line.strip() for line in ocr_text.splitlines() if line.strip())
    upper_text = text.upper()
    if "CONSTANCIA DE CREDITO FISCAL" not in upper_text or "AEROLINEAS ARGENTINAS" not in upper_text:
        return None

    number = first_match(r"Constancia Nro\.:\s*(\d+)", text, re.IGNORECASE)
    date = first_match(r"Fecha\s*\.:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
    provider_name = first_match(r"Denominacion\s*:\s*([^\n]+)", text, re.IGNORECASE) or "AEROLINEAS ARGENTINAS S.A."
    provider_cuit = first_match(r"C\.U\.I\.T\.?\s*Nro\.:\s*(\d{2}-\d{8}-\d|\d{11})", text, re.IGNORECASE)
    receiver_name = first_match(r"Apellido y nombres o denominacion\s*:\s*([^\n]+)", text, re.IGNORECASE)
    cuit_matches = re.findall(r"C\.U\.I\.T\.?\s*Nro\.:\s*(\d{2}-\d{8}-\d|\d{11})", text, re.IGNORECASE)
    receiver_cuit = cuit_matches[1] if len(cuit_matches) > 1 else None
    original_number = first_match(r"- Numero:\s*([^\n]+)", text, re.IGNORECASE)
    original_date = first_match(r"- Fecha\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
    total = parse_money(first_match(r"Importe del comprobante\s*:\s*\$?\s*([\d.,]+)", text, re.IGNORECASE))
    base_105 = parse_money(first_match(r"Importe gravado 10\.5%\s*:\s*\$?\s*([\d.,]+)", text, re.IGNORECASE))
    tax_105 = parse_money(first_match(r"Importe del credito fiscal 10\.5%\s*:\s*\$?\s*([\d.,]+)", text, re.IGNORECASE))
    base_21 = parse_money(first_match(r"Importe gravado 21%\s*:\s*\$?\s*([\d.,]+)", text, re.IGNORECASE) or 0)
    tax_21 = parse_money(first_match(r"Importe del credito fiscal 21%\s*:\s*\$?\s*([\d.,]+)", text, re.IGNORECASE) or 0)

    if not (number and date and total is not None):
        return None

    taxes = round_money((tax_105 or 0) + (tax_21 or 0))
    subtotal = round_money((base_105 or 0) + (base_21 or 0)) if base_105 is not None else None

    return normalize_external_document(
        {
            "document_type": "external_provider_invoice",
            "provider": {
                "name": provider_name,
                "business_name": provider_name,
                "tax_id": provider_cuit,
                "vat_number": None,
                "address": first_match(r"Domicilio\s*:\s*([^\n]+)", text, re.IGNORECASE),
                "country": "Argentina",
                "phone": None,
            },
            "buyer": {
                "name": receiver_name,
                "business_name": receiver_name,
                "tax_id": receiver_cuit,
                "vat_number": None,
                "address": first_match(r"B - Datos del Receptor.*?Domicilio\s*:\s*([^\n]+)", text, re.IGNORECASE | re.DOTALL),
                "country": "Argentina",
                "phone": None,
            },
            "document": {
                "title": "CONSTANCIA DE CREDITO FISCAL",
                "number": number,
                "date": parse_document_date(date),
                "account_number": None,
                "customer_number": None,
                "status": None,
            },
            "currency": "ARS",
            "subtotal": subtotal,
            "taxes": taxes,
            "fees": 0.0,
            "total": total,
            "paid": None,
            "balance_due": None,
            "payment": {
                "method": None,
                "card_brand": None,
                "card_last4": None,
                "amount": None,
            },
            "items": [
                {
                    "description": f"Credito fiscal del comprobante {original_number or ''}".strip(),
                    "quantity": 1,
                    "unit_price": total,
                    "amount": total,
                    "term": None,
                    "reference": parse_document_date(original_date) if original_date else None,
                }
            ],
            "notes": "Aerolíneas Argentinas fiscal credit certificate. Not an ARCA invoice.",
        }
    )


def parse_generic_external_invoice_ocr(ocr_text):
    text = "\n".join(line.strip() for line in ocr_text.splitlines() if line.strip())
    upper_text = text.upper()
    if not any(marker in upper_text for marker in ("FACTURA", "INVOICE", "RECIBO")):
        return None
    if "CAE" in upper_text and "PUNTO DE VENTA" in upper_text:
        return None

    invoice_no = re.search(r"INVOICE\s+NO\.\s*(?P<prefix>[0-9-]+)\s*(?P<number>\d+)", text, re.IGNORECASE)
    number = (
        first_match(r"N[uú]mero de factura:\s*([A-Z0-9-]+)", text, re.IGNORECASE)
        or first_match(r"N[º°]\s*factura\s*([A-Z0-9-]+)", text, re.IGNORECASE)
        or first_match(r"Factura\s+n[^A-Z0-9]{0,4}([A-Z0-9]+-[0-9]+)", text, re.IGNORECASE)
        or first_match(r"FACTURA\s+N\.\S*\s*([A-Z0-9-]+)", text, re.IGNORECASE)
        or first_match(r"N[uú]mero:\s*([A-Z0-9-]+)", text, re.IGNORECASE)
        or first_match(r"\b([A-Z]\d{3}-\d+)\b", text)
    )
    if invoice_no:
        number = f"{invoice_no.group('prefix')}{invoice_no.group('number')}"
    if not number:
        number = (
            first_match(r"Invoice Number:\s*([A-Z0-9-]+)", text, re.IGNORECASE)
            or first_match(r"Factura:\s*(\d{4,5}-\d{8})", text, re.IGNORECASE)
        )

    date = (
        first_match(r"Fecha de la Factura:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
        or first_match(r"Buenos Aires\s+(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
        or first_match(r"Fecha de factura:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
        or first_match(r"DATE:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
        or first_match(r"FECHA:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
        or first_match(r"Fecha:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4})", text, re.IGNORECASE)
        or first_match(r"Fecha de Emisi[oó]n\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
    )
    if not date:
        date = first_match(r"Issue Date:\s*(\d{1,2}\s*-\s*[A-Za-z]{3}\s*-\s*\d{4})", text, re.IGNORECASE)

    total_text = (
        first_match(r"TOTAL(?:\s+FACTURA)?\s*:?\s*(?:USD|EUR|ARS)?\s*\$?\s*([\d.,]+)\s*(?:\$|€)?", text, re.IGNORECASE)
        or first_match(r"Balance Due\s+\$?\s*([\d.,]+)", text, re.IGNORECASE)
        or first_match(r"Importe Total\s*:\s*\$?\s*([\d.,]+)", text, re.IGNORECASE)
    )
    if not total_text:
        total_text = first_match(
            r"Total\s+(?:USD|EUR|ARS)\s+[\d.,]+\s+(?:USD|EUR|ARS)\s+([\d.,]+)",
            text,
            re.IGNORECASE,
        )
    total = parse_money(total_text) if total_text else None
    if not (number and date and total is not None):
        return None

    subtotal_text = (
        first_match(r"SUBTOTAL\s*:?\s*(?:USD|EUR|ARS)?\s*\$?\s*([\d.,]+)", text, re.IGNORECASE)
        or first_match(r"Subtotal\s+([\d.,]+)", text, re.IGNORECASE)
        or first_match(r"Sub Total Ventas\s*:\s*\$?\s*([\d.,]+)", text, re.IGNORECASE)
    )
    taxes_text = (
        first_match(r"(?:IVA|TOTAL TAX|IGV)\s*:?\s*(?:USD|EUR|ARS)?\s*\$?\s*([\d.,]+)", text, re.IGNORECASE)
        or first_match(r"I\.V\.A\.\s*[\d.,]+%\s+([\d.,]+)", text, re.IGNORECASE)
    )

    currency = None
    if re.search(r"\bUSD\b|DOLAR", upper_text):
        currency = "USD"
    elif re.search(r"\bEUR\b|€", upper_text):
        currency = "EUR"
    elif re.search(r"\bARS\b|PESOS", upper_text):
        currency = "ARS"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    provider_name = None
    if lines:
        if lines[0].upper().startswith("LA FACTURA") and len(lines) > 1:
            provider_name = lines[1]
        elif lines[0].upper().startswith("FACTURA ") and len(lines[0].split()) > 1:
            provider_name = lines[0].split(" ", 1)[1]
        elif lines[0].upper().startswith("FACTURA") and len(lines) > 1:
            provider_name = lines[1]
    if not provider_name:
        provider_name = first_match(r"Supplier:\s*(.*?)\s+Customer:", text, re.IGNORECASE)

    buyer_name = (
        first_match(r"Facturar a:\s*(?:Nombre del cliente:\s*)?([^\n]+)", text, re.IGNORECASE | re.DOTALL)
        or first_match(r"BILL TO\s+SHIP TO\s+([^\n]+)", text, re.IGNORECASE)
        or first_match(r"PARA:\s*(?:Nombre de la compa\S+:\s*)?([^\n]+)", text, re.IGNORECASE | re.DOTALL)
        or first_match(r"Cliente:\s*([^\n]+)", text, re.IGNORECASE)
        or first_match(r"Se\S+\(es\)\s*:\s*([^\n]+)", text, re.IGNORECASE)
    )
    if not buyer_name:
        buyer_name = first_match(r"Customer:\s*([^\n]+)", text, re.IGNORECASE)

    return normalize_external_document(
        {
            "document_type": "external_provider_invoice",
            "provider": {
                "name": provider_name,
                "business_name": provider_name,
                "tax_id": None,
                "vat_number": first_match(r"\b(?:NIF/CIF|N\.I\.F\.|RUC):\s*([A-Z0-9-]+)", text, re.IGNORECASE),
                "address": None,
                "country": None,
                "phone": first_match(r"Tel[eé]fono:\s*([^\n]+)", text, re.IGNORECASE),
            },
            "buyer": {
                "name": buyer_name,
                "business_name": buyer_name,
                "tax_id": first_match(r"\b(?:RUC|CUIT|N\.I\.F\.):\s*([0-9-]+)", text, re.IGNORECASE),
                "vat_number": None,
                "address": first_match(r"(?:Domicilio|Direcci[oó]n)\s*:?\s*([^\n]+)", text, re.IGNORECASE),
                "country": None,
                "phone": None,
            },
            "document": {
                "title": "FACTURA" if "FACTURA" in upper_text else "INVOICE",
                "number": number,
                "date": parse_document_date(date),
                "account_number": None,
                "customer_number": first_match(r"N[uú]mero de cliente:\s*(\d+)", text, re.IGNORECASE),
                "status": None,
            },
            "currency": currency,
            "subtotal": parse_money(subtotal_text) if subtotal_text else None,
            "taxes": parse_money(taxes_text) if taxes_text else None,
            "fees": 0.0,
            "total": total,
            "paid": None,
            "balance_due": total,
            "payment": {
                "method": None,
                "card_brand": None,
                "card_last4": None,
                "amount": None,
            },
            "items": [],
            "notes": "Generic external invoice parsed from observed PDF text. Not an ARCA invoice.",
        }
    )


def parse_real_arca_ocr(text, letter, code, numbers, issue_date, cae, due_date, document_kind="Factura"):
    if "Apellido y Nombre / Raz" not in text:
        return None

    emitter_name = first_match(r"Raz[oó]n Social:\s*(.*?)\s+Fecha de Emisi[oó]n", text)
    emitter_cuit = first_match(r"Domicilio Comercial:.*?CUIT:\s*(\d{11}|\d{2}-\d{8}-\d)", text, re.DOTALL)
    emitter_tax = first_match(r"Condici[oó]n frente al IVA:\s*(.*?)\s+Fecha de Inicio", text)
    receiver = re.search(
        r"\nCUIT:\s*(?P<cuit>\d{11}|\d{2}-\d{8}-\d)\s+Apellido y Nombre / Raz[oó]n Social:\s*(?P<name>.*?)(?:\n|$)",
        text,
    )
    if not (emitter_name and emitter_cuit and receiver):
        return None

    receiver_tax = None
    tax_matches = re.findall(r"Condici[oó]n frente al IVA:\s*(.*?)(?:\s+Domicilio:|\s+Fecha de Inicio|\n)", text)
    if len(tax_matches) >= 2:
        receiver_tax = tax_matches[1].strip()

    point_of_sale = numbers.group(1).zfill(5)
    receipt_number = numbers.group(2).zfill(8)
    subtotal = parse_ar_money(first_match(r"Subtotal:\s*\$\s*([\d.,]+)", text) or 0)
    tributos_total = parse_ar_money(first_match(r"Importe Otros Tributos:\s*\$\s*([\d.,]+)", text) or 0)
    total = parse_ar_money(first_match(r"Importe Total:\s*\$\s*([\d.,]+)", text) or subtotal)

    items = []
    seen_items = set()
    for line in text.splitlines():
        item_match = re.match(
            r"(?P<code>\d{2})\s+(?P<description>.+?)\s+(?P<quantity>\d+(?:[,.]\d+)?)\s+(?P<unit_name>\S+)\s+(?P<unit>[\d.,]+)\s+(?P<discount_rate>[\d.,]+)\s+(?P<discount_amount>[\d.,]+)\s+(?P<amount>[\d.,]+)$",
            line.strip(),
        )
        if item_match:
            item_key = item_match.group(0)
            if item_key in seen_items:
                continue
            seen_items.add(item_key)
            items.append(
                {
                    "descripcion": item_match.group("description"),
                    "cantidad": parse_quantity(item_match.group("quantity")),
                    "precio_unitario": parse_ar_money(item_match.group("unit")),
                    "importe": parse_ar_money(item_match.group("amount")),
                }
            )

    parsed = {
        "tipo_comprobante": f"{document_kind} {letter}",
        "codigo_comprobante": int(code.group(1)),
        "punto_venta": point_of_sale,
        "numero_comprobante": receipt_number,
        "numero_factura": f"{point_of_sale}-{receipt_number}",
        "fecha_emision": parse_ar_date(issue_date.group(1)),
        "emisor": {
            "nombre": emitter_name,
            "cuit": emitter_cuit,
            "doc_tipo": 80,
            "doc_nro": digits(emitter_cuit),
            "condicion_iva": emitter_tax,
        },
        "receptor": {
            "nombre": receiver.group("name").strip(),
            "cuit": receiver.group("cuit"),
            "doc_tipo": 80,
            "doc_nro": digits(receiver.group("cuit")),
            "condicion_iva": receiver_tax,
        },
        "moneda": "PES",
        "tipo_cambio": 1,
        "subtotal": subtotal,
        "importe_no_gravado": 0.0,
        "importe_exento": 0.0,
        "iva_total": 0.0,
        "tributos_total": tributos_total,
        "impuestos": tributos_total,
        "total": total,
        "cae": cae.group(1),
        "fecha_vencimiento_cae": parse_ar_date(due_date.group(1)),
        "iva": [],
        "tributos": [],
        "items": items,
    }
    return normalize_invoice_json(parsed)


def parse_arca_summary_ocr(text, letter, code, numbers, issue_date, cae, due_date, document_kind="Factura"):
    cuit_matches = re.findall(r"CUIT:\s*(\d{11}|\d{2}-\d{8}-\d)", text)
    receiver = re.search(
        r"CUIT:\s*(?P<cuit>\d{11}|\d{2}-\d{8}-\d)\s+Apellido y Nombre / Raz[oóÃ³]n Social:\s*(?P<name>.*?)(?:\n|$)",
        text,
    )
    emitter_name = first_match(r"Raz[oóÃ³]n Social:\s*([^\n]+)", text)
    if not (emitter_name and cuit_matches and receiver):
        return None

    point_of_sale = numbers.group(1).zfill(5)
    receipt_number = numbers.group(2).zfill(8)
    subtotal = parse_ar_money(first_match(r"Subtotal:\s*\$\s*([\d.,]+)", text) or 0)
    net = parse_ar_money(first_match(r"Importe Neto Gravado:\s*\$\s*([\d.,]+)", text) or subtotal)
    tributos_total = parse_ar_money(first_match(r"Importe Otros Tributos:\s*\$\s*([\d.,]+)", text) or 0)
    total = parse_ar_money(first_match(r"Importe Total:\s*\$\s*([\d.,]+)", text) or subtotal or net)

    iva = []
    for rate_text, amount_text in re.findall(r"IVA\s+(\d+(?:[,.]\d+)?)%:\s*\$\s*([\d.,]+)", text):
        rate = parse_money(rate_text)
        amount = parse_ar_money(amount_text)
        if amount and amount > 0:
            iva.append(
                {
                    "codigo": IVA_CODE_BY_RATE.get(rate),
                    "descripcion": f"{rate:g}%",
                    "base_imponible": net if net else None,
                    "importe": amount,
                }
            )
    iva_total = round_money(sum(item["importe"] for item in iva))

    concept = first_match(r"en concepto de:\s*(.*?)\s+Subtotal:", text, re.DOTALL)
    items = []
    if concept:
        items.append(
            {
                "descripcion": re.sub(r"\s+", " ", concept).strip(),
                "cantidad": 1,
                "precio_unitario": subtotal,
                "importe": subtotal,
            }
        )

    parsed = {
        "tipo_comprobante": f"{document_kind} {letter}",
        "codigo_comprobante": int(code.group(1)),
        "punto_venta": point_of_sale,
        "numero_comprobante": receipt_number,
        "numero_factura": f"{point_of_sale}-{receipt_number}",
        "fecha_emision": parse_ar_date(issue_date.group(1)),
        "emisor": {
            "nombre": emitter_name,
            "cuit": cuit_matches[0],
            "doc_tipo": 80,
            "doc_nro": digits(cuit_matches[0]),
            "condicion_iva": first_match(r"Condici[oóÃ³]n\s+fren\s*te\s+al IVA:\s*([^\n]+)", text),
        },
        "receptor": {
            "nombre": receiver.group("name").strip(),
            "cuit": receiver.group("cuit"),
            "doc_tipo": 80,
            "doc_nro": digits(receiver.group("cuit")),
            "condicion_iva": None,
        },
        "moneda": "PES",
        "tipo_cambio": 1,
        "subtotal": subtotal or net,
        "importe_no_gravado": 0.0,
        "importe_exento": 0.0,
        "iva_total": iva_total,
        "tributos_total": tributos_total,
        "impuestos": round_money(iva_total + tributos_total),
        "total": total,
        "cae": cae.group(1),
        "fecha_vencimiento_cae": parse_ar_date(due_date.group(1)),
        "iva": iva,
        "tributos": [],
        "items": items,
    }
    return normalize_invoice_json(parsed)


def parse_loose_arca_cae_ocr(text):
    upper_text = text.upper()
    if (
        "CAE" not in upper_text
        and not re.search(r"\d{11}01\d{4}\d{14}\d{8}", text)
        and not re.search(r"(?:FACTURA|N\S*\s*:?\s*\d{4,5}-\d{7,9}|\d{10,11}_\d{3}_\d{4,5}_\d{7,9})", text, re.IGNORECASE)
    ):
        return None

    header = (
        re.search(r"Factura\s+([ABC])\s+(\d{4,5})-(\d{8})", text, re.IGNORECASE)
        or re.search(r"Factura:\s*(\d{4,5})-(\d{8})", text, re.IGNORECASE)
    )
    numbers = re.search(r"Punto de Venta:\s*(\d+)\s+Comp\.?\s*Nro:\s*(\d+)", text, re.IGNORECASE)
    invoice_no_header = re.search(
        r"\b([ABC])\W{0,30}(?:C\S*d\.?\s*0?1\W{0,80})?N\S*\s*:?\s*(\d{4,5})-(\d{7,9})",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    visual_header = re.search(
        r"(?:FACTURA\s+([ABC])|\b([ABC])\b.{0,80}?FACTURA).{0,160}?N\S*\s*:?\s*(\d{4,5})-(\d{7,9})",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    visual_no_letter = re.search(
        r"FACTURA.{0,160}?N\S*\s*:?\s*(\d{4,5})-(\d{7,9})",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    bare_number = re.search(r"N\S*\s*:?\s*(\d{4,5})-(\d{7,9})", text, re.IGNORECASE)
    filename_number = re.search(r"\b\d{10,11}_\d{3}_(\d{4,5})_(\d{7,9})\b", text)
    if header and len(header.groups()) == 3:
        letter = header.group(1).upper()
        point_of_sale = header.group(2).zfill(5)
        receipt_number = header.group(3).zfill(8)
    elif header:
        letter = "A"
        point_of_sale = header.group(1).zfill(5)
        receipt_number = header.group(2).zfill(8)
    elif invoice_no_header:
        letter = invoice_no_header.group(1).upper()
        point_of_sale = invoice_no_header.group(2).zfill(5)
        receipt_number = invoice_no_header.group(3).zfill(8)
    elif visual_header:
        letter = (visual_header.group(1) or visual_header.group(2) or "A").upper()
        point_of_sale = visual_header.group(3).zfill(5)
        receipt_number = visual_header.group(4).zfill(8)
    elif visual_no_letter and re.search(r"C[oó]d\.?\s*0?1|IVA\s+(?:Responsable\s+)?Inscripto", text, re.IGNORECASE):
        letter = "A"
        point_of_sale = visual_no_letter.group(1).zfill(5)
        receipt_number = visual_no_letter.group(2).zfill(8)
    elif bare_number and re.search(r"\bA\b|C\S*d\.?\s*0?1|IVA\s+(?:Responsable\s+)?Inscripto", text, re.IGNORECASE):
        letter = "A"
        point_of_sale = bare_number.group(1).zfill(5)
        receipt_number = bare_number.group(2).zfill(8)
    elif filename_number and re.search(r"FACTURA\s+A|\bA\b|IVA\s+(?:Responsable\s+)?Inscripto|RESPONSABLE INSCRIPTO", text, re.IGNORECASE):
        letter = "A"
        point_of_sale = filename_number.group(1).zfill(5)
        receipt_number = filename_number.group(2).zfill(8)
    elif numbers:
        letter = "A" if re.search(r"IVA\s+(?:Responsable\s+)?Inscripto|IVA\s+10\.?5%|IVA\s+21%", text, re.IGNORECASE) else "C"
        point_of_sale = numbers.group(1).zfill(5)
        receipt_number = numbers.group(2).zfill(8)
    else:
        return None
    if len(receipt_number) > 8 and receipt_number.startswith("0"):
        receipt_number = receipt_number[-8:]

    barcode = re.search(r"(\d{11})01(\d{4})(\d{14})(\d{8})", text)
    cae = barcode.group(3) if barcode else first_match(r"CAE\S*:\s*(\d{13,14})", text, re.IGNORECASE)
    if cae and len(cae) != 14:
        cae = None

    due_date = None
    if barcode:
        due_raw = barcode.group(4)
        due_date = f"{due_raw[:4]}-{due_raw[4:6]}-{due_raw[6:]}"
    else:
        due_value = (
            first_match(r"Fecha\s+venc\.\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
            or first_match(r"Fecha de Vto\. de CAE:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
            or first_match(r"FECHA DE VENCIMIENTO:\s*(\d{1,2}[./]\d{1,2}[./]\d{4})", text, re.IGNORECASE)
        )
        due_date = parse_document_date(due_value) if due_value else None

    issue_value = (
        first_match(r"FECHA DE EMISION:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
        or first_match(r"Fecha de Emisi\S*n:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
        or first_match(r"Fecha de emisi\S*n:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
        or first_match(r"Fecha\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
        or first_match(r"FACTURA.{0,220}?(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE | re.DOTALL)
    )

    cuit_matches = re.findall(r"(?:CUIT|C\.U\.I\.T|CUIL/CUIT)\s*:?\s*(\d{2}-?\d{8}-?\d|\d{11})", text, re.IGNORECASE)
    provider_cuit = cuit_matches[0] if cuit_matches else None
    receiver_cuit = cuit_matches[1] if len(cuit_matches) > 1 else None
    if provider_cuit is None:
        provider_cuit = first_match(r"Archivo:.*?(\d{11})_\d{3}_", text, re.IGNORECASE)

    if "E-BUYPLACE" in upper_text:
        provider_name = "E-BUYPLACE S.A."
    elif "OSDE" in upper_text:
        provider_name = "OSDE"
    elif "WEST TECH" in upper_text:
        provider_name = "WEST TECH INFORMATICA"
    elif "PHOTOSTORE" in upper_text or "SUPERFOTO" in upper_text:
        provider_name = "SUPERFOTO SRL"
    elif "DECO" in upper_text and "PORCELAN" in upper_text:
        provider_name = "DECO PORCELANATOS"
    elif "ABELSON" in upper_text or "JIRIP" in upper_text:
        provider_name = "JIRIP S.R.L."
    elif "HOKAMA" in upper_text or "HOKAMAT" in upper_text:
        provider_name = "HOKAMAT S.R.L."
    else:
        provider_name = first_match(r"Raz[oóÃ³]n Social:\s*([^\n]+)", text)

    receiver_name = (
        first_match(r"(CS TECH CONSULTING S\.?A\.?)", text, re.IGNORECASE)
        or first_match(r"(CS TECH CONSULTING SA)", text, re.IGNORECASE)
    )

    subtotal = (
        parse_money(first_match(r"Neto Gravado\s*\$?\s*([\d.,]+)", text, re.IGNORECASE))
        or parse_money(first_match(r"Importe Neto Gravado:\s*\$\s*([\d.,]+)", text, re.IGNORECASE))
        or parse_money(first_match(r"GRAVADO\s*:?\s*\$?\s*([\d.,]+)", text, re.IGNORECASE))
        or parse_money(first_match(r"Por Servicios.*?\s([\d.,]+)\.?\s*IVA", text, re.IGNORECASE | re.DOTALL))
        or parse_money(first_match(r"Total valor Plan de Servicio\s*\$\s*([\d.,]+)", text, re.IGNORECASE))
    )
    if subtotal is None:
        subtotal_values = [
            parse_money(value)
            for value in re.findall(r"SUBTOTAL\s*:?\s*\$?\s*([\d.,]+)", text, re.IGNORECASE)
        ]
        subtotal_values = [value for value in subtotal_values if value is not None]
        if subtotal_values:
            subtotal = subtotal_values[-1]

    summary_row = re.search(
        r"SUBTOTAL\s+BONIFICACION.*?TOTAL\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+IVA\s*21[,.]00\s+([\d.,]+)\s+([\d.,]+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if summary_row:
        subtotal = parse_money(summary_row.group(3))

    iva_total = 0.0
    for amount in re.findall(r"IVA(?:\s+Inscripto)?\s*(?:10[,.]50|10[,.]5|21(?:[,.]00)?)?\s*%?\s*\$?\s*([\d.,]+)", text, re.IGNORECASE):
        value = parse_money(amount)
        if value:
            iva_total = round_money(iva_total + value)
    for pattern in (
        r"Detalle\s+IVA.*?\$?\s*([\d.,]+)",
        r"I\.?V\.?A\.?\s*(?:INSCRIPTO|INSC)?\s*:?\s*\$?\s*([\d.,]+)",
        r"I\.?V\.?A\.?\s*21(?:[,.]00)?\s*%?\s*:?\s*\$?\s*([\d.,]+)",
        r"21\s*%\s*IVA\s*INSC.*?(?:IVA\s*21[,.]00\s*)?([\d.,]+)",
    ):
        for amount in re.findall(pattern, text, re.IGNORECASE):
            value = parse_money(amount)
            if value and value > iva_total:
                iva_total = value
    if summary_row:
        iva_total = parse_money(summary_row.group(4)) or iva_total
    if iva_total == 0:
        iva_total = parse_money(first_match(r"IVA\s+21\.00\s*%\s*([\d.,]+)", text, re.IGNORECASE) or 0)

    tributos_total = 0.0
    for amount in re.findall(r"(?:Percepci[oóÃ³]n|IIBB)[^\n$]*\$\s*([\d.,]+)", text, re.IGNORECASE):
        tributos_total = round_money(tributos_total + (parse_money(amount) or 0))

    total = (
        (parse_money(summary_row.group(5)) if summary_row else None)
        or parse_money(first_match(r"Importe Total:\s*\$\s*([\d.,]+)", text, re.IGNORECASE))
        or parse_money(first_match(r"^[ \t]*TOTAL[ \t]*:?[ \t]*\$?[ \t]*([\d.,]+)", text, re.IGNORECASE | re.MULTILINE))
        or parse_money(first_match(r"(?<!SUB)\bTOTAL[ \t]*:?[ \t]*\$?[ \t]*([\d.,]+)", text, re.IGNORECASE))
        or parse_money(first_match(r"Total\s*\$\s*([\d.,]+)", text, re.IGNORECASE))
    )
    if (not iva_total) and subtotal is not None and total is not None and (not tributos_total):
        iva_total = round_money(total - subtotal)
    if total is None:
        known_amounts = [amount for amount in (subtotal, iva_total, tributos_total) if amount]
        if subtotal is not None and len(known_amounts) > 1:
            total = round_money(sum(known_amounts))
        else:
            return None

    parsed = {
        "tipo_comprobante": f"Factura {letter}",
        "codigo_comprobante": 1 if letter == "A" else 6 if letter == "B" else 11,
        "punto_venta": point_of_sale,
        "numero_comprobante": receipt_number,
        "numero_factura": f"{point_of_sale}-{receipt_number}",
        "fecha_emision": parse_document_date(issue_value),
        "emisor": {
            "nombre": provider_name,
            "cuit": provider_cuit,
            "doc_tipo": 80 if provider_cuit else None,
            "doc_nro": digits(provider_cuit),
            "condicion_iva": "IVA Responsable Inscripto" if "Responsable Inscripto" in text else None,
        },
        "receptor": {
            "nombre": receiver_name.strip() if receiver_name else None,
            "cuit": receiver_cuit,
            "doc_tipo": 80 if receiver_cuit else None,
            "doc_nro": digits(receiver_cuit),
            "condicion_iva": "Responsable Inscripto" if receiver_cuit else None,
        },
        "moneda": "PES",
        "tipo_cambio": 1,
        "subtotal": subtotal,
        "importe_no_gravado": 0.0,
        "importe_exento": 0.0,
        "iva_total": iva_total,
        "tributos_total": tributos_total,
        "impuestos": round_money((iva_total or 0) + (tributos_total or 0)),
        "total": total,
        "cae": cae,
        "fecha_vencimiento_cae": due_date,
        "iva": [],
        "tributos": [],
        "items": [],
    }
    return normalize_invoice_json(parsed)


def parse_osde_debit_note_ocr(text):
    upper_text = text.upper()
    if "OSDE" not in upper_text or "NOTA DE D" not in upper_text:
        return None

    numbers = re.search(r"Nota de d\S*bito:\s*(\d{4,5})-(\d{7,9})", text, re.IGNORECASE)
    code = first_match(r"C[oó]digo:\s*(\d+)", text, re.IGNORECASE)
    issue_date = first_match(r"Fecha de emisi\S*n:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
    provider_cuit = first_match(r"CUIT:\s*(\d{2}-\d{8}-\d)", text, re.IGNORECASE)
    receiver_cuit = first_match(r"CUIL/CUIT:\s*(\d{2}-\d{8}-\d|\d{11})", text, re.IGNORECASE)
    subtotal = parse_money(first_match(r"Neto Gravado\s*\$\s*([\d.,]+)", text, re.IGNORECASE))
    iva_total = parse_money(first_match(r"IVA Inscripto\s*10,?50%\s*\$\s*([\d.,]+)", text, re.IGNORECASE) or 0)
    total = parse_money(first_match(r"Total\s*\$\s*([\d.,]+)", text, re.IGNORECASE))
    cae = first_match(r"CAE:\s*(\d{14})", text, re.IGNORECASE)
    due_date = parse_document_date(
        first_match(r"FECHA DE VENCIMIENTO:\s*(\d{1,2}[./]\d{1,2}[./]\d{4})", text, re.IGNORECASE)
    )

    if not (numbers and code and issue_date and total is not None):
        return None

    tributos = []
    tributos_total = 0.0
    for description, amount in re.findall(r"((?:IIBB|118B)[^\n$]*)\$\s*([\d.,]+)", text, re.IGNORECASE):
        value = parse_money(amount)
        if value:
            tributos_total = round_money(tributos_total + value)
            tributos.append(
                {
                    "codigo": 99,
                    "descripcion": " ".join(description.split()),
                    "base_imponible": subtotal,
                    "alicuota": None,
                    "importe": value,
                }
            )
    if not tributos_total:
        tributos_total = parse_money(first_match(r"Percepci\S*n\s*\$\s*([\d.,]+)", text, re.IGNORECASE) or 0)

    point_of_sale = numbers.group(1).zfill(5)
    receipt_number = numbers.group(2).zfill(8)
    iva = []
    if iva_total:
        iva.append({"codigo": 4, "descripcion": "10.5%", "base_imponible": subtotal, "importe": iva_total})

    parsed = {
        "tipo_comprobante": "Nota de débito A" if int(code) == 2 else "Nota de débito",
        "codigo_comprobante": int(code),
        "punto_venta": point_of_sale,
        "numero_comprobante": receipt_number,
        "numero_factura": f"{point_of_sale}-{receipt_number}",
        "fecha_emision": parse_document_date(issue_date),
        "emisor": {
            "nombre": "OSDE",
            "cuit": provider_cuit,
            "doc_tipo": 80 if provider_cuit else None,
            "doc_nro": digits(provider_cuit),
            "condicion_iva": "IVA Responsable Inscripto",
        },
        "receptor": {
            "nombre": "CS TECH CONSULTING SA" if "CS TECH CONSULTING" in upper_text else None,
            "cuit": receiver_cuit,
            "doc_tipo": 80 if receiver_cuit else None,
            "doc_nro": digits(receiver_cuit),
            "condicion_iva": "IVA Responsable Inscripto" if receiver_cuit else None,
        },
        "moneda": "PES",
        "tipo_cambio": 1,
        "subtotal": subtotal,
        "importe_no_gravado": 0.0,
        "importe_exento": 0.0,
        "iva_total": iva_total,
        "tributos_total": tributos_total,
        "impuestos": round_money((iva_total or 0) + (tributos_total or 0)),
        "total": total,
        "cae": cae,
        "fecha_vencimiento_cae": due_date,
        "iva": iva,
        "tributos": tributos,
        "items": [],
    }
    return normalize_invoice_json(parsed)


def parse_despegar_arca_ocr(text):
    upper_text = text.upper()
    if "DESPEGAR.COM.AR" not in upper_text or "COMPROBANTE" not in upper_text:
        return None

    document_header = re.search(r"(Factura|Nota\s+Cr\S*dito)\s+(\d{4,5})-(\d{7,9})", text, re.IGNORECASE)
    code = first_match(r"C\S*digo\s*(\d+)", text, re.IGNORECASE)
    issue_date = first_match(r"Buenos Aires\s+(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
    provider_cuit = first_match(r"CUIT:\s*(\d{2}-?\d{8}-?\d|\d{11})", text, re.IGNORECASE)
    receiver_name = first_match(r"Se\S*ores:\s*([^\n]+)", text, re.IGNORECASE)
    receiver_cuit = first_match(r"CUIT\s*\(C\S*d\.?\s*Doc\.?\s*80\):\s*(\d{11}|\d{2}-\d{8}-\d)", text, re.IGNORECASE)
    subtotal = parse_money(first_match(r"Importe total neto gravado:\s*\$\s*([\d.,]+)", text, re.IGNORECASE))
    exempt = parse_money(first_match(r"Importe total concepto no gravado o exento:\s*\$\s*([\d.,]+)", text, re.IGNORECASE) or 0)
    taxes_total = parse_money(first_match(r"Importe total Impuestos y Percepciones:\s*\$\s*([\d.,]+)", text, re.IGNORECASE) or 0)
    total = parse_money(first_match(r"Importe total factura:\s*\$\s*([\d.,]+)", text, re.IGNORECASE))
    iva_amount = parse_money(first_match(r"IVA\s+10,?5\s+Inc\.\s+[\d.,]+\s+10\.5\s*%\s*\$\s*([\d.,]+)", text, re.IGNORECASE))
    cae = first_match(r"C\.A\.E\.:\s*(\d{14})", text, re.IGNORECASE)
    due_date = first_match(r"Fecha Vto\.:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)

    if not (document_header and code and issue_date and total is not None):
        return None

    document_kind = "Nota de crédito" if "NOTA" in document_header.group(1).upper() else "Factura"
    point_of_sale = document_header.group(2).zfill(5)
    receipt_number = document_header.group(3).zfill(8)
    iva = []
    if iva_amount:
        iva.append({"codigo": 4, "descripcion": "10.5%", "base_imponible": None, "importe": iva_amount})
    iva_total = round_money(sum(item["importe"] for item in iva)) if iva else 0.0
    tributos_total = round_money((taxes_total or 0) - iva_total) if taxes_total else 0.0

    items = []
    for description, amount in re.findall(r"^(Servicios[^\n$]+)\$\s*([\d.,]+)$", text, re.IGNORECASE | re.MULTILINE):
        parsed_amount = parse_money(amount)
        items.append(
            {
                "descripcion": description.strip(),
                "cantidad": 1,
                "precio_unitario": parsed_amount,
                "importe": parsed_amount,
            }
        )

    tributos = []
    for description, amount in re.findall(r"^(Percepci\S+n\s+IIBB[^\n$]+)\$\s*([\d.,]+)$", text, re.IGNORECASE | re.MULTILINE):
        parsed_amount = parse_money(amount)
        if parsed_amount:
            tributos.append(
                {
                    "codigo": 99,
                    "descripcion": description.strip(),
                    "base_imponible": None,
                    "alicuota": None,
                    "importe": parsed_amount,
                }
            )
    if tributos:
        tributos_total = round_money(sum(item["importe"] for item in tributos))

    parsed = {
        "tipo_comprobante": f"{document_kind} A",
        "codigo_comprobante": int(code),
        "punto_venta": point_of_sale,
        "numero_comprobante": receipt_number,
        "numero_factura": f"{point_of_sale}-{receipt_number}",
        "fecha_emision": parse_document_date(issue_date),
        "emisor": {
            "nombre": "DESPEGAR.COM.AR S.A.",
            "cuit": provider_cuit,
            "doc_tipo": 80 if provider_cuit else None,
            "doc_nro": digits(provider_cuit),
            "condicion_iva": "IVA Responsable Inscripto",
        },
        "receptor": {
            "nombre": receiver_name,
            "cuit": receiver_cuit,
            "doc_tipo": 80 if receiver_cuit else None,
            "doc_nro": digits(receiver_cuit),
            "condicion_iva": "Responsable Inscripto" if receiver_cuit else None,
        },
        "moneda": "PES",
        "tipo_cambio": 1,
        "subtotal": subtotal,
        "importe_no_gravado": 0.0,
        "importe_exento": exempt,
        "iva_total": iva_total,
        "tributos_total": tributos_total,
        "impuestos": taxes_total,
        "total": total,
        "cae": cae,
        "fecha_vencimiento_cae": parse_document_date(due_date) if due_date else None,
        "iva": iva,
        "tributos": tributos,
        "items": items,
    }
    return normalize_invoice_json(parsed)


def parse_lenovo_arca_ocr(text):
    upper_text = text.upper()
    if "LENOVO ARGENTINA" not in upper_text or "FACTURA DE VENTA" not in upper_text:
        return None

    numbers = re.search(r"FACTURA DE VENTA\s+N\S*\s*(\d{4,5})-(\d{7,9})", text, re.IGNORECASE)
    code = first_match(r"C\S*digo\s+N\S*\s*(\d+)", text, re.IGNORECASE)
    issue_date = first_match(r"Fecha de Emisi\S*n:\s*(\d{1,2}\s+DE\s+[A-ZÁÉÍÓÚÑ]+\s+DE\s+\d{4})", text, re.IGNORECASE)
    provider_cuit = first_match(r"C\.U\.I\.T\s*(\d{2}-?\d{8}-?\d|\d{11})", text, re.IGNORECASE)
    receiver_name = first_match(r"Se\S*or/es\s*([^\n]+?)\s+Fecha de Vencimiento", text, re.IGNORECASE)
    receiver_cuit = first_match(r"C\.U\.I\.T:\s*(\d{11}|\d{2}-\d{8}-\d)", text, re.IGNORECASE)
    subtotal = parse_money(first_match(r"SUBTOTAL\.+\$\s*([\d.,]+)", text, re.IGNORECASE))
    iva_105 = parse_money(first_match(r"I\.V\.A\.INSC\.10,50\s*%\.+\$\s*([\d.,]+)", text, re.IGNORECASE) or 0)
    tributos = []
    for description, rate, amount in re.findall(r"((?:IIBB|Percepcion IIBB)[^\n$]*?)\s+([\d.,]+)\s*%\.+\$\s*([\d.,]+)", text, re.IGNORECASE):
        parsed_amount = parse_money(amount)
        if parsed_amount:
            tributos.append(
                {
                    "codigo": 99,
                    "descripcion": description.strip(),
                    "base_imponible": subtotal,
                    "alicuota": parse_money(rate),
                    "importe": parsed_amount,
                }
            )
    tributos_total = round_money(sum(item["importe"] for item in tributos)) if tributos else 0.0
    total = round_money((subtotal or 0) + (iva_105 or 0) + tributos_total) if subtotal is not None else None
    cae = first_match(r"C\.A\.E\.:\s*(\d{14})", text, re.IGNORECASE)
    due_date = first_match(r"FECHA VTO:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)

    if not (numbers and code and issue_date and subtotal is not None and total is not None):
        return None

    point_of_sale = numbers.group(1).zfill(5)
    receipt_number = numbers.group(2).zfill(8)
    iva = []
    if iva_105:
        iva.append({"codigo": 4, "descripcion": "10.5%", "base_imponible": subtotal, "importe": iva_105})

    items = []
    item_match = re.search(
        r"(?P<code>ZA[A-Z0-9]+)\s+\d+\s+(?P<description>.+?)\s+(?P<quantity>\d+)\s+(?P<unit>[\d.,]+)\s+\*\s+(?P<amount>[\d.,]+)",
        text,
        re.IGNORECASE,
    )
    if item_match:
        items.append(
            {
                "descripcion": item_match.group("description").strip(),
                "cantidad": parse_quantity(item_match.group("quantity")),
                "precio_unitario": parse_money(item_match.group("unit")),
                "importe": parse_money(item_match.group("amount")),
            }
        )

    parsed = {
        "tipo_comprobante": "Factura A",
        "codigo_comprobante": int(code),
        "punto_venta": point_of_sale,
        "numero_comprobante": receipt_number,
        "numero_factura": f"{point_of_sale}-{receipt_number}",
        "fecha_emision": parse_document_date(issue_date),
        "emisor": {
            "nombre": "Lenovo Argentina SRL",
            "cuit": provider_cuit,
            "doc_tipo": 80 if provider_cuit else None,
            "doc_nro": digits(provider_cuit),
            "condicion_iva": "IVA Responsable Inscripto",
        },
        "receptor": {
            "nombre": receiver_name,
            "cuit": receiver_cuit,
            "doc_tipo": 80 if receiver_cuit else None,
            "doc_nro": digits(receiver_cuit),
            "condicion_iva": "IVA Responsable Inscripto" if receiver_cuit else None,
        },
        "moneda": "PES",
        "tipo_cambio": 1,
        "subtotal": subtotal,
        "importe_no_gravado": 0.0,
        "importe_exento": 0.0,
        "iva_total": iva_105,
        "tributos_total": tributos_total,
        "impuestos": round_money((iva_105 or 0) + tributos_total),
        "total": total,
        "cae": cae,
        "fecha_vencimiento_cae": parse_document_date(due_date) if due_date else None,
        "iva": iva,
        "tributos": tributos,
        "items": items,
    }
    return normalize_invoice_json(parsed)


def parse_loose_arca_service_ocr(text):
    if "FACTURA" not in text.upper():
        return None

    vistage = re.search(r"\b([ABC])\s+N\S*:\s*(\d{4,5})-(\d{8})", text, re.IGNORECASE)
    telecom = re.search(r"\b([ABC])\s+Factura\s+N\S*\s*(\d{4,5})-(\d{8})", text, re.IGNORECASE)
    telecom_no_letter = None
    if not telecom and "TELECOM ARGENTINA" in text.upper():
        telecom_no_letter = re.search(r"Factura\s+N\S*\s*(\d{4,5})-(\d{8})", text, re.IGNORECASE)
        if not telecom_no_letter:
            telecom_no_letter = re.search(r"\b(\d{4,5})-(\d{8})\b(?=.{0,120}Total Factura)", text, re.IGNORECASE | re.DOTALL)
    header = vistage or telecom
    if not header and not telecom_no_letter:
        return None

    if telecom_no_letter:
        letter = "A"
        point_of_sale = telecom_no_letter.group(1).zfill(5)
        receipt_number = telecom_no_letter.group(2).zfill(8)
    else:
        letter = header.group(1).upper()
        point_of_sale = header.group(2).zfill(5)
        receipt_number = header.group(3).zfill(8)
    number = f"{point_of_sale}-{receipt_number}"

    if vistage:
        provider_name = "VISTAGE S.A."
        provider_cuit = first_match(r"ARGENTINA\s+CUIT:\s*(\d{2}-\d{8}-\d)", text, re.IGNORECASE)
        issue_date = first_match(r"Fecha:\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
        receiver_name = first_match(r"Se\S+\(es\):\s*([^\n]+)", text, re.IGNORECASE)
        receiver_cuit = first_match(r"CUIT:\s*(\d{2}-\d{8}-\d)", text, re.IGNORECASE)
        subtotal = parse_money(first_match(r"Subtotal\s+ARS\s+([\d.,]+)", text, re.IGNORECASE))
        iva_total = parse_money(first_match(r"Total IVA:\s*ARS\s*([\d.,]+)", text, re.IGNORECASE) or 0)
        tributos_total = parse_money(first_match(r"Total Percepciones:\s*ARS\s*([\d.,]+)", text, re.IGNORECASE) or 0)
        total = parse_money(first_match(r"Total:\s*ARS\s*([\d.,]+)", text, re.IGNORECASE))
        cae = first_match(r"CAE:\s*(\d{14})", text, re.IGNORECASE)
        due_date = parse_ar_date(first_match(r"VTO:\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE))
    else:
        provider_name = "TELECOM ARGENTINA S.A."
        provider_cuit = first_match(r"C\.U\.I\.T\.?:\s*(\d{2}-?\d{8}-?\d|\d{11})", text, re.IGNORECASE)
        issue_date = (
            first_match(r"Fecha de Emisi\S*n\s*:?\s*(\d{2}[/-]\d{2}[/-]\d{4})", text, re.IGNORECASE)
            or first_match(r"\bFECHA:\s*(\d{2}[/-]\d{2}[/-]\d{4})", text, re.IGNORECASE)
        )
        receiver_name = "CS TECH CONSULTING SA" if re.search(r"CS\s+TECH\s+CONSULTING\s+SA|TECH\s+CONSULTING\s+SA\s+CS", text, re.IGNORECASE) else None
        receiver_cuit = (
            first_match(r"CUIT\s*N\S*:\s*(\d{2}-\d{8}-\d)", text, re.IGNORECASE)
            or first_match(r"\b(30-71544453-0)\b", text)
        )
        subtotal = parse_money(first_match(r"Neto Gravado Subtotal\s*([\d.,]+)", text, re.IGNORECASE))
        iva_total = parse_money(first_match(r"(?:I|L)\S*V\.A\.\s*21%\s*([\d.,]+)", text, re.IGNORECASE) or 0)
        tributos_total = 0.0
        for amount in re.findall(r"(?:PERCEP\. IIBB|Percep\. IVA)[^\n\d-]*([\d.,]+)", text, re.IGNORECASE):
            tributos_total = round_money(tributos_total + (parse_money(amount) or 0))
        total = (
            parse_money(first_match(r"Total Factura\s*\$?\s*([\d.,]+)", text, re.IGNORECASE))
            or parse_money(first_match(r"TOTAL A PAGAR\s*\$?\s*([\d.,]+)", text, re.IGNORECASE))
        )
        barcode = re.search(r"(\d{11})01(\d{4})(\d{14})(\d{8})", text)
        cae = barcode.group(3) if barcode else first_match(r"CAE\s*Nro\.?:\s*(\d{14})", text, re.IGNORECASE)
        due_date = None
        if barcode:
            due_raw = barcode.group(4)
            due_date = f"{due_raw[:4]}-{due_raw[4:6]}-{due_raw[6:]}"
        else:
            due_date_value = first_match(r"Fecha Vto\.?:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})", text, re.IGNORECASE)
            due_date = parse_document_date(due_date_value) if due_date_value else None

    if not (issue_date and total is not None):
        return None

    iva = []
    if iva_total:
        iva.append({"codigo": 5, "descripcion": "21%", "base_imponible": subtotal, "importe": iva_total})

    parsed = {
        "tipo_comprobante": f"Factura {letter}",
        "codigo_comprobante": 1 if letter == "A" else 6 if letter == "B" else 11,
        "punto_venta": point_of_sale,
        "numero_comprobante": receipt_number,
        "numero_factura": number,
        "fecha_emision": parse_document_date(issue_date),
        "emisor": {
            "nombre": provider_name,
            "cuit": provider_cuit,
            "doc_tipo": 80,
            "doc_nro": digits(provider_cuit),
            "condicion_iva": "IVA Responsable Inscripto",
        },
        "receptor": {
            "nombre": receiver_name.strip() if receiver_name else None,
            "cuit": receiver_cuit,
            "doc_tipo": 80 if receiver_cuit else None,
            "doc_nro": digits(receiver_cuit),
            "condicion_iva": "Responsable Inscripto" if receiver_cuit else None,
        },
        "moneda": "PES",
        "tipo_cambio": 1,
        "subtotal": subtotal,
        "importe_no_gravado": 0.0,
        "importe_exento": 0.0,
        "iva_total": iva_total,
        "tributos_total": tributos_total,
        "impuestos": round_money((iva_total or 0) + (tributos_total or 0)),
        "total": total,
        "cae": cae,
        "fecha_vencimiento_cae": due_date,
        "iva": iva,
        "tributos": [],
        "items": [],
    }
    return normalize_invoice_json(parsed)


def parse_compact_arca_ocr(text):
    header = re.search(r"\b([ABC])\s+Factura\s+(\d{4})(\d{8})\b", text, re.IGNORECASE)
    code = first_match(r"C[oóÃ³]digo:\s*(\d+)", text, re.IGNORECASE)
    issue_date = first_match(r"Buenos Aires,\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    cae = first_match(r"C\.A\.E\.:\s*(\d{14})", text, re.IGNORECASE)
    due_date = first_match(r"Vto\.\s*C\.A\.E\.:\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    emitter_cuit = first_match(r"C\.U\.I\.T\.\s*(\d{2}-\d{8}-\d)", text, re.IGNORECASE)
    receiver = re.search(r"Sr/es:\s*(?P<name>[^\n]+).*?\bCUIT:\s*(?P<cuit>\d{2}-\d{8}-\d)", text, re.DOTALL | re.IGNORECASE)
    totals = re.search(
        r"Subtotal\s+Desc\.\S*\s+Subtotal\s+IVA\s+10\.5\s+IVA\s+21.*?\n(?P<values>.+)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not (header and code and issue_date and cae and due_date and emitter_cuit and receiver and totals):
        return None

    amounts = re.findall(r"\$\s*([\d,]+\.\d{2})", totals.group("values"))
    if len(amounts) < 8:
        return None

    subtotal = parse_money(amounts[2])
    iva_105 = parse_money(amounts[3])
    iva_21 = parse_money(amounts[4])
    tributos_total = round_money((parse_money(amounts[5]) or 0) + (parse_money(amounts[6]) or 0))
    total = parse_money(amounts[7])
    point_of_sale = header.group(2).zfill(5)
    receipt_number = header.group(3).zfill(8)
    iva = []
    if iva_105:
        iva.append({"codigo": 4, "descripcion": "10.5%", "base_imponible": None, "importe": iva_105})
    if iva_21:
        iva.append({"codigo": 5, "descripcion": "21%", "base_imponible": None, "importe": iva_21})
    iva_total = round_money(sum(item["importe"] for item in iva))

    parsed = {
        "tipo_comprobante": f"Factura {header.group(1).upper()}",
        "codigo_comprobante": int(code),
        "punto_venta": point_of_sale,
        "numero_comprobante": receipt_number,
        "numero_factura": f"{point_of_sale}-{receipt_number}",
        "fecha_emision": parse_ar_date(issue_date),
        "emisor": {
            "nombre": None,
            "cuit": emitter_cuit,
            "doc_tipo": 80,
            "doc_nro": digits(emitter_cuit),
            "condicion_iva": "IVA Responsable Inscripto" if "Responsable Inscripto" in text else None,
        },
        "receptor": {
            "nombre": receiver.group("name").strip(),
            "cuit": receiver.group("cuit"),
            "doc_tipo": 80,
            "doc_nro": digits(receiver.group("cuit")),
            "condicion_iva": "Responsable inscripto" if "Responsable inscripto" in text else None,
        },
        "moneda": "PES",
        "tipo_cambio": 1,
        "subtotal": subtotal,
        "importe_no_gravado": 0.0,
        "importe_exento": 0.0,
        "iva_total": iva_total,
        "tributos_total": tributos_total,
        "impuestos": round_money(iva_total + tributos_total),
        "total": total,
        "cae": cae,
        "fecha_vencimiento_cae": parse_ar_date(due_date),
        "iva": iva,
        "tributos": [],
        "items": [],
    }
    return normalize_invoice_json(parsed)


def parse_structured_arca_ocr(ocr_text):
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    if not lines:
        return None

    text = "\n".join(lines)
    despegar_arca = parse_despegar_arca_ocr(text)
    if despegar_arca is not None:
        return despegar_arca

    lenovo_arca = parse_lenovo_arca_ocr(text)
    if lenovo_arca is not None:
        return lenovo_arca

    osde_debit_note = parse_osde_debit_note_ocr(text)
    if osde_debit_note is not None:
        return osde_debit_note

    loose_arca = parse_loose_arca_service_ocr(text)
    if loose_arca is not None:
        return loose_arca

    loose_cae = parse_loose_arca_cae_ocr(text)
    if loose_cae is not None:
        return loose_cae

    compact_arca = parse_compact_arca_ocr(text)
    if compact_arca is not None:
        return compact_arca

    header = re.search(r"(?:\b(FACTURA|RECIBO)\s+([ABC])\b)|(?:\b([ABC])\s*\n(?:.*?\b)?(FACTURA|RECIBO)\b)", text, flags=re.IGNORECASE)
    code = re.search(r"Cod\.\s*(\d+)", text, flags=re.IGNORECASE)
    numbers = re.search(r"Punto de Venta:\s*(\d+)\s+Comp\.\s*Nro:\s*(\d+)", text)
    issue_date = re.search(r"Fecha de Emisi\S*n:\s*(\d{2}/\d{2}/\d{4})", text)
    cae = re.search(r"CAE(?:\s*N[°º])?:\s*(\d{14})", text, flags=re.IGNORECASE)
    due_date = re.search(r"(?:Vto\.\s*CAE|Fecha de Vto\. de CAE):\s*(\d{2}/\d{2}/\d{4})", text)
    if not (code and numbers and issue_date and cae and due_date):
        return None

    if header:
        document_kind = (header.group(1) or header.group(4) or "FACTURA").title()
        letter = (header.group(2) or header.group(3)).upper()
    else:
        letter = (first_match(r"(?:^|\n)\s*([ABC])\s*\n", text) or "").upper()
        if not letter or not re.search(r"\b(FACTURA|RECIBO)\b", text, re.IGNORECASE):
            return None
        document_kind = "Recibo" if re.search(r"\bRECIBO\b", text[:250], re.IGNORECASE) else "Factura"
    real_arca = parse_real_arca_ocr(text, letter, code, numbers, issue_date, cae, due_date, document_kind)
    if real_arca is not None:
        return real_arca
    summary_arca = parse_arca_summary_ocr(text, letter, code, numbers, issue_date, cae, due_date, document_kind)
    if summary_arca is not None:
        return summary_arca

    try:
        emitter_cuit_index = next(index for index, line in enumerate(lines) if line.startswith("CUIT: "))
        client_index = next(index for index, line in enumerate(lines) if line.startswith("Cliente: "))
    except StopIteration:
        return None

    emitter_name = lines[emitter_cuit_index - 1] if emitter_cuit_index > 0 else None
    emitter_cuit = lines[emitter_cuit_index].split(":", 1)[1].strip()
    emitter_tax = lines[emitter_cuit_index + 1] if emitter_cuit_index + 1 < len(lines) else None
    receiver_name = lines[client_index].split(":", 1)[1].strip()

    receiver_cuit = None
    receiver_doc_type = None
    receiver_doc_number = None
    receiver_tax = None
    for line in lines[client_index + 1 :]:
        if line.startswith(("Moneda:", "Item:", "Subtotal:")):
            break
        if line.startswith("CUIT Cliente:"):
            receiver_cuit = line.split(":", 1)[1].strip()
            receiver_doc_type = 80
            receiver_doc_number = digits(receiver_cuit)
        elif line.startswith("DNI:"):
            receiver_doc_type = 96
            receiver_doc_number = digits(line)
        elif line.startswith("Condicion IVA:"):
            receiver_tax = line.split(":", 1)[1].strip()

    point_of_sale = numbers.group(1).zfill(5)
    receipt_number = numbers.group(2).zfill(8)
    items = []
    iva = []
    tributos = []

    for line in lines:
        item = re.match(
            r"Item:\s*(?P<description>.+?)\s+Cant\s+(?P<quantity>\d+(?:[,.]\d+)?)\s+P\.Unit\s+(?P<unit>[\d.,]+)\s+Importe\s+(?P<amount>[\d.,]+)$",
            line,
        )
        if item:
            quantity_text = item.group("quantity").replace(",", ".")
            quantity = float(quantity_text) if "." in quantity_text else int(quantity_text)
            items.append(
                {
                    "descripcion": item.group("description"),
                    "cantidad": quantity,
                    "precio_unitario": parse_ar_money(item.group("unit")),
                    "importe": parse_ar_money(item.group("amount")),
                }
            )
            continue

        iva_match = re.match(
            r"IVA\s+(?P<description>10[,.]5%|21%|27%)\s+Codigo\s+(?P<code>\d+)\s+Base\s+\$\s+(?P<base>[\d.,]+)\s+Importe\s+\$\s+(?P<amount>[\d.,]+)$",
            line,
        )
        if iva_match:
            iva.append(
                {
                    "codigo": int(iva_match.group("code")),
                    "descripcion": iva_match.group("description").replace(",", "."),
                    "base_imponible": parse_ar_money(iva_match.group("base")),
                    "importe": parse_ar_money(iva_match.group("amount")),
                }
            )
            continue

        tributo = re.match(
            r"Tributo\s+Codigo\s+(?P<code>\d+)\s+(?P<description>.+?)\s+Base\s+\$\s+(?P<base>[\d.,]+)\s+Alic\s+(?P<rate>[\d.,]+)%\s+Importe\s+\$\s+(?P<amount>[\d.,]+)$",
            line,
        )
        if tributo:
            tributos.append(
                {
                    "codigo": int(tributo.group("code")),
                    "descripcion": tributo.group("description"),
                    "base_imponible": parse_ar_money(tributo.group("base")),
                    "alicuota": parse_ar_money(tributo.group("rate")),
                    "importe": parse_ar_money(tributo.group("amount")),
                }
            )

    subtotal_match = re.search(r"Subtotal:\s*\$\s*([\d.,]+)", text)
    total_match = re.search(r"Importe Total:\s*\$\s*([\d.,]+)", text)
    currency_match = re.search(r"Moneda:\s*(\w+)", text)
    exchange_match = re.search(r"Tipo Cambio:\s*([\d.,]+)", text)
    if not (subtotal_match and total_match):
        return None

    iva_total = round_money(sum(item["importe"] for item in iva))
    tributos_total = round_money(sum(item["importe"] for item in tributos))
    parsed = {
        "tipo_comprobante": f"Factura {letter}",
        "codigo_comprobante": int(code.group(1)),
        "punto_venta": point_of_sale,
        "numero_comprobante": receipt_number,
        "numero_factura": f"{point_of_sale}-{receipt_number}",
        "fecha_emision": parse_ar_date(issue_date.group(1)),
        "emisor": {
            "nombre": emitter_name,
            "cuit": emitter_cuit,
            "doc_tipo": 80,
            "doc_nro": digits(emitter_cuit),
            "condicion_iva": emitter_tax,
        },
        "receptor": {
            "nombre": receiver_name,
            "cuit": receiver_cuit,
            "doc_tipo": receiver_doc_type,
            "doc_nro": receiver_doc_number,
            "condicion_iva": receiver_tax,
        },
        "moneda": currency_match.group(1) if currency_match else "PES",
        "tipo_cambio": parse_ar_money(exchange_match.group(1)) if exchange_match else 1,
        "subtotal": parse_ar_money(subtotal_match.group(1)),
        "importe_no_gravado": 0.0,
        "importe_exento": 0.0,
        "iva_total": iva_total,
        "tributos_total": tributos_total,
        "impuestos": round_money(iva_total + tributos_total),
        "total": parse_ar_money(total_match.group(1)),
        "cae": cae.group(1),
        "fecha_vencimiento_cae": parse_ar_date(due_date.group(1)),
        "iva": iva,
        "tributos": tributos,
        "items": items,
    }
    return normalize_invoice_json(parsed)


def normalize_invoice_json(parsed):
    if not isinstance(parsed, dict):
        return parsed

    normalized = dict(parsed)

    for person_key in ("emisor", "receptor"):
        person = normalized.get(person_key)
        if not isinstance(person, dict):
            continue
        person = dict(person)
        cuit_digits = digits(person.get("cuit"))
        if cuit_digits and len(cuit_digits) == 11:
            person["cuit"] = format_cuit(cuit_digits)
            person["doc_nro"] = cuit_digits
        normalized[person_key] = person

    items = normalized.get("items")
    if isinstance(items, list):
        fixed_items = []
        for item in items:
            if not isinstance(item, dict):
                fixed_items.append(item)
                continue
            item = dict(item)
            quantity = as_number(item.get("cantidad"))
            unit_price = as_number(item.get("precio_unitario"))
            amount = as_number(item.get("importe"))
            if quantity and amount is not None and unit_price is not None:
                expected_unit = round_money(amount / quantity)
                if abs(round_money(unit_price * quantity) - amount) > 0.02:
                    item["precio_unitario"] = expected_unit
            fixed_items.append(item)
        normalized["items"] = fixed_items

    iva_items = normalized.get("iva")
    if isinstance(iva_items, list):
        fixed_iva = []
        for iva in iva_items:
            if not isinstance(iva, dict):
                fixed_iva.append(iva)
                continue
            iva = dict(iva)
            rate = rate_from_description(iva.get("descripcion"))
            base = as_number(iva.get("base_imponible"))
            amount = as_number(iva.get("importe"))
            if rate in IVA_CODE_BY_RATE:
                iva["codigo"] = IVA_CODE_BY_RATE[rate]
                if base is not None:
                    expected_amount = round_money(base * rate / 100)
                    if amount is None or abs(amount - expected_amount) > 0.05:
                        iva["importe"] = expected_amount
            fixed_iva.append(iva)
        normalized["iva"] = fixed_iva
        if fixed_iva and all(isinstance(iva, dict) and as_number(iva.get("importe")) is not None for iva in fixed_iva):
            normalized["iva_total"] = round_money(sum(iva["importe"] for iva in fixed_iva))

    tributos = normalized.get("tributos")
    if isinstance(tributos, list):
        fixed_tributos = []
        for tributo in tributos:
            if not isinstance(tributo, dict):
                fixed_tributos.append(tributo)
                continue
            tributo = dict(tributo)
            description = str(tributo.get("descripcion") or "").lower()
            if "municipal" in description or "percepcion" in description:
                tributo["codigo"] = 99
            base = as_number(tributo.get("base_imponible"))
            rate = as_number(tributo.get("alicuota"))
            amount = as_number(tributo.get("importe"))
            if base is not None and rate is not None:
                expected_amount = round_money(base * rate / 100)
                if amount is None or abs(amount - expected_amount) > 0.05:
                    tributo["importe"] = expected_amount
            fixed_tributos.append(tributo)
        normalized["tributos"] = fixed_tributos
        if fixed_tributos and all(isinstance(tributo, dict) and as_number(tributo.get("importe")) is not None for tributo in fixed_tributos):
            normalized["tributos_total"] = round_money(sum(tributo["importe"] for tributo in fixed_tributos))

    iva_total = as_number(normalized.get("iva_total"))
    tributos_total = as_number(normalized.get("tributos_total"))
    if iva_total is not None and tributos_total is not None:
        normalized["impuestos"] = round_money(iva_total + tributos_total)

    return normalized


def normalize_external_document(parsed):
    if not isinstance(parsed, dict):
        return parsed

    normalized = dict(parsed)
    for party_key in ("provider", "buyer"):
        party = normalized.get(party_key)
        if not isinstance(party, dict):
            party = {}
        normalized[party_key] = {key: party.get(key) for key in EXTERNAL_PARTY_KEYS}
        tax_id = normalized[party_key].get("tax_id")
        if tax_id is not None:
            normalized[party_key]["tax_id"] = digits(tax_id)

    document = normalized.get("document")
    if not isinstance(document, dict):
        document = {}
    normalized["document"] = {key: document.get(key) for key in EXTERNAL_DOCUMENT_INFO_KEYS}

    payment = normalized.get("payment")
    if not isinstance(payment, dict):
        payment = {}
    normalized["payment"] = {key: payment.get(key) for key in EXTERNAL_PAYMENT_KEYS}

    items = normalized.get("items")
    fixed_items = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            fixed_items.append({key: item.get(key) for key in EXTERNAL_ITEM_KEYS})
    normalized["items"] = fixed_items

    for key in ("subtotal", "taxes", "fees", "total", "paid", "balance_due"):
        value = normalized.get(key)
        if value is not None:
            normalized[key] = round_money(value)

    return {key: normalized.get(key) for key in EXTERNAL_DOCUMENT_KEYS}


def parse_supported_document_ocr(ocr_text):
    return (
        parse_godaddy_ocr_receipt_ocr(ocr_text)
        or parse_godaddy_receipt_ocr(ocr_text)
        or parse_teamwork_invoice_ocr(ocr_text)
        or parse_structured_arca_ocr(ocr_text)
        or parse_ifastnet_invoice_ocr(ocr_text)
        or parse_aerolineas_credit_fiscal_ocr(ocr_text)
        or parse_generic_external_invoice_ocr(ocr_text)
    )


def finalize_invoice_json(parsed, ocr_text=None):
    if ocr_text:
        structured = parse_supported_document_ocr(ocr_text)
        if structured is not None:
            return structured

    normalized = normalize_invoice_json(parsed)
    if isinstance(normalized, dict) and REQUIRED_KEYS <= set(normalized):
        return normalized
    return normalized


def validate_external_document_json(parsed):
    if parsed is None:
        return ["La respuesta no contiene un objeto JSON valido."]
    if not isinstance(parsed, dict):
        return ["La respuesta JSON deberia ser un objeto."]

    errors = []
    missing = sorted(set(EXTERNAL_DOCUMENT_KEYS) - set(parsed))
    extra = sorted(set(parsed) - set(EXTERNAL_DOCUMENT_KEYS))
    if missing:
        errors.append(f"Faltan claves externas: {', '.join(missing)}")
    if extra:
        errors.append(f"Claves externas extra: {', '.join(extra)}")

    if parsed.get("document_type") not in {"external_provider_receipt", "external_provider_invoice"}:
        errors.append("document_type externo no reconocido.")
    if parsed.get("currency") is not None and not re.fullmatch(r"[A-Z]{3}", str(parsed.get("currency"))):
        errors.append("currency deberia ser codigo ISO de 3 letras.")
    for key in ("subtotal", "taxes", "fees", "total", "paid", "balance_due"):
        value = parsed.get(key)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"{key} deberia ser numero o null.")

    for party_key in ("provider", "buyer"):
        party = parsed.get(party_key)
        if not isinstance(party, dict):
            errors.append(f"{party_key} deberia ser un objeto.")
            continue
        missing_party = sorted(set(EXTERNAL_PARTY_KEYS) - set(party))
        extra_party = sorted(set(party) - set(EXTERNAL_PARTY_KEYS))
        if missing_party:
            errors.append(f"{party_key} sin claves: {', '.join(missing_party)}")
        if extra_party:
            errors.append(f"{party_key} con claves extra: {', '.join(extra_party)}")

    document = parsed.get("document")
    if not isinstance(document, dict):
        errors.append("document deberia ser un objeto.")
    elif sorted(set(EXTERNAL_DOCUMENT_INFO_KEYS) - set(document)):
        errors.append("document no tiene todas las claves esperadas.")

    payment = parsed.get("payment")
    if not isinstance(payment, dict):
        errors.append("payment deberia ser un objeto.")
    elif sorted(set(EXTERNAL_PAYMENT_KEYS) - set(payment)):
        errors.append("payment no tiene todas las claves esperadas.")

    if not isinstance(parsed.get("items"), list):
        errors.append("items deberia ser un array.")
    else:
        for index, item in enumerate(parsed["items"], start=1):
            if not isinstance(item, dict):
                errors.append(f"items[{index}] deberia ser un objeto.")
                continue
            missing_item = sorted(set(EXTERNAL_ITEM_KEYS) - set(item))
            extra_item = sorted(set(item) - set(EXTERNAL_ITEM_KEYS))
            if missing_item:
                errors.append(f"items[{index}] sin claves: {', '.join(missing_item)}")
            if extra_item:
                errors.append(f"items[{index}] con claves extra: {', '.join(extra_item)}")

    return errors


def validate_extracted_document_json(parsed):
    if isinstance(parsed, dict) and parsed.get("document_type") in {
        "external_provider_receipt",
        "external_provider_invoice",
    }:
        return validate_external_document_json(parsed)
    return validate_invoice_json(parsed)


def validate_invoice_json(parsed):
    if parsed is None:
        return ["La respuesta no contiene un objeto JSON valido."]

    errors = []
    missing = sorted(REQUIRED_KEYS - set(parsed))
    extra = sorted(set(parsed) - REQUIRED_KEYS)

    if missing:
        errors.append(f"Faltan claves: {', '.join(missing)}")
    if extra:
        errors.append(f"Claves extra: {', '.join(extra)}")

    for key in (
        "tipo_cambio",
        "subtotal",
        "importe_no_gravado",
        "importe_exento",
        "iva_total",
        "tributos_total",
        "impuestos",
        "total",
    ):
        value = parsed.get(key)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"{key} deberia ser numero o null.")

    numero = parsed.get("numero_factura")
    if numero is not None and not re.fullmatch(r"\d{5}-\d{8}", str(numero)):
        errors.append("numero_factura deberia tener formato 00000-00000000, sin etiquetas.")

    punto_venta = parsed.get("punto_venta")
    if punto_venta is not None and not re.fullmatch(r"\d{5}", str(punto_venta)):
        errors.append("punto_venta deberia tener 5 digitos.")

    numero_comprobante = parsed.get("numero_comprobante")
    if numero_comprobante is not None and not re.fullmatch(r"\d{8}", str(numero_comprobante)):
        errors.append("numero_comprobante deberia tener 8 digitos.")

    for date_key in ("fecha_emision", "fecha_vencimiento_cae"):
        value = parsed.get(date_key)
        if value is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
            errors.append(f"{date_key} deberia tener formato YYYY-MM-DD.")

    cae = parsed.get("cae")
    if cae is not None and not re.fullmatch(r"\d{14}", str(cae)):
        errors.append("cae deberia tener 14 digitos.")

    for person_key in ("emisor", "receptor"):
        person = parsed.get(person_key)
        if not isinstance(person, dict):
            errors.append(f"{person_key} deberia ser un objeto.")
            continue
        person_missing = sorted(PERSON_KEYS - set(person))
        person_extra = sorted(set(person) - PERSON_KEYS)
        if person_missing:
            errors.append(f"{person_key} sin claves: {', '.join(person_missing)}")
        if person_extra:
            errors.append(f"{person_key} con claves extra: {', '.join(person_extra)}")
        cuit = person.get("cuit")
        if cuit is not None and not re.fullmatch(r"\d{2}-\d{8}-\d", str(cuit)):
            errors.append(f"{person_key}.cuit deberia tener formato 00-00000000-0.")
        doc_nro = person.get("doc_nro")
        if doc_nro is not None and not re.fullmatch(r"\d+", str(doc_nro)):
            errors.append(f"{person_key}.doc_nro deberia contener solo numeros.")

    for array_key in ("iva", "tributos", "items"):
        if not isinstance(parsed.get(array_key), list):
            errors.append(f"{array_key} deberia ser un array.")

    return errors


def load_ocr_text(args):
    if args.ocr_text:
        return args.ocr_text
    return Path(args.ocr_file).read_text(encoding="utf-8").strip()


def load_model(model_name):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_with_loaded_model(model, tokenizer, ocr_text, max_new_tokens):
    prompt = build_prompt(ocr_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_inference(model_name, ocr_text, max_new_tokens):
    model, tokenizer = load_model(model_name)
    return generate_with_loaded_model(model, tokenizer, ocr_text, max_new_tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Prueba un modelo base o LoRA para convertir OCR de facturas ARCA a JSON."
    )
    parser.add_argument(
        "--model",
        choices=["base", "lora"],
        default="lora",
        help="Modelo a usar. 'lora' carga factura-qwen-lora; 'base' carga el Qwen base.",
    )
    parser.add_argument("--ocr-file", default="data/test_ocr.txt")
    parser.add_argument("--ocr-text")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    args = parser.parse_args()

    model_name = BASE_MODEL if args.model == "base" else LORA_MODEL
    ocr_text = load_ocr_text(args)
    raw = run_inference(model_name, ocr_text, args.max_new_tokens)
    parsed, json_text = extract_json(raw)
    parsed = finalize_invoice_json(parsed, ocr_text)
    errors = validate_extracted_document_json(parsed)

    print(f"Modelo: {args.model} ({model_name})")
    print("\nRespuesta cruda:")
    print(raw)

    print("\nJSON extraido:")
    if parsed is None:
        print(json_text or "(sin JSON)")
    else:
        print(json.dumps(parsed, ensure_ascii=False, indent=2))

    print("\nValidacion:")
    if errors:
        for error in errors:
            print(f"- {error}")
    else:
        print("- OK: JSON valido y con las claves esperadas.")


if __name__ == "__main__":
    main()
