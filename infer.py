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


def as_number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def round_money(value):
    return round(float(value) + 1e-9, 2)


def parse_ar_money(value):
    return round_money(str(value).replace(".", "").replace(",", "."))


def parse_ar_date(value):
    return datetime.strptime(value, "%d/%m/%Y").date().isoformat()


def rate_from_description(description):
    if description is None:
        return None
    match = re.search(r"10[,.]5|21|27", str(description))
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def parse_structured_arca_ocr(ocr_text):
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    if not lines:
        return None

    text = "\n".join(lines)
    header = re.search(r"FACTURA\s+([ABC])", text, flags=re.IGNORECASE)
    code = re.search(r"Cod\.\s*(\d+)", text)
    numbers = re.search(r"Punto de Venta:\s*(\d+)\s+Comp\.\s*Nro:\s*(\d+)", text)
    issue_date = re.search(r"Fecha de Emision:\s*(\d{2}/\d{2}/\d{4})", text)
    cae = re.search(r"CAE:\s*(\d{14})", text)
    due_date = re.search(r"Vto\.\s*CAE:\s*(\d{2}/\d{2}/\d{4})", text)
    if not (header and code and numbers and issue_date and cae and due_date):
        return None

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
    letter = header.group(1).upper()

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
        if all(isinstance(iva, dict) and as_number(iva.get("importe")) is not None for iva in fixed_iva):
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
        if all(isinstance(tributo, dict) and as_number(tributo.get("importe")) is not None for tributo in fixed_tributos):
            normalized["tributos_total"] = round_money(sum(tributo["importe"] for tributo in fixed_tributos))

    iva_total = as_number(normalized.get("iva_total"))
    tributos_total = as_number(normalized.get("tributos_total"))
    if iva_total is not None and tributos_total is not None:
        normalized["impuestos"] = round_money(iva_total + tributos_total)

    return normalized


def finalize_invoice_json(parsed, ocr_text=None):
    if ocr_text:
        structured = parse_structured_arca_ocr(ocr_text)
        if structured is not None:
            return structured

    normalized = normalize_invoice_json(parsed)
    if isinstance(normalized, dict) and REQUIRED_KEYS <= set(normalized):
        return normalized
    return normalized


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
    errors = validate_invoice_json(parsed)

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
