import argparse
import json
import re
from pathlib import Path

from unsloth import FastLanguageModel


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

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, text[index : index + end]

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None, None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, match.group(0)

    return parsed, match.group(0)


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
