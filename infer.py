import argparse
import json
import re
from pathlib import Path

from unsloth import FastLanguageModel


BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
LORA_MODEL = "factura-qwen-lora"
MAX_SEQ_LENGTH = 1024
DEFAULT_INSTRUCTION = (
    "Converti este texto OCR de una factura en un unico objeto JSON valido. "
    "No inventes datos. Si falta un dato, usa null. "
    "No agregues texto antes o despues del JSON. "
    "Usa exactamente estas claves: tipo_comprobante, numero_factura, "
    "empresa_emisora, identificacion_emisora, cliente, fecha, subtotal, "
    "impuestos, total, moneda. "
    "No uses claves distintas como comp_nro, importe_total o total_factura. "
    "Devuelve valores limpios: numero_factura sin etiquetas como Nro o Comp, "
    "identificacion_emisora sin la palabra CUIT, y cliente sin el prefijo Cliente. "
    "La fecha debe estar en formato YYYY-MM-DD. "
    "Los importes deben ser numeros sin simbolo de moneda. "
    "La moneda debe ser ARS si la factura esta en pesos argentinos."
)
REQUIRED_KEYS = {
    "tipo_comprobante",
    "numero_factura",
    "empresa_emisora",
    "identificacion_emisora",
    "cliente",
    "fecha",
    "subtotal",
    "impuestos",
    "total",
    "moneda",
}


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
        return ["La respuesta no contiene un objeto JSON válido."]

    errors = []
    missing = sorted(REQUIRED_KEYS - set(parsed))
    extra = sorted(set(parsed) - REQUIRED_KEYS)

    if missing:
        errors.append(f"Faltan claves: {', '.join(missing)}")
    if extra:
        errors.append(f"Claves extra: {', '.join(extra)}")

    for key in ("subtotal", "impuestos", "total"):
        value = parsed.get(key)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"{key} debería ser número o null.")

    numero = parsed.get("numero_factura")
    if numero is not None and not re.fullmatch(r"\d{4}-\d{8}", str(numero)):
        errors.append("numero_factura deberia tener formato 0000-00000000, sin etiquetas.")

    cuit = parsed.get("identificacion_emisora")
    if cuit is not None and not re.fullmatch(r"\d{2}-\d{8}-\d", str(cuit)):
        errors.append("identificacion_emisora deberia tener formato CUIT limpio 00-00000000-0.")

    cliente = parsed.get("cliente")
    if isinstance(cliente, str) and re.match(r"^\s*cliente\s*:", cliente, flags=re.IGNORECASE):
        errors.append("cliente no deberia incluir el prefijo 'Cliente:'.")

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
        description="Prueba un modelo base o LoRA para convertir OCR de facturas a JSON."
    )
    parser.add_argument(
        "--model",
        choices=["base", "lora"],
        default="lora",
        help="Modelo a usar. 'lora' carga factura-qwen-lora; 'base' carga el Qwen base.",
    )
    parser.add_argument("--ocr-file", default="data/test_ocr.txt")
    parser.add_argument("--ocr-text")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    model_name = BASE_MODEL if args.model == "base" else LORA_MODEL
    ocr_text = load_ocr_text(args)
    raw = run_inference(model_name, ocr_text, args.max_new_tokens)
    parsed, json_text = extract_json(raw)
    errors = validate_invoice_json(parsed)

    print(f"Modelo: {args.model} ({model_name})")
    print("\nRespuesta cruda:")
    print(raw)

    print("\nJSON extraído:")
    if parsed is None:
        print(json_text or "(sin JSON)")
    else:
        print(json.dumps(parsed, ensure_ascii=False, indent=2))

    print("\nValidación:")
    if errors:
        for error in errors:
            print(f"- {error}")
    else:
        print("- OK: JSON válido y con las claves esperadas.")


if __name__ == "__main__":
    main()
