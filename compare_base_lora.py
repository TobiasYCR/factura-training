import argparse
import gc
import json
from pathlib import Path

import torch

from infer import (
    BASE_MODEL,
    LORA_MODEL,
    extract_json,
    run_inference,
    validate_invoice_json,
)


def unload_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_result(label, model_name, raw):
    parsed, json_text = extract_json(raw)
    errors = validate_invoice_json(parsed)

    print("=" * 80)
    print(f"{label}: {model_name}")
    print("-" * 80)
    print("Respuesta cruda:")
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
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compara el Qwen base contra el LoRA entrenado para factura OCR -> JSON."
    )
    parser.add_argument("--ocr-file", default="data/test_ocr.txt")
    parser.add_argument("--ocr-text")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    if args.ocr_text:
        ocr_text = args.ocr_text
    else:
        ocr_text = Path(args.ocr_file).read_text(encoding="utf-8").strip()

    print("Texto OCR usado:")
    print(ocr_text)
    print()

    base_raw = run_inference(BASE_MODEL, ocr_text, args.max_new_tokens)
    print_result("BASE", BASE_MODEL, base_raw)
    unload_cuda()

    lora_raw = run_inference(LORA_MODEL, ocr_text, args.max_new_tokens)
    print_result("LORA", LORA_MODEL, lora_raw)
    unload_cuda()


if __name__ == "__main__":
    main()
