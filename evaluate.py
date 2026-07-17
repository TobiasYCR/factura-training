import argparse
import gc
import json

import torch

from infer import (
    BASE_MODEL,
    LORA_MODEL,
    extract_json,
    generate_with_loaded_model,
    load_model,
    validate_invoice_json,
)


def unload_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_line_number"] = line_number
            yield item


def parse_expected(item):
    expected = item.get("output")
    if isinstance(expected, str):
        return json.loads(expected)
    return expected


def compare_fields(actual, expected):
    rows = []
    keys = sorted(set(expected) | set(actual or {}))
    for key in keys:
        expected_value = expected.get(key)
        actual_value = None if actual is None else actual.get(key)
        rows.append((key, actual_value == expected_value, expected_value, actual_value))
    return rows


def evaluate_model(label, model_name, examples, max_new_tokens):
    model, tokenizer = load_model(model_name)
    total_fields = 0
    ok_fields = 0
    valid_json_count = 0

    print("=" * 80)
    print(f"Evaluando {label}: {model_name}")

    for index, item in enumerate(examples, start=1):
        expected = parse_expected(item)
        raw = generate_with_loaded_model(model, tokenizer, item["input"], max_new_tokens)
        actual, _ = extract_json(raw)
        validation_errors = validate_invoice_json(actual)
        rows = compare_fields(actual, expected)
        passed = sum(1 for _, ok, _, _ in rows if ok)

        total_fields += len(rows)
        ok_fields += passed
        if actual is not None and not validation_errors:
            valid_json_count += 1

        print("-" * 80)
        print(f"Caso {index} (linea {item['_line_number']}): {passed}/{len(rows)} campos OK")
        if validation_errors:
            print("Validacion:")
            for error in validation_errors:
                print(f"- {error}")

        for key, ok, expected_value, actual_value in rows:
            marker = "OK" if ok else "FAIL"
            print(f"{marker} {key}: esperado={expected_value!r} obtenido={actual_value!r}")

    print("-" * 80)
    print(f"JSON validos: {valid_json_count}/{len(examples)}")
    print(f"Campos correctos: {ok_fields}/{total_fields}")
    print()

    del model
    del tokenizer
    unload_cuda()


def main():
    parser = argparse.ArgumentParser(
        description="Evalua campo por campo el modelo base, el LoRA o ambos."
    )
    parser.add_argument("--model", choices=["base", "lora", "both"], default="both")
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    examples = list(read_jsonl(args.eval_file))
    if not examples:
        raise SystemExit(f"No hay ejemplos en {args.eval_file}")

    if args.model in ("base", "both"):
        evaluate_model("BASE", BASE_MODEL, examples, args.max_new_tokens)
    if args.model in ("lora", "both"):
        evaluate_model("LORA", LORA_MODEL, examples, args.max_new_tokens)


if __name__ == "__main__":
    main()
