import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer import (
    BASE_MODEL,
    LORA_MODEL,
    build_prompt,
    extract_json,
    finalize_invoice_json,
    load_model,
    normalize_external_document,
    normalize_invoice_json,
    parse_supported_document_ocr,
    validate_extracted_document_json,
)


def compact_json(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line"] = line_number
            rows.append(row)
    return rows


def normalize_for_expected(parsed, expected):
    if not isinstance(parsed, dict):
        return parsed
    if isinstance(expected, dict) and expected.get("document_type"):
        return normalize_external_document(parsed)
    return normalize_invoice_json(parsed)


def generate(model, tokenizer, instruction, ocr_text, max_new_tokens):
    prompt = build_prompt(ocr_text, instruction=instruction)
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


def flatten(data, prefix=""):
    if isinstance(data, dict):
        values = {}
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            values.update(flatten(value, child_prefix))
        return values
    if isinstance(data, list):
        values = {f"{prefix}.__len__": len(data)}
        for index, value in enumerate(data):
            values.update(flatten(value, f"{prefix}[{index}]"))
        return values
    return {prefix: data}


def values_equal(left, right):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 0.02
    return left == right


def compare_fields(predicted, expected):
    predicted_flat = flatten(predicted)
    expected_flat = flatten(expected)
    all_keys = sorted(set(predicted_flat) | set(expected_flat))
    ok = 0
    mismatches = []
    for key in all_keys:
        predicted_value = predicted_flat.get(key)
        expected_value = expected_flat.get(key)
        if values_equal(predicted_value, expected_value):
            ok += 1
        else:
            mismatches.append(
                {
                    "field": key,
                    "expected": expected_value,
                    "predicted": predicted_value,
                }
            )
    return ok, len(all_keys), mismatches


def filename_from_input(text):
    first_line = str(text).splitlines()[0].strip() if str(text).splitlines() else ""
    if first_line.lower().startswith("archivo:"):
        return first_line.split(":", 1)[1].strip()
    return None


def evaluate_row_model(row, model, tokenizer, args):
    raw = generate(
        model,
        tokenizer,
        row.get("instruction"),
        row["input"],
        args.max_new_tokens,
    )
    parsed, json_text = extract_json(raw)
    expected = json.loads(row["output"])
    parsed = normalize_for_expected(parsed, expected)
    errors = validate_extracted_document_json(parsed)
    return parsed, raw, json_text, errors


def evaluate_row_production(row):
    expected = json.loads(row["output"])
    parsed = parse_supported_document_ocr(row["input"])
    parsed = normalize_for_expected(parsed, expected)
    errors = validate_extracted_document_json(parsed)
    return parsed, None, None, errors


def parse_args():
    parser = argparse.ArgumentParser(description="Evalua real_eval.jsonl contra modelo o flujo parser.")
    parser.add_argument("--eval-file", default="data/real_eval.jsonl")
    parser.add_argument("--model", choices=["base", "lora"], default="lora")
    parser.add_argument(
        "--mode",
        choices=["model", "production"],
        default="model",
        help="model evalua el LoRA puro; production evalua parser/reglas sin cargar modelo.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default="data/eval_results.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(args.eval_file)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No hay filas para evaluar en {args.eval_file}")

    model = tokenizer = None
    if args.mode == "model":
        model_name = BASE_MODEL if args.model == "base" else LORA_MODEL
        model, tokenizer = load_model(model_name)
    else:
        model_name = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    valid_count = 0
    exact_count = 0
    field_ok = 0
    field_total = 0
    arca_count = external_count = 0
    arca_valid = external_valid = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as out_file:
        for index, row in enumerate(rows, start=1):
            row_started = time.perf_counter()
            expected = json.loads(row["output"])
            is_external = bool(expected.get("document_type"))
            external_count += int(is_external)
            arca_count += int(not is_external)

            if args.mode == "model":
                predicted, raw, json_text, errors = evaluate_row_model(row, model, tokenizer, args)
            else:
                predicted, raw, json_text, errors = evaluate_row_production(row)

            field_matches, field_count, mismatches = compare_fields(predicted, expected)
            is_valid = not errors
            is_exact = field_count > 0 and not mismatches
            valid_count += int(is_valid)
            exact_count += int(is_exact)
            field_ok += field_matches
            field_total += field_count
            arca_valid += int(is_valid and not is_external)
            external_valid += int(is_valid and is_external)

            item = {
                "line": row["_line"],
                "filename": filename_from_input(row["input"]),
                "ok_schema": is_valid,
                "exact": is_exact,
                "type": "external" if is_external else "arca",
                "predicted": predicted,
                "field_matches": field_matches,
                "field_total": field_count,
                "field_accuracy": round(field_matches / field_count, 4) if field_count else 0,
                "errors": errors,
                "mismatches": mismatches[:20],
                "elapsed_ms": round((time.perf_counter() - row_started) * 1000, 2),
            }
            if raw is not None:
                item["raw_model_response"] = raw
                item["json_text"] = json_text
            out_file.write(compact_json(item) + "\n")

            status = "OK" if is_valid else "FAIL"
            print(
                f"{index:03d}/{len(rows):03d} {status} "
                f"{item['type']} fields {field_matches}/{field_count} "
                f"{item['elapsed_ms']} ms"
            )
            if errors:
                print("  " + " | ".join(errors[:3]))

    elapsed = round(time.perf_counter() - started, 2)
    print("-" * 80)
    print(f"Modo: {args.mode}")
    if model_name:
        print(f"Modelo: {args.model} ({model_name})")
    print(f"Filas: {len(rows)}")
    print(f"JSON/schema OK: {valid_count}/{len(rows)} ({valid_count / len(rows):.1%})")
    print(f"Exactos: {exact_count}/{len(rows)} ({exact_count / len(rows):.1%})")
    print(f"Campos OK: {field_ok}/{field_total} ({field_ok / field_total:.1%})")
    if arca_count:
        print(f"ARCA schema OK: {arca_valid}/{arca_count} ({arca_valid / arca_count:.1%})")
    if external_count:
        print(f"Externos schema OK: {external_valid}/{external_count} ({external_valid / external_count:.1%})")
    print(f"Tiempo total: {elapsed}s")
    print(f"Detalle: {out_path}")


if __name__ == "__main__":
    main()
