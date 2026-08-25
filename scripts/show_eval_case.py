import argparse
import json
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("line", line_number)
            rows.append(row)
    return rows


def filename_from_input(text):
    lines = str(text).splitlines()
    first_line = lines[0].strip() if lines else ""
    if first_line.lower().startswith("archivo:"):
        return first_line.split(":", 1)[1].strip()
    return None


def load_eval_rows(path):
    if not path:
        return {}
    rows_by_line = {}
    for row in load_jsonl(path):
        rows_by_line[row["line"]] = row
    return rows_by_line


def main():
    parser = argparse.ArgumentParser(description="Muestra el detalle completo de un caso de evaluacion.")
    parser.add_argument("results_file", help="Archivo generado por scripts/evaluate_real_dataset.py")
    parser.add_argument("--eval-file", help="Archivo real_eval.jsonl usado para recuperar input/output originales.")
    parser.add_argument("--contains", required=True, help="Texto a buscar en el nombre del archivo.")
    parser.add_argument("--show-json", action="store_true", help="Muestra expected/predicted completos.")
    args = parser.parse_args()

    results = load_jsonl(args.results_file)
    eval_rows = load_eval_rows(args.eval_file)
    needle = args.contains.lower()

    matches = []
    for result in results:
        eval_row = eval_rows.get(result.get("line"), {})
        filename = result.get("filename") or filename_from_input(eval_row.get("input", ""))
        if filename and needle in filename.lower():
            matches.append((result, eval_row, filename))

    if not matches:
        raise SystemExit(f"No encontre casos que contengan: {args.contains}")

    for result, eval_row, filename in matches:
        print("=" * 80)
        print(f"Archivo: {filename}")
        print(f"Linea: {result.get('line')}")
        print(f"Tipo: {result.get('type')}")
        print(f"Schema OK: {result.get('ok_schema')}")
        print(f"Exacto: {result.get('exact')}")
        print(f"Campos: {result.get('field_matches')}/{result.get('field_total')} ({result.get('field_accuracy'):.1%})")

        errors = result.get("errors") or []
        if errors:
            print("\nErrores:")
            for error in errors:
                print(f"- {error}")

        mismatches = result.get("mismatches") or []
        if mismatches:
            print("\nMismatches:")
            for mismatch in mismatches:
                print(
                    "- "
                    f"{mismatch.get('field')}: "
                    f"esperado={mismatch.get('expected')!r} "
                    f"obtenido={mismatch.get('predicted')!r}"
                )

        if args.show_json:
            expected = json.loads(eval_row.get("output", "{}"))
            print("\nExpected:")
            print(json.dumps(expected, ensure_ascii=False, indent=2))
            print("\nPredicted:")
            print(json.dumps(result.get("predicted"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
