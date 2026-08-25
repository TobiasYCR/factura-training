import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_field(field):
    field = str(field)
    return field.replace(".__len__", "")


def main():
    parser = argparse.ArgumentParser(description="Resume errores y campos flojos de una evaluacion JSONL.")
    parser.add_argument("results_file", help="Archivo generado por scripts/evaluate_real_dataset.py")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = load_jsonl(args.results_file)
    if not rows:
        raise SystemExit(f"No hay filas en {args.results_file}")

    total = len(rows)
    exact = sum(1 for row in rows if row.get("exact"))
    schema_ok = sum(1 for row in rows if row.get("ok_schema"))
    field_ok = sum(row.get("field_matches", 0) for row in rows)
    field_total = sum(row.get("field_total", 0) for row in rows)

    by_type = defaultdict(list)
    for row in rows:
        by_type[row.get("type", "unknown")].append(row)

    field_mismatches = Counter()
    weak_rows = []
    schema_errors = Counter()

    for row in rows:
        for error in row.get("errors", []):
            schema_errors[error] += 1
        for mismatch in row.get("mismatches", []):
            field_mismatches[clean_field(mismatch.get("field"))] += 1
        weak_rows.append(
            (
                row.get("field_accuracy", 0),
                row.get("line"),
                row.get("filename"),
                row.get("type", "unknown"),
                row.get("field_matches", 0),
                row.get("field_total", 0),
                row.get("errors", []),
                row.get("mismatches", [])[:5],
            )
        )

    print(f"Archivo: {args.results_file}")
    print(f"Filas: {total}")
    print(f"Schema OK: {schema_ok}/{total} ({schema_ok / total:.1%})")
    print(f"Exactos: {exact}/{total} ({exact / total:.1%})")
    field_accuracy = field_ok / field_total if field_total else 0
    print(f"Campos OK: {field_ok}/{field_total} ({field_accuracy:.1%})")
    print()

    print("Por tipo:")
    for doc_type, items in sorted(by_type.items()):
        count = len(items)
        type_schema_ok = sum(1 for row in items if row.get("ok_schema"))
        type_exact = sum(1 for row in items if row.get("exact"))
        type_field_ok = sum(row.get("field_matches", 0) for row in items)
        type_field_total = sum(row.get("field_total", 0) for row in items)
        type_field_accuracy = type_field_ok / type_field_total if type_field_total else 0
        print(
            f"- {doc_type}: schema {type_schema_ok}/{count} ({type_schema_ok / count:.1%}), "
            f"exactos {type_exact}/{count} ({type_exact / count:.1%}), "
            f"campos {type_field_ok}/{type_field_total} ({type_field_accuracy:.1%})"
        )
    print()

    if schema_errors:
        print(f"Errores de schema mas repetidos (top {args.top}):")
        for error, count in schema_errors.most_common(args.top):
            print(f"- {count}x {error}")
        print()

    print(f"Campos con mas diferencias (top {args.top}):")
    for field, count in field_mismatches.most_common(args.top):
        print(f"- {count}x {field}")
    print()

    print(f"Casos prioritarios por menor field_accuracy (top {args.top}):")
    for accuracy, line, filename, doc_type, matches, count, errors, mismatches in sorted(weak_rows)[: args.top]:
        label = filename or f"linea {line}"
        print(f"- {label} | {doc_type} | {matches}/{count} ({accuracy:.1%})")
        for error in errors[:2]:
            print(f"  error: {error}")
        for mismatch in mismatches[:3]:
            print(
                "  mismatch: "
                f"{mismatch.get('field')} esperado={mismatch.get('expected')!r} "
                f"obtenido={mismatch.get('predicted')!r}"
            )


if __name__ == "__main__":
    main()
