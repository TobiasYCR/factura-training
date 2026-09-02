import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Resume logs JSONL de factura-training.")
    parser.add_argument("log_file", nargs="?", default="logs/extractions.jsonl")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    rows = read_jsonl(args.log_file)
    if not rows:
        raise SystemExit("Log vacio o inexistente.")

    total = len(rows)
    ok_count = sum(1 for row in rows if row.get("ok"))
    review_count = sum(1 for row in rows if row.get("requires_review"))
    sources = Counter(row.get("source") or "none" for row in rows)
    types = Counter(row.get("document_type") or row.get("tipo_comprobante") or "unknown" for row in rows)
    errors = Counter(error for row in rows for error in row.get("errors") or [])
    warnings = Counter(warning for row in rows for warning in row.get("warnings") or [])
    weak_fields = Counter()
    for row in rows:
        for field, score in (row.get("field_confidence") or {}).items():
            if score == 0:
                weak_fields[field] += 1

    print(f"Requests: {total}")
    print(f"OK: {ok_count}/{total} ({ok_count / total:.1%})")
    print(f"Requieren revision: {review_count}/{total} ({review_count / total:.1%})")
    print("\nFuentes:")
    for source, count in sources.most_common(args.top):
        print(f"- {source}: {count}")
    print("\nTipos:")
    for doc_type, count in types.most_common(args.top):
        print(f"- {doc_type}: {count}")
    print("\nWarnings:")
    for warning, count in warnings.most_common(args.top):
        print(f"- {count}x {warning}")
    print("\nErrores:")
    for error, count in errors.most_common(args.top):
        print(f"- {count}x {error}")
    print("\nCampos flojos:")
    for field, count in weak_fields.most_common(args.top):
        print(f"- {count}x {field}")


if __name__ == "__main__":
    main()
