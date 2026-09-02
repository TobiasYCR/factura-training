import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import extract_document, extract_upload_text


def safe_stem(path):
    return "".join(char if char.isalnum() or char in "._- " else "_" for char in path.stem).strip()


def compare_values(expected, actual, prefix=""):
    mismatches = []
    if isinstance(expected, dict):
        actual = actual if isinstance(actual, dict) else {}
        for key, expected_value in expected.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            mismatches.extend(compare_values(expected_value, actual.get(key), child_prefix))
        return mismatches

    if isinstance(expected, list):
        actual = actual if isinstance(actual, list) else []
        if len(expected) != len(actual):
            mismatches.append({"field": f"{prefix}.__len__", "expected": len(expected), "actual": len(actual)})
            return mismatches
        for index, expected_item in enumerate(expected):
            mismatches.extend(compare_values(expected_item, actual[index], f"{prefix}[{index}]"))
        return mismatches

    if expected != actual:
        mismatches.append({"field": prefix, "expected": expected, "actual": actual})
    return mismatches


def process_pdf(path, expected_dir, args):
    text, extractor = extract_upload_text(
        path.read_bytes(),
        path.name,
        force_ocr=args.force_ocr,
        ocr_lang=args.ocr_lang,
        ocr_dpi=args.ocr_dpi,
        ocr_multipass=args.ocr_multipass,
    )
    result = extract_document(text, filename=path.name, min_confidence=args.min_confidence)
    expected_path = expected_dir / f"{safe_stem(path)}.json"
    row = {
        "file": str(path),
        "expected": str(expected_path),
        "ok": result["ok"],
        "requires_review": result.get("requires_review", False),
        "confidence": result.get("confidence"),
        "errors": result.get("errors", []),
        "warnings": result.get("warnings", []),
        "text_extractor": extractor,
        "mismatches": [],
    }

    if expected_path.exists():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        row["mismatches"] = compare_values(expected, result["data"])
        row["passed"] = result["ok"] and not row["mismatches"] and not row["requires_review"]
    else:
        row["passed"] = False
        row["missing_expected"] = True
        if args.write_missing_expected:
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_text(json.dumps(result["data"], ensure_ascii=False, indent=2), encoding="utf-8")
            row["missing_expected_written"] = True
    return row


def main():
    parser = argparse.ArgumentParser(description="Corre PDFs de regresion y compara contra JSON esperado.")
    parser.add_argument("input_dir", nargs="?", default="data/regression_pdfs")
    parser.add_argument("--expected-dir", default="data/regression_expected")
    parser.add_argument("--pattern", default="*.pdf")
    parser.add_argument("--output", default="tmp/regression_results.jsonl")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--ocr-lang", default="spa+eng")
    parser.add_argument("--ocr-dpi", type=int, default=220)
    parser.add_argument("--ocr-multipass", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.82)
    parser.add_argument("--write-missing-expected", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    expected_dir = Path(args.expected_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob(args.pattern), key=lambda item: item.name.lower())
    if not paths:
        raise SystemExit(f"No se encontraron PDFs en {input_dir} con pattern {args.pattern!r}.")

    passed = 0
    with output.open("w", encoding="utf-8") as file:
        for index, path in enumerate(paths, start=1):
            row = process_pdf(path, expected_dir, args)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            passed += int(row["passed"])
            status = "OK" if row["passed"] else "FAIL"
            print(f"{index:03d}/{len(paths):03d} {status} {path.name}")
            if row.get("missing_expected"):
                print("  Falta JSON esperado.")
            for mismatch in row["mismatches"][:3]:
                print(f"  {mismatch['field']}: esperado={mismatch['expected']!r} actual={mismatch['actual']!r}")
            if row["warnings"]:
                print("  Warnings: " + " | ".join(row["warnings"][:3]))

    print("-" * 80)
    print(f"Pasaron: {passed}/{len(paths)}")
    print(f"Resultado: {output}")
    if passed != len(paths):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
