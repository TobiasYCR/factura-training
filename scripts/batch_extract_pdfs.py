import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import extract_document, extract_upload_text
from ocr import DEFAULT_OCR_MULTIPASS, OcrUnavailableError


def safe_output_name(path):
    safe = "".join(char if char.isalnum() or char in "._- " else "_" for char in path.stem)
    return safe.strip() or "document"


def document_label(data):
    if not isinstance(data, dict):
        return None
    if data.get("document_type"):
        document = data.get("document") or {}
        return {
            "type": data.get("document_type"),
            "number": document.get("number"),
            "date": document.get("date"),
            "total": data.get("total"),
            "currency": data.get("currency"),
        }
    return {
        "type": data.get("tipo_comprobante"),
        "number": data.get("numero_factura"),
        "date": data.get("fecha_emision"),
        "total": data.get("total"),
        "currency": data.get("moneda"),
    }


def iter_files(input_dir, pattern):
    return sorted(Path(input_dir).glob(pattern), key=lambda path: path.name.lower())


def process_file(path, args):
    started = time.perf_counter()
    result = {
        "file": str(path),
        "filename": path.name,
        "ok": False,
        "elapsed_ms": None,
        "text_extractor": None,
        "ocr_text_length": 0,
        "source": None,
        "label": None,
        "errors": [],
    }

    try:
        file_bytes = path.read_bytes()
        first_multipass = args.ocr_policy == "robust" or args.ocr_multipass
        text, text_extractor = extract_upload_text(
            file_bytes,
            path.name,
            force_ocr=args.force_ocr,
            ocr_lang=args.ocr_lang,
            ocr_dpi=args.ocr_dpi,
            ocr_multipass=first_multipass,
        )
        ocr_attempts = [text_extractor]
        parser_text = f"Archivo: {path.name}\n{text}"
        extracted = extract_document(
            parser_text,
            use_model=args.use_model,
            model_choice=args.model,
            max_new_tokens=args.max_new_tokens,
            min_confidence=args.min_confidence,
        )
        if (
            args.ocr_policy == "auto"
            and not (text_extractor.get("method") == "ocr" and text_extractor.get("multipass"))
            and (not extracted["ok"] or extracted["confidence"] < args.min_confidence)
        ):
            retry_text, retry_extractor = extract_upload_text(
                file_bytes,
                path.name,
                force_ocr=True,
                ocr_lang=args.ocr_lang,
                ocr_dpi=args.ocr_dpi,
                ocr_multipass=True,
            )
            retry_parser_text = f"Archivo: {path.name}\n{retry_text}"
            retry_extracted = extract_document(
                retry_parser_text,
                use_model=args.use_model,
                model_choice=args.model,
                max_new_tokens=args.max_new_tokens,
                min_confidence=args.min_confidence,
            )
            ocr_attempts.append(retry_extractor)
            if (retry_extracted["ok"] and not extracted["ok"]) or retry_extracted["confidence"] >= extracted["confidence"]:
                text = retry_text
                text_extractor = retry_extractor
                parser_text = retry_parser_text
                extracted = retry_extracted
        if args.write_ocr:
            output_text = args.output_dir / f"{safe_output_name(path)}.txt"
            output_text.write_text(text, encoding="utf-8")
            result["ocr_text_file"] = str(output_text)

        result.update(
            {
                "ok": extracted["ok"],
                "text_extractor": text_extractor,
                "ocr_attempts": ocr_attempts,
                "ocr_text_length": len(text),
                "source": extracted["source"],
                "label": document_label(extracted["data"]),
                "errors": extracted["errors"],
            }
        )
        if args.write_json:
            output_json = args.output_dir / f"{safe_output_name(path)}.json"
            output_json.write_text(
                json.dumps(extracted["data"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result["json_file"] = str(output_json)
    except OcrUnavailableError as error:
        result["errors"].append(str(error))
    except Exception as error:
        result["errors"].append(str(error))

    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Extrae JSON en lote desde PDFs o imagenes.")
    parser.add_argument("input_dir", help="Carpeta con PDFs/imagenes.")
    parser.add_argument("--pattern", default="*.pdf")
    parser.add_argument("--output-dir", default="data/real_invoices_analysis")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--ocr-lang", default="spa+eng")
    parser.add_argument("--ocr-dpi", type=int, default=220)
    parser.add_argument("--ocr-policy", choices=["fast", "auto", "robust"], default="fast")
    parser.add_argument("--ocr-multipass", action="store_true", default=DEFAULT_OCR_MULTIPASS)
    parser.add_argument("--no-ocr-multipass", action="store_false", dest="ocr_multipass")
    parser.add_argument("--use-model", action="store_true")
    parser.add_argument("--model", choices=["base", "lora"], default="lora")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--min-confidence", type=float, default=0.82)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--write-json", action="store_true")
    parser.add_argument("--write-ocr", action="store_true", help="Guarda el texto extraido/OCR de cada archivo.")
    parser.add_argument(
        "--failed-from",
        help="Procesa solo archivos fallidos listados en un batch_summary.jsonl anterior.",
    )
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = iter_files(args.input_dir, args.pattern)
    if args.failed_from:
        failed_paths = set()
        with open(args.failed_from, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                if not item.get("ok"):
                    failed_paths.add(str(Path(item["file"])))
        paths = [path for path in paths if str(path) in failed_paths]

    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No se encontraron archivos con pattern {args.pattern!r} en {args.input_dir}")

    summary_path = args.output_dir / "batch_summary.jsonl"
    ok_count = 0
    parser_count = 0
    model_count = 0
    ocr_count = 0
    robust_retry_count = 0

    with summary_path.open("w", encoding="utf-8") as summary_file:
        for index, path in enumerate(paths, start=1):
            item = process_file(path, args)
            summary_file.write(json.dumps(item, ensure_ascii=False) + "\n")

            ok_count += int(item["ok"])
            parser_count += int(item["source"] == "parser")
            model_count += int(item["source"] == "model")
            text_extractor = item.get("text_extractor") or {}
            ocr_count += int(text_extractor.get("method") == "ocr")
            robust_retry_count += int(len(item.get("ocr_attempts") or []) > 1)
            label = item.get("label") or {}
            status = "OK" if item["ok"] else "FAIL"
            print(
                f"{index:03d}/{len(paths):03d} {status} "
                f"{Path(item['file']).name} | {label.get('type')} | {label.get('number')} | "
                f"{item['elapsed_ms']} ms"
            )
            if item["errors"]:
                print("  " + " | ".join(item["errors"][:3]))

    print("-" * 80)
    print(f"Archivos: {len(paths)}")
    print(f"OK: {ok_count}")
    print(f"Parser: {parser_count}")
    print(f"Modelo: {model_count}")
    print(f"OCR visual: {ocr_count}")
    print(f"Reintentos OCR robusto: {robust_retry_count}")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
