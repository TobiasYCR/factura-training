import argparse
import json
import random
from pathlib import Path


ARCA_INSTRUCTION = (
    "Converti este texto OCR de una factura ARCA en un unico objeto JSON valido. "
    "No inventes datos: si falta un dato usa null; para iva, tributos e items usa array vacio. "
    "Usa exactamente el schema ARCA del proyecto y normaliza fechas, CUIT, punto de venta, numero y montos."
)

EXTERNAL_INSTRUCTION = (
    "Converti este texto OCR de un comprobante externo o recibo en un unico objeto JSON valido. "
    "No inventes datos: si falta un dato usa null; para items usa array vacio. "
    "Usa el schema external_provider del proyecto y conserva moneda, numero, proveedor, comprador, pagos y totales."
)


def compact_json(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def instruction_for(data):
    if isinstance(data, dict) and data.get("document_type"):
        return EXTERNAL_INSTRUCTION
    return ARCA_INSTRUCTION


def load_pairs(input_dir):
    input_dir = Path(input_dir)
    rows = []
    for json_path in sorted(input_dir.glob("*.json"), key=lambda path: path.name.lower()):
        if json_path.name == "batch_summary.jsonl":
            continue
        txt_path = json_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        ocr_text = txt_path.read_text(encoding="utf-8").strip()
        if not ocr_text or data is None:
            continue
        rows.append(
            {
                "filename": json_path.with_suffix(".pdf").name,
                "instruction": instruction_for(data),
                "input": f"Archivo: {json_path.with_suffix('.pdf').name}\n{ocr_text}",
                "output": compact_json(data),
            }
        )
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(compact_json(row) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Construye datasets JSONL reales desde OCR + JSON parseado.")
    parser.add_argument("--input-dir", default="data/real_invoices_analysis")
    parser.add_argument("--train-out", default="data/real_train.jsonl")
    parser.add_argument("--eval-out", default="data/real_eval.jsonl")
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_pairs(args.input_dir)
    if not rows:
        raise SystemExit(f"No encontre pares .txt/.json en {args.input_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    eval_count = max(1, round(len(rows) * args.eval_ratio)) if len(rows) > 1 else 0
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]

    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)

    external_count = sum(1 for row in rows if "external_provider" in row["output"])
    print(f"Ejemplos reales: {len(rows)}")
    print(f"Train: {len(train_rows)} -> {args.train_out}")
    print(f"Eval: {len(eval_rows)} -> {args.eval_out}")
    print(f"ARCA: {len(rows) - external_count}")
    print(f"Externos: {external_count}")


if __name__ == "__main__":
    main()
