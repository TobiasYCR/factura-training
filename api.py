import argparse
import io
import json
import re
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from infer import (
    BASE_MODEL,
    LORA_MODEL,
    extract_json,
    finalize_invoice_json,
    generate_with_loaded_model,
    load_model,
    parse_supported_document_ocr,
    validate_extracted_document_json,
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MODEL_CACHE = {}


def json_response(handler, status, payload):
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "si", "sí", "on"}


def parse_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_pdf_text(pdf_bytes):
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(page for page in pages if page.strip()).strip()
        if text:
            return text, "pdfplumber"
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(page for page in pages if page.strip()).strip()
        if text:
            return text, "pypdf"
    except Exception:
        pass

    return "", None


def parse_multipart_form(body, content_type):
    boundary_match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not boundary_match:
        raise ValueError("No se encontro boundary multipart.")

    boundary = boundary_match.group(1).encode("utf-8")
    fields = {}
    files = {}

    for raw_part in body.split(b"--" + boundary):
        part = raw_part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].strip()
        if b"\r\n\r\n" not in part:
            continue

        raw_headers, raw_value = part.split(b"\r\n\r\n", 1)
        raw_value = raw_value.rstrip(b"\r\n")
        headers = raw_headers.decode("utf-8", errors="replace")
        disposition = next(
            (line for line in headers.splitlines() if line.lower().startswith("content-disposition:")),
            "",
        )
        name = first_header_param(disposition, "name")
        filename = first_header_param(disposition, "filename")
        if not name:
            continue
        if filename:
            files[name] = {"filename": filename, "content": raw_value}
        else:
            fields[name] = raw_value.decode("utf-8", errors="replace")

    return fields, files


def first_header_param(header, name):
    match = re.search(rf'{re.escape(name)}="([^"]*)"', header)
    if match:
        return match.group(1)
    match = re.search(rf"{re.escape(name)}=([^;\s]+)", header)
    return match.group(1) if match else None


def get_model(model_choice):
    model_name = BASE_MODEL if model_choice == "base" else LORA_MODEL
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = load_model(model_name)
    return model_name, MODEL_CACHE[model_name]


def extract_document(ocr_text, use_model=False, model_choice="lora", max_new_tokens=900):
    started = time.perf_counter()
    raw_model_response = None
    parsed = parse_supported_document_ocr(ocr_text)
    extraction_source = "parser" if parsed is not None else None
    model_name = None

    if parsed is None and use_model:
        model_name, (model, tokenizer) = get_model(model_choice)
        raw_model_response = generate_with_loaded_model(model, tokenizer, ocr_text, max_new_tokens)
        model_parsed, _ = extract_json(raw_model_response)
        parsed = finalize_invoice_json(model_parsed, ocr_text)
        extraction_source = "model"

    if parsed is None:
        parsed = finalize_invoice_json(None, ocr_text)

    errors = validate_extracted_document_json(parsed)
    return {
        "ok": not errors,
        "source": extraction_source,
        "model": model_name,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "errors": errors,
        "data": parsed,
        "raw_model_response": raw_model_response,
    }


class InvoiceApiHandler(BaseHTTPRequestHandler):
    server_version = "FacturaTrainingAPI/0.1"

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "service": "factura-training-api",
                    "endpoints": ["GET /health", "POST /extract"],
                },
            )
            return
        json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Endpoint no encontrado."})

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/extract":
            json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Endpoint no encontrado."})
            return

        try:
            request = self.read_extract_request()
            result = extract_document(
                request["ocr_text"],
                use_model=request["use_model"],
                model_choice=request["model"],
                max_new_tokens=request["max_new_tokens"],
            )
            result["input"] = {
                "filename": request["filename"],
                "text_extractor": request["text_extractor"],
                "ocr_text_length": len(request["ocr_text"]),
            }
            json_response(self, HTTPStatus.OK if result["ok"] else HTTPStatus.UNPROCESSABLE_ENTITY, result)
        except ValueError as error:
            json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
        except Exception as error:
            json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(error)})

    def read_extract_request(self):
        content_length = parse_int(self.headers.get("Content-Length"), 0)
        if content_length <= 0:
            raise ValueError("La request no tiene body.")
        if content_length > MAX_UPLOAD_BYTES:
            raise ValueError("El archivo supera el limite de 25 MB.")

        body = self.rfile.read(content_length)
        content_type = self.headers.get("Content-Type", "")
        query = parse_qs(urlparse(self.path).query)

        fields = {}
        files = {}
        if content_type.startswith("multipart/form-data"):
            fields, files = parse_multipart_form(body, content_type)
        elif content_type.startswith("application/json"):
            fields = json.loads(body.decode("utf-8"))
        else:
            raise ValueError("Usa multipart/form-data o application/json.")

        model = get_first(fields, query, "model", "lora")
        if model not in {"base", "lora"}:
            raise ValueError("model debe ser 'base' o 'lora'.")

        use_model = parse_bool(get_first(fields, query, "use_model", False), default=False)
        max_new_tokens = parse_int(get_first(fields, query, "max_new_tokens", 900), 900)
        ocr_text = get_first(fields, query, "ocr_text", "")
        filename = None
        text_extractor = None

        uploaded = files.get("file") or files.get("pdf")
        if uploaded:
            filename = uploaded["filename"]
            suffix = Path(filename).suffix.lower()
            if suffix != ".pdf":
                raise ValueError("Por ahora el endpoint acepta archivos PDF.")
            ocr_text, text_extractor = extract_pdf_text(uploaded["content"])
            if not ocr_text:
                raise ValueError(
                    "No se pudo extraer texto del PDF. Si es escaneado como imagen, falta conectar OCR real."
                )

        if not str(ocr_text).strip():
            raise ValueError("Envia un PDF en el campo 'file' o texto OCR en 'ocr_text'.")

        return {
            "filename": filename,
            "text_extractor": text_extractor,
            "ocr_text": str(ocr_text).strip(),
            "use_model": use_model,
            "model": model,
            "max_new_tokens": max_new_tokens,
        }

    def log_message(self, format, *args):
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))


def get_first(fields, query, key, default=None):
    if key in fields:
        return fields[key]
    if key in query and query[key]:
        return query[key][0]
    return default


def main():
    parser = argparse.ArgumentParser(description="Endpoint local para extraer JSON desde PDFs de facturas.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), InvoiceApiHandler)
    print(f"Factura Training API escuchando en http://{args.host}:{args.port}")
    print("Healthcheck: GET /health")
    print("Extraccion:  POST /extract con multipart field file=@factura.pdf")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
