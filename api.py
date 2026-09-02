import argparse
import io
import json
import os
import re
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from infer import (
    BASE_MODEL,
    LORA_MODEL,
    assess_document_quality,
    build_arca_invoice_identifier,
    build_display_description,
    clean_arca_description_candidate,
    complete_monthly_consulting_description,
    derive_iva_percentage,
    extract_json,
    finalize_invoice_json,
    generate_with_loaded_model,
    load_model,
    parse_supported_document_ocr,
    validate_extracted_document_json,
)
from ocr import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANG,
    DEFAULT_OCR_MULTIPASS,
    OcrUnavailableError,
    get_ocr_status,
    ocr_image_bytes,
    ocr_pdf_bytes,
)

MAX_UPLOAD_BYTES = int(os.environ.get("FACTURA_MAX_UPLOAD_MB", "25")) * 1024 * 1024
MODEL_CACHE = {}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
API_KEY = os.environ.get("FACTURA_API_KEY")
LOG_FILE = os.environ.get("FACTURA_LOG_FILE")


def looks_like_broken_embedded_text(text):
    if not text:
        return False
    cid_count = text.count("(cid:")
    if cid_count >= 10:
        return True
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    readable = sum(1 for char in compact if char.isalnum() or char in ".,:$-/")
    return len(compact) > 500 and readable / len(compact) < 0.55


def json_response(handler, status, payload):
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def is_missing_value(value):
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, list) and not value:
        return True
    return False


def document_completeness(data):
    if not isinstance(data, dict):
        return 0.0

    scalar_total = 0
    scalar_present = 0

    def visit(value):
        nonlocal scalar_total, scalar_present
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, list):
            if not value:
                scalar_total += 1
                return
            for item in value:
                visit(item)
            return
        scalar_total += 1
        scalar_present += int(not is_missing_value(value))

    visit(data)
    if scalar_total == 0:
        return 0.0
    return scalar_present / scalar_total


def add_arca_integration_fields(data, source_text=None):
    """Expose fields consumed by the administrative invoice screen."""
    if not isinstance(data, dict):
        return data

    enriched = dict(data)
    if not data.get("document_type"):
        invoice_identifier = build_arca_invoice_identifier(data)
        iva_percentage = derive_iva_percentage(data)
        if invoice_identifier:
            enriched["numero_factura_completo"] = invoice_identifier
        if iva_percentage is not None:
            enriched["iva_porcentaje"] = iva_percentage

    description = build_display_description(data, source_text) or enriched.get("descripcion")
    if description and not data.get("document_type"):
        description = clean_arca_description_candidate(description) or description
        description = complete_monthly_consulting_description(
            description,
            source_text,
            enriched.get("fecha_emision"),
        )
    if description:
        enriched["descripcion"] = description
    return enriched


def calculate_confidence(parsed, errors, source, used_model, warnings=None):
    if errors:
        return 0.0
    if not isinstance(parsed, dict):
        return 0.0
    completeness = document_completeness(parsed)
    if source == "parser":
        base = 0.55 + (0.43 * completeness)
    elif used_model:
        base = 0.50 + (0.32 * completeness)
    else:
        base = 0.35 + (0.25 * completeness)
    warning_penalty = min(0.25, 0.04 * len(warnings or []))
    return round(max(0.0, base - warning_penalty), 4)


def append_log(event):
    if not LOG_FILE:
        return
    path = Path(LOG_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "si", "on"}


def parse_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_embedded_pdf_text(pdf_bytes):
    errors = []

    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(page for page in pages if page.strip()).strip()
        if text:
            return text, {"method": "embedded_text", "engine": "pdfplumber", "errors": []}
    except Exception as error:
        errors.append(f"pdfplumber: {error}")

    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(page for page in pages if page.strip()).strip()
        if text:
            return text, {"method": "embedded_text", "engine": "pypdf", "errors": errors}
    except Exception as error:
        errors.append(f"pypdf: {error}")

    return "", {"method": "embedded_text", "engine": None, "errors": errors}


def extract_upload_text(
    file_bytes,
    filename,
    force_ocr=False,
    ocr_lang=DEFAULT_OCR_LANG,
    ocr_dpi=DEFAULT_OCR_DPI,
    ocr_multipass=DEFAULT_OCR_MULTIPASS,
):
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        if not force_ocr:
            text, meta = extract_embedded_pdf_text(file_bytes)
            if text and not looks_like_broken_embedded_text(text):
                return text, meta
        text, ocr_meta = ocr_pdf_bytes(file_bytes, lang=ocr_lang, dpi=ocr_dpi, multipass=ocr_multipass)
        return text, {"method": "ocr", **ocr_meta}

    if suffix in IMAGE_EXTENSIONS:
        text, ocr_meta = ocr_image_bytes(file_bytes, lang=ocr_lang, multipass=ocr_multipass)
        return text, {"method": "ocr", **ocr_meta}

    raise ValueError("Por ahora el endpoint acepta PDF o imagenes png/jpg/tiff/bmp/webp.")


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


def should_run_model(parsed, errors, use_model, model_policy, confidence, min_confidence):
    if not use_model:
        return False
    if model_policy == "always":
        return True
    if parsed is None or errors:
        return True
    if model_policy == "low_confidence" and confidence < min_confidence:
        return True
    return False


def extract_document(
    ocr_text,
    filename=None,
    use_model=False,
    model_choice="lora",
    max_new_tokens=900,
    model_policy="fallback",
    min_confidence=0.82,
):
    started = time.perf_counter()
    raw_model_response = None
    parser_text = f"Archivo: {filename}\n{ocr_text}" if filename else ocr_text
    parsed = parse_supported_document_ocr(parser_text)
    extraction_source = "parser" if parsed is not None else None
    model_name = None
    initial_errors = validate_extracted_document_json(parsed) if parsed is not None else []
    initial_warnings = assess_document_quality(parsed, parser_text) if parsed is not None else []
    initial_confidence = calculate_confidence(parsed, initial_errors, extraction_source, False, initial_warnings)

    if should_run_model(parsed, initial_errors, use_model, model_policy, initial_confidence, min_confidence):
        parser_parsed = parsed
        parser_source = extraction_source
        parser_errors = initial_errors
        parser_confidence = initial_confidence
        parser_warnings = initial_warnings
        model_name, (model, tokenizer) = get_model(model_choice)
        raw_model_response = generate_with_loaded_model(model, tokenizer, ocr_text, max_new_tokens)
        model_parsed, _ = extract_json(raw_model_response)
        parsed = finalize_invoice_json(model_parsed, parser_text)
        extraction_source = "model"
        model_errors = validate_extracted_document_json(parsed)
        model_warnings = assess_document_quality(parsed, parser_text)
        model_confidence = calculate_confidence(parsed, model_errors, extraction_source, True, model_warnings)
        if parser_parsed is not None and not parser_errors and parser_confidence >= model_confidence:
            parsed = parser_parsed
            extraction_source = parser_source
            initial_warnings = parser_warnings
            raw_model_response = None
        else:
            initial_warnings = model_warnings

    if parsed is None:
        parsed = finalize_invoice_json(None, parser_text)

    errors = validate_extracted_document_json(parsed)
    data = add_arca_integration_fields(parsed, parser_text)
    warnings = assess_document_quality(data, parser_text)
    used_model = extraction_source == "model"
    return {
        "ok": not errors,
        "source": extraction_source,
        "model": model_name,
        "confidence": calculate_confidence(data, errors, extraction_source, used_model, warnings),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "errors": errors,
        "warnings": warnings,
        "data": data,
        "raw_model_response": raw_model_response,
    }


class InvoiceApiHandler(BaseHTTPRequestHandler):
    server_version = "FacturaTrainingAPI/0.1"

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
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
                    "ocr": get_ocr_status(),
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
        if not self.is_authorized():
            json_response(self, HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "API key invalida o faltante."})
            return

        try:
            request_id = str(uuid.uuid4())
            request = self.read_extract_request()
            result = extract_document(
                request["ocr_text"],
                filename=request["filename"],
                use_model=request["use_model"],
                model_choice=request["model"],
                max_new_tokens=request["max_new_tokens"],
                model_policy=request["model_policy"],
                min_confidence=request["min_confidence"],
            )
            ocr_attempts = [request["text_extractor"]] if request["text_extractor"] else []
            if (
                request.get("ocr_policy") == "auto"
                and request.get("file_bytes")
                and not (
                    (request.get("text_extractor") or {}).get("method") == "ocr"
                    and (request.get("text_extractor") or {}).get("multipass")
                )
                and (not result["ok"] or result["confidence"] < request["min_confidence"])
            ):
                retry_text, retry_extractor = extract_upload_text(
                    request["file_bytes"],
                    request["filename"],
                    force_ocr=True,
                    ocr_lang=request["ocr_lang"],
                    ocr_dpi=request["ocr_dpi"],
                    ocr_multipass=True,
                )
                retry_result = extract_document(
                    retry_text,
                    filename=request["filename"],
                    use_model=request["use_model"],
                    model_choice=request["model"],
                    max_new_tokens=request["max_new_tokens"],
                    model_policy=request["model_policy"],
                    min_confidence=request["min_confidence"],
                )
                ocr_attempts.append(retry_extractor)
                if (retry_result["ok"] and not result["ok"]) or retry_result["confidence"] >= result["confidence"]:
                    request["ocr_text"] = retry_text
                    request["text_extractor"] = retry_extractor
                    result = retry_result
            parser_text = (
                f"Archivo: {request['filename']}\n{request['ocr_text']}"
                if request["filename"]
                else request["ocr_text"]
            )
            result["data"] = add_arca_integration_fields(result["data"], parser_text)
            result["warnings"] = assess_document_quality(result["data"], parser_text)
            result["confidence"] = calculate_confidence(
                result["data"],
                result["errors"],
                result["source"],
                result["source"] == "model",
                result["warnings"],
            )
            result["input"] = {
                "filename": request["filename"],
                "text_extractor": request["text_extractor"],
                "ocr_policy": request["ocr_policy"],
                "ocr_attempts": ocr_attempts,
                "ocr_text_length": len(request["ocr_text"]),
            }
            result["request_id"] = request_id
            append_log(
                {
                    "request_id": request_id,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "ok": result["ok"],
                    "source": result["source"],
                    "model": result["model"],
                    "confidence": result["confidence"],
                    "elapsed_ms": result["elapsed_ms"],
                    "errors": result["errors"],
                    "warnings": result["warnings"],
                    "input": result["input"],
                    "document_type": result["data"].get("document_type") if isinstance(result["data"], dict) else None,
                    "tipo_comprobante": result["data"].get("tipo_comprobante") if isinstance(result["data"], dict) else None,
                }
            )
            json_response(self, HTTPStatus.OK if result["ok"] else HTTPStatus.UNPROCESSABLE_ENTITY, result)
        except ValueError as error:
            json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
        except OcrUnavailableError as error:
            json_response(
                self,
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "ok": False,
                    "error": str(error),
                    "ocr": get_ocr_status(),
                    "install_hint": (
                        "Instala Tesseract localmente. En Ubuntu/WSL: "
                        "sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng"
                    ),
                },
            )
        except Exception as error:
            json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(error)})

    def is_authorized(self):
        if not API_KEY:
            return True
        provided = self.headers.get("X-API-Key") or ""
        return provided == API_KEY

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
        model_policy = str(get_first(fields, query, "model_policy", "fallback") or "fallback")
        if model_policy not in {"fallback", "low_confidence", "always"}:
            raise ValueError("model_policy debe ser 'fallback', 'low_confidence' o 'always'.")
        force_ocr = parse_bool(get_first(fields, query, "force_ocr", False), default=False)
        ocr_policy = str(get_first(fields, query, "ocr_policy", "auto") or "auto")
        if ocr_policy not in {"auto", "fast", "robust"}:
            raise ValueError("ocr_policy debe ser 'auto', 'fast' o 'robust'.")
        max_new_tokens = parse_int(get_first(fields, query, "max_new_tokens", 900), 900)
        try:
            min_confidence = float(get_first(fields, query, "min_confidence", 0.82))
        except (TypeError, ValueError):
            min_confidence = 0.82
        ocr_lang = str(get_first(fields, query, "ocr_lang", DEFAULT_OCR_LANG) or DEFAULT_OCR_LANG)
        ocr_dpi = parse_int(get_first(fields, query, "ocr_dpi", DEFAULT_OCR_DPI), DEFAULT_OCR_DPI)
        ocr_multipass_raw = get_first(fields, query, "ocr_multipass", None)
        if ocr_multipass_raw is None:
            ocr_multipass = ocr_policy == "robust"
        else:
            ocr_multipass = parse_bool(ocr_multipass_raw, default=DEFAULT_OCR_MULTIPASS)
        ocr_text = get_first(fields, query, "ocr_text", "")
        filename = None
        text_extractor = None

        uploaded = files.get("file") or files.get("pdf")
        if uploaded:
            filename = uploaded["filename"]
            ocr_text, text_extractor = extract_upload_text(
                uploaded["content"],
                filename,
                force_ocr=force_ocr,
                ocr_lang=ocr_lang,
                ocr_dpi=ocr_dpi,
                ocr_multipass=ocr_multipass,
            )
            if not ocr_text:
                raise ValueError("No se pudo extraer texto del archivo.")

        if not str(ocr_text).strip():
            raise ValueError("Envia un PDF en el campo 'file' o texto OCR en 'ocr_text'.")

        return {
            "filename": filename,
            "text_extractor": text_extractor,
            "ocr_text": str(ocr_text).strip(),
            "file_bytes": uploaded["content"] if uploaded else None,
            "ocr_lang": ocr_lang,
            "ocr_dpi": ocr_dpi,
            "ocr_policy": ocr_policy,
            "use_model": use_model,
            "model": model,
            "model_policy": model_policy,
            "min_confidence": min_confidence,
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
    if API_KEY:
        print("API key: requerida por header X-API-Key")
    if LOG_FILE:
        print(f"Logs JSONL: {LOG_FILE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
