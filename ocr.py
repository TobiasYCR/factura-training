import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


DEFAULT_OCR_LANG = os.environ.get("OCR_LANG", "spa+eng")
DEFAULT_OCR_DPI = int(os.environ.get("OCR_DPI", "220"))
DEFAULT_TESSERACT_PSM = os.environ.get("OCR_PSM", "6")


class OcrUnavailableError(RuntimeError):
    pass


def find_tesseract():
    configured = os.environ.get("TESSERACT_CMD")
    if configured and Path(configured).exists():
        return configured
    return shutil.which("tesseract")


def get_ocr_status():
    tesseract = find_tesseract()
    status = {
        "engine": "tesseract",
        "available": bool(tesseract),
        "command": tesseract,
        "default_lang": DEFAULT_OCR_LANG,
        "default_dpi": DEFAULT_OCR_DPI,
    }
    if not tesseract:
        status["message"] = "Tesseract no esta instalado o no esta en PATH."
        return status

    try:
        result = subprocess.run(
            [tesseract, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        first_line = (result.stdout or result.stderr).splitlines()[0]
        status["version"] = first_line.strip()
    except Exception as error:
        status["available"] = False
        status["message"] = str(error)
    return status


def render_pdf_pages(pdf_bytes, dpi=DEFAULT_OCR_DPI, max_pages=None):
    errors = []
    try:
        from pdf2image import convert_from_bytes

        pages = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="png")
        if max_pages is not None:
            pages = pages[:max_pages]
        return pages
    except Exception as error:
        errors.append(f"pdf2image: {error}")

    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(pdf_bytes)
        page_count = len(pdf)
        if max_pages is not None:
            page_count = min(page_count, max_pages)
        scale = dpi / 72
        return [pdf[index].render(scale=scale).to_pil().convert("RGB") for index in range(page_count)]
    except Exception as error:
        errors.append(f"pypdfium2: {error}")

    raise OcrUnavailableError("No se pudo renderizar el PDF para OCR. " + " | ".join(errors))


def load_image(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def image_to_text(image, lang=DEFAULT_OCR_LANG, psm=DEFAULT_TESSERACT_PSM, timeout=60):
    tesseract = find_tesseract()
    if not tesseract:
        raise OcrUnavailableError(
            "Tesseract no esta instalado o no esta en PATH. Instala Tesseract localmente y volve a probar."
        )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as file:
        image_path = Path(file.name)

    try:
        image.save(image_path)
        command = [
            tesseract,
            str(image_path),
            "stdout",
            "-l",
            lang,
            "--psm",
            str(psm),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "Tesseract fallo sin mensaje.").strip()
            if lang != "eng" and "Error opening data file" in message:
                return image_to_text(image, lang="eng", psm=psm, timeout=timeout)
            raise OcrUnavailableError(message)
        return result.stdout.strip()
    finally:
        image_path.unlink(missing_ok=True)


def ocr_pdf_bytes(pdf_bytes, lang=DEFAULT_OCR_LANG, dpi=DEFAULT_OCR_DPI, max_pages=None):
    pages = render_pdf_pages(pdf_bytes, dpi=dpi, max_pages=max_pages)
    texts = []
    for index, page in enumerate(pages, start=1):
        text = image_to_text(page, lang=lang)
        if text:
            texts.append(f"--- OCR PAGE {index} ---\n{text}")
    return "\n\n".join(texts).strip(), {
        "engine": "tesseract",
        "lang": lang,
        "dpi": dpi,
        "pages": len(pages),
    }


def ocr_image_bytes(image_bytes, lang=DEFAULT_OCR_LANG):
    image = load_image(image_bytes)
    text = image_to_text(image, lang=lang)
    return text.strip(), {
        "engine": "tesseract",
        "lang": lang,
        "dpi": None,
        "pages": 1,
    }
