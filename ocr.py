import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


DEFAULT_OCR_LANG = os.environ.get("OCR_LANG", "spa+eng")
DEFAULT_OCR_DPI = int(os.environ.get("OCR_DPI", "220"))
DEFAULT_TESSERACT_PSM = os.environ.get("OCR_PSM", "6")
DEFAULT_OCR_MULTIPASS = os.environ.get("OCR_MULTIPASS", "0").lower() not in {"0", "false", "no"}


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
        "multipass": DEFAULT_OCR_MULTIPASS,
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


def preprocess_image_variants(image):
    base = image.convert("RGB")
    variants = [("original", base)]

    gray = ImageOps.grayscale(base)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.7)
    gray = ImageEnhance.Sharpness(gray).enhance(1.4)
    enhanced = gray.convert("RGB")
    variants.append(("enhanced", enhanced))

    denoised = gray.filter(ImageFilter.MedianFilter(size=3))
    threshold = denoised.point(lambda pixel: 255 if pixel > 175 else 0, mode="1").convert("RGB")
    variants.append(("threshold", threshold))
    return variants


def score_ocr_text(text):
    value = str(text or "")
    if not value.strip():
        return -10_000

    upper = value.upper()
    score = min(len(value), 12_000) / 100
    score += 35 * len(re.findall(r"\b\d{2}-?\d{8}-?\d\b", value))
    score += 30 * len(re.findall(r"\bCAE\b|\bCAI\b", upper))
    score += 25 * len(re.findall(r"\bFACTURA\b|\bRECIBO\b|\bNOTA\s+DE\s+(?:CREDITO|DEBITO)\b", upper))
    score += 18 * len(re.findall(r"\b(?:TOTAL|SUBTOTAL|NETO\s+GRAVADO|IVA|PERCEP|TRIBUTOS?)\b", upper))
    score += 12 * len(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", value))
    score += 8 * len(re.findall(r"\$?\s*\d{1,3}(?:\.\d{3})*,\d{2}|\$?\s*\d+\.\d{2}", value))
    score -= 20 * value.count("�")
    score -= 3 * len(re.findall(r"[|{}\[\]~]{2,}", value))
    return score


def run_tesseract(image, lang=DEFAULT_OCR_LANG, psm=DEFAULT_TESSERACT_PSM, timeout=60):
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


def image_to_text(image, lang=DEFAULT_OCR_LANG, psm=DEFAULT_TESSERACT_PSM, timeout=60):
    return run_tesseract(image, lang=lang, psm=psm, timeout=timeout)


def image_to_best_text(image, lang=DEFAULT_OCR_LANG, timeout=60, multipass=DEFAULT_OCR_MULTIPASS):
    passes = [("original", DEFAULT_TESSERACT_PSM, image)]
    if multipass:
        passes = [
            (variant_name, psm, variant_image)
            for variant_name, variant_image in preprocess_image_variants(image)
            for psm in ("6", "4")
        ]

    results = []
    best = None
    for variant_name, psm, variant_image in passes:
        try:
            text = run_tesseract(variant_image, lang=lang, psm=psm, timeout=timeout)
        except OcrUnavailableError:
            if results:
                continue
            raise
        score = score_ocr_text(text)
        result = {
            "variant": variant_name,
            "psm": str(psm),
            "score": round(score, 2),
            "length": len(text or ""),
        }
        results.append(result)
        if best is None or score > best[0]:
            best = (score, text or "", result)

    if best is None:
        return "", {"variant": "none", "psm": None, "score": 0, "passes": results}
    return best[1].strip(), {**best[2], "passes": results}


def ocr_pdf_bytes(pdf_bytes, lang=DEFAULT_OCR_LANG, dpi=DEFAULT_OCR_DPI, max_pages=None, multipass=DEFAULT_OCR_MULTIPASS):
    pages = render_pdf_pages(pdf_bytes, dpi=dpi, max_pages=max_pages)
    texts = []
    page_results = []
    for index, page in enumerate(pages, start=1):
        text, page_meta = image_to_best_text(page, lang=lang, multipass=multipass)
        page_results.append({"page": index, **page_meta})
        if text:
            texts.append(f"--- OCR PAGE {index} ---\n{text}")
    return "\n\n".join(texts).strip(), {
        "engine": "tesseract",
        "lang": lang,
        "dpi": dpi,
        "pages": len(pages),
        "multipass": multipass,
        "page_results": page_results,
    }


def ocr_image_bytes(image_bytes, lang=DEFAULT_OCR_LANG, multipass=DEFAULT_OCR_MULTIPASS):
    image = load_image(image_bytes)
    text, image_meta = image_to_best_text(image, lang=lang, multipass=multipass)
    return text.strip(), {
        "engine": "tesseract",
        "lang": lang,
        "dpi": None,
        "pages": 1,
        "multipass": multipass,
        "page_results": [{"page": 1, **image_meta}],
    }
