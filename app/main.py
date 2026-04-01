from io import BytesIO

import easyocr
import fitz
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError


MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_PDF_PAGES = 3
PDF_RENDER_SCALE = 1.5
MAX_IMAGE_WIDTH = 1800
MAX_IMAGE_HEIGHT = 1800
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

app = FastAPI(title="bank-ocr")
reader = easyocr.Reader(["ko", "en"], gpu=False)


class HealthResponse(BaseModel):
    status: str


class OcrExtractResponse(BaseModel):
    success: bool
    filename: str
    content_type: str
    extracted_text: str
    lines: list[str]
    line_count: int


def validate_filename(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    lower_name = filename.lower()
    if not any(lower_name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Only jpg, jpeg, and png files are supported",
        )

    return filename


def extract_lines_from_image_bytes(file_bytes: bytes) -> list[str]:
    image = Image.open(BytesIO(file_bytes)).convert("RGB")

    if image.width > MAX_IMAGE_WIDTH or image.height > MAX_IMAGE_HEIGHT:
        image.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT))

    results = reader.readtext(np.array(image))
    return [text.strip() for _, text, _ in results if text.strip()]


def extract_lines_from_pdf_bytes(file_bytes: bytes) -> list[str]:
    lines: list[str] = []

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf_document:
            page_count = min(pdf_document.page_count, MAX_PDF_PAGES)

            for page_index in range(page_count):
                page = pdf_document.load_page(page_index)
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)
                )
                image_bytes = pixmap.tobytes("png")
                lines.extend(extract_lines_from_image_bytes(image_bytes))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail="Invalid PDF file") from exc

    return lines


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ocr/extract", response_model=OcrExtractResponse)
async def extract_text(file: UploadFile = File(...)) -> OcrExtractResponse:
    filename = validate_filename(file.filename)

    if not file.content_type or (
        not file.content_type.startswith("image/")
        and file.content_type != "application/pdf"
    ):
        raise HTTPException(status_code=400, detail="Only image or pdf uploads are supported")

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size must be 10MB or less")

        if filename.lower().endswith(".pdf"):
            lines = extract_lines_from_pdf_bytes(file_bytes)
        else:
            lines = extract_lines_from_image_bytes(file_bytes)

        extracted_text = " ".join(lines)

        if not lines:
            raise HTTPException(status_code=422, detail="No text detected from image")

        return OcrExtractResponse(
            success=True,
            filename=filename,
            content_type=file.content_type,
            extracted_text=extracted_text,
            lines=lines,
            line_count=len(lines),
        )
    except HTTPException:
        raise
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {exc}") from exc
