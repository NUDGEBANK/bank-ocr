from io import BytesIO

import easyocr
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image


app = FastAPI(title="bank-ocr")
reader = easyocr.Reader(["ko", "en"], gpu=False)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ocr/extract")
async def extract_text(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        results = reader.readtext(image)
        lines = [text for _, text, _ in results]

        return {
            "success": True,
            "filename": file.filename,
            "extracted_text": " ".join(lines),
            "lines": lines,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {exc}") from exc
