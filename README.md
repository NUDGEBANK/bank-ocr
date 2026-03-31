# bank-ocr

FastAPI based OCR server for certificate verification.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Endpoint

- `GET /health`
- `POST /ocr/extract`

Current version accepts image files such as `jpg`, `jpeg`, `png`, and `pdf`.
