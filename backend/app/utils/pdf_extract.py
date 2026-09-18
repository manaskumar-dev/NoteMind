from io import BytesIO

from fastapi import UploadFile, HTTPException
from pypdf import PdfReader

from app.config import settings

MAX_BYTES = settings.max_pdf_size_mb * 1024 * 1024


def validate_and_extract_text(file: UploadFile, raw: bytes) -> str:
    """Validates content-type/size, then extracts text. Raises 400 on any violation."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=400, detail=f"PDF exceeds {settings.max_pdf_size_mb}MB limit")
    if not raw.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    try:
        reader = PdfReader(BytesIO(raw))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read PDF contents")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")
    return text[:50_000]
