from fastapi import APIRouter, UploadFile, File
import fitz  # PyMuPDF

router = APIRouter()

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    pdf_bytes = await file.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in doc:
        text += page.get_text() + "\n"

    doc.close()

    title = file.filename.replace(".pdf", "")

    return {
        "title": title,
        "content": text.strip()
    }