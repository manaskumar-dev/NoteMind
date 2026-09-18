from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.utils.pdf_extract import validate_and_extract_text
from app import models, schemas

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/pdf", response_model=schemas.NoteOut, status_code=201)
async def upload_pdf(
    file: UploadFile = File(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    raw = await file.read()
    text = validate_and_extract_text(file, raw)

    note = models.Note(
        owner_id=user.id,
        title=file.filename.rsplit(".", 1)[0][:200] or "Untitled PDF",
        content_md=text,
        source="pdf",
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return note
