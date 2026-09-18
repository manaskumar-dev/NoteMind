from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.utils.rate_limit import enforce_rate_limit
from app.utils.gemini_client import summarize_text, refine_grammar
from app import models, schemas

router = APIRouter(prefix="/ai", tags=["ai"])


def _owned_note(note_id: int, user: models.User, db: Session) -> models.Note:
    note = db.query(models.Note).filter(models.Note.id == note_id).first()
    if not note or note.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.post("/summarize", response_model=schemas.AIResultOut)
def summarize(payload: schemas.SummarizeRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    enforce_rate_limit(user.id)
    note = _owned_note(payload.note_id, user, db)
    result = summarize_text(note.content_md)
    return schemas.AIResultOut(note_id=note.id, result=result)


@router.post("/refine", response_model=schemas.AIResultOut)
def refine(payload: schemas.RefineRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    enforce_rate_limit(user.id)
    note = _owned_note(payload.note_id, user, db)
    result = refine_grammar(note.content_md)
    return schemas.AIResultOut(note_id=note.id, result=result)
