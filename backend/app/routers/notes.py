from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.utils.markdown_sanitize import render_markdown_safe
from app.utils.spaced_repetition import next_review_date, advance_stage
from app import models, schemas

router = APIRouter(prefix="/notes", tags=["notes"])


def _get_owned_note(note_id: int, user: models.User, db: Session) -> models.Note:
    """Fetches a note and enforces ownership — the single choke point every route uses."""
    note = db.query(models.Note).filter(models.Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    if note.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized for this note")
    return note


def _resolve_tags(names: list[str], user: models.User, db: Session) -> list[models.Tag]:
    tags = []
    for name in names:
        tag = db.query(models.Tag).filter(models.Tag.owner_id == user.id, models.Tag.name == name).first()
        if not tag:
            tag = models.Tag(owner_id=user.id, name=name)
            db.add(tag)
            db.flush()
        tags.append(tag)
    return tags


@router.post("", response_model=schemas.NoteOut, status_code=status.HTTP_201_CREATED)
def create_note(payload: schemas.NoteCreate, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    note = models.Note(owner_id=user.id, title=payload.title, content_md=payload.content_md)
    note.tags = _resolve_tags(payload.tags, user, db)
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


@router.get("", response_model=list[schemas.NoteOut])
def list_notes(
    q: str | None = None,
    tag: str | None = None,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Search is plain SQL LIKE on title/content — no embeddings, per project scope."""
    query = db.query(models.Note).filter(models.Note.owner_id == user.id)
    if q:
        like = f"%{q}%"
        query = query.filter((models.Note.title.like(like)) | (models.Note.content_md.like(like)))
    if tag:
        query = query.join(models.Note.tags).filter(models.Tag.name == tag.strip().lower())
    return query.order_by(models.Note.updated_at.desc()).all()


@router.get("/{note_id}", response_model=schemas.NoteOut)
def get_note(note_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _get_owned_note(note_id, user, db)


@router.get("/{note_id}/render", response_model=schemas.NoteRenderedOut)
def render_note(note_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    note = _get_owned_note(note_id, user, db)
    return schemas.NoteRenderedOut(id=note.id, title=note.title, content_html=render_markdown_safe(note.content_md))


@router.put("/{note_id}", response_model=schemas.NoteOut)
def update_note(note_id: int, payload: schemas.NoteUpdate, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    note = _get_owned_note(note_id, user, db)
    if payload.title is not None:
        note.title = payload.title
    if payload.content_md is not None:
        note.content_md = payload.content_md
    if payload.tags is not None:
        note.tags = _resolve_tags(payload.tags, user, db)
    db.commit()
    db.refresh(note)
    return note


@router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_note(note_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    note = _get_owned_note(note_id, user, db)
    db.delete(note)
    db.commit()


# ---- Reminders (spaced repetition) ----

@router.post("/{note_id}/reminder/start", response_model=schemas.ReminderOut)
def start_reminder(note_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    note = _get_owned_note(note_id, user, db)
    if note.reminder:
        raise HTTPException(status_code=400, detail="Reminder already active for this note")
    reminder = models.Reminder(note_id=note.id, stage=0, next_review_at=next_review_date(0))
    db.add(reminder)
    db.commit()
    db.refresh(reminder)
    return reminder


@router.post("/{note_id}/reminder/complete", response_model=schemas.ReminderOut)
def complete_review(note_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    note = _get_owned_note(note_id, user, db)
    if not note.reminder:
        raise HTTPException(status_code=404, detail="No active reminder for this note")
    reminder = note.reminder
    reminder.stage = advance_stage(reminder.stage)
    reminder.next_review_at = next_review_date(reminder.stage)
    db.commit()
    db.refresh(reminder)
    return reminder


@router.get("/due/list", response_model=list[schemas.NoteOut])
def due_notes(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return (
        db.query(models.Note)
        .join(models.Reminder)
        .filter(models.Note.owner_id == user.id, models.Reminder.next_review_at <= datetime.utcnow())
        .all()
    )
