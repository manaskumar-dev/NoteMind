from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Note
from revision_scheduler import calculate_next_revision
from datetime import datetime

router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create Note
@router.post("/notes")
def create_note(note: dict, db: Session = Depends(get_db)):
    db_note = Note(
        title=note.get("title", "Untitled"),
        content=note.get("content", "")
    )
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    return db_note

# Get All Notes
@router.get("/notes")
def get_notes(db: Session = Depends(get_db)):
    return db.query(Note).all()

# Get Single Note
@router.get("/notes/{note_id}")
def get_note(note_id: int, db: Session = Depends(get_db)):
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note

# Update Note
@router.put("/notes/{note_id}")
def update_note(note_id: int, updated_data: dict, db: Session = Depends(get_db)):
    note = db.query(Note).filter(Note.id == note_id).first()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note.title = updated_data.get("title", note.title)
    note.content = updated_data.get("content", note.content)

    db.commit()
    db.refresh(note)

    return note

# Delete Note
@router.delete("/notes/{note_id}")
def delete_note(note_id: int, db: Session = Depends(get_db)):
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    db.delete(note)
    db.commit()
    return {"message": "Note deleted successfully"}

# Schedule Revision (Spaced Repetition)
@router.post("/schedule-revision/{note_id}")
def schedule_revision(note_id: int, db: Session = Depends(get_db)):
    note = db.query(Note).filter(Note.id == note_id).first()

    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note.review_count += 1
    note.last_reviewed = datetime.utcnow()
    note.next_revision = calculate_next_revision(note.review_count)

    db.commit()
    db.refresh(note)

    return note