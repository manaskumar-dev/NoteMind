from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app import models, schemas

router = APIRouter(prefix="/tags", tags=["tags"])


@router.get("", response_model=list[schemas.TagOut])
def list_tags(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(models.Tag).filter(models.Tag.owner_id == user.id).order_by(models.Tag.name).all()
