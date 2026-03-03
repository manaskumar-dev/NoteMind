from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from database import Base

class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="Untitled")
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_reviewed = Column(DateTime, nullable=True)
    next_revision = Column(DateTime, nullable=True)
    review_count = Column(Integer, default=0)