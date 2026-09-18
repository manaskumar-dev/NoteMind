from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator


# ---- Auth ----
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---- Tags ----
class TagOut(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


# ---- Reminders ----
class ReminderOut(BaseModel):
    stage: int
    next_review_at: datetime
    completed: bool

    class Config:
        from_attributes = True


# ---- Notes ----
class NoteCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content_md: str = Field(default="", max_length=50_000)
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, v: list[str]) -> list[str]:
        return [t.strip().lower() for t in v if t.strip()][:20]


class NoteUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    content_md: str | None = Field(default=None, max_length=50_000)
    tags: list[str] | None = None

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return [t.strip().lower() for t in v if t.strip()][:20]


class NoteOut(BaseModel):
    id: int
    title: str
    content_md: str
    source: str
    created_at: datetime
    updated_at: datetime
    tags: list[TagOut]
    reminder: ReminderOut | None = None

    class Config:
        from_attributes = True


class NoteRenderedOut(BaseModel):
    """Note with sanitized HTML for safe display."""
    id: int
    title: str
    content_html: str


# ---- AI ----
class SummarizeRequest(BaseModel):
    note_id: int


class RefineRequest(BaseModel):
    note_id: int


class AIResultOut(BaseModel):
    note_id: int
    result: str
