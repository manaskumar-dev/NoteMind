import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import Base, engine
from app.routers import auth, notes, tags, upload, ai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("notes_app")

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Notes API")

# Strict CORS: explicit origin list from .env, never "*" while allow_credentials=True
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Logs the error server-side only; client never sees stack traces or internals."""
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(auth.router)
app.include_router(notes.router)
app.include_router(tags.router)
app.include_router(upload.router)
app.include_router(ai.router)


@app.get("/health")
def health():
    return {"status": "ok"}
