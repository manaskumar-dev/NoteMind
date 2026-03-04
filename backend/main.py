from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import upload


# Database
from database import Base, engine

# Routes
from routes import notes
from routes import ai

# Create FastAPI app
app = FastAPI(
    title="NoteMind API",
    description="AI-Powered Knowledge & Memory Management System",
    version="1.0.0"
)

# Create database tables automatically
Base.metadata.create_all(bind=engine)

# Enable CORS (for frontend connection later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(notes.router)
app.include_router(ai.router)
app.include_router(upload.router)

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "NoteMind Backend Running Successfully 🚀"
    }