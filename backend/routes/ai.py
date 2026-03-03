from fastapi import APIRouter
from transformers import pipeline

router = APIRouter()

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

@router.post("/summarize")
def summarize(data: dict):
    text = data.get("content", "")

    if not text:
        return {"error": "No content provided"}

    result = summarizer(
    text,
    max_length=50,
    min_length=15,
    do_sample=False
)

    return {"summary": result[0]["summary_text"]}