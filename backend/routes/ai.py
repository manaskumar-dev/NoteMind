from fastapi import APIRouter
from pydantic import BaseModel
from transformers import pipeline

router = APIRouter()

class TextInput(BaseModel):
    content: str


# Load summarization model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)
# summarizer = pipeline(
#     "summarization",
#     model="sshleifer/distilbart-cnn-12-6"
# )


@router.post("/summarize")
def summarize(data: TextInput):

    text = data.content.strip()

    result = summarizer(
        text,
        max_length=40,
        min_length=12,
        do_sample=False
    )

    summary = result[0]["summary_text"]

    return {"summary": summary}


@router.post("/improve")
def improve(data: TextInput):

    text = data.content.strip()

    prompt = (
        "Improve the following study notes. "
        "Fix grammar and spelling mistakes, remove repeated sentences, "
        "remove unnecessary words, and make the text clearer while keeping the original meaning.\n\n"
        f"{text}"
    )

    result = summarizer(
        prompt,
        max_length=200,
        min_length=60,
        do_sample=False
    )

    improved_text = result[0]["summary_text"]

    return {"improved": improved_text}
