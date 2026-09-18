# AI Notes — Study Note Manager

FastAPI + SQLite backend, vanilla HTML/CSS/JS frontend, Gemini for AI (free tier).

## Run the backend
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in SECRET_KEY and GEMINI_API_KEY
uvicorn app.main:app --reload --port 8000
```
Get a free Gemini key at https://aistudio.google.com/apikey.
Generate `SECRET_KEY` with: `python3 -c "import secrets; print(secrets.token_hex(32))"`.

## Run the frontend
Serve `frontend/` as static files (do not open via `file://`, since CORS needs a real origin):
```bash
cd frontend
python3 -m http.server 5500
```
Open http://localhost:5500. Make sure `CORS_ORIGINS` in `.env` includes that origin.

## Project layout
```
backend/app/
  config.py         settings loaded strictly from .env
  database.py        SQLAlchemy engine/session
  models.py           User, Note, Tag, Reminder (ORM only, no raw SQL)
  schemas.py         Pydantic request/response validation
  security.py         bcrypt hashing + JWT
  dependencies.py    get_current_user (JWT -> user, used on every protected route)
  routers/
    auth.py            register/login
    notes.py           CRUD, ownership checks, LIKE search, spaced-repetition reminders
    tags.py            list tags
    upload.py          PDF upload -> validated -> extracted -> note
    ai.py               Gemini summarize/refine, rate-limited
  utils/
    markdown_sanitize.py  markdown -> HTML -> bleach allowlist (XSS prevention)
    pdf_extract.py         type/size/magic-byte validation + pypdf extraction
    spaced_repetition.py   1 -> 3 -> 7 -> 15 day interval logic
    rate_limit.py           in-memory per-user fixed-window limiter
    gemini_client.py        strict, injection-resistant prompts
  main.py              CORS (explicit origins, no wildcard+credentials), global
                        exception handler (no stack traces to client)
frontend/
  index.html, css/style.css
  js/api.js    fetch wrapper, JWT in Authorization header
  js/auth.js    login/register
  js/notes.js  CRUD, search, tags, autosave (debounced), AI actions, reminders, PDF upload
  js/app.js    view bootstrap
```

## Key design decisions (interview notes)
- **Ownership checks**: every note/reminder route re-fetches the row and compares `owner_id`
  to the JWT-derived user, never trusts a client-supplied user id.
- **XSS**: raw markdown is stored; HTML is only produced by `render_markdown_safe`, which
  pipes `markdown` output through a `bleach` allowlist. The client-side preview also
  escapes text before any regex-based formatting, so it can't inject HTML either.
- **PDF validation**: checks declared content-type, byte size against `MAX_PDF_SIZE_MB`,
  and the `%PDF-` magic header before parsing, so junk/oversized/mislabeled files 400 out
  early rather than reaching `pypdf`.
- **Secrets**: `Settings` (pydantic-settings) reads only from `.env`; nothing is hardcoded,
  nothing is logged (the global exception handler logs only the exception type + route).
- **Rate limiting**: AI endpoints only, per-user fixed-window counter in memory — enough
  for a demo; would move to Redis for multi-process deployments.
- **AI prompts**: summarize and refine prompts explicitly tell the model to ignore any
  instructions embedded in the note content, mitigating prompt injection from note text.
- **Search**: plain SQL `LIKE` on title/content via SQLAlchemy `.filter()`, no raw SQL,
  no vector store — matches the stated scope.
- **Spaced repetition**: a `Reminder` row per note stores `stage` (0-3) and
  `next_review_at`; completing a review advances the stage and recomputes the date from
  the fixed `[1, 3, 7, 15]` day table — no external scheduler needed for a demo.
