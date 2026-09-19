# NoteMind — AI-Powered Study Note Manager

**NoteMind** is a full-stack note-taking app built for students and lifelong learners. It combines a rich-text editor, spaced-repetition reminders, and Google Gemini–powered AI tools (summarize, grammar fix, plain-language explanations) to turn passive notes into an active study workflow.

**Live demo:** [https://note-mind-c7m9.vercel.app](https://note-mind-c7m9.vercel.app)
**API:** [https://notemind-1.onrender.com](https://notemind-1.onrender.com) · [Health check](https://notemind-1.onrender.com/health)

> Backend is hosted on Render's free tier and may take 30–50 seconds to wake up on the first request after inactivity.

---

## Features

- **Auth** — JWT-based registration/login with bcrypt password hashing, "Remember me" (persistent vs. session-only tokens)
- **Rich-text notes** — bold/italic/underline, multi-color highlighting, keyboard shortcuts (`Ctrl+S` to save, `Ctrl+B/I/U` for formatting), live word/character count, autosave
- **AI tools** (Gemini, rate-limited per user)
  - **Summarize** — condenses a note, appended as a new section so original notes are never lost
  - **Fix Grammar** — corrects the full note in place
  - **Explain Simply** — rewrites a note in plain language with a real-life analogy
  - **Explore a Topic** — ask about *any* topic, not just your own notes, with a search-style input
- **Spaced repetition** — a `1 → 3 → 7 → 15` day review schedule per note, with a focused **Review Session** mode (title → reveal → Mark Reviewed / Skip) pulling directly from due notes
- **PDF import** — upload a PDF, extract its text, and it becomes a new note automatically
- **Organization** — tagging, search (title/content), relative timestamps ("2h ago")
- **Dark mode** — theme toggle with persistence, full component coverage (editor, menus, autofill fields)
- **Settings** — profile popover (Claude-style) with a compact modal for display name and theme, not a full-page takeover

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, SQLAlchemy, SQLite, Pydantic Settings |
| Auth | JWT (`python-jose`), `bcrypt` password hashing |
| AI | Google Gemini API (`google-generativeai`) |
| PDF parsing | `pypdf` |
| Sanitization | `markdown` + `bleach` (XSS-safe HTML rendering) |
| Frontend | Vanilla HTML / CSS / JavaScript (no framework, no build step) |
| Hosting | Render (API) · Vercel (static frontend) |

---

## Project Structure

```
backend/
  app/
    main.py                 App entrypoint — CORS, global exception handler, /health
    config.py                Settings loaded strictly from environment variables
    database.py               SQLAlchemy engine/session
    models.py                  User, Note, Tag, Reminder ORM models
    schemas.py                Pydantic request/response schemas
    security.py                Password hashing + JWT issuing/verification
    dependencies.py           get_current_user — JWT → authenticated user
    routers/
      auth.py                   Register / login
      notes.py                  CRUD, ownership checks, search, reminders
      tags.py                    List tags
      upload.py                 PDF upload → validate → extract → create note
      ai.py                      Summarize / grammar fix / explain (rate-limited)
    utils/
      markdown_sanitize.py     Markdown → HTML via a strict bleach allowlist
      pdf_extract.py             Content-type, size, and magic-byte validation
      spaced_repetition.py      1 → 3 → 7 → 15 day interval scheduling
      rate_limit.py               Per-user fixed-window limiter (in-memory)
      gemini_client.py           Injection-resistant prompts for each AI tool
  requirements.txt
  .env.example
  runtime.txt                 Pins Python version for reproducible builds

frontend/
  index.html
  css/style.css
  js/
    api.js        Fetch wrapper, JWT in Authorization header
    auth.js        Login / register / password visibility toggle
    dashboard.js  Home page stats and recent notes
    notes.js       Notes CRUD, editor, AI panel, reminders, review session, settings
    notify.js       Browser notification permission/reminders
    toast.js        Toast notification system
    app.js          View routing, theme, profile menu, settings modal
```

---

## Running Locally

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in SECRET_KEY and GEMINI_API_KEY
uvicorn app.main:app --reload --port 8000
```

- Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- Generate a secret key: `python3 -c "import secrets; print(secrets.token_hex(32))"`
- Verify it's running: `http://localhost:8000/health` → `{"status": "ok", "db": "ok"}`

### Frontend

```bash
cd frontend
python3 -m http.server 5500
```

Open `http://localhost:5500`. Do **not** open `index.html` directly via `file://` — the backend's CORS policy requires a real HTTP origin.

Confirm your local backend's `.env` includes this origin:
```
CORS_ORIGINS=http://localhost:5500,http://127.0.0.1:5500
```

---

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `SECRET_KEY` | JWT signing secret — random, never committed | `a1b2c3...` (64 chars) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime | `60` |
| `DATABASE_URL` | SQLAlchemy connection string | `sqlite:///./notes.db` |
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `GEMINI_MODEL` | Gemini model name | `gemini-3.1-flash-lite` |
| `CORS_ORIGINS` | Comma-separated allowed origins (no trailing slash) | `https://your-app.vercel.app` |
| `MAX_PDF_SIZE_MB` | PDF upload size limit | `5` |
| `AI_RATE_LIMIT_PER_MINUTE` | Per-user AI call limit | `5` |

---

## Deployment

**Backend (Render)**
1. New Web Service → connect the repo, root directory `backend`
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add the environment variables above (`cors_origins` = your Vercel URL, no trailing slash)
5. A `runtime.txt` pins the Python version so `pip` installs prebuilt wheels instead of compiling from source

**Frontend (Vercel)**
1. New Project → import the repo, root directory `frontend`
2. Framework preset: **Other** (static site, no build step)
3. Update `API_BASE` in `js/api.js` to your Render URL before deploying

> ⚠️ SQLite on Render's free tier is stored on an ephemeral filesystem — data resets on redeploy/restart. For persistent data, switch `DATABASE_URL` to a managed Postgres instance (no code changes needed beyond adding `psycopg2-binary`).

---

## Key Design Decisions

- **Ownership-based authorization** — every note/reminder route re-fetches the row and compares `owner_id` against the JWT-derived user; the client never supplies a trusted user ID.
- **XSS prevention** — raw markdown is stored; HTML is only ever produced by a sanitizer that pipes `markdown` output through a `bleach` allowlist. The client-side preview also escapes text before formatting.
- **PDF validation** — checks declared content-type, size against `MAX_PDF_SIZE_MB`, and the `%PDF-` magic header before parsing, so malformed or oversized files fail fast.
- **Secrets management** — `pydantic-settings` reads exclusively from environment variables; nothing is hardcoded or logged (the global exception handler logs only the exception type and route, never a stack trace, to the client).
- **Rate limiting** — AI endpoints use a per-user in-memory fixed-window counter; sufficient for a single-instance deployment, would move to Redis for a multi-process setup.
- **Prompt injection resistance** — every Gemini prompt explicitly instructs the model to treat note/topic content as data, not instructions, mitigating injection attempts embedded in user text.
- **Search** — plain SQL `LIKE` via SQLAlchemy `.filter()`; no raw SQL, no vector store — a deliberate scope choice, not an oversight, given the dataset size.
- **Spaced repetition** — a `Reminder` row per note stores a `stage` (0–3) and `next_review_at`; completing a review advances the stage and recomputes the date from a fixed `[1, 3, 7, 15]` day table, no external scheduler required.
- **Stateless auth** — no server-side session store; JWTs carry their own validity, so the API can scale horizontally without sticky sessions.

## Known Limitations & Future Work

- No refresh token / rotation — a single short-lived JWT is used for simplicity; a production system would add refresh tokens with a revocation store.
- No version history for notes — would require a second table and conflict handling for concurrent autosave.
- SQLite is not suited for concurrent multi-instance deployment — a Postgres migration is a natural next step.
- No collaborative/multi-user note sharing — the current ownership model assumes a single owner per note by design.

---

## License

MIT — free to use, modify, and learn from.
