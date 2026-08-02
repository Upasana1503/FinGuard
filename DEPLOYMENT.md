# Deploying FinGuard

Two independent deployments: the FastAPI backend (does the actual guardrail
inference) and the Streamlit frontend (talks to the backend over HTTP). Deploy
the backend first — the frontend needs its URL.

## 1. Database — Neon (free Postgres)

1. Create an account at neon.tech, create a new project.
2. Copy the connection string it gives you (starts with `postgresql://`).
3. Keep it — you'll paste it into the backend's environment variables next.

## 2. Backend — Render

1. Push this repo to GitHub (already done if you're reading this from the
   FinGuard repo).
2. On render.com: New → Web Service → connect the repo.
3. Root directory: `backend`
4. Environment: **Docker** (it'll pick up `backend/Dockerfile` automatically).
5. Environment variables: add `DATABASE_URL` = the Neon connection string
   from step 1.
6. Instance type: pick one with **at least 2GB RAM** — Qwen2.5-1.5B needs
   real memory headroom; Render's free 512MB tier will OOM on model load.
7. Deploy. First boot will be slow (~1-2 min: downloading model weights +
   the 3-call warmup baked into startup) — that's expected, only happens
   once per deploy, not per request.
8. Once live, note the backend's public URL (e.g.
   `https://finguard-api.onrender.com`) — the frontend needs it.

Sanity check once deployed:
```bash
curl https://YOUR-BACKEND-URL.onrender.com/health
# -> {"status":"ok"}
```

## 3. Frontend — Streamlit Community Cloud

1. On share.streamlit.io: New app → connect the repo.
2. Main file path: `frontend/streamlit_app.py`
3. In the app's "Secrets" settings, add:
   ```
   FINGUARD_BACKEND_URL = "https://YOUR-BACKEND-URL.onrender.com"
   ```
4. Deploy. That's the live site you share/link from your resume.

## Local development (no external services needed)

Backend (uses a local SQLite file automatically if `DATABASE_URL` isn't set):
```bash
cd backend
pip install -r requirements.txt   # or reuse an existing venv with torch/transformers/sklearn already installed
uvicorn app.main:app --reload --port 8123
```

Frontend, in a second terminal:
```bash
cd frontend
pip install -r requirements.txt
FINGUARD_BACKEND_URL=http://127.0.0.1:8123 streamlit run streamlit_app.py
```

## Notes on what's NOT in this product (on purpose)

- **No multi-turn/trajectory tracking.** Built and tested in the research
  repo (`src/probe_v4.py`), found not to actually discriminate malicious
  from benign sessions yet. Shipping it here would mean shipping a feature
  that doesn't work — left out until it's actually validated.
- **Rate limiting is in-memory, per-process** (`backend/app/rate_limit.py`).
  Fine for a single Render instance; if this ever needs to scale to
  multiple instances, that needs to move to Redis or the database.
- **Policy-category attribution (the "why was this flagged" categories) is
  a known weak point** — in testing it correctly separates malicious from
  benign (~solid), but picking the *specific* right category among the 8
  is only accurate about 1 in 8 times. The binary flag is trustworthy; the
  named category next to it is a best guess, not a certainty — the
  disclaimer field in every response says this explicitly.
