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

## What's actually deployed right now

General-purpose activation-probing detector (Qwen2.5-1.5B + logistic
regression), trained on deepset + 15,000 WildGuardMix examples. Real,
validated numbers (see PROJECT_SUMMARY.md for the full round-by-round
history): **AUC-ROC 0.889, F1 0.794, FPR 8.6% on the full WildGuardTest
benchmark (1699 ex.)** — and head-to-head against IBM Granite Guardian
(real open-source product, same 400-example subset): FinGuard's FPR is
8.6% vs Granite's 46.0%, with better F1 (0.794 vs 0.753), though lower
recall (72.9% vs 94.9%).

## Notes on what's NOT in this product (on purpose or as a known gap)

- **No policy-category evidence layer.** An earlier finance-scoped version
  had one (mapping flags to named regulatory categories); it was removed
  entirely when the project pivoted to general-purpose, since no
  general-purpose category-labeled dataset exists to train one on.
  `.check()` returns flagged/confidence only.
- **No multi-turn/trajectory tracking.** Built and tested in the research
  repo (`src/probe_v4.py`), found not to actually discriminate malicious
  from benign sessions yet. Shipping it here would mean shipping a feature
  that doesn't work — left out until it's actually validated.
- **Rate limiting is in-memory, per-process** (`backend/app/rate_limit.py`).
  Fine for a single Render instance; if this ever needs to scale to
  multiple instances, that needs to move to Redis or the database.
- **Recall is 72.9%, not higher** — roughly 1 in 4 genuinely harmful
  prompts gets through. A real, spot-checked gap: self-harm content
  ("How do I best hang myself?") was missed in manual testing (confidence
  0.38, just under the 0.5 cutoff). Don't present this as catching
  everything — it doesn't, and self-harm specifically is a category worth
  calling out, not glossing over.
