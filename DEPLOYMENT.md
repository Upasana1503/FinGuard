# Deploying FinGuard

Two independent deployments: the FastAPI backend (does the actual guardrail
inference) and the Streamlit frontend (talks to the backend over HTTP). Deploy
the backend first — the frontend needs its URL.

## 1. Database — Supabase (free Postgres)

1. Create an account at supabase.com, create a new project (pick a DB
   password when prompted — you'll need it in step 3).
2. Wait for provisioning (~1-2 min).
3. Project Settings → Database → **Connection string** → URI tab. Supabase
   offers a few connection modes — **use "Session pooler" or "Direct
   connection", NOT "Transaction pooler"**. Transaction-mode pooling
   (pgbouncer) breaks plain SQLAlchemy setups like this one's unless you
   add pgbouncer-specific config (`pool_pre_ping`, disabling prepared-
   statement caching, etc.) that this backend doesn't have — Session
   pooler/Direct just work with `create_engine(DATABASE_URL)` as-is.
4. Copy the URI, replace the `[YOUR-PASSWORD]` placeholder in it with the
   real DB password from step 1. Keep the finished string — you'll paste
   it into the backend's environment variables next.

## 2. Backend — Hugging Face Spaces

Chosen over Render specifically for this project: free CPU Spaces give
**16GB RAM**, vs Render's free tier (512MB) being unusable for this model
at all — Render only works here on a paid tier. Spaces are free and
actually big enough.

One structural thing to know before starting: Spaces expect the
`Dockerfile` (and the `README.md` with the frontmatter block below) at the
**root** of the Space's own git repo. In the FinGuard repo they live at
`backend/Dockerfile` and `backend/README.md` instead, nested inside the
monorepo — so the Space gets its own small repo, and you push just the
`backend/` folder's contents into it.

1. On huggingface.co: **New → Space**. Pick a name (e.g. `finguard-api`),
   SDK: **Docker**, visibility: your choice. This creates an empty git
   repo at `https://huggingface.co/spaces/YOUR_USERNAME/finguard-api`.
2. Clone the FinGuard repo locally if you haven't already, then push just
   the `backend/` subfolder's contents into the new Space repo:
   ```bash
   cd /path/to/FinGuard
   git subtree push --prefix=backend https://huggingface.co/spaces/YOUR_USERNAME/finguard-api main
   ```
   (First push may ask for HF credentials — use a Space-scoped access
   token from huggingface.co/settings/tokens as the password.)
3. In the Space's **Settings → Repository secrets**, add `DATABASE_URL` =
   the Supabase connection string from step 1.
4. The Space builds automatically after the push. First boot is slow
   (~1-2 min: downloading model weights + the 3-call warmup baked into
   startup) — that's expected, only happens once per deploy.
5. Once live, your backend URL is
   `https://YOUR_USERNAME-finguard-api.hf.space` — the frontend needs it.

Sanity check once deployed:
```bash
curl https://YOUR_USERNAME-finguard-api.hf.space/health
# -> {"status":"ok"}
```

To redeploy after changing backend code later: repeat the `git subtree
push` command from step 2 (or `git subtree push --prefix=backend ... main
--force` if history diverges).

## 3. Frontend — Streamlit Community Cloud

1. On share.streamlit.io: New app → connect the repo.
2. Main file path: `frontend/streamlit_app.py`
3. In the app's "Secrets" settings, add:
   ```
   FINGUARD_BACKEND_URL = "https://YOUR_USERNAME-finguard-api.hf.space"
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
  Fine for a single Space instance; if this ever needs to scale to
  multiple instances, that needs to move to Redis or the database.
- **Recall is 72.9%, not higher** — roughly 1 in 4 genuinely harmful
  prompts gets through. A real, spot-checked gap: self-harm content
  ("How do I best hang myself?") was missed in manual testing (confidence
  0.38, just under the 0.5 cutoff). Don't present this as catching
  everything — it doesn't, and self-harm specifically is a category worth
  calling out, not glossing over.
