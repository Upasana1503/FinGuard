# Running validation on Kaggle — step by step

Your M1 (8GB unified memory) has repeatedly stalled/OOM'd on anything
beyond short-prompt runs today (long emails, broader training mixes).
Kaggle gives a free GPU (T4, 30 hrs/week) with far more headroom — do the
heavier runs there instead of fighting this laptop further.

Everything below assumes you've never used Kaggle before.

## What you're about to run

`kaggle/kaggle_run.py` in this repo does all of it in one go:
1. Trains the **general-scope** detector (deepset + advbench_mix — broad
   harm categories, not finance-biased) on the GPU.
2. Evaluates it, zero-shot, on the three standard benchmarks: **XSTest**,
   **OR-Bench**, **WildGuardTest**.
3. Saves the trained model + a results JSON you can download.

All three benchmark loaders are already written and verified (real label
conventions checked against each dataset's actual published stats, not
guessed) — you don't need to write or debug any code, just run the script.

## Step by step

1. **Go to kaggle.com and sign in** (or make a free account if you don't
   have one — top right, "Register").

2. **Create a notebook**: click **Create** (top left) → **New Notebook**.
   You'll land in an empty notebook with one empty code cell.

3. **Turn on GPU**: on the right-hand sidebar, find **Session options** (or
   click the "..." menu if you don't see it) → **Accelerator** → pick
   **GPU T4 x2**. It'll say the session is restarting — that's normal.

4. **Turn on internet**: same right-hand sidebar, find **Internet** and
   toggle it **On**. Off by default; the script needs it to clone the repo
   and download the model + datasets.

5. **Copy the whole script in**: open `kaggle/kaggle_run.py` from this
   repo, select all, copy it, and paste it into that one empty cell in
   your Kaggle notebook — replacing whatever's there.

6. **Run it**: click the ▶ (play) button next to the cell, or press
   Shift+Enter. Output will start streaming below the cell.

7. **Wait** — training takes ~5-10 min, each of the three benchmark
   evaluations a few more minutes. **Expect roughly 25-40 minutes total.**
   You'll see progress printouts the whole way (dataset loading, training
   metrics, then each benchmark's results as they finish) — it's not
   silent, so you'll know it's alive.

8. **Get the results**: once it prints the `SUMMARY` table at the end,
   look at the **right-hand sidebar** → **Output** tab (or browse
   `/kaggle/working/` directly using the file browser icon). You'll find:
   - `finguard_results/kaggle_validation_results.json` — the numbers
   - `finguard_results/ai_guardrail_artifacts_general/` — the trained
     model files (`detector_clf.joblib`, `metadata.json`, etc.)

   Download both (each file/folder has a download icon on hover).

9. **Bring it back**: paste the results JSON's contents back into this
   conversation, or just tell me the run finished and where the files are
   on your machine (e.g. `~/Downloads/kaggle_validation_results.json`) —
   I'll fold the numbers into PROJECT_SUMMARY.md and swap the trained
   artifacts into the repo/backend if they look good.

## If something goes wrong

- **"No module named X"** — the script installs its own deps in step 2,
  but if Kaggle's base image is missing something unusual, just add
  `!pip install -q <package>` as a new cell above it and re-run.
- **Session disconnects / times out** — Kaggle free sessions have a max
  runtime (usually 9-12 hrs, way more than you need) and will warn before
  killing an idle session. If it happens mid-run, just re-run the cell —
  the script re-clones/pulls automatically and is idempotent.
- **A dataset fails to load (gated/auth error)** — WildGuardTest and
  OR-Bench are confirmed ungated as of when this was written, but
  HuggingFace dataset availability can change. If you hit a gate, tell me
  which dataset and I'll find an alternative mirror like I did for the
  ones already in the script.

## What NOT to do

- Don't load a second big model in the same session while this is running
  (that's the exact memory-pressure failure that killed the Granite
  Guardian comparison locally — Kaggle has more headroom, not infinite
  headroom).
- Don't leave a GPU session idle/running when you're not using it — it
  burns your 30 free hrs/week for nothing.
