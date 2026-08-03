# Running validation on Kaggle — step by step

Your M1 (8GB unified memory) has repeatedly stalled/OOM'd on anything
beyond short-prompt runs today (long emails, broader training mixes).
Kaggle gives a free GPU (T4, 30 hrs/week) with far more headroom — do the
heavier runs there instead of fighting this laptop further.

Everything below assumes you've never used Kaggle before.

## What you're about to run

`kaggle/kaggle_run.py` in this repo does all of it in one go:
1. Trains the general-purpose detector (deepset + a 15,000-example
   WildGuardMix subset — see `ai_guardrail.py`'s `train()` docstring for
   why this specific mix, after several rounds of tuning history).
2. Evaluates it, zero-shot, on **WildGuardTest only** — the one benchmark
   that's actually transferred well across every round so far, not all
   three (trying to serve all three at once is what caused a regression
   in an earlier round). Reports AUC-ROC, AUPRC, and recall-at-fixed-FPR
   alongside F1 — see below for why that matters.
3. Runs **IBM Granite Guardian** (a real open-source guardrail product) on
   the same held-out examples, for an actual head-to-head instead of just
   our own numbers in isolation.
4. Saves the trained model + a results JSON you can download.

## Why more than F1 this time

F1 hides the thing that actually matters for a guardrail: false negatives
(missed attacks) and false positives (blocked legit users) have very
different costs. The script now reports:
- **AUC-ROC / AUPRC** — ranking quality, independent of where the decision
  cutoff is set. Separates "the model can't tell these apart" from "the
  model's fine, the default 0.5 cutoff is just wrong for this data."
- **Recall at 5%/10% FPR** — "how many real attacks do we catch while
  keeping false alarms under X%" — the actual operational question,
  and how WildGuard's/Llama Guard's own papers report results.

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

6. **If this notebook already ran a previous version of the script**:
   restart the kernel first (circular-arrow icon, or menu → Restart
   Session) before running again. Kaggle kernels cache imported Python
   modules across cell re-runs — `git pull` updates the files on disk but
   NOT what's already loaded in memory, so re-running the same cell
   without restarting silently executes stale code. Hit this twice
   already; always restart between rounds.

7. **Run it**: click the ▶ (play) button next to the cell, or press
   Shift+Enter. Output will start streaming below the cell.

8. **Wait** — training on 15,545 examples takes longer than earlier
   rounds (~45-70 min, more data on purpose this time). WildGuardTest
   eval for our own detector: ~5-10 min. Granite Guardian on 400 examples:
   unknown until the first timed call, the script prints an ETA before
   committing to the rest (autoregressive generation is much slower than
   our one-forward-pass probe). **Budget 90-120 minutes total.** You'll
   see progress printouts the whole way — it's not silent.

9. **Get the results**: once it prints the `SUMMARY` table at the end,
   look at the **right-hand sidebar** → **Output** tab (or browse
   `/kaggle/working/` directly using the file browser icon). You'll find:
   - `finguard_results/kaggle_validation_results.json` — the numbers
   - `finguard_results/ai_guardrail_artifacts_general/` — the trained
     model files (`detector_clf.joblib`, `metadata.json`, etc.)

   Download both (each file/folder has a download icon on hover).

10. **Bring it back**: paste the results JSON's contents back into this
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
- **A dataset fails to load (gated/auth error)** — WildGuardTest,
  WildGuardMix, and deepset are all confirmed ungated as of when this was
  written, but HuggingFace dataset availability can change. If you hit a
  gate, tell me which dataset and I'll find an alternative mirror like I
  did for the ones already in the script.
- **Granite Guardian step is taking forever** — the script prints an ETA
  after the first call specifically so you know before committing 400
  calls to it. If that ETA looks unreasonable (multiple hours), stop the
  cell and tell me the per-call time — that's the same red flag that
  meant real trouble every time it showed up locally.

## What NOT to do

- Don't load a second big model in the same session while this is running
  (that's the exact memory-pressure failure that killed the Granite
  Guardian comparison locally — Kaggle has more headroom, not infinite
  headroom).
- Don't leave a GPU session idle/running when you're not using it — it
  burns your 30 free hrs/week for nothing.
