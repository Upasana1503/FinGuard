"""
FinGuard -- Kaggle validation run (round 5).

Paste this ENTIRE file into a single Kaggle notebook cell and run it.
See ../KAGGLE_VALIDATION.md for the click-by-click setup steps if you
haven't done that yet.

Scope narrowed on purpose (see PROJECT_SUMMARY.md for the full history):
  - No more finance/cybersec anywhere -- general-purpose only.
  - ONE benchmark (WildGuardTest), not three. It's the only one that's
    consistently transferred well across every round so far, and trying
    to serve XSTest/OR-Bench at the same time is what caused round 4's
    regression (tuning the training mix for one benchmark hurt another).
  - Reports threshold-independent metrics (AUC-ROC, AUPRC) and a
    threshold-optimal F1, not just default-cutoff F1 -- every prior round
    only reported F1 at the classifier's built-in 0.5 cutoff, which can't
    tell you whether "bad F1" means bad separability or just a bad cutoff.
  - Runs IBM Granite Guardian (a real open-source guardrail product) on
    the same held-out examples for an actual head-to-head, not just our
    own numbers in isolation. This stalled/OOM'd locally every time
    (memory contention loading two models on an 8GB laptop) -- Kaggle's
    dedicated GPU shouldn't have that problem, but the two models are
    still loaded sequentially with memory freed between them, same
    lesson learned from the local failures either way.

What this does, in order:
  1. Clone/pull the repo.
  2. Install anything missing.
  3. Train the detector: deepset + a 15,000-example WildGuardMix subset
     (the only thing that's consistently helped -- see train() in
     ai_guardrail.py for the full tuning history).
  4. Evaluate zero-shot on WildGuardTest: AUC-ROC, AUPRC, default-
     threshold metrics, best-F1-threshold metrics, recall at 5%/10% FPR.
  5. Free the detector's model from memory, then run Granite Guardian on
     a subset of the same held-out examples (GRANITE_EVAL_N below --
     autoregressive generate() calls are much slower than our one-forward-
     pass probe, so this is capped for time; raise it if you have budget).
  6. Save everything to /kaggle/working/ and print a side-by-side summary.

Runtime estimate on a T4: training on 15,545 examples will take longer
than any prior round (~45-70 min) -- it's more data than we've used
before, on purpose. WildGuardTest eval (our detector): ~5-10 min. Granite
Guardian on GRANITE_EVAL_N examples: unknown until first timed call, the
script times the first one and prints an ETA before committing to the
rest. Budget 90-120 min total.
"""

import json
import os
import subprocess
import sys
import time

REPO_URL = "https://github.com/Upasana1503/FinGuard.git"
REPO_DIR = "/kaggle/working/FinGuard"
OUT_DIR = "/kaggle/working/finguard_results"
GRANITE_EVAL_N = 400  # subset of WildGuardTest to run Granite Guardian on (generate() is slow)

# ---------------------------------------------------------------------------
# 1. Clone / update the repo
# ---------------------------------------------------------------------------
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Install anything missing
# ---------------------------------------------------------------------------
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "torch", "datasets", "scikit-learn", "joblib"], check=True)

# ---------------------------------------------------------------------------
# 3. Train the general-purpose detector
# ---------------------------------------------------------------------------
from ai_guardrail import ActivationGuardrail  # noqa: E402

print("=" * 70)
print("Training general-purpose detector on GPU (deepset + 15k WildGuardMix) ...")
print("=" * 70)

guardrail = ActivationGuardrail()
train_metadata = guardrail.train(batch_size=32, wildguardmix_samples=15000)
print("\nTraining metadata:")
print(json.dumps(train_metadata, indent=2))

# ---------------------------------------------------------------------------
# 4. Zero-shot evaluation on WildGuardTest -- full threshold/AUC picture
# ---------------------------------------------------------------------------
from benchmark_v2 import load_wildguardtest  # noqa: E402
from probe_v3 import extract_activations_batch, evaluate_with_threshold_metrics  # noqa: E402
import numpy as np  # noqa: E402

print(f"\n{'=' * 70}\nEvaluating on WildGuardTest\n{'=' * 70}")
wgt_examples = load_wildguardtest(None)
wgt_texts = [t for t, _ in wgt_examples]
wgt_labels = np.array([label for _, label in wgt_examples])
print(f"Loaded {len(wgt_examples)} examples ({int(wgt_labels.sum())} harmful / "
      f"{len(wgt_labels) - int(wgt_labels.sum())} unharmful)")

X_wgt = extract_activations_batch(wgt_texts, guardrail.model, guardrail.tokenizer,
                                   guardrail.device, guardrail.layer, batch_size=32)
finguard_metrics = evaluate_with_threshold_metrics(guardrail.clf, X_wgt, wgt_labels)
print("\nFinGuard on WildGuardTest (full threshold/AUC picture):")
print(json.dumps(finguard_metrics, indent=2))

# ---------------------------------------------------------------------------
# 5. Free the detector's model, then run Granite Guardian on a subset
# ---------------------------------------------------------------------------
import gc  # noqa: E402

print("\nFreeing FinGuard's model from memory before loading Granite Guardian ...")
del guardrail.model, guardrail.clf
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass

from guardrail_granite import load_granite, run_granite_guardrail  # noqa: E402

print(f"\n{'=' * 70}\nRunning Granite Guardian 3.0-2B on {GRANITE_EVAL_N} WildGuardTest examples\n{'=' * 70}")
granite_model, granite_tokenizer, granite_device = load_granite()

# Same fixed-seed shuffle as everywhere else in this project, so the subset
# is representative rather than "whatever happened to be first in the file".
rng = np.random.RandomState(42)
subset_idx = rng.choice(len(wgt_examples), size=min(GRANITE_EVAL_N, len(wgt_examples)), replace=False)
granite_texts = [wgt_texts[i] for i in subset_idx]
granite_labels = wgt_labels[subset_idx]

start = time.time()
first_result = run_granite_guardrail(granite_texts[0], granite_model, granite_tokenizer, granite_device)
per_call = time.time() - start
eta_min = per_call * (len(granite_texts) - 1) / 60
print(f"First call took {per_call:.1f}s -> ETA for remaining {len(granite_texts) - 1} calls: ~{eta_min:.1f} min")

granite_preds = [0 if first_result.allowed else 1]
for i, text in enumerate(granite_texts[1:], start=2):
    result = run_granite_guardrail(text, granite_model, granite_tokenizer, granite_device)
    granite_preds.append(0 if result.allowed else 1)
    if i % 50 == 0:
        print(f"  {i}/{len(granite_texts)}")

granite_preds = np.array(granite_preds)
tp = int(((granite_preds == 1) & (granite_labels == 1)).sum())
fp = int(((granite_preds == 1) & (granite_labels == 0)).sum())
tn = int(((granite_preds == 0) & (granite_labels == 0)).sum())
fn = int(((granite_preds == 0) & (granite_labels == 1)).sum())
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
granite_metrics = {
    "n_examples": len(granite_texts), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
    "false_positive_rate": round(fpr, 4), "accuracy": round((tp + tn) / len(granite_texts), 4),
}
print("\nGranite Guardian on the same WildGuardTest subset:")
print(json.dumps(granite_metrics, indent=2))

# ---------------------------------------------------------------------------
# 6. Save everything + print summary
# ---------------------------------------------------------------------------
all_results = {
    "train_metadata": train_metadata,
    "finguard_wildguardtest_full": finguard_metrics,
    "granite_guardian_wildguardtest_subset": granite_metrics,
    "granite_eval_n": GRANITE_EVAL_N,
}

results_path = os.path.join(OUT_DIR, "kaggle_validation_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)

import shutil  # noqa: E402
artifacts_src = os.path.join(REPO_DIR, "logs", "ai_guardrail_artifacts")
artifacts_dst = os.path.join(OUT_DIR, "ai_guardrail_artifacts_general")
if os.path.exists(artifacts_src):
    shutil.copytree(artifacts_src, artifacts_dst, dirs_exist_ok=True)

print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
print("FinGuard on full WildGuardTest (1699 ex.):")
print(f"  AUC-ROC: {finguard_metrics['auc_roc']}  AUPRC: {finguard_metrics['auprc']}")
print(f"  Default-threshold F1: {finguard_metrics['default_threshold']['f1']}  "
      f"FPR: {finguard_metrics['default_threshold']['false_positive_rate']}")
print(f"  Best-F1-threshold F1: {finguard_metrics['best_f1_threshold']['f1']}  "
      f"(threshold={finguard_metrics['best_f1_threshold']['threshold']})")
print(f"  Recall at 5% FPR: {finguard_metrics['recall_at_5pct_fpr']}  "
      f"Recall at 10% FPR: {finguard_metrics['recall_at_10pct_fpr']}")
print(f"\nGranite Guardian on the SAME {GRANITE_EVAL_N}-example subset:")
print(f"  F1: {granite_metrics['f1']}  FPR: {granite_metrics['false_positive_rate']}  "
      f"Recall: {granite_metrics['recall']}")

print(f"\nResults saved to {results_path}")
print(f"Artifacts saved to {artifacts_dst}")
print("\nDownload both from the notebook's Output/Data pane (top right) and bring them back.")
