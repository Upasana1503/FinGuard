"""
FinGuard -- Kaggle validation run.

Paste this ENTIRE file into a single Kaggle notebook cell and run it.
See ../KAGGLE_VALIDATION.md for the click-by-click setup steps (creating
the notebook, turning on GPU + internet, etc.) if you haven't done that yet.

What this does, in order:
  1. Clones the FinGuard repo (or pulls latest if already cloned).
  2. Installs anything missing beyond Kaggle's preinstalled packages.
  3. Trains the GENERAL-scope detector (deepset + advbench_mix -- no
     finance-only bias) on the GPU, batched for speed.
  4. Evaluates it zero-shot (no retraining) on the three standard
     benchmarks: XSTest, OR-Bench, WildGuardTest.
  5. Saves everything (trained artifacts + a results JSON + a printed
     summary table) to /kaggle/working/ so you can download it and bring
     the numbers back.

Runtime estimate on a T4: training ~5-10 min, each benchmark eval a
few minutes -- expect ~25-40 min total, much faster than anything we
managed locally today.
"""

import json
import os
import subprocess
import sys

REPO_URL = "https://github.com/Upasana1503/FinGuard.git"
REPO_DIR = "/kaggle/working/FinGuard"
OUT_DIR = "/kaggle/working/finguard_results"

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
# 2. Install anything missing (Kaggle ships torch/transformers/datasets/
#    scikit-learn/joblib already -- this is just a safety net)
# ---------------------------------------------------------------------------
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "torch", "datasets", "scikit-learn", "joblib"], check=True)

# ---------------------------------------------------------------------------
# 3. Train the general-scope detector
# ---------------------------------------------------------------------------
from ai_guardrail import ActivationGuardrail  # noqa: E402

print("=" * 70)
print("Training general-scope detector on GPU ...")
print("=" * 70)

guardrail = ActivationGuardrail()
train_metadata = guardrail.train(batch_size=32)
print("\nTraining metadata:")
print(json.dumps(train_metadata, indent=2))

# ---------------------------------------------------------------------------
# 4. Zero-shot evaluation on the three standard benchmarks
# ---------------------------------------------------------------------------
from benchmark_v2 import load_xstest, load_or_bench, load_wildguardtest  # noqa: E402
from probe_v3 import extract_activations_batch  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score  # noqa: E402
import numpy as np  # noqa: E402


def evaluate(name, loader_fn, **loader_kwargs):
    print(f"\n{'=' * 70}\nEvaluating on {name}\n{'=' * 70}")
    examples = loader_fn(**loader_kwargs)
    texts = [t for t, _ in examples]
    y = np.array([label for _, label in examples])
    print(f"Loaded {len(examples)} examples ({int(y.sum())} positive / {len(y) - int(y.sum())} negative)")

    X = extract_activations_batch(texts, guardrail.model, guardrail.tokenizer,
                                   guardrail.device, guardrail.layer, batch_size=32)
    preds = guardrail.clf.predict(X)

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / len(y)

    metrics = {
        "n_examples": len(examples), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": round(float(precision), 4), "recall": round(float(recall), 4),
        "f1": round(float(f1), 4), "false_positive_rate": round(float(fpr), 4),
        "accuracy": round(float(accuracy), 4),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


all_results = {"train_metadata": train_metadata}
all_results["xstest"] = evaluate("XSTest (450 ex.)", load_xstest, max_samples=None)
all_results["or_bench"] = evaluate("OR-Bench (or-bench-hard-1k + or-bench-toxic, ~1974 ex.)", load_or_bench, max_samples=None)
all_results["wildguardtest"] = evaluate("WildGuardTest (1699 ex.)", load_wildguardtest, max_samples=None)

# ---------------------------------------------------------------------------
# 5. Save everything + print summary
# ---------------------------------------------------------------------------
results_path = os.path.join(OUT_DIR, "kaggle_validation_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)

# Copy the trained artifacts out too, so they can be downloaded and swapped
# into backend/app/guardrail_core/artifacts/ locally afterward.
import shutil  # noqa: E402
artifacts_src = os.path.join(REPO_DIR, "logs", "ai_guardrail_artifacts")
artifacts_dst = os.path.join(OUT_DIR, "ai_guardrail_artifacts_general")
if os.path.exists(artifacts_src):
    shutil.copytree(artifacts_src, artifacts_dst, dirs_exist_ok=True)

print(f"\n{'=' * 70}\nSUMMARY -- general-scope detector, zero-shot on standard benchmarks\n{'=' * 70}")
print(f"{'benchmark':<16}{'precision':<12}{'recall':<10}{'f1':<10}{'FPR':<10}{'accuracy'}")
for name in ["xstest", "or_bench", "wildguardtest"]:
    m = all_results[name]
    print(f"{name:<16}{m['precision']:<12}{m['recall']:<10}{m['f1']:<10}{m['false_positive_rate']:<10}{m['accuracy']}")

print(f"\nResults saved to {results_path}")
print(f"Artifacts saved to {artifacts_dst}")
print("\nDownload both from the notebook's Output/Data pane (top right) and bring them back.")
