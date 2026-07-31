"""
Guardrail v2 — Benchmark harness.

Evaluates guardrail_v1 against real, published attack/benign datasets
and reports precision/recall/F1/false-positive-rate/latency — not just
pass/fail on a handful of hand-picked examples.

IMPORTANT (read this before trusting your numbers):
  "When Benchmarks Lie" (arXiv 2602.14161) shows that single-class
  datasets (100% malicious OR 100% benign) let a classifier "cheat" by
  learning superficial distributional cues rather than real attack
  signal. So this harness deliberately MIXES a malicious-only dataset
  (AdvBench) with a benign-only dataset (a slice of OpenOrca), rather
  than relying on one pre-mixed dataset alone. Compare your metrics
  across BOTH the pre-labeled mixed dataset and this manual mix —
  if they diverge a lot, that's a real finding worth writing up.

Usage:
    python benchmark_v2.py --dataset deepset       # quick, pre-labeled
    python benchmark_v2.py --dataset advbench_mix  # harder, manually mixed
    python benchmark_v2.py --dataset both
"""

import argparse
import json
import os
import time
from dataclasses import asdict

from guardrail_v1 import run_guardrail


# ---------------------------------------------------------------------------
# Dataset loaders — each returns list of (text, label) where label:
#   1 = malicious/attack, 0 = benign
# ---------------------------------------------------------------------------

def load_deepset(max_samples: int = None):
    """Pre-labeled mixed injection/benign dataset, 546 examples."""
    from datasets import load_dataset

    ds = load_dataset("deepset/prompt-injections")["train"]
    examples = [(row["text"], int(row["label"])) for row in ds]
    if max_samples:
        examples = examples[:max_samples]
    return examples


ADVBENCH_CSV_URL = (
    "https://github.com/llm-attacks/llm-attacks/raw/refs/heads/main/"
    "data/advbench/harmful_behaviors.csv"
)


def load_advbench_mix(n_malicious: int = 200, n_benign: int = 200):
    """
    Manual mix: AdvBench (100% malicious harmful-behavior prompts) +
    a slice of OpenOrca (100% benign instruction prompts).
    Harder / more realistic than a pre-mixed dataset — see module docstring.

    AdvBench pulled directly from the original public GitHub CSV
    (no HF gating/login required). OpenOrca still uses `datasets`.
    """
    import csv
    import io
    import ssl
    import urllib.request

    try:
        with urllib.request.urlopen(ADVBENCH_CSV_URL) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError:
        # Fallback for the common macOS "certificate verify failed" issue
        # (python.org installs don't always link system CA certs).
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(ADVBENCH_CSV_URL, context=ctx) as resp:
            raw = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(raw))
    malicious = [(row["goal"], 1) for row in reader][:n_malicious]

    from datasets import load_dataset

    orca = load_dataset("Open-Orca/OpenOrca", split=f"train[:{n_benign}]")
    benign = [(row["question"], 0) for row in orca][:n_benign]

    combined = malicious + benign
    return combined


def load_minimal_pairs_flat(max_samples: int = None):
    """FinSec-MinPairs (see minimal_pairs.py) as flat (text, label) pairs.
    Contrastive benign/malicious pairs sharing surface lexicon -- the fair
    test of intent-disambiguation, unlike deepset (see PROJECT_SUMMARY Sec.9)."""
    from minimal_pairs import load_minimal_pairs

    return load_minimal_pairs(max_samples)


DATASET_LOADERS = {
    "deepset": load_deepset,
    "advbench_mix": load_advbench_mix,
    "minimal_pairs": load_minimal_pairs_flat,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(examples, use_classifier: bool = False):
    """
    Run guardrail on every example, compare predicted (blocked=1/allowed=0)
    against ground-truth label, compute standard metrics.
    """
    tp = fp = tn = fn = 0
    latencies = []
    errors = []  # store (text, true_label, predicted, reasons) for false pos/neg

    for text, true_label in examples:
        try:
            result = run_guardrail(text, use_classifier=use_classifier)
        except Exception as e:
            errors.append({"text": text[:100], "error": str(e)})
            continue

        predicted = 0 if result.allowed else 1  # blocked = flagged as malicious
        latencies.append(result.latency_ms)

        if predicted == 1 and true_label == 1:
            tp += 1
        elif predicted == 1 and true_label == 0:
            fp += 1
            errors.append({
                "type": "false_positive", "text": text[:150],
                "reasons": result.reasons,
            })
        elif predicted == 0 and true_label == 0:
            tn += 1
        elif predicted == 0 and true_label == 1:
            fn += 1
            errors.append({
                "type": "false_negative", "text": text[:150],
                "reasons": result.reasons,
            })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # % of benign wrongly blocked
    accuracy = (tp + tn) / len(examples) if examples else 0.0

    return {
        "n_examples": len(examples),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 4) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 4) if latencies else 0,
        "errors_sample": errors[:20],  # keep report readable; full list saved separately
        "n_pipeline_errors": len(errors) - len([e for e in errors if "type" in e]),
    }


def print_report(name: str, metrics: dict):
    print(f"\n{'='*60}")
    print(f"  Benchmark: {name}")
    print(f"{'='*60}")
    print(f"  N examples:        {metrics['n_examples']}")
    print(f"  Precision:         {metrics['precision']}")
    print(f"  Recall:            {metrics['recall']}")
    print(f"  F1:                {metrics['f1']}")
    print(f"  False positive rate: {metrics['false_positive_rate']}  <-- watch this closely")
    print(f"  Accuracy:          {metrics['accuracy']}")
    print(f"  Avg latency:       {metrics['avg_latency_ms']} ms")
    print(f"  P95 latency:       {metrics['p95_latency_ms']} ms")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")
    if metrics["errors_sample"]:
        print(f"\n  Sample errors (first {len(metrics['errors_sample'])}):")
        for e in metrics["errors_sample"][:5]:
            print(f"    [{e.get('type','?')}] {e.get('text','')[:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deepset", "advbench_mix", "minimal_pairs", "both"],
                         default="deepset")
    parser.add_argument("--use_classifier", action="store_true",
                         help="Also run the toxicity classifier layer (slower, needs transformers+torch)")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    targets = ["deepset", "advbench_mix", "minimal_pairs"] if args.dataset == "both" else [args.dataset]
    all_results = {}

    NEEDS_MAX_SAMPLES = {"deepset", "minimal_pairs"}

    for name in targets:
        print(f"\nLoading dataset: {name} ...")
        loader = DATASET_LOADERS[name]
        examples = loader(args.max_samples) if name in NEEDS_MAX_SAMPLES else loader()
        print(f"Loaded {len(examples)} examples. Running guardrail...")

        start = time.time()
        metrics = evaluate(examples, use_classifier=args.use_classifier)
        metrics["wall_clock_seconds"] = round(time.time() - start, 2)
        all_results[name] = metrics
        print_report(name, metrics)

    out_path = os.path.join(logs_dir, "benchmark_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results (incl. all errors) saved to {os.path.abspath(out_path)}")