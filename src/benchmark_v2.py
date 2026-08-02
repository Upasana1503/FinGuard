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


def load_financial_fraud_text(max_samples: int = None):
    """
    lakpriya/financial_fraud_textual_dataset -- 3999 real emails, roughly
    balanced: 2000 fraud/scam (classic advance-fee "bank manager needs help
    moving $5M" style letters), 1999 legitimate business correspondence.
    Independently collected, not authored by this project.

    IMPORTANT FRAMING CAVEAT: this tests something ADJACENT to the deployed
    task, not identical to it. FinSec-MinPairs tests "is a USER'S REQUEST TO
    AN AGENT malicious" (e.g. "transfer funds without telling anyone").
    This dataset instead tests "is this EMAIL BODY a financial scam" --
    different structure, same broad domain (financial-crime-flavored
    language vs ordinary financial business language). Report results with
    that distinction stated, not as if it were the same benchmark task.

    No dataset card/citation on the HF repo -- treat as "a real-world fraud
    email corpus found on HF," not a peer-reviewed named benchmark.

    label: 1 = fraud/scam email, 0 = legitimate.
    """
    from datasets import load_dataset

    ds = load_dataset("lakpriya/financial_fraud_textual_dataset")["train"]
    examples = [(row["Text"], int(row["Class"])) for row in ds]
    if max_samples:
        examples = examples[:max_samples]
    return examples


def load_or_bench(max_samples: int = None, n_toxic: int = None):
    """
    OR-Bench (Cui et al.) -- large over-refusal benchmark, second leg of
    the standard trio alongside XSTest.

    Verified real structure (checked directly, not assumed): the HF repo
    `bench-llm/or-bench` has three configs --
      - or-bench-80k: 80,359 safe-but-scary-sounding prompts (label 0)
      - or-bench-hard-1k: 1,319 of the HARDEST safe prompts, a curated
        subset of the 80k (label 0) -- used here by default since 80k is
        unnecessarily huge for a validation pass
      - or-bench-toxic: 655 genuinely unsafe prompts (label 1)

    label: 1 = genuinely unsafe (or-bench-toxic), 0 = safe-but-scary
    (or-bench-hard-1k). Pass max_samples to cap the SAFE side if you want
    a smaller/faster run; n_toxic caps the toxic side independently
    (defaults to using all 655).
    """
    from datasets import load_dataset

    safe_ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")["train"]
    toxic_ds = load_dataset("bench-llm/or-bench", "or-bench-toxic")["train"]

    safe_examples = [(row["prompt"], 0) for row in safe_ds]
    toxic_examples = [(row["prompt"], 1) for row in toxic_ds]

    if max_samples:
        safe_examples = safe_examples[:max_samples]
    if n_toxic:
        toxic_examples = toxic_examples[:n_toxic]

    return safe_examples + toxic_examples


def load_wildguardtest(max_samples: int = None):
    """
    WildGuardTest (Han et al., 2024) -- the standard comparison point most
    published guardrail papers report against.

    allenai's own repo (allenai/wildguardmix) is gated; using the verified
    ungated mirror `walledai/WildGuardTest` instead (checked directly:
    1725 rows, fields prompt/adversarial/label, label in
    {"harmful": 754, "unharmful": 945, None: 26} -- the 26 None-label rows
    are dropped since they can't be scored).

    label: 1 = harmful, 0 = unharmful.
    """
    from datasets import load_dataset

    ds = load_dataset("walledai/WildGuardTest")["train"]
    examples = [(row["prompt"], 1 if row["label"] == "harmful" else 0)
                for row in ds if row["label"] in ("harmful", "unharmful")]
    if max_samples:
        examples = examples[:max_samples]
    return examples


def load_xstest(max_samples: int = None):
    """
    XSTest (Röttger et al.) -- the canonical over-refusal / pseudo-harm
    benchmark. 450 prompts: 250 SAFE ones worded to sound risky (e.g. "How
    can I kill a Python process?"), 200 genuinely unsafe "contrast" prompts
    using the same trigger words for real harm.

    Label convention verified against actual category counts, not guessed:
    types prefixed "contrast_" are the 200 unsafe prompts (8 types x 25);
    everything else is the 250 safe-but-scary-sounding prompts (10 types x
    25) -- these counts match the published XSTest paper's 250/200 split
    exactly, confirming the mapping is right before trusting any numbers
    computed from it.

    label: 1 = should be blocked (contrast/unsafe), 0 = should be allowed
    (safe-sounding-risky). FPR on this dataset IS the over-refusal rate --
    the metric this benchmark exists to measure.
    """
    from datasets import load_dataset

    ds = load_dataset("natolambert/xstest-v2-copy")["prompts"]
    examples = [(row["prompt"], 1 if row["type"].startswith("contrast_") else 0) for row in ds]
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
    "xstest": load_xstest,
    "financial_fraud_text": load_financial_fraud_text,
    "or_bench": load_or_bench,
    "wildguardtest": load_wildguardtest,
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
    parser.add_argument("--dataset", choices=["deepset", "advbench_mix", "minimal_pairs", "xstest",
                                               "financial_fraud_text", "or_bench", "wildguardtest", "both"],
                         default="deepset")
    parser.add_argument("--use_classifier", action="store_true",
                         help="Also run the toxicity classifier layer (slower, needs transformers+torch)")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    targets = ["deepset", "advbench_mix", "minimal_pairs"] if args.dataset == "both" else [args.dataset]
    all_results = {}

    NEEDS_MAX_SAMPLES = {"deepset", "minimal_pairs", "xstest", "financial_fraud_text", "or_bench", "wildguardtest"}

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