"""
Head-to-head: ai_guardrail.py (this project's final product) vs a real
open-source guardrail product (IBM Granite Guardian 3.0-2B) vs the v1 regex
baseline, all evaluated on the SAME held-out examples.

This is the credibility check the project was missing (see PROJECT_SUMMARY.md
Sec.13): every result up to now compared this project's own versions against
each other (v1 vs v3, v3 vs v4). This script is the first comparison against
an actual deployed guardrail product, not another iteration of the same
codebase.

Fairness notes (read before trusting the numbers):
  - The held-out test split comes from ai_guardrail.train()'s own train/test
    split (saved to logs/ai_guardrail_artifacts/held_out_test_split.json) --
    these examples were NOT used to train ai_guardrail's detector, so it has
    no unfair advantage from memorization.
  - Granite Guardian is used zero-shot (it was never trained or fine-tuned
    on deepset or FinSec-MinPairs by this project) -- it's being scored on
    its out-of-the-box general "harm" risk definition, exactly as IBM ships
    it. That's the correct way to evaluate an off-the-shelf product, but it
    does mean it's not domain-adapted the way ai_guardrail is.
  - v1 is included as the cheap-baseline reference point, not as a serious
    competitor -- its numbers here are expected to be poor, consistent with
    every other result in this project.

Usage:
    python compare_products.py                  # uses/creates ai_guardrail artifacts, full held-out set
    python compare_products.py --max_eval 80     # cap eval set size for a faster run
"""

import argparse
import json
import os
import time

from dataclasses import asdict


def compute_metrics(examples, predictions, latencies):
    """examples: list of (text, true_label). predictions: parallel list of 0/1.
    Same metric definitions as benchmark_v2.evaluate() for direct comparability."""
    tp = fp = tn = fn = 0
    for (_, true_label), pred in zip(examples, predictions):
        if pred == 1 and true_label == 1:
            tp += 1
        elif pred == 1 and true_label == 0:
            fp += 1
        elif pred == 0 and true_label == 0:
            tn += 1
        elif pred == 0 and true_label == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / len(examples) if examples else 0.0

    return {
        "n_examples": len(examples), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4), "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 4) if latencies else 0,
    }


def run_v1(examples):
    from guardrail_v1 import run_guardrail

    predictions, latencies = [], []
    for text, _ in examples:
        result = run_guardrail(text, use_classifier=False)
        predictions.append(0 if result.allowed else 1)
        latencies.append(result.latency_ms)
    return predictions, latencies


def run_ai_guardrail(examples, guardrail):
    predictions, latencies = [], []
    for text, _ in examples:
        record = guardrail.check(text)
        predictions.append(1 if record["flagged"] else 0)
        latencies.append(record["latency_ms"])
    return predictions, latencies


def run_granite(examples, model, tokenizer, device):
    from guardrail_granite import run_granite_guardrail

    predictions, latencies = [], []
    for i, (text, _) in enumerate(examples):
        result = run_granite_guardrail(text, model, tokenizer, device)
        predictions.append(0 if result.allowed else 1)
        latencies.append(result.latency_ms)
        if (i + 1) % 20 == 0:
            print(f"  granite-guardian: {i + 1}/{len(examples)}")
    return predictions, latencies


if __name__ == "__main__":
    from ai_guardrail import ActivationGuardrail, ARTIFACTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_eval", type=int, default=None,
                         help="Cap the held-out eval set size for a faster run")
    parser.add_argument("--retrain", action="store_true",
                         help="Force ai_guardrail to retrain (regenerates the held-out split too)")
    args = parser.parse_args()

    guardrail = ActivationGuardrail()
    if args.retrain:
        guardrail.train()
    else:
        guardrail.ensure_ready()

    test_split_path = os.path.join(ARTIFACTS_DIR, "held_out_test_split.json")
    if not os.path.exists(test_split_path):
        print("No held-out split found -- training ai_guardrail to generate one.")
        guardrail.train()

    with open(test_split_path) as f:
        raw = json.load(f)
    examples = [(r["text"], r["label"]) for r in raw]
    if args.max_eval:
        examples = examples[:args.max_eval]
    n_pos = sum(l for _, l in examples)
    print(f"\nHeld-out comparison set: {len(examples)} examples ({n_pos} malicious / {len(examples) - n_pos} benign)")

    print("\n=== Running v1 (regex baseline) ===")
    v1_preds, v1_lat = run_v1(examples)
    v1_metrics = compute_metrics(examples, v1_preds, v1_lat)

    print("\n=== Running ai_guardrail.py (this project's final product) ===")
    ai_preds, ai_lat = run_ai_guardrail(examples, guardrail)
    ai_metrics = compute_metrics(examples, ai_preds, ai_lat)

    # Free Qwen (ai_guardrail's backbone) before loading Granite Guardian --
    # this machine has 8GB unified memory; holding both a 1.5B and a 2B model
    # in RAM at once was pushing the system into heavy swap, which is why
    # earlier runs took minutes per Granite call instead of the ~30-60s a
    # single resident model needs. One model in memory at a time.
    import gc

    print("\nFreeing ai_guardrail's model from memory before loading Granite Guardian ...")
    del guardrail.model, guardrail.clf, guardrail.directions
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    print("\n=== Loading + running Granite Guardian 3.0-2B (real open-source product) ===")
    from guardrail_granite import load_granite

    granite_model, granite_tokenizer, granite_device = load_granite()
    granite_preds, granite_lat = run_granite(examples, granite_model, granite_tokenizer, granite_device)
    granite_metrics = compute_metrics(examples, granite_preds, granite_lat)

    results = {
        "held_out_n": len(examples),
        "v1_regex_baseline": v1_metrics,
        "ai_guardrail_final_product": ai_metrics,
        "granite_guardian_3_2b": granite_metrics,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(f"\n{'=' * 72}")
    print(f"  Head-to-head on {len(examples)} held-out examples")
    print(f"{'=' * 72}")
    print(f"{'system':<28}{'precision':<12}{'recall':<10}{'F1':<10}{'FPR':<10}{'latency(ms)'}")
    for name, m in [("v1 (regex)", v1_metrics), ("ai_guardrail (ours)", ai_metrics),
                     ("granite-guardian-3.0-2b", granite_metrics)]:
        print(f"{name:<28}{m['precision']:<12}{m['recall']:<10}{m['f1']:<10}"
              f"{m['false_positive_rate']:<10}{m['avg_latency_ms']}")

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, "compare_products_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {os.path.abspath(out_path)}")
