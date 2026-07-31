"""
Diagnostic: control test for shortcut learning / leakage.

If your probe gets suspiciously perfect scores, the FIRST thing to check
is whether it would ALSO get a suspiciously good score on RANDOM labels.
If yes -> something is leaking (duplicate/near-duplicate examples across
train/test, a trivial confound like text length, etc.) and your real
result is NOT trustworthy yet.

This directly implements the sanity check "When Benchmarks Lie"
(arXiv 2602.14161) warns you to run before believing a benchmark result.

Usage:
    python diagnose_leakage.py --dataset deepset --layer 8 --max_samples 150
"""

import argparse
import random

import numpy as np

from probe_v3 import load_model, build_activation_dataset, train_and_eval_probe


def check_near_duplicates(examples, threshold=0.9):
    """
    Cheap duplicate/template check: how many example PAIRS share a large
    fraction of their words? High overlap between many pairs suggests
    template-based generation, which risks train/test leakage under a
    random split.
    """
    texts = [t for t, _ in examples]
    word_sets = [set(t.lower().split()) for t in texts]

    near_dupe_pairs = 0
    total_checked = 0
    # Sample-limited pairwise check (full O(n^2) is fine for n<300, cap otherwise)
    n = min(len(texts), 200)
    for i in range(n):
        for j in range(i + 1, n):
            total_checked += 1
            a, b = word_sets[i], word_sets[j]
            if not a or not b:
                continue
            overlap = len(a & b) / max(len(a | b), 1)
            if overlap >= threshold:
                near_dupe_pairs += 1

    rate = near_dupe_pairs / total_checked if total_checked else 0
    print(f"\n[Duplicate check] {near_dupe_pairs}/{total_checked} pairs "
          f"(first {n} examples) share >={threshold*100:.0f}% word overlap "
          f"({rate*100:.2f}% of checked pairs)")
    if rate > 0.02:
        print("  -> WARNING: notable template/near-duplicate rate detected. "
              "This can cause train/test leakage under a random split.")
    return rate


def run_control_test(examples, model, tokenizer, device, layer, test_size=0.3):
    from sklearn.model_selection import train_test_split

    real_labels = [l for _, l in examples]
    shuffled_labels = real_labels.copy()
    random.seed(1)
    random.shuffle(shuffled_labels)

    shuffled_examples = [(examples[i][0], shuffled_labels[i]) for i in range(len(examples))]

    train_ex, test_ex = train_test_split(
        shuffled_examples, test_size=test_size, random_state=42,
        stratify=shuffled_labels,
    )

    print("\n[Control test] Training probe on RANDOMIZED labels "
          "(true signal removed -- should score near chance, ~0.5 accuracy)")
    X_train, y_train = build_activation_dataset(train_ex, model, tokenizer, device, layer)
    X_test, y_test = build_activation_dataset(test_ex, model, tokenizer, device, layer)
    metrics, _ = train_and_eval_probe(X_train, y_train, X_test, y_test)

    print(f"\nControl (random-label) result: {metrics}")
    if metrics["f1"] > 0.65:
        print("  -> RED FLAG: probe still scores well on MEANINGLESS labels. "
              "Your real result is very likely inflated by leakage, not genuine signal.")
    else:
        print("  -> Good: probe performs near chance on random labels, as expected. "
              "This is evidence (not proof) your real result reflects real signal.")
    return metrics


if __name__ == "__main__":
    from benchmark_v2 import DATASET_LOADERS

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deepset", "advbench_mix", "minimal_pairs"], default="deepset")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} ...")
    examples = DATASET_LOADERS[args.dataset](args.max_samples)
    print(f"Loaded {len(examples)} examples "
          f"({sum(l for _, l in examples)} positive / {len(examples)} total).")

    check_near_duplicates(examples)

    model, tokenizer, device = load_model(args.model)
    run_control_test(examples, model, tokenizer, device, args.layer)