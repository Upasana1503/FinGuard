"""
General-purpose vs finance-scoped detector -- head-to-head, same model,
same architecture, different training data, same held-out eval (XSTest).

Context: after seeing XSTest recall near-zero and a bad false-positive rate
on a long-form fraud-email dataset, the question came up -- would training
on GENERAL harm data (not finance-scoped) fix this? This script tests that
directly instead of arguing about it.

Two detectors trained from scratch, both LogisticRegressionCV over the same
layer-8 Qwen2.5 activations, same code path (train_and_eval_probe):
  - "general": deepset (546, prompt injection) + advbench_mix (400, broad
    harmful-behavior categories) -- NO finance data at all.
  - "finance": deepset (546) + FinSec-MinPairs (64, finance-only contrastive
    pairs) -- what's actually shipped in the product today.

Both evaluated zero-shot (no retraining) on the FULL XSTest set (450) --
the real question being answered: does broadening the TRAINING data fix the
over-refusal-adjacent problems, or is FinGuard's current shape (short
prompts, finance-specific) actually orthogonal to that?

Deliberately does NOT touch ai_guardrail.py's shipped artifacts -- this is
a standalone experiment, not a retrain-in-place. Nothing here overwrites
what's deployed until/unless the results say it should.

Usage:
    python general_vs_finance_experiment.py --layer 8
"""

import argparse
import json
import os

import numpy as np


def evaluate_zero_shot(clf, X, y):
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    preds = clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / len(y)
    return {
        "precision": round(float(precision), 4), "recall": round(float(recall), 4),
        "f1": round(float(f1), 4), "false_positive_rate": round(float(fpr), 4),
        "accuracy": round(float(accuracy), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from benchmark_v2 import load_deepset, load_advbench_mix, load_xstest
    from minimal_pairs import load_minimal_pairs
    from probe_v3 import load_model, build_activation_dataset, train_and_eval_probe

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)

    print("=" * 70)
    print("Loading training data for both detectors ...")
    deepset_ex = load_deepset(None)
    advbench_ex = load_advbench_mix()
    finance_ex = load_minimal_pairs(None)

    general_examples = deepset_ex + advbench_ex
    finance_examples = deepset_ex + finance_ex
    print(f"general: deepset({len(deepset_ex)}) + advbench_mix({len(advbench_ex)}) = {len(general_examples)}")
    print(f"finance: deepset({len(deepset_ex)}) + FinSec-MinPairs({len(finance_ex)}) = {len(finance_examples)}")

    results = {}
    trained_clfs = {}

    for name, examples in [("general", general_examples), ("finance", finance_examples)]:
        print(f"\n{'=' * 70}\nTraining '{name}' detector\n{'=' * 70}")
        train_ex, test_ex = train_test_split(
            examples, test_size=args.test_size, random_state=42,
            stratify=[label for _, label in examples],
        )
        X_train, y_train = build_activation_dataset(train_ex, model, tokenizer, device, args.layer, batch_size=args.batch_size)
        X_test, y_test = build_activation_dataset(test_ex, model, tokenizer, device, args.layer, batch_size=args.batch_size)
        metrics, clf = train_and_eval_probe(X_train, y_train, X_test, y_test)
        print(f"'{name}' held-out metrics (own test split): {json.dumps(metrics, indent=2)}")
        trained_clfs[name] = clf
        results[f"{name}_own_heldout"] = metrics

    print(f"\n{'=' * 70}\nZero-shot evaluation: both detectors on FULL XSTest (450, never trained on)\n{'=' * 70}")
    xstest_ex = load_xstest(None)
    X_xstest, y_xstest = build_activation_dataset(xstest_ex, model, tokenizer, device, args.layer, batch_size=args.batch_size)

    for name, clf in trained_clfs.items():
        metrics = evaluate_zero_shot(clf, X_xstest, y_xstest)
        print(f"\n'{name}' detector on XSTest: {json.dumps(metrics, indent=2)}")
        results[f"{name}_on_xstest"] = metrics

    print(f"\n{'=' * 70}\nSide-by-side on XSTest\n{'=' * 70}")
    print(f"{'metric':<12}{'general':<12}{'finance':<12}")
    for metric in ["precision", "recall", "f1", "false_positive_rate", "accuracy"]:
        g = results["general_on_xstest"][metric]
        f = results["finance_on_xstest"][metric]
        print(f"{metric:<12}{g:<12}{f:<12}")

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, "general_vs_finance_xstest_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {os.path.abspath(out_path)}")
