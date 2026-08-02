"""
Guardrail v3 — Activation probe.

Instead of reading the PROMPT TEXT (like v1), this reads the model's
INTERNAL ACTIVATIONS while it processes the prompt, and trains a simple
classifier (logistic regression) on those internal vectors.

Core idea:
  text  ->  [transformer layers, each producing a hidden vector]  ->  output text
                              ^
                    we tap in here (mid-layer), BEFORE the model
                    finishes "deciding" what words to say

Why this might beat v1: v1 (regex) only catches phrasings it was
explicitly written to catch. A paraphrase sails through untouched.
The activation probe reads something closer to "intent," which may
survive paraphrasing even when the surface words change completely.

Model: defaults to Qwen2.5-1.5B-Instruct — small enough to run on an
M1 Mac (CPU or `mps` backend) or free-tier Colab/Kaggle GPU. Swap
--model to a bigger one (e.g. meta-llama/Llama-3.2-8B-Instruct) once
you move to Kaggle for final numbers.

Usage:
    python probe_v3.py --dataset deepset --layer 8
    python probe_v3.py --dataset deepset --layer_sweep   # try multiple layers, report best
"""

import argparse
import json
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# 1. Model loading + activation extraction
# ---------------------------------------------------------------------------

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    print(f"Loading {model_name} on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right-padding is required for a causal LM: with attention_mask passed
    # through, real tokens keep the exact same positions/attention pattern
    # as the unpadded case. Left-padding would shift real tokens to later
    # positions and risk subtly different hidden states -- don't chance it.
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _base_transformer(model):
    """
    Returns the bare transformer stack (no LM head) if the wrapper exposes
    one, e.g. Qwen2ForCausalLM.model -> Qwen2Model.

    Why this matters: calling the full CausalLM wrapper computes lm_head
    logits over the ENTIRE vocabulary (~150k tokens for Qwen2.5) for every
    token position in the batch, even though we only ever read
    hidden_states and never look at the logits. For batch=32 x seq_len=256
    x vocab=150000 in float16, that's a ~2.3GB tensor we throw away --
    which is exactly what blew out MPS memory on a long-text batch (see
    the financial-fraud-email eval: 3999 multi-paragraph emails). Skipping
    the LM head entirely removes that allocation and is strictly faster,
    with identical hidden_states math.
    """
    return model.model if hasattr(model, "model") else model


def extract_activation(text: str, model, tokenizer, device: str, layer: int) -> np.ndarray:
    """
    Run one forward pass, pull the hidden state at `layer`, mean-pool
    across tokens to get a single fixed-size vector per prompt.

    layer=0 is the embedding layer; layer=N is the last transformer layer.
    Mid layers (roughly 1/3 to 2/3 through the stack) tend to encode
    semantic/intent information best — that's the empirical claim you're
    testing with --layer_sweep.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = _base_transformer(model)(**inputs, output_hidden_states=True)

    # Cast to float32 BEFORE reducing, not after. Residual-stream activations
    # have known outlier dimensions that can run large in magnitude; summing
    # ~256 of them in float16 (max ~65504) can overflow to inf, which then
    # poisons the classifier (confirmed on Kaggle's CUDA float16 math --
    # MPS's float16 apparently tolerated it, CUDA's didn't. Real bug, not a
    # platform quirk: the fix is dtype hygiene, not a CUDA workaround).
    hidden = outputs.hidden_states[layer].float()  # (batch=1, seq_len, hidden_dim)
    pooled = hidden.mean(dim=1).squeeze(0)          # mean-pool over tokens -> (hidden_dim,)
    return pooled.cpu().numpy()


def extract_activations_batch(texts, model, tokenizer, device, layer, batch_size=16, verbose=True):
    """
    Batched version of extract_activation -- one forward pass per BATCH
    instead of per text. Same math, much faster on GPU (Kaggle T4 etc.):
    a 1.5B model's activations for a batch of 16-32 short prompts cost
    almost nothing extra over batch=1, since the model weights (not the
    activations) dominate memory.

    Correctness note: naive `hidden.mean(dim=1)` on a padded batch would
    average in the padding positions' hidden states, silently corrupting
    every vector's pooling. This masks padding out BEFORE averaging, using
    attention_mask, so each text's vector only pools over its own real
    tokens -- identical math to calling extract_activation() one at a time,
    just batched.
    """
    import torch

    all_vecs = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=256, padding=True
        ).to(device)
        with torch.no_grad():
            outputs = _base_transformer(model)(**inputs, output_hidden_states=True)

        # float32 before the reduction, not after -- see extract_activation's
        # comment: summing ~256 float16 residual-stream values can overflow
        # to inf (confirmed on Kaggle CUDA), especially here where batching
        # adds even more terms into the same sum.
        hidden = outputs.hidden_states[layer].float()               # (batch, seq_len, hidden_dim)
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (batch, seq_len, 1)
        summed = (hidden * mask).sum(dim=1)                         # (batch, hidden_dim)
        counts = mask.sum(dim=1).clamp(min=1e-9)                    # (batch, 1) -- real-token count per text
        pooled = summed / counts

        all_vecs.append(pooled.cpu().numpy())

        done = min(start + batch_size, n)
        if verbose and (done % 50 < batch_size or done == n):
            print(f"  extracted {done}/{n}")

    return np.concatenate(all_vecs, axis=0)


def build_activation_dataset(examples, model, tokenizer, device, layer, batch_size=16):
    """examples: list of (text, label). Returns X (n, hidden_dim), y (n,)."""
    texts = [t for t, _ in examples]
    y = np.array([label for _, label in examples])
    X = extract_activations_batch(texts, model, tokenizer, device, layer, batch_size=batch_size)
    return X, y


# ---------------------------------------------------------------------------
# 2. Probe training + evaluation
# ---------------------------------------------------------------------------

def train_and_eval_probe(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    # LogisticRegressionCV automatically searches for the regularization
    # strength (C) via internal cross-validation on the TRAINING data only.
    # This matters here because hidden_dim (~1500+) >> n_train (~100s),
    # which makes plain LogisticRegression prone to finding spurious
    # correlations even in noise -- exactly what your control test flagged.
    # Guard against inf/nan in the activation matrices before handing them
    # to sklearn -- fails with a message that points at the actual cause
    # (extraction-side numerical issue) instead of sklearn's generic
    # "contains infinity" error with no indication of where it came from.
    for name, arr in [("X_train", X_train), ("X_test", X_test)]:
        if not np.all(np.isfinite(arr)):
            n_bad = np.count_nonzero(~np.isfinite(arr))
            raise ValueError(
                f"{name} contains {n_bad} non-finite value(s) (inf/nan) -- this means activation "
                f"extraction overflowed, most likely a float16 reduction issue (see "
                f"extract_activation/extract_activations_batch in this file). Not a data problem."
            )

    clf = LogisticRegressionCV(
        max_iter=2000, class_weight="balanced", cv=5, Cs=10,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / len(y_test)

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "false_positive_rate": round(float(fpr), 4),
        "accuracy": round(float(accuracy), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }, clf


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from benchmark_v2 import DATASET_LOADERS  # reuse your v2 dataset loaders

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deepset", "advbench_mix", "minimal_pairs", "xstest"], default="deepset")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=8, help="Which hidden layer to probe")
    parser.add_argument("--layer_sweep", action="store_true",
                         help="Try several layers, report which works best")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Cap dataset size for faster local runs (e.g. 150)")
    parser.add_argument("--batch_size", type=int, default=16,
                         help="Activation-extraction batch size. Bigger = faster on GPU "
                              "(Kaggle T4 etc.), doesn't meaningfully increase memory since "
                              "model weights dominate, not activations. Lower this if you "
                              "genuinely hit OOM on a small/CPU-only box.")
    args = parser.parse_args()

    from sklearn.model_selection import train_test_split

    print(f"Loading dataset: {args.dataset} ...")
    examples = DATASET_LOADERS[args.dataset](args.max_samples)
    print(f"Loaded {len(examples)} examples.")

    model, tokenizer, device = load_model(args.model)

    # Split BEFORE extracting activations -- avoids leaking test examples
    # into anything (not strictly needed here since extraction has no
    # learned params, but this is the correct habit to build now).
    train_ex, test_ex = train_test_split(
        examples, test_size=args.test_size, random_state=42,
        stratify=[label for _, label in examples],
    )
    print(f"Train: {len(train_ex)}  Test: {len(test_ex)}")

    layers_to_try = args.layer_sweep and range(2, model.config.num_hidden_layers, 4) or [args.layer]

    results = {}
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for layer in layers_to_try:
        print(f"\n--- Layer {layer} ---")
        start = time.time()
        X_train, y_train = build_activation_dataset(train_ex, model, tokenizer, device, layer, batch_size=args.batch_size)
        X_test, y_test = build_activation_dataset(test_ex, model, tokenizer, device, layer, batch_size=args.batch_size)
        metrics, clf = train_and_eval_probe(X_train, y_train, X_test, y_test)
        metrics["extraction_time_sec"] = round(time.time() - start, 2)
        results[f"layer_{layer}"] = metrics
        print(json.dumps(metrics, indent=2))

    out_path = os.path.join(logs_dir, f"probe_v3_{args.dataset}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {os.path.abspath(out_path)}")

    if len(results) > 1:
        best_layer = max(results, key=lambda k: results[k]["f1"])
        print(f"\nBest layer by F1: {best_layer} -> {results[best_layer]}")