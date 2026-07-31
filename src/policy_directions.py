"""
Policy-direction probe — the technical core of v5.

Existing detection work (v3/v4, and the literature surveyed in
PROJECT_SUMMARY.md Sec.8: TrajGuard, LAD, NeuroFilter, AgentDoG) outputs a
single scalar/binary "harmful or not." That's not what a compliance officer
under Fed/FDIC/OCC model-risk-management guidance can cite as evidence.

This module replaces v5's old plan (rule-based text -> policy-category
mapping, i.e. just another regex layer bolted after the fact) with something
actually derived from the model's internal state: for each named policy
category (unauthorized_fund_transfer, credential_bypass_intrusion, ...),
learn a DIRECTION in activation space via difference-of-means:

    direction_c = mean(activations of malicious examples in category c)
                - mean(activations of benign examples in category c)

This is the same technique used in the "refusal direction" line of work
(Arditi et al.) precisely because it's stable in low-data regimes where a
full classifier would overfit -- which matters here since FinSec-MinPairs
has only 8 pairs (16 examples) per category.

Given a NEW flagged prompt, we project its activation onto every category
direction (cosine similarity) and rank them. The output is a decomposed,
named attribution ("0.71 unauthorized_fund_transfer, 0.12 pii_exfiltration")
instead of an opaque score -- an evidence trail that's actually read off the
model's internals, not reconstructed after the fact from the surface text.

Detection (is this malicious at all?) is deliberately left to v3's
validated probe -- this module only answers "given something was flagged,
which named policy direction does it move along, and by how much."

Usage:
    python policy_directions.py --layer 8 --test_size 0.25
"""

import argparse
import json
import os

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


# ---------------------------------------------------------------------------
# 1. Training: diff-of-means direction per category
# ---------------------------------------------------------------------------

def train_category_directions(pairs_meta, model, tokenizer, device, layer, test_size=0.25, seed=42):
    """
    pairs_meta: list of dicts from minimal_pairs.load_pairs_with_meta()
      (pair_id, domain, category, benign_text, malicious_text).

    Splits PAIRS (not raw examples) into train/test per category so the
    held-out evaluation never sees a pair used to build its own direction.
    With only 8 pairs/category this is a small-n pilot -- say so in any
    write-up, don't oversell the held-out numbers.

    Returns: (directions: {category: unit_vector}, eval_metrics: dict)
    """
    import random
    from collections import defaultdict
    from probe_v3 import extract_activation

    random.seed(seed)
    by_category = defaultdict(list)
    for p in pairs_meta:
        by_category[p["category"]].append(p)

    train_pairs, test_pairs = [], []
    for category, plist in by_category.items():
        shuffled = plist[:]
        random.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_size))
        test_pairs += shuffled[:n_test]
        train_pairs += shuffled[n_test:]

    # Cache activations once per unique text (benign/malicious texts don't overlap).
    print(f"Extracting activations for {len(train_pairs) + len(test_pairs)} pairs "
          f"({2 * (len(train_pairs) + len(test_pairs))} texts) at layer {layer} ...")
    act_cache = {}

    def get_act(text):
        if text not in act_cache:
            act_cache[text] = extract_activation(text, model, tokenizer, device, layer)
        return act_cache[text]

    directions = {}
    train_counts = {}
    for category, plist in by_category.items():
        cat_train = [p for p in train_pairs if p["category"] == category]
        if not cat_train:
            continue
        benign_acts = np.stack([get_act(p["benign_text"]) for p in cat_train])
        malicious_acts = np.stack([get_act(p["malicious_text"]) for p in cat_train])
        direction = _unit(malicious_acts.mean(axis=0) - benign_acts.mean(axis=0))
        directions[category] = direction
        train_counts[category] = len(cat_train)

    # Held-out evaluation: for each test pair's malicious example, does the
    # TOP-attributed direction match the true category? And does the
    # malicious member always score higher on its own direction than the
    # paired benign member (the minimal-pair discrimination test, but for
    # the direction itself rather than a downstream classifier)?
    category_attribution_correct = 0
    own_direction_discriminates = 0
    per_pair_results = []
    for p in test_pairs:
        b_act = get_act(p["benign_text"])
        m_act = get_act(p["malicious_text"])
        sims = {cat: cosine_similarity(m_act, d) for cat, d in directions.items()}
        top_category = max(sims, key=sims.get)
        correct = top_category == p["category"]
        category_attribution_correct += int(correct)

        own_dir = directions[p["category"]]
        b_sim = cosine_similarity(b_act, own_dir)
        m_sim = cosine_similarity(m_act, own_dir)
        discriminates = m_sim > b_sim
        own_direction_discriminates += int(discriminates)

        per_pair_results.append({
            "pair_id": p["pair_id"], "true_category": p["category"],
            "top_attributed_category": top_category, "attribution_correct": correct,
            "benign_sim_own_direction": round(b_sim, 4),
            "malicious_sim_own_direction": round(m_sim, 4),
            "own_direction_discriminates": discriminates,
        })

    n_test = len(test_pairs)
    metrics = {
        "layer": layer,
        "n_categories": len(directions),
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": n_test,
        "train_pairs_per_category": train_counts,
        "category_attribution_accuracy": round(category_attribution_correct / n_test, 4) if n_test else None,
        "own_direction_pair_discrimination": round(own_direction_discriminates / n_test, 4) if n_test else None,
        "per_pair_results": per_pair_results,
    }
    return directions, metrics


# ---------------------------------------------------------------------------
# 2. Scoring a new prompt against saved directions
# ---------------------------------------------------------------------------

def score_text(text, model, tokenizer, device, layer, directions):
    """Returns list of (category, cosine_similarity) sorted descending."""
    from probe_v3 import extract_activation

    act = extract_activation(text, model, tokenizer, device, layer)
    sims = [(cat, cosine_similarity(act, d)) for cat, d in directions.items()]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims


def score_activation(act, directions):
    """Same as score_text but takes an already-extracted activation vector."""
    sims = [(cat, cosine_similarity(act, d)) for cat, d in directions.items()]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims


# ---------------------------------------------------------------------------
# 3. Persistence
# ---------------------------------------------------------------------------

def save_directions(path, directions, layer):
    categories = list(directions.keys())
    matrix = np.stack([directions[c] for c in categories])
    np.savez(path, matrix=matrix, categories=np.array(categories), layer=layer)


def load_directions(path):
    data = np.load(path, allow_pickle=False)
    categories = data["categories"].tolist()
    matrix = data["matrix"]
    layer = int(data["layer"])
    directions = {cat: matrix[i] for i, cat in enumerate(categories)}
    return directions, layer


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from probe_v3 import load_model
    from minimal_pairs import load_pairs_with_meta

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--test_size", type=float, default=0.25)
    args = parser.parse_args()

    pairs_meta = load_pairs_with_meta()
    print(f"Loaded {len(pairs_meta)} minimal pairs across "
          f"{len(set(p['category'] for p in pairs_meta))} policy categories.")

    model, tokenizer, device = load_model(args.model)

    directions, metrics = train_category_directions(
        pairs_meta, model, tokenizer, device, args.layer, test_size=args.test_size
    )

    print(json.dumps({k: v for k, v in metrics.items() if k != "per_pair_results"}, indent=2))

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    directions_path = os.path.join(logs_dir, f"policy_directions_layer{args.layer}.npz")
    save_directions(directions_path, directions, args.layer)
    print(f"\nSaved {len(directions)} policy directions to {os.path.abspath(directions_path)}")

    metrics_path = os.path.join(logs_dir, f"policy_directions_layer{args.layer}_eval.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved eval metrics to {os.path.abspath(metrics_path)}")
