"""
Guardrail v4 — Trajectory-aware activation probe.

v3 classified ONE message using its activation vector.
v4 classifies the LAST message in a SESSION (sequence of turns), using
not just its own activation vector, but how that vector has DRIFTED
relative to earlier turns in the same session.

Why this might help: a message can look ambiguous in isolation but be
much more (or less) suspicious in context. E.g. "transfer the funds now"
means something different after a session that's been escalating toward
unauthorized access vs. after a session that's clearly an authenticated
routine banking chat.

IMPORTANT LIMITATION (state this honestly in your report):
Real multi-turn attack datasets are hard to source/build. This script
constructs SYNTHETIC sessions by stitching together single-turn examples
from your existing dataset: most sessions are all-benign, some sessions
are benign turns followed by one injection turn (simulating escalation).
This is a reasonable starting point, NOT a substitute for a real
multi-turn dataset -- say so explicitly if you write this up.

Usage:
    python probe_v4.py --dataset deepset --session_len 3 --max_sessions 150
"""

import argparse
import json
import os
import random

import numpy as np

from probe_v3 import load_model, extract_activation, train_and_eval_probe, get_device


# ---------------------------------------------------------------------------
# 1. Synthetic session construction
# ---------------------------------------------------------------------------

def build_synthetic_sessions(examples, session_len=3, n_sessions=150, seed=42):
    """
    examples: list of (text, label) from your existing loaders (v2).
    Returns: list of (session_turns: list[str], session_label: int)
      session_label = 1 if the session ends in a malicious turn, else 0.

    Roughly half the sessions are all-benign; half are benign turns
    followed by one malicious final turn (the escalation pattern).
    """
    random.seed(seed)
    benign = [t for t, l in examples if l == 0]
    malicious = [t for t, l in examples if l == 1]

    if len(malicious) < 5:
        raise ValueError(
            f"Only {len(malicious)} malicious examples available -- "
            "too few to build meaningful sessions. Try a larger dataset."
        )

    sessions = []
    for i in range(n_sessions):
        make_malicious_session = i % 2 == 0
        turns = random.sample(benign, k=session_len - 1)
        if make_malicious_session:
            turns.append(random.choice(malicious))
            label = 1
        else:
            turns.append(random.choice(benign))
            label = 0
        sessions.append((turns, label))

    random.shuffle(sessions)
    return sessions


# ---------------------------------------------------------------------------
# 2. Trajectory feature extraction
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(1 - np.dot(a, b) / denom)


def extract_contextual_turn_vectors(session_turns, model, tokenizer, device, layer):
    """
    Returns one activation vector per turn, where turn i's vector comes from
    running the model over the CONVERSATION SO FAR (turns[0..i] joined), not
    turn i in isolation.

    CORRECTNESS NOTE (found while building the trajectory-drift visualization,
    see PROJECT_SUMMARY.md Sec.9/10): the original version of this function
    called extract_activation() on each turn independently. That means every
    turn's vector was computed with ZERO knowledge of the turns before it --
    there was no mechanism by which earlier turns could shape the model's
    representation of a later one. "Drift" computed from those vectors was
    therefore near-zero and directionless (empirically ~0.0004-0.001,
    indistinguishable between malicious and benign sessions) -- not because
    the trajectory hypothesis is false, but because the feature literally
    could not encode trajectory. This fixes that: each vector is now
    extracted with the full preceding context in the same forward pass, so a
    later turn's representation can actually be influenced by what came
    before it.

    Limitation: extract_activation() truncates at 256 tokens, so very long or
    many-turn sessions will lose early context -- fine for the session
    lengths used so far (session_len<=4, short turns), worth revisiting if
    session length grows.
    """
    vectors = []
    for i in range(len(session_turns)):
        context_text = "\n\n".join(session_turns[: i + 1])
        vectors.append(extract_activation(context_text, model, tokenizer, device, layer))
    return vectors


def extract_trajectory_features(session_turns, model, tokenizer, device, layer):
    """
    For a session (list of turn texts), extract:
      - the LAST turn's raw activation vector (same as v3 would use alone)
      - drift: cosine distance between last turn and FIRST turn
      - step_drift: cosine distance between last turn and PREVIOUS turn
      - cumulative_drift: sum of turn-to-turn distances across the session

    Final feature vector = [last_turn_vector, drift, step_drift, cumulative_drift]
    This lets the probe use both "what does the last message look like"
    (v3's signal) AND "how has this session been trending" (the new signal).
    """
    turn_vectors = extract_contextual_turn_vectors(session_turns, model, tokenizer, device, layer)

    first_vec = turn_vectors[0]
    prev_vec = turn_vectors[-2] if len(turn_vectors) > 1 else turn_vectors[0]
    last_vec = turn_vectors[-1]

    drift = cosine_distance(last_vec, first_vec)
    step_drift = cosine_distance(last_vec, prev_vec)
    cumulative_drift = sum(
        cosine_distance(turn_vectors[i], turn_vectors[i - 1])
        for i in range(1, len(turn_vectors))
    )

    return np.concatenate([last_vec, [drift, step_drift, cumulative_drift]])


def build_trajectory_dataset(sessions, model, tokenizer, device, layer):
    X, y = [], []
    for i, (turns, label) in enumerate(sessions):
        feat = extract_trajectory_features(turns, model, tokenizer, device, layer)
        X.append(feat)
        y.append(label)
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(sessions)} sessions")
    return np.array(X), np.array(y)


def build_lastturn_only_dataset(sessions, model, tokenizer, device, layer):
    """For a fair v3-vs-v4 comparison: same sessions, but ONLY the last
    turn's raw vector (no trajectory features) -- this is what v3 alone
    would see if given just the final message."""
    X, y = [], []
    for turns, label in sessions:
        vec = extract_activation(turns[-1], model, tokenizer, device, layer)
        X.append(vec)
        y.append(label)
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from benchmark_v2 import DATASET_LOADERS
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deepset", "advbench_mix", "minimal_pairs"], default="deepset")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--session_len", type=int, default=3)
    parser.add_argument("--max_sessions", type=int, default=150)
    parser.add_argument("--test_size", type=float, default=0.3)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} ...")
    examples = DATASET_LOADERS[args.dataset](None)
    print(f"Loaded {len(examples)} single-turn examples.")

    sessions = build_synthetic_sessions(
        examples, session_len=args.session_len, n_sessions=args.max_sessions
    )
    print(f"Built {len(sessions)} synthetic sessions "
          f"({sum(l for _, l in sessions)} end-in-malicious).")

    train_sessions, test_sessions = train_test_split(
        sessions, test_size=args.test_size, random_state=42,
        stratify=[label for _, label in sessions],
    )

    model, tokenizer, device = load_model(args.model)

    print("\n=== Extracting TRAJECTORY features (v4) ===")
    X_train_traj, y_train = build_trajectory_dataset(train_sessions, model, tokenizer, device, args.layer)
    X_test_traj, y_test = build_trajectory_dataset(test_sessions, model, tokenizer, device, args.layer)
    metrics_v4, _ = train_and_eval_probe(X_train_traj, y_train, X_test_traj, y_test)

    print("\n=== Extracting LAST-TURN-ONLY features (v3 baseline, same sessions) ===")
    X_train_last, _ = build_lastturn_only_dataset(train_sessions, model, tokenizer, device, args.layer)
    X_test_last, _ = build_lastturn_only_dataset(test_sessions, model, tokenizer, device, args.layer)
    metrics_v3, _ = train_and_eval_probe(X_train_last, y_train, X_test_last, y_test)

    print("\n" + "=" * 60)
    print("  v3 (last turn only) vs v4 (full trajectory)")
    print("=" * 60)
    print("v3:", json.dumps(metrics_v3, indent=2))
    print("v4:", json.dumps(metrics_v4, indent=2))

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, f"probe_v4_{args.dataset}_comparison.json")
    with open(out_path, "w") as f:
        json.dump({"v3_last_turn_only": metrics_v3, "v4_trajectory": metrics_v4}, f, indent=2)
    print(f"\nSaved comparison to {os.path.abspath(out_path)}")