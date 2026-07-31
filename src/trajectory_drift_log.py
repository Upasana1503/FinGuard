"""
Raw per-turn trajectory-drift logger — feeds the visualization requested
alongside the head-to-head product comparison.

probe_v4.py only ever saved the FINAL aggregate drift features (one number
per session) because that's all the classifier needs. To actually SEE
whether malicious sessions drift further/faster than benign ones, we need
the full per-turn sequence, not just the endpoint summary. This script
re-runs session construction + activation extraction (reusing probe_v4's
synthetic-session builder, now pointed at FinSec-MinPairs instead of
deepset -- FinSec-MinPairs is the dataset actually built to make the
trajectory hypothesis testable, see PROJECT_SUMMARY.md Sec.9) and logs the
distance-from-first-turn and distance-from-previous-turn at EVERY turn, for
every session, so it can be plotted as a line chart.

Usage:
    python trajectory_drift_log.py --session_len 4 --n_sessions 40 --layer 8
"""

import argparse
import json
import os


if __name__ == "__main__":
    from benchmark_v2 import DATASET_LOADERS
    from probe_v3 import load_model
    from probe_v4 import build_synthetic_sessions, cosine_distance, extract_contextual_turn_vectors

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--session_len", type=int, default=4)
    parser.add_argument("--n_sessions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    examples = DATASET_LOADERS["minimal_pairs"](None)
    print(f"Loaded {len(examples)} FinSec-MinPairs examples.")

    sessions = build_synthetic_sessions(
        examples, session_len=args.session_len, n_sessions=args.n_sessions, seed=args.seed
    )
    print(f"Built {len(sessions)} synthetic sessions "
          f"({sum(l for _, l in sessions)} end-in-malicious).")

    model, tokenizer, device = load_model(args.model)

    logged_sessions = []
    for idx, (turns, label) in enumerate(sessions):
        vecs = extract_contextual_turn_vectors(turns, model, tokenizer, device, args.layer)
        drift_from_first = [0.0] + [round(cosine_distance(vecs[i], vecs[0]), 4) for i in range(1, len(vecs))]
        step_drift = [0.0] + [round(cosine_distance(vecs[i], vecs[i - 1]), 4) for i in range(1, len(vecs))]
        logged_sessions.append({
            "session_index": idx,
            "label": label,
            "n_turns": len(turns),
            "turns_preview": [t[:80] for t in turns],
            "drift_from_first_turn": drift_from_first,
            "step_drift": step_drift,
        })
        if (idx + 1) % 10 == 0:
            print(f"  processed {idx + 1}/{len(sessions)} sessions")

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, "trajectory_drift_sessions.json")
    with open(out_path, "w") as f:
        json.dump({
            "layer": args.layer, "session_len": args.session_len,
            "n_sessions": len(sessions), "sessions": logged_sessions,
        }, f, indent=2)
    print(f"\nSaved per-turn drift trajectories to {os.path.abspath(out_path)}")
