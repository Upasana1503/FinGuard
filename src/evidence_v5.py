"""
Guardrail v5 — Policy-cited evidence generation, wired to real activations.

Earlier plan for v5 (see PROJECT_SUMMARY.md) was a rule-based layer: take a
flagged prompt's TEXT, pattern-match it into a policy category after the
fact. That's just another regex classifier wearing a compliance costume —
it inherits the exact "surface text vs. intent" weakness v1 already has.

This version wires v5 to what the model actually did internally:

  1. v3-style binary probe (LogisticRegressionCV over mid-layer activations)
     decides IF a prompt is flagged -- the validated mechanism (F1=0.975 on
     deepset, see PROJECT_SUMMARY.md Sec.9).
  2. IF flagged, policy_directions.py's diff-of-means category directions
     decide WHICH named policy category the activation moves along, and by
     how much (cosine similarity) -- an attribution read off the SAME
     activation vector, not reconstructed from text.
  3. A small static POLICY_REFERENCE table maps each category to an
     illustrative regulatory framework citation, so the output is something
     a compliance reviewer can actually act on instead of a bare score.

IMPORTANT — this is a research prototype demonstrating the evidence-
generation CONCEPT, not verified legal/compliance guidance. The citations
below name real frameworks but are illustrative category tags chosen by the
author, not the output of a legal-citation-retrieval system. Say this
explicitly in any write-up.

Usage:
    python evidence_v5.py --layer 8                 # demo on held-out pairs
    python evidence_v5.py --layer 8 --prompt "..."   # score a single prompt
"""

import argparse
import json
import os
import time

import numpy as np

POLICY_REFERENCE = {
    "unauthorized_fund_transfer": {
        "label": "Unauthorized Fund Transfer",
        "reference": "Interagency (Fed/FDIC/OCC) guidance on payment/wire-transfer dual-authorization "
                      "controls; internal treasury-ops segregation-of-duties policy.",
    },
    "insider_trading_facilitation": {
        "label": "Insider Trading / MNPI Misuse",
        "reference": "SEC Rule 10b-5 and Regulation FD (trading on or tipping material non-public information).",
    },
    "structuring_money_laundering": {
        "label": "Structuring / Money Laundering",
        "reference": "Bank Secrecy Act, 31 U.S.C. Sec.5324 (structuring transactions to evade "
                      "reporting requirements); FinCEN AML program requirements.",
    },
    "account_takeover_finance": {
        "label": "Account Takeover / Authentication Bypass (Financial)",
        "reference": "GLBA Safeguards Rule (customer authentication controls); "
                      "interagency guidance on identity verification for account access.",
    },
    "credential_bypass_intrusion": {
        "label": "Credential Bypass / Unauthorized Access",
        "reference": "NIST SP 800-53 AC-7 / IA-2 (authentication enforcement); "
                      "Computer Fraud and Abuse Act (unauthorized access), absent documented authorization.",
    },
    "malware_development": {
        "label": "Malicious Software Development",
        "reference": "NIST SP 800-53 SI-3 (malicious code protection); CISA guidance on malware "
                      "development/distribution absent an authorized research or defensive-testing scope.",
    },
    "data_exfiltration": {
        "label": "Data Exfiltration",
        "reference": "GLBA Safeguards Rule / NIST SP 800-53 SC-7 (boundary protection); "
                      "applicable breach-notification statutes if customer data is involved.",
    },
    "social_engineering_phishing": {
        "label": "Social Engineering / Phishing",
        "reference": "FTC guidance on deceptive practices; CISA phishing-prevention guidance; "
                      "internal security-awareness policy on impersonation.",
    },
}

EVIDENCE_DISCLAIMER = (
    "Research-prototype output. Policy category mapping is an illustrative "
    "taxonomy chosen by the author, not a verified legal/compliance "
    "citation-retrieval result. Do not use as compliance advice."
)


def generate_evidence(text, activation, flagged, flag_confidence, directions,
                       top_k=3, min_similarity=0.05):
    """
    Build the structured evidence record for one prompt.

    activation: raw activation vector for `text` (already extracted, so
      callers running this over many prompts only pay the forward-pass cost
      once per prompt regardless of how many downstream layers use it).
    directions: {category: unit_vector} from policy_directions.train_category_directions
      (or policy_directions.load_directions).
    """
    from policy_directions import score_activation

    record = {
        "prompt_preview": text[:200],
        "flagged": bool(flagged),
        "flag_confidence": round(float(flag_confidence), 4),
        "disclaimer": EVIDENCE_DISCLAIMER,
    }

    if not flagged:
        record["policy_attribution"] = []
        return record

    ranked = score_activation(activation, directions)
    attributions = []
    for category, sim in ranked[:top_k]:
        if sim < min_similarity:
            continue
        ref = POLICY_REFERENCE.get(category, {"label": category, "reference": "(no reference mapped)"})
        attributions.append({
            "category": category,
            "policy_label": ref["label"],
            "policy_reference": ref["reference"],
            "activation_cosine_similarity": round(float(sim), 4),
        })
    record["policy_attribution"] = attributions
    return record


# ---------------------------------------------------------------------------
# End-to-end pipeline: train the v3 detector + v5 directions, then evidence
# ---------------------------------------------------------------------------

def build_pipeline(model, tokenizer, device, layer, pairs_meta, test_size=0.25):
    """Trains BOTH the v3-style binary detector and the v5 policy directions
    on the same FinSec-MinPairs split, at the same layer, so activations are
    only extracted once per text. Returns (clf, directions, held_out_pairs)."""
    from sklearn.model_selection import train_test_split
    from probe_v3 import extract_activation, train_and_eval_probe
    from policy_directions import train_category_directions

    directions, direction_metrics = train_category_directions(
        pairs_meta, model, tokenizer, device, layer, test_size=test_size
    )

    flat = []
    for p in pairs_meta:
        flat.append((p["benign_text"], 0))
        flat.append((p["malicious_text"], 1))
    train_flat, test_flat = train_test_split(
        flat, test_size=test_size, random_state=42, stratify=[l for _, l in flat]
    )
    X_train = np.stack([extract_activation(t, model, tokenizer, device, layer) for t, _ in train_flat])
    y_train = np.array([l for _, l in train_flat])
    X_test = np.stack([extract_activation(t, model, tokenizer, device, layer) for t, _ in test_flat])
    y_test = np.array([l for _, l in test_flat])
    detector_metrics, clf = train_and_eval_probe(X_train, y_train, X_test, y_test)

    return clf, directions, direction_metrics, detector_metrics


def run_on_prompt(text, model, tokenizer, device, layer, clf, directions):
    from probe_v3 import extract_activation

    act = extract_activation(text, model, tokenizer, device, layer)
    pred = int(clf.predict(act.reshape(1, -1))[0])
    proba = clf.predict_proba(act.reshape(1, -1))[0][1] if hasattr(clf, "predict_proba") else float(pred)
    return generate_evidence(text, act, flagged=bool(pred), flag_confidence=proba, directions=directions)


if __name__ == "__main__":
    from probe_v3 import load_model
    from minimal_pairs import load_pairs_with_meta

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--prompt", type=str, default=None,
                         help="Score a single ad-hoc prompt instead of the demo set")
    args = parser.parse_args()

    pairs_meta = load_pairs_with_meta()
    model, tokenizer, device = load_model(args.model)

    print("Training v3 detector + v5 policy directions on FinSec-MinPairs "
          f"(layer {args.layer}) ...")
    clf, directions, direction_metrics, detector_metrics = build_pipeline(
        model, tokenizer, device, args.layer, pairs_meta, test_size=args.test_size
    )
    print("\nDetector (v3-style) held-out metrics:")
    print(json.dumps(detector_metrics, indent=2))
    print("\nPolicy-direction held-out metrics:")
    print(json.dumps({k: v for k, v in direction_metrics.items() if k != "per_pair_results"}, indent=2))

    if args.prompt:
        demo_prompts = [args.prompt]
    else:
        # Demo on a handful of malicious minimal-pair examples across categories.
        demo_prompts = [p["malicious_text"] for p in pairs_meta[::8]]  # one per category

    print(f"\n{'=' * 60}\n  Evidence output\n{'=' * 60}")
    records = []
    for text in demo_prompts:
        record = run_on_prompt(text, model, tokenizer, device, args.layer, clf, directions)
        print(json.dumps(record, indent=2))
        records.append(record)

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, f"evidence_v5_layer{args.layer}_demo.json")
    with open(out_path, "w") as f:
        json.dump({
            "detector_metrics": detector_metrics,
            "direction_metrics": {k: v for k, v in direction_metrics.items() if k != "per_pair_results"},
            "evidence_records": records,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)
    print(f"\nSaved to {os.path.abspath(out_path)}")
