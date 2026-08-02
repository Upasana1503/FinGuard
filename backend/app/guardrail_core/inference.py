"""
FinGuard inference core — the product's guardrail engine.

Trimmed, self-contained version of the research repo's ai_guardrail.py +
probe_v3.py + policy_directions.py + evidence_v5.py, stripped of everything
training-related. This module only LOADS pre-trained artifacts (shipped in
./artifacts/) and runs inference. No sklearn training, no dataset loaders,
no torch autograd needed at request time.

Deliberately excludes trajectory-drift / multi-turn tracking: that
component was built and tested in the research repo and found NOT to
discriminate malicious from benign sessions (see PROJECT_SUMMARY.md).
Shipping it here would mean shipping something that doesn't work -- so the
product only exposes the validated single-message check.
"""

import json
import os
import time

import numpy as np

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

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
}

EVIDENCE_DISCLAIMER = (
    "Policy category mapping is an illustrative taxonomy, not verified legal/compliance "
    "advice. Use flagged output as a signal for human review, not an automated legal determination."
)


def get_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class GuardrailEngine:
    """Loads the model + trained artifacts ONCE (at FastAPI startup) and
    serves .check(text) calls cheaply after that -- one forward pass per
    call, no reloading."""

    def __init__(self, model_name: str = MODEL_NAME, artifacts_dir: str = ARTIFACTS_DIR):
        import joblib
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with open(os.path.join(artifacts_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        self.layer = self.metadata["layer"]

        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            output_hidden_states=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.clf = joblib.load(os.path.join(artifacts_dir, "detector_clf.joblib"))

        directions_data = np.load(os.path.join(artifacts_dir, "policy_directions.npz"))
        categories = directions_data["categories"].tolist()
        matrix = directions_data["matrix"]
        self.directions = {cat: matrix[i] for i, cat in enumerate(categories)}

    def extract_activation(self, text: str) -> np.ndarray:
        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        # Call the bare transformer (skip the LM head) -- we only read
        # hidden_states, never logits, and computing lm_head over the full
        # ~150k vocab for every request is pure wasted memory/compute (see
        # src/probe_v3.py's _base_transformer for the full explanation --
        # this is what caused a real OOM on long-text batches).
        base = self.model.model if hasattr(self.model, "model") else self.model
        with torch.no_grad():
            outputs = base(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[self.layer]
        pooled = hidden.mean(dim=1).squeeze(0)
        return pooled.float().cpu().numpy()

    def check(self, text: str, top_k: int = 3, min_similarity: float = 0.05) -> dict:
        start = time.perf_counter()
        act = self.extract_activation(text)

        pred = int(self.clf.predict(act.reshape(1, -1))[0])
        proba = float(self.clf.predict_proba(act.reshape(1, -1))[0][1])

        record = {
            "flagged": bool(pred),
            "flag_confidence": round(proba, 4),
            "policy_attribution": [],
            "disclaimer": EVIDENCE_DISCLAIMER,
        }

        if pred:
            sims = [(cat, cosine_similarity(act, d)) for cat, d in self.directions.items()]
            sims.sort(key=lambda x: x[1], reverse=True)
            for category, sim in sims[:top_k]:
                if sim < min_similarity:
                    continue
                ref = POLICY_REFERENCE.get(category, {"label": category, "reference": "(no reference mapped)"})
                record["policy_attribution"].append({
                    "category": category,
                    "policy_label": ref["label"],
                    "policy_reference": ref["reference"],
                    "activation_cosine_similarity": round(float(sim), 4),
                })

        record["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return record


_engine = None


def get_engine() -> GuardrailEngine:
    """Lazy singleton -- loaded once per process (FastAPI worker)."""
    global _engine
    if _engine is None:
        _engine = GuardrailEngine()
    return _engine
