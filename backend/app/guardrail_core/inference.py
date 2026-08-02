"""
FinGuard inference core — the product's guardrail engine.

Trimmed, self-contained version of the research repo's ai_guardrail.py +
probe_v3.py, stripped of everything training-related. This module only
LOADS a pre-trained artifact (shipped in ./artifacts/) and runs inference.
No sklearn training, no dataset loaders, no torch autograd needed at
request time.

General-purpose scope: detector trained on deepset + advbench_mix (broad
harm categories, not domain-biased). No policy-category evidence layer --
that was a finance-specific R&D feature (still in the research repo's
policy_directions.py/evidence_v5.py, not wired into this shipped product).

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


def get_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class GuardrailEngine:
    """Loads the model + trained classifier ONCE (at FastAPI startup) and
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

    def check(self, text: str) -> dict:
        start = time.perf_counter()
        act = self.extract_activation(text)

        pred = int(self.clf.predict(act.reshape(1, -1))[0])
        proba = float(self.clf.predict_proba(act.reshape(1, -1))[0][1])

        return {
            "flagged": bool(pred),
            "flag_confidence": round(proba, 4),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }


_engine = None


def get_engine() -> GuardrailEngine:
    """Lazy singleton -- loaded once per process (FastAPI worker)."""
    global _engine
    if _engine is None:
        _engine = GuardrailEngine()
    return _engine
