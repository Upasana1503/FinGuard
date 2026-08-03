---
title: FinGuard API
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# FinGuard API

Activation-based AI guardrail for LLM agents. General-purpose harmful-intent
detector — reads the LLM's internal activations to judge intent instead of
matching keywords in prompt text, so it holds up better against paraphrasing
than a regex filter. Built on Qwen2.5-1.5B-Instruct.

Real validated numbers (full history in the main repo's `PROJECT_SUMMARY.md`):
AUC-ROC 0.889, F1 0.794, FPR 8.6% on the WildGuardTest benchmark (1699
examples, zero-shot). Head-to-head against IBM Granite Guardian on the same
held-out data: FinGuard's false-positive rate is 8.6% vs Granite's 46.0%.

**Endpoints**: `POST /v1/signup` (get an API key) · `POST /v1/check` (score a
prompt) · `GET /v1/usage` · `GET /health`

Source + full research history: https://github.com/Upasana1503/FinGuard
