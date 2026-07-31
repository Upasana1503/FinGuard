# Project Master Summary — Activation-Based Guardrails for Ambiguous, Domain-Specific Intent Detection

*Use this document to resume this project in a new conversation with full context. Paste it as-is at the start of a new chat.*

---

## 1. Who I am / context

Undergrad preparing for placements, targeting **AIML SDE and AIML Research roles**, aiming this project at the **finance and cybersecurity domains**. Compute: MacBook M1 (local dev) + Kaggle free-tier GPU (final runs). Comfortable with math/ML fundamentals but learning transformer internals and research methodology as I go. No dataset-building experience going in — LLM-assisted dataset generation + existing benchmarks are the practical path.

## 2. End goal

Produce a **research-grade (not necessarily top-tier-venue, but arXiv/workshop-quality) project** that:
- Demonstrates a real, defensible technical contribution (not just an applied wrapper)
- Produces actual numeric results comparing my approach against existing open-source guardrails
- Is honest about its scope and limitations
- Doubles as a strong SDE portfolio piece (working code, benchmarks, clean repo) AND a research-role talking point (a real, citable gap, real experiments, honest negative results where they occurred)

## 3. The problem statement (final, as of this document)

> Guardrails deployed for autonomous LLM agents in regulated, high-stakes domains (finance, cybersecurity) face two compounding, jointly-unaddressed failures: (1) surface-text/single-step detection conflates domain-specific professional language with genuine malicious intent — a "pseudo-harm" / "over-refusal" problem explicitly named as underexplored in existing literature (documented true-positive rates as low as 43.1% on complex violations) — and (2) even when detection succeeds, guardrails produce opaque binary/scalar outputs that cannot satisfy the auditable, policy-citable evidence regulators now require for agentic systems under interagency model-risk-management guidance (Fed/FDIC/OCC).
>
> This project addresses both by (a) evaluating pseudo-harm/over-refusal specifically in finance/cybersecurity contexts using established benchmarks (XSTest, OR-Bench) extended with a small domain-specific slice, (b) testing whether activation-based detection (reading internal model representations rather than surface text) reduces this false-positive problem relative to text-based baselines, and optionally (c) generating structured, policy-mapped evidence trails as the detection output, rather than an opaque score.

## 4. The precise hypothesis being tested

1. **Primary**: Activation-based detection outperforms text/regex-based detection at identifying harmful intent, especially on ambiguous, context-dependent language (domain jargon) where surface wording is misleading.
2. **Secondary**: Incorporating conversational trajectory (activation drift across multiple turns), not just the current message, further improves disambiguation specifically on ambiguous cases — general single-turn attacks don't show this benefit (already confirmed — see Results).
3. **Tertiary (added, v5/v6)**: A named, per-policy-category direction in activation space (diff-of-means) can (a) separate benign/malicious WITHIN a category even when the two share surface lexicon (confirmed, see §9), and (b) the resulting binary probe is robust to synonym substitution specifically — i.e. it is reading something closer to intent than lexicon, not just a more expensive keyword matcher (confirmed, see §9 robustness table) — though it is NOT robust to token-fragmenting obfuscation or distractor-dilution attacks (also confirmed, an honest limitation, not hidden).

## 5. Important honest framing

This is **not a novel detection architecture** — activation-based single/multi-turn detection already exists in the literature (see §8, "already covered" column). My defensible contribution is narrower:
- **FinSec-MinPairs**: a hand-authored contrastive minimal-pair benchmark (64 pairs / 128 examples, 8 finance+cybersec policy categories) where each pair shares the same trigger keywords/action and differs ONLY in stated authorization/context. This is the dataset contribution that replaces the earlier vague "extend XSTest with a slice" plan — a controlled variable gives a much cleaner causal claim than a random domain-specific slice, and it directly fixes the reason v4's trajectory result was inconclusive (deepset isn't ambiguous enough — see §9)
- Applying/evaluating activation probing specifically on **domain-specific pseudo-harm** (finance/cybersecurity) via FinSec-MinPairs, which no existing benchmark or paper targets directly
- **Diff-of-means policy-direction attribution** (v5, `policy_directions.py`): instead of a rule-based text→category mapping (the original v5 plan, which was really just another regex layer), each named policy category gets a direction in activation space learned via difference-of-means (same technique as "refusal direction" interpretability work). A flagged prompt's evidence is a decomposed, ranked cosine-similarity attribution over these directions, not an opaque score — and it's read off the same activation vector the detector used, not reconstructed from text after the fact. Existing detection-focused papers (TrajGuard, LAD, NeuroFilter, AgentDoG) do not produce this
- An honestly-reported empirical comparison, including negative/inconclusive results where they occurred (v4 trajectory on deepset; category-attribution accuracy on FinSec-MinPairs, both below)

## 6. System architecture (build order)

| Stage | File | What it does | Status |
|---|---|---|---|
| v1 | `guardrail_v1.py` | Regex/rule-based baseline: PII, injection patterns, finance-sensitive-action detection | ✅ Built, tested |
| v2 | `benchmark_v2.py` | Evaluation harness: precision/recall/F1/FPR/latency on real datasets (deepset, AdvBench-mix) | ✅ Built, tested |
| v3 | `probe_v3.py` | Activation probe: extracts mid-layer hidden states from a small open LLM (Qwen2.5-1.5B-Instruct), trains logistic regression (CV-regularized) to classify harmful/benign from internal activations | ✅ Built, tested, validated |
| v4 | `probe_v4.py` | Trajectory-aware probe: extracts activations across a session of turns, adds drift/step-drift/cumulative-drift features, compares against last-turn-only baseline | ✅ Built, tested; result inconclusive on current dataset (see §9) |
| Diagnostic | `diagnose_leakage.py` | Control test: trains probe on randomized labels to check for leakage/shortcut learning; near-duplicate/template detector | ✅ Built, used successfully to catch a real bug |
| Dataset | `data/minimal_pairs.json`, `minimal_pairs.py` | FinSec-MinPairs: 64 hand-authored contrastive pairs (128 ex.) across 8 finance/cybersec policy categories; loader + pair-discrimination-accuracy metric | ✅ Built, wired into `benchmark_v2.py` DATASET_LOADERS as `minimal_pairs` |
| v5 (core) | `policy_directions.py` | Diff-of-means direction per policy category in activation space; cosine-similarity scoring/attribution; held-out eval (category-attribution accuracy, within-category pair discrimination) | ✅ Built, run on FinSec-MinPairs (layer 8) — see §9 |
| v5 (evidence) | `evidence_v5.py` | Orchestrator: v3-style probe decides IF flagged, `policy_directions.py` decides WHICH category + cosine score, static table maps category → illustrative policy/regulatory reference | ✅ Built, wired to real activations end-to-end (rewritten from the earlier rule-based/demo-data plan), run on FinSec-MinPairs — see §9 |
| v6 | `robustness_v6.py` | Adversarial robustness check: roleplay-wrap, leetspeak, zero-width-space insertion, synonym substitution, distractor-padding (dilution attack targeting v3's mean-pooling specifically) vs v1 and v3 detection rate | ✅ Built, run on FinSec-MinPairs (layer 8) — see §9 |
| **Final product** | `ai_guardrail.py` | The packaged deliverable: single `ActivationGuardrail` class (`.check()` / `.check_session()`), wrapping the validated pieces — v3-style detector trained on combined deepset+FinSec-MinPairs (674 ex.), v5's policy-direction evidence, trained artifacts persisted to `logs/ai_guardrail_artifacts/` (joblib + npz, no retraining per call). Trajectory drift is attached to `.check_session()` output as an explicitly-labeled experimental/informational signal, NOT used to gate the decision (still unvalidated, see below) | ✅ Built, trained, run — see §9 |
| Comparison baseline | `guardrail_granite.py` | Wrapper for **IBM Granite Guardian 3.0-2B** (ungated, real deployed open-source guardrail product used in IBM watsonx.governance) using its prescribed prompt-template + Yes/No-token scoring, matching the same result interface as `guardrail_v1.run_guardrail` | ✅ Built, verified correct (correctly scored a benign and a malicious smoke-test prompt) |
| Comparison harness | `compare_products.py` | Head-to-head: v1 vs `ai_guardrail.py` vs `guardrail_granite.py`, all evaluated on the exact same held-out split (`ai_guardrail`'s own reserved test set, so no system has a training-data advantage) | ✅ Built, run — see §9 (small n, see fairness note there) |
| Diagnostic (new) | `trajectory_drift_log.py` | Logs the FULL per-turn drift sequence (not just the final aggregate features v4 trains on) for synthetic sessions built from FinSec-MinPairs, for visualization | ✅ Built, run, charted — see §9 |

**Note**: v5 is explicitly a research prototype demonstrating the evidence-generation *concept*, not a production system — the policy/regulatory references in `evidence_v5.py`'s `POLICY_REFERENCE` table are an illustrative taxonomy chosen by the author, not a verified legal-citation-retrieval result (the module says this in its own disclaimer field). The end product of this project is "a validated technique + honest benchmark results + a proof-of-concept evidence layer," not a deployable guardrail.

## 7. Tech stack

- **Model**: Qwen2.5-1.5B-Instruct (open weights, no gating/license approval needed, small enough for M1 local dev, swappable for larger models on Kaggle for final numbers)
- **Local dev**: M1 MacBook, VSCode, Python venv, PyTorch with `mps` backend
- **Final/heavier runs**: Kaggle free-tier (30 GPU-hrs/week, T4, longer sessions than Colab)
- **Libraries**: transformers, torch, scikit-learn (LogisticRegressionCV), datasets, numpy

## 8. Gap-checking log (research ideas checked against literature — method to reuse for future ideas)

This is the actual skill being practiced: **every "novel" idea was cross-checked against current literature before being trusted.** Log format: idea → verdict.

| Idea | Verdict |
|---|---|
| Generic "activation probing for safety" | Already done (multiple papers, 2604.18901 etc.) |
| Multi-turn/trajectory activation drift detection | Already done — LAD (2604.28129), TrajGuard (2604.07727), NeuroFilter |
| Graph-based agent guardrails | Already done — SentinelAgent, Agentproof |
| "Recovery instead of refusal" | Already done — TRIAD (2606.05805) |
| Root-cause/diagnostic guardrails | Already done — AgentDoG |
| Context reducing false positives (text-based) | Already done — CAPTURE (2505.12368) |
| Safety/utility/latency tradeoff study (general) | Already claimed — "No Free Lunch with Guardrails" (2504.00441) |
| **Pseudo-harm named as unsolved by NFL paper itself** | **Confirmed open** — explicitly flagged, not solved by NFL |
| **Domain-specific (finance/cybersec) pseudo-harm benchmark** | **Confirmed open, now built** — FinSec-MinPairs (64 contrastive pairs, 8 categories); see §9 for a first result on it |
| **Activation-based (not text-based) context disambiguation** | **Confirmed open** — CAPTURE does this with text, not activations. Not yet tested with v4-style trajectory features on FinSec-MinPairs (next step, §13) |
| **Policy-citable, audit-grade evidence generation** | **Confirmed open, now built (pilot)** — diff-of-means policy directions (`policy_directions.py`) give real activation-derived category attribution; within-category discrimination is strong (1.0 on held-out pairs) but cross-category attribution accuracy is weak (0.125) at current dataset size — an honest limitation, see §9 |

**Method for finding these**: read Limitations/Future Work sections deliberately; triangulate across subfields that don't normally cite each other (interpretability + compliance/governance, in this case); before trusting any "novel" idea (mine or another LLM's), search for it explicitly and look for the closest existing paper.

## 9. Results so far (the actual numbers)

**v1 vs v3 on deepset/prompt-injections (546 examples, full dataset):**

| Method | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| v1 (regex baseline) | 1.0 | 0.069 | 0.129 | Catches only exact-phrasing attacks; paraphrases sail through |
| v3 (activation probe, best layer) | 1.0 | 0.951 | 0.975 | ~7x F1 improvement over v1 |

**This is the strongest, most defensible result of the project so far.**

**v3 vs v4 (trajectory) on synthetic sessions built from deepset:**

| Method | F1 | Notes |
|---|---|---|
| v3-style (last turn only) | 0.978 | |
| v4 (trajectory features added) | 0.989 | Improvement is real-direction but only 1 example difference out of 90 test cases — **not statistically conclusive** |

**Why v4's improvement is inconclusive**: deepset's individual attack messages are unambiguous enough on their own that v3 alone already scores near-ceiling (~0.98), leaving little room for trajectory context to add value. **deepset is the wrong dataset to prove the trajectory/disambiguation hypothesis** — it doesn't contain genuinely ambiguous, context-dependent cases. This is a real, useful negative finding, not a dead end.

**v1 vs v3 on FinSec-MinPairs (128 examples, 64 contrastive pairs, layer 8):**

| Method | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| v1 (regex baseline) | 1.0 | 0.016 | 0.031 | Catches 1/64 malicious examples — worse than on deepset, because pairs deliberately avoid literal injection phrasing ("ignore previous instructions") and use natural professional language instead |
| v3 (activation probe, layer 8) | 0.95 | 1.0 | 0.974 | Held-out F1 essentially matches the deepset result, on a benchmark specifically designed to be hard for surface-lexicon methods |

**This is now the sharper headline result** — the v1/v3 gap on FinSec-MinPairs (F1 0.031 → 0.974, ~31x) is larger than on deepset (~7x), and it's the fair test the project was missing: same trigger words in both classes, so the gap can't be explained by v1 just not knowing the right regex.

**Pair-discrimination sanity check** (does a detector actually use context, or just trigger-keyword matching?): a naive keyword-matcher baseline (flags any text containing transfer/bypass/malware/exploit/phish/launder/withdraw) scores **1/64 = 0.016 pair-discrimination accuracy** on FinSec-MinPairs — confirms the pairs are genuinely lexically confusable and the benchmark is doing its job.

**Policy-direction attribution (v5, `policy_directions.py`, layer 8, 48 train / 16 test pairs):**

| Metric | Value | Notes |
|---|---|---|
| Own-direction pair discrimination | 1.0 | On held-out pairs, the malicious member always scores higher on its OWN category's diff-of-means direction than its paired benign member — the binary signal within a category is very clean |
| Category-attribution accuracy | 0.125 | On held-out malicious examples, the TOP-ranked category direction matches the true category only 12.5% of the time — cosine similarities across categories are all very high (>0.99) and close together, suggesting the directions are dominated by a shared "general harm" component rather than cleanly separated per-category axes |

**Honest read**: the diff-of-means approach validates cleanly as a binary signal (matches v3's detection numbers) but does NOT yet deliver reliable fine-grained category attribution — likely an artifact of small n (6 training pairs/category). Candidate fixes for next iteration: orthogonalize each category direction against the shared/global harm direction before scoring, or grow FinSec-MinPairs per category before re-evaluating. Reporting this as-is rather than only showing the flattering number.

**Adversarial robustness (v6, `robustness_v6.py`, FinSec-MinPairs, layer 8, detection rate on 19 held-out malicious examples):**

| Transform | v1 (regex) | v3 (probe) | Notes |
|---|---|---|---|
| clean | 0.0 | 1.0 | baseline |
| synonym_substitute | 0.0 | 1.0 | v3 fully unaffected — direct evidence for "intent not lexicon," the core claim of the project |
| leetspeak | 0.0 | 0.684 | character-level obfuscation partially corrupts the representation |
| zero_width_insert | 0.0 | 0.421 | token-fragmenting attack meaningfully hurts v3 |
| distractor_pad | 0.0 | 0.316 | dilution attack (designed specifically against v3's mean-pooling) works as hypothesized |
| roleplay_wrap | 1.0 | 0.316 | interesting split: the "you are now..." framing accidentally triggers v1's `role_override` regex (100% detection) while simultaneously diluting v3's mean-pooled signal enough to drop its detection to 0.316 |

**Honest read**: the "activation probe beats regex" story from §9's main result is NOT universal — it's a claim about semantic paraphrase specifically (synonym_substitute confirms it cleanly). Under token-fragmenting or dilution attacks, v3's mean-pooling is a real, specific weakness, and v1's exact-match regex is (perversely) invariant to some of these by construction. This is a genuine limitation worth reporting, not a result to bury — and a natural pointer toward a mitigation to try next (e.g. attention-weighted pooling instead of mean-pooling, or pooling only over the final N tokens) rather than plain mean-pooling.

**Final product (`ai_guardrail.py`) detector, trained on combined deepset + FinSec-MinPairs (674 examples, 135 held out, layer 8):**

| Metric | Value |
|---|---|
| Precision | 0.93 |
| Recall | 1.0 |
| F1 | 0.9637 |
| Accuracy | 0.9704 |
| TP/FP/TN/FN | 53 / 4 / 78 / 0 |

Consistent with every earlier activation-probe result in this project (deepset-only: F1 0.975; FinSec-MinPairs-only: F1 0.974) — combining both training sources for the shipped product didn't cost accuracy.

**Head-to-head vs a real open-source guardrail product** (`compare_products.py`, IBM Granite Guardian 3.0-2B, ungated and the only general-purpose open guardrail this environment could actually download without a gated-model HF token — Llama Guard 3 / WildGuard / ShieldGemma all require one): in progress / see follow-up run for numbers. Fairness notes that apply regardless of the final numbers: Granite Guardian is scored zero-shot on its stock "harm" risk definition (never trained on this project's data, exactly how IBM ships it); `ai_guardrail` is scored on a held-out split it was never trained on; v1 is included as the cheap-baseline reference point, not a serious competitor. **Practical constraint worth recording**: Granite Guardian's per-call latency on this M1 setup was ~130-390s/prompt (both MPS and CPU) because unlike the probes, it does autoregressive `generate()` rather than one forward pass — this capped the head-to-head to a small held-out sample rather than the full 135-example split. A larger run belongs on Kaggle GPU, consistent with §7's original compute-tier plan (heavy runs were always meant to go there, not local).

**Trajectory-drift visualization — a second real bug caught while building it**: `probe_v4.py`'s `extract_trajectory_features` computed each turn's activation independently (`extract_activation(turn_text, ...)` per turn, no memory of earlier turns) — meaning "drift" was structurally incapable of reflecting conversational trajectory, since no turn's representation could be shaped by what came before it. This was caught by plotting the raw per-turn numbers (`trajectory_drift_log.py`) and noticing they were tiny (~0.0004-0.001) and directionless. Fixed via a new `extract_contextual_turn_vectors()` (`probe_v4.py`) that extracts turn *i*'s vector from the joined text of turns 0..i, so later turns' representations are actually conditioned on earlier ones — used now by `probe_v4.py`, `ai_guardrail.py`'s `.check_session()`, and `trajectory_drift_log.py`.

**After the fix**: drift now grows sensibly with turn count (turn 1 → turn 3 average rises from ~0.002 to ~0.014-0.015) — mechanically correct. But it still does **not** separate malicious from benign sessions (malicious avg ≈0.0140 vs benign avg ≈0.0154 at the final turn — no meaningful gap, benign if anything slightly higher). Diagnosis: `build_synthetic_sessions` (used unchanged from the original v4 design) splices together *randomly sampled, topically unrelated* benign turns before the final turn — ordinary topic-jumping already produces plenty of drift regardless of how the session ends, swamping any signal from the final turn's category. **Not yet fixed** — the fix would be building sessions where the earlier turns stay within ONE FinSec-MinPairs category (topically coherent escalation) rather than random category mixing, which the dataset's own `category` field makes possible but hasn't been tried. Full interactive chart (40 sessions, individual + average lines, raw data table) built and reviewed as part of this pass.

## 10. Known failure points / mistakes caught (important to document honestly)

1. **First v3/v4 run was broken**: flat, suspiciously perfect (1.0) scores across every layer, on a small (`max_samples=150`), unshuffled subset. Caught by noticing the implausible flatness across layers, then confirmed/ruled out via a randomized-label control test (`diagnose_leakage.py`).
2. **Bug: `max_samples` truncated without shuffling** — fixed by shuffling before truncation in `benchmark_v2.py`'s dataset loaders.
3. **Bug: `diagnose_leakage.py`'s default `--max_samples` was 150** instead of `None`, so a "full dataset" control run silently still used only 150 examples. Fixed.
4. **Overfitting risk from high-dimensional activations (~1500+ dims) vs small sample sizes (~100s)** — addressed by switching from plain `LogisticRegression` to `LogisticRegressionCV` (cross-validated regularization strength).
5. **AdvBench on HuggingFace is gated** (requires login/token) — switched to pulling the original public CSV directly from GitHub instead.
6. **SSL certificate errors on macOS python.org installs** — standard fix (`Install Certificates.command`) plus a `certifi`-based fallback added to the code.
7. **v4's trajectory features were computed from context-blind activations**: each turn's vector came from running the model on that turn ALONE, so no turn's representation could reflect anything about earlier turns — "drift" was measuring noise by construction, not conversational trajectory. Caught by logging and looking at the raw per-turn numbers (near-zero, no trend) instead of trusting the aggregate feature. Fixed with `extract_contextual_turn_vectors()` in `probe_v4.py` (see §9); this means every v4 result reported before this fix (including the deepset v3-vs-v4 comparison in §9) was computed on the flawed extraction and should be treated as superseded, not just "inconclusive due to an easy dataset" as originally concluded.

## 11. Datasets — what to use for what

| Dataset | Role |
|---|---|
| `deepset/prompt-injections` (HF, 546 ex.) | Used for v1-v4 initial development and the core "regex vs probe" comparison |
| AdvBench (public GitHub CSV) | Secondary, harder malicious-only set, mixed with OpenOrca benign for a second data point |
| **FinSec-MinPairs** (`data/minimal_pairs.json`, hand-authored, 64 pairs / 128 ex.) | **The dataset contribution.** Contrastive finance/cybersec pairs sharing trigger lexicon, differing only in authorization/context. Used for the sharper v1-vs-v3 comparison, the policy-direction eval (v5), and the adversarial robustness check (v6). This is what §13's old "small XSTest extension" plan turned into — the pair design gives a controlled variable instead of a random domain slice |
| **WildGuardMix / WildGuardTest** (Han et al., 2024, ~92k ex.) | **The standard, most-cited benchmark for comparing against real published open-source guardrails** (Llama Guard, WildGuard, ShieldGemma, etc.) — use this for the main "compare my method numerically against existing guardrails" claim |
| **XSTest** (Röttger et al.) | **The canonical over-refusal / pseudo-harm benchmark** — benign prompts with trigger words that shouldn't be blocked. Plan: use as the base, extend with a small finance/cybersecurity-specific slice as the dataset contribution, rather than building from scratch |
| **OR-Bench** | Second, larger dedicated over-refusal benchmark; cross-check against XSTest |
| ToxicChat | Third standard leg of the typical eval trio (WildGuardTest + ToxicChat + XSTest), used in most recent guardrail papers |

**Not yet done**: downloading/running against WildGuardTest, XSTest, OR-Bench. This is the immediate next step.

## 12. Reading list (papers to actually read, roughly in dependency order)

**Fundamentals (read first if terminology is unfamiliar)**
- Jay Alammar, "The Illustrated Transformer" / "The Illustrated GPT-2" (blog, not a paper — explains tokens/attention/residual stream in plain terms)
- 3Blue1Brown neural network series (YouTube)
- Andrej Karpathy, "Neural Networks: Zero to Hero" (YouTube/GitHub)

**Guardrails foundations**
- NeMo Guardrails (Rebedea et al., 2023)
- OWASP Top 10 for LLM Applications

**Detection / benchmarks**
- AdvBench / Zou et al. 2023 (original adversarial jailbreak paper)
- HarmBench (Mazeika et al., 2024)
- XSTest (Röttger et al.) — over-refusal benchmark
- WildGuard (Han et al., 2024) — arXiv 2406.18495

**Activation-based detection (core technical basis)**
- "Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams" — arXiv 2604.18901
- CAPTURE (context-aware over-defense) — arXiv 2505.12368
- LAD — arXiv 2604.28129; TrajGuard — arXiv 2604.07727 (multi-turn trajectory detection, prior art to cite/build on)
- Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (2024) — the diff-of-means direction-extraction technique `policy_directions.py` (v5) is built on; cite directly as the method's origin, not as prior art on THIS project's application of it to per-category policy attribution

**Evaluation methodology**
- "When Benchmarks Lie: Evaluating Malicious Prompt Classifiers Under True Distribution Shift" — arXiv 2602.14161 (directly relevant — this is the exact failure mode encountered in §10.1)

**Compliance framing**
- "Type-Checked Compliance... Lean 4" — arXiv 2604.01483 (skim for the "probabilistic isn't enough for regulation" argument; the formal-verification method itself is out of scope)

**Prior art to explicitly cite as "related work, not reinvented"**
- TRIAD (2606.05805), AgentDoG, SentinelAgent, Agentproof, NeuroFilter

## 13. Immediate next steps (in order)

**Done since last revision**: FinSec-MinPairs built (was step 4 below); v5 wired to real activations via diff-of-means policy directions (was step 6); a first adversarial robustness pass exists (was step 7, in spirit — v6 tests transform robustness rather than multi-seed variance, both still needed); packaged final product (`ai_guardrail.py`) built and trained; first real head-to-head against an actual open-source guardrail product (Granite Guardian 3.0-2B) built and run (small n, see §9); the v4 context-blind-activation bug found and fixed; trajectory-drift visualized and the fix's honest limits documented.

**New priorities from this pass**:
0a. Re-run the granite-guardian head-to-head at full held-out size (135 examples) on Kaggle GPU — this machine's ~130-390s/call `generate()` latency made only a small-n run practical locally (§9)
0b. If a HF token with accepted licenses becomes available, add Llama Guard 3 / WildGuard / ShieldGemma as additional comparison points in `compare_products.py` — Granite Guardian was the only ungated option found
0c. Build category-consistent synthetic sessions (reuse FinSec-MinPairs' `category` field so earlier turns in a session stay topically coherent, not randomly mixed) and re-run `trajectory_drift_log.py` + `probe_v4.py` — the real fair test of whether trajectory drift separates malicious/benign now that the context-blind-extraction bug (§9/§10) is fixed

1. Rerun `diagnose_leakage.py` with the fixed default on the full deepset dataset — close the loop on trustworthiness (still open)
2. Run `probe_v4.py --dataset minimal_pairs` — FinSec-MinPairs is the genuinely-ambiguous dataset the trajectory hypothesis needed; deepset was confirmed the wrong dataset for this (§9), this is the fair test
3. Fix policy-direction category-attribution accuracy (currently 0.125, §9): try orthogonalizing each category direction against the shared/global harm direction before scoring, and/or grow FinSec-MinPairs beyond 8 pairs/category
4. Grow FinSec-MinPairs itself (target ~15-20 pairs/category) — current n (6 train pairs/category) is almost certainly why category attribution is noisy; more data is the first thing to rule out before concluding the linear-direction approach doesn't separate categories
5. Run `robustness_v6.py` on deepset and/or WildGuardTest too, not just FinSec-MinPairs — check whether the synonym-robust/dilution-vulnerable pattern (§9) holds generally or is specific to this benchmark
6. Try a pooling fix for the dilution vulnerability found in §9 (attention-weighted pooling or last-N-token pooling instead of mean-pooling over the whole sequence) and re-run v6 to see if it closes the gap
7. Download and run v1 + v3 against **WildGuardTest** — first real comparison against published guardrail numbers
8. Download **XSTest** (and/or OR-Bench) — cross-check FinSec-MinPairs findings against the standard over-refusal benchmarks
9. Multi-seed / cross-validated robustness check on any close comparisons (v3 vs v4, category-attribution accuracy) before reporting them as conclusive — n is still small everywhere in §9's new results
10. Write up: honest methodology section including the leakage catch (§10) and the v5/v6 honest limitations (§9: weak category attribution, mean-pooling dilution vulnerability) as demonstrations of rigor, not hidden mistakes

## 14. How to resume with an LLM

Paste this entire document at the start of a new conversation. It contains: the goal, the exact hypothesis, what's built, what's proven, what failed and was fixed, which datasets to use next, and the reading list. Ask directly for help on whichever numbered next step (§13) you're on.
