# Running the standard benchmarks on Kaggle

Your M1 (8GB unified memory) is fine for the activation probe itself, but
downloading + running full external guardrail models for comparison (like
we hit with Granite Guardian) eats the whole machine. Kaggle gives you a
free GPU (T4, 30 hrs/week) and ~13GB+ RAM with none of that pressure — do
the validation runs there instead.

## What "validation on the standard benchmarks" means here

Your project has been validated so far on:
- `deepset/prompt-injections` (546 ex., real published dataset) ✅
- FinSec-MinPairs (your own 64-example finance benchmark) ✅

Not yet validated on the three benchmarks guardrail papers actually get
judged against:
- **XSTest** — the over-refusal / pseudo-harm benchmark (this is the one
  closest to your actual thesis — benign prompts that LOOK risky)
- **OR-Bench** — larger, newer over-refusal benchmark, cross-checks XSTest
- **WildGuardTest** — the standard comparison point most guardrail papers
  report against (may be gated on HuggingFace — check when you get there;
  if it asks you to log in / accept terms, you'll need a free HF account +
  token, same as any gated model)

## Step-by-step on Kaggle

1. **Create a new Notebook** on kaggle.com (not a Script) — Settings →
   Accelerator → **GPU T4 x2** (or whichever T4 option is offered on the
   free tier).

2. **Enable internet access** in the notebook settings (needed to pull the
   model + datasets from HuggingFace) — off by default on Kaggle.

3. **Upload your repo** — easiest path: add your GitHub repo as a Kaggle
   "Dataset" via File → Add Data → GitHub, pointing at
   `https://github.com/Upasana1503/FinGuard`, OR just `git clone` it in
   the first notebook cell:
   ```python
   !git clone https://github.com/Upasana1503/FinGuard.git
   %cd FinGuard/src
   !pip install -q transformers torch scikit-learn datasets joblib
   ```

4. **Add loaders for the new benchmarks.** Your `benchmark_v2.py` currently
   only knows `deepset`, `advbench_mix`, and `minimal_pairs`. You'll need to
   add loader functions for XSTest and OR-Bench, same pattern as
   `load_deepset()`:
   ```python
   def load_xstest(max_samples=None):
       from datasets import load_dataset
       ds = load_dataset("natolambert/xstest-v2-copy")["prompts"]  # check exact HF path/config when you get there
       # XSTest's own convention: prompt_type starting with "contrast_" = should be refused (label 1),
       # everything else = safe (label 0). Confirm this mapping against XSTest's paper/dataset card before trusting it.
       examples = [(row["prompt"], 0 if row["type"].startswith("contrast") else 1) for row in ds]
       if max_samples:
           examples = examples[:max_samples]
       return examples
   ```
   (Dataset card details change — search "xstest huggingface" when you're
   actually there and confirm the exact config name and label convention
   before running anything. Don't trust the snippet above blindly, verify
   against the real dataset card.)

5. **Run the same scripts you already have**, just pointed at the new
   dataset name, e.g.:
   ```python
   !python benchmark_v2.py --dataset xstest
   !python probe_v3.py --dataset xstest --layer_sweep
   ```

## Batching + subsetting (the actual fix for OOM/slowness)

`probe_v3.py` and `diagnose_leakage.py` now extract activations in BATCHES
instead of one text at a time (`--batch_size`, default 16) — verified this
produces numerically identical results (cosine similarity >0.999999 against
the old one-at-a-time path; the tiny float16 difference is just
batching-order rounding noise, not a bug). This is a real speedup, not just
a memory workaround: a 1.5B model's activations for a batch of 16-32 short
prompts cost almost nothing extra over batch=1, since model *weights*
dominate memory, not activations.

Practical settings for Kaggle's T4:
```bash
# raise batch_size on GPU -- 32-64 should be comfortable for a 1.5B model
python probe_v3.py --dataset xstest --layer 8 --batch_size 32

# WildGuardTest is ~92k examples -- don't run the full set blind on your
# first pass. Subset first to sanity-check the loader/labels are right,
# THEN scale up:
python probe_v3.py --dataset wildguardtest --max_samples 2000 --batch_size 32   # smoke test
python probe_v3.py --dataset wildguardtest --batch_size 32                       # full run, once the smoke test looks right
```

If you genuinely hit OOM (unlikely on a T4 for a 1.5B model, but possible
if you're also holding a second model in memory — see the warning below),
lower `--batch_size` back down rather than reducing `--max_samples`;
subsetting changes what you're measuring, batch size doesn't.

6. **Save results before your session ends** — Kaggle sessions aren't
   permanent. Either commit the notebook (File → Save Version) so the
   output logs persist, or explicitly write results to `/kaggle/working/`
   and download them before closing.

7. **Bring the numbers back here** — once you have real XSTest/OR-Bench/
   WildGuardTest results, paste them back into this conversation (or just
   tell me you're ready) and I'll fold them into PROJECT_SUMMARY.md and,
   if the numbers hold up, retrain the shipped product's artifacts on the
   validated data before you deploy.

## What NOT to do on Kaggle

- Don't try to load two big models in the same session at once (this is
  exactly what killed the Granite Guardian comparison locally — same
  memory-pressure failure mode applies on Kaggle too, just with a higher
  ceiling before it bites).
- Don't leave a session idle with the GPU accelerator on — burns your
  30 free hrs/week for nothing.
