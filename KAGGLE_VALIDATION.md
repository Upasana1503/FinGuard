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
