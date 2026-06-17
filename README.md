# Multilingual Health QA in Low-Resource African Languages

Reproducible solution for the **Zindi / Mak-AI Multilingual Health Question Answering**
challenge: answer maternal, sexual & reproductive health (MSRH) questions in **Akan, Amharic,
Luganda, Swahili and English**, each answer in the same language as its question.

**Best public leaderboard score: 0.598** · Local ROUGE portion: 0.370 (from a 0.315 baseline).

> ▶ **Run on Colab:** open `notebooks/HASH_HealthQA.ipynb` — the single unified notebook covering
> the entire challenge (EDA → retrieval → fine-tuning → submission). Replace the repo URL in the
> first cell, then *Run all* (use a GPU runtime for the fine-tuning section). Add a Colab badge once
> the repo is on GitHub:
> `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<user>/<repo>/blob/main/notebooks/HASH_HealthQA.ipynb)`

## The governing insight
The metric is `0.37·ROUGE-1 F1 + 0.37·ROUGE-L F1 + 0.26·LLM-judge` → **74% is lexical overlap**
with a specific reference answer. In this templated domain the highest-value approach is
**retrieval** of lexically-close reference answers; generation is layered on only where
retrieval is structurally capped. Every decision is justified against this metric.

## Approach (summary)
1. **Per-subset character n-gram TF-IDF retrieval** over a **global** answer bank.
2. **Multilingual semantic rerank** (`paraphrase-multilingual-mpnet-base-v2`) of top-k candidates.
3. **Ghana input-hybrid** (splice question into answer) for the weak configs.
4. **Val-as-bank** for the Test submission (provided labelled data as retrieval memory).
5. **mT5 fine-tuning** (full + LoRA) for the Ghana configs, **per-subset Val-gated** (kept only if
   it beats retrieval — it did not, and that negative result is documented).

A validated **`public ≈ local + 0.096`** calibration enabled fully-offline iteration.

## Repository layout
```
src/
  qa_common.py            seed, data loading, local ROUGE scoreboard, submission writer
  retrieval.py            per-subset routing + global-bank TF-IDF retrieval (top-k)
  rerank.py               multilingual sentence-embedding rerank
  hybrid.py               Ghana input×retrieval hybrid
  baseline_retrieval.py   reproduces the 0.315 baseline
  make_submission_v*.py   build submissions (v2 routing, v3 rerank, v4 +hybrid, v5 +gen-gate)
  train_generator.py      mT5/AfriTeVa fine-tuning (learning curves, LoRA, seeded)
  exp_*.py                logged experiments (routing, val-bank, rerank, hybrid, diagnostics, voting, gating)
  make_figures.py         regenerate all report figures
  build_unified_notebook.py / build_report_html.py     regenerate the notebook / report HTML
notebooks/                HASH_HealthQA.ipynb — single unified end-to-end Colab notebook
reports/                  REPORT.md, figures/
submissions/              generated Zindi submissions (4-col format)
EXPERIMENTS.md            full experiment log (every idea + Val score + insight)
requirements.txt / requirements.lock.txt
Data/                     Train.csv, Val.csv, Test.csv (not committed)
SampleSubmission.csv
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place Train.csv, Val.csv, Test.csv in Data/  and SampleSubmission.csv at the root
```

## Reproduce the results
```bash
source .venv/bin/activate
python src/baseline_retrieval.py        # -> local portion 0.3150 (sanity floor)
python src/make_submission_v3.py        # rerank pipeline (downloads MPNet once)
python src/make_submission_v4.py        # + Ghana hybrid -> submissions/submission_v4.csv (best)
python src/make_figures.py              # -> reports/figures/*.png
```
Fine-tuning (GPU) runs in the fine-tuning section of `notebooks/HASH_HealthQA.ipynb`; it writes
`val_gen.csv`/`test_gen.csv` into `generator_outputs/<run>/`, then:
```bash
python src/make_submission_v5.py --gen-dir generator_outputs/mt5small_full
```

## Results
| Stage | Local portion (/0.74) | Public |
|---|---:|---:|
| Baseline (char_wb retrieval) | 0.3150 | — |
| + routing + global bank | 0.3292 | — |
| + MPNet rerank | 0.3572 | 0.579 |
| **+ Ghana hybrid (best)** | **0.3701** | **0.598** |

See `reports/REPORT.md` for full analysis, figures, and the 14 documented experiments.

## Reproducibility & compliance
Seed fixed (=42) everywhere; pinned `requirements.lock.txt`; deterministic submission generation.
Only open-licensed packages/models (Apache-2.0 / MIT); challenge data is CC-BY-SA 4.0. No paid
services, no AutoML, no private data. A tooling/assistance disclosure is provided at the end of the report.
