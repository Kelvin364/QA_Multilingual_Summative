# Multilingual Health QA in Low-Resource African Languages

Reproducible solution for the **Zindi / Mak-AI Multilingual Health Question Answering**
challenge: answer maternal, sexual & reproductive health (MSRH) questions in **Akan, Amharic,
Luganda, Swahili and English**, each answer in the same language as its question.

**Best public leaderboard score: 0.598** · Local ROUGE portion: 0.370 (from a 0.315 baseline).

[![Open In Colab](https://drive.google.com/file/d/1Km8lQxZzjxUVbUl0MPqpEBXDPr1EpCdK/view?usp=sharing)

>  **Run on Colab:** open `notebooks/End-to-End-Notebook.ipynb` (badge above) — the unified
> end-to-end notebook covering the whole challenge (EDA → retrieval → fine-tuning → submission),
> and now a **"Best 8 experiments" section that runs each experiment live**. *Run all* on a GPU
> runtime. The GPU-only fine-tuning can also be run on its own via
> [`notebooks/colab_finetune_ghana.ipynb`](https://colab.research.google.com/github/Kelvin364/QA_Multilingual_Summative/blob/main/notebooks/colab_finetune_ghana.ipynb).

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
  make_figures.py         regenerate all figures into figures/
  build_best_experiments.py  rebuild the notebook's "Best 8 experiments" section + the HTML report
notebooks/
  End-to-End-Notebook.ipynb   unified end-to-end Colab notebook (EDA → retrieval → fine-tune →
                              submission), incl. the live "Best 8 experiments" section
  colab_finetune_ghana.ipynb  standalone GPU notebook for the mT5 fine-tuning
figures/                  all generated figures (fig1–fig9)
reports/
  best_experiments.html       standalone HTML report of the 8 best experiments (self-contained)
  experiment_logs/            genuine captured stdout from each experiment run
submissions/              generated Zindi submissions (4-col format)
EXPERIMENTS.md            full experiment log (every idea + Val score + insight)
requirements.txt / requirements.lock.txt
Data/                     Train.csv, Val.csv, Test.csv (not committed — challenge data)
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
python src/make_figures.py              # -> figures/*.png
```
Fine-tuning (GPU) runs in the fine-tuning section of `notebooks/End-to-End-Notebook.ipynb` (or the
standalone `notebooks/colab_finetune_ghana.ipynb`); it writes `val_gen.csv`/`test_gen.csv` into
`generator_outputs/<run>/`, then:
```bash
python src/make_submission_v5.py --gen-dir generator_outputs/mt5small_full
```

### Best 8 experiments (live proof of execution)
Each experiment runs end-to-end and prints a genuine scoreboard (no hardcoded tables):
```bash
python src/baseline_retrieval.py   # 1. baseline floor          -> 0.3150
python src/exp_a_routing.py        # 2. routing + global bank   -> 0.3292
python src/exp_b_valbank.py        # 3. Val-bank fold-in        -> +0.0104
python src/exp_a0_diagnostic.py    # 4. feasibility diagnostic  (Ghana oracle@20 ~0.39)
python src/exp_c_rerank.py         # 5. MPNet rerank            -> 0.3572  (downloads MPNet)
python src/exp_d_hybrid.py         # 6. Ghana hybrid (best)     -> 0.3701
python src/exp_e_voting.py         # 7. voting (rejected)       -> worse, documented
# 8. mT5 fine-tune (full + LoRA) -> generator_outputs/ (GPU; see the notebook)
python src/build_best_experiments.py   # rebuild reports/best_experiments.html + notebook section
```
The captured outputs live in `reports/experiment_logs/`; the rendered report is
`reports/best_experiments.html`.

## Results
| Stage | Local portion (/0.74) | Public |
|---|---:|---:|
| Baseline (char_wb retrieval) | 0.3150 | — |
| + routing + global bank | 0.3292 | — |
| + MPNet rerank | 0.3572 | 0.579 |
| **+ Ghana hybrid (best)** | **0.3701** | **0.598** |

See `report document` for the 8 headline experiments (with live run output and
figures), `EXPERIMENTS.md` for the full log of all ~14 experiments, and the academic report (PDF,
submitted separately) for the complete analysis and discussion.

## Reproducibility & compliance
Seed fixed (=42) everywhere; pinned `requirements.lock.txt`; deterministic submission generation.
Only open-licensed packages/models (Apache-2.0 / MIT); challenge data is CC-BY-SA 4.0. No paid
services, no AutoML, no private data. A tooling/assistance disclosure is provided at the end of the report.
