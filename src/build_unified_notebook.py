"""
build_unified_notebook.py — generate notebooks/HASH_HealthQA.ipynb: ONE notebook covering the
entire challenge end-to-end (setup -> EDA -> retrieval -> fine-tuning -> gating -> submission ->
results). Cells are ENVIRONMENT-ADAPTIVE: on Colab they clone the repo, upload data and train on
GPU; run locally (repo + Data present, runs already cached) they skip those and reuse artifacts —
so the notebook is executable in both places. Run: python src/build_unified_notebook.py
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "notebooks", "HASH_HealthQA.ipynb")
REPO = "https://github.com/<your-username>/hash-multilingual-health-qa.git"

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
 md("# Multilingual Health QA in Low-Resource African Languages\n"
    "### Unified end-to-end notebook (EDA → retrieval → fine-tuning → submission)\n\n"
    "**Governing insight:** the metric is `0.37·ROUGE-1 + 0.37·ROUGE-L + 0.26·LLM-judge` → **74% is\n"
    "lexical overlap**, so the system is **retrieval-first**, with fine-tuning layered on the configs\n"
    "where retrieval is structurally capped (≈99%-unique answers). Best public score: **0.598**.\n\n"
    "Cells adapt to the environment: on **Colab** they clone the repo, upload data and train on GPU;\n"
    "run **locally** (repo + `Data/` present) they reuse cached artifacts. Use a GPU runtime on Colab."),

 md("## 1. Setup"),
 code("import os, sys\n"
      "IN_COLAB = 'google.colab' in sys.modules or os.path.isdir('/content')\n"
      "if not os.path.isdir('src') and IN_COLAB:\n"
      f"    !git clone {REPO}\n"
      "    %cd hash-multilingual-health-qa\n"
      "if os.path.isdir('src'):\n"
      "    sys.path.append('src')\n"
      "try:\n"
      "    import pandas, sklearn, rouge_score  # already present locally\n"
      "except ImportError:\n"
      "    !pip install -q -r requirements.txt\n"
      "print('setup complete | cwd:', os.getcwd())"),

 md("## 2. Data\n`Train.csv`, `Val.csv`, `Test.csv` (challenge files, not redistributed in the repo)."),
 code("import os\n"
      "os.makedirs('Data', exist_ok=True)\n"
      "have = all(os.path.exists(f'Data/{f}') for f in ['Train.csv', 'Val.csv', 'Test.csv'])\n"
      "if not have:\n"
      "    try:\n"
      "        from google.colab import files\n"
      "        up = files.upload()\n"
      "        for n in up:\n"
      "            if n.endswith('.csv') and n != 'SampleSubmission.csv':\n"
      "                os.replace(n, f'Data/{n}')\n"
      "    except Exception:\n"
      "        print('Place Train.csv, Val.csv, Test.csv in Data/')\n"
      "print('Data/:', sorted(os.listdir('Data')))"),

 md("## 3. Exploratory Data Analysis\nThe decisive variable is **answer uniqueness** per subset."),
 code("import pandas as pd\n"
      "tr = pd.read_csv('Data/Train.csv'); va = pd.read_csv('Data/Val.csv'); te = pd.read_csv('Data/Test.csv')\n"
      "print('shapes:', tr.shape, va.shape, te.shape)\n"
      "dup = (tr.groupby('subset').output.nunique() / tr.groupby('subset').size()).sort_values()\n"
      "print('\\nanswer uniqueness (unique/rows) per subset:'); print(dup.round(3))\n"
      "print('\\ntest mix %:'); print((te.subset.value_counts(normalize=True) * 100).round(1))"),
 code("!python src/make_figures.py\n"
      "from IPython.display import Image, display\n"
      "for f in ['fig1_subset_distribution.png', 'fig2_answer_dup_rate.png', 'fig3_answer_length.png']:\n"
      "    display(Image(f'reports/figures/{f}'))"),

 md("## 4. Retrieval pipeline\nBaseline → per-subset routing + global bank → MPNet rerank → Ghana hybrid (each Val-gated)."),
 code("# 4a. Baseline floor (CPU): per-language char_wb TF-IDF nearest-answer\n!python src/baseline_retrieval.py"),
 code("# 4b. Best retrieval submission: rerank + Ghana hybrid (+ Val-bank for Test)\n"
      "#     prints the honest Train-only Val scoreboard, writes submissions/submission_v4.csv\n"
      "!python src/make_submission_v4.py"),
 code("for f in ['fig4_experiment_progression.png', 'fig5_per_subset_v5.png',\n"
      "          'fig6_retrieval_vs_oracle.png', 'fig7_calibration.png']:\n"
      "    display(Image(f'reports/figures/{f}'))"),

 md("## 5. Fine-tuning (GPU) — mT5 on the Ghana configs\n"
    "Retrieval is capped on Ghana (oracle@20 ≈ 0.39). We fine-tune mT5-small (full + LoRA/PEFT).\n"
    "If a run already exists in `generator_outputs/` it is reused; otherwise it trains (GPU, ~15–25 min)."),
 code("import os\n"
      "if os.path.exists('generator_outputs/mt5small_full/val_gen.csv'):\n"
      "    import json\n"
      "    print('Reusing existing mt5small_full run:')\n"
      "    print(json.dumps(json.load(open('generator_outputs/mt5small_full/metrics.json'))['per_subset'], indent=2))\n"
      "else:\n"
      "    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
      "    !PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train_generator.py \\\n"
      "        --data-dir Data --output-dir generator_outputs --run-name mt5small_full \\\n"
      "        --model-name google/mt5-small --subsets Aka_Gha,Eng_Gha \\\n"
      "        --epochs 2 --train-bs 16 --grad-accum 1 --lr 1e-3 \\\n"
      "        --max-input 96 --max-target 160 --gen-len 160 --beams 2 --seed 42"),
 code("import os\n"
      "if os.path.exists('generator_outputs/mt5small_lora/val_gen.csv'):\n"
      "    print('Reusing existing mt5small_lora (LoRA/PEFT) run.')\n"
      "else:\n"
      "    !PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train_generator.py \\\n"
      "        --data-dir Data --output-dir generator_outputs --run-name mt5small_lora \\\n"
      "        --model-name google/mt5-small --subsets Aka_Gha,Eng_Gha --use-lora \\\n"
      "        --epochs 2 --train-bs 16 --grad-accum 1 --lr 5e-4 \\\n"
      "        --max-input 96 --max-target 160 --gen-len 160 --beams 2 --seed 42"),
 code("for f in ['fig8_learning_curves.png', 'fig9_generation_vs_retrieval.png']:\n"
      "    display(Image(f'reports/figures/{f}'))"),

 md("## 6. Gate + final submission\nReplace a Ghana answer with generation ONLY if it beats retrieval on that subset's Val."),
 code("!python src/make_submission_v5.py --gen-dir generator_outputs/mt5small_full\n"
      "import pandas as pd\n"
      "s = pd.read_csv('submissions/submission_v5.csv')\n"
      "print('final submission rows:', len(s), '| columns:', list(s.columns))"),

 md("## 7. Results\n"
    "| Stage | Local portion (/0.74) | Public |\n|---|---:|---:|\n"
    "| Baseline | 0.3150 | — |\n| + routing + global bank | 0.3292 | — |\n"
    "| + MPNet rerank | 0.3572 | 0.579 |\n| **+ Ghana hybrid (best)** | **0.3701** | **0.598** |\n\n"
    "Validated mapping **`public ≈ local + 0.096`**. The ceiling is set by answer uniqueness, not\n"
    "model quality — the central, evidence-backed finding. Full analysis, the 14 documented\n"
    "experiments, ethics and future work: `reports/REPORT.md` / `REPORT.html`."),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {OUT} ({len(cells)} cells)")
