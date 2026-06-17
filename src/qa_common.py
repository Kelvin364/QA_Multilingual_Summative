"""
qa_common.py — shared, reproducible utilities for the HASH Multilingual Health QA project.
All experiments import from here so data loading, the local ROUGE scoreboard, the
submission writer, and the seed are identical everywhere.

The local scoreboard mimics Zindi's metric for the 74% lexical portion:
    rouge_portion = 0.37 * ROUGE-1 F1 + 0.37 * ROUGE-L F1     (max 0.74)
We cannot compute the 0.26 LLM-judge term locally, so we optimise the part we can
measure and report R1 / RL / portion. We also report the unweighted mean ROUGE
(= (R1+RL)/2) so numbers are directly comparable to other write-ups.

Only open-source packages (pandas, scikit-learn, numpy, rouge-score). Seed fixed.
"""
from __future__ import annotations
import os
import re
import random
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Fix every RNG we might touch so a re-run reproduces the score exactly."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional (not needed for pure retrieval)
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

# ---- paths -------------------------------------------------------------
# Resolve relative to this file so scripts run from any working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)                       # QA_Summative_ML/
DATA_DIR = os.path.join(ROOT, "Data")
SAMPLE_SUB = os.path.join(ROOT, "SampleSubmission.csv")

def load_data():
    """Return (train, val, test) DataFrames. Columns: ID, input, output[, subset]."""
    tr = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    va = pd.read_csv(os.path.join(DATA_DIR, "Val.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    return tr, va, te

# ---- local scoreboard (mimics Zindi's ROUGE metric) -------------------
_scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

def rouge_portion(refs, preds):
    """Return (r1, rl, portion, mean_rouge) averaged over all rows."""
    r1 = rl = 0.0
    for ref, pred in zip(refs, preds):
        s = _scorer.score(str(ref), str(pred))
        r1 += s["rouge1"].fmeasure
        rl += s["rougeL"].fmeasure
    n = len(refs)
    r1 /= n; rl /= n
    portion = 0.37 * r1 + 0.37 * rl     # = 74% of the leaderboard score
    mean_rouge = 0.5 * (r1 + rl)        # comparable to other write-ups
    return r1, rl, portion, mean_rouge

def score_report(val_df, preds, label="model"):
    """Print overall + per-subset scoreboard and return a tidy DataFrame."""
    refs = val_df.output.values
    r1, rl, portion, mean = rouge_portion(refs, preds)
    print(f"\n=== {label} : LOCAL VAL ===")
    print(f"OVERALL  R1={r1:.4f}  RL={rl:.4f}  portion(/0.74)={portion:.4f}  mean_rouge={mean:.4f}")
    rows = [{"subset": "OVERALL", "n": len(refs), "R1": r1, "RL": rl,
             "portion": portion, "mean_rouge": mean}]
    preds = np.asarray(preds, dtype=object)
    for sub in sorted(val_df.subset.unique()):
        m = (val_df.subset == sub).values
        sr1, srl, sp, sm = rouge_portion(refs[m], preds[m])
        rows.append({"subset": sub, "n": int(m.sum()), "R1": sr1, "RL": srl,
                     "portion": sp, "mean_rouge": sm})
        print(f"  {sub:8s} n={int(m.sum()):5d}  R1={sr1:.4f}  RL={srl:.4f}  portion={sp:.4f}")
    return pd.DataFrame(rows)

# ---- submission writer (exact Zindi 4-column format) ------------------
_WS = re.compile(r"\s+")

def clean_answer(text) -> str:
    """Collapse all whitespace (incl. embedded \\r / \\n) to single spaces.

    Some training answers contain carriage returns/newlines that fragment the CSV
    on re-read (and would break Zindi's parser). Single-line answers are robust for
    any parser and do not change ROUGE, which is token-based.
    """
    if text is None:
        return ""
    return _WS.sub(" ", str(text)).strip()

def write_submission(test_df, preds, path):
    """Write ID, TargetRLF1, TargetR1F1, TargetLLM with identical text per row."""
    preds = [clean_answer(p) for p in preds]
    sub = pd.DataFrame({
        "ID": test_df.ID.values,
        "TargetRLF1": preds,
        "TargetR1F1": preds,
        "TargetLLM": preds,
    })
    assert len(sub) == len(test_df), "row count mismatch"
    assert list(sub.columns) == ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]
    sub.to_csv(path, index=False)
    print(f"Wrote {path}  ({len(sub)} rows)")
    return sub
