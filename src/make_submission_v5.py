"""
make_submission_v5.py — merge fine-tuned generation into our best retrieval submission (v4),
gated per subset on Val. We REPLACE a weak-config answer with generation ONLY if generation
beats retrieval on that subset's Val mean-ROUGE. Never a blind replacement. Seed fixed.

Expects Colab outputs in outputs/generator/{val_gen.csv,test_gen.csv}.
Reuses cached rerank Val preds (outputs/rerank_val_preds.csv) + hybrid for the retrieval side.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from qa_common import set_seed, load_data, rouge_portion, score_report, write_submission, ROOT
from hybrid import apply_hybrid

_ap = argparse.ArgumentParser()
_ap.add_argument("--gen-dir", default=os.path.join(ROOT, "outputs", "generator"),
                 help="dir with the chosen run's val_gen.csv & test_gen.csv")
_args = _ap.parse_args()

set_seed()
tr, va, te = load_data()
GEN_DIR = _args.gen_dir
val_gen = pd.read_csv(os.path.join(GEN_DIR, "val_gen.csv"))
test_gen = pd.read_csv(os.path.join(GEN_DIR, "test_gen.csv"))

# --- retrieval side (v4): rerank base + Ghana hybrid, Val ---
cache = os.path.join(ROOT, "outputs", "rerank_val_preds.csv")
assert os.path.exists(cache), "run exp_d_hybrid.py first to cache rerank Val preds"
val_base = pd.read_csv(cache)["pred"].astype(str).values
val_retr = apply_hybrid(va, val_base)            # retrieval+hybrid per row (Val order = va order)

def mean_rouge_mask(mask, preds):
    return rouge_portion(va.output.values[mask], np.asarray(preds, dtype=object)[mask])[3]

# --- per-subset gate: does generation beat retrieval on Val? ---
gen_val_map = dict(zip(val_gen.ID, val_gen.gen_pred.astype(str)))
keep_gen = {}
print("Per-subset Val gate (generation vs retrieval mean-ROUGE):")
for s in val_gen.subset.unique():
    m = (va.subset == s).values
    retr_m = mean_rouge_mask(m, val_retr)
    gen_preds_full = np.array([gen_val_map.get(i, "") for i in va.ID], dtype=object)
    gen_m = mean_rouge_mask(m, gen_preds_full)
    win = gen_m > retr_m
    keep_gen[s] = win
    print(f"  {s}: retrieval={retr_m:.4f}  generation={gen_m:.4f}  -> use {'GENERATION' if win else 'retrieval'}")

# --- build gated Val preds + report ---
val_final = val_retr.copy()
for i, sid in enumerate(va.ID):
    s = va.subset.iloc[i]
    if keep_gen.get(s) and sid in gen_val_map:
        val_final[i] = gen_val_map[sid]
score_report(va, val_final, label="v5: retrieval+hybrid with Val-gated generation")

# --- apply same gate to Test, build submission ---
# Test retrieval side = v4 submission (already written); load it
v4 = pd.read_csv(os.path.join(ROOT, "submissions", "submission_v4.csv"))
test_pred_map = dict(zip(v4.ID, v4.TargetLLM.astype(str)))
gen_test_map = dict(zip(test_gen.ID, test_gen.gen_pred.astype(str)))
sub_te = te.copy()
final_test = []
for sid, s in zip(sub_te.ID, sub_te.subset):
    if keep_gen.get(s) and sid in gen_test_map:
        final_test.append(gen_test_map[sid])
    else:
        final_test.append(test_pred_map.get(sid, ""))
write_submission(sub_te, final_test, os.path.join(ROOT, "submissions", "submission_v5.csv"))
