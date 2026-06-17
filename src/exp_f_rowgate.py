"""
Phase F — ROW-LEVEL gate: use generation only where retrieval is weak (low top-1 similarity).
Subset-average said generation loses, but per-row, generation may win where retrieval has no good
match. We sweep a retrieval-confidence threshold on Ghana Val and check if the gated combo beats
pure retrieval. Uses existing data (rerank cache + mt5small_full val_gen) — no retraining.

For each Ghana subset:
  retrieval_pred = rerank+hybrid (cache);  gen_pred = mt5small_full;  conf = tfidf top-1 cosine
  rule: use gen if conf < t else retrieval.  Report best t vs pure retrieval, and row-oracle ceiling.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from qa_common import set_seed, load_data, rouge_portion, ROOT
from retrieval import _topk_for_subset, cfg_for
from hybrid import apply_hybrid

set_seed()
tr, va, te = load_data()

def m1(ref, pred):
    return rouge_portion([ref], [pred])[3]

# retrieval side (v5): rerank base (cache) + hybrid, aligned to va order
cache = pd.read_csv(os.path.join(ROOT, "outputs", "rerank_val_preds.csv"))["pred"].astype(str).values
val_retr = apply_hybrid(va, cache)
# generation side
vg = pd.read_csv(os.path.join(ROOT, "generator_outputs", "mt5small_full", "val_gen.csv"))
gen_map = dict(zip(vg.ID, vg.gen_pred.astype(str)))

for sub in ["Aka_Gha", "Eng_Gha"]:
    m = (va.subset == sub).values
    idx = np.where(m)[0]
    vaL = va[m]
    # tfidf top-1 confidence per row (global bank, routed cfg)
    _, sim = _topk_for_subset(tr, vaL.input, cfg_for(sub), 1, None, True)
    conf = sim[:, 0]
    refs = va.output.values
    retr = np.array([m1(refs[i], val_retr[i]) for i in idx])
    gen = np.array([m1(refs[i], gen_map.get(va.ID.iloc[i], "")) for i in idx])

    pure = retr.mean()
    oracle = np.maximum(retr, gen).mean()
    print(f"\n=== {sub}  n={m.sum()} ===")
    print(f"  pure retrieval      : {pure:.4f}")
    print(f"  pure generation     : {gen.mean():.4f}")
    print(f"  row-oracle(max)     : {oracle:.4f}   (ceiling if we always picked the better one)")
    # sweep confidence threshold
    best = (pure, None)
    for t in np.quantile(conf, np.linspace(0.05, 0.95, 19)):
        gated = np.where(conf < t, gen, retr).mean()
        if gated > best[0]:
            best = (gated, t)
    if best[1] is not None:
        frac = (conf < best[1]).mean()
        print(f"  best gated combo    : {best[0]:.4f}  (use gen when conf<{best[1]:.3f}, ~{frac*100:.0f}% of rows)  Δ={best[0]-pure:+.4f}")
    else:
        print(f"  best gated combo    : no threshold beats pure retrieval")
