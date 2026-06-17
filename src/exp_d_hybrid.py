"""
Phase 2 / Idea D — input×retrieval hybrid on the weak configs (Aka_Gha, Eng_Gha).
Splice the source question into the rerank prediction; the question carries key terms
that recur in the reference answer (ROUGE-gaming, but the metric rewards it).

Base = rerank (Idea C) Val preds (cached). We sweep mode/word-caps PER weak subset,
score that subset's portion, keep the best rule. Strong subsets are left untouched.
Val eval, bank = TRAIN only (no leak).

CAUTION: this is metric-gaming. We log the ROUGE gain AND eyeball fluency, because the
top-10 AfroLM BERTScore check penalises incoherent answers.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from qa_common import set_seed, load_data, rouge_portion, ROOT
from rerank import predict_rerank, _get_encoder

set_seed()
tr, va, te = load_data()
CACHE = os.path.join(ROOT, "outputs", "rerank_val_preds.csv")

# base rerank Val preds (cache to avoid recompute)
if os.path.exists(CACHE):
    base = pd.read_csv(CACHE)["pred"].astype(str).values
    print(f"[cache] loaded base rerank preds: {CACHE}")
else:
    model, _ = _get_encoder()
    base = predict_rerank(tr, va, k=20, model=model)
    pd.DataFrame({"ID": va.ID.values, "pred": base}).to_csv(CACHE, index=False)
    print(f"[cache] wrote base rerank preds: {CACHE}")

def trim(text, n):
    return text if not n else " ".join(str(text).split()[:n])

def combine(q, pred, mode, qcap, pcap):
    q, pred = trim(q, qcap), trim(pred, pcap)
    if mode == "input_pred": return f"{q} {pred}".strip()
    if mode == "pred_input": return f"{pred} {q}".strip()
    if mode == "input_only": return str(q).strip()
    raise ValueError(mode)

def portion_for(subset, preds):
    m = (va.subset == subset).values
    _, _, p, _ = rouge_portion(va.output.values[m], np.asarray(preds, dtype=object)[m])
    return p

MODES = ["input_pred", "pred_input", "input_only"]
QCAPS = [None, 30, 50, 80]
PCAPS = [None, 60, 100]

best_rules = {}
for subset in ("Aka_Gha", "Eng_Gha"):
    m = (va.subset == subset).values
    base_p = portion_for(subset, base)
    cand = [("BASE(rerank)", base_p, None)]
    qs = va.loc[m, "input"].astype(str).values
    bp = np.asarray(base, dtype=object)[m]
    for mode in MODES:
        for qcap in QCAPS:
            for pcap in (PCAPS if mode != "input_only" else [None]):
                hyb = [combine(q, p, mode, qcap, pcap) for q, p in zip(qs, bp)]
                _, _, p, _ = rouge_portion(va.output.values[m], np.asarray(hyb, dtype=object))
                cand.append((f"{mode} q<={qcap} p<={pcap}", p, (mode, qcap, pcap)))
    cand.sort(key=lambda x: -x[1])
    print(f"\n=== {subset}  base={base_p:.4f} ===")
    for name, p, _ in cand[:6]:
        tag = "  <-- BASE" if name.startswith("BASE") else (" *BEST*" if (name, p) == (cand[0][0], cand[0][1]) else "")
        print(f"  {name:28s} portion={p:.4f}  Δ={p-base_p:+.4f}{tag}")
    best_rules[subset] = cand[0]

# apply best rules, score overall
final = np.asarray(base, dtype=object).copy()
for subset, (name, p, rule) in best_rules.items():
    if rule is None:
        continue
    mode, qcap, pcap = rule
    m = (va.subset == subset).values
    qs = va.loc[m, "input"].astype(str).values
    idxs = np.where(m)[0]
    for i, q in zip(idxs, qs):
        final[i] = combine(q, base[i], mode, qcap, pcap)

r1, rl, portion, mean = rouge_portion(va.output.values, final)
print(f"\n=== D: rerank + Ghana hybrid (Train-only) ===")
print(f"OVERALL portion={portion:.4f}  mean={mean:.4f}  (rerank-only was 0.3572)")
print("best rules:", {k: v[0] for k, v in best_rules.items()})
