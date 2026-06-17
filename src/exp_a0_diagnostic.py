"""
Phase A0 — feasibility diagnostic: WHY are the weak configs low, and can generation help?
For each subset we report:
  - dup_rate      : unique answers / rows (low repeat => templated => retrieval-friendly)
  - top1_sim      : mean cosine of the chosen (nearest) train question (retrieval confidence)
  - cur_mean      : current rerank+hybrid mean-ROUGE (from v4 logic, approx via top-1)
  - oracle@20     : best mean-ROUGE achievable among top-20 TF-IDF candidates (retrieval ceiling)
Interpretation:
  oracle@20 HIGH but cur LOW  -> better ranking/selection wins (cheap).
  oracle@20 LOW               -> right answer not in candidate pool -> needs generation OR
                                 answers aren't question-determined (0.70 likely infeasible).
Memory-lean: one subset at a time, Train-only bank, gc between. No neural model loaded.
"""
from __future__ import annotations
import gc
import numpy as np
from qa_common import set_seed, load_data, rouge_portion
from retrieval import _topk_for_subset, cfg_for

set_seed()
tr, va, te = load_data()

def mean_rouge_one(ref, pred):
    _, _, _, m = rouge_portion([ref], [pred])
    return m

K = 20
print(f"{'subset':9s} {'n':>5s} {'dup_rate':>8s} {'top1_sim':>8s} {'cur_mean':>8s} {'oracle@20':>9s} {'gap':>6s}")
for subset in ("Aka_Gha", "Eng_Gha", "Amh_Eth", "Lug_Uga", "Eng_Uga"):
    trL = tr[tr.subset == subset]
    vaL = va[va.subset == subset]
    dup = trL.output.nunique() / len(trL)
    cfg = cfg_for(subset)
    # global bank = full train (matches our config); retrieve top-K
    idx, sim = _topk_for_subset(tr, vaL.input, cfg, K, None, True)
    bank_ans = tr.output.values
    refs = vaL.output.values
    cur, orc, top1 = [], [], []
    for j in range(len(vaL)):
        cands = [bank_ans[c] for c in idx[j]]
        cur.append(mean_rouge_one(refs[j], cands[0]))
        orc.append(max(mean_rouge_one(refs[j], c) for c in cands))
        top1.append(sim[j][0])
    cur_m, orc_m, top1_m = np.mean(cur), np.mean(orc), np.mean(top1)
    print(f"{subset:9s} {len(vaL):5d} {dup:8.3f} {top1_m:8.3f} {cur_m:8.3f} {orc_m:9.3f} {orc_m-cur_m:6.3f}")
    del idx, sim, cur, orc, top1; gc.collect()

print("\nLegend: gap = oracle@20 - cur_mean (recoverable by better selection alone).")
print("If oracle@20 for Ghana is HIGH -> rerank/selection can win (no GPU needed).")
print("If oracle@20 for Ghana is LOW  -> answer not retrievable -> generation needed / or infeasible.")
