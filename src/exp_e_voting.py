"""
Phase E — capture strong-config oracle headroom via better candidate SELECTION.
Lean version: compare selection strategies over the TF-IDF top-k pool (no slow oracle).
Bar = our current v5 rerank+hybrid per-subset mean-ROUGE.

  tfidf_top1   : nearest single candidate
  vote         : most-frequent answer among top-k (tie-break summed similarity)
  vote_simwt   : answer with max summed similarity across its top-k occurrences
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from qa_common import set_seed, load_data, rouge_portion
from retrieval import predict

set_seed()
tr, va, te = load_data()
K = 30

# current v5 (rerank+hybrid) per-subset mean-ROUGE, to beat
CURRENT = {"Eng_Uga": 0.690, "Lug_Uga": 0.548, "Swa_Ken": 0.638, "Eng_Ken": 0.738,
           "Eng_Eth": 0.566, "Aka_Gha": 0.356, "Eng_Gha": 0.332}

def m1(ref, pred):
    return rouge_portion([ref], [pred])[3]

_, cand = predict(tr, va, routing=True, bank_scope="global", k=K, return_candidates=True)
refs = va.output.values
subs = va.subset.values

def select(strategy, answers, sims):
    if strategy == "tfidf_top1":
        return answers[0]
    freq = defaultdict(float); cnt = defaultdict(int)
    for a, s in zip(answers, sims):
        cnt[a] += 1; freq[a] += s
    if strategy == "vote":
        return max(answers, key=lambda a: (cnt[a], freq[a]))
    if strategy == "vote_simwt":
        return max(freq, key=lambda a: freq[a])

strategies = ["tfidf_top1", "vote", "vote_simwt"]
print(f"{'subset':9s} {'current':>8s} " + " ".join(f"{s:>11s}" for s in strategies))
agg = {s: [] for s in strategies}
for sub in ["Eng_Uga", "Lug_Uga", "Swa_Ken", "Eng_Ken", "Eng_Eth", "Aka_Gha", "Eng_Gha"]:
    idx = np.where(subs == sub)[0]
    sc = {s: [] for s in strategies}
    for i in idx:
        a, s = cand["answers"][i], cand["sims"][i]
        for st in strategies:
            sc[st].append(m1(refs[i], select(st, a, s)))
    best = max(strategies, key=lambda st: np.mean(sc[st]))
    row = " ".join(f"{np.mean(sc[st]):11.4f}" + ("*" if st == best else " ") for st in strategies)
    print(f"{sub:9s} {CURRENT[sub]:8.3f} {row}")
    for st in strategies: agg[st].extend(sc[st])
print("-" * 60)
print(f"{'OVERALL':9s} {'0.500':>8s} " + " ".join(f"{np.mean(agg[st]):11.4f} " for st in strategies))
print("(* = best strategy for that subset; 'current' = v5 rerank+hybrid mean-ROUGE)")
