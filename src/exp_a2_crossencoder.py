"""
Phase A2 (probe) — can a cross-encoder reranker capture the strong-config oracle headroom?
A0 showed Eng_Uga oracle@20=0.886 but our bi-encoder rerank only reaches ~0.69.
Test a LIGHT multilingual cross-encoder scoring (query_question, candidate_ANSWER) over
top-20 TF-IDF candidates. RAM-safe: small model (~470MB), 300-row slice per subset.

Compares per slice:
  tfidf_top1   : nearest TF-IDF answer (no rerank)
  bi_q         : current bi-encoder pick (question-question) — approximated by oracle of our pick? no:
                 we just report tfidf_top1 vs cross-encoder; bi-encoder full pipeline already logged.
  cross_enc    : cross-encoder pick by (query, candidate answer)
  oracle@20    : ceiling
"""
from __future__ import annotations
import numpy as np
from qa_common import set_seed, load_data, rouge_portion
from retrieval import predict

set_seed()
tr, va, te = load_data()
CE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilingual, ~470MB, open
N = 300

def mean1(ref, pred):
    return rouge_portion([ref], [pred])[3]

from sentence_transformers import CrossEncoder
import torch
dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[A2] cross-encoder={CE_MODEL} device={dev}")
ce = CrossEncoder(CE_MODEL, device=dev, max_length=512)

for subset in ("Eng_Uga", "Aka_Gha"):
    sl = va[va.subset == subset].head(N)
    preds, cand = predict(tr, sl, routing=True, bank_scope="global", k=20, return_candidates=True)
    refs = sl.output.values
    qs = sl.input.astype(str).values
    t1, cer, orc = [], [], []
    for j in range(len(sl)):
        cands = cand["answers"][j]
        t1.append(mean1(refs[j], cands[0]))
        orc.append(max(mean1(refs[j], c) for c in cands))
        scores = ce.predict([(qs[j], str(c)[:1000]) for c in cands])
        cer.append(mean1(refs[j], cands[int(np.argmax(scores))]))
    print(f"{subset:8s} (n={len(sl)})  tfidf_top1={np.mean(t1):.3f}  cross_enc={np.mean(cer):.3f}  oracle@20={np.mean(orc):.3f}")
