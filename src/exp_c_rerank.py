"""
Phase 2 / Idea C — MPNet semantic rerank of top-k TF-IDF candidates.
Val eval, bank = TRAIN only (no leak). Compare to A2 retrieval-only = 0.3292.
"""
from __future__ import annotations
import time
from qa_common import set_seed, load_data, score_report
from rerank import predict_rerank, _get_encoder

set_seed()
tr, va, te = load_data()

model, _ = _get_encoder()
t = time.time()
preds = predict_rerank(tr, va, k=20, model=model)
df = score_report(va, preds, label="C: MPNet rerank (k=20, per-subset w) vs A2=0.3292")
print(f"[elapsed {time.time()-t:.0f}s]")
