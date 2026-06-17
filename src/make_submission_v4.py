"""
Build submission_v4.csv — best retrieval-track config:
MPNet rerank (C) + Ghana input-hybrid (D) + Val-bank for Test (B).
Reuses cached Val rerank preds for the report; computes rerank for Test.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from qa_common import set_seed, load_data, score_report, write_submission, ROOT
from rerank import predict_rerank, _get_encoder
from hybrid import apply_hybrid

set_seed()
tr, va, te = load_data()
CACHE = os.path.join(ROOT, "outputs", "rerank_val_preds.csv")

# 1) Val report (honest, Train-only): reuse cached rerank base + hybrid
model = None
if os.path.exists(CACHE):
    val_base = pd.read_csv(CACHE)["pred"].astype(str).values
    print(f"[cache] loaded val rerank preds")
else:
    model, _ = _get_encoder()
    val_base = predict_rerank(tr, va, k=20, model=model)
val_final = apply_hybrid(va, val_base)
score_report(va, val_final, label="v4: rerank + Ghana hybrid (Train-only eval)")

# 2) Test: rerank over Train+Val bank, then hybrid
if model is None:
    model, _ = _get_encoder()
bank = pd.concat([tr, va], ignore_index=True)
test_base = predict_rerank(bank, te, k=20, model=model)
test_final = apply_hybrid(te, test_base)
write_submission(te, test_final, os.path.join(ROOT, "submissions", "submission_v4.csv"))
