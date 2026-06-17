"""
Build submission_v3.csv — best Phase-2 config: MPNet rerank (Idea C) + Val-bank (Idea B).
Val eval (honest, Train-only bank) printed for the record; Test bank = Train + Val.
"""
from __future__ import annotations
import os
import pandas as pd
from qa_common import set_seed, load_data, score_report, write_submission, ROOT
from rerank import predict_rerank, _get_encoder

set_seed()
tr, va, te = load_data()
model, _ = _get_encoder()

# 1) honest Val score (eval bank = Train only)
val_preds = predict_rerank(tr, va, k=20, model=model)
score_report(va, val_preds, label="v3: MPNet rerank (Train-only eval)")

# 2) Test submission: bank = Train + Val
bank = pd.concat([tr, va], ignore_index=True)
test_preds = predict_rerank(bank, te, k=20, model=model)
write_submission(te, test_preds, os.path.join(ROOT, "submissions", "submission_v3.csv"))
