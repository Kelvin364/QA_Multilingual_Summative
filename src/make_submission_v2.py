"""
Build submission_v2.csv — best Phase-2 retrieval config.
Config: per-subset routing + GLOBAL bank + NFC normalization (Idea A),
        Test bank = Train + Val (Idea B).  Seed fixed.
Val eval (honest, Train-only bank) is printed for the record.
"""
from __future__ import annotations
import os
import pandas as pd
from qa_common import set_seed, load_data, score_report, write_submission, ROOT
from retrieval import predict

set_seed()
tr, va, te = load_data()

# 1) Honest Val score: eval bank = TRAIN only (no leak)
val_preds = predict(tr, va, routing=True, bank_scope="global")
score_report(va, val_preds, label="v2: routing + global (Train-only eval)")

# 2) Test submission: bank = Train + Val (Idea B)
bank = pd.concat([tr, va], ignore_index=True)
test_preds = predict(bank, te, routing=True, bank_scope="global")
write_submission(te, test_preds, os.path.join(ROOT, "submissions", "submission_v2.csv"))
