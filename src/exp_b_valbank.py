"""
Phase 2 / Idea B — estimate the gain from folding Val into the retrieval bank.
Clean simulation (no leak): hold out Val_B; compare scoring Val_B when the bank is
Train vs Train+Val_A. Val_B is never in any bank. Real Test gain is >= this (we add
all 6,686 Val rows for Test, vs only ~3,343 Val_A here). Config = best A2.
"""
from __future__ import annotations
import pandas as pd
from qa_common import set_seed, load_data, rouge_portion
from retrieval import predict

set_seed()
tr, va, te = load_data()

va_a = va.sample(frac=0.5, random_state=42)
va_b = va.drop(va_a.index)
bank_plus = pd.concat([tr, va_a], ignore_index=True)

def score(bank, label):
    preds = predict(bank, va_b, routing=True, bank_scope="global")
    r1, rl, portion, mean = rouge_portion(va_b.output.values, preds)
    print(f"{label:34s} portion={portion:.4f}  mean={mean:.4f}")
    return portion

print(f"held-out Val_B n={len(va_b)}  (bank+ adds Val_A n={len(va_a)})\n")
p0 = score(tr, "bank = Train only")
p1 = score(bank_plus, "bank = Train + Val_A")
print(f"\nDelta from adding Val to bank: {p1 - p0:+.4f}  (Test gain >= this)")
