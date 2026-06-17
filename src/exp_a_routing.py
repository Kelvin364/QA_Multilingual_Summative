"""
Phase 2 / Idea A — per-subset TF-IDF routing.
Val-gated matrix over: routing on/off, bank scope (per_subset vs global), accent strip.
Eval bank = TRAIN ONLY (no leakage). Prints overall + weak-subset deltas vs 0.3150.
"""
from __future__ import annotations
import time
from qa_common import set_seed, load_data, rouge_portion
from retrieval import predict

set_seed()
tr, va, te = load_data()
BASE = 0.3150

def run(label, **kw):
    t = time.time()
    preds = predict(tr, va, **kw)
    r1, rl, portion, mean = rouge_portion(va.output.values, preds)
    # weak-subset portions
    weak = {}
    for s in ("Aka_Gha", "Eng_Gha"):
        m = (va.subset == s).values
        _, _, p, _ = rouge_portion(va.output.values[m], preds[m])
        weak[s] = p
    flag = "BEATS" if portion > BASE else "below"
    print(f"{label:52s} portion={portion:.4f} ({flag} {BASE})  mean={mean:.4f}  "
          f"Aka={weak['Aka_Gha']:.4f} EngGha={weak['Eng_Gha']:.4f}  [{time.time()-t:.0f}s]")
    return portion, preds

print("baseline reference: portion=0.3150  Aka=0.2237 EngGha=0.1864\n")
# control: default cfg, per-subset bank (≈ baseline but raw-char instead of char_wb)
run("ctrl  default-cfg  per_subset  no-strip", routing=False, bank_scope="per_subset")
# Idea A variants
run("A1    routing       per_subset  no-strip", routing=True, bank_scope="per_subset")
run("A2    routing       GLOBAL      no-strip", routing=True, bank_scope="global")
run("A3    routing       GLOBAL      strip-unicode", routing=True, bank_scope="global", strip_accents="unicode")
run("A4    routing       per_subset  strip-unicode", routing=True, bank_scope="per_subset", strip_accents="unicode")
