"""
retrieval.py — configurable TF-IDF retrieval with per-subset routing.

Re-implemented in our own style from the Phase-1 specs. Supports:
  - per-subset vectorizer routing (analyzer / ngram / min_df / sublinear_tf ...)
  - bank scope: "per_subset" (separate per language) vs "global" (all languages share
    one answer bank — the cross-language Ghana lever; reference found this helps)
  - text normalization toggle (NFC) and accent-stripping toggle
  - top-k retrieval (for later semantic rerank), nearest-1 by default

Leakage rule lives with the caller: for Val eval pass bank=Train only; for the Test
submission pass bank=Train+Val.
"""
from __future__ import annotations
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---- per-subset routing table (extracted in Phase 1; our starting prior) ----
DEFAULT_CFG = dict(analyzer="char", ngram_min=2, ngram_max=4, min_df=2,
                   max_features=100000, sublinear_tf=True)
ROUTING = {
    "Aka_Gha": dict(analyzer="char", ngram_min=1, ngram_max=4, min_df=5, sublinear_tf=True),
    "Eng_Gha": dict(analyzer="char", ngram_min=1, ngram_max=3, min_df=5, sublinear_tf=True),
    "Eng_Ken": dict(analyzer="char", ngram_min=1, ngram_max=3, min_df=2, sublinear_tf=True),
    "Eng_Uga": dict(analyzer="char", ngram_min=1, ngram_max=3, min_df=2, sublinear_tf=True),
    "Eng_Eth": dict(analyzer="char", ngram_min=2, ngram_max=5, min_df=2, sublinear_tf=True),
    "Lug_Uga": dict(analyzer="char", ngram_min=1, ngram_max=4, min_df=5, sublinear_tf=True),
    "Amh_Eth": dict(analyzer="char_wb", ngram_min=3, ngram_max=5, min_df=1, sublinear_tf=False),
    # Swa_Ken -> DEFAULT_CFG
}

def cfg_for(subset: str) -> dict:
    return {**DEFAULT_CFG, **ROUTING.get(subset, {})}

# ---- normalization -----------------------------------------------------
def make_normalizer(nfc: bool = True):
    """Return a preprocessor: collapse whitespace, optional Unicode NFC."""
    def norm(text: str) -> str:
        t = str(text)
        if nfc:
            t = unicodedata.normalize("NFC", t)
        return " ".join(t.split())
    return norm

def build_vectorizer(cfg: dict, strip_accents=None, nfc: bool = True) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=(cfg["ngram_min"], cfg["ngram_max"]),
        min_df=cfg.get("min_df", 1),
        max_df=cfg.get("max_df", 1.0),
        max_features=cfg.get("max_features", 100000),
        sublinear_tf=cfg.get("sublinear_tf", False),
        use_idf=True, smooth_idf=True, norm="l2", lowercase=True,
        strip_accents=strip_accents,                 # None | "unicode" | "ascii"
        preprocessor=make_normalizer(nfc),
    )

# ---- core retrieval ----------------------------------------------------
def _topk_for_subset(bank_df, query_inputs, cfg, k, strip_accents, nfc):
    """Return (topk_idx [nq,k] into bank_df rows, topk_sim [nq,k])."""
    vec = build_vectorizer(cfg, strip_accents=strip_accents, nfc=nfc)
    Xb = vec.fit_transform(bank_df.input.astype(str))
    Xq = vec.transform(query_inputs.astype(str))
    sims = linear_kernel(Xq, Xb)                     # cosine (L2-normed)
    if k == 1:
        idx = sims.argmax(axis=1)[:, None]
    else:
        kk = min(k, sims.shape[1])
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
        idx = np.take_along_axis(part, order, axis=1)
    topk_sim = np.take_along_axis(sims, idx, axis=1)
    return idx, topk_sim

def predict(bank_df, query_df, routing=True, bank_scope="global", k=1,
            strip_accents=None, nfc=True, return_candidates=False):
    """Nearest-answer retrieval with per-subset routing.

    bank_scope: "global" = retrieve over all-language bank; "per_subset" = restrict
    bank to the query's own subset. routing=False uses DEFAULT_CFG for every subset.
    Returns preds (object array). If return_candidates, also returns a dict with
    per-row top-k answers and sims (for reranking).
    """
    preds = np.empty(len(query_df), dtype=object)
    cand = {"answers": [None] * len(query_df), "sims": [None] * len(query_df)} if return_candidates else None
    qpos = {idx: i for i, idx in enumerate(query_df.index)}  # row index -> position

    for subset in query_df.subset.unique():
        cfg = cfg_for(subset) if routing else dict(DEFAULT_CFG)
        bank = bank_df if bank_scope == "global" else bank_df[bank_df.subset == subset]
        if len(bank) == 0:
            bank = bank_df
        qmask = query_df.subset == subset
        q = query_df.loc[qmask]
        idx, sim = _topk_for_subset(bank, q.input, cfg, k, strip_accents, nfc)
        bank_ans = bank.output.values
        nn1 = idx[:, 0]
        positions = [qpos[ix] for ix in q.index]
        for j, pos in enumerate(positions):
            preds[pos] = bank_ans[nn1[j]]
            if return_candidates:
                cand["answers"][pos] = [bank_ans[c] for c in idx[j]]
                cand["sims"][pos] = sim[j].tolist()
    if return_candidates:
        return preds, cand
    return preds
