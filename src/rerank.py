"""
rerank.py — top-k TF-IDF recall → multilingual sentence-embedding rerank (Idea C).

For each query we take the top-k TF-IDF candidates (global bank, per-subset routing),
then re-score them by semantic similarity of query-question ↔ candidate-question using
an open multilingual encoder. Final score blends the two per subset:
    score = (1 - w) * tfidf_cos + w * semantic_cos
w per subset (Phase-1 table). w=1.0 → pure semantic ranking among TF-IDF candidates.

Encoder: sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (Apache-2.0, open).
Efficiency: encode the whole bank's questions ONCE and look up candidate embeddings.
"""
from __future__ import annotations
import numpy as np
from retrieval import _topk_for_subset, cfg_for, DEFAULT_CFG

SEM_W = {"Aka_Gha": 0.5, "Amh_Eth": 0.2, "Eng_Eth": 0.4, "Eng_Gha": 0.6,
         "Eng_Ken": 1.0, "Eng_Uga": 1.0, "Lug_Uga": 0.1, "Swa_Ken": 0.5}
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def _get_encoder():
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rerank] encoder={MODEL_NAME} device={dev}")
    return SentenceTransformer(MODEL_NAME, device=dev), dev

def _encode_unique(model, texts, batch_size=64):
    """Encode unique strings once; return (lookup dict text->row, matrix)."""
    uniq = list(dict.fromkeys(texts))
    emb = model.encode(uniq, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    return {t: i for i, t in enumerate(uniq)}, emb

def predict_rerank(bank_df, query_df, k=20, routing=True, sem_w=None,
                   strip_accents=None, nfc=True, model=None):
    """Rerank top-k TF-IDF candidates by semantic similarity. Returns preds (object array)."""
    sem_w = SEM_W if sem_w is None else sem_w
    if model is None:
        model, _ = _get_encoder()

    # encode every bank question + query question once
    bank_q = bank_df.input.astype(str).tolist()
    q_q = query_df.input.astype(str).tolist()
    bank_lut, bank_emb = _encode_unique(model, bank_q)
    q_lut, q_emb = _encode_unique(model, q_q)
    bank_ans = bank_df.output.values

    preds = np.empty(len(query_df), dtype=object)
    qpos = {idx: i for i, idx in enumerate(query_df.index)}

    for subset in query_df.subset.unique():
        cfg = cfg_for(subset) if routing else dict(DEFAULT_CFG)
        w = float(sem_w.get(subset, 0.4))
        qmask = query_df.subset == subset
        q = query_df.loc[qmask]
        idx, tfidf_sim = _topk_for_subset(bank_df, q.input, cfg, k, strip_accents, nfc)  # global bank
        positions = [qpos[ix] for ix in q.index]
        q_texts = q.input.astype(str).tolist()
        for j, pos in enumerate(positions):
            cand_pos = idx[j]                                   # bank rows
            qe = q_emb[q_lut[q_texts[j]]]
            ce = bank_emb[[bank_lut[bank_q[c]] for c in cand_pos]]
            sem = ce @ qe                                       # cosine (normalized)
            blend = (1 - w) * tfidf_sim[j] + w * sem
            preds[pos] = bank_ans[cand_pos[int(blend.argmax())]]
    return preds
