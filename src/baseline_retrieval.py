"""
Baseline 1 — per-language char-n-gram TF-IDF nearest-question retrieval.
Ported from proj/baseline_retrieval.py with paths fixed to Data/ and seed locked
via qa_common. This is our floor: reproduce ~0.3150 portion on local Val.

Idea: answers are heavily templated, so for each query question we return the
answer of the most cosine-similar TRAIN question (within the same language).
"""
from __future__ import annotations
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from qa_common import set_seed, load_data, score_report, write_submission, ROOT

set_seed()

def retrieve(train_df, queries):
    """Fit char_wb 2-5 TF-IDF on train questions; return nearest train answer per query."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    Xtr = vec.fit_transform(train_df.input.astype(str))
    Xq = vec.transform(queries.astype(str))
    sims = linear_kernel(Xq, Xtr)            # cosine (tfidf is L2-normed)
    nn = sims.argmax(axis=1)
    return train_df.output.values[nn]

def predict_all(train_df, target_df):
    """Predict answers for target_df, fitting a separate model per language."""
    preds = np.empty(len(target_df), dtype=object)
    for lang in target_df.subset.unique():
        trL = train_df[train_df.subset == lang]
        mask = (target_df.subset == lang).values
        preds[mask] = retrieve(trL, target_df.loc[mask, "input"])
    return preds

if __name__ == "__main__":
    tr, va, te = load_data()

    # 1) local Val scoreboard (retrieve from TRAIN only — no leakage)
    val_preds = predict_all(tr, va)
    score_report(va, val_preds, label="baseline_char_wb_2_5")

    # 2) Test submission in the exact 4-column format
    test_preds = predict_all(tr, te)
    write_submission(te, test_preds, os.path.join(ROOT, "submissions", "submission_baseline.csv"))
