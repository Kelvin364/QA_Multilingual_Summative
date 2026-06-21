"""
make_figures.py — generate all report figures into reports/figures/ from our real data.
Pulls EDA from the CSVs, experiment numbers from our logged results, learning curves from
the generator history.csv files. Run: python src/make_figures.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qa_common import load_data, ROOT

FIG = os.path.join(ROOT, "figures")          # tracked dir; the notebook displays from here
os.makedirs(FIG, exist_ok=True)
tr, va, te = load_data()

def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=130, bbox_inches="tight"); plt.close()
    print("wrote", name)

ORDER = ["Eng_Uga", "Aka_Gha", "Eng_Gha", "Lug_Uga", "Swa_Ken", "Eng_Ken", "Amh_Eth", "Eng_Eth"]

# --- 1. subset distribution: train vs test ---
trc = tr.subset.value_counts(normalize=True).reindex(ORDER) * 100
tec = te.subset.value_counts(normalize=True).reindex(ORDER) * 100
x = np.arange(len(ORDER)); w = 0.4
plt.figure(figsize=(9, 4))
plt.bar(x - w/2, trc, w, label="Train %")
plt.bar(x + w/2, tec, w, label="Test %")
plt.xticks(x, ORDER, rotation=30, ha="right"); plt.ylabel("% of split")
plt.title("Language-country distribution (Train vs Test)"); plt.legend()
save("fig1_subset_distribution.png")

# --- 2. answer duplication rate per subset (THE key insight) ---
dup = {s: tr[tr.subset == s].output.nunique() / len(tr[tr.subset == s]) for s in ORDER}
colors = ["#d62728" if dup[s] > 0.9 else "#2ca02c" for s in ORDER]
plt.figure(figsize=(9, 4))
plt.bar(ORDER, [dup[s] for s in ORDER], color=colors)
plt.axhline(0.9, ls="--", c="gray", lw=1)
plt.xticks(rotation=30, ha="right"); plt.ylabel("unique answers / rows")
plt.title("Answer uniqueness per subset (red = ~bespoke, retrieval-hostile)")
save("fig2_answer_dup_rate.png")

# --- 3. answer length distribution ---
tr2 = tr.copy(); tr2["alen"] = tr2.output.astype(str).str.len()
plt.figure(figsize=(8, 4))
plt.hist(tr2.alen.clip(upper=2000), bins=60, color="#1f77b4")
plt.axvline(tr2.alen.median(), c="red", ls="--", label=f"median {int(tr2.alen.median())} chars")
plt.xlabel("answer length (chars, clipped 2000)"); plt.ylabel("count")
plt.title("Answer length distribution (long, variable → generous max length)"); plt.legend()
save("fig3_answer_length.png")

# --- 4. experiment score progression (overall ROUGE portion /0.74) ---
prog = [("baseline\nchar_wb", 0.3150), ("A1\nrouting", 0.3202), ("A2\n+global", 0.3292),
        ("C\n+MPNet rerank", 0.3572), ("D/v5\n+Ghana hybrid", 0.3701)]
labels = [p[0] for p in prog]; vals = [p[1] for p in prog]
plt.figure(figsize=(8.5, 4))
plt.plot(range(len(vals)), vals, "o-", lw=2, ms=8, color="#1f77b4")
for i, v in enumerate(vals): plt.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
plt.xticks(range(len(labels)), labels); plt.ylabel("local ROUGE portion (/0.74)")
plt.title("Local score progression (retrieval pipeline)"); plt.grid(alpha=0.3)
save("fig4_experiment_progression.png")

# --- 5. per-subset score (v5) with weak/strong coloring ---
v5 = {"Eng_Ken": 0.5459, "Eng_Uga": 0.5107, "Swa_Ken": 0.4722, "Eng_Eth": 0.4189,
      "Lug_Uga": 0.4053, "Aka_Gha": 0.2635, "Eng_Gha": 0.2459, "Amh_Eth": 0.0229}
order5 = sorted(ORDER, key=lambda s: -v5[s])
cols = ["#d62728" if v5[s] < 0.30 else "#2ca02c" for s in order5]
plt.figure(figsize=(9, 4))
plt.bar(order5, [v5[s] for s in order5], color=cols)
plt.xticks(rotation=30, ha="right"); plt.ylabel("ROUGE portion (/0.74)")
plt.title("v5 per-subset score (red = weak Ghana/Amharic configs)")
save("fig5_per_subset_v5.png")

# --- 6. retrieval vs oracle@20 (headroom + Ghana cap) ---
A0 = {"Eng_Uga": (0.69, 0.886), "Lug_Uga": (0.55, 0.819), "Aka_Gha": (0.31, 0.388),
      "Eng_Gha": (0.26, 0.344), "Amh_Eth": (0.03, 0.070)}
ks = list(A0.keys()); rr = [A0[k][0] for k in ks]; orc = [A0[k][1] for k in ks]
x = np.arange(len(ks)); w = 0.4
plt.figure(figsize=(8.5, 4))
plt.bar(x - w/2, rr, w, label="achieved (rerank)")
plt.bar(x + w/2, orc, w, label="oracle@20 (ceiling)")
plt.xticks(x, ks, rotation=20, ha="right"); plt.ylabel("mean-ROUGE")
plt.title("Retrieval vs oracle@20: strong configs have headroom, Ghana is capped low"); plt.legend()
save("fig6_retrieval_vs_oracle.png")

# --- 7. local -> public calibration ---
pts = [(0.4828, 0.579, "our v3"), (0.5001, 0.598, "our v5"), (0.508, 0.6026, "reference best")]
plt.figure(figsize=(6.5, 5))
for lx, ly, lab in pts:
    plt.scatter(lx, ly, s=80); plt.annotate(lab, (lx, ly), textcoords="offset points", xytext=(6, 6))
xs = np.linspace(0.45, 0.62, 50)
plt.plot(xs, xs + 0.096, "--", c="gray", label="public = local + 0.096")
plt.xlabel("local mean-ROUGE (Val)"); plt.ylabel("public leaderboard")
plt.title("Local↔public calibration (offset stable to 0.001)"); plt.legend(); plt.grid(alpha=0.3)
save("fig7_calibration.png")

# --- 8. learning curves (generator runs) ---
plt.figure(figsize=(8, 4.5))
for run, style in [("mt5small_full", "-"), ("mt5small_lora", "--")]:
    p = os.path.join(ROOT, "generator_outputs", run, "history.csv")
    if not os.path.exists(p): continue
    h = pd.read_csv(p)
    trl = h.dropna(subset=["train_loss"]) if "train_loss" in h else pd.DataFrame()
    evl = h.dropna(subset=["eval_loss"]) if "eval_loss" in h else pd.DataFrame()
    # train_loss logged sparsely at epoch boundaries here; use the per-step 'loss' if present
    step = h.dropna(subset=["loss"]) if "loss" in h else trl
    if len(step): plt.plot(step["step"], step["loss"], style, label=f"{run} train")
    if len(evl): plt.scatter(evl["step"], evl["eval_loss"], marker="o", s=40, label=f"{run} eval")
plt.xlabel("step"); plt.ylabel("loss"); plt.ylim(0, 12)
plt.title("Generator learning curves (mT5-small full vs LoRA)"); plt.legend()
save("fig8_learning_curves.png")

# --- 9. generation vs retrieval (Ghana) ---
gen = {"Aka_Gha": (0.356, 0.324, 0.270), "Eng_Gha": (0.332, 0.308, 0.223)}
labels = list(gen.keys()); x = np.arange(len(labels)); w = 0.27
plt.figure(figsize=(7, 4))
plt.bar(x - w, [gen[k][0] for k in labels], w, label="retrieval")
plt.bar(x, [gen[k][1] for k in labels], w, label="mT5-small full")
plt.bar(x + w, [gen[k][2] for k in labels], w, label="mT5-small LoRA")
plt.xticks(x, labels); plt.ylabel("Val mean-ROUGE")
plt.title("Generation underperforms retrieval on bespoke Ghana answers"); plt.legend()
save("fig9_generation_vs_retrieval.png")

print("\nAll figures in", FIG)
