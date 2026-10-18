#!/usr/bin/env python
"""Step 3: visual QC of exported datasets -> ../qc/*.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

FS = 250
root = Path(__file__).resolve().parents[1]
ds, out = root / "dataset", root / "qc"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

X, M = {}, {}
for sp in ["train", "val", "test"]:
    X[sp] = np.load(ds / "single" / f"{sp}_X.npy")
    M[sp] = pd.read_csv(ds / "single" / f"{sp}_meta.csv")
t = np.arange(2500) / FS


def pick(cond, n):
    items = [(sp, i) for sp in M for i in np.flatnonzero(cond(M[sp]).to_numpy())]
    if not items:
        return []
    idx = rng.choice(len(items), size=min(n, len(items)), replace=False)
    return [items[j] for j in idx]


def plot_grid(items, title, fn):
    if not items:
        return
    fig, axs = plt.subplots(len(items), 2, figsize=(14, 2.1 * len(items)), squeeze=False, sharex=True)
    for r, (sp, i) in enumerate(items):
        m, x = M[sp].iloc[i], X[sp][i]
        names = m.sig_names.split(",")
        for c in range(2):
            axs[r, c].plot(t, x[c], lw=0.7)
            axs[r, c].set_ylabel(f"{names[c]} (mV)", fontsize=7)
            axs[r, c].tick_params(labelsize=7)
            if c == 0 and r == 0:
                pass
        axs[r, 0].set_title(f"{m.clip_id} {m.subtype} {m.record} {sp} arr={m.with_arrhythmia} "
                            f"src={m.clip_source if isinstance(m.clip_source, str) else ''} "
                            f"t0={m.start_sample / FS:.0f}s", fontsize=7, loc="left")
        if m.is_pos and m.subtype.startswith("pos"):
            for c in range(2):
                axs[r, c].axvline(5, color="r", ls=":", lw=0.8)   # window centre = AST peak
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / fn, dpi=110)
    plt.close(fig)


plot_grid(pick(lambda d: (d.is_pos == 1) & (d.clip_source == "peak"), 5), "Positive (peak-centred, red = AST/AT peak)", "pos_peak.png")
plot_grid(pick(lambda d: d.clip_source == "fallback_window", 4), "Positive fallback windows", "pos_fallback.png")
plot_grid(pick(lambda d: d.subtype == "neg_posture", 4), "Negative: posture / axis shift", "neg_posture.png")
plot_grid(pick(lambda d: d.subtype == "neg_arrhythmia", 4), "Negative: arrhythmia", "neg_arrhythmia.png")
plot_grid(pick(lambda d: d.subtype == "neg_tachy", 3), "Negative: sinus tachycardia", "neg_tachy.png")
plot_grid(pick(lambda d: d.subtype == "neg_clean", 3), "Negative: clean sinus", "neg_clean.png")

# ---- paired: Previous (left) vs Current (right) ----
rows = []
for sp in ["train", "val", "test"]:
    pm = pd.read_csv(ds / "paired" / f"{sp}_meta.csv")
    cur, prev = np.load(ds / "paired" / f"{sp}_cur.npy"), np.load(ds / "paired" / f"{sp}_prev.npy")
    assert len(pm) == len(cur) == len(prev)
    # cross-check: paired Current must equal the single-snapshot clip with the same clip_id
    mm = M[sp].reset_index().set_index("clip_id")
    for j, cid in enumerate(pm.clip_id):
        assert np.array_equal(cur[j], X[sp][mm.loc[cid, "index"]]), "pair/current mismatch"
    for j in range(len(pm)):
        rows.append((sp, j, pm.iloc[j], cur[j], prev[j]))
sel = [rows[i] for i in rng.choice([k for k, r in enumerate(rows) if r[2].is_pos == 1], 4, replace=False)] + \
      [rows[i] for i in rng.choice([k for k, r in enumerate(rows) if r[2].is_pos == 0], 3, replace=False)]
fig, axs = plt.subplots(len(sel), 2, figsize=(14, 2.3 * len(sel)), sharex=True)
for r, (sp, j, m, cur, prev) in enumerate(sel):
    axs[r, 0].plot(t, prev[0], lw=0.7, color="gray"); axs[r, 0].plot(t, prev[1] - 3, lw=0.7, color="gray")
    axs[r, 1].plot(t, cur[0], lw=0.7); axs[r, 1].plot(t, cur[1] - 3, lw=0.7)
    axs[r, 0].set_title(f"PREVIOUS ({m.prev_type}, {m.prev_gap_s / 60:.1f} min before)  {m.record}", fontsize=7, loc="left")
    axs[r, 1].set_title(f"CURRENT {m.clip_id} {m.subtype}", fontsize=7, loc="left")
    for a in axs[r]:
        a.tick_params(labelsize=7)
fig.suptitle("Paired samples (ch0 top, ch1 shifted -3 mV)")
fig.tight_layout()
fig.savefig(out / "paired.png", dpi=110)
plt.close(fig)

# ---- numeric sanity ----
allx = np.concatenate([X[s] for s in X])
print("amplitude mV  min/median-std/max:", allx.min(), np.median(allx.std(axis=2)), allx.max())
print("clips with |value|>10 mV:", int((np.abs(allx) > 10).any(axis=(1, 2)).sum()))
print("figures written to", out)
