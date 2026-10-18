#!/usr/bin/env python
"""Step 1: EDB annotation parsing, patient-level split, and clip-budget statistics.

Outputs (under ../outputs):
  records_summary.csv   per-record stats (patient_id, split, durations, window budgets)
  episodes.csv          every ischemic (ST/T uppercase) episode and axis-shift (lowercase) episode
  patient_split.csv     patient_id -> split, member records
  window_budget.csv     per-record candidate non-overlapping 10 s windows per class
  step1_report.md       human-readable summary
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

FS = 250
WIN_S = 10
BUFFER_S = 120
# e0129/e0133/e0161: patient identity ambiguous (same clinical profile) -> removed to avoid leakage
EXCLUDE_RECORDS = {"e0129", "e0133", "e0161"}
ECTOPIC = set("VSaJF")            # PVC, SVPB, aberrated APB, nodal, fusion
BEAT_SYMS = set("NLRBAaJSVrFejnE/fQ?|")  # '|' = isolated artefact; counted as non-normal
EPI_RE = re.compile(r"^(\()?(A)?(ST|T|st|t)([01])([+-]{1,2})(\d+)?(\))?$")


# ----------------------------------------------------------------------------
def parse_episodes(rec, ann, n_samples):
    """Return list of dicts for ischemic (upper) and axis-shift (lower) episodes.

    Edge cases (flagged in `incomplete`): episodes still open at record end, episodes whose
    begin precedes the excerpt (orphan end), and re-opened keys (previous closed at new begin).
    """
    open_ev, peaks, eps = {}, defaultdict(list), []

    def emit(typ, lead, d, b, s, inc):
        key = (typ, lead, d)
        pk = [p for p in peaks.pop(key, []) if b <= p[0] <= s]
        pk_s, pk_mag = (pk[0] if pk else ((b + s) // 2, None))
        eps.append(dict(record=rec, kind="ischemia" if typ.isupper() else "axis_shift",
                        etype=typ.upper(), lead=int(lead), direction=d,
                        start=b, end=s, peak=pk_s, peak_uv=pk_mag,
                        has_peak=bool(pk), incomplete=inc))

    order = np.argsort(ann.sample, kind="stable")
    for i in order:
        aux = (ann.aux_note[i] or "").strip().strip("\x00")
        m = EPI_RE.match(aux)
        if not m:
            continue
        beg, peak, typ, lead, d, mag, end = m.groups()
        if len(d) == 2:      # extreme-T sub-interval, nested in a normal T episode
            continue
        key = (typ, lead, d)
        s = int(ann.sample[i])
        if beg:
            if key in open_ev:           # previous never closed
                emit(typ, lead, d, open_ev.pop(key), s, "reopened")
            open_ev[key] = s
        elif peak:
            peaks[key].append((s, int(mag) if mag else None))
        elif end:
            if key in open_ev:
                emit(typ, lead, d, open_ev.pop(key), s, "")
            else:                        # begin before excerpt start
                emit(typ, lead, d, 0, s, "orphan_end")
    for (typ, lead, d), b in open_ev.items():   # still open at the end of the excerpt
        emit(typ, lead, d, b, n_samples, "open_at_end")
    return eps



def timeline(rec, ann, n_sec):
    """Per-second boolean arrays + beat list from an annotation object."""
    sm, sy, ax, sub = ann.sample, ann.symbol, ann.aux_note, ann.subtype
    # noise state
    noisy = np.zeros(n_sec, bool)
    afib = np.zeros(n_sec, bool)
    nz = [(int(sm[i]), int(sub[i])) for i in range(len(sm)) if sy[i] == "~"]
    nz = [(0, 0)] + nz + [(n_sec * FS, 0)]
    for (a, st), (b, _) in zip(nz[:-1], nz[1:]):
        if st != 0:
            noisy[a // FS: int(np.ceil(b / FS))] = True
    # rhythm
    rh = [(int(sm[i]), (ax[i] or "").strip("\x00").strip()) for i in range(len(sm)) if sy[i] == "+"]
    rh = rh + [(n_sec * FS, "")]
    for (a, r), (b, _) in zip(rh[:-1], rh[1:]):
        if r.startswith("(AFIB"):
            afib[a // FS: int(np.ceil(b / FS))] = True
    # beats
    bi = [i for i in range(len(sm)) if sy[i] in BEAT_SYMS]
    bt = np.array([sm[i] for i in bi]) / FS
    bs = np.array([sy[i] for i in bi])
    return noisy, afib, bt, bs


def mask_from(intervals, n_sec, pad=0):
    m = np.zeros(n_sec, bool)
    for a, b in intervals:
        m[max(0, int(a / FS) - pad): min(n_sec, int(np.ceil(b / FS)) + pad)] = True
    return m


def classify_windows(n_sec, noisy, afib, bt, bs, isch, buf, axis):
    """Non-overlapping 10 s grid windows -> class label (or None)."""
    out = []
    for t in range(0, n_sec - WIN_S + 1, WIN_S):
        sl = slice(t, t + WIN_S)
        if noisy[sl].any():
            out.append((t, "noisy")); continue
        if isch[sl].any():
            out.append((t, "ischemia_span")); continue
        if buf[sl].any():
            out.append((t, "buffer")); continue
        if axis[sl].any():
            out.append((t, "posture" if axis[sl].all() else "axis_edge")); continue
        k = (bt >= t) & (bt < t + WIN_S)
        syms = bs[k]
        times = bt[k]
        hr = 60.0 / np.diff(times).mean() if len(times) >= 3 else np.nan
        if afib[sl].any() or any(s in ECTOPIC for s in syms):
            out.append((t, "arrhythmia")); continue
        if np.isnan(hr):
            out.append((t, "sparse")); continue
        if 100 <= hr <= 130 and all(s == "N" for s in syms):
            out.append((t, "tachy")); continue
        if hr < 100 and all(s == "N" for s in syms):
            out.append((t, "clean")); continue
        out.append((t, "other"))
    return out


# ----------------------------------------------------------------------------
def patient_key(hdr):
    """EDB headers carry no patient id; identify repeat subjects by identical clinical comment block."""
    lines = [" ".join(c.lstrip("#").split()) for c in hdr.comments if c.strip()]
    return "|".join(sorted(lines))


def split_patients(pat_df, seed=42, n_trials=40000, sizes=(55, 12, 12)):
    """Random search for a patient-level 55/12/12 partition balanced on key counts."""
    rng = np.random.default_rng(seed)
    cols = ["n_pos", "n_posture", "n_arr", "n_tachy", "n_clean"]
    X = pat_df[cols].to_numpy(float)
    tot = X.sum(0)
    frac = np.array(sizes) / sum(sizes)
    has_post = pat_df["n_posture"].to_numpy() > 0
    has_pos = pat_df["n_pos"].to_numpy() > 0
    best, best_s = None, 1e18
    n = len(pat_df)
    for _ in range(n_trials):
        p = rng.permutation(n)
        idx = [p[:sizes[0]], p[sizes[0]:sizes[0] + sizes[1]], p[sizes[0] + sizes[1]:]]
        if not all(has_post[i].any() for i in idx[1:]):
            continue
        if not all(has_pos[i].any() for i in idx):
            continue
        s = 0.0
        for i, f in zip(idx, frac):
            s += (((X[i].sum(0) / np.maximum(tot, 1)) - f) ** 2).sum()
        if s < best_s:
            best, best_s = idx, s
    assert best is not None, "no feasible split found"
    lab = np.empty(n, object)
    for i, name in zip(best, ["train", "val", "test"]):
        lab[i] = name
    return lab, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edb", default="/Users/bingerliang/Downloads/european-st-t-database-1.0.0")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "outputs"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    edb, out = Path(a.edb), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = [r.strip() for r in (edb / "RECORDS").read_text().split() if r.strip()]
    recs = [r for r in recs if r not in EXCLUDE_RECORDS]
    print(f"{len(recs)} records")

    rows, epi_rows, win_rows = [], [], []
    for r in recs:
        hdr = wfdb.rdheader(str(edb / r))
        ann = wfdb.rdann(str(edb / r), "atr")
        n_sec = hdr.sig_len // FS
        eps = parse_episodes(r, ann, hdr.sig_len)
        isch_eps = [e for e in eps if e["kind"] == "ischemia"]
        axis_eps = [e for e in eps if e["kind"] == "axis_shift"]
        noisy, afib, bt, bs = timeline(r, ann, n_sec)
        isch = mask_from([(e["start"], e["end"]) for e in isch_eps], n_sec)
        buf = mask_from([(e["start"], e["end"]) for e in isch_eps], n_sec, pad=BUFFER_S) & ~isch
        axis = mask_from([(e["start"], e["end"]) for e in axis_eps], n_sec)
        wins = classify_windows(n_sec, noisy, afib, bt, bs, isch, buf, axis)
        cnt = defaultdict(int)
        for t, c in wins:
            cnt[c] += 1
            win_rows.append(dict(record=r, win_start_s=t, cls=c))
        # positives: window centred on peak must lie in record and avoid bad noise
        pos_ok = 0
        for e in isch_eps:
            c = e["peak"] / FS
            t0 = int(round(c - WIN_S / 2))
            usable = 0 <= t0 and t0 + WIN_S <= n_sec and not noisy[t0:t0 + WIN_S].any()
            e["pos_usable"] = usable
            if usable:
                sl = slice(t0, t0 + WIN_S)
                k = (bt >= t0) & (bt < t0 + WIN_S)
                e["pos_with_arrhythmia"] = bool(afib[sl].any() or any(s in ECTOPIC for s in bs[k]))
                pos_ok += 1
        epi_rows += eps
        rows.append(dict(
            record=r, patient_key=patient_key(hdr), n_sec=n_sec,
            n_isch_ep=len(isch_eps), n_st=sum(e["etype"] == "ST" for e in isch_eps),
            n_t=sum(e["etype"] == "T" for e in isch_eps), n_pos=pos_ok,
            n_pos_arr=sum(bool(e.get("pos_with_arrhythmia")) for e in isch_eps),
            n_axis_ep=len(axis_eps),
            sec_isch=int(isch.sum()), sec_buffer=int(buf.sum()), sec_axis=int(axis.sum()),
            sec_noisy=int(noisy.sum()), sec_afib=int(afib.sum()),
            n_ectopic=int(sum(s in ECTOPIC for s in bs)),
            n_clean=cnt["clean"], n_arr=cnt["arrhythmia"], n_posture=cnt["posture"],
            n_tachy=cnt["tachy"], n_axis_edge=cnt["axis_edge"], n_buffer_win=cnt["buffer"],
            n_noisy_win=cnt["noisy"], n_sparse=cnt["sparse"], n_other=cnt["other"]))

    df = pd.DataFrame(rows)
    # patient grouping (heuristic, see report)
    keys = {k: f"P{i + 1:02d}" for i, k in enumerate(sorted(df.patient_key.unique()))}
    df["patient_id"] = df.patient_key.map(keys)
    pat = df.groupby("patient_id").agg(
        records=("record", lambda x: ",".join(x)), n_rec=("record", "size"),
        n_pos=("n_pos", "sum"), n_posture=("n_posture", "sum"), n_arr=("n_arr", "sum"),
        n_tachy=("n_tachy", "sum"), n_clean=("n_clean", "sum")).reset_index()
    sizes = (55, 12, 12) if len(pat) == 79 else tuple(
        int(round(len(pat) * f)) for f in (0.7, 0.15, 0.15))
    if sum(sizes) != len(pat):
        sizes = (len(pat) - 2 * sizes[1], sizes[1], sizes[1])
    lab, sc = split_patients(pat, a.seed, sizes=sizes)
    pat["split"] = lab
    df = df.merge(pat[["patient_id", "split"]], on="patient_id")

    df.drop(columns="patient_key").to_csv(out / "records_summary.csv", index=False)
    pat.to_csv(out / "patient_split.csv", index=False)
    pd.DataFrame(epi_rows).to_csv(out / "episodes.csv", index=False)
    wdf = pd.DataFrame(win_rows).merge(df[["record", "patient_id", "split"]], on="record")
    wdf.to_csv(out / "window_budget.csv", index=False)

    # ---------------- report ----------------
    ep = pd.DataFrame(epi_rows)
    ie = ep[ep.kind == "ischemia"]
    L = []
    L.append("# Step 1 report: EDB annotation parsing & budget\n")
    L.append(f"- records: {len(df)}, patient groups (heuristic): {len(pat)}, split sizes {sizes}")
    L.append(f"- patients with >1 record: {int((pat.n_rec > 1).sum())}: "
             + "; ".join(f"{p.patient_id}={p.records}" for p in pat[pat.n_rec > 1].itertuples()))
    L.append(f"- ischemic episodes (upper-case, paired): {len(ie)}  "
             f"(ST {int((ie.etype == 'ST').sum())}, T {int((ie.etype == 'T').sum())}); "
             f"with peak annotation {int(ie.has_peak.sum())}; "
             f"incomplete-boundary episodes: {ie[ie.incomplete != ''].incomplete.value_counts().to_dict()}")
    L.append(f"- axis-shift episodes (lower-case): {int((ep.kind == 'axis_shift').sum())} "
             f"in records {sorted(ep[ep.kind == 'axis_shift'].record.unique())}")
    tot_h = lambda c: df[c].sum() / 3600
    L.append("\n## Total duration by state (hours; seconds-resolution masks)\n")
    L.append("| state | hours |\n|---|---|")
    for lbl, c in [("record total", "n_sec"), ("ischemia span", "sec_isch"),
                   ("buffer ±2min", "sec_buffer"), ("axis-shift span", "sec_axis"),
                   ("bad noise (n/u)", "sec_noisy"), ("AFIB rhythm", "sec_afib")]:
        L.append(f"| {lbl} | {tot_h(c):.2f} |")
    L.append("\n## Candidate non-overlapping 10 s windows (grid) by class\n")
    wc = wdf.groupby(["cls", "split"]).size().unstack(fill_value=0)
    wc["total"] = wc.sum(axis=1)
    L.append(wc.to_markdown())
    L.append("\n## Positive clips (centred on AST/AT peak, noise-free)\n")
    pos = df.groupby("split")[["n_isch_ep", "n_pos", "n_pos_arr"]].sum()
    pos["patients"] = pat.groupby("split").size()
    L.append(pos.to_markdown())
    L.append("\n## Negative budget needed vs. available (ratio 1:2 and 1:3)\n")
    rows2 = []
    for sp in ["train", "val", "test"]:
        npos = int(pos.loc[sp, "n_pos"])
        av = wc[sp]
        for ratio in (2, 3):
            need = npos * ratio
            req = dict(clean=0.50, arrhythmia=0.25, posture=0.15, tachy=0.10)
            lim = min(av.get(k, 0) / v for k, v in req.items()) if npos else 0
            rows2.append(dict(split=sp, ratio=f"1:{ratio}", n_pos=npos, neg_needed=need,
                              max_neg_given_mix=int(lim),
                              bottleneck=min(req, key=lambda k: av.get(k, 0) / req[k])))
    L.append(pd.DataFrame(rows2).to_markdown(index=False))
    L.append("\n## Split balance (patients / pos / windows)\n")
    L.append(pat.groupby("split")[["n_pos", "n_posture", "n_arr", "n_tachy", "n_clean"]]
             .sum().assign(patients=pat.groupby("split").size()).to_markdown())
    (out / "step1_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
