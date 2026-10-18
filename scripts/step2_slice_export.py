#!/usr/bin/env python
"""Step 2: slice 10 s clips and export the single-snapshot and paired (Previous, Current) datasets.

Requires outputs of step1 (patient_split.csv, records_summary.csv, episodes.csv, window_budget.csv).

Exports to ../dataset:
  single/{split}_X.npy        float32 (N, 2, 2500), mV, channel order = record header order
  single/{split}_meta.csv     patient_id, clip_id, subtype, is_pos + provenance
  single/all_meta.csv
  paired/{split}_cur.npy / {split}_prev.npy   float32 (M, 2, 2500)
  paired/{split}_meta.csv     pair_id, clip_id, prev_* provenance
  paired/all_meta.csv
  dataset_summary.md
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

FS, WIN_S = 250, 10
WIN = FS * WIN_S
NEG_MIX = {"clean": 0.50, "arrhythmia": 0.25, "posture": 0.15, "tachy": 0.10}
PREV_MIX = [("clean", 0.65), ("arrhythmia", 0.25), ("buffer", 0.10)]  # buffer = pre/post-event edge
PREV_LOOKBACK_S = 1800   # prefer a Previous from the 30 min before the target


def balanced_pick(pool, quota, rng):
    """pool: DataFrame(patient_id, record, win_start_s). Round-robin over patients, random windows."""
    by_p = {p: g.sample(frac=1, random_state=int(rng.integers(1 << 31))).to_dict("records")
            for p, g in pool.groupby("patient_id")}
    out = []
    while len(out) < quota and by_p:
        for p in list(rng.permutation(list(by_p))):
            if len(out) >= quota:
                break
            out.append(by_p[p].pop())
            if not by_p[p]:
                del by_p[p]
    return out


def cluster_events(g):
    """Merge ischemic episodes (ST/T, both leads) of one record whose spans overlap."""
    g = g.sort_values("start")
    groups, cur, cur_end = [], [], -1
    for x in g.itertuples():
        if cur and x.start <= cur_end:
            cur.append(x.Index); cur_end = max(cur_end, x.end)
        else:
            if cur:
                groups.append(cur)
            cur, cur_end = [x.Index], x.end
    if cur:
        groups.append(cur)
    return [g.loc[ix] for ix in groups]


def overlap_score(x, ev):
    """Total seconds that episode x overlaps with the other episodes of its event."""
    tot = 0
    for y in ev.itertuples():
        if y.Index == x.Index:
            continue
        tot += max(0, min(x.end, y.end) - max(x.start, y.start))
    return tot / FS


def template(sig, beats, t0, min_beats=5):
    """Median beat (R-40 ms .. R+400 ms) of a 10 s clip, PR-segment baseline removed. (2, 110)."""
    b = beats[(beats >= t0 * FS + 20) & (beats < (t0 + WIN_S) * FS - 110)]
    if len(b) < min_beats:
        return None
    segs = np.stack([sig[:, i - 10:i + 100] - sig[:, i - 15:i - 8].mean(axis=1, keepdims=True)
                     for i in b])
    return np.median(segs, axis=0)


def template_diff(a, b):
    """Max |difference| (mV) over both leads in the ST-T portion (R+40 ms .. R+400 ms)."""
    return float(np.abs(a[:, 20:] - b[:, 20:]).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra_gap_s", type=int, default=300, help="min peak spacing for extra clips in one event")
    ap.add_argument("--min_morph_mv", type=float, default=0.1, help="min ST-T template difference (mV)")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--edb", default="/Users/bingerliang/Downloads/european-st-t-database-1.0.0")
    ap.add_argument("--step1", default=str(root / "outputs"))
    ap.add_argument("--out", default=str(root / "dataset"))
    ap.add_argument("--neg_ratio", type=int, default=3, help="negatives per positive (2 or 3)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    edb, s1, out = Path(a.edb), Path(a.step1), Path(a.out)
    (out / "single").mkdir(parents=True, exist_ok=True)
    (out / "paired").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    pat = pd.read_csv(s1 / "patient_split.csv")
    recs = pd.read_csv(s1 / "records_summary.csv")[["record", "patient_id", "split"]]
    eps_all = pd.read_csv(s1 / "episodes.csv")
    eps_all = eps_all[eps_all.kind == "ischemia"].merge(recs, on="record")
    wb = pd.read_csv(s1 / "window_budget.csv")
    assert (pat.groupby("split").patient_id.nunique().sum()) == pat.patient_id.nunique()

    # ---------------- 1a. events & positive clips ----------------
    pos_rows, ev_stats = [], defaultdict(int)
    span_wins = {r: g.win_start_s.to_numpy() for r, g in wb[wb.cls == "ischemia_span"].groupby("record")}
    for r in eps_all.record.unique():
        span_wins.setdefault(r, np.array([], int))
    for r, g in eps_all.groupby("record"):
        rec = wfdb.rdrecord(str(edb / r))
        sig = rec.p_signal.T.astype(np.float32)
        ann = wfdb.rdann(str(edb / r), "atr")
        beats = np.array([s for s, y in zip(ann.sample, ann.symbol) if y == "N"])
        tmpl_cache = {}
        for k, ev in enumerate(cluster_events(g)):
            eid = f"{r}_E{k:02d}"
            ev_stats["events"] += 1
            usable = ev[ev.pos_usable == True].copy()  # noqa: E712
            ev_start = int(ev.start.min() / FS)
            ev_end = int(np.ceil(ev.end.max() / FS))
            types = "+".join(sorted(set(f"{x.etype}{x.lead}{x.direction}" for x in ev.itertuples())))
            if usable.empty:
                # all peak-centred windows are noisy: use the noise-free grid window (class
                # 'ischemia_span') lying fully inside the event, nearest to the best-overlap peak
                cand = span_wins[r][(span_wins[r] >= ev_start) & (span_wins[r] + WIN_S <= ev_end)]
                if len(cand) == 0:
                    ev_stats["events_dropped_noisy"] += 1
                    continue
                sc = [overlap_score(x, ev) for x in ev.itertuples()]
                best = list(ev.itertuples())[int(np.argmax(sc))]
                t0 = int(cand[np.argmin(np.abs(cand - (best.peak / FS - WIN_S / 2)))])
                inb = (ann.sample >= t0 * FS) & (ann.sample < (t0 + WIN_S) * FS)
                arr = any(y in "VSaJF" for y in np.array(ann.symbol)[inb])
                ev_stats["events_fallback_window"] += 1
                pos_rows.append(dict(
                    split=best.split, patient_id=best.patient_id, record=r, start_s=t0, is_pos=1,
                    subtype=f"pos_{best.etype}{best.lead}{best.direction}", cls="ischemia",
                    peak_uv=best.peak_uv, with_arrhythmia=int(arr),
                    ep_incomplete=best.incomplete if isinstance(best.incomplete, str) else "",
                    event_id=eid, event_role="primary", clip_source="fallback_window",
                    event_types=types, n_ep_in_event=len(ev), event_start_s=ev_start,
                    event_end_s=ev_end))
                continue
            usable["score"] = [overlap_score(x, ev) for x in usable.itertuples()]
            usable = usable.sort_values(["score", "peak"], ascending=[False, True])
            chosen = []
            for x in usable.itertuples():
                t0 = int(round(x.peak / FS - WIN_S / 2))
                if chosen:
                    if min(abs(x.peak - c.peak) for c in chosen) < a.extra_gap_s * FS:
                        continue
                    if t0 not in tmpl_cache:
                        tmpl_cache[t0] = template(sig, beats, t0)
                    tm = tmpl_cache[t0]
                    ok = tm is not None
                    for c in chosen:
                        tc = tmpl_cache[int(round(c.peak / FS - WIN_S / 2))]
                        if ok and tc is not None and template_diff(tm, tc) < a.min_morph_mv:
                            ok = False
                    if not ok:
                        ev_stats["extra_rejected_similar"] += 1
                        continue
                else:
                    tmpl_cache[t0] = template(sig, beats, t0)
                chosen.append(x)
                pos_rows.append(dict(
                    split=x.split, patient_id=x.patient_id, record=r, start_s=t0, is_pos=1,
                    subtype=f"pos_{x.etype}{x.lead}{x.direction}", cls="ischemia", peak_uv=x.peak_uv,
                    with_arrhythmia=int(bool(x.pos_with_arrhythmia)),
                    ep_incomplete=x.incomplete if isinstance(x.incomplete, str) else "",
                    event_id=eid, event_role="primary" if len(chosen) == 1 else "extra", clip_source="peak",
                    event_types=types, n_ep_in_event=len(ev), event_start_s=ev_start, event_end_s=ev_end))
            ev_stats["extra_clips"] += len(chosen) - 1
    pos_df = pd.DataFrame(pos_rows)

    # ---------------- 1b. negatives ----------------
    clips = pos_df.to_dict("records")
    for sp in ["train", "val", "test"]:
        n_pos = int((pos_df.split == sp).sum())
        for cls, frac in NEG_MIX.items():
            quota = int(round(n_pos * a.neg_ratio * frac))
            pool = wb[(wb.split == sp) & (wb.cls == cls)]
            pick = balanced_pick(pool, quota, rng)
            if len(pick) < quota:
                print(f"WARNING {sp}/{cls}: only {len(pick)}/{quota} windows available")
            for w in pick:
                clips.append(dict(split=sp, patient_id=w["patient_id"], record=w["record"],
                                  start_s=int(w["win_start_s"]), is_pos=0,
                                  subtype=f"neg_{cls}", cls=cls, peak_uv=np.nan,
                                  with_arrhythmia=int(cls == "arrhythmia"), ep_incomplete="",
                                  event_id="", event_role="", clip_source="", event_types="", n_ep_in_event=0,
                                  event_start_s=np.nan, event_end_s=np.nan))
    cl = pd.DataFrame(clips)
    cl = cl.sort_values(["split", "is_pos", "record", "start_s"], ascending=[True, False, True, True])
    cl["clip_id"] = [f"C{i:05d}" for i in range(len(cl))]
    cl = cl.reset_index(drop=True)
    # no two positive windows of one record may overlap
    for _, g in cl[cl.is_pos == 1].groupby("record"):
        assert (np.diff(np.sort(g.start_s.to_numpy())) >= WIN_S).all(), "overlapping positives"

    # ---------------- 2. Previous selection (one per event; per clip for negatives) ----------------
    pool_by_rec = {}
    for (r, cls), g in wb[wb.cls.isin(["clean", "arrhythmia", "buffer"])].groupby(["record", "cls"]):
        pool_by_rec[(r, cls)] = g.win_start_s.to_numpy()
    used = set()
    event_prev = {}

    def draw_prev(record, ref_s):
        want = rng.choice([k for k, _ in PREV_MIX], p=[v for _, v in PREV_MIX])
        for k in [want] + [k for k, _ in PREV_MIX if k != want]:
            cand = pool_by_rec.get((record, k), np.array([], int))
            cand = cand[cand + WIN_S <= ref_s]                 # strictly before the reference time
            fresh = np.array([c for c in cand if (record, int(c)) not in used], int)
            cand = fresh if len(fresh) else cand               # de-duplicate when possible
            if len(cand):
                near = cand[cand >= ref_s - PREV_LOOKBACK_S]
                s = int(rng.choice(near if len(near) else cand))
                used.add((record, s))
                return s, k, k == want
        return None

    prev_rows = []
    for c in cl.itertuples():
        if c.is_pos:    # reference = event onset (true pre-onset baseline), shared inside the event
            if c.event_id not in event_prev:
                event_prev[c.event_id] = draw_prev(c.record, int(c.event_start_s))
            prev_rows.append(event_prev[c.event_id])
        else:
            prev_rows.append(draw_prev(c.record, c.start_s))
    cl["prev_start_s"] = [x[0] if x else np.nan for x in prev_rows]
    cl["prev_type"] = [x[1] if x else "" for x in prev_rows]
    cl["prev_type_wanted_ok"] = [int(x[2]) if x else 0 for x in prev_rows]
    cl["prev_gap_s"] = cl.start_s - cl.prev_start_s - WIN_S
    print("event stats:", dict(ev_stats))

    # ---------------- 3. cut waveforms ----------------
    X = np.zeros((len(cl), 2, WIN), np.float32)
    P = np.zeros_like(X)
    sig_names = {}
    for r, g in cl.groupby("record"):
        rec = wfdb.rdrecord(str(edb / r))
        sig = rec.p_signal.T.astype(np.float32)    # mV (200 adu/mV -> physical units already)
        sig_names[r] = ",".join(rec.sig_name)
        for i, row in g.iterrows():
            s0 = row.start_s * FS
            X[i] = sig[:, s0:s0 + WIN]
            if not np.isnan(row.prev_start_s):
                q0 = int(row.prev_start_s) * FS
                P[i] = sig[:, q0:q0 + WIN]
    cl["sig_names"] = cl.record.map(sig_names)
    cl["start_sample"] = cl.start_s * FS
    assert np.isfinite(X).all() and (X.std(axis=2) > 0).all(), "bad clip (nan/flat)"

    # ---------------- 4. export ----------------
    cols = ["patient_id", "clip_id", "subtype", "is_pos", "split", "record", "start_sample",
            "cls", "with_arrhythmia", "peak_uv", "event_id", "event_role", "clip_source", "event_types",
            "n_ep_in_event", "ep_incomplete", "sig_names"]
    for sp in ["train", "val", "test"]:
        m = cl[cl.split == sp]
        np.save(out / "single" / f"{sp}_X.npy", X[m.index.to_numpy()])
        m[cols].reset_index(drop=True).to_csv(out / "single" / f"{sp}_meta.csv", index=False)
        pm = m[m.prev_start_s.notna()].copy()
        pm["pair_id"] = [f"PR{i:05d}" for i in range(len(pm))] if sp == "train" else \
            [f"PR_{sp}_{i:04d}" for i in range(len(pm))]
        pm["prev_start_sample"] = (pm.prev_start_s * FS).astype(int)
        np.save(out / "paired" / f"{sp}_cur.npy", X[pm.index.to_numpy()])
        np.save(out / "paired" / f"{sp}_prev.npy", P[pm.index.to_numpy()])
        pm[["pair_id", "clip_id", "patient_id", "subtype", "is_pos", "split", "record", "start_sample",
            "prev_start_sample", "prev_type", "prev_type_wanted_ok", "prev_gap_s"]] \
            .reset_index(drop=True).to_csv(out / "paired" / f"{sp}_meta.csv", index=False)
    cl[cols].to_csv(out / "single" / "all_meta.csv", index=False)
    cl[cl.prev_start_s.notna()][["clip_id", "patient_id", "subtype", "is_pos", "split", "record",
                                 "prev_type", "prev_gap_s"]].to_csv(out / "paired" / "all_meta.csv", index=False)

    # leakage assertion: patient -> exactly one split
    assert cl.groupby("patient_id").split.nunique().max() == 1

    # ---------------- 5. summary ----------------
    L = ["# Dataset summary\n", f"neg_ratio=1:{a.neg_ratio}, seed={a.seed}\n"]
    L.append(pd.crosstab([cl.split, cl.subtype.str.split("_").str[0]], cl.cls).to_markdown())
    L.append("\n## Single: count per split / is_pos\n")
    L.append(pd.crosstab(cl.split, cl.is_pos).to_markdown())
    L.append("\n## Positive subtype\n")
    L.append(pd.crosstab(cl[cl.is_pos == 1].split, cl[cl.is_pos == 1].subtype).to_markdown())
    L.append("\n## Paired: available / dropped (no earlier window) per split\n")
    L.append(pd.DataFrame({"clips": cl.groupby("split").size(),
                           "paired": cl[cl.prev_start_s.notna()].groupby("split").size()}).to_markdown())
    L.append("\n## Previous type actually used (fraction)\n")
    L.append(pd.crosstab(cl.split, cl.prev_type, normalize="index").round(3).to_markdown())
    L.append(f"\nMedian Previous gap (s): {cl.prev_gap_s.median():.0f}; "
             f"extra clips (long events): {int((cl.event_role == 'extra').sum())}; "
             f"events with a positive clip: {cl[cl.is_pos == 1].event_id.nunique()}")
    L.append(f"\nPatients per split: {cl.groupby('split').patient_id.nunique().to_dict()}")
    (out / "dataset_summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
