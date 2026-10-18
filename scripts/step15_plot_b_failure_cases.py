#!/usr/bin/env python
"""Step 15: Generate Waterfall and Waveform plots for cases where Model B failed (B worse than A).

1. Baseline Poisoning (B misses true ischemia because baseline was contaminated with arrhythmia):
   - Patient P35 (e0206), pair PR_test_0037, GT=1, A=0.672 -> B=0.065
2. Pseudo-Dynamic Shift (B hallucinates acute shift on completely flat clean ECG):
   - Patient P21 (e0106), pair PR_test_0056, GT=0, A=0.072 -> B=0.729
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from interpret.glassbox import ExplainableBoostingClassifier
from step11_raw_concat_ab_experiment import build_raw_concat_data
from step14_concordant_divergent_waterfalls import plot_concordant_waterfall


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs/models/waterfall"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    splits, cur_feats, concat_feats = build_raw_concat_data()
    tr_cur, y_tr = splits["train"]["cur_X"], splits["train"]["y"]
    te_cur = splits["test"]["cur_X"]
    tr_cat = splits["train"]["concat_X"]
    te_cat = splits["test"]["concat_X"]
    
    ebm_a = ExplainableBoostingClassifier(interactions=15, learning_rate=0.03, max_leaves=3, min_samples_leaf=10, random_state=42, n_jobs=-1)
    ebm_a.fit(tr_cur, y_tr)
    
    ebm_b = ExplainableBoostingClassifier(interactions=15, learning_rate=0.03, max_leaves=3, min_samples_leaf=10, random_state=42, n_jobs=-1)
    ebm_b.fit(tr_cat, y_tr)
    
    th_a, th_b = 0.180, 0.240
    intercept_a = float(ebm_a.intercept_[0])
    intercept_b = float(ebm_b.intercept_[0])
    
    all_preds = pd.read_csv(root / "outputs/models/ebm_all_test_predictions.csv")
    cur_waves = np.load(root / "dataset/paired/test_cur.npy")
    prev_waves = np.load(root / "dataset/paired/test_prev.npy")
    
    local_exp_a = ebm_a.explain_local(te_cur)
    local_exp_b = ebm_b.explain_local(te_cat)
    
    failure_scenarios = [
        {
            "idx": 37,
            "filename": "failure_case1_baseline_poisoning_P35.png",
            "desc": "Baseline Poisoning: Arrhythmia Baseline Suppresses True Ischemia Alarm (A=0.67 -> B=0.07)"
        },
        {
            "idx": 56,
            "filename": "failure_case2_pseudo_dynamic_shift_P21.png",
            "desc": "Pseudo-Dynamic Shift: Flat Clean ECG Hallucinates Alarm Due to Old Baseline (A=0.07 -> B=0.73)"
        }
    ]
    
    for sc in failure_scenarios:
        i = sc["idx"]
        row = all_preds.loc[i]
        c_wave = cur_waves[i]
        p_wave = prev_waves[i]
        
        plot_concordant_waterfall(
            i, row, c_wave, p_wave,
            local_exp_a.data(i), intercept_a, th_a,
            local_exp_b.data(i), intercept_b, th_b,
            sc["desc"],
            out_dir / sc["filename"]
        )
        
    print("Failure mode plots generated successfully!")


if __name__ == "__main__":
    main()
