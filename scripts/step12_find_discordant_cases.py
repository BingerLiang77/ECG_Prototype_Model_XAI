#!/usr/bin/env python
"""Step 12: Identify and Export Discordant Predictions between Model A and Model B.

Extracts all test cases where Model A (Single Snapshot) and Model B (Concat with Baseline) disagree:
1. A=1, B=0 (A Positive, B Negative):
   - B corrects A's false alarm (Ground truth is Negative)
   - B misses subtle ischemia that A caught (Ground truth is Positive)
2. A=0, B=1 (A Negative, B Positive):
   - B catches subtle dynamic ischemia that A missed (Ground truth is Positive)
   - B generates a new false alarm (Ground truth is Negative)

Saves all metadata, probabilities, thresholds, window locations, and baseline info.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from step11_raw_concat_ab_experiment import build_raw_concat_data


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs/models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 75)
    print("STEP 12: IDENTIFYING DISCORDANT CASES (MODEL A vs MODEL B)")
    print("=" * 75)
    
    splits, cur_feats, concat_feats = build_raw_concat_data()
    tr_cur, y_tr = splits["train"]["cur_X"], splits["train"]["y"]
    val_cur, y_val = splits["val"]["cur_X"], splits["val"]["y"]
    te_cur, y_te = splits["test"]["cur_X"], splits["test"]["y"]
    
    tr_cat = splits["train"]["concat_X"]
    val_cat = splits["val"]["concat_X"]
    te_cat = splits["test"]["concat_X"]
    
    # Metadata
    df_te_single = pd.read_csv(root / "dataset/single/test_meta.csv")
    df_te_paired = pd.read_csv(root / "dataset/paired/test_meta.csv")
    meta = pd.merge(
        df_te_paired,
        df_te_single[["clip_id", "event_id", "event_role", "peak_uv", "event_types", "with_arrhythmia"]],
        on="clip_id",
        how="left"
    )
    
    # 1. Fit Model A
    print("Training Model A (Single)...")
    ebm_a = ExplainableBoostingClassifier(
        interactions=15, learning_rate=0.03, max_leaves=3, min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    ebm_a.fit(tr_cur, y_tr)
    v_prob_a = ebm_a.predict_proba(val_cur)[:, 1]
    t_prob_a = ebm_a.predict_proba(te_cur)[:, 1]
    
    # 2. Fit Model B
    print("Training Model B (Concat with Baseline)...")
    ebm_b = ExplainableBoostingClassifier(
        interactions=15, learning_rate=0.03, max_leaves=3, min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    ebm_b.fit(tr_cat, y_tr)
    v_prob_b = ebm_b.predict_proba(val_cat)[:, 1]
    t_prob_b = ebm_b.predict_proba(te_cat)[:, 1]
    
    # 3. Calibrate Thresholds on Validation
    th_a, th_b = 0.5, 0.5
    best_ba_a, best_ba_b = -1.0, -1.0
    for th in np.linspace(0.05, 0.95, 91):
        ba_a = balanced_accuracy_score(y_val, (v_prob_a >= th).astype(int))
        if ba_a > best_ba_a:
            best_ba_a, th_a = ba_a, th
        ba_b = balanced_accuracy_score(y_val, (v_prob_b >= th).astype(int))
        if ba_b > best_ba_b:
            best_ba_b, th_b = ba_b, th
            
    print(f"Optimal Thresholds -> Model A: {th_a:.3f}, Model B: {th_b:.3f}")
    
    # 4. Predictions on Test
    pred_a = (t_prob_a >= th_a).astype(int)
    pred_b = (t_prob_b >= th_b).astype(int)
    
    # 5. Build Comprehensive Test Results Table
    test_results = meta.copy()
    test_results["ground_truth"] = y_te
    test_results["prob_A"] = t_prob_a.round(4)
    test_results["prob_B"] = t_prob_b.round(4)
    test_results["pred_A"] = pred_a
    test_results["pred_B"] = pred_b
    test_results["threshold_A"] = round(th_a, 3)
    test_results["threshold_B"] = round(th_b, 3)
    
    # Human-readable timestamps (sampling rate 250 Hz)
    test_results["cur_time_hhmmss"] = test_results["start_sample"].apply(
        lambda s: f"{int(s//(250*3600)):02d}:{int((s%(250*3600))//(250*60)):02d}:{int((s%(250*60))//250):02d}"
    )
    test_results["prev_time_hhmmss"] = test_results["prev_start_sample"].apply(
        lambda s: f"{int(s//(250*3600)):02d}:{int((s%(250*3600))//(250*60)):02d}:{int((s%(250*60))//250):02d}"
    )
    
    # Discordance classification
    def categorize_discordance(row):
        pa, pb, gt = row["pred_A"], row["pred_B"], row["ground_truth"]
        if pa == pb:
            return "Concordant"
        elif pa == 1 and pb == 0:
            if gt == 0:
                return "A_pos_B_neg (B corrects A false alarm)"
            else:
                return "A_pos_B_neg (B misses ischemia caught by A)"
        else: # pa == 0 and pb == 1
            if gt == 1:
                return "A_neg_B_pos (B catches subtle ischemia missed by A)"
            else:
                return "A_neg_B_pos (B false alarm)"
                
    test_results["discordance_category"] = test_results.apply(categorize_discordance, axis=1)
    
    # Filter discordant cases
    discordant_df = test_results[test_results["pred_A"] != test_results["pred_B"]].copy()
    discordant_df.sort_values(by=["discordance_category", "patient_id", "start_sample"], inplace=True)
    
    # Select cleanly ordered columns
    cols_to_save = [
        "pair_id", "patient_id", "record", "event_id", "subtype", "ground_truth",
        "pred_A", "prob_A", "threshold_A",
        "pred_B", "prob_B", "threshold_B",
        "discordance_category",
        "start_sample", "cur_time_hhmmss",
        "prev_start_sample", "prev_time_hhmmss", "prev_type", "prev_gap_s",
        "peak_uv", "event_role", "event_types", "with_arrhythmia"
    ]
    discordant_clean = discordant_df[cols_to_save]
    
    # Export CSVs
    out_csv = out_dir / "ebm_discordant_cases.csv"
    discordant_clean.to_csv(out_csv, index=False)
    test_results.to_csv(out_dir / "ebm_all_test_predictions.csv", index=False)
    
    # Stats
    print("\n" + "=" * 75)
    print("DISCORDANCE SUMMARY STATS")
    print("=" * 75)
    cat_counts = discordant_df["discordance_category"].value_counts()
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count} cases ({count/len(test_results)*100:.1f}%)")
    print(f"\nTotal discordant cases saved to: {out_csv} ({len(discordant_df)} rows)")


if __name__ == "__main__":
    main()
