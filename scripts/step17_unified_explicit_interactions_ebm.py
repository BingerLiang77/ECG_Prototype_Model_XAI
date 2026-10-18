#!/usr/bin/env python
"""Step 17: Unified Dual-Mode EBM with Explicit Cross-Temporal Interactions.

Tests whether explicitly specifying high-value cross-temporal interaction pairs
(from Model B) restores the serial diagnostic advantage in a unified model,
overcoming the 50% missing-baseline dilution suffered by FAST greedy search.
"""
import sys
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parent))

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix
)
from step16_unified_dual_mode_ebm import (
    construct_dual_mode_datasets, evaluate_metrics,
    plot_consistency_scatter, plot_paired_waterfall_same_ecg
)

# 14 Cross-Temporal pairs from Model B + 6 Intra-Current pairs from Model A
EXPLICIT_INTERACTIONS = [
    # --- 14 Cross-Temporal Interactions (Serial Dynamics) ---
    ("cur_ch0_j_dev_mv", "prev_ch0_j_dev_mv"),
    ("cur_ch0_j_dev_mv", "prev_ch1_s_depth_mv"),
    ("cur_ch0_st_integral_mv_ms", "prev_ch0_st_integral_mv_ms"),
    ("cur_ch0_st_integral_mv_ms", "prev_ch1_rs_ratio"),
    ("cur_ch0_t_amp_mv", "prev_ch0_st_integral_mv_ms"),
    ("cur_ch0_t_amp_mv", "prev_ch0_t_amp_mv"),
    ("cur_ch0_t_amp_mv", "prev_ch1_rs_ratio"),
    ("cur_ch0_t_to_r_ratio", "prev_ch0_t_to_r_ratio"),
    ("cur_ch1_j_dev_mv", "prev_ch1_j_dev_mv"),
    ("cur_ch1_st_integral_mv_ms", "prev_ch1_st_integral_mv_ms"),
    ("cur_ch1_st_integral_mv_ms", "prev_ch1_t_to_r_ratio"),
    ("cur_ch1_t_amp_mv", "prev_ch1_t_amp_mv"),
    ("cur_ch1_t_to_r_ratio", "prev_ch1_t_to_r_ratio"),
    ("cur_spatial_theta_st_deg", "prev_spatial_theta_st_deg"),
    # --- 6 Intra-Current Interactions (Instantaneous Morphology) ---
    ("cur_ch1_st_dev_mv", "cur_ch1_s_depth_mv"),
    ("cur_rr_std_ms", "cur_ch0_j_dev_mv"),
    ("cur_ch0_s_depth_mv", "cur_ch0_rs_ratio"),
    ("cur_ch1_s_depth_mv", "cur_ch1_t_to_r_ratio"),
    ("cur_ch1_st_integral_mv_ms", "cur_ch1_s_depth_mv"),
    ("cur_ch1_j_dev_mv", "cur_ch1_t_symmetry_idx"),
]


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs/models"
    wf_dir = out_dir / "waterfall"
    out_dir.mkdir(parents=True, exist_ok=True)
    wf_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STEP 17: UNIFIED DUAL-MODE EBM WITH EXPLICIT CROSS-TEMPORAL INTERACTIONS")
    print(f"Total Explicit Interactions: {len(EXPLICIT_INTERACTIONS)} (14 cross-temporal + 6 intra-current)")
    print("=" * 80)
    
    # 1. Construct Dual-Mode Data
    processed_splits, prev_medians, feature_cols = construct_dual_mode_datasets()
    
    X_tr_with = processed_splits["train"]["df_with"]
    X_tr_without = processed_splits["train"]["df_without"]
    y_tr = processed_splits["train"]["y"]
    
    X_tr_combined = pd.concat([X_tr_with, X_tr_without], axis=0).reset_index(drop=True)
    y_tr_combined = np.concatenate([y_tr, y_tr])
    
    X_val_with = processed_splits["val"]["df_with"]
    X_val_without = processed_splits["val"]["df_without"]
    y_val = processed_splits["val"]["y"]
    X_val_combined = pd.concat([X_val_with, X_val_without], axis=0).reset_index(drop=True)
    y_val_combined = np.concatenate([y_val, y_val])
    
    # 2. Train EBM with Explicit Interactions
    print("\nTraining EBM with Explicit Interactions on combined dataset (2,222 rows x 73 cols)...")
    ebm_explicit = ExplainableBoostingClassifier(
        interactions=EXPLICIT_INTERACTIONS,
        learning_rate=0.03,
        max_leaves=3,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    ebm_explicit.fit(X_tr_combined, y_tr_combined)
    intercept = float(ebm_explicit.intercept_[0])
    
    # Save Model
    joblib.dump(ebm_explicit, out_dir / "unified_explicit_ebm.joblib")
    print(f"Saved model to: {out_dir / 'unified_explicit_ebm.joblib'}")
    
    # 3. Global Feature Importance
    exp_global = ebm_explicit.explain_global()
    global_names = exp_global.data()["names"]
    global_scores = exp_global.data()["scores"]
    global_df = pd.DataFrame({"feature": global_names, "importance": global_scores}).sort_values("importance", ascending=False)
    global_df.to_csv(out_dir / "unified_explicit_ebm_global_importance.csv", index=False)
    
    print("\n" + "=" * 80)
    print("TOP 20 GLOBAL FEATURES (EXPLICIT INTERACTIONS EBM):")
    print("=" * 80)
    print(global_df.head(20).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("ALL CROSS-TEMPORAL INTERACTIONS (CUR & PREV) SCORES:")
    print("=" * 80)
    cross_df = global_df[global_df["feature"].str.contains("prev") & global_df["feature"].str.contains("&")]
    print(cross_df.to_string(index=False))
    
    # 4. Calibrate Thresholds on Validation
    val_probs_combined = ebm_explicit.predict_proba(X_val_combined)[:, 1]
    best_th, best_ba = 0.5, -1.0
    for th in np.linspace(0.05, 0.95, 91):
        ba = balanced_accuracy_score(y_val_combined, (val_probs_combined >= th).astype(int))
        if ba > best_ba:
            best_ba = ba
            best_th = th
            
    print(f"\nValidation Optimal Unified Threshold: {best_th:.3f} (Val Balanced Acc: {best_ba:.4f})")
    
    # 5. Paired Inference on Test Set (203 unique ECG clips)
    X_te_with = processed_splits["test"]["df_with"]
    X_te_without = processed_splits["test"]["df_without"]
    y_te = processed_splits["test"]["y"]
    df_te = processed_splits["test"]["meta"]
    n_test = len(y_te)
    
    probs_with = ebm_explicit.predict_proba(X_te_with)[:, 1]
    probs_without = ebm_explicit.predict_proba(X_te_without)[:, 1]
    
    res_with, preds_with = evaluate_metrics(probs_with, y_te, best_th, df_te)
    res_without, preds_without = evaluate_metrics(probs_without, y_te, best_th, df_te)
    
    res_df = pd.DataFrame({
        "Explicit EBM (Without Baseline)": res_without,
        "Explicit EBM (With Baseline)": res_with,
    })
    res_df["Delta (With - Without)"] = res_df["Explicit EBM (With Baseline)"] - res_df["Explicit EBM (Without Baseline)"]
    
    print("\n" + "=" * 80)
    print("EXPLICIT INTERACTIONS DUAL-MODE TEST PERFORMANCE (SAME MODEL)")
    print("=" * 80)
    print(res_df.round(4).to_string())
    res_df.to_csv(out_dir / "unified_explicit_ebm_test_results.csv")
    
    # 6. Consistency Stats
    prob_delta = probs_with - probs_without
    abs_prob_delta = np.abs(prob_delta)
    decision_flips = preds_with != preds_without
    
    test_paired = df_te.copy()
    test_paired["ground_truth"] = y_te
    test_paired["prob_without"] = probs_without.round(4)
    test_paired["prob_with"] = probs_with.round(4)
    test_paired["prob_delta"] = prob_delta.round(4)
    test_paired["abs_prob_delta"] = abs_prob_delta.round(4)
    test_paired["pred_without"] = preds_without
    test_paired["pred_with"] = preds_with
    test_paired["decision_flip"] = decision_flips.astype(int)
    
    test_paired["cur_time_hhmmss"] = test_paired["start_sample"].apply(
        lambda s: f"{int(s//(250*3600)):02d}:{int((s%(250*3600))//(250*60)):02d}:{int((s%(250*60))//250):02d}"
    )
    
    def categorize_flip(r):
        pw, pwo, gt = r["pred_with"], r["pred_without"], r["ground_truth"]
        if pw == pwo:
            return "Concordant Decision"
        elif pwo == 0 and pw == 1:
            return "With-Baseline Alarms (0 -> 1)" + (" [Corrected Miss]" if gt==1 else " [New False Alarm]")
        else:
            return "With-Baseline Suppresses (1 -> 0)" + (" [Corrected False Alarm]" if gt==0 else " [Missed Ischemia]")
            
    test_paired["flip_category"] = test_paired.apply(categorize_flip, axis=1)
    test_paired.to_csv(out_dir / "unified_explicit_ebm_paired_predictions.csv", index=False)
    
    print("\n" + "=" * 80)
    print("PREDICTION CONSISTENCY STATS (203 IDENTICAL TEST ECGs)")
    print("=" * 80)
    print(f"Mean Absolute Probability Delta: {abs_prob_delta.mean():.4f}")
    print(f"Median Absolute Probability Delta: {np.median(abs_prob_delta):.4f}")
    print(f"Max Absolute Probability Delta: {abs_prob_delta.max():.4f}")
    print(f"Total Decision Flips: {decision_flips.sum()} / {n_test} ({decision_flips.mean()*100:.1f}%)")
    print(f"Concordant Decisions: {(~decision_flips).sum()} / {n_test} ({(~decision_flips).mean()*100:.1f}%)")
    print("\nFlip Breakdown:")
    for cat, count in test_paired["flip_category"].value_counts().items():
        print(f"  {cat}: {count} ({count/n_test*100:.1f}%)")
        
    plot_consistency_scatter(test_paired, best_th, out_dir / "unified_explicit_ebm_prob_scatter.png")
    
    # 7. Waterfall Plots for Key Cases
    print("\nGenerating local explanations for paired waterfalls...")
    local_exp_without = ebm_explicit.explain_local(X_te_without)
    local_exp_with = ebm_explicit.explain_local(X_te_with)
    
    sample_cases = [
        {"pair_id": "PR_test_0004", "name": "explicit_waterfall_case1_ischemia_boost_P21"},
        {"pair_id": "PR_test_0192", "name": "explicit_waterfall_case2_corrected_false_alarm_P55"},
        {"pair_id": "PR_test_0179", "name": "explicit_waterfall_case3_posture_new_false_alarm_P48"},
        {"pair_id": "PR_test_0108", "name": "explicit_waterfall_case4_normal_stability_P31"}
    ]
    
    for sc in sample_cases:
        matches = test_paired[test_paired["pair_id"] == sc["pair_id"]]
        if len(matches) > 0:
            idx = matches.index[0]
            row = test_paired.loc[idx]
            plot_paired_waterfall_same_ecg(
                row,
                local_exp_without.data(idx),
                local_exp_with.data(idx),
                intercept,
                best_th,
                wf_dir / f"{sc['name']}.png"
            )
            
    print("\nStep 17 explicit interactions experiment completed successfully!")


if __name__ == "__main__":
    main()
