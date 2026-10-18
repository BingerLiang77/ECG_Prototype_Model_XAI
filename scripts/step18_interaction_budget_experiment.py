#!/usr/bin/env python
"""Step 18: Interaction Budget Exploration in Unified Dual-Mode EBM.

Compares:
1. Auto FAST Interactions = 50 (testing if larger budget allows cross-temporal terms to emerge naturally)
2. Explicit Canonical Longitudinal Interactions = 50 (all 36 canonical cur_x & prev_x pairs + 14 intra-current pairs)
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
    balanced_accuracy_score, f1_score
)
from step16_unified_dual_mode_ebm import construct_dual_mode_datasets, evaluate_metrics

def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs/models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STEP 18: EXPANDING INTERACTION BUDGET (BUDGET = 50)")
    print("=" * 80)
    
    # 1. Datasets
    processed_splits, prev_medians, feature_cols = construct_dual_mode_datasets()
    cur_cols = [c for c in feature_cols if c.startswith("cur_")]
    
    X_tr_with = processed_splits["train"]["df_with"]
    X_tr_without = processed_splits["train"]["df_without"]
    y_tr = processed_splits["train"]["y"]
    X_tr_comb = pd.concat([X_tr_with, X_tr_without], axis=0).reset_index(drop=True)
    y_tr_comb = np.concatenate([y_tr, y_tr])
    
    X_val_with = processed_splits["val"]["df_with"]
    X_val_without = processed_splits["val"]["df_without"]
    y_val = processed_splits["val"]["y"]
    X_val_comb = pd.concat([X_val_with, X_val_without], axis=0).reset_index(drop=True)
    y_val_comb = np.concatenate([y_val, y_val])
    
    X_te_with = processed_splits["test"]["df_with"]
    X_te_without = processed_splits["test"]["df_without"]
    y_te = processed_splits["test"]["y"]
    df_te = processed_splits["test"]["meta"]
    
    # Construct 50 Explicit Interactions:
    # 36 Canonical Longitudinal Pairs: (cur_feat, prev_feat) for all 36 features!
    canonical_pairs = []
    for c in cur_cols:
        p = c.replace("cur_", "prev_")
        if p in feature_cols:
            canonical_pairs.append((c, p))
            
    # Top 14 Intra-Current Pairs from single-ECG analysis
    intra_current_pairs = [
        ("cur_ch1_st_dev_mv", "cur_ch1_s_depth_mv"),
        ("cur_rr_std_ms", "cur_ch0_j_dev_mv"),
        ("cur_ch0_s_depth_mv", "cur_ch0_rs_ratio"),
        ("cur_ch1_s_depth_mv", "cur_ch1_t_to_r_ratio"),
        ("cur_ch1_st_integral_mv_ms", "cur_ch1_s_depth_mv"),
        ("cur_ch1_j_dev_mv", "cur_ch1_t_symmetry_idx"),
        ("cur_ch0_st_slope_mv_s", "cur_ch0_st_integral_mv_ms"),
        ("cur_rr_std_ms", "cur_reciprocal_st_product"),
        ("cur_rr_std_ms", "cur_ch1_r_amp_mv"),
        ("cur_ch1_j_dev_mv", "cur_ch1_s_depth_mv"),
        ("cur_hr_bpm", "cur_ch1_qtc_bazett_ms"),
        ("cur_ch1_st_dev_mv", "cur_ch1_st_integral_mv_ms"),
        ("cur_ch1_st_integral_mv_ms", "cur_ch1_qt_ms"),
        ("cur_ch1_t_amp_mv", "cur_ch1_r_amp_mv"),
    ]
    explicit_50 = canonical_pairs + intra_current_pairs
    print(f"Constructed Explicit 50 Interactions: {len(canonical_pairs)} Canonical Longitudinal + {len(intra_current_pairs)} Intra-Current")
    
    # Experiment A: Auto FAST = 50
    print("\n" + "-" * 60)
    print("Training Model 1: Auto FAST interactions=50...")
    ebm_auto50 = ExplainableBoostingClassifier(
        interactions=50, learning_rate=0.03, max_leaves=3, min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    ebm_auto50.fit(X_tr_comb, y_tr_comb)
    
    # Check how many cross-temporal interactions FAST picked
    exp_auto50 = ebm_auto50.explain_global()
    names_a50 = exp_auto50.data()["names"]
    int_a50 = [n for n in names_a50 if " & " in n]
    cross_a50 = [n for n in int_a50 if "prev" in n]
    print(f"Auto FAST picked {len(int_a50)} interactions, of which {len(cross_a50)} are cross-temporal!")
    if len(cross_a50) > 0:
        print("Cross-temporal terms picked by FAST 50:")
        for cn in cross_a50:
            print(f"  {cn}")
            
    # Experiment B: Explicit 50
    print("\n" + "-" * 60)
    print("Training Model 2: Explicit Interactions = 50 (36 Canonical + 14 Intra)...")
    ebm_exp50 = ExplainableBoostingClassifier(
        interactions=explicit_50, learning_rate=0.03, max_leaves=3, min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    ebm_exp50.fit(X_tr_comb, y_tr_comb)
    
    # Evaluate Both on Test Set
    models = {
        "Auto FAST (interactions=50)": ebm_auto50,
        "Explicit 50 (36 Canonical + 14 Intra)": ebm_exp50,
    }
    
    summary_rows = []
    for mname, m in models.items():
        # Calibrate on validation
        val_probs = m.predict_proba(X_val_comb)[:, 1]
        best_th, best_ba = 0.5, -1.0
        for th in np.linspace(0.05, 0.95, 91):
            ba = balanced_accuracy_score(y_val_comb, (val_probs >= th).astype(int))
            if ba > best_ba:
                best_ba = ba
                best_th = th
                
        p_wo = m.predict_proba(X_te_without)[:, 1]
        p_w = m.predict_proba(X_te_with)[:, 1]
        
        r_wo, _ = evaluate_metrics(p_wo, y_te, best_th, df_te)
        r_w, _ = evaluate_metrics(p_w, y_te, best_th, df_te)
        
        delta_auc = r_w["AUROC"] - r_wo["AUROC"]
        delta_prc = r_w["AUPRC"] - r_wo["AUPRC"]
        abs_delta = np.abs(p_w - p_wo)
        flips = (p_w >= best_th) != (p_wo >= best_th)
        
        summary_rows.append({
            "Model": mname,
            "Mode": "Without Baseline",
            "AUROC": r_wo["AUROC"],
            "AUPRC": r_wo["AUPRC"],
            "Balanced_Acc": r_wo["Balanced_Acc"],
            "Sensitivity": r_wo["Sensitivity"],
            "Specificity": r_wo["Specificity"],
            "FPR_Clean": r_wo["FPR_Clean"],
            "Mean_Delta_P": abs_delta.mean(),
            "Flips": flips.sum(),
        })
        summary_rows.append({
            "Model": mname,
            "Mode": "With Baseline",
            "AUROC": r_w["AUROC"],
            "AUPRC": r_w["AUPRC"],
            "Balanced_Acc": r_w["Balanced_Acc"],
            "Sensitivity": r_w["Sensitivity"],
            "Specificity": r_w["Specificity"],
            "FPR_Clean": r_w["FPR_Clean"],
            "Delta_AUROC": delta_auc,
            "Delta_AUPRC": delta_prc,
        })
        
    res_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("BUDGET = 50 COMPARISON RESULTS")
    print("=" * 80)
    print(res_df.to_string(index=False))
    res_df.to_csv(out_dir / "interaction_budget_50_comparison.csv", index=False)
    
    # Save explicit 50 model
    joblib.dump(ebm_exp50, out_dir / "unified_explicit_50_ebm.joblib")
    print("\nSaved unified_explicit_50_ebm.joblib successfully!")

if __name__ == "__main__":
    main()
