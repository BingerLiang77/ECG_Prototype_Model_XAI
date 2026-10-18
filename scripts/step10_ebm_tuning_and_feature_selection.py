#!/usr/bin/env python
"""Step 10: Systematic Feature Selection & Hyperparameter Tuning for EBM.

Pipeline:
1. Feature Selection (conducted strictly on Train split):
   - Redundancy filter (drop collinear features with |r| > 0.98).
   - ExtraTrees / Mutual-Info importance ranking on Train set.
   - Compare top-K candidate feature sets (e.g. 30, 45, 60, all) on Validation Balanced Accuracy.
2. Hyperparameter Grid Tuning (on Validation set):
   - interactions: [5, 10, 15, 25]
   - learning_rate: [0.015, 0.03, 0.05]
   - max_leaves: [2, 3]
   - min_samples_leaf: [5, 10]
   - outer_bags: [8, 14]
3. Validation Decision Threshold Calibration:
   - Sweep threshold in [0.05, 0.95] for optimal Balanced Accuracy.
4. Test-set Benchmarking & Side-by-side comparison with Logistic Regression & XGBoost.
5. Export global importance and top interaction terms.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import time
from itertools import product
warnings.filterwarnings("ignore")

from sklearn.ensemble import ExtraTreesClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix
)

from step8_ebm_ab_test import load_data, META_COLS


def select_features_by_importance(X_tr, y_tr, top_k=45, corr_thresh=0.98):
    """Filter collinear features and rank remaining by ExtraTrees importance on Train."""
    # 1. Collinearity filter
    corr = X_tr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > corr_thresh)]
    cols_kept = [c for c in X_tr.columns if c not in drop_cols]
    
    # 2. Importance ranking on Train only
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    clf.fit(X_tr[cols_kept], y_tr)
    imp = pd.Series(clf.feature_importances_, index=cols_kept).sort_values(ascending=False)
    
    selected = imp.head(top_k).index.tolist()
    return selected, imp


def find_best_threshold(val_probs, y_val):
    best_th = 0.5
    best_ba = -1.0
    for th in np.linspace(0.05, 0.95, 91):
        preds = (val_probs >= th).astype(int)
        ba = balanced_accuracy_score(y_val, preds)
        if ba > best_ba:
            best_ba = ba
            best_th = th
    return best_th, best_ba


def tune_ebm(X_tr, y_tr, X_val, y_val, feature_k_list, param_grid, random_state=42):
    """Grid search across feature set sizes and tree hyperparameters using Validation Balanced Acc."""
    trials = []
    best_trial = None
    best_score = -1.0
    best_model = None
    best_th = 0.5
    best_feats = None

    for k in feature_k_list:
        if k is None or k >= X_tr.shape[1]:
            cur_feats = X_tr.columns.tolist()
            feat_label = f"all_{len(cur_feats)}"
        else:
            cur_feats, _ = select_features_by_importance(X_tr, y_tr, top_k=k)
            feat_label = f"top_{k}"
            
        X_tr_sub = X_tr[cur_feats]
        X_val_sub = X_val[cur_feats]
        
        for params in param_grid:
            ebm = ExplainableBoostingClassifier(
                interactions=params["interactions"],
                learning_rate=params["learning_rate"],
                max_leaves=params["max_leaves"],
                min_samples_leaf=params["min_samples_leaf"],
                outer_bags=params.get("outer_bags", 8),
                random_state=random_state,
                n_jobs=-1
            )
            t0 = time.time()
            ebm.fit(X_tr_sub, y_tr)
            fit_time = time.time() - t0
            
            val_probs = ebm.predict_proba(X_val_sub)[:, 1]
            val_auroc = roc_auc_score(y_val, val_probs)
            val_auprc = average_precision_score(y_val, val_probs)
            th, val_ba = find_best_threshold(val_probs, y_val)
            
            record = {
                "feat_label": feat_label,
                "n_feats": len(cur_feats),
                **params,
                "val_ba": val_ba,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
                "best_threshold": th,
                "fit_time_s": fit_time
            }
            trials.append(record)
            
            if val_ba > best_score:
                best_score = val_ba
                best_trial = record
                best_model = ebm
                best_th = th
                best_feats = cur_feats
                
    return best_model, best_trial, best_feats, best_th, pd.DataFrame(trials)


def evaluate_test(model, threshold, features, X_te, y_te, te_df):
    X_sub = X_te[features]
    probs = model.predict_proba(X_sub)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    auroc = roc_auc_score(y_te, probs)
    auprc = average_precision_score(y_te, probs)
    acc = accuracy_score(y_te, preds)
    ba = balanced_accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, zero_division=0)
    
    cm = confusion_matrix(y_te, preds)
    sens = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    
    subtypes = ["neg_posture", "neg_arrhythmia", "neg_tachy", "neg_clean"]
    sub_fpr = {}
    for s in subtypes:
        m = (te_df["subtype"] == s).values
        sub_fpr[s] = float(np.mean(preds[m] == 1)) if m.sum() > 0 else np.nan
        
    return {
        "AUROC": auroc, "AUPRC": auprc, "Accuracy": acc, "Sensitivity": sens,
        "Specificity": spec, "Balanced_Acc": ba, "F1": f1,
        "FPR_Posture": sub_fpr.get("neg_posture", 0.0),
        "FPR_Arrhythmia": sub_fpr.get("neg_arrhythmia", 0.0),
        "FPR_Tachycardia": sub_fpr.get("neg_tachy", 0.0),
        "FPR_Clean": sub_fpr.get("neg_clean", 0.0)
    }, probs, preds


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs/models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 75)
    print("STEP 10: EBM SYSTEMATIC FEATURE SELECTION & HYPERPARAMETER TUNING")
    print("=" * 75)
    
    # ----------------------------------------------------
    # Hyperparameter Grid Definition
    # ----------------------------------------------------
    param_grid = [
        {"interactions": 5, "learning_rate": 0.03, "max_leaves": 3, "min_samples_leaf": 10, "outer_bags": 8},
        {"interactions": 10, "learning_rate": 0.03, "max_leaves": 3, "min_samples_leaf": 10, "outer_bags": 8},
        {"interactions": 15, "learning_rate": 0.03, "max_leaves": 3, "min_samples_leaf": 10, "outer_bags": 8},
        {"interactions": 20, "learning_rate": 0.03, "max_leaves": 3, "min_samples_leaf": 10, "outer_bags": 8},
        {"interactions": 10, "learning_rate": 0.015, "max_leaves": 3, "min_samples_leaf": 10, "outer_bags": 8},
        {"interactions": 15, "learning_rate": 0.015, "max_leaves": 3, "min_samples_leaf": 10, "outer_bags": 8},
        {"interactions": 10, "learning_rate": 0.05, "max_leaves": 3, "min_samples_leaf": 10, "outer_bags": 8},
        {"interactions": 10, "learning_rate": 0.03, "max_leaves": 2, "min_samples_leaf": 10, "outer_bags": 8},
        {"interactions": 15, "learning_rate": 0.03, "max_leaves": 2, "min_samples_leaf": 10, "outer_bags": 8},
        {"interactions": 10, "learning_rate": 0.03, "max_leaves": 3, "min_samples_leaf": 5, "outer_bags": 8},
        {"interactions": 15, "learning_rate": 0.03, "max_leaves": 3, "min_samples_leaf": 5, "outer_bags": 8},
        {"interactions": 15, "learning_rate": 0.03, "max_leaves": 3, "min_samples_leaf": 10, "outer_bags": 14},
    ]
    
    # ----------------------------------------------------
    # 1. Tuning Model A (Single Snapshot)
    # ----------------------------------------------------
    print("\n>>> TUNING MODEL A (SINGLE SNAPSHOT)...")
    (X_tr_a, y_tr_a, df_tr_a), (X_val_a, y_val_a, df_val_a), (X_te_a, y_te_a, df_te_a), feat_a = load_data("single")
    feature_k_a = [25, 35, None] # candidate subsets
    
    model_a, best_trial_a, best_feats_a, best_th_a, log_a = tune_ebm(
        X_tr_a, y_tr_a, X_val_a, y_val_a, feature_k_a, param_grid
    )
    print(f"  Best Model A: Feats={best_trial_a['feat_label']} ({best_trial_a['n_feats']}), "
          f"Interactions={best_trial_a['interactions']}, LR={best_trial_a['learning_rate']}, "
          f"Leaves={best_trial_a['max_leaves']}, MinLeaf={best_trial_a['min_samples_leaf']}, Bags={best_trial_a['outer_bags']}")
    print(f"  Val Balanced Acc: {best_trial_a['val_ba']:.4f}, Val AUROC: {best_trial_a['val_auroc']:.4f}, Thresh: {best_th_a:.3f}")
    
    res_a, probs_a, preds_a = evaluate_test(model_a, best_th_a, best_feats_a, X_te_a, y_te_a, df_te_a)
    
    # ----------------------------------------------------
    # 2. Tuning Model B (Paired Dynamic Alert)
    # ----------------------------------------------------
    print("\n>>> TUNING MODEL B (PAIRED DYNAMIC ALERT)...")
    (X_tr_b, y_tr_b, df_tr_b), (X_val_b, y_val_b, df_val_b), (X_te_b, y_te_b, df_te_b), feat_b = load_data("paired")
    feature_k_b = [30, 45, 60, None] # candidate subsets
    
    model_b, best_trial_b, best_feats_b, best_th_b, log_b = tune_ebm(
        X_tr_b, y_tr_b, X_val_b, y_val_b, feature_k_b, param_grid
    )
    print(f"  Best Model B: Feats={best_trial_b['feat_label']} ({best_trial_b['n_feats']}), "
          f"Interactions={best_trial_b['interactions']}, LR={best_trial_b['learning_rate']}, "
          f"Leaves={best_trial_b['max_leaves']}, MinLeaf={best_trial_b['min_samples_leaf']}, Bags={best_trial_b['outer_bags']}")
    print(f"  Val Balanced Acc: {best_trial_b['val_ba']:.4f}, Val AUROC: {best_trial_b['val_auroc']:.4f}, Thresh: {best_th_b:.3f}")
    
    res_b, probs_b, preds_b = evaluate_test(model_b, best_th_b, best_feats_b, X_te_b, y_te_b, df_te_b)
    
    # ----------------------------------------------------
    # 3. Test-Set Benchmark Table
    # ----------------------------------------------------
    cmp_df = pd.DataFrame([res_a, res_b], index=["Tuned EBM Model A (Single)", "Tuned EBM Model B (Paired Dynamic)"]).T
    cmp_df["Absolute Gain (B - A)"] = cmp_df["Tuned EBM Model B (Paired Dynamic)"] - cmp_df["Tuned EBM Model A (Single)"]
    
    print("\n" + "=" * 75)
    print("TUNED EBM TEST-SET BENCHMARK RESULTS")
    print("=" * 75)
    print(cmp_df.round(4).to_string())
    
    # ----------------------------------------------------
    # 4. Compare with Baseline Models (LR, Default EBM, XGBoost)
    # ----------------------------------------------------
    lr_res = pd.read_csv(out_dir / "ab_test_logistic_regression_results.csv", index_col=0)
    ebm_def_res = pd.read_csv(out_dir / "ebm_ab_test_results.csv", index_col=0)
    xgb_res = pd.read_csv(out_dir / "xgboost_ab_test_results.csv", index_col=0)
    
    bench_all = pd.DataFrame({
        "LR (Paired)": lr_res["Model B (Paired Dynamic)"],
        "EBM Default (Paired)": ebm_def_res["EBM Model B (Paired Dynamic)"],
        "EBM Tuned (Paired)": cmp_df["Tuned EBM Model B (Paired Dynamic)"],
        "XGBoost Tuned (Paired)": xgb_res["XGB Model B (Paired Dynamic)"]
    })
    
    print("\n" + "=" * 75)
    print("ALL-MODELS FINAL BENCHMARK ON MODEL B (PAIRED DYNAMIC ALERT)")
    print("=" * 75)
    print(bench_all.round(4).to_string())
    
    # ----------------------------------------------------
    # 5. Extract Feature Importance & Interactions
    # ----------------------------------------------------
    print("\n" + "=" * 75)
    print("TOP 15 MOST INFLUENTIAL TERMS IN TUNED EBM (MODEL B)")
    print("=" * 75)
    exp = model_b.explain_global()
    imp_df = pd.DataFrame({
        "term": exp.data()["names"],
        "importance": exp.data()["scores"]
    }).sort_values(by="importance", ascending=False)
    print(imp_df.head(15).to_string(index=False))
    
    # Save artifacts
    cmp_df.to_csv(out_dir / "ebm_tuned_ab_test_results.csv")
    bench_all.to_csv(out_dir / "ebm_comparison_all_models.csv")
    imp_df.to_csv(out_dir / "ebm_tuned_feature_importance.csv", index=False)
    log_a.to_csv(out_dir / "ebm_tuning_log_model_a.csv", index=False)
    log_b.to_csv(out_dir / "ebm_tuning_log_model_b.csv", index=False)
    
    print("\nTuned EBM benchmark finished and exported successfully!")


if __name__ == "__main__":
    main()
