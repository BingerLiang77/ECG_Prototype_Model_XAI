#!/usr/bin/env python
"""Step 11: Raw Concat ECG (Current + Previous without manual differences) A/B Experiment.

Setup:
- Model A (Single Snapshot): Uses only Current ECG features (36 features).
- Model B (Raw Concat): Uses [Current ECG (36) + Previous Baseline ECG (36)] directly concatenated (72 features).
  NO manual difference features (delta_*), NO absolute deformations (abs_*), NO manual ratios.
- Models evaluated:
  1. Logistic Regression (balanced, threshold tuned on Val)
  2. Explainable Boosting Machine (EBM Tuned, interactions=15, threshold tuned on Val)
  3. XGBoost (Tuned Ensemble, threshold tuned on Val)
- Evaluates on locked Test set (12 patients):
  - AUROC, AUPRC, Accuracy, Balanced Accuracy, Sensitivity, Specificity, F1
  - Hard negative subtypes: Posture, Arrhythmia, Tachycardia, Clean
  - Contrast with previous Handcrafted Difference Model B.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from interpret.glassbox import ExplainableBoostingClassifier
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix
)


def build_raw_concat_data():
    root = Path(__file__).resolve().parents[1]
    feat_dir = root / "outputs/features"
    
    splits_data = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(feat_dir / f"paired_{split}_features.csv")
        base_names = [c[4:] for c in df.columns if c.startswith("cur_")]
        
        # Current ECG features (36)
        cur_cols = [f"cur_{k}" for k in base_names]
        cur_df = df[cur_cols].copy()
        
        # Previous Baseline ECG features (36) reconstructed from cur - delta
        prev_dict = {}
        for k in base_names:
            if f"delta_{k}" in df.columns:
                prev_dict[f"prev_{k}"] = df[f"cur_{k}"] - df[f"delta_{k}"]
            else:
                prev_dict[f"prev_{k}"] = 0.0
        prev_df = pd.DataFrame(prev_dict, index=df.index)
        
        # Concat features (72)
        concat_df = pd.concat([cur_df, prev_df], axis=1)
        
        y = df["is_pos"].values
        splits_data[split] = {
            "cur_X": cur_df,
            "concat_X": concat_df,
            "y": y,
            "meta_df": df
        }
    return splits_data, cur_cols, list(concat_df.columns)


def tune_and_eval(clf, X_tr, y_tr, X_val, y_val, X_te, y_te, df_te):
    clf.fit(X_tr, y_tr)
    v_probs = clf.predict_proba(X_val)[:, 1]
    t_probs = clf.predict_proba(X_te)[:, 1]
    
    best_th, best_ba = 0.5, -1.0
    for th in np.linspace(0.05, 0.95, 91):
        preds = (v_probs >= th).astype(int)
        ba = balanced_accuracy_score(y_val, preds)
        if ba > best_ba:
            best_ba = ba
            best_th = th
            
    t_preds = (t_probs >= best_th).astype(int)
    cm = confusion_matrix(y_te, t_preds)
    
    subtypes = ["neg_posture", "neg_arrhythmia", "neg_tachy", "neg_clean"]
    sub_fpr = {s: float(np.mean(t_preds[(df_te["subtype"] == s).values] == 1)) for s in subtypes}
    
    metrics = {
        "AUROC": roc_auc_score(y_te, t_probs),
        "AUPRC": average_precision_score(y_te, t_probs),
        "Accuracy": accuracy_score(y_te, t_preds),
        "Balanced_Acc": balanced_accuracy_score(y_te, t_preds),
        "Sensitivity": cm[1, 1] / cm[1].sum(),
        "Specificity": cm[0, 0] / cm[0].sum(),
        "F1": f1_score(y_te, t_preds, zero_division=0),
        "Threshold": best_th,
        "FPR_Posture": sub_fpr["neg_posture"],
        "FPR_Arrhythmia": sub_fpr["neg_arrhythmia"],
        "FPR_Tachycardia": sub_fpr["neg_tachy"],
        "FPR_Clean": sub_fpr["neg_clean"]
    }
    return metrics, clf, t_probs, t_preds


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs/models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STEP 11: RAW CONCAT EXPERIMENT (CURRENT + PREVIOUS WITHOUT MANUAL DELTA)")
    print("=" * 80)
    
    splits, cur_feats, concat_feats = build_raw_concat_data()
    print(f"Dataset ready: Current features = {len(cur_feats)}, Concat features = {len(concat_feats)}")
    
    tr_cur, tr_cat, y_tr = splits["train"]["cur_X"], splits["train"]["concat_X"], splits["train"]["y"]
    val_cur, val_cat, y_val = splits["val"]["cur_X"], splits["val"]["concat_X"], splits["val"]["y"]
    te_cur, te_cat, y_te = splits["test"]["cur_X"], splits["test"]["concat_X"], splits["test"]["y"]
    df_te = splits["test"]["meta_df"]
    
    results = {}
    
    # =========================================================================
    # 1. LOGISTIC REGRESSION
    # =========================================================================
    print("\n--- 1. Training Logistic Regression (Model A vs Model B Raw Concat) ---")
    pipe_lr_a = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42))])
    res_lr_a, _, _, _ = tune_and_eval(pipe_lr_a, tr_cur, y_tr, val_cur, y_val, te_cur, y_te, df_te)
    
    pipe_lr_b = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42))])
    res_lr_b, _, _, _ = tune_and_eval(pipe_lr_b, tr_cat, y_tr, val_cat, y_val, te_cat, y_te, df_te)
    
    results["LR_Model_A (Single)"] = res_lr_a
    results["LR_Model_B (Raw Concat)"] = res_lr_b
    
    # =========================================================================
    # 2. EXPLAINABLE BOOSTING MACHINE (EBM TUNED)
    # =========================================================================
    print("\n--- 2. Training EBM Tuned Glass-box (Model A vs Model B Raw Concat) ---")
    ebm_a = ExplainableBoostingClassifier(interactions=15, learning_rate=0.03, max_leaves=3, min_samples_leaf=10, random_state=42, n_jobs=-1)
    res_ebm_a, _, _, _ = tune_and_eval(ebm_a, tr_cur, y_tr, val_cur, y_val, te_cur, y_te, df_te)
    
    ebm_b = ExplainableBoostingClassifier(interactions=15, learning_rate=0.03, max_leaves=3, min_samples_leaf=10, random_state=42, n_jobs=-1)
    res_ebm_b, fitted_ebm_b, _, _ = tune_and_eval(ebm_b, tr_cat, y_tr, val_cat, y_val, te_cat, y_te, df_te)
    
    results["EBM_Model_A (Single)"] = res_ebm_a
    results["EBM_Model_B (Raw Concat)"] = res_ebm_b
    
    # =========================================================================
    # 3. XGBOOST (TUNED ENSEMBLE)
    # =========================================================================
    print("\n--- 3. Training XGBoost Tuned (Model A vs Model B Raw Concat) ---")
    xgb_a = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=1.0, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss")
    res_xgb_a, _, _, _ = tune_and_eval(xgb_a, tr_cur, y_tr, val_cur, y_val, te_cur, y_te, df_te)
    
    xgb_b = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, scale_pos_weight=1.0, subsample=1.0, colsample_bytree=0.8, random_state=42, eval_metric="logloss")
    res_xgb_b, _, _, _ = tune_and_eval(xgb_b, tr_cat, y_tr, val_cat, y_val, te_cat, y_te, df_te)
    
    results["XGB_Model_A (Single)"] = res_xgb_a
    results["XGB_Model_B (Raw Concat)"] = res_xgb_b
    
    # =========================================================================
    # 4. EXPORT & COMPARISON TABLE
    # =========================================================================
    res_df = pd.DataFrame(results)
    res_df["LR_Gain (B-A)"] = res_df["LR_Model_B (Raw Concat)"] - res_df["LR_Model_A (Single)"]
    res_df["EBM_Gain (B-A)"] = res_df["EBM_Model_B (Raw Concat)"] - res_df["EBM_Model_A (Single)"]
    res_df["XGB_Gain (B-A)"] = res_df["XGB_Model_B (Raw Concat)"] - res_df["XGB_Model_A (Single)"]
    
    print("\n" + "=" * 80)
    print("RAW CONCAT A/B TEST RESULTS (NO MANUAL DIFFERENCES)")
    print("=" * 80)
    print(res_df.round(4).to_string())
    
    res_df.to_csv(out_dir / "raw_concat_ab_results.csv")
    
    # Inspect EBM Concat Top Terms (check if it learned cur x prev interaction)
    exp = fitted_ebm_b.explain_global()
    terms = pd.DataFrame({
        "term": exp.data()["names"],
        "importance": exp.data()["scores"]
    }).sort_values("importance", ascending=False)
    terms.to_csv(out_dir / "ebm_raw_concat_feature_importance.csv", index=False)
    
    print("\nTop 15 Most Influential Terms in EBM (Raw Concat):")
    print(terms.head(15).to_string(index=False))
    print("\nExperiment finished successfully!")


if __name__ == "__main__":
    main()
