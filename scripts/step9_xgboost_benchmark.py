#!/usr/bin/env python
"""Step 9: XGBoost Benchmark & Full Hyperparameter Tuning on identical data.

Pipeline:
1. Loads exact same features + absolute deformations as Logistic Regression & EBM.
2. Grid search / Validation Tuning for XGBoost:
   - max_depth (3, 5, 7)
   - learning_rate (0.01, 0.03, 0.1)
   - scale_pos_weight (1.0, 3.0)
   - colsample_bytree & subsample (0.8, 1.0)
   - Optimal decision threshold calibration on Validation set for Balanced Accuracy.
3. Test Set Evaluation:
   - AUROC, AUPRC, Balanced Accuracy, Sensitivity, Specificity, F1
   - Hard Negative Subtype Stress Tests: Posture, Arrhythmia, Tachycardia, Clean
4. Global Feature Importance extraction.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix
)

# Reuse exact data loader from step8
from step8_ebm_ab_test import load_data


def tune_and_train_xgb(X_tr, y_tr, X_val, y_val, random_state=42):
    """Grid search over key XGBoost hyperparameters using Validation Balanced Accuracy."""
    param_grid = [
        {"max_depth": 3, "learning_rate": 0.03, "scale_pos_weight": 3.0, "subsample": 0.8},
        {"max_depth": 3, "learning_rate": 0.05, "scale_pos_weight": 1.0, "subsample": 0.8},
        {"max_depth": 4, "learning_rate": 0.03, "scale_pos_weight": 3.0, "subsample": 0.8},
        {"max_depth": 4, "learning_rate": 0.05, "scale_pos_weight": 1.0, "subsample": 0.8},
        {"max_depth": 5, "learning_rate": 0.03, "scale_pos_weight": 3.0, "subsample": 0.8},
        {"max_depth": 5, "learning_rate": 0.05, "scale_pos_weight": 1.0, "subsample": 0.8},
        {"max_depth": 3, "learning_rate": 0.10, "scale_pos_weight": 2.5, "subsample": 1.0},
        {"max_depth": 4, "learning_rate": 0.10, "scale_pos_weight": 1.0, "subsample": 1.0},
    ]
    
    best_score = -1.0
    best_params = None
    best_model = None
    best_thresh = 0.5
    
    for p in param_grid:
        clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            scale_pos_weight=p["scale_pos_weight"],
            subsample=p["subsample"],
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss"
        )
        clf.fit(X_tr, y_tr)
        val_probs = clf.predict_proba(X_val)[:, 1]
        
        # Tune threshold for Balanced Accuracy on Validation Set
        for th in np.linspace(0.1, 0.9, 81):
            preds = (val_probs >= th).astype(int)
            ba = balanced_accuracy_score(y_val, preds)
            if ba > best_score:
                best_score = ba
                best_params = p
                best_thresh = th
                best_model = clf
                
    return best_model, best_params, best_thresh, best_score


def evaluate_test(model, threshold, X_te, y_te, te_df):
    probs = model.predict_proba(X_te)[:, 1]
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
    
    print("=" * 70)
    print("STEP 9: XGBOOST A/B BENCHMARK (IDENTICAL DATA & TUNING)")
    print("=" * 70)
    
    # 1. Model A: Single
    print("\n--- Training Model A: XGBoost Single-Snapshot ---")
    (X_tr_a, y_tr_a, df_tr_a), (X_val_a, y_val_a, df_val_a), (X_te_a, y_te_a, df_te_a), feat_a = load_data("single")
    model_a, params_a, th_a, val_ba_a = tune_and_train_xgb(X_tr_a, y_tr_a, X_val_a, y_val_a)
    print(f"  Best Model A params: {params_a}, threshold: {th_a:.3f} (Val Balanced Acc: {val_ba_a:.4f})")
    res_a, probs_a, preds_a = evaluate_test(model_a, th_a, X_te_a, y_te_a, df_te_a)
    
    # 2. Model B: Paired Dynamic
    print("\n--- Training Model B: XGBoost Dynamic-Alert (with Previous) ---")
    (X_tr_b, y_tr_b, df_tr_b), (X_val_b, y_val_b, df_val_b), (X_te_b, y_te_b, df_te_b), feat_b = load_data("paired")
    model_b, params_b, th_b, val_ba_b = tune_and_train_xgb(X_tr_b, y_tr_b, X_val_b, y_val_b)
    print(f"  Best Model B params: {params_b}, threshold: {th_b:.3f} (Val Balanced Acc: {val_ba_b:.4f})")
    res_b, probs_b, preds_b = evaluate_test(model_b, th_b, X_te_b, y_te_b, df_te_b)
    
    # 3. Benchmark Comparison Table
    cmp_df = pd.DataFrame([res_a, res_b], index=["XGB Model A (Single)", "XGB Model B (Paired Dynamic)"]).T
    cmp_df["Absolute Gain (B - A)"] = cmp_df["XGB Model B (Paired Dynamic)"] - cmp_df["XGB Model A (Single)"]
    
    print("\n" + "=" * 70)
    print("XGBOOST TEST-SET BENCHMARK RESULTS")
    print("=" * 70)
    print(cmp_df.round(4).to_string())
    
    # 4. Top Feature Importances (Model B)
    print("\n" + "=" * 70)
    print("TOP 15 MOST INFLUENTIAL FEATURES IN XGBOOST (MODEL B)")
    print("=" * 70)
    imp_series = pd.Series(model_b.feature_importances_, index=feat_b).sort_values(ascending=False)
    print(imp_series.head(15).round(4).to_string())
    
    # Save
    cmp_df.to_csv(out_dir / "xgboost_ab_test_results.csv")
    imp_series.to_frame(name="Gain").to_csv(out_dir / "xgboost_feature_importance.csv")
    print("\nBenchmark saved successfully!")


if __name__ == "__main__":
    main()
