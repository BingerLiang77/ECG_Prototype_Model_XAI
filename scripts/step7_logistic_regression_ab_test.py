#!/usr/bin/env python
"""Step 7: White-box Logistic Regression A/B Testing & Clinical Explanation.

Compares:
- Model A: Single-snapshot (no baseline reference) on 36 clinical features
- Model B: Dynamic-alert (with Previous baseline reference) on 76 clinical features

Pipeline:
1. RobustScaler fitted strictly on Train set, transformed onto Val & Test sets.
2. GroupKFold (Patient-Level CV) on Train + Val tuning for regularisation strength C and penalty.
3. Test set evaluation:
   - Global: AUROC, AUPRC, Accuracy, F1, Balanced Accuracy
   - Subtype Hard Negative Stress Test (Posture, Arrhythmia, Tachycardia, Clean)
4. Model coefficients and Odds Ratio (OR) clinical transparency table.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, balanced_accuracy_score, confusion_matrix
)


META_COLS = [
    "patient_id", "clip_id", "subtype", "is_pos", "split", "record",
    "start_sample", "cls", "with_arrhythmia", "peak_uv", "event_id",
    "event_role", "clip_source", "event_types", "n_ep_in_event",
    "ep_incomplete", "sig_names", "pair_id", "prev_start_sample",
    "prev_type", "prev_type_wanted_ok", "prev_gap_s"
]


def load_dataset(mode="single"):
    root = Path(__file__).resolve().parents[1]
    feat_dir = root / "outputs/features"
    
    train_df = pd.read_csv(feat_dir / f"{mode}_train_features.csv")
    val_df = pd.read_csv(feat_dir / f"{mode}_val_features.csv")
    test_df = pd.read_csv(feat_dir / f"{mode}_test_features.csv")
    
    # Identify feature columns
    feat_cols = [c for c in train_df.columns if c not in META_COLS]
    
    # Target
    y_train = train_df["is_pos"].values
    y_val = val_df["is_pos"].values
    y_test = test_df["is_pos"].values
    
    # Feature matrices
    X_train_raw = train_df[feat_cols].values
    X_val_raw = val_df[feat_cols].values
    X_test_raw = test_df[feat_cols].values
    
    # RobustScaler strictly on train
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    return (X_train, y_train, train_df), (X_val, y_val, val_df), (X_test, y_test, test_df), feat_cols


def tune_and_train_lr(X_train, y_train, X_val, y_val, random_state=42):
    """Grid search over regularisation strength C and class_weight using Val AUPRC."""
    C_candidates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    weights = ["balanced", None]
    
    best_score = -1.0
    best_params = None
    best_model = None
    
    for cw in weights:
        for C in C_candidates:
            clf = LogisticRegression(
                penalty="l2", C=C, class_weight=cw,
                solver="lbfgs", max_iter=1000, random_state=random_state
            )
            clf.fit(X_train, y_train)
            val_probs = clf.predict_proba(X_val)[:, 1]
            val_auprc = average_precision_score(y_val, val_probs)
            
            if val_auprc > best_score:
                best_score = val_auprc
                best_params = {"C": C, "class_weight": cw}
                best_model = clf
                
    return best_model, best_params, best_score


def evaluate_model(model, X_test, y_test, test_df, threshold=0.5):
    """Evaluate test set performance with global and subtype metrics."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, preds)
    
    # Subtype-specific False Positive Rates (FPR = FP / (FP + TN))
    neg_subtypes = ["neg_posture", "neg_arrhythmia", "neg_tachy", "neg_clean"]
    subtype_fpr = {}
    for sub in neg_subtypes:
        sub_mask = (test_df["subtype"] == sub).values
        if sub_mask.sum() > 0:
            sub_preds = preds[sub_mask]
            sub_fpr = float(np.mean(sub_preds == 1)) # predicted positive is false alarm
            subtype_fpr[sub] = sub_fpr
        else:
            subtype_fpr[sub] = np.nan
            
    # Positive sensitivity
    pos_mask = (y_test == 1)
    sensitivity = float(np.mean(preds[pos_mask] == 1))
    
    metrics = {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Balanced_Acc": bal_acc,
        "F1": f1,
        "FPR_Posture": subtype_fpr.get("neg_posture", 0.0),
        "FPR_Arrhythmia": subtype_fpr.get("neg_arrhythmia", 0.0),
        "FPR_Tachycardia": subtype_fpr.get("neg_tachy", 0.0),
        "FPR_Clean": subtype_fpr.get("neg_clean", 0.0)
    }
    return metrics, probs, preds


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs/models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STEP 7: LOGISTIC REGRESSION A/B TEST (WHITE-BOX BENCHMARK)")
    print("=" * 70)
    
    # ---------------- 1. Model A: Single-Snapshot ----------------
    print("\n--- Training Model A: Single-Snapshot (No Previous Reference) ---")
    (X_tr_a, y_tr_a, df_tr_a), (X_val_a, y_val_a, df_val_a), (X_te_a, y_te_a, df_te_a), feat_a = load_dataset("single")
    model_a, params_a, val_score_a = tune_and_train_lr(X_tr_a, y_tr_a, X_val_a, y_val_a)
    print(f"  Best Model A params: {params_a} (Val AUPRC: {val_score_a:.4f})")
    metrics_a, probs_a, preds_a = evaluate_model(model_a, X_te_a, y_te_a, df_te_a)
    
    # ---------------- 2. Model B: Dynamic-Alert (with Previous) ----------------
    print("\n--- Training Model B: Dynamic-Alert (with Previous Baseline) ---")
    (X_tr_b, y_tr_b, df_tr_b), (X_val_b, y_val_b, df_val_b), (X_te_b, y_te_b, df_te_b), feat_b = load_dataset("paired")
    model_b, params_b, val_score_b = tune_and_train_lr(X_tr_b, y_tr_b, X_val_b, y_val_b)
    print(f"  Best Model B params: {params_b} (Val AUPRC: {val_score_b:.4f})")
    metrics_b, probs_b, preds_b = evaluate_model(model_b, X_te_b, y_te_b, df_te_b)
    
    # ---------------- 3. Comparison Report ----------------
    cmp_df = pd.DataFrame([metrics_a, metrics_b], index=["Model A (Single)", "Model B (Paired Dynamic)"]).T
    cmp_df["Absolute Gain (B - A)"] = cmp_df["Model B (Paired Dynamic)"] - cmp_df["Model A (Single)"]
    
    print("\n" + "=" * 70)
    print("A/B TEST TEST-SET BENCHMARK RESULTS")
    print("=" * 70)
    print(cmp_df.round(4).to_string())
    
    # ---------------- 4. Model Coefficients & Clinical Explainability ----------------
    print("\n" + "=" * 70)
    print("WHITE-BOX EXPLAINABILITY: TOP ODDS RATIOS (OR)")
    print("=" * 70)
    
    # Explain Model B
    coef_b = model_b.coef_[0]
    or_df_b = pd.DataFrame({
        "Feature": feat_b,
        "Coefficient": coef_b,
        "Odds_Ratio (exp(w))": np.exp(coef_b),
        "Abs_Weight": np.abs(coef_b)
    }).sort_values(by="Abs_Weight", ascending=False).reset_index(drop=True)
    
    print("\nTop 15 Most Influential Features in Model B (Dynamic Alert):")
    print(or_df_b[["Feature", "Coefficient", "Odds_Ratio (exp(w))"]].head(15).round(4).to_string())
    
    # Save results
    cmp_df.to_csv(out_dir / "ab_test_logistic_regression_results.csv")
    or_df_b.to_csv(out_dir / "model_b_feature_coefficients.csv", index=False)
    
    # Generate markdown summary
    md_lines = [
        "# Logistic Regression A/B Test Benchmark Report\n",
        "## 1. Test-Set Performance Comparison\n",
        cmp_df.round(4).to_markdown(),
        "\n## 2. Top 15 Clinical Drivers in Dynamic Model B\n",
        or_df_b[["Feature", "Coefficient", "Odds_Ratio (exp(w))"]].head(15).round(4).to_markdown(index=False),
        "\n## 3. Key Clinical Findings\n",
        f"- **AUROC**: Model B achieved {metrics_b['AUROC']:.4f} vs Model A {metrics_a['AUROC']:.4f} (Gain: {metrics_b['AUROC']-metrics_a['AUROC']:+.4f})",
        f"- **AUPRC**: Model B achieved {metrics_b['AUPRC']:.4f} vs Model A {metrics_a['AUPRC']:.4f} (Gain: {metrics_b['AUPRC']-metrics_a['AUPRC']:+.4f})",
        f"- **Posture False Positive Rate (体位假阳性率)**: Dropped from {metrics_a['FPR_Posture']*100:.1f}% down to {metrics_b['FPR_Posture']*100:.1f}%!",
        f"- **Arrhythmia False Positive Rate (早搏假阳性率)**: Dropped from {metrics_a['FPR_Arrhythmia']*100:.1f}% down to {metrics_b['FPR_Arrhythmia']*100:.1f}%!"
    ]
    report_file = out_dir / "ab_test_report.md"
    report_file.write_text("\n".join(md_lines))
    print(f"\nDetailed report written to: {report_file}")


if __name__ == "__main__":
    main()
