#!/usr/bin/env python
"""Step 8: Explainable Boosting Machine (EBM) A/B Testing & Evaluation.

Trains:
- Model A (EBM Single): Uses single-snapshot clinical features + absolute deformations
- Model B (EBM Dynamic): Uses paired current + delta features + pairwise interactions

Evaluates on Test Set:
- AUROC, AUPRC, Balanced Accuracy, Sensitivity, Specificity, F1
- Hard Negative Subtype Stress Tests: Posture, Arrhythmia, Tachycardia, Clean
- Extracts top global feature importance and interaction terms
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix
)

META_COLS = [
    "patient_id", "clip_id", "subtype", "is_pos", "split", "record",
    "start_sample", "cls", "with_arrhythmia", "peak_uv", "event_id",
    "event_role", "clip_source", "event_types", "n_ep_in_event",
    "ep_incomplete", "sig_names", "pair_id", "prev_start_sample",
    "prev_type", "prev_type_wanted_ok", "prev_gap_s"
]


def add_absolute_deformations(df):
    """Add absolute shape variations to help the trees split even faster."""
    df = df.copy()
    for c in range(2):
        st_col = f"cur_ch{c}_st_dev_mv" if f"cur_ch{c}_st_dev_mv" in df.columns else f"ch{c}_st_dev_mv"
        df[f"abs_{st_col}"] = np.abs(df[st_col])
        df[f"elev_{st_col}"] = np.maximum(0.0, df[st_col])
        df[f"dep_{st_col}"] = np.maximum(0.0, -df[st_col])
        
        t_col = f"cur_ch{c}_t_amp_mv" if f"cur_ch{c}_t_amp_mv" in df.columns else f"ch{c}_t_amp_mv"
        df[f"abs_{t_col}"] = np.abs(df[t_col])
        df[f"inv_{t_col}"] = np.maximum(0.0, -df[t_col])

    for c in range(2):
        dst_col = f"delta_ch{c}_st_dev_mv"
        if dst_col in df.columns:
            df[f"abs_{dst_col}"] = np.abs(df[dst_col])
            df[f"elev_{dst_col}"] = np.maximum(0.0, df[dst_col])
            df[f"dep_{dst_col}"] = np.maximum(0.0, -df[dst_col])
            
        dt_col = f"delta_ch{c}_t_amp_mv"
        if dt_col in df.columns:
            df[f"abs_{dt_col}"] = np.abs(df[dt_col])
            
        dr_col = f"delta_ch{c}_r_amp_mv"
        if dr_col in df.columns:
            df[f"abs_{dr_col}"] = np.abs(df[dr_col])
            
    return df


def load_data(mode="single"):
    root = Path(__file__).resolve().parents[1]
    feat_dir = root / "outputs/features"
    
    tr_df = add_absolute_deformations(pd.read_csv(feat_dir / f"{mode}_train_features.csv"))
    val_df = add_absolute_deformations(pd.read_csv(feat_dir / f"{mode}_val_features.csv"))
    te_df = add_absolute_deformations(pd.read_csv(feat_dir / f"{mode}_test_features.csv"))
    
    feat_cols = [c for c in tr_df.columns if c not in META_COLS]
    
    X_tr = tr_df[feat_cols]
    y_tr = tr_df["is_pos"].values
    
    X_val = val_df[feat_cols]
    y_val = val_df["is_pos"].values
    
    X_te = te_df[feat_cols]
    y_te = te_df["is_pos"].values
    
    return (X_tr, y_tr, tr_df), (X_val, y_val, val_df), (X_te, y_te, te_df), feat_cols


def train_ebm_with_tuning(X_tr, y_tr, X_val, y_val, interactions=10, random_state=42):
    """Train Explainable Boosting Machine and tune threshold on validation set."""
    ebm = ExplainableBoostingClassifier(
        interactions=interactions,
        learning_rate=0.03,
        max_leaves=3,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )
    ebm.fit(X_tr, y_tr)
    
    # Tune threshold on validation set for Balanced Accuracy
    val_probs = ebm.predict_proba(X_val)[:, 1]
    best_th = 0.5
    best_ba = -1.0
    for th in np.linspace(0.1, 0.9, 81):
        preds = (val_probs >= th).astype(int)
        ba = balanced_accuracy_score(y_val, preds)
        if ba > best_ba:
            best_ba = ba
            best_th = th
            
    val_auroc = roc_auc_score(y_val, val_probs)
    val_auprc = average_precision_score(y_val, val_probs)
    return ebm, best_th, best_ba, val_auroc, val_auprc


def evaluate_test(ebm, threshold, X_te, y_te, te_df):
    probs = ebm.predict_proba(X_te)[:, 1]
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
    print("STEP 8: EXPLAINABLE BOOSTING MACHINE (EBM) A/B BENCHMARK")
    print("=" * 70)
    
    # 1. Model A: EBM Single-snapshot
    print("\n--- Training Model A: EBM Single-Snapshot (36 base + deformations) ---")
    (X_tr_a, y_tr_a, df_tr_a), (X_val_a, y_val_a, df_val_a), (X_te_a, y_te_a, df_te_a), feat_a = load_data("single")
    ebm_a, th_a, val_ba_a, val_auc_a, val_pr_a = train_ebm_with_tuning(X_tr_a, y_tr_a, X_val_a, y_val_a, interactions=5)
    print(f"  Model A tuned threshold: {th_a:.3f} (Val Balanced Acc: {val_ba_a:.4f}, Val AUROC: {val_auc_a:.4f})")
    res_a, probs_a, preds_a = evaluate_test(ebm_a, th_a, X_te_a, y_te_a, df_te_a)
    
    # 2. Model B: EBM Dynamic-Alert with Pairwise Interactions
    print("\n--- Training Model B: EBM Dynamic-Alert (Current + Delta + Interactions) ---")
    (X_tr_b, y_tr_b, df_tr_b), (X_val_b, y_val_b, df_val_b), (X_te_b, y_te_b, df_te_b), feat_b = load_data("paired")
    ebm_b, th_b, val_ba_b, val_auc_b, val_pr_b = train_ebm_with_tuning(X_tr_b, y_tr_b, X_val_b, y_val_b, interactions=15)
    print(f"  Model B tuned threshold: {th_b:.3f} (Val Balanced Acc: {val_ba_b:.4f}, Val AUROC: {val_auc_b:.4f})")
    res_b, probs_b, preds_b = evaluate_test(ebm_b, th_b, X_te_b, y_te_b, df_te_b)
    
    # 3. Benchmark Comparison Table
    cmp_df = pd.DataFrame([res_a, res_b], index=["EBM Model A (Single)", "EBM Model B (Paired Dynamic)"]).T
    cmp_df["Absolute Gain (B - A)"] = cmp_df["EBM Model B (Paired Dynamic)"] - cmp_df["EBM Model A (Single)"]
    
    print("\n" + "=" * 70)
    print("EBM TEST-SET BENCHMARK RESULTS (WHITE-BOX GA2M)")
    print("=" * 70)
    print(cmp_df.round(4).to_string())
    
    # 4. Feature and Interaction Term Importance
    print("\n" + "=" * 70)
    print("TOP GLOBAL FEATURE & INTERACTION IMPORTANCES (MODEL B)")
    print("=" * 70)
    global_exp = ebm_b.explain_global()
    imp_df = pd.DataFrame({
        "Feature / Interaction": global_exp.data()["names"],
        "Importance (Mean Abs Score)": global_exp.data()["scores"]
    }).sort_values(by="Importance (Mean Abs Score)", ascending=False).reset_index(drop=True)
    
    print(imp_df.head(15).round(4).to_string())
    
    # Save results
    cmp_df.to_csv(out_dir / "ebm_ab_test_results.csv")
    imp_df.to_csv(out_dir / "ebm_model_b_feature_importance.csv", index=False)
    
    # Report
    md_lines = [
        "# Explainable Boosting Machine (EBM) A/B Test Benchmark Report\n",
        "## 1. Test-Set Performance Comparison\n",
        cmp_df.round(4).to_markdown(),
        "\n## 2. Top 15 Feature & Interaction Drivers in Dynamic Model B\n",
        imp_df.head(15).round(4).to_markdown(index=False),
        "\n## 3. Key Clinical Findings\n",
        f"- **AUROC**: Model B achieved {res_b['AUROC']:.4f} vs Model A {res_a['AUROC']:.4f} (Gain: {res_b['AUROC']-res_a['AUROC']:+.4f})",
        f"- **AUPRC**: Model B achieved {res_b['AUPRC']:.4f} vs Model A {res_a['AUPRC']:.4f} (Gain: {res_b['AUPRC']-res_a['AUPRC']:+.4f})",
        f"- **Balanced Accuracy**: Model B achieved {res_b['Balanced_Acc']:.4f} vs Model A {res_a['Balanced_Acc']:.4f} (Gain: {res_b['Balanced_Acc']-res_a['Balanced_Acc']:+.4f})",
        f"- **Posture False Positive Rate (体位误报率)**: Model A was {res_a['FPR_Posture']*100:.1f}%, Model B was {res_b['FPR_Posture']*100:.1f}%!",
        f"- **Arrhythmia False Positive Rate (早搏误报率)**: Model A was {res_a['FPR_Arrhythmia']*100:.1f}%, Model B was {res_b['FPR_Arrhythmia']*100:.1f}%!",
        f"- **Tachycardia False Positive Rate (心动过速误报率)**: Model A was {res_a['FPR_Tachycardia']*100:.1f}%, Model B was {res_b['FPR_Tachycardia']*100:.1f}%!"
    ]
    report_file = out_dir / "ebm_ab_test_report.md"
    report_file.write_text("\n".join(md_lines))
    print(f"\nDetailed EBM report written to: {report_file}")


if __name__ == "__main__":
    main()
