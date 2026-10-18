#!/usr/bin/env python
"""Step 16: Unified Dual-Mode EBM Training & Evaluation (With vs Without Baseline).

1. Data Construction:
   - For every ECG clip, create two versions:
     * Mode With Baseline: [X_cur, X_prev, baseline_available=1], label y
     * Mode Without Baseline: [X_cur, Imputed_Median(X_prev), baseline_available=0], label y
   - Imputation: strictly computed on training set X_prev median.
2. Training:
   - Train a single unified ExplainableBoostingClassifier on the combined training set (2,222 rows).
   - Tune optimal threshold(s) on validation set.
3. Paired Testing:
   - On the 203 locked test ECG clips, run inference in both modes.
   - Evaluate:
     * Prediction Consistency: probability delta |P_with - P_without|, decision flip rate.
     * Performance Comparison: AUROC, AUPRC, Balanced Acc, Sens, Spec in both modes.
     * Explanation Consistency: local feature attribution shift for identical current ECG.
   - Render:
     * Probability scatter plot (P_without vs P_with).
     * Side-by-side Waterfall comparison on representative cases.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import joblib

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, confusion_matrix
)
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from step11_raw_concat_ab_experiment import build_raw_concat_data


def construct_dual_mode_datasets():
    splits, cur_feats, concat_feats = build_raw_concat_data()
    prev_feats = [c for c in concat_feats if c.startswith("prev_")]
    
    # 1. Compute training set median for previous baseline features
    X_tr_prev = splits["train"]["concat_X"][prev_feats]
    prev_medians = X_tr_prev.median()
    
    processed_splits = {}
    for sp in ["train", "val", "test"]:
        cur_df = splits[sp]["cur_X"].copy()
        prev_df = splits[sp]["concat_X"][prev_feats].copy()
        y = splits[sp]["y"]
        meta = splits[sp]["meta_df"]
        n = len(cur_df)
        
        # Mode 1: With Baseline
        df_with = pd.concat([cur_df, prev_df], axis=1)
        df_with["baseline_available"] = 1
        
        # Mode 2: Without Baseline (impute prev_feats with training median)
        df_without = cur_df.copy()
        for c in prev_feats:
            df_without[c] = prev_medians[c]
        df_without["baseline_available"] = 0
        
        # Match column order
        cols_order = list(df_with.columns)
        df_without = df_without[cols_order]
        
        processed_splits[sp] = {
            "df_with": df_with,
            "df_without": df_without,
            "y": y,
            "meta": meta,
            "cols_order": cols_order
        }
        
    return processed_splits, prev_medians, cols_order


def evaluate_metrics(probs, y, threshold, df_meta):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, preds)
    sens = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else np.nan
    spec = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else np.nan
    
    subtypes = ["neg_posture", "neg_arrhythmia", "neg_tachy", "neg_clean"]
    sub_fpr = {}
    for s in subtypes:
        m = (df_meta["subtype"] == s).values
        sub_fpr[s] = float(np.mean(preds[m] == 1)) if m.sum() > 0 else np.nan
        
    return {
        "AUROC": roc_auc_score(y, probs),
        "AUPRC": average_precision_score(y, probs),
        "Accuracy": accuracy_score(y, preds),
        "Balanced_Acc": balanced_accuracy_score(y, preds),
        "Sensitivity": sens,
        "Specificity": spec,
        "F1": f1_score(y, preds, zero_division=0),
        "Threshold": threshold,
        "FPR_Posture": sub_fpr.get("neg_posture", 0.0),
        "FPR_Arrhythmia": sub_fpr.get("neg_arrhythmia", 0.0),
        "FPR_Tachycardia": sub_fpr.get("neg_tachy", 0.0),
        "FPR_Clean": sub_fpr.get("neg_clean", 0.0)
    }, preds


def plot_consistency_scatter(test_df, th_unified, save_path):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    
    pos_mask = test_df["ground_truth"] == 1
    neg_mask = test_df["ground_truth"] == 0
    
    # Scatter points
    ax.scatter(
        test_df.loc[neg_mask, "prob_without"],
        test_df.loc[neg_mask, "prob_with"],
        color="#2980b9", alpha=0.65, edgecolors="none", s=45, label="Ground Truth = Non-Ischemia (0)"
    )
    ax.scatter(
        test_df.loc[pos_mask, "prob_without"],
        test_df.loc[pos_mask, "prob_with"],
        color="#e74c3c", alpha=0.85, edgecolors="black", linewidth=0.5, s=60, label="Ground Truth = Ischemia (1)"
    )
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.7, label="Equal Probability (y = x)")
    
    # Decision threshold lines
    ax.axvline(th_unified, color="#7f8c8d", linestyle=":", lw=1.2, label=f"Threshold = {th_unified:.3f}")
    ax.axhline(th_unified, color="#7f8c8d", linestyle=":", lw=1.2)
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Predicted Probability WITHOUT Baseline (Model Mode: Single/首诊)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Probability WITH Baseline (Model Mode: Serial/动态)", fontsize=11, fontweight="bold")
    ax.set_title("Prediction Consistency: Identical ECG Tested With vs Without Baseline\n(Single Unified EBM Model)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved probability scatter plot to: {save_path}")


def plot_paired_waterfall_same_ecg(
    row, exp_without_data, exp_with_data, intercept, th, save_path
):
    """Render side-by-side Waterfall charts for the SAME ECG under two input modes."""
    fig, (ax_wo, ax_w) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=200)
    
    def render_bar(ax, exp_data, title, prob, pred, is_with):
        scores = exp_data["scores"]
        names = exp_data["names"]
        vals = exp_data["values"]
        
        tuples = sorted(zip(names, scores, vals), key=lambda x: abs(x[1]), reverse=True)
        top_k = tuples[:9]
        other_sum = sum(x[1] for x in tuples[9:])
        
        items = [("Base Intercept", intercept, None)] + top_k
        if abs(other_sum) > 0.001:
            items.append(("Other Residual Features", other_sum, None))
            
        y_labels = []
        bar_scores = []
        bar_colors = []
        
        for name, score, val in items:
            if name == "Base Intercept":
                y_labels.append("Base Intercept")
                bar_colors.append("#7f8c8d")
            elif name == "Other Residual Features":
                y_labels.append(name)
                bar_colors.append("#95a5a6")
            else:
                val_str = f" = {val:.2f}" if isinstance(val, (int, float, np.number)) else ""
                y_labels.append(f"{name}{val_str}")
                bar_colors.append("#e74c3c" if score > 0 else "#2980b9")
            bar_scores.append(score)
            
        y_pos = np.arange(len(y_labels))[::-1]
        bars = ax.barh(y_pos, bar_scores, color=bar_colors, alpha=0.88, edgecolor="black", height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=8.5)
        ax.axvline(0, color="black", linestyle="-", lw=1.0)
        
        for bar, score in zip(bars, bar_scores):
            w = bar.get_width()
            align = "left" if w >= 0 else "right"
            offset = 0.02 if w >= 0 else -0.02
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{score:+.3f}",
                    va="center", ha=align, fontsize=8, fontweight="bold")
                    
        total_logit = intercept + sum(scores)
        verdict = "POSITIVE (Ischemia)" if pred == 1 else "NEGATIVE (Normal)"
        verdict_color = "#c0392b" if pred == 1 else "#27ae60"
        
        ax.set_title(
            f"[{title}]\n"
            f"Logit: {total_logit:+.3f} | Pred Prob: {prob:.3f} (Threshold: {th:.3f})\n"
            f"Decision: {verdict}",
            fontsize=10.5, fontweight="bold", color=verdict_color, pad=8
        )
        ax.set_xlabel("Contribution to Log-Odds (Red: +Ischemia, Blue: -Normal)", fontsize=9.5)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        
        x_min, x_max = ax.get_xlim()
        span = x_max - x_min
        ax.set_xlim([x_min - span * 0.15, x_max + span * 0.15])

    render_bar(ax_wo, exp_without_data, "MODE WITHOUT BASELINE (Single / Imputed)", row["prob_without"], row["pred_without"], False)
    render_bar(ax_w, exp_with_data, "MODE WITH BASELINE (Serial / Real Baseline)", row["prob_with"], row["pred_with"], True)
    
    gt_str = "POSITIVE (Ischemia)" if row["ground_truth"] == 1 else "NEGATIVE (Non-Ischemic)"
    plt.suptitle(
        f"SAME ECG EXPLANATION CONSISTENCY: {row['pair_id']} (Patient {row['patient_id']} - {row['record']})\n"
        f"Time: {row['cur_time_hhmmss']} | Subtype: {row['subtype']} | Ground Truth: {gt_str}",
        fontsize=12, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved paired waterfall to: {save_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs/models"
    wf_dir = out_dir / "waterfall"
    out_dir.mkdir(parents=True, exist_ok=True)
    wf_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STEP 16: UNIFIED DUAL-MODE EBM TRAINING & PAIRED TESTING")
    print("=" * 80)
    
    # 1. Construct Dual-Mode Data
    processed_splits, prev_medians, feature_cols = construct_dual_mode_datasets()
    print(f"Features: {len(feature_cols)} total (36 current + 36 baseline + 1 baseline_available)")
    
    # Build combined training set (with + without)
    X_tr_with = processed_splits["train"]["df_with"]
    X_tr_without = processed_splits["train"]["df_without"]
    y_tr = processed_splits["train"]["y"]
    
    X_tr_combined = pd.concat([X_tr_with, X_tr_without], axis=0).reset_index(drop=True)
    y_tr_combined = np.concatenate([y_tr, y_tr])
    print(f"Training set combined shape: {X_tr_combined.shape} (Positives: {y_tr_combined.sum()})")
    
    # Validation combined
    X_val_with = processed_splits["val"]["df_with"]
    X_val_without = processed_splits["val"]["df_without"]
    y_val = processed_splits["val"]["y"]
    X_val_combined = pd.concat([X_val_with, X_val_without], axis=0).reset_index(drop=True)
    y_val_combined = np.concatenate([y_val, y_val])
    
    # 2. Train Single Unified EBM
    print("\nTraining Unified Dual-Mode EBM...")
    ebm = ExplainableBoostingClassifier(
        interactions=15,
        learning_rate=0.03,
        max_leaves=3,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    ebm.fit(X_tr_combined, y_tr_combined)
    intercept = float(ebm.intercept_[0])
    
    # 3. Calibrate Thresholds on Validation
    val_probs_combined = ebm.predict_proba(X_val_combined)[:, 1]
    val_probs_with = ebm.predict_proba(X_val_with)[:, 1]
    val_probs_without = ebm.predict_proba(X_val_without)[:, 1]
    
    # Find unified optimal threshold
    best_th_unified, best_ba_unified = 0.5, -1.0
    for th in np.linspace(0.05, 0.95, 91):
        ba = balanced_accuracy_score(y_val_combined, (val_probs_combined >= th).astype(int))
        if ba > best_ba_unified:
            best_ba_unified = ba
            best_th_unified = th
            
    print(f"Validation Optimal Unified Threshold: {best_th_unified:.3f} (Val Balanced Acc: {best_ba_unified:.4f})")
    
    # 4. Paired Inference on Test Set (203 unique ECG clips)
    X_te_with = processed_splits["test"]["df_with"]
    X_te_without = processed_splits["test"]["df_without"]
    y_te = processed_splits["test"]["y"]
    df_te = processed_splits["test"]["meta"]
    n_test = len(y_te)
    
    probs_with = ebm.predict_proba(X_te_with)[:, 1]
    probs_without = ebm.predict_proba(X_te_without)[:, 1]
    
    # Evaluate metrics under unified threshold
    res_with, preds_with = evaluate_metrics(probs_with, y_te, best_th_unified, df_te)
    res_without, preds_without = evaluate_metrics(probs_without, y_te, best_th_unified, df_te)
    
    res_df = pd.DataFrame({
        "Unified EBM (Without Baseline)": res_without,
        "Unified EBM (With Baseline)": res_with,
    })
    res_df["Delta (With - Without)"] = res_df["Unified EBM (With Baseline)"] - res_df["Unified EBM (Without Baseline)"]
    
    print("\n" + "=" * 80)
    print("UNIFIED DUAL-MODE EBM TEST PERFORMANCE (SAME MODEL)")
    print("=" * 80)
    print(res_df.round(4).to_string())
    res_df.to_csv(out_dir / "unified_ebm_test_results.csv")
    
    # 5. Prediction Consistency Analysis
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
    
    # Flip categories
    def categorize_flip(r):
        pw, pwo, gt = r["pred_with"], r["pred_without"], r["ground_truth"]
        if pw == pwo:
            return "Concordant Decision"
        elif pwo == 0 and pw == 1:
            return "With-Baseline Alarms (0 -> 1)" + (" [Corrected Miss]" if gt==1 else " [New False Alarm]")
        else: # pwo == 1 and pw == 0
            return "With-Baseline Suppresses (1 -> 0)" + (" [Corrected False Alarm]" if gt==0 else " [Missed Ischemia]")
            
    test_paired["flip_category"] = test_paired.apply(categorize_flip, axis=1)
    test_paired.to_csv(out_dir / "unified_ebm_paired_predictions.csv", index=False)
    
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
        
    # Plot Scatter
    plot_consistency_scatter(test_paired, best_th_unified, out_dir / "unified_ebm_prob_scatter.png")
    
    # Save trained unified EBM model
    joblib.dump(ebm, out_dir / "unified_ebm.joblib")
    print(f"Saved unified EBM model to: {out_dir / 'unified_ebm.joblib'}")
    
    # Save Global Feature Importance
    exp_global = ebm.explain_global()
    global_names = exp_global.data()["names"]
    global_scores = exp_global.data()["scores"]
    global_df = pd.DataFrame({"feature": global_names, "importance": global_scores}).sort_values("importance", ascending=False)
    global_df.to_csv(out_dir / "unified_ebm_global_importance.csv", index=False)
    print("\n" + "=" * 80)
    print("TOP 15 GLOBAL FEATURES IN UNIFIED EBM:")
    print("=" * 80)
    print(global_df.head(15).to_string(index=False))

    # 6. Explanation Consistency: Generate Paired Waterfalls for Representative ECGs
    print("\nGenerating local explanations for paired waterfalls...")
    local_exp_without = ebm.explain_local(X_te_without)
    local_exp_with = ebm.explain_local(X_te_with)
    
    # Select illustrative cases:
    # 1. P21 pos_ST1- (Ischemia confidence boosted: 0.813 -> 0.909)
    # 2. P55 neg_clean (Massive false alarm suppressed by baseline: 0.676 -> 0.188, flipped 1 -> 0)
    # 3. P48 neg_posture (Posture artifact in baseline induces false alarm: 0.162 -> 0.613, flipped 0 -> 1)
    # 4. P31 neg_clean (Normal stable concordant: 0.008 -> 0.005)
    sample_cases = [
        {"pair_id": "PR_test_0004", "name": "unified_waterfall_case1_ischemia_boost_P21"},
        {"pair_id": "PR_test_0192", "name": "unified_waterfall_case2_corrected_false_alarm_P55"},
        {"pair_id": "PR_test_0179", "name": "unified_waterfall_case3_posture_new_false_alarm_P48"},
        {"pair_id": "PR_test_0108", "name": "unified_waterfall_case4_normal_stability_P31"}
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
                best_th_unified,
                wf_dir / f"{sc['name']}.png"
            )
            
    print("\nStep 16 unified dual-mode experiment completed successfully!")


if __name__ == "__main__":
    main()
