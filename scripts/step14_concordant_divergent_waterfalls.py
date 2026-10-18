#!/usr/bin/env python
"""Step 14: Waterfall Analysis for Concordant Predictions with Large Probability Divergence.

Analyzes cases where Model A and Model B agree on the classification verdict,
but have massive probability gaps (|P_B - P_A| >= 0.40):
1. Confidence Surge on True Positives (A hesitant ~0.20 -> B confident ~0.80+)
2. Dramatic De-escalation on Posture/Artifact (A overreacting ~0.80 -> B calmed to ~0.27)
3. True Negative Reassurance (A borderline ~0.16 -> B completely safe ~0.01)

Renders multi-panel figures:
- Top: 10s Raw ECG Waveforms (Current vs Baseline, Lead 0 and Lead 1)
- Bottom: Exact EBM Additive Waterfall Charts for Model A and Model B
Exports CSV and publication-ready PNGs.
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


def render_waterfall(ax, exp_data, intercept, th, model_name, prob, pred):
    scores = exp_data["scores"]
    names = exp_data["names"]
    vals = exp_data["values"]
    
    term_tuples = sorted(zip(names, scores, vals), key=lambda x: abs(x[1]), reverse=True)
    top_k = term_tuples[:9]
    other_sum = sum(x[1] for x in term_tuples[9:])
    
    display_items = [("Base Intercept", intercept, None)] + top_k
    if abs(other_sum) > 0.001:
        display_items.append(("Other Residual Features", other_sum, None))
        
    y_labels = []
    bar_scores = []
    bar_colors = []
    
    for name, score, val in display_items:
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
    calc_prob = 1.0 / (1.0 + np.exp(-total_logit))
    verdict = "POSITIVE (Alarm)" if pred == 1 else "NEGATIVE (Normal)"
    verdict_color = "#c0392b" if pred == 1 else "#27ae60"
    
    ax.set_title(
        f"[{model_name} Local Explanation]\n"
        f"Logit: {total_logit:+.3f} | Pred Prob: {calc_prob:.3f} (Threshold: {th:.3f})\n"
        f"Final Decision: {verdict}",
        fontsize=10.5, fontweight="bold", color=verdict_color, pad=8
    )
    ax.set_xlabel("Contribution to Log-Odds (Red: +Ischemia, Blue: -Normal)", fontsize=9.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    
    x_min, x_max = ax.get_xlim()
    span = x_max - x_min
    ax.set_xlim([x_min - span * 0.15, x_max + span * 0.15])


def plot_concordant_waterfall(
    idx, row, cur_wave, prev_wave,
    exp_a_data, intercept_a, th_a,
    exp_b_data, intercept_b, th_b,
    scenario_desc, save_path
):
    fig = plt.figure(figsize=(18, 12), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.4], hspace=0.32, wspace=0.25)
    
    fs = 250
    t = np.arange(2500) / fs
    gt_str = "POSITIVE (Ischemia)" if row["ground_truth"] == 1 else "NEGATIVE (Non-Ischemic)"
    
    # 1. Top Left: Current Waveform
    ax_cur = fig.add_subplot(gs[0, 0])
    ax_cur.plot(t, cur_wave[0], color="#d35400", label="Lead 0 (Channel 0)", lw=1.2)
    ax_cur.plot(t, cur_wave[1], color="#8e44ad", label="Lead 1 (Channel 1)", lw=1.2)
    ax_cur.set_title(
        f"[CURRENT WINDOW] Clip: {row['clip_id']} | Time: {row['cur_time_hhmmss']} | Patient: {row['patient_id']} ({row['record']})\n"
        f"Subtype: {row['subtype']} | Ground Truth: {gt_str}",
        fontsize=11, fontweight="bold", pad=8
    )
    ax_cur.set_xlabel("Time (seconds)", fontsize=10)
    ax_cur.set_ylabel("Amplitude (mV)", fontsize=10)
    ax_cur.set_xlim([0, 10])
    ax_cur.grid(True, linestyle="--", alpha=0.5)
    ax_cur.legend(loc="upper right", fontsize=9)
    
    # 2. Top Right: Baseline Waveform
    ax_prev = fig.add_subplot(gs[0, 1])
    ax_prev.plot(t, prev_wave[0], color="#2980b9", label="Baseline Lead 0", lw=1.2)
    ax_prev.plot(t, prev_wave[1], color="#16a085", label="Baseline Lead 1", lw=1.2)
    ax_prev.set_title(
        f"[PREVIOUS BASELINE WINDOW] Time: {row['prev_time_hhmmss']} | Type: {row['prev_type']}\n"
        f"Time Elapsed Since Baseline: {row['prev_gap_s']:.0f} seconds ({row['prev_gap_s']/60:.1f} min)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax_prev.set_xlabel("Time (seconds)", fontsize=10)
    ax_prev.set_ylabel("Amplitude (mV)", fontsize=10)
    ax_prev.set_xlim([0, 10])
    ax_prev.grid(True, linestyle="--", alpha=0.5)
    ax_prev.legend(loc="upper right", fontsize=9)
    
    # 3. Bottom: Waterfalls
    ax_wf_a = fig.add_subplot(gs[1, 0])
    render_waterfall(ax_wf_a, exp_a_data, intercept_a, th_a, "Model A (Single Snapshot)", row["prob_A"], row["pred_A"])
    
    ax_wf_b = fig.add_subplot(gs[1, 1])
    render_waterfall(ax_wf_b, exp_b_data, intercept_b, th_b, "Model B (Concat with Baseline)", row["prob_B"], row["pred_B"])
    
    prob_delta = row["prob_B"] - row["prob_A"]
    plt.suptitle(
        f"CONCORDANT CASE WITH LARGE PROBABILITY DIVERGENCE: {row['pair_id']} (Patient {row['patient_id']} - {row['record']})\n"
        f"Scenario: {scenario_desc} | Prob A = {row['prob_A']:.3f} -> Prob B = {row['prob_B']:.3f} (Δ = {prob_delta:+.3f})",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs/models/waterfall"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STEP 14: CONCORDANT CASES WITH LARGE PROBABILITY DIVERGENCE")
    print("=" * 80)
    
    splits, cur_feats, concat_feats = build_raw_concat_data()
    tr_cur, y_tr = splits["train"]["cur_X"], splits["train"]["y"]
    val_cur, y_val = splits["val"]["cur_X"], splits["val"]["y"]
    te_cur, y_te = splits["test"]["cur_X"], splits["test"]["y"]
    
    tr_cat = splits["train"]["concat_X"]
    val_cat = splits["val"]["concat_X"]
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
    
    concordant = all_preds[all_preds["pred_A"] == all_preds["pred_B"]].copy()
    concordant["prob_diff"] = (concordant["prob_B"] - concordant["prob_A"]).abs()
    concordant.sort_values(by="prob_diff", ascending=False, inplace=True)
    
    # Save CSV
    concordant.to_csv(out_dir / "concordant_large_divergence_cases.csv", index=False)
    print(f"Saved {len(concordant)} concordant cases to {out_dir / 'concordant_large_divergence_cases.csv'}")
    
    local_exp_a = ebm_a.explain_local(te_cur)
    local_exp_b = ebm_b.explain_local(te_cat)
    
    # 4 Key Representative Scenarios:
    # 1. P08 pos_T0+: A barely alarms (0.199) -> B surges to 0.814 (+0.615)
    # 2. P48 neg_posture: A screams false alarm (0.780) -> B calms down to 0.275 (-0.505)
    # 3. P01 pos_ST1-: A moderate (0.487) -> B reaches extreme certainty 0.908 (+0.420)
    # 4. P35 neg_clean: A suspicious (0.164) -> B reassures to near zero 0.018 (-0.146)
    scenarios = [
        {
            "idx": 14,
            "filename": "concordant_case1_confidence_surge_P08.png",
            "desc": "True Positive: A Barely Crosses Threshold (0.199) -> B Surges to High Confidence (0.814)"
        },
        {
            "idx": 177,
            "filename": "concordant_case2_posture_deescalation_P48.png",
            "desc": "Posture Artifact: A Overreacts with High Alarm (0.780) -> B De-escalates towards Baseline (0.275)"
        },
        {
            "idx": 48,
            "filename": "concordant_case3_certainty_boost_P01.png",
            "desc": "Severe Ischemia: A Moderate Alarm (0.487) -> B Achieves 91% Certainty (0.908)"
        },
        {
            "idx": 134,
            "filename": "concordant_case4_clean_reassurance_P35.png",
            "desc": "Normal Clean: A Suspicious near Threshold (0.164) -> B Confirms Absolute Reassurance (0.018)"
        }
    ]
    
    for sc in scenarios:
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
        
    print("\nAll concordant waterfall charts generated successfully!")


if __name__ == "__main__":
    main()
