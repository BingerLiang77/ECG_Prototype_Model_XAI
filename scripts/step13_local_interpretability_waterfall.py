#!/usr/bin/env python
"""Step 13: Local Interpretability & Waterfall Decomposition for Discordant ECG Cases.

For cases where Model A and Model B disagree:
1. Reconstructs exact additive logit contributions (Waterfall) for both Model A and Model B.
2. Plots multi-panel clinical interpretability figures:
   - Top: Raw 10-second ECG Waveforms (Current window vs Baseline window, Lead 0 and Lead 1).
   - Bottom Left: Model A Local Explanation (Horizontal Waterfall Bar Chart).
   - Bottom Right: Model B Local Explanation (Horizontal Waterfall Bar Chart showing Cross-Temporal Interactions).
3. Exports high-resolution PNGs and a master CSV of local feature contributions for all 50 discordant cases.
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
from sklearn.metrics import balanced_accuracy_score
from step11_raw_concat_ab_experiment import build_raw_concat_data


def plot_paired_waterfall(
    idx, row, cur_wave, prev_wave,
    exp_a_data, intercept_a, th_a,
    exp_b_data, intercept_b, th_b,
    save_path
):
    """Generate a combined clinical visualization: Waveforms + Side-by-Side Waterfall Charts."""
    fig = plt.figure(figsize=(18, 12), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.4], hspace=0.32, wspace=0.25)
    
    fs = 250
    t = np.arange(2500) / fs
    
    gt_str = "POSITIVE (Ischemia)" if row["ground_truth"] == 1 else "NEGATIVE (Non-Ischemic)"
    
    # -------------------------------------------------------------
    # 1. Top Left: Current ECG Waveform
    # -------------------------------------------------------------
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
    
    # -------------------------------------------------------------
    # 2. Top Right: Previous Baseline ECG Waveform
    # -------------------------------------------------------------
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
    
    # -------------------------------------------------------------
    # Helper to render horizontal waterfall
    # -------------------------------------------------------------
    def render_waterfall(ax, exp_data, intercept, th, model_name, prob, pred):
        scores = exp_data["scores"]
        names = exp_data["names"]
        vals = exp_data["values"]
        
        # Sort terms by magnitude
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
        
        # Annotate bar values
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
        
        # Add some margin for text labels
        x_min, x_max = ax.get_xlim()
        span = x_max - x_min
        ax.set_xlim([x_min - span * 0.15, x_max + span * 0.15])

    # -------------------------------------------------------------
    # 3. Bottom Left: Model A Waterfall
    # -------------------------------------------------------------
    ax_wf_a = fig.add_subplot(gs[1, 0])
    render_waterfall(ax_wf_a, exp_a_data, intercept_a, th_a, "Model A (Single Snapshot)", row["prob_A"], row["pred_A"])
    
    # -------------------------------------------------------------
    # 4. Bottom Right: Model B Waterfall
    # -------------------------------------------------------------
    ax_wf_b = fig.add_subplot(gs[1, 1])
    render_waterfall(ax_wf_b, exp_b_data, intercept_b, th_b, "Model B (Concat with Baseline)", row["prob_B"], row["pred_B"])
    
    plt.suptitle(
        f"DISCORDANT CASE WATERFALL ANALYSIS: {row['pair_id']} (Patient {row['patient_id']} - {row['record']})\n"
        f"Category: {row['discordance_category']}",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved waterfall plot to: {save_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs/models/waterfall"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 75)
    print("STEP 13: LOCAL INTERPRETABILITY & WATERFALL ANALYSIS")
    print("=" * 75)
    
    splits, cur_feats, concat_feats = build_raw_concat_data()
    tr_cur, y_tr = splits["train"]["cur_X"], splits["train"]["y"]
    val_cur, y_val = splits["val"]["cur_X"], splits["val"]["y"]
    te_cur, y_te = splits["test"]["cur_X"], splits["test"]["y"]
    
    tr_cat = splits["train"]["concat_X"]
    val_cat = splits["val"]["concat_X"]
    te_cat = splits["test"]["concat_X"]
    
    # 1. Fit models
    ebm_a = ExplainableBoostingClassifier(interactions=15, learning_rate=0.03, max_leaves=3, min_samples_leaf=10, random_state=42, n_jobs=-1)
    ebm_a.fit(tr_cur, y_tr)
    v_prob_a = ebm_a.predict_proba(val_cur)[:, 1]
    
    ebm_b = ExplainableBoostingClassifier(interactions=15, learning_rate=0.03, max_leaves=3, min_samples_leaf=10, random_state=42, n_jobs=-1)
    ebm_b.fit(tr_cat, y_tr)
    v_prob_b = ebm_b.predict_proba(val_cat)[:, 1]
    
    th_a = 0.180
    th_b = 0.240
    
    intercept_a = float(ebm_a.intercept_[0])
    intercept_b = float(ebm_b.intercept_[0])
    
    # 2. Load metadata & raw waveforms
    discordant_cases = pd.read_csv(root / "outputs/models/ebm_discordant_cases.csv")
    all_preds = pd.read_csv(root / "outputs/models/ebm_all_test_predictions.csv")
    cur_waves = np.load(root / "dataset/paired/test_cur.npy")
    prev_waves = np.load(root / "dataset/paired/test_prev.npy")
    
    # Compute local explanations for entire test set
    print("Computing local explanations for Model A and Model B...")
    local_exp_a = ebm_a.explain_local(te_cur)
    local_exp_b = ebm_b.explain_local(te_cat)
    
    # 3. Generate Visual Waterfall Plots for 4 Prime Exemplars
    key_cases = [
        # Case 1: Hero case - B catches subtle ischemia missed by A (GT=1)
        {"idx": 11, "name": "case1_subtle_ischemia_caught_by_B", "label": "Hero Case: B Catches Subtle Ischemia Missed by A"},
        # Case 2: Hero case - B corrects chronic baseline false alarm (GT=0)
        {"idx": 64, "name": "case2_clean_false_alarm_corrected_by_B", "label": "Hero Case: B Corrects False Alarm on Chronic Flat Waveform"},
        # Case 3: Another major correction - P08 chronic deviation corrected by B (GT=0)
        {"idx": 71, "name": "case3_chronic_st_elevation_corrected_by_B", "label": "Hero Case: B Corrects A's Severe False Alarm on P08"},
        # Case 4: False alarm by B - P21 clean window over-estimated by B (GT=0)
        {"idx": 55, "name": "case4_b_false_alarm_overreaction", "label": "Error Analysis: B False Alarm Over-Reaction"},
    ]
    
    for case in key_cases:
        i = case["idx"]
        row = all_preds.loc[i]
        c_wave = cur_waves[i]
        p_wave = prev_waves[i]
        
        save_file = out_dir / f"{case['name']}.png"
        plot_paired_waterfall(
            i, row, c_wave, p_wave,
            local_exp_a.data(i), intercept_a, th_a,
            local_exp_b.data(i), intercept_b, th_b,
            save_file
        )
        
    # 4. Extract top drivers for all 50 discordant cases into CSV
    records = []
    for _, d_row in discordant_cases.iterrows():
        pair_id = d_row["pair_id"]
        i = all_preds[all_preds["pair_id"] == pair_id].index[0]
        
        d_a = local_exp_a.data(i)
        d_b = local_exp_b.data(i)
        
        top_pos_a = sorted(zip(d_a["names"], d_a["scores"]), key=lambda x: x[1], reverse=True)[:3]
        top_neg_a = sorted(zip(d_a["names"], d_a["scores"]), key=lambda x: x[1])[:3]
        
        top_pos_b = sorted(zip(d_b["names"], d_b["scores"]), key=lambda x: x[1], reverse=True)[:3]
        top_neg_b = sorted(zip(d_b["names"], d_b["scores"]), key=lambda x: x[1])[:3]
        
        records.append({
            "pair_id": pair_id,
            "patient_id": d_row["patient_id"],
            "record": d_row["record"],
            "subtype": d_row["subtype"],
            "ground_truth": d_row["ground_truth"],
            "discordance_category": d_row["discordance_category"],
            "prob_A": d_row["prob_A"],
            "pred_A": d_row["pred_A"],
            "prob_B": d_row["prob_B"],
            "pred_B": d_row["pred_B"],
            "Model_A_Top_Pos_Driver": f"{top_pos_a[0][0]} (+{top_pos_a[0][1]:.3f})",
            "Model_A_Top_Neg_Driver": f"{top_neg_a[0][0]} ({top_neg_a[0][1]:.3f})",
            "Model_B_Top_Pos_Driver": f"{top_pos_b[0][0]} (+{top_pos_b[0][1]:.3f})",
            "Model_B_Top_Neg_Driver": f"{top_neg_b[0][0]} ({top_neg_b[0][1]:.3f})",
        })
        
    contrib_df = pd.DataFrame(records)
    contrib_df.to_csv(out_dir / "discordant_top_drivers_summary.csv", index=False)
    print(f"Top drivers summary exported to: {out_dir / 'discordant_top_drivers_summary.csv'}")
    print("\nLocal Interpretability Waterfall generation complete!")


if __name__ == "__main__":
    main()
