#!/usr/bin/env python
"""Step 19: Synthetic Missingness Ablation Experiment.

Empirically tests the Gradient Preemption & Feature Dilution Hypothesis:
How does the fraction of missing baseline ECGs (replaced with median constant)
in training data affect:
1. The number of cross-temporal interactions selected by FAST (out of 15)
2. The global importance ratio of baseline (prev_) vs current (cur_) features
3. The individual ranking and importance score of key baseline features (e.g. prev_ch0_j_dev_mv)
4. The test AUROC and baseline gain (Delta AUROC = AUROC_with - AUROC_without)
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

# Add scripts directory to path
sys.path.append(str(Path(__file__).resolve().parent))
from step11_raw_concat_ab_experiment import build_raw_concat_data


def run_synthetic_missingness_ablation():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs/models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STEP 19: SYNTHETIC MISSINGNESS ABLATION EXPERIMENT")
    print("=" * 80)
    
    # 1. Load Data
    splits, cur_feats, concat_feats = build_raw_concat_data()
    prev_feats = [c for c in concat_feats if c.startswith("prev_")]
    
    X_tr_raw = splits["train"]["concat_X"].copy()
    y_tr = splits["train"]["y"]
    
    X_val_raw = splits["val"]["concat_X"].copy()
    y_val = splits["val"]["y"]
    
    X_te_raw = splits["test"]["concat_X"].copy()
    y_te = splits["test"]["y"]
    
    # Training set median for prev features
    prev_medians = X_tr_raw[prev_feats].median()
    
    # Construct fixed test sets
    # Mode with baseline (paired)
    X_te_with = X_te_raw.copy()
    X_te_with["baseline_available"] = 1
    
    # Mode without baseline (imputed with median)
    X_te_without = X_te_raw.copy()
    for col in prev_feats:
        X_te_without[col] = prev_medians[col]
    X_te_without["baseline_available"] = 0
    
    # Define Missingness Ratios
    missing_ratios = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    
    results = []
    
    for k in missing_ratios:
        print(f"\n---> Running Ablation for Missingness Ratio k = {k*100:.0f}%...")
        
        # 1. Mask Training Set
        X_tr_k = X_tr_raw.copy()
        rng_tr = np.random.RandomState(42)
        is_missing_tr = rng_tr.rand(len(X_tr_k)) < k
        for col in prev_feats:
            X_tr_k.loc[is_missing_tr, col] = prev_medians[col]
        X_tr_k["baseline_available"] = (~is_missing_tr).astype(int)
        
        # 2. Mask Validation Set
        X_val_k = X_val_raw.copy()
        rng_val = np.random.RandomState(42)
        is_missing_val = rng_val.rand(len(X_val_k)) < k
        for col in prev_feats:
            X_val_k.loc[is_missing_val, col] = prev_medians[col]
        X_val_k["baseline_available"] = (~is_missing_val).astype(int)
        
        # 3. Train EBM with standard interactions=15
        ebm = ExplainableBoostingClassifier(
            interactions=15,
            learning_rate=0.03,
            max_leaves=3,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        ebm.fit(X_tr_k, y_tr)
        
        # 4. Evaluate on Test Set
        probs_with = ebm.predict_proba(X_te_with)[:, 1]
        probs_without = ebm.predict_proba(X_te_without)[:, 1]
        
        auroc_with = roc_auc_score(y_te, probs_with)
        auprc_with = average_precision_score(y_te, probs_with)
        auroc_without = roc_auc_score(y_te, probs_without)
        auprc_without = average_precision_score(y_te, probs_without)
        delta_auroc = auroc_with - auroc_without
        
        # Calibrate threshold on validation set
        val_probs = ebm.predict_proba(X_val_k)[:, 1]
        best_th, best_ba = 0.5, -1.0
        for th in np.linspace(0.05, 0.95, 91):
            ba = balanced_accuracy_score(y_val, (val_probs >= th).astype(int))
            if ba > best_ba:
                best_ba = ba
                best_th = th
                
        ba_with = balanced_accuracy_score(y_te, (probs_with >= best_th).astype(int))
        ba_without = balanced_accuracy_score(y_te, (probs_without >= best_th).astype(int))
        
        # 5. Extract Feature Importance & Rankings
        exp = ebm.explain_global()
        names = exp.data()["names"]
        scores = exp.data()["scores"]
        
        # Create sorted dataframe of all terms
        imp_df = pd.DataFrame({"name": names, "score": scores}).sort_values("score", ascending=False).reset_index(drop=True)
        imp_df["rank"] = imp_df.index + 1
        
        # Count cross-temporal interactions (cur & prev)
        interaction_names = [n for n in names if "&" in n]
        cross_interactions = [n for n in interaction_names if ("cur_" in n and "prev_" in n)]
        n_cross = len(cross_interactions)
        
        # Sum of main effects
        cur_main_scores = [s for n, s in zip(names, scores) if (n.startswith("cur_") and "&" not in n)]
        prev_main_scores = [s for n, s in zip(names, scores) if (n.startswith("prev_") and "&" not in n)]
        
        sum_cur_main = float(np.sum(cur_main_scores))
        sum_prev_main = float(np.sum(prev_main_scores))
        prev_importance_ratio = sum_prev_main / (sum_cur_main + sum_prev_main) if (sum_cur_main + sum_prev_main) > 0 else 0.0
        
        # Key baseline feature: prev_ch0_j_dev_mv
        key_feat = "prev_ch0_j_dev_mv"
        if key_feat in imp_df["name"].values:
            row_k = imp_df[imp_df["name"] == key_feat].iloc[0]
            key_rank = int(row_k["rank"])
            key_score = float(row_k["score"])
        else:
            key_rank = len(imp_df)
            key_score = 0.0
            
        # Key current feature: cur_ch1_t_symmetry_idx
        cur_key = "cur_ch1_t_symmetry_idx"
        cur_score = float(imp_df[imp_df["name"] == cur_key]["score"].iloc[0]) if cur_key in imp_df["name"].values else 0.0
        
        print(f"  Cross Interactions Picked: {n_cross} / 15")
        print(f"  Baseline Feature Importance Ratio: {prev_importance_ratio*100:.2f}%")
        print(f"  Key Feature '{key_feat}': Rank {key_rank}, Score {key_score:.4f}")
        print(f"  Paired Test AUROC: {auroc_with:.4f} (Delta vs Unpaired: {delta_auroc:+.4f})")
        
        results.append({
            "missing_ratio": k,
            "missing_pct": f"{int(k*100)}%",
            "n_cross_interactions": n_cross,
            "n_intra_interactions": len(interaction_names) - n_cross,
            "sum_cur_importance": sum_cur_main,
            "sum_prev_importance": sum_prev_main,
            "prev_importance_ratio": prev_importance_ratio,
            "key_prev_j_dev_rank": key_rank,
            "key_prev_j_dev_score": key_score,
            "key_cur_t_sym_score": cur_score,
            "test_paired_auroc": auroc_with,
            "test_unpaired_auroc": auroc_without,
            "delta_auroc": delta_auroc,
            "test_paired_auprc": auprc_with,
            "test_paired_ba": ba_with,
            "val_best_threshold": best_th
        })
        
    res_df = pd.DataFrame(results)
    csv_path = out_dir / "synthetic_missingness_ablation_results.csv"
    res_df.to_csv(csv_path, index=False)
    print("\n" + "=" * 80)
    print("SYNTHETIC MISSINGNESS ABLATION SUMMARY TABLE:")
    print("=" * 80)
    disp_cols = [
        "missing_pct", "n_cross_interactions", "prev_importance_ratio",
        "key_prev_j_dev_rank", "key_prev_j_dev_score", "test_paired_auroc", "delta_auroc"
    ]
    print(res_df[disp_cols].to_string(index=False))
    print(f"\nSaved CSV to: {csv_path}")
    
    # Plot Publication-Grade Figure
    plot_path = out_dir / "synthetic_missingness_ablation_curves.png"
    plot_ablation_curves(res_df, plot_path)
    print(f"Saved figure to: {plot_path}")
    print("=" * 80)


def plot_ablation_curves(df, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=250)
    plt.subplots_adjust(hspace=0.28, wspace=0.24)
    
    x = df["missing_ratio"] * 100
    
    # Subplot 1: Number of Cross-Temporal Interactions
    ax1 = axes[0, 0]
    ax1.plot(x, df["n_cross_interactions"], marker="o", color="#d35400", lw=2.5, markersize=8, label="Cross-Temporal Pairs (Cur & Prev)")
    ax1.plot(x, df["n_intra_interactions"], marker="s", color="#7f8c8d", lw=1.8, linestyle="--", markersize=6, label="Intra-Current Pairs (Cur & Cur)")
    ax1.set_title("(A) Automated Interaction Selection vs. Missingness", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Missing Baseline Fraction in Training Set (%)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Interactions Selected (out of 15)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim([-0.5, 16])
    ax1.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="center right", fontsize=9.5)
    
    # Subplot 2: Total Feature Importance Ratio
    ax2 = axes[0, 1]
    ax2.plot(x, df["prev_importance_ratio"] * 100, marker="^", color="#2980b9", lw=2.5, markersize=8, label="Baseline Feature Share (%)")
    ax2.set_title("(B) Baseline Feature Importance Share vs. Missingness", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Missing Baseline Fraction in Training Set (%)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Total Prev Importance Share (%)", fontsize=10.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9.5)
    
    # Subplot 3: Individual Key Feature Rank & Score
    ax3 = axes[1, 0]
    color_score = "#27ae60"
    color_rank = "#8e44ad"
    
    ax3.plot(x, df["key_prev_j_dev_score"], marker="D", color=color_score, lw=2.5, markersize=7, label="Importance Score")
    ax3.set_xlabel("Missing Baseline Fraction in Training Set (%)", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Importance Score (prev_ch0_j_dev_mv)", color=color_score, fontsize=10.5, fontweight="bold")
    ax3.tick_params(axis="y", labelcolor=color_score)
    ax3.grid(True, linestyle="--", alpha=0.5)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, df["key_prev_j_dev_rank"], marker="v", color=color_rank, lw=2, linestyle="-.", markersize=7, label="Global Rank")
    ax3_twin.set_ylabel("Global Rank (Lower is Better)", color=color_rank, fontsize=10.5, fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor=color_rank)
    ax3_twin.invert_yaxis()
    ax3.set_title("(C) Key Feature 'prev_ch0_j_dev_mv' Dilution", fontsize=12, fontweight="bold", pad=8)
    
    # Subplot 4: Paired Test AUROC & Delta Gain
    ax4 = axes[1, 1]
    ax4.plot(x, df["test_paired_auroc"], marker="o", color="#c0392b", lw=2.5, markersize=8, label="Paired Test AUROC")
    ax4.plot(x, df["test_unpaired_auroc"], marker="x", color="#34495e", lw=1.8, linestyle=":", markersize=7, label="Unpaired Test AUROC")
    ax4.set_title("(D) Diagnostic Performance & Baseline Gain vs. Missingness", fontsize=12, fontweight="bold", pad=8)
    ax4.set_xlabel("Missing Baseline Fraction in Training Set (%)", fontsize=10.5, fontweight="bold")
    ax4.set_ylabel("AUROC", fontsize=10.5, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)
    
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x, df["delta_auroc"], width=4.5, color="#16a085", alpha=0.35, label=r"$\Delta$ AUROC Gain")
    ax4_twin.set_ylabel(r"Baseline Gain ($\Delta$ AUROC)", color="#16a085", fontsize=10.5, fontweight="bold")
    ax4_twin.tick_params(axis="y", labelcolor="#16a085")
    ax4_twin.set_ylim([-0.01, max(df["delta_auroc"].max() * 1.5, 0.08)])
    
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    lines_4t, labels_4t = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines_4 + lines_4t, labels_4 + labels_4t, loc="lower left", fontsize=9)
    
    plt.suptitle("Synthetic Missingness Ablation: Empirical Proof of Gradient Preemption & Feature Dilution", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_synthetic_missingness_ablation()
