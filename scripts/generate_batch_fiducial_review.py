#!/usr/bin/env python
"""Generate an extensive multi-case batch of fiducial delineation plots for manual review.

Generates 12 representative examples across:
- Positive ST depressions (ST-)
- Positive ST elevations (ST+)
- Positive T wave changes (T+, T-)
- Negative Posture/axis shifts
- Negative Arrhythmias (PVC/SVPB)
- Negative Sinus Tachycardia (tachy)
- Negative Normal clean baselines
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import our delineator module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step4_fiducial_delineator import bandpass_filter, extract_median_beat, delineate_beat, FS


def main():
    root = Path(__file__).resolve().parents[1]
    X_train = np.load(root / "dataset/single/train_X.npy")
    m_train = pd.read_csv(root / "dataset/single/train_meta.csv")
    out_dir = root / "qc"
    out_dir.mkdir(exist_ok=True)
    
    rng = np.random.default_rng(123)
    
    # Select 12 varied and clinically representative categories
    subtypes_to_sample = [
        # Positives
        "pos_ST0-", "pos_ST1-", 
        "pos_ST0+", "pos_ST1+", 
        "pos_T0+", "pos_T1+", 
        "pos_T0-", "pos_T1-",
        # Negatives
        "neg_posture", "neg_posture",
        "neg_arrhythmia",
        "neg_tachy"
    ]
    
    selected_indices = []
    for st in subtypes_to_sample:
        sub_df = m_train[m_train.subtype == st]
        # pick one randomly
        pick_idx = rng.choice(sub_df.index)
        selected_indices.append(pick_idx)
        
    n_cases = len(selected_indices)
    fig, axs = plt.subplots(n_cases, 2, figsize=(15, 2.7 * n_cases))
    
    for row, idx in enumerate(selected_indices):
        clip = X_train[idx]
        meta = m_train.iloc[idx]
        sig_names = meta.sig_names.split(",")
        
        # 1. Filter clip
        clip_filt = np.zeros_like(clip)
        for c in range(2):
            clip_filt[c] = bandpass_filter(clip[c])
            
        # 2. Extract median beat
        med_beat, r_peaks, hr_bpm, r_idx = extract_median_beat(clip_filt)
        t_axis = (np.arange(med_beat.shape[1]) - r_idx) / FS * 1000.0  # ms relative to R
        
        for c in range(2):
            lead_name = sig_names[c]
            fid = delineate_beat(med_beat[c], r_idx, hr_bpm)
            
            ax = axs[row, c]
            ax.plot(t_axis, med_beat[c], color="black", lw=1.4, label="Median Beat")
            
            # Draw baseline
            ax.axhline(fid["v_base"], color="gray", ls="--", lw=0.8, label=f"Isoelectric PQ ({fid['v_base']:.2f}mV)")
            
            # Plot fiducials
            pts = [
                ("PQ", fid["pq"], "purple", "s"),
                ("R", fid["r"], "blue", "^"),
                ("S", fid["s"], "navy", "v"),
                ("J", fid["j"], "green", "o"),
                ("ST80" if hr_bpm <= 120 else "ST60", fid["st"], "red", "D"),
                ("T_peak", fid["t"], "darkorange", "P")
            ]
            for label, pt, col, marker in pts:
                ax.scatter(t_axis[pt], med_beat[c, pt], color=col, s=40, marker=marker, zorder=5, label=label)
                
            # Annotate ST dev and slope
            st_name = "ST80" if hr_bpm <= 120 else "ST60"
            st_text = f"ST_dev: {fid['st_dev_mv']*1000:+.0f}uV | Slope: {fid['st_slope_mv_s']:+.2f}mV/s | T_amp: {fid['t_amp_mv']*1000:+.0f}uV"
            ax.set_title(f"[{meta.clip_id}] {meta.subtype} ({meta.record}) - {lead_name} | HR: {hr_bpm:.0f}bpm\n{st_text}", fontsize=8)
            ax.set_xlabel("Time from R-peak (ms)", fontsize=7)
            ax.set_ylabel("Amplitude (mV)", fontsize=7)
            ax.grid(True, ls=":", alpha=0.5)
            if row == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")
                
    fig.tight_layout()
    plot_path = out_dir / "fiducial_points_batch_12cases.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print("Batch review plot successfully generated at:", plot_path)


if __name__ == "__main__":
    main()
