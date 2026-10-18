#!/usr/bin/env python
"""Step 4: Fiducial points detection and median beat template extraction.

Tests extracting median beat and fiducial points (PQ, R, S, J, ST60/80, T_peak)
on a small set of representative clips and generates a verification plot.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FS = 250  # Sampling frequency: 250 Hz (1 sample = 4 ms)


def bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=FS, order=2):
    """Zero-phase Butterworth bandpass filter to remove baseline wander and EMG noise."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, sig, axis=-1)


def detect_r_peaks(lead_sig, fs=FS):
    """Pan-Tompkins inspired robust R-peak detector for a 10s lead signal."""
    # 1. Bandpass filter 5-15 Hz to isolate QRS energy
    nyq = 0.5 * fs
    b, a = butter(1, [5.0 / nyq, 15.0 / nyq], btype="band")
    filtered = filtfilt(b, a, lead_sig)
    
    # 2. Derivative
    diff = np.diff(filtered, prepend=filtered[0])
    
    # 3. Squaring
    squared = diff ** 2
    
    # 4. Moving integration window (~120ms = 30 samples)
    win_len = int(0.12 * fs)
    kernel = np.ones(win_len) / win_len
    integrated = np.convolve(squared, kernel, mode="same")
    
    # 5. Peak finding with minimum distance of 300ms (max HR 200 bpm)
    min_dist = int(0.3 * fs)
    threshold = 0.25 * np.max(integrated)
    peaks, _ = find_peaks(integrated, distance=min_dist, height=threshold)
    
    # Refine R-peaks by searching local extremum in original lead_sig within +-40ms of integrated peaks
    refined_peaks = []
    win_search = int(0.04 * fs)
    for p in peaks:
        left = max(0, p - win_search)
        right = min(len(lead_sig), p + win_search + 1)
        sub = lead_sig[left:right]
        # Check whether R peak is positive or negative dominant
        idx = np.argmax(sub) if np.max(sub) - np.mean(sub) > np.mean(sub) - np.min(sub) else np.argmin(sub)
        refined_peaks.append(left + idx)
        
    return np.array(refined_peaks, dtype=int)


def extract_median_beat(clip_sig, fs=FS, pre_ms=200, post_ms=500):
    """Extract median beat template across normal beats for a 2-channel 10s clip.
    
    Returns:
        median_beat: (2, n_samples) template centered at R-peak
        r_peaks: detected R-peaks in clip
        hr_bpm: estimated heart rate
    """
    pre_samples = int(pre_ms / 1000.0 * fs)   # 50 samples (200ms)
    post_samples = int(post_ms / 1000.0 * fs) # 125 samples (500ms)
    beat_len = pre_samples + post_samples     # 175 samples (700ms)
    
    # Use lead with higher dynamic range to find R peaks
    var0, var1 = np.var(clip_sig[0]), np.var(clip_sig[1])
    lead_idx = 0 if var0 >= var1 else 1
    r_peaks = detect_r_peaks(clip_sig[lead_idx], fs)
    
    # Filter R-peaks that have full pre and post window
    valid_peaks = [r for r in r_peaks if (r - pre_samples >= 0) and (r + post_samples < clip_sig.shape[1])]
    
    if len(valid_peaks) < 3:
        # Fallback to center-based pseudo beat if too few peaks
        hr_bpm = 75.0
        median_beat = np.zeros((2, beat_len), dtype=np.float32)
        return median_beat, valid_peaks, hr_bpm, pre_samples
        
    # Calculate HR
    rr_intervals = np.diff(valid_peaks) / fs
    hr_bpm = 60.0 / np.median(rr_intervals)
    
    # Collect beat windows and take median
    beats = []
    for r in valid_peaks:
        seg = clip_sig[:, r - pre_samples : r + post_samples]
        beats.append(seg)
    median_beat = np.median(np.stack(beats, axis=0), axis=0)
    
    return median_beat, valid_peaks, hr_bpm, pre_samples


def delineate_beat(template_lead, r_idx, hr_bpm, fs=FS):
    """Locate fiducial points (PQ, R, S, J, ST60/80, T_peak) on a single lead template.
    
    template_lead: 1D array of median beat
    r_idx: sample index of R peak in template (usually pre_samples)
    """
    # 1. R point
    # Search within +-20ms of r_idx for exact peak
    w_r = int(0.02 * fs)
    sub_r = template_lead[max(0, r_idx - w_r) : min(len(template_lead), r_idx + w_r + 1)]
    r_pt = max(0, r_idx - w_r) + np.argmax(np.abs(sub_r - np.median(template_lead)))
    
    # 2. PQ Junction (Isoelectric Baseline Reference)
    # Search 30ms to 90ms before R-peak for flat PR segment
    pq_start = max(0, r_pt - int(0.09 * fs))
    pq_end = max(0, r_pt - int(0.03 * fs))
    if pq_end > pq_start:
        diff_pr = np.abs(np.diff(template_lead[pq_start:pq_end]))
        # Point with smallest local slope in PR segment
        best_offset = np.argmin(diff_pr)
        pq_pt = pq_start + best_offset
    else:
        pq_pt = max(0, r_pt - int(0.05 * fs))
    v_base = template_lead[pq_pt]
    
    # 3. Q and S points (QRS wave extrema)
    # Search Q before R:
    q_start = max(0, r_pt - int(0.06 * fs))
    q_pt = q_start + np.argmin(template_lead[q_start:r_pt + 1])
    
    # Search S after R:
    s_start = r_pt
    s_end = min(len(template_lead), r_pt + int(0.08 * fs))
    s_pt = s_start + np.argmin(template_lead[s_start:s_end])
    
    # 4. J point (QRS offset / ST onset)
    # In clinical cardiology, J point is the point where the QRS wave completely finishes returning 
    # to the isoelectric level or transitions into the ST segment.
    # For a deep S wave (or QS wave), the ascending limb rises back towards v_base.
    # J point must be after the steep recovery slope has finished (diff drops to near-zero baseline slope).
    
    diff_lead = np.diff(template_lead)  # 1st derivative (slope)
    
    # Identify the steep QRS recovery limb starting from S point
    j_search_start = s_pt + 1
    j_search_end = min(len(template_lead) - 1, s_pt + int(0.12 * fs))
    
    # Find maximum recovery slope on the ascending limb after S
    recov_slopes = np.abs(diff_lead[j_search_start:min(len(diff_lead), s_pt + int(0.06 * fs))])
    max_recov_slope = np.max(recov_slopes) if len(recov_slopes) > 0 else 0.05
    
    j_pt = j_search_end
    # Search forward until the slope drops below 8% of max recovery slope OR enters the flat ST segment
    for i in range(j_search_start, j_search_end):
        # A true J point is both relatively flat AND has mostly recovered towards v_base
        slope_flat = np.abs(diff_lead[i]) <= max(0.008, 0.08 * max_recov_slope)
        # Check if the next 3 samples stay flat (confirming ST segment rather than mid-QRS inflection)
        next_slopes = np.abs(diff_lead[i:min(len(diff_lead), i + 4)])
        if slope_flat and (len(next_slopes) == 0 or np.mean(next_slopes) <= max(0.012, 0.12 * max_recov_slope)):
            j_pt = i
            break
            
    # Safeguard: J-point should not be significantly below baseline if signal already crossed baseline
    if template_lead[j_pt] < v_base - 0.1 and j_pt < j_search_end:
        # Check if baseline is crossed shortly after
        crossings = np.where(np.diff(np.signbit(template_lead[j_pt:j_search_end] - v_base)))[0]
        if len(crossings) > 0:
            j_pt = j_pt + crossings[0] + 1
        
    # 5. ST evaluation point (ST60 if HR > 120, else ST80)
    st_offset_ms = 60 if hr_bpm > 120 else 80
    st_pt = min(len(template_lead) - 1, j_pt + int(st_offset_ms / 1000.0 * fs))
    
    # 6. T peak
    # Search from J + 60ms to end of template (up to 450ms after R)
    t_start = min(len(template_lead) - 1, j_pt + int(0.06 * fs))
    t_end = min(len(template_lead), r_pt + int(0.45 * fs))
    if t_end > t_start:
        sub_t = template_lead[t_start:t_end] - v_base
        # Find peak with maximum deflection from baseline (positive or inverted)
        idx_t = np.argmax(np.abs(sub_t))
        t_pt = t_start + idx_t
    else:
        t_pt = min(len(template_lead) - 1, j_pt + int(0.18 * fs))
        
    fiducials = {
        "pq": pq_pt,
        "r": r_pt,
        "s": s_pt,
        "j": j_pt,
        "st": st_pt,
        "t": t_pt,
        "v_base": v_base,
        "st_dev_mv": template_lead[st_pt] - v_base,
        "j_dev_mv": template_lead[j_pt] - v_base,
        "t_amp_mv": template_lead[t_pt] - v_base,
        "st_slope_mv_s": (template_lead[st_pt] - template_lead[j_pt]) / (st_offset_ms / 1000.0)
    }
    return fiducials


def main():
    root = Path(__file__).resolve().parents[1]
    X_train = np.load(root / "dataset/single/train_X.npy")
    m_train = pd.read_csv(root / "dataset/single/train_meta.csv")
    out_dir = root / "qc"
    out_dir.mkdir(exist_ok=True)
    
    # Pick 4 diverse examples: ST depression, ST elevation, T inversion, normal clean
    idx_st_dep = m_train[m_train.subtype == "pos_ST1-"].index[0]
    idx_st_elev = m_train[m_train.subtype == "pos_ST0+"].index[0]
    idx_t_inv = m_train[m_train.subtype == "pos_T0-"].index[0]
    idx_clean = m_train[m_train.subtype == "neg_clean"].index[0]
    
    test_indices = [idx_st_dep, idx_st_elev, idx_t_inv, idx_clean]
    
    fig, axs = plt.subplots(len(test_indices), 2, figsize=(14, 3.2 * len(test_indices)))
    
    for row, idx in enumerate(test_indices):
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
            ax.plot(t_axis, med_beat[c], color="black", lw=1.5, label="Median Beat")
            
            # Draw baseline
            ax.axhline(fid["v_base"], color="gray", ls="--", lw=0.9, label=f"Isoelectric PQ ({fid['v_base']:.2f}mV)")
            
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
                ax.scatter(t_axis[pt], med_beat[c, pt], color=col, s=45, marker=marker, zorder=5, label=label)
                
            # Annotate ST dev
            st_text = f"ST_dev: {fid['st_dev_mv']*1000:.0f}uV | Slope: {fid['st_slope_mv_s']:.2f}mV/s"
            ax.set_title(f"{meta.clip_id} ({meta.subtype}) - {lead_name} | HR: {hr_bpm:.0f}bpm\n{st_text}", fontsize=9)
            ax.set_xlabel("Time from R-peak (ms)", fontsize=8)
            ax.set_ylabel("Amplitude (mV)", fontsize=8)
            ax.grid(True, ls=":", alpha=0.6)
            if row == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")
                
    fig.tight_layout()
    plot_path = out_dir / "fiducial_points_verification.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print("Verification plot successfully generated at:", plot_path)


if __name__ == "__main__":
    main()
