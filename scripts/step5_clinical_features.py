#!/usr/bin/env python
"""Step 5: Comprehensive Clinical Feature Extractor for ECG Delineation.

Calculates all foundational and advanced clinical features from delineated fiducials:
1. ST-segment features (ST60/80 deviation, J deviation, ST slope, ST integral)
2. T-wave features (T amplitude, T/R ratio, T symmetry index, T-end, QTc via Bazett & Fridericia)
3. QRS depolarization & electrical axis features (QRS duration, Q depth, pathological Q ratio, R amp, S depth, R/S ratio, spatial electrical angle theta)
4. Dynamic & global rhythm features (heart rate, RR interval std, tachycardia flag)
5. Inter-lead reciprocal ischemia features (ST cross-product)
6. Pairwise delta features for Task 2 (including ST/HR index, Delta ST, Delta angle theta, Delta QTc)
"""
import numpy as np
from pathlib import Path
import sys

# Import our robust delineator from step4
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step4_fiducial_delineator import bandpass_filter, extract_median_beat, delineate_beat, FS


def find_t_end(lead_sig, t_pt, v_base, fs=FS):
    """Find T-wave end using the clinical tangential intersection method.
    
    Draws a tangent at the steepest slope of the T-wave descending limb
    and finds its intersection with the isoelectric baseline v_base.
    """
    n = len(lead_sig)
    # Search window after T peak: up to 180ms
    search_end = min(n - 1, t_pt + int(0.18 * fs))
    if search_end <= t_pt + 2:
        return min(n - 1, t_pt + int(0.10 * fs))
        
    diff = np.diff(lead_sig)
    # Sign of T wave relative to baseline
    t_sign = 1 if lead_sig[t_pt] >= v_base else -1
    
    # Descending limb: if T positive, slope is most negative; if inverted, slope is most positive
    limb_slopes = diff[t_pt:search_end]
    steep_idx = np.argmin(limb_slopes) if t_sign == 1 else np.argmax(limb_slopes)
    steep_pt = t_pt + steep_idx
    steep_slope = diff[steep_pt]
    
    if abs(steep_slope) < 1e-4:
        # Fallback to curvature or amplitude threshold (within 5% of v_base)
        t_end = steep_pt
        for i in range(steep_pt, search_end):
            if abs(lead_sig[i] - v_base) < 0.02:
                t_end = i
                break
        return t_end
        
    # Line equation: y - y0 = slope * (x - x0) -> intersection with y = v_base:
    # x_intercept = x0 + (v_base - y0) / slope
    x_intercept = steep_pt + (v_base - lead_sig[steep_pt]) / steep_slope
    t_end = int(round(np.clip(x_intercept, t_pt + 1, min(n - 1, t_pt + int(0.25 * fs)))))
    return t_end


def extract_clinical_features_single(clip_2ch, fs=FS):
    """Extract full clinical electrophysiological feature dictionary from a 2-channel 10s clip.
    
    clip_2ch: (2, 2500) array in mV
    Returns:
        feats: dict containing all numerical clinical features
    """
    # 1. Filter
    clip_filt = np.zeros_like(clip_2ch)
    for c in range(2):
        clip_filt[c] = bandpass_filter(clip_2ch[c], fs=fs)
        
    # 2. Median beat extraction
    med_beat, r_peaks, hr_bpm, r_idx = extract_median_beat(clip_filt, fs=fs)
    
    # Global rhythm metrics
    if len(r_peaks) >= 2:
        rr_intervals_ms = np.diff(r_peaks) / fs * 1000.0
        rr_std_ms = float(np.std(rr_intervals_ms))
        mean_rr_s = float(np.mean(rr_intervals_ms) / 1000.0)
    else:
        rr_std_ms = 0.0
        mean_rr_s = 60.0 / max(1.0, hr_bpm)
        
    feats = {
        "hr_bpm": float(hr_bpm),
        "rr_std_ms": rr_std_ms,
        "is_tachycardia": 1.0 if hr_bpm > 100.0 else 0.0
    }
    
    fids = []
    # 3. Delineate both leads
    for c in range(2):
        lead_sig = med_beat[c]
        fid = delineate_beat(lead_sig, r_idx, hr_bpm, fs=fs)
        fids.append(fid)
        
        prefix = f"ch{c}_"
        v_base = fid["v_base"]
        
        # --- A. ST Segment Features ---
        st_dev = fid["st_dev_mv"]
        j_dev = fid["j_dev_mv"]
        st_slope = fid["st_slope_mv_s"]
        
        # ST integral: area between J and T_peak
        j_pt, t_pt = fid["j"], fid["t"]
        if t_pt > j_pt:
            trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
            st_integral = float(trapz_fn(lead_sig[j_pt:t_pt] - v_base) * (1000.0 / fs)) # mV*ms
        else:
            st_integral = 0.0
            
        feats[prefix + "st_dev_mv"] = float(st_dev)
        feats[prefix + "j_dev_mv"] = float(j_dev)
        feats[prefix + "st_slope_mv_s"] = float(st_slope)
        feats[prefix + "st_integral_mv_ms"] = st_integral
        
        # --- B. T-wave & Repolarization Features ---
        t_amp = fid["t_amp_mv"]
        feats[prefix + "t_amp_mv"] = float(t_amp)
        
        # T-end and QTc
        t_end = find_t_end(lead_sig, t_pt, v_base, fs=fs)
        fid["t_end"] = t_end
        
        # QT interval (Q point to T-end)
        q_pt = fid["pq"]  # or q_pt from QRS onset
        qt_ms = max(10.0, (t_end - q_pt) * (1000.0 / fs))
        # Bazett: QTc = QT / sqrt(RR_s), Fridericia: QTc = QT / cbrt(RR_s)
        qtc_bazett = qt_ms / np.sqrt(max(0.2, mean_rr_s))
        qtc_fridericia = qt_ms / (max(0.2, mean_rr_s) ** (1.0 / 3.0))
        
        feats[prefix + "qt_ms"] = float(qt_ms)
        feats[prefix + "qtc_bazett_ms"] = float(qtc_bazett)
        feats[prefix + "qtc_fridericia_ms"] = float(qtc_fridericia)
        
        # T-wave symmetry index: (T_peak - J) / (T_end - T_peak)
        t_rise = max(1, t_pt - j_pt)
        t_fall = max(1, t_end - t_pt)
        t_symmetry = float(t_rise / t_fall)
        feats[prefix + "t_symmetry_idx"] = float(np.clip(t_symmetry, 0.1, 10.0))
        
        # --- C. QRS Depolarization & Electrical Axis Features ---
        r_pt = fid["r"]
        s_pt = fid["s"]
        r_amp = float(lead_sig[r_pt] - v_base)
        s_depth = float(lead_sig[s_pt] - v_base)
        
        # Q wave
        q_search = lead_sig[max(0, r_pt - int(0.06 * fs)):r_pt]
        q_depth = float(np.min(q_search) - v_base) if len(q_search) else 0.0
        
        qrs_dur_ms = float((j_pt - q_pt) * (1000.0 / fs))
        rs_ratio = float(r_amp / (abs(s_depth) + 1e-3))
        t_to_r_ratio = float(t_amp / (abs(r_amp) + 1e-3))
        
        # Pathological Q wave flag: Q >= 40ms or Q/R >= 25%
        q_ratio = abs(q_depth) / (abs(r_amp) + 1e-3)
        pathological_q = 1.0 if (q_depth < -0.05 and q_ratio >= 0.25) else 0.0
        
        feats[prefix + "qrs_dur_ms"] = qrs_dur_ms
        feats[prefix + "r_amp_mv"] = r_amp
        feats[prefix + "s_depth_mv"] = s_depth
        feats[prefix + "rs_ratio"] = float(np.clip(rs_ratio, -20.0, 20.0))
        feats[prefix + "t_to_r_ratio"] = float(np.clip(t_to_r_ratio, -10.0, 10.0))
        feats[prefix + "pathological_q"] = pathological_q

    # --- D. Inter-lead Spatial & Reciprocal Features ---
    # Spatial QRS electrical angle theta = arctan2(ch1_R, ch0_R) in degrees
    r0 = feats["ch0_r_amp_mv"]
    r1 = feats["ch1_r_amp_mv"]
    theta_qrs_deg = float(np.degrees(np.arctan2(r1, r0)))
    feats["spatial_theta_qrs_deg"] = theta_qrs_deg
    
    # Spatial ST electrical angle theta_st = arctan2(ch1_st, ch0_st)
    st0 = feats["ch0_st_dev_mv"]
    st1 = feats["ch1_st_dev_mv"]
    theta_st_deg = float(np.degrees(np.arctan2(st1, st0)))
    feats["spatial_theta_st_deg"] = theta_st_deg
    
    # Reciprocal ischemia sign (negative if one elevates and one depresses)
    feats["reciprocal_st_product"] = float(st0 * st1)
    
    return feats


def compute_paired_features(feats_cur, feats_prev, time_gap_s):
    """Compute differential Delta features for Task 2 (Current vs Previous)."""
    paired_feats = {}
    # 1. Base Current features
    for k, v in feats_cur.items():
        paired_feats["cur_" + k] = v
        
    # 2. Pure Delta (Net Change)
    for k in feats_cur.keys():
        paired_feats["delta_" + k] = feats_cur[k] - feats_prev[k]
        
    # 3. Advanced Exercise/Dynamic Correction Features
    delta_hr = paired_feats["delta_hr_bpm"]
    for c in range(2):
        delta_st_uv = paired_feats[f"delta_ch{c}_st_dev_mv"] * 1000.0  # in uV
        # ST / Heart Rate Index (uV / bpm)
        if abs(delta_hr) >= 1.0:
            st_hr_index = float(delta_st_uv / delta_hr)
        else:
            # If HR didn't change, physiological slope is neutral
            st_hr_index = 0.0
        paired_feats[f"ch{c}_st_hr_index_uV_bpm"] = float(np.clip(st_hr_index, -100.0, 100.0))
        
    # Spatial Axis Shift (degrees)
    delta_theta_qrs = abs(feats_cur["spatial_theta_qrs_deg"] - feats_prev["spatial_theta_qrs_deg"])
    if delta_theta_qrs > 180.0:
        delta_theta_qrs = 360.0 - delta_theta_qrs
    paired_feats["delta_spatial_theta_qrs_deg"] = float(delta_theta_qrs)
    
    paired_feats["prev_time_gap_s"] = float(time_gap_s)
    return paired_feats


if __name__ == "__main__":
    # Self-test on one example
    X_train = np.load(Path(__file__).resolve().parents[1] / "dataset/single/train_X.npy")
    test_clip = X_train[0]
    feats = extract_clinical_features_single(test_clip)
    print(f"Extracted {len(feats)} clinical features successfully from a single clip:")
    for k, v in list(feats.items())[:10]:
        print(f"  {k}: {v:.4f}")
    print("  ...")
    print(f"  spatial_theta_qrs_deg: {feats['spatial_theta_qrs_deg']:.2f}°")
    print(f"  reciprocal_st_product: {feats['reciprocal_st_product']:.6f}")
