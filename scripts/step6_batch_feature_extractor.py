#!/usr/bin/env python
"""Step 6: Batch extract all clinical features for single and paired datasets.

Processes:
- dataset/single/{train,val,test}_X.npy -> outputs/features/single_{split}_features.csv
- dataset/paired/{train,val,test}_{cur,prev}.npy -> outputs/features/paired_{split}_features.csv

Preserves all metadata columns (patient_id, clip_id, subtype, is_pos, etc.)
alongside the newly engineered clinical features.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import our feature extractor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step5_clinical_features import extract_clinical_features_single, compute_paired_features, FS


def process_single_split(split, data_dir, out_dir):
    print(f"[{split.upper()}] Starting single-snapshot feature extraction...")
    t0 = time.time()
    x_path = data_dir / "single" / f"{split}_X.npy"
    m_path = data_dir / "single" / f"{split}_meta.csv"
    
    X = np.load(x_path)
    meta = pd.read_csv(m_path)
    n = len(X)
    assert len(meta) == n, f"Mismatch in {split} single"
    
    records = []
    for i in range(n):
        feats = extract_clinical_features_single(X[i], fs=FS)
        records.append(feats)
        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"  [{split.upper()} Single] Processed {i + 1}/{n} clips ({time.time() - t0:.1f}s)")
            
    feat_df = pd.DataFrame(records)
    out_df = pd.concat([meta, feat_df], axis=1)
    
    out_file = out_dir / f"single_{split}_features.csv"
    out_df.to_csv(out_file, index=False)
    print(f"[{split.upper()}] Single features saved to {out_file} (shape: {out_df.shape})")
    return out_file


def process_paired_split(split, data_dir, out_dir):
    print(f"[{split.upper()}] Starting paired dynamic-alert feature extraction...")
    t0 = time.time()
    cur_path = data_dir / "paired" / f"{split}_cur.npy"
    prev_path = data_dir / "paired" / f"{split}_prev.npy"
    m_path = data_dir / "paired" / f"{split}_meta.csv"
    
    cur_X = np.load(cur_path)
    prev_X = np.load(prev_path)
    meta = pd.read_csv(m_path)
    n = len(cur_X)
    assert len(prev_X) == n and len(meta) == n, f"Mismatch in {split} paired"
    
    records = []
    for i in range(n):
        f_cur = extract_clinical_features_single(cur_X[i], fs=FS)
        f_prev = extract_clinical_features_single(prev_X[i], fs=FS)
        time_gap = meta.loc[i, "prev_gap_s"] if "prev_gap_s" in meta.columns else 0.0
        p_feats = compute_paired_features(f_cur, f_prev, time_gap_s=time_gap)
        records.append(p_feats)
        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"  [{split.upper()} Paired] Processed {i + 1}/{n} pairs ({time.time() - t0:.1f}s)")
            
    feat_df = pd.DataFrame(records)
    out_df = pd.concat([meta, feat_df], axis=1)
    
    out_file = out_dir / f"paired_{split}_features.csv"
    out_df.to_csv(out_file, index=False)
    print(f"[{split.upper()}] Paired features saved to {out_file} (shape: {out_df.shape})")
    return out_file


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "dataset"
    out_dir = root / "outputs/features"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ["test", "val", "train"]  # Process small ones first for fast feedback
    
    # 1. Single-snapshot
    for sp in splits:
        process_single_split(sp, data_dir, out_dir)
        
    # 2. Paired dynamic-alert
    for sp in splits:
        process_paired_split(sp, data_dir, out_dir)
        
    print("\nAll feature tables generated successfully in:", out_dir)


if __name__ == "__main__":
    main()
