# Longitudinal ECG Ischemia Detection: Experimental Data Tables Notebook

**Brief Intro**:  
Evaluated on the European ST-T Database (EDB) across 87 long-term ambulatory records from 77 unique patients under patient-level zero-leakage isolation. The following tables document the quantitative benchmarks, global feature rankings, and interaction selection logs in the exact sequence reported.

---

## 1. Performance Comparison Across Model Architectures and Input Formats (Section 3.1)

Evaluated on the held-out 12-patient test set (203 paired samples) using validation-set calibrated decision thresholds:

| Model Architecture | Input Configuration | AUROC | AUPRC | Balanced Acc | Sensitivity | Specificity | F1-Score | FPR Posture | FPR Tachycardia |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **Model A (Single)** | 0.6594 | 0.4509 | 56.73% | 52.94% | 60.53% | 0.3913 | 78.26% | 53.33% |
| | **Model B-Diff (Engineered)** | 0.7675 | 0.5412 | 70.79% | 78.43% | 63.16% | 0.5442 | 95.65% | 26.67% |
| | **Model B-Concat (Raw)** | 0.5921 | 0.3132 | 58.06% | 50.98% | 65.13% | 0.4000 | 95.65% | 40.00% |
| **XGBoost** | **Model A (Single)** | 0.6654 | 0.4067 | 62.62% | 62.75% | 62.50% | 0.4571 | 100.0% | 40.00% |
| | **Model B-Diff (Engineered)** | 0.8270 | 0.6653 | 71.49% | 66.67% | 76.32% | 0.5620 | 95.65% | 33.33% |
| | **Model B-Concat (Raw)** | 0.7377 | 0.5179 | 63.59% | 68.63% | 58.55% | 0.4698 | 100.0% | 26.67% |
| **EBM (GA2M)** | **Model A (Single)** | 0.7377 | 0.6096 | 64.90% | 70.59% | 59.21% | 0.4832 | 91.30% | 40.00% |
| | **Model B-Diff (Engineered)** | 0.8046 | 0.6121 | 76.39% | 78.43% | 74.34% | 0.6154 | 82.61% | 13.33% |
| | **Model B-Concat (Raw)** | 0.8256 | 0.7166 | 74.11% | 70.59% | 77.63% | 0.5950 | 69.57% | 13.33% |

---

## 2. Top 15 Global Terms: Univariate and Interaction Terms Combined (Section 4.1)

Mean absolute score of each univariate function $f_i$ or interaction term $f_{ij}$ across the training set:

| Rank | Model A (Single Snapshot) Term | Term Type | Score | Model B (Raw Concat) Term | Term Type | Score |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| **1** | `cur_ch1_t_symmetry_idx` | Univariate | 0.3710 | `cur_rr_std_ms` | Univariate | 0.3091 |
| **2** | `cur_ch1_t_to_r_ratio` | Univariate | 0.3603 | `cur_ch0_j_dev_mv` | Univariate | 0.2699 |
| **3** | `cur_rr_std_ms` | Univariate | 0.3512 | `cur_hr_bpm` | Univariate | 0.2673 |
| **4** | `cur_rr_std_ms & cur_ch0_j_dev_mv` | Interaction | 0.2791 | `cur_ch1_st_dev_mv & cur_ch1_s_depth_mv` | Interaction | 0.2634 |
| **5** | `cur_ch0_j_dev_mv` | Univariate | 0.2548 | `cur_ch1_t_to_r_ratio` | Univariate | 0.2410 |
| **6** | `cur_ch0_s_depth_mv & cur_ch0_rs_ratio` | Interaction | 0.2337 | `prev_hr_bpm` (Baseline) | Univariate | 0.2157 |
| **7** | `cur_ch0_rs_ratio` | Univariate | 0.2249 | `prev_ch1_r_amp_mv` (Baseline) | Univariate | 0.2119 |
| **8** | `cur_ch0_st_slope & cur_ch0_st_integral` | Interaction | 0.2090 | `prev_ch0_j_dev_mv` (Baseline) | Univariate | 0.1965 |
| **9** | `cur_ch1_st_dev_mv & cur_ch1_s_depth_mv` | Interaction | 0.2040 | `prev_ch1_s_depth_mv` (Baseline) | Univariate | 0.1910 |
| **10** | `cur_ch1_st_slope_mv_s` | Univariate | 0.1999 | `prev_ch1_t_to_r_ratio` (Baseline) | Univariate | 0.1877 |
| **11** | `cur_ch1_r_amp_mv` | Univariate | 0.1991 | `cur_ch1_st_integral_mv_ms` | Univariate | 0.1831 |
| **12** | `cur_ch1_s_depth_mv & cur_ch1_t_to_r` | Interaction | 0.1974 | `cur_ch1_t_symmetry_idx` | Univariate | 0.1754 |
| **13** | `cur_ch1_s_depth_mv` | Univariate | 0.1969 | `cur_ch1_t_to_r & prev_ch1_t_to_r` | Cross-Temporal | 0.1727 |
| **14** | `cur_rr_std_ms & cur_reciprocal_st_prod` | Interaction | 0.1942 | `cur_ch1_st_integral & prev_ch1_t_to_r` | Cross-Temporal | 0.1704 |
| **15** | `cur_rr_std_ms & cur_ch1_r_amp_mv` | Interaction | 0.1902 | `cur_ch1_t_amp_mv & prev_ch1_t_amp_mv` | Cross-Temporal | 0.1696 |

---

## 3. Top 10 Univariate Main Effects Comparison (Section 4.2)

Isolated univariate main effects showing the emergence of baseline reference anchors in Model B:

| Rank | Model A (Single) Feature | Score | Model B (Concat) Feature | Score | Context Comparison |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **1** | `cur_ch1_t_symmetry_idx` | 0.3710 | `cur_rr_std_ms` | 0.3091 | Model A prioritizes static morphology; Model B shifts to HRV |
| **2** | `cur_ch1_t_to_r_ratio` | 0.3603 | `cur_ch0_j_dev_mv` | 0.2699 | J-point elevation/depression increases in rank |
| **3** | `cur_rr_std_ms` | 0.3512 | `cur_hr_bpm` | 0.2673 | Instantaneous heart rate contribution |
| **4** | `cur_ch0_j_dev_mv` | 0.2548 | `cur_ch1_t_to_r_ratio` | 0.2410 | Repolarization to depolarization ratio |
| **5** | `cur_ch0_rs_ratio` | 0.2249 | `prev_hr_bpm` | 0.2157 | Baseline heart rate reference term |
| **6** | `cur_ch1_st_slope_mv_s` | 0.1999 | `prev_ch1_r_amp_mv` | 0.2119 | Baseline R-wave amplitude reference |
| **7** | `cur_ch1_r_amp_mv` | 0.1991 | `prev_ch0_j_dev_mv` | 0.1966 | Baseline J-point position reference |
| **8** | `cur_ch1_s_depth_mv` | 0.1969 | `prev_ch1_s_depth_mv` | 0.1910 | Baseline S-wave depth reference |
| **9** | `cur_hr_bpm` | 0.1826 | `prev_ch1_t_to_r_ratio` | 0.1877 | Baseline T/R ratio reference |
| **10** | `cur_ch0_t_to_r_ratio` | 0.1786 | `cur_ch1_st_integral_mv_ms` | 0.1831 | Integrated ST segment displacement |

---

## 4. Interaction Terms Selected by EBM FAST Algorithm (Section 4.3)

All 15 pairwise interaction terms automatically selected out of 2,556 candidate pairs:

| Rank | Model A (Single Snapshot) Interaction | Score | Model B (Raw Concat) Interaction | Score |
| :---: | :--- | :---: | :--- | :---: |
| **1** | `cur_rr_std & cur_ch0_j_dev` | 0.2791 | `cur_ch1_st_dev & cur_ch1_s_depth` (Single snapshot) | 0.2634 |
| **2** | `cur_ch0_s_depth & cur_ch0_rs_ratio` | 0.2337 | `cur_ch1_t_to_r & prev_ch1_t_to_r` (Cross-temporal T/R) | 0.1727 |
| **3** | `cur_ch0_st_slope & cur_ch0_st_integral` | 0.2090 | `cur_ch1_st_integral & prev_ch1_t_to_r` (Cross-temporal) | 0.1704 |
| **4** | `cur_ch1_st_dev & cur_ch1_s_depth` | 0.2040 | `cur_ch1_t_amp & prev_ch1_t_amp` (Cross-temporal T amplitude) | 0.1696 |
| **5** | `cur_ch1_s_depth & cur_ch1_t_to_r` | 0.1974 | `cur_ch0_j_dev & prev_ch0_j_dev` (Cross-temporal J point) | 0.1605 |
| **6** | `cur_rr_std & cur_reciprocal_st_prod` | 0.1942 | `cur_ch0_st_integ & prev_ch0_st_integ` (Cross-temporal ST integral) | 0.1519 |
| **7** | `cur_rr_std & cur_ch1_r_amp` | 0.1902 | `cur_ch0_j_dev & prev_ch1_s_depth` (Cross-temporal) | 0.1515 |
| **8** | `cur_ch1_j_dev & cur_ch1_s_depth` | 0.1690 | `cur_ch1_j_dev & prev_ch1_j_dev` (Cross-temporal lead 1 J point) | 0.1499 |
| **9** | `cur_hr_bpm & cur_ch1_qtc_bazett` | 0.1680 | `cur_ch1_st_integ & prev_ch1_st_integ` (Cross-temporal lead 1 ST integral) | 0.1478 |
| **10** | `cur_ch1_st_integ & cur_ch1_s_depth` | 0.1578 | `cur_ch0_t_amp & prev_ch0_t_amp` (Cross-temporal lead 0 T wave) | 0.1474 |
| **11** | `cur_ch1_st_dev & cur_ch1_st_integ` | 0.1463 | `cur_ch0_t_amp & prev_ch1_rs_ratio` (Cross-temporal) | 0.1363 |
| **12** | `cur_ch1_st_integ & cur_ch1_qt` | 0.1381 | `cur_ch0_st_integ & prev_ch1_rs_ratio` (Cross-temporal) | 0.1192 |
| **13** | `cur_ch1_t_amp & cur_ch1_r_amp` | 0.1252 | `cur_ch0_t_to_r & prev_ch0_t_to_r` (Cross-temporal lead 0 T/R ratio) | 0.1145 |
| **14** | `cur_ch1_r_amp & cur_ch1_t_to_r` | 0.1095 | `cur_ch0_t_amp & prev_ch0_st_integ` (Cross-temporal) | 0.1014 |
| **15** | `cur_ch1_st_integ & cur_ch1_r_amp` | 0.0841 | `cur_spatial_theta_st & prev_spatial_theta_st` (Cross-temporal axis) | 0.0909 |

---

## 5. Quantitative Comparison Across Search Configurations (Section 7.1)

Unified dual-mode EBM evaluated under missing-baseline (unpaired) and present-baseline (paired) test modes:

| Experiment Setup | Interaction Budget | Cross-Temporal Selected | Test Input Mode | AUROC | $\Delta$ AUROC | Balanced Accuracy | Sensitivity | Specificity |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Step 16: Greedy Search** | Auto FAST (budget=15) | 0 / 15 | Without Baseline (Unpaired) | 0.7399 | - | 66.96% | 45.10% | 88.82% |
| | | | With Baseline (Paired) | 0.7430 | +0.0031 | 66.63% | 45.10% | 88.16% |
| **Step 18-A: Expanded Budget** | Auto FAST (budget=50) | 0 / 50 | Without Baseline (Unpaired) | 0.7722 | - | 65.01% | 37.25% | 92.76% |
| | | | With Baseline (Paired) | 0.7757 | +0.0035 | 66.00% | 37.25% | 94.74% |
| **Step 17: Explicit Selection** | 14 Cross + 6 Within | 14 / 20 | Without Baseline (Unpaired) | 0.7247 | - | 67.30% | 41.18% | 93.42% |
| | | | With Baseline (Paired) | 0.7574 | +0.0326 | 66.32% | 37.25% | 95.39% |
| **Step 18-B: Explicit Full Set** | 36 Cross + 14 Within | 36 / 50 | Without Baseline (Unpaired) | 0.7314 | - | 69.85% | 64.71% | 75.00% |
| | | | With Baseline (Paired) | 0.7568 | +0.0254 | 70.19% | 62.75% | 77.63% |
| **Reference: Dedicated Model B** | 100% Paired (No Imputation) | 13 / 15 | Dedicated Paired Only | 0.8256 | - | 74.11% | 66.67% | 81.58% |

---

## 6. Baseline Feature Importance Suppression (Section 7.2.1)

Comparison of feature scores and rankings between dedicated Model B (100% paired) and unified Model Step 16 (50% imputed):

| Feature Category | Feature Name | Dedicated Model B (100% Paired) | Unified Model Step 16 (50% Imputed) | Observation |
| :--- | :--- | :---: | :---: | :--- |
| **Current** | `cur_ch1_t_symmetry_idx` | 1.3739 (Rank 1) | 1.3739 (Rank 1) | Maintains primary status |
| **Current** | `cur_ch1_t_to_r_ratio` | 1.2144 (Rank 2) | 1.2144 (Rank 2) | Maintains primary status |
| **Baseline** | `prev_ch0_j_dev_mv` | 0.2078 (Rank 12) | 0.0699 (Rank 51) | 66% score reduction |
| **Baseline** | `prev_ch1_s_depth_mv` | 0.1969 (Rank 14) | 0.0666 (Rank 53) | 66% score reduction |
| **Baseline** | `prev_ch1_r_amp_mv` | 0.1853 (Rank 16) | 0.1002 (Rank 50) | 46% score reduction |
| **Indicator** | `baseline_available` | - (Constant 1) | 0.0144 (Rank 84) | Minimal contribution |

---

## 7. Automated Interaction Selection Divergence (Section 7.2.2)

Comparison of interaction terms selected by FAST under 100% paired data versus 50% imputed missing baseline data:

| Rank | Dedicated Model B (Budget=15) | Unified Step 16 (Auto FAST=15) | Unified Step 18-A (Auto FAST=50) |
| :---: | :--- | :--- | :--- |
| **Top 1** | `cur_ch1_st_dev & cur_ch1_s_depth` (Single snapshot) | `cur_ch1_st_integral & cur_ch1_s_depth` (Single snapshot) | `cur_ch1_st_integral & cur_ch1_s_depth` (Single snapshot) |
| **Top 2** | `cur_ch1_t_to_r & prev_ch1_t_to_r` (Cross-temporal) | `cur_ch1_j_dev & cur_ch1_t_symmetry` (Single snapshot) | `cur_ch1_j_dev & cur_ch1_t_symmetry` (Single snapshot) |
| **Top 3** | `cur_ch1_st_integ & prev_ch1_t_to_r` (Cross-temporal) | `cur_ch1_r_amp & cur_ch1_t_to_r` (Single snapshot) | `cur_ch1_r_amp & cur_ch1_t_to_r` (Single snapshot) |
| **Top 4** | `cur_ch1_t_amp & prev_ch1_t_amp` (Cross-temporal) | `cur_ch0_rs_ratio & cur_ch1_qtc_frid` (Single snapshot) | `cur_ch0_rs_ratio & cur_ch1_qtc_frid` (Single snapshot) |
| **Top 5** | `cur_ch0_j_dev & prev_ch0_j_dev` (Cross-temporal) | `cur_ch1_st_dev & cur_ch1_t_amp` (Single snapshot) | `cur_ch1_st_dev & cur_ch1_t_amp` (Single snapshot) |
| **...** | 13 of 15 terms are cross-temporal | 0 of 15 terms are cross-temporal | 0 of 50 terms are cross-temporal |
| **Cross-Temporal Ratio** | **86.7% (13 / 15)** | **0.0% (0 / 15)** | **0.0% (0 / 50)** |
