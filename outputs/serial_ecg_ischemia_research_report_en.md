# Experimental Report: Explainable Boosting Machines for Longitudinal ECG Myocardial Ischemia Detection

**Project**: Interpretable Dynamic Ischemia Detection Using Explainable Boosting Machines (EBM) and Longitudinal ECG Pairing  
**Directory**: `/Users/bingerliang/antigravity_projects/edb_ischemia_project`  
**Focus**: Evaluating the impact of patient-specific historical baseline ECGs on interpretable generalized additive models under strict patient-level isolation.

---

## 1. Clinical Background & Hypotheses

### 1.1 Clinical Background
1. **Alarm Fatigue**:  
   Static threshold analysis on single ECG recordings frequently triggers false alarms due to inter-individual anatomical and physiological baseline differences (e.g., pre-existing ST elevation, early repolarization, and tachycardia-induced J-point depression).
2. **Black-Box Limitations**:  
   Deep neural networks (e.g., ResNet, LSTM) applied to time-series ECG data present two practical challenges:
   - *Lack of traceable attribution*: Inability to decompose predictions into specific electrophysiological wave fiducials.
   - *Shortcut learning*: Recent studies indicate sequence models may exploit visit frequency and sampling intervals rather than true electrophysiological progression.

### 1.2 Research Hypotheses
- **Hypothesis 1**: Without temporal interval metadata or frequency cues, can an intrinsically interpretable generalized additive model with interactions (EBM / $\text{GA}^2\text{M}$) learn dynamic longitudinal relationships from patient-specific baseline ECGs?
- **Hypothesis 2**: How does the addition of historical baseline ECGs alter model feature weights, and how do manually engineered difference metrics compare with raw feature concatenation?

---

## 2. Experimental Setup & Benchmark Protocol

### 2.1 Patient-Level Data Partitioning
- **Data Source**: European ST-T Database (EDB), 2-lead ambulatory ECG records, sampled at 250 Hz.
- **Data Cleaning**: Three records with ambiguous patient background (`e0129`, `e0133`, `e0161`) were excluded, leaving 87 records from 77 unique patients.
- **Patient-Level Split**:
  - **Training Set**: 53 patients, 1,111 paired samples (277 ischemic positive).
  - **Validation Set**: 12 patients, 232 paired samples (56 ischemic positive; used for hyperparameter tuning and threshold calibration).
  - **Test Set**: 12 patients, 203 paired samples (51 ischemic positive; held-out evaluation).

### 2.2 Feature Extraction & Model Configurations
1. **Fiducial Point Delineation**:  
   Bandpass filtering and beat template aggregation were applied to identify the isoelectric baseline ($V_{base}$), $R/S$ peaks, $J$ point, $ST_{60/80}$ offsets, $T$ peak, and $T$ end (tangent method).
2. **Electrophysiological Feature Set (36 features per ECG)**:
   - *Morphological metrics*: $ST_{dev}$ (mV), $J_{dev}$ (mV), $ST_{slope}$ (mV/s), $ST_{integral}$ (mV·ms).
   - *Repolarization and interval metrics*: QT, QTc (Bazett and Fridericia corrections), T amplitude, T/R energy ratio, T-wave symmetry index, pathological Q wave.
   - *Spatial and axis metrics*: R/S amplitude ratio, 2-lead spatial axis angles $\theta_{QRS}$, $\theta_{ST}$, reciprocal ST product.
3. **Three Input Configurations**:
   - **Model A (Single Snapshot)**: 36 physical features from the current 10-second segment only.
   - **Model B-Diff (Engineered Differences)**: Current features plus precomputed difference metrics ($\Delta ST, \Delta T, \Delta HR, \Delta \theta$, and absolute magnitudes $|\Delta ST|, |\Delta T|$; 94 features total).
   - **Model B-Concat (Raw Concat)**: Direct concatenation of current (36 features) and baseline (36 features), 72 features total, without precomputed differences.

---

## 3. Macro Benchmark Results

All models were evaluated on the held-out 12-patient test set (203 samples) using validation-set calibrated decision thresholds.

### 3.1 Performance Comparison Across Model Architectures and Input Formats

| Model Architecture | Input Configuration | AUROC | AUPRC | Balanced Acc | Sensitivity | Specificity | F1-Score | FPR Posture | FPR Tachycardia |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Model A (Single) | 0.6594 | 0.4509 | 56.73% | 52.94% | 60.53% | 0.3913 | 78.26% | 53.33% |
| | Model B-Diff (Engineered) | 0.7675 | 0.5412 | 70.79% | 78.43% | 63.16% | 0.5442 | 95.65% | 26.67% |
| | Model B-Concat (Raw) | 0.5921 | 0.3132 | 58.06% | 50.98% | 65.13% | 0.4000 | 95.65% | 40.00% |
| **XGBoost** | Model A (Single) | 0.6654 | 0.4067 | 62.62% | 62.75% | 62.50% | 0.4571 | 100.0% | 40.00% |
| | Model B-Diff (Engineered) | 0.8270 | 0.6653 | 71.49% | 66.67% | 76.32% | 0.5620 | 95.65% | 33.33% |
| | Model B-Concat (Raw) | 0.7377 | 0.5179 | 63.59% | 68.63% | 58.55% | 0.4698 | 100.0% | 26.67% |
| **EBM ($\text{GA}^2\text{M}$)** | Model A (Single) | 0.7377 | 0.6096 | 64.90% | 70.59% | 59.21% | 0.4832 | 91.30% | 40.00% |
| | Model B-Diff (Engineered) | 0.8046 | 0.6121 | 76.39% | 78.43% | 74.34% | 0.6154 | 82.61% | 13.33% |
| | Model B-Concat (Raw) | 0.8256 | 0.7166 | 74.11% | 70.59% | 77.63% | 0.5950 | 69.57% | 13.33% |

### 3.2 Observed Model Behaviors

1. **Linear Model and Explicit Differences**:  
   In linear logistic regression, AUROC was 0.7675 with engineered difference terms, but decreased to 0.5921 under raw concatenation. Linear models cannot represent non-monotonic absolute deviations ($|x_{cur} - x_{prev}|$) without explicit transformation terms.
2. **Decision Trees and Axis Alignment**:  
   XGBoost achieved an AUROC of 0.8270 with engineered differences, but dropped to 0.7377 under raw concatenation. Standard decision tree splits are orthogonal to coordinate axes ($x_{cur} > c$ or $x_{prev} > c$), requiring multiple stair-step approximations to model diagonal boundaries ($x_{cur} - x_{prev} > \delta$).
3. **EBM Generalized Additive Representation**:  
   EBM formulation $g(E[y]) = \sum f_i(x_i) + \sum f_{ij}(x_i, x_j)$ supports smooth relative differences through opposing univariate spline functions and pairwise interaction terms. Under raw concatenation, EBM achieved 0.8256 AUROC and 0.7166 AUPRC without manual difference calculation.

---

## 4. Global Interpretability Analysis

Feature importance values extracted from EBM illustrate shifts in model weighting following the introduction of baseline measurements.

### 4.1 Top 15 Global Terms (Univariate and Interaction Terms Combined)

| Rank | Model A (Single Snapshot) Term | Term Type | Score | Model B (Raw Concat) Term | Term Type | Score |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| 1 | `cur_ch1_t_symmetry_idx` | Univariate | 0.3710 | `cur_rr_std_ms` | Univariate | 0.3091 |
| 2 | `cur_ch1_t_to_r_ratio` | Univariate | 0.3603 | `cur_ch0_j_dev_mv` | Univariate | 0.2699 |
| 3 | `cur_rr_std_ms` | Univariate | 0.3512 | `cur_hr_bpm` | Univariate | 0.2673 |
| 4 | `cur_rr_std_ms & cur_ch0_j_dev_mv` | Interaction | 0.2791 | `cur_ch1_st_dev_mv & cur_ch1_s_depth_mv` | Interaction | 0.2634 |
| 5 | `cur_ch0_j_dev_mv` | Univariate | 0.2548 | `cur_ch1_t_to_r_ratio` | Univariate | 0.2410 |
| 6 | `cur_ch0_s_depth_mv & cur_ch0_rs_ratio` | Interaction | 0.2337 | `prev_hr_bpm` | Univariate | 0.2157 |
| 7 | `cur_ch0_rs_ratio` | Univariate | 0.2249 | `prev_ch1_r_amp_mv` | Univariate | 0.2119 |
| 8 | `cur_ch0_st_slope & cur_ch0_st_integral` | Interaction | 0.2090 | `prev_ch0_j_dev_mv` | Univariate | 0.1965 |
| 9 | `cur_ch1_st_dev_mv & cur_ch1_s_depth_mv` | Interaction | 0.2040 | `prev_ch1_s_depth_mv` | Univariate | 0.1910 |
| 10 | `cur_ch1_st_slope_mv_s` | Univariate | 0.1999 | `prev_ch1_t_to_r_ratio` | Univariate | 0.1877 |
| 11 | `cur_ch1_r_amp_mv` | Univariate | 0.1991 | `cur_ch1_st_integral_mv_ms` | Univariate | 0.1831 |
| 12 | `cur_ch1_s_depth_mv & cur_ch1_t_to_r` | Interaction | 0.1974 | `cur_ch1_t_symmetry_idx` | Univariate | 0.1754 |
| 13 | `cur_ch1_s_depth_mv` | Univariate | 0.1969 | `cur_ch1_t_to_r & prev_ch1_t_to_r` | Cross-Temporal | 0.1727 |
| 14 | `cur_rr_std_ms & cur_reciprocal_st_prod` | Interaction | 0.1942 | `cur_ch1_st_integral & prev_ch1_t_to_r` | Cross-Temporal | 0.1704 |
| 15 | `cur_rr_std_ms & cur_ch1_r_amp_mv` | Interaction | 0.1902 | `cur_ch1_t_amp_mv & prev_ch1_t_amp_mv` | Cross-Temporal | 0.1696 |

### 4.2 Top 10 Univariate Main Effects Comparison

| Rank | Model A (Single) Feature | Score | Model B (Concat) Feature | Score | Context Comparison |
| :---: | :--- | :---: | :--- | :---: | :--- |
| 1 | `cur_ch1_t_symmetry_idx` | 0.3710 | `cur_rr_std_ms` | 0.3091 | Model A prioritizes static morphology; Model B shifts to heart rate variability |
| 2 | `cur_ch1_t_to_r_ratio` | 0.3603 | `cur_ch0_j_dev_mv` | 0.2699 | J-point elevation/depression increases in rank |
| 3 | `cur_rr_std_ms` | 0.3512 | `cur_hr_bpm` | 0.2673 | Instantaneous heart rate contribution |
| 4 | `cur_ch0_j_dev_mv` | 0.2548 | `cur_ch1_t_to_r_ratio` | 0.2410 | Repolarization to depolarization ratio |
| 5 | `cur_ch0_rs_ratio` | 0.2249 | `prev_hr_bpm` | 0.2157 | Baseline heart rate reference term |
| 6 | `cur_ch1_st_slope_mv_s` | 0.1999 | `prev_ch1_r_amp_mv` | 0.2119 | Baseline R-wave amplitude reference |
| 7 | `cur_ch1_r_amp_mv` | 0.1991 | `prev_ch0_j_dev_mv` | 0.1966 | Baseline J-point position reference |
| 8 | `cur_ch1_s_depth_mv` | 0.1969 | `prev_ch1_s_depth_mv` | 0.1910 | Baseline S-wave depth reference |
| 9 | `cur_hr_bpm` | 0.1826 | `prev_ch1_t_to_r_ratio` | 0.1877 | Baseline T/R ratio reference |
| 10 | `cur_ch0_t_to_r_ratio` | 0.1786 | `cur_ch1_st_integral_mv_ms` | 0.1831 | Integrated ST segment displacement |

### 4.3 Interaction Terms Selected by EBM

Out of 2,556 possible pairwise candidate interactions, EBM's FAST algorithm selected 15 interaction terms for each model.

| Rank | Model A (Single Snapshot) Interaction | Score | Model B (Raw Concat) Interaction | Score |
| :---: | :--- | :---: | :--- | :---: |
| 1 | `cur_rr_std & cur_ch0_j_dev` | 0.2791 | `cur_ch1_st_dev & cur_ch1_s_depth` | 0.2634 |
| 2 | `cur_ch0_s_depth & cur_ch0_rs_ratio` | 0.2337 | `cur_ch1_t_to_r & prev_ch1_t_to_r` | 0.1727 |
| 3 | `cur_ch0_st_slope & cur_ch0_st_integral` | 0.2090 | `cur_ch1_st_integral & prev_ch1_t_to_r` | 0.1704 |
| 4 | `cur_ch1_st_dev & cur_ch1_s_depth` | 0.2040 | `cur_ch1_t_amp & prev_ch1_t_amp` | 0.1696 |
| 5 | `cur_ch1_s_depth & cur_ch1_t_to_r` | 0.1974 | `cur_ch0_j_dev & prev_ch0_j_dev` | 0.1605 |
| 6 | `cur_rr_std & cur_reciprocal_st_prod` | 0.1942 | `cur_ch0_st_integ & prev_ch0_st_integ` | 0.1519 |
| 7 | `cur_rr_std & cur_ch1_r_amp` | 0.1902 | `cur_ch0_j_dev & prev_ch1_s_depth` | 0.1515 |
| 8 | `cur_ch1_j_dev & cur_ch1_s_depth` | 0.1690 | `cur_ch1_j_dev & prev_ch1_j_dev` | 0.1499 |
| 9 | `cur_hr_bpm & cur_ch1_qtc_bazett` | 0.1680 | `cur_ch1_st_integ & prev_ch1_st_integ` | 0.1478 |
| 10 | `cur_ch1_st_integ & cur_ch1_s_depth` | 0.1578 | `cur_ch0_t_amp & prev_ch0_t_amp` | 0.1474 |
| 11 | `cur_ch1_st_dev & cur_ch1_st_integ` | 0.1463 | `cur_ch0_t_amp & prev_ch1_rs_ratio` | 0.1363 |
| 12 | `cur_ch1_st_integ & cur_ch1_qt` | 0.1381 | `cur_ch0_st_integ & prev_ch1_rs_ratio` | 0.1192 |
| 13 | `cur_ch1_t_amp & cur_ch1_r_amp` | 0.1252 | `cur_ch0_t_to_r & prev_ch0_t_to_r` | 0.1145 |
| 14 | `cur_ch1_r_amp & cur_ch1_t_to_r` | 0.1095 | `cur_ch0_t_amp & prev_ch0_st_integ` | 0.1014 |
| 15 | `cur_ch1_st_integ & cur_ch1_r_amp` | 0.0841 | `cur_spatial_theta_st & prev_spatial_theta_st` | 0.0909 |

**Summary**: In Model B, 13 of the 15 automatically selected terms were cross-temporal (combining current and baseline features), and 8 pairs matched identical features across time points (e.g., `cur_T & prev_T`, `cur_J & prev_J`).

---

## 5. Local Waterfall Analysis & Case Reviews

Across the 203 test samples, Model A and Model B produced divergent classification outcomes on 50 samples (24.6% discordance rate). Below are representative examples illustrating local attribution.

### 5.1 Case 1: Detection of Low-Amplitude Ischemia
- **Sample**: Patient `P04` (`e0127`), onset at 01:09:34, baseline at 00:43:10 (26-minute interval), true label $y = 1$.
- **Model Predictions**: $P_A = 0.0674$ (below threshold 0.180) $\rightarrow$ $P_B = 0.7496$ (positive prediction).
- **Attribution**: The absolute ST elevation was 0.08 mV. Cross-temporal term `cur_ch0_t_amp & prev_ch0_t_amp` contributed +0.776 log-odds based on relative variation.
- **Figure**: [case1_subtle_ischemia_caught_by_B.png](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/waterfall/case1_subtle_ischemia_caught_by_B.png)

### 5.2 Case 2: Correction of Tachycardia-Associated False Alarm
- **Sample**: Patient `P35` (`e0206`), window at 00:26:40 (HR = 118 bpm), true label $y = 0$ (normal).
- **Model Predictions**: $P_A = 0.8463$ (false positive) $\rightarrow$ $P_B = 0.2022$ (adjusted below threshold).
- **Attribution**: Tachycardia-associated J-point shift produced positive log-odds in Model A via `cur_rr_std & cur_j_dev` (+0.965). In Model B, baseline terms `prev_qt` (-0.701) and `prev_j_dev` (-0.650) supplied negative offsets (-1.35 total), adjusting the predicted probability.
- **Figure**: [case3_chronic_st_elevation_corrected_by_B.png](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/waterfall/case3_chronic_st_elevation_corrected_by_B.png)

### 5.3 Case 3: Probability Shift in Concordant Positive Case
- **Sample**: Patient `P08` (`e0151`), onset at 01:21:25, true label $y = 1$.
- **Model Predictions**: $P_A = 0.1992$ (near threshold) $\rightarrow$ $P_B = 0.8139$.
- **Attribution**: Cross-temporal term `cur_ch1_t_amp & prev_ch1_t_amp` added +0.912 log-odds relative to baseline.
- **Figure**: [concordant_case1_confidence_surge_P08.png](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/waterfall/concordant_case1_confidence_surge_P08.png)

---

## 6. Analysis of Failure Modes

Among the 51 true positive events in the test set, 5 events (9.8%) detected by Model A were missed by Model B. Review of these cases indicates three recurring patterns:

### 6.1 Baseline Conduction Abnormality
- **Sample**: Patient `P35` (`e0206`), episodes `PR_test_0036` to `0038`.
- **Predictions**: Model A detected the events ($P_A = 0.27 - 0.67$), whereas Model B predicted below threshold ($P_B = 0.04 - 0.07$).
- **Observation**: The assigned baseline window at 00:00:50 contained ventricular conduction abnormalities. The model attributed current morphological deviations to pre-existing abnormality, yielding negative log-odds contributions (e.g., `prev_qt` -0.65).
- **Figure**: [failure_case1_baseline_poisoning_P35.png](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/waterfall/failure_case1_baseline_poisoning_P35.png)

### 6.2 Post-Ischemic Transition Baseline
- **Sample**: Patient `P21` (`e0106`), window `PR_test_0056` (true normal, $y = 0$).
- **Predictions**: $P_A = 0.0723$ (true negative) $\rightarrow$ $P_B = 0.7286$ (false positive).
- **Observation**: The baseline window occurred during recovery from an earlier ischemic episode. The term `cur_st_integ & prev_st_integ` registered an artificial relative offset (+0.914 log-odds) against normal resting morphology.
- **Figure**: [failure_case2_pseudo_dynamic_shift_P21.png](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/waterfall/failure_case2_pseudo_dynamic_shift_P21.png)

### 6.3 Posture-Induced Axis Rotation
False positive rates on posture changes (`neg_posture`) remained elevated across all tested models lacking spatial geometric constraints (Logistic Regression: 95.6%, EBM: 82.6%, XGBoost: 95.6%).

---

## 7. Exploratory Unified Dual-Mode Experiments & Scalability Limits

To test whether a single EBM model can handle both single-input (unpaired) and paired inputs, models were trained on 2,222 samples (with median imputation and a binary `baseline_available` flag) and evaluated on the 203 test samples under missing-baseline and available-baseline conditions.

### 7.1 Quantitative Comparison Across Search Configurations

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

- Scatter plot comparison: [`outputs/models/unified_explicit_ebm_prob_scatter.png`](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/unified_explicit_ebm_prob_scatter.png)
- Waterfall plot: [`outputs/models/waterfall/explicit_waterfall_case1_ischemia_boost_P21.png`](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/waterfall/explicit_waterfall_case1_ischemia_boost_P21.png)

### 7.2 Mechanism: Interaction Selection Under Partial Missingness

Comparing dedicated Model B (100% paired) with the unified model (50% imputed) revealed two structural observations:

#### (1) Baseline Feature Importance Suppression

| Feature Category | Feature Name | Dedicated Model B (100% Paired) | Unified Model Step 16 (50% Imputed) | Observation |
| :--- | :--- | :---: | :---: | :--- |
| Current | `cur_ch1_t_symmetry_idx` | 1.3739 (Rank 1) | 1.3739 (Rank 1) | Unchanged |
| Current | `cur_ch1_t_to_r_ratio` | 1.2144 (Rank 2) | 1.2144 (Rank 2) | Unchanged |
| Baseline | `prev_ch0_j_dev_mv` | 0.2078 (Rank 12) | 0.0699 (Rank 51) | 66% reduction in importance score |
| Baseline | `prev_ch1_s_depth_mv` | 0.1969 (Rank 14) | 0.0666 (Rank 53) | 66% reduction in importance score |
| Baseline | `prev_ch1_r_amp_mv` | 0.1853 (Rank 16) | 0.1002 (Rank 50) | 46% reduction in importance score |
| Indicator | `baseline_available` | - | 0.0144 (Rank 84) | Minimal contribution |

#### (2) Automated Interaction Selection Divergence

| Rank | Dedicated Model B (Budget=15) | Unified Step 16 (Auto FAST=15) | Unified Step 18-A (Auto FAST=50) |
| :---: | :--- | :--- | :--- |
| Top 1 | `cur_ch1_st_dev & cur_ch1_s_depth` (Single) | `cur_ch1_st_integral & cur_ch1_s_depth` (Single) | `cur_ch1_st_integral & cur_ch1_s_depth` (Single) |
| Top 2 | `cur_ch1_t_to_r & prev_ch1_t_to_r` (Cross) | `cur_ch1_j_dev & cur_ch1_t_symmetry` (Single) | `cur_ch1_j_dev & cur_ch1_t_symmetry` (Single) |
| Top 3 | `cur_ch1_st_integ & prev_ch1_t_to_r` (Cross) | `cur_ch1_r_amp & cur_ch1_t_to_r` (Single) | `cur_ch1_r_amp & cur_ch1_t_to_r` (Single) |
| Top 4 | `cur_ch1_t_amp & prev_ch1_t_amp` (Cross) | `cur_ch0_rs_ratio & cur_ch1_qtc_frid` (Single) | `cur_ch0_rs_ratio & cur_ch1_qtc_frid` (Single) |
| Top 5 | `cur_ch0_j_dev & prev_ch0_j_dev` (Cross) | `cur_ch1_st_dev & cur_ch1_t_amp` (Single) | `cur_ch1_st_dev & cur_ch1_t_amp` (Single) |
| ... | 13 of 15 terms are cross-temporal | 0 of 15 terms are cross-temporal | 0 of 50 terms are cross-temporal |
| Cross-Temporal Ratio | 86.7% (13 / 15) | 0.0% (0 / 15) | 0.0% (0 / 50) |

Under 50% missing baseline data, greedy interaction search prioritized fully observed within-current pairs over cross-temporal pairs, resulting in negligible baseline gains (+0.0031 AUROC). Explicit domain-informed pair injection (Step 17 and Step 18-B) recovered performance (+0.0326 AUROC gain, 62.75% sensitivity).

### 7.3 Architectural Scalability Considerations

1. **Multi-Condition Extension**:  
   Predefining pairwise interactions is feasible for isolated ischemic ST-T analysis, but scales combinatorially when expanding across diverse cardiac pathologies (arrhythmias, hypertrophy, conduction blocks).
2. **Variable-Length Longitudinal Sequences**:  
   Static tabular concatenation becomes unwieldy when accommodating variable numbers of historical ECGs ($t-1, \dots, t-n$) with non-uniform recording intervals and missing data.

---

## 8. Decoupled Evidence Retrieval & Diagnostic Glass-Box Framework

To separate sequential history processing from interpretable diagnosis, a decoupled two-stage architecture is considered:

```
                  Variable-Length Historical ECG Sequence
                       [ECG_1, ECG_2, ..., ECG_t-1]
                                    │
                                    ▼
          ┌───────────────────────────────────────────────────┐
          │             Evidence Retrieval Model              │
          │    - Sequence model (e.g., Attention / RNN)       │
          │    - Role: Identify relevant historical ECGs and  │
          │      features; no direct diagnostic prediction    │
          │    - Output: Fixed K evidence slots + provenance  │
          └─────────────────────────┬─────────────────────────┘
                                    │
                                    ▼ Structured evidence
                    [Selected Historical Features (K items)]
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          [Current ECG Features]         [Retrieved Evidence]
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
          ┌───────────────────────────────────────────────────┐
          │               Diagnostic Glass-Box                │
          │    - Explainable additive model (e.g., EBM)       │
          │    - Fixed input dimension (36 + K)               │
          │    - Outputs final prediction and attribution     │
          └─────────────────────────┬─────────────────────────┘
                                    ▼
                 Final Prediction & Verifiable Evidence Chain
```

### 8.1 Key Technical Considerations
1. **Shortcut Prevention**:  
   Ensure the retrieval stage does not pass visit frequency, total sequence length, or recording interval cues to the diagnostic model.
2. **Training Objectives**:  
   Define independent training or gating objectives for evidence selection without introducing diagnostic leakage.
3. **Spatial Geometric Constraints**:  
   Incorporate lead projection terms ($\Delta \theta_{QRS}^\perp$) to distinguish frontal plane electrical axis rotation from ischemic vector displacement.

---

## Data & Model File Index

- Full test predictions: [`outputs/models/ebm_all_test_predictions.csv`](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/ebm_all_test_predictions.csv)
- Dual-mode paired predictions: [`outputs/models/unified_explicit_ebm_paired_predictions.csv`](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/unified_explicit_ebm_paired_predictions.csv)
- Interaction budget comparison: [`outputs/models/interaction_budget_50_comparison.csv`](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/interaction_budget_50_comparison.csv)
- Discordant cases (N=50): [`outputs/models/ebm_discordant_cases.csv`](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/ebm_discordant_cases.csv)
- High divergence cases (N=20): [`outputs/models/waterfall/concordant_large_divergence_cases.csv`](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/waterfall/concordant_large_divergence_cases.csv)
- Top driver summary: [`outputs/models/waterfall/discordant_top_drivers_summary.csv`](file:///Users/bingerliang/antigravity_projects/edb_ischemia_project/outputs/models/waterfall/discordant_top_drivers_summary.csv)
