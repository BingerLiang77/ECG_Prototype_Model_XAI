# Explainable Boosting Machine (EBM) A/B Test Benchmark Report

## 1. Test-Set Performance Comparison

|                 |   EBM Model A (Single) |   EBM Model B (Paired Dynamic) |   Absolute Gain (B - A) |
|:----------------|-----------------------:|-------------------------------:|------------------------:|
| AUROC           |                 0.7313 |                         0.8002 |                  0.0689 |
| AUPRC           |                 0.6203 |                         0.5793 |                 -0.041  |
| Accuracy        |                 0.7389 |                         0.7537 |                  0.0148 |
| Sensitivity     |                 0.4902 |                         0.7255 |                  0.2353 |
| Specificity     |                 0.8224 |                         0.7632 |                 -0.0592 |
| Balanced_Acc    |                 0.6563 |                         0.7443 |                  0.088  |
| F1              |                 0.4854 |                         0.5968 |                  0.1113 |
| FPR_Posture     |                 0.3913 |                         0.8261 |                  0.4348 |
| FPR_Arrhythmia  |                 0.0526 |                         0.0789 |                  0.0263 |
| FPR_Tachycardia |                 0.2    |                         0.1333 |                 -0.0667 |
| FPR_Clean       |                 0.1711 |                         0.1579 |                 -0.0132 |

## 2. Top 15 Feature & Interaction Drivers in Dynamic Model B

| Feature / Interaction                            |   Importance (Mean Abs Score) |
|:-------------------------------------------------|------------------------------:|
| abs_delta_ch0_r_amp_mv & abs_delta_ch1_t_amp_mv  |                        0.4505 |
| cur_rr_std_ms & cur_reciprocal_st_product        |                        0.3216 |
| cur_hr_bpm & delta_ch0_t_amp_mv                  |                        0.3038 |
| delta_ch1_t_to_r_ratio                           |                        0.2862 |
| cur_ch1_t_to_r_ratio                             |                        0.2732 |
| abs_cur_ch0_st_dev_mv & abs_delta_ch1_t_amp_mv   |                        0.272  |
| delta_hr_bpm                                     |                        0.2715 |
| prev_time_gap_s                                  |                        0.2691 |
| cur_ch1_t_symmetry_idx                           |                        0.2423 |
| delta_ch1_st_slope_mv_s                          |                        0.2363 |
| abs_cur_ch1_t_amp_mv                             |                        0.2334 |
| abs_delta_ch0_t_amp_mv                           |                        0.2306 |
| cur_ch0_t_to_r_ratio & prev_time_gap_s           |                        0.2251 |
| abs_delta_ch0_st_dev_mv & abs_delta_ch1_t_amp_mv |                        0.2232 |
| delta_ch1_t_amp_mv & delta_ch1_s_depth_mv        |                        0.2229 |

## 3. Key Clinical Findings

- **AUROC**: Model B achieved 0.8002 vs Model A 0.7313 (Gain: +0.0689)
- **AUPRC**: Model B achieved 0.5793 vs Model A 0.6203 (Gain: -0.0410)
- **Balanced Accuracy**: Model B achieved 0.7443 vs Model A 0.6563 (Gain: +0.0880)
- **Posture False Positive Rate (体位误报率)**: Model A was 39.1%, Model B was 82.6%!
- **Arrhythmia False Positive Rate (早搏误报率)**: Model A was 5.3%, Model B was 7.9%!
- **Tachycardia False Positive Rate (心动过速误报率)**: Model A was 20.0%, Model B was 13.3%!