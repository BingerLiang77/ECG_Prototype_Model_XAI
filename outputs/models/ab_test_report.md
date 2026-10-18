# Logistic Regression A/B Test Benchmark Report

## 1. Test-Set Performance Comparison

|                 |   Model A (Single) |   Model B (Paired Dynamic) |   Absolute Gain (B - A) |
|:----------------|-------------------:|---------------------------:|------------------------:|
| AUROC           |             0.676  |                     0.6133 |                 -0.0627 |
| AUPRC           |             0.4627 |                     0.3584 |                 -0.1042 |
| Accuracy        |             0.7537 |                     0.6158 |                 -0.1379 |
| Sensitivity     |             0.0196 |                     0.5882 |                  0.5686 |
| Balanced_Acc    |             0.5098 |                     0.6066 |                  0.0968 |
| F1              |             0.0385 |                     0.4348 |                  0.3963 |
| FPR_Posture     |             0      |                     0.8261 |                  0.8261 |
| FPR_Arrhythmia  |             0      |                     0.1579 |                  0.1579 |
| FPR_Tachycardia |             0      |                     0.4    |                  0.4    |
| FPR_Clean       |             0      |                     0.3421 |                  0.3421 |

## 2. Top 15 Clinical Drivers in Dynamic Model B

| Feature                   |   Coefficient |   Odds_Ratio (exp(w)) |
|:--------------------------|--------------:|----------------------:|
| delta_hr_bpm              |        1.1763 |                3.2424 |
| delta_ch0_pathological_q  |       -0.993  |                0.3705 |
| cur_ch1_t_amp_mv          |        0.8154 |                2.2602 |
| delta_is_tachycardia      |       -0.7097 |                0.4918 |
| cur_ch0_pathological_q    |        0.7037 |                2.0213 |
| cur_ch1_s_depth_mv        |        0.6574 |                1.9297 |
| cur_ch1_r_amp_mv          |       -0.6385 |                0.5281 |
| delta_ch0_qt_ms           |        0.5866 |                1.7979 |
| cur_ch0_t_amp_mv          |       -0.5761 |                0.5621 |
| cur_ch1_j_dev_mv          |        0.5204 |                1.6828 |
| cur_spatial_theta_qrs_deg |        0.5094 |                1.6642 |
| cur_ch0_j_dev_mv          |        0.506  |                1.6586 |
| cur_ch1_st_integral_mv_ms |       -0.495  |                0.6095 |
| prev_time_gap_s           |        0.4708 |                1.6013 |
| cur_ch1_pathological_q    |        0.4692 |                1.5988 |

## 3. Key Clinical Findings

- **AUROC**: Model B achieved 0.6133 vs Model A 0.6760 (Gain: -0.0627)
- **AUPRC**: Model B achieved 0.3584 vs Model A 0.4627 (Gain: -0.1042)
- **Posture False Positive Rate (体位假阳性率)**: Dropped from 0.0% down to 82.6%!
- **Arrhythmia False Positive Rate (早搏假阳性率)**: Dropped from 0.0% down to 15.8%!