# Step 1 report: EDB annotation parsing & budget

- records: 87, patient groups (heuristic): 77, split sizes (53, 12, 12)
- patients with >1 record: 6: P05=e0154,e0155; P09=e0123,e0124,e0125,e0126; P14=e0136,e0139; P29=e0147,e0148; P41=e0162,e0163; P57=e0118,e0119,e0121,e0122
- ischemic episodes (upper-case, paired): 736  (ST 359, T 377); with peak annotation 736; incomplete-boundary episodes: {'open_at_end': 11, 'orphan_end': 3}
- axis-shift episodes (lower-case): 17 in records ['e0509', 'e0601', 'e0611', 'e0613', 'e0615']

## Total duration by state (hours; seconds-resolution masks)

| state | hours |
|---|---|
| record total | 174.00 |
| ischemia span | 45.22 |
| buffer ±2min | 19.79 |
| axis-shift span | 2.31 |
| bad noise (n/u) | 19.41 |
| AFIB rhythm | 0.35 |

## Candidate non-overlapping 10 s windows (grid) by class

| cls           |   test |   train |   val |   total |
|:--------------|-------:|--------:|------:|--------:|
| arrhythmia    |    255 |    1023 |   145 |    1423 |
| axis_edge     |      3 |       5 |     2 |      10 |
| buffer        |    725 |    4207 |   898 |    5830 |
| clean         |   4784 |   21544 |  4204 |   30532 |
| ischemia_span |   1817 |    9533 |  2158 |   13508 |
| noisy         |   1469 |    7204 |  1010 |    9683 |
| other         |      0 |      41 |     9 |      50 |
| posture       |    197 |     528 |    63 |     788 |
| tachy         |    110 |     555 |   151 |     816 |

## Positive clips (centred on AST/AT peak, noise-free)

| split   |   n_isch_ep |   n_pos |   n_pos_arr |   patients |
|:--------|------------:|--------:|------------:|-----------:|
| test    |          99 |      91 |           8 |         12 |
| train   |         531 |     432 |          19 |         53 |
| val     |         106 |      97 |           6 |         12 |

## Negative budget needed vs. available (ratio 1:2 and 1:3)

| split   | ratio   |   n_pos |   neg_needed |   max_neg_given_mix | bottleneck   |
|:--------|:--------|--------:|-------------:|--------------------:|:-------------|
| train   | 1:2     |     432 |          864 |                3520 | posture      |
| train   | 1:3     |     432 |         1296 |                3520 | posture      |
| val     | 1:2     |      97 |          194 |                 420 | posture      |
| val     | 1:3     |      97 |          291 |                 420 | posture      |
| test    | 1:2     |      91 |          182 |                1020 | arrhythmia   |
| test    | 1:3     |      91 |          273 |                1020 | arrhythmia   |

## Split balance (patients / pos / windows)

| split   |   n_pos |   n_posture |   n_arr |   n_tachy |   n_clean |   patients |
|:--------|--------:|------------:|--------:|----------:|----------:|-----------:|
| test    |      91 |         197 |     255 |       110 |      4784 |         12 |
| train   |     432 |         528 |    1023 |       555 |     21544 |         53 |
| val     |      97 |          63 |     145 |       151 |      4204 |         12 |