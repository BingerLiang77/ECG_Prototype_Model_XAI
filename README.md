# ECG XAI: Longitudinal Interpretable ECG Analysis for Myocardial Ischemia Detection

This repository contains the experimental pipeline, empirical benchmark results, interpretability visualizations, and next-generation model proposals for **explainable, longitudinal electrocardiogram (ECG) analysis**.

Evaluated on the **European ST-T Database (EDB)** under strict subject-level zero-leakage isolation.

---

## 📑 Research Reports & Model Proposals

### 1. Longitudinal Empirical Research Report (全景实验研究报告)
Comprehensive experimental evaluation comparing single-snapshot baseline models (Model A) against longitudinal paired dynamic models (Model B), detailing global feature importance, local waterfall attributions, failure mode taxonomy, and missingness ablation:
- 🇺🇸 **English Version**: [`outputs/serial_ecg_ischemia_research_report_en.md`](outputs/serial_ecg_ischemia_research_report_en.md)
- 🇨🇳 **Chinese Version**: [`outputs/serial_ecg_ischemia_research_report.md`](outputs/serial_ecg_ischemia_research_report.md)

### 2. Next-Generation Longitudinal Architecture Proposal (纵向可解释模型方案)
Proposed 4-step framework integrating Cross-Attention temporal relevance ($\alpha_t$) and bidirectionally bounded feature weighting ($\beta_{t,f}$) into a glass-box decision maker with exact linear decomposition:
- 🇺🇸 **English Proposal**: [`outputs/longitudinal_interpretable_ecg_model_proposal_en.md`](outputs/longitudinal_interpretable_ecg_model_proposal_en.md)
- 🇨🇳 **Chinese Proposal**: [`outputs/longitudinal_interpretable_ecg_model_proposal_zh.md`](outputs/longitudinal_interpretable_ecg_model_proposal_zh.md)

### 3. Benchmark Data Tables Notebook (实验数据纯享草稿)
Condensed reference notebook retaining all 7 experimental benchmark tables, feature ranking comparisons, and interaction logs without extended discussion:
- 🇺🇸 **English Notebook**: [`outputs/experiment_tables_draft_en.md`](outputs/experiment_tables_draft_en.md)

---

## 🔬 Key Experimental Findings

1. **Intrinsically Interpretable Superiority**:  
   Under raw feature concatenation (without manual feature engineering), Explainable Boosting Machines (EBM / $\text{GA}^2\text{M}$) achieved **0.8256 AUROC** and **0.7166 AUPRC** on locked patient test sets, matching or exceeding black-box deep ensembles while providing fully transparent local waterfall explanations.

2. **Empirical Proof of Feature Dilution & Gradient Preemption**:  
   Synthetic missingness ablation demonstrated that flat tabular concatenation with constant imputation systematically dampens baseline importance:
   - Increasing baseline missingness from 0% to 50% reduces automated cross-temporal interaction selection from **14/15 down to 0/15**.
   - Baseline feature importance share steadily decays from **42.2% down to 24.8%**, and key physiological feature `prev_ch0_j_dev_mv` drops from **Rank 9 to Rank 51**.
   - See: [`outputs/models/synthetic_missingness_ablation_curves.png`](outputs/models/synthetic_missingness_ablation_curves.png)

3. **Local Clinical Waterfall Attribution**:  
   Exact log-odds decomposition explains why longitudinal baseline pairing successfully rescues subtle ischemic changes (+0.776 log-odds boost) and suppresses tachycardia-induced false alarms (-1.35 log-odds adjustment).

---

## 📂 Repository Structure

```
.
├── outputs/
│   ├── serial_ecg_ischemia_research_report_en.md     # Full Research Report (English)
│   ├── serial_ecg_ischemia_research_report.md        # Full Research Report (Chinese)
│   ├── longitudinal_interpretable_ecg_model_proposal_en.md  # Proposal (English)
│   ├── longitudinal_interpretable_ecg_model_proposal_zh.md  # Proposal (Chinese)
│   ├── executive_summary.md                          # High-level summary
│   ├── features/                                     # Delineated electrophysiological feature tables
│   └── models/
│       ├── synthetic_missingness_ablation_curves.png # Step 19 ablation curves
│       ├── synthetic_missingness_ablation_results.csv# Step 19 ablation data
│       ├── unified_explicit_ebm_prob_scatter.png     # Prediction scatter plots
│       └── waterfall/                                # Case study waterfall charts
└── scripts/
    ├── step1_parse_split_stats.py                    # Dataset parsing & subject isolation split
    ├── ...
    ├── step11_raw_concat_ab_experiment.py            # Raw concatenation A/B benchmark
    ├── ...
    ├── step16_unified_dual_mode_ebm.py               # Unified dual-mode EBM experiment
    ├── step17_unified_explicit_interactions_ebm.py   # Explicit interaction injection
    ├── step18_interaction_budget_experiment.py       # Interaction budget exploration
    └── step19_synthetic_missingness_ablation.py      # Missingness ablation experiment
```

---

## 🛠 Reproduction

```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # (or numpy, pandas, scikit-learn, interpret, xgboost, matplotlib)

# Run raw concatenation A/B experiment
python scripts/step11_raw_concat_ab_experiment.py

# Run synthetic missingness ablation
python scripts/step19_synthetic_missingness_ablation.py
```
