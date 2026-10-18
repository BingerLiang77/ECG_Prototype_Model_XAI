# A Longitudinal Interpretable Model for ECG Analysis

## 1. Research Idea

The goal is to develop an interpretable model that predicts the current ECG outcome by incorporating both the current ECG and a variable number of historical ECGs.

Unlike conventional models that treat each ECG independently, the proposed approach aims to leverage longitudinal information while maintaining traceability to individual historical ECGs and their features.

## 2. Proposed Model Architecture

**Step 1 — Feature extraction**

Extract a fixed set of \(F\) clinical or morphological features from each ECG. The number of historical ECGs \(T\) may vary between patients.

**Step 2 — Temporal weighting**

Use a Cross-Attention-based mechanism to evaluate the relevance of each historical ECG to the current ECG. Apply Softmax to obtain temporal weights, \(\alpha_t\), across historical records.

**Step 3 — Feature-level weighting**

Estimate a separate feature weight, \(\beta_{t,f}\), for each feature in each historical ECG. These weights can be positive or negative, allowing the model to represent evidence in different directions.

**Step 4 — Feature aggregation**

Combine the weighted values of the same feature across historical ECGs:

\[
z_f = \sum_{t=1}^{T} \alpha_t \beta_{t,f} x_{t,f}
\]

This produces a fixed-length vector of \(F\) aggregated features, regardless of the number of historical ECGs.

## 3. Alternative Prediction Models

Two prediction models will be considered, depending on the trade-off between predictive flexibility and evidence-level interpretability.

### Option A — Linear Model

Use a linear classifier, such as logistic regression, to predict the current ECG outcome:

\[
\operatorname{logit}(p) = b + \sum_{f=1}^{F} w_f z_f
\]

Substituting the aggregation formula gives an exact decomposition of the model's logit:

\[
\operatorname{logit}(p) = b + \sum_f \sum_t C_{t,f}
\]

where

\[
C_{t,f} = w_f \alpha_t \beta_{t,f} x_{t,f}
\]

Each \(C_{t,f}\) represents the contribution of a specific feature in a specific historical ECG to the final logit, under the defined model structure.

**Advantages:**
- Transparent mathematical structure.
- Exact decomposition of the logit into historical feature-level contributions.
- Low computational complexity.

**Limitation:** A standard linear model cannot capture arbitrary nonlinear relationships or feature interactions without explicitly adding suitable terms.

### Option B — Explainable Boosting Machine (EBM)

Use EBM to predict the outcome from the aggregated feature vector.

EBM can capture nonlinear feature effects and selected feature interactions, providing richer explanations at the aggregated-feature level.

**Advantages:**
- More flexible than a standard linear model.
- Can reveal nonlinear relationships between aggregated ECG features and the prediction.
- Can model selected interactions between aggregated features.

**Limitation:** Because EBM is nonlinear, its aggregated-feature contributions generally cannot be uniquely and directly decomposed into individual historical ECG contributions using the same simple formula as the linear model.

## 4. Interpretability and Traceability

The proposed architecture aims to provide two complementary levels of explanation:

- **Aggregated-feature explanations:** Identify which features and feature interactions influence the prediction.
- **Historical-evidence traceability:** Record temporal weights, feature weights, weighted evidence, timestamps, and source ECG identifiers.

The linear model provides a direct mathematical link between individual historical feature contributions and the final logit. EBM provides richer nonlinear explanations at the aggregated-feature level, but further attribution methods or architectural constraints may be needed to quantify individual historical evidence contributions.

The learned weights should not automatically be interpreted as clinical importance. Their stability and faithfulness must be evaluated.

## 5. Initial Evaluation Plan

1. Compare a current-ECG-only baseline with the longitudinal model.
2. Compare the linear predictor and EBM in terms of predictive performance.
3. Evaluate temporal and feature-weight stability.
4. Test model behaviour when selected historical evidence is removed or perturbed.
5. Examine whether the learned relationships are consistent with plausible ECG patterns.

Patient-level data splitting will be used to avoid leakage between training and evaluation sets.

## 6. Main Research Question

Can a longitudinal ECG model improve prediction by selectively weighting historical ECGs and their features, while maintaining a transparent and verifiable link between the historical evidence and the final prediction?

The initial priority is to establish the value of longitudinal information and evidence selection. The choice between a linear predictor and EBM will then be informed by predictive performance and the required level of interpretability.
