# Repository Structure

```
README.md
LICENSE
CITATION.cff
requirements.txt
src/
├── pca_fdr_nested_cv_V4_tailFDR.py
├── uci_loader.py
├── parkinson_loader.py
├── generate_low_variance_pc_synthetic.py
├── run_over_lowvar_synthetics_V4.py
├── generate_synthetic_imbalance_nxprop.py
├── run_synthetic_imbalance_nxprop.py
├── pca_fdr_sensitivity_tau_alpha.py
├── run_pca_fdr_sensitivity_tau_alpha.py
└── pca_geometry_diagnostics.py
analysis/
├── summarize_and_plot_lowvar_nested_v2.py
├── summarize_synthetic_imbalance_nxprop.py
├── summarize_pca_geometry_results.py
└── hybrid_vs_auxiliary_baseline.py
results/
```

> **Note:** The `results/` directory is intentionally empty in the repository. Large intermediate outputs are not versioned. Tested with Python 3.10.12.

---

## 1. Main Real-World Benchmark

### Script
`src/pca_fdr_nested_cv_V4_tailFDR.py`

This is the main experimental pipeline used for the real-world benchmark. The paper evaluates 17 binary classification datasets covering a broad range of sample sizes, dimensionalities, class distributions, and feature-to-sample ratios.

The benchmark includes datasets from:
- UCI Machine Learning Repository
- OpenML
- scikit-learn
- PC-GITA Parkinson voice data

### Methods Compared
- **ALL + linear SVM**: All standardized original predictors are supplied directly to the classifier.
- **PCA-$\\tau$ + linear SVM**: Only the principal components required to reach the cumulative explained-variance threshold $\\tau$ are retained.
- **HYBRID + linear SVM**: The PCA-$\\tau$ core is retained and FDR-significant tail PCs are selectively rescued. If no tail PC satisfies the rescue criterion, the selected HYBRID component set is identical to the PCA-$\\tau$ core, although classifier hyperparameter tuning is still performed independently for the two methods.
- **PLS-DA**: Partial least squares discriminant analysis is included as a supervised latent-space baseline.

### Cross-Validation
For the real-world datasets:
- **Outer cross-validation**: 5 folds
- **Outer repetitions**: 10
- **Total outer evaluations**: 50 per method and dataset
- **Inner cross-validation**: 5 folds

All data-dependent operations are fitted independently inside the corresponding training partition:
- Missing-value imputation
- Standardization
- PCA
- PCA truncation
- ANOVA
- Benjamini–Hochberg correction
- Rescue-set construction
- Classifier tuning

This avoids information leakage from validation or test samples.

### Group-Aware Validation
For datasets with repeated observations from the same subject, subject-level splitting is used so that observations from the same individual never appear simultaneously in training and validation/test partitions.

This applies to:
- PC-GITA representations
- UCI174 Parkinson

### Linear SVM Tuning
The regularization parameter is selected from:
$$C \in \{0.001, 0.01, 0.1, 1, 10\}$$

The optimal value is selected by maximizing mean inner-cross-validation MCC.

### PLS-DA Tuning
Candidate numbers of latent components are:
$$\{1, 2, 3, 5, 8, 10, 15, 20\}$$

restricted when necessary by the dimensionality and sample size of the training partition.

### Evaluation Metrics
The outer test folds report:
- Matthews correlation coefficient (MCC)
- ROC-AUC
- F1 score

Only MCC is used for inner model selection.

---

## 2. Controlled Low-Variance Synthetic Experiment

### Generator
`src/generate_low_variance_pc_synthetic.py`

### Runner
`src/run_over_lowvar_synthetics_V4.py`

### Summary and Plots
`analysis/summarize_and_plot_lowvar_nested_v2.py`

### Objective
This experiment isolates the failure mode targeted by HYBRID: discriminative information located in a low-variance direction that falls outside the PCA-$\\tau$ core.

Each dataset contains:
- $n = 400$
- $p = 20$

A block of 10 nuisance variables follows an equicorrelation structure with:
$$\rho = 0.70$$

Variables $x_{11}$ and $x_{12}$ are generated with strong positive correlation:
$$\rho(x_{11}, x_{12}) = 0.98$$

The discriminative signal is introduced exclusively along the low-variance contrast direction:
$$u_{\\text{minus}} = \frac{x_{11} - x_{12}}{\sqrt{2}}$$

Signal strength is controlled through the standardized within-class effect size:
$$d \in \{0, 0.5, 1\}$$

Ten independently generated datasets are produced for each value of $d$, yielding 30 synthetic datasets.

Each dataset is evaluated using:
- Outer 5-fold cross-validation
- 5 repetitions
- Inner 5-fold cross-validation

The experiment evaluates whether HYBRID can recover predictive information deliberately placed outside the variance-based PCA core.

---

## 3. Sample-Size & Class-Imbalance Synthetic Experiment

### Generator
`src/generate_synthetic_imbalance_nxprop.py`

### Runner
`src/run_synthetic_imbalance_nxprop.py`

### Summary
`analysis/summarize_synthetic_imbalance_nxprop.py`

### Objective
This experiment studies how the probability of tail-PC rescue changes with:
- Total sample size
- Minority-class proportion

The feature dimension is fixed at:
$$p = 100$$

Sample size varies over:
$$n \in \{200, 400, 800\}$$

Minority-class proportion varies over:
$$\{0.40, 0.20, 0.10, 0.05\}$$

Ten independently generated datasets are produced for every combination, yielding:
$$3 \times 4 \times 10 = 120 \text{ synthetic datasets}$$

### Latent Variance Spectrum
Data are generated in a latent principal-component space with smoothly decaying variance:
$$\lambda_j = 3 \cdot \exp\left(-\frac{j - 1}{28}\right) + 0.08 \quad \text{for } j = 1, \dots, 100$$

The spectrum parameters were chosen heuristically to generate a gradual decay with a non-negligible low-variance tail.

Discriminative information is planted in:
- PC5
- PC40
- PC85

with standardized class effect:
$$d = 0.80$$

The latent representation is transformed using a fixed orthogonal rotation, followed by Gaussian measurement noise with:
$$\sigma_{\\text{noise}} = 0.05$$

Only sample size and class proportion vary across conditions.

### Validation
Each dataset is evaluated using:
- Outer 5-fold cross-validation
- 10 repetitions
- Inner 5-fold cross-validation

The primary outcome is rescue frequency rather than predictive performance.

---

## 4. Sensitivity to $\\tau$ and $\\alpha_{\\text{FDR}}$

### Main Script
`src/pca_fdr_sensitivity_tau_alpha.py`

### Batch Runner
`src/run_pca_fdr_sensitivity_tau_alpha.py`

This analysis evaluates sensitivity to the two main HYBRID design parameters.

### Variance-Threshold Sensitivity
Across the 17 real-world datasets:
$$\\tau \in \{0.80, 0.90, 0.95\}$$
$$\\alpha_{\\text{FDR}} = 0.05$$

The primary quantity is the absolute MCC of HYBRID rather than only:
$$\Delta\text{MCC} = \text{MCC}_{\\text{HYBRID}} - \text{MCC}_{\\text{PCA}_\\tau}$$

This distinction is important because lowering $\\tau$ may weaken the PCA baseline and artificially inflate the apparent rescue gain.

The sensitivity analysis shows that:
- Lower $\\tau$ values leave a larger candidate tail for rescue.
- Higher $\\tau$ values retain a richer unsupervised PCA core.
- At $\\tau = 0.95$, HYBRID often becomes nearly indistinguishable from PCA-$\\tau$ because little tail space remains for rescue.
- $\\tau = 0.90$ is therefore interpreted as an intermediate operating point rather than as a universally optimal threshold.

### FDR-Threshold Sensitivity
For four diagnostic datasets:
- PC-GITA AWPE4
- clean1 / OpenML 40665
- Spambase
- UCI174 Parkinson

the analysis evaluates:
$$\\alpha_{\\text{FDR}} \in \{0.01, 0.05, 0.10\}$$
for each $\\tau \in \{0.80, 0.90, 0.95\}$.

Increasing $\\alpha_{\\text{FDR}}$ generally increases the number of rescued PCs, but does not produce a consistent monotonic improvement in predictive performance. The preferred FDR level is dataset dependent.

---

## 5. Diagnostic Analysis of Rescue Behavior

### Per-Fold Diagnostics
`src/pca_geometry_diagnostics.py`

This script produces detailed fold-level PCA and rescue diagnostics, including files such as:
- `*_all_pc_diagnostics.csv`
- `*_all_pc_loadings.csv`
- `*_eigen_spectrum.csv`
- `*_fold_geometry.csv`

### Summary
`analysis/summarize_pca_geometry_results.py`

This script aggregates the diagnostic outputs and produces:
- `*_geometry_summary.csv`

These summaries include quantities used in the paper's diagnostic comparison, such as:
- Severity of PCA truncation
- Rescue frequency
- Relative rescue size
- Adjusted $q$-values
- Median $q$-value of rescued components

The paper focuses on four representative datasets:
- **PC-GITA AWPE4**: successful rescue
- **clean1 / OpenML 40665**: successful rescue
- **Spambase**: frequent but ineffective rescue
- **UCI174 Parkinson**: unfavorable rescue

The objective is explanatory rather than predictive: the analysis does not attempt to derive a general rule for identifying in advance when HYBRID will succeed.

---

## 6. Auxiliary Baseline Comparison

### Script
`analysis/hybrid_vs_auxiliary_baseline.py`

This script generates the compact comparison between HYBRID and the two auxiliary baselines:
- **ALL**
- **PLS-DA**

For each dataset, differences in:
- MCC
- ROC-AUC
- F1

are summarized across the two auxiliary baselines.

---

## 7. Data Availability

Most benchmark datasets are obtained from public repositories including:
- UCI Machine Learning Repository
- OpenML
- scikit-learn

The PC-GITA Parkinson voice representations may require separate access according to the terms of the original data source.

Dataset loaders are provided in:
- `src/uci_loader.py`
- `src/parkinson_loader.py`

Users should download datasets from their original sources and configure local paths as required by the scripts. Large raw datasets are not redistributed in this repository.

---

## 8. Installation

A Python environment can be created from:

```bash
pip install -r requirements.txt
```

The code was developed and tested using Python 3.x.

Main dependencies include:
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `statsmodels`

Exact versions used for the archived release are listed in `requirements.txt`.

---

## 9. Reproducibility Notes

The repository contains the scripts required to reproduce the main analyses reported in the paper.

Some exploratory scripts used during method development, debugging, smoke testing, and discarded experimental variants are intentionally not included. The repository therefore represents the final reproducibility pipeline rather than the complete development history.

Because repeated cross-validation estimates are statistically dependent, paired performance differences are interpreted descriptively rather than as independent observations for formal hypothesis testing.

---

## 10. Citation

If you use this code or the method in your work, please cite:

```bibtex
@article{cuesta2026pca,
  title={PCA-$\\tau$ with ANOVA/FDR Rescue},
  author={Cuesta, J.},
  year={2026}
}
```

A permanent software DOI will be added after the archived release is deposited in Zenodo.

See also: `CITATION.cff`

---

## 11. License

See the `LICENSE` file.

---

## 12. AI-Assisted Preparation

OpenAI ChatGPT was used during the preparation of the associated manuscript to assist with methodological discussion, code review and debugging, figure preparation, alternative analyses, manuscript editing, and language refinement.

All generated suggestions, code, analyses, and text were critically reviewed, validated, and, where appropriate, modified by the author, who retained full responsibility for the scientific decisions, interpretation, and conclusions of the work.
