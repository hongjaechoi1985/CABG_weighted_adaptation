# CABG Weighted Adaptation: Cross-Hospital Mortality Prediction

Research analysis code for evaluating **source-to-target transportability and weighted adaptation** of a random-forest model for in-hospital mortality after coronary artery bypass grafting (CABG), using Seoul National University Hospital (SNUH) as the source and Boramae Medical Center (BRM) as the target.

## Scope

The provided workflow includes:

1. Source-cohort training/internal validation (80:20 split) and out-of-fold, development-stage intercept calibration.
2. Direct external validation in a fixed BRM evaluation sample, with a separate BRM adaptation sample (30:70 target split).
3. Class-balanced, group-contribution-weighted pooled random-forest retraining for target weight `alpha` from 0 to 1 in increments of 0.1.
4. Unweighted pooled, globally class-balanced pooled, target-only and recalibration comparisons.
5. AUROC, AUPRC, Brier score, bootstrap intervals and calibration curves.
6. Adaptation-fraction sensitivity analysis (10%, 20%, 30%) and 100 repeated, stratified target 30:70 splits.
7. Feature- and domain-level SHAP analysis, stability summaries, explanation-drift metrics and SHAP dependence plots.

**Scope limitation:** This release contains the final random-forest weighted-adaptation analysis. It does **not** contain the earlier five-algorithm model-selection pipeline.

## Data availability and privacy

**No patient-level data are distributed with this repository.** The source and target hospital datasets are subject to institutional and patient privacy restrictions. Access is governed by the respective data custodians and applicable approvals. This code cannot reproduce the published numerical results without the original authorized datasets.

The analysis uses two CSV inputs, described in [`datasets/README.md`](datasets/README.md). The predictor names and target (`inhosp_death`) are declared in `code/analysis.py` under `feature_cols`, `target_col`, and `hospital_col`. The data dictionary, clinical units, and required code values should be verified locally. No synthetic data are included.

## Installation

Python 3 is required. Create an isolated virtual environment, then run:

```bash
pip install -r requirements.txt
```

Package versions are intentionally not claimed to match the original analysis environment. For strict environment replication, pin the package versions from the actual analysis machine and run the pipeline again.

## Running

Provide restricted data locally (default paths below), or set environment variables to the authorized CSV locations:

```bash
export CABG_MAIN_DATA="/secure/path/to/Total_TL_CABG_data_cleaned_BRM_holdout70.csv"
export CABG_ADAPT_DATA="/secure/path/to/BRM_CABG_adaptation_30pct.csv"
export CABG_OUTPUT_DIR="outputs/calibrated_framework_seed1127"
python code/analysis.py
```

On Windows PowerShell, use `$env:CABG_MAIN_DATA = 'C:\secure\...'` and similarly for other variables. An output-free notebook version is also provided at `code/analysis.ipynb`. Running the full pipeline may be computationally intensive due to repeated model fitting and SHAP computation.

Outputs are written under `outputs/`, including tables, figures, joblib models and SHAP summaries. This directory is ignored by Git. Review all generated files before any separate disclosure or sharing.

## Methodological interpretation

- Source intercept calibration is estimated with out-of-fold SNUH training predictions, then applied without refitting to source internal validation and BRM external evaluation.
- Target-specific recalibration uses BRM adaptation data rather than BRM fixed evaluation labels.
- In the fixed-split trajectory, the `alpha` with the highest observed BRM evaluation AUROC is selected as a **descriptive representative model** for subsequent comparisons and SHAP displays. This is **not** an independently validated optimal weight, since the fixed evaluation outcomes influence that selection.
- The repeated 30:70 target splits are intended to assess split robustness. Their 2.5th–97.5th percentile distributions should not be interpreted as conventional 95% confidence intervals.
- SHAP computation attempts interventional TreeExplainer and falls back to tree-path-dependent explanations if the first method fails; inspect runtime warnings before comparing output across environments.

## Disclosure checklist

Do not upload patient-level data, identifiers, fitted model objects, raw prediction arrays, patient-level explanations, notebook outputs, local access credentials or institutional network paths. The `.gitignore` provides a safeguard but cannot remove sensitive material already committed in Git history.

## Citation and license

Add the final article DOI/citation and a code release identifier once available. No license is included at this stage; select an appropriate open-source license after checking institutional intellectual-property requirements. Public source visibility alone does not automatically grant a reuse license.
