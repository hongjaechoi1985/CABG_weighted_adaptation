"""CABG weighted adaptation analysis. Requires restricted-access institutional data.

Source data are NOT included in this repository. See README.md for expected inputs.
"""


# 0. Imports, reproducibility, and configuration
# ============================================================
import os
import json
import random
import platform
import warnings
from pathlib import Path
from datetime import datetime

try:
    from IPython.display import display
except ImportError:
    display = print

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau

from sklearn import __version__ as sklearn_version
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 1127
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Primary data files used in the original analysis
MAIN_DATA_PATH = Path(
    os.environ.get("CABG_MAIN_DATA", "datasets/BRM_subset/Total_TL_CABG_data_cleaned_BRM_holdout70.csv")
)
BRM_ADAPT_PATH = Path(
    os.environ.get("CABG_ADAPT_DATA", "datasets/BRM_subset/BRM_CABG_adaptation_30pct.csv")
)

OUTPUT_DIR = Path(os.environ.get("CABG_OUTPUT_DIR", "outputs/calibrated_framework_seed1127"))
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
SHAP_DIR = OUTPUT_DIR / "shap"
for directory in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, MODEL_DIR, SHAP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Analysis settings
A_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)
PRIMARY_ALPHA = None  # representative alpha is selected descriptively from the fixed-split trajectory
BRM_ADAPTATION_FRACTION = 0.30
N_REPEATED_SPLITS = 100       # Increase to 200 for the final sensitivity analysis if desired
N_BOOTSTRAP = 2000
N_SHAP_REPEATS = 30           # Computationally intensive; increase only after a pilot run
N_FEATURES_PER_DOMAIN = 4
SHAP_MAX_EVALUATION_N = None  # None = use the full evaluation set
RUN_FIXED_SHAP_ALL_ALPHA = True

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 5,
    "class_weight": None,
    "n_jobs": -1,
}

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", None)

print("Base seed:", SEED)
print("Output directory:", OUTPUT_DIR.resolve())



# Revised-framework settings
N_ALPHA_SELECTION_REPEATS = 10
ALPHA_SELECTION_FOLDS = 5  # approximately one event per validation fold
FRACTION_GRID = [0.10, 0.20, 0.30]
N_FRACTION_REPEATS = 50
RF_RANDOM_STATE_FIXED = SEED  # held fixed to isolate BRM split variability


# Development-stage calibration settings
SOURCE_CALIBRATION_FOLDS = 5
SOURCE_CALIBRATION_METHOD = "intercept_only"
CALIBRATION_BINS = 3


# 1. Variable definitions and data loading

feature_cols = [
    "sex",
    "Hypertension",
    "Diabetes_mellitus",
    "Cerebrovascular_disease",
    "Coronary_artery_disease",
    "Atrial_fibrillation",
    "Chronic_kidney_disease",
    "Valvular_heart_disease",
    "COPD",
    "Aortic_aneurysm",
    "Heart_failure",
    "age",
    "BMI",
    "albumin",
    "alp",
    "alt",
    "ast",
    "bt",
    "bun",
    "ck",
    "ckmb",
    "creatinine",
    "crp",
    "fio2",
    "hb",
    "hco3",
    "hct",
    "hr",
    "lacate",
    "nibp_dbp",
    "nibp_mbp",
    "nibp_sbp",
    "paco2",
    "pao2",
    "ph",
    "platelet",
    "ptinr",
    "rr",
    "spo2",
    "total_bilirubin",
    "total_protein",
    "troponin_i",
    "wbc",
]

target_col = "inhosp_death"
hospital_col = "hospital"

binary_cols = [
    "sex",
    "Hypertension",
    "Diabetes_mellitus",
    "Cerebrovascular_disease",
    "Coronary_artery_disease",
    "Atrial_fibrillation",
    "Chronic_kidney_disease",
    "Valvular_heart_disease",
    "COPD",
    "Aortic_aneurysm",
    "Heart_failure",
]
continuous_cols = [column for column in feature_cols if column not in binary_cols]

domain_map = {
    "Patient characteristics": ["age", "sex", "BMI"],
    "Comorbidities": [
        "Hypertension", "Diabetes_mellitus", "Cerebrovascular_disease",
        "Coronary_artery_disease", "Atrial_fibrillation",
        "Chronic_kidney_disease", "Valvular_heart_disease", "COPD",
        "Aortic_aneurysm", "Heart_failure",
    ],
    "Hemodynamics / vital signs": [
        "bt", "hr", "rr", "nibp_sbp", "nibp_dbp", "nibp_mbp", "spo2", "fio2",
    ],
    "Laboratory results": [
        "albumin", "alp", "alt", "ast", "bun", "ck", "ckmb", "creatinine",
        "crp", "hb", "hco3", "hct", "lacate", "paco2", "pao2", "ph",
        "platelet", "ptinr", "total_bilirubin", "total_protein", "troponin_i", "wbc",
    ],
}

# Publication labels used consistently across SHAP summary, forest, and
# dependence plots. Dictionary keys always remain the model-input names.
FEATURE_DISPLAY_NAMES = {
    "sex": "Sex",
    "Hypertension": "Hypertension",
    "Diabetes_mellitus": "Diabetes mellitus",
    "Cerebrovascular_disease": "Cerebrovascular disease",
    "Coronary_artery_disease": "Coronary artery disease",
    "Atrial_fibrillation": "Atrial fibrillation",
    "Chronic_kidney_disease": "Chronic kidney disease",
    "Valvular_heart_disease": "Valvular heart disease",
    "COPD": "COPD",
    "Aortic_aneurysm": "Aortic aneurysm",
    "Heart_failure": "Heart failure",
    "age": "Age",
    "BMI": "Body mass index",
    "albumin": "Albumin",
    "alp": "Alkaline phosphatase",
    "alt": "Alanine transaminase",
    "ast": "Aspartate transaminase",
    "bt": "Body temperature",
    "bun": "Blood urea nitrogen",
    "ck": "Creatine kinase",
    "ckmb": "Creatine kinase-MB",
    "creatinine": "Creatinine",
    "crp": "C-reactive protein",
    "fio2": "Fraction of inspired oxygen",
    "hb": "Hemoglobin",
    "hco3": "Bicarbonate",
    "hct": "Hematocrit",
    "hr": "Heart rate",
    "lacate": "Lactate",
    "nibp_dbp": "Diastolic blood pressure",
    "nibp_mbp": "Mean blood pressure",
    "nibp_sbp": "Systolic blood pressure",
    "paco2": "PaCO2",
    "pao2": "PaO2",
    "ph": "pH",
    "platelet": "Platelet count",
    "ptinr": "Prothrombin time",
    "rr": "Respiratory rate",
    "spo2": "Peripheral oxygen saturation",
    "total_bilirubin": "Total bilirubin",
    "total_protein": "Total protein",
    "troponin_i": "Troponin I",
    "wbc": "White blood cell count",
}

main_df = pd.read_csv(MAIN_DATA_PATH)
brm_fixed_adapt_df = pd.read_csv(BRM_ADAPT_PATH)
main_df.columns = main_df.columns.astype(str).str.strip()
brm_fixed_adapt_df.columns = brm_fixed_adapt_df.columns.astype(str).str.strip()

required_main = feature_cols + [target_col, hospital_col]
required_adapt = feature_cols + [target_col]
missing_main = sorted(set(required_main) - set(main_df.columns))
missing_adapt = sorted(set(required_adapt) - set(brm_fixed_adapt_df.columns))
if missing_main or missing_adapt:
    raise ValueError({"missing_main": missing_main, "missing_adapt": missing_adapt})

snuh_df = main_df.loc[main_df[hospital_col] == 1].copy()
brm_fixed_eval_df = main_df.loc[main_df[hospital_col] == 2].copy()

# The two BRM files are expected to be disjoint partitions of the original BRM cohort.
brm_full_df = pd.concat(
    [brm_fixed_adapt_df[required_adapt], brm_fixed_eval_df[required_adapt]],
    axis=0,
    ignore_index=True,
)

# Optional overlap check when an identifier is available in both files.
candidate_id_cols = ["subjectId", "subject_id", "person_id", "patient_id"]
for id_col in candidate_id_cols:
    if id_col in brm_fixed_adapt_df.columns and id_col in brm_fixed_eval_df.columns:
        overlap = set(brm_fixed_adapt_df[id_col].dropna()) & set(brm_fixed_eval_df[id_col].dropna())
        if overlap:
            raise ValueError(f"BRM adaptation/evaluation overlap detected in {id_col}: {len(overlap)}")
        print(f"No BRM overlap detected using {id_col}.")
        break

print("SNUH full cohort:", snuh_df.shape, "events:", int(snuh_df[target_col].sum()))
print("BRM fixed adaptation:", brm_fixed_adapt_df.shape, "events:", int(brm_fixed_adapt_df[target_col].sum()))
print("BRM fixed evaluation:", brm_fixed_eval_df.shape, "events:", int(brm_fixed_eval_df[target_col].sum()))
print("BRM reconstructed full cohort:", brm_full_df.shape, "events:", int(brm_full_df[target_col].sum()))


# 2. Preprocessing and model helpers


def clean_feature_types(df):
    result = df.copy()
    result["sex"] = (
        result["sex"].astype("string").str.strip().str.upper().map({
            "M": 1, "MALE": 1, "F": 0, "FEMALE": 0,
            "1": 1, "1.0": 1, "0": 0, "0.0": 0,
        })
    )
    for column in binary_cols:
        if column != "sex":
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in continuous_cols:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def fit_preprocessor(X_train_raw):
    X_train = clean_feature_types(X_train_raw[feature_cols])
    all_missing = [column for column in feature_cols if X_train[column].isna().all()]
    if all_missing:
        raise ValueError(f"All-missing predictors in SNUH training data: {all_missing}")

    binary_imputer = SimpleImputer(strategy="most_frequent")
    continuous_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_binary = binary_imputer.fit_transform(X_train[binary_cols])
    X_continuous = continuous_imputer.fit_transform(X_train[continuous_cols])
    scaler.fit(X_continuous)

    return {
        "binary_imputer": binary_imputer,
        "continuous_imputer": continuous_imputer,
        "scaler": scaler,
        "binary_cols": binary_cols,
        "continuous_cols": continuous_cols,
        "feature_cols": feature_cols,
    }


def transform_features(X_raw, preprocessor):
    X = clean_feature_types(X_raw[feature_cols])
    X_binary = pd.DataFrame(
        preprocessor["binary_imputer"].transform(X[binary_cols]),
        columns=binary_cols,
        index=X.index,
    ).astype(int)
    X_continuous_imputed = pd.DataFrame(
        preprocessor["continuous_imputer"].transform(X[continuous_cols]),
        columns=continuous_cols,
        index=X.index,
    )
    X_continuous = pd.DataFrame(
        preprocessor["scaler"].transform(X_continuous_imputed),
        columns=continuous_cols,
        index=X.index,
    )
    return pd.concat([X_binary, X_continuous], axis=1)[feature_cols]


def make_rf(seed=SEED):
    return RandomForestClassifier(random_state=int(seed), **RF_PARAMS)


def make_class_balanced_weight(y):
    y = np.asarray(y, dtype=int)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return np.ones(len(y), dtype=float)
    weights = np.zeros(len(y), dtype=float)
    weights[y == 1] = 0.5 * len(y) / n_pos
    weights[y == 0] = 0.5 * len(y) / n_neg
    return weights


def make_group_contribution_weight(y_source, y_target, alpha, scale_to_total_n=True):
    """Assign total group contributions (1-alpha) and alpha, then split each group 50:50 by outcome."""
    y_source = np.asarray(y_source, dtype=int)
    y_target = np.asarray(y_target, dtype=int)
    n_total = len(y_source) + len(y_target)
    source_total = (1.0 - float(alpha)) * (n_total if scale_to_total_n else 1.0)
    target_total = float(alpha) * (n_total if scale_to_total_n else 1.0)

    def within_group(y, total):
        if total <= 0:
            return np.zeros(len(y), dtype=float)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        if n_pos == 0 or n_neg == 0:
            return np.repeat(total / len(y), len(y))
        w = np.zeros(len(y), dtype=float)
        w[y == 1] = 0.5 * total / n_pos
        w[y == 0] = 0.5 * total / n_neg
        return w

    return np.concatenate([within_group(y_source, source_total), within_group(y_target, target_total)])


def metric_row(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    prevalence = float(np.mean(y_true))
    brier = brier_score_loss(y_true, y_prob)
    null_brier = prevalence * (1.0 - prevalence)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Brier": brier,
        "Null_Brier_eval_prevalence": null_brier,
        "Brier_skill_score": 1.0 - brier / null_brier if null_brier > 0 else np.nan,
    }


def clip_probabilities(probabilities, epsilon=1e-6):
    return np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0 - epsilon)


def fit_recalibrator(y, p, estimate_slope=True):
    """Unpenalized logistic recalibration on logit predictions; returns intercept and slope."""
    y = np.asarray(y, dtype=int)
    x = logit(clip_probabilities(p))
    if len(np.unique(y)) < 2:
        return {"intercept": np.nan, "slope": np.nan, "success": False}

    def negative_log_likelihood(parameters):
        intercept, slope = parameters
        fitted = clip_probabilities(expit(intercept + slope * x))
        return -np.sum(y * np.log(fitted) + (1 - y) * np.log(1 - fitted))

    if estimate_slope:
        result = minimize(
            negative_log_likelihood,
            x0=np.array([0.0, 1.0]),
            method="L-BFGS-B",
            bounds=[(-20, 20), (-10, 10)],
        )
        intercept, slope = result.x
    else:
        def intercept_objective(intercept):
            fitted = clip_probabilities(expit(intercept + x))
            return -np.sum(y * np.log(fitted) + (1 - y) * np.log(1 - fitted))

        result = minimize_scalar(intercept_objective, bounds=(-20, 20), method="bounded")
        intercept, slope = float(result.x), 1.0

    return {"intercept": float(intercept), "slope": float(slope), "success": bool(result.success)}


def apply_recalibrator(p, parameters):
    return expit(
        parameters["intercept"]
        + parameters["slope"] * logit(clip_probabilities(p))
    )


def calibration_statistics(y_true, y_prob):
    full = fit_recalibrator(y_true, y_prob, estimate_slope=True)
    intercept_only = fit_recalibrator(y_true, y_prob, estimate_slope=False)
    return {
        "calibration_intercept": intercept_only["intercept"],
        "calibration_slope": full["slope"],
        "calibration_model_intercept": full["intercept"],
        "calibration_fit_success": full["success"],
    }


def stratified_bootstrap_metric_ci(y_true, y_prob, metric_function, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Stratified percentile bootstrap CI with fixed event and non-event counts."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    positive = np.where(y_true == 1)[0]
    negative = np.where(y_true == 0)[0]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        sampled = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        estimates.append(metric_function(y_true[sampled], y_prob[sampled]))
    estimates = np.asarray(estimates, dtype=float)
    return {
        "point": float(metric_function(y_true, y_prob)),
        "bootstrap_mean": float(np.mean(estimates)),
        "bootstrap_sd": float(np.std(estimates, ddof=1)),
        "ci_lower": float(np.quantile(estimates, 0.025)),
        "ci_upper": float(np.quantile(estimates, 0.975)),
        "n_bootstrap": int(len(estimates)),
    }


def paired_bootstrap_difference_ci(y_true, p_reference, p_candidate, metric_function, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Paired stratified bootstrap for candidate minus reference performance."""
    y_true = np.asarray(y_true, dtype=int)
    p_reference = np.asarray(p_reference, dtype=float)
    p_candidate = np.asarray(p_candidate, dtype=float)
    positive = np.where(y_true == 1)[0]
    negative = np.where(y_true == 0)[0]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        sampled = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        differences.append(
            metric_function(y_true[sampled], p_candidate[sampled])
            - metric_function(y_true[sampled], p_reference[sampled])
        )
    differences = np.asarray(differences, dtype=float)
    point = metric_function(y_true, p_candidate) - metric_function(y_true, p_reference)
    return {
        "point_difference": float(point),
        "ci_lower": float(np.quantile(differences, 0.025)),
        "ci_upper": float(np.quantile(differences, 0.975)),
        "probability_difference_gt_0": float(np.mean(differences > 0)),
    }




def fit_weighted_model(X_source, y_source, X_target, y_target, alpha, seed):
    X_pool = pd.concat(
        [X_source.reset_index(drop=True), X_target.reset_index(drop=True)],
        ignore_index=True,
    )
    y_pool = pd.concat(
        [pd.Series(y_source), pd.Series(y_target)], ignore_index=True
    ).astype(int)
    weights = make_group_contribution_weight(y_source, y_target, alpha)
    keep = weights > 0
    model = make_rf(seed)
    model.fit(X_pool.loc[keep], y_pool.loc[keep], sample_weight=weights[keep])
    return model


# 3. Fixed source and BRM data preparation

# ============================================================
# Source model development and development-stage calibration
# ============================================================
X_snuh = snuh_df[feature_cols].copy()
y_snuh = snuh_df[target_col].astype(int).copy()

X_source_train_raw, X_source_internal_raw, y_source_train, y_source_internal = train_test_split(
    X_snuh,
    y_snuh,
    test_size=0.20,
    random_state=SEED,
    stratify=y_snuh,
)

X_source_train_raw = X_source_train_raw.reset_index(drop=True)
X_source_internal_raw = X_source_internal_raw.reset_index(drop=True)
y_source_train = y_source_train.reset_index(drop=True).astype(int)
y_source_internal = y_source_internal.reset_index(drop=True).astype(int)

# Generate out-of-fold predictions strictly within the SNUH training set.
# Preprocessing is refitted within each fold to avoid preprocessing leakage.
n_source_events = int(y_source_train.sum())
source_calibration_folds = min(SOURCE_CALIBRATION_FOLDS, n_source_events)
if source_calibration_folds < 2:
    raise ValueError("At least two SNUH training events are required for OOF calibration.")

source_cv = StratifiedKFold(
    n_splits=source_calibration_folds,
    shuffle=True,
    random_state=SEED,
)

p_source_train_oof_raw = np.full(len(y_source_train), np.nan, dtype=float)

for fold_id, (fold_train_idx, fold_valid_idx) in enumerate(
    source_cv.split(X_source_train_raw, y_source_train)
):
    X_fold_train_raw = X_source_train_raw.iloc[fold_train_idx].reset_index(drop=True)
    X_fold_valid_raw = X_source_train_raw.iloc[fold_valid_idx].reset_index(drop=True)
    y_fold_train = y_source_train.iloc[fold_train_idx].reset_index(drop=True)

    fold_preprocessor = fit_preprocessor(X_fold_train_raw)
    X_fold_train = transform_features(X_fold_train_raw, fold_preprocessor).reset_index(drop=True)
    X_fold_valid = transform_features(X_fold_valid_raw, fold_preprocessor).reset_index(drop=True)

    fold_model = make_rf(RF_RANDOM_STATE_FIXED)
    fold_model.fit(
        X_fold_train,
        y_fold_train,
        sample_weight=make_class_balanced_weight(y_fold_train),
    )
    p_source_train_oof_raw[fold_valid_idx] = fold_model.predict_proba(X_fold_valid)[:, 1]

if np.isnan(p_source_train_oof_raw).any():
    raise RuntimeError("Some SNUH training observations did not receive OOF predictions.")

# Intercept-only calibration is preferred because the number of deaths is limited
# and outcome-balanced sample weights primarily alter the probability intercept.
source_recalibrator = fit_recalibrator(
    y_source_train,
    p_source_train_oof_raw,
    estimate_slope=False,
)
if not source_recalibrator["success"]:
    raise RuntimeError("Source-model intercept calibration did not converge.")

# Fit the final preprocessing pipeline and RF on the complete SNUH training set.
preprocessor = fit_preprocessor(X_source_train_raw)
X_source_train = transform_features(X_source_train_raw, preprocessor).reset_index(drop=True)
X_source_internal = transform_features(X_source_internal_raw, preprocessor).reset_index(drop=True)

X_brm_fixed_adapt = transform_features(
    brm_fixed_adapt_df[feature_cols], preprocessor
).reset_index(drop=True)
y_brm_fixed_adapt = brm_fixed_adapt_df[target_col].reset_index(drop=True).astype(int)

X_brm_fixed_eval = transform_features(
    brm_fixed_eval_df[feature_cols], preprocessor
).reset_index(drop=True)
y_brm_fixed_eval = brm_fixed_eval_df[target_col].reset_index(drop=True).astype(int)

source_model = make_rf(RF_RANDOM_STATE_FIXED)
source_model.fit(
    X_source_train,
    y_source_train,
    sample_weight=make_class_balanced_weight(y_source_train),
)

# Raw RF scores
p_source_internal_raw = source_model.predict_proba(X_source_internal)[:, 1]
p_source_fixed_adapt_raw = source_model.predict_proba(X_brm_fixed_adapt)[:, 1]
p_source_fixed_eval_raw = source_model.predict_proba(X_brm_fixed_eval)[:, 1]

# Locked development-stage calibration applied without refitting.
p_source_train_oof_calibrated = apply_recalibrator(
    p_source_train_oof_raw, source_recalibrator
)
p_source_internal_calibrated = apply_recalibrator(
    p_source_internal_raw, source_recalibrator
)
p_source_fixed_adapt_calibrated = apply_recalibrator(
    p_source_fixed_adapt_raw, source_recalibrator
)
p_source_fixed_eval_calibrated = apply_recalibrator(
    p_source_fixed_eval_raw, source_recalibrator
)

# Backward-compatible aliases: these now represent the deployable calibrated
# source pipeline rather than uncalibrated RF scores.
p_source_internal = p_source_internal_calibrated
p_source_fixed_adapt = p_source_fixed_adapt_calibrated
p_source_fixed_eval = p_source_fixed_eval_calibrated

joblib.dump(preprocessor, MODEL_DIR / "source_preprocessor_seed1127.joblib")
joblib.dump(source_model, MODEL_DIR / "source_only_class_balanced_RF.joblib")
joblib.dump(source_recalibrator, MODEL_DIR / "source_intercept_calibrator.joblib")

source_calibration_parameter_df = pd.DataFrame([{
    "calibration_method": SOURCE_CALIBRATION_METHOD,
    "n_folds": source_calibration_folds,
    "n_training": len(y_source_train),
    "n_training_events": int(y_source_train.sum()),
    "intercept": source_recalibrator["intercept"],
    "slope": source_recalibrator["slope"],
    "fit_success": source_recalibrator["success"],
}])

source_calibration_performance_df = pd.DataFrame([
    {
        "evaluation": "SNUH training OOF",
        "probability": "Raw",
        **metric_row(y_source_train, p_source_train_oof_raw),
        **calibration_statistics(y_source_train, p_source_train_oof_raw),
    },
    {
        "evaluation": "SNUH training OOF",
        "probability": "Internally calibrated",
        **metric_row(y_source_train, p_source_train_oof_calibrated),
        **calibration_statistics(y_source_train, p_source_train_oof_calibrated),
    },
    {
        "evaluation": "SNUH internal validation",
        "probability": "Raw",
        **metric_row(y_source_internal, p_source_internal_raw),
        **calibration_statistics(y_source_internal, p_source_internal_raw),
    },
    {
        "evaluation": "SNUH internal validation",
        "probability": "Internally calibrated",
        **metric_row(y_source_internal, p_source_internal_calibrated),
        **calibration_statistics(y_source_internal, p_source_internal_calibrated),
    },
    {
        "evaluation": "BRM direct external validation",
        "probability": "Raw",
        **metric_row(y_brm_fixed_eval, p_source_fixed_eval_raw),
        **calibration_statistics(y_brm_fixed_eval, p_source_fixed_eval_raw),
    },
    {
        "evaluation": "BRM direct external validation",
        "probability": "Source-calibrated",
        **metric_row(y_brm_fixed_eval, p_source_fixed_eval_calibrated),
        **calibration_statistics(y_brm_fixed_eval, p_source_fixed_eval_calibrated),
    },
])

source_calibration_parameter_df.to_csv(
    TABLE_DIR / "source_calibration_parameter.csv", index=False
)
source_calibration_performance_df.to_csv(
    TABLE_DIR / "source_calibration_performance.csv", index=False
)

print("SNUH training set:", len(y_source_train), "events:", int(y_source_train.sum()))
print("SNUH internal validation:", len(y_source_internal), "events:", int(y_source_internal.sum()))
print("Fixed BRM adaptation set:", len(y_brm_fixed_adapt), "events:", int(y_brm_fixed_adapt.sum()))
print("Fixed BRM evaluation set:", len(y_brm_fixed_eval), "events:", int(y_brm_fixed_eval.sum()))
print("Locked source calibration intercept:", source_recalibrator["intercept"])
display(source_calibration_parameter_df)
display(source_calibration_performance_df)


# 4. Fixed-split alpha trajectory, representative model, and comparators

# ============================================================
# Fixed BRM 30:70 analysis: complete alpha-performance trajectory
# ============================================================
# The same fixed 70% BRM evaluation set is used to describe every alpha.
# The alpha with the highest observed AUROC is carried forward as the
# representative weighted model for calibration and SHAP analyses. Because
# this evaluation set also informs that choice, the selected alpha is not an
# independently validated optimum and all between-alpha results are exploratory.

internal_validation_df = source_calibration_performance_df.loc[
    source_calibration_performance_df["evaluation"] == "SNUH internal validation"
].copy()
internal_validation_df.to_csv(TABLE_DIR / "SNUH_internal_validation_RF.csv", index=False)

fixed_models = {"SNUH-only": source_model}
fixed_predictions_raw = {
    "SNUH-only": p_source_fixed_eval_raw,
}
fixed_predictions = {
    "SNUH-only": p_source_fixed_eval_calibrated,
    "SNUH-only raw": p_source_fixed_eval_raw,
}
fixed_results = [
    {
        "method": "SNUH-only raw",
        "alpha": 0.0,
        "calibration_source": "None",
        **metric_row(y_brm_fixed_eval, p_source_fixed_eval_raw),
        **calibration_statistics(y_brm_fixed_eval, p_source_fixed_eval_raw),
    },
    {
        "method": "SNUH-only",
        "alpha": 0.0,
        "calibration_source": "SNUH training OOF intercept",
        **metric_row(y_brm_fixed_eval, p_source_fixed_eval_calibrated),
        **calibration_statistics(y_brm_fixed_eval, p_source_fixed_eval_calibrated),
    },
]

for alpha in A_GRID:
    alpha = float(alpha)
    label = f"Weighted alpha={alpha:.1f}"
    model = fit_weighted_model(
        X_source_train, y_source_train,
        X_brm_fixed_adapt, y_brm_fixed_adapt,
        alpha=alpha, seed=RF_RANDOM_STATE_FIXED,
    )
    probability = model.predict_proba(X_brm_fixed_eval)[:, 1]
    fixed_models[label] = model
    fixed_predictions_raw[label] = probability
    fixed_predictions[label] = probability
    fixed_results.append({
        "method": label, "alpha": alpha,
        "calibration_source": "None (raw RF output)",
        **metric_row(y_brm_fixed_eval, probability),
        **calibration_statistics(y_brm_fixed_eval, probability),
    })

fixed_weighted_trajectory_df = pd.DataFrame(
    [row for row in fixed_results if str(row["method"]).startswith("Weighted alpha=")]
).sort_values("alpha")
representative_row = fixed_weighted_trajectory_df.sort_values(
    ["AUROC", "alpha"], ascending=[False, True]
).iloc[0]
PRIMARY_ALPHA = float(representative_row["alpha"])
selected_label = f"Weighted alpha={PRIMARY_ALPHA:.1f}"
selected_model = fixed_models[selected_label]
p_selected_eval_raw = fixed_predictions_raw[selected_label]

X_fixed_pool = pd.concat([X_source_train, X_brm_fixed_adapt], ignore_index=True)
y_fixed_pool = pd.concat([y_source_train, y_brm_fixed_adapt], ignore_index=True)

unweighted_model = make_rf(RF_RANDOM_STATE_FIXED)
unweighted_model.fit(X_fixed_pool, y_fixed_pool)
p_unweighted = unweighted_model.predict_proba(X_brm_fixed_eval)[:, 1]
fixed_models["Unweighted pooling"] = unweighted_model
fixed_predictions["Unweighted pooling"] = p_unweighted
fixed_results.append({
    "method": "Unweighted pooling",
    "alpha": len(y_brm_fixed_adapt) / len(y_fixed_pool),
    **metric_row(y_brm_fixed_eval, p_unweighted),
    **calibration_statistics(y_brm_fixed_eval, p_unweighted),
})

natural_balanced_model = make_rf(RF_RANDOM_STATE_FIXED)
natural_balanced_model.fit(
    X_fixed_pool, y_fixed_pool,
    sample_weight=make_class_balanced_weight(y_fixed_pool),
)
p_natural_balanced = natural_balanced_model.predict_proba(X_brm_fixed_eval)[:, 1]
fixed_models["Natural pooling + global class balance"] = natural_balanced_model
fixed_predictions["Natural pooling + global class balance"] = p_natural_balanced
fixed_results.append({
    "method": "Natural pooling + global class balance",
    "alpha": len(y_brm_fixed_adapt) / len(y_fixed_pool),
    **metric_row(y_brm_fixed_eval, p_natural_balanced),
    **calibration_statistics(y_brm_fixed_eval, p_natural_balanced),
})

brm_only_model = make_rf(RF_RANDOM_STATE_FIXED)
brm_only_model.fit(X_brm_fixed_adapt, y_brm_fixed_adapt)
p_brm_only = brm_only_model.predict_proba(X_brm_fixed_eval)[:, 1]
fixed_models["BRM-only unweighted"] = brm_only_model
fixed_predictions["BRM-only unweighted"] = p_brm_only
fixed_results.append({
    "method": "BRM-only unweighted", "alpha": 1.0,
    **metric_row(y_brm_fixed_eval, p_brm_only),
    **calibration_statistics(y_brm_fixed_eval, p_brm_only),
})

# Recalibration is estimated only from out-of-fold predictions in the fixed
# 30% adaptation set. The fixed 70% evaluation outcomes are not used to fit it.
selection_folds = min(
    ALPHA_SELECTION_FOLDS,
    int(y_brm_fixed_adapt.sum()),
    int((1 - y_brm_fixed_adapt).sum()),
)
if selection_folds < 2:
    raise ValueError("At least two events and two non-events are required for OOF recalibration.")

rskf_recal = RepeatedStratifiedKFold(
    n_splits=selection_folds,
    n_repeats=N_ALPHA_SELECTION_REPEATS,
    random_state=SEED,
)
oof_recalibration_records = []
for fold_number, (target_train_idx, target_valid_idx) in enumerate(
    rskf_recal.split(X_brm_fixed_adapt, y_brm_fixed_adapt)
):
    model = fit_weighted_model(
        X_source_train, y_source_train,
        X_brm_fixed_adapt.iloc[target_train_idx].reset_index(drop=True),
        y_brm_fixed_adapt.iloc[target_train_idx].reset_index(drop=True),
        alpha=PRIMARY_ALPHA, seed=RF_RANDOM_STATE_FIXED,
    )
    probability = model.predict_proba(X_brm_fixed_adapt.iloc[target_valid_idx])[:, 1]
    for row_index, p in zip(target_valid_idx, probability):
        oof_recalibration_records.append({
            "row_index": int(row_index),
            "y": int(y_brm_fixed_adapt.iloc[row_index]),
            "probability": float(p),
        })

selected_oof_patient_df = (
    pd.DataFrame(oof_recalibration_records)
    .groupby("row_index", as_index=False)
    .agg(y=("y", "first"), probability=("probability", "mean"))
    .sort_values("row_index")
)
recal_intercept = fit_recalibrator(
    selected_oof_patient_df["y"], selected_oof_patient_df["probability"], estimate_slope=False
)
recal_full = fit_recalibrator(
    selected_oof_patient_df["y"], selected_oof_patient_df["probability"], estimate_slope=True
)

for label, parameters in {
    f"{selected_label} + intercept recalibration": recal_intercept,
    f"{selected_label} + intercept/slope recalibration": recal_full,
}.items():
    p_eval = apply_recalibrator(p_selected_eval_raw, parameters)
    fixed_predictions[label] = p_eval
    fixed_results.append({
        "method": label, "alpha": PRIMARY_ALPHA,
        "calibration_source": "BRM adaptation OOF",
        "adaptation_recalibration_intercept": parameters["intercept"],
        "adaptation_recalibration_slope": parameters["slope"],
        **metric_row(y_brm_fixed_eval, p_eval),
        **calibration_statistics(y_brm_fixed_eval, p_eval),
    })

fixed_results_df = pd.DataFrame(fixed_results)
fixed_results_df.to_csv(TABLE_DIR / "fixed_split_all_methods_performance_calibration.csv", index=False)
fixed_weighted_trajectory_df.to_csv(TABLE_DIR / "fixed_split_alpha_trajectory.csv", index=False)
joblib.dump(selected_model, MODEL_DIR / f"representative_weighted_RF_alpha_{PRIMARY_ALPHA:.1f}.joblib")

print(f"Representative alpha by highest observed fixed-split AUROC: {PRIMARY_ALPHA:.1f}")
print("This alpha is descriptive/exploratory, not an independently validated optimum.")
display(internal_validation_df)
display(fixed_weighted_trajectory_df)
display(fixed_results_df.sort_values("AUROC", ascending=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, metric in zip(axes, ["AUROC", "AUPRC", "Brier"]):
    ax.plot(
        fixed_weighted_trajectory_df["alpha"] * 100,
        fixed_weighted_trajectory_df[metric], marker="o", linewidth=2,
    )
    ax.axvline(PRIMARY_ALPHA * 100, color="tab:blue", linestyle="--", alpha=0.7)
    ax.set_xlabel("BRM adaptation weight (%)")
    ax.set_ylabel(metric)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fixed_split_alpha_performance_trajectory.png", dpi=300, bbox_inches="tight")
plt.show()


## Bootstrap confidence intervals and calibration

# ============================================================
# Bootstrap CIs and calibration on the fixed evaluation set
# ============================================================
primary_methods = [
    "SNUH-only",
    "Unweighted pooling",
    selected_label,
    f"{selected_label} + intercept recalibration",
]

bootstrap_rows = []
for method_index, method in enumerate(primary_methods):
    probabilities = fixed_predictions[method]
    auc_ci = stratified_bootstrap_metric_ci(
        y_brm_fixed_eval,
        probabilities,
        roc_auc_score,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED + method_index * 10 + 1,
    )
    auprc_ci = stratified_bootstrap_metric_ci(
        y_brm_fixed_eval,
        probabilities,
        average_precision_score,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED + method_index * 10 + 2,
    )
    bootstrap_rows.append({
        "method": method,
        "AUROC": auc_ci["point"],
        "AUROC_bootstrap_mean": auc_ci["bootstrap_mean"],
        "AUROC_bootstrap_SD": auc_ci["bootstrap_sd"],
        "AUROC_CI_lower": auc_ci["ci_lower"],
        "AUROC_CI_upper": auc_ci["ci_upper"],
        "AUPRC": auprc_ci["point"],
        "AUPRC_bootstrap_mean": auprc_ci["bootstrap_mean"],
        "AUPRC_bootstrap_SD": auprc_ci["bootstrap_sd"],
        "AUPRC_CI_lower": auprc_ci["ci_lower"],
        "AUPRC_CI_upper": auprc_ci["ci_upper"],
        "Brier": brier_score_loss(y_brm_fixed_eval, probabilities),
    })

bootstrap_fixed_df = pd.DataFrame(bootstrap_rows)
bootstrap_fixed_df.to_csv(TABLE_DIR / "fixed_evaluation_bootstrap_CI.csv", index=False)
display(bootstrap_fixed_df)

# Calibration plot: development-stage calibration, transport, adaptation,
# and target-specific intercept recalibration.
calibration_series = [
    {
        "label": "SNUH internal validation",
        "y": y_source_internal,
        "p": p_source_internal_calibrated,
        "color": "#7F7F7F",
        "marker": "o",
        "linestyle": "--",
    },
    {
        "label": "Direct external validation",
        "y": y_brm_fixed_eval,
        "p": p_source_fixed_eval_calibrated,
        "color": "#4C78A8",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "label": f"{int(PRIMARY_ALPHA * 100)}% BRM-weighted",
        "y": y_brm_fixed_eval,
        "p": p_selected_eval_raw,
        "color": "#F28E2B",
        "marker": "s",
        "linestyle": "-",
    },
    {
        "label": f"{int(PRIMARY_ALPHA * 100)}% BRM-weighted + recalibration",
        "y": y_brm_fixed_eval,
        "p": fixed_predictions[f"{selected_label} + intercept recalibration"],
        "color": "#B07AA1",
        "marker": "^",
        "linestyle": "-",
    },
]

calibration_points = []
for series in calibration_series:
    observed, predicted = calibration_curve(
        series["y"], series["p"], n_bins=CALIBRATION_BINS, strategy="quantile"
    )
    for bin_index, (x_value, y_value) in enumerate(zip(predicted, observed)):
        calibration_points.append({
            "model": series["label"],
            "bin": bin_index + 1,
            "mean_predicted_probability": float(x_value),
            "observed_event_proportion": float(y_value),
        })

calibration_points_df = pd.DataFrame(calibration_points)
calibration_points_df.to_csv(
    TABLE_DIR / "calibration_plot_points.csv", index=False
)

CALIBRATION_MAX = 0.40
fig, ax = plt.subplots(figsize=(6.2, 5.0))
for series in calibration_series:
    points = calibration_points_df.loc[
        calibration_points_df["model"] == series["label"]
    ]
    ax.plot(
        points["mean_predicted_probability"],
        points["observed_event_proportion"],
        color=series["color"],
        marker=series["marker"],
        linestyle=series["linestyle"],
        linewidth=1.8,
        markersize=5,
        label=series["label"],
    )

ax.plot(
    [0, CALIBRATION_MAX], [0, CALIBRATION_MAX],
    color="#333333", linestyle=":", linewidth=1.2,
    label="_nolegend_",
)
ax.set_xlim(0, CALIBRATION_MAX)
ax.set_ylim(0, CALIBRATION_MAX)
ax.set_aspect("equal", adjustable="box")
ticks = np.arange(0, CALIBRATION_MAX + 0.001, 0.10)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed event proportion")
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "Figure_2B_calibration_corrected.png", dpi=600, bbox_inches="tight")
plt.savefig(FIGURE_DIR / "Figure_2B_calibration_corrected.pdf", bbox_inches="tight")
plt.show()


# 5. Adaptation-fraction sensitivity without altering fixed evaluation membership


# ============================================================
# 10%, 20%, and 30% adaptation-fraction sensitivity analysis
# ============================================================
# Only the original fixed 30% adaptation set is sampled. The original 70%
# evaluation set is never moved into adaptation. Results are descriptive because
# the same fixed evaluation set is used across the fraction/alpha grid.

def nested_stratified_subsets(y, fractions_within_pool, seed):
    y = pd.Series(y).reset_index(drop=True).astype(int)
    rng = np.random.default_rng(seed)
    ordered = {}
    for cls in [0, 1]:
        idx = np.flatnonzero(y.to_numpy() == cls)
        ordered[cls] = rng.permutation(idx)
    subsets = {}
    for frac in sorted(fractions_within_pool):
        chosen = []
        for cls in [0, 1]:
            n_take = max(1, int(round(len(ordered[cls]) * frac)))
            chosen.extend(ordered[cls][:n_take].tolist())
        subsets[frac] = np.array(sorted(chosen), dtype=int)
    return subsets

# Fractions relative to the fixed 30% adaptation pool: 1/3, 2/3, and all.
within_pool_fraction = {0.10: 1/3, 0.20: 2/3, 0.30: 1.0}
fraction_rows = []

for repeat_id in range(N_FRACTION_REPEATS):
    subsets = nested_stratified_subsets(
        y_brm_fixed_adapt,
        [within_pool_fraction[x] for x in FRACTION_GRID],
        seed=SEED + 5000 + repeat_id,
    )
    for total_fraction in FRACTION_GRID:
        pool_fraction = within_pool_fraction[total_fraction]
        indices = subsets[pool_fraction]
        X_adapt_sub = X_brm_fixed_adapt.iloc[indices].reset_index(drop=True)
        y_adapt_sub = y_brm_fixed_adapt.iloc[indices].reset_index(drop=True)
        for alpha in A_GRID:
            model = fit_weighted_model(
                X_source_train, y_source_train,
                X_adapt_sub, y_adapt_sub,
                alpha=float(alpha), seed=RF_RANDOM_STATE_FIXED,
            )
            p_eval = model.predict_proba(X_brm_fixed_eval)[:, 1]
            fraction_rows.append({
                "repeat": repeat_id,
                "adaptation_fraction_total_BRM": total_fraction,
                "adaptation_n": len(y_adapt_sub),
                "adaptation_events": int(y_adapt_sub.sum()),
                "alpha": float(alpha),
                **metric_row(y_brm_fixed_eval, p_eval),
            })

fraction_results_df = pd.DataFrame(fraction_rows)
fraction_summary_df = (
    fraction_results_df.groupby(["adaptation_fraction_total_BRM", "alpha"], as_index=False)
    .agg(
        n_repeats=("repeat", "nunique"),
        adaptation_n_mean=("adaptation_n", "mean"),
        adaptation_events_mean=("adaptation_events", "mean"),
        AUROC_mean=("AUROC", "mean"), AUROC_SD=("AUROC", "std"),
        AUROC_q025=("AUROC", lambda x: x.quantile(0.025)),
        AUROC_q975=("AUROC", lambda x: x.quantile(0.975)),
        AUPRC_mean=("AUPRC", "mean"), AUPRC_SD=("AUPRC", "std"),
        Brier_mean=("Brier", "mean"), Brier_SD=("Brier", "std"),
    )
)
fraction_results_df.to_csv(TABLE_DIR / "adaptation_fraction_10_20_30_all_results.csv", index=False)
fraction_summary_df.to_csv(TABLE_DIR / "adaptation_fraction_10_20_30_summary.csv", index=False)
display(fraction_summary_df)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for total_fraction, group in fraction_summary_df.groupby("adaptation_fraction_total_BRM"):
    label = f"{int(total_fraction * 100)}% adaptation"
    axes[0].plot(group["alpha"] * 100, group["AUROC_mean"], marker="o", label=label)
    axes[1].plot(group["alpha"] * 100, group["AUPRC_mean"], marker="o", label=label)
    axes[2].plot(group["alpha"] * 100, group["Brier_mean"], marker="o", label=label)
for ax, ylabel in zip(axes, ["Mean AUROC", "Mean AUPRC", "Mean Brier score"]):
    ax.set_xlabel("BRM adaptation weight (%)")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "adaptation_fraction_10_20_30_performance.png", dpi=300, bbox_inches="tight")
plt.show()

# 6. Repeated stratified 30:70 BRM splits

# This analysis intentionally changes the BRM adaptation/evaluation membership using seeds `1127 + repeat_id`. It therefore does **not** retain one global fixed evaluation set. Within each repetition, however, the 70% evaluation subset is held out from training and is shared by every method and alpha. These results quantify split variability and are reported using mean, SD, and the empirical 2.5th–97.5th percentile range—not as a formal 95% confidence interval.


X_brm_full_raw = brm_full_df[feature_cols].copy()
y_brm_full = brm_full_df[target_col].astype(int).copy()
X_brm_full = transform_features(X_brm_full_raw, preprocessor)


def fit_weighted_model(X_source, y_source, X_target, y_target, alpha, seed):
    X_pool = pd.concat([X_source.reset_index(drop=True), X_target.reset_index(drop=True)], ignore_index=True)
    y_pool = pd.concat([pd.Series(y_source), pd.Series(y_target)], ignore_index=True).astype(int)
    weights = make_group_contribution_weight(y_source, y_target, alpha)
    keep = weights > 0
    model = make_rf(seed)
    model.fit(X_pool.loc[keep], y_pool.loc[keep], sample_weight=weights[keep])
    return model


def append_result(rows, repeat_id, split_seed, method, alpha, y_eval, p_eval, n_adapt, events_adapt, n_eval, events_eval):
    rows.append({
        "repeat": repeat_id,
        "split_seed": split_seed,
        "method": method,
        "alpha": alpha,
        "adaptation_n": n_adapt,
        "adaptation_events": events_adapt,
        "evaluation_n": n_eval,
        "evaluation_events": events_eval,
        **metric_row(y_eval, p_eval),
        **calibration_statistics(y_eval, p_eval),
    })


repeated_rows = []
for repeat_id in range(N_REPEATED_SPLITS):
    split_seed = SEED + repeat_id
    adapt_indices, eval_indices = train_test_split(
        np.arange(len(y_brm_full)),
        train_size=BRM_ADAPTATION_FRACTION,
        random_state=split_seed,
        stratify=y_brm_full,
    )
    X_adapt = X_brm_full.iloc[adapt_indices].reset_index(drop=True)
    y_adapt = y_brm_full.iloc[adapt_indices].reset_index(drop=True)
    X_eval = X_brm_full.iloc[eval_indices].reset_index(drop=True)
    y_eval = y_brm_full.iloc[eval_indices].reset_index(drop=True)

    model_seed = RF_RANDOM_STATE_FIXED  # fixed to isolate BRM split variability
    n_adapt, events_adapt = len(y_adapt), int(y_adapt.sum())
    n_eval, events_eval = len(y_eval), int(y_eval.sum())

    # Source-only reference. RF seed is held fixed; only BRM split composition varies.
    source_repeat_model = make_rf(model_seed)
    source_repeat_model.fit(
        X_source_train,
        y_source_train,
        sample_weight=make_class_balanced_weight(y_source_train),
    )
    p_source_adapt_raw = source_repeat_model.predict_proba(X_adapt)[:, 1]
    p_source_eval_raw = source_repeat_model.predict_proba(X_eval)[:, 1]
    p_source_adapt = apply_recalibrator(p_source_adapt_raw, source_recalibrator)
    p_source_eval = apply_recalibrator(p_source_eval_raw, source_recalibrator)
    append_result(
        repeated_rows, repeat_id, split_seed, "SNUH-only", 0.0,
        y_eval, p_source_eval, n_adapt, events_adapt, n_eval, events_eval,
    )

    # Proposed weighted alpha grid
    for alpha in A_GRID:
        model = fit_weighted_model(
            X_source_train, y_source_train, X_adapt, y_adapt,
            alpha=alpha, seed=model_seed,
        )
        p_eval = model.predict_proba(X_eval)[:, 1]
        append_result(
            repeated_rows, repeat_id, split_seed, "Weighted adaptation", alpha,
            y_eval, p_eval, n_adapt, events_adapt, n_eval, events_eval,
        )

    # True unweighted pooling
    X_pool = pd.concat([X_source_train, X_adapt], ignore_index=True)
    y_pool = pd.concat([y_source_train, y_adapt], ignore_index=True)
    unweighted = make_rf(model_seed)
    unweighted.fit(X_pool, y_pool)
    append_result(
        repeated_rows, repeat_id, split_seed, "Unweighted pooling", np.nan,
        y_eval, unweighted.predict_proba(X_eval)[:, 1],
        n_adapt, events_adapt, n_eval, events_eval,
    )

    # Natural pooling with global class balance, reported separately from true unweighted pooling
    natural_balanced = make_rf(model_seed)
    natural_balanced.fit(X_pool, y_pool, sample_weight=make_class_balanced_weight(y_pool))
    append_result(
        repeated_rows, repeat_id, split_seed, "Natural pooling + global class balance", np.nan,
        y_eval, natural_balanced.predict_proba(X_eval)[:, 1],
        n_adapt, events_adapt, n_eval, events_eval,
    )

    # Target-only unweighted
    brm_only = make_rf(model_seed)
    brm_only.fit(X_adapt, y_adapt)
    append_result(
        repeated_rows, repeat_id, split_seed, "BRM-only unweighted", 1.0,
        y_eval, brm_only.predict_proba(X_eval)[:, 1],
        n_adapt, events_adapt, n_eval, events_eval,
    )

    # Recalibration comparators
    for label, estimate_slope in [
        ("SNUH-only + intercept recalibration", False),
        ("SNUH-only + intercept/slope recalibration", True),
    ]:
        parameters = fit_recalibrator(y_adapt, p_source_adapt, estimate_slope=estimate_slope)
        p_eval = apply_recalibrator(p_source_eval, parameters)
        append_result(
            repeated_rows, repeat_id, split_seed, label, np.nan,
            y_eval, p_eval, n_adapt, events_adapt, n_eval, events_eval,
        )

    if (repeat_id + 1) % 10 == 0:
        print(f"Completed repeated split {repeat_id + 1}/{N_REPEATED_SPLITS}")

repeated_results_df = pd.DataFrame(repeated_rows)
repeated_results_df.to_csv(TABLE_DIR / "repeated_BRM_split_all_results.csv", index=False, encoding="utf-8-sig")
print("Repeated result rows:", len(repeated_results_df))

## Repeated-split summaries

def summarize_distribution(data, group_columns, metrics=("AUROC", "AUPRC", "Brier", "Brier_skill_score")):
    records = []
    for keys, group in data.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            records.append({
                **base,
                "metric": metric,
                "n": len(values),
                "mean": np.mean(values),
                "SD": np.std(values, ddof=1),
                "median": np.median(values),
                "q025": np.quantile(values, 0.025),
                "q975": np.quantile(values, 0.975),
            })
    return pd.DataFrame(records)


weighted_repeated = repeated_results_df.loc[
    repeated_results_df["method"] == "Weighted adaptation"
].copy()
alpha_summary_df = summarize_distribution(weighted_repeated, ["alpha"])
alpha_summary_df.to_csv(TABLE_DIR / "repeated_split_alpha_performance_summary.csv", index=False, encoding="utf-8-sig")

primary_comparison_df = pd.concat([
    repeated_results_df.loc[
        repeated_results_df["method"].isin([
            "SNUH-only",
            "Unweighted pooling",
            "Natural pooling + global class balance",
            "BRM-only unweighted",
            "SNUH-only + intercept recalibration",
            "SNUH-only + intercept/slope recalibration",
        ])
    ],
    repeated_results_df.loc[
        (repeated_results_df["method"] == "Weighted adaptation")
        & np.isclose(repeated_results_df["alpha"], PRIMARY_ALPHA)
    ].assign(method=f"Weighted adaptation alpha={PRIMARY_ALPHA:.1f}"),
], ignore_index=True)

primary_summary_df = summarize_distribution(primary_comparison_df, ["method"])
primary_summary_df.to_csv(TABLE_DIR / "repeated_split_primary_methods_summary.csv", index=False, encoding="utf-8-sig")

# Paired differences by repeat: representative fixed-split alpha versus comparator methods
performance_wide = primary_comparison_df.pivot_table(
    index="repeat", columns="method", values=["AUROC", "AUPRC", "Brier"], aggfunc="first"
)
paired_rows = []
candidate = f"Weighted adaptation alpha={PRIMARY_ALPHA:.1f}"
for reference in ["SNUH-only", "Unweighted pooling", "Natural pooling + global class balance"]:
    for metric in ["AUROC", "AUPRC", "Brier"]:
        difference = performance_wide[(metric, candidate)] - performance_wide[(metric, reference)]
        paired_rows.append({
            "candidate": candidate,
            "reference": reference,
            "metric": metric,
            "n_repeats": difference.notna().sum(),
            "mean_difference": difference.mean(),
            "SD_difference": difference.std(ddof=1),
            "median_difference": difference.median(),
            "q025": difference.quantile(0.025),
            "q975": difference.quantile(0.975),
            "proportion_candidate_greater": float((difference > 0).mean()) if metric != "Brier" else float((difference < 0).mean()),
        })

paired_repeated_df = pd.DataFrame(paired_rows)
paired_repeated_df.to_csv(TABLE_DIR / "repeated_split_paired_differences.csv", index=False, encoding="utf-8-sig")

# Exploratory best-alpha distribution
best_alpha_df = (
    weighted_repeated.sort_values(["repeat", "AUROC"], ascending=[True, False])
    .groupby("repeat", as_index=False)
    .first()[["repeat", "alpha", "AUROC", "AUPRC", "Brier"]]
    .rename(columns={"alpha": "best_alpha_by_AUROC"})
)
best_alpha_counts = (
    best_alpha_df["best_alpha_by_AUROC"].value_counts().sort_index()
    .rename_axis("best_alpha_by_AUROC").reset_index(name="count")
)
best_alpha_counts["proportion"] = best_alpha_counts["count"] / N_REPEATED_SPLITS
best_alpha_df.to_csv(TABLE_DIR / "repeated_split_best_alpha_each_repeat_exploratory.csv", index=False, encoding="utf-8-sig")
best_alpha_counts.to_csv(TABLE_DIR / "repeated_split_best_alpha_distribution_exploratory.csv", index=False, encoding="utf-8-sig")

# ============================================================
# Display repeated-split results for all alpha values: 0.1–1.0
# ============================================================

alpha_display_df = (
    alpha_summary_df.loc[
        alpha_summary_df["alpha"].between(0.1, 1.0)
    ]
    .pivot(
        index="alpha",
        columns="metric",
        values=[
            "n",
            "mean",
            "SD",
            "median",
            "q025",
            "q975",
        ],
    )
)

# Flatten multi-level column names
alpha_display_df.columns = [
    f"{metric}_{stat}"
    for stat, metric in alpha_display_df.columns
]

# Arrange columns by performance metric
ordered_columns = []

for metric in [
    "AUROC",
    "AUPRC",
    "Brier",
    "Brier_skill_score",
]:
    for stat in [
        "n",
        "mean",
        "SD",
        "median",
        "q025",
        "q975",
    ]:
        column = f"{metric}_{stat}"
        if column in alpha_display_df.columns:
            ordered_columns.append(column)

alpha_display_df = (
    alpha_display_df[ordered_columns]
    .reset_index()
    .sort_values("alpha")
    .reset_index(drop=True)
)

# Save the complete alpha table
alpha_display_df.to_csv(
    TABLE_DIR /
    "repeated_split_alpha_0.1_to_0.9_performance_table.csv",
    index=False,
    encoding="utf-8-sig",
)

display(alpha_display_df)

# Other summary tables
display(primary_summary_df)
display(paired_repeated_df)
display(best_alpha_counts)


## Repeated-split performance figures
def get_metric_summary(summary_df, metric):
    return summary_df.loc[summary_df["metric"] == metric].sort_values("alpha")


fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, metric, y_label in zip(
    axes,
    ["AUROC", "AUPRC", "Brier"],
    ["AUROC", "AUPRC", "Brier score"],
):
    plot_data = get_metric_summary(alpha_summary_df, metric)
    ax.plot(plot_data["alpha"] * 100, plot_data["mean"], marker="o", linewidth=2)
    ax.fill_between(
        plot_data["alpha"] * 100,
        plot_data["q025"],
        plot_data["q975"],
        alpha=0.20,
    )
    ax.axvline(PRIMARY_ALPHA * 100, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("BRM adaptation weight (%)")
    ax.set_ylabel(y_label)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Performance across repeated stratified BRM splits")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "repeated_split_alpha_performance.png", dpi=300, bbox_inches="tight")
plt.savefig(FIGURE_DIR / "repeated_split_alpha_performance.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8.5, 5.2))
box_order = [
    "SNUH-only",
    "Unweighted pooling",
    "Natural pooling + global class balance",
    "SNUH-only + intercept recalibration",
    f"Weighted adaptation alpha={PRIMARY_ALPHA:.1f}",
    "BRM-only unweighted",
]
box_data = [
    primary_comparison_df.loc[primary_comparison_df["method"] == method, "AUROC"].dropna()
    for method in box_order
]
ax.boxplot(box_data, labels=box_order, showfliers=False)
ax.set_ylabel("AUROC")
ax.set_title("AUROC distribution across repeated BRM splits")
ax.tick_params(axis="x", rotation=35)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "repeated_split_primary_methods_AUROC_boxplot.png", dpi=300, bbox_inches="tight")
plt.savefig(FIGURE_DIR / "repeated_split_primary_methods_AUROC_boxplot.pdf", bbox_inches="tight")
plt.show()

try:
    import shap
except ImportError as error:
    raise ImportError(
        "The SHAP section requires the `shap` package. "
        "Install it in the notebook environment before running this section."
    ) from error


# ============================================================
# 1. Extract positive-class SHAP values
# ============================================================

def extract_positive_class_shap(shap_output):

    values = (
        shap_output.values
        if hasattr(shap_output, "values")
        else shap_output
    )

    if isinstance(values, list):
        values = (
            values[1]
            if len(values) == 2
            else values[0]
        )

    values = np.asarray(values)

    if values.ndim == 3:

        if values.shape[-1] == 2:
            values = values[:, :, 1]

        elif values.shape[0] == 2:
            values = values[1, :, :]

        else:
            raise ValueError(
                f"Unexpected 3D SHAP array shape: "
                f"{values.shape}"
            )

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP array shape: "
            f"{values.shape}"
        )

    return values


# ============================================================
# 2. Corrected normalized SHAP importance
# Interventional TreeExplainer
# ============================================================

def compute_normalized_shap_importance(
    model,
    X_eval,
    X_background=None,
    max_background=100,
    seed=1127,
):

    X_eval = (
        X_eval
        .copy()
        .astype(float)
        .reset_index(drop=True)
    )

    if X_background is None:
        X_background = X_eval.copy()
    else:
        X_background = (
            X_background
            .copy()
            .astype(float)
            .reset_index(drop=True)
        )

    # Use a deterministic background sample
    if len(X_background) > max_background:

        X_bg = X_background.sample(
            n=max_background,
            random_state=seed,
        )

    else:
        X_bg = X_background.copy()

    try:

        explainer = shap.TreeExplainer(
            model,
            data=X_bg,
            feature_perturbation="interventional",
            model_output="raw",
        )

        try:
            shap_output = explainer.shap_values(
                X_eval,
                check_additivity=False,
            )

        except TypeError:
            shap_output = explainer.shap_values(
                X_eval
            )

    except Exception as error:

        print(
            "[WARNING] Interventional TreeExplainer "
            f"failed: {error}"
        )

        print(
            "Fallback to tree_path_dependent."
        )

        explainer = shap.TreeExplainer(
            model,
            feature_perturbation=(
                "tree_path_dependent"
            ),
            model_output="raw",
        )

        try:
            shap_output = explainer.shap_values(
                X_eval,
                check_additivity=False,
            )

        except TypeError:
            shap_output = explainer.shap_values(
                X_eval
            )

    values = extract_positive_class_shap(
        shap_output
    )

    mean_absolute = np.mean(
        np.abs(values),
        axis=0,
    )

    total_importance = mean_absolute.sum()

    if total_importance > 0:
        normalized = (
            mean_absolute /
            total_importance
        )
    else:
        normalized = np.zeros_like(
            mean_absolute
        )

    normalized_importance = pd.Series(
        normalized,
        index=X_eval.columns,
        name="normalized_importance",
    )

    return normalized_importance, values


# ============================================================
# 3. Feature-to-domain mapping
# ============================================================

feature_to_domain = {
    feature: domain
    for domain, features in domain_map.items()
    for feature in features
}

unmapped_features = sorted(
    set(feature_cols) -
    set(feature_to_domain)
)

if unmapped_features:
    raise ValueError(
        "Features missing from domain_map: "
        f"{unmapped_features}"
    )


# ============================================================
# 4. Repeated-split SHAP analysis
# ============================================================

shap_feature_rows = []
shap_domain_rows = []
explanation_drift_rows = []


for repeat_id in range(N_SHAP_REPEATS):

    split_seed = SEED + repeat_id

    adapt_indices, eval_indices = (
        train_test_split(
            np.arange(len(y_brm_full)),
            train_size=(
                BRM_ADAPTATION_FRACTION
            ),
            random_state=split_seed,
            stratify=y_brm_full,
        )
    )

    X_adapt = (
        X_brm_full
        .iloc[adapt_indices]
        .reset_index(drop=True)
    )

    y_adapt = (
        y_brm_full
        .iloc[adapt_indices]
        .reset_index(drop=True)
    )

    X_eval_full = (
        X_brm_full
        .iloc[eval_indices]
        .reset_index(drop=True)
    )

    # Keep the full evaluation set available
    # for construction of the background sample.
    X_background = X_eval_full.copy()

    X_eval = X_eval_full.copy()

    # Optional restriction of the SHAP evaluation sample
    if (
        SHAP_MAX_EVALUATION_N is not None
        and len(X_eval) >
        SHAP_MAX_EVALUATION_N
    ):

        evaluation_rng = (
            np.random.default_rng(
                SEED + 50000 + repeat_id
            )
        )

        selected = evaluation_rng.choice(
            len(X_eval),
            size=SHAP_MAX_EVALUATION_N,
            replace=False,
        )

        X_eval = (
            X_eval
            .iloc[selected]
            .reset_index(drop=True)
        )


    # Fixed RF seed isolates variation
    # attributable to the BRM data split.
    model_seed = RF_RANDOM_STATE_FIXED


    # --------------------------------------------------------
    # SNUH-only model
    # --------------------------------------------------------

    source_repeat = make_rf(
        model_seed
    )

    source_repeat.fit(
        X_source_train,
        y_source_train,
        sample_weight=(
            make_class_balanced_weight(
                y_source_train
            )
        ),
    )


    # --------------------------------------------------------
    # Weighted model
    # --------------------------------------------------------

    weighted_repeat = (
        fit_weighted_model(
            X_source_train,
            y_source_train,
            X_adapt,
            y_adapt,
            alpha=PRIMARY_ALPHA,
            seed=model_seed,
        )
    )


    # --------------------------------------------------------
    # SHAP calculation
    #
    # Both models use:
    # - the same X_eval
    # - the same background population
    # - the same background-sampling seed
    # --------------------------------------------------------

    shap_seed = SEED + repeat_id

    importance_source, _ = (
        compute_normalized_shap_importance(
            source_repeat,
            X_eval,
            X_background=X_background,
            max_background=100,
            seed=shap_seed,
        )
    )

    importance_weighted, _ = (
        compute_normalized_shap_importance(
            weighted_repeat,
            X_eval,
            X_background=X_background,
            max_background=100,
            seed=shap_seed,
        )
    )


    # --------------------------------------------------------
    # Explanation-drift metrics
    # --------------------------------------------------------

    source_rank = (
        importance_source.rank(
            ascending=False,
            method="average",
        )
    )

    weighted_rank = (
        importance_weighted.rank(
            ascending=False,
            method="average",
        )
    )

    tau = kendalltau(
        source_rank,
        weighted_rank,
        nan_policy="omit",
    ).statistic

    js_distance = jensenshannon(
        (
            importance_source.to_numpy()
            + 1e-12
        ),
        (
            importance_weighted.to_numpy()
            + 1e-12
        ),
        base=2,
    )

    explanation_drift_rows.append({
        "repeat": repeat_id,
        "split_seed": split_seed,
        "kendall_tau": tau,
        "jensen_shannon_distance":
            float(js_distance),
        "jensen_shannon_divergence":
            float(js_distance ** 2),
    })


    # --------------------------------------------------------
    # Feature-level contribution
    # --------------------------------------------------------

    for feature in feature_cols:

        source_value = float(
            importance_source[feature]
        )

        weighted_value = float(
            importance_weighted[feature]
        )

        shap_feature_rows.append({
            "repeat": repeat_id,
            "split_seed": split_seed,
            "feature": feature,
            "domain":
                feature_to_domain[feature],
            "source_contribution_percent":
                source_value * 100,
            "weighted_contribution_percent":
                weighted_value * 100,
            "change_percentage_points":
                (
                    weighted_value -
                    source_value
                ) * 100,
        })


    # --------------------------------------------------------
    # Domain-level contribution
    # --------------------------------------------------------

    for domain, features in domain_map.items():

        available_features = [
            feature
            for feature in features
            if feature in
            importance_source.index
        ]

        if not available_features:
            raise ValueError(
                "No available predictors for "
                f"domain: {domain}"
            )

        source_value = float(
            importance_source
            .loc[available_features]
            .sum()
        )

        weighted_value = float(
            importance_weighted
            .loc[available_features]
            .sum()
        )

        shap_domain_rows.append({
            "repeat": repeat_id,
            "split_seed": split_seed,
            "domain": domain,
            "source_contribution_percent":
                source_value * 100,
            "weighted_contribution_percent":
                weighted_value * 100,
            "change_percentage_points":
                (
                    weighted_value -
                    source_value
                ) * 100,
        })


    if (repeat_id + 1) % 5 == 0:
        print(
            "Completed SHAP repeat "
            f"{repeat_id + 1}/"
            f"{N_SHAP_REPEATS}"
        )


# ============================================================
# 5. Save repeated-split results
# ============================================================

shap_feature_repeats_df = pd.DataFrame(
    shap_feature_rows
)

shap_domain_repeats_df = pd.DataFrame(
    shap_domain_rows
)

explanation_drift_df = pd.DataFrame(
    explanation_drift_rows
)


shap_feature_repeats_df.to_csv(
    SHAP_DIR /
    "SHAP_feature_contribution_repeated_splits.csv",
    index=False,
    encoding="utf-8-sig",
)

shap_domain_repeats_df.to_csv(
    SHAP_DIR /
    "SHAP_domain_contribution_repeated_splits.csv",
    index=False,
    encoding="utf-8-sig",
)

explanation_drift_df.to_csv(
    SHAP_DIR /
    "explanation_drift_metrics_repeated_splits.csv",
    index=False,
    encoding="utf-8-sig",
)


print(
    "Repeated-split SHAP analysis completed "
    "using the interventional TreeExplainer."
)


## SHAP stability summaries and top-predictor union

def summarize_change(group):
    values = group["change_percentage_points"].dropna().to_numpy(dtype=float)
    return pd.Series({
        "n_repeats": len(values),
        "source_mean_percent": group["source_contribution_percent"].mean(),
        "weighted_mean_percent": group["weighted_contribution_percent"].mean(),
        "mean_change_pp": np.mean(values),
        "SD_change_pp": np.std(values, ddof=1),
        "change_q025": np.quantile(values, 0.025),
        "change_q975": np.quantile(values, 0.975),
        "proportion_change_positive": np.mean(values > 0),
    })


feature_stability_df = (
    shap_feature_repeats_df.groupby(["domain", "feature"], as_index=False)
    .apply(summarize_change)
    .reset_index(drop=True)
)
domain_stability_df = (
    shap_domain_repeats_df.groupby("domain", as_index=False)
    .apply(summarize_change)
    .reset_index(drop=True)
)

# Prespecified selection rule for the supplementary feature-level plot:
# union of the top 10 predictors in the SNUH-only and representative
# weighted models. Predictors present in both lists are retained once.
TOP_N_SHAP_FEATURES = 10

feature_stability_df["absolute_mean_change_pp"] = (
    feature_stability_df["mean_change_pp"].abs()
)
feature_stability_df["source_rank"] = (
    feature_stability_df["source_mean_percent"]
    .rank(ascending=False, method="min")
    .astype(int)
)
feature_stability_df["weighted_rank"] = (
    feature_stability_df["weighted_mean_percent"]
    .rank(ascending=False, method="min")
    .astype(int)
)

source_top_features = set(
    feature_stability_df
    .nsmallest(TOP_N_SHAP_FEATURES, "source_rank")["feature"]
)
weighted_top_features = set(
    feature_stability_df
    .nsmallest(TOP_N_SHAP_FEATURES, "weighted_rank")["feature"]
)
selected_feature_union = source_top_features | weighted_top_features

selected_features_df = feature_stability_df.loc[
    feature_stability_df["feature"].isin(selected_feature_union)
].copy()

def identify_top_feature_membership(feature):
    in_source = feature in source_top_features
    in_weighted = feature in weighted_top_features
    if in_source and in_weighted:
        return "Top 10 in both models"
    if in_source:
        return "Top 10 in SNUH-only"
    return f"Top 10 in {int(round(PRIMARY_ALPHA * 100))}% BRM-weighted"

selected_features_df["selection_group"] = (
    selected_features_df["feature"].map(identify_top_feature_membership)
)
selected_features_df["best_model_rank"] = (
    selected_features_df[["source_rank", "weighted_rank"]].min(axis=1)
)
selected_features_df = selected_features_df.sort_values(
    ["domain", "best_model_rank", "absolute_mean_change_pp"],
    ascending=[True, True, False],
).reset_index(drop=True)
selected_features_df["selection_rule"] = (
    "Union of top 10 predictors by mean normalized SHAP contribution "
    "in the SNUH-only and representative weighted models"
)

drift_summary_df = pd.DataFrame({
    "metric": ["Kendall tau", "Jensen-Shannon distance", "Jensen-Shannon divergence"],
    "mean": [
        explanation_drift_df["kendall_tau"].mean(),
        explanation_drift_df["jensen_shannon_distance"].mean(),
        explanation_drift_df["jensen_shannon_divergence"].mean(),
    ],
    "SD": [
        explanation_drift_df["kendall_tau"].std(ddof=1),
        explanation_drift_df["jensen_shannon_distance"].std(ddof=1),
        explanation_drift_df["jensen_shannon_divergence"].std(ddof=1),
    ],
    "q025": [
        explanation_drift_df["kendall_tau"].quantile(0.025),
        explanation_drift_df["jensen_shannon_distance"].quantile(0.025),
        explanation_drift_df["jensen_shannon_divergence"].quantile(0.025),
    ],
    "q975": [
        explanation_drift_df["kendall_tau"].quantile(0.975),
        explanation_drift_df["jensen_shannon_distance"].quantile(0.975),
        explanation_drift_df["jensen_shannon_divergence"].quantile(0.975),
    ],
})

feature_stability_df.to_csv(SHAP_DIR / "SHAP_feature_change_stability_summary.csv", index=False, encoding="utf-8-sig")
domain_stability_df.to_csv(SHAP_DIR / "SHAP_domain_change_stability_summary.csv", index=False, encoding="utf-8-sig")
selected_features_df.to_csv(
    SHAP_DIR / "SHAP_top10_union_feature_change_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
drift_summary_df.to_csv(SHAP_DIR / "explanation_drift_metrics_summary.csv", index=False, encoding="utf-8-sig")

display(domain_stability_df)
display(selected_features_df)
display(drift_summary_df)

# ============================================================
# SHAP stability figures
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.lines import Line2D


# ------------------------------------------------------------
# 1. Common settings
# ------------------------------------------------------------

DOMAIN_ORDER = [
    "Laboratory results",
    "Hemodynamics / vital signs",
    "Patient characteristics",
    "Comorbidities",
]

DOMAIN_COLORS = {
    "Laboratory results": "#8C8C8C",
    "Hemodynamics / vital signs": "#2E8B57",
    "Patient characteristics": "#4169A8",
    "Comorbidities": "#C49A00",
}


# ------------------------------------------------------------
# 2. Automatic font selection
# ------------------------------------------------------------

available_fonts = {
    font.name
    for font in font_manager.fontManager.ttflist
}

FIGURE_FONT = (
    "Arial"
    if "Arial" in available_fonts
    else "DejaVu Sans"
)

plt.rcParams.update({
    "font.family": FIGURE_FONT,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

print(f"Figure font: {FIGURE_FONT}")


# ============================================================
# 3. Domain-level SHAP changes across repeated splits
# ============================================================

domain_plot = (
    domain_stability_df.copy()
    .set_index("domain")
    .reindex(DOMAIN_ORDER)
    .dropna(how="all")
    .reset_index()
)

domain_plot["error_lower"] = (
    domain_plot["mean_change_pp"]
    - domain_plot["change_q025"]
)

domain_plot["error_upper"] = (
    domain_plot["change_q975"]
    - domain_plot["mean_change_pp"]
)

fig, ax = plt.subplots(
    figsize=(7.4, 4.8)
)

y_positions = np.arange(
    len(domain_plot)
)

for y_position, (_, row) in zip(
    y_positions,
    domain_plot.iterrows(),
):
    domain = str(row["domain"])
    color = DOMAIN_COLORS.get(
        domain,
        "#4C78A8",
    )

    ax.errorbar(
        row["mean_change_pp"],
        y_position,
        xerr=np.array([
            [row["error_lower"]],
            [row["error_upper"]],
        ]),
        fmt="o",
        markersize=8,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=1.2,
        ecolor=color,
        elinewidth=2.0,
        capsize=4,
        capthick=1.5,
        zorder=3,
    )

ax.axvline(
    0,
    color="#666666",
    linestyle="--",
    linewidth=1.1,
    zorder=1,
)

ax.set_yticks(y_positions)
ax.set_yticklabels(
    domain_plot["domain"]
)
ax.invert_yaxis()

ax.set_xlabel(
    "Change in normalized SHAP contribution "
    "(percentage points)"
)
ax.set_ylabel("")

ax.set_title(
    "Domain-level SHAP changes across repeated splits"
)

ax.spines[
    ["top", "right"]
].set_visible(False)

ax.grid(
    axis="x",
    color="#E6E6E6",
    linewidth=0.7,
    alpha=0.7,
    zorder=0,
)

ax.tick_params(
    axis="both",
    width=0.9,
    length=4,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "SHAP_domain_change_stability.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    FIGURE_DIR /
    "SHAP_domain_change_stability.pdf",
    bbox_inches="tight",
    facecolor="white",
)

plt.show()
plt.close(fig)


# ============================================================
# 4. Feature-level SHAP change forest plot
# ============================================================

feature_plot = (
    selected_features_df.copy()
)

feature_plot["feature_label"] = (
    feature_plot["feature"]
    .map(FEATURE_DISPLAY_NAMES)
    .fillna(
        feature_plot["feature"]
        .str.replace(
            "_",
            " ",
            regex=False,
        )
    )
)

# Apply the same domain order used in the other figures
feature_plot["domain"] = pd.Categorical(
    feature_plot["domain"],
    categories=DOMAIN_ORDER,
    ordered=True,
)

feature_plot = (
    feature_plot
    .sort_values(
        [
            "domain",
            "best_model_rank",
            "absolute_mean_change_pp",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)

feature_plot["error_lower"] = (
    feature_plot["mean_change_pp"]
    - feature_plot["change_q025"]
)

feature_plot["error_upper"] = (
    feature_plot["change_q975"]
    - feature_plot["mean_change_pp"]
)

fig, ax = plt.subplots(
    figsize=(
        9.0,
        max(
            6.0,
            0.43 * len(feature_plot),
        ),
    )
)

y_positions = np.arange(
    len(feature_plot)
)

feature_plot["y_position"] = (
    y_positions
)

for domain in DOMAIN_ORDER:

    group = feature_plot.loc[
        feature_plot["domain"]
        == domain
    ]

    if group.empty:
        continue

    color = DOMAIN_COLORS[domain]

    ax.errorbar(
        group["mean_change_pp"],
        group["y_position"],
        xerr=np.vstack([
            group["error_lower"],
            group["error_upper"],
        ]),
        fmt="o",
        markersize=7,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=1.0,
        color=color,
        ecolor=color,
        elinewidth=1.8,
        capsize=3,
        capthick=1.3,
        label=domain,
        zorder=3,
    )

ax.axvline(
    0,
    color="#666666",
    linestyle="--",
    linewidth=1.1,
    zorder=1,
)

ax.set_yticks(y_positions)

ax.set_yticklabels(
    feature_plot["feature_label"]
)

ax.invert_yaxis()

ax.set_xlabel(
    "Change in normalized SHAP contribution "
    "(percentage points)"
)

ax.set_ylabel("")

ax.spines[
    ["top", "right"]
].set_visible(False)

ax.grid(
    axis="x",
    color="#E6E6E6",
    linewidth=0.7,
    alpha=0.7,
    zorder=0,
)

ax.tick_params(
    axis="both",
    width=0.9,
    length=4,
)


# Fixed domain legend order
available_domains = set(
    feature_plot["domain"]
    .dropna()
    .astype(str)
)

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=DOMAIN_COLORS[domain],
        markeredgecolor=DOMAIN_COLORS[domain],
        markersize=7,
        label=domain,
    )
    for domain in DOMAIN_ORDER
    if domain in available_domains
]

ax.legend(
    handles=legend_handles,
    frameon=False,
    loc="lower right",
    fontsize=9,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "SHAP_top10_union_feature_change_forest.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    FIGURE_DIR /
    "SHAP_top10_union_feature_change_forest.pdf",
    bbox_inches="tight",
    facecolor="white",
)

plt.show()
plt.close(fig)


# ============================================================
# 5. Explanation-drift metric distributions
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.0, 4.2),
)

# Kendall tau distribution
axes[0].hist(
    explanation_drift_df[
        "kendall_tau"
    ].dropna(),
    bins=12,
    color="#4C78A8",
    edgecolor="white",
    linewidth=0.8,
)

axes[0].set_xlabel(
    "Kendall tau"
)

axes[0].set_ylabel(
    "Number of repeated splits"
)

axes[0].set_title(
    "Global importance-rank agreement"
)


# Jensen–Shannon distance distribution
axes[1].hist(
    explanation_drift_df[
        "jensen_shannon_distance"
    ].dropna(),
    bins=12,
    color="#F28E2B",
    edgecolor="white",
    linewidth=0.8,
)

axes[1].set_xlabel(
    "Jensen–Shannon distance"
)

axes[1].set_ylabel(
    "Number of repeated splits"
)

axes[1].set_title(
    "Normalized importance-distribution shift"
)

for ax in axes:

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    ax.tick_params(
        axis="both",
        width=0.9,
        length=4,
    )

    ax.grid(
        axis="y",
        color="#E6E6E6",
        linewidth=0.7,
        alpha=0.6,
        zorder=0,
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "explanation_drift_metric_distributions.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    FIGURE_DIR /
    "explanation_drift_metric_distributions.pdf",
    bbox_inches="tight",
    facecolor="white",
)

plt.show()
plt.close(fig)


# ============================================================


# 8. Fixed-split SHAP summary dot plots and dependence plots
# ============================================================
# These figures use the same fixed BRM evaluation observations, deterministic
# background sample, and interventional TreeExplainer for both models.
# Probability recalibration is not applied to SHAP because it does not alter
# the fitted random-forest decision structure.

SHAP_SUMMARY_MAX_DISPLAY = 10
SHAP_BACKGROUND_MAX_N = 100
SHAP_FIGURE_DPI = 600


def restore_original_feature_units(X_transformed, fitted_preprocessor):
    """Return an imputed display DataFrame with continuous variables restored
    to their original clinical units. Row order and feature order are retained.
    """
    X_display = X_transformed.copy().astype(float).reset_index(drop=True)
    X_display.loc[:, continuous_cols] = fitted_preprocessor["scaler"].inverse_transform(
        X_display[continuous_cols]
    )
    return X_display[feature_cols]


def calculate_fixed_model_shap(
    model,
    X_model,
    X_background,
    seed,
):
    """Calculate raw positive-class SHAP values using the locked method."""
    importance, shap_values = compute_normalized_shap_importance(
        model=model,
        X_eval=X_model,
        X_background=X_background,
        max_background=SHAP_BACKGROUND_MAX_N,
        seed=seed,
    )
    return importance, np.asarray(shap_values, dtype=float)


def save_shap_summary_dot_plot(
    shap_values,
    X_display,
    output_stem,
    max_display=10,
):
    """Save one publication-ready SHAP beeswarm/dot summary plot."""
    display_data = X_display.rename(columns=FEATURE_DISPLAY_NAMES)
    plt.figure(figsize=(8.2, 6.2))
    shap.summary_plot(
        shap_values,
        display_data,
        plot_type="dot",
        max_display=max_display,
        show=False,
        plot_size=None,
        color_bar_label="Feature value",
    )
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(8.2, 6.2)
    ax.set_xlabel("SHAP value", fontsize=12)
    ax.tick_params(axis="both", labelsize=10.5)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=SHAP_FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def summarize_fixed_shap_ranking(shap_values, model_label):
    """Return global mean-absolute-SHAP ranking for one fixed model."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    contribution = mean_abs_shap / mean_abs_shap.sum()
    ranking = pd.DataFrame({
        "model": model_label,
        "feature": feature_cols,
        "feature_label": [FEATURE_DISPLAY_NAMES.get(x, x) for x in feature_cols],
        "mean_absolute_shap": mean_abs_shap,
        "normalized_contribution_percent": contribution * 100,
    }).sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def _binned_shap_trend(x, y, n_bins=20):
    """Calculate a deterministic quantile-binned trend for visualization."""
    trend_df = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    if trend_df.empty:
        return np.array([]), np.array([])
    unique_n = trend_df["x"].nunique()
    if unique_n <= 2:
        grouped = trend_df.groupby("x", as_index=False)["y"].mean().sort_values("x")
    else:
        q = min(int(n_bins), int(unique_n))
        trend_df["bin"] = pd.qcut(trend_df["x"], q=q, duplicates="drop")
        grouped = trend_df.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean")).reset_index(drop=True)
    return grouped["x"].to_numpy(), grouped["y"].to_numpy()


def plot_shap_dependence_comparison(
    feature,
    X_original,
    shap_source,
    shap_weighted,
    output_dir=FIGURE_DIR,
    source_label="SNUH-only",
    weighted_label=None,
    n_bins=20,
    point_size=25,
    point_alpha=0.45,
    xlim=None,
    ylim=None,
    show_title=False,
):
    """Compare predictor-SHAP relationships using original clinical units.

    Parameters
    ----------
    feature : str
        Internal model-input name, e.g., 'paco2', 'hco3', 'creatinine', or 'BMI'.
    X_original : pandas.DataFrame
        Imputed feature matrix expressed in original clinical units.
    shap_source, shap_weighted : numpy.ndarray
        SHAP matrices calculated on the same rows and feature order.
    """
    if feature not in feature_cols:
        raise KeyError(f"Unknown feature: {feature}. Choose one of: {feature_cols}")
    if len(X_original) != len(shap_source) or len(X_original) != len(shap_weighted):
        raise ValueError("X_original and both SHAP matrices must contain the same rows.")

    weighted_label = weighted_label or f"{int(round(PRIMARY_ALPHA * 100))}% BRM-weighted"
    feature_index = feature_cols.index(feature)
    x = X_original[feature].to_numpy(dtype=float)
    y_source = shap_source[:, feature_index]
    y_weighted = shap_weighted[:, feature_index]

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.scatter(x, y_source, s=point_size, alpha=point_alpha, color="#4C78A8", edgecolors="none", label=source_label)
    ax.scatter(x, y_weighted, s=point_size, alpha=point_alpha, color="#F28E2B", edgecolors="none", label=weighted_label)

    for y_values, color in [(y_source, "#4C78A8"), (y_weighted, "#F28E2B")]:
        trend_x, trend_y = _binned_shap_trend(x, y_values, n_bins=n_bins)
        ax.plot(trend_x, trend_y, color=color, linewidth=2.2)

    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"{FEATURE_DISPLAY_NAMES.get(feature, feature)} original value", fontsize=12)
    ax.set_ylabel("SHAP value", fontsize=12)
    if show_title:
        ax.set_title(FEATURE_DISPLAY_NAMES.get(feature, feature))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()

    safe_feature = feature.lower().replace(" ", "_")
    output_stem = Path(output_dir) / f"SHAP_dependence_{safe_feature}_source_vs_weighted"
    fig.savefig(output_stem.with_suffix(".png"), dpi=SHAP_FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    return output_stem


# Fixed evaluation data in original clinical units for colors and x axes.
X_brm_fixed_eval_original = restore_original_feature_units(
    X_brm_fixed_eval,
    preprocessor,
)

# Both models are explained on exactly the same fixed observations and with
# the same deterministic background-sampling seed.
source_fixed_importance, shap_source_fixed = calculate_fixed_model_shap(
    source_model,
    X_brm_fixed_eval,
    X_brm_fixed_eval,
    seed=SEED,
)
weighted_fixed_importance, shap_weighted_fixed = calculate_fixed_model_shap(
    selected_model,
    X_brm_fixed_eval,
    X_brm_fixed_eval,
    seed=SEED,
)

source_ranking_df = summarize_fixed_shap_ranking(shap_source_fixed, "SNUH-only")
weighted_ranking_df = summarize_fixed_shap_ranking(
    shap_weighted_fixed,
    f"{int(round(PRIMARY_ALPHA * 100))}% BRM-weighted",
)
fixed_shap_ranking_df = pd.concat([source_ranking_df, weighted_ranking_df], ignore_index=True)
fixed_shap_ranking_df.to_csv(
    SHAP_DIR / "fixed_evaluation_SHAP_feature_rankings.csv",
    index=False,
    encoding="utf-8-sig",
)

save_shap_summary_dot_plot(
    shap_source_fixed,
    X_brm_fixed_eval_original,
    FIGURE_DIR / "SHAP_summary_dot_SNUH_only_top10",
    max_display=SHAP_SUMMARY_MAX_DISPLAY,
)
save_shap_summary_dot_plot(
    shap_weighted_fixed,
    X_brm_fixed_eval_original,
    FIGURE_DIR / f"SHAP_summary_dot_weighted_alpha_{PRIMARY_ALPHA:.1f}_top10",
    max_display=SHAP_SUMMARY_MAX_DISPLAY,
)

display(source_ranking_df.head(SHAP_SUMMARY_MAX_DISPLAY))
display(weighted_ranking_df.head(SHAP_SUMMARY_MAX_DISPLAY))

# Optional examples: uncomment only the clinically relevant predictors selected
# for the supplementary dependence analysis.
# plot_shap_dependence_comparison(
#     feature="paco2",
#     X_original=X_brm_fixed_eval_original,
#     shap_source=shap_source_fixed,
#     shap_weighted=shap_weighted_fixed,
# )
# plot_shap_dependence_comparison(
#     feature="hco3",
#     X_original=X_brm_fixed_eval_original,
#     shap_source=shap_source_fixed,
#     shap_weighted=shap_weighted_fixed,
# )
# plot_shap_dependence_comparison(
#     feature="creatinine",
#     X_original=X_brm_fixed_eval_original,
#     shap_source=shap_source_fixed,
#     shap_weighted=shap_weighted_fixed,
# )
# plot_shap_dependence_comparison(
#     feature="BMI",
#     X_original=X_brm_fixed_eval_original,
#     shap_source=shap_source_fixed,
#     shap_weighted=shap_weighted_fixed,
# )

# 9. Reproducibility metadata
metadata = {
    "generated_at": datetime.now().isoformat(),
    "base_seed": SEED,
    "selected_alpha": PRIMARY_ALPHA,
    "representative_alpha_rule": "Highest observed AUROC across the complete alpha grid on the fixed BRM evaluation set",
    "fixed_evaluation_used_for_representative_alpha_choice": True,
    "representative_alpha_interpretation": "Descriptive/exploratory; not an independently validated optimal alpha",
    "repeated_split_role": "Robustness/split-variability analysis, not a globally fixed test set",
    "fraction_sensitivity": FRACTION_GRID,
    "rf_parameters": RF_PARAMS,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn_version,
    "python": platform.python_version(),
}
(OUTPUT_DIR / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
display(metadata)

# Analysis outputs

# The following cell lists the principal files generated by the revised workflow.
# Raw and calibrated probabilities are intentionally retained separately so that
# discrimination, probability calibration, and model interpretation are not conflated.


analysis_output_manifest = pd.DataFrame([
    ["Source calibration parameter", TABLE_DIR / "source_calibration_parameter.csv"],
    ["Source raw/calibrated performance", TABLE_DIR / "source_calibration_performance.csv"],
    ["Fixed alpha trajectory", TABLE_DIR / "fixed_split_alpha_trajectory.csv"],
    ["All fixed-split methods", TABLE_DIR / "fixed_split_all_methods_performance_calibration.csv"],
    ["Bootstrap performance intervals", TABLE_DIR / "fixed_evaluation_bootstrap_CI.csv"],
    ["Calibration plotting points", TABLE_DIR / "calibration_plot_points.csv"],
    ["Adaptation fraction results", TABLE_DIR / "adaptation_fraction_10_20_30_summary.csv"],
    ["Repeated split alpha results", TABLE_DIR / "repeated_split_alpha_performance_summary.csv"],
    ["Repeated split method comparison", TABLE_DIR / "repeated_split_primary_methods_summary.csv"],
    ["Repeated split paired differences", TABLE_DIR / "repeated_split_paired_differences.csv"],
    ["Repeated split best-alpha distribution", TABLE_DIR / "repeated_split_best_alpha_distribution_exploratory.csv"],
    ["Repeated SHAP domain results", SHAP_DIR / "SHAP_domain_contribution_repeated_splits.csv"],
    ["Repeated SHAP feature results", SHAP_DIR / "SHAP_feature_contribution_repeated_splits.csv"],
    ["Explanation-drift metrics", SHAP_DIR / "explanation_drift_metrics_repeated_splits.csv"],
    ["Corrected calibration figure", FIGURE_DIR / "Figure_2B_calibration_corrected.png"],
], columns=["output", "path"])

analysis_output_manifest["exists"] = analysis_output_manifest["path"].map(
    lambda path: Path(path).exists()
)
analysis_output_manifest.to_csv(
    OUTPUT_DIR / "analysis_output_manifest.csv", index=False
)
display(analysis_output_manifest)


# ============================================================



