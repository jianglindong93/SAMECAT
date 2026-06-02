#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import optuna

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


# In[2]:


def load_data(path, div_type, bmd_site, mask_path=None, mask_name=None,
              only_clinicvar=False, use_selected_microbes=False):
    subject_id = pd.read_csv(path + 'subject_id_' + div_type + '.csv')
    if subject_id.columns[0] != 'subject_id':
        subject_id.rename(columns={subject_id.columns[0]: 'subject_id'}, inplace=True)

    bmd_data = pd.read_csv(path + 'bmd_' + div_type + '.csv')
    Y = bmd_data[[bmd_site]]

    X = pd.read_csv(path + 'clinical_var_' + div_type + '.csv')

    ra_microbes_data = None
    if not only_clinicvar:
        ra_microbes_data = pd.read_csv(path + 'microbe_comp_' + div_type + '.csv')
        if use_selected_microbes:
            if mask_path is None or mask_name is None:
                raise ValueError("mask_path and mask_name must be provided when use_selected_microbes=True")
            selected_microbes = pd.read_csv(mask_path + 'microbe_names_' + mask_name + '.csv')
            selected_microbes = selected_microbes['species'].to_list()
            ra_microbes_data = ra_microbes_data[selected_microbes]

    return subject_id, ra_microbes_data, X, Y


# In[3]:


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _find_param_column(df: pd.DataFrame, candidates):
    """Return the first matching column from candidates (case-insensitive), else None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def to_numpy(data):
    if data is None:
        return None
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    return np.asarray(data)


# -------------------------
# CLR pipeline
# -------------------------
def zero_handling_inplace(X):
    nonzero_vals = X[X > 0]
    if nonzero_vals.size == 0:
        raise ValueError("Array contains no non-zero values.")
    min_nonzero = nonzero_vals.min()
    replacement = 0.5 * min_nonzero
    X[X == 0] = replacement


def normalize_rows_to_sum1(X):
    # assumes X > 0
    X_sum = X.sum(axis=1, keepdims=True)
    if np.any(X_sum == 0):
        raise ValueError("Some rows sum to zero; cannot normalize.")
    return X / X_sum


def clr_transform(X):
    X_log = np.log(X)
    return X_log - X_log.mean(axis=1, keepdims=True)


def process_mra(mra, conduct_clr_transformation=True):
    mra = to_numpy(mra)
    if mra is None:
        return None
    mra = mra.astype(float, copy=True)
    if conduct_clr_transformation:
        zero_handling_inplace(mra)
        mra = normalize_rows_to_sum1(mra)
        mra = clr_transform(mra)
    return mra


# In[4]:


def detect_continuous_cols(X_df: pd.DataFrame):
    """Numeric columns with >2 unique values (ignores NaN); keeps binary/one-hot unscaled."""
    cols = []
    for c in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[c]):
            if X_df[c].nunique(dropna=True) > 2:
                cols.append(c)
    return cols


def fit_transform_X_scaler(X_fit_df, X_apply_df, covariate_cols=None, auto_detect_continuous=True):
    """
    Fit StandardScaler on selected continuous covariates from X_fit_df, transform both fit/apply.
    Returns: (X_fit_scaled_np, X_apply_scaled_np, cols_scaled, scaler_or_None)
    """
    if not isinstance(X_fit_df, pd.DataFrame) or not isinstance(X_apply_df, pd.DataFrame):
        raise TypeError("For robust covariate scaling, X_fit_df and X_apply_df must be pandas DataFrames.")

    if covariate_cols is not None:
        cols = [c for c in covariate_cols if c in X_fit_df.columns]
        missing = [c for c in covariate_cols if c not in X_fit_df.columns]
        if missing:
            raise ValueError(f"These covariate_cols are missing in X columns: {missing}")
    else:
        cols = detect_continuous_cols(X_fit_df) if auto_detect_continuous else []

    X_fit_scaled = X_fit_df.copy()
    X_apply_scaled = X_apply_df.copy()

    if len(cols) == 0:
        return (X_fit_scaled.to_numpy(dtype=float),
                X_apply_scaled.to_numpy(dtype=float),
                cols,
                None)

    scaler = StandardScaler()
    scaler.fit(X_fit_scaled[cols].to_numpy(dtype=float))

    X_fit_scaled.loc[:, cols] = scaler.transform(X_fit_scaled[cols].to_numpy(dtype=float))
    X_apply_scaled.loc[:, cols] = scaler.transform(X_apply_scaled[cols].to_numpy(dtype=float))

    return (X_fit_scaled.to_numpy(dtype=float),
            X_apply_scaled.to_numpy(dtype=float),
            cols,
            scaler)


def fit_transform_mra_scaler(mra_fit, mra_apply, standardize_mra_after_clr=True):
    """
    Fit StandardScaler on mra_fit (after CLR), transform both fit/apply.
    Returns: (mra_fit_scaled, mra_apply_scaled, scaler_or_None)
    """
    if mra_fit is None or mra_apply is None:
        return None, None, None
    if not standardize_mra_after_clr:
        return mra_fit, mra_apply, None

    scaler = StandardScaler()
    scaler.fit(mra_fit)
    return scaler.transform(mra_fit), scaler.transform(mra_apply), scaler


def build_inputs_scaled(
    X_fit, mra_fit, y_fit,
    X_apply, mra_apply, y_apply,
    conduct_clr_transformation=True,
    covariate_cols=None,
    auto_detect_continuous=True,
    standardize_mra_after_clr=True
):
    """
    Fit scalers on FIT split (tune), apply to APPLY split (test).
    - scales selected/auto-detected continuous covariates in X
    - applies CLR to mra (if provided), then scales mra features
    - concatenates X + mra if mra exists, else uses only X
    Returns: (input_fit, y_fit_np, input_apply, y_apply_np, meta_dict)
    """
    # Ensure X are DataFrames for robust scaling
    if not isinstance(X_fit, pd.DataFrame):
        X_fit = pd.DataFrame(to_numpy(X_fit), columns=[f"x{i}" for i in range(to_numpy(X_fit).shape[1])])
    if not isinstance(X_apply, pd.DataFrame):
        X_apply = pd.DataFrame(to_numpy(X_apply), columns=[f"x{i}" for i in range(to_numpy(X_apply).shape[1])])

    # y
    y_fit_np = to_numpy(y_fit).flatten()
    y_apply_np = to_numpy(y_apply).flatten()

    # X scaling (fit on fit split)
    X_fit_scaled, X_apply_scaled, cols_scaled, _ = fit_transform_X_scaler(
        X_fit, X_apply, covariate_cols=covariate_cols, auto_detect_continuous=auto_detect_continuous
    )

    # mra CLR + scaling (optional)
    mra_fit_clr = process_mra(mra_fit, conduct_clr_transformation=conduct_clr_transformation) if mra_fit is not None else None
    mra_apply_clr = process_mra(mra_apply, conduct_clr_transformation=conduct_clr_transformation) if mra_apply is not None else None

    use_mra = (mra_fit_clr is not None) and (mra_apply_clr is not None)

    if use_mra:
        mra_fit_scaled, mra_apply_scaled, _ = fit_transform_mra_scaler(
            mra_fit_clr, mra_apply_clr, standardize_mra_after_clr=standardize_mra_after_clr
        )
        input_fit = np.concatenate([X_fit_scaled, mra_fit_scaled], axis=1)
        input_apply = np.concatenate([X_apply_scaled, mra_apply_scaled], axis=1)
    else:
        input_fit = X_fit_scaled
        input_apply = X_apply_scaled

    meta = {
        "use_mra": bool(use_mra),
        "scaled_covariates": cols_scaled,
        "n_covariates_scaled": int(len(cols_scaled)),
        "standardize_mra_after_clr": bool(standardize_mra_after_clr) if use_mra else False,
        "n_features_total": int(input_fit.shape[1]),
    }
    return input_fit, y_fit_np, input_apply, y_apply_np, meta


# -------------------------
# ElasticNet evaluation (unchanged except uses already-prepared inputs)
# -------------------------
def en_validation_results(tune, X_train, y_train, X_test, y_test):
    alpha_col = _find_param_column(tune, ["alpha", "Alpha", "best_alpha", "enet_alpha"])
    l1_col = _find_param_column(tune, ["l1_ratio", "l1ratio", "L1_ratio", "best_l1_ratio", "enet_l1_ratio"])

    if alpha_col is None or l1_col is None:
        raise ValueError(
            "Elastic Net requires hyperparameter columns for alpha and l1_ratio in the tuning summary.\n"
            f"Detected alpha column: {alpha_col}\n"
            f"Detected l1_ratio column: {l1_col}\n"
        )

    results = []
    context_cols = [c for c in ["division"] if c in tune.columns]

    for _, row in tune.iterrows():
        case_id = int(row["case_id"])
        alpha = float(row[alpha_col])
        l1_ratio = float(row[l1_col])

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=10000
        )

        model.fit(X_train, y_train)

        yhat_train = model.predict(X_train)
        yhat_test = model.predict(X_test)

        out = {
            "case_id": case_id,
            **({c: row[c] for c in context_cols} if context_cols else {}),
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "RMSE_train": rmse(y_train, yhat_train),
            #"R2_train": float(r2_score(y_train, yhat_train)),
            "R2_train": float((pearsonr(y_train, yhat_train)[0])**2),
            "RMSE_test": rmse(y_test, yhat_test),
            #"R2_test": float(r2_score(y_test, yhat_test)),
            "R2_test": float((pearsonr(y_test, yhat_test)[0])**2),
            "n_features": int(X_train.shape[1]),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        }
        results.append(out)

    return pd.DataFrame(results).sort_values("case_id").reset_index(drop=True)


# In[5]:


bmd_sites = ["NECK_BMD", "HTOT_BMD", "spine_total_bmd", "R_13_BMD"]
model_list = ["en"]

root_path = "root_path/"
tune_path = root_path + "summarized_results_path/"
train_data_path = root_path + "bgi_data_folder/"
lc_data_path = root_path + "lc_data_folder/"
summarized_results_path = root_path + "summarized_results_path/"
os.makedirs(summarized_results_path, exist_ok=True)

only_clinicvar = False
use_selected_microbes = True
mask_path = "mask_path/"
mask_name = 'wozeroonlyspecies'

# Choose ONE of the two modes:
# (A) explicitly scale a specific list of covariates
explicit_covariates = ["age", "bmi", "Milk_num_week", "Yogurt_num_week", "PC1", "PC2"]

# (B) or auto-detect continuous covariates (numeric & >2 unique values)
auto_detect_continuous = True  # set True to use mode (B)
covariate_cols = explicit_covariates if not auto_detect_continuous else None


# In[6]:


for bmd_site in bmd_sites:
    subject_id_tune, mra_tune, X_tune, Y_tune = load_data(
        train_data_path, "tu", bmd_site,
        mask_path, mask_name=mask_name,
        only_clinicvar=only_clinicvar,
        use_selected_microbes=use_selected_microbes
    )

    subject_id_test, mra_test, X_test, Y_test = load_data(
        lc_data_path, "lc", bmd_site,
        mask_path, mask_name=mask_name,
        only_clinicvar=only_clinicvar,
        use_selected_microbes=use_selected_microbes
    )

    # ---- NEW: build standardized inputs (fit scalers on TUNE, apply to TEST) ----
    input_X_tune, y_tune, input_X_test, y_test, meta = build_inputs_scaled(
        X_fit=X_tune, mra_fit=mra_tune, y_fit=Y_tune,
        X_apply=X_test, mra_apply=mra_test, y_apply=Y_test,
        conduct_clr_transformation=True,
        covariate_cols=covariate_cols,
        auto_detect_continuous=auto_detect_continuous,
        standardize_mra_after_clr=True
    )

    for model_name in model_list:
        tune = pd.read_csv(
            tune_path + "all_species_" + bmd_site + "_" + model_name + "_summarized_results.csv"
        ).copy()
        tune.insert(0, "case_id", np.arange(len(tune), dtype=int))

        if model_name == "en":
            results_df = en_validation_results(tune, input_X_tune, y_tune, input_X_test, y_test)
        else:
            raise ValueError("This script section currently wired for ElasticNet ('en') only.")

        # (optional) append scaling metadata columns for traceability
        results_df["use_mra"] = meta["use_mra"]
        results_df["n_covariates_scaled"] = meta["n_covariates_scaled"]
        results_df["standardize_mra_after_clr"] = meta["standardize_mra_after_clr"]
        results_df["n_features_total"] = meta["n_features_total"]

        results_df.to_csv(
            summarized_results_path + bmd_site + "_" + model_name + "_lc_validation_ml.csv",
            index=False
        )


# In[ ]:




