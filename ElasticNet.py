#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import optuna
import os


# In[ ]:


def load_data(path1, div_type, bmd_site, mask_path = None, mask_name = None,
              only_clinicvar = False, use_selected_microbes = False):
    subject_id = pd.read_csv(path1 + 'subject_id_' + div_type + '.csv')
    if subject_id.columns[0] != 'subject_id':
        subject_id.rename(columns={subject_id.columns[0]: 'subject_id'}, inplace=True)
    bmd_data = pd.read_csv(path1 + 'bmd_' + div_type + '.csv')
    Y = bmd_data[[bmd_site]]
    X = pd.read_csv(path1 + 'clinical_var_' + div_type + '.csv')
    ra_microbes_data = None
    if only_clinicvar == False:
        ra_microbes_data = pd.read_csv(path1 + 'microbe_comp_' + div_type + '.csv')
        if use_selected_microbes:
            selected_microbes = pd.read_csv(mask_path + 'microbe_names_' + mask_name + '.csv')
            selected_microbes = selected_microbes['species'].to_list()
            ra_microbes_data = ra_microbes_data[selected_microbes]

    return(subject_id, ra_microbes_data, X, Y)


# In[ ]:


def optimize_elastic_net(
    X_train, mra_train, y_train,
    X_valid, mra_valid, y_valid,
    X_tune,  mra_tune,  y_tune,
    X_test,  mra_test,  y_test,
    n_trials=100,
    conduct_clr_transformation=True,
    covariate_cols=None,              # e.g. ["age","bmi","Milk_num_week","Yogurt_num_week","PC1","PC2"]
    auto_detect_continuous=True,      # if covariate_cols is None, detect numeric cols with >2 unique values
    standardize_mra_after_clr=True,   # standardize CLR(mra)
    eps_strategy="half_min_nonzero",  # how to replace zeros before log
):
    """
    Optimize Elastic Net hyperparameters using Optuna, then train on tuning set and evaluate on test set.
    With: standardization for selected/auto-detected clinical covariates + CLR(mra) wt/wo standardization.
    """

    # -----------------------
    # Helpers
    # -----------------------
    def to_numpy(data):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.to_numpy()
        return np.asarray(data)

    def zero_handling_inplace(X):
        # X is numpy array; modifies in place
        nonzero_vals = X[X > 0]
        if nonzero_vals.size == 0:
            raise ValueError("Array contains no non-zero values.")
        min_nonzero = nonzero_vals.min()
        replacement = 0.5 * min_nonzero
        X[X == 0] = replacement

    def normalize_rows_to_sum1(X):
        X_sum = X.sum(axis=1, keepdims=True)
        # guard against divide by zero
        if np.any(X_sum == 0):
            raise ValueError("Some rows sum to zero after zero handling; cannot normalize.")
        return X / X_sum

    def clr_transform(X):
        # expects strictly positive
        X_log = np.log(X)
        return X_log - X_log.mean(axis=1, keepdims=True)

    def prep_mra(mra):
        m = to_numpy(mra).astype(float, copy=True)
        if conduct_clr_transformation:
            if eps_strategy == "half_min_nonzero":
                zero_handling_inplace(m)
            else:
                raise ValueError(f"Unknown eps_strategy: {eps_strategy}")
            m = normalize_rows_to_sum1(m)
            m = clr_transform(m)
        return m

    def ensure_df(X):
        # Keep DataFrame if already; otherwise create one with generic names
        if isinstance(X, pd.DataFrame):
            return X.copy()
        X_np = to_numpy(X)
        return pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])

    def detect_continuous_cols(X_df):
        # numeric columns with >2 unique values (ignores NaN in uniqueness)
        cols = []
        for c in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[c]):
                nunq = X_df[c].nunique(dropna=True)
                if nunq > 2:
                    cols.append(c)
        return cols

    def standardize_covariates_fit_transform(X_fit_df, X_apply_dfs, cols_to_scale):
        """
        Fit scaler on X_fit_df[cols_to_scale] and transform those cols in each df in X_apply_dfs.
        Returns: scaler, transformed_dfs
        """
        scaler = StandardScaler()
        if len(cols_to_scale) == 0:
            # nothing to scale
            return None, [df.copy() for df in X_apply_dfs]

        X_fit = X_fit_df[cols_to_scale].to_numpy(dtype=float)
        scaler.fit(X_fit)

        out = []
        for df in X_apply_dfs:
            df2 = df.copy()
            df2.loc[:, cols_to_scale] = scaler.transform(df2[cols_to_scale].to_numpy(dtype=float))
            out.append(df2)
        return scaler, out

    def standardize_matrix_fit_transform(M_fit, M_apply_list):
        scaler = StandardScaler()
        scaler.fit(M_fit)
        return scaler, [scaler.transform(M) for M in M_apply_list]

    # -----------------------
    # 1) Prepare X as DataFrames
    # -----------------------
    Xtr_df = ensure_df(X_train)
    Xva_df = ensure_df(X_valid)
    Xtu_df = ensure_df(X_tune)
    Xte_df = ensure_df(X_test)

    # Decide which covariates to standardize
    if covariate_cols is not None:
        cols_to_scale = [c for c in covariate_cols if c in Xtr_df.columns]
        missing = [c for c in covariate_cols if c not in Xtr_df.columns]
        if len(missing) > 0:
            raise ValueError(f"These covariate_cols are missing in X_train columns: {missing}")
    else:
        cols_to_scale = detect_continuous_cols(Xtr_df) if auto_detect_continuous else []

    print(cols_to_scale)

    # -----------------------
    # 2) Prepare mra (CLR etc.)
    # -----------------------
    mtr = prep_mra(mra_train)
    mva = prep_mra(mra_valid)
    mtu = prep_mra(mra_tune)
    mte = prep_mra(mra_test)

    # Targets to numpy
    ytr = to_numpy(y_train).flatten()
    yva = to_numpy(y_valid).flatten()
    ytu = to_numpy(y_tune).flatten()
    yte = to_numpy(y_test).flatten()

    # ============================================================
    # OPTUNA PHASE: fit scalers on TRAIN, apply to VALID
    # ============================================================
    _, [Xtr_scaled_df, Xva_scaled_df] = standardize_covariates_fit_transform(
        Xtr_df, [Xtr_df, Xva_df], cols_to_scale
    )

    Xtr_scaled = Xtr_scaled_df.to_numpy(dtype=float)
    Xva_scaled = Xva_scaled_df.to_numpy(dtype=float)

    if standardize_mra_after_clr:
        _, [mtr_scaled, mva_scaled] = standardize_matrix_fit_transform(mtr, [mtr, mva])
    else:
        mtr_scaled, mva_scaled = mtr, mva

    input_train = np.concatenate((Xtr_scaled, mtr_scaled), axis=1)
    input_valid = np.concatenate((Xva_scaled, mva_scaled), axis=1)

    summarized_results_dict = {}

    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-5, 1e2, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        model.fit(input_train, ytr)

        y_valid_pred = model.predict(input_valid)
        rmse_valid = mean_squared_error(yva, y_valid_pred) ** 0.5
        return rmse_valid

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    summarized_results_dict.update(best_params)
    summarized_results_dict["scaled_covariates"] = cols_to_scale
    summarized_results_dict["n_covariates_scaled"] = len(cols_to_scale)
    summarized_results_dict["standardize_mra_after_clr"] = bool(standardize_mra_after_clr)

    # ============================================================
    # FINAL MODEL PHASE: fit scalers on TUNE, apply to TEST
    # ============================================================
    _, [Xtu_scaled_df, Xte_scaled_df] = standardize_covariates_fit_transform(
        Xtu_df, [Xtu_df, Xte_df], cols_to_scale
    )
    Xtu_scaled = Xtu_scaled_df.to_numpy(dtype=float)
    Xte_scaled = Xte_scaled_df.to_numpy(dtype=float)

    if standardize_mra_after_clr:
        _, [mtu_scaled, mte_scaled] = standardize_matrix_fit_transform(mtu, [mtu, mte])
    else:
        mtu_scaled, mte_scaled = mtu, mte

    input_tune = np.concatenate((Xtu_scaled, mtu_scaled), axis=1)
    input_test = np.concatenate((Xte_scaled, mte_scaled), axis=1)

    final_model = ElasticNet(**best_params, max_iter=10000)
    final_model.fit(input_tune, ytu)

    y_tune_pred = final_model.predict(input_tune)
    y_test_pred = final_model.predict(input_test)

    rmse_tune = mean_squared_error(ytu, y_tune_pred) ** 0.5
    rmse_test = mean_squared_error(yte, y_test_pred) ** 0.5

    r_tune = pearsonr(ytu, y_tune_pred)[0]
    r_test = pearsonr(yte, y_test_pred)[0]

    summarized_results_dict.update({
        'RMSE_Tuning_Set': rmse_tune,
        'RMSE_Testing_Set': rmse_test,
        'R2_Tuning_Set': float(r_tune ** 2),
        'R2_Testing_Set': float(r_test ** 2),
    })

    return summarized_results_dict


# In[ ]:


data_path = "data_path/"
microbes_mask_path = "mask_path/"
div_list = np.char.add('tune_', np.array(list(range(1, 11, 1))).astype('str')).tolist()
bmd_site_list = ["NECK_BMD", "HTOT_BMD", "spine_total_bmd", "R_13_BMD"]
summarized_results_path = "summarized_results_path/"
os.makedirs(summarized_results_path, exist_ok=True)


# In[ ]:


PB_methods = ["all_species"]
only_clinicvar = False
use_selected_microbes = True
mask_name = 'mask_name'
for bmd_site in bmd_site_list:
    for PB in PB_methods:
        subject_id_tune, mra_tune, X_tune, Y_tune = load_data(data_path + "train_test_split/",
                                                              "tu",
                                                              bmd_site,
                                                              microbes_mask_path,
                                                              mask_name = mask_name,
                                                              only_clinicvar = only_clinicvar,
                                                              use_selected_microbes = use_selected_microbes)

        subject_id_test, mra_test, X_test, Y_test = load_data(data_path + "train_test_split/",
                                                              "te",
                                                              bmd_site,
                                                              microbes_mask_path,
                                                              mask_name = mask_name,
                                                              only_clinicvar = only_clinicvar,
                                                              use_selected_microbes = use_selected_microbes)
        div_track = []
        summarized_results_cache = []
        for div in div_list:
            print(f'Running on {div}')
            subject_id_train, mra_train, X_train, Y_train = load_data(data_path + div + "/",
                                                                      "tr_" + div,
                                                                      bmd_site,
                                                                      microbes_mask_path,
                                                                      mask_name = mask_name,
                                                                      only_clinicvar = only_clinicvar,
                                                                      use_selected_microbes = use_selected_microbes)

            subject_id_valid, mra_valid, X_valid, Y_valid = load_data(data_path + div + "/",
                                                                      "val_" + div,
                                                                      bmd_site,
                                                                      microbes_mask_path,
                                                                      mask_name = mask_name,
                                                                      only_clinicvar = only_clinicvar,
                                                                      use_selected_microbes = use_selected_microbes)
            summarized_results_dict = optimize_elastic_net(X_train, mra_train, Y_train,
                                                           X_valid, mra_valid, Y_valid,
                                                           X_tune, mra_tune, Y_tune,
                                                           X_test, mra_test, Y_test,
                                                           covariate_cols=None,
                                                           auto_detect_continuous=True,
                                                           standardize_mra_after_clr=True)
            summarized_results_cache.append(summarized_results_dict)
            div_track.append(div)
        div_track_dic = {'division': div_track}
        div_tract_cache = pd.DataFrame(data = div_track_dic)
        summarized_results_cache = pd.DataFrame.from_dict(summarized_results_cache)
        summarized_results_cache = pd.concat([div_tract_cache, summarized_results_cache], axis = 1)
        summarized_results_cache.to_csv(summarized_results_path + PB + "_" + bmd_site + "_en_summarized_results.csv", index=False)


# In[ ]:




