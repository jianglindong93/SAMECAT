#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from xgboost import XGBRegressor


# In[12]:


def load_data(path, div_type, bmd_site, mask_path = None, mask_name = None,
              only_clinicvar = False, use_selected_microbes = False):
    subject_id = pd.read_csv(path + 'subject_id_' + div_type + '.csv')
    if subject_id.columns[0] != 'subject_id':
        subject_id.rename(columns={subject_id.columns[0]: 'subject_id'}, inplace=True)
    bmd_data = pd.read_csv(path + 'bmd_' + div_type + '.csv')
    Y = bmd_data[[bmd_site]]
    X = pd.read_csv(path + 'clinical_var_' + div_type + '.csv')
    ra_microbes_data = None
    if only_clinicvar == False:
        ra_microbes_data = pd.read_csv(path + 'microbe_comp_' + div_type + '.csv')
        if use_selected_microbes:
            selected_microbes = pd.read_csv(mask_path + 'microbe_names_' + mask_name + '.csv')
            selected_microbes = selected_microbes['species'].to_list()
            ra_microbes_data = ra_microbes_data[selected_microbes]

    return(subject_id, ra_microbes_data, X, Y)


# In[3]:


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# In[4]:


def _none_if_nan(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def _find_param_column(df: pd.DataFrame, candidates):
    """Return the first matching column from candidates (case-insensitive), else None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _parse_bool(x):
    x = _none_if_nan(x)
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
        raise ValueError(f"Unrecognized boolean value: {x!r}")
    if isinstance(x, (int, np.integer, float, np.floating)):
        return bool(int(x))
    return bool(x)


def _parse_max_features(x):
    """
    max_features: int, float in (0,1], 'sqrt', 'log2', or None.
    """
    x = _none_if_nan(x)
    if x is None:
        return None

    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"none", "null", "nan", ""}:
            return None
        if s in {"sqrt", "log2", "auto"}:
            return s
        # numeric string
        try:
            v = float(s)
            if v.is_integer():
                return int(v)
            return v
        except Exception:
            raise ValueError(f"Unrecognized max_features value: {x!r}")

    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if float(x).is_integer():
            return int(x)
        return float(x)

    return x


def _as_int_if_possible(x):
    x = _none_if_nan(x)
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return int(x)
    return x


# In[5]:


def build_rf_params(row: pd.Series) -> dict:
    """
    Build RF params from *one tuning row*.
    Only uses params present in the row.
    """
    params = {}

    if "n_estimators" in row:
        params["n_estimators"] = int(row["n_estimators"])
    if "max_depth" in row:
        md = _none_if_nan(row["max_depth"])
        params["max_depth"] = None if md is None else int(md)
    if "min_samples_split" in row:
        params["min_samples_split"] = int(row["min_samples_split"])
    if "min_samples_leaf" in row:
        params["min_samples_leaf"] = int(row["min_samples_leaf"])
    if "max_features" in row:
        params["max_features"] = _parse_max_features(row["max_features"])
    if "bootstrap" in row:
        b = _parse_bool(row["bootstrap"])
        if b is not None:
            params["bootstrap"] = b
    if "criterion" in row:
        params["criterion"] = str(row["criterion"])
    if "min_impurity_decrease" in row:
        params["min_impurity_decrease"] = float(row["min_impurity_decrease"])
    if "max_leaf_nodes" in row:
        mln = _none_if_nan(row["max_leaf_nodes"])
        params["max_leaf_nodes"] = None if mln is None else int(mln)
    if "ccp_alpha" in row:
        params["ccp_alpha"] = float(row["ccp_alpha"])

    return params

def build_xgb_params(row: pd.Series) -> dict:
    """
    Build XGBRegressor params from one tuning row.
    This function is intentionally permissive: it only consumes params that exist in the row.
    Add/remove mappings here to fit your exact tuning output.
    """
    params = {}

    # Core XGBoost hyperparams (common in tuning files)
    if "n_estimators" in row:
        params["n_estimators"] = int(row["n_estimators"])
    if "learning_rate" in row:
        params["learning_rate"] = float(row["learning_rate"])
    if "max_depth" in row:
        params["max_depth"] = int(row["max_depth"])
    if "min_child_weight" in row:
        params["min_child_weight"] = float(row["min_child_weight"])
    if "subsample" in row:
        params["subsample"] = float(row["subsample"])
    if "colsample_bytree" in row:
        params["colsample_bytree"] = float(row["colsample_bytree"])
    if "gamma" in row:
        params["gamma"] = float(row["gamma"])
    if "alpha" in row:
        params["alpha"] = float(row["alpha"])
    if "lambda" in row:
        params["lambda"] = float(row["lambda"])

    # Sometimes present
    if "max_delta_step" in row:
        v = _none_if_nan(row["max_delta_step"])
        if v is not None:
            params["max_delta_step"] = float(v)
    if "colsample_bylevel" in row:
        params["colsample_bylevel"] = float(row["colsample_bylevel"])
    if "colsample_bynode" in row:
        params["colsample_bynode"] = float(row["colsample_bynode"])
    if "scale_pos_weight" in row:
        # not typical for regression, but keep if present (harmless unless you set it oddly)
        v = _none_if_nan(row["scale_pos_weight"])
        if v is not None:
            params["scale_pos_weight"] = float(v)

    # If your tuning file contains "booster" or "tree_method"
    if "booster" in row:
        params["booster"] = str(row["booster"])
    if "tree_method" in row:
        params["tree_method"] = str(row["tree_method"])
    if "grow_policy" in row:
        params["grow_policy"] = str(row["grow_policy"])
    if "max_leaves" in row:
        v = _as_int_if_possible(row["max_leaves"])
        if v is not None:
            params["max_leaves"] = int(v)

    # Early stopping (optional): only used if you wire it into .fit()
    # Keeping it here as a placeholder.
    return params


# In[6]:


def to_numpy(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.to_numpy()
    return np.array(data)

def zero_handling(X):
    nonzero_vals = X[X > 0]
    if nonzero_vals.size == 0:
        raise ValueError("Array contains no non-zero values.")
    min_nonzero = nonzero_vals.min()
    # compute replacement = half of that
    replacement = 0.5 * min_nonzero
    # replace zeros in place
    X[X == 0] = replacement
    #self.X[self.X == 0] = 1

def normalize(X):
    zero_handling(X)
    if np.any(X==0):
        raise ValueError("Exist zero values in mRA matrix.")
    X_sum = X.sum(axis=1, keepdims=True)
    X_normalized = X / X_sum
    return X_normalized

def clr_transform(X):
    X_log = np.log(X)
    X_clr = X_log - X_log.mean(axis=1, keepdims=True)
    return X_clr

def process_input_data(covar, mra, y, conduct_clr_transformation=True):
    covar = to_numpy(covar)
    mra = to_numpy(mra) # Convert mra_train to numpy array
    if conduct_clr_transformation:
        mra = normalize(mra)
        mra = clr_transform(mra)
    input_X = np.concatenate((covar, mra), axis=1)
    y = to_numpy(y).flatten()
    return input_X, y


# In[7]:


def rf_validation_results(tune, X_train, y_train, X_test, y_test):
    # =========================
    # TRAIN + EVALUATE FOR EACH TUNING ROW
    # =========================
    results = []

    # Keep some context columns if present (optional)
    context_cols = [c for c in ["division"] if c in tune.columns]
    #metric_cols = [c for c in tune.columns if any(k in c.lower() for k in ["rmse", "r2", "mae", "mse"])]

    for _, row in tune.iterrows():
        case_id = int(row["case_id"])
        rf_params = build_rf_params(row)

        model = RandomForestRegressor(
            #random_state=0,
            n_jobs=-1,
            **rf_params
        )

        model.fit(X_train, y_train)
        yhat_test = model.predict(X_test)
        yhat_train = model.predict(X_train)

        out = {
            "case_id": case_id,
            **({c: row[c] for c in context_cols} if context_cols else {}),
            **rf_params,
            "RMSE_train": rmse(y_train, yhat_train),
            "R2_train": float((pearsonr(y_train, yhat_train)[0])**2),
            "RMSE_test": rmse(y_test, yhat_test),
            "R2_test": float((pearsonr(y_test, yhat_test)[0])**2),
        }

        # Attach any tuning metrics for reference (keeps your original tuning summary info)
        #for c in metric_cols:
        #    out[f"tune_{c}"] = row[c]

        results.append(out)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("case_id").reset_index(drop=True)
    return results_df


# In[8]:


def xgb_validation_results(tune, X_train, y_train, X_test, y_test):
    # Optional carry-through columns
    context_cols = [c for c in ["division"] if c in tune.columns]
    #metric_cols = [c for c in tune.columns if any(k in c.lower() for k in ["rmse", "r2", "mae", "mse"])]

    # =========================
    # TRAIN + EVALUATE EACH CASE
    # =========================
    results = []
    
    for _, row in tune.iterrows():
        case_id = int(row["case_id"])
        xgb_params = build_xgb_params(row)

        model = XGBRegressor(
            objective="reg:squarederror",
            #random_state=0,
            n_jobs=-1,
            # For newer xgboost versions, this avoids some warnings:
            verbosity=0,
            **xgb_params
        )

        model.fit(X_train, y_train)
        
        yhat_train = model.predict(X_train)
        yhat_test = model.predict(X_test)

        out = {
            "case_id": case_id,
            **({c: row[c] for c in context_cols} if context_cols else {}),
            **xgb_params,
            "RMSE_train": rmse(y_train, yhat_train),
            "R2_train": float((pearsonr(y_train, yhat_train)[0])**2),
            "RMSE_test": rmse(y_test, yhat_test),
            "R2_test": float((pearsonr(y_test, yhat_test)[0])**2),
            "n_features": int(X_train.shape[1]),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        }

        # Attach tuning metrics for reference (no ranking applied)
        #for c in metric_cols:
        #    out[f"tune_{c}"] = row[c]

        results.append(out)

    results_df = pd.DataFrame(results).sort_values("case_id").reset_index(drop=True)
    return results_df


# In[10]:


bmd_sites = ["NECK_BMD", "HTOT_BMD", "spine_total_bmd", "R_13_BMD"]
model_list = ["rrf", "xgboost"]
root_path = "root_path/"
tune_path = root_path + "summarized_results_path/"
train_data_path = root_path + "bgi_data_folder/"
lc_data_path = root_path + "lc_data_folder/"
summarized_results_path = root_path + "summarized_results_path/"
os.makedirs(summarized_results_path, exist_ok=True)


# In[13]:


only_clinicvar = False
use_selected_microbes = True
mask_path = "mask_path/"
mask_name = 'wozeroonlyspecies'
for bmd_site in bmd_sites:
    subject_id_tune, mra_tune, X_tune, Y_tune = load_data(train_data_path,
                                                          "tu",
                                                          bmd_site,
                                                          mask_path,
                                                          mask_name = mask_name,
                                                          only_clinicvar = only_clinicvar,
                                                          use_selected_microbes = use_selected_microbes)

    subject_id_test, mra_test, X_test, Y_test = load_data(lc_data_path,
                                                          "lc",
                                                          bmd_site,
                                                          mask_path,
                                                          mask_name = mask_name,
                                                          only_clinicvar = only_clinicvar,
                                                          use_selected_microbes = use_selected_microbes)
    input_X_tune, y_tune = process_input_data(X_tune, mra_tune, Y_tune)
    input_X_test, y_test = process_input_data(X_test, mra_test, Y_test)
    for model_name in model_list:
        tune = pd.read_csv(tune_path + "all_species_" + bmd_site + "_" + model_name + "_summarized_results.csv").copy()
        tune.insert(0, "case_id", np.arange(len(tune), dtype=int))
        if model_name == "rrf":
            results_df = rf_validation_results(tune, input_X_tune, y_tune, input_X_test, y_test)
        elif model_name == "xgboost":
            results_df = xgb_validation_results(tune, input_X_tune, y_tune, input_X_test, y_test)
        else:
            raise ValueError("Model name not recognized or not in standard format.")
        results_df.to_csv(summarized_results_path + bmd_site + "_" + model_name + "_lc_validation_ml.csv", index = False)


# In[ ]:




