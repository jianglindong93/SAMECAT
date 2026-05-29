#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import optuna
from optuna.integration import XGBoostPruningCallback
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


def optimize_xgboost(X_train, mra_train, y_train,
                     X_valid, mra_valid, y_valid,
                     X_tune, mra_tune, y_tune,
                     X_test, mra_test, y_test,
                     conduct_clr_transformation = True,
                     n_trials = 100, max_epochs = 1000):
    """
    Optimize XGBoost hyperparameters using Optuna with Hyperband, train on the combined tuning set,
    and evaluate on both tuning and testing sets.

    Parameters:
    - X_train (pd.DataFrame or np.ndarray): Features for the training set.
    - y_train (pd.Series or np.ndarray): BMD scores for the training set.
    - X_val (pd.DataFrame or np.ndarray): Features for the validation set.
    - y_val (pd.Series or np.ndarray): BMD scores for the validation set.
    - X_tune (pd.DataFrame or np.ndarray): Features for the tuning set.
    - y_tune (pd.Series or np.ndarray): BMD scores for the tuning set.
    - X_test (pd.DataFrame or np.ndarray): Features for the testing set.
    - y_test (pd.Series or np.ndarray): BMD scores for the testing set.
    - summarized_results_dict (dict): Stored best hyperparameters and results from Phase 1.
    - n_trials (int): Number of hyperparameter optimization trials. Default is 100.
    - max_epochs (int): Maximum number of boosting rounds (epochs). Default is 100.

    Returns:
    - dict: Contains best parameters for Phase 1 & 2,
            Phase 1 model losses for tuning and testing sets,
            Phase 2 MSE for tuning and testing sets, and R² for the testing set.
    """

    # Ensure input data is in NumPy array format
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

    X_train = to_numpy(X_train)
    mra_train = to_numpy(mra_train) # Convert mra_train to numpy array
    if conduct_clr_transformation:
        mra_train = normalize(mra_train)
        mra_train = clr_transform(mra_train)
    input_train = np.concatenate((X_train, mra_train), axis=1)
    y_train = to_numpy(y_train).flatten()

    X_valid = to_numpy(X_valid)
    mra_valid = to_numpy(mra_valid) # Convert mra_valid to numpy array
    if conduct_clr_transformation:
        mra_valid = normalize(mra_valid)
        mra_valid = clr_transform(mra_valid)
    input_valid = np.concatenate((X_valid, mra_valid), axis=1)
    y_valid = to_numpy(y_valid).flatten()

    X_tune = to_numpy(X_tune)
    mra_tune = to_numpy(mra_tune) # Convert mra_tune to numpy array
    if conduct_clr_transformation:
        mra_tune = normalize(mra_tune)
        mra_tune = clr_transform(mra_tune)
    input_tune = np.concatenate((X_tune, mra_tune), axis=1)
    y_tune = to_numpy(y_tune).flatten()

    X_test = to_numpy(X_test)
    mra_test = to_numpy(mra_test) # Convert mra_test to numpy array
    if conduct_clr_transformation:
        mra_test = normalize(mra_test)
        mra_test = clr_transform(mra_test)
    input_test = np.concatenate((X_test, mra_test), axis=1)
    y_test = to_numpy(y_test).flatten()

    summarized_results_dict = {}
    def objective(trial):
        # Define the hyperparameter search space
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'tree_method': 'auto',
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, max_epochs),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }

        # Create DMatrix for training and validation
        dtrain = xgb.DMatrix(input_train, label = y_train)
        dvalid = xgb.DMatrix(input_valid, label = y_valid)

        pruning_callback = XGBoostPruningCallback(trial, "validation-rmse")

        # Train the model
        model = xgb.train(
            param,
            dtrain,
            num_boost_round = param['n_estimators'],
            evals = [(dvalid, 'validation')],
            early_stopping_rounds = 10,
            verbose_eval = False,
            callbacks=[pruning_callback]
        )

        # Predict on the validation set
        y_valid_pred = model.predict(dvalid)

        # Calculate RMSE
        rmse = mean_squared_error(y_valid, y_valid_pred)**0.5
        return rmse

    # Set up the Optuna study with Hyperband
    pruner = optuna.pruners.HyperbandPruner(min_resource = 10)
    study = optuna.create_study(direction = 'minimize', pruner = pruner)
    study.optimize(objective, n_trials = n_trials)

    # Retrieve the best hyperparameters
    best_params = study.best_params
    summarized_results_dict.update(best_params)

    # Train the final model on the entire tuning set
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(input_tune, y_tune)

    # Predictions
    y_tune_pred = final_model.predict(input_tune)
    y_test_pred = final_model.predict(input_test)

    # Calculate MSE
    rmse_tune = mean_squared_error(y_tune, y_tune_pred)**0.5
    rmse_test = mean_squared_error(y_test, y_test_pred)**0.5


    # Calculate R² for the testing set using squared Pearson correlation
    r_tune, _ = pearsonr(y_tune, y_tune_pred)
    r_squared_tune = r_tune ** 2
    r_test, _ = pearsonr(y_test, y_test_pred)
    r_squared_test = r_test ** 2


    summarized_results_dict.update({
        'RMSE_Tuning_Set': rmse_tune,
        'RMSE_Testing_Set': rmse_test,
        'R2_Tuning_Set': r_squared_tune,
        'R2_Testing_Set': r_squared_test
    })


    # Return the summarized results
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
            summarized_results_dict = optimize_xgboost(X_train, mra_train, Y_train,
                                                       X_valid, mra_valid, Y_valid,
                                                       X_tune, mra_tune, Y_tune,
                                                       X_test, mra_test, Y_test)
            summarized_results_cache.append(summarized_results_dict)
            div_track.append(div)
        div_track_dic = {'division': div_track}
        div_tract_cache = pd.DataFrame(data = div_track_dic)
        summarized_results_cache = pd.DataFrame.from_dict(summarized_results_cache)
        summarized_results_cache = pd.concat([div_tract_cache, summarized_results_cache], axis = 1)
        summarized_results_cache.to_csv(summarized_results_path + PB + "_" + bmd_site + "_xgboost_summarized_results.csv", index=False)


# In[ ]:




