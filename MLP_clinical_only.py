#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn.functional import relu
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna
import os
from copy import deepcopy
from scipy.stats import pearsonr
from torch.utils.data import Dataset

import torch.nn.utils.prune as prune

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


"""
Codes related with MIOSTONE network were adapted from https://github.com/batmen-lab/MIOSTONE.

Modifications were made to better accommodate our analyses.
"""

class MIOSTONEDataset(Dataset):
    """
    Handles Metagenomic relative abudance data + all preprocessing.
    """

    def __init__(self, subject_id, X, meta, y, features):
        self.subject_id = subject_id
        self.X = X
        self.meta = meta
        self.y = y
        self.features = features
        self.normalized = False
        self.clr_transformed = False
        self.data_adapted = False

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @classmethod
    def init_from_files(cls, master_path, div_type, bmd_site):
        subject_id = pd.read_csv(master_path + "subject_id_" + div_type + ".csv")
        data = pd.read_csv(master_path + "microbe_comp_" + div_type + ".csv")
        meta = pd.read_csv(master_path + "clinical_var_" + div_type + ".csv")
        bmd_data = pd.read_csv(master_path + "bmd_" + div_type + ".csv")
        y = bmd_data[[bmd_site]]
        features = ["s__" + col.split(".s__")[-1] for col in data.columns]
        
        #X = data.values.astype(np.float32)
        X = data.values
        meta = meta.values
        y = y.values
        
        return cls(subject_id, X, meta, y, features)
    
    def zero_handling(self):
        nonzero_vals = self.X[self.X > 0]
        if nonzero_vals.size == 0:
            raise ValueError("Array contains no non-zero values.")
        min_nonzero = nonzero_vals.min()
        # compute replacement = half of that
        replacement = 0.5 * min_nonzero
        # replace zeros in place
        self.X[self.X == 0] = replacement
        #self.X[self.X == 0] = 1

    def normalize(self):
        if self.normalized:
            raise ValueError("Dataset is already normalized")
        self.zero_handling()
        self.X_sum = self.X.sum(axis=1, keepdims=True)
        self.X = self.X / self.X_sum
        self.normalized = True

    def clr_transform(self):
        if self.clr_transformed:
            raise ValueError("Dataset is already clr-transformed")
        if self.normalized:
            self.X = np.log(self.X)
        else:
            self.X = np.log1p(self.X)
        self.X = self.X - self.X.mean(axis=1, keepdims=True)
        self.clr_transformed = True
    
    def data_adaptation(self, dtype):
        if self.data_adapted:
            raise ValueError("Dataset is already adapted")
        self.X = torch.from_numpy(self.X).type(dtype)
        self.meta = torch.from_numpy(self.meta).type(dtype)
        self.y = torch.from_numpy(self.y).type(dtype)
        if torch.cuda.is_available():
            self.X = self.X.cuda()
            self.meta = self.meta.cuda()
            self.y = self.y.cuda()
        self.data_adapted = True


# In[3]:


def mse_loss(pred_x, x):
    batch_size = x.size(0)
    assert batch_size != 0
    mse_loss_val = F.mse_loss(pred_x, x, reduction='sum').div(batch_size)
    
    return mse_loss_val

def get_r2(x, pred_x):
    r, _ = pearsonr(x, pred_x)
    r2 = r**2
    
    return r2


# In[4]:


class MLP_clinical_only(nn.Module):
    def __init__(self, 
                 input_vcovar_n1, 
                 #vcovar_level1_dim, 
                 fuse_level_dim, 
                 dc_h_dim):
        super(MLP_clinical_only, self).__init__()
        
        # clinical encoder
        self.vcovar_encoder = nn.Sequential(
            nn.Linear(input_vcovar_n1, fuse_level_dim),
            #nn.BatchNorm1d(fuse_level_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        ## BMD prediction heads
        # htot_bmd prediction
        self.decoder = nn.Sequential(
            nn.Linear(fuse_level_dim, dc_h_dim),
            #nn.BatchNorm1d(dc_h_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(dc_h_dim, 1)
        )
        
    def forward(self, clinical_data):
        vcovar_code = self.vcovar_encoder(clinical_data)
        pred_htot = self.decoder(vcovar_code)
        
        return pred_htot


# In[5]:


class Objective:
    def __init__(self, master_path, div, bmd_site, dtype = torch.float64):
        self.master_path = master_path
        self.div = div
        self.bmd_site = bmd_site
        self.dtype = dtype
        
    def __call__(self, trial):
        # Hyperparameter suggestions
        learning_rate2 = trial.suggest_float('learning_rate2', 1e-5, 1e-1, log=True)
        l2 = trial.suggest_float('l2', 1e-3, 1, log=True)
        p2_epoch_num = trial.suggest_int('p2_epoch_num', 100, 500, step = 100)
        
        # Data loader
        loaded_data_train = MIOSTONEDataset.init_from_files(self.master_path + self.div + '/', 
                                                            'tr_' + self.div,
                                                            self.bmd_site)
        
        loaded_data_valid = MIOSTONEDataset.init_from_files(self.master_path + self.div + '/', 
                                                            'val_' + self.div, 
                                                            self.bmd_site)
        
        input_vcovar_n1 = loaded_data_train.meta.shape[1]
        # Model, loss function, optimization
        model = MLP_clinical_only(input_vcovar_n1 = input_vcovar_n1, 
                                  fuse_level_dim = trial.suggest_int('fuse_level_dim', 2, 16, step = 2), 
                                  dc_h_dim = trial.suggest_int('dc_h_dim', 2, 8, step = 2))
        model = model.to(dtype=torch.float64, device=DEVICE)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate2, weight_decay = l2)
        scheduler = CosineAnnealingLR(optimizer, T_max=p2_epoch_num)
        
        loaded_data_train.data_adaptation(self.dtype)
        loaded_data_valid.data_adaptation(self.dtype)
        
        for epoch2 in range(1, p2_epoch_num + 1):
            model.train()
            optimizer.zero_grad()
            
            pred_htot = model(loaded_data_train.meta)
            loss = mse_loss(pred_htot, loaded_data_train.y)
            
            loss.backward()
            optimizer.step()
            
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                pred_htot_valid = model(loaded_data_valid.meta)
                loss_valid = mse_loss(pred_htot_valid, loaded_data_valid.y)
                
            trial.report(loss_valid, epoch2)
            if trial.should_prune():
                print(f'Trial {trial.number} pruned at phase 2 epoch {epoch2}.')
                raise optuna.exceptions.TrialPruned()
            
            if epoch2 % 100 == 0:
                print(f'Phase 2 - Epoch [{epoch2}/{p2_epoch_num}], Training Loss: {loss.item():.4f}, Validation Loss: {loss_valid.item():.4f}')
        
        return loss_valid


# In[6]:


def testMLP_clinical_only(master_path, div, bmd_site, 
                          best_params_dict, init_model_params_dict, model_path, 
                          dtype = torch.float64, save_pred_results = True):
    summarized_results_dict = {}
    summarized_results_dict.update(best_params_dict)
    
    loaded_data_tune = MIOSTONEDataset.init_from_files(master_path + 'train_test_split/', 
                                                       'tu',
                                                       bmd_site)
        
    loaded_data_test = MIOSTONEDataset.init_from_files(master_path + 'train_test_split/', 
                                                       'te', 
                                                       bmd_site)
    
    tune_num_subject = loaded_data_tune.X.shape[0]
    test_num_subject = loaded_data_test.X.shape[0]
    input_vcovar_n1 = loaded_data_tune.meta.shape[1]
    
    p2_epoch_num = best_params_dict['p2_epoch_num']
    learning_rate2 = best_params_dict['learning_rate2']
    l2 = best_params_dict['l2']
    
    init_model_params = {key: best_params_dict[key] for key in init_model_params_dict if key in best_params_dict}
    
    model = MLP_clinical_only(input_vcovar_n1 = input_vcovar_n1, 
                              **init_model_params)
    
    os.makedirs(model_path, exist_ok=True)
    model = model.to(dtype=torch.float64, device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate2, weight_decay = l2)
    scheduler = CosineAnnealingLR(optimizer, T_max=p2_epoch_num)
    
    loaded_data_tune.data_adaptation(dtype)
    loaded_data_test.data_adaptation(dtype)
        
    for epoch2 in range(1, p2_epoch_num + 1):
        model.train()
        optimizer.zero_grad()
            
        pred_htot = model(loaded_data_tune.meta)
        loss = mse_loss(pred_htot, loaded_data_tune.y)
            
        loss.backward()
        optimizer.step()
            
        scheduler.step()
            
        model.eval()
        with torch.no_grad():
            pred_htot_test = model(loaded_data_test.meta)
            loss_test = mse_loss(pred_htot_test, loaded_data_test.y)
            
        if epoch2 % 100 == 0:
            print(f'Phase 2 - Epoch [{epoch2}/{p2_epoch_num}], Training Loss: {loss.item():.4f}, Testing Loss: {loss_test.item():.4f}')
    
    torch.save(model.state_dict(), model_path + div + '_' + bmd_site + '_optparam_mlp_clinical_only_testing.pt')
    
    tune_bmd_dict = {bmd_site: np.array(loaded_data_tune.y.detach().cpu().numpy()).reshape(tune_num_subject)}
    tune_pred_dict = {'subject_id': pd.DataFrame.to_numpy(loaded_data_tune.subject_id).reshape(tune_num_subject),
                      'pred_' + bmd_site: np.array(pred_htot.detach().cpu().numpy()).reshape(tune_num_subject)}
    tune_pred_cache = pd.DataFrame.from_dict(tune_pred_dict)
    
    rmse_r2_dict = {}
    tune_rmse_dict = {'Tuning RMSE': np.sqrt(np.array(loss.detach().cpu().numpy()))}
    tune_r2_dict = {'Tuning R2': get_r2(tune_bmd_dict.get(bmd_site), tune_pred_dict.get('pred_' + bmd_site))}
    
    rmse_r2_dict.update(tune_rmse_dict)
    rmse_r2_dict.update(tune_r2_dict)
    
    test_bmd_dict = {bmd_site: np.array(loaded_data_test.y.detach().cpu().numpy()).reshape(test_num_subject)}
    test_pred_dict = {'subject_id': pd.DataFrame.to_numpy(loaded_data_test.subject_id).reshape(test_num_subject),
                      'pred_' + bmd_site: np.array(pred_htot_test.detach().cpu().numpy()).reshape(test_num_subject)}
    test_pred_cache = pd.DataFrame.from_dict(test_pred_dict)
    
    test_rmse_dict = {'Testing RMSE': np.sqrt(np.array(loss_test.detach().cpu().numpy()))}
    test_r2_dict = {'Testing R2': get_r2(test_bmd_dict.get(bmd_site), test_pred_dict.get('pred_' + bmd_site))}
    
    rmse_r2_dict.update(test_rmse_dict)
    rmse_r2_dict.update(test_r2_dict)
    
    if save_pred_results:
        pred_save_path = master_path + div + '/prediction_results/'
        os.makedirs(pred_save_path, exist_ok=True)
        tune_pred_cache.to_csv(pred_save_path + div + '_' + bmd_site + '_tune_set_mlp_clinical_only_pred_results.csv', index=False)
        test_pred_cache.to_csv(pred_save_path + div + '_' + bmd_site + '_test_set_mlp_clinical_only_pred_results.csv', index=False)
    
    summarized_results_dict.update(rmse_r2_dict)
    
    return summarized_results_dict


# In[9]:


root_path = 'root_path/'
master_path = root_path + 'data_folder/'
div_list = np.char.add('tune_', np.array(list(range(1, 11, 1))).astype('str')).tolist()
model_path = root_path + 'saved_models_mlp_clinical_only/'
os.makedirs(model_path, exist_ok=True)
init_model_params_dict = {'fuse_level_dim', 'dc_h_dim'}
summarized_results_path = root_path + 'summarized_results_mlp_clinical_only/'
os.makedirs(summarized_results_path, exist_ok=True)
bmd_site = 'bmd_site' #NECK_BMD, HTOT_BMD, spine_total_bmd, R_13_BMD


# In[10]:


div_track = []
summarized_results_cache = []
for div in div_list:
    print(f'Running on {div}')
    pruner = optuna.pruners.HyperbandPruner(min_resource = 20)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    objective = Objective(master_path, div, bmd_site)
    study.optimize(objective, n_trials=100)
    
    best_params_dict = study.best_trial.params
    summarized_results_dict = testMLP_clinical_only(master_path, div, bmd_site, 
                                                    best_params_dict, init_model_params_dict, model_path)
    summarized_results_cache.append(summarized_results_dict)
    div_track.append(div)

div_track_dic = {'division': div_track}
div_tract_cache = pd.DataFrame(data = div_track_dic)

summarized_results_cache = pd.DataFrame.from_dict(summarized_results_cache)
    
summarized_results_cache = pd.concat([div_tract_cache, summarized_results_cache], axis = 1)
summarized_results_cache.to_csv(summarized_results_path + bmd_site + '_mlp_clinical_only_summarized_result.csv', index=False)


# In[ ]:




