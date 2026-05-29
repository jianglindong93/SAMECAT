#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from ete3 import Tree
from torch.utils.data import Dataset
from captum.module import (BinaryConcreteStochasticGates,
                           GaussianStochasticGates)
import torch.nn.utils.prune as prune

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


"""
Codes related with MIOSTONE network were adapted from https://github.com/batmen-lab/MIOSTONE.

Modifications were made to better accommodate our analyses.
"""

class MIOSTONETree:
    """
    Attributes:
        ete_tree (ete3.Tree): An ete3 Tree instance.
        depths (dict): A dictionary mapping feature names to their depths in the tree.
        max_depth (int): The maximum depth of the tree.
    """

    def __init__(self, ete_tree):
        self.ete_tree = ete_tree
        self.depths = {}
        self.max_depth = 0
        self.taxonomic_ranks = [
            "Kingdom", "Phylum", "Class", "Order",
            "Family", "Genus", "Species"
        ]

    @classmethod
    def init_from_nwk(cls, nwk_file):
        """
        Initialize from a Newick file.
        """
        # ete3 will detect it's a file path
        t = Tree(nwk_file, format=1)
        t.name = "root"
        for node in t.traverse():
            # set branch length of root to zero
            if node.is_root():
                node.dist = 0.0
            else:
                node.dist = 1.0
        return cls(t)

    def prune(self, features):
        leaves = set(self.ete_tree.get_leaves())
        while any(leaf.name not in features for leaf in leaves):
            for leaf in leaves:
                if leaf.name not in features:
                    leaf.delete(prevent_nondicotomic=False)
            leaves = set(self.ete_tree.get_leaves())

    def compute_depths(self):
        """
        Populate self.depths[node.name] = depth, and self.max_depth.
        """
        for node in self.ete_tree.traverse("levelorder"):
            if node.is_root():
                self.depths[node.name] = 0
            else:
                self.depths[node.name] = self.depths[node.up.name] + 1
            self.max_depth = max(self.max_depth, self.depths[node.name])

    def compute_indices(self):
        """
        Assign each node an index within its depth level.
        """
        self.indices = {}
        curr_depth = 0
        curr_id = 0
        for node in self.ete_tree.traverse("levelorder"):
            d = self.depths[node.name]
            if d > curr_depth:
                curr_depth = d
                curr_id = 0
            self.indices[node.name] = curr_id
            curr_id += 1


# In[4]:


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
    def init_from_files(cls, master_path, div_type, bmd_site, use_mask = False, mask_path = None, mask_name = None):
        subject_id = pd.read_csv(master_path + 'subject_id_' + div_type + '.csv')
        data = pd.read_csv(master_path + 'microbe_comp_' + div_type + ".csv")
        if use_mask:
            selected_microbes = pd.read_csv(mask_path + 'microbe_names_' + mask_name + '.csv')
            selected_microbes = selected_microbes['species'].to_list()
            data = data[selected_microbes]
        meta = pd.read_csv(master_path + 'clinical_var_' + div_type + '.csv')
        bmd_data = pd.read_csv(master_path + 'bmd_' + div_type + '.csv')
        y = bmd_data[[bmd_site]]
        features = ['s__' + col.split('.s__')[-1] for col in data.columns]
        
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

    def order_features_by_tree(self, tree: MIOSTONETree):
        leaf_names = tree.ete_tree.get_leaf_names()
        idxs = [self.features.index(n) for n in leaf_names]
        self.X = self.X[:, idxs]
        self.features = np.array(leaf_names)
    
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


# In[14]:


def rmse_loss(pred_x, x):
    batch_size = x.size(0)
    assert batch_size != 0
    mse_loss_val = F.mse_loss(pred_x, x, reduction='sum').div(batch_size)
    rmse_loss_val = torch.sqrt(mse_loss_val)
    
    return rmse_loss_val

def get_r2(x, pred_x):
    r, _ = pearsonr(x, pred_x)
    r2 = r**2
    
    return r2


# In[6]:


class MIOSTONELayer(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 gate_type, 
                 gate_param,
                 connections,
                 prune_mode):
        super(MIOSTONELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_type = gate_type
        self.gate_param = gate_param
        self.connections = connections
        self.prune_mode = prune_mode
        self.x_linear = None
        self.l0_reg = None

        # Initialize the layer
        self._init_layer()

    def _init_layer(self):
        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.LeakyReLU()
        )
        # Linear layer
        self.linear = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
        )

        # Gate layer
        if self.gate_type == "concrete":
            self.gate_mask = self._generate_gate_mask()
            self.gate_layer = BinaryConcreteStochasticGates(n_gates=len(self.connections),
                                                           mask=self.gate_mask,
                                                           temperature=self.gate_param)
        elif self.gate_type == "gaussian":
            self.gate_mask = self._generate_gate_mask()
            self.gate_layer = GaussianStochasticGates(n_gates=len(self.connections),
                                                        mask=self.gate_mask,
                                                        std=self.gate_param)
            
        # Prune the network based on the connections
        self._apply_pruning()

    def _generate_gate_mask(self):
        mask = torch.zeros(self.out_features, dtype=torch.int64)
        value = 0
        for _, output_indices in self.connections.values():
            for output_index in output_indices:
                mask[output_index] = value
            value += 1

        return mask

    def _apply_pruning(self):
        # If the prune mode is random, generate random connections
        if self.prune_mode == "random":
            self._generate_random_connections()
        # Define a custom prune method for each layer
        prune.custom_from_mask(self.mlp[0], name='weight', mask=self._generate_pruning_mask())
        prune.custom_from_mask(self.linear[0], name='weight', mask=self._generate_pruning_mask())
        # Remove the original weight parameter
        prune.remove(self.mlp[0], 'weight')
        prune.remove(self.linear[0], 'weight')

    def _generate_random_connections(self):
        connections = {}
        all_input_indices = [mapping[0] for mapping in self.connections.values()]
        for ete_node, (_, output_indices) in self.connections.items():
            idx = random.randint(0, len(all_input_indices) - 1)
            input_indices = all_input_indices[idx]
            connections[ete_node] = (input_indices, output_indices)
            all_input_indices = all_input_indices[:idx] + all_input_indices[idx + 1:]

        self.connections = connections

    def _generate_pruning_mask(self):
        # Start with a mask of all zeros (all connections pruned)
        mask = torch.zeros((self.out_features, self.in_features), dtype=torch.int64)

        # Iterate over the connections at the current depth and set the corresponding elements in the mask to 1
        for input_indices, output_indices in self.connections.values():
            for input_index in input_indices:
                for output_index in output_indices:
                    mask[output_index, input_index] = 1

        return mask

    def forward(self, x, x_linear):
        # Apply the MLP layer
        x_mlp = self.mlp(x)

        # Apply the linear layer
        self.x_linear = self.linear(x_linear)
        
        # Apply the linear layer with the gate values
        if self.gate_type == "deterministic":
            gate_values = self.gate_param
            self.l0_reg = torch.tensor(0.0).to(x.device)
        else:
            input_size = x_mlp.size()
            batch_size = input_size[0]

            gate_values = self.gate_layer._sample_gate_values(batch_size)

            # hard-sigmoid rectification z=min(1,max(0,_z))
            gate_values = torch.clamp(gate_values, min=0, max=1)

            # use expand_as not expand/broadcast_to which do not work with torch.fx
            input_mask = self.gate_layer.mask.expand_as(x_mlp)

            # flatten all dim except batch to gather from gate values
            flattened_mask = input_mask.reshape(batch_size, -1)
            gate_values = torch.gather(gate_values, 1, flattened_mask)

            # reshape gates(batch_size, n_elements) into input_size for point-wise mul
            gate_values = gate_values.reshape(input_size)

            prob_density = self.gate_layer._get_gate_active_probs()
            if self.gate_layer.reg_reduction == "sum":
                l0_reg = prob_density.sum()
            elif self.gate_layer.reg_reduction == "mean":
                l0_reg = prob_density.mean()
            else:
                l0_reg = prob_density

            l0_reg *= self.gate_layer.reg_weight
            self.l0_reg = l0_reg

        # Apply the gate values
        x_mlp_gated = gate_values * x_mlp
        x_linear_gated = (1 - gate_values) * self.x_linear

        x_gated = x_mlp_gated + x_linear_gated

        return x_gated
    
class MIOSTONEModel(nn.Module):
    def __init__(self, 
                 tree,
                 out_features,
                 node_min_dim,
                 node_dim_func,
                 node_dim_func_param, 
                 node_gate_type,
                 node_gate_param,
                 prune_mode):
        super(MIOSTONEModel, self).__init__()
        self.out_features = out_features
        self.node_min_dim = node_min_dim
        self.node_dim_func = node_dim_func
        self.node_dim_func_param = node_dim_func_param
        self.node_gate_type = node_gate_type
        self.node_gate_param = node_gate_param
        self.prune_mode = prune_mode
        self.hidden_layers = None
        self.output_layer = None
        self.total_l0_reg = None

        # Initialize the architecture based on the tree
        connections, layer_dims = self._init_architecture(tree)

        # Build the model based on the architecture
        self._build_model(connections, layer_dims)

    def _init_architecture(self, tree):
        # Define the node dimension function
        def dim_func(x, node_dim_func, node_dim_func_param, depth):
            if node_dim_func == "linear":
                coeff = node_dim_func_param ** (tree.max_depth - depth)
                return int(coeff * x)
            elif node_dim_func == "const":
                return int(node_dim_func_param)

        # Initialize dictionary for connections and layer dimensions
        layer_connections = [{} for _ in range(tree.max_depth + 1)]
        layer_dims = [None for _ in range(tree.max_depth + 1)]

        curr_index = 0
        curr_depth = tree.max_depth
        prev_layer_out_features = 0
        for ete_node in reversed(list(tree.ete_tree.traverse("levelorder"))):
            node_depth = tree.depths[ete_node.name]
            if node_depth != curr_depth:
                layer_dims[curr_depth] = (prev_layer_out_features, curr_index)
                curr_depth = node_depth
                prev_layer_out_features = curr_index
                curr_index = 0

            if ete_node.is_leaf():
                layer_connections[curr_depth][ete_node.name] = ([], [curr_index])
                curr_index += 1
                continue

            children = ete_node.get_children()

            # Calculate input indices
            input_indices = []
            for child in children:
                child_output_indices = layer_connections[node_depth + 1][child.name][1]
                input_indices.extend(child_output_indices)

            # Calculate output dimensions and indices
            node_out_features = max(self.node_min_dim, 
                                    dim_func(self.node_min_dim * len(list(ete_node.get_leaves())),
                                            self.node_dim_func, 
                                            self.node_dim_func_param, 
                                            node_depth))
            output_indices = list(range(curr_index, curr_index + node_out_features))
            curr_index += node_out_features

            # Store in connections
            layer_connections[curr_depth][ete_node.name] = (input_indices, output_indices)

        # Append the dimension of the last layer
        layer_dims[0] = (prev_layer_out_features, curr_index)

        # Remove the layer dimension of the leaf nodes
        layer_dims = layer_dims[:-1]

        return layer_connections, layer_dims

    def _build_model(self, layer_connections, layer_dims):
        # Initialize the hidden layers
        self.hidden_layers = nn.ModuleList()
        for depth, (in_features, out_features) in enumerate(layer_dims):
            # Get the connections for the current layer
            connections = layer_connections[depth]

            # Initialize the layer
            layer = MIOSTONELayer(in_features, 
                                  out_features, 
                                  self.node_gate_type, 
                                  self.node_gate_param, 
                                  connections,
                                  prune_mode=self.prune_mode)
            self.hidden_layers.append(layer)
            
        # Initialize the output layer
        output_layer_in_features = layer_dims[0][1] 
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(output_layer_in_features),
            nn.Linear(output_layer_in_features, self.out_features),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        # Initialize the total l0 regularization
        self.total_l0_reg = torch.tensor(0.0).to(x.device)

        # Initialize the linear layer input
        x_linear = x

        # Iterate over the layers
        for layer in reversed(self.hidden_layers):
            # Apply the layer
            x = layer(x, x_linear)

            # Update the linear layer input
            x_linear = layer.x_linear
            layer.x_linear = None

            # Update the total l0 regularization
            self.total_l0_reg += layer.l0_reg
            layer.l0_reg = None

        # Apply the output layer
        x = self.output_layer(x)

        return x
    
    def get_total_l0_reg(self):
        return self.total_l0_reg

class TaxoConcat(nn.Module):
    def __init__(self, 
                 tree,
                 mra_out_dim,
                 node_min_dim,
                 node_dim_func,
                 node_dim_func_param, 
                 node_gate_type,
                 node_gate_param,
                 prune_mode, 
                 input_vcovar_n1, 
                 #vcovar_level1_dim, 
                 vcovar_hidden_dim, 
                 dc_h_dim):
        super(TaxoConcat, self).__init__()
        
        ## balance --> hidden feature construction
        # metagenome htot encoder
        self.mra_encoder = MIOSTONEModel(tree, 
                                         mra_out_dim,
                                         node_min_dim,
                                         node_dim_func,
                                         node_dim_func_param, 
                                         node_gate_type,
                                         node_gate_param,
                                         prune_mode)
        
        # clinical encoder
        self.vcovar_encoder = nn.Sequential(
            nn.Linear(input_vcovar_n1, vcovar_hidden_dim),
            #nn.BatchNorm1d(fuse_level_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        ## BMD prediction heads
        # htot_bmd prediction
        self.decoder = nn.Sequential(
            nn.Linear(mra_out_dim + vcovar_hidden_dim, dc_h_dim),
            #nn.BatchNorm1d(dc_h_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(dc_h_dim, 1)
        )
        
    def forward(self, mgs_data, clinical_data):
        mgs_code = self.mra_encoder(mgs_data)
        vcovar_code = self.vcovar_encoder(clinical_data)
        concat_code = torch.cat((mgs_code, vcovar_code), dim = 1)
        
        pred_htot = self.decoder(concat_code)
        
        return (pred_htot, self.mra_encoder.get_total_l0_reg())


# In[15]:


class Objective:
    def __init__(self, tree_path, master_path, div, bmd_site, 
                 use_mask = False, mask_name = None, dtype = torch.float64):
        self.tree_path = tree_path
        self.master_path = master_path
        self.div = div
        self.bmd_site = bmd_site
        self.use_mask = use_mask
        self.mask_name = mask_name
        self.dtype = dtype
        
    def __call__(self, trial):
        # Hyperparameter suggestions
        learning_rate2 = trial.suggest_float('learning_rate2', 1e-5, 1e-1, log=True)
        l2 = trial.suggest_float('l2', 1e-3, 1, log=True)
        p2_epoch_num = trial.suggest_int('p2_epoch_num', 100, 1000, step = 100)
        lambda_2 = trial.suggest_float('lambda_2', 1e-3, 2e-1, log=True)
        
        # Data loader
        if self.use_mask:
            miostone_tree = MIOSTONETree.init_from_nwk(self.tree_path + 'taxa_tree_' + self.mask_name + '.nwk')
        else:
            miostone_tree = MIOSTONETree.init_from_nwk(self.tree_path + 'taxa_tree.nwk')
        miostone_tree.compute_depths()
        miostone_tree.compute_indices()
        
        loaded_data_train = MIOSTONEDataset.init_from_files(self.master_path + self.div + '/', 
                                                            'tr_' + self.div,
                                                            self.bmd_site,
                                                            use_mask = self.use_mask,
                                                            mask_path = self.tree_path,
                                                            mask_name = self.mask_name)
        loaded_data_train.normalize()
        loaded_data_train.clr_transform()
        loaded_data_train.order_features_by_tree(miostone_tree)
        
        loaded_data_valid = MIOSTONEDataset.init_from_files(self.master_path + self.div + '/', 
                                                            'val_' + self.div, 
                                                            self.bmd_site,
                                                            use_mask = self.use_mask,
                                                            mask_path = self.tree_path,
                                                            mask_name = self.mask_name)
        loaded_data_valid.normalize()
        loaded_data_valid.clr_transform()
        loaded_data_valid.order_features_by_tree(miostone_tree)
        
        input_vcovar_n1 = loaded_data_train.meta.shape[1]
        # Model, loss function, optimization
        model = TaxoConcat(tree = miostone_tree,
                           mra_out_dim = trial.suggest_int('mra_out_dim', 2, 16, step = 2),
                           node_min_dim = 1,
                           node_dim_func = 'linear',
                           node_dim_func_param = 0.6, 
                           node_gate_type = 'concrete',
                           node_gate_param = 0.3,
                           prune_mode = 'taxonomy', 
                           input_vcovar_n1 = input_vcovar_n1, 
                           vcovar_hidden_dim = trial.suggest_int('vcovar_hidden_dim', 2, 16, step = 2), 
                           dc_h_dim = trial.suggest_int('dc_h_dim', 2, 8, step = 2))
        model = model.to(dtype=torch.float64, device=DEVICE)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate2, weight_decay = l2)
        scheduler = CosineAnnealingLR(optimizer, T_max=p2_epoch_num)
        
        loaded_data_train.data_adaptation(self.dtype)
        loaded_data_valid.data_adaptation(self.dtype)
        
        for epoch2 in range(1, p2_epoch_num + 1):
            model.train()
            optimizer.zero_grad()
            
            pred_htot, total_l0_reg = model(loaded_data_train.X, loaded_data_train.meta)
            pred_loss = rmse_loss(pred_htot, loaded_data_train.y)
            loss = pred_loss + lambda_2*total_l0_reg
            loss_train = loss.item()
            
            loss.backward()
            optimizer.step()
            
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                pred_htot_valid, _ = model(loaded_data_valid.X, loaded_data_valid.meta)
                loss_valid = rmse_loss(pred_htot_valid, loaded_data_valid.y)
                
            trial.report(loss_valid, epoch2)
            if trial.should_prune():
                print(f'Trial {trial.number} pruned at phase 2 epoch {epoch2}.')
                raise optuna.exceptions.TrialPruned()
            
            if epoch2 % 100 == 0:
                print(f'{self.div} - Epoch [{epoch2}/{p2_epoch_num}], Oervall training loss: {loss_train:.4f}, Training Loss: {pred_loss.item():.4f}, Validation Loss: {loss_valid.item():.4f}')
        
        return loss_valid


# In[16]:


def testTaxoConcat(tree_path, master_path, div, bmd_site, 
                   best_params_dict, init_model_params_dict, model_path, 
                   use_mask = False, mask_name = None, 
                    dtype = torch.float64, save_pred_results = True):
    summarized_results_dict = {}
    summarized_results_dict.update(best_params_dict)
    
    if use_mask:
        miostone_tree = MIOSTONETree.init_from_nwk(tree_path + 'taxa_tree_' + mask_name + '.nwk')
    else:
        miostone_tree = MIOSTONETree.init_from_nwk(tree_path + 'taxa_tree.nwk')
    miostone_tree.compute_depths()
    miostone_tree.compute_indices()
    loaded_data_tune = MIOSTONEDataset.init_from_files(master_path + 'train_test_split/', 
                                                       'tu',
                                                       bmd_site,
                                                       use_mask = use_mask,
                                                       mask_path = tree_path,
                                                       mask_name = mask_name)
    loaded_data_tune.normalize()
    loaded_data_tune.clr_transform()
    loaded_data_tune.order_features_by_tree(miostone_tree)
        
    loaded_data_test = MIOSTONEDataset.init_from_files(master_path + 'train_test_split/', 
                                                       'te', 
                                                       bmd_site,
                                                       use_mask = use_mask,
                                                       mask_path = tree_path,
                                                       mask_name = mask_name)
    loaded_data_test.normalize()
    loaded_data_test.clr_transform()
    loaded_data_test.order_features_by_tree(miostone_tree)
    
    tune_num_subject = loaded_data_tune.X.shape[0]
    test_num_subject = loaded_data_test.X.shape[0]
    input_vcovar_n1 = loaded_data_tune.meta.shape[1]
    
    p2_epoch_num = best_params_dict['p2_epoch_num']
    learning_rate2 = best_params_dict['learning_rate2']
    l2 = best_params_dict['l2']
    lambda_2 = best_params_dict['lambda_2']
    
    init_model_params = {key: best_params_dict[key] for key in init_model_params_dict if key in best_params_dict}
    
    model = TaxoConcat(tree = miostone_tree,
                       node_min_dim = 1,
                       node_dim_func = 'linear',
                       node_dim_func_param = 0.6, 
                       node_gate_type = 'concrete',
                       node_gate_param = 0.3,
                       prune_mode = 'taxonomy', 
                       input_vcovar_n1 = input_vcovar_n1,
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
            
        pred_htot, total_l0_reg = model(loaded_data_tune.X, loaded_data_tune.meta)
        pred_loss = rmse_loss(pred_htot, loaded_data_tune.y)
        loss = pred_loss + lambda_2*total_l0_reg
        loss_tune = loss.item()
            
        loss.backward()
        optimizer.step()
            
        scheduler.step()
            
        model.eval()
        with torch.no_grad():
            pred_htot_test, _ = model(loaded_data_test.X, loaded_data_test.meta)
            loss_test = rmse_loss(pred_htot_test, loaded_data_test.y)
            
        if epoch2 % 100 == 0:
            print(f'{div} Testing Stage - Epoch [{epoch2}/{p2_epoch_num}], Oervall training loss: {loss_tune:.4f}, Training Loss: {pred_loss.item():.4f}, Testing Loss: {loss_test.item():.4f}')
    
    torch.save(model.state_dict(), model_path + div + '_' + bmd_site + '_optparam_taxoconcat_testing.pt')
    
    tune_bmd_dict = {bmd_site: np.array(loaded_data_tune.y.detach().cpu().numpy()).reshape(tune_num_subject)}
    tune_pred_dict = {'subject_id': pd.DataFrame.to_numpy(loaded_data_tune.subject_id).reshape(tune_num_subject),
                      'pred_' + bmd_site: np.array(pred_htot.detach().cpu().numpy()).reshape(tune_num_subject)}
    tune_pred_cache = pd.DataFrame.from_dict(tune_pred_dict)
    
    rmse_r2_dict = {}
    tune_rmse_dict = {'Tuning RMSE': np.array(pred_loss.detach().cpu().numpy())}
    tune_r2_dict = {'Tuning R2': get_r2(tune_bmd_dict.get(bmd_site), tune_pred_dict.get('pred_' + bmd_site))}
    
    rmse_r2_dict.update(tune_rmse_dict)
    rmse_r2_dict.update(tune_r2_dict)
    
    test_bmd_dict = {bmd_site: np.array(loaded_data_test.y.detach().cpu().numpy()).reshape(test_num_subject)}
    test_pred_dict = {'subject_id': pd.DataFrame.to_numpy(loaded_data_test.subject_id).reshape(test_num_subject),
                      'pred_' + bmd_site: np.array(pred_htot_test.detach().cpu().numpy()).reshape(test_num_subject)}
    test_pred_cache = pd.DataFrame.from_dict(test_pred_dict)
    
    test_rmse_dict = {'Testing RMSE': np.array(loss_test.detach().cpu().numpy())}
    test_r2_dict = {'Testing R2': get_r2(test_bmd_dict.get(bmd_site), test_pred_dict.get('pred_' + bmd_site))}
    
    rmse_r2_dict.update(test_rmse_dict)
    rmse_r2_dict.update(test_r2_dict)
    
    if save_pred_results:
        pred_save_path = master_path + div + '/prediction_results/'
        os.makedirs(pred_save_path, exist_ok=True)
        tune_pred_cache.to_csv(pred_save_path + div + '_' + bmd_site + '_tune_set_taxoconcat_pred_results.csv', index=False)
        test_pred_cache.to_csv(pred_save_path + div + '_' + bmd_site + '_test_set_taxoconcat_pred_results.csv', index=False)
    
    summarized_results_dict.update(rmse_r2_dict)
    
    return summarized_results_dict


# In[22]:


root_path = 'root_path/'
master_path = root_path + 'data_folder/'
tree_path = master_path + 'tree_folder/'
div_list = np.char.add('tune_', np.array(list(range(1, 11, 1))).astype('str')).tolist()
model_path = root_path + 'saved_models_taxoconcat/'
os.makedirs(model_path, exist_ok=True)
init_model_params_dict = {'mra_out_dim', 'vcovar_hidden_dim', 'dc_h_dim'}
summarized_results_path = root_path + 'summarized_results_taxoconcat/'
os.makedirs(summarized_results_path, exist_ok=True)
bmd_site = 'bmd_site' #NECK_BMD, HTOT_BMD, spine_total_bmd, R_13_BMD
use_mask = True
mask_name = 'mask_name'


# In[23]:


div_track = []
summarized_results_cache = []
for div in div_list:
    #print(f'Running on {div}')
    pruner = optuna.pruners.HyperbandPruner(min_resource = 20)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    objective = Objective(tree_path, master_path, div, bmd_site, use_mask = use_mask, mask_name = mask_name)
    study.optimize(objective, n_trials=100)
    
    best_params_dict = study.best_trial.params
    summarized_results_dict = testTaxoConcat(tree_path, master_path, div, bmd_site, 
                                              best_params_dict, init_model_params_dict, model_path,
                                              use_mask = use_mask, mask_name = mask_name)
    summarized_results_cache.append(summarized_results_dict)
    div_track.append(div)

div_track_dic = {'division': div_track}
div_tract_cache = pd.DataFrame(data = div_track_dic)

summarized_results_cache = pd.DataFrame.from_dict(summarized_results_cache)
    
summarized_results_cache = pd.concat([div_tract_cache, summarized_results_cache], axis = 1)
summarized_results_cache.to_csv(summarized_results_path + bmd_site + '_taxoconcat_summarized_results.csv', index=False)


# In[ ]:




