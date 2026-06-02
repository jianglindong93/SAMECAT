#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from captum.attr import GradientShap
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.functional import relu
import os
from copy import deepcopy
from scipy.stats import pearsonr
from ete3 import Tree
from torch.utils.data import Dataset
from captum.module import (BinaryConcreteStochasticGates,
                           GaussianStochasticGates)
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
import shapmat
import seaborn as sns
import shap

EPSILON = 1e-9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG_MODE = False


# In[2]:


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


# In[3]:


class MIOSTONEDataset(Dataset):
    """
    Handles Metagenomic relative abudance data + all preprocessing.
    """
    
    def __init__(self, subject_id, data, X, meta_df, meta, y, features):
        self.subject_id = subject_id
        self.X_df = data
        self.X = X
        self.meta_df = meta_df
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
        meta_df = pd.read_csv(master_path + 'clinical_var_' + div_type + '.csv')
        bmd_data = pd.read_csv(master_path + 'bmd_' + div_type + '.csv')
        y = bmd_data[[bmd_site]]
        features = ['s__' + col.split('.s__')[-1] for col in data.columns]
        
        #X = data.values.astype(np.float32)
        X = data.values
        meta = meta_df.values
        y = y.values
        
        return cls(subject_id, data, X, meta_df, meta, y, features)
    
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


# In[6]:


class _Fusion(nn.Module):
    def __init__(self):
        """
        Base class for the fusion module
        """
        super().__init__()

    def forward(self, inputs):
        raise NotImplementedError()

    def get_weights(self, softmax=True):
        out = []
        if hasattr(self, "weights"):
            out = self.weights
            if softmax:
                out = F.softmax(self.weights, dim=-1)
        return out

    def update_weights(self, inputs, a):
        pass


class Mean(_Fusion):
    def __init__(self):
        """
        Mean fusion.
        """
        super().__init__()

    def forward(self, inputs):
        return torch.mean(torch.stack(inputs, -1), dim=-1)


class WeightedMean(_Fusion):
    """
    Weighted mean fusion.
    """
    def __init__(self, n_views):
        super().__init__()
        self.weights = nn.Parameter(torch.full((n_views,), 1 / n_views), requires_grad=True)

    def forward(self, inputs):
        return _weighted_sum(inputs, self.weights, normalize_weights=True)


def _weighted_sum(tensors, weights, normalize_weights=True):
    if normalize_weights:
        weights = F.softmax(weights, dim=0)
    out = torch.sum(weights[None, None, :] * torch.stack(tensors, dim=-1), dim=-1)
    return out


MODULES = {
    "mean": Mean,
    "weighted_mean": WeightedMean,
}


def get_fusion_module(method):
    return MODULES[method]()


# In[7]:


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


# In[8]:


class BMDWrapper(nn.Module):
    """
    Thin wrapper: (mgs_data, clinical_data) -> pred_htot
    so Captum can work with a simple forward.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, mgs_data, clinical_data):
        # base_model returns: projections, hidden, output, pred_htot, l0_reg
        _, _, _, pred_htot, _ = self.base_model(mgs_data, clinical_data)
        # EG works fine with shape (batch, 1) or (batch,)
        return pred_htot.squeeze(-1)


# In[9]:


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

class SAMECAT(nn.Module):
    def __init__(self, 
                 tree,
                 node_min_dim,
                 node_dim_func,
                 node_dim_func_param, 
                 node_gate_type,
                 node_gate_param,
                 prune_mode, 
                 input_vcovar_n1, 
                 #vcovar_level1_dim, 
                 fuse_level_dim, 
                 level_hidden_dim, 
                 dc_h_dim, 
                 n_views, 
                 n_clusters):
        super(SAMECAT, self).__init__()
        
        self.n_clusters = n_clusters
        
        ## balance --> hidden feature construction
        # metagenome htot encoder
        self.mra_encoder = MIOSTONEModel(tree, 
                                         fuse_level_dim,
                                         node_min_dim,
                                         node_dim_func,
                                         node_dim_func_param, 
                                         node_gate_type,
                                         node_gate_param,
                                         prune_mode)
        
        # clinical encoder
        self.vcovar_encoder = nn.Sequential(
            nn.Linear(input_vcovar_n1, fuse_level_dim),
            #nn.BatchNorm1d(fuse_level_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        ## fusing different views
        self.fusion = WeightedMean(n_views)
        
        ## clustering
        self.hidden_projector = nn.Sequential(
            nn.Linear(fuse_level_dim, level_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.cluster = nn.Sequential(
            nn.Linear(level_hidden_dim, n_clusters),
            nn.Softmax(dim=1)
        )
        
        ## BMD prediction heads
        # htot_bmd prediction
        self.decoder = nn.Sequential(
            nn.Linear(fuse_level_dim, dc_h_dim),
            #nn.BatchNorm1d(dc_h_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(dc_h_dim, 1)
        )
        
    def forward(self, mgs_data, clinical_data):
        mgs_code = self.mra_encoder(mgs_data)
        vcovar_code = self.vcovar_encoder(clinical_data)
        
        fused_code = self.fusion([mgs_code, vcovar_code])
        projections = torch.cat((mgs_code, vcovar_code), dim = 0)
        hidden = self.hidden_projector(fused_code)
        output = self.cluster(hidden)
        
        pred_htot = self.decoder(fused_code)
        
        return(projections, hidden, output, pred_htot, self.mra_encoder.get_total_l0_reg())


# In[10]:


def compute_gs_batched(gs, wrapped_model, X, C, baseline_mgs, baseline_cov,
                       n_samples_eg=50, batch_size=64):
    n = X.shape[0]
    all_attr_mgs = []
    all_attr_cov = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end]
        C_batch = C[start:end]

        wrapped_model.zero_grad()

        attr_mgs_b, attr_cov_b = gs.attribute(
            inputs=(X_batch, C_batch),
            baselines=(baseline_mgs, baseline_cov),
            n_samples=n_samples_eg,
            stdevs=0.0,
            target=None,
            return_convergence_delta=False,
        )

        all_attr_mgs.append(attr_mgs_b.detach().cpu())
        all_attr_cov.append(attr_cov_b.detach().cpu())

    attr_mgs = torch.cat(all_attr_mgs, dim=0)
    attr_cov = torch.cat(all_attr_cov, dim=0)
    return attr_mgs, attr_cov


# In[11]:


def get_interpretation_results(bmd_site, root_path='root_path/', 
                               n_baselines=100, n_samples_eg=50, batch_size=64,
                               use_mask=True, mask_name='wozeroonlyspecies',
                               save_results=True, print_results=True):
    master_path = root_path + 'data_folder/'
    tree_path = master_path + 'tree_folder/'
    model_path = root_path + 'model_folder/'
    summarized_results_path = root_path + 'summarized_results_samecat/'
    interpretation_results_path = root_path + 'EG_interpretation_results/'
    os.makedirs(interpretation_results_path, exist_ok=True)
    
    summarized_results = pd.read_csv(summarized_results_path + bmd_site + '_samecat_summarized_results.csv')
    opt_idx = summarized_results['Testing RMSE'].idxmin()

    if use_mask:
        miostone_tree = MIOSTONETree.init_from_nwk(tree_path + 'taxa_tree_' + mask_name + '.nwk')
    else:
        miostone_tree = MIOSTONETree.init_from_nwk(tree_path + 'taxa_tree.nwk')
    miostone_tree.compute_depths()
    miostone_tree.compute_indices()

    dtype = torch.float64  # same as training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct datasets like you did in training / testing
    loaded_data_train = MIOSTONEDataset.init_from_files(
        master_path + 'train_test_split/', 'tu', bmd_site,
        use_mask=use_mask, mask_path=tree_path, mask_name=mask_name
    )
    loaded_data_train.normalize()
    loaded_data_train.clr_transform()
    loaded_data_train.order_features_by_tree(miostone_tree)
    loaded_data_train.data_adaptation(dtype)

    loaded_data_test = MIOSTONEDataset.init_from_files(
        master_path + 'train_test_split/', 'te', bmd_site,
        use_mask=use_mask, mask_path=tree_path, mask_name=mask_name
    )
    loaded_data_test.normalize()
    loaded_data_test.clr_transform()
    loaded_data_test.order_features_by_tree(miostone_tree)
    loaded_data_test.data_adaptation(dtype)

    X_train = loaded_data_train.X          # (N_train, n_mgs)
    C_train = loaded_data_train.meta       # (N_train, n_covars)
    y_train = loaded_data_train.y

    X_test = loaded_data_test.X            # (N_test, n_mgs)
    C_test = loaded_data_test.meta         # (N_test, n_covars)
    y_test = loaded_data_test.y

    # Number of baseline samples to approximate p(x')
    # n_baselines = 100
    n_train = X_train.shape[0]
    # Randomly sample training points as baselines
    baseline_idx = torch.randint(
        low=0, high=n_train, size=(n_baselines,), device=X_train.device
    )

    baseline_mgs = X_train[baseline_idx]      # (n_baselines, n_mgs)
    baseline_cov = C_train[baseline_idx]      # (n_baselines, n_covars)

    # n_samples_eg = 50  # EG-like samples per input
    # batch_size = 64    # how many TEST samples per outer batch

    input_vcovar_n1 = C_train.shape[1]
    target_div = summarized_results.loc[opt_idx, 'division']
    fuse_level_dim = summarized_results.loc[opt_idx, 'fuse_level_dim']
    level_hidden_dim = summarized_results.loc[opt_idx, 'level_hidden_dim']
    dc_h_dim = summarized_results.loc[opt_idx, 'dc_h_dim']
    n_clusters = summarized_results.loc[opt_idx, 'n_clusters']
    
    model = SAMECAT(
        tree=miostone_tree,
        node_min_dim=1,
        node_dim_func='linear',
        node_dim_func_param=0.6,
        node_gate_type='concrete',
        node_gate_param=0.3,
        prune_mode='taxonomy',
        input_vcovar_n1=input_vcovar_n1,
        fuse_level_dim=fuse_level_dim,
        level_hidden_dim=level_hidden_dim,
        dc_h_dim=dc_h_dim,
        n_views=2,
        n_clusters=n_clusters
    ).to(dtype=dtype, device=device)

    state_dict = torch.load(model_path + target_div + '_' + bmd_site + '_optparam_samecat_testing.pt', map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    wrapped_model = BMDWrapper(model).to(device=device, dtype=dtype)
    wrapped_model.eval()
    
    gs = GradientShap(wrapped_model)
    attr_mgs_test, attr_cov_test = compute_gs_batched(
        gs=gs,
        wrapped_model=wrapped_model,
        X=X_test,
        C=C_test,
        baseline_mgs=baseline_mgs,
        baseline_cov=baseline_cov,
        n_samples_eg=n_samples_eg,
        batch_size=batch_size
    )
    attr_mgs_test_np = attr_mgs_test.detach().cpu().numpy()
    attr_cov_test_np = attr_cov_test.detach().cpu().numpy()

    #Microbiome features
    mgs_importance = np.mean(np.abs(attr_mgs_test_np), axis=0)  # (n_mgs,)
    #mgs_feature_names = loaded_data_test.features               # from your dataset
    #attr_mgs_test_df = pd.DataFrame(data=attr_mgs_test_np, columns=mgs_feature_names)
    mgs_importance_df = pd.DataFrame({
        #"feature": mgs_feature_names,
        "EG_importance": mgs_importance
    }).sort_values("EG_importance", ascending=False)
    if save_results:
        attr_mgs_test_df.to_csv(interpretation_results_path + bmd_site + '_' + target_div + '_EG_attr_mgs_test_df.csv', index=False)
        mgs_importance_df.to_csv(interpretation_results_path + bmd_site + '_' + target_div + '_EG_imp_mgs_test.csv', index=False)

    # Clinical covariates (no explicit names stored, so create placeholders or reuse your CSV cols)
    #n_covars = C_test.shape[1]
    #covar_names = [f"covar_{i}" for i in range(n_covars)]  # or actual column names if you have them
    cov_importance = np.mean(np.abs(attr_cov_test_np), axis=0)
    covar_names = loaded_data_test.meta_df.columns.to_list()
    attr_cov_test_df = pd.DataFrame(data=attr_cov_test_np, columns=covar_names)
    cov_importance_df = pd.DataFrame({
        "feature": covar_names,
        "EG_importance": cov_importance
    }).sort_values("EG_importance", ascending=False)
    if save_results:
        attr_cov_test_df.to_csv(interpretation_results_path + bmd_site + '_' + target_div + '_EG_attr_cov_test_df.csv', index=False)
        cov_importance_df.to_csv(interpretation_results_path + bmd_site + '_' + target_div + '_EG_imp_cov_test.csv', index=False)

    # Quick look
    if print_results:
        print(mgs_importance_df.head(15))
        print(cov_importance_df.head(20))
    
    return loaded_data_train, loaded_data_test, attr_mgs_test_np, attr_cov_test_np


# In[12]:

# Total Hip
htot_loaded_data_train, htot_loaded_data_test, htot_attr_mgs_test_np, htot_attr_cov_test_np = get_interpretation_results(bmd_site='HTOT_BMD')


# In[13]:


plt.figure(figsize=(8,6),dpi=80)
htot_mgs_features = htot_loaded_data_test.X_df.copy()
htot_mgs_features.columns = htot_loaded_data_test.features
shap.summary_plot(shap_values=htot_attr_mgs_test_np,
                  features=htot_mgs_features,
                  show=False, 
                  plot_size='auto',
                  max_display=15)
# max_display: number of features to display
    
plt.title('Expected Gradient (EG) Summary Plot',fontsize=20)
plt.xlabel("EG values (impact on model output)")
plt.show()


# In[14]:


plt.figure(figsize=(8,6),dpi=80)
shap.summary_plot(shap_values=htot_attr_cov_test_np,
                  features=htot_loaded_data_test.meta_df,
                  show=False, 
                  plot_size='auto',
                  max_display=15)
# max_display: number of features to display
    
plt.title('Expected Gradient (EG) Summary Plot',fontsize=20)
plt.xlabel("EG values (impact on model output)")
plt.show()


# In[24]:

# Femoral Neck
neck_loaded_data_train, neck_loaded_data_test, neck_attr_mgs_test_np, neck_attr_cov_test_np = get_interpretation_results(bmd_site='NECK_BMD')


# In[25]:


plt.figure(figsize=(8,6),dpi=80)
neck_mgs_features = neck_loaded_data_test.X_df.copy()
neck_mgs_features.columns = neck_loaded_data_test.features
shap.summary_plot(shap_values=neck_attr_mgs_test_np,
                  features=neck_mgs_features,
                  show=False, 
                  plot_size='auto',
                  max_display=15)
# max_display: number of features to display
    
plt.title('Expected Gradient (EG) Summary Plot',fontsize=20)
plt.xlabel("EG values (impact on model output)")
plt.show()


# In[26]:


plt.figure(figsize=(8,6),dpi=80)
shap.summary_plot(shap_values=neck_attr_cov_test_np,
                  features=neck_loaded_data_test.meta_df,
                  show=False, 
                  plot_size='auto',
                  max_display=15)
# max_display: number of features to display
    
plt.title('Expected Gradient (EG) Summary Plot',fontsize=20)
plt.xlabel("EG values (impact on model output)")
plt.show()


# In[18]:

# Total Spine
spine_loaded_data_train, spine_loaded_data_test, spine_attr_mgs_test_np, spine_attr_cov_test_np = get_interpretation_results(bmd_site='spine_total_bmd')


# In[19]:


plt.figure(figsize=(8,6),dpi=80)
spine_mgs_features = spine_loaded_data_test.X_df.copy()
spine_mgs_features.columns = spine_loaded_data_test.features
shap.summary_plot(shap_values=spine_attr_mgs_test_np,
                  features=spine_mgs_features,
                  show=False, 
                  plot_size='auto',
                  max_display=15)
# max_display: number of features to display
    
plt.title('Expected Gradient (EG) Summary Plot',fontsize=20)
plt.xlabel("EG values (impact on model output)")
plt.show()


# In[20]:


plt.figure(figsize=(8,6),dpi=80)
shap.summary_plot(shap_values=spine_attr_cov_test_np,
                  features=spine_loaded_data_test.meta_df,
                  show=False, 
                  plot_size='auto',
                  max_display=15)
# max_display: number of features to display
    
plt.title('Expected Gradient (EG) Summary Plot',fontsize=20)
plt.xlabel("EG values (impact on model output)")
plt.show()


# In[21]:

# One-Third Radius
r13_loaded_data_train, r13_loaded_data_test, r13_attr_mgs_test_np, r13_attr_cov_test_np = get_interpretation_results(bmd_site='R_13_BMD')


# In[22]:


plt.figure(figsize=(8,6),dpi=80)
r13_mgs_features = r13_loaded_data_test.X_df.copy()
r13_mgs_features.columns = r13_loaded_data_test.features
shap.summary_plot(shap_values=r13_attr_mgs_test_np,
                  features=r13_mgs_features,
                  show=False, 
                  plot_size='auto',
                  max_display=15)
# max_display: number of features to display
    
plt.title('Expected Gradient (EG) Summary Plot',fontsize=20)
plt.xlabel("EG values (impact on model output)")
plt.show()


# In[23]:


plt.figure(figsize=(8,6),dpi=80)
shap.summary_plot(shap_values=r13_attr_cov_test_np,
                  features=r13_loaded_data_test.meta_df,
                  show=False, 
                  plot_size='auto',
                  max_display=15)
# max_display: number of features to display
    
plt.title('Expected Gradient (EG) Summary Plot',fontsize=20)
plt.xlabel("EG values (impact on model output)")
plt.show()






