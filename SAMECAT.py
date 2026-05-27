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
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
import optuna
import os
from copy import deepcopy
from scipy.stats import pearsonr
from ete3 import Tree
from torch.utils.data import Dataset
from captum.module import (BinaryConcreteStochasticGates,
                           GaussianStochasticGates)
import torch.nn.utils.prune as prune

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
    Adapted from https://github.com/batmen-lab/MIOSTONE.

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
        
        X = data.values
        meta = meta.values
        y = y.values
        
        return cls(subject_id, X, meta, y, features)
    
    def zero_handling(self):
        nonzero_vals = self.X[self.X > 0]
        if nonzero_vals.size == 0:
            raise ValueError("Array contains no non-zero values.")
        min_nonzero = nonzero_vals.min()
        # compute replacement = half of min_nonzero
        replacement = 0.5 * min_nonzero
        # replace zeros in place
        self.X[self.X == 0] = replacement

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


# In[4]:


"""
Codes related with deep divergence-based clustering and contrastive learning 
were adapted from https://github.com/DanielTrosten/mvc/tree/main/src/lib.

Modifications were made to better accommodate our analysis.
"""

def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=EPSILON):
    """
    Compute a Gaussian kernel matrix from a distance matrix.

    :param dist: Disatance matrix
    :type dist: th.Tensor
    :param rel_sigma: Multiplication factor for the sigma hyperparameter
    :type rel_sigma: float
    :param min_sigma: Minimum value for sigma. For numerical stability.
    :type min_sigma: float
    :return: Kernel matrix
    :rtype: th.Tensor
    """
    # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
    dist = relu(dist)
    sigma2 = rel_sigma * torch.median(dist)
    # Disable gradient for sigma
    sigma2 = sigma2.detach()
    sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
    k = torch.exp(- dist / (2 * sigma2))
    return k


def vector_kernel(x, rel_sigma=0.15):
    """
    Compute a kernel matrix from the rows of a matrix.

    :param x: Input matrix
    :type x: th.Tensor
    :param rel_sigma: Multiplication factor for the sigma hyperparameter
    :type rel_sigma: float
    :return: Kernel matrix
    :rtype: th.Tensor
    """
    return kernel_from_distance_matrix(cdist(x, x), rel_sigma)


def cdist(X, Y):
    """
    Pairwise distance between rows of X and rows of Y.

    :param X: First input matrix
    :type X: th.Tensor
    :param Y: Second input matrix
    :type Y: th.Tensor
    :return: Matrix containing pairwise distances between rows of X and rows of Y
    :rtype: th.Tensor
    """
    xyT = X @ torch.t(Y)
    x2 = torch.sum(X**2, dim=1, keepdim=True)
    y2 = torch.sum(Y**2, dim=1, keepdim=True)
    d = x2 - 2 * xyT + torch.t(y2)
    return d


# In[5]:


def triu(X):
    # Sum of strictly upper triangular part
    return torch.sum(torch.triu(X, diagonal=1))


def _atleast_epsilon(X, eps=1e-9):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return torch.where(X < eps, X.new_tensor(eps), X)


def d_cs(A, K, n_clusters):
    """
    Cauchy-Schwarz divergence.

    :param A: Cluster assignment matrix
    :type A:  th.Tensor
    :param K: Kernel matrix
    :type K: th.Tensor
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :return: CS-divergence
    :rtype: th.Tensor
    """
    nom = torch.t(A) @ K @ A
    dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=1e-9**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / torch.sqrt(dnom_squared))
    return d


# In[6]:


class _Fusion(nn.Module):
    def __init__(self):
        """
        Base class for the fusion module

        :param cfg: Fusion config. See config.defaults.Fusion
        :param input_sizes: Input shapes
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

        :param cfg: Fusion config. See config.defaults.Fusion
        :param input_sizes: Input shapes
        """
        super().__init__()

    def forward(self, inputs):
        return torch.mean(torch.stack(inputs, -1), dim=-1)


class WeightedMean(_Fusion):
    """
    Weighted mean fusion.

    :param cfg: Fusion config. See config.defaults.Fusion
    :param input_sizes: Input shapes
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


def DDC1(output, hidden, net):
    return d_cs(output, hidden_kernel(hidden), net.n_clusters)


def DDC2(output):
    n = output.size(0)
    return 2 / (n * (n - 1)) * triu(output @ torch.t(output))


def DDC3(output, hidden, net):
    m = torch.exp(-cdist(output, torch.eye(net.n_clusters, dtype=torch.float64, device=DEVICE)))
    return d_cs(m, hidden_kernel(hidden), net.n_clusters)


def contrastive_loss(pos, neg):
    """
    Computes the contrastive loss using the log-softmax formulation 
    where the positive pair is excluded from the denominator:

    L = - mean(log(exp(pos) / sum(exp(neg))))

    Args:
        inputs (torch.Tensor): pos: Positive pair distances (shape: [batch_size])
                               neg: Negative pair distances (shape: [batch_size, num_negatives])

    Returns:
        torch.Tensor: The computed contrastive loss.
    """

    # Compute exponentials
    exp_pos = torch.exp(pos)  # exp(pos)
    exp_neg_sum = torch.sum(torch.exp(neg), dim=1)  # sum(exp(neg))

    # Compute log probabilities
    log_prob = torch.log(exp_pos / exp_neg_sum)

    # Compute mean negative log-likelihood
    loss = -torch.mean(log_prob)

    return loss


def contrastive_loss_without_negative_sampling(input_logit):
    """
    Computes contrastive loss where:
    - Diagonal elements are positive pair distances.
    - Off-diagonal elements are negative pair distances.

    L = - mean(log(exp(D_ii) / sum(exp(D_ij) for j ≠ i)))

    Args:
        matrix (torch.Tensor): A square matrix of shape (batch_size, batch_size)
                               where matrix[i, i] is the positive distance,
                               and matrix[i, j] (j ≠ i) are negative distances.

    Returns:
        torch.Tensor: The computed contrastive loss.
    """

    # Extract positive pair distances from diagonal
    pos_distances = torch.diagonal(input_logit)  # Shape: (batch_size,)

    # Compute exponentials of positive distances
    exp_pos = torch.exp(pos_distances)

    # Compute exponentials of all elements
    exp_input = torch.exp(input_logit)

    # Compute sum over negative distances (excluding diagonal)
    exp_neg_sum = torch.sum(exp_input, dim=1) - exp_pos  # Exclude diagonal

    # Compute log probabilities
    log_prob = torch.log(exp_pos / exp_neg_sum)

    # Compute mean negative log-likelihood
    loss = -torch.mean(log_prob)

    return loss

large_num = 1e9
class Contrastive:
    def __init__(self, n_clusters, contrastive_similarity = "cos", negative_samples_ratio = 0.25):
        """
        Contrastive loss function

        """
        super().__init__()
        self.large_num = large_num
        self.n_clusters = n_clusters
        # Select which implementation to use
        if negative_samples_ratio == -1:
            self._loss_func = self._loss_without_negative_sampling
        else:
            self.eye = torch.eye(n_clusters, dtype=torch.float64,device=DEVICE)
            self._loss_func = self._loss_with_negative_sampling

        # Set similarity function
        if contrastive_similarity == "cos":
            self.similarity_func = self._cosine_similarity
        elif contrastive_similarity == "gauss":
            self.similarity_func = vector_kernel
        else:
            raise RuntimeError(f"Invalid contrastive similarity: {contrastive_similarity}")
        self.negative_samples_ratio = negative_samples_ratio

    @staticmethod
    def _norm(mat):
        return F.normalize(mat, p=2, dim=1)

    @staticmethod
    def get_weight(net):
        w = torch.min(F.softmax(net.fusion.weights.detach(), dim=0))
        return w

    @classmethod
    def _normalized_projections(cls, projections):
        n = projections.size(0) // 2
        h1, h2 = projections[:n], projections[n:]
        h2 = cls._norm(h2)
        h1 = cls._norm(h1)
        return n, h1, h2

    @classmethod
    def _cosine_similarity(cls, projections):
        h = cls._norm(projections)
        return h @ h.t()
    
    def ensure_diverse_clusters(self, assignments):
        """
        Ensures that at least one subject is assigned to a different cluster if all samples
        are initially assigned to the same group.

        Args:
            assignments (torch.Tensor): A 1D tensor of cluster assignments (shape: [num_samples]).

        Returns:
            torch.Tensor: Updated cluster assignments.
        """
        unique_clusters = torch.unique(assignments)

        # If all samples are in the same cluster
        if unique_clusters.numel() == 1:
            #print(f"All samples assigned to cluster {unique_clusters.item()}! Reassigning one sample...")
        
            # Randomly pick one sample index
            random_index = torch.randint(0, assignments.shape[0], (1,)).item()

            # Pick a different cluster than the current one
            current_cluster = unique_clusters.item()
            possible_clusters = [i for i in range(self.n_clusters) if i != current_cluster]
            new_cluster = possible_clusters[torch.randint(0, len(possible_clusters), (1,)).item()]

            # Assign the new cluster to the selected subject
            assignments[random_index] = new_cluster
            #print(f"Subject at index {random_index} reassigned to cluster {new_cluster}")

        return assignments

    def _draw_negative_samples(self, output, v, pos_indices):
        """
        Construct set of negative samples.

        :param output: Model clustering output
        :type output: torch.Tensor
        :param v: Number of views
        :type v: int
        :param pos_indices: Row indices of the positive samples in the concatenated similarity matrix
        :type pos_indices: torch.Tensor
        :return: Indices of negative samples
        :rtype: th.Tensor
        """
        cat = output.detach().argmax(dim=1)
        cat = self.ensure_diverse_clusters(cat)
        cat = torch.cat(v * [cat], dim=0)
        #print("cat unique values: ", torch.unique(cat))

        weights = (1 - self.eye[cat])[:, cat[[pos_indices]]].T
        #print("Weights min:", weights.min(), "Weights max:", weights.max(), "Sum:", weights.sum(dim=1))
        #assert (weights >= 0).all(), "Weights contain negative values!"
        #assert (weights.sum(dim=1) > 0).all(), "Some weight sums are zero!"
        n_negative_samples = int(self.negative_samples_ratio * cat.size(0))
        negative_sample_indices = torch.multinomial(weights, n_negative_samples, replacement=True)
        if DEBUG_MODE:
            self._check_negative_samples_valid(cat, pos_indices, negative_sample_indices)
        return negative_sample_indices

    @staticmethod
    def _check_negative_samples_valid(cat, pos_indices, neg_indices):
        pos_cats = cat[pos_indices].view(-1, 1)
        neg_cats = cat[neg_indices]
        assert (pos_cats != neg_cats).detach().cpu().numpy().all()

    @staticmethod
    def _get_positive_samples(logits, v, n):
        """
        Get positive samples

        :param logits: Input similarities
        :type logits: th.Tensor
        :param v: Number of views
        :type v: int
        :param n: Number of samples per view (batch size)
        :type n: int
        :return: Similarities of positive pairs, and their indices
        :rtype: Tuple[th.Tensor, th.Tensor]
        """
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = torch.diagonal(logits, offset=diagonal_offset)
            _lower = torch.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = torch.arange(0, diag_length)
            _lower_inds = torch.arange(i * n, v * n)
            if DEBUG_MODE:
                assert _upper.size() == _lower.size() == _upper_inds.size() == _lower_inds.size() == (diag_length,)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = torch.cat(diagonals, dim=0)
        pos_inds = torch.cat(inds, dim=0)
        return pos, pos_inds

    def _loss_with_negative_sampling(self, output, projections, net, v, tau = 0.1, adaptive_contrastive_weight = True, delta = 0.1):
        """
        Contrastive loss implementation with negative sampling.

        """
        n = output.size(0)
        logits = self.similarity_func(projections) / tau

        pos, pos_inds = self._get_positive_samples(logits, v, n)
        neg_inds = self._draw_negative_samples(output, v, pos_inds)
        #print("neg_inds size: ", neg_inds.size())
        #print("logits size: ", logits.size())
        neg = logits[pos_inds.view(-1, 1), neg_inds]
        
        loss = contrastive_loss(pos, neg)

        if adaptive_contrastive_weight:
            loss *= self.get_weight(net)

        return delta * loss

    def _loss_without_negative_sampling(self, projections, net, v, tau = 0.1, adaptive_contrastive_weight = True, delta = 0.1):
        """
        Contrastive loss implementation without negative sampling.

        """
        assert v == 2, "Contrastive loss without negative sampling only supports 2 views."
        n, h1, h2 = self._normalized_projections(projections)

        masks = torch.eye(n, dtype=torch.float64, device=DEVICE)

        logits_aa = ((h1 @ h1.t()) / tau) - masks * self.large_num
        logits_bb = ((h2 @ h2.t()) / tau) - masks * self.large_num

        logits_ab = (h1 @ h2.t()) / tau
        logits_ba = (h2 @ h1.t()) / tau
        
        input_a = torch.cat((logits_ab, logits_aa), dim=1)
        input_b = torch.cat((logits_ba, logits_bb), dim=1)
        
        loss_a = contrastive_loss_without_negative_sampling(input_a)
        loss_b = contrastive_loss_without_negative_sampling(input_b)

        loss = (loss_a + loss_b)

        if adaptive_contrastive_weight:
            loss *= self.get_weight(net)

        return delta * loss


# ======================================================================================================================
# Extra functions
# ======================================================================================================================

def hidden_kernel(hidden, rel_sigma = 0.15):
    return vector_kernel(hidden, rel_sigma)


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


# In[9]:


class Objective:
    def __init__(self, tree_path, master_path, div, bmd_site,
                 n_views = 2, use_mask = False, mask_name = None, 
                 dtype = torch.float64):
        self.tree_path = tree_path
        self.master_path = master_path
        self.div = div
        self.bmd_site = bmd_site
        self.n_views = n_views
        self.use_mask = use_mask
        self.mask_name = mask_name
        self.dtype = dtype
        
    def __call__(self, trial):
        # Hyperparameter suggestions
        learning_rate1 = trial.suggest_float('learning_rate1', 1e-5, 1e-1, log=True)
        learning_rate2 = trial.suggest_float('learning_rate2', 1e-5, 1e-1, log=True)
        l2 = trial.suggest_float('l2', 1e-3, 1, log=True)
        p1_epoch_num = trial.suggest_int('p1_epoch_num', 80, 200, step = 20)
        p2_epoch_num = trial.suggest_int('p2_epoch_num', 100, 800, step = 100)
        n_clusters = trial.suggest_int('n_clusters', 2, 16)
        lambda_1 = trial.suggest_float('lambda_1', 1e-3, 10, log=True)
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
        model = SAMECAT(tree = miostone_tree,
                        node_min_dim = 1,
                        node_dim_func = 'linear',
                        node_dim_func_param = 0.6, 
                        node_gate_type = 'concrete',
                        node_gate_param = 0.3,
                        prune_mode = 'taxonomy', 
                        input_vcovar_n1 = input_vcovar_n1, 
                        fuse_level_dim = trial.suggest_int('fuse_level_dim', 2, 16, step = 2), 
                        level_hidden_dim = trial.suggest_int('level_hidden_dim', 2, 16, step = 2), 
                        dc_h_dim = trial.suggest_int('dc_h_dim', 2, 8, step = 2), 
                        n_views = self.n_views, 
                        n_clusters = n_clusters)
        model = model.to(dtype=torch.float64, device=DEVICE)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate1, weight_decay = l2)
        scheduler_phase1 = CosineAnnealingLR(optimizer, T_max=p1_epoch_num)
        scheduler_phase2 = CosineAnnealingLR(optimizer, T_max=p2_epoch_num)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_phase1, scheduler_phase2], milestones=[p1_epoch_num])
        
        loaded_data_train.data_adaptation(self.dtype)
        loaded_data_valid.data_adaptation(self.dtype)
        contrastive = Contrastive(n_clusters)
        for epoch1 in range(1, p1_epoch_num + 1):
            model.train()
            optimizer.zero_grad()
            
            _, hidden, output, _, _ = model(loaded_data_train.X, loaded_data_train.meta)
            DDC1_loss = DDC1(output, hidden, model) 
            DDC2_loss = DDC2(output) 
            DDC3_loss = DDC3(output, hidden, model)
            loss = DDC1_loss + DDC2_loss + DDC3_loss
            
            loss.backward()
            optimizer.step()
            loss_train = loss.item()
            
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                _, hidden_valid, output_valid, _, _ = model(loaded_data_valid.X, loaded_data_valid.meta)
                DDC1_loss_valid = DDC1(output_valid, hidden_valid, model) 
                DDC2_loss_valid = DDC2(output_valid) 
                DDC3_loss_valid = DDC3(output_valid, hidden_valid, model)
                loss_valid = DDC1_loss_valid + DDC2_loss_valid + DDC3_loss_valid
            if epoch1 % 100 == 0:
                print(f'Phase 1 - Epoch [{epoch1}/{p1_epoch_num}], Training Loss: {loss_train:.4f}, Validation Loss: {loss_valid.item():.4f}')
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate2
        
        for epoch2 in range(1, p2_epoch_num + 1):
            model.train()
            optimizer.zero_grad()
            
            projections, _, output, pred_htot, total_l0_reg = model(loaded_data_train.X, loaded_data_train.meta)
            pred_loss = rmse_loss(pred_htot, loaded_data_train.y)
            contrastive_loss = contrastive._loss_with_negative_sampling(output, projections, model, self.n_views)
            loss = pred_loss + lambda_1*contrastive_loss + lambda_2*total_l0_reg
            
            loss.backward()
            optimizer.step()
            loss_train = loss.item()
            
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                _, _, _, pred_htot_valid, _ = model(loaded_data_valid.X, loaded_data_valid.meta)
                loss_valid = rmse_loss(pred_htot_valid, loaded_data_valid.y)
                
            trial.report(loss_valid, epoch2)
            if trial.should_prune():
                print(f'Trial {trial.number} pruned at phase 2 epoch {epoch2}.')
                raise optuna.exceptions.TrialPruned()
            
            if epoch2 % 100 == 0:
                print(f'Phase 2 - Epoch [{epoch2}/{p2_epoch_num}], Oervall training loss: {loss_train:.4f}, Training Loss: {pred_loss.item():.4f}, Validation Loss: {loss_valid.item():.4f}')
        
        return loss_valid


# In[10]:


def testSAMECAT(tree_path, master_path, div, bmd_site, 
                best_params_dict, init_model_params_dict, model_path, 
                n_views = 2, use_mask = False, mask_name = None, 
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
    
    p1_epoch_num = best_params_dict['p1_epoch_num']
    p2_epoch_num = best_params_dict['p2_epoch_num']
    learning_rate1 = best_params_dict['learning_rate1']
    learning_rate2 = best_params_dict['learning_rate2']
    l2 = best_params_dict['l2']
    n_clusters = best_params_dict['n_clusters']
    lambda_1 = best_params_dict['lambda_1']
    lambda_2 = best_params_dict['lambda_2']
    
    init_model_params = {key: best_params_dict[key] for key in init_model_params_dict if key in best_params_dict}
    
    model = SAMECAT(tree = miostone_tree,
                    node_min_dim = 1,
                    node_dim_func = 'linear',
                    node_dim_func_param = 0.6, 
                    node_gate_type = 'concrete',
                    node_gate_param = 0.3,
                    prune_mode = 'taxonomy', 
                    input_vcovar_n1 = input_vcovar_n1, 
                    n_views = n_views, 
                    n_clusters = n_clusters,
                    **init_model_params)
    
    os.makedirs(model_path, exist_ok=True)
    model = model.to(dtype=torch.float64, device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate1, weight_decay = l2)
    scheduler_phase1 = CosineAnnealingLR(optimizer, T_max=p1_epoch_num)
    scheduler_phase2 = CosineAnnealingLR(optimizer, T_max=p2_epoch_num)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_phase1, scheduler_phase2], milestones=[p1_epoch_num])
    
    loaded_data_tune.data_adaptation(dtype)
    loaded_data_test.data_adaptation(dtype)
    contrastive = Contrastive(n_clusters)
    for epoch1 in range(1, p1_epoch_num + 1):
        model.train()
        optimizer.zero_grad()
            
        _, hidden, output, _, _ = model(loaded_data_tune.X, loaded_data_tune.meta)
        DDC1_loss = DDC1(output, hidden, model) 
        DDC2_loss = DDC2(output) 
        DDC3_loss = DDC3(output, hidden, model)
        loss = DDC1_loss + DDC2_loss + DDC3_loss
            
        loss.backward()
        optimizer.step()
        loss_tune = loss.item()
            
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            _, hidden_test, output_test, _, _ = model(loaded_data_test.X, loaded_data_test.meta)
            DDC1_loss_test = DDC1(output_test, hidden_test, model) 
            DDC2_loss_test = DDC2(output_test) 
            DDC3_loss_test = DDC3(output_test, hidden_test, model)
            loss_test = DDC1_loss_test + DDC2_loss_test + DDC3_loss_test
        if epoch1 % 100 == 0:
            print(f'Phase 1 - Epoch [{epoch1}/{p1_epoch_num}], Training Loss: {loss_tune:.4f}, Testing Loss: {loss_test.item():.4f}')
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate2
        
    for epoch2 in range(1, p2_epoch_num + 1):
        model.train()
        optimizer.zero_grad()
            
        projections, _, output, pred_htot, total_l0_reg = model(loaded_data_tune.X, loaded_data_tune.meta)
        pred_loss = rmse_loss(pred_htot, loaded_data_tune.y)
        contrastive_loss = contrastive._loss_with_negative_sampling(output, projections, model, n_views)
        loss = pred_loss + lambda_1*contrastive_loss + lambda_2*total_l0_reg
            
        loss.backward()
        optimizer.step()
        loss_tune = loss.item()
            
        scheduler.step()
            
        model.eval()
        with torch.no_grad():
            _, _, _, pred_htot_test, _ = model(loaded_data_test.X, loaded_data_test.meta)
            loss_test = rmse_loss(pred_htot_test, loaded_data_test.y)
            
        if epoch2 % 100 == 0:
            print(f'Phase 2 - Epoch [{epoch2}/{p2_epoch_num}], Overall training Loss: {loss_tune:.4f}, Training Loss: {pred_loss.item():.4f}, Testing Loss: {loss_test.item():.4f}')
    
    torch.save(model.state_dict(), model_path + div + '_' + bmd_site + '_optparam_samecat_testing.pt')
    
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
        tune_pred_cache.to_csv(pred_save_path + div + '_' + bmd_site + '_tune_set_pred_results.csv', index=False)
        test_pred_cache.to_csv(pred_save_path + div + '_' + bmd_site + '_test_set_pred_results.csv', index=False)
    
    summarized_results_dict.update(rmse_r2_dict)
    
    return summarized_results_dict


# In[22]:


root_path = 'root_path/'
master_path = root_path + 'data_folder/'
tree_path = master_path + 'tree_folder/'
div_list = np.char.add('tune_', np.array(list(range(1, 11, 1))).astype('str')).tolist()
model_path = root_path + 'saved_models_samecat/'
os.makedirs(model_path, exist_ok=True)
init_model_params_dict = {'fuse_level_dim', 'level_hidden_dim', 'dc_h_dim'}
summarized_results_path = root_path + 'summarized_results_samecat/'
os.makedirs(summarized_results_path, exist_ok=True)
bmd_site = 'bmd_site' #NECK_BMD, HTOT_BMD, spine_total_bmd, R_13_BMD
use_mask = True
mask_name = 'mask_name'


# In[35]:


div_track = []
summarized_results_cache = []
for div in div_list:
    print(f'Running on {div}')
    pruner = optuna.pruners.HyperbandPruner(min_resource = 20)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    objective = Objective(tree_path, master_path, div, bmd_site, use_mask = use_mask, mask_name = mask_name)
    study.optimize(objective, n_trials=100)
    
    best_params_dict = study.best_trial.params
    summarized_results_dict = testSAMECAT(tree_path, master_path, div, bmd_site, 
                                          best_params_dict, init_model_params_dict, model_path,
                                          use_mask = use_mask, mask_name = mask_name)
    summarized_results_cache.append(summarized_results_dict)
    div_track.append(div)

div_track_dic = {'division': div_track}
div_tract_cache = pd.DataFrame(data = div_track_dic)

summarized_results_cache = pd.DataFrame.from_dict(summarized_results_cache)
    
summarized_results_cache = pd.concat([div_tract_cache, summarized_results_cache], axis = 1)
summarized_results_cache.to_csv(summarized_results_path + bmd_site + '_samecat_summarized_results.csv', index=False)





