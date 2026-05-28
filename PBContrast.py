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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device("cpu")
EPSILON = 1e-9
DEBUG_MODE = False
dtype = torch.float64


# In[2]:


def load_data(path, div_type, PB, bmd_site, dtype, use_mask = False, mask_path = None, mask_name = None, conduct_clr_transformation = False):
    def to_numpy(data):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.to_numpy(dtype='float64')
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
    
    subject_id = pd.read_csv(path + 'subject_id_' + div_type + '.csv')
    ra_microbes_cache = pd.read_csv(path + 'microbe_comp_' + div_type + '.csv')
    if use_mask:
        selected_microbes = pd.read_csv(mask_path + 'microbe_names_' + mask_name + '.csv')
        selected_microbes = selected_microbes['species'].to_list()
        ra_microbes_cache = ra_microbes_cache[selected_microbes]
    bmd_data = pd.read_csv(path + 'bmd_' + div_type + '.csv')
    clinical_data = pd.read_csv(path + 'clinical_var_' + div_type + '.csv')

    ra_score = to_numpy(ra_microbes_cache)
    ra_score = normalize(ra_score)
    if conduct_clr_transformation:
        ra_score = clr_transform(ra_score)
    else:
        ra_score = np.log(ra_score)
    clinical_var = to_numpy(clinical_data)
    bmd = to_numpy(bmd_data.loc[:, [bmd_site]])
    
    RA_SCORE = torch.from_numpy(ra_score).type(dtype)
    CLINICAL_VAR = torch.from_numpy(clinical_var).type(dtype)
    BMD = torch.from_numpy(bmd).type(dtype)
    if torch.cuda.is_available():
        RA_SCORE = RA_SCORE.cuda()
        CLINICAL_VAR = CLINICAL_VAR.cuda()
        BMD = BMD.cuda()
        
    return(subject_id, RA_SCORE, CLINICAL_VAR, BMD)

def load_balances(path, PB, bmd_name, dtype, condition = '_'):
    balance_mask = pd.read_csv(path + PB + condition + bmd_name + '_only_selected_bals.csv', index_col = 0).to_numpy(dtype='float64')
    balance_mask = torch.from_numpy(balance_mask).type(dtype)
    balance_mask = torch.transpose(balance_mask, 0, 1)
    
    ###if gpu is being used
    if torch.cuda.is_available():
        balance_mask = balance_mask.cuda()
    ###
    
    return balance_mask


# In[4]:


"""
Codes related with deep divergence-based clustering and contrastive learning 
were adapted from https://github.com/DanielTrosten/mvc/tree/main/src/lib.

Modifications were made to better accommodate our analyses.
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
    Y = Y.to(dtype=X.dtype, device=X.device)
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


# In[24]:


def DDC1(output, hidden, net):
    return d_cs(output, hidden_kernel(hidden), net.n_clusters)


def DDC2(output):
    n = output.size(0)
    return 2 / (n * (n - 1)) * triu(output @ torch.t(output))


def DDC3(output, hidden, net):
    I = torch.eye(net.n_clusters, dtype=output.dtype, device=output.device)
    m = torch.exp(-cdist(output, I))
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
            self.eye = torch.eye(n_clusters, device=DEVICE)
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

        masks = torch.eye(n, device=DEVICE)

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


def mse_loss(pred_x, x):
    batch_size = x.size(0)
    assert batch_size != 0
    mse_loss_val = F.mse_loss(pred_x, x, reduction='sum').div(batch_size)
    
    return mse_loss_val


def rmse_loss(pred_x, x):
    batch_size = x.size(0)
    assert batch_size != 0
    mse_loss_val = F.mse_loss(pred_x, x, reduction='sum').div(batch_size)
    rmse_loss_val = torch.sqrt(mse_loss_val)
    
    return rmse_loss_val


def orth_loss(z_a, z_b, eps=1e-9):
    batch_size = z_a.size(0)

    # Compute column means
    mu_a = torch.mean(z_a, dim=0, keepdim=True)  # shape (1, D)
    mu_b = torch.mean(z_b, dim=0, keepdim=True)
    
    # Center
    z_a_centered = z_a - mu_a                    # shape (N, D)
    z_b_centered = z_b - mu_b

    # Compute column‐wise std (unbiased? usually we use population std: division by N)
    # Add a tiny eps for numerical stability
    eps = 1e-10
    sigma_a = torch.sqrt(torch.mean(z_a_centered ** 2, dim=0, keepdim=True) + eps)  # (1, D)
    sigma_b = torch.sqrt(torch.mean(z_b_centered ** 2, dim=0, keepdim=True) + eps)

    # Normalize to unit variance
    z_a_norm = z_a_centered / sigma_a    # (N, D)
    z_b_norm = z_b_centered / sigma_b    # (N, D)

    # 2) Compute cross‐correlation matrix C of shape (D, D)
    #    C_{ij} = (1/N) * sum_n [ z_a_norm[n,i] * z_b_norm[n,j] ]
    c = (z_a_norm.T @ z_b_norm) / batch_size  # (D, D)
    
    mean_sum_of_squares = torch.mean(c**2)
    orthogonal_loss = torch.sqrt(mean_sum_of_squares + eps)
    
    return orthogonal_loss

def get_r2(x, pred_x):
    r, _ = pearsonr(x, pred_x)
    r2 = r**2
    
    return r2


# In[8]:


class MTL_mgs_bmd(nn.Module):
    def __init__(self, input_v1_n1, input_v5_n1, vbal_level1_dim, vcovar_level1_dim, 
                 fuse_level2_dim, level_hidden_dim, dc_h_dim, 
                 n_views, n_clusters, bal,
                 dropout_rate_1=0.01):
        super(MTL_mgs_bmd, self).__init__()
        
        self.n_clusters = n_clusters
        self.bal = bal
        
        ## balance --> hidden feature construction
        # metagenome mRA encoder pre
        self.v1_encoder_pre = nn.Sequential(
            nn.Linear(input_v1_n1, vbal_level1_dim),
            #nn.BatchNorm1d(vbal_level1_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # clinical encoder pre
        self.v5_encoder_pre = nn.Sequential(
            nn.Linear(input_v5_n1, vcovar_level1_dim),
            #nn.BatchNorm1d(vcovar_level1_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        ## feature integration
        # metagenome mRA encoder integrate
        self.v1_encoder_integrate = nn.Sequential(
            nn.Dropout(dropout_rate_1),
            nn.Linear(vbal_level1_dim, fuse_level2_dim),
            #nn.BatchNorm1d(fuse_level2_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # clinical encoder integrate
        self.v5_encoder_integrate = nn.Sequential(
            nn.Dropout(dropout_rate_1),
            nn.Linear(vcovar_level1_dim, fuse_level2_dim),
            #nn.BatchNorm1d(fuse_level2_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        ## fusing different views
        self.fusion = WeightedMean(n_views)
        
        ## clustering
        self.hidden_projector = nn.Sequential(
            nn.Linear(fuse_level2_dim, level_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.cluster = nn.Sequential(
            nn.Linear(level_hidden_dim, n_clusters),
            nn.Softmax(dim=1)
        )
        
        ## BMD prediction heads
        self.decoder_1 = nn.Sequential(
            nn.Linear(fuse_level2_dim, dc_h_dim),
            nn.BatchNorm1d(dc_h_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(dc_h_dim, 1)
        )
        
    def forward(self, mgs_ra, clinical_var):
        bal_score = torch.mm(mgs_ra, self.bal)
        
        v1_code_pre = self.v1_encoder_pre(bal_score)
        v5_code_pre = self.v5_encoder_pre(clinical_var)
        
        v1_code_integrate = self.v1_encoder_integrate(v1_code_pre)
        v5_code_integrate = self.v5_encoder_integrate(v5_code_pre)
        fused_code = self.fusion([v1_code_integrate, v5_code_integrate])
        projections = torch.cat((v1_code_integrate, v5_code_integrate), dim = 0)
        hidden = self.hidden_projector(fused_code)
        output = self.cluster(hidden)
        
        pred_bmd = self.decoder_1(fused_code)
        
        return(projections, hidden, output, fused_code, pred_bmd)


# In[25]:


class Objective:
    def __init__(self, data_path, balance_path, 
                 use_mask, mask_path, mask_name, condition,
                 div, PB, bmd_site, bmd_name, n_views = 2, dtype = torch.float64):
        self.data_path = data_path
        self.balance_path = balance_path
        self.div = div
        self.PB = PB
        self.bmd_site = bmd_site
        self.bmd_name = bmd_name
        self.n_views = n_views
        self.dtype = dtype
        self.use_mask = use_mask
        self.mask_path = mask_path
        self.mask_name = mask_name
        self.condition = condition
        
    def __call__(self, trial):
        # Hyperparameter suggestions
        learning_rate1 = trial.suggest_float('learning_rate1', 1e-5, 1e-1, log=True)
        learning_rate2 = trial.suggest_float('learning_rate2', 1e-5, 1e-1, log=True)
        l2 = trial.suggest_float('l2', 1e-3, 1, log=True)
        p1_epoch_num = trial.suggest_int('p1_epoch_num', 80, 200, step = 20)
        p2_epoch_num = trial.suggest_int('p2_epoch_num', 100, 800, step = 100)
        n_clusters = trial.suggest_int('n_clusters', 2, 16)
        lambda_1 = trial.suggest_float('lambda_1', 1e-3, 10, log=True)
        
        # Data loader
        subject_id_train, ra_score_train, clinical_var_train, bmd_train = load_data(self.data_path + self.div + "/", 
                                                                                    "tr_" + self.div, 
                                                                                    self.PB, 
                                                                                    self.bmd_site,
                                                                                    self.dtype, 
                                                                                    use_mask = self.use_mask, 
                                                                                    mask_path = self.mask_path, 
                                                                                    mask_name = self.mask_name)
        
        subject_id_valid, ra_score_valid, clinical_var_valid, bmd_valid = load_data(self.data_path + self.div + "/", 
                                                                                    "val_" + self.div, 
                                                                                    self.PB, 
                                                                                    self.bmd_site,
                                                                                    self.dtype, 
                                                                                    use_mask = self.use_mask, 
                                                                                    mask_path = self.mask_path, 
                                                                                    mask_name = self.mask_name)
        balance_mask = load_balances(self.balance_path, self.PB, self.bmd_name, self.dtype, self.condition)
        
        # Model, loss function, optimization
        input_v1_n1 = balance_mask.size(1)
        input_v5_n1 = clinical_var_train.size(1)
        model = MTL_mgs_bmd(input_v1_n1 = input_v1_n1, 
                            input_v5_n1 = input_v5_n1, 
                            vbal_level1_dim = trial.suggest_int('vbal_level1_dim', 2, 16, step = 2), 
                            vcovar_level1_dim = trial.suggest_int('vcovar_level1_dim', 2, 16, step = 2), 
                            fuse_level2_dim = trial.suggest_int('fuse_level2_dim', 2, 8, step = 2), 
                            level_hidden_dim = trial.suggest_int('level_hidden_dim', 2, 4, step = 2), 
                            dc_h_dim = trial.suggest_int('dc_h_dim', 2, 4, step = 2), 
                            n_views = self.n_views, 
                            n_clusters = n_clusters,
                            bal = balance_mask)
        model = model.double()
        if torch.cuda.is_available():
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr = learning_rate1, weight_decay = l2)
        scheduler_phase1 = CosineAnnealingLR(optimizer, T_max=p1_epoch_num)
        scheduler_phase2 = CosineAnnealingLR(optimizer, T_max=p2_epoch_num)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_phase1, scheduler_phase2], milestones=[p1_epoch_num])
        
        contrastive = Contrastive(n_clusters)
        for epoch1 in range(1, p1_epoch_num + 1):
            model.train()
            optimizer.zero_grad()
            
            _, hidden, output, _, _ = model(ra_score_train, clinical_var_train)
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
                _, hidden_valid, output_valid, _, _ = model(ra_score_valid, clinical_var_valid)
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
            
            projections, _, output, fused_code, pred_bmd = model(ra_score_train, clinical_var_train)
            pred_loss = rmse_loss(pred_bmd, bmd_train)
            contrastive_loss = contrastive._loss_with_negative_sampling(output, projections, model, self.n_views)
            loss = pred_loss + lambda_1*contrastive_loss
            
            loss.backward()
            optimizer.step()
            loss_train = loss.item()
            
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                projections_valid, _, output_valid, fused_code_valid, pred_bmd_valid = model(ra_score_valid, clinical_var_valid)
                loss_valid = rmse_loss(pred_bmd_valid, bmd_valid)
                
            trial.report(loss_valid, epoch2)
            if trial.should_prune():
                print(f'Trial {trial.number} pruned at phase 2 epoch {epoch2}.')
                raise optuna.exceptions.TrialPruned()
            
            if epoch2 % 100 == 0:
                print(f'{self.div} Phase 2 - Epoch [{epoch2}/{p2_epoch_num}], Overall Training Loss: {loss_train:.4f}, Training Loss: {pred_loss.item():.4f}, Validation Loss: {loss_valid.item():.4f}')
        
        return loss_valid


# In[32]:


def testMTL_mgs_bmd(data_path, balance_path, PB, bmd_site, bmd_name,  
                    use_mask, mask_path, mask_name, condition,
                    best_params_dict, init_model_params_dict, model_path, div, 
                    n_views = 2, dtype = torch.float64, save_pred_results = True):
    summarized_results_dict = {}
    summarized_results_dict.update(best_params_dict)
    
    subject_id_tune, ra_score_tune, clinical_var_tune, bmd_tune = load_data(data_path + 'train_test_split/', 
                                                                            'tu', 
                                                                            PB, 
                                                                            bmd_site, 
                                                                            dtype, 
                                                                            use_mask = use_mask, 
                                                                            mask_path = mask_path, 
                                                                            mask_name = mask_name)
        
    subject_id_test, ra_score_test, clinical_var_test, bmd_test = load_data(data_path + 'train_test_split/', 
                                                                            'te', 
                                                                            PB, 
                                                                            bmd_site, 
                                                                            dtype, 
                                                                            use_mask = use_mask, 
                                                                            mask_path = mask_path, 
                                                                            mask_name = mask_name)
    balance_mask = load_balances(balance_path, PB, bmd_name, dtype, condition)
    tune_num_subject = subject_id_tune.shape[0]
    test_num_subject = subject_id_test.shape[0]
    input_v1_n1 = balance_mask.size(1)
    input_v5_n1 = clinical_var_tune.size(1)
    p1_epoch_num = best_params_dict['p1_epoch_num']
    p2_epoch_num = best_params_dict['p2_epoch_num']
    learning_rate1 = best_params_dict['learning_rate1']
    learning_rate2 = best_params_dict['learning_rate2']
    l2 = best_params_dict['l2']
    n_clusters = best_params_dict['n_clusters']
    lambda_1 = best_params_dict['lambda_1']
    
    init_model_params = {key: best_params_dict[key] for key in init_model_params_dict if key in best_params_dict}
    
    model = MTL_mgs_bmd(input_v1_n1 = input_v1_n1,
                        input_v5_n1 = input_v5_n1, 
                        n_views = n_views, 
                        n_clusters = n_clusters,
                        bal = balance_mask,
                        **init_model_params)
    model = model.double()
    os.makedirs(model_path, exist_ok=True)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate1, weight_decay = l2)
    scheduler_phase1 = CosineAnnealingLR(optimizer, T_max=p1_epoch_num)
    scheduler_phase2 = CosineAnnealingLR(optimizer, T_max=p2_epoch_num)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_phase1, scheduler_phase2], milestones=[p1_epoch_num])
    
    contrastive = Contrastive(n_clusters)
    for epoch1 in range(1, p1_epoch_num + 1):
        model.train()
        optimizer.zero_grad()
            
        _, hidden, output, _, _ = model(ra_score_tune, clinical_var_tune)
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
            _, hidden_test, output_test, _, _ = model(ra_score_test, clinical_var_test)
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
            
        projections, _, output, fused_code, pred_bmd = model(ra_score_tune, clinical_var_tune)
        pred_loss = rmse_loss(pred_bmd, bmd_tune)
        contrastive_loss = contrastive._loss_with_negative_sampling(output, projections, model, n_views)
        loss = pred_loss + lambda_1*contrastive_loss
            
        loss.backward()
        optimizer.step()
        loss_tune = loss.item()
            
        scheduler.step()
            
        model.eval()
        with torch.no_grad():
            projections_test, _, output_test, fused_code_test, pred_bmd_test = model(ra_score_test, clinical_var_test)
            loss_test = rmse_loss(pred_bmd_test, bmd_test)
            
        if epoch2 % 100 == 0:
            print(f'{div} Testing Stage Phase 2 - Epoch [{epoch2}/{p2_epoch_num}], Overall Training Loss: {loss_tune:.4f}, Training Loss: {pred_loss.item():.4f}, Testing Loss: {loss_test.item():.4f}')
    
    torch.save(model.state_dict(), model_path + div + condition + PB + '_PBContrast_optparam_mtl_mgs_bmd_testing.pt')
    
    tune_bmd = np.asarray(bmd_tune.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    tune_bmd_dict = {bmd_name: tune_bmd}
    tune_bmd_pred = np.asarray(pred_bmd.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    tune_pred_dict = {'pred_' + bmd_name: tune_bmd_pred}
    tune_pred_cache = pd.DataFrame.from_dict(tune_pred_dict)
    
    rmse_r2_dict = {}
    tune_rmse_dict = {'Tuning RMSE': np.array(pred_loss.detach().cpu().numpy())}
    tune_r2_dict = {'Tuning R2': get_r2(tune_bmd_dict.get(bmd_name), tune_pred_dict.get('pred_' + bmd_name))}
    
    rmse_r2_dict.update(tune_rmse_dict)
    rmse_r2_dict.update(tune_r2_dict)
    
    test_bmd = np.asarray(bmd_test.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    test_bmd_dict = {bmd_name: test_bmd}
    test_bmd_pred = np.asarray(pred_bmd_test.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    test_pred_dict = {'pred_' + bmd_name: test_bmd_pred}
    test_pred_cache = pd.DataFrame.from_dict(test_pred_dict)
    
    test_rmse_dict = {'Testing RMSE': np.array(loss_test.detach().cpu().numpy())}
    test_r2_dict = {'Testing R2': get_r2(test_bmd_dict.get(bmd_name), test_pred_dict.get('pred_' + bmd_name))}
    
    rmse_r2_dict.update(test_rmse_dict)
    rmse_r2_dict.update(test_r2_dict)
    
    if save_pred_results:
        pred_save_path = data_path + div + '/prediction_results/'
        os.makedirs(pred_save_path, exist_ok=True)
        tune_pred_cache.to_csv(pred_save_path + div + '_' + bmd_site + '_' + PB + '_PBContrast_tune_set_pred_results.csv', index=False)
        test_pred_cache.to_csv(pred_save_path + div + '_' + bmd_site + '_' + PB + '_PBContrast_test_set_pred_results.csv', index=False)
    
    summarized_results_dict.update(rmse_r2_dict)
    
    return summarized_results_dict


# In[20]:


root_path = 'root_path/'
data_path = root_path + 'data_folder/'
mask_path = data_path + 'tree_folder/'
use_mask = True
mask_name = 'prev_filtered'
balance_path = root_path + 'selected_balances/'
condition = '_prevfiltered_'
bmd_site = 'bmd_site' #NECK_BMD, HTOT_BMD, spine_total_bmd, R_13_BMD
if bmd_site == 'NECK_BMD':
    bmd_name = 'fneck'
elif bmd_site == 'HTOT_BMD':
    bmd_name = 'htot'
elif bmd_site == 'spine_total_bmd':
    bmd_name = 'spine'
elif bmd_site == 'R_13_BMD':
    bmd_name = 'R13'
else:
    raise ValueError("Unrecorded BMD type.")
div_list = np.char.add('tune_', np.array(list(range(1, 11, 1))).astype('str')).tolist()
PB_methods = ['PCA_PB']
model_path = root_path + 'saved_models_pbcontrast/'
init_model_params_dict = {'vbal_level1_dim', 'vcovar_level1_dim', 'fuse_level2_dim', 
                          'level_hidden_dim', 'dc_h_dim'}
summarized_results_path = root_path + 'summarized_results_pbcontrast/'
os.makedirs(summarized_results_path, exist_ok=True)


# In[33]:


for PB in PB_methods:
    div_track = []
    summarized_results_cache = []
    for div in div_list:
        print(f'Running on {div}')
        pruner = optuna.pruners.HyperbandPruner(min_resource = 50)
        study = optuna.create_study(direction='minimize', pruner=pruner)
        objective = Objective(data_path, balance_path, use_mask, mask_path, mask_name, condition, div, PB, bmd_site, bmd_name)
        study.optimize(objective, n_trials=100)
    
        best_params_dict = study.best_trial.params
        summarized_results_dict = testMTL_mgs_bmd(data_path, balance_path, PB, bmd_site, bmd_name,
                                                  use_mask, mask_path, mask_name, condition,
                                                  best_params_dict, init_model_params_dict, model_path, div)
        summarized_results_cache.append(summarized_results_dict)
        div_track.append(div)

    div_track_dic = {'division': div_track}
    div_tract_cache = pd.DataFrame(data = div_track_dic)

    summarized_results_cache = pd.DataFrame.from_dict(summarized_results_cache)
    
    summarized_results_cache = pd.concat([div_tract_cache, summarized_results_cache], axis = 1)
    summarized_results_cache.to_csv(summarized_results_path + PB + condition + bmd_name + "_pbcontrast_summarized_results.csv", index=False)


# In[ ]:




