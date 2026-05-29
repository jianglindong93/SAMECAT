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
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
import optuna
import os
from copy import deepcopy
from scipy.stats import pearsonr
from ete3 import Tree
from torch.utils.data import Dataset
from captum.module import (BinaryConcreteStochasticGates,
                           GaussianStochasticGates)
import torch.nn.utils.prune as prune
from scipy.spatial import distance
from sklearn.covariance import LedoitWolf
from scipy.linalg import cholesky
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import connected_components
import warnings

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

    def __init__(self, subject_id, X_df, X, meta_df, meta, y, features):
        self.subject_id = subject_id
        self.X_df = X_df
        self.X = X
        self.meta_df = meta_df
        self.meta = meta
        self.y = y
        self.features = features
        #self.num_classes = len(np.unique(y))
        #self.class_weight = len(y) / (self.num_classes * np.bincount(y))
        self.normalized = False
        self.clr_transformed = False
        self.data_adapted = False
        #self.standardized = False
        #self.one_hot_encoded = False
        #self.tree_matrix_repr = False

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @classmethod
    def init_from_files(cls, master_path, div_type, bmd_sites, use_mask = False, mask_path = None, mask_name = None):
        subject_id = pd.read_csv(master_path + 'subject_id_' + div_type + '.csv')
        data = pd.read_csv(master_path + 'microbe_comp_' + div_type + ".csv")
        if use_mask:
            selected_microbes = pd.read_csv(mask_path + 'microbe_names_' + mask_name + '.csv')
            selected_microbes = selected_microbes['species'].to_list()
            data = data[selected_microbes]
        meta_df = pd.read_csv(master_path + 'clinical_var_' + div_type + '.csv')
        bmd_data = pd.read_csv(master_path + 'bmd_' + div_type + '.csv')
        features = ['s__' + col.split('.s__')[-1] for col in data.columns]

        #X = data.values.astype(np.float32)
        #meta = meta.values
        y = [bmd_data[[site]].values for site in bmd_sites]

        X_df = data
        X = data.values
        meta = meta_df.values
        #y = [bmd_data[[site]] for site in bmd_sites]

        return cls(subject_id, X_df, X, meta_df, meta, y, features)

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
        self.y = [torch.from_numpy(specific_bmd).type(dtype) for specific_bmd in self.y]
        if torch.cuda.is_available():
            self.X = self.X.cuda()
            self.meta = self.meta.cuda()
            self.y = [specific_bmd.cuda() for specific_bmd in self.y]
        self.data_adapted = True


# In[4]:


class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, save_model = False, model_path=None):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.epoch_count = 0
        self.best_epoch_num = 1
        self.early_stop = False
        self.min_loss = None
        self.delta = delta
        self.save_model = save_model
        self.model_path = model_path

    def __call__(self, loss, model):
        if self.min_loss is None:
            self.epoch_count += 1
            self.best_epoch_num = self.epoch_count
            self.min_loss = loss
            if self.save_model:
                self.save_checkpoint(model)
        elif loss > self.min_loss - self.delta:
            self.epoch_count += 1
            self.counter += 1
            if self.counter % 10 == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.epoch_count += 1
            self.best_epoch_num = self.epoch_count
            self.min_loss = loss
            if self.verbose:
                print(f'Validation accuracy increased ({self.max_acc:.6f} --> {acc:.6f}).  Saving model ...')
            if self.save_model:
                self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.model_path)


# In[5]:


def gower_mahalanobis_with_grouped_race_small(
    X: pd.DataFrame,
    continuous_cols,
    binary_cols,
    cauc_col="race_caucasian",
    afam_col="race_african_american",
    race_nom_col="race_nominal",
    max_n_for_full=6000,            # safety
    norm_cont="quantile",           # "quantile" or "minmax"
    cont_quantile=0.95,
    dtype=np.float64
):
    n = len(X)
    if n > max_n_for_full:
        raise MemoryError(f"n={n} too large for a full N×N matrix in-memory (limit {max_n_for_full}).")

    X = X.copy()

    # --- build race nominal from 2 dummies ---
    c = X[cauc_col].astype(float)
    a = X[afam_col].astype(float)
    race = pd.Series(index=X.index, dtype=object)
    race[(c == 1) & (a == 0)] = "Caucasian"
    race[(c == 0) & (a == 1)] = "African American"
    race[(c == 0) & (a == 0)] = "Asian"
    X[race_nom_col] = race

    # remove race dummies from binary set
    binary_cols = [b for b in binary_cols if b not in {cauc_col, afam_col}]
    nominal_cols = [race_nom_col]

    D_parts, W_parts = [], []

    # --- Continuous (Mahalanobis with shrinkage) ---
    if len(continuous_cols) > 0:
        Xc = X[continuous_cols].astype(dtype).to_numpy()
        # Simple mean impute
        if np.isnan(Xc).any():
            col_means = np.nanmean(Xc, axis=0)
            idx = np.where(np.isnan(Xc))
            Xc[idx] = np.take(col_means, idx[1])

        # shrinkage covariance for stability
        lw = LedoitWolf().fit(Xc)
        VI = lw.precision_.astype(dtype, copy=False)

        # pairwise Mahalanobis distances
        Dcont = distance.cdist(Xc, Xc, metric='mahalanobis', VI=VI).astype(dtype, copy=False)

        # normalize continuous distances to [0,1] in a robust way
        if norm_cont == "quantile":
            # estimate high quantile to avoid needing full min/max sweep
            # sample pairs to estimate q
            rng = np.random.default_rng(0)
            m = min(200000, n*(n-1)//2)  # up to 200k pairs
            ii = rng.integers(0, n, size=m)
            jj = rng.integers(0, n, size=m)
            mask = ii != jj
            q = np.quantile(Dcont[ii[mask], jj[mask]], cont_quantile)
            scale = max(q, 1e-6)
            Dcont_norm = np.clip(Dcont / scale, 0, 1)
        else:
            dmin = float(Dcont.min())
            dmax = float(Dcont.max())
            Dcont_norm = (Dcont - dmin) / (dmax - dmin + 1e-12)

        D_parts.append(Dcont_norm)
        W_parts.append(len(continuous_cols))

    # --- Binary (Hamming) ---
    if len(binary_cols) > 0:
        Xb = X[binary_cols].copy()
        for ccol in binary_cols:
            mode_val = Xb[ccol].mode(dropna=True)
            fill = mode_val.iloc[0] if not mode_val.empty else 0
            Xb[ccol] = Xb[ccol].fillna(fill)
        Xb = Xb.astype(dtype).to_numpy()
        Dbin = distance.cdist(Xb, Xb, metric='hamming').astype(dtype, copy=False)
        D_parts.append(Dbin)
        W_parts.append(len(binary_cols))

    # --- Nominal (race) ---
    nmat = np.zeros((n, n), dtype=dtype)
    denom = np.zeros((n, n), dtype=dtype)
    col = X[race_nom_col].astype(object).to_numpy()
    valid = ~pd.isna(col)
    same = (col[:, None] == col[None, :])
    both = (valid[:, None] & valid[None, :])
    d = np.where(both, 1.0 - same.astype(dtype), np.nan)
    mask = ~np.isnan(d)
    nmat[mask] += d[mask].astype(dtype, copy=False)
    denom[mask] += 1.0
    Dnom = np.where(denom > 0, nmat / denom, 0.0).astype(dtype, copy=False)
    D_parts.append(Dnom)
    W_parts.append(1)  # counts as ONE variable

    # --- Combine (Gower-style weighted average) ---
    total_w = float(sum(W_parts))
    D = sum(w * Dp for w, Dp in zip(W_parts, D_parts)) / (total_w + 1e-12)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    return pd.DataFrame(D, index=X.index, columns=X.index)


# In[6]:


class Node:
    def __init__(self, id, left=None, right=None, is_leaf=False, leaf_id=None):
        self.id = id
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.leaf_id = leaf_id

def _scipy_to_nodes(scipy_node):
    if scipy_node.is_leaf():
        return Node(scipy_node.id, is_leaf=True, leaf_id=scipy_node.id)
    return Node(
        scipy_node.id,
        left=_scipy_to_nodes(scipy_node.left),
        right=_scipy_to_nodes(scipy_node.right),
        is_leaf=False,
    )

def _collect_leaves(node):
    # returns list of leaf indices under this node
    if node.is_leaf:
        return [node.leaf_id]
    return _collect_leaves(node.left) + _collect_leaves(node.right)

def balance_basis_from_linkage(Z, feature_names):
    """
    Build a (p × (p-1)) balance basis from a SciPy linkage (on features).
    feature_names: list of species names in the same order used to compute Z.
    """
    # SciPy leaves are indexed 0..p-1 in the order provided to linkage
    p = len(feature_names)
    scipy_root = to_tree(Z, rd=False)
    root = _scipy_to_nodes(scipy_root)

    basis_cols = []

    # DFS over internal nodes to build balances; skip leaves
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            continue
        # push children for traversal
        stack.append(node.right)
        stack.append(node.left)

        left_leaves = _collect_leaves(node.left)
        right_leaves = _collect_leaves(node.right)
        r = len(left_leaves)
        s = len(right_leaves)
        if r == 0 or s == 0:
            continue

        v = np.zeros(p, dtype=float)
        v[left_leaves] =  1.0 / r
        v[right_leaves] = -1.0 / s

        # balance scaling
        scale = np.sqrt((r * s) / (r + s))
        v *= scale
        basis_cols.append(v)

    # Stack in the order we visited (gives (p-1) vectors); shape (p, p-1)
    B = np.column_stack(basis_cols)
    # (Optional) make columns exactly orthonormal in the Euclidean sense after log:
    # they already have the correct Aitchison balance scaling; no further orthonormalization needed.
    return B  # shape: (p, p-1)


# In[7]:


def mahalanobis_knn_from_balances(
    X_bal,
    k=15,
    data_type = np.float64,
    random_state=0
):
    """
    Compute Mahalanobis-based KNN from a balances matrix.

    Parameters
    ----------
    X_bal : pd.DataFrame, shape (n_samples, n_features)
        Balances (or other continuous features) with samples as rows.
    k : int
        Number of neighbors (excluding self).

    Returns
    -------
    distances : (n, k) ndarray
        Distances from each sample to its k nearest neighbors.
    indices : (n, k) ndarray
        Indices of the k nearest neighbors for each sample.
    VI : (p, p) ndarray
        Precision matrix estimated by Ledoit-Wolf.
    L : (p, p) ndarray
        Cholesky factor such that VI = L @ L.T (lower-triangular).
    X_whiten : (n, p) ndarray
        Whitened data matrix X_bal @ L.
    """
    # Convert to numpy
    Xc = X_bal.to_numpy(dtype=float)
    n, p = Xc.shape

    # Simple mean imputation if any NaNs
    if np.isnan(Xc).any():
        col_means = np.nanmean(Xc, axis=0)
        idx = np.where(np.isnan(Xc))
        Xc[idx] = np.take(col_means, idx[1])

    # Ledoit-Wolf shrinkage covariance & precision
    lw = LedoitWolf().fit(Xc)
    VI = lw.precision_  # p x p, SPD

    # Cholesky factor: VI = L @ L.T (lower triangular)
    L = cholesky(VI, lower=True)

    # Whiten rows: Mahalanobis(Xc; VI) == Euclidean(Xc @ L)
    X_whiten = Xc @ L   # (n, p)

    # Euclidean KNN in whitened space = Mahalanobis KNN in original space
    nbrs = NearestNeighbors(
        n_neighbors=k+1,    # +1 for self
        metric='euclidean',
        algorithm='auto',
        n_jobs=-1
    ).fit(X_whiten)

    distances, indices = nbrs.kneighbors(X_whiten)
    # First neighbor is self (distance 0); drop it
    distances = distances[:, 1:]
    indices   = indices[:, 1:]

    return distances, indices, VI, L, X_whiten


# In[8]:


def knn_from_precomputed_distance(D, k=15):
    """
    Build KNN indices & distances from a precomputed sample-sample distance matrix.

    Parameters
    ----------
    D : (n, n) array_like
        Precomputed symmetric distance matrix (e.g., mixed Hamming + Mahalanobis).
        Must have D[i, i] = 0.
    k : int
        Number of neighbors to keep for each sample.

    Returns
    -------
    distances : (n, k) ndarray
        Distances from each sample to its k nearest neighbors.
    indices : (n, k) ndarray
        Indices of the k nearest neighbors for each sample.
    """
    D = np.asarray(D)
    n = D.shape[0]
    assert D.shape == (n, n), "D must be square"

    # argsort along each row: smallest to largest
    # first element is i itself (distance 0), so we skip that
    order = np.argsort(D, axis=1)[:, 1:k+1]  # (n, k)
    indices = order
    distances = np.take_along_axis(D, indices, axis=1)

    return distances, indices


# In[9]:


def geodesic_from_knn(
    indices,
    distances,
    symmetrize="or",
    directed=False
):
    """
    Construct geodesic (shortest-path) distances from a KNN graph.

    Parameters
    ----------
    indices : (n, k) int ndarray
        Neighbor indices for each sample.
    distances : (n, k) float ndarray
        Corresponding edge weights.
    symmetrize : {"or", "and"}
        How to symmetrize the directed KNN graph if directed=False:
            "or"  : keep edge if i->j OR j->i exists (union-of-kNN).
            "and" : keep edge only if i->j AND j->i exist (mutual kNN).
    directed : bool
        If True, keep the graph directed.
        If False, symmetrize according to `symmetrize`.

    Returns
    -------
    D_geo : (n, n) ndarray
        Geodesic (shortest-path) distances.
    A : csr_matrix, shape (n, n)
        Sparse adjacency matrix used for the graph.
    """
    indices = np.asarray(indices)
    distances = np.asarray(distances)
    n, k = indices.shape

    # 1) Build directed adjacency matrix A from KNN edges (i -> j)
    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    data = distances.ravel()
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    # 2) Symmetrize if we want an undirected manifold
    if not directed:
        if symmetrize == "or":
            # union of edges: keep min weight where both directions exist,
            # and keep single-direction edges as well
            A_sym_min = A.minimum(A.T)   # edges where both exist
            A_union = A + A.T            # edges where at least one direction exists
            # where A_sym_min is zero but A_union nonzero, use those union weights
            A = A_sym_min + (A_union.multiply(A_sym_min == 0))
        elif symmetrize == "and":
            # mutual kNN: keep only edges present in both directions
            A = A.minimum(A.T)
        else:
            raise ValueError("symmetrize must be 'or' or 'and'")

    # 3) Geodesic distances = all-pairs shortest paths on weighted graph
    D_geo = shortest_path(
        A,
        directed=directed,
        return_predecessors=False,
        unweighted=False
    )

    return D_geo, A


# In[10]:


def check_geodesic_sanity(D_geo, A, verbose=True):
    """
    Basic sanity checks on the geodesic distance matrix and graph.

    Parameters
    ----------
    D_geo : (n, n) ndarray
        Geodesic distance matrix.
    A : csr_matrix
        Adjacency matrix used to construct D_geo.

    Returns
    -------
    stats : dict
        Summary of checks (symmetry, zero_diag, components, inf counts).
    """
    n = D_geo.shape[0]

    # 1) Symmetry check
    sym_ok = np.allclose(D_geo, D_geo.T, atol=1e-8, equal_nan=True)

    # 2) Zero diagonal
    diag = np.diag(D_geo)
    diag_ok = np.allclose(diag, 0.0, atol=1e-8, equal_nan=True)

    # 3) Infinities (unreachable pairs)
    mask_offdiag = ~np.eye(n, dtype=bool)
    num_inf = np.isinf(D_geo[mask_offdiag]).sum()
    total_pairs = mask_offdiag.sum()
    frac_inf = num_inf / max(total_pairs, 1)

    # 4) Connected components (on the undirected version of A)
    n_components, labels = connected_components(A, directed=False)
    comp_sizes = np.bincount(labels)

    if verbose:
        print("=== Geodesic sanity check ===")
        print(f"Symmetric D_geo      : {sym_ok}")
        print(f"Zero diagonal        : {diag_ok}")
        print(f"# of components      : {n_components}")
        print(f"Component sizes      : {comp_sizes}")
        print(f"# of inf distances   : {num_inf} / {total_pairs} "
              f"({frac_inf:.4%} of off-diagonal entries)")
        if n_components > 1:
            print("Warning: graph is disconnected; consider increasing k.")
        if frac_inf > 0:
            print("Warning: some pairs are unreachable (infinite geodesic distance).")

    return {
        "symmetric": sym_ok,
        "zero_diag": diag_ok,
        "n_components": n_components,
        "component_sizes": comp_sizes,
        "num_inf": num_inf,
        "frac_inf": frac_inf,
        "labels": labels,
    }


# In[11]:


def geodesic_to_kernel_mds(D, dtype=np.float64):
    """
    Convert a geodesic distance matrix D (n x n) into a kernel K (n x n)
    via classical MDS: K = -0.5 * J D^2 J, where J is the centering matrix.
    """
    D = np.asarray(D, dtype=dtype)
    n = D.shape[0]
    assert D.shape == (n, n), "D must be square"

    # Replace inf / nan with max finite distance to avoid explosions
    mask_bad = ~np.isfinite(D)
    if mask_bad.any():
        max_finite = np.nanmax(D[~mask_bad])
        D = D.copy()
        D[mask_bad] = max_finite

    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    K = -0.5 * J @ D2 @ J
    return K

def normalize_kernel(K, dtype=np.float64):
    """
    Simple normalization: shift to be nonnegative and scale to [0, 1].
    """
    K = np.asarray(K, dtype=dtype)
    K = K - K.min()
    K = K / (K.max() + 1e-12)
    return K


# In[12]:


def prime_dual_align_from_geodesics(
    D_geo_x,
    D_geo_y,
    dx=None,
    dy=None,
    device=None,
    dtype=torch.float64,    # good default for stability
    epoch_pd=3000,
    rho=0.5,                # base rho; scheduled inside
    epsilon=0.05,           # step/relaxation size
    log_pd=100,
    integration_type="MultiOmics",
    delay=200,
    verbose=True,
    # early stopping params
    use_early_stop=True,
    tol_align=5e-4,
    tol_constr=2e-3,
    tol_F=5e-4,
    min_consecutive=5,
    # NEW: prior regularization on F
    lambda_F=0.0           # e.g. 0.05 or 0.1 to keep F near F0
):
    """
    Prime–Dual + Adam manifold alignment between two views,
    starting from their geodesic distance matrices.

    Parameters
    ----------
    D_geo_x : (m, m) ndarray
        Geodesic distances for view X (e.g., mixed covariates).
    D_geo_y : (n, n) ndarray
        Geodesic distances for view Y (e.g., microbiome balances).
    dx, dy : float, optional
        Scale factors for the two views; if None, both default to 1.0.
        They define the initial alpha = sqrt(dy/dx).
    device : "cpu" or "cuda" or torch.device
    dtype : torch.dtype
        e.g., torch.float32 or torch.float64.
    epoch_pd : int
        Maximum number of iterations.
    rho : float
        Base penalty parameter; the effective rho is scheduled over time.
    epsilon : float
        Relaxation / update step size.
    log_pd : int
        Logging frequency.
    integration_type : str
        If "MultiOmics", alpha is updated by trace ratio.
    delay : int
        Iteration at which to start updating alpha.
    use_early_stop : bool
        Whether to stop when convergence criteria are met.
    tol_align, tol_constr, tol_F : float
        Tolerances for alignment error, constraint residuals, and ΔF.
    min_consecutive : int
        Number of consecutive epochs that must satisfy all criteria
        before stopping early.
    lambda_F : float
        Strength of quadratic regularization toward F0:
        (lambda_F / 2) * ||F - F0||_F^2.
        F0 = I if m == n, uniform otherwise.

    Returns
    -------
    F_np : (m, n) ndarray
        Alignment matrix F.
    """

    # ---------- Device ----------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # ---------- 1) From geodesic distances to normalized kernels ----------
    Kx_np = geodesic_to_kernel_mds(D_geo_x)
    Ky_np = geodesic_to_kernel_mds(D_geo_y)

    Kx_np = normalize_kernel(Kx_np)
    Ky_np = normalize_kernel(Ky_np)

    m = Kx_np.shape[0]
    n = Ky_np.shape[0]
    assert Kx_np.shape == (m, m), "Kx must be square"
    assert Ky_np.shape == (n, n), "Ky must be square"

    Kx = torch.from_numpy(Kx_np).to(device=device, dtype=dtype)
    Ky = torch.from_numpy(Ky_np).to(device=device, dtype=dtype)

    # dx, dy for initial scaling a
    if dx is None:
        dx = 1.0
    if dy is None:
        dy = 1.0

    a = torch.tensor(np.sqrt(dy / dx), dtype=dtype, device=device)

    # ---------- 2) Initialize primal / dual variables ----------
    # Only use Identity if you are SURE the data is paired/sorted
    is_paired = (m == n) and (integration_type == "MultiOmics") # or some user flag

    if is_paired:
        F = torch.eye(m, n, dtype=dtype, device=device)
    else:
        F = torch.ones((m, n), dtype=dtype, device=device) / n

    F0 = F.clone()

    Im = torch.ones((m, 1), dtype=dtype, device=device)
    In = torch.ones((n, 1), dtype=dtype, device=device)
    Lambda = torch.zeros((n, 1), dtype=dtype, device=device)
    Mu = torch.zeros((m, 1), dtype=dtype, device=device)
    S = torch.zeros((n, 1), dtype=dtype, device=device)

    # Adam parameters
    pho1 = 0.9
    pho2 = 0.999
    delta = torch.tensor(1e-8, dtype=dtype, device=device)
    Fst_moment = torch.zeros((m, n), dtype=dtype, device=device)
    Snd_moment = torch.zeros((m, n), dtype=dtype, device=device)

    # for early stopping
    F_prev = None       # will be set after first iteration
    consec = 0          # consecutive epochs meeting stopping criteria

    eps_small = torch.tensor(1e-12, dtype=dtype, device=device)

    # ---------- 3) Prime–dual iterations ----------
    i = 0
    while i < epoch_pd:
        # --- rho schedule: start softer, ramp up to rho ---
        t = i / float(epoch_pd)
        rho_t = rho * (0.1 + 0.9 * t)   # from 0.1*rho to 1.0*rho

        # --- 3.1) Gradient wrt F using OLD F ---
        FKy = torch.mm(F, Ky)

        # residuals for constraints based on OLD F
        r1_old = torch.mm(F, In) - Im         # (m, 1) row-sum residual
        r2_old = torch.mm(F.t(), Im) - In + S # (n, 1) col/slack residual

        # penalty term in gradient using residuals
        constraint_term = rho_t * (
            torch.mm(r1_old, In.t())      # (m, 1) @ (1, n) -> (m, n)
            + torch.mm(Im, r2_old.t())    # (m, 1) @ (1, n) -> (m, n)
        )

        grad = (
            4.0 * torch.mm(FKy, torch.mm(F.t(), FKy))
            - 4.0 * a * torch.mm(Kx, FKy)
            + torch.mm(Mu, In.t())
            + torch.mm(Im, Lambda.t())
            + constraint_term
        )

        # NEW: prior regularization toward F0
        if lambda_F > 0.0:
            grad = grad + lambda_F * (F - F0)

        # --- 3.2) Adam + projection update for F ---
        i += 1
        Fst_moment = pho1 * Fst_moment + (1.0 - pho1) * grad
        Snd_moment = pho2 * Snd_moment + (1.0 - pho2) * (grad * grad)
        hat_Fst_moment = Fst_moment / (1.0 - (pho1 ** i))
        hat_Snd_moment = Snd_moment / (1.0 - (pho2 ** i))
        grad_adam = hat_Fst_moment / (torch.sqrt(hat_Snd_moment) + delta)

        F_tmp = F - grad_adam
        F_tmp = torch.clamp(F_tmp, min=0.0)
        F = (1.0 - epsilon) * F + epsilon * F_tmp

        # --- 3.3) Recompute residuals with NEW F ---
        r1 = torch.mm(F, In) - Im
        r2 = torch.mm(F.t(), Im) - In + S

        # --- 3.4) Update slack S using NEW r2 ---
        grad_s = Lambda + rho_t * r2
        s_tmp = S - grad_s
        s_tmp = torch.clamp(s_tmp, min=0.0)
        S = (1.0 - epsilon) * S + epsilon * s_tmp

        # --- 3.5) Update dual variables using NEW r1, r2 ---
        Mu = Mu + epsilon * r1
        Lambda = Lambda + epsilon * r2

        # --- 3.6) Optional: update scaling factor a ---
        if integration_type == "MultiOmics" and i >= delay:
            num = torch.trace(torch.mm(Kx, torch.mm(torch.mm(F, Ky), F.t())))
            den = torch.trace(torch.mm(Kx, Kx)) + eps_small
            a = num / den

        # --- 3.7) Diagnostics (alignment, constraints, F-change) ---
        approx = torch.mm(torch.mm(F, Ky), F.t())
        align_err = torch.norm(a * Kx - approx) / (torch.norm(a * Kx) + eps_small)

        row_res = torch.norm(r1) / (m ** 0.5)
        col_res = torch.norm(r2) / (n ** 0.5)

        if F_prev is None:
            delta_F = torch.tensor(float("inf"), dtype=dtype, device=device)
        else:
            delta_F = torch.norm(F - F_prev) / (torch.norm(F_prev) + eps_small)

        # SAFE CLONE: F_prev does NOT change when F changes later
        F_prev = F.detach().clone()

        # --- 3.8) Logging ---
        if verbose and (i % log_pd == 0 or i == epoch_pd):
            print(
                f"epoch:[{i}/{epoch_pd}] "
                f"err:{align_err.item():.4f} "
                f"alpha:{float(a):.4f} "
                f"row_res:{row_res.item():.4e} "
                f"col_res:{col_res.item():.4e} "
                f"dF:{delta_F.item():.4e} "
                f"rho_t:{rho_t:.3f}"
            )

        # --- 3.9) Early stopping with consecutive epochs ---
        if use_early_stop:
            criteria_met = (
                (align_err < tol_align) and
                (row_res   < tol_constr) and
                (col_res   < tol_constr) and
                (delta_F   < tol_F)
            )

            if criteria_met:
                consec += 1
            else:
                consec = 0

            if consec >= min_consecutive:
                if verbose:
                    print(
                        f"Converged at epoch {i} "
                        f"(criteria met for {consec} consecutive epochs)."
                    )
                break

    F_np = F.detach()
    return F_np


# In[13]:


def get_manifold_alignment(loaded_data):
    # Mixed distance for clinical variables
    bin_vars = [c for c in loaded_data.meta_df.columns if loaded_data.meta_df[c].nunique() == 2]
    cont_vars = [c for c in loaded_data.meta_df.columns if c not in bin_vars]
    D_covar = gower_mahalanobis_with_grouped_race_small(
                loaded_data.meta_df,
                continuous_cols=cont_vars,
                binary_cols=bin_vars,
                cauc_col="race_cauc",
                afam_col="race_afri",
                race_nom_col="race_nominal"
               )

    # Balance construction from microbiome relative abundance
    if not loaded_data.normalized:
        raise ValueError("Dataset needs to be normalized")
    if not loaded_data.clr_transformed:
        raise ValueError("Dataset needs to be clr-transformed")
    Z = linkage(loaded_data.X.T, method="average", metric="euclidean")
    root_expl = to_tree(Z, rd=False)
    B_basis = balance_basis_from_linkage(Z, loaded_data.features)
    balances = loaded_data.X @ B_basis
    balances = pd.DataFrame(balances, index = loaded_data.X_df.index,
                            columns=[f"bal_{i+1}" for i in range(B_basis.shape[1])])
    distance_covar, indices_covar = knn_from_precomputed_distance(D_covar)
    distances_bal, indices_bal, VI_bal, L_bal, X_whiten_bal = mahalanobis_knn_from_balances(
                                                                balances,
                                                                k=15  # tune this
                                                              )

    # Geodesic distance matrix from KNN
    D_geo_covar, A_covar = geodesic_from_knn(
                             indices_covar,
                             distance_covar,
                             symmetrize="or",   # or "and" for mutual kNN
                             directed=False
                            )
    D_geo_bal, A_bal = geodesic_from_knn(
                         indices_bal,
                         distances_bal,
                         symmetrize="or",   # or "and" for mutual kNN
                         directed=False
                        )

    # Alignment matrix
    F = prime_dual_align_from_geodesics(
          D_geo_x=D_geo_covar,
          D_geo_y=D_geo_bal,
          dtype=torch.float64,
          epoch_pd=3000,
          rho=0.5,
          epsilon=0.05,
          log_pd=200,
          integration_type="MultiOmics",
          delay=300,
          verbose=True,
          use_early_stop=True,
          tol_align=5e-4,
          tol_constr=2e-3,
          tol_F=5e-4,
          min_consecutive=5,
          lambda_F=0.1,   # try 0.05–0.2 and see how peaked F becomes
        )

    return F


# In[14]:


def mse_loss(pred_x, x, get_sqrt = False):
    batch_size = x.size(0)
    assert batch_size != 0
    mse_loss_val = F.mse_loss(pred_x, x, reduction='sum').div(batch_size)
    if get_sqrt:
        mse_loss_val = torch.sqrt(mse_loss_val + 1e-6)

    return mse_loss_val

def get_r2(x, pred_x):
    r, _ = pearsonr(x, pred_x)
    r2 = r**2

    return r2


# In[15]:


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
                l0_reg = prob_density.sum() / self.out_features
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

class LassoNetDecoder(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features=1):
        super().__init__()

        # Path 1: The Skip Layer (The "Linear" part)
        # We need bias=True to capture the baseline intercept
        self.skip = nn.Linear(in_features, out_features, bias=True)

        # Path 2: The Non-Linear Backbone
        # We separate the first layer (W0) because we need its weights
        # for the specific LASSONet hierarchical penalty.
        self.W0 = nn.Linear(in_features, hidden_dim)

        self.backbone = nn.Sequential(
            self.W0,
            nn.LeakyReLU(negative_slope=0.01), # or LeakyReLU/Softplus
            #nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        # LASSONet Output = Linear(x) + NonLinear(x)
        return self.skip(x) + self.backbone(x)

class TaxoMA(nn.Module):
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
                 dc_h_dim=None,
                 sum_alignment=True,
                 vectorize_delta=False,
                 lambda_delta=1e-3,
                 apply_lasso=False,
                 apply_lassonet=True):
        super(TaxoMA, self).__init__()

        # clinical encoder
        self.vcovar_encoder = nn.Sequential(
            nn.Linear(input_vcovar_n1, fuse_level_dim),
            #nn.BatchNorm1d(fuse_level_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # metagenome encoder
        self.mra_encoder = MIOSTONEModel(tree,
                                         fuse_level_dim,
                                         node_min_dim,
                                         node_dim_func,
                                         node_dim_func_param,
                                         node_gate_type,
                                         node_gate_param,
                                         prune_mode)

        self.logit_PF = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.sum_alignment = sum_alignment
        self.vectorize_delta = vectorize_delta
        self.g = fuse_level_dim
        self.log_delta = None
        self.delta_gate_X = None
        self.delta_gate_Y = None
        self.lambda_delta = lambda_delta
        self.apply_lasso = apply_lasso
        self.apply_lassonet = apply_lassonet
        if self.sum_alignment:
            if self.vectorize_delta:
                self.delta_gate_X = nn.Linear(2 * fuse_level_dim, fuse_level_dim)
                self.delta_gate_Y = nn.Linear(2 * fuse_level_dim, fuse_level_dim)
            else:
                self.log_delta = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.pred_hidden_dim = None
        if not self.sum_alignment:
            self.pred_hidden_dim = 4 * fuse_level_dim
        else:
            self.pred_hidden_dim = 2 * fuse_level_dim
        ## BMD prediction heads
        ## htot_bmd prediction
        #self.decoder = nn.Sequential(
        #    nn.BatchNorm1d(self.pred_hidden_dim),
        #    nn.Linear(self.pred_hidden_dim, dc_h_dim),
        #    #nn.BatchNorm1d(dc_h_dim),
        #    nn.LeakyReLU(negative_slope=0.01),
        #    nn.Linear(dc_h_dim, 1)
        #)
        self.decnorm = nn.BatchNorm1d(self.pred_hidden_dim)
        self.predlayer = nn.Linear(self.pred_hidden_dim, 1, bias=True)
        if self.apply_lassonet:
            self.decoder = LassoNetDecoder(
                in_features=self.pred_hidden_dim,
                hidden_dim=dc_h_dim,
                out_features=1
            )
        else:
            self.decoder = nn.Sequential(
                #self.decnorm,
                #nn.Dropout(p=0.2),
                self.predlayer
            )

    def aggregate_latent_spaces(self, X, Y, F_corr):
        # Messages
        FY = F_corr @ Y        # (N, g): Y -> X
        FX = F_corr.t() @ X    # (N, g): X -> Y

        if not self.sum_alignment:
            return FY, FX
        else:
            N = X.shape[0]
            device = X.device
            dtype  = X.dtype
            ones = torch.ones(N, 1, device=device, dtype=dtype)
            eps = torch.tensor(1e-12, device=device, dtype=dtype)
            # Row / col sums
            sX = F_corr @ ones         # (N, 1)
            sY = F_corr.t() @ ones     # (N, 1)
            if not self.vectorize_delta:
                # --- M^X (aggregated X) ---
                delta = torch.exp(self.log_delta)
                # numerator: X + δ F Y
                num_X = X + delta * FY  # (n_x, g)
                # denominator per sample: 1 + δ Σ_j F[i,j]
                denom_X = 1.0 + delta * sX  # (n_x, 1)
                Mx = num_X / (denom_X + eps)  # broadcast row-wise

                # --- M^Y (aggregated Y) ---
                delta_inv = 1.0 / (delta + eps)
                # numerator: Y + δ^-1 F^T X
                num_Y = Y + delta_inv * FX  # (n_y, g)
                # denominator per sample: 1 + δ^-1 Σ_i F[i,j]
                denom_Y = 1.0 + delta_inv * sY  # (n_y, 1)
                My = num_Y / (denom_Y + eps)

                return Mx, My
            else:
                # ----- δ_X gating -----
                gate_input_X = torch.cat([X, FY], dim=1)        # (N, 2g)
                delta_logits_X = self.delta_gate_X(gate_input_X)  # (N, g)
                delta_X = F.softplus(delta_logits_X) + eps      # (N, g), >0

                denom_X = 1.0 + delta_X * sX     # broadcast sX: (N, 1) -> (N, g)
                num_X   = X + delta_X * FY
                Mx = num_X / (denom_X + eps)

                # ----- δ_Y gating -----
                gate_input_Y = torch.cat([Y, FX], dim=1)        # (N, 2g)
                delta_logits_Y = self.delta_gate_Y(gate_input_Y)
                delta_Y = F.softplus(delta_logits_Y) + eps    # (N, g)

                denom_Y = 1.0 + delta_Y * sY
                num_Y   = Y + delta_Y * FX
                My = num_Y / (denom_Y + eps)

                if self.lambda_delta is not None:
                    # ----- Regularization on δ -----
                    # log δ near 0  => δ near 1 (neutral)
                    log_delta_X = torch.log(delta_X)
                    log_delta_Y = torch.log(delta_Y)

                    reg_X = (log_delta_X ** 2).mean()
                    reg_Y = (log_delta_Y ** 2).mean()
                    reg_delta = self.lambda_delta * (reg_X + reg_Y)

                    return Mx, My, reg_delta
                else:
                    return Mx, My

    def forward(self, clinical_data, mgs_data, F_raw):
        vcovar_code = self.vcovar_encoder(clinical_data)
        mgs_code = self.mra_encoder(mgs_data)

        N = clinical_data.shape[0]
        F_prior = torch.eye(N, device=clinical_data.device, dtype=clinical_data.dtype)

        eps = 1e-12
        F = F_raw.to(device=vcovar_code.device, dtype=vcovar_code.dtype)
        F.requires_grad_(False)
        # Row-normalize F: each row sums to ~1
        row_sums = F.sum(dim=1, keepdim=True)            # (N, 1)
        P_data = F / (row_sums + eps)                     # (N, N)
        ## Column-normalize F: each column sums to ~1
        #col_sums = F.sum(dim=0, keepdim=True)            # (1, N)
        #P_data_col = F / (col_sums + eps)                     # (N, N)

        alpha = torch.sigmoid(self.logit_PF)  # in (0,1)
        corr = alpha * P_data + (1.0 - alpha) * F_prior

        reg_delta = None
        if not self.sum_alignment:
            mgs_aligned_on_clinical, vcovar_aligned_on_mgs = self.aggregate_latent_spaces(vcovar_code, mgs_code, corr)
            union_code = torch.cat(
                [vcovar_code, mgs_aligned_on_clinical, mgs_code, vcovar_aligned_on_mgs],
                dim=1
            )
        else:
            if self.lambda_delta is not None:
                vcovar_aggregated, mgs_aggregated, reg_delta = self.aggregate_latent_spaces(vcovar_code, mgs_code, corr)
            else:
                vcovar_aggregated, mgs_aggregated = self.aggregate_latent_spaces(vcovar_code, mgs_code, corr)
            union_code = torch.cat(
                [vcovar_aggregated, mgs_aggregated],
                dim=1
            )

        pred_bmd = self.decoder(self.decnorm(union_code))

        # --- RETURN ---
        # We return a dictionary or tuple containing the weights needed for the loss
        results = {
            "pred": pred_bmd,
            "reg_l0": self.mra_encoder.get_total_l0_reg()
        }

        if reg_delta is not None:
            results["reg_delta"] = reg_delta

        if self.apply_lasso:
            results["lasso_weight"] = self.predlayer.weight

        if self.apply_lassonet:
            results["lasso_skip_weight"] = self.decoder.skip.weight
            results["lasso_nonlin_weight"] = self.decoder.W0.weight

        return results


# In[26]:


def load_data_and_compute_alignment_trva(tree_path, master_path, div, bmd_site,
                                         use_mask = False, mask_name = None):
    if use_mask:
        miostone_tree = MIOSTONETree.init_from_nwk(tree_path + 'taxa_tree_' + mask_name + '.nwk')
    else:
        miostone_tree = MIOSTONETree.init_from_nwk(tree_path + 'taxa_tree.nwk')
    miostone_tree.compute_depths()
    miostone_tree.compute_indices()

    loaded_data_train = MIOSTONEDataset.init_from_files(master_path + div + '/',
                                                        'tr_' + div,
                                                        bmd_site,
                                                        use_mask = use_mask,
                                                        mask_path = tree_path,
                                                        mask_name = mask_name)
    loaded_data_train.normalize()
    loaded_data_train.clr_transform()
    loaded_data_train.order_features_by_tree(miostone_tree)
    F_train = get_manifold_alignment(loaded_data_train)

    loaded_data_valid = MIOSTONEDataset.init_from_files(master_path + div + '/',
                                                        'val_' + div,
                                                        bmd_site,
                                                        use_mask = use_mask,
                                                        mask_path = tree_path,
                                                        mask_name = mask_name)
    loaded_data_valid.normalize()
    loaded_data_valid.clr_transform()
    loaded_data_valid.order_features_by_tree(miostone_tree)
    F_valid = get_manifold_alignment(loaded_data_valid)

    return(miostone_tree, loaded_data_train, loaded_data_valid, F_train, F_valid)

class Objective:
    def __init__(self,
                 miostone_tree,
                 loaded_data_train, loaded_data_valid,
                 F_train, F_valid,
                 sum_alignment=True,
                 vectorize_delta=False,
                 delta_reg=False,
                 apply_lasso=False,
                 apply_lassonet=True,
                 dtype = torch.float64):
        self.miostone_tree = miostone_tree
        self.loaded_data_train = loaded_data_train
        self.loaded_data_valid = loaded_data_valid
        self.F_train = F_train
        self.F_valid = F_valid
        self.sum_alignment = sum_alignment
        self.vectorize_delta = vectorize_delta
        self.delta_reg = delta_reg
        self.apply_lasso = apply_lasso
        self.apply_lassonet = apply_lassonet
        self.dtype = dtype

    def __call__(self, trial):
        # Hyperparameter suggestions
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 5e-2, log=True)
        l2 = trial.suggest_float('l2', 5e-3, 1e-1, log=True)
        epoch_num = 800
        #epoch_num = trial.suggest_int('epoch_num', 80, 200, step = 20)
        lambda_l0 = trial.suggest_float('lambda_l0', 5e-3, 1e-2, log=True)
        lambda_delta = None
        if self.delta_reg:
            lambda_delta = trial.suggest_float('lambda_delta', 1e-6, 1e-4, log=True)
        lambda_lasso = None
        M = None
        if self.apply_lasso:
            lambda_lasso = trial.suggest_float('lambda_lasso', 1e-4, 1e-2, log=True)
        if self.apply_lassonet:
            lambda_lasso = trial.suggest_float('lambda_lasso', 1e-4, 1e-2, log=True)
            M = trial.suggest_float('M', 10., 20., log=True)

        input_vcovar_n1 = self.loaded_data_train.meta.shape[1]
        # Model, loss function, optimization
        model = TaxoMA(tree = miostone_tree,
                       node_min_dim = 1,
                       node_dim_func = 'linear',
                       node_dim_func_param = 0.6,
                       node_gate_type = 'concrete',
                       node_gate_param = 0.3,
                       prune_mode = 'taxonomy',
                       input_vcovar_n1 = input_vcovar_n1,
                       #fuse_level_dim = trial.suggest_int('fuse_level_dim', 4, 8, step = 2),
                       fuse_level_dim = trial.suggest_categorical('fuse_level_dim', [4]),
                       dc_h_dim = trial.suggest_int('dc_h_dim', 2, 4, step = 2) if self.apply_lassonet else None,
                       sum_alignment = self.sum_alignment,
                       vectorize_delta = self.vectorize_delta,
                       lambda_delta = lambda_delta,
                       apply_lasso = self.apply_lasso,
                       apply_lassonet = self.apply_lassonet)
        model = model.to(dtype=torch.float64, device=DEVICE)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = l2)
        #scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',       # We want to minimize loss
            factor=0.5,       # Reduce LR by half when plateauing
            patience=15,      # Wait 15 epochs before reducing (prevents knee-jerk reactions)
            verbose=True,     # Print a message when LR is updated
            min_lr=1e-6       # Lower bound to prevent LR from becoming 0
        )

        if not self.loaded_data_train.data_adapted:
            self.loaded_data_train.data_adaptation(self.dtype)
        if not self.loaded_data_valid.data_adapted:
            self.loaded_data_valid.data_adaptation(self.dtype)
        early_stopping = EarlyStopping(patience = 60)

        # Warmup configuration
        warmup_epochs = 20 # Don't apply full sparsity immediately

        for epoch in range(1, epoch_num + 1):
            model.train()
            optimizer.zero_grad()

            # --- CALCULATE WARMUP FACTOR (0.0 -> 1.0) ---
            if epoch < warmup_epochs:
                warmup_factor = epoch / warmup_epochs
            else:
                warmup_factor = 1.0

            outputs = model(self.loaded_data_train.meta, self.loaded_data_train.X, self.F_train)
            pred_loss = mse_loss(outputs["pred"], self.loaded_data_train.y[0])
            scaled_pred_loss = pred_loss * 1.0

            total_loss = scaled_pred_loss # Start with just prediction loss

            # Add delta reg
            if self.delta_reg:
                total_loss += lambda_delta * outputs["reg_delta"]
            
            # 2. Add L0 Reg (Structural Sparsity) WITH WARMUP
            # This solves the "fighting" issue by keeping it zero at the start
            if "reg_l0" in outputs:
                total_loss += (lambda_l0 * warmup_factor) * outputs["reg_l0"]
            
            # 3. Add Lasso / LassoNet terms WITH WARMUP
            #current_lambda_lasso = lambda_lasso * warmup_factor
            
            # Add Lasso / LassoNet terms with WARMED UP lambda
            if self.apply_lasso:
                current_lambda_lasso = lambda_lasso * warmup_factor
                lasso_w = outputs["lasso_weight"]
                l1_loss = torch.norm(lasso_w, p=1) / model.pred_hidden_dim
                total_loss += current_lambda_lasso * l1_loss

            elif self.apply_lassonet:
                current_lambda_lasso = lambda_lasso * warmup_factor
                skip_w = outputs["lasso_skip_weight"]
                nonlin_w = outputs["lasso_nonlin_weight"]

                l1_loss = torch.norm(skip_w, p=1) / model.pred_hidden_dim

                nonlin_norm = torch.norm(nonlin_w, p=2, dim=0)
                skip_abs = torch.abs(skip_w).squeeze()
                hierarchy_loss = torch.sum(torch.relu(nonlin_norm - M * skip_abs)) / model.pred_hidden_dim

                # Apply current_lambda_lasso here
                total_loss += current_lambda_lasso * l1_loss + hierarchy_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss_train = total_loss.item()

            #scheduler.step()

            model.eval()
            with torch.no_grad():
                outputs_valid = model(self.loaded_data_valid.meta, self.loaded_data_valid.X, self.F_valid)
                pred_loss_valid = mse_loss(outputs_valid["pred"], self.loaded_data_valid.y[0])
            if epoch > warmup_epochs:
                scheduler.step(pred_loss_valid)
            if epoch >= 100:
                early_stopping(pred_loss_valid, model)
            if early_stopping.early_stop:
                print(f'Early stopping, number of epochs: [{epoch}/{epoch_num}]')
                break
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{epoch_num}], Overall Training Loss: {total_loss_train:.4f}, Prediction Training Loss: {torch.sqrt(pred_loss).item():.4f}, Prediction Validation Loss: {torch.sqrt(pred_loss_valid).item():.4f}')

        return pred_loss_valid
        #return early_stopping.min_loss.item()


# In[27]:


def load_data_and_compute_alignment_tute(tree_path, master_path, bmd_site,
                                         use_mask = False, mask_name = None):
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
    F_tune = get_manifold_alignment(loaded_data_tune)

    loaded_data_test = MIOSTONEDataset.init_from_files(master_path + 'train_test_split/',
                                                       'te',
                                                       bmd_site,
                                                       use_mask = use_mask,
                                                       mask_path = tree_path,
                                                       mask_name = mask_name)
    loaded_data_test.normalize()
    loaded_data_test.clr_transform()
    loaded_data_test.order_features_by_tree(miostone_tree)
    F_test = get_manifold_alignment(loaded_data_test)

    return(miostone_tree, loaded_data_tune, loaded_data_test, F_tune, F_test)

def test_go(miostone_tree, loaded_data_tune, loaded_data_test, F_tune, F_test, bmd_site,
            div, best_params_dict, init_model_params_dict, model_path,
            sum_alignment=True, vectorize_delta=False, delta_reg=False, apply_lasso=False, apply_lassonet=True,
            dtype = torch.float64, save_pred_results = True):
    summarized_results_dict = {}
    summarized_results_dict.update(best_params_dict)

    tune_num_subject = loaded_data_tune.X.shape[0]
    test_num_subject = loaded_data_test.X.shape[0]
    input_vcovar_n1 = loaded_data_tune.meta.shape[1]

    #epoch_num = best_params_dict['epoch_num']
    epoch_num = 800
    learning_rate = best_params_dict['learning_rate']
    l2 = best_params_dict['l2']
    lambda_l0 = best_params_dict['lambda_l0']
    lambda_delta = None
    if delta_reg:
        lambda_delta = best_params_dict['lambda_delta']
    lambda_lasso = None
    M = None
    if apply_lasso:
        lambda_lasso = best_params_dict['lambda_lasso']
    if apply_lassonet:
        lambda_lasso = best_params_dict['lambda_lasso']
        M = best_params_dict['M']
    init_model_params = {key: best_params_dict[key] for key in init_model_params_dict if key in best_params_dict}

    model = TaxoMA(tree = miostone_tree,
                   node_min_dim = 1,
                   node_dim_func = 'linear',
                   node_dim_func_param = 0.6,
                   node_gate_type = 'concrete',
                   node_gate_param = 0.3,
                   prune_mode = 'taxonomy',
                   input_vcovar_n1 = input_vcovar_n1,
                   sum_alignment = sum_alignment,
                   vectorize_delta = vectorize_delta,
                   lambda_delta = lambda_delta,
                   apply_lasso = apply_lasso,
                   apply_lassonet = apply_lassonet,
                   **init_model_params)

    os.makedirs(model_path, exist_ok=True)
    model = model.to(dtype=torch.float64, device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = l2)
    #scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       # We want to minimize loss
        factor=0.5,       # Reduce LR by half when plateauing
        patience=15,      # Wait 10 epochs before reducing (prevents knee-jerk reactions)
        verbose=True,     # Print a message when LR is updated
        min_lr=1e-6       # Lower bound to prevent LR from becoming 0
    )

    if not loaded_data_tune.data_adapted:
        loaded_data_tune.data_adaptation(dtype)
    if not loaded_data_test.data_adapted:
        loaded_data_test.data_adaptation(dtype)
    early_stopping = EarlyStopping(patience = 60)

    # Warmup configuration
    warmup_epochs = 20

    for epoch in range(1, epoch_num + 1):
        model.train()
        optimizer.zero_grad()

        # --- CALCULATE WARMUP FACTOR (0.0 -> 1.0) ---
        if epoch < warmup_epochs:
            warmup_factor = epoch / warmup_epochs
        else:
            warmup_factor = 1.0

        outputs = model(loaded_data_tune.meta, loaded_data_tune.X, F_tune)
        pred_loss = mse_loss(outputs["pred"], loaded_data_tune.y[0])
        scaled_pred_loss = pred_loss * 1.0

        total_loss = scaled_pred_loss # Start with just prediction loss

        # Add delta reg
        if delta_reg:
            total_loss += lambda_delta * outputs["reg_delta"]
        
        # 2. Add L0 Reg (Structural Sparsity) WITH WARMUP
        # This solves the "fighting" issue by keeping it zero at the start
        if "reg_l0" in outputs:
            total_loss += (lambda_l0 * warmup_factor) * outputs["reg_l0"]
            
        # 3. Add Lasso / LassoNet terms WITH WARMUP
        #current_lambda_lasso = lambda_lasso * warmup_factor

        # Add Lasso / LassoNet terms with WARMED UP lambda
        if apply_lasso:
            current_lambda_lasso = lambda_lasso * warmup_factor
            lasso_w = outputs["lasso_weight"]
            l1_loss = torch.norm(lasso_w, p=1) / model.pred_hidden_dim
            total_loss += current_lambda_lasso * l1_loss

        elif apply_lassonet:
            current_lambda_lasso = lambda_lasso * warmup_factor
            skip_w = outputs["lasso_skip_weight"]
            nonlin_w = outputs["lasso_nonlin_weight"]

            l1_loss = torch.norm(skip_w, p=1) / model.pred_hidden_dim

            nonlin_norm = torch.norm(nonlin_w, p=2, dim=0)
            skip_abs = torch.abs(skip_w).squeeze()
            hierarchy_loss = torch.sum(torch.relu(nonlin_norm - M * skip_abs)) / model.pred_hidden_dim

            # Apply current_lambda_lasso here
            total_loss += current_lambda_lasso * l1_loss + hierarchy_loss

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss_tune = total_loss.item()

        #scheduler.step()

        model.eval()
        with torch.no_grad():
            outputs_test = model(loaded_data_test.meta, loaded_data_test.X, F_test)
            pred_loss_test = mse_loss(outputs_test["pred"], loaded_data_test.y[0])
        if epoch > warmup_epochs:
            scheduler.step(pred_loss_test)
        if epoch >= 100:
            early_stopping(pred_loss_test, model)
        if early_stopping.early_stop:
            print(f'Early stopping, number of epochs: [{epoch}/{epoch_num}]')
            break
        if epoch % 100 == 0:
            print(f'Testing Stage Epoch [{epoch}/{epoch_num}], Overall Training Loss: {total_loss_tune:.4f}, Prediction Training Loss: {torch.sqrt(pred_loss).item():.4f}, Prediction Testing Loss: {torch.sqrt(pred_loss_test).item():.4f}')

    torch.save(model.state_dict(), model_path + div + '_' + bmd_site + '_optparam_taxoma_testing.pt')

    tune_bmd_dict = {bmd_site: np.array(loaded_data_tune.y[0].detach().cpu().numpy()).reshape(tune_num_subject)}
    tune_pred_dict = {'subject_id': pd.DataFrame.to_numpy(loaded_data_tune.subject_id).reshape(tune_num_subject),
                      'pred_' + bmd_site: np.array(outputs["pred"].detach().cpu().numpy()).reshape(tune_num_subject)}
    tune_pred_cache = pd.DataFrame.from_dict(tune_pred_dict)

    rmse_r2_dict = {}
    tune_rmse_dict = {'Tuning RMSE': np.array(torch.sqrt(pred_loss).detach().cpu().numpy())}
    tune_r2_dict = {'Tuning R2': get_r2(tune_bmd_dict.get(bmd_site), tune_pred_dict.get('pred_' + bmd_site))}

    rmse_r2_dict.update(tune_rmse_dict)
    rmse_r2_dict.update(tune_r2_dict)

    test_bmd_dict = {bmd_site: np.array(loaded_data_test.y[0].detach().cpu().numpy()).reshape(test_num_subject)}
    test_pred_dict = {'subject_id': pd.DataFrame.to_numpy(loaded_data_test.subject_id).reshape(test_num_subject),
                      'pred_' + bmd_site: np.array(outputs_test["pred"].detach().cpu().numpy()).reshape(test_num_subject)}
    test_pred_cache = pd.DataFrame.from_dict(test_pred_dict)

    test_rmse_dict = {'Testing RMSE': np.array(torch.sqrt(pred_loss_test).detach().cpu().numpy())}
    test_r2_dict = {'Testing R2': get_r2(test_bmd_dict.get(bmd_site), test_pred_dict.get('pred_' + bmd_site))}

    rmse_r2_dict.update(test_rmse_dict)
    rmse_r2_dict.update(test_r2_dict)

    if save_pred_results:
        pred_save_path = master_path + div + '/prediction_results/'
        os.makedirs(pred_save_path, exist_ok=True)
        tune_pred_cache.to_csv(pred_save_path + div + '_' + bmd_site + '_tune_set_pred_TaxoMA_results_2.csv', index=False)
        test_pred_cache.to_csv(pred_save_path + div + '_' + bmd_site + '_test_set_pred_TaxoMA_results_2.csv', index=False)

    summarized_results_dict.update(rmse_r2_dict)

    return summarized_results_dict


# In[28]:


# Profile codes:
#1 - ali-T,vec-F,delreg-F,lasso-F,lassonet-T;
#2 - ali-T,vec-F,delreg-F,lasso-F,lassonet-F;
#3 - ali-F,vec-F,delreg-F,lasso-F,lassonet-T;
#4 - ali-F,vec-F,delreg-F,lasso-F,lassonet-F;
#5 - ali-T,vec-T,delreg-T,lasso-F,lassonet-F;
#6 - ali-T,vec-T,delreg-T,lasso-F,lassonet-T;
#7 - ali-T,vec-T,delreg-T,lasso-T,lassonet-F;
#8 - ali-T,vec-T,delreg-F,lasso-F,lassonet-F

root_path = 'root_path/'
master_path = root_path + 'data_folder/'
tree_path = master_path + 'tree_folder/'
div_list = np.char.add('tune_', np.array(list(range(1, 11, 1))).astype('str')).tolist()
model_path = root_path + 'saved_models_taxoma_2/'
os.makedirs(model_path, exist_ok=True)

sum_alignment=True
vectorize_delta=False
delta_reg=False
apply_lasso=False
apply_lassonet=False
if not sum_alignment:
    assert vectorize_delta == False, "No delta vectorization needed when no sum alignment."
    assert delta_reg == False, "No delta regularization needed when no sum alignment."
if not vectorize_delta:
    assert delta_reg == False, "No delta regularization needed when no vectorization."
if apply_lasso:
    assert apply_lassonet == False, "Already have Lasso, no LassoNet needed."
if apply_lassonet:
    assert apply_lasso == False, "Already have LassoNet, no Lasso needed."

if apply_lassonet:
    init_model_params_dict = {'fuse_level_dim', 'dc_h_dim'}
else:
    init_model_params_dict = {'fuse_level_dim'}

summarized_results_path = root_path + 'summarized_results_taxoma_2/'
os.makedirs(summarized_results_path, exist_ok=True)
bmd_site = ['bmd_site'] #NECK_BMD, HTOT_BMD, spine_total_bmd, R_13_BMD
use_mask = True
mask_name = 'mask_name'


# In[29]:


div_track = []
summarized_results_cache = []
miostone_tree, loaded_data_tune, loaded_data_test, F_tune, F_test = load_data_and_compute_alignment_tute(tree_path,
                                                                                                         master_path,
                                                                                                         bmd_site,
                                                                                                         use_mask,
                                                                                                         mask_name)
for div in div_list:
    print(f'Running on {div}')
    pruner = optuna.pruners.HyperbandPruner(min_resource = 50)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    _, loaded_data_train, loaded_data_valid, F_train, F_valid = load_data_and_compute_alignment_trva(tree_path,
                                                                                                     master_path,
                                                                                                     div,
                                                                                                     bmd_site,
                                                                                                     use_mask,
                                                                                                     mask_name)
    objective = Objective(miostone_tree, loaded_data_train, loaded_data_valid, F_train, F_valid,
                          sum_alignment = sum_alignment,
                          vectorize_delta = vectorize_delta,
                          delta_reg = delta_reg,
                          apply_lasso = apply_lasso,
                          apply_lassonet = apply_lassonet)
    study.optimize(objective, n_trials=100)

    best_params_dict = study.best_trial.params

    summarized_results_dict = test_go(miostone_tree, loaded_data_tune, loaded_data_test, F_tune, F_test, bmd_site[0],
                                      div, best_params_dict, init_model_params_dict, model_path,
                                      sum_alignment = sum_alignment,
                                      vectorize_delta = vectorize_delta,
                                      delta_reg = delta_reg,
                                      apply_lasso = apply_lasso,
                                      apply_lassonet = apply_lassonet)
    summarized_results_cache.append(summarized_results_dict)
    div_track.append(div)

div_track_dic = {'division': div_track}
div_tract_cache = pd.DataFrame(data = div_track_dic)

summarized_results_cache = pd.DataFrame.from_dict(summarized_results_cache)

summarized_results_cache = pd.concat([div_tract_cache, summarized_results_cache], axis = 1)
summarized_results_cache.to_csv(summarized_results_path + bmd_site[0] + '_taxoma_summarized_results_2.csv', index=False)


# In[ ]:




