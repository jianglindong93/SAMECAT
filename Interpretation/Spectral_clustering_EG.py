#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ====== INITIAL SETTINGS ======
EG_CSV_PATH = 'EG_results_path/EG_attr_mgs_test_df.csv'
OUT_DIR = 'output_results_path/'

# Sweep grid (start small, then expand)
K_NN_LIST     = [15, 20, 30, 40]
EDGE_THR_LIST = [0.05, 0.10, 0.15]
K_MOD_LIST    = [5, 6, 7, 8, 9, 10, 11, 12]

N_BOOT = 100
SEEDS  = [0]
MERGE_CORR = 0.90

USE_ABS_SIMILARITY = False  # True = |cosine|; False = signed cosine
USE_ABS_MERGE = False       # True = merge by |corr|; False = merge only by +corr

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
print("Output folder:", OUT_DIR)


# In[8]:


def signed_log_transform(E_raw: np.ndarray):
    abs_vals = np.abs(E_raw)
    nonzero = abs_vals[abs_vals > 0]
    eps = float(np.median(nonzero)) if nonzero.size else 1e-12
    E_t = np.sign(E_raw) * np.log1p(abs_vals / eps)
    return E_t, eps

def build_affinity_cosine_knn(X: np.ndarray, k_nn: int, edge_thr: float, use_abs: bool = True):
    Xn = normalize(X, axis=0)
    S = Xn.T @ Xn
    W = np.abs(S) if use_abs else S.copy()
    np.fill_diagonal(W, 1.0)

    n_feats = W.shape[0]
    k_eff = min(k_nn + 1, n_feats)

    top_idx = np.argpartition(W, -k_eff, axis=1)[:, -k_eff:]
    row_ids = np.repeat(np.arange(n_feats), k_eff)
    col_ids = top_idx.reshape(-1)
    w = W[np.repeat(np.arange(n_feats), k_eff), col_ids]

    not_self = row_ids != col_ids
    keep = not_self & (w > edge_thr)

    A = csr_matrix((w[keep], (row_ids[keep], col_ids[keep])), shape=(n_feats, n_feats))
    A = (A + A.T) * 0.5
    A.setdiag(1.0)
    A.eliminate_zeros()
    return A

def spectral_cluster(A: csr_matrix, k_modules: int, seed: int):
    sc = SpectralClustering(
        n_clusters=k_modules,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed
    )
    return sc.fit_predict(A).astype(int)

def module_signature_mean(E_t: np.ndarray, labels: np.ndarray):
    k = int(labels.max()) + 1
    Sig = np.zeros((E_t.shape[0], k), dtype=float)
    for m in range(k):
        cols = np.where(labels == m)[0]
        if cols.size:
            Sig[:, m] = E_t[:, cols].mean(axis=1)
    return Sig

def module_module_corr(Sig: np.ndarray):
    if Sig.shape[1] <= 1:
        return np.ones((Sig.shape[1], Sig.shape[1]))
    Sig_z = StandardScaler().fit_transform(Sig)
    C = np.corrcoef(Sig_z, rowvar=False)
    C = np.clip(C, -1, 1)
    np.fill_diagonal(C, 1.0)
    return C

def merge_modules_by_corr(C: np.ndarray, merge_corr: float, use_abs: bool = True):
    if C.shape[0] <= 1:
        return np.arange(C.shape[0], dtype=int)
    W = np.abs(C) if use_abs else C.copy()
    adj = (W >= merge_corr).astype(int)
    np.fill_diagonal(adj, 1)
    _, comp_labels = connected_components(csr_matrix(adj), directed=False, connection="weak")
    return comp_labels.astype(int)

def graph_connectivity_stats(A: csr_matrix):
    offdiag = A.copy()
    offdiag.setdiag(0.0)
    offdiag.eliminate_zeros()
    degrees = np.array(offdiag.sum(axis=1)).ravel()
    isolated = int(np.sum(degrees == 0))
    n_comp, _ = connected_components(offdiag, directed=False, connection="weak")
    return int(n_comp), isolated

def merged_module_count_from_signatures(Sig: np.ndarray, merge_corr: float, use_abs: bool = True):
    C = module_module_corr(Sig)
    map_old_to_new = merge_modules_by_corr(C, merge_corr=merge_corr, use_abs=use_abs)
    return int(np.unique(map_old_to_new).size)


# In[9]:


df = pd.read_csv(EG_CSV_PATH)
E_raw = df.to_numpy(dtype=float)
feat_names = df.columns.to_list()

print("EG matrix shape:", E_raw.shape)  # (samples, features)


# In[10]:


results = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)

    # Reference clustering on full data (per config) happens inside loop
    for k_nn in K_NN_LIST:
        for thr in EDGE_THR_LIST:
            for k_mod in K_MOD_LIST:
                # full data
                E_t, eps = signed_log_transform(E_raw)
                X = StandardScaler().fit_transform(E_t)
                A = build_affinity_cosine_knn(X, k_nn=k_nn, edge_thr=thr, use_abs=USE_ABS_SIMILARITY)
                n_comp, iso = graph_connectivity_stats(A)

                ref_labels = spectral_cluster(A, k_modules=k_mod, seed=seed)
                Sig_ref = module_signature_mean(E_t, ref_labels)
                merged_count = merged_module_count_from_signatures(Sig_ref, merge_corr=MERGE_CORR, use_abs=USE_ABS_MERGE)

                # bootstrap stability
                nmi_scores, ari_scores = [], []
                for b in range(N_BOOT):
                    idx = rng.integers(0, E_raw.shape[0], size=E_raw.shape[0])
                    E_b = E_raw[idx, :]

                    E_t_b, _ = signed_log_transform(E_b)
                    X_b = StandardScaler().fit_transform(E_t_b)
                    A_b = build_affinity_cosine_knn(X_b, k_nn=k_nn, edge_thr=thr, use_abs=USE_ABS_SIMILARITY)
                    lab_b = spectral_cluster(A_b, k_modules=k_mod, seed=seed + 1000 + b)

                    nmi_scores.append(normalized_mutual_info_score(ref_labels, lab_b))
                    ari_scores.append(adjusted_rand_score(ref_labels, lab_b))

                results.append({
                    "seed": seed,
                    "k_nn": k_nn,
                    "edge_thr": thr,
                    "k_modules": k_mod,
                    "eps_median_absEG": eps,
                    "n_components": n_comp,
                    "isolated_nodes": iso,
                    "merged_module_count": merged_count,
                    "mean_nmi": float(np.mean(nmi_scores)),
                    "std_nmi": float(np.std(nmi_scores, ddof=1)) if len(nmi_scores) > 1 else 0.0,
                    "mean_ari": float(np.mean(ari_scores)),
                    "std_ari": float(np.std(ari_scores, ddof=1)) if len(ari_scores) > 1 else 0.0,
                })

sweep_df = pd.DataFrame(results)

# Aggregate results -> one row per config
agg = (
    sweep_df.groupby(["k_nn", "edge_thr", "k_modules"], as_index=False)
            .agg(
                mean_nmi=("mean_nmi", "mean"),
                std_nmi=("mean_nmi", "std"),
                mean_ari=("mean_ari", "mean"),
                std_ari=("mean_ari", "std"),
                n_components=("n_components", "median"),
                isolated_nodes=("isolated_nodes", "median"),
                merged_module_count=("merged_module_count", "median"),
                seeds_used=("seed", "nunique"),
            )
)
agg[["std_nmi", "std_ari"]] = agg[["std_nmi", "std_ari"]].fillna(0.0)

# Simple ranking: prioritize stability + graph health
#agg["score"] = (
#    1.00 * agg["mean_nmi"]
#    + 0.25 * agg["mean_ari"]
#    - 0.02 * (agg["n_components"] - 1).clip(lower=0)
#    - 0.001 * agg["isolated_nodes"]
#    - 0.02 * (agg["merged_module_count"] - 6).abs()
#)

#agg = agg.sort_values("score", ascending=False).reset_index(drop=True)

def _robust_scale(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Robust z-score using MAD (less sensitive to outliers than mean/std)."""
    med = x.median()
    mad = (x - med).abs().median()
    return (x - med) / (1.4826 * mad + eps)

def add_robust_score(
    agg: pd.DataFrame,
    baseline: dict = None,
    jitter: float = 0.30,         # "reasonable reweighting": ±30%
    n_draws: int = 500,
    topk: int = 5,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """
    Adds robust scoring columns to `agg`:
      - score_mean: mean score across reweightings
      - score_p10:  10th percentile score (conservative)
      - win_rate:   fraction of draws where this row is top-1
      - topk_rate:  fraction of draws where this row is in top-k
      - rank_std:   variability of rank across draws

    Returns (augmented_agg, diagnostics_dict).
    """

    df = agg.copy()

    # ---- 1) Normalize metrics so weights are not hostage to raw units ----
    # "Good" metrics (higher is better)
    df["z_mean_nmi"] = _robust_scale(df["mean_nmi"])
    df["z_mean_ari"] = _robust_scale(df["mean_ari"])

    # "Bad" metrics (lower is better) => multiply by -1 after scaling
    df["z_frag"] = -_robust_scale((df["n_components"] - 1).clip(lower=0))
    df["z_isol"] = -_robust_scale(df["isolated_nodes"])
    df["z_moddev"] = -_robust_scale((df["merged_module_count"] - 6).abs())

    feats = ["z_mean_nmi", "z_mean_ari", "z_frag", "z_isol", "z_moddev"]

    # Baseline weights on the *normalized* features
    if baseline is None:
        baseline = {
            "z_mean_nmi": 1.00,
            "z_mean_ari": 0.25,
            "z_frag":     0.02,
            "z_isol":     0.001,
            "z_moddev":   0.02
        }

    w0 = np.array([baseline[c] for c in feats], dtype=float)

    # ---- 2) Draw "reasonable" alternative weights around baseline ----
    rng = np.random.default_rng(random_state)
    # multiplicative jitter: w = w0 * (1 + u), u ~ Uniform(-jitter, +jitter)
    U = rng.uniform(-jitter, +jitter, size=(n_draws, len(feats)))
    W = w0[None, :] * (1.0 + U)

    # Optional: keep all weights nonnegative (common for utility functions)
    W = np.clip(W, 0.0, None)

    # Optional: normalize weights to sum to 1 for interpretability
    W = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    Z = df[feats].to_numpy(dtype=float)  # shape: (n_settings, n_feats)

    # ---- 3) Compute score matrix: settings x draws ----
    # scores[i, d] = sum_j Z[i, j] * W[d, j]
    scores = Z @ W.T  # shape: (n_settings, n_draws)

    # Summaries per setting
    df["score_mean"] = scores.mean(axis=1)
    df["score_p10"]  = np.quantile(scores, 0.10, axis=1)  # conservative robustness
    df["score_std"]  = scores.std(axis=1)

    # ---- 4) Robustness diagnostics: win-rate, top-k rate, rank variability ----
    winners = scores.argmax(axis=0)  # index of best setting per draw
    win_counts = np.bincount(winners, minlength=len(df))
    df["win_rate"] = win_counts / n_draws

    # top-k membership per draw
    topk_idx = np.argpartition(scores, -topk, axis=0)[-topk:, :]  # (topk, n_draws)
    topk_counts = np.bincount(topk_idx.reshape(-1), minlength=len(df))
    df["topk_rate"] = topk_counts / n_draws

    # rank std: compute rank per draw (0 best); then std across draws
    # For efficiency: ranks = argsort(argsort(-scores)) along axis 0
    ranks = np.argsort(np.argsort(-scores, axis=0), axis=0)
    df["rank_std"] = ranks.std(axis=1)

    diagnostics = {
        "baseline_weights": baseline,
        "jitter": jitter,
        "n_draws": n_draws,
        "topk": topk,
        "best_by_mean": df.sort_values(["score_mean", "score_p10"], ascending=False).head(1),
        "best_by_p10":  df.sort_values(["score_p10", "score_mean"], ascending=False).head(1),
    }

    return df, diagnostics

agg2, diag = add_robust_score(
    agg,
    jitter=0.30,     # ±30% “reasonable reweighting”
    n_draws=100,
    topk=5,
    random_state=0,
)

best = agg2.sort_values(["score_p10", "score_mean", "win_rate"], ascending=False).reset_index(drop=True)

# Export
sweep_out = Path(OUT_DIR) / "eg_sweep_results.csv"
best.to_csv(sweep_out, index=False)

print("Saved:", sweep_out)
display(best.head(15))


# In[11]:


best_case = best.iloc[0].to_dict()
k_nn_best = int(best_case["k_nn"])
thr_best = float(best_case["edge_thr"])
k_mod_best = int(best_case["k_modules"])

print("Recommended config:", {"k_nn": k_nn_best, "edge_thr": thr_best, "k_modules": k_mod_best})

# Full-data clustering with best config
E_t, eps = signed_log_transform(E_raw)
X = StandardScaler().fit_transform(E_t)
A = build_affinity_cosine_knn(X, k_nn=k_nn_best, edge_thr=thr_best, use_abs=USE_ABS_SIMILARITY)
labels = spectral_cluster(A, k_modules=k_mod_best, seed=0)

Sig = module_signature_mean(E_t, labels)
C = module_module_corr(Sig)

map_old_to_new = merge_modules_by_corr(C, merge_corr=MERGE_CORR, use_abs=USE_ABS_MERGE)
merged_labels = map_old_to_new[labels]

# Feature stats
mean_abs = np.mean(np.abs(E_raw), axis=0)
mean_signed = np.mean(E_raw, axis=0)

assign = pd.DataFrame({
    "feature": feat_names,
    "module": labels,
    "merged_module": merged_labels,
    "mean_abs_EG": mean_abs,
    "mean_signed_EG": mean_signed,
})

# Exports
final_dir = Path(OUT_DIR) / "eg_final_outputs"
final_dir.mkdir(parents=True, exist_ok=True)

assign.to_csv(final_dir / "final_assignments_original_and_merged.csv", index=False)

# Sizes
(assign["module"].value_counts().sort_index()
 .rename("n_features").reset_index()
 .rename(columns={"index":"module"})
).to_csv(final_dir / "module_sizes_original.csv", index=False)

(assign["merged_module"].value_counts().sort_index()
 .rename("n_features").reset_index()
 .rename(columns={"index":"merged_module"})
).to_csv(final_dir / "module_sizes_merged.csv", index=False)

# Top features per merged module
top_n = 30
top = (assign.sort_values(["merged_module","mean_abs_EG"], ascending=[True, False])
            .groupby("merged_module").head(top_n).reset_index(drop=True))
top.to_csv(final_dir / "top_features_per_merged_module.csv", index=False)

# Directionality summary (risk vs protective)
direction = (
    assign.groupby("merged_module")
          .agg(
              n_features=("feature","count"),
              mean_signed=("mean_signed_EG","mean"),
              frac_pos=("mean_signed_EG", lambda x: float(np.mean(x > 0))),
              frac_neg=("mean_signed_EG", lambda x: float(np.mean(x < 0))),
              mean_abs=("mean_abs_EG","mean"),
          )
          .reset_index()
)
def direction_class(row):
    if row["mean_signed"] > 0 and row["frac_pos"] >= 0.75:
        return "pro-BMD"
    if row["mean_signed"] < 0 and row["frac_neg"] >= 0.75:
        return "anti-BMD"
    return "mixed/ambiguous"
direction["direction_class"] = direction.apply(direction_class, axis=1)
direction.to_csv(final_dir / "merged_module_directionality_summary.csv", index=False)

# Module signatures (original + merged)
Sig_df = pd.DataFrame(Sig, columns=[f"module_{i}" for i in range(Sig.shape[1])])
Sig_df.to_csv(final_dir / "module_signatures_original.csv", index=False)

merged_ids = np.unique(merged_labels)
SigM = np.zeros((E_raw.shape[0], merged_ids.size))
for j, m in enumerate(merged_ids):
    cols = np.where(merged_labels == m)[0]
    SigM[:, j] = E_t[:, cols].mean(axis=1)
SigM_df = pd.DataFrame(SigM, columns=[f"merged_{m}" for m in merged_ids])
SigM_df.to_csv(final_dir / "module_signatures_merged.csv", index=False)

# Correlation matrix and merge map
pd.DataFrame(C, index=[f"module_{i}" for i in range(C.shape[0])],
                columns=[f"module_{i}" for i in range(C.shape[0])]) \
  .to_csv(final_dir / "module_module_correlation_original.csv")

pd.Series(map_old_to_new, name="merged_id").to_csv(final_dir / "merge_map_oldModule_to_mergedModule.csv",
                                                   index_label="original_module")

print("Saved final outputs to:", final_dir)
display(direction.sort_values("merged_module"))
display(top.head(20))


# In[12]:


rng = np.random.default_rng(0)

nmi_k, ari_k, nmi_m, ari_m = [], [], [], []
for b in range(100):
    idx = rng.integers(0, E_raw.shape[0], size=E_raw.shape[0])
    E_b = E_raw[idx, :]

    E_t_b, _ = signed_log_transform(E_b)
    X_b = StandardScaler().fit_transform(E_t_b)
    A_b = build_affinity_cosine_knn(X_b, k_nn=k_nn_best, edge_thr=thr_best, use_abs=USE_ABS_SIMILARITY)
    lab_b = spectral_cluster(A_b, k_modules=k_mod_best, seed=1000 + b)

    Sig_b = module_signature_mean(E_t_b, lab_b)
    C_b = module_module_corr(Sig_b)
    map_b = merge_modules_by_corr(C_b, merge_corr=MERGE_CORR, use_abs=USE_ABS_MERGE)
    lab_b_m = map_b[lab_b]

    nmi_k.append(normalized_mutual_info_score(labels, lab_b))
    ari_k.append(adjusted_rand_score(labels, lab_b))
    nmi_m.append(normalized_mutual_info_score(merged_labels, lab_b_m))
    ari_m.append(adjusted_rand_score(merged_labels, lab_b_m))

stability = pd.DataFrame({
    "bootstrap_id": np.arange(100),
    "NMI_originalK": nmi_k,
    "ARI_originalK": ari_k,
    "NMI_merged": nmi_m,
    "ARI_merged": ari_m,
})
stability.to_csv(final_dir / "bootstrap_stability_scores.csv", index=False)

summary = pd.DataFrame({
    "metric": ["NMI_originalK", "ARI_originalK", "NMI_merged", "ARI_merged"],
    "mean": [np.mean(nmi_k), np.mean(ari_k), np.mean(nmi_m), np.mean(ari_m)],
    "std":  [np.std(nmi_k, ddof=1), np.std(ari_k, ddof=1), np.std(nmi_m, ddof=1), np.std(ari_m, ddof=1)],
    "k_nn": [k_nn_best]*4,
    "edge_thr": [thr_best]*4,
    "k_modules": [k_mod_best]*4,
    "merge_corr": [MERGE_CORR]*4,
})
summary.to_csv(final_dir / "stability_summary.csv", index=False)

print("Saved:", final_dir / "bootstrap_stability_scores.csv")
print("Saved:", final_dir / "stability_summary.csv")
display(summary)


# In[ ]:




