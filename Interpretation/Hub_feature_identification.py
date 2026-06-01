#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colors
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


master_path = "master_path/"


# In[ ]:


def signed_log_transform(E_raw: np.ndarray):
    abs_vals = np.abs(E_raw)
    nonzero = abs_vals[abs_vals > 0]
    eps = float(np.median(nonzero)) if nonzero.size else 1e-12
    E_t = np.sign(E_raw) * np.log1p(abs_vals / eps)
    return E_t, eps

def signed_log_transform_with_eps(E_raw: np.ndarray, eps: float):
    eps = float(eps) if (eps is not None and np.isfinite(eps) and eps > 0) else 1e-12
    return np.sign(E_raw) * np.log1p(np.abs(E_raw) / eps)

def _safe_pearson(x: np.ndarray, y: np.ndarray, min_n: int = 5) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return np.nan
    x = x[m].astype(float)
    y = y[m].astype(float)

    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    if (not np.isfinite(sx)) or (not np.isfinite(sy)) or sx == 0.0 or sy == 0.0:
        return np.nan

    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0.0:
        return np.nan
    return float(np.sum(x * y) / denom)

def _coerce_module_index(s: pd.Series) -> pd.Series:
    """
    Coerce module indices like 0.0 -> 0; keep NaN as NaN.
    """
    s = pd.to_numeric(s, errors="coerce")
    return s.round().astype("Int64")

def _add_site_column(df: pd.DataFrame, site_name: str) -> pd.DataFrame:
    """
    Rerun-safe: assigns/overwrites df['site'] and moves it to the first column.
    """
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    df["site"] = site_name
    cols = ["site"] + [c for c in df.columns if c != "site"]
    return df.loc[:, cols]


# In[ ]:


def load_site_inputs(
    eg_csv: str,
    assign_csv: str,
    module_col: str = "merged_module",   # FINAL module index
    feature_col: str = "feature",
):
    E_df = pd.read_csv(eg_csv)
    assign_df = pd.read_csv(assign_csv)

    assign_df[feature_col] = assign_df[feature_col].astype(str)
    assign_df[module_col] = _coerce_module_index(assign_df[module_col])
    assign_df = assign_df.dropna(subset=[module_col]).copy()

    # overlap features
    eg_cols = [c for c in E_df.columns]
    common = sorted(set(eg_cols).intersection(set(assign_df[feature_col])))

    if len(common) == 0:
        raise ValueError(
            f"No overlapping features between EG columns and assignment feature_col='{feature_col}'."
        )

    # subset and order EG
    E_df = E_df[common].copy()

    # map feature -> merged_module (int)
    feat2mod = assign_df.set_index(feature_col)[module_col].to_dict()
    modules = np.array([int(feat2mod[f]) for f in common], dtype=int)
    feature_names = np.array(common, dtype=object)

    return E_df, assign_df, feature_names, modules


# In[ ]:


def compute_cs_leave_one_out(
    E_t: np.ndarray,                 # n_subjects x p_features (already transformed)
    feature_names: np.ndarray,        # length p
    modules: np.ndarray,              # length p (int module id per feature)
    min_subjects: int = 5,
    min_module_size: int = 2,
):
    n, p = E_t.shape
    rows = []

    for m in pd.unique(modules):
        idx = np.where(modules == m)[0]
        msize = len(idx)

        if msize < min_module_size:
            for j in idx:
                rows.append({
                    "feature": feature_names[j],
                    "module": int(m),
                    "module_size": msize,
                    "CS": np.nan,
                    "status": "module_too_small"
                })
            continue

        E_mod = E_t[:, idx]  # n x msize
        finite_mod = np.isfinite(E_mod)
        sums = np.nansum(E_mod, axis=1)                 # length n
        counts = finite_mod.sum(axis=1).astype(float)   # length n

        for j in idx:
            x = E_t[:, j]
            finite_x = np.isfinite(x)

            sums_loo = sums - np.where(finite_x, x, 0.0)
            counts_loo = counts - finite_x.astype(float)

            y = np.full(n, np.nan, dtype=float)
            ok = counts_loo >= 1
            y[ok] = sums_loo[ok] / counts_loo[ok]

            r = _safe_pearson(x, y, min_n=min_subjects)

            rows.append({
                "feature": feature_names[j],
                "module": int(m),
                "module_size": msize,
                "CS": r,
                "status": "ok" if np.isfinite(r) else "undefined"
            })

    cs_df = pd.DataFrame(rows)
    cs_df["signed_CS"] = cs_df["CS"]  # explicit signed hub score
    return cs_df


# In[ ]:


def get_top_hubs_per_module(
    cs_df: pd.DataFrame,
    top_k: int = 20,
    require_positive: bool = False,
):
    out = []
    for m, sub in cs_df.groupby("module", sort=True):
        s = sub[np.isfinite(sub["signed_CS"])].copy()
        if require_positive:
            s = s[s["signed_CS"] > 0]
        s = s.sort_values("signed_CS", ascending=False).head(top_k)
        s["rank"] = np.arange(1, len(s) + 1)
        out.append(s)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# In[ ]:


def compute_cs_and_hubs_for_site(
    site_name: str,
    eg_csv: str,
    assign_csv: str,
    module_col: str = "merged_module",  # fixed
    top_k: int = 20,
    min_subjects: int = 5,
    min_module_size: int = 2,
    require_positive: bool = False,
):
    E_df, assign_df, feature_names, modules = load_site_inputs(
        eg_csv=eg_csv,
        assign_csv=assign_csv,
        module_col=module_col
    )

    E_raw = E_df.to_numpy(dtype=float)
    E_t, eps = signed_log_transform(E_raw)

    cs_df = compute_cs_leave_one_out(
        E_t=E_t,
        feature_names=feature_names,
        modules=modules,
        min_subjects=min_subjects,
        min_module_size=min_module_size
    )
    cs_df["eps_signedlog"] = eps
    cs_df = _add_site_column(cs_df, site_name)

    hubs_df = get_top_hubs_per_module(cs_df, top_k=top_k, require_positive=require_positive)
    hubs_df = _add_site_column(hubs_df, site_name)

    return cs_df, hubs_df, eps


# In[ ]:


def adaptive_top_k(module_size: int, base_k: int = 20, frac: float = 0.10, method: str = "max_frac"):
    """
    Compute adaptive K for a module.
    Options:
      - method="max_frac": K = max(base_k, ceil(frac * module_size))
      - method="sqrt":     K = max(base_k, ceil(sqrt(module_size)))
      - method="frac":     K = ceil(frac * module_size)  (no base_k floor)
    """
    module_size = int(module_size)
    if module_size <= 0:
        return 1

    if method == "max_frac":
        return int(max(base_k, np.ceil(frac * module_size)))
    elif method == "sqrt":
        return int(max(base_k, np.ceil(np.sqrt(module_size))))
    elif method == "frac":
        return int(max(1, np.ceil(frac * module_size)))
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_module_quantile_stability(
    CS_boot: np.ndarray,       # B x p
    modules: np.ndarray,       # p (int)
    q: float = 0.90,
    min_defined_frac: float = 0.90,
):
    """
    Relative-to-module distribution threshold stability:
    In each bootstrap b and module m, compute thr = quantile_q(CS among defined features in module m).
    Feature j is a "hit" in bootstrap b if CS_boot[b, j] >= thr (and finite).

    Returns DataFrame with:
      - thr_stability_q : hits/B
      - defined_frac    : #defined / B
      - n_defined
    """
    B, p = CS_boot.shape
    hit = np.zeros(p, dtype=int)
    n_defined = np.isfinite(CS_boot).sum(axis=0)

    for m in np.unique(modules):
        idx = np.where(modules == m)[0]
        if idx.size == 0:
            continue

        for b in range(B):
            vals = CS_boot[b, idx]
            ok = np.isfinite(vals)
            if ok.sum() < 2:
                continue
            thr = np.quantile(vals[ok], q)
            # count hits for features in this module at this bootstrap
            hit[idx] += (np.isfinite(vals) & (vals >= thr)).astype(int)

    out = pd.DataFrame({
        "feature_idx": np.arange(p),
        "thr_stability_q": hit / float(B),
        "n_defined": n_defined,
        "defined_frac": n_defined / float(B),
    })

    out = out[out["defined_frac"] >= float(min_defined_frac)].copy()
    return out

def bootstrap_hub_stability(
    site_name: str,
    eg_csv: str,
    assign_csv: str,
    module_col: str = "merged_module",
    B: int = 300,

    # --- adaptive-K controls ---
    base_k: int = 20,
    k_frac: float = 0.10,
    k_method: str = "max_frac",   # "max_frac" | "sqrt" | "frac"

    # --- relative-threshold controls ---
    q: float = 0.90,              # module quantile threshold per bootstrap
    min_defined_frac: float = 0.90,

    random_state: int = 0,
    min_subjects: int = 5,
    min_module_size: int = 2,
):
    # ---- load & align ----
    E_df, assign_df, feature_names, modules = load_site_inputs(
        eg_csv=eg_csv,
        assign_csv=assign_csv,
        module_col=module_col
    )

    E_raw_full = E_df.to_numpy(dtype=float)
    _, eps_full = signed_log_transform(E_raw_full)  # compute eps once, reuse in bootstraps

    n, p = E_raw_full.shape
    rng = np.random.default_rng(random_state)

    # store signed CS across bootstraps
    CS_boot = np.full((B, p), np.nan, dtype=float)

    # topK hits (adaptive K per module)
    topk_hits = np.zeros(p, dtype=int)

    # module index mapping
    unique_mods = np.unique(modules)
    mod_to_idx = {int(m): np.where(modules == m)[0] for m in unique_mods}
    mod_sizes = {m: len(idx) for m, idx in mod_to_idx.items()}
    mod_K = {m: adaptive_top_k(mod_sizes[m], base_k=base_k, frac=k_frac, method=k_method)
             for m in mod_to_idx.keys()}

    for b in range(B):
        samp = rng.integers(0, n, size=n)

        # fixed-eps sign-log for consistency
        E_t_b = signed_log_transform_with_eps(E_raw_full[samp, :], eps_full)

        cs_b = compute_cs_leave_one_out(
            E_t=E_t_b,
            feature_names=feature_names,
            modules=modules,
            min_subjects=min_subjects,
            min_module_size=min_module_size
        )

        # align signed CS back to feature order
        cs_map = cs_b.set_index("feature")["signed_CS"].to_dict()
        CS_boot[b, :] = np.array([cs_map.get(f, np.nan) for f in feature_names], dtype=float)

        # ---- adaptive topK hits per module ----
        for m, midx in mod_to_idx.items():
            if len(midx) < min_module_size:
                continue

            vals = CS_boot[b, midx]
            ok = np.isfinite(vals)
            if ok.sum() == 0:
                continue

            K_m = min(mod_K[m], ok.sum())  # cannot exceed #defined

            order = np.argsort(vals[ok])[::-1]  # signed CS descending
            chosen_local = np.where(ok)[0][order[:K_m]]
            topk_hits[midx[chosen_local]] += 1

    # ---- summary stats ----
    mean_cs = np.nanmean(CS_boot, axis=0)
    sd_cs = np.nanstd(CS_boot, axis=0, ddof=1)
    n_defined = np.isfinite(CS_boot).sum(axis=0)

    stability_topK = topk_hits / float(B)

    boot_df = pd.DataFrame({
        "feature": feature_names,
        "module": modules.astype(int),
        "module_size": [mod_sizes[int(m)] for m in modules.astype(int)],
        "K_adaptive": [mod_K[int(m)] for m in modules.astype(int)],
        "mean_CS": mean_cs,
        "sd_CS": sd_cs,
        "n_defined": n_defined,
        "defined_frac": n_defined / float(B),
        "stability_topK_adaptive": stability_topK,
        "B": B,
        "eps_signedlog": eps_full,
        "q_threshold": q,
        "k_method": k_method,
        "k_frac": k_frac,
        "base_k": base_k,
    })

    boot_df = _add_site_column(boot_df, site_name)

    # ---- relative-to-module quantile stability ----
    thr_df = compute_module_quantile_stability(
        CS_boot=CS_boot,
        modules=modules.astype(int),
        q=q,
        min_defined_frac=min_defined_frac
    ).rename(columns={"feature_idx": "_idx"})

    # merge threshold stability into boot_df by row index
    boot_df = boot_df.reset_index(drop=True)
    boot_df["_idx"] = np.arange(len(boot_df))
    boot_df = boot_df.merge(thr_df[["_idx", "thr_stability_q"]], on="_idx", how="left")
    boot_df.drop(columns=["_idx"], inplace=True)

    # ---- ranking per module: (thr_stability_q desc, mean_CS desc) ----
    # This is the ranking I recommend when you have "crowded hub plateaus".
    boot_df["rank_thr_then_mean"] = np.nan
    for m, idx in boot_df.groupby("module", sort=True).groups.items():
        g = boot_df.loc[idx].copy()
        g = g.sort_values(["thr_stability_q", "mean_CS"], ascending=[False, False])
        boot_df.loc[g.index, "rank_thr_then_mean"] = np.arange(1, len(g) + 1)

    # also keep an adaptive-topK based rank if you want it
    boot_df["rank_topK_then_mean"] = np.nan
    for m, idx in boot_df.groupby("module", sort=True).groups.items():
        g = boot_df.loc[idx].copy()
        g = g.sort_values(["stability_topK_adaptive", "mean_CS"], ascending=[False, False])
        boot_df.loc[g.index, "rank_topK_then_mean"] = np.arange(1, len(g) + 1)

    return boot_df


# In[ ]:


# /*target_div*/: "Determined by the internal evaluation procedure, see EG_computation.py"
site_inputs = {
    "Femoral neck": (master_path + "NECK_BMD_" + "/*target_div_fneck*/" + "_EG_attr_mgs_test_df.csv",
                     master_path + "eg_spectral_outputs/FNECK/eg_final_outputs/fneck_final_assignments_original_and_merged.csv"),
    "Total hip":    (master_path + "HTOT_BMD_" + "/*target_div_htot*/" + "_EG_attr_mgs_test_df.csv",
                     master_path + "eg_spectral_outputs/HTOT/eg_final_outputs/htot_final_assignments_original_and_merged.csv"),
    "Total spine":  (master_path + "spine_total_bmd_" + "/*target_div_stot*/" + "_EG_attr_mgs_test_df.csv",
                     master_path + "eg_spectral_outputs/STOT/eg_final_outputs/stot_final_assignments_original_and_merged.csv"),
    "1/3 radius":   (master_path + "R_13_BMD_" + "/*target_div_r13*/" + "_EG_attr_mgs_test_df.csv",
                     master_path + "eg_spectral_outputs/R13/eg_final_outputs/r13_final_assignments_original_and_merged.csv"),
}
all_cs, all_hubs, all_boot = [], [], []

for site, (eg_fp, mod_fp) in site_inputs.items():
    cs_df, hubs_df, eps = compute_cs_and_hubs_for_site(
        site_name=site,
        eg_csv=eg_fp,
        assign_csv=mod_fp,
        module_col="merged_module",
        top_k=20,
        require_positive=False
    )
    all_cs.append(cs_df.copy())
    all_hubs.append(hubs_df.copy())

    boot_df = bootstrap_hub_stability(
        site_name=site,
        eg_csv=eg_fp,
        assign_csv=mod_fp,
        module_col="merged_module",
        B=300,
    
        # adaptive K
        base_k=20,
        k_frac=0.10,
        k_method="max_frac",

        # relative threshold stability
        q=0.90,
        min_defined_frac=0.90,

        random_state=0
    )
    all_boot.append(boot_df.copy())

all_cs = pd.concat(all_cs, ignore_index=True)
all_hubs = pd.concat(all_hubs, ignore_index=True)
all_boot = pd.concat(all_boot, ignore_index=True)

all_cs.to_csv(master_path + "eg_spectral_outputs/CS_per_feature_per_module_per_site.csv", index=False)
all_hubs.to_csv(master_path + "eg_spectral_outputs/Top_hub_features_per_module_per_site.csv", index=False)
all_boot.to_csv(master_path + "eg_spectral_outputs/Bootstrap_hub_stability_B300.csv", index=False)


# In[ ]:


hub_candidates = (
    all_boot.query("defined_frac >= 0.90")
            .query("thr_stability_q >= 0.70")
            .query("mean_CS >= 0.5")
            .sort_values(["site", "module", "thr_stability_q", "mean_CS"],
                         ascending=[True, True, False, False])
)

hub_candidates.to_csv(master_path + "eg_spectral_outputs/Hub_features_selected_by_bootstrap_B300.csv", index=False)

