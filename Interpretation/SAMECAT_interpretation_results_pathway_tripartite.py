#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


master_path = "master_path/"
output_cloud = "output_results_path/"


# In[3]:


def load_site_top10_table(fp, site_name,
                          module_col="module",
                          pathway_col="pathway",
                          rho_col="rho",
                          simplify_pathway="id"):  # keep as "id" for matrix keys
    df = pd.read_csv(fp).copy()
    df["site"] = site_name

    # Parse "PWY-xxxx: Full pathway name"
    s = df[pathway_col].astype(str)

    # Split on first ":" only
    df["pathway_id"] = s.str.split(":", n=1).str[0].str.strip()

    # If ":" exists, take RHS; otherwise fallback to ID
    rhs = s.str.split(":", n=1).str[1]
    df["pathway_name"] = rhs.fillna(df["pathway_id"]).astype(str).str.strip()

    # Choose key used in matrices
    if simplify_pathway == "id":
        df["pathway_key"] = df["pathway_id"]
    else:
        # full string as key (rarely recommended due to inconsistencies)
        df["pathway_key"] = s.str.strip()

    # Unique module label across sites (critical)
    df["module_uid"] = df["site"].astype(str) + " | M" + df[module_col].astype(str)

    out = df[["site", "module_uid", "pathway_key", "pathway_id", "pathway_name", rho_col]].rename(
        columns={rho_col: "rho"}
    )
    out["rho"] = pd.to_numeric(out["rho"], errors="coerce")
    out = out.dropna(subset=["rho", "pathway_key", "module_uid", "site"])
    return out

def build_pathway_label_map(long_df):
    """
    Returns dict: pathway_id -> pathway_name.
    If duplicates exist, keeps the most frequent name for that ID.
    """
    tmp = long_df.dropna(subset=["pathway_id", "pathway_name"]).copy()
    # most common name per id
    label_map = (
        tmp.groupby("pathway_id")["pathway_name"]
           .agg(lambda x: x.value_counts().index[0])
           .to_dict()
    )
    return label_map

def build_module_pathway_matrix(long_df, value_col="rho", agg="max_abs_signed"):
    """
    Returns module x pathway matrix.

    agg options:
      - "mean": mean rho per (module, pathway)
      - "max_abs_signed": choose entry with max |rho|, keep its sign
      - "max": max rho (not abs)
    """
    df = long_df.copy()

    if agg == "mean":
        g = df.groupby(["module_uid", "pathway_key"])[value_col].mean()
    elif agg == "max":
        g = df.groupby(["module_uid", "pathway_key"])[value_col].max()
    elif agg == "max_abs_signed":
        # pick row with max abs(rho) within each (module, pathway)
        df["_abs"] = df[value_col].abs()
        idx = df.groupby(["module_uid", "pathway_key"])["_abs"].idxmax()
        g = df.loc[idx].set_index(["module_uid", "pathway_key"])[value_col]
        df = df.drop(columns=["_abs"])
    else:
        raise ValueError("Unknown agg")

    mat = g.unstack("pathway_key").fillna(0.0)
    return mat

def build_site_pathway_matrix(long_df, value_col="rho", agg="max_abs_signed"):
    """
    Site x pathway matrix by aggregating across modules within site.
    """
    df = long_df.copy()

    if agg == "mean":
        g = df.groupby(["site", "pathway_key"])[value_col].mean()
    elif agg == "max":
        g = df.groupby(["site", "pathway_key"])[value_col].max()
    elif agg == "max_abs_signed":
        df["_abs"] = df[value_col].abs()
        idx = df.groupby(["site", "pathway_key"])["_abs"].idxmax()
        g = df.loc[idx].set_index(["site", "pathway_key"])[value_col]
        df = df.drop(columns=["_abs"])
    else:
        raise ValueError("Unknown agg")

    mat = g.unstack("pathway_key").fillna(0.0)
    return mat

def _cluster_order(mat, metric="correlation", method="average"):
    """
    Returns list of index labels in dendrogram leaf order.
    """
    X = mat.to_numpy(dtype=float)
    if mat.shape[0] <= 1:
        return list(mat.index)
    # pdist on rows
    d = pdist(X, metric=metric)
    Z = linkage(d, method=method)
    order = leaves_list(Z)
    return list(mat.index[order])

def tripartite_order(long_df,
                     metric="correlation",
                     method="average",
                     module_agg="max_abs_signed",
                     site_agg="max_abs_signed",
                     pathway_order_within_module="abs_rho_desc"):
    """
    Returns:
      site_order, module_order, pathway_order (global concatenated),
      plus a mapping for separators.
    """
    # Matrices
    mod_mat = build_module_pathway_matrix(long_df, agg=module_agg)   # module_uid x pathway
    site_mat = build_site_pathway_matrix(long_df, agg=site_agg)      # site x pathway

    # 1) Sites clustered
    site_order = _cluster_order(site_mat, metric=metric, method=method)

    module_order = []
    pathway_order = []
    separators = []  # list of (site, start_idx, end_idx)

    # 2) For each site, cluster its modules
    for site in site_order:
        mods = sorted(long_df.loc[long_df["site"] == site, "module_uid"].unique())
        sub_mod_mat = mod_mat.loc[mod_mat.index.intersection(mods)]

        mods_ordered = _cluster_order(sub_mod_mat, metric=metric, method=method)
        module_order.extend(mods_ordered)

        # 3) Within each module, order pathways (top to bottom)
        site_start = len(pathway_order)
        for mu in mods_ordered:
            # get pathways present in this module
            row = mod_mat.loc[mu]
            present = row[row != 0].copy()

            if len(present) == 0:
                continue

            if pathway_order_within_module == "abs_rho_desc":
                paths = present.reindex(present.abs().sort_values(ascending=False).index).index.tolist()
            elif pathway_order_within_module == "rho_desc":
                paths = present.sort_values(ascending=False).index.tolist()
            else:
                raise ValueError("Unknown pathway_order_within_module")

            # keep global uniqueness while preserving first occurrence
            for p in paths:
                if p not in pathway_order:
                    pathway_order.append(p)

        site_end = len(pathway_order)
        separators.append((site, site_start, site_end))

    return site_order, module_order, pathway_order, separators, mod_mat

def plot_tripartite_heatmap(long_df,
                            value_col="rho",
                            metric="correlation",
                            method="average",
                            figsize=(12, 14),
                            module_agg="max_abs_signed",
                            site_agg="max_abs_signed",
                            show_pathway_name=True,
                            append_id_if_missing=True):

    site_order, module_order, pathway_order, separators, mod_mat = tripartite_order(
        long_df,
        metric=metric,
        method=method,
        module_agg=module_agg,
        site_agg=site_agg,
        pathway_order_within_module="abs_rho_desc"
    )

    # Build pathway x module matrix
    M = mod_mat.reindex(index=module_order, columns=pathway_order).T  # pathways x modules

    # Replace pathway IDs with human-readable names for display
    if show_pathway_name:
        label_map = build_pathway_label_map(long_df)

        def _label(pid):
            name = label_map.get(pid, None)
            if name is None or name == "" or name == pid:
                return f"{pid}" if not append_id_if_missing else f"{pid}"
            # If you want BOTH name and ID, change to: f"{name} ({pid})"
            #return name
            return f"{name} ({pid})"

        display_index = [ _label(pid) for pid in M.index.tolist() ]
        M.index = display_index  # only affects display (matrix values/order unchanged)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M.to_numpy(dtype=float), aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(value_col)

    ax.set_title("Tripartite hierarchy: Pathway → Site-specific module → Site (values = ρ)", pad=12)
    ax.set_xlabel("Site-specific modules (clustered within clustered sites)")
    ax.set_ylabel("Pathways (ordered by within-module |ρ|, nested by site/module)")

    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels(M.columns.tolist(), rotation=90, fontsize=7)

    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels(M.index.tolist(), fontsize=7)

    # separators between sites
    mod_to_site = pd.Series({m: m.split(" | ")[0] for m in module_order})
    current = mod_to_site.iloc[0]
    for j in range(1, len(module_order)):
        if mod_to_site.iloc[j] != current:
            ax.axvline(j - 0.5, color="white", linewidth=2.0)
            current = mod_to_site.iloc[j]

    # site labels across the top
    spans = []
    for s in site_order:
        cols = [i for i, m in enumerate(module_order) if m.startswith(s + " | ")]
        if cols:
            spans.append((s, min(cols), max(cols)))
    for s, a, b in spans:
        ax.text((a + b) / 2, -1.5, s, ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    return fig, ax, M, (site_order, module_order, pathway_order)


# In[5]:


site_files = {
    "Femoral neck": master_path + "eg_spectral_outputs/FNECK/module_pathway_results/fneck_representative_pathways_top10_per_module.csv",
    "Total hip":    master_path + "eg_spectral_outputs/HTOT/module_pathway_results/htot_representative_pathways_top10_per_module.csv",
    "Total spine":  master_path + "eg_spectral_outputs/STOT/module_pathway_results/stot_representative_pathways_top10_per_module.csv",
    "1/3 radius":   master_path + "eg_spectral_outputs/R13/module_pathway_results/r13_representative_pathways_top10_per_module.csv",
}

dfs = []
for site, fp in site_files.items():
    dfs.append(load_site_top10_table(fp, site_name=site, module_col="module", pathway_col="pathway", rho_col="rho", simplify_pathway="id"))

long_df = pd.concat(dfs, ignore_index=True)

fig, ax, M, orders = plot_tripartite_heatmap(long_df, figsize=(14, 16), show_pathway_name=True)

fig.savefig(
    output_cloud + "fig7_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()






