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


# In[3]:


def build_feature_site_matrix(
    hubs_csv_or_df,
    site_col="site",
    feature_col="feature",
    site_order=None,
    mode="weighted",                 # "binary" | "weighted"
    value_col=None,                  # e.g. "mean_CS", "signed_CS", "CS", "stability_topK", "thr_stability_q"
    agg="max",                       # "max" | "mean" | "median"
    # filtering
    min_sites=1,                     # keep features present in >= min_sites
    top_n=None,                      # keep top_n rows after ordering
    # ordering (consistent across modes)
    order_by=("n_sites", "row_sum"), # ("n_sites","row_sum") or ("row_sum","n_sites")
):
    """
    Returns:
      mat: DataFrame (rows=features, cols=sites) with 0/1 (binary) or aggregated values (weighted)
      meta: DataFrame with n_sites and row_sum (for ordering/debug)
    """

    # load
    if isinstance(hubs_csv_or_df, str):
        df = pd.read_csv(hubs_csv_or_df)
    else:
        df = hubs_csv_or_df.copy()

    df = df.dropna(subset=[site_col, feature_col]).copy()
    df[site_col] = df[site_col].astype(str)
    df[feature_col] = df[feature_col].astype(str)

    if site_order is None:
        site_order = list(pd.unique(df[site_col]))

    # choose value column automatically if not provided
    if mode == "weighted":
        if value_col is None:
            for cand in ["mean_CS", "signed_CS", "CS", "stability_topK_adaptive", "stability_topK", "thr_stability_q"]:
                if cand in df.columns:
                    value_col = cand
                    break
        if value_col is None:
            raise ValueError("mode='weighted' but no suitable value_col found. Provide value_col explicitly.")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # aggregate within (feature, site): module labels differ, so collapse them away
    if mode == "binary":
        g = df.groupby([feature_col, site_col]).size().rename("v").reset_index()
        g["v"] = 1
    else:
        if agg == "max":
            g = df.groupby([feature_col, site_col])[value_col].max().rename("v").reset_index()
        elif agg == "mean":
            g = df.groupby([feature_col, site_col])[value_col].mean().rename("v").reset_index()
        elif agg == "median":
            g = df.groupby([feature_col, site_col])[value_col].median().rename("v").reset_index()
        else:
            raise ValueError("agg must be 'max', 'mean', or 'median'.")

    mat = g.pivot(index=feature_col, columns=site_col, values="v").fillna(0.0)

    # ensure site order + include missing sites as zero columns if needed
    for s in site_order:
        if s not in mat.columns:
            mat[s] = 0.0
    mat = mat[site_order]

    # filter by min sites (presence is non-zero)
    n_sites = (mat > 0).sum(axis=1)
    mat = mat.loc[n_sites >= int(min_sites)].copy()

    # ordering metadata
    meta = pd.DataFrame({
        "feature": mat.index,
        "n_sites": (mat > 0).sum(axis=1),
        "row_sum": mat.sum(axis=1),
    }).set_index("feature")

    # consistent ordering across modes
    if order_by == ("n_sites", "row_sum"):
        idx = meta.sort_values(["n_sites", "row_sum"], ascending=[False, False]).index
    elif order_by == ("row_sum", "n_sites"):
        idx = meta.sort_values(["row_sum", "n_sites"], ascending=[False, False]).index
    else:
        raise ValueError("order_by must be ('n_sites','row_sum') or ('row_sum','n_sites').")

    mat = mat.loc[idx]
    meta = meta.loc[idx]

    if top_n is not None:
        mat = mat.head(int(top_n))
        meta = meta.head(int(top_n))

    return mat, meta, value_col

def plot_feature_site_heatmap(
    hubs_csv_or_df,
    site_order=None,
    mode="weighted",                 # "binary" | "weighted"
    value_col=None,                  # optional; auto-picked if None
    agg="max",
    min_sites=1,
    top_n=80,
    figsize=(10, 14),
    title=None,
    show_n_sites_bar=True,
    n_sites_bar_width=0.18,          # fraction of heatmap width
    x_tick_rotation=35,
    y_fontsize=8,
):
    mat, meta, used_value_col = build_feature_site_matrix(
        hubs_csv_or_df,
        site_order=site_order,
        mode=mode,
        value_col=value_col,
        agg=agg,
        min_sites=min_sites,
        top_n=top_n,
        order_by=("n_sites", "row_sum"),
    )

    if title is None:
        if mode == "binary":
            title = "Hub feature presence across skeletal sites"
        else:
            title = f"Hub feature strength across skeletal sites ({agg} of {used_value_col})"

    # figure layout
    if show_n_sites_bar:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, n_sites_bar_width], wspace=0.15)
        ax = fig.add_subplot(gs[0, 0])
        ax_bar = fig.add_subplot(gs[0, 1], sharey=ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_bar = None

    data = mat.to_numpy(dtype=float)

    if mode == "binary":
        # two-box legend (0/1)
        cmap = ListedColormap(["white", "black"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        im = ax.imshow((data > 0).astype(int), aspect="auto", cmap=cmap, norm=norm)

        # custom binary legend boxes
        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                       markerfacecolor="white", markeredgecolor="black", label="0 (absent)"),
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                       markerfacecolor="black", markeredgecolor="black", label="1 (present)"),
        ]
        ax.legend(handles=handles, frameon=False, loc="upper right", title="Presence")

    else:
        im = ax.imshow(data, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(used_value_col)

    # ticks/labels
    ax.set_title(title, pad=12)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=x_tick_rotation, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist(), fontsize=y_fontsize)

    ax.set_xlabel("Skeletal site")
    ax.set_ylabel("Hub features")

    # gridlines (light)
    ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # optional n_sites bar
    if show_n_sites_bar and ax_bar is not None:
        ns = meta["n_sites"].to_numpy()
        y = np.arange(len(ns))
        ax_bar.barh(y, ns)
        ax_bar.set_xlabel("#sites")
        ax_bar.set_xlim(0, max(4, int(ns.max())))
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels([])  # avoid duplicate labels
        ax_bar.invert_yaxis()
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

    # make y order top-to-bottom consistent
    ax.invert_yaxis()

    return fig, ax, mat, meta, used_value_col

def build_feature_site_matrix_from_bootstrap_selected(
    bootstrap_csv_or_df,
    site_col="site",
    feature_col="feature",

    # --- selection filters ---
    use_metric="thr_stability_q",     # "thr_stability_q" | "stability_topK_adaptive" | "stability_topK"
    stability_cutoff=0.70,
    min_defined_frac=0.90,            # if defined_frac exists
    min_mean_CS=None,                 # optional, e.g. 0.20 or 0.90
    modules_include=None,             # optional list[int]
    sites_include=None,               # optional list[str] (filter)

    # --- matrix controls ---
    site_order=None,
    mode="weighted",                  # "binary" | "weighted"
    value_col="mean_CS",              # for weighted mode; can be "mean_CS" or e.g. "thr_stability_q"
    agg="max",                        # "max" | "mean" | "median"
    min_sites=1,                      # keep features present in >= min_sites sites
    top_n=80,                         # keep top rows after ordering
    order_by=("n_sites", "row_sum"),  # consistent ordering across modes
):
    """
    Returns:
      mat: DataFrame rows=features cols=sites
      meta: DataFrame with n_sites, row_sum
      filtered_df: filtered long table actually used
    """
    # load
    if isinstance(bootstrap_csv_or_df, str):
        df = pd.read_csv(bootstrap_csv_or_df)
    else:
        df = bootstrap_csv_or_df.copy()

    for c in [site_col, feature_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in input data.")

    df = df.dropna(subset=[site_col, feature_col]).copy()
    df[site_col] = df[site_col].astype(str)
    df[feature_col] = df[feature_col].astype(str)

    # optional site filter
    if sites_include is not None:
        df = df[df[site_col].isin(sites_include)].copy()

    # module filter if requested
    if modules_include is not None:
        if "module" not in df.columns:
            raise ValueError("modules_include provided but 'module' column not found.")
        df = df[df["module"].isin(modules_include)].copy()

    # defined_frac filter
    if min_defined_frac is not None and "defined_frac" in df.columns:
        df["defined_frac"] = pd.to_numeric(df["defined_frac"], errors="coerce")
        df = df[df["defined_frac"] >= float(min_defined_frac)].copy()

    # mean_CS filter
    if min_mean_CS is not None:
        if "mean_CS" not in df.columns:
            raise ValueError("min_mean_CS provided but 'mean_CS' column not found.")
        df["mean_CS"] = pd.to_numeric(df["mean_CS"], errors="coerce")
        df = df[df["mean_CS"] >= float(min_mean_CS)].copy()

    # stability filter
    if use_metric not in df.columns:
        raise ValueError(f"use_metric='{use_metric}' not found in columns: {list(df.columns)}")
    df[use_metric] = pd.to_numeric(df[use_metric], errors="coerce")
    df = df[df[use_metric] >= float(stability_cutoff)].copy()

    if len(df) == 0:
        raise ValueError("No rows remain after filtering. Relax stability/defined_frac/mean_CS thresholds.")

    # site order
    if site_order is None:
        preferred = ["Femoral neck", "Total hip", "Total spine", "1/3 radius"]
        present = [s for s in preferred if s in set(df[site_col])]
        rest = [s for s in pd.unique(df[site_col]) if s not in present]
        site_order = present + rest

    # build long values (collapse module dimension within each site)
    if mode == "binary":
        g = df.groupby([feature_col, site_col]).size().rename("v").reset_index()
        g["v"] = 1
    else:
        if value_col not in df.columns:
            raise ValueError(f"value_col='{value_col}' not found in the input table.")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        if agg == "max":
            g = df.groupby([feature_col, site_col])[value_col].max().rename("v").reset_index()
        elif agg == "mean":
            g = df.groupby([feature_col, site_col])[value_col].mean().rename("v").reset_index()
        elif agg == "median":
            g = df.groupby([feature_col, site_col])[value_col].median().rename("v").reset_index()
        else:
            raise ValueError("agg must be 'max', 'mean', or 'median'.")

    mat = g.pivot(index=feature_col, columns=site_col, values="v").fillna(0.0)

    # ensure all sites exist as columns
    for s in site_order:
        if s not in mat.columns:
            mat[s] = 0.0
    mat = mat[site_order]

    # filter by min_sites
    n_sites = (mat > 0).sum(axis=1)
    mat = mat.loc[n_sites >= int(min_sites)].copy()

    if len(mat) == 0:
        raise ValueError("No features remain after applying min_sites. Lower min_sites or relax filters.")

    meta = pd.DataFrame({
        "feature": mat.index,
        "n_sites": (mat > 0).sum(axis=1),
        "row_sum": mat.sum(axis=1),
    }).set_index("feature")

    # consistent ordering
    if order_by == ("n_sites", "row_sum"):
        idx = meta.sort_values(["n_sites", "row_sum"], ascending=[False, False]).index
    elif order_by == ("row_sum", "n_sites"):
        idx = meta.sort_values(["row_sum", "n_sites"], ascending=[False, False]).index
    else:
        raise ValueError("order_by must be ('n_sites','row_sum') or ('row_sum','n_sites').")

    mat = mat.loc[idx]
    meta = meta.loc[idx]

    if top_n is not None:
        mat = mat.head(int(top_n))
        meta = meta.head(int(top_n))

    return mat, meta, df

def order_rows_by_site_presence(mat, meta, secondary="row_sum"):
    """
    Put features present in more sites at the top.
    secondary:
      - "row_sum": tie-break by sum across sites (for weighted)
      - "max": tie-break by max across sites
      - "none": only n_sites
    """
    n_sites = (mat > 0).sum(axis=1)

    if secondary == "row_sum":
        sec = mat.sum(axis=1)
    elif secondary == "max":
        sec = mat.max(axis=1)
    else:
        sec = pd.Series(0.0, index=mat.index)

    order = (
        pd.DataFrame({"n_sites": n_sites, "sec": sec}, index=mat.index)
          .sort_values(["n_sites", "sec"], ascending=[False, False])
          .index
    )

    mat2 = mat.loc[order]
    meta2 = meta.loc[order] if meta is not None else None
    return mat2, meta2

def plot_feature_site_heatmap_from_bootstrap_selected(
    bootstrap_csv_or_df,
    site_order=None,

    # selection filters
    use_metric="thr_stability_q",
    stability_cutoff=0.70,
    min_defined_frac=0.90,
    min_mean_CS=None,

    # matrix controls
    mode="weighted",                 # "binary" | "weighted"
    value_col="mean_CS",
    agg="max",
    min_sites=1,
    top_n=80,

    # plotting
    figsize=(10, 14),
    title=None,
    x_tick_rotation=35,
    y_fontsize=8,
    show_n_sites_bar=True,
    n_sites_bar_width=0.20,
):
    mat, meta, filtered_df = build_feature_site_matrix_from_bootstrap_selected(
        bootstrap_csv_or_df,
        site_order=site_order,
        use_metric=use_metric,
        stability_cutoff=stability_cutoff,
        min_defined_frac=min_defined_frac,
        min_mean_CS=min_mean_CS,
        mode=mode,
        value_col=value_col,
        agg=agg,
        min_sites=min_sites,
        top_n=top_n
    )

    # ---- ensure rows ordered with more-site-shared at TOP ----
    # (this is usually already done in build_feature_site_matrix_from_bootstrap_selected,
    # but keeping it here is fine and explicit)
    n_sites_vec = (mat > 0).sum(axis=1)
    if mode == "binary":
        strength = n_sites_vec.astype(float)
    else:
        strength = mat.sum(axis=1)  # tie-breaker within same n_sites

    order = (
        pd.DataFrame({"n_sites": n_sites_vec, "strength": strength}, index=mat.index)
          .sort_values(["n_sites", "strength"], ascending=[False, False])
          .index
    )
    mat = mat.loc[order]
    meta = meta.loc[order]

    if title is None:
        if mode == "binary":
            title = f"Selected hub feature presence across sites\n({use_metric} ≥ {stability_cutoff})"
        else:
            title = f"Selected hub feature strength across sites ({agg} {value_col})\n({use_metric} ≥ {stability_cutoff})"

    # ---- layout with optional n_sites bar ----
    if show_n_sites_bar:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, n_sites_bar_width], wspace=0.18)
        ax = fig.add_subplot(gs[0, 0])
        ax_bar = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_bar = None

    data = mat.to_numpy(dtype=float)

    # KEY FIX: origin="upper" puts first row at the TOP.
    if mode == "binary":
        cmap = ListedColormap(["white", "black"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        ax.imshow((data > 0).astype(int), aspect="auto", cmap=cmap, norm=norm, origin="upper")

        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                       markerfacecolor="white", markeredgecolor="black", label="0 (absent)"),
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                       markerfacecolor="black", markeredgecolor="black", label="1 (present)"),
        ]
        ax.legend(handles=handles, frameon=False, loc="upper right", title="Presence")
    else:
        im = ax.imshow(data, aspect="auto", origin="upper")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(value_col)

    # ---- axis labels + ticks ----
    ax.set_title(title, pad=12)
    ax.set_xlabel("Skeletal site")
    ax.set_ylabel("Selected hub features")

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=x_tick_rotation, ha="right")

    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist(), fontsize=y_fontsize)

    # ---- light gridlines ----
    ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # IMPORTANT: remove invert_yaxis(); it was flipping your sorted order.
    # ax.invert_yaxis()

    # ---- n_sites bar (optional) ----
    if ax_bar is not None:
        ns = meta["n_sites"].to_numpy()
        y = np.arange(len(ns))
        ax_bar.barh(y, ns)
        ax_bar.set_xlabel("#sites")
        ax_bar.set_xlim(0, max(4, int(ns.max())))
        ax_bar.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

        # Align bar panel to heatmap panel
        ax_bar.set_ylim(ax.get_ylim())

    ax.tick_params(axis="y", which="both", labelleft=True)
    ax.yaxis.set_ticks_position("left")
    if ax_bar is not None:
        ax_bar.tick_params(axis="y", which="both", labelleft=False)

    return fig, ax, mat, meta, filtered_df


# In[7]:


def extract_species_short_from_full_taxonomy(tax_string):
    """
    From:
      k__Bacteria.p__Firmicutes....g__Lactobacillus.s__Lactobacillus_farciminis
    return:
      s__Lactobacillus_farciminis
    """
    if pd.isna(tax_string):
        return np.nan
    parts = str(tax_string).split(".")
    for p in reversed(parts):
        if p.startswith("s__"):
            return p
    return np.nan


def get_selected_feature_taxonomy_table(selected_features, microbe_names_csv_or_df, taxonomy_col="species"):
    """
    Match selected feature names (short species names like s__X) to full taxonomy strings
    in microbe_names_prev_filtered2.csv.

    Returns
    -------
    tax_df : DataFrame with columns:
        - feature
        - full_taxonomy
        - k, p, c, o, f, g, s
    """
    if isinstance(microbe_names_csv_or_df, str):
        micro = pd.read_csv(microbe_names_csv_or_df)
    else:
        micro = microbe_names_csv_or_df.copy()

    if taxonomy_col not in micro.columns:
        raise ValueError(f"Column '{taxonomy_col}' not found in microbe names table.")

    micro = micro.copy()
    micro["feature"] = micro[taxonomy_col].apply(extract_species_short_from_full_taxonomy)

    # keep only selected features
    selected_features = pd.Index(selected_features.astype(str) if isinstance(selected_features, pd.Series) else list(map(str, selected_features)))
    tax_df = micro[micro["feature"].isin(selected_features)].copy()

    # deduplicate if needed
    tax_df = tax_df.drop_duplicates(subset=["feature"])

    # check missing
    found = set(tax_df["feature"])
    missing = [f for f in selected_features if f not in found]
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} selected features were not found in taxonomy table.\n"
            f"Examples: {missing[:10]}"
        )

    # parse taxonomic ranks
    rank_prefixes = {
        "k__": "k",
        "p__": "p",
        "c__": "c",
        "o__": "o",
        "f__": "f",
        "g__": "g",
        "s__": "s",
    }

    def parse_ranks(tax_string):
        out = {v: None for v in rank_prefixes.values()}
        for part in str(tax_string).split("."):
            for prefix, short in rank_prefixes.items():
                if part.startswith(prefix):
                    out[short] = part
                    break
        return out

    parsed = tax_df[taxonomy_col].apply(parse_ranks).apply(pd.Series)
    tax_df = pd.concat([tax_df.rename(columns={taxonomy_col: "full_taxonomy"}), parsed], axis=1)

    # preserve input order only if needed later
    tax_df["feature"] = tax_df["feature"].astype(str)

    return tax_df[["feature", "full_taxonomy", "k", "p", "c", "o", "f", "g", "s"]]


def build_ete3_taxonomy_tree_from_tax_df(tax_df, feature_col="feature"):
    """
    Build a rooted taxonomy tree with ete3 from parsed taxonomy table.

    Leaves are named by the short species name (feature), e.g. s__Lactobacillus_farciminis.
    Internal nodes use taxonomy labels such as p__Firmicutes, g__Lactobacillus, etc.

    Returns
    -------
    tree : ete3.Tree
    """
    try:
        from ete3 import Tree
    except ImportError:
        raise ImportError("ete3 is not installed. Please run: pip install ete3")

    tree = Tree()
    tree.name = "root"

    # store created nodes by full path
    node_lookup = {("root",): tree}

    rank_cols = ["k", "p", "c", "o", "f", "g", "s"]

    for _, row in tax_df.iterrows():
        path = ["root"]
        parent = tree

        for rc in rank_cols:
            label = row[rc]
            if pd.isna(label) or label is None:
                continue

            # leaf should be the selected feature name, not necessarily raw row["s"]
            if rc == "s":
                label = row[feature_col]

            candidate_path = tuple(path + [label])

            if candidate_path not in node_lookup:
                child = parent.add_child(name=label)
                node_lookup[candidate_path] = child
            else:
                child = node_lookup[candidate_path]

            parent = child
            path.append(label)

    return tree


def get_tree_leaf_order(tree):
    """
    Return leaf order from left-to-right traversal.
    """
    return [leaf.name for leaf in tree.iter_leaves()]


def compute_tree_plot_coords(tree, leaf_order):
    """
    Compute x/y coordinates for each node.

    y is determined by the supplied leaf_order so that the tree aligns exactly
    with the heatmap row order.
    """
    leaf_y = {name: i for i, name in enumerate(leaf_order)}
    x = {}
    y = {}

    def assign_x(node, depth=0):
        x[node] = depth
        for ch in node.children:
            assign_x(ch, depth + 1)

    def assign_y(node):
        if node.is_leaf():
            y[node] = leaf_y[node.name]
        else:
            for ch in node.children:
                assign_y(ch)
            y[node] = np.mean([y[ch] for ch in node.children])

    assign_x(tree)
    assign_y(tree)

    return x, y


def make_phylum_color_map(tax_df, default_unknown="#999999", cmap_name="tab20"):
    """
    Build a dict: phylum_label -> color
    Example phylum labels: p__Firmicutes, p__Proteobacteria, ...
    """
    phyla = (
        tax_df["p"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    phyla = sorted(phyla)

    cmap = plt.get_cmap(cmap_name, max(len(phyla), 1))
    phylum_to_color = {
        ph: cmap(i) for i, ph in enumerate(phyla)
    }
    phylum_to_color[None] = default_unknown
    return phylum_to_color


def annotate_tree_phylum(tree, tax_df, feature_col="feature", phylum_col="p"):
    """
    Annotate each node with a node.phylum attribute.

    Rules:
    - leaf phylum is taken from tax_df via feature -> phylum
    - internal node phylum is assigned only if ALL descendant leaves
      belong to the same phylum
    - otherwise node.phylum = None
    """
    feature_to_phylum = (
        tax_df[[feature_col, phylum_col]]
        .drop_duplicates()
        .set_index(feature_col)[phylum_col]
        .to_dict()
    )

    def assign(node):
        if node.is_leaf():
            node.add_feature("phylum", feature_to_phylum.get(node.name, None))
            return node.phylum

        child_phyla = [assign(ch) for ch in node.children]
        child_phyla_nonnull = [p for p in child_phyla if p is not None]

        if len(child_phyla_nonnull) == 0:
            node.add_feature("phylum", None)
        elif len(set(child_phyla_nonnull)) == 1 and len(child_phyla_nonnull) == len(child_phyla):
            node.add_feature("phylum", child_phyla_nonnull[0])
        else:
            node.add_feature("phylum", None)

        return node.phylum

    assign(tree)
    return tree


def get_node_branch_color(node, phylum_to_color, mixed_color="black"):
    """
    Color a branch by the phylum of the child subtree.
    Mixed subtree -> mixed_color
    """
    ph = getattr(node, "phylum", None)
    return phylum_to_color.get(ph, mixed_color) if ph is not None else mixed_color


def draw_ete3_tree_on_axis(
    ax,
    tree,
    leaf_order,
    linecolor="black",
    linewidth=0.8,
    show_leaf_labels=False,
    leaf_fontsize=8,
    color_by_phylum=False,
    phylum_to_color=None,
    mixed_color="black"
):
    """
    Draw a rectangular phylogenetic/taxonomic tree on a matplotlib axis.

    If color_by_phylum=True, each branch is colored according to the phylum
    of the child subtree when that subtree is phylum-homogeneous.
    """
    x, y = compute_tree_plot_coords(tree, leaf_order)

    # draw edges
    for node in tree.traverse("preorder"):
        if node.is_leaf():
            continue

        child_ys = [y[ch] for ch in node.children]
        x0 = x[node]

        # parent vertical connector:
        # color by this node if homogeneous, otherwise mixed_color
        if color_by_phylum:
            parent_color = get_node_branch_color(node, phylum_to_color, mixed_color=mixed_color)
        else:
            parent_color = linecolor

        ax.plot([x0, x0], [min(child_ys), max(child_ys)], color=parent_color, lw=linewidth)

        # horizontal connectors to children:
        # color by child's subtree phylum
        for ch in node.children:
            if color_by_phylum:
                edge_color = get_node_branch_color(ch, phylum_to_color, mixed_color=mixed_color)
            else:
                edge_color = linecolor

            ax.plot([x0, x[ch]], [y[ch], y[ch]], color=edge_color, lw=linewidth)

    # optional leaf labels
    if show_leaf_labels:
        max_x = max(x.values()) if len(x) > 0 else 0
        for leaf in tree.iter_leaves():
            ax.text(max_x + 0.1, y[leaf], leaf.name, va="center", ha="left", fontsize=leaf_fontsize)

    # cosmetics
    ax.set_ylim(-0.5, len(leaf_order) - 0.5)
    max_x = max(x.values()) if len(x) > 0 else 0
    ax.set_xlim(-0.02, max_x + 0.02)
    ax.margins(x=0)
    ax.invert_yaxis()  # match imshow(origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def format_species_label(s):
    """
    Optional formatting for display only.
    Keeps the original names recognizable while making them cleaner.
    """
    s = str(s)
    # keep the s__ prefix if you want; otherwise uncomment the next line
    # s = s.replace("s__", "")
    return s

def plot_feature_site_heatmap_with_phylo_tree_from_bootstrap_selected(
    bootstrap_csv_or_df,
    microbe_names_csv_or_df,
    site_order=None,

    # selection filters
    use_metric="thr_stability_q",
    stability_cutoff=0.70,
    min_defined_frac=0.90,
    min_mean_CS=None,

    # matrix controls
    mode="weighted",                 # "binary" | "weighted"
    value_col="mean_CS",
    agg="max",
    min_sites=1,
    top_n=80,

    # plotting
    figsize=(16, 14),
    title=None,
    x_tick_rotation=35,
    y_fontsize=8,
    show_n_sites_bar=True,

    # panel widths
    tree_width_ratio=0.26,
    label_width_ratio=0.95,
    heatmap_width_ratio=1.00,
    n_sites_bar_width=0.22,

    # spacing
    panel_wspace=0.015,

    # colorbar controls
    cbar_height_frac=0.36,   # fraction of heatmap axis height
    cbar_width=0.012,        # figure fraction
    cbar_pad=0.018,          # gap to the right of the site-count bar
    label_x=0.00,                    # left anchor of species labels inside label axis
    reorder_rows_by_tree=True,
    
    # new tree-color controls
    color_tree_by_phylum=True,
    phylum_cmap_name="tab20",
    mixed_branch_color="black",
):
    # ---------------------------------------------------
    # Step 1: build selected feature x site matrix
    # ---------------------------------------------------
    mat, meta, filtered_df = build_feature_site_matrix_from_bootstrap_selected(
        bootstrap_csv_or_df,
        site_order=site_order,
        use_metric=use_metric,
        stability_cutoff=stability_cutoff,
        min_defined_frac=min_defined_frac,
        min_mean_CS=min_mean_CS,
        mode=mode,
        value_col=value_col,
        agg=agg,
        min_sites=min_sites,
        top_n=top_n
    )

    # ---------------------------------------------------
    # Step 2: get taxonomy table for selected features
    # ---------------------------------------------------
    tax_df = get_selected_feature_taxonomy_table(
        selected_features=mat.index.tolist(),
        microbe_names_csv_or_df=microbe_names_csv_or_df,
        taxonomy_col="species"
    )

    # ---------------------------------------------------
    # Step 3: build tree
    # ---------------------------------------------------
    tree = build_ete3_taxonomy_tree_from_tax_df(tax_df, feature_col="feature")
    phylum_to_color = None
    if color_tree_by_phylum:
        tree = annotate_tree_phylum(tree, tax_df, feature_col="feature", phylum_col="p")
        phylum_to_color = make_phylum_color_map(
            tax_df,
            default_unknown="#999999",
            cmap_name=phylum_cmap_name
        )

    # ---------------------------------------------------
    # Step 4: reorder rows by tree if requested
    # ---------------------------------------------------
    tree_leaf_order = get_tree_leaf_order(tree)
    tree_leaf_order = [x for x in tree_leaf_order if x in mat.index]

    if reorder_rows_by_tree:
        mat = mat.loc[tree_leaf_order]
        meta = meta.loc[tree_leaf_order]
    else:
        tree_leaf_order = mat.index.tolist()

    # ---------------------------------------------------
    # Title
    # ---------------------------------------------------
    if title is None:
        if mode == "binary":
            title = f"Selected hub feature presence across sites\n({use_metric} ≥ {stability_cutoff})"
        else:
            title = f"Selected hub feature strength across sites ({agg} {value_col})\n({use_metric} ≥ {stability_cutoff})"

    # ---------------------------------------------------
    # Step 5: layout
    # Final order:
    # tree | species labels | heatmap | #sites bar
    # colorbar will be added manually later
    # ---------------------------------------------------
    if show_n_sites_bar:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            1, 4,
            width_ratios=[
                tree_width_ratio,
                label_width_ratio,
                heatmap_width_ratio,
                n_sites_bar_width
            ],
            wspace=panel_wspace
        )

        ax_tree  = fig.add_subplot(gs[0, 0])
        ax_label = fig.add_subplot(gs[0, 1])
        ax       = fig.add_subplot(gs[0, 2])
        ax_bar   = fig.add_subplot(gs[0, 3], sharey=ax)
        ax_cbar  = None

    else:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[
                tree_width_ratio,
                label_width_ratio,
                heatmap_width_ratio
            ],
            wspace=panel_wspace
        )

        ax_tree  = fig.add_subplot(gs[0, 0])
        ax_label = fig.add_subplot(gs[0, 1])
        ax       = fig.add_subplot(gs[0, 2])
        ax_bar   = None
        ax_cbar  = None

    data = mat.to_numpy(dtype=float)

    # ---------------------------------------------------
    # Step 6: heatmap
    # ---------------------------------------------------
    if mode == "binary":
        cmap = ListedColormap(["white", "black"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        ax.imshow((data > 0).astype(int), aspect="auto", cmap=cmap, norm=norm, origin="upper")

        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                       markerfacecolor="white", markeredgecolor="black", label="0 (absent)"),
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                       markerfacecolor="black", markeredgecolor="black", label="1 (present)"),
        ]
        ax.legend(handles=handles, frameon=False, loc="upper right", title="Presence")
    else:
        im = ax.imshow(data, aspect="auto", origin="upper")

    ax.set_title(title, pad=12)
    ax.set_xlabel("Skeletal site")
    ax.set_ylabel("")

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=x_tick_rotation, ha="right")

    # remove species labels from heatmap axis
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)

    # gridlines
    ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ---------------------------------------------------
    # Step 7: dedicated species-label axis
    # This axis sits BETWEEN tree and heatmap
    # ---------------------------------------------------
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(-0.5, mat.shape[0] - 0.5)
    ax_label.invert_yaxis()

    # turn off axis furniture
    ax_label.set_xticks([])
    ax_label.set_yticks([])
    for spine in ax_label.spines.values():
        spine.set_visible(False)

    # draw labels manually, left-aligned
    display_labels = [format_species_label(s) for s in mat.index.tolist()]
    y_positions = np.arange(mat.shape[0])

    for y, lab in zip(y_positions, display_labels):
        ax_label.text(
            label_x, y, lab,
            ha="left", va="center",
            fontsize=y_fontsize,
            clip_on=False
        )

    # optional visual label for the block
    ax_label.set_title("", pad=12)

    # ---------------------------------------------------
    # Step 8: tree
    # ---------------------------------------------------
    draw_ete3_tree_on_axis(
        ax_tree,
        tree=tree,
        leaf_order=mat.index.tolist(),
        linecolor="black",
        linewidth=0.8,
        show_leaf_labels=False,
        color_by_phylum=color_tree_by_phylum,
        phylum_to_color=phylum_to_color,
        mixed_color=mixed_branch_color
    )
    ax_tree.set_title("Taxonomy", pad=12)

    # ---------------------------------------------------
    # Step 9: #sites bar immediately next to heatmap
    # ---------------------------------------------------
    if ax_bar is not None:
        ns = meta["n_sites"].to_numpy()
        y = np.arange(len(ns))
        ax_bar.barh(y, ns)
        ax_bar.set_xlabel("#sites")
        #ax_bar.set_xlim(0, max(4, int(ns.max())) * 1.05)
        ax_bar.set_xlim(0, max(4, int(ns.max())))
        ax_bar.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
    
    # ---------------------------------------------------
    # Step 9.5: add a smaller manual colorbar
    # ---------------------------------------------------
    # place colorbar to the right of the #sites bar if present,
    # otherwise to the right of the heatmap
    anchor_ax = ax_bar if ax_bar is not None else ax

    bbox = anchor_ax.get_position()
    heat_bbox = ax.get_position()

    cbar_h = heat_bbox.height * cbar_height_frac
    cbar_y = heat_bbox.y0 + (heat_bbox.height - cbar_h) / 2
    cbar_x = bbox.x1 + cbar_pad

    ax_cbar = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_h])
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(value_col)

    # ---------------------------------------------------
    # Step 10: align y extents across all axes
    # ---------------------------------------------------
    ax_tree.set_ylim(ax.get_ylim())
    ax_label.set_ylim(ax.get_ylim())
    if ax_bar is not None:
        ax_bar.set_ylim(ax.get_ylim())
    
    if color_tree_by_phylum and phylum_to_color is not None:
        legend_items = []
        for ph in sorted([k for k in phylum_to_color.keys() if k is not None]):
            legend_items.append(
                plt.Line2D([0], [0], color=phylum_to_color[ph], lw=2, label=ph.replace("p__", ""))
            )

        if len(legend_items) > 0:
            fig.legend(
                handles=legend_items,
                title="Phylum",
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(0.87, 0.95),
                fontsize=7,
                title_fontsize=8,
                ncol=1
            )

    fig.subplots_adjust(left=0.08, right=0.89, top=0.93, bottom=0.08)
    
    return fig, ax_tree, ax_label, ax, ax_bar, ax_cbar, mat, meta, filtered_df, tax_df, tree


# In[12]:


fp_hub = master_path + "eg_spectral_outputs/Hub_features_selected_by_bootstrap_B300.csv"
fp_tax = master_path + "microbe_names_wtzeroonlyspecies.csv"
site_order = ["Femoral neck", "Total hip", "Total spine", "1/3 radius"]

fig, ax_tree, ax_label, ax, ax_bar, ax_cbar, mat, meta, filtered_df, tax_df, tree =     plot_feature_site_heatmap_with_phylo_tree_from_bootstrap_selected(
        bootstrap_csv_or_df=fp_hub,
        microbe_names_csv_or_df=fp_tax,
        site_order=site_order,
        use_metric="thr_stability_q",
        stability_cutoff=0.70,
        min_defined_frac=0.90,
        min_mean_CS=None,
        mode="weighted",
        value_col="mean_CS",
        agg="max",
        min_sites=1,
        top_n=80,
        figsize=(14, 14),
        y_fontsize=8,
        tree_width_ratio=0.25,
        label_width_ratio=0.49,
        heatmap_width_ratio=1.00,
        n_sites_bar_width=0.22,
        cbar_height_frac=0.34,
        cbar_width=0.010,
        cbar_pad=0.006,
        panel_wspace=0.03,
        reorder_rows_by_tree=True,
        color_tree_by_phylum=True,
        phylum_cmap_name="tab20",
        mixed_branch_color="black"
    )

output_pdf = "output_results_path/hub_species_summary_plot_300dpi.pdf"
fig.savefig(
    output_pdf,
    bbox_inches="tight"
)

plt.show()


# In[14]:


output_pdf = "output_results_path/hub_species_summary_plot.pdf"
output_png = "output_results_path/hub_species_summary_plot_300dpi.png"

fig.canvas.draw()

fig.savefig(
    output_pdf,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_png,
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[ ]:




