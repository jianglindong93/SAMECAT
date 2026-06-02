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


# In[3]:


master_path = "master_path/"
output_cloud = "output_results_path/"


# In[2]:


def plot_module_pathway_effect_vs_stability(
    df,
    module_col="module",
    pathway_col="pathway",
    rho_col="rho",
    stability_col="stability",
    mean_cov_col="mean_cov",
    figsize=(10, 6),
    size_scale=500,
    alpha=0.85,
    edgecolor="black",
    linewidth=0.5,
    # Cutoff lines
    add_cutoff_lines=True,
    stability_cutoff=0.90,
    effect_cutoff=0.20,
    cutoff_line_kwargs=None,
    # Label control
    label_top_n=3,
    label_by="abs_rho_times_stability",
    label_fontsize=9,
    label_x_pad_axes=0.02,          # padding for xlim
    label_min_dy_axes=0.03,
    label_in_topright_only=True,
    # Leader line + label polish
    leader_style="elbow",            # "straight" or "elbow"
    leader_linewidth=1.4,            # thicker = clearer
    leader_color="0.25",
    leader_alpha=0.9,
    leader_shrinkB=0,                # <<< key: avoid shortening
    label_x_offset_frac=0.04,        # <<< key: longer leader line (was 0.01)
    add_text_halo=True,              # optional polish
    halo_linewidth=3,
    halo_color="white",
    # Legend control
    legend_outside=True,
    legend_anchor=(1.30, 1.00),
    legend_loc="upper left",
    legend_fontsize=10,
    legend_title="Module",
    legend_markerscale=0.65,
    legend_handletextpad=0.6,
    legend_labelspacing=0.4,
    legend_borderaxespad=0.0,
    right_margin=0.72,               # reserved space for legend/labels
    title=None,
):
    """
    Scatter plot of module–pathway pairs: |rho| vs bootstrap stability.
      - Dot size ∝ mean coverage
      - Color and marker encode module
      - Optional cutoff lines
      - Labels for top-N strongest pathways with collision control
      - Clear leader lines + optional text halo
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Colors & markers (redundant encoding: color + shape)
    modules = sorted(df[module_col].unique())
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(modules)}
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(modules)}

    # ---- plot per-module ----
    for m in modules:
        sub = df[df[module_col] == m]
        xm = sub[rho_col].astype(float).abs().to_numpy()
        ym = sub[stability_col].astype(float).to_numpy()
        covm = sub[mean_cov_col].astype(float).clip(lower=0).to_numpy()
        sm = (covm + 1e-6) * float(size_scale)

        ax.scatter(
            xm, ym,
            s=sm,
            color=color_map[m],
            marker=marker_map[m],
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            label=f"Module {m}",
        )

    # ---- cutoff lines ----
    if add_cutoff_lines:
        if cutoff_line_kwargs is None:
            cutoff_line_kwargs = dict(linestyle="--", linewidth=1, color="gray", alpha=0.7)
        if stability_cutoff is not None:
            ax.axhline(stability_cutoff, **cutoff_line_kwargs)
        if effect_cutoff is not None:
            ax.axvline(effect_cutoff, **cutoff_line_kwargs)

    # ---- axes labels/title ----
    ax.set_xlabel("|Spearman ρ|", fontsize=12)
    ax.set_ylabel("Bootstrap stability", fontsize=12)
    if title is None:
        title = "Top module–pathway pairs: effect vs stability\n(dot size ∝ mean pathway coverage)"
    ax.set_title(title, fontsize=13)

    # ---- x-limits: create some right-side room ----
    xvals = df[rho_col].astype(float).abs().to_numpy()
    xmin, xmax = float(np.nanmin(xvals)), float(np.nanmax(xvals))
    xr = xmax - xmin if xmax > xmin else 1.0
    left_pad_frac = 0.05   # <<< NEW: extra space so markers don't touch border
    ax.set_xlim(
        xmin - left_pad_frac * xr,
        xmax + label_x_pad_axes * xr
    )

    # Optional fine polish
    ax.margins(x=0.01)

    # ---- label top-N (with spacing control) ----
    if label_top_n and label_top_n > 0:
        abs_rho = df[rho_col].astype(float).abs()
        stab = df[stability_col].astype(float)

        score = abs_rho * stab if label_by == "abs_rho_times_stability" else abs_rho
        top = df.assign(_score=score).sort_values("_score", ascending=False)

        # label only high-confidence quadrant
        if label_in_topright_only and (stability_cutoff is not None) and (effect_cutoff is not None):
            top = top[(top[stability_col] >= stability_cutoff) & (top[rho_col].abs() >= effect_cutoff)]

        # avoid duplicate pathway labels
        if pathway_col in top.columns:
            top = top.drop_duplicates(subset=[pathway_col], keep="first")

        top = top.head(int(label_top_n)).copy()
        top = top.sort_values(stability_col, ascending=False)

        placed_y_axes = []

        for _, r in top.iterrows():
            xi = abs(float(r[rho_col]))
            yi = float(r[stability_col])
            label = str(r[pathway_col])

            # Convert point to axes coords for collision checks
            x_axes, y_axes = ax.transAxes.inverted().transform(ax.transData.transform((xi, yi)))

            # Adjust y in axes coords to keep separation
            y_axes_adj = y_axes
            for py in placed_y_axes:
                if abs(y_axes_adj - py) < label_min_dy_axes:
                    y_axes_adj = (py - label_min_dy_axes) if (y_axes_adj > 0.5) else (py + label_min_dy_axes)

            # Clamp
            y_axes_adj = min(max(y_axes_adj, 0.02), 0.98)

            # Convert adjusted axes y back to data coords for label position
            x_data_adj, y_data_adj = ax.transData.inverted().transform(
                ax.transAxes.transform((x_axes, y_axes_adj))
            )

            # Leader line style
            if leader_style == "elbow":
                arrowprops = dict(
                    arrowstyle="-",
                    linewidth=leader_linewidth,
                    color=leader_color,
                    alpha=leader_alpha,
                    connectionstyle="angle3,angleA=0,angleB=90",
                    shrinkA=0,
                    shrinkB=leader_shrinkB,
                )
            else:
                arrowprops = dict(
                    arrowstyle="-",
                    linewidth=leader_linewidth,
                    color=leader_color,
                    alpha=leader_alpha,
                    shrinkA=0,
                    shrinkB=leader_shrinkB,
                )

            # Place label farther right to lengthen leader line
            ann = ax.annotate(
                label,
                xy=(xi, yi),
                xytext=(x_data_adj + label_x_offset_frac * xr, y_data_adj),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=label_fontsize,
                arrowprops=arrowprops,
                color="black",
                annotation_clip=False,
            )

            # Optional polish: white halo behind text for readability
            if add_text_halo:
                ann.set_path_effects([pe.withStroke(linewidth=halo_linewidth, foreground=halo_color)])

            placed_y_axes.append(y_axes_adj)

    # ---- legend ----
    if legend_outside:
        ax.legend(
            title=legend_title,
            bbox_to_anchor=legend_anchor,
            loc=legend_loc,
            frameon=False,
            fontsize=legend_fontsize,
            markerscale=legend_markerscale,
            handletextpad=legend_handletextpad,
            labelspacing=legend_labelspacing,
            borderaxespad=legend_borderaxespad,
        )
        fig.subplots_adjust(right=right_margin)
    else:
        ax.legend(
            title=legend_title,
            frameon=False,
            fontsize=legend_fontsize,
            markerscale=legend_markerscale,
            handletextpad=legend_handletextpad,
            labelspacing=legend_labelspacing,
        )

    return fig, ax


# In[9]:


top_per_module = pd.read_csv(master_path + "eg_spectral_outputs/FNECK/module_pathway_results/fneck_representative_pathways_top10_per_module.csv")
fig, ax = plot_module_pathway_effect_vs_stability(
    top_per_module,
    module_col="module",
    pathway_col="pathway",
    label_top_n=4,
    legend_anchor=(1.32, 0.50),
    legend_markerscale=0.65,
    # Key changes for longer, clearer leader lines:
    label_x_offset_frac=0.04,
    leader_shrinkB=0,
    leader_linewidth=1.4,
    leader_style="elbow",
    # Optional polish:
    add_text_halo=True,
)

fig.canvas.draw()
fig.savefig(
    output_cloud + "fneck_figS7.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[10]:


top_per_module = pd.read_csv(master_path + "eg_spectral_outputs/HTOT/module_pathway_results/htot_representative_pathways_top10_per_module.csv")
fig, ax = plot_module_pathway_effect_vs_stability(
    top_per_module,
    module_col="module",
    pathway_col="pathway",
    label_top_n=4,
    legend_anchor=(1.32, 0.50),
    legend_markerscale=0.65,
    # Key changes for longer, clearer leader lines:
    label_x_offset_frac=0.04,
    leader_shrinkB=0,
    leader_linewidth=1.4,
    leader_style="elbow",
    # Optional polish:
    add_text_halo=True,
)

fig.canvas.draw()
fig.savefig(
    output_cloud + "htot_figS7.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[11]:


top_per_module = pd.read_csv(master_path + "eg_spectral_outputs/STOT/module_pathway_results/stot_representative_pathways_top10_per_module.csv")
fig, ax = plot_module_pathway_effect_vs_stability(
    top_per_module,
    module_col="module",
    pathway_col="pathway",
    label_top_n=4,
    legend_anchor=(1.32, 0.50),
    legend_markerscale=0.65,
    # Key changes for longer, clearer leader lines:
    label_x_offset_frac=0.04,
    leader_shrinkB=0,
    leader_linewidth=1.4,
    leader_style="elbow",
    # Optional polish:
    add_text_halo=True,
)

fig.canvas.draw()
fig.savefig(
    output_cloud + "stot_figS7.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[12]:


top_per_module = pd.read_csv(master_path + "eg_spectral_outputs/R13/module_pathway_results/r13_representative_pathways_top10_per_module.csv")
fig, ax = plot_module_pathway_effect_vs_stability(
    top_per_module,
    module_col="module",
    pathway_col="pathway",
    label_top_n=4,
    legend_anchor=(1.32, 0.50),
    legend_markerscale=0.65,
    # Key changes for longer, clearer leader lines:
    label_x_offset_frac=0.04,
    leader_shrinkB=0,
    leader_linewidth=1.4,
    leader_style="elbow",
    # Optional polish:
    add_text_halo=True,
)

fig.canvas.draw()
fig.savefig(
    output_cloud + "r13_figS7.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()






