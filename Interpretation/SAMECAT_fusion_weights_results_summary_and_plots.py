#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


# In[3]:


master_path = "master_path/"
output_cloud = "output_results_path/"


# In[4]:


# FNECK
fneck_combined_df = pd.read_csv(master_path + "NECK_BMD_samecat_fusion_weights.csv")

# HTOT
htot_combined_df = pd.read_csv(master_path + "HTOT_BMD_samecat_fusion_weights.csv")

# STOT
stot_combined_df = pd.read_csv(master_path + "spine_total_bmd_samecat_fusion_weights.csv")

# R13
r13_combined_df = pd.read_csv(master_path + "R_13_BMD_samecat_fusion_weights.csv")


# In[5]:


def plot_fusion_weights_by_split(mgs_weights,
                                 clinical_weights,
                                 labels=None,
                                 title="Fusion weights by tuning split (mgs vs clinical)",
                                 save_path=None):
    """
    Plot grouped bar chart comparing mgs and clinical fusion weights
    across splits, and include Wilcoxon signed-rank p-value for
    testing mgs_weight - 0.5 = 0 in the legend.

    Parameters
    ----------
    mgs_weights : array-like
        Normalized mgs weights (length n_splits)
    clinical_weights : array-like
        Normalized clinical weights (length n_splits)
    labels : list of str, optional
        Labels for splits (default: split_1 ... split_n)
    title : str
        Plot title
    save_path : str, optional
        If provided, saves the figure to this path

    Returns
    -------
    dict
        Wilcoxon statistic and p-value
    """

    mgs = np.array(mgs_weights, dtype=float)
    clin = np.array(clinical_weights, dtype=float)

    if len(mgs) != len(clin):
        raise ValueError("mgs_weights and clinical_weights must have same length")

    n = len(mgs)

    if labels is None:
        labels = [f"split_{i+1}" for i in range(n)]

    # Wilcoxon signed-rank test: H0 median(mgs - 0.5) = 0
    diff = mgs - 0.5
    w_stat, p_value = stats.wilcoxon(diff)
    p_text = f"Wilcoxon (mgs-0.5=0): p={p_value:.3g}"

    x = np.arange(n)
    width = 0.38

    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, mgs, width, label=f"mgs_weight ({p_text})")
    plt.bar(x + width/2, clin, width, label="clinical_weight")

    plt.axhline(0.5, linestyle="--")
    plt.ylim(0, 1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Normalized fusion weight")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path,
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0.05
        )

    plt.show()

    return {"wilcoxon_stat": w_stat, "p_value": p_value}


# In[10]:


df = fneck_combined_df

mgs_weights = df["Normalized mgs weight"].to_numpy()
clinical_weights = df["Normalized clinical weight"].to_numpy()
labels = df["division"].astype(str).tolist()

result = plot_fusion_weights_by_split(
    mgs_weights,
    clinical_weights,
    labels=labels,
    title="(a) FNECK BMD: Fusion weights across tuning splits",
    save_path=output_cloud + "fneck_figS5_300dpi.png"
)

result


# In[7]:


df = htot_combined_df

mgs_weights = df["Normalized mgs weight"].to_numpy()
clinical_weights = df["Normalized clinical weight"].to_numpy()
labels = df["division"].astype(str).tolist()

result = plot_fusion_weights_by_split(
    mgs_weights,
    clinical_weights,
    labels=labels,
    title="(b) HTOT BMD: Fusion weights across tuning splits",
    save_path=output_cloud + "htot_figS5_300dpi.png"
)

result


# In[8]:


df = stot_combined_df

mgs_weights = df["Normalized mgs weight"].to_numpy()
clinical_weights = df["Normalized clinical weight"].to_numpy()
labels = df["division"].astype(str).tolist()

result = plot_fusion_weights_by_split(
    mgs_weights,
    clinical_weights,
    labels=labels,
    title="(c) STOT BMD: Fusion weights across tuning splits",
    save_path=output_cloud + "stot_figS5_300dpi.png"
)

result


# In[9]:


df = r13_combined_df

mgs_weights = df["Normalized mgs weight"].to_numpy()
clinical_weights = df["Normalized clinical weight"].to_numpy()
labels = df["division"].astype(str).tolist()

result = plot_fusion_weights_by_split(
    mgs_weights,
    clinical_weights,
    labels=labels,
    title="(d) R13 BMD: Fusion weights across tuning splits",
    save_path=output_cloud + "r13_figS5_300dpi.png"
)

result







