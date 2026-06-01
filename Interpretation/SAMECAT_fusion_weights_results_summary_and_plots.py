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


# In[6]:


fneck_combined_df


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


# In[14]:


def plot_fusion_weights_by_split_from_df(
    df: pd.DataFrame,
    *,
    weight_type: str = "normalized",   # "normalized" or "raw"
    mgs_col: str = None,
    clinical_col: str = None,
    label_col: str = "division",
    title: str = None,
    save_path: str = None,
    # testing controls
    test: str = "wilcoxon",            # "wilcoxon" or "ttest"
    test_mode: str = "mgs_vs_clinical",# "mgs_vs_clinical" or "mgs_vs_value"
    test_value: float = 0.5,           # used only when test_mode="mgs_vs_value"
    alternative: str = "two-sided",    # "two-sided", "greater", "less"
):
    """
    Plot grouped bars of mgs vs clinical weights per split, and include a paired-test p-value in the legend.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing weight columns.
    weight_type : {"normalized","raw"}
        Convenience selector for default column names.
    mgs_col, clinical_col : str or None
        Explicit column names. If None, inferred from weight_type.
    label_col : str or None
        Column for x-axis labels. If None or missing, uses split_1..split_n.
    title : str or None
        Title for the plot. If None, auto-generated.
    save_path : str or None
        If provided, saves figure to this path.
    test : {"wilcoxon","ttest"}
        Statistical test to use.
    test_mode : {"mgs_vs_clinical","mgs_vs_value"}
        - "mgs_vs_clinical": paired comparison between mgs and clinical vectors
        - "mgs_vs_value": one-sample test on (mgs - test_value)
    test_value : float
        Reference value for one-sample test (only used for test_mode="mgs_vs_value").
    alternative : {"two-sided","greater","less"}
        Alternative hypothesis direction (SciPy compatible).

    Returns
    -------
    dict with keys: test, test_mode, statistic, p_value, n, mgs_col, clinical_col
    """

    # --- choose columns ---
    if mgs_col is None or clinical_col is None:
        if weight_type.lower() == "normalized":
            default_mgs = "Normalized mgs weight"
            default_clin = "Normalized clinical weight"
        elif weight_type.lower() == "raw":
            default_mgs = "Raw mgs weight"
            default_clin = "Raw clinical weight"
        else:
            raise ValueError("weight_type must be 'normalized' or 'raw' (or provide mgs_col/clinical_col).")

        mgs_col = default_mgs if mgs_col is None else mgs_col
        clinical_col = default_clin if clinical_col is None else clinical_col

    if mgs_col not in df.columns or clinical_col not in df.columns:
        raise KeyError(f"Missing columns. Need '{mgs_col}' and '{clinical_col}' in df.columns.")

    # --- extract vectors (drop NaNs pairwise) ---
    tmp = df[[mgs_col, clinical_col]].copy()
    tmp = tmp.dropna()
    mgs = tmp[mgs_col].to_numpy(dtype=float)
    clin = tmp[clinical_col].to_numpy(dtype=float)

    if len(mgs) == 0:
        raise ValueError("No valid (non-NaN) weight pairs available after dropping NaNs.")

    n = len(mgs)

    # --- labels ---
    if label_col is not None and label_col in df.columns:
        labels = df.loc[tmp.index, label_col].astype(str).tolist()
    else:
        labels = [f"split_{i+1}" for i in range(n)]

    # --- run test ---
    test = test.lower()
    test_mode = test_mode.lower()

    if test_mode == "mgs_vs_clinical":
        if test == "wilcoxon":
            stat, p = stats.wilcoxon(mgs, clin, alternative=alternative)
            test_text = f"Wilcoxon paired (mgs vs clinical), p={p:.3g}"
        elif test == "ttest":
            res = stats.ttest_rel(mgs, clin, alternative=alternative)
            stat, p = res.statistic, res.pvalue
            test_text = f"Paired t-test (mgs vs clinical), p={p:.3g}"
        else:
            raise ValueError("test must be 'wilcoxon' or 'ttest'.")

    elif test_mode == "mgs_vs_value":
        diff = mgs - float(test_value)
        if test == "wilcoxon":
            stat, p = stats.wilcoxon(diff, alternative=alternative)
            test_text = f"Wilcoxon (mgs-{test_value:g}=0), p={p:.3g}"
        elif test == "ttest":
            res = stats.ttest_1samp(diff, popmean=0.0, alternative=alternative)
            stat, p = res.statistic, res.pvalue
            test_text = f"1-sample t-test (mgs-{test_value:g}=0), p={p:.3g}"
        else:
            raise ValueError("test must be 'wilcoxon' or 'ttest'.")
    else:
        raise ValueError("test_mode must be 'mgs_vs_clinical' or 'mgs_vs_value'.")

    # --- plot ---
    x = np.arange(n)
    width = 0.38

    if title is None:
        title = f"Fusion weights by split ({weight_type})"

    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, mgs, width, label=f"mgs_weight ({test_text})")
    plt.bar(x + width/2, clin, width, label="clinical_weight")

    # optional reference line: only makes sense for normalized weights
    if weight_type.lower() == "normalized":
        plt.axhline(0.5, linestyle="--", linewidth=1)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Weight")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()

    return {
        "test": test,
        "test_mode": test_mode,
        "statistic": float(stat),
        "p_value": float(p),
        "n": int(n),
        "mgs_col": mgs_col,
        "clinical_col": clinical_col,
    }


# In[16]:


df = fneck_combined_df

res = plot_fusion_weights_by_split_from_df(
    df,
    weight_type="raw",
    test="ttest",                 # <-- paired t-test
    test_mode="mgs_vs_clinical",  # <-- compare raw mgs vs raw clinical
    alternative="two-sided",
    label_col="division",
    title="NECK BMD: Raw fusion weights (paired t-test)",
    save_path=None,
)

print(res)  # includes statistic + p_value


# In[ ]:




