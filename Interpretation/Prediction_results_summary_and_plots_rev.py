#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


# In[2]:


master_path = "master_path/"
output_cloud = "output_results_path/"


# In[3]:


## part1
# FNECK
result_path = master_path + "fig2/FNECK/"
p1_fneck_combined_dfs = [pd.read_csv(result_path + "femoral_neck_samecat_results.csv"),
                         pd.read_csv(result_path + "femoral_neck_samecat_filtered_species_results.csv"),
                         pd.read_csv(result_path + "femoral_neck_miostone_results.csv"),
                         pd.read_csv(result_path + "femoral_neck_mlp_results.csv")]
# HTOT
result_path = master_path + "fig2/HTOT/"
p1_htot_combined_dfs = [pd.read_csv(result_path + "hip_total_samecat_results.csv"),
                        pd.read_csv(result_path + "hip_total_samecat_filtered_species_results.csv"),
                        pd.read_csv(result_path + "hip_total_miostone_results.csv"),
                        pd.read_csv(result_path + "hip_total_mlp_results.csv")]

# STOT
result_path = master_path + "fig2/STOT/"
p1_stot_combined_dfs = [pd.read_csv(result_path + "spine_total_samecat_results.csv"),
                        pd.read_csv(result_path + "spine_total_samecat_filtered_species_results.csv"),
                        pd.read_csv(result_path + "spine_total_miostone_results.csv"),
                        pd.read_csv(result_path + "spine_total_mlp_results.csv")]

# R13
result_path = master_path + "fig2/R13/"
p1_r13_combined_dfs = [pd.read_csv(result_path + "1_3_radius_samecat_results.csv"),
                       pd.read_csv(result_path + "1_3_radius_samecat_filtered_species_results.csv"),
                       pd.read_csv(result_path + "1_3_radius_miostone_results.csv"),
                       pd.read_csv(result_path + "1_3_radius_mlp_results.csv")]


# In[4]:


## part2
# FNECK
result_path = master_path + "fig3/FNECK/"
p2_fneck_combined_dfs = [pd.read_csv(result_path + "femoral_neck_samecat_results.csv"),
                         pd.read_csv(result_path + "femoral_neck_taxoconcat_results.csv"),
                         pd.read_csv(result_path + "femoral_neck_samecat_cl_only_results.csv"),
                         pd.read_csv(result_path + "femoral_neck_taxoma_results.csv"),
                         pd.read_csv(result_path + "PCA_PB_femoral_neck_PBContrast_results.csv")]

# HTOT
result_path = master_path + "fig3/HTOT/"
p2_htot_combined_dfs = [pd.read_csv(result_path + "hip_total_samecat_results.csv"),
                        pd.read_csv(result_path + "hip_total_taxoconcat_results.csv"),
                        pd.read_csv(result_path + "hip_total_samecat_cl_only_results.csv"),
                        pd.read_csv(result_path + "hip_total_taxoma_results.csv"),
                        pd.read_csv(result_path + "PCA_PB_hip_total_PBContrast_results.csv")]

# STOT
result_path = master_path + "fig3/STOT/"
p2_stot_combined_dfs = [pd.read_csv(result_path + "spine_total_samecat_results.csv"),
                        pd.read_csv(result_path + "spine_total_taxoconcat_results.csv"),
                        pd.read_csv(result_path + "spine_total_samecat_cl_only_results.csv"),
                        pd.read_csv(result_path + "spine_total_taxoma_results.csv"),
                        pd.read_csv(result_path + "PCA_PB_spine_total_PBContrast_results.csv")]

# R13
result_path = master_path + "fig3/R13/"
p2_r13_combined_dfs = [pd.read_csv(result_path + "1_3_radius_samecat_results.csv"),
                       pd.read_csv(result_path + "1_3_radius_taxoconcat_results.csv"),
                       pd.read_csv(result_path + "1_3_radius_samecat_cl_only_results.csv"),
                       pd.read_csv(result_path + "1_3_radius_taxoma_results.csv"),
                       pd.read_csv(result_path + "PCA_PB_1_3_radius_PBContrast_results.csv")]


# In[5]:


## part3
# FNECK
result_path = master_path + "fig4/FNECK/"
p3_fneck_combined_dfs = [pd.read_csv(result_path + "femoral_neck_samecat_results.csv"),
                         pd.read_csv(result_path + "all_species_femoral_neck_en_results.csv"),
                         pd.read_csv(result_path + "all_species_femoral_neck_rrf_results.csv"),
                         pd.read_csv(result_path + "all_species_femoral_neck_xgboost_results.csv")]

# HTOT
result_path = master_path + "fig4/HTOT/"
p3_htot_combined_dfs = [pd.read_csv(result_path + "hip_total_samecat_results.csv"),
                        pd.read_csv(result_path + "all_species_hip_total_en_results.csv"),
                        pd.read_csv(result_path + "all_species_hip_total_rrf_results.csv"),
                        pd.read_csv(result_path + "all_species_hip_total_xgboost_results.csv")]

# STOT
result_path = master_path + "fig4/STOT/"
p3_stot_combined_dfs = [pd.read_csv(result_path + "spine_total_samecat_results.csv"),
                        pd.read_csv(result_path + "all_species_spine_total_en_results.csv"),
                        pd.read_csv(result_path + "all_species_spine_total_rrf_results.csv"),
                        pd.read_csv(result_path + "all_species_spine_total_xgboost_results.csv")]

# R13
result_path = master_path + "fig4/R13/"
p3_r13_combined_dfs = [pd.read_csv(result_path + "1_3_radius_samecat_results.csv"),
                       pd.read_csv(result_path + "all_species_1_3_radius_en_results.csv"),
                       pd.read_csv(result_path + "all_species_1_3_radius_rrf_results.csv"),
                       pd.read_csv(result_path + "all_species_1_3_radius_xgboost_results.csv")]


# In[6]:


## part4
# FNECK
result_path = master_path + "fig5/FNECK/"
p4_fneck_combined_dfs = [pd.read_csv(result_path + "NECK_BMD_samecat_lc_validation_ml.csv"),
                         pd.read_csv(result_path + "NECK_BMD_en_lc_validation_ml.csv"),
                         pd.read_csv(result_path + "NECK_BMD_rrf_lc_validation_ml.csv"),
                         pd.read_csv(result_path + "NECK_BMD_xgboost_lc_validation_ml.csv")]

# HTOT
result_path = master_path + "fig5/HTOT/"
p4_htot_combined_dfs = [pd.read_csv(result_path + "HTOT_BMD_samecat_lc_validation_ml.csv"),
                        pd.read_csv(result_path + "HTOT_BMD_en_lc_validation_ml.csv"),
                        pd.read_csv(result_path + "HTOT_BMD_rrf_lc_validation_ml.csv"),
                        pd.read_csv(result_path + "HTOT_BMD_xgboost_lc_validation_ml.csv")]

# STOT
result_path = master_path + "fig5/STOT/"
p4_stot_combined_dfs = [pd.read_csv(result_path + "spine_total_bmd_samecat_lc_validation_ml.csv"),
                        pd.read_csv(result_path + "spine_total_bmd_en_lc_validation_ml.csv"),
                        pd.read_csv(result_path + "spine_total_bmd_rrf_lc_validation_ml.csv"),
                        pd.read_csv(result_path + "spine_total_bmd_xgboost_lc_validation_ml.csv")]

# R13
result_path = master_path + "fig5/R13/"
p4_r13_combined_dfs = [pd.read_csv(result_path + "R_13_BMD_samecat_lc_validation_ml.csv"),
                       pd.read_csv(result_path + "R_13_BMD_en_lc_validation_ml.csv"),
                       pd.read_csv(result_path + "R_13_BMD_rrf_lc_validation_ml.csv"),
                       pd.read_csv(result_path + "R_13_BMD_xgboost_lc_validation_ml.csv")]


# In[7]:


## sup_part1
# FNECK
result_path = master_path + "figS3/FNECK/"
sup_p1_fneck_en_combined_dfs = [pd.read_csv(result_path + "clinical_only_femoral_neck_en_results.csv"),
                                pd.read_csv(result_path + "femoral_neck_mlp_results.csv"),
                                pd.read_csv(result_path + "all_species_femoral_neck_en_results.csv")]

sup_p1_fneck_rf_combined_dfs = [pd.read_csv(result_path + "clinical_only_femoral_neck_rrf_results.csv"),
                                pd.read_csv(result_path + "femoral_neck_mlp_results.csv"),
                                pd.read_csv(result_path + "all_species_femoral_neck_rrf_results.csv")]

sup_p1_fneck_xgboost_combined_dfs = [pd.read_csv(result_path + "clinical_only_femoral_neck_xgboost_results.csv"),
                                     pd.read_csv(result_path + "femoral_neck_mlp_results.csv"),
                                     pd.read_csv(result_path + "all_species_femoral_neck_xgboost_results.csv")]

# HTOT
result_path = master_path + "figS3/HTOT/"
sup_p1_htot_en_combined_dfs = [pd.read_csv(result_path + "clinical_only_hip_total_en_results.csv"),
                               pd.read_csv(result_path + "hip_total_mlp_results.csv"),
                               pd.read_csv(result_path + "all_species_hip_total_en_results.csv")]

sup_p1_htot_rf_combined_dfs = [pd.read_csv(result_path + "clinical_only_hip_total_rrf_results.csv"),
                               pd.read_csv(result_path + "hip_total_mlp_results.csv"),
                               pd.read_csv(result_path + "all_species_hip_total_rrf_results.csv")]

sup_p1_htot_xgboost_combined_dfs = [pd.read_csv(result_path + "clinical_only_hip_total_xgboost_results.csv"),
                                    pd.read_csv(result_path + "hip_total_mlp_results.csv"),
                                    pd.read_csv(result_path + "all_species_hip_total_xgboost_results.csv")]

# STOT
result_path = master_path + "figS3/STOT/"
sup_p1_stot_en_combined_dfs = [pd.read_csv(result_path + "clinical_only_spine_total_en_results.csv"),
                               pd.read_csv(result_path + "spine_total_mlp_results.csv"),
                               pd.read_csv(result_path + "all_species_spine_total_en_results.csv")]

sup_p1_stot_rf_combined_dfs = [pd.read_csv(result_path + "clinical_only_spine_total_rrf_results.csv"),
                               pd.read_csv(result_path + "spine_total_mlp_results.csv"),
                               pd.read_csv(result_path + "all_species_spine_total_rrf_results.csv")]

sup_p1_stot_xgboost_combined_dfs = [pd.read_csv(result_path + "clinical_only_spine_total_xgboost_results.csv"),
                                    pd.read_csv(result_path + "spine_total_mlp_results.csv"),
                                    pd.read_csv(result_path + "all_species_spine_total_xgboost_results.csv")]

# R13
result_path = master_path + "figS3/R13/"
sup_p1_r13_en_combined_dfs = [pd.read_csv(result_path + "clinical_only_1_3_radius_en_results.csv"),
                              pd.read_csv(result_path + "1_3_radius_mlp_results.csv"),
                              pd.read_csv(result_path + "all_species_1_3_radius_en_results.csv")]

sup_p1_r13_rf_combined_dfs = [pd.read_csv(result_path + "clinical_only_1_3_radius_rrf_results.csv"),
                              pd.read_csv(result_path + "1_3_radius_mlp_results.csv"),
                              pd.read_csv(result_path + "all_species_1_3_radius_rrf_results.csv")]

sup_p1_r13_xgboost_combined_dfs = [pd.read_csv(result_path + "clinical_only_1_3_radius_xgboost_results.csv"),
                                   pd.read_csv(result_path + "1_3_radius_mlp_results.csv"),
                                   pd.read_csv(result_path + "all_species_1_3_radius_xgboost_results.csv")]


# In[8]:


## sup_part2
# FNECK
result_path = master_path + "figS4/FNECK/"
sup_part2_fneck_combined_dfs = [pd.read_csv(result_path + "femoral_neck_samecat_results.csv"),
                                pd.read_csv(result_path + "clinical_only_femoral_neck_en_results.csv"),
                                pd.read_csv(result_path + "clinical_only_femoral_neck_rrf_results.csv"),
                                pd.read_csv(result_path + "clinical_only_femoral_neck_xgboost_results.csv")]

# HTOT
result_path = master_path + "figS4/HTOT/"
sup_part2_htot_combined_dfs = [pd.read_csv(result_path + "hip_total_samecat_results.csv"),
                               pd.read_csv(result_path + "clinical_only_hip_total_en_results.csv"),
                               pd.read_csv(result_path + "clinical_only_hip_total_rrf_results.csv"),
                               pd.read_csv(result_path + "clinical_only_hip_total_xgboost_results.csv")]

# STOT
result_path = master_path + "figS4/STOT/"
sup_part2_stot_combined_dfs = [pd.read_csv(result_path + "spine_total_samecat_results.csv"),
                               pd.read_csv(result_path + "clinical_only_spine_total_en_results.csv"),
                               pd.read_csv(result_path + "clinical_only_spine_total_rrf_results.csv"),
                               pd.read_csv(result_path + "clinical_only_spine_total_xgboost_results.csv")]

# R13
result_path = master_path + "figS4/R13/"
sup_part2_r13_combined_dfs = [pd.read_csv(result_path + "1_3_radius_samecat_results.csv"),
                              pd.read_csv(result_path + "clinical_only_1_3_radius_en_results_wstd.csv"),
                              pd.read_csv(result_path + "clinical_only_1_3_radius_rrf_results.csv"),
                              pd.read_csv(result_path + "clinical_only_1_3_radius_xgboost_results.csv")]


# In[9]:


## sup_part3
# FNECK
result_path = master_path + "figS2/FNECK/"
sup_part3_fneck_combined_dfs = [pd.read_csv(result_path + "PCA_PB_femoral_neck_PBContrast_results.csv"),
                                pd.read_csv(result_path + "femoral_neck_samecat_filtered_species_results.csv"),
                                pd.read_csv(result_path + "femoral_neck_mlp_results.csv")]

# HTOT
result_path = master_path + "figS2/HTOT/"
sup_part3_htot_combined_dfs = [pd.read_csv(result_path + "PCA_PB_hip_total_PBContrast_results.csv"),
                               pd.read_csv(result_path + "hip_total_samecat_filtered_species_results.csv"),
                               pd.read_csv(result_path + "hip_total_mlp_results.csv")]

# STOT
result_path = master_path + "figS2/STOT/"
sup_part3_stot_combined_dfs = [pd.read_csv(result_path + "PCA_PB_spine_total_PBContrast_results.csv"),
                               pd.read_csv(result_path + "spine_total_samecat_filtered_species_results.csv"),
                               pd.read_csv(result_path + "spine_total_mlp_results.csv")]

# R13
result_path = master_path + "figS2/R13/"
sup_part3_r13_combined_dfs = [pd.read_csv(result_path + "PCA_PB_1_3_radius_PBContrast_results.csv"),
                              pd.read_csv(result_path + "1_3_radius_samecat_filtered_species_results.csv"),
                              pd.read_csv(result_path + "1_3_radius_mlp_results.csv")]


# In[10]:


## sup_part4
# FNECK
result_path = master_path + "figS1/FNECK/"
sup_part4_fneck_combined_dfs = [pd.read_csv(result_path + "femoral_neck_samecat_filtered_species_results.csv"),
                                pd.read_csv(result_path + "femoral_neck_miostone_results.csv"),
                                pd.read_csv(result_path + "femoral_neck_mlp_results.csv")]
# HTOT
result_path = master_path + "figS1/HTOT/"
sup_part4_htot_combined_dfs = [pd.read_csv(result_path + "hip_total_samecat_filtered_species_results.csv"),
                               pd.read_csv(result_path + "hip_total_miostone_results.csv"),
                               pd.read_csv(result_path + "hip_total_mlp_results.csv")]

# STOT
result_path = master_path + "figS1/STOT/"
sup_part4_stot_combined_dfs = [pd.read_csv(result_path + "spine_total_samecat_filtered_species_results.csv"),
                               pd.read_csv(result_path + "spine_total_miostone_results.csv"),
                               pd.read_csv(result_path + "spine_total_mlp_results.csv")]

# R13
result_path = master_path + "figS1/R13/"
sup_part4_r13_combined_dfs = [pd.read_csv(result_path + "1_3_radius_samecat_filtered_species_results.csv"),
                              pd.read_csv(result_path + "1_3_radius_miostone_results.csv"),
                              pd.read_csv(result_path + "1_3_radius_mlp_results.csv")]


# In[11]:


def _normalize_colname(s: str) -> str:
    """Normalize a column name for fuzzy matching."""
    return "".join(ch.lower() for ch in s if ch.isalnum())

def _pick_col(df: pd.DataFrame, candidates, *, prefer_exact=True, allow_fuzzy=True, metric_name="metric"):
    """
    Pick a column from df given candidate names.
    - candidates: str or list[str]
    """
    if isinstance(candidates, str):
        candidates = [candidates]

    # 1) exact match
    if prefer_exact:
        for c in candidates:
            if c in df.columns:
                return c

    # 2) fuzzy match: ignore case/underscores/spaces/symbols
    if allow_fuzzy:
        norm_map = {_normalize_colname(c): c for c in df.columns}
        for cand in candidates:
            key = _normalize_colname(cand)
            if key in norm_map:
                return norm_map[key]

    # not found
    return None


def build_rmse_r2_dfs_robust(
    combined_dfs,
    column_names,
    rmse_col=None,
    r2_col=None,
    rmse_candidates=("Testing RMSE", "RMSE_Testing_Set", "RMSE testing set", "rmse_test", "test_rmse"),
    r2_candidates=("Testing R2", "R2_Testing_Set", "R2 testing set", "r2_test", "test_r2"),
    prefer_exact=True,
    allow_fuzzy=True,
):
    """
    Robustly build RMSE and R2 DataFrames from a list of result DataFrames,
    handling differing column name conventions across inputs.

    Parameters
    ----------
    combined_dfs : list[pd.DataFrame]
    column_names : list[str]
    rmse_col, r2_col : str or None
        If provided, treated as highest-priority candidate (can still fall back if missing).
    rmse_candidates, r2_candidates : tuple/list[str]
        Candidate column names to search for each metric.
    prefer_exact : bool
        Try exact matches first.
    allow_fuzzy : bool
        If True, also match after normalizing (case/underscore/space-insensitive).

    Returns
    -------
    rmse_df : pd.DataFrame
    r2_df : pd.DataFrame
    """

    if len(combined_dfs) != len(column_names):
        raise ValueError(
            f"Number of DataFrames ({len(combined_dfs)}) must match number of column names ({len(column_names)})."
        )

    # Build candidate lists (explicit rmse_col/r2_col take priority if provided)
    rmse_cands = ([rmse_col] if rmse_col else []) + list(rmse_candidates)
    r2_cands   = ([r2_col] if r2_col else []) + list(r2_candidates)

    rmse_series = []
    r2_series = []
    missing = []

    for idx, df in enumerate(combined_dfs):
        rmse_key = _pick_col(df, rmse_cands, prefer_exact=prefer_exact, allow_fuzzy=allow_fuzzy, metric_name="RMSE")
        r2_key   = _pick_col(df, r2_cands,   prefer_exact=prefer_exact, allow_fuzzy=allow_fuzzy, metric_name="R2")

        if rmse_key is None or r2_key is None:
            missing.append(
                (idx, rmse_key, r2_key, list(df.columns))
            )
            continue

        rmse_series.append(df[rmse_key])
        r2_series.append(df[r2_key])

    if missing:
        msg_lines = ["Some inputs are missing required columns:"]
        for idx, rmse_key, r2_key, cols in missing:
            msg_lines.append(
                f"- combined_dfs[{idx}]: RMSE col found={rmse_key}, R2 col found={r2_key}. "
                f"Available columns: {cols}"
            )
        msg_lines.append(
            "Tip: pass rmse_candidates / r2_candidates with the exact names present in your data."
        )
        raise KeyError("\n".join(msg_lines))

    rmse_df = pd.concat(rmse_series, axis=1)
    rmse_df.columns = column_names

    r2_df = pd.concat(r2_series, axis=1)
    r2_df.columns = column_names

    return rmse_df, r2_df


# In[12]:


sns.set_style("white")

def add_sig_bracket(ax, x1, x2, y, h, text, fs=7, lw=0.8, text_pad=0.0):
    ax.plot([x1, x1, x2, x2],
            [y,  y + h, y + h, y],
            lw=lw, c="black", clip_on=False)
    ax.text((x1 + x2) / 2,
            y + h + text_pad,
            text,
            ha="center", va="bottom",
            fontsize=fs, color="black",
            clip_on=False)

def annotate_mean_sd_and_brackets_clean_v2(
    ax,
    df,
    positions,                 # <-- NEW
    baseline_col="TaxoCAT",
    test="ttest",
    show_only_significant=True,
    alpha=0.05,
    mean_sd_fs=7,
    p_fs=7,
    mean_offset_frac=0.050,
    bracket_zone_gap_frac=0.090,
    bracket_step_frac=0.080,
    bracket_h_frac=0.015,
    text_pad_frac=0.015,
):
    df = df.apply(pd.to_numeric, errors="coerce")
    means = df.mean(skipna=True)
    sds   = df.std(skipna=True)
    col_tops = df.max(skipna=True)

    # p-values vs baseline (paired)
    p_values = {}
    if baseline_col in df.columns:
        for col in df.columns:
            if col == baseline_col:
                continue

            x = df[baseline_col].to_numpy()
            y = df[col].to_numpy()

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if len(x) < 2:
                p_values[col] = np.nan
                continue

            try:
                if test == "ttest":
                    p_values[col] = ttest_rel(x, y).pvalue
                elif test == "wilcoxon":
                    if np.allclose(x - y, 0):
                        p_values[col] = np.nan
                    else:
                        p_values[col] = wilcoxon(
                            x, y,
                            zero_method="pratt",
                            alternative="two-sided",
                            mode="auto"
                        ).pvalue
                else:
                    raise ValueError(f"Unknown test: {test}")
            except Exception:
                p_values[col] = np.nan

    # Current y-scale (after boxplot exists)
    y_min, y_max = ax.get_ylim()
    yrng = y_max - y_min
    if not np.isfinite(yrng) or yrng == 0:
        yrng = 1.0

    # 1) mean ± SD labels (anchored to group tops) — USE positions[i]
    label_tops = []
    mean_offset = mean_offset_frac * yrng
    for i, col in enumerate(df.columns):
        top_i = col_tops.get(col, np.nan)
        if not np.isfinite(top_i):
            continue
        y_text = top_i + mean_offset
        ax.text(positions[i], y_text, f"{means[col]:.3f} ± {sds[col]:.3f}",
                ha="center", va="bottom",
                fontsize=mean_sd_fs, color="blue", clip_on=False)
        label_tops.append(y_text)

    if len(label_tops) == 0 or baseline_col not in df.columns:
        return

    # comparisons (baseline vs others) — keep significant + NaN
    x0_idx = list(df.columns).index(baseline_col)
    comps = []
    for i, col in enumerate(df.columns):
        if col == baseline_col:
            continue
        p = p_values.get(col, np.nan)
        if show_only_significant:
            if np.isfinite(p) and p >= alpha:
                continue
        comps.append((abs(i - x0_idx), i, col, p))
    comps.sort(key=lambda t: t[0])

    # 2) headroom and bracket zone
    needed_frac = (bracket_zone_gap_frac
                   + (len(comps) + 1) * bracket_step_frac
                   + bracket_h_frac
                   + text_pad_frac
                   + 0.10)
    ax.set_ylim(y_min, y_max + needed_frac * yrng)

    y_min2, y_max2 = ax.get_ylim()
    yrng2 = y_max2 - y_min2
    if not np.isfinite(yrng2) or yrng2 == 0:
        yrng2 = 1.0

    bracket_gap = bracket_zone_gap_frac * yrng2
    step_gap    = bracket_step_frac * yrng2
    bracket_h   = bracket_h_frac * yrng2
    text_pad    = text_pad_frac * yrng2

    bracket_zone_start = max(label_tops) + bracket_gap

    # 3) Draw brackets — USE positions[x0_idx] and positions[i]
    y = bracket_zone_start
    for _, i, col, p in comps:
        if not np.isfinite(p):
            ptxt = "p=NA"
        else:
            ptxt = "p<1e-10" if p < 1e-10 else f"p={p:.3g}"

        add_sig_bracket(
            ax,
            positions[x0_idx], positions[i],
            y=y, h=bracket_h, text=ptxt,
            fs=p_fs, lw=0.8, text_pad=text_pad
        )
        y += step_gap


def draw_boxplot_matplotlib(ax, df, positions, box_width=0.42, colors=None):
    """
    Matplotlib boxplot with explicit positions. Colors are optional.
    """
    data = [df[col].dropna().to_numpy() for col in df.columns]
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=True,
        whis=1.5
    )

    # Style
    for element in ["whiskers", "caps", "medians"]:
        for artist in bp[element]:
            artist.set_linewidth(0.8)
            artist.set_color("0.2")

    for flier in bp["fliers"]:
        flier.set_markersize(3)
        flier.set_markeredgecolor("0.2")
        flier.set_markerfacecolor("white")

    # Fill colors
    if colors is None:
        # fallback palette resembling your earlier look
        colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("0.2")
        patch.set_linewidth(0.8)

    return bp

def plot_rmse_r2_panels(
    rmse_df,
    r2_df,
    title,
    baseline_col="TaxoCAT",
    spacing=0.72,
    box_width=0.42,
    figsize=(6.0, 5.1),
    panel_labels=("a", "b"),
    xtick_fontsize=9,
):
    """
    Plot a 2-panel (RMSE, R²) figure with explicit x-positions, mean±SD annotations,
    and paired statistical comparisons.

    Parameters
    ----------
    rmse_df : pd.DataFrame
        RMSE values (columns = models, rows = splits).
    r2_df : pd.DataFrame
        R² values (columns = models, rows = splits).
    title : str
        Figure title (e.g., "FNECK").
    baseline_col : str, default="TaxoCAT"
        Baseline model for paired comparisons.
    spacing : float, default=0.72
        Distance between category centers.
    box_width : float, default=0.42
        Width of each boxplot.
    figsize : tuple, default=(6.0, 5.1)
        Figure size in inches.
    panel_labels : tuple, default=("a", "b")
        Labels for subpanels.
    xtick_fontsize : int, default=9
        Font size for x-axis tick labels.
    """

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, x=0.10, y=0.965, ha="left",
                 fontsize=10, fontweight="bold")

    cols = list(rmse_df.columns)
    n = len(cols)

    # Explicit x positions (uniform spacing by default)
    positions = np.arange(n) * spacing

    # -----------------
    # Panel a: RMSE
    # -----------------
    ax = axes[0]
    draw_boxplot_matplotlib(ax, rmse_df, positions, box_width=box_width)
    ax.set_ylabel("RMSE")
    ax.text(-0.08, 1.02, panel_labels[0],
            transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom")

    annotate_mean_sd_and_brackets_clean_v2(
        ax,
        rmse_df,
        positions,
        baseline_col=baseline_col,
        test="ttest"
    )

    # -----------------
    # Panel b: R²
    # -----------------
    ax = axes[1]
    draw_boxplot_matplotlib(ax, r2_df, positions, box_width=box_width)
    ax.set_ylabel("R²")
    ax.text(-0.08, 1.02, panel_labels[1],
            transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom")

    annotate_mean_sd_and_brackets_clean_v2(
        ax,
        r2_df,
        positions,
        baseline_col=baseline_col,
        test="wilcoxon"
    )

    # X ticks (explicit positions)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(cols, rotation=45, ha="right",
                             fontsize=xtick_fontsize)

    # Styling
    for ax in axes:
        ax.margins(x=0.01)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        top=0.90,
        bottom=0.20,
        hspace=0.22
    )

    return fig, axes


# In[13]:


## part1
column_names = ["SAMECAT", "SAMECAT (filtered species)", "SAMECAT (mgs only)", "SAMECAT (clin only)"]


# In[14]:


# FNECK
skeletal_site = "FNECK"
master_dfs = p1_fneck_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("a", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig2.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig2_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[15]:


# HTOT
skeletal_site = "HTOT"
master_dfs = p1_htot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("b", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig2.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig2_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[16]:


# STOT
skeletal_site = "STOT"
master_dfs = p1_stot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("c", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig2.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig2_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[17]:


# R13
skeletal_site = "R13"
master_dfs = p1_r13_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("d", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig2.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig2_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[19]:


## part2
column_names = ["SAMECAT", "TaxoConcat", "SAMECAT (CL-classic)", "TaxoMA", "PBContrast"]


# In[20]:


# FNECK
skeletal_site = "FNECK"
master_dfs = p2_fneck_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("a", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[21]:


# HTOT
skeletal_site = "HTOT"
master_dfs = p2_htot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("b", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[22]:


# STOT
skeletal_site = "STOT"
master_dfs = p2_stot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("c", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[23]:


# R13
skeletal_site = "R13"
master_dfs = p2_r13_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("d", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[24]:


## part3
column_names = ["SAMECAT", "Elastic Net", "Random Forest", "XGBoost"]


# In[25]:


# FNECK
skeletal_site = "FNECK"
master_dfs = p3_fneck_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("a", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig4.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig4_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[26]:


# HTOT
skeletal_site = "HTOT"
master_dfs = p3_htot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("b", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig4.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig4_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[27]:


# STOT
skeletal_site = "STOT"
master_dfs = p3_stot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("c", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig4.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig4_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[28]:


# R13
skeletal_site = "R13"
master_dfs = p3_r13_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("d", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig4.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig4_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[29]:


## sup_part1
en_column_names = ["Elastic Net (clin only)", "SAMECAT (clin only)", "Elastic Net"]
rf_column_names = ["Random Forest (clin only)", "SAMECAT (clin only)", "Random Forest"]
xgb_column_names = ["XGBoost (clin only)", "SAMECAT (clin only)", "XGBoost"]


# In[30]:


# FNECK
skeletal_site = "(a) FNECK"
en_master_dfs = sup_p1_fneck_en_combined_dfs
rf_master_dfs = sup_p1_fneck_rf_combined_dfs
xgb_master_dfs = sup_p1_fneck_xgboost_combined_dfs


# In[32]:


# EN
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=en_master_dfs,
    column_names=en_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=en_column_names[0],
    panel_labels=("1.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_EN_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_EN_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[33]:


# RF
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=rf_master_dfs,
    column_names=rf_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=" ",
    baseline_col=rf_column_names[0],
    panel_labels=("2.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_RF_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_RF_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[34]:


# XGBoost
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=xgb_master_dfs,
    column_names=xgb_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=" ",
    baseline_col=xgb_column_names[0],
    panel_labels=("3.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_XGB_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_XGB_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[35]:


# HTOT
skeletal_site = "(b) HTOT"
en_master_dfs = sup_p1_htot_en_combined_dfs
rf_master_dfs = sup_p1_htot_rf_combined_dfs
xgb_master_dfs = sup_p1_htot_xgboost_combined_dfs


# In[36]:


# EN
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=en_master_dfs,
    column_names=en_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=en_column_names[0],
    panel_labels=("1.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_EN_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_EN_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[37]:


# RF
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=rf_master_dfs,
    column_names=rf_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=" ",
    baseline_col=rf_column_names[0],
    panel_labels=("2.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_RF_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_RF_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[38]:


# XGBoost
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=xgb_master_dfs,
    column_names=xgb_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=" ",
    baseline_col=xgb_column_names[0],
    panel_labels=("3.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_XGB_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_XGB_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[39]:


# STOT
skeletal_site = "(c) STOT"
en_master_dfs = sup_p1_stot_en_combined_dfs
rf_master_dfs = sup_p1_stot_rf_combined_dfs
xgb_master_dfs = sup_p1_stot_xgboost_combined_dfs


# In[40]:


# EN
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=en_master_dfs,
    column_names=en_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=en_column_names[0],
    panel_labels=("1.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_EN_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_EN_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[41]:


# RF
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=rf_master_dfs,
    column_names=rf_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=" ",
    baseline_col=rf_column_names[0],
    panel_labels=("2.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_RF_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_RF_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[42]:


# XGBoost
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=xgb_master_dfs,
    column_names=xgb_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=" ",
    baseline_col=xgb_column_names[0],
    panel_labels=("3.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_XGB_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_XGB_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[43]:


# R13
skeletal_site = "(d) R13"
en_master_dfs = sup_p1_r13_en_combined_dfs
rf_master_dfs = sup_p1_r13_rf_combined_dfs
xgb_master_dfs = sup_p1_r13_xgboost_combined_dfs


# In[44]:


# EN
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=en_master_dfs,
    column_names=en_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=en_column_names[0],
    panel_labels=("1.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_EN_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_EN_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[45]:


# RF
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=rf_master_dfs,
    column_names=rf_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=" ",
    baseline_col=rf_column_names[0],
    panel_labels=("2.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_RF_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_RF_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[46]:


# XGBoost
dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=xgb_master_dfs,
    column_names=xgb_column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=" ",
    baseline_col=xgb_column_names[0],
    panel_labels=("3.", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_XGB_figS3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S3/" + skeletal_site + "_XGB_figS3_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[47]:


## sup_part2
column_names = ["SAMECAT", "Elastic Net (clin only)", "Random Forest (clin only)", "XGBoost (clin only)"]


# In[49]:


# FNECK
skeletal_site = "FNECK"
master_dfs = sup_part2_fneck_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("a", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S4/" + skeletal_site + "_figS4.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S4/" + skeletal_site + "_figS4_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[50]:


# HTOT
skeletal_site = "HTOT"
master_dfs = sup_part2_htot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("b", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S4/" + skeletal_site + "_figS4.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S4/" + skeletal_site + "_figS4_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[51]:


# STOT
skeletal_site = "STOT"
master_dfs = sup_part2_stot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("c", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S4/" + skeletal_site + "_figS4.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S4/" + skeletal_site + "_figS4_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[52]:


# R13
skeletal_site = "R13"
master_dfs = sup_part2_r13_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("d", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + "Fig S4/" + skeletal_site + "_figS4.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + "Fig S4/" + skeletal_site + "_figS4_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[53]:


## sup_part3
#column_names = ["TaxoCAT (filtered species)", "PBContrast", "TaxoCAT (clin only)"]
column_names = ["PBContrast", "SAMECAT (filtered species)", "SAMECAT (clin only)"]


# In[54]:


# FNECK
skeletal_site = "FNECK"
master_dfs = sup_part3_fneck_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("a", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_figS1.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_figS1_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[55]:


# HTOT
skeletal_site = "HTOT"
master_dfs = sup_part3_htot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("b", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_figS1.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_figS1_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[56]:


# STOT
skeletal_site = "STOT"
master_dfs = sup_part3_stot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("c", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_figS1.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_figS1_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[57]:


# R13
skeletal_site = "R13"
master_dfs = sup_part3_r13_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("d", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_figS1.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_figS1_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[58]:


## part4
column_names = ["SAMECAT", "Elastic Net", "Random Forest", "XGBoost"]


# In[59]:


# FNECK
skeletal_site = "LC-FNECK"
master_dfs = p4_fneck_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("a", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig5.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig5_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[60]:


# HTOT
skeletal_site = "LC-HTOT"
master_dfs = p4_htot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("b", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig5.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig5_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[61]:


# STOT
skeletal_site = "LC-STOT"
master_dfs = p4_stot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("c", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig5.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig5_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[62]:


# R13
skeletal_site = "LC-R13"
master_dfs = p4_r13_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("d", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_fig5.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_fig5_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[13]:


## sup_part4
column_names = ["SAMECAT (filtered species)", "SAMECAT (mgs only)", "SAMECAT (clin only)"]


# In[14]:


# FNECK
skeletal_site = "FNECK"
master_dfs = sup_part4_fneck_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("a", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_figS1-2.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_figS1-2_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[15]:


# HTOT
skeletal_site = "HTOT"
master_dfs = sup_part4_htot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("b", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_figS1-2.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_figS1-2_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[16]:


# STOT
skeletal_site = "STOT"
master_dfs = sup_part4_stot_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("c", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_figS1-2.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_figS1-2_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[17]:


# R13
skeletal_site = "R13"
master_dfs = sup_part4_r13_combined_dfs

dfs_rmse, dfs_r2 = build_rmse_r2_dfs_robust(
    combined_dfs=master_dfs,
    column_names=column_names
)

fig, axes = plot_rmse_r2_panels(
    rmse_df=dfs_rmse,
    r2_df=dfs_r2,
    title=skeletal_site,
    baseline_col=column_names[0],
    panel_labels=("d", " "),
    spacing=0.72
)

fig.canvas.draw()

fig.savefig(
    output_cloud + skeletal_site + "_figS1-2.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    output_cloud + skeletal_site + "_figS1-2_300dpi.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()


# In[ ]:




