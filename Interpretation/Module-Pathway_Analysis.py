#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt


# In[66]:


root_path = 'root_path/'
root_path_pathway = root_path + 'humann2_data/merged_all_results_post_alignment/'
root_path_eg = root_path + 'EG_interpretation_results/'
bmd_site = 'bmd_site' #FNECK, HTOT, STOT, R13
root_path_spectral = root_path_eg + "eg_spectral_outputs/" + bmd_site + "/eg_final_outputs/"
bmd_site_path = None
target_div = 'determined by the internal evaluation, see EG_computation.py'
if bmd_site == 'FNECK':
    bmd_site_path = 'NECK_BMD_' + target_div
elif bmd_site == 'HTOT':
    bmd_site_path = 'HTOT_BMD_' + target_div
elif bmd_site == 'STOT':
    bmd_site_path = 'spine_total_bmd_' + target_div
elif bmd_site == 'R13':
    bmd_site_path = 'R_13_BMD_' + target_div
else:
    raise ValueError("Unrecognized BMD site.")
PATH_ID = 'ID_path'
PATH_PATHAB = root_path_pathway + 'bgi_merged_pathabrelab.csv'
PATH_PATHCOV = root_path_pathway + 'bgi_merged_pathcov.csv'
PATH_EG = root_path_eg + bmd_site_path + '_EG_attr_mgs_test_df.csv'
PATH_ASSIGN = root_path_spectral + 'final_assignments_original_and_merged.csv'


# In[67]:


def is_stratified(pathway_str: str) -> bool:
    return isinstance(pathway_str, str) and ("|" in pathway_str)

def split_pathway_and_taxon(pathway_str: str):
    """
    HUMAnN stratified rows look like:
      'PWYNAME: description|g__Genus.s__Species_name'
    Returns:
      base_pathway (str), taxon (str)
    """
    base, taxon = pathway_str.split("|", 1)
    return base.strip(), taxon.strip()

def humann_taxon_to_species_feature(taxon: str):
    """
    Convert HUMAnN taxon string to the species feature naming used in EG / assignments.
    Example taxon:
      'g__Acinetobacter.s__Acinetobacter_bouvetii' -> 's__Acinetobacter_bouvetii'
    If no 's__' part exists, returns None.
    """
    if not isinstance(taxon, str):
        return None
    m = re.search(r"(s__[^|]+)$", taxon)
    if not m:
        return None
    return m.group(1).strip()


# In[68]:


assign = pd.read_csv(PATH_ASSIGN)

# Use merged_module by default (more stable, less fragmented)
species_to_module = dict(zip(assign["feature"].astype(str), assign["merged_module"].astype(int)))

modules = sorted(assign["merged_module"].unique())
print("Modules:", modules)
print("N species with module labels:", len(species_to_module))


# In[69]:


subject_id = pd.read_csv(PATH_ID + 'subject_id_te.csv')
subject_cols = subject_id["sampleID"].to_list()
print("N subjects:", len(subject_cols), "Example:", subject_cols[:5])

EG = pd.read_csv(PATH_EG)
# EG should be (n_subjects x n_species). If it has no index, use pathway column order:
if EG.shape[0] == len(subject_cols):
    EG.index = subject_cols
else:
    raise ValueError(f"EG rows ({EG.shape[0]}) do not match subjects in pathab ({len(subject_cols)}). "
                     "Please ensure EG has subject IDs or same row order.")

# keep only species that have module assignments
EG = EG.loc[:, [c for c in EG.columns if c in species_to_module]]
print("EG shape after filtering species with module labels:", EG.shape)


# In[70]:


def compute_module_eg(EG: pd.DataFrame, species_to_module: dict, use_abs: bool = False):
    """
    Returns: DataFrame (n_subjects x n_modules)
    """
    X = EG.copy()
    if use_abs:
        X = X.abs()

    # map each species column to its module
    mod_labels = pd.Series({sp: species_to_module[sp] for sp in X.columns}, name="module")

    # group columns by module and average
    module_EG = X.groupby(mod_labels, axis=1).mean()
    module_EG = module_EG.reindex(sorted(module_EG.columns), axis=1)
    module_EG.columns = [f"Module_{m}" for m in module_EG.columns]
    return module_EG

module_EG = compute_module_eg(EG, species_to_module, use_abs=False)
print(module_EG.shape)
module_EG.head()


# In[35]:


pathab = pd.read_csv(PATH_PATHAB)
#pathab = pd.read_csv(PATH_PATHAB, sep="\t")
#pathab_run = pathab[["Pathway"] + subject_cols]
#pathab_run.shape
pathab.shape


# In[29]:


path_ab_filt = pathab_run[
    (pathab_run.drop(columns="Pathway") > 0).mean(axis=1) != 0
].reset_index(drop=True)

path_ab_filt.to_csv(root_path_pathway + "path_ab_filt.csv", index=False)
print(f"Original pathways: {pathab_run.shape[0]}")
print(f"Retained pathways: {path_ab_filt.shape[0]}")


# In[36]:


pathcov = pd.read_csv(PATH_PATHCOV)
#pathcov = pd.read_csv(PATH_PATHCOV, sep="\t")
#pathcov_run = pathcov[["Pathway"] + subject_cols]
#pathcov_run.shape
pathcov.shape


# In[14]:


from pathlib import Path

PATH_PATHCOV = Path(PATH_PATHCOV)

with PATH_PATHCOV.open("r", encoding="utf-8", errors="replace") as f:
    header = f.readline().rstrip("\n\r")
    expected = header.count("\t") + 1

    bad = []
    for i, line in enumerate(f, start=2):  # start=2 since header is line 1
        n = line.rstrip("\n\r").count("\t") + 1
        if n != expected:
            bad.append((i, n))
            if len(bad) >= 20:  # cap output
                break

print("Expected fields:", expected)
print("First bad lines (line_number, fields):", bad[:20])


# In[34]:


path_cov_filt = pathcov_run[
    (pathcov_run.drop(columns="Pathway") > 0).mean(axis=1) != 0
].reset_index(drop=True)

path_cov_filt.to_csv(root_path_pathway + "path_cov_filt.csv", index=False)
print(f"Original pathways: {pathcov_run.shape[0]}")
print(f"Retained pathways: {path_cov_filt.shape[0]}")


# In[71]:


def _clean_pathway_series(s: pd.Series) -> pd.Series:
    # Fix common HUMAnN file artifacts: CRLF, trailing spaces, occasional quotes
    s = s.astype(str)
    s = s.str.replace("\r", "", regex=False).str.strip()
    s = s.str.strip('"').str.strip("'")
    return s

def collapse_stratified_to_module_matrix(
    table_path: str,
    species_to_module: dict,
    subject_cols: list,
    chunksize: int = 20000,
    agg: str = "sum",               # "sum" for abundance; "mean" pattern for coverage via sum+count
    #sep: str = "\t",
    pathway_col_candidates=("Pathway", "# Pathway", "pathway", "PWY"),
):
    """
    Reads HUMAnN pathway TSV and collapses stratified rows into (module, base_pathway) aggregates.

    Returns:
      agg_sum: DataFrame indexed by (module, base_pathway), columns=subjects
      agg_cnt: DataFrame indexed by (module, base_pathway), counts of contributing species rows
              (only meaningful when agg == "mean")
    """

    agg_sum = None
    agg_cnt = None

    #reader = pd.read_csv(
    #    table_path,
    #    sep=sep,
    #    chunksize=chunksize,
    #    dtype=str,          # read as str first; convert numeric later (safer for weird files)
    #    low_memory=False
    #)
    
    reader = pd.read_csv(table_path, chunksize=chunksize, dtype=str)

    for chunk_i, chunk in enumerate(reader):
        # identify pathway column robustly
        pathway_col = None
        for c in pathway_col_candidates:
            if c in chunk.columns:
                pathway_col = c
                break
        if pathway_col is None:
            raise ValueError(
                f"Could not find a pathway column in {table_path}. "
                f"Columns seen: {list(chunk.columns)[:20]}"
            )

        # keep only needed columns (pathway + subjects that exist)
        keep_subjects = [c for c in subject_cols if c in chunk.columns]
        if chunk_i == 0 and len(keep_subjects) < len(subject_cols):
            missing = sorted(set(subject_cols) - set(keep_subjects))
            print(f"[WARN] {len(missing)} subject columns not found in {table_path}. Example: {missing[:5]}")

        cols = [pathway_col] + keep_subjects
        chunk = chunk[cols].copy()

        # clean pathway field
        chunk[pathway_col] = _clean_pathway_series(chunk[pathway_col])

        # drop unstratified rows; we only want "base|taxon"
        strat_mask = chunk[pathway_col].astype(str).str.contains("|", regex=False)
        chunk = chunk.loc[strat_mask].copy()
        if chunk.empty:
            continue

        # split base_pathway and taxon
        base_tax = chunk[pathway_col].str.split("|", n=1, expand=True)
        chunk["base_pathway"] = _clean_pathway_series(base_tax[0])
        chunk["taxon"] = _clean_pathway_series(base_tax[1])

        # map HUMAnN taxon to your EG feature naming
        # NOTE: this depends on your existing helper
        chunk["species"] = chunk["taxon"].map(humann_taxon_to_species_feature)

        # map species to module
        chunk["module"] = chunk["species"].map(species_to_module)
        chunk = chunk.dropna(subset=["module"])
        if chunk.empty:
            continue
        chunk["module"] = chunk["module"].astype(int)

        # convert numeric subject columns
        for c in keep_subjects:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        # group by (module, base_pathway)
        gb = chunk.groupby(["module", "base_pathway"], sort=False)[keep_subjects]
        part_sum = gb.sum(min_count=1)  # min_count keeps all-NA rows as NA instead of 0

        if agg_sum is None:
            agg_sum = part_sum
        else:
            agg_sum = agg_sum.add(part_sum, fill_value=0)

        if agg == "mean":
            # count contributing species rows per (module, base_pathway) for each subject:
            # count non-NA entries across stratified rows (species) within that group
            part_cnt = gb.count()
            if agg_cnt is None:
                agg_cnt = part_cnt
            else:
                agg_cnt = agg_cnt.add(part_cnt, fill_value=0)

    if agg_sum is None:
        # nothing read
        empty = pd.DataFrame(columns=subject_cols)
        empty.index = pd.MultiIndex.from_arrays([[], []], names=["module", "base_pathway"])
        return empty, (empty.copy() if agg == "mean" else None)

    # enforce clean MultiIndex names
    agg_sum.index = pd.MultiIndex.from_tuples(agg_sum.index, names=["module", "base_pathway"])

    if agg == "mean":
        agg_cnt.index = pd.MultiIndex.from_tuples(agg_cnt.index, names=["module", "base_pathway"])
        return agg_sum, agg_cnt
    else:
        return agg_sum, None

# 7a) Module pathway abundance (sum)
mod_pathab_sum, _ = collapse_stratified_to_module_matrix(
    PATH_PATHAB, species_to_module, subject_cols, chunksize=20000, agg="sum"
)

# 7b) Module pathway coverage (mean = sum / count)
mod_pathcov_sum, mod_pathcov_cnt = collapse_stratified_to_module_matrix(
    PATH_PATHCOV, species_to_module, subject_cols, chunksize=20000, agg="mean"
)
mod_pathcov_mean = mod_pathcov_sum.div(mod_pathcov_cnt, axis=0)

def filter_by_prevalence(df: pd.DataFrame, prev_min: float = 0.10) -> pd.DataFrame:
    """
    Prevalence computed across columns (subjects): fraction with value > 0.
    NaN is ignored automatically by mean().
    """
    prev = (df > 0).mean(axis=1)  # NaN -> ignored
    return df.loc[prev > prev_min].copy()

mod_pathab_sum = filter_by_prevalence(mod_pathab_sum, prev_min=0.)
mod_pathcov_mean = filter_by_prevalence(mod_pathcov_mean, prev_min=0.)

print("Collapsed module-pathway abundance:", mod_pathab_sum.shape)
print("Collapsed module-pathway coverage:", mod_pathcov_mean.shape)


# In[72]:


def compute_prevalence(cov_vec: pd.Series, cov_thr: float) -> float:
    return float((cov_vec >= cov_thr).mean())

def associate_module_pathways(
    module_EG: pd.DataFrame,
    mod_pathab_sum: pd.DataFrame,
    mod_pathcov_mean: pd.DataFrame,
    cov_thr: float = 0.10,
    prev_thr: float = 0.50,
    min_abs_rho: float = 0.10,
    fdr_q: float = 0.10
):
    """
    Returns a long DataFrame with correlations, p-values, coverage prevalence, and q-values.
    """
    
    # 1) Hard alignment guard
    common_idx = mod_pathab_sum.index.intersection(mod_pathcov_mean.index)
    if len(common_idx) == 0:
        raise ValueError("No overlap between mod_pathab_sum.index and mod_pathcov_mean.index. Parsing likely broken.")

    if len(common_idx) < len(mod_pathab_sum.index):
        print(f"[WARN] Dropping {len(mod_pathab_sum.index) - len(common_idx)} abundance rows not found in coverage index.")
    if len(common_idx) < len(mod_pathcov_mean.index):
        print(f"[WARN] Dropping {len(mod_pathcov_mean.index) - len(common_idx)} coverage rows not found in abundance index.")

    mod_pathab_sum = mod_pathab_sum.loc[common_idx]
    mod_pathcov_mean = mod_pathcov_mean.loc[common_idx]
    
    results = []
    # mod_pathab_sum index is (module, base_pathway)
    for m in sorted({idx[0] for idx in mod_pathab_sum.index}):
        m_name = f"Module_{m}"
        if m_name not in module_EG.columns:
            continue

        x = module_EG[m_name].loc[subject_cols].astype(float).values

        # subset pathways for this module
        #idx_m = [idx for idx in mod_pathab_sum.index if idx[0] == m]
        idx_m = common_idx[common_idx.get_level_values(0) == m]
        ab_m = mod_pathab_sum.loc[idx_m]
        cov_m = mod_pathcov_mean.loc[idx_m]

        # compute per-pathway stats
        rows = []
        for (mm, pwy), y_series in ab_m.iterrows():
            cov_series = cov_m.loc[(mm, pwy)]

            prev = compute_prevalence(cov_series, cov_thr=cov_thr)
            if prev < prev_thr:
                continue

            y = y_series.loc[subject_cols].astype(float).values
            rho, pval = spearmanr(x, y, nan_policy="omit")

            if np.isnan(rho):
                continue
            if abs(rho) < min_abs_rho:
                continue

            rows.append((m, pwy, rho, pval, prev, float(cov_series.mean())))

        if not rows:
            continue

        dfm = pd.DataFrame(rows, columns=["module","pathway","rho","pval","prevalence","mean_cov"])
        # FDR within each module
        dfm["qval"] = multipletests(dfm["pval"].values, method="fdr_bh")[1]
        dfm = dfm[dfm["qval"] <= fdr_q].copy()

        results.append(dfm)

    if results:
        out = pd.concat(results, axis=0, ignore_index=True)
        return out.sort_values(["module","qval","rho"], ascending=[True, True, False])
    else:
        return pd.DataFrame(columns=["module","pathway","rho","pval","prevalence","mean_cov","qval"])


# In[73]:


def bootstrap_stability(
    x: pd.Series,
    y: pd.Series,
    n_boot: int = 300,
    alpha: float = 0.05,
    random_state: int = 0
):
    rng = np.random.default_rng(random_state)
    n = len(x)
    full_rho, _ = spearmanr(x.values, y.values, nan_policy="omit")
    if np.isnan(full_rho):
        return np.nan, np.nan

    full_sign = np.sign(full_rho) if full_rho != 0 else 0.0
    hit = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # bootstrap subjects with replacement
        xb = x.values[idx]
        yb = y.values[idx]
        rho_b, p_b = spearmanr(xb, yb, nan_policy="omit")
        if np.isnan(rho_b):
            continue
        if (p_b <= alpha) and (np.sign(rho_b) == full_sign):
            hit += 1
    stability = hit / n_boot
    return float(full_rho), float(stability)


# In[74]:


def tune_thresholds(
    module_EG,
    mod_pathab_sum,
    mod_pathcov_mean,
    cov_thr_grid=(0.05, 0.10, 0.20),
    prev_thr_grid=(0.30, 0.50, 0.70),
    min_abs_rho_grid=(0.10, 0.15, 0.20),
    fdr_q=0.10,
    n_boot=200
):
    rows = []
    for cov_thr in cov_thr_grid:
        for prev_thr in prev_thr_grid:
            for min_abs_rho in min_abs_rho_grid:
                assoc = associate_module_pathways(
                    module_EG, mod_pathab_sum, mod_pathcov_mean,
                    cov_thr=cov_thr, prev_thr=prev_thr, min_abs_rho=min_abs_rho, fdr_q=fdr_q
                )
                if assoc.empty:
                    rows.append((cov_thr, prev_thr, min_abs_rho, 0, np.nan, np.nan))
                    continue

                # compute stability for a manageable subset: top N per module by qval then |rho|
                assoc2 = assoc.copy()
                assoc2["abs_rho"] = assoc2["rho"].abs()
                assoc2 = assoc2.sort_values(["module","qval","abs_rho"], ascending=[True, True, False])

                # take top 10 per module for stability estimation
                keep = assoc2.groupby("module").head(10).copy()

                stabs = []
                for _, r in keep.iterrows():
                    m = int(r["module"])
                    pwy = r["pathway"]
                    x = module_EG[f"Module_{m}"].loc[subject_cols].astype(float)
                    y = mod_pathab_sum.loc[(m, pwy), subject_cols].astype(float)
                    _, stab = bootstrap_stability(x, y, n_boot=n_boot, alpha=0.05, random_state=0)
                    stabs.append(stab)

                rows.append((
                    cov_thr, prev_thr, min_abs_rho,
                    assoc.shape[0],
                    float(np.nanmedian(stabs)) if len(stabs) else np.nan,
                    float(np.nanmean(stabs)) if len(stabs) else np.nan
                ))

    out = pd.DataFrame(rows, columns=[
        "cov_thr","prev_thr","min_abs_rho","n_pairs_selected","median_stability_top10","mean_stability_top10"
    ])
    # Prefer configs with non-trivial selections and high stability
    out["score"] = (
        out["median_stability_top10"].fillna(0)
        - 0.0005 * (out["n_pairs_selected"] - 200).abs()  # mild preference for reasonable size
    )
    return out.sort_values("score", ascending=False)


# In[75]:


tuning = tune_thresholds(
    module_EG, mod_pathab_sum, mod_pathcov_mean,
    cov_thr_grid=(0.05, 0.10, 0.20, 0.30),
    prev_thr_grid=(0.30, 0.50, 0.70),
    min_abs_rho_grid=(0.10, 0.15, 0.20),
    fdr_q=0.10,
    n_boot=200
)
tuning.head(10)


# In[76]:


best = tuning.iloc[0].to_dict()
best


# In[77]:


def build_final_representative_table(
    module_EG, mod_pathab_sum, mod_pathcov_mean,
    cov_thr, prev_thr, min_abs_rho, fdr_q,
    n_boot=500, alpha=0.05
):
    assoc = associate_module_pathways(
        module_EG, mod_pathab_sum, mod_pathcov_mean,
        cov_thr=cov_thr, prev_thr=prev_thr, min_abs_rho=min_abs_rho, fdr_q=fdr_q
    )
    if assoc.empty:
        return assoc

    # bootstrap stability for all selected pairs (may be heavy; OK for 363 subjects if list isn't huge)
    stabs = []
    for _, r in assoc.iterrows():
        m = int(r["module"])
        pwy = r["pathway"]
        x = module_EG[f"Module_{m}"].loc[subject_cols].astype(float)
        y = mod_pathab_sum.loc[(m, pwy), subject_cols].astype(float)
        _, stab = bootstrap_stability(x, y, n_boot=n_boot, alpha=alpha, random_state=0)
        stabs.append(stab)

    assoc = assoc.copy()
    assoc["stability"] = stabs

    # Composite score (multiplicative “AND” logic)
    assoc["PRS"] = (
        assoc["rho"].abs()
        * np.log1p(assoc["mean_cov"].clip(lower=0))
        * assoc["stability"].fillna(0)
    )

    return assoc.sort_values(["module","PRS"], ascending=[True, False])


final_tbl = build_final_representative_table(
    module_EG, mod_pathab_sum, mod_pathcov_mean,
    cov_thr=float(best["cov_thr"]),
    prev_thr=float(best["prev_thr"]),
    min_abs_rho=float(best["min_abs_rho"]),
    fdr_q=0.10,
    n_boot=300,  # increase to 500–1000 in your final run
    alpha=0.05
)

final_tbl.head(20)


# In[78]:


topK = 10
top_per_module = final_tbl.groupby("module").head(topK).copy()
top_per_module


# In[79]:


# Build matrix of signed rho for top pathways
pivot = top_per_module.pivot_table(index="pathway", columns="module", values="rho", aggfunc="first")

plt.figure(figsize=(10, max(4, 0.25*len(pivot))))
plt.imshow(pivot.fillna(0).values, aspect="auto")
plt.colorbar(label="Spearman rho (ModuleEG vs ModulePathAb)")
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xticks(range(len(pivot.columns)), [f"M{c}" for c in pivot.columns], rotation=0)
plt.title("Top representative pathways per module (signed association)")
plt.tight_layout()
plt.show()


# In[80]:


def plot_top_prs_for_module(df, module_id, topK=10):
    d = df[df["module"] == module_id].sort_values("PRS", ascending=False).head(topK)
    if d.empty:
        return
    plt.figure(figsize=(10, 0.4*len(d)+2))
    plt.barh(d["pathway"][::-1], d["PRS"][::-1])
    plt.title(f"Module {module_id}: Top {topK} representative pathways (PRS)")
    plt.xlabel("PRS = |rho| * log(1+mean_cov) * stability")
    plt.tight_layout()
    plt.show()

for m in sorted(top_per_module["module"].unique()):
    plot_top_prs_for_module(final_tbl, m, topK=10)


# In[81]:


plt.figure(figsize=(10,6))
x = top_per_module["rho"].abs().values
y = top_per_module["stability"].values
s = (top_per_module["mean_cov"].clip(lower=0).values + 1e-6) * 500  # scale dot sizes

plt.scatter(x, y, s=s)
plt.xlabel("|Spearman rho|")
plt.ylabel("Bootstrap stability")
plt.title("Top module–pathway pairs: effect vs stability (dot size ~ mean coverage)")
plt.tight_layout()
plt.show()


# In[82]:


OUTDIR = "D:/metagenome_new_analysis/Multi-task_mgs_bmd/ContrastMTL/EG interpretations/EG_interpretation_results/eg_spectral_outputs/" + bmd_site + "/module_pathway_results"
os.makedirs(OUTDIR, exist_ok=True)

final_tbl.to_csv(os.path.join(OUTDIR, "representative_pathways_full_table.csv"), index=False)
top_per_module.to_csv(os.path.join(OUTDIR, f"representative_pathways_top{topK}_per_module.csv"), index=False)

tuning.to_csv(os.path.join(OUTDIR, "threshold_tuning_summary.csv"), index=False)

print("Saved results to:", OUTDIR)


# In[ ]:




