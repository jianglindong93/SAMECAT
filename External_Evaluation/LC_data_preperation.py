#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# In[2]:


# Dot-aware: Table 1 uses '.' between ranks; we also allow other common separators defensively.
SPECIES_RE = re.compile(r"(?:^|[.\|;,\s])\s*(s__[^.\|;,\s]+)", flags=re.IGNORECASE)

def extract_species_token(name: str) -> str:
    """
    Extract 's__...' from a taxonomy string, or return the species token itself if already species-only.
    Returns "" if not found.
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).strip()
    if s.lower().startswith("s__"):
        return s
    m = SPECIES_RE.search(s)
    return m.group(1).strip() if m else ""

def normalize_species_token(tok: str) -> str:
    """
    Normalize for matching: lowercase, trim, spaces->underscore, collapse underscores.
    """
    if not tok:
        return ""
    t = tok.strip().lower()
    t = t.replace(" ", "_")
    t = re.sub(r"_+", "_", t)
    return t

def species_key(colname: str) -> str:
    return normalize_species_token(extract_species_token(colname))


def align_table2_to_table1_by_species_hardened(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    fill_value: float = 0.0,
    mode_df1_missing: str = "error",       # "error" or "warn"
    mode_df2_missing: str = "error",       # "error" or "drop"
    mode_df1_duplicate_keys: str = "error",# "error" or "warn"
    mode_df2_duplicate_keys: str = "sum",  # "sum" | "first" | "error"
    require_numeric: bool = True,
    require_nonnegative: bool = True,
) -> Tuple[List[str], pd.DataFrame, Dict]:
    """
    Align df2 to df1 by species-level keys (s__...), keeping df1's original column names and order.
    """
    # ---------------------------
    # Basic validation
    # ---------------------------
    if mode_df1_missing not in {"error", "warn"}:
        raise ValueError("mode_df1_missing must be 'error' or 'warn'")
    if mode_df2_missing not in {"error", "drop"}:
        raise ValueError("mode_df2_missing must be 'error' or 'drop'")
    if mode_df1_duplicate_keys not in {"error", "warn"}:
        raise ValueError("mode_df1_duplicate_keys must be 'error' or 'warn'")
    if mode_df2_duplicate_keys not in {"sum", "first", "error"}:
        raise ValueError("mode_df2_duplicate_keys must be 'sum', 'first', or 'error'")

    # ---------------------------
    # Numeric validation/coercion (optional)
    # ---------------------------
    def _ensure_numeric(df: pd.DataFrame, which: str) -> pd.DataFrame:
        if not require_numeric:
            return df
        out = df.copy()
        bad = []
        for c in out.columns:
            if not pd.api.types.is_numeric_dtype(out[c]):
                coerced = pd.to_numeric(out[c], errors="coerce")
                # If coercion would introduce NaNs for existing non-null values, treat as unsafe
                if coerced.isna().any() and out[c].notna().any():
                    bad.append(c)
                out[c] = coerced
        if bad:
            raise ValueError(
                f"{which} has non-numeric columns that cannot be safely coerced: "
                f"{bad[:10]}{' ...' if len(bad) > 10 else ''}"
            )
        return out

    df1n = _ensure_numeric(df1, "df1")
    df2n = _ensure_numeric(df2, "df2")

    if require_nonnegative:
        if (df1n.values < 0).any():
            raise ValueError("df1 contains negative values; unexpected for relative abundance.")
        if (df2n.values < 0).any():
            raise ValueError("df2 contains negative values; unexpected for relative abundance.")

    # ---------------------------
    # Build species keys
    # ---------------------------
    df1_cols = list(df1n.columns)
    df2_cols = list(df2n.columns)

    df1_keys_in_order = [species_key(c) for c in df1_cols]
    df2_keys_all = [species_key(c) for c in df2_cols]

    df1_missing_cols = [c for c, k in zip(df1_cols, df1_keys_in_order) if k == ""]
    df2_missing_cols = [c for c, k in zip(df2_cols, df2_keys_all) if k == ""]

    # Missing-key handling
    if df1_missing_cols:
        msg = f"df1 has {len(df1_missing_cols)} columns with no s__ token. Examples: {df1_missing_cols[:10]}"
        if mode_df1_missing == "error":
            raise ValueError(msg)

    if df2_missing_cols:
        msg = f"df2 has {len(df2_missing_cols)} columns with no s__ token. Examples: {df2_missing_cols[:10]}"
        if mode_df2_missing == "error":
            raise ValueError(msg)

    # Drop df2 missing-key columns if requested
    dropped_df2_missing = 0
    if df2_missing_cols and mode_df2_missing == "drop":
        valid_mask = [k != "" for k in df2_keys_all]
        df2_work = df2n.loc[:, valid_mask].copy()
        df2_keys = [k for k in df2_keys_all if k != ""]
        dropped_df2_missing = len(df2_missing_cols)
    else:
        df2_work = df2n.copy()
        df2_keys = df2_keys_all

    # Assign species keys to df2
    df2_work.columns = df2_keys

    # ---------------------------
    # Duplicate species keys
    # ---------------------------
    # df1 duplicates: ambiguous mapping because multiple df1 columns share the same species key
    df1_key_to_cols: Dict[str, List[str]] = {}
    for c, k in zip(df1_cols, df1_keys_in_order):
        if k:
            df1_key_to_cols.setdefault(k, []).append(c)
    df1_dups = {k: cols for k, cols in df1_key_to_cols.items() if len(cols) > 1}
    if df1_dups:
        msg = f"df1 has duplicate species keys (ambiguous). Example: {next(iter(df1_dups.items()))}"
        if mode_df1_duplicate_keys == "error":
            raise ValueError(msg)

    # df2 duplicates: multiple columns collapse to the same species key
    df2_dup_keys = df2_work.columns[df2_work.columns.duplicated()].unique().tolist()
    if df2_dup_keys:
        if mode_df2_duplicate_keys == "error":
            raise ValueError(f"df2 has duplicate species keys after normalization: {df2_dup_keys[:10]}")
        if mode_df2_duplicate_keys == "sum":
            df2_work = df2_work.groupby(level=0, axis=1).sum()
        else:  # "first"
            df2_work = df2_work.loc[:, ~df2_work.columns.duplicated(keep="first")]

    # ---------------------------
    # Align df2 into df1 space by species keys
    # ---------------------------
    df2_aligned_norm = df2_work.reindex(columns=df1_keys_in_order, fill_value=fill_value)

    # Restore df1's original taxonomy-rich column names and order
    df2_aligned = df2_aligned_norm.copy()
    df2_aligned.columns = df1_cols

    # Common species reported as df1 columns
    df2_key_set = set(df2_work.columns)
    common_df1_cols = [c for c, k in zip(df1_cols, df1_keys_in_order) if k and k in df2_key_set]

    report = {
        "n_df1_cols": len(df1_cols),
        "n_df2_cols": len(df2_cols),
        "n_common_species_keys": len(set(species_key(c) for c in common_df1_cols)),
        "df1_missing_s_token_count": len(df1_missing_cols),
        "df2_missing_s_token_count": len(df2_missing_cols),
        "df2_missing_dropped_count": dropped_df2_missing,
        "df1_duplicate_species_keys_count": len(df1_dups),
        "df2_duplicate_species_keys_count": len(df2_dup_keys),
        "df1_missing_examples": df1_missing_cols[:20],
        "df2_missing_examples": df2_missing_cols[:20],
        "df2_duplicate_keys_examples": df2_dup_keys[:20],
        # If df1 duplicates are only warned, keep them for audit:
        "df1_duplicate_keys_map": df1_dups if mode_df1_duplicate_keys == "warn" else {},
    }

    return common_df1_cols, df2_aligned, report


# In[3]:


path1 = "bgi_data_folder/"
ab1 = pd.read_csv(path1 + "microbe_comp_te.csv")
path2 = "lc_data_folder/"
ab2 = pd.read_csv(path2 + "microbe_comp_all_lc.csv")


# In[4]:


common_cols, ab2_aligned, report = align_table2_to_table1_by_species_hardened(
    ab1, ab2,
    fill_value=0.0,
    mode_df1_missing="error",
    mode_df2_missing="error",          # table2 should be species-only; fail fast if not
    mode_df1_duplicate_keys="error",   # strongly recommended
    mode_df2_duplicate_keys="sum"      # safe if duplicates ever occur
)

print(report)


# In[6]:


# common_cols: list of df1 column names (full taxonomy strings)
common_df = pd.DataFrame(
    {"species": common_cols}
)

common_df.to_csv(
    path2 + "common_species_bgilc.csv",
    index=False
)


# In[7]:


ab2_aligned.to_csv(path2 + "microbe_comp_lc.csv", index=False)


# In[ ]:




