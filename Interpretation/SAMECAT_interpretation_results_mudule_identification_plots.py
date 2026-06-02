#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


master_path = "master_path/"
output_cloud = "output_results_path/"


# In[ ]:


def plot_multi_site_similarity_with_module_pies(
    site_configs,
    n_cols=2,
    figsize=(20, 24),
    dpi=300,
    global_title="Feature cosine similarity (signed-log EG)",
    global_title_fontsize=24,
    global_title_y=0.965,
    panel_title_fontsize=18,
    xlabel_fontsize=18,
    xlabel_labelpad=6,
    bracket_label_fontsize=12,
    bracket_linewidth=1.2,
    boundary_lw=1.5,
    pie_title_fontsize=18,
    pie_text_fontsize=18,
    pie_ann_fontsize=18,
    cbar_label_fontsize=18,
    cbar_tick_fontsize=14,
    cbar_box=(0.92, 0.26, 0.018, 0.34),  # [left, bottom, width, height]
    # fixed pie sizing across all panels
    pie_box_w=0.22,
    pie_box_h=0.30,
    pie_row1_y=0.63,
    pie_row2_y=0.08,
    # one global legend
    show_global_pie_legend=True,
    global_pie_legend_anchor=(0.50, 0.545),
    global_pie_legend_fontsize=16,
    save_path_png=None,
    save_path_pdf=None,
    module_col_preference=("merged_module", "module"),
    feature_col="feature",
    max_features=None,
):
    """
    Multi-panel figure:
      - bracketed module spans
      - cosine similarity heatmap
      - fixed-size pie charts arranged in 2 rows x 3 cols below each heatmap
      - one shared colorbar
      - one shared Positive/Negative legend
    """

    def _signed_log_transform(E_raw):
        abs_vals = np.abs(E_raw)
        nonzero = abs_vals[abs_vals > 0]
        eps = float(np.median(nonzero)) if nonzero.size else 1e-12
        E_t = np.sign(E_raw) * np.log1p(abs_vals / eps)
        return E_t, eps

    def _prepare_module_ordered_similarity(
        eg_csv,
        module_csv,
        feature_col="feature",
        module_col_preference=("merged_module", "module"),
        max_features=None,
    ):
        eg_df = pd.read_csv(eg_csv, index_col=0).apply(pd.to_numeric, errors="coerce")

        mods = pd.read_csv(module_csv)
        if feature_col not in mods.columns:
            feature_col_use = mods.columns[0]
        else:
            feature_col_use = feature_col

        module_col = None
        for c in module_col_preference:
            if c in mods.columns:
                module_col = c
                break
        if module_col is None:
            candidate_cols = [c for c in mods.columns if c != feature_col_use]
            if not candidate_cols:
                raise ValueError(f"No module column found in {module_csv}")
            module_col = candidate_cols[0]

        mods = mods[[feature_col_use, module_col]].dropna().copy()
        mods[module_col] = mods[module_col].astype(int)

        common_features = eg_df.columns.intersection(mods[feature_col_use])
        eg_df = eg_df[common_features]

        mod_map = mods.set_index(feature_col_use)[module_col].to_dict()
        modules = pd.Series({f: mod_map[f] for f in common_features})

        if max_features is not None and len(common_features) > max_features:
            keep = eg_df.mean(axis=0).abs().sort_values(ascending=False).head(max_features).index
            eg_df = eg_df[keep]
            modules = modules.loc[keep]

        E_raw = eg_df.values
        E_t, eps = _signed_log_transform(E_raw)
        S = cosine_similarity(E_t.T)

        mean_abs_t = np.abs(E_t).mean(axis=0)
        order_df = pd.DataFrame({
            "feature": eg_df.columns,
            "module": modules.values,
            "mean_abs_t": mean_abs_t
        }).sort_values(["module", "mean_abs_t"], ascending=[True, False])

        ordered_features = order_df["feature"].tolist()
        idx = [eg_df.columns.get_loc(f) for f in ordered_features]
        S_ord = S[np.ix_(idx, idx)]
        mod_ord = order_df["module"].values.astype(int)

        return {
            "S_ord": S_ord,
            "mod_ord": mod_ord,
            "eps": eps,
            "n_features": eg_df.shape[1],
            "n_samples": eg_df.shape[0],
            "ordered_features": ordered_features,
        }

    def _plot_compact_pie(
        row,
        ax,
        title=None,
        class_to_color=None,
        pie_text_fontsize=18,
        ann_fontsize=18,
        title_fontsize=18,
    ):
        if class_to_color is None:
            class_to_color = {
                "pro-BMD": "darkred",
                "anti-BMD": "darkblue",
                "mixed": "dimgray",
                "neutral": "dimgray",
                "mixed/ambiguous": "dimgray",
                "pro-bmd": "darkred",
                "anti-bmd": "darkblue",
            }

        frac_pos = float(row["frac_pos"])
        frac_neg = float(row["frac_neg"])

        ax.pie(
            [frac_pos, frac_neg],
            startangle=90,
            counterclock=False,
            autopct=lambda p: f"{p:.1f}%",
            pctdistance=0.7,
            radius=1.0,
            wedgeprops=dict(edgecolor="white", linewidth=1.0),
            textprops=dict(fontsize=pie_text_fontsize),
        )

        ax.set(aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])

        mean_signed = float(row["mean_signed"])
        mean_abs = float(row["mean_abs"])

        annotation_text = (
            f"n={int(row['n_features'])}\n"
            f"signed={mean_signed:+.1e}\n"
            f"abs={mean_abs:+.1e}\n"
            f"{row['direction_class']}"
        )

        ax.text(
            0,
            -1.18,
            annotation_text,
            ha="center",
            va="top",
            fontsize=ann_fontsize,
        )

        if title is not None:
            dclass = str(row["direction_class"])
            tcolor = class_to_color.get(dclass, "black")
            ax.set_title(title, fontsize=title_fontsize, pad=6, color=tcolor)

    def _add_module_pie_grid_fixed_size(
        ax_pies,
        summary_df,
        module_col="merged_module",
        pie_box_w=0.22,
        pie_box_h=0.30,
        pie_row1_y=0.63,
        pie_row2_y=0.08,
        pie_title_fontsize=18,
        pie_text_fontsize=18,
        pie_ann_fontsize=18,
    ):
        ax_pies.set_axis_off()

        tmp = summary_df.copy()
        if module_col not in tmp.columns:
            if "module" in tmp.columns:
                module_col_use = "module"
            elif "merged_module" in tmp.columns:
                module_col_use = "merged_module"
            else:
                raise ValueError("No module column found in pie summary dataframe.")
        else:
            module_col_use = module_col

        tmp[module_col_use] = tmp[module_col_use].astype(int)
        tmp = tmp.sort_values(module_col_use).reset_index(drop=True)

        # fixed 2x3 positions
        x_centers = [1/6, 3/6, 5/6]
        y_centers = [pie_row1_y, pie_row2_y]

        max_slots = 6
        n_modules = min(len(tmp), max_slots)

        for i in range(n_modules):
            row_idx = i // 3
            col_idx = i % 3

            cx = x_centers[col_idx]
            cy = y_centers[row_idx]

            left = cx - pie_box_w / 2
            bottom = cy - pie_box_h / 2

            ax_in = ax_pies.inset_axes([left, bottom, pie_box_w, pie_box_h])

            row = tmp.iloc[i]
            _plot_compact_pie(
                row=row,
                ax=ax_in,
                title=f"Module {int(row[module_col_use])}",
                pie_text_fontsize=pie_text_fontsize,
                ann_fontsize=pie_ann_fontsize,
                title_fontsize=pie_title_fontsize,
            )

    def _draw_one_panel(
        fig,
        subspec,
        S_ord,
        mod_ord,
        pie_summary_df,
        site_name,
        panel_letter=None,
        boundary_lw=1.5,
        panel_title_fontsize=18,
        xlabel_fontsize=18,
        xlabel_labelpad=6,
        bracket_label_fontsize=12,
        bracket_linewidth=1.2,
        pie_title_fontsize=18,
        pie_text_fontsize=18,
        pie_ann_fontsize=18,
        pie_box_w=0.22,
        pie_box_h=0.30,
        pie_row1_y=0.63,
        pie_row2_y=0.08,
    ):
        mod_ord = np.asarray(mod_ord).astype(int)
        n = S_ord.shape[0]

        change_idx = np.where(mod_ord[1:] != mod_ord[:-1])[0] + 1
        boundaries = change_idx.tolist()

        starts = np.r_[0, change_idx]
        ends = np.r_[change_idx, n]
        seg_mods = mod_ord[starts]
        mids = (starts + ends - 1) / 2.0

        inner = GridSpecFromSubplotSpec(
            3, 1,
            subplot_spec=subspec,
            height_ratios=[0.06, 1.0, 1.60],
            hspace=0.05
        )

        ax_top = fig.add_subplot(inner[0, 0])
        ax_main = fig.add_subplot(inner[1, 0])
        ax_pies = fig.add_subplot(inner[2, 0])

        panel_title = site_name if panel_letter is None else f"({panel_letter}) {site_name}"
        ax_top.set_title(panel_title, fontsize=panel_title_fontsize, pad=12)

        im = ax_main.imshow(
            S_ord,
            aspect="auto",
            interpolation="nearest",
            vmin=-1,
            vmax=1
        )

        ax_main.set_xlabel(
            "Features (module-ordered)",
            fontsize=xlabel_fontsize,
            labelpad=xlabel_labelpad
        )
        ax_main.set_ylabel("")
        ax_main.set_xticks([])
        ax_main.set_yticks([])

        for b in boundaries:
            ax_main.axhline(b - 0.5, color="white", linewidth=boundary_lw)
            ax_main.axvline(b - 0.5, color="white", linewidth=boundary_lw)

        ax_top.set_xlim(-0.5, n - 0.5)
        ax_top.set_ylim(0, 1)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for spine in ax_top.spines.values():
            spine.set_visible(False)

        y_top = 0.42
        y_bottom = 0.22
        for start, end, mod, mid in zip(starts, ends, seg_mods, mids):
            x0 = start - 0.5
            x1 = end - 0.5
            ax_top.plot([x0, x1], [y_top, y_top], color="black", lw=bracket_linewidth, clip_on=False)
            ax_top.plot([x0, x0], [y_bottom, y_top], color="black", lw=bracket_linewidth, clip_on=False)
            ax_top.plot([x1, x1], [y_bottom, y_top], color="black", lw=bracket_linewidth, clip_on=False)
            ax_top.text(mid, y_top + 0.08, str(mod), ha="center", va="bottom", fontsize=bracket_label_fontsize)

        _add_module_pie_grid_fixed_size(
            ax_pies=ax_pies,
            summary_df=pie_summary_df,
            module_col="merged_module",
            pie_box_w=pie_box_w,
            pie_box_h=pie_box_h,
            pie_row1_y=pie_row1_y,
            pie_row2_y=pie_row2_y,
            pie_title_fontsize=pie_title_fontsize,
            pie_text_fontsize=pie_text_fontsize,
            pie_ann_fontsize=pie_ann_fontsize,
        )

        return im

    n_panels = len(site_configs)
    n_rows = math.ceil(n_panels / n_cols)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.suptitle(global_title, fontsize=global_title_fontsize, y=global_title_y)

    outer = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        hspace=0.22,
        wspace=0.12
    )

    ims = []

    for i, cfg in enumerate(site_configs):
        r = i // n_cols
        c = i % n_cols

        feature_col_use = cfg.get("feature_col", feature_col)
        module_col_pref_use = cfg.get("module_col_preference", module_col_preference)
        max_features_use = cfg.get("max_features", max_features)

        prepared = _prepare_module_ordered_similarity(
            eg_csv=cfg["eg_csv"],
            module_csv=cfg["module_csv"],
            feature_col=feature_col_use,
            module_col_preference=module_col_pref_use,
            max_features=max_features_use,
        )

        pie_summary_df = pd.read_csv(cfg["pie_csv"])

        im = _draw_one_panel(
            fig=fig,
            subspec=outer[r, c],
            S_ord=prepared["S_ord"],
            mod_ord=prepared["mod_ord"],
            pie_summary_df=pie_summary_df,
            site_name=cfg["site_name"],
            panel_letter=cfg.get("panel_letter", None),
            boundary_lw=cfg.get("boundary_lw", boundary_lw),
            panel_title_fontsize=cfg.get("panel_title_fontsize", panel_title_fontsize),
            xlabel_fontsize=cfg.get("xlabel_fontsize", xlabel_fontsize),
            xlabel_labelpad=cfg.get("xlabel_labelpad", xlabel_labelpad),
            bracket_label_fontsize=cfg.get("bracket_label_fontsize", bracket_label_fontsize),
            bracket_linewidth=cfg.get("bracket_linewidth", bracket_linewidth),
            pie_title_fontsize=cfg.get("pie_title_fontsize", pie_title_fontsize),
            pie_text_fontsize=cfg.get("pie_text_fontsize", pie_text_fontsize),
            pie_ann_fontsize=cfg.get("pie_ann_fontsize", pie_ann_fontsize),
            pie_box_w=cfg.get("pie_box_w", pie_box_w),
            pie_box_h=cfg.get("pie_box_h", pie_box_h),
            pie_row1_y=cfg.get("pie_row1_y", pie_row1_y),
            pie_row2_y=cfg.get("pie_row2_y", pie_row2_y),
        )
        ims.append(im)

    # one shared colorbar
    cax = fig.add_axes(cbar_box)
    cb = fig.colorbar(ims[0], cax=cax)
    cb.set_label("Cosine similarity", fontsize=cbar_label_fontsize)
    cb.ax.tick_params(labelsize=cbar_tick_fontsize)

    # one shared pie legend
    if show_global_pie_legend:
        handles = [
            Patch(facecolor="#1f77b4", edgecolor="none", label="Positive EG"),
            Patch(facecolor="#ff7f0e", edgecolor="none", label="Negative EG"),
        ]
        fig.legend(
            handles=handles,
            loc="center",
            bbox_to_anchor=global_pie_legend_anchor,
            ncol=1,
            frameon=False,
            fontsize=global_pie_legend_fontsize,
        )

    if save_path_png is not None:
        fig.savefig(
            save_path_png,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05
        )

    if save_path_pdf is not None:
        fig.savefig(
            save_path_pdf,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.05
        )

    plt.show()
    return fig


# In[ ]:


# /*target_div*/: "Determined by the internal evaluation procedure, see EG_computation.py"
site_configs = [
    {
        "site_name": "Femoral Neck",
        "panel_letter": "a",
        "eg_csv": master_path + "NECK_BMD_" + "/*target_div_fneck*/" + "_EG_attr_mgs_test_df.csv",
        "module_csv": master_path + "eg_spectral_outputs/FNECK/eg_final_outputs/fneck_final_assignments_original_and_merged.csv",
        "pie_csv": master_path + "eg_spectral_outputs/FNECK/eg_final_outputs/merged_module_directionality_summary.csv",
    },
    {
        "site_name": "Total Hip",
        "panel_letter": "b",
        "eg_csv": master_path + "HTOT_BMD_" + "/*target_div_htot*/" + "_EG_attr_mgs_test_df.csv",
        "module_csv": master_path + "eg_spectral_outputs/HTOT/eg_final_outputs/htot_final_assignments_original_and_merged.csv",
        "pie_csv": master_path + "eg_spectral_outputs/HTOT/eg_final_outputs/merged_module_directionality_summary.csv",
    },
    {
        "site_name": "Total Spine",
        "panel_letter": "c",
        "eg_csv": master_path + "spine_total_bmd_" + "/*target_div_stot*/" + "_EG_attr_mgs_test_df.csv",
        "module_csv": master_path + "eg_spectral_outputs/STOT/eg_final_outputs/stot_final_assignments_original_and_merged.csv",
        "pie_csv": master_path + "eg_spectral_outputs/STOT/eg_final_outputs/merged_module_directionality_summary.csv",
    },
    {
        "site_name": "1/3 Radius",
        "panel_letter": "d",
        "eg_csv": master_path + "R_13_BMD_" + "/*target_div_r13*/" + "_EG_attr_mgs_test_df.csv",
        "module_csv": master_path + "eg_spectral_outputs/R13/eg_final_outputs/r13_final_assignments_original_and_merged.csv",
        "pie_csv": master_path + "eg_spectral_outputs/R13/eg_final_outputs/merged_module_directionality_summary.csv",
    },
]

fig = plot_multi_site_similarity_with_module_pies(
    site_configs=site_configs,
    n_cols=2,
    figsize=(22, 28),
    dpi=300,
    global_title="Feature cosine similarity (signed-log EG)",
    global_title_fontsize=24,
    global_title_y=0.935,
    panel_title_fontsize=18,
    xlabel_fontsize=18,
    xlabel_labelpad=6,
    bracket_label_fontsize=16,
    pie_title_fontsize=18,
    pie_text_fontsize=18,
    pie_ann_fontsize=18,
    pie_box_w=0.22,
    pie_box_h=0.30,
    pie_row1_y=0.65,
    pie_row2_y=0.05,
    show_global_pie_legend=True,
    #global_pie_legend_anchor=(0.50, 0.545),
    global_pie_legend_anchor=(0.98, 0.46),
    global_pie_legend_fontsize=16,
    cbar_box=(0.95, 0.5, 0.018, 0.34),
    save_path_png=output_cloud + "multi_site_similarity_with_module_pies.png",
    save_path_pdf=output_cloud + "multi_site_similarity_with_module_pies.pdf",
)






