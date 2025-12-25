# analyze_decisions_step8.py — Step 8: Decision & robustness summaries from Step 7 outputs
# Inputs:
#   experiments/intervention_results_with_deltas.csv
#   experiments/summary_by_strategy_tau_k.csv
#   experiments/stats_vs_random.csv                (optional use in the report)
#   preprocessed/thread_descriptives.csv          (optional; for stratification)
#
# Outputs (experiments/decisions/):
#   decision_matrix_k*.png      — clear-improvement grid per k (ΔAUC_M)
#   bps_auc_vs_tau_k*.png       — benefit-per-seed vs τ (ΔAUC_M)
#   top_cells_table.tex         — LaTeX table of best cells by ΔAUC_M (and by ΔM_final)
#   stratified_size_*.png       — stratified effect by thread size terciles (if available)
#   stratified_duration_*.png   — stratified effect by duration terciles (if available)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("experiments")
IN_SUM = BASE / "summary_by_strategy_tau_k.csv"
IN_PER = BASE / "intervention_results_with_deltas.csv"
IN_STATS = BASE / "stats_vs_random.csv"
DESC = Path("preprocessed") / "thread_descriptives.csv"

OUT_DIR = BASE / "decisions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Load
print(">>> analyze_decisions_step8.py STARTED <<<")
summ = pd.read_csv(IN_SUM)
per  = pd.read_csv(IN_PER)
stats = pd.read_csv(IN_STATS) if IN_STATS.exists() else None

# Add benefit-per-seed
summ["bps_aucM"]   = -summ["mean_dAUC_M"]   / summ["k"]   # more negative AUC is better → higher BPS
summ["bps_finalM"] = -summ["mean_dM_final"] / summ["k"]

# Clear-improvement flags (95% CI strictly below 0)
summ["sig_aucM"]   = (summ["mean_dAUC_M"]   + summ["ci95_dAUC_M"]) < 0
summ["sig_finalM"] = (summ["mean_dM_final"] + summ["ci95_dM_final"]) < 0

# ---------------- Decision matrices (per k)
def plot_grid_for_k(df, k, metric="mean_dAUC_M", sig_col="sig_aucM", title=None, fname=None):
    d = df[df["k"] == k].copy()
    # order axes
    taus = sorted(d["tau_min"].unique())
    strats = ["earliest","hubs","bridges","community","random"]
    # grid of means
    grid = np.full((len(strats), len(taus)), np.nan)
    sigs = np.zeros_like(grid, dtype=bool)
    for i, s in enumerate(strats):
        for j, t in enumerate(taus):
            row = d[(d["strategy"]==s) & (d["tau_min"]==t)]
            if not row.empty:
                grid[i, j] = float(row[metric].iloc[0])
                sigs[i, j] = bool(row[sig_col].iloc[0])
    # plot
    plt.figure(figsize=(6.5, 3.8), dpi=160)
    im = plt.imshow(grid, aspect="auto", origin="upper", cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=metric)
    plt.xticks(range(len(taus)), taus); plt.xlabel("τ (minutes)")
    plt.yticks(range(len(strats)), strats); plt.ylabel("strategy")
    ttl = title or f"{metric} by strategy × τ (k={k})\n(negative is better)"
    plt.title(ttl)
    # overlay markers for “clear improvement” (CI below 0)
    for i in range(len(strats)):
        for j in range(len(taus)):
            if sigs[i, j]:
                plt.text(j, i, "★", ha="center", va="center", fontsize=10, color="black")
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

for k in sorted(summ["k"].unique()):
    plot_grid_for_k(
        summ, k,
        metric="mean_dAUC_M", sig_col="sig_aucM",
        title=f"ΔAUC_M (negative better) — k={k}",
        fname=OUT_DIR / f"decision_matrix_k{k}.png"
    )

# ---------------- Benefit-per-seed vs τ (lines per strategy) for each k
def bps_plot(df, k, metric="bps_aucM", ylabel="Benefit-per-seed (−ΔAUC_M / k)", fname=None):
    d = df[df["k"]==k].copy()
    taus = sorted(d["tau_min"].unique())
    strats = ["earliest","hubs","bridges","community","random"]
    plt.figure(figsize=(6.8, 3.6), dpi=160)
    for s in strats:
        y = []
        for t in taus:
            sel = d.loc[(d["strategy"]==s) & (d["tau_min"]==t), metric]
            y.append(float(sel.iloc[0]) if not sel.empty else np.nan)

        plt.plot(taus, y, marker="o", label=s)
    plt.xlabel("τ (minutes)"); plt.ylabel(ylabel)
    plt.title(f"Cost-effectiveness vs delay — k={k}")
    plt.legend(frameon=False, ncol=3)
    plt.tight_layout()
    if fname:
        plt.savefig(fname); plt.close()
    else:
        plt.show()

for k in sorted(summ["k"].unique()):
    bps_plot(summ, k, metric="bps_aucM",
             ylabel="Benefit-per-seed (−ΔAUC_M / k)",
             fname=OUT_DIR / f"bps_auc_vs_tau_k{k}.png")

# ---------------- Top cells → LaTeX tables (by ΔAUC_M and by ΔM_final)
def top_cells_table(df, topn=10, metric="mean_dAUC_M", ci="ci95_dAUC_M"):
    d = (df[["strategy","tau_min","k",metric,ci,"sig_aucM","sig_finalM"]]
         .copy()
         .sort_values(metric))  # ascending since negative is good
    d = d.rename(columns={
        "tau_min": r"$\tau$",
        "k": r"$k$",
        "strategy": "strategy",
        metric: metric,
        ci: "ci95"
    }).head(topn)
    return d

topA = top_cells_table(summ, topn=12, metric="mean_dAUC_M", ci="ci95_dAUC_M")
topF = top_cells_table(summ, topn=12, metric="mean_dM_final", ci="ci95_dM_final")

with open(OUT_DIR / "top_cells_table.tex", "w") as f:
    f.write("% Auto-generated by Step 8\n")
    f.write("\\begin{table}[t]\n\\centering\n")
    f.write("\\caption{Best cells by mean $\\Delta$AUC$_M$ (more negative is better). Stars mark 95\\% CIs entirely below 0.}\n")
    f.write(topA.to_latex(index=False, float_format=lambda x: f"{x:.3g}"))
    f.write("\\vspace{1ex}\n")
    f.write("\\caption{Best cells by mean $\\Delta M_{final}$ (more negative is better).}\n")
    f.write(topF.to_latex(index=False, float_format=lambda x: f"{x:.3g}"))
    f.write("\\end{table}\n")

# ---------------- Optional stratification (by thread size/duration if available)
if DESC.exists():
    desc = pd.read_csv(DESC)
    # Expect columns like: thread_id, n_nodes (size), duration_min (or similar)
    key = "thread_id"
    if key in per.columns and key in desc.columns:
        merged = (per.merge(desc[[key] + [c for c in desc.columns if c != key]], on=key, how="left")
                    .dropna(subset=["dAUC_M"]))
        # Make size and duration terciles if available
        if "n_nodes" in merged.columns:
            merged["size_bin"] = pd.qcut(merged["n_nodes"], q=3, labels=["small","medium","large"])
        if "duration_min" in merged.columns:
            merged["dur_bin"] = pd.qcut(merged["duration_min"], q=3, labels=["short","medium","long"])

        def strat_plot(df, group_col, title, fname):
            # average ΔAUC_M by τ within each bin, aggregated over k and strategy or filter if desired
            if group_col not in df.columns:
                return
            # Example: show earliest vs hubs vs community for k=3 as a compact view
            view = df[(df["k"]==3) & (df["strategy"].isin(["earliest","hubs","community"]))]
            taus = sorted(view["tau_min"].unique())
            bins = df[group_col].dropna().unique()
            for b in bins:
                plt.figure(figsize=(6.6,3.6), dpi=160)
                sub = view[view[group_col]==b]
                for s in ["earliest","hubs","community"]:
                    y = [sub[(sub["strategy"]==s)&(sub["tau_min"]==t)]["dAUC_M"].mean() if not sub[(sub["strategy"]==s)&(sub["tau_min"]==t)].empty else np.nan
                         for t in taus]
                    plt.plot(taus, y, marker="o", label=s)
                plt.axhline(0, color="k", linewidth=0.8, alpha=0.5)
                plt.xlabel("τ (minutes)"); plt.ylabel("mean ΔAUC_M (negative better)")
                plt.title(f"{title}: {b}")
                plt.legend(frameon=False)
                plt.tight_layout()
                plt.savefig(OUT_DIR / f"{fname}_{b}.png"); plt.close()

        if "size_bin" in merged.columns:
            strat_plot(merged, "size_bin", "Stratified by thread size (k=3)", "stratified_size_k3")
        if "dur_bin" in merged.columns:
            strat_plot(merged, "dur_bin", "Stratified by duration (k=3)", "stratified_duration_k3")

print(f"Saved decision matrices, BPS plots, and LaTeX table to {OUT_DIR}")
print(">>> Step 8 COMPLETE <<<")
