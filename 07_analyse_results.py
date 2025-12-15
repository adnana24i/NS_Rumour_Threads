# analyze_results.py  (Step 7 — Evaluation & Reporting)

import json, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy import stats
import matplotlib.pyplot as plt

# ---------------- I/O ----------------
DATA_DIR = Path("preprocessed")
CALIB_DIR = Path("calibration")
EXP_DIR  = Path("experiments")
FIG_DIR  = EXP_DIR / "figs"
FIG_DIR.mkdir(exist_ok=True, parents=True)

TWEETS = DATA_DIR / "tweets_clean.csv"
EDGES  = DATA_DIR / "edges_clean.csv"
RES_PT = EXP_DIR / "intervention_results_per_thread.csv"
RES_AG = EXP_DIR / "intervention_results_aggregate.csv"
CALIB  = CALIB_DIR / "calibrated_ic_parameters.json"

OUT_BASELINE   = EXP_DIR / "baseline_no_intervention.csv"
OUT_WITHDELTA  = EXP_DIR / "intervention_results_with_deltas.csv"
OUT_SUMMARY    = EXP_DIR / "summary_by_strategy_tau_k.csv"
OUT_STATS      = EXP_DIR / "stats_vs_random.csv"

# ---------------- Config ----------------
BIN_MIN = 5          # minutes per bin (must match Step 6)
N_BASELINE_SIM = 20  # sims per thread for the no-intervention baseline (fast)

# ---------------- Helpers (reused from Step 6, trimmed) ----------------
def auc_trapezoid(y):
    if len(y) <= 1:
        return float(y[-1]) if len(y) else 0.0
    y = np.asarray(y, float)
    return float(np.trapz(y, dx=1.0))

def prebuild_threads_fast(edges_df, tweets_df):
    tw = tweets_df[['tweet_id', 'author_id', 'created_at', 'thread_id']].copy()
    tw['tweet_id'] = tw['tweet_id'].astype(str)
    tw['author_id'] = tw['author_id'].astype(str)

    e = edges_df[['thread_id','parent_tweet_id','child_tweet_id']].copy()
    e['parent_tweet_id'] = e['parent_tweet_id'].astype(str)
    e['child_tweet_id']  = e['child_tweet_id'].astype(str)

    # attach parent/child authors and child times
    e = e.merge(tw[['tweet_id','author_id']].rename(
                columns={'tweet_id':'parent_tweet_id','author_id':'parent_author'}),
                on='parent_tweet_id', how='left')
    e = e.merge(tw[['tweet_id','author_id','created_at']].rename(
                columns={'tweet_id':'child_tweet_id','author_id':'child_author','created_at':'child_time'}),
                on='child_tweet_id', how='left')

    e = e.dropna(subset=['child_time']).sort_values(['thread_id','child_time'])

    t0_series = e.groupby('thread_id')['child_time'].transform('min')
    dt_min = (e['child_time'] - t0_series).dt.total_seconds() / 60.0
    e['dt_min'] = dt_min.astype(float)

    cache = {}
    for tid, g in e.groupby('thread_id', sort=False):
        arr_parent = g['parent_author'].astype(str).to_numpy()
        arr_child  = g['child_author'].astype(str).to_numpy()
        arr_dt     = g['dt_min'].to_numpy()
        source_author = str(g['parent_author'].iloc[0])
        users_thr = tw.loc[tw['thread_id'] == tid, 'author_id'].astype(str).unique()
        cache[tid] = (arr_parent, arr_child, arr_dt, source_author, set(users_thr))
    return cache

def simulate_thread_ic_fast(par, chi, dts, source_author, tau_min, seeds_F,
                            lam_M, lam_F, n_bins, bin_min=5, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    users = np.unique(np.concatenate([par, chi]))
    idx = {u:i for i,u in enumerate(users)}
    U = users.size

    state = np.zeros(U, dtype=np.int8)  # 0=S, 1=M, 2=F
    if source_author in idx:
        state[idx[source_author]] = 1

    seeded = False
    max_bin = n_bins - 1
    M_curve = np.zeros(n_bins, dtype=np.int32)
    rnd = rng.random(len(dts))

    for i in range(len(dts)):
        t = dts[i]

        if (not seeded) and t >= tau_min:
            for s in seeds_F:
                j = idx.get(s, None)
                if j is not None and state[j] == 0:
                    state[j] = 2
            seeded = True

        u = idx.get(par[i], None)
        v = idx.get(chi[i], None)
        if v is not None and state[v] == 0 and u is not None:
            if state[u] == 1 and rnd[i] < lam_M:
                state[v] = 1
            elif state[u] == 2 and rnd[i] < lam_F:
                state[v] = 2

        b = int(min(max_bin, max(0, int(t // bin_min))))
        M_curve[b] = (state == 1).sum()

    for i in range(1, n_bins):
        if M_curve[i] < M_curve[i-1]:
            M_curve[i] = M_curve[i-1]
    return M_curve

# ---------------- Load data & calibration ----------------
print(">>> analyze_results.py STARTED <<<")
tweets = pd.read_csv(TWEETS, parse_dates=['created_at'])
edges  = pd.read_csv(EDGES)
res_pt = pd.read_csv(RES_PT)
res_ag = pd.read_csv(RES_AG)

with open(CALIB, "r") as f:
    calib = json.load(f)

lam_false = float(calib["lambdas"].get("false", {}).get("mean", 0.15))
lam_unver = float(calib["lambdas"].get("unverified", {}).get("mean", 0.12))
LAM_F_FACTOR = 0.9
lam_F_false = lam_false * LAM_F_FACTOR
lam_F_unver = lam_unver * LAM_F_FACTOR

# thread -> veracity bucket
ver_by_thread = (tweets
                 .dropna(subset=["thread_id"])
                 .assign(v=tweets["veracity_normalized"].astype(str).str.lower())
                 .groupby("thread_id")["v"].first().to_dict())

# ---------------- Compute no-intervention baseline ----------------
print("Prebuilding per-thread arrays for baseline...")
cache = prebuild_threads_fast(edges, tweets)
threads = list(cache.keys())

baseline_rows = []
rng = np.random.default_rng(123)

print("Simulating no-intervention baseline per thread...")
for tid in tqdm(threads, total=len(threads)):
    par, chi, dts, source_author, _ = cache[tid]
    if par.size == 0:
        continue
    max_dt = float(dts.max())
    n_bins = max(1, int(math.floor(max_dt / BIN_MIN)) + 1)

    vnorm = ver_by_thread.get(tid, "unverified")
    if vnorm == "false":
        lam_M, lam_F = lam_false, lam_F_false
    else:
        lam_M, lam_F = lam_unver, lam_F_unver

    curves = []
    for _ in range(N_BASELINE_SIM):
        c = simulate_thread_ic_fast(
            par, chi, dts, source_author,
            tau_min=np.inf,          # never seed (no intervention)
            seeds_F=set(),
            lam_M=lam_M, lam_F=lam_F,
            n_bins=n_bins, bin_min=BIN_MIN, rng=rng
        )
        curves.append(c)
    curves = np.stack(curves, axis=0)

    aucs   = [auc_trapezoid(c) for c in curves]
    finals = curves[:, -1]
    peaks  = curves.max(axis=1)
    peak_t = curves.argmax(axis=1)

    baseline_rows.append({
        "thread_id": tid,
        "n_bins": n_bins,
        "auc_M_base": float(np.mean(aucs)),
        "M_final_base": float(np.mean(finals)),
        "peak_M_base": float(np.mean(peaks)),
        "peak_t_base": float(np.mean(peak_t)),
        "runs_base": int(N_BASELINE_SIM),
        "veracity_bucket": vnorm
    })

baseline = pd.DataFrame(baseline_rows)
baseline.to_csv(OUT_BASELINE, index=False)
print(f"Saved baseline → {OUT_BASELINE}")

# ---------------- Merge baseline with interventions & compute deltas ----------------
df = res_pt.merge(baseline, on=["thread_id","n_bins"], how="inner")

def delta(col):
    return df[col] - df[col.replace("_mean","_base")]

df["dAUC_M"]     = df["auc_M_mean"]   - df["auc_M_base"]
df["dM_final"]   = df["M_final_mean"] - df["M_final_base"]
df["dPeak_M"]    = df["peak_M_mean"]  - df["peak_M_base"]
df["dPeak_t"]    = df["peak_t_mean"]  - df["peak_t_base"]

df.to_csv(OUT_WITHDELTA, index=False)
print(f"Saved results with deltas → {OUT_WITHDELTA}")

# ---------------- Summaries by strategy, tau, k ----------------
summary = (df
           .groupby(["strategy","tau_min","k"], as_index=False)
           .agg(mean_dAUC_M = ("dAUC_M","mean"),
                ci95_dAUC_M = ("dAUC_M", lambda x: 1.96*np.std(x, ddof=1)/np.sqrt(max(1,len(x)))),
                mean_dM_final = ("dM_final","mean"),
                ci95_dM_final = ("dM_final", lambda x: 1.96*np.std(x, ddof=1)/np.sqrt(max(1,len(x)))),
                n_threads = ("thread_id","nunique"))
          )
summary.to_csv(OUT_SUMMARY, index=False)
print(f"Saved summary → {OUT_SUMMARY}")

# ---------------- Paired tests vs. RANDOM (same tau,k) ----------------
rows = []
for (tau, k), sub in df.groupby(["tau_min","k"]):
    # pick threads where both 'random' and a given strategy exist
    base = sub[sub["strategy"]=="random"][["thread_id","dAUC_M","dM_final"]].rename(
        columns={"dAUC_M":"dAUC_M_rand","dM_final":"dM_final_rand"})
    for strat in sorted(sub["strategy"].unique()):
        if strat == "random": 
            continue
        comp = sub[sub["strategy"]==strat][["thread_id","dAUC_M","dM_final"]]
        joint = comp.merge(base, on="thread_id", how="inner")
        if len(joint) < 20:  # skip tiny sample
            continue
        # note: smaller dAUC_M is better (more negative)
        stat_auc, p_auc = stats.wilcoxon(joint["dAUC_M"], joint["dAUC_M_rand"], zero_method="wilcox", alternative="less")
        stat_fin, p_fin = stats.wilcoxon(joint["dM_final"], joint["dM_final_rand"], zero_method="wilcox", alternative="less")
        rows.append({
            "strategy": strat, "tau_min": tau, "k": k,
            "n_pairs": len(joint),
            "wilcoxon_dAUC_M_stat": float(stat_auc), "wilcoxon_dAUC_M_p": float(p_auc),
            "wilcoxon_dM_final_stat": float(stat_fin), "wilcoxon_dM_final_p": float(p_fin)
        })
stats_df = pd.DataFrame(rows)
stats_df.to_csv(OUT_STATS, index=False)
print(f"Saved stats vs random → {OUT_STATS}")

# ---------------- (Optional) Quick plots ----------------
try:
    piv = summary.pivot_table(index="tau_min", columns="k", values="mean_dAUC_M")
    for strat in df["strategy"].unique():
        piv = (df[df["strategy"]==strat]
               .groupby(["tau_min","k"])["dAUC_M"].mean()
               .unstack("k").sort_index())
        plt.figure(figsize=(6.0,4.2), dpi=140)
        plt.title(f"Mean ΔAUC_M (intervention – baseline): {strat}")
        plt.imshow(piv.values, aspect="auto", origin="lower")
        plt.colorbar(label="ΔAUC_M (negative = better)")
        plt.xlabel("k (seeds)")
        plt.ylabel("τ (minutes)")
        plt.xticks(range(len(piv.columns)), piv.columns)
        plt.yticks(range(len(piv.index)), piv.index)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"heatmap_dAUC_{strat}.png")
        plt.close()
    print(f"Saved heatmaps in {FIG_DIR}")
except Exception as e:
    print("Plotting skipped:", e)

print(">>> Step 7 COMPLETE <<<")
