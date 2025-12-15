# run_interventions.py  (Step 6 — Counterfactual seeding experiments under IC)

import json, math, random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------------- I/O ----------------
DATA_DIR = Path("preprocessed")
CALIB_DIR = Path("calibration")
OUT_DIR = Path("experiments")
OUT_DIR.mkdir(exist_ok=True)

TWEETS = DATA_DIR / "tweets_clean.csv"
EDGES  = DATA_DIR / "edges_clean.csv"
USERF  = DATA_DIR / "G_user_features.csv"
DESC   = DATA_DIR / "thread_descriptives.csv"
CALIB  = CALIB_DIR / "calibrated_ic_parameters.json"   # from Step 5

# ---------------- Config ----------------
# Seeding options
TAU_MINUTES = [0, 15, 30, 60]          # when to inject fact-check seeds after thread starts
SEED_BUDGETS = [1, 3, 5, 10]           # number of seeds
STRATEGIES = ["earliest", "hubs", "bridges", "community", "random"]

# Diffusion
N_SIM = 50                              # simulations per (thread, strategy, tau, k)
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# Binning for prevalence curves / AUC
BIN_MIN = 5

# ---------------- Helpers ----------------
def auc_trapezoid(y):
    # y is list/np.array over equal bins (area under curve)
    if len(y) <= 1:
        return float(y[-1]) if len(y) else 0.0
    y = np.asarray(y, float)
    return float(np.trapz(y, dx=1.0))

def discretize_minutes(series, t0):
    return (((series - t0).dt.total_seconds()) / 60.0).astype(float)

def build_thread_edges(df_edges, df_tw, thread_id):
    e = df_edges[df_edges["thread_id"] == thread_id].copy()
    if e.empty: 
        return None
    # timestamps at child (exposure time)
    t_map = df_tw.set_index("tweet_id")["created_at"].to_dict()
    e["t"] = e["child_tweet_id"].map(t_map)
    e = e.dropna(subset=["t"]).sort_values("t")
    if e.empty:
        return None
    t0 = e["t"].iloc[0]
    e["dt_min"] = discretize_minutes(e["t"], t0)
    # active users in this thread
    users = pd.unique(pd.concat([
        df_tw.loc[df_tw["thread_id"] == thread_id, "author_id"].astype(str),
        e["parent_author"].astype(str),
        e["child_author"].astype(str),
    ], ignore_index=True).dropna())
    return e, t0, set(users)

def pick_seeds(strategy, k, tau_min, e_thread, users_in_thread, df_user):
    # Eligible users: those who have appeared by tau_min (observed so far)
    seen = set(e_thread.loc[e_thread["dt_min"] <= tau_min, "child_author"].astype(str).tolist())
    seen |= {e_thread.iloc[0]["parent_author"]}  # include source
    eligible = [u for u in users_in_thread if u in seen]
    if not eligible:
        return set()

    # --- ensure we only rank among eligible users that have features
    sub_feat = df_user.loc[df_user["user_id"].isin(eligible)].copy()
    if strategy in {"hubs", "bridges", "community"} and sub_feat.empty:
        # fallback when no eligible users have feature rows yet (type mismatch or cold-start)
        # earliest responders is a safe, reproducible fallback
        order = e_thread.drop_duplicates("child_author")
        order = order[order["dt_min"] <= tau_min]["child_author"].astype(str).tolist()
        return set(order[:k])

    if strategy == "earliest":
        # first k distinct responders (child_author order)
        order = e_thread.drop_duplicates("child_author")
        order = order[order["dt_min"] <= tau_min]["child_author"].astype(str).tolist()
        return set(order[:k])

    if strategy == "hubs":
        cand = sub_feat.sort_values(["pagerank","in_degree_w"], ascending=False)["user_id"].tolist()
        return set(cand[:k])

    if strategy == "bridges":
        cand = sub_feat.sort_values(["participation_coeff","betweenness"], ascending=False)["user_id"].tolist()
        return set(cand[:k])

    if strategy == "community":
        # allocate seeds proportional to community mass among eligible users with features
        counts = sub_feat["community_id"].value_counts()
        if counts.empty:
            # fallback to hubs among eligible-with-features
            cand = sub_feat.sort_values(["pagerank","in_degree_w"], ascending=False)["user_id"].tolist()
            return set(cand[:k])
        alloc = {}
        for cid, cnt in counts.items():
            share = cnt / counts.sum()
            alloc[cid] = max(0, int(round(share * k)))
        # fix rounding to exactly k
        diff = k - sum(alloc.values())
        for cid in counts.index[:abs(diff)]:
            alloc[cid] = alloc.get(cid, 0) + (1 if diff > 0 else -1)
        seeds = set()
        # pick within each community by PageRank (then in_degree)
        idx_user = df_user.set_index("user_id")
        for cid, take in alloc.items():
            if take <= 0:
                continue
            bucket = sub_feat[sub_feat["community_id"] == cid]["user_id"].tolist()
            bucket = idx_user.loc[bucket].sort_values(["pagerank","in_degree_w"], ascending=False).index.tolist()
            seeds |= set(bucket[:take])
        # if rounding/availability gave fewer than k, top-up with hubs among the remainder
        if len(seeds) < k:
            remain = [u for u in sub_feat["user_id"].tolist() if u not in seeds]
            remain_sorted = idx_user.loc[remain].sort_values(["pagerank","in_degree_w"], ascending=False).index.tolist()
            for u in remain_sorted:
                if len(seeds) >= k: break
                seeds.add(u)
        return set(list(seeds)[:k])

    if strategy == "random":
        cand = list(eligible)
        random.shuffle(cand)
        return set(cand[:k])

    return set()

########################################################
def simulate_thread_ic_fast(par, chi, dts, source_author, tau_min, seeds_F, lam_M, lam_F, n_bins, bin_min=5):
    """
    par, chi, dts: np arrays of equal length for a thread (parent_author, child_author, dt_min)
    """
    # map users to compact indices
    users = np.unique(np.concatenate([par, chi]))
    idx = {u:i for i,u in enumerate(users)}
    U = users.size

    # states: 0=S, 1=M, 2=F
    state = np.zeros(U, dtype=np.int8)
    if source_author in idx:
        state[idx[source_author]] = 1

    # mark seeds once dt >= tau
    seeded = False

    max_bin = n_bins - 1
    M_curve = np.zeros(n_bins, dtype=np.int32)

    # pre-draw randoms (one per edge)
    rnd = np.random.rand(len(dts))

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
        # incremental update is faster than summing every time:
        # keep a running count of M
        if i == 0:
            m_now = (state == 1).sum()
        else:
            m_now = (state == 1).sum()
        M_curve[b] = m_now

    # forward fill bins
    for i in range(1, n_bins):
        if M_curve[i] < M_curve[i-1]:
            M_curve[i] = M_curve[i-1]
    return M_curve
########################################################

def simulate_thread_ic(e_thread, tau_min, seeds_F, lam_M, lam_F, n_bins):
    """
    Competitive IC on observed contact sequence:
      - initial M at source parent
      - after tau_min, force 'seeds_F' to F
      - on each edge u->v at time t: if S[v] and state[u]==M (resp. F), adopt with lam_M (resp. lam_F)
      - states exclusive; no forgetting
    Returns prevalence curve of M over 'n_bins' bins (bin index = floor(dt / BIN_MIN)).
    """
    # states: 0 S, 1 M, 2 F
    users = pd.unique(pd.concat([e_thread["parent_author"].astype(str), e_thread["child_author"].astype(str)])).tolist()
    state = {u:0 for u in users}
    # source tweet's author becomes M at t0
    source_author = str(e_thread.iloc[0]["parent_author"])
    if source_author in state:
        state[source_author] = 1

    max_bin = n_bins - 1
    M_curve = np.zeros(n_bins, dtype=int)

    def bin_of(dt):
        return int(min(max_bin, max(0, math.floor(dt / BIN_MIN))))

    for _, row in e_thread.iterrows():
        t = row["dt_min"]
        u = str(row["parent_author"])
        v = str(row["child_author"])

        # apply seeding when we reach/ pass tau_min (lazy trigger)
        if "seeded" not in state:
            if t >= tau_min:
                for s in seeds_F:
                    if s in state and state[s]==0:
                        state[s] = 2
                state["seeded"] = True

        # expose v from u
        if v in state and state[v]==0:
            if state[u]==1 and np.random.rand() < lam_M:
                state[v]=1
            elif state[u]==2 and np.random.rand() < lam_F:
                state[v]=2

        # record M count at this bin
        b = bin_of(t)
        M_curve[b] = sum(1 for x in state.values() if x==1)

    # fill forward (some bins may not get an update)
    for i in range(1, n_bins):
        if M_curve[i] < M_curve[i-1]:
            M_curve[i] = M_curve[i-1]
    return M_curve

# ---------------- Load data ----------------
print(">>> run_interventions.py STARTED <<<")

tweets = pd.read_csv(TWEETS, parse_dates=["created_at"])
edges = pd.read_csv(EDGES)
users = pd.read_csv(USERF)
desc  = pd.read_csv(DESC)
with open(CALIB, "r") as f:
    calib = json.load(f)

lam_false = float(calib["lambdas"].get("false", {}).get("mean", 0.15))
lam_unver = float(calib["lambdas"].get("unverified", {}).get("mean", 0.12))
# assume fact-check spread rate comparable or slightly lower; expose as a knob
LAM_F_FACTOR = 0.9
lam_F_false = lam_false * LAM_F_FACTOR
lam_F_unver = lam_unver * LAM_F_FACTOR

# --- CRITICAL: align user ID types so feature lookups succeed
tweets["author_id"] = tweets["author_id"].astype(str)
users["user_id"] = users["user_id"].astype(str)
# (optional) community_id may be float if -1 present; make it int-like but tolerant
if "community_id" in users.columns:
    try:
        users["community_id"] = users["community_id"].astype("Int64").fillna(-1).astype(int)
    except Exception:
        pass

# Map tweet->author and thread
tw2author = tweets.set_index("tweet_id")["author_id"].astype(str).to_dict()
edges["parent_author"] = edges["parent_tweet_id"].map(tw2author)
edges["child_author"]  = edges["child_tweet_id"].map(tw2author)

# Build per-thread index
threads = sorted(edges["thread_id"].dropna().unique().tolist())

########################################################
# ---------- FAST PREBUILD: per-thread edge arrays (one-time) ----------
def prebuild_threads_fast(edges_df, tweets_df):
    # Map IDs → author & time (vectorized joins, no per-thread .loc inside loop)
    tw = tweets_df[['tweet_id', 'author_id', 'created_at']].copy()
    tw['tweet_id'] = tw['tweet_id'].astype(str)
    tw['author_id'] = tw['author_id'].astype(str)

    e = edges_df[['thread_id','parent_tweet_id','child_tweet_id']].copy()
    e['parent_tweet_id'] = e['parent_tweet_id'].astype(str)
    e['child_tweet_id']  = e['child_tweet_id'].astype(str)

    # attach parent/child authors
    e = e.merge(tw[['tweet_id','author_id']].rename(columns={'tweet_id':'parent_tweet_id','author_id':'parent_author'}),
                on='parent_tweet_id', how='left')
    e = e.merge(tw[['tweet_id','author_id','created_at']].rename(
                columns={'tweet_id':'child_tweet_id','author_id':'child_author','created_at':'child_time'}),
                on='child_tweet_id', how='left')

    # drop rows with missing times
    e = e.dropna(subset=['child_time']).sort_values(['thread_id','child_time'])

    # per-thread first time
    t0_series = e.groupby('thread_id')['child_time'].transform('min')
    dt_min = (e['child_time'] - t0_series).dt.total_seconds() / 60.0
    e['dt_min'] = dt_min.astype(float)

    # pack into arrays per thread
    cache = {}
    for tid, g in e.groupby('thread_id', sort=False):
        arr_parent = g['parent_author'].astype(str).to_numpy()
        arr_child  = g['child_author'].astype(str).to_numpy()
        arr_dt     = g['dt_min'].to_numpy()
        # source author = parent of the first edge in time
        source_author = str(g['parent_author'].iloc[0])
        # users seen in this thread (authors from tweets table)
        users_thr = tweets_df.loc[tweets_df['thread_id'] == tid, 'author_id'].astype(str).unique()
        cache[tid] = (arr_parent, arr_child, arr_dt, source_author, set(users_thr))
    return cache
########################################################

# ---------------- Run experiments ----------------
results = []
agg_counter = Counter()
# Prebuild all threads once (fast)
thread_cache = prebuild_threads_fast(edges, tweets)
threads = list(thread_cache.keys())

print("Running intervention experiments...")
ver_by_thread = (tweets
                 .dropna(subset=["thread_id"])
                 .assign(v=tweets["veracity_normalized"].astype(str).str.lower())
                 .groupby("thread_id")["v"]
                 .first()
                 .to_dict())
for thread_id in tqdm(threads, total=len(threads)):
    par, chi, dts, source_author, users_in_thr = thread_cache[thread_id]
    if par.size == 0:
        continue

    max_dt = float(dts.max())
    n_bins = max(1, int(math.floor(max_dt / BIN_MIN)) + 1)

    # pick lambdas by thread veracity (default to 'unverified')
    vnorm = ver_by_thread.get(thread_id, "unverified")
    if vnorm == "false":
        lam_M, lam_F = lam_false, lam_F_false
    else:
        lam_M, lam_F = lam_unver, lam_F_unver

    for tau in TAU_MINUTES:
        for k in SEED_BUDGETS:
            for strat in STRATEGIES:
                # Build a tiny DataFrame view only for selecting seeds among 'seen' users
                e_tmp = pd.DataFrame({'parent_author': par, 'child_author': chi, 'dt_min': dts})
                seeds = pick_seeds(strat, k, tau, e_tmp, users_in_thr, users)
                if not seeds:
                    results.append({
                        "thread_id": thread_id, "strategy": strat, "tau_min": tau, "k": k,
                        "n_bins": n_bins, "auc_M_mean": np.nan, "auc_M_std": np.nan,
                        "M_final_mean": np.nan, "M_final_std": np.nan,
                        "peak_M_mean": np.nan, "peak_t_mean": np.nan, "runs": 0
                    })
                    continue

                curves = []
                for _ in range(N_SIM):
                    M_curve = simulate_thread_ic_fast(
                        par, chi, dts, source_author,
                        tau, seeds, lam_M, lam_F, n_bins, BIN_MIN
                    )
                    curves.append(M_curve)
                curves = np.stack(curves, axis=0)

                aucs   = [auc_trapezoid(c) for c in curves]
                finals = curves[:, -1]
                peaks  = curves.max(axis=1)
                peak_t = curves.argmax(axis=1)

                results.append({
                    "thread_id": thread_id,
                    "strategy": strat,
                    "tau_min": tau,
                    "k": k,
                    "n_bins": n_bins,
                    "auc_M_mean": float(np.mean(aucs)),
                    "auc_M_std": float(np.std(aucs)),
                    "M_final_mean": float(np.mean(finals)),
                    "M_final_std": float(np.std(finals)),
                    "peak_M_mean": float(np.mean(peaks)),
                    "peak_t_mean": float(np.mean(peak_t)),
                    "runs": int(N_SIM)
                })

# save per-thread results
per_thread = pd.DataFrame(results)
per_thread.to_csv(OUT_DIR / "intervention_results_per_thread.csv", index=False)

# aggregate (mean over threads) per (strategy, tau, k)
agg = (per_thread
       .groupby(["strategy","tau_min","k"], dropna=False)[["auc_M_mean","M_final_mean","peak_M_mean","peak_t_mean"]]
       .mean()
       .reset_index())
agg.to_csv(OUT_DIR / "intervention_results_aggregate.csv", index=False)

print("Saved:")
print("  - experiments/intervention_results_per_thread.csv")
print("  - experiments/intervention_results_aggregate.csv")
print(">>> Step 6 COMPLETE <<<")
