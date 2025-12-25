# run_interventions_threshold.py — Step 9b: Counterfactual seeding under Threshold model
import json, math, random
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------------- I/O ----------------
DATA_DIR = Path("preprocessed")
CALIB_DIR = Path("calibration")
OUT_DIR  = Path("experiments_threshold"); OUT_DIR.mkdir(exist_ok=True)

TWEETS = DATA_DIR/"tweets_clean.csv"
EDGES  = DATA_DIR/"edges_clean.csv"
USERF  = DATA_DIR/"G_user_features_pruned.csv"  # use pruned features for targeting
CALIB  = CALIB_DIR/"calibrated_threshold_parameters.json"

TAU_MINUTES = [0, 15, 30, 60]
SEED_BUDGETS = [1, 3, 5, 10]
STRATEGIES = ["earliest","hubs","bridges","community","random"]

BIN_MIN = 5
N_SIM = 30
RNG_SEED = 7
random.seed(RNG_SEED); np.random.seed(RNG_SEED)

print(">>> run_interventions_threshold.py STARTED <<<")

# ---------------- Load
tweets = pd.read_csv(TWEETS, parse_dates=["created_at"])
edges  = pd.read_csv(EDGES)
users  = pd.read_csv(USERF)
with open(CALIB,"r") as f: calib = json.load(f)

theta_false = float(calib["thetas"].get("false",{}).get("mean", 0.25))
theta_unver = float(calib["thetas"].get("unverified",{}).get("mean", 0.28))
THETA_F_FACTOR = 0.95  # usually ≥ θ_M (harder to flip), adjust if desired
thetaF_false = min(1.0, theta_false * THETA_F_FACTOR)
thetaF_unver = min(1.0, theta_unver * THETA_F_FACTOR)

# Map tweet->author and thread
tw2author = tweets.set_index("tweet_id")["author_id"].astype(str).to_dict()
edges["parent_author"] = edges["parent_tweet_id"].map(tw2author).astype(str)
edges["child_author"]  = edges["child_tweet_id"].map(tw2author).astype(str)

# Per-thread veracity
ver_by_thread = (tweets.dropna(subset=["thread_id"])
                 .assign(v=tweets["veracity_normalized"].astype(str).str.lower())
                 .groupby("thread_id")["v"].first().to_dict())

# ---------------- Seed selector (reuse your Step-6 logic)
def pick_seeds(strategy, k, tau_min, e_thread, users_in_thread, df_user):
    seen = set(e_thread.loc[e_thread["dt_min"] <= tau_min, "child_author"].astype(str).tolist())
    seen |= {str(e_thread.iloc[0]["parent_author"])}
    eligible = [u for u in users_in_thread if u in seen]
    if not eligible: return set()

    if strategy == "earliest":
        order = e_thread.drop_duplicates("child_author")
        order = order[order["dt_min"] <= tau_min]["child_author"].astype(str).tolist()
        return set(order[:k])

    if strategy == "hubs":
        cand = df_user.loc[df_user["user_id"].isin(eligible)].sort_values(["pagerank","in_degree_w"], ascending=False)["user_id"].tolist()
        return set(cand[:k])

    if strategy == "bridges":
        cand = df_user.loc[df_user["user_id"].isin(eligible)].sort_values(["participation_coeff","betweenness"], ascending=False)["user_id"].tolist()
        return set(cand[:k])

    if strategy == "community":
        sub = df_user.loc[df_user["user_id"].isin(eligible), ["user_id","community_id"]]
        counts = sub["community_id"].value_counts()
        if counts.empty:
            return pick_seeds("hubs", k, tau_min, e_thread, users_in_thread, df_user)
        alloc = {}
        for cid, cnt in counts.items():
            share = cnt / counts.sum()
            alloc[cid] = max(0, int(round(share * k)))
        diff = k - sum(alloc.values())
        for cid in counts.index[:abs(diff)]:
            alloc[cid] = alloc.get(cid,0) + (1 if diff>0 else -1)
        seeds=set()
        for cid,take in alloc.items():
            if take<=0: continue
            bucket=sub[sub["community_id"]==cid]["user_id"].tolist()
            bucket=df_user.set_index("user_id").loc[bucket].sort_values(["pagerank","in_degree_w"], ascending=False).index.tolist()
            seeds |= set(bucket[:take])
        return set(list(seeds)[:k])

    if strategy == "random":
        cand = list(eligible); random.shuffle(cand)
        return set(cand[:k])

    return set()

# ---------------- Prebuild per-thread arrays
def prebuild_threads_fast(edges_df, tweets_df):
    tw = tweets_df[['tweet_id','author_id','created_at']].copy()
    tw['tweet_id'] = tw['tweet_id'].astype(str)
    tw['author_id'] = tw['author_id'].astype(str)
    e = edges_df[['thread_id','parent_tweet_id','child_tweet_id']].copy()
    e['parent_tweet_id'] = e['parent_tweet_id'].astype(str)
    e['child_tweet_id']  = e['child_tweet_id'].astype(str)
    e = e.merge(tw[['tweet_id','author_id']].rename(columns={'tweet_id':'parent_tweet_id','author_id':'parent_author'}), on='parent_tweet_id', how='left')
    e = e.merge(tw[['tweet_id','author_id','created_at']].rename(columns={'tweet_id':'child_tweet_id','author_id':'child_author','created_at':'child_time'}), on='child_tweet_id', how='left')
    e = e.dropna(subset=['child_time']).sort_values(['thread_id','child_time'])
    t0 = e.groupby('thread_id')['child_time'].transform('min')
    e['dt_min'] = ((e['child_time']-t0).dt.total_seconds()/60.0).astype(float)
    cache={}
    # also collect per-thread set of authors (eligibility universe)
    authors_by_thread = tweets_df.dropna(subset=["thread_id"]).copy()
    authors_by_thread["author_id"] = authors_by_thread["author_id"].astype(str)
    thr2users = authors_by_thread.groupby("thread_id")["author_id"].apply(lambda s:set(s.astype(str))).to_dict()
    for tid,g in e.groupby('thread_id', sort=False):
        par = g['parent_author'].astype(str).to_numpy()
        chi = g['child_author'].astype(str).to_numpy()
        dts = g['dt_min'].to_numpy()
        source_author = str(g['parent_author'].iloc[0])
        users_thr = thr2users.get(tid, set(pd.unique(np.concatenate([par,chi])).astype(str)))
        cache[tid]=(par,chi,dts,source_author,users_thr)
    return cache

cache = prebuild_threads_fast(edges, tweets)
threads = list(cache.keys())
users["user_id"] = users["user_id"].astype(str)

# ---------------- Threshold simulator (competitive)
def simulate_thread_threshold(par, chi, dts, source_author, tau_min, seeds_F, theta_M, theta_F, n_bins, bin_min=5):
    users_all = pd.unique(np.concatenate([par,chi])).astype(str)
    idx = {u:i for i,u in enumerate(users_all)}
    U = len(users_all)
    S = np.zeros(U, dtype=np.int8)       # 0=S, 1=M, 2=F
    seen = np.zeros(U, dtype=np.int32)
    m_in = np.zeros(U, dtype=np.int32)
    f_in = np.zeros(U, dtype=np.int32)

    if source_author in idx: S[idx[source_author]] = 1

    seeded=False
    max_bin = n_bins-1
    M_curve = np.zeros(n_bins, dtype=np.int32)

    for i in range(len(dts)):
        t = dts[i]
        if (not seeded) and t >= tau_min:
            for s in seeds_F:
                j = idx.get(s, None)
                if j is not None and S[j]==0: S[j]=2
            seeded=True

        u = idx.get(par[i], None); v = idx.get(chi[i], None)
        if u is None or v is None: continue

        # exposure counts
        seen[v] += 1
        if S[u]==1: m_in[v] += 1
        elif S[u]==2: f_in[v] += 1

        # threshold checks (exclusive states; M vs F)
        if S[v]==0 and seen[v]>0:
            fracM = m_in[v] / seen[v]
            fracF = f_in[v] / seen[v]
            if fracM >= theta_M and fracF < theta_F:
                S[v]=1
            elif fracF >= theta_F and fracM < theta_M:
                S[v]=2
            elif fracM >= theta_M and fracF >= theta_F:
                # tie-break: give advantage to F (or randomize)
                S[v]=2

        b = int(min(max_bin, max(0, int(t//bin_min))))
        M_curve[b] = int((S==1).sum())

    for k in range(1, n_bins):
        if M_curve[k] < M_curve[k-1]: M_curve[k]=M_curve[k-1]
    return M_curve

def auc_trap(y):
    if len(y)<=1: return float(y[-1]) if len(y) else 0.0
    y = np.asarray(y, float); return float(np.trapz(y, dx=1.0))

# ---------------- Run grid
results=[]
print("Running Threshold interventions...")
for tid in tqdm(threads, total=len(threads)):
    par,chi,dts,source,users_thr = cache[tid]
    if par.size==0: continue
    max_dt = float(dts.max())
    n_bins = max(1, int(math.floor(max_dt/BIN_MIN))+1)

    v = ver_by_thread.get(tid, "unverified")
    if v=="false":
        thM, thF = theta_false, thetaF_false
    else:
        thM, thF = theta_unver, thetaF_unver

    e_tmp = pd.DataFrame({"parent_author":par, "child_author":chi, "dt_min":dts})
    for tau in TAU_MINUTES:
        for k in SEED_BUDGETS:
            for strat in STRATEGIES:
                seeds = pick_seeds(strat, k, tau, e_tmp, users_thr, users)
                if not seeds:
                    results.append({"thread_id":tid,"strategy":strat,"tau_min":tau,"k":k,
                                    "n_bins":n_bins,"auc_M_mean":np.nan,"M_final_mean":np.nan,
                                    "peak_M_mean":np.nan,"peak_t_mean":np.nan,"runs":0})
                    continue
                curves=[]
                for _ in range(N_SIM):
                    c = simulate_thread_threshold(par,chi,dts,source,tau,seeds, thM, thF, n_bins, BIN_MIN)
                    curves.append(c)
                C = np.stack(curves, axis=0)
                aucs = np.array([auc_trap(c) for c in C])
                finals = C[:,-1]; peaks = C.max(axis=1); peak_t = C.argmax(axis=1)

                results.append({"thread_id":tid,"strategy":strat,"tau_min":tau,"k":k,"n_bins":n_bins,
                                "auc_M_mean":float(aucs.mean()), "M_final_mean":float(finals.mean()),
                                "peak_M_mean":float(peaks.mean()), "peak_t_mean":float(peak_t.mean()),
                                "runs":int(N_SIM)})

per_thread = pd.DataFrame(results)
per_thread.to_csv(OUT_DIR/"intervention_threshold_per_thread.csv", index=False)

agg = (per_thread.groupby(["strategy","tau_min","k"], dropna=False)
       [["auc_M_mean","M_final_mean","peak_M_mean","peak_t_mean"]].mean().reset_index())
agg.to_csv(OUT_DIR/"intervention_threshold_aggregate.csv", index=False)

print("Saved:")
print(f"  - {OUT_DIR/'intervention_threshold_per_thread.csv'}")
print(f"  - {OUT_DIR/'intervention_threshold_aggregate.csv'}")
print(">>> Step 9b COMPLETE <<<")
