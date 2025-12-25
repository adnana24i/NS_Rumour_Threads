# calibrate_threshold.py — Step 9a: Calibrate θ for a single-contagion Threshold model
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DATA_DIR = Path("preprocessed")
OUT_DIR  = Path("calibration"); OUT_DIR.mkdir(exist_ok=True)

TWEETS = DATA_DIR/"tweets_clean.csv"
EDGES  = DATA_DIR/"edges_clean.csv"
TIME_BIN_MIN = 60
THETA_GRID = np.linspace(0.05, 0.6, 12)  # candidate thresholds
N_SIM = 10
RNG = np.random.default_rng(42)

def normalize_id(x):
    if pd.isna(x) or str(x).strip().lower() in {"nan",""}: return None
    return str(x).strip()

print(">>> calibrate_threshold.py STARTED <<<")
tweets = pd.read_csv(TWEETS, parse_dates=["created_at"])
edges  = pd.read_csv(EDGES)

for c in ["tweet_id","thread_id"]:
    tweets[c] = tweets[c].apply(normalize_id)
for c in ["parent_tweet_id","child_tweet_id","thread_id"]:
    edges[c] = edges[c].apply(normalize_id)
edges = edges.dropna(subset=["parent_tweet_id","child_tweet_id"])

tmap = tweets.set_index("tweet_id")["created_at"].to_dict()
vmap = tweets.set_index("tweet_id")["veracity_normalized"].astype(str).str.lower().to_dict()

def obs_curve_for_thread(e_thr):
    e = e_thr.copy()
    e["t"] = e["child_tweet_id"].map(tmap)
    e = e.dropna(subset=["t"]).sort_values("t")
    if e.empty: return None
    t0 = e["t"].iloc[0]
    bins = ((e["t"] - t0).dt.total_seconds()/60.0/TIME_BIN_MIN).astype(int)
    return bins.value_counts().sort_index().cumsum().values

# Threshold simulation on observed contact order (single contagion: only M)
def simulate_threshold_single(par, chi, dts, theta, n_bins, bin_min):
    users = pd.unique(np.concatenate([par, chi])).astype(str)
    idx = {u:i for i,u in enumerate(users)}
    U = len(users)
    state   = np.zeros(U, dtype=np.int8)  # 0=S, 1=M
    seen_in = np.zeros(U, dtype=np.int32) # exposures seen
    m_in    = np.zeros(U, dtype=np.int32) # M exposures among seen

    # source parent becomes M at first contact
    if par.size>0 and par[0] in idx: state[idx[par[0]]] = 1

    max_bin = n_bins-1
    curve = np.zeros(n_bins, dtype=np.int32)

    for i in range(len(dts)):
        u = idx.get(par[i]); v = idx.get(chi[i])
        if u is None or v is None: continue

        # exposure updates counts for v
        seen_in[v] += 1
        if state[u] == 1: m_in[v] += 1

        # adoption check
        if state[v] == 0 and seen_in[v] > 0:
            if (m_in[v] / seen_in[v]) >= theta:
                state[v] = 1

        b = int(min(max_bin, max(0, int(dts[i]//bin_min))))
        curve[b] = (state==1).sum()

    for k in range(1, n_bins):
        if curve[k] < curve[k-1]: curve[k] = curve[k-1]
    return curve

# Prebuild per-thread arrays (parents, children, dts)
def prebuild(edges_df, tweets_df):
    tw = tweets_df[["tweet_id","author_id","created_at"]].copy()
    tw["tweet_id"] = tw["tweet_id"].astype(str)
    e = edges_df[["thread_id","parent_tweet_id","child_tweet_id"]].copy()
    e["parent_tweet_id"] = e["parent_tweet_id"].astype(str)
    e["child_tweet_id"]  = e["child_tweet_id"].astype(str)
    e = e.merge(tw[["tweet_id","author_id"]].rename(columns={"tweet_id":"parent_tweet_id","author_id":"parent_author"}), on="parent_tweet_id", how="left")
    e = e.merge(tw[["tweet_id","author_id","created_at"]].rename(columns={"tweet_id":"child_tweet_id","author_id":"child_author","created_at":"child_time"}), on="child_tweet_id", how="left")
    e = e.dropna(subset=["child_time"]).sort_values(["thread_id","child_time"])
    t0 = e.groupby("thread_id")["child_time"].transform("min")
    e["dt_min"] = ((e["child_time"]-t0).dt.total_seconds()/60.0).astype(float)
    cache = {}
    for tid,g in e.groupby("thread_id", sort=False):
        par = g["parent_author"].astype(str).to_numpy()
        chi = g["child_author"].astype(str).to_numpy()
        dts = g["dt_min"].to_numpy()
        cache[tid]=(par,chi,dts)
    return cache

cache = prebuild(edges, tweets)

# Build observed curves per veracity
threads_by_ver = {"false":[], "unverified":[]}
for tid, (par,chi,dts) in tqdm(cache.items(), total=len(cache), desc="Prepare threads"):
    # observed curve
    # rebuild minimal df to reuse obs_curve_for_thread
    df = pd.DataFrame({"child_tweet_id":chi})
    df["t"] = df["child_tweet_id"].map(tmap)
    if df["t"].isna().all(): continue
    t0 = df["t"].min()
    bins = ((df["t"] - t0).dt.total_seconds()/60.0/TIME_BIN_MIN).astype(int)
    obs = bins.value_counts().sort_index().cumsum().values
    if obs is None or len(obs)==0: continue
    # veracity from earliest parent tweet in this thread
    # approximate with first edge's parent tweet id → tweets veracity
    # edges had tweet ids; map via inverse:
    # we don't have parent tweet id here; fallback: thread root author veracity via tweets table
    # safer: use the earliest child's parent tweet id from raw edges
    pass

# We need veracity per thread; reuse the tweets table
vt = (tweets.dropna(subset=["thread_id"])
            .assign(v=tweets["veracity_normalized"].astype(str).str.lower())
            .groupby("thread_id")["v"].first().to_dict())

results = {"model":"Threshold (single contagion)", "time_bin_minutes": TIME_BIN_MIN, "n_simulations": N_SIM, "thetas": {}}
theta_by_ver = {"false":[], "unverified":[]}

for tid,(par,chi,dts) in tqdm(cache.items(), total=len(cache), desc="Calibrating"):
    v = vt.get(tid, "unverified")
    # observed curve (fast)
    # build times from dts
    if len(dts)==0: continue
    n_bins = max(1, int(math.floor(dts.max()/TIME_BIN_MIN))+1)
    # approximate observed cumulative as one increment per edge in bin order
    hist = np.zeros(n_bins, dtype=int)
    bins = np.minimum(n_bins-1, (dts//TIME_BIN_MIN).astype(int))
    for b in bins: hist[int(b)] += 1
    obs = np.cumsum(hist)

    # grid search θ
    best_theta, best_loss = None, np.inf
    for th in THETA_GRID:
        sims = []
        for _ in range(N_SIM):
            sims.append(simulate_threshold_single(par,chi,dts, th, n_bins, TIME_BIN_MIN))
        sim = np.mean(np.vstack(sims), axis=0)
        m = min(len(sim), len(obs))
        loss = ((sim[:m]-obs[:m])**2).mean()
        if loss < best_loss: best_loss, best_theta = loss, th
    theta_by_ver.setdefault(v, []).append(best_theta)

for v, arr in theta_by_ver.items():
    if len(arr)==0: continue
    results["thetas"][v] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n_threads": int(len(arr))}

with open(OUT_DIR/"calibrated_threshold_parameters.json","w") as f:
    json.dump(results, f, indent=2)

print("Saved → calibration/calibrated_threshold_parameters.json")
print(">>> Step 9a COMPLETE <<<")
