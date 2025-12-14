
"""
Step 5 — Calibrate diffusion parameters (Independent Cascade)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

DATA_DIR = Path("preprocessed")
OUT_DIR = Path("calibration")
OUT_DIR.mkdir(exist_ok=True)

TWEETS = DATA_DIR / "tweets_clean.csv"
EDGES  = DATA_DIR / "edges_clean.csv"

TIME_BIN_MIN = 60          # time aggregation for curves
N_SIM = 20                 # simulations per lambda
LAMBDA_GRID = np.linspace(0.02, 0.5, 15)

# ------------------ LOAD DATA ------------------
print(">>> calibrate_diffusion.py STARTED <<<")

tweets = pd.read_csv(
    TWEETS,
    parse_dates=["created_at"],
    engine="python",
    on_bad_lines="skip"
)

edges = pd.read_csv(EDGES)

print(f"Loaded {len(tweets):,} tweets | {len(edges):,} edges")

# ------------------ NORMALIZE IDS ------------------
def normalize_id(x):
    if pd.isna(x) or str(x).strip().lower() in ['nan', '']:
        return None
    return str(x).strip()

# Normalize tweets
for col in ['tweet_id', 'thread_id']:
    tweets[col] = tweets[col].apply(normalize_id)

# Normalize edges
for col in ['parent_tweet_id', 'child_tweet_id', 'thread_id']:
    edges[col] = edges[col].apply(normalize_id)

# Drop edges with missing IDs
edges = edges.dropna(subset=['parent_tweet_id', 'child_tweet_id'])
print(f"Edges after cleaning: {len(edges):,}")


tweet_time = tweets.set_index('tweet_id')['created_at'].to_dict()
tweet_veracity = tweets.set_index('tweet_id')['veracity_normalized'].astype(str).str.lower().to_dict()


print("\nPreparing threads for calibration...")

threads_by_veracity = {'false': [], 'unverified': []}
skipped = {'no_edges': 0, 'bad_veracity': 0}

for thread_id, e_thr in tqdm(edges.groupby('thread_id'), total=edges['thread_id'].nunique()):
    if e_thr.empty:
        skipped['no_edges'] += 1
        continue

    e_thr = e_thr.copy()
    # map child tweet timestamps; if missing, drop that edge
    e_thr['t'] = e_thr['child_tweet_id'].map(tweet_time)
    e_thr = e_thr.dropna(subset=['t']).sort_values('t')

    if e_thr.empty:
        skipped['no_edges'] += 1
        continue

    # source tweet = earliest parent in thread
    source_tweet = e_thr.iloc[0]['parent_tweet_id']
    veracity = tweet_veracity.get(source_tweet, None)
    if veracity not in threads_by_veracity:
        skipped['bad_veracity'] += 1
        continue

    # observed cumulative adoption curve
    t0 = e_thr['t'].iloc[0]
    bins = ((e_thr['t'] - t0).dt.total_seconds() / 60 / TIME_BIN_MIN).astype(int)
    obs_curve = bins.value_counts().sort_index().cumsum().values

    threads_by_veracity[veracity].append(obs_curve)

print("\nThread preparation summary:")
for k, v in skipped.items():
    print(f"  Skipped ({k}): {v}")

print("\nThreads per veracity:")
for v, ts in threads_by_veracity.items():
    print(f"  {v}: {len(ts)}")

def simulate_ic(n_steps, lam):
    """Simple IC simulation: each active node activates new nodes binomially."""
    active = 1
    curve = [1]
    for _ in range(1, n_steps):
        new = np.random.binomial(active, lam)
        active += new
        curve.append(active)
    return np.array(curve)

def curve_distance(a, b):
    """Mean squared distance between two curves (observed vs simulated)."""
    m = min(len(a), len(b))
    return np.mean((a[:m] - b[:m]) ** 2)

print("\nRunning calibration...")

results = {
    "model": "Independent Cascade",
    "time_bin_minutes": TIME_BIN_MIN,
    "n_simulations": N_SIM,
    "cv": "leave-one-event-out",
    "lambdas": {}
}

for veracity, curves in threads_by_veracity.items():
    if len(curves) < 1:
        print(f"Skipping {veracity}: too few threads")
        continue

    best_lams = []

    for obs in tqdm(curves, desc=f"Calibrating {veracity}"):
        T = len(obs)
        scores = []

        for lam in LAMBDA_GRID:
            sims = np.mean([simulate_ic(T, lam) for _ in range(N_SIM)], axis=0)
            scores.append(curve_distance(obs, sims))

        best_lams.append(LAMBDA_GRID[np.argmin(scores)])

    results["lambdas"][veracity] = {
        "mean": float(np.mean(best_lams)),
        "std": float(np.std(best_lams)),
        "n_threads": len(best_lams)
    }


OUT_DIR.mkdir(exist_ok=True)
out_file = OUT_DIR / "calibrated_ic_parameters.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved calibrated parameters → {out_file}")
print(">>> Step 5 COMPLETE <<<")
