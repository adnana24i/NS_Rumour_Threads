import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import timedelta
from tqdm.auto import tqdm


# ---------- I/O ----------
DATA_DIR = Path("preprocessed")  # <- uses outputs from your preprocess.py
EDGES = DATA_DIR / "edges_clean.csv"
TWEETS = DATA_DIR / "tweets_clean.csv"
THREADS = DATA_DIR / "threads_clean.csv"

OUT_USER_FEATS = DATA_DIR / "G_user_features.csv"
OUT_THREAD_DESC = DATA_DIR / "thread_descriptives.csv"
OUT_GROWTH = DATA_DIR / "growth_curves.csv"

print("Loading cleaned data from:", DATA_DIR.resolve())
edges = pd.read_csv(EDGES)
tweets = pd.read_csv(TWEETS, parse_dates=["created_at"])
threads = pd.read_csv(THREADS)

# ---------- Step 3: Build auxiliary user→user graph & features ----------
# (a) aggregate directed interactions (src_user -> dst_user)
print("\n[Step 3] Building user→user graph...")
edges['weight'] = 1.0  # each parent->child interaction counts as 1; customize if desired
user_edges = (
    edges.groupby(['src_user', 'dst_user'], as_index=False)['weight']
    .sum()
    .query('src_user.notna() and dst_user.notna()')
)
user_edges['src_user'] = user_edges['src_user'].astype(str)
user_edges['dst_user'] = user_edges['dst_user'].astype(str)

# (b) directed weighted graph
G = nx.DiGraph()
G.add_weighted_edges_from(user_edges[['src_user','dst_user','weight']].itertuples(index=False, name=None))
print(f"  Users (nodes): {G.number_of_nodes():,} | Interactions (edges): {G.number_of_edges():,}")

# (c) core centralities (weighted)
# weighted degrees
in_deg_w  = dict(G.in_degree(weight='weight'))
out_deg_w = dict(G.out_degree(weight='weight'))

# PageRank (weighted)
pagerank = nx.pagerank(G, weight='weight')

# Betweenness: use edge length = 1/weight on an undirected view of G
# (betweenness on directed graphs can be very sparse; bridging is often assessed undirected)
UG_for_btw = nx.Graph()
for u, v, w in tqdm(G.edges(data='weight'),
                    desc="Build undirected for betweenness",
                    total=G.number_of_edges()):
    if UG_for_btw.has_edge(u, v):
        UG_for_btw[u][v]['weight'] += w
    else:
        UG_for_btw.add_edge(u, v, weight=w)


# define "length" as inverse weight so stronger ties are shorter
for u, v, data in tqdm(UG_for_btw.edges(data=True),
                       desc="Set edge lengths",
                       total=UG_for_btw.number_of_edges()):
    w = data.get('weight', 1.0)
    data['length'] = 1.0 / max(w, 1e-9)

#########################################################
# betweenness = nx.betweenness_centrality(UG_for_btw, weight='length', normalized=True)

# --- clean up before core decomposition ---
# remove self-loops on the undirected graph (required by k_core/core_number)
UG_for_btw.remove_edges_from(nx.selfloop_edges(UG_for_btw))


# --- AFTER (fast approximate on LCC) ---
# --- FAST BETWEenness: compute on the 2-core, then LCC, with a smaller sample ---
# --- fast betweenness on a pruned subgraph ---
UG_core = nx.k_core(UG_for_btw, k=2)  # strip leaves; faster & betweenness-carrying core
print(f"  2-core reduction: {UG_for_btw.number_of_nodes():,} -> {UG_core.number_of_nodes():,} nodes")

if UG_core.number_of_nodes() == 0 or UG_core.number_of_edges() == 0:
    betweenness = {n: 0.0 for n in UG_for_btw.nodes()}
else:
    UG_cc_nodes = max(nx.connected_components(UG_core), key=len)
    UG_cc = UG_core.subgraph(UG_cc_nodes).copy()
    n_cc = UG_cc.number_of_nodes()

    k_sample = min(96, max(32, int(np.sqrt(n_cc) / 3)))  # shrink if still slow
    print(f"  Betweenness on 2-core LCC: nodes={n_cc:,}, sample k={k_sample}")

    betweenness_cc = nx.betweenness_centrality(
        UG_cc, k=k_sample, weight='length', normalized=True, seed=42
    )
    betweenness = {n: 0.0 for n in UG_for_btw.nodes()}
    betweenness.update(betweenness_cc)


# after building G (the directed graph), also strip any self-loops there
G.remove_edges_from(nx.selfloop_edges(G))



#################################################################################
# (d) community detection + participation coefficient
# Try Louvain (python-louvain); fall back to greedy modularity communities
try:
    import community as community_louvain  # python-louvain
    part = community_louvain.best_partition(UG_for_btw, weight='weight', resolution=1.0, random_state=42)
    community_id = part
    algo_used = "louvain"
except Exception:
    comms = list(nx.algorithms.community.greedy_modularity_communities(UG_for_btw, weight='weight'))
    # map node -> community id
    community_id = {}
    for cid, S in enumerate(comms):
        for n in S:
            community_id[n] = cid
    algo_used = "greedy_modularity"
print(f"  Community detection: {algo_used}")

# participation coefficient (Guimera-Amaral)
# First compute weighted degree per node into each community on the undirected graph
from collections import defaultdict

deg_by_comm = defaultdict(lambda: defaultdict(float))  # node -> comm -> weight_sum
total_deg = defaultdict(float)

for u, v, data in tqdm(UG_for_btw.edges(data=True),
                       desc="Sum weights by community",
                       total=UG_for_btw.number_of_edges()):
    w = data.get('weight', 1.0)
    cu, cv = community_id.get(u, -1), community_id.get(v, -1)
    deg_by_comm[u][cv] += w
    deg_by_comm[v][cu] += w
    total_deg[u] += w
    total_deg[v] += w


def participation_coeff(node):
    k = total_deg.get(node, 0.0)
    if k <= 0:
        return 0.0
    s = 0.0
    for c, w in deg_by_comm[node].items():
        s += (w / k) ** 2
    return 1.0 - s

participation = {n: participation_coeff(n)
                 for n in tqdm(UG_for_btw.nodes(),
                               desc="Participation coeff",
                               total=UG_for_btw.number_of_nodes())}

# (e) assemble and save user features
users = sorted(G.nodes())
user_df = pd.DataFrame({
    'user_id': users,
    'in_degree_w': [in_deg_w.get(u, 0.0) for u in users],
    'out_degree_w': [out_deg_w.get(u, 0.0) for u in users],
    'pagerank': [pagerank.get(u, 0.0) for u in users],
    'betweenness': [betweenness.get(u, 0.0) for u in users],
    'community_id': [community_id.get(u, -1) for u in users],
    'participation_coeff': [participation.get(u, 0.0) for u in users],
})
user_df.to_csv(OUT_USER_FEATS, index=False)
print(f"  Saved user features → {OUT_USER_FEATS}")

# ---------- Step 4: Descriptive baselines per thread ----------
print("\n[Step 4] Descriptive baselines per thread (structural virality + growth curves)")

# helper: mean pairwise distance (structural virality) for a (connected) cascade tree
def structural_virality_undirected(G_tree_undirected):
    n = G_tree_undirected.number_of_nodes()
    if n < 2:
        return 0.0
    # all-pairs shortest paths (on trees this is O(n))
    lengths = dict(nx.all_pairs_shortest_path_length(G_tree_undirected))
    total = 0
    for i, dists in lengths.items():
        for j, dij in dists.items():
            if j > i:
                total += dij
    return (2.0 * total) / (n * (n - 1))

# Prepare outputs
desc_rows = []
growth_rows = []

# Pre-index helpful maps
tweet2author = tweets.set_index('tweet_id')['author_id'].astype(str).to_dict()
tweet2thread = tweets.set_index('tweet_id')['thread_id'].to_dict()
thread_groups = tweets.groupby('thread_id')

for thread_id, df in tqdm(thread_groups,
                          desc="Per-thread baselines",
                          total=thread_groups.ngroups):
    df = df.sort_values('created_at')
    # growth curve (cumulative tweets vs. time at each tweet timestamp)
    # (optional: resample to minute grid—here we record at each observed tweet time)
    cum = np.arange(1, len(df) + 1)
    for t, c in zip(df['created_at'], cum):
        growth_rows.append({'thread_id': thread_id, 'timestamp': t.isoformat(), 'cum_count': int(c)})

    # build cascade tree from parent->child edges within this thread
    e_thr = edges[edges['thread_id'] == thread_id][['parent_tweet_id','child_tweet_id']].astype(str)
    # Nodes in thread
    nodes_in_thread = set(df['tweet_id'].astype(str))
    # Filter edges to those present
    e_thr = e_thr[e_thr['parent_tweet_id'].isin(nodes_in_thread) & e_thr['child_tweet_id'].isin(nodes_in_thread)]

    # Directed tree
    T_dir = nx.DiGraph()
    T_dir.add_nodes_from(nodes_in_thread)
    T_dir.add_edges_from(e_thr.itertuples(index=False, name=None))

    # Undirected for distances (some threads may not be perfect trees after filtering; take largest component)
    T_undir = T_dir.to_undirected()
    if T_undir.number_of_nodes() == 0:
        sv = 0.0
    else:
        # largest connected component
        largest_cc_nodes = max(nx.connected_components(T_undir), key=len)
        Tcc = T_undir.subgraph(largest_cc_nodes).copy()
        sv = structural_virality_undirected(Tcc)

    # early growth rate: tweets per minute in first 15 minutes from first tweet
    t0 = df['created_at'].min()
    t15 = t0 + timedelta(minutes=15)
    early_count = (df['created_at'] <= t15).sum()
    early_rate_per_min = early_count / 15.0

    # pull thread-level basics from threads.csv when available
    meta = threads[threads['thread_id'] == thread_id]
    n_nodes = int(meta['n_nodes'].iloc[0]) if not meta.empty and 'n_nodes' in meta else int(len(df))
    depth = int(meta['depth'].iloc[0]) if not meta.empty and 'depth' in meta else int(nx.dag_longest_path_length(T_dir)) if nx.is_directed_acyclic_graph(T_dir) else np.nan
    duration_min = float(meta['duration_min'].iloc[0]) if not meta.empty and 'duration_min' in meta else (df['created_at'].max() - df['created_at'].min()).total_seconds() / 60.0

    desc_rows.append({
        'thread_id': thread_id,
        'n_nodes': n_nodes,
        'depth': depth,
        'duration_min': duration_min,
        'structural_virality': sv,
        'early_15min_count': int(early_count),
        'early_15min_rate_per_min': early_rate_per_min
    })

# write outputs
desc_df = pd.DataFrame(desc_rows)
growth_df = pd.DataFrame(growth_rows).sort_values(['thread_id','timestamp'])

desc_df.to_csv(OUT_THREAD_DESC, index=False)
growth_df.to_csv(OUT_GROWTH, index=False)

print(f"  Saved thread descriptives → {OUT_THREAD_DESC}")
print(f"  Saved growth curves      → {OUT_GROWTH}")

print("\nDone.")
