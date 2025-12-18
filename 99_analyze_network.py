# analyze_network.py — Network structure report for the user→user graph
# Inputs:
#   preprocessed/edges_clean.csv        (must contain: src_user, dst_user, thread_id)  OR (parent_author, child_author)
#   preprocessed/G_user_features.csv    (from Step 3; user_id + centralities/communities)
# Outputs (folder: network_analysis/):
#   metrics_overview.json
#   tables/*.csv (top nodes, components, k-core sizes, community sizes)
#   figs/*.png   (degree/strength CCDFs, clustering, assortativity, correlation heatmap, layout, etc.)

from pathlib import Path
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from networkx.algorithms.richclub import rich_club_coefficient


# ---------------- I/O ----------------
DATA_DIR = Path("preprocessed")
OUT_DIR  = Path("network_analysis")
FIGS     = OUT_DIR / "figs"
TABLES   = OUT_DIR / "tables"
OUT_DIR.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

EDGES = DATA_DIR / "edges_clean.csv"
USERF = DATA_DIR / "G_user_features.csv"

# ---------------- Helpers ----------------
def read_edges(path: Path):
    df = pd.read_csv(path)
    # Support either (src_user, dst_user) or (parent_author, child_author)
    if {"src_user","dst_user"}.issubset(df.columns):
        src_col, dst_col = "src_user", "dst_user"
    elif {"parent_author","child_author"}.issubset(df.columns):
        src_col, dst_col = "parent_author", "child_author"
    else:
        raise ValueError("edges_clean.csv must have either (src_user,dst_user) or (parent_author,child_author)")
    df = df[[src_col, dst_col]].rename(columns={src_col: "src_user", dst_col: "dst_user"})
    df["src_user"] = df["src_user"].astype(str)
    df["dst_user"] = df["dst_user"].astype(str)
    # aggregate weights
    df["weight"] = 1.0
    df = df.groupby(["src_user","dst_user"], as_index=False)["weight"].sum()
    # drop self-loops
    df = df[df["src_user"] != df["dst_user"]]
    return df

def ccdf(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return np.array([1.0]), np.array([1.0])
    xs = np.sort(vals)
    ys = 1.0 - np.arange(1, len(xs)+1)/len(xs)
    return xs, ys

def safe_modularity(Gu, communities):
    try:
        import networkx.algorithms.community as nxcom
        return nxcom.modularity(G=Gu, communities=communities, weight="weight")
    except Exception:
        return np.nan

def draw_ccdf(ax, data, title, xlabel):
    x, y = ccdf(data)
    if len(x) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return
    ax.loglog(x, np.maximum(y, 1e-6), marker='.', linestyle='none')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CCDF")
    
def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if x.size == 0:
        return np.nan
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    # Gini = 1 + 1/n - 2 * sum(cumx)/(n * cumx[-1])
    return 1.0 + 1.0/n - 2.0 * np.sum(cumx) / (n * cumx[-1]) if cumx[-1] > 0 else 0.0

def freeman_degree_centralization(deg):
    """Normalized Freeman centralization (undirected)"""
    vals = np.asarray(list(deg.values()), dtype=float)
    if vals.size < 3:
        return np.nan
    kmax = vals.max()
    num = np.sum(kmax - vals)
    n = vals.size
    den = (n - 1) * (n - 2)
    return num / den if den > 0 else np.nan


# ---------------- Load ----------------
print(">>> analyze_network.py STARTED <<<")
edges = read_edges(EDGES)
# --- PRUNE nodes with in-degree==1 and out-degree==0 (on the directed view) ---
in_counts  = edges.groupby("dst_user").size()
out_counts = edges.groupby("src_user").size()

cand    = set(in_counts[in_counts == 1].index) - set(out_counts.index)  # in=1 & out=0
to_drop = pd.Index(cand)

edges_pruned = edges[~edges["src_user"].isin(to_drop) & ~edges["dst_user"].isin(to_drop)].copy()
print(f"Pruning: removed {len(to_drop)} nodes; edges {len(edges)} → {len(edges_pruned)}")

# use the pruned edge list from here on
edges = edges_pruned
# -------------------------------------------------------------------------------

users = pd.read_csv(USERF).rename(columns={"user_id":"node"})
users["node"] = users["node"].astype(str)

# Directed graph (weights = interaction count)
G = nx.DiGraph()
G.add_weighted_edges_from(edges[["src_user","dst_user","weight"]].itertuples(index=False, name=None))
print(f"Directed graph: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")

# Undirected projection for clustering/community/betweenness-like stats
UG = nx.Graph()
for u, v, w in tqdm(G.edges(data="weight"), total=G.number_of_edges(), desc="Build undirected projection"):
    if UG.has_edge(u, v):
        UG[u][v]["weight"] += w
    else:
        UG.add_edge(u, v, weight=w)



# ---------------- Basic metrics ----------------
metrics = {}
metrics["n_nodes"]  = G.number_of_nodes()
metrics["n_edges"]  = G.number_of_edges()
metrics["density_directed"]  = nx.density(G)
metrics["density_undirected"] = nx.density(UG)
metrics["reciprocity"] = nx.reciprocity(G)


# Degree/strength distributions
in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())
in_w    = dict(G.in_degree(weight="weight"))
out_w   = dict(G.out_degree(weight="weight"))

# ---------------- Hubness analysis ----------------
# Choose which notion of "hub": attention hubs = in-degree (directed),
# structural hubs = degree on UG. We'll compute both.
deg_ug = dict(UG.degree())                         # undirected degree
in_deg_vals  = np.array(list(in_deg.values()))
deg_ug_vals  = np.array(list(deg_ug.values()))

metrics["gini_in_degree"]  = float(gini(in_deg_vals))
metrics["gini_ug_degree"]  = float(gini(deg_ug_vals))
metrics["freeman_centralization_UG"] = float(freeman_degree_centralization(deg_ug))

# Rich-club coefficient φ(k) on UG (unweighted)
phi = rich_club_coefficient(UG, normalized=False, Q=100)  # dict: k -> φ(k)
phi_series = pd.Series(phi).sort_index()
phi_series.to_csv(TABLES / "rich_club_phi.csv", header=["phi"])

# Per-community hub dominance: max degree / sum degree within community
if "comm_detected" in users.columns:
    users["deg_ug"] = users["node"].map(deg_ug).fillna(0).astype(float)
    hdom_rows = []
    for cid, sub in users.groupby("comm_detected"):
        if len(sub) == 0:
            continue
        s = sub["deg_ug"].sum()
        m = sub["deg_ug"].max()
        hdom = float(m / s) if s > 0 else np.nan
        h = sub.sort_values("deg_ug", ascending=False).head(1)["node"].values[0] if len(sub) else None
        hdom_rows.append({"community": int(cid), "n_nodes": int(len(sub)),
                          "sum_deg": float(s), "max_deg": float(m),
                          "hub_node": h, "hub_dominance": hdom})
    pd.DataFrame(hdom_rows).sort_values("hub_dominance", ascending=False)\
      .to_csv(TABLES / "community_hub_dominance.csv", index=False)


# Strongly/weakly connected components
wcc_sizes = [len(c) for c in nx.weakly_connected_components(G)]
scc_sizes = [len(c) for c in nx.strongly_connected_components(G)]
metrics["wcc_count"] = len(wcc_sizes)
metrics["wcc_max"]   = int(max(wcc_sizes) if wcc_sizes else 0)
metrics["scc_count"] = len(scc_sizes)
metrics["scc_max"]   = int(max(scc_sizes) if scc_sizes else 0)

# Clustering/transitivity (undirected, weighted ignored for clustering coefficient here)
metrics["transitivity"] = nx.transitivity(UG)
metrics["avg_clustering"] = nx.average_clustering(UG)

# Assortativity (degree-degree) on UG to avoid direction issues
try:
    metrics["assortativity_degree"] = nx.degree_assortativity_coefficient(UG, x='out', y='in')  # fall back if raises
except Exception:
    metrics["assortativity_degree"] = nx.degree_assortativity_coefficient(UG)

# k-core sizes on UG
kcore_sizes = []
for k in [1,2,3,4,5,10,20,50]:
    core = nx.k_core(UG, k=k)
    if core.number_of_nodes() > 0:
        kcore_sizes.append((k, core.number_of_nodes(), core.number_of_edges()))
pd.DataFrame(kcore_sizes, columns=["k","n_nodes","n_edges"]).to_csv(TABLES / "kcore_sizes.csv", index=False)

# Communities (Louvain if available, else greedy modularity)
print("Community detection...")
try:
    import community as community_louvain
    part = community_louvain.best_partition(UG, weight="weight", random_state=42)
    algo = "louvain"
except Exception:
    from networkx.algorithms.community import greedy_modularity_communities
    comms = list(greedy_modularity_communities(UG, weight="weight"))
    part = {n:i for i, C in enumerate(comms) for n in C}
    algo = "greedy_modularity"

users["comm_detected"] = users["node"].map(part).fillna(-1).astype(int)
comm_counts = users["comm_detected"].value_counts().rename_axis("community").reset_index(name="n_nodes")
comm_counts.to_csv(TABLES / "community_sizes.csv", index=False)

# Modularity (approximate if using greedy)
if algo == "louvain":
    # reconstruct communities list
    comm_map = {}
    for n, c in part.items():
        comm_map.setdefault(c, []).append(n)
    modularity = safe_modularity(UG, list(comm_map.values()))
else:
    # for greedy we already had comms variable; re-make if missing
    try:
        modularity = safe_modularity(UG, comms)
    except NameError:
        modularity = np.nan
metrics["community_algo"] = algo
metrics["modularity"] = float(modularity) if modularity == modularity else None

# ---------------- Correlations among centralities/features ----------------
# Merge graph-derived degrees with user features from Step 3
feat = users.copy()
feat["in_degree"]   = feat["node"].map(in_deg).fillna(0).astype(float)
feat["out_degree"]  = feat["node"].map(out_deg).fillna(0).astype(float)
feat["in_strength"] = feat["node"].map(in_w).fillna(0).astype(float)
feat["out_strength"] = feat["node"].map(out_w).fillna(0).astype(float)

# choose feature columns present
cand_cols = [c for c in ["in_degree","out_degree","in_strength","out_strength",
                         "pagerank","betweenness","participation_coeff"] if c in feat.columns]
corr = feat[cand_cols].corr(method="spearman")
corr.to_csv(TABLES / "centrality_correlations.csv")

# ---------------- Top nodes tables ----------------
top_hubs = (feat.sort_values(["pagerank","in_strength"], ascending=False)
            [["node","in_degree","in_strength","pagerank"]].head(50))
top_bridges = (feat.sort_values(["participation_coeff","betweenness"], ascending=False)
               [["node","out_degree","out_strength","betweenness","participation_coeff"]].head(50))
top_hubs.to_csv(TABLES / "top50_hubs.csv", index=False)
top_bridges.to_csv(TABLES / "top50_bridges.csv", index=False)

# ---------------- Plots ----------------

# Lorenz curve for in-degree (directed) and UG degree (side-by-side figures)
def lorenz_points(vals):
    v = np.sort(np.asarray(vals, dtype=float))
    v = v[v >= 0]
    if v.size == 0: 
        return np.array([0,1]), np.array([0,1])
    cum = np.cumsum(v)
    cum = cum / cum[-1]
    x = np.linspace(0, 1, len(cum), endpoint=True)
    return np.concatenate([[0], x]), np.concatenate([[0], cum])

# (a) Lorenz for in-degree
xL, yL = lorenz_points(in_deg_vals)
plt.figure(figsize=(5.2,4), dpi=160)
plt.plot(xL, yL, label="in-degree")
plt.plot([0,1], [0,1], "--", linewidth=1)
plt.title(f"Lorenz curve (in-degree) — Gini={metrics['gini_in_degree']:.2f}")
plt.xlabel("Cumulative share of nodes")
plt.ylabel("Cumulative share of degree")
plt.tight_layout(); plt.savefig(FIGS / "lorenz_in_degree.png"); plt.close()

# (b) Lorenz for UG degree
xL, yL = lorenz_points(deg_ug_vals)
plt.figure(figsize=(5.2,4), dpi=160)
plt.plot(xL, yL, label="UG degree")
plt.plot([0,1], [0,1], "--", linewidth=1)
plt.title(f"Lorenz curve (UG degree) — Gini={metrics['gini_ug_degree']:.2f}")
plt.xlabel("Cumulative share of nodes")
plt.ylabel("Cumulative share of degree")
plt.tight_layout(); plt.savefig(FIGS / "lorenz_ug_degree.png"); plt.close()

# Rich-club φ(k)
plt.figure(figsize=(6,4), dpi=150)
plt.plot(phi_series.index, phi_series.values, marker='.')
plt.xscale("log")
plt.xlabel("k (degree threshold)")
plt.ylabel("rich-club φ(k)")
plt.title("Rich-club coefficient (UG)")
plt.tight_layout(); plt.savefig(FIGS / "rich_club_phi.png"); plt.close()

# Top-20 hubs by in-degree and by PageRank (if available)
plt.figure(figsize=(7.5,4.5), dpi=150)
top20_in = pd.Series(in_deg).sort_values(ascending=False).head(20)
plt.bar(range(len(top20_in)), top20_in.values)
plt.xticks(range(len(top20_in)), [str(k)[:8] for k in top20_in.index], rotation=45, ha="right")
plt.ylabel("in-degree"); plt.title("Top-20 attention hubs (directed)")
plt.tight_layout(); plt.savefig(FIGS / "top20_in_degree.png"); plt.close()

if "pagerank" in users.columns:
    tmp = users.dropna(subset=["pagerank"]).copy()
    top20_pr = tmp.sort_values("pagerank", ascending=False).head(20)
    plt.figure(figsize=(7.5,4.5), dpi=150)
    plt.bar(range(len(top20_pr)), top20_pr["pagerank"].values)
    plt.xticks(range(len(top20_pr)), [str(x)[:8] for x in top20_pr["node"]], rotation=45, ha="right")
    plt.ylabel("PageRank"); plt.title("Top-20 hubs by PageRank")
    plt.tight_layout(); plt.savefig(FIGS / "top20_pagerank.png"); plt.close()

print("Rendering plots...")
plt.figure(figsize=(6,4), dpi=150)
draw_ccdf(plt.gca(), list(in_deg.values()), "In-degree CCDF (directed)", "in-degree")
plt.tight_layout(); plt.savefig(FIGS / "ccdf_in_degree.png"); plt.close()

plt.figure(figsize=(6,4), dpi=150)
draw_ccdf(plt.gca(), list(out_deg.values()), "Out-degree CCDF (directed)", "out-degree")
plt.tight_layout(); plt.savefig(FIGS / "ccdf_out_degree.png"); plt.close()

plt.figure(figsize=(6,4), dpi=150)
draw_ccdf(plt.gca(), list(in_w.values()), "In-strength CCDF (directed, weighted)", "in-strength")
plt.tight_layout(); plt.savefig(FIGS / "ccdf_in_strength.png"); plt.close()

plt.figure(figsize=(6,4), dpi=150)
draw_ccdf(plt.gca(), list(out_w.values()), "Out-strength CCDF (directed, weighted)", "out-strength")
plt.tight_layout(); plt.savefig(FIGS / "ccdf_out_strength.png"); plt.close()

# Component size bars (weakly connected components on G)
wcc_sizes_sorted = sorted(wcc_sizes, reverse=True)[:25]
plt.figure(figsize=(7,4), dpi=150)
plt.bar(range(len(wcc_sizes_sorted)), wcc_sizes_sorted)
plt.title("Largest weakly connected components (size)")
plt.xlabel("component rank"); plt.ylabel("|V|")
plt.tight_layout(); plt.savefig(FIGS / "wcc_sizes_top25.png"); plt.close()

# Centrality correlation heatmap
plt.figure(figsize=(5,4), dpi=160)
im = plt.imshow(corr.values, origin="lower", aspect="auto")
plt.xticks(range(len(cand_cols)), cand_cols, rotation=45, ha="right")
plt.yticks(range(len(cand_cols)), cand_cols)
plt.colorbar(im, fraction=0.046, pad=0.04, label="Spearman ρ")
plt.title("Centrality/feature correlations")
plt.tight_layout(); plt.savefig(FIGS / "centrality_correlations.png"); plt.close()

# Scatter: in-strength vs PageRank (log-log)
if "pagerank" in feat.columns:
    plt.figure(figsize=(6,4), dpi=150)
    xs = feat["in_strength"].replace(0, np.nan)
    ys = feat["pagerank"].replace(0, np.nan)
    plt.loglog(xs, ys, ".", alpha=0.25)
    plt.xlabel("in-strength (weighted in-degree)")
    plt.ylabel("PageRank")
    plt.title("Influence proxy vs. exposure (log–log)")
    plt.tight_layout(); plt.savefig(FIGS / "scatter_instrength_pagerank.png"); plt.close()

# Community-colored layout on a sampled subgraph (k-core or top connected component)
print("Drawing layout on sampled subgraph...")
sample_nodes = None
try:
    U2 = nx.k_core(UG, k=5)
    if U2.number_of_nodes() < 1000:
        sample_nodes = list(U2.nodes())
    else:
        # take 1000 nodes by random sample from 5-core to keep layout readable
        rng = np.random.default_rng(42)
        sample_nodes = rng.choice(list(U2.nodes()), size=1000, replace=False)
    H = UG.subgraph(sample_nodes).copy()
except Exception:
    # fallback: largest connected component up to 1000 nodes
    lcc = max(nx.connected_components(UG), key=len)
    lcc = list(lcc)
    sample_nodes = lcc[:min(1000, len(lcc))]
    H = UG.subgraph(sample_nodes).copy()

pos = nx.spring_layout(H, k=1/np.sqrt(max(1, H.number_of_nodes())), weight="weight", seed=42, iterations=100)
# map communities (if unknown -> -1)
comm_map = users.set_index("node")["comm_detected"].to_dict()
node_colors = [comm_map.get(n, -1) for n in H.nodes()]
# Normalize community IDs for colormap
unique_vals = {c:i for i,c in enumerate(sorted(set(node_colors)))}
colors = [unique_vals[c] for c in node_colors]

plt.figure(figsize=(7,7), dpi=160)
nx.draw_networkx_nodes(H, pos, node_size=10, node_color=colors, cmap="tab20", linewidths=0)
nx.draw_networkx_edges(H, pos, alpha=0.05, width=0.5)
plt.axis("off")
plt.title("Community-colored sample of UG (k-core/LCC)")
plt.tight_layout(); plt.savefig(FIGS / "layout_community_sample.png"); plt.close()

# ---------------- Save overview ----------------
metrics["in_degree_mean"]   = float(np.mean(list(in_deg.values())))
metrics["out_degree_mean"]  = float(np.mean(list(out_deg.values())))
metrics["in_strength_mean"] = float(np.mean(list(in_w.values())))
metrics["out_strength_mean"]= float(np.mean(list(out_w.values())))
with open(OUT_DIR / "metrics_overview.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save component tables
pd.DataFrame({"wcc_size": wcc_sizes}).sort_values("wcc_size", ascending=False)\
  .to_csv(TABLES / "wcc_sizes.csv", index=False)
pd.DataFrame({"scc_size": scc_sizes}).sort_values("scc_size", ascending=False)\
  .to_csv(TABLES / "scc_sizes.csv", index=False)

print("Saved:")
print(f"  - {OUT_DIR/'metrics_overview.json'}")
print(f"  - {TABLES/'kcore_sizes.csv'}, {TABLES/'community_sizes.csv'}, centrality tables + components")
print(f"  - Figures in {FIGS}")
print(">>> Network analysis COMPLETE <<<")
