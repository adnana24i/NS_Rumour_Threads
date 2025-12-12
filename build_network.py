import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


INPUT_DIR = Path("ns_rumour_threads/preprocessed")
OUTPUT_DIR = Path("network_data")
OUTPUT_DIR.mkdir(exist_ok=True)

TWEETS_FILE = INPUT_DIR / "tweets_clean.csv"
EDGES_FILE = INPUT_DIR / "edges_clean.csv"
THREADS_FILE = INPUT_DIR / "threads_clean.csv"


print(f"\nInput directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print("-" * 80)

print(f"Load data ...")
try:
    tweets_df = pd.read_csv(TWEETS_FILE)
    edges_df = pd.read_csv(EDGES_FILE)
    threads_df = pd.read_csv(THREADS_FILE)
    
    print(f"Loaded {len(tweets_df):,} tweets")
    print(f"Loaded {len(edges_df):,} edges")
    print(f"Loaded {len(threads_df):,} threads")
    
    # Parse timestamps
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], utc=True)
    edges_df['t_edge'] = pd.to_datetime(edges_df['t_edge'], utc=True)
    
    print(f"Parsed timestamps")
    
except FileNotFoundError as e:
    print(f"Error: Could not find input files")
    print(f"  {e}")
    raise
except Exception as e:
    print(f"Error: {e}")
    raise

print("\nBuild tweet index")
print("-" * 80)

# Fast lookup: tweet_id -> {author_id, timestamp, thread_id, text, ...}
tweet_index = {}

for _, row in tweets_df.iterrows():
    tweet_id = str(row['tweet_id'])
    tweet_index[tweet_id] = {
        'author_id': str(row['author_id']),
        'timestamp': row['created_at'],
        'thread_id': str(row['thread_id']),
        'text': row['text'],
        'event': row['event'],
        'category': row['category'],
        'veracity': row['veracity_normalized'],
        'is_rumour': row['is_rumour_normalized'],
        'parent_tweet_id': str(row['parent_tweet_id']) if pd.notna(row['parent_tweet_id']) and row['parent_tweet_id'] != '' else None,
    }

print(f" Built index for {len(tweet_index):,} tweets")

# Quick validation
missing_authors = sum(1 for t in tweet_index.values() if not t['author_id'] or t['author_id'] == 'nan')
missing_timestamps = sum(1 for t in tweet_index.values() if pd.isna(t['timestamp']))

if missing_authors > 0:
    print(f" Warning: {missing_authors} tweets missing author_id")
if missing_timestamps > 0:
    print(f"Warning: {missing_timestamps} tweets missing timestamp")

print("\n Build temporal contact sequences ")
print("-" * 80)

# Group edges by thread
edges_by_thread = edges_df.groupby('thread_id')

temporal_sequences = {}
thread_stats = []

total_threads = len(edges_by_thread)
processed = 0

for thread_id, thread_edges in edges_by_thread:
    processed += 1
    
    # Sort edges by timestamp to create temporal sequence
    thread_edges_sorted = thread_edges.sort_values('t_edge').copy()
    
    # Find source tweet (the root with no parent, or earliest tweet)
    thread_tweets = tweets_df[tweets_df['thread_id'] == thread_id]
    
    # Source is typically the tweet with no parent or earliest timestamp
    source_candidates = thread_tweets[
        (thread_tweets['parent_tweet_id'].isna()) | 
        (thread_tweets['parent_tweet_id'] == '') |
        (thread_tweets['parent_tweet_id'] == 'nan')
    ]
    
    if len(source_candidates) > 0:
        source_tweet_id = str(source_candidates.iloc[0]['tweet_id'])
    else:
        # Fallback: earliest tweet in thread
        source_tweet_id = str(thread_tweets.sort_values('created_at').iloc[0]['tweet_id'])
    
    # Get thread metadata from first tweet
    source_info = tweet_index.get(source_tweet_id, {})
    event = source_info.get('event', 'unknown')
    category = source_info.get('category', 'unknown')
    veracity = source_info.get('veracity', 'Unverified')
    is_rumour = source_info.get('is_rumour', None)
    
    # Build contact sequence: list of (parent_id, child_id, timestamp, src_user, dst_user)
    contact_sequence = []
    
    for _, edge in thread_edges_sorted.iterrows():
        contact_sequence.append({
            'parent_tweet_id': str(edge['parent_tweet_id']),
            'child_tweet_id': str(edge['child_tweet_id']),
            'timestamp': edge['t_edge'],
            'src_user': str(edge['src_user']),
            'dst_user': str(edge['dst_user']),
        })
    
    # Store temporal sequence with metadata
    temporal_sequences[str(thread_id)] = {
        'thread_id': str(thread_id),
        'source_tweet_id': source_tweet_id,
        'event': event,
        'category': category,
        'veracity': veracity,
        'is_rumour': is_rumour,
        'n_edges': len(contact_sequence),
        'start_time': contact_sequence[0]['timestamp'] if contact_sequence else None,
        'end_time': contact_sequence[-1]['timestamp'] if contact_sequence else None,
        'contact_sequence': contact_sequence,
    }
    
    # Collect stats for summary
    duration_sec = (contact_sequence[-1]['timestamp'] - contact_sequence[0]['timestamp']).total_seconds() if len(contact_sequence) > 1 else 0
    
    thread_stats.append({
        'thread_id': str(thread_id),
        'source_tweet_id': source_tweet_id,
        'event': event,
        'category': category,
        'veracity': veracity,
        'is_rumour': is_rumour,
        'n_edges': len(contact_sequence),
        'n_unique_users': len(set([c['src_user'] for c in contact_sequence] + [c['dst_user'] for c in contact_sequence])),
        'duration_seconds': duration_sec,
        'start_time': contact_sequence[0]['timestamp'] if contact_sequence else None,
        'end_time': contact_sequence[-1]['timestamp'] if contact_sequence else None,
    })
    
    if processed % 100 == 0 or processed == total_threads:
        print(f"  Progress: {processed}/{total_threads} threads processed", end='\r')

print(f"\n Built temporal sequences for {len(temporal_sequences):,} threads")


print("\n Build aux structures ")
print("-" * 80)

# User-to-threads mapping (for cross-thread analysis)
print("\n User-to-threads mapping:")
user_to_threads = defaultdict(set)

for thread_id, seq_data in temporal_sequences.items():
    for contact in seq_data['contact_sequence']:
        user_to_threads[contact['src_user']].add(thread_id)
        user_to_threads[contact['dst_user']].add(thread_id)

print(f"  Indexed {len(user_to_threads):,} unique users")
print(f"  Avg threads per user: {np.mean([len(t) for t in user_to_threads.values()]):.2f}")


print("\n Computing thread network statistics:")

thread_network_stats = []

for thread_id, seq_data in temporal_sequences.items():
    contacts = seq_data['contact_sequence']
    
    # Get all nodes (users) in this thread
    all_users = set()
    edges_list = []
    
    for c in contacts:
        all_users.add(c['src_user'])
        all_users.add(c['dst_user'])
        edges_list.append((c['src_user'], c['dst_user']))
    
    # Basic network stats
    n_nodes = len(all_users)
    n_edges = len(edges_list)
    
    # Degree distribution
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    
    for src, dst in edges_list:
        out_degree[src] += 1
        in_degree[dst] += 1
    
    max_out_degree = max(out_degree.values()) if out_degree else 0
    max_in_degree = max(in_degree.values()) if in_degree else 0
    avg_out_degree = np.mean(list(out_degree.values())) if out_degree else 0
    avg_in_degree = np.mean(list(in_degree.values())) if in_degree else 0
    
    # Identify source node
    source_user = tweet_index[seq_data['source_tweet_id']]['author_id']
    source_out_degree = out_degree.get(source_user, 0)
    
    thread_network_stats.append({
        'thread_id': thread_id,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'max_out_degree': max_out_degree,
        'max_in_degree': max_in_degree,
        'avg_out_degree': avg_out_degree,
        'avg_in_degree': avg_in_degree,
        'source_user': source_user,
        'source_out_degree': source_out_degree,
    })

network_stats_df = pd.DataFrame(thread_network_stats)
print(f" Computed network statistics for {len(network_stats_df):,} threads")

print("\nSave output ")
print("-" * 80)

# Save temporal sequences 
sequences_file = OUTPUT_DIR / "temporal_sequences.pkl"
with open(sequences_file, 'wb') as f:
    pickle.dump(temporal_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f" Saved temporal sequences as pkl file {sequences_file}")


# Save tweet index 
index_file = OUTPUT_DIR / "tweet_index.pkl"
with open(index_file, 'wb') as f:
    pickle.dump(tweet_index, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved tweet index as pkl files {index_file}")


#  Save user-to-threads mapping 
user_threads_file = OUTPUT_DIR / "user_to_threads.pkl"
# Convert sets to lists for JSON compatibility later if needed
user_to_threads_serializable = {k: list(v) for k, v in user_to_threads.items()}
with open(user_threads_file, 'wb') as f:
    pickle.dump(user_to_threads_serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f" Saved user-to-threads mapping as pkl file {user_threads_file}")

#  Save thread metadata 
metadata_df = pd.DataFrame(thread_stats)
metadata_file = OUTPUT_DIR / "thread_metadata.csv"
metadata_df.to_csv(metadata_file, index=False)
print(f" Saved thread metadata as csv {metadata_file}")

#  Save network statistics 
network_stats_file = OUTPUT_DIR / "thread_network_stats.csv"
network_stats_df.to_csv(network_stats_file, index=False)
print(f" Saved network statistics as csv file {network_stats_file}")

print("\n Validation")
print("-" * 80)

# Basic validation
print("\n Data Integrity Check:")
print(f"  Threads with temporal sequences: {len(temporal_sequences):,}")
print(f"  Threads in metadata: {len(metadata_df):,}")
print(f"  Match: {'yes' if len(temporal_sequences) == len(metadata_df) else 'no'}")

# Check all edges reference valid tweets
print("\n Edge Validity:")
invalid_edges = 0
for thread_id, seq_data in temporal_sequences.items():
    for contact in seq_data['contact_sequence']:
        parent_id = contact['parent_tweet_id']
        child_id = contact['child_tweet_id']
        if parent_id not in tweet_index:
            invalid_edges += 1
        if child_id not in tweet_index:
            invalid_edges += 1

if invalid_edges == 0:
    print(f"  All edges reference valid tweets")
else:
    print(f" {invalid_edges} edge endpoints not found in tweet index")

# Summary statistics
print("\n  Statistics:")
print(f"\nThread-level:")
print(f"  Total threads: {len(metadata_df):,}")
print(f"  Avg edges per thread: {metadata_df['n_edges'].mean():.1f}")
print(f"  Median edges per thread: {metadata_df['n_edges'].median():.1f}")
print(f"  Max edges in a thread: {metadata_df['n_edges'].max():,}")
print(f"  Avg unique users per thread: {metadata_df['n_unique_users'].mean():.1f}")
print(f"  Avg duration (minutes): {metadata_df['duration_seconds'].mean() / 60:.1f}")

print(f"\nVeracity distribution:")
veracity_counts = metadata_df['veracity'].value_counts()
for veracity, count in veracity_counts.items():
    pct = 100 * count / len(metadata_df)
    print(f"  {veracity}: {count:,} ({pct:.1f}%)")

print(f"\nNetwork statistics:")
print(f"  Avg nodes per thread: {network_stats_df['n_nodes'].mean():.1f}")
print(f"  Avg out-degree: {network_stats_df['avg_out_degree'].mean():.2f}")
print(f"  Avg in-degree: {network_stats_df['avg_in_degree'].mean():.2f}")
print(f"  Max out-degree seen: {network_stats_df['max_out_degree'].max()}")
print(f"  Avg source out-degree: {network_stats_df['source_out_degree'].mean():.2f}")
