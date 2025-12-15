import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("ns_rumour_threads/parsed")  # Change to your actual path
TWEETS_FILE = DATA_DIR / "tweets.csv" 
EDGES_FILE = DATA_DIR / "edges.csv"
THREADS_FILE = DATA_DIR / "threads.csv"

# Output directory for cleaned data
OUTPUT_DIR = Path("preprocessed")
OUTPUT_DIR.mkdir(exist_ok=True)

# Filter configuration
FILTER_CONFIG = {
    'keep_english_only': True,        
    'min_thread_size': 3,             
    'deduplicate_users': False,        
    'remove_invalid_timestamps': True,
}

print("\nLoading data...")
print("-" * 80)

try:
    tweets_df = pd.read_csv(TWEETS_FILE)
    edges_df = pd.read_csv(EDGES_FILE)
    threads_df = pd.read_csv(THREADS_FILE)
    print(f" Loaded tweets.csv: {len(tweets_df):,} rows")
    print(f"Loaded edges.csv: {len(edges_df):,} rows")
    print(f"Loaded threads.csv: {len(threads_df):,} rows")
except FileNotFoundError as e:
    print(f"Error: Couldn't find data files")
    print(f"  {e}")
    raise

# 2.2 Check tweet IDs in edges exist in tweets
print("\nEdge Validity Check")
tweet_ids = set(tweets_df['tweet_id'].astype(str))

edges_df['parent_tweet_id'] = edges_df['parent_tweet_id'].astype(str)
edges_df['child_tweet_id'] = edges_df['child_tweet_id'].astype(str)

parent_exists = edges_df['parent_tweet_id'].isin(tweet_ids)
child_exists = edges_df['child_tweet_id'].isin(tweet_ids)
both_exist = parent_exists & child_exists

pct_parent = 100 * parent_exists.sum() / len(edges_df) if len(edges_df) > 0 else 0
pct_child = 100 * child_exists.sum() / len(edges_df) if len(edges_df) > 0 else 0
pct_both = 100 * both_exist.sum() / len(edges_df) if len(edges_df) > 0 else 0

print(f"  Parent tweets exist in tweets.csv: {parent_exists.sum():,}/{len(edges_df):,} ({pct_parent:.1f}%)")
print(f"  Child tweets exist in tweets.csv:  {child_exists.sum():,}/{len(edges_df):,} ({pct_child:.1f}%)")
print(f"  Both parent & child exist:         {both_exist.sum():,}/{len(edges_df):,} ({pct_both:.1f}%)")

if pct_both < 90:
    print(" Warning: Less than 90 percent of edges have both endpoints in tweets.csv")

# Check for valid timestamps
print("\n Valid timestamps check")
tweets_df['created_at_parsed'] = pd.to_datetime(tweets_df['created_at'], errors='coerce', utc=True)
valid_timestamps = tweets_df['created_at_parsed'].notna()
pct_valid_ts = 100 * valid_timestamps.sum() / len(tweets_df) if len(tweets_df) > 0 else 0

print(f" Tweets with valid timestamps: {valid_timestamps.sum():,}/{len(tweets_df):,} ({pct_valid_ts:.1f}%)")
if pct_valid_ts < 95:
    print(" Warning: More than 5 percent of tweets lack valid timestamps")

# 2.4 Check for language field
print("\n Language Check:")
has_lang = tweets_df['lang'].notna() & (tweets_df['lang'] != '')
pct_lang = 100 * has_lang.sum() / len(tweets_df) if len(tweets_df) > 0 else 0

print(f"  Tweets with language specified: {has_lang.sum():,}/{len(tweets_df):,} ({pct_lang:.1f}%)")

if has_lang.sum() > 0:
    lang_dist = tweets_df.loc[has_lang, 'lang'].value_counts().head(10)
    print(f"\n  Top 10 languages:")
    for lang, count in lang_dist.items():
        print(f"    {lang}: {count:,} ({100*count/has_lang.sum():.1f}%)")

# 2.5 Check for missing critical fields
print("\n Missing fields:")
critical_fields = {
    'tweets': ['tweet_id', 'thread_id', 'author_id', 'created_at', 'text'],
    'edges': ['parent_tweet_id', 'child_tweet_id', 'src_user', 'dst_user', 't_edge'],
    'threads': ['thread_id', 'event', 'category']
}

for table_name, fields in critical_fields.items():
    df = {'tweets': tweets_df, 'edges': edges_df, 'threads': threads_df}[table_name]
    print(f"\n  {table_name}.csv:")
    for field in fields:
        if field in df.columns:
            missing = df[field].isna().sum()
            pct_missing = 100 * missing / len(df) if len(df) > 0 else 0
            print(f" {field}: {missing:,} missing ({pct_missing:.1f}%)")
        else:
            print(f"  {field}: column missing!!!")


print("\nLabel normalization")
print("-" * 80)

def normalize_veracity(val):
    if pd.isna(val) or val == '' or val == 'nan':
        return 'Unverified'
    
    val_str = str(val).strip().lower()
    
    if val_str in ['true', '1', '1.0', 'verified']:
        return 'True'
    elif val_str in ['false', '0', '0.0']:
        return 'False'
    elif val_str in ['unverified', 'unknown', 'none']:
        return 'Unverified'
    else:
        return 'Unverified'  


print("\n Normalizing tweets.csv:")
if 'veracity' in tweets_df.columns:
    original_dist = tweets_df['veracity'].value_counts()
    print(f"  Original veracity distribution:")
    for val, count in original_dist.items():
        print(f"    {val}: {count:,}")
    
    tweets_df['veracity_normalized'] = tweets_df['veracity'].apply(normalize_veracity)
    
    normalized_dist = tweets_df['veracity_normalized'].value_counts()
    print(f"\n  Normalized veracity distribution:")
    for val, count in normalized_dist.items():
        print(f"    {val}: {count:,}")


# Normalize in threads
print("\n Normalizing threads.csv:")
if 'veracity' in threads_df.columns:
    threads_df['veracity_normalized'] = threads_df['veracity'].apply(normalize_veracity)
    
    normalized_dist = threads_df['veracity_normalized'].value_counts()
    print(f"  Thread veracity distribution:")
    for val, count in normalized_dist.items():
        print(f"    {val}: {count:,}")


# Normalize is_rumour field
print("\n Normalizing is_rumour field:")
for df, name in [(tweets_df, 'tweets'), (threads_df, 'threads')]:
    if 'is_rumour' in df.columns:
        df['is_rumour_normalized'] = df['is_rumour'].apply(
            lambda x: True if str(x).lower() in ['true', '1', '1.0'] else False if str(x).lower() in ['false', '0', '0.0'] else None
        )
        dist = df['is_rumour_normalized'].value_counts(dropna=False)
        print(f"  {name}.csv is_rumour distribution:")
        for val, count in dist.items():
            print(f"    {val}: {count:,}")


print("\nDeleting non relevant data")
print("-" * 80)

tweets_filtered = tweets_df.copy()
edges_filtered = edges_df.copy()
threads_filtered = threads_df.copy()

initial_tweet_count = len(tweets_filtered)
initial_edge_count = len(edges_filtered)
initial_thread_count = len(threads_filtered)

# 4.1 Keep English only
if FILTER_CONFIG['keep_english_only']:
    print("\n Filtering for English tweets only:")
    before = len(tweets_filtered)
    tweets_filtered = tweets_filtered[tweets_filtered['lang'] == 'en']
    after = len(tweets_filtered)
    print(f"  Tweets: {before:,} → {after:,} (removed {before-after:,})")
    
    # Update edges to only include tweets that remain
    remaining_tweet_ids = set(tweets_filtered['tweet_id'].astype(str))
    edges_filtered = edges_filtered[
        edges_filtered['parent_tweet_id'].isin(remaining_tweet_ids) &
        edges_filtered['child_tweet_id'].isin(remaining_tweet_ids)
    ]
    print(f"  Edges: {initial_edge_count:,} → {len(edges_filtered):,}")

# 4.2 Remove invalid timestamps
if FILTER_CONFIG['remove_invalid_timestamps']:
    print("\n Removing tweets with invalid timestamps:")
    before = len(tweets_filtered)
    tweets_filtered = tweets_filtered[tweets_filtered['created_at_parsed'].notna()]
    after = len(tweets_filtered)
    print(f"  Tweets: {before:,} → {after:,} (removed {before-after:,})")
    
    # Update edges
    remaining_tweet_ids = set(tweets_filtered['tweet_id'].astype(str))
    edges_filtered = edges_filtered[
        edges_filtered['parent_tweet_id'].isin(remaining_tweet_ids) &
        edges_filtered['child_tweet_id'].isin(remaining_tweet_ids)
    ]
    print(f"  Edges: {initial_edge_count:,} → {len(edges_filtered):,}")

# 4.3 Filter by minimum thread size
if FILTER_CONFIG['min_thread_size'] > 1:
    print(f"\n Filtering threads with less than {FILTER_CONFIG['min_thread_size']} tweets:")
    
    # Count tweets per thread
    thread_sizes = tweets_filtered.groupby('thread_id').size()
    valid_threads = thread_sizes[thread_sizes >= FILTER_CONFIG['min_thread_size']].index
    
    before_tweets = len(tweets_filtered)
    before_threads = threads_filtered['thread_id'].nunique()
    
    tweets_filtered = tweets_filtered[tweets_filtered['thread_id'].isin(valid_threads)]
    threads_filtered = threads_filtered[threads_filtered['thread_id'].isin(valid_threads)]
    
    # Update edges
    remaining_tweet_ids = set(tweets_filtered['tweet_id'].astype(str))
    edges_filtered = edges_filtered[
        edges_filtered['parent_tweet_id'].isin(remaining_tweet_ids) &
        edges_filtered['child_tweet_id'].isin(remaining_tweet_ids)
    ]
    
    after_tweets = len(tweets_filtered)
    after_threads = threads_filtered['thread_id'].nunique()
    
    print(f"  Threads: {before_threads:,} → {after_threads:,} (removed {before_threads-after_threads:,})")
    print(f"  Tweets: {before_tweets:,} → {after_tweets:,}")
    print(f"  Edges: {initial_edge_count:,} → {len(edges_filtered):,}")



print("Filtering results")
print("-" * 80)
print(f"Initial --> Final:")
print(f"  Tweets:  {initial_tweet_count:,} → {len(tweets_filtered):,} ({100*len(tweets_filtered)/initial_tweet_count:.1f}% retained)")
print(f"  Edges:   {initial_edge_count:,} → {len(edges_filtered):,} ({100*len(edges_filtered)/initial_edge_count:.1f}% retained)")
print(f"  Threads: {initial_thread_count:,} → {len(threads_filtered):,} ({100*len(threads_filtered)/initial_thread_count:.1f}% retained)")


print("\nFinal validation")
print("-" * 80)

# Recalculate integrity metrics on filtered data
print("\n Edge validity in filtered dataset:")
tweet_ids_filtered = set(tweets_filtered['tweet_id'].astype(str))

parent_exists_f = edges_filtered['parent_tweet_id'].isin(tweet_ids_filtered)
child_exists_f = edges_filtered['child_tweet_id'].isin(tweet_ids_filtered)
both_exist_f = parent_exists_f & child_exists_f

pct_both_f = 100 * both_exist_f.sum() / len(edges_filtered) if len(edges_filtered) > 0 else 0
print(f"  Both parent and child exist: {both_exist_f.sum():,}/{len(edges_filtered):,} ({pct_both_f:.1f}%)")

if pct_both_f < 99:
    print(" Warning: Some edges still reference non-existent tweets")

# Thread size distribution
print("\nThread size distribution (filtered):")
thread_sizes_filtered = tweets_filtered.groupby('thread_id').size()
size_stats = thread_sizes_filtered.describe()
print(f"  Mean: {size_stats['mean']:.1f}")
print(f"  Median: {size_stats['50%']:.1f}")
print(f"  Min: {int(size_stats['min'])}")
print(f"  Max: {int(size_stats['max'])}")

# Veracity distribution
print("\n Veracity distribution (filtered):")
veracity_dist = tweets_filtered['veracity_normalized'].value_counts()
for val, count in veracity_dist.items():
    pct = 100 * count / len(tweets_filtered)
    print(f"  {val}: {count:,} ({pct:.1f}%)")

print("\nSave preprocessed")
print("-" * 80)

# Select final columns to save
tweets_final = tweets_filtered[[
    'event', 'category', 'thread_id', 'tweet_id', 'parent_tweet_id', 'author_id',
    'created_at_parsed', 'text', 'lang', 'retweet_count', 'favorite_count',
    'veracity_normalized', 'is_rumour_normalized'
]].copy()
tweets_final.rename(columns={'created_at_parsed': 'created_at'}, inplace=True)

edges_final = edges_filtered[[
    'event', 'category', 'thread_id', 'parent_tweet_id', 'child_tweet_id',
    'src_user', 'dst_user', 't_edge'
]].copy()

threads_final = threads_filtered[[
    'event', 'category', 'thread_id', 'n_nodes', 'depth', 'duration_min',
    'veracity_normalized', 'is_rumour_normalized'
]].copy()

tweets_out = OUTPUT_DIR / "tweets_clean.csv"
edges_out = OUTPUT_DIR / "edges_clean.csv"
threads_out = OUTPUT_DIR / "threads_clean.csv"

tweets_final.to_csv(tweets_out, index=False)
edges_final.to_csv(edges_out, index=False)
threads_final.to_csv(threads_out, index=False)

print(f"Saved {len(tweets_final):,} tweets in {tweets_out}")
print(f" Saved {len(edges_final):,} edges in {edges_out}")
print(f" Saved {len(threads_final):,} threads in {threads_out}")


print("\n" + "=" * 80)
print("Preprocessing report")
print("=" * 80)

summary = f"""
Integrity:
  - Initial tweets: {initial_tweet_count:,}
  - Initial edges: {initial_edge_count:,}
  - Initial threads: {initial_thread_count:,}
  
After removals:
  - Final tweets: {len(tweets_final):,} ({100*len(tweets_final)/initial_tweet_count:.1f}%)
  - Final edges: {len(edges_final):,} ({100*len(edges_final)/initial_edge_count:.1f}%)
  - Final threads: {len(threads_final):,} ({100*len(threads_final)/initial_thread_count:.1f}%)

Threads stats:
  - Mean size: {thread_sizes_filtered.mean():.1f} tweets
  - Median size: {thread_sizes_filtered.median():.1f} tweets
  - Largest thread: {thread_sizes_filtered.max()} tweets

output files:
  - {tweets_out}
  - {edges_out}
  - {threads_out}
 
"""
print(summary)



print("done!")
