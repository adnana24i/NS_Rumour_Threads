"""
parse_pheme_debug.py
Parse a PHEME dataset into three CSVs with verbose diagnostics and logging.

Outputs (always written; even if empty):
  - tweets.csv   (one row per tweet: source + reactions)
  - edges.csv    (parent_tweet -> child_tweet with src/dst authors and child timestamp)
  - threads.csv  (per-thread stats: size, depth, duration, labels)

Example:
  python parse_pheme_debug.py --root "PATH/TO/all-rnr-annotated-threads" --out parsed --workers 8 --limit 50 --verbose

Requirements:
  - Python 3.9+
  - pandas
Optional:
  - orjson (faster JSON)
"""

import argparse
import sys
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

# Optional speedup for JSON decoding
try:
    import orjson as _fastjson
except Exception:
    _fastjson = None

TWITTER_TS = "%a %b %d %H:%M:%S %z %Y"


# ----------------------------
# Logging helpers
# ----------------------------

class Logger:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def info(self, msg: str):
        print(f"[INFO] {msg}")

    def warn(self, msg: str):
        print(f"[WARN] {msg}", file=sys.stderr)

    def error(self, msg: str):
        print(f"[ERROR] {msg}", file=sys.stderr)

    def debug(self, msg: str):
        if self.verbose:
            print(f"[DEBUG] {msg}")


# ----------------------------
# Diagnostics containers
# ----------------------------

@dataclass
class ThreadDiag:
    event: str
    category: str
    thread_id: str
    source_present: bool = False
    reactions_count: int = 0
    annotation_present: bool = False
    structure_present: bool = False
    structure_pairs: int = 0
    tweets_loaded: int = 0
    edges_from_structure: int = 0
    edges_from_reply_link: int = 0
    skipped_missing_auth: int = 0
    skipped_missing_time: int = 0
    json_load_errors: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class GlobalDiag:
    events_found: int = 0
    threads_enumerated: int = 0
    threads_parsed: int = 0
    total_tweets: int = 0
    total_edges: int = 0
    total_threads_rows: int = 0
    total_json_errors: int = 0
    total_missing_structure: int = 0
    total_empty_structure: int = 0
    total_threads_no_tweets: int = 0
    total_skipped_auth: int = 0
    total_skipped_time: int = 0
    total_edges_from_structure: int = 0
    total_edges_from_reply: int = 0


# ----------------------------
# Robust I/O helpers
# ----------------------------

def clean_json_list(dir_path: Path, log: Logger) -> List[Path]:
    """Return only real JSON files (skip macOS AppleDouble sidecars like '._*')."""
    if not dir_path.exists():
        log.debug(f"Directory not found (ok if optional): {dir_path}")
        return []
    files = [p for p in dir_path.glob("*.json") if not p.name.startswith("._")]
    return files


def load_json(path: Path, diag: ThreadDiag, log: Logger) -> Optional[dict]:
    """UTF-8/BOM tolerant loader; uses orjson if available. Returns None on error."""
    try:
        if not path.exists():
            diag.notes.append(f"Missing file: {path.name}")
            log.debug(f"Missing JSON: {path}")
            return None
        if _fastjson is not None:
            return _fastjson.loads(path.read_bytes())
        import json
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        diag.json_load_errors += 1
        log.warn(f"Failed to load JSON: {path} -> {e}")
        log.debug(traceback.format_exc())
        return None


def parse_ts(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, TWITTER_TS)
    except Exception:
        return None


def is_event_dir(d: Path) -> bool:
    return d.is_dir() and d.name.endswith("-all-rnr-threads")


def autodetect_root(start: Path, log: Logger) -> Optional[Path]:
    cand = start / "all-rnr-annotated-threads"
    if cand.is_dir():
        log.info(f"Auto-detected root at {cand}")
        return cand
    for d in start.iterdir():
        if d.is_dir():
            try:
                if any(is_event_dir(x) for x in d.iterdir()):
                    log.info(f"Auto-detected root at {d}")
                    return d
            except PermissionError:
                continue
    try:
        if any(is_event_dir(x) for x in start.iterdir()):
            log.info(f"Treating current directory as root: {start}")
            return start
    except PermissionError:
        pass
    return None


def validate_root(root: Path, log: Logger) -> Tuple[bool, str, int]:
    if not root.exists():
        return False, f"Root path does not exist: {root}", 0
    events = [d for d in root.iterdir() if is_event_dir(d)]
    if not events:
        return False, f"No event folders found under: {root}", 0
    has_threads = False
    for ev in events:
        ok = False
        for cat in ("rumours", "non-rumours"):
            p = ev / cat
            if p.exists() and any(x.is_dir() for x in p.iterdir()):
                ok = True
                has_threads = True
        if not ok:
            log.warn(f"Event has no threads in expected places: {ev}")
    if not has_threads:
        return False, "No threads found in any event.", len(events)
    return True, "Root looks valid.", len(events)


def enumerate_threads(root: Path) -> List[Tuple[str, str, Path]]:
    triples: List[Tuple[str, str, Path]] = []
    for event_dir in [d for d in root.iterdir() if is_event_dir(d)]:
        for category in ("rumours", "non-rumours"):
            cat = event_dir / category
            if not cat.exists():
                continue
            for thr in [d for d in cat.iterdir() if d.is_dir()]:
                triples.append((event_dir.name, category, thr))
    return triples


# ----------------------------
# Per-thread parsing with diagnostics
# ----------------------------

def parse_thread(event: str, category: str, thr_dir: Path, log: Logger) -> Tuple[List[dict], List[dict], dict, ThreadDiag]:
    diag = ThreadDiag(event=event, category=category, thread_id=thr_dir.name)

    # Load annotation and structure
    ann_path = thr_dir / "annotation.json"
    struct_path = thr_dir / "structure.json"
    ann = load_json(ann_path, diag, log) or {}
    struct = load_json(struct_path, diag, log) or {}
    diag.annotation_present = bool(ann)
    diag.structure_present = bool(struct)

    if not diag.structure_present:
        diag.notes.append("structure.json missing or empty")

    # Labels (vary by PHEME release)
    is_rumour = ann.get("is_rumour")
    veracity_bool = ann.get("true")  # sometimes 1/0 or True/False
    veracity: Optional[str] = None
    if isinstance(veracity_bool, (bool, int)):
        veracity = "True" if bool(veracity_bool) else "False"

    # Load tweets
    src_files = clean_json_list(thr_dir / "source-tweets", log)
    diag.source_present = len(src_files) > 0
    source = load_json(src_files[0], diag, log) if src_files else None

    react_files = clean_json_list(thr_dir / "reactions", log)
    diag.reactions_count = len(react_files)
    reactions = [load_json(p, diag, log) for p in react_files]
    reactions = [r for r in reactions if r]

    tweets: List[dict] = []
    if source:
        tweets.append(source)
    tweets.extend(reactions)

    # Build lookups
    id2tweet: Dict[str, dict] = {}
    id2author: Dict[str, str] = {}
    id2time: Dict[str, Optional[datetime]] = {}

    for tw in tweets:
        tid = str(tw.get("id") or tw.get("id_str") or "")
        if not tid:
            diag.notes.append("Tweet without id/id_str skipped")
            continue
        id2tweet[tid] = tw
        user = tw.get("user", {}) or {}
        id2author[tid] = str(user.get("id") or user.get("id_str") or "")
        id2time[tid] = parse_ts(tw.get("created_at", "")) if tw.get("created_at") else None

    diag.tweets_loaded = len(id2tweet)

    # Tweets rows
    thread_id = thr_dir.name
    tweets_rows: List[dict] = []
    for tid, tw in id2tweet.items():
        user = tw.get("user", {}) or {}
        tweets_rows.append({
            "event": event,
            "category": category,
            "thread_id": thread_id,
            "is_rumour": is_rumour,
            "veracity": veracity,
            "tweet_id": tid,
            "parent_tweet_id": str(tw.get("in_reply_to_status_id") or tw.get("in_reply_to_status_id_str") or ""),
            "author_id": str(user.get("id") or user.get("id_str") or ""),
            "created_at": id2time[tid],
            "text": tw.get("text", ""),
            "lang": tw.get("lang", ""),
            "retweet_count": tw.get("retweet_count"),
            "favorite_count": tw.get("favorite_count"),
        })

    # Edges rows
    edges_rows: List[dict] = []

    def maybe_add_edge(parent_id: str, child_id: str, source_tag: str):
        src_user = id2author.get(parent_id, "")
        dst_user = id2author.get(child_id, "")
        t_edge = id2time.get(child_id)
        if not (src_user and dst_user):
            diag.skipped_missing_auth += 1
            return
        if not t_edge:
            diag.skipped_missing_time += 1
            return
        edges_rows.append({
            "event": event,
            "category": category,
            "thread_id": thread_id,
            "parent_tweet_id": parent_id,
            "child_tweet_id": child_id,
            "src_user": src_user,
            "dst_user": dst_user,
            "t_edge": t_edge,
        })
        if source_tag == "struct":
            diag.edges_from_structure += 1
        else:
            diag.edges_from_reply_link += 1

    # From structure.json
    if isinstance(struct, dict) and struct:
        pair_count = 0
        for parent, children in struct.items():
            if not isinstance(children, list):
                continue
            for child in children:
                pair_count += 1
                maybe_add_edge(str(parent), str(child), source_tag="struct")
        diag.structure_pairs = pair_count
        if pair_count == 0:
            diag.notes.append("structure.json present but had 0 parent->children pairs")
    else:
        diag.structure_pairs = 0

    # Fallback from in_reply_to_*
    for child_id, tw in id2tweet.items():
        pid = tw.get("in_reply_to_status_id") or tw.get("in_reply_to_status_id_str")
        if not pid:
            continue
        parent_id = str(pid)
        if parent_id in id2tweet:
            maybe_add_edge(parent_id, child_id, source_tag="reply")

    # Thread stats
    def depth_from(node: str) -> int:
        if not isinstance(struct, dict) or not struct:
            return 1
        kids = struct.get(node, [])
        if not isinstance(kids, list) or not kids:
            return 1
        return 1 + max(depth_from(str(c)) for c in kids)

    size = len(id2tweet)
    all_parents = set(struct.keys()) if isinstance(struct, dict) else set()
    all_children = {c for v in struct.values() if isinstance(struct, dict) and isinstance(v, list) for c in v}
    roots = list(all_parents - all_children)
    if not roots and id2tweet:
        roots = [tid for tid, tw in id2tweet.items()
                 if not (tw.get("in_reply_to_status_id") or tw.get("in_reply_to_status_id_str"))] or [next(iter(id2tweet))]

    depth = 0
    for r in roots:
        try:
            depth = max(depth, depth_from(str(r)))
        except Exception:
            pass

    times = [t for t in id2time.values() if isinstance(t, datetime)]
    duration_min = ((max(times) - min(times)).total_seconds() / 60.0) if times else None

    summary_row = {
        "event": event,
        "category": category,
        "thread_id": thread_id,
        "is_rumour": is_rumour,
        "veracity": veracity,
        "n_nodes": size,
        "depth": depth,
        "duration_min": duration_min,
    }

    if diag.tweets_loaded == 0:
        diag.notes.append("No tweets loaded in this thread (no source/reactions readable).")

    return tweets_rows, edges_rows, summary_row, diag


# ----------------------------
# Whole-corpus build (parallel optional)
# ----------------------------

def build_tables(root: Path, limit_per_category_per_event: Optional[int], workers: int, log: Logger):
    triples = enumerate_threads(root)
    if not triples:
        log.error("No thread directories found. Check your --root path and dataset layout.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    # Optionally cap per (event, category)
    if limit_per_category_per_event is not None:
        log.info(f"Applying per (event,category) limit = {limit_per_category_per_event}")
        kept, seen = [], {}
        for e, c, t in triples:
            key = (e, c)
            if seen.get(key, 0) < limit_per_category_per_event:
                kept.append((e, c, t))
                seen[key] = seen.get(key, 0) + 1
        triples = kept

    total = len(triples)
    log.info(f"Parsing {total} thread folders with workers={workers}")

    all_tweets: List[dict] = []
    all_edges: List[dict] = []
    all_threads: List[dict] = []
    diags: List[ThreadDiag] = []

    def _worker(args):
        e, c, t = args
        try:
            return parse_thread(e, c, t, log)
        except Exception as ex:
            td = ThreadDiag(event=e, category=c, thread_id=t.name, notes=[f"CRASH: {ex}"])
            td.json_load_errors += 1
            log.error(f"Exception in thread {t}: {ex}")
            log.debug(traceback.format_exc())
            # return empty rows but include diag so we can see the failure
            return [], [], {"event": e, "category": c, "thread_id": t.name}, td

    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(_worker, triple) for triple in triples]
            done = 0
            for fut in as_completed(futures):
                tr, er, sr, diag = fut.get(timeout=None)
                all_tweets.extend(tr)
                all_edges.extend(er)
                all_threads.append(sr)
                diags.append(diag)
                done += 1
                if done % 100 == 0 or log.verbose:
                    log.info(f"Progress: {done}/{total} threads parsed "
                             f"(tweets+={len(tr)}, edges+={len(er)})")
    else:
        # Single-threaded (easier to debug)
        for i, triple in enumerate(triples, 1):
            tr, er, sr, diag = _worker(triple)
            all_tweets.extend(tr)
            all_edges.extend(er)
            all_threads.append(sr)
            diags.append(diag)
            if i % 100 == 0 or log.verbose:
                log.info(f"Progress: {i}/{total} threads parsed (tweets+={len(tr)}, edges+={len(er)})")

    tweets_df = pd.DataFrame(all_tweets)
    edges_df = pd.DataFrame(all_edges)
    threads_df = pd.DataFrame(all_threads)

    return tweets_df, edges_df, threads_df, diags


def summarize_and_log(tweets_df: pd.DataFrame,
                      edges_df: pd.DataFrame,
                      threads_df: pd.DataFrame,
                      diags: List[ThreadDiag],
                      events_count: int,
                      log: Logger):
    g = GlobalDiag()
    g.events_found = events_count
    g.threads_parsed = len(diags)
    g.total_tweets = len(tweets_df)
    g.total_edges = len(edges_df)
    g.total_threads_rows = len(threads_df)

    for d in diags:
        g.total_json_errors += d.json_load_errors
        if not d.structure_present:
            g.total_missing_structure += 1
        if d.structure_present and d.structure_pairs == 0:
            g.total_empty_structure += 1
        if d.tweets_loaded == 0:
            g.total_threads_no_tweets += 1
        g.total_skipped_auth += d.skipped_missing_auth
        g.total_skipped_time += d.skipped_missing_time
        g.total_edges_from_structure += d.edges_from_structure
        g.total_edges_from_reply += d.edges_from_reply_link

    log.info("--- Summary ---")
    log.info(f"Events found:                 {g.events_found}")
    log.info(f"Threads parsed:               {g.threads_parsed}")
    log.info(f"Tweets total (rows):          {g.total_tweets}")
    log.info(f"Edges total (rows):           {g.total_edges}")
    log.info(f"Threads summary rows:         {g.total_threads_rows}")
    log.info(f"JSON load errors (sum):       {g.total_json_errors}")
    log.info(f"Threads w/ no structure.json: {g.total_missing_structure}")
    log.info(f"structure.json with 0 pairs:  {g.total_empty_structure}")
    log.info(f"Threads with 0 tweets:        {g.total_threads_no_tweets}")
    log.info(f"Edges from structure:         {g.total_edges_from_structure}")
    log.info(f"Edges from in_reply_to:       {g.total_edges_from_reply}")
    log.info(f"Edges skipped (no authors):   {g.total_skipped_auth}")
    log.info(f"Edges skipped (no time):      {g.total_skipped_time}")

    # Heuristics to explain empty CSVs
    if g.total_tweets == 0:
        log.error("No tweets were loaded. Likely causes: wrong --root, cloud placeholders not downloaded, or permissions.")
    if g.total_edges == 0:
        if g.total_missing_structure + g.total_empty_structure > 0:
            log.warn("No edges built. Many threads lack usable structure.json. Relying on in_reply_to fallback.")
        if g.total_edges_from_reply == 0:
            log.error("No edges from in_reply_to either. Likely the tweet JSONs for parents/children are missing locally "
                      "(e.g., not downloaded from OneDrive) or in_reply_to_* fields are absent in this PHEME copy.")


def ensure_out_dir(out: Path, log: Logger) -> bool:
    try:
        out.mkdir(parents=True, exist_ok=True)
        # quick writability check
        test_file = out / ".write_test"
        with open(test_file, "w", encoding="utf-8") as fh:
            fh.write("ok")
        test_file.unlink(missing_ok=True)
        return True
    except Exception as e:
        log.error(f"Cannot write to output directory {out}: {e}")
        log.debug(traceback.format_exc())
        return False


def save_csvs(tweets_df: pd.DataFrame, edges_df: pd.DataFrame, threads_df: pd.DataFrame, out: Path, log: Logger):
    # Normalize datetimes to ISO strings (avoid tz-naive surprises in CSVs)
    if "created_at" in tweets_df.columns and not tweets_df.empty:
        tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"], utc=True).astype(str)
    if "t_edge" in edges_df.columns and not edges_df.empty:
        edges_df["t_edge"] = pd.to_datetime(edges_df["t_edge"], utc=True).astype(str)

    # Always write files, even if empty, to make the outcome explicit.
    tweets_path = out / "tweets.csv"
    edges_path = out / "edges.csv"
    threads_path = out / "threads.csv"

    try:
        tweets_df.to_csv(tweets_path, index=False)
        log.info(f"Wrote {len(tweets_df):,} rows -> {tweets_path}")
    except Exception as e:
        log.error(f"Failed to write {tweets_path}: {e}")
        log.debug(traceback.format_exc())

    try:
        edges_df.to_csv(edges_path, index=False)
        log.info(f"Wrote {len(edges_df):,} rows -> {edges_path}")
    except Exception as e:
        log.error(f"Failed to write {edges_path}: {e}")
        log.debug(traceback.format_exc())

    try:
        threads_df.to_csv(threads_path, index=False)
        log.info(f"Wrote {len(threads_df):,} rows -> {threads_path}")
    except Exception as e:
        log.error(f"Failed to write {threads_path}: {e}")
        log.debug(traceback.format_exc())


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse PHEME dataset into CSVs with diagnostics.")
    ap.add_argument("--root", type=str, default=None,
                    help="Path to PHEME root (folder containing event dirs ending with -all-rnr-threads). "
                         "If omitted, try auto-detect under current directory.")
    ap.add_argument("--out", type=str, default="parsed",
                    help="Output directory for CSVs (default: ./parsed)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional limit per event/category (quick test)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker threads for parsing (I/O-bound). Default: 1 (easier debug).")
    ap.add_argument("--verbose", action="store_true",
                    help="Print debug details while running.")
    args = ap.parse_args()

    log = new_logger = Logger(verbose=args.verbose)

    # Resolve root
    if args.root:
        root = Path(args.root)
    else:
        root = autodetect_root(Path.cwd(), log)

    if root is None:
        log.error("Could not auto-detect dataset root. Pass --root PATH/TO/all-rnr-annotated-threads")
        sys.exit(2)

    ok, msg, events_count = validate_root(root, log)
    if not ok:
        log.error(msg)
        sys.exit(2)
    log.info(f"Root: {root} | {msg} | Events: {events_count}")

    out = Path(args.out)
    if not ensure_out_dir(out, log):
        sys.exit(3)

    tweets_df, edges_df, threads_df, diags = build_tables(root, args.limit, args.workers, log)
    summarize_and_log(tweets_df, edges_df, threads_df, diags, events_count, log)
    save_csvs(tweets_df, edges_df, threads_df, out, log)

    # Post-run hints if something is off
    if len(tweets_df) == 0:
        log.warn("tweets.csv is empty. Check that your dataset files are fully present (not cloud placeholders) and readable.")
    if len(edges_df) == 0:
        log.warn("edges.csv is empty. Inspect logs above: if many threads lack structure.json or in_reply_to_* parents "
                 "aren't present locally, you may need to fully extract/sync the dataset or verify JSON fields.")

    log.info("Done.")

if __name__ == "__main__":
    main()
