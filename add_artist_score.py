"""
add_artist_score.py - add an 'artist_hits' feature to the labeled data.

artist_hits = number of the artist's OTHER Billboard Hot 100 songs
(the song's own chart appearance is excluded, so its label does not leak).

Usage (same flags you used for build_dataset.py):
  python add_artist_score.py --billboard charts.csv --min-year 1990 --max-year 2018

Needs build_dataset.py in the same folder. Original files are backed up as
data/*_noartist.csv, and data/full_labeled.csv + data/balanced_labeled.csv are updated.
"""
import argparse
import os
import shutil

import pandas as pd

from build_dataset import clean_title, primary_artist

ap = argparse.ArgumentParser()
ap.add_argument("--billboard", required=True)
ap.add_argument("--min-year", type=int)
ap.add_argument("--max-year", type=int)
args = ap.parse_args()

bb = pd.read_csv(args.billboard)
tcol = next(c for c in ["song", "title", "track"] if c in bb.columns)
acol = next(c for c in ["artist", "artists"] if c in bb.columns)
dcol = next((c for c in ["date", "chart_date"] if c in bb.columns), None)

if dcol and (args.min_year or args.max_year):
    yr = pd.to_datetime(bb[dcol], errors="coerce").dt.year
    if args.min_year:
        bb = bb[yr >= args.min_year]
    if args.max_year:
        bb = bb[yr <= args.max_year]

bb["artist_key"] = bb[acol].map(primary_artist)
bb["song_key"] = bb[tcol].map(clean_title)
songs = bb.drop_duplicates(["song_key", "artist_key"])
hits_per_artist = songs.groupby("artist_key").size().to_dict()
print(f"Artists with at least one hit: {len(hits_per_artist)}")


def load(name):
    """Read the original (pre-artist) version, backing it up on first run."""
    path, backup = f"data/{name}.csv", f"data/{name}_noartist.csv"
    if not os.path.exists(backup):
        shutil.copy(path, backup)
    return pd.read_csv(backup)


full = load("full_labeled")
full["artist_key"] = full["match_key"].str.rsplit("|", n=1).str[-1]
full["artist_hits"] = (full["artist_key"].map(hits_per_artist).fillna(0)
                       - full["label"]).clip(lower=0).astype(int)
full.drop(columns="artist_key").to_csv("data/full_labeled.csv", index=False)

bal = load("balanced_labeled")
lookup = full.drop_duplicates("match_key").set_index("match_key")["artist_hits"]
bal["artist_hits"] = bal["match_key"].map(lookup).astype(int)
bal.to_csv("data/balanced_labeled.csv", index=False)

print("\nMean artist_hits by label (balanced data):")
print(bal.groupby("label")["artist_hits"].mean().round(2).to_string())
print("\nDone. Now add \"artist_hits\" to FEATURES in train.py and tune.py.")
