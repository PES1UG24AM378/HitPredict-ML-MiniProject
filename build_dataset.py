"""
build_dataset.py - merge Spotify audio features with Billboard Hot 100 labels.

Usage:
  python build_dataset.py --spotify spotify.csv --billboard billboard.csv

Optional:
  --min-year 1990 --max-year 2018   (filter Billboard chart dates)
  --spotify-title-col track_name --spotify-artist-col artists
  --billboard-title-col song --billboard-artist-col artist --billboard-date-col date

Outputs (in ./data/):
  full_labeled.csv      all Spotify songs with label (hit=1 / 0), imbalanced
  balanced_labeled.csv  all hits + equal number of random non-hits (1:1)
"""
import argparse
import os
import re

import pandas as pd

FEATURES = ["danceability", "energy", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "loudness", "tempo",
            "duration_ms"]


def clean_title(t):
    t = str(t).lower()
    t = re.sub(r"[\(\[].*?[\)\]]", " ", t)           # drop (...) and [...]
    t = re.split(r"\s(?:feat\.?|featuring|ft\.?|with)\s", t)[0]
    t = re.sub(r"\s-\s.*$", "", t)                   # drop "- remastered" etc.
    t = re.sub(r"[^a-z0-9 ]", "", t)                 # strip punctuation
    return re.sub(r"\s+", " ", t).strip()


def primary_artist(a):
    a = str(a).lower()
    a = re.sub(r"[\[\]'\"]", "", a)                  # list-string artifacts
    a = re.split(r";|,|\s(?:feat\.?|featuring|ft\.?|with|x|&|and)\s", a)[0]
    a = re.sub(r"^the\s", "", a.strip())
    a = re.sub(r"[^a-z0-9 ]", "", a)
    return re.sub(r"\s+", " ", a).strip()


def pick(df, given, candidates, what):
    if given:
        return given
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"Could not find {what} column. Columns: {list(df.columns)}. "
                     f"Pass it explicitly with the matching flag.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spotify", required=True)
    p.add_argument("--billboard", required=True)
    p.add_argument("--spotify-title-col")
    p.add_argument("--spotify-artist-col")
    p.add_argument("--billboard-title-col")
    p.add_argument("--billboard-artist-col")
    p.add_argument("--billboard-date-col")
    p.add_argument("--min-year", type=int)
    p.add_argument("--max-year", type=int)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sp = pd.read_csv(args.spotify)
    bb = pd.read_csv(args.billboard)
    print(f"Spotify rows: {len(sp)} | Billboard rows: {len(bb)}")

    s_t = pick(sp, args.spotify_title_col, ["track_name", "name", "title"], "Spotify title")
    s_a = pick(sp, args.spotify_artist_col, ["artists", "artist", "artist_name"], "Spotify artist")
    b_t = pick(bb, args.billboard_title_col, ["song", "title", "track"], "Billboard title")
    b_a = pick(bb, args.billboard_artist_col, ["artist", "artists"], "Billboard artist")

    # Optional year filter on the Billboard side
    d_col = args.billboard_date_col or next((c for c in ["date", "chart_date", "year"] if c in bb.columns), None)
    if d_col and (args.min_year or args.max_year):
        yr = pd.to_datetime(bb[d_col], errors="coerce").dt.year
        if d_col == "year":
            yr = bb[d_col]
        if args.min_year:
            bb = bb[yr >= args.min_year]
        if args.max_year:
            bb = bb[yr <= args.max_year]
        print(f"Billboard rows after year filter: {len(bb)}")

    # Build match keys
    sp["match_key"] = sp[s_t].map(clean_title) + "|" + sp[s_a].map(primary_artist)
    bb["match_key"] = bb[b_t].map(clean_title) + "|" + bb[b_a].map(primary_artist)

    # Dedupe Spotify (same song appears on albums/compilations); keep most popular
    if "popularity" in sp.columns:
        sp = sp.sort_values("popularity", ascending=False)
    sp = sp.drop_duplicates("match_key").reset_index(drop=True)

    hit_keys = set(bb["match_key"].unique())
    sp["label"] = sp["match_key"].isin(hit_keys).astype(int)

    n_hits, n_total = sp["label"].sum(), len(sp)
    print(f"Unique Billboard songs: {len(hit_keys)}")
    print(f"Matched hits in Spotify data: {n_hits} of {n_total} songs "
          f"({n_hits / n_total:.1%})")
    print(f"Billboard match rate: {n_hits / len(hit_keys):.1%} "
          "(low is normal: many hits are missing from the Spotify file)")

    os.makedirs("data", exist_ok=True)
    sp.to_csv("data/full_labeled.csv", index=False)

    # Balance 1:1 by undersampling non-hits (as the paper did)
    hits = sp[sp["label"] == 1]
    non = sp[sp["label"] == 0].sample(n=len(hits), random_state=args.seed)
    bal = pd.concat([hits, non]).sample(frac=1, random_state=args.seed)
    bal.to_csv("data/balanced_labeled.csv", index=False)
    print(f"Saved data/full_labeled.csv ({len(sp)}) and "
          f"data/balanced_labeled.csv ({len(bal)})")

    present = [f for f in FEATURES if f in sp.columns]
    print("Audio feature columns found:", present)


if __name__ == "__main__":
    main()
