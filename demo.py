"""
demo.py - predict the hit probability of a song.

Usage:
  python demo.py --song "blinding lights"     look up a song in the dataset
  python demo.py                              type in audio features by hand

Needs results/best_model.joblib (from train.py or tune.py).
"""
import argparse

import joblib
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--song", help="song title (partial match) to look up in data/full_labeled.csv")
args = ap.parse_args()

bundle = joblib.load("results/best_model.joblib")
model, feats = bundle["model"], bundle["features"]
data = pd.read_csv("data/full_labeled.csv")


def show(title, artist, row, actual=None):
    p = model.predict_proba(pd.DataFrame([row[feats]]))[0, 1]
    print(f"\n{title} - {artist}")
    print(f"Hit probability: {p:.1%}  ->  predicted: {'HIT' if p >= 0.5 else 'NOT A HIT'}")
    if actual is not None:
        print(f"Actual Billboard label: {'HIT' if actual == 1 else 'not a hit'}")


if args.song:
    tcol = "track_name" if "track_name" in data.columns else "name"
    acol = "artists" if "artists" in data.columns else "artist"
    m = data[data[tcol].astype(str).str.lower().str.contains(args.song.lower(), regex=False)]
    if m.empty:
        raise SystemExit("Song not found in the dataset. Run without --song to enter features by hand.")
    for _, r in m.head(5).iterrows():
        show(r[tcol], r[acol], r, r["label"])
    print("\nNote: songs from the dataset may have been in the training data; "
          "use manual input to test unseen songs.")
else:
    print("Enter audio features (press Enter to use the dataset median).")
    row = {}
    for f in feats:
        med = float(data[f].median())
        v = input(f"  {f} [{med:.3g}]: ").strip()
        row[f] = float(v) if v else med
    show("Custom song", "manual input", pd.Series(row))
