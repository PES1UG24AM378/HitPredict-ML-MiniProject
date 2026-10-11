# HitPredict: Predicting Billboard Hits from Spotify Audio Features

ML Mini-Project (UE24CS352A) — Problem 21
Team: Shahid, Vilas

Predicts whether a song will chart on the Billboard Hot 100 using its Spotify
audio features and the artist's chart history.

## Results

| Model | Setting | Test Acc | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (tuned) | Balanced 50/50 | 0.912 | 0.939 | 0.881 | 0.958 |
| Random Forest (class-weighted) | Realistic (2% hit rate) | 0.948 | 0.262 | 0.840 | 0.955 |

Adding `artist_hits` (the artist's count of other Billboard hits) raised
ROC-AUC from 0.822 to 0.958 and was the single strongest feature — stronger
than any audio feature alone, matching the finding in the original CS229
paper this project is based on.

## Setup

```bash
pip install pandas scikit-learn matplotlib joblib
```

## Datasets

Download both from Kaggle and place the CSVs in this folder:
- **Spotify audio features**: "Spotify Tracks Dataset" by Yash → `dataset.csv`
- **Billboard Hot 100**: "Billboard 'The Hot 100' Songs" by Dhruvil Dave → `charts.csv`

## How to run (in order)

```bash
# 1. Merge Spotify + Billboard, create labeled datasets
python build_dataset.py --spotify dataset.csv --billboard charts.csv --min-year 1990 --max-year 2018

# 2. Add artist chart-history feature
python add_artist_score.py --billboard charts.csv --min-year 1990 --max-year 2018

# 3. Train baseline models
python train.py

# 4. Hyperparameter tuning + evaluation on realistic (imbalanced) data
python tune.py

# 5. Try the demo
python demo.py --song "shape of you"      # look up a song in the dataset
python demo.py                             # enter audio features manually
```

## Project structure

```
build_dataset.py       Merges Spotify + Billboard data, labels hits
add_artist_score.py    Adds artist_hits feature (artist's other Billboard hits)
train.py                Trains 5 baseline models, saves metrics + confusion matrices
tune.py                 Hyperparameter search + imbalanced-data evaluation
demo.py                 CLI demo: predicts hit probability for a song
data/                   Generated datasets (not in repo — regenerate via scripts)
results/                Metrics, plots, and the trained model (best_model.joblib)
```

## Features used

Audio: danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, duration_ms, key, mode,
time_signature.
Artist history: artist_hits (count of the artist's other Billboard Hot 100
songs, excluding the song itself, to avoid label leakage).

`popularity` is intentionally excluded — it is a consequence of being a hit,
not a predictor of it.

## Limitations

- Non-hit songs are randomly sampled from all eras/genres, while hits are
  1990-2018 mainstream chart songs — some features may partly separate
  "mainstream pop" from "everything else" rather than hit from non-hit.
- `artist_hits` is computed from the artist's *other* chart songs in the same
  window, not strictly prior to each song's release date (release-date data
  was not available in the Spotify file used).
- On realistic, imbalanced data, precision is moderate (~26%) even though
  recall is high (~84%) — the model is tuned to catch most real hits at the
  cost of some false positives, which fits a screening use case better than
  a high-confidence one.

## Reference

Based on: Davies, Mehrabian, Tarassenko, "HitPredict: Predicting Billboard
Hits Using Spotify Data", Stanford CS229, 2018.
