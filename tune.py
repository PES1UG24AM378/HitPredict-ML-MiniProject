"""
tune.py - (1) tune Random Forest and SVM on the balanced data,
          (2) evaluate on the full IMBALANCED data with class weights.

Usage: python tune.py
Outputs in ./results/: tuning_results.csv, imbalanced_results.csv, best_model.joblib (updated)
Note: the SVM search can take a few minutes.
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
            "key", "mode", "time_signature", "artist_hits"]


def load(path):
    df = pd.read_csv(path)
    feats = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=feats + ["label"])
    return df[feats], df["label"], feats


def report(name, model, Xtr, ytr, Xte, yte):
    pred, prob = model.predict(Xte), model.predict_proba(Xte)[:, 1]
    return {
        "model": name,
        "train_acc": round(accuracy_score(ytr, model.predict(Xtr)), 3),
        "test_acc": round(accuracy_score(yte, pred), 3),
        "precision": round(precision_score(yte, pred, zero_division=0), 3),
        "recall": round(recall_score(yte, pred), 3),
        "f1": round(f1_score(yte, pred), 3),
        "roc_auc": round(roc_auc_score(yte, prob), 3),
        "pr_auc": round(average_precision_score(yte, prob), 3),
    }


# ---------- Part 1: tuning on balanced data ----------
X, y, feats = load("data/balanced_labeled.csv")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

searches = {
    "Random Forest (tuned)": RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 300, 500],
         "max_depth": [4, 6, 8, 10, 12],
         "min_samples_leaf": [5, 10, 20, 40],
         "max_features": ["sqrt", 0.5]},
        n_iter=20, cv=5, scoring="roc_auc", random_state=SEED, n_jobs=-1),
    "SVM RBF (tuned)": RandomizedSearchCV(
        make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
        {"svc__C": [0.1, 0.5, 1, 2, 5, 10],
         "svc__gamma": ["scale", 0.01, 0.03, 0.1]},
        n_iter=12, cv=5, scoring="roc_auc", random_state=SEED, n_jobs=-1),
}

rows, best = [], {}
for name, s in searches.items():
    s.fit(Xtr, ytr)
    print(f"{name}: best params = {s.best_params_} | cv AUC = {s.best_score_:.3f}")
    rows.append(report(name, s.best_estimator_, Xtr, ytr, Xte, yte))
    best[name] = s.best_estimator_

tuned = pd.DataFrame(rows)
tuned.to_csv("results/tuning_results.csv", index=False)
print("\nTuned models (balanced data):\n", tuned.to_string(index=False))

top = tuned.sort_values("roc_auc", ascending=False).iloc[0]["model"]
joblib.dump({"model": best[top], "features": feats}, "results/best_model.joblib")
print(f"\nSaved best tuned model: {top} -> results/best_model.joblib")

# ---------- Part 2: full imbalanced data + class weights ----------
Xf, yf, _ = load("data/full_labeled.csv")
print(f"\nFull data: {len(yf)} songs, hit rate {yf.mean():.2%}")
Xtr, Xte, ytr, yte = train_test_split(Xf, yf, test_size=0.25, stratify=yf, random_state=SEED)

imb = {
    "RF balanced weights": RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=10,
        class_weight="balanced", random_state=SEED, n_jobs=-1),
    "LogReg balanced weights": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "LogReg NO weights (baseline)": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000)),
}
rows = []
for name, m in imb.items():
    m.fit(Xtr, ytr)
    rows.append(report(name, m, Xtr, ytr, Xte, yte))

res = pd.DataFrame(rows)
res.to_csv("results/imbalanced_results.csv", index=False)
print("\nImbalanced data results:\n", res.to_string(index=False))
print("\nOn imbalanced data, judge by precision, recall, F1 and PR-AUC, not accuracy "
      "(predicting 'non-hit' for everything already gives high accuracy).")
