"""
train.py - train and compare models for hit prediction.

Usage:
  python train.py
  python train.py --data data/balanced_labeled.csv

Needs: pip install pandas scikit-learn matplotlib joblib
Outputs (in ./results/): metrics.csv, confusion matrices, feature importance, best_model.joblib
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Audio features only. Do NOT add 'popularity' - it is leakage
# (popularity is mostly a consequence of being a hit).
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
            "key", "mode", "time_signature", "artist_hits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/balanced_labeled.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    feats = [f for f in FEATURES if f in df.columns]
    print("Using features:", feats)
    df = df.dropna(subset=feats + ["label"])
    X, y = df[feats], df["label"]
    print(f"Rows: {len(df)} | hits: {int(y.sum())} | non-hits: {int((y == 0).sum())}")

    # Stratified 75/25 split, same as the paper
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.seed)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=args.seed)),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10,
                                                random_state=args.seed, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=args.seed),
        "Neural Net (MLP)": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=800, random_state=args.seed)),
    }

    os.makedirs("results", exist_ok=True)
    rows, fitted = [], {}
    for name, m in models.items():
        cv = cross_val_score(m, Xtr, ytr, cv=5, scoring="roc_auc").mean()
        m.fit(Xtr, ytr)
        pred, prob = m.predict(Xte), m.predict_proba(Xte)[:, 1]
        rows.append({
            "model": name,
            "train_acc": accuracy_score(ytr, m.predict(Xtr)),
            "test_acc": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred),
            "recall": recall_score(yte, pred),
            "f1": f1_score(yte, pred),
            "roc_auc": roc_auc_score(yte, prob),
            "cv_auc_train": cv,
        })
        fitted[name] = m
        disp = ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["non-hit", "hit"])
        disp.ax_.set_title(name)
        plt.savefig(f"results/cm_{name.split()[0].lower()}.png", dpi=120, bbox_inches="tight")
        plt.close()

    res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).round(3)
    res.to_csv("results/metrics.csv", index=False)
    print("\n", res.to_string(index=False))
    print("\n(train_acc much higher than test_acc = overfitting)")

    # Feature importance from Random Forest
    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
    imp.plot.barh(title="Random Forest feature importance")
    plt.savefig("results/feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\nTop features:\n", imp.sort_values(ascending=False).head(5).round(3).to_string())

    best = res.iloc[0]["model"]
    joblib.dump({"model": fitted[best], "features": feats}, "results/best_model.joblib")
    print(f"\nBest model by ROC-AUC: {best} -> saved results/best_model.joblib")


if __name__ == "__main__":
    main()
