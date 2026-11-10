"""Train five CredibilityLens experiments and save the selected model.

Run: python train.py
Or:  python train.py --csv path/to/news.csv

CSV columns: text,label (optional title). Labels should identify FAKE and REAL.
"""
import argparse
import glob
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core import clean_text, prep, style_vector, build_baseline

STYLE_W = 0.25

def load_dataset(csv_path):
    if csv_path is None:
        import kagglehub
        folder = kagglehub.dataset_download("antonioskokiantonis/newscsv")
        candidates = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
        preferred = [p for p in candidates if os.path.basename(p).lower() == "news.csv"]
        candidates = preferred or candidates
        if not candidates:
            raise FileNotFoundError("No CSV file was found in the downloaded Kaggle dataset.")
        csv_path = candidates[0]

    df = pd.read_csv(csv_path)
    required = {"text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    if "title" in df.columns:
        df["text"] = df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[df["label"].isin(["FAKE", "REAL"])].reset_index(drop=True)
    if df["label"].nunique() != 2:
        raise ValueError("The dataset must contain both FAKE and REAL labels.")
    return df

def fit_eval(name, x_train, x_test, y_train, y_test):
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = {
        "setup": name,
        "accuracy": round(accuracy_score(y_test, predicted) * 100, 2),
        "f1_fake": round(f1_score(y_test, predicted, pos_label="FAKE") * 100, 2),
    }
    return model, metrics

def main(csv_path=None):
    df = load_dataset(csv_path)
    print("Rows:", len(df))
    print(df["label"].value_counts())

    print("Extracting writing-style features...")
    style = np.vstack([style_vector(text) for text in df["text"]])
    raw_clean = df["text"].map(clean_text)
    masked = df["text"].map(prep)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42,
        stratify=df["label"],
    )
    y_train = df["label"].values[train_idx]
    y_test = df["label"].values[test_idx]

    scaler = StandardScaler()
    style_train = csr_matrix(scaler.fit_transform(style[train_idx]) * STYLE_W)
    style_test = csr_matrix(scaler.transform(style[test_idx]) * STYLE_W)

    results = []

    _, metrics = fit_eval("Style only", style_train, style_test, y_train, y_test)
    results.append(metrics)

    raw_vectorizer = TfidfVectorizer(
        stop_words="english", max_df=0.7, min_df=2, max_features=50000
    )
    raw_train = raw_vectorizer.fit_transform(raw_clean.iloc[train_idx])
    raw_test = raw_vectorizer.transform(raw_clean.iloc[test_idx])

    _, metrics = fit_eval("Text only (raw)", raw_train, raw_test, y_train, y_test)
    results.append(metrics)
    _, metrics = fit_eval(
        "Hybrid (raw)", hstack([raw_train, style_train]),
        hstack([raw_test, style_test]), y_train, y_test,
    )
    results.append(metrics)

    vectorizer = TfidfVectorizer(
        stop_words="english", max_df=0.7, min_df=2, max_features=50000
    )
    masked_train = vectorizer.fit_transform(masked.iloc[train_idx])
    masked_test = vectorizer.transform(masked.iloc[test_idx])

    _, metrics = fit_eval("Text only (masked)", masked_train, masked_test, y_train, y_test)
    results.append(metrics)
    model, metrics = fit_eval(
        "Hybrid (masked)", hstack([masked_train, style_train]),
        hstack([masked_test, style_test]), y_train, y_test,
    )
    results.append(metrics)

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))
    shortcut_gap = results[2]["accuracy"] - results[4]["accuracy"]
    print(f"\nShortcut gap (hybrid raw - hybrid masked): {shortcut_gap:.2f} accuracy points")

    predictions = model.predict(hstack([masked_test, style_test]))
    print("\nSelected model classification report:")
    print(classification_report(y_test, predictions))

    ax = results_df.plot.barh(
        x="setup", y="accuracy", legend=False, figsize=(8, 4), color="#4C78A8"
    )
    ax.set_xlim(max(0, results_df["accuracy"].min() - 10), 100)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("")
    ax.set_title("CredibilityLens: Ablation and Shortcut Audit")
    plt.tight_layout()
    plt.savefig("ablation.png", dpi=160)
    plt.close()

    baseline = build_baseline(style[train_idx], y_train == "REAL")
    joblib.dump(
        {
            "model": model,
            "tfidf": vectorizer,
            "scaler": scaler,
            "baseline": baseline,
            "style_w": STYLE_W,
        },
        "bundle.joblib",
    )
    with open("results.json", "w", encoding="utf-8") as file:
        json.dump(
            {"table": results, "shortcut_gap": round(float(shortcut_gap), 2)},
            file, indent=2,
        )
    print("Saved bundle.joblib, results.json, and ablation.png.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    main(parser.parse_args().csv)
