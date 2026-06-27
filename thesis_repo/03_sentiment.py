# 03_sentiment.py — Sentiment analysis using cardiffnlp/twitter-roberta
# Run: python 03_sentiment.py

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

PROC_DIR  = "data_processed"
ALL_OUT   = os.path.join(PROC_DIR, "messages_clean.parquet")
SENT_OUT  = os.path.join(PROC_DIR, "messages_clean_plus_sentiment.parquet")


def run_sentiment():
    if not os.path.exists(ALL_OUT):
        raise SystemExit(f"Missing: {ALL_OUT} — run 02_clean.py first.")
    df = pd.read_parquet(ALL_OUT)

    _device = "mps" if torch.backends.mps.is_available() else -1
    clf = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True, max_length=512, padding=True,
        device=_device,
    )

    labels, scores = [], []
    for txt in tqdm(df["text"].astype(str), desc="Sentiment"):
        r = clf(txt)[0]
        labels.append(r["label"])
        scores.append(r["score"])

    df["sentiment_label"] = labels
    df["sentiment_score"]  = scores

    os.makedirs(PROC_DIR, exist_ok=True)
    df.to_parquet(SENT_OUT, index=False)
    print(f"Sentiment done: {len(df):,} rows  →  {SENT_OUT}")
    print(df["sentiment_label"].value_counts())
    return df


def main():
    run_sentiment()
    print("\nDone. Next: python 04_keywords.py")


if __name__ == "__main__":
    main()