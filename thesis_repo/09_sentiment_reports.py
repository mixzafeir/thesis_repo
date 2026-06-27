# 09_sentiment_reports.py — Sentiment analysis plots
# Run: python 09_sentiment_reports.py
# Depends on: 03_sentiment.py (messages_clean_plus_sentiment.parquet)

import os
import pandas as pd
import matplotlib.pyplot as plt

PROC_DIR  = "data_processed"
SENT_OUT  = os.path.join(PROC_DIR, "messages_clean_plus_sentiment.parquet")
PLOTS_DIR = os.path.join(PROC_DIR, "plots", "sentiment")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_sentiment_distribution(df):
    ax = df["sentiment_label"].value_counts().plot(kind="bar")
    ax.set_title("Sentiment distribution")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sentiment_overall.png"), dpi=200)
    plt.close()
    print("Saved: sentiment_overall.png")


def plot_sentiment_by_platform(df):
    ct = pd.crosstab(df["platform"], df["sentiment_label"])
    ax = ct.div(ct.sum(axis=1), axis=0).mul(100).plot(kind="bar", stacked=True, rot=0)
    ax.set_title("Sentiment distribution by platform (%)")
    ax.set_ylabel("% of messages")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sentiment_by_platform.png"), dpi=200)
    plt.close()
    print("Saved: sentiment_by_platform.png")


def main():
    if not os.path.exists(SENT_OUT):
        raise SystemExit(f"Missing: {SENT_OUT} — run 03_sentiment.py first.")

    ensure_dir(PLOTS_DIR)
    df = pd.read_parquet(SENT_OUT)
    print(f"Loaded {len(df):,} rows")
    print("\nSentiment distribution:")
    print(df["sentiment_label"].value_counts().to_string())

    plot_sentiment_distribution(df)
    plot_sentiment_by_platform(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
