"""
Zero-shot stance detection using DeBERTa.
Run after: 02_clean.py
Run: python 06_zeroshot.py
Adds a 'stance_zeroshot' column and saves a new parquet.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

PROC_DIR     = "data_processed"
ALL_OUT      = os.path.join(PROC_DIR, "messages_clean.parquet")
ZEROSHOT_OUT = os.path.join(PROC_DIR, "messages_clean_plus_zeroshot.parquet")
CHECKPOINT   = os.path.join(PROC_DIR, "zeroshot_checkpoint.parquet")

MODEL      = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
LABELS     = ["pro_palestine", "pro_israel", "neutral"]
BATCH_SIZE = 8    # small batches to keep memory under control
SAVE_EVERY = 5000 # checkpoint every N rows


def main():
    if not os.path.exists(ALL_OUT):
        raise SystemExit(f"Missing input file: {ALL_OUT}\nRun 02_clean.py first.")

    df = pd.read_parquet(ALL_OUT)
    print(f"Loaded {len(df):,} rows from {ALL_OUT}")

    # Resume from checkpoint if available
    start_idx = 0
    all_labels = [None] * len(df)
    if os.path.exists(CHECKPOINT):
        ckpt = pd.read_parquet(CHECKPOINT)
        done = ckpt["stance_zeroshot"].notna().sum()
        if done > 0:
            print(f"Resuming from checkpoint — {done:,} rows already done.")
            start_idx = done
            for i, lbl in enumerate(ckpt["stance_zeroshot"].tolist()):
                all_labels[i] = lbl

    print(f"Loading model: {MODEL}")
    classifier = pipeline("zero-shot-classification", model=MODEL, device=-1)

    texts = df["text"].fillna("").tolist()

    print("Running zero-shot classification …")
    with tqdm(total=len(texts), initial=start_idx, unit="post") as pbar:
        for i in range(start_idx, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            results = classifier(
                batch,
                candidate_labels=LABELS,
                multi_label=False,
                batch_size=BATCH_SIZE,
            )
            for j, r in enumerate(results):
                all_labels[i + j] = r["labels"][0]
            pbar.update(len(batch))

            # Save checkpoint
            if (i + BATCH_SIZE) % SAVE_EVERY < BATCH_SIZE:
                df["stance_zeroshot"] = all_labels
                df.to_parquet(CHECKPOINT, index=False)

    df["stance_zeroshot"] = all_labels

    dist = df["stance_zeroshot"].value_counts()
    print("\nZero-shot stance distribution:")
    print(dist.to_string())

    df.to_parquet(ZEROSHOT_OUT, index=False)
    print(f"\nSaved → {ZEROSHOT_OUT}")

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
        print("Checkpoint removed.")


if __name__ == "__main__":
    main()
