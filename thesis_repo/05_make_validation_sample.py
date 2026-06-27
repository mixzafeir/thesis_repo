# 05_make_validation_sample.py — Generate validation_sample_1000.csv for manual labeling
# Run ONCE after 04_keywords.py, before manual labeling.
# WARNING: Do NOT rerun after you have filled in manual_label — it will overwrite your work.
# Run: python 05_make_validation_sample.py

import os
import pandas as pd

PROC_DIR   = "data_processed"
STANCE_OUT = os.path.join(PROC_DIR, "messages_plus_sentiment_keywords.parquet")
VALID_OUT  = os.path.join(PROC_DIR, "validation_sample_1000.csv")


def main():
    if not os.path.exists(STANCE_OUT):
        raise SystemExit(f"Missing: {STANCE_OUT} — run 04_keywords.py first.")

    if os.path.exists(VALID_OUT):
        confirm = input(f"{VALID_OUT} already exists. Overwrite? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return

    df = pd.read_parquet(STANCE_OUT)
    parts = []
    for label in ["pro_palestine", "pro_israel", "neutral"]:
        sub = df[df["stance_label"] == label]
        n   = min(333, len(sub))
        if n == 0:
            print(f"Warning: no rows for '{label}'")
            continue
        parts.append(sub.sample(n, random_state=42))

    sample = pd.concat(parts).sample(frac=1, random_state=42).copy()
    sample = sample[["row_id", "text"]].copy()
    sample["manual_label"] = ""
    sample.to_csv(VALID_OUT, index=False, encoding="utf-8")
    print(f"Saved: {VALID_OUT}  ({len(sample)} rows)")
    print(sample["manual_label"].value_counts())


if __name__ == "__main__":
    main()
