# 01_collect.py — Load Telegram JSON files into messages_clean.parquet
# Run: python 01_collect.py

import os
import json
import glob
import pandas as pd
from tqdm import tqdm

RAW_DIR  = "data_raw"
PROC_DIR = "data_processed"
ALL_OUT  = os.path.join(PROC_DIR, "messages_clean.parquet")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_telegram_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    channel = os.path.splitext(os.path.basename(path))[0]
    for item in data:
        msg = item.get("message", "")
        if not msg or not isinstance(msg, str):
            continue
        rows.append({
            "platform":      "telegram",
            "source":        channel,
            "post_id":       str(item.get("id")),
            "created_at":    item.get("timestamp"),
            "text":          msg,
            "channel_label": item.get("label", ""),
        })
    return pd.DataFrame(rows)


def run_yield_report():
    print("\n── Channel yield report ──")
    totals = {"pro_palestine": 0, "pro_israel": 0, "unknown": 0}
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RAW_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)
        label = data[0].get("label", "unknown") if data else "unknown"
        totals[label] = totals.get(label, 0) + len(data)
        print(f"  {fname:<45} {len(data):>6,}  [{label}]")
    print(f"\n  pro_palestine total : {totals['pro_palestine']:>6,}")
    print(f"  pro_israel total    : {totals['pro_israel']:>6,}")
    print(f"  Grand total         : {sum(totals.values()):>6,}")


def main():
    paths = glob.glob(os.path.join(RAW_DIR, "*.json"))
    if not paths:
        raise SystemExit("No .json files found in data_raw.")

    dfs = [load_telegram_file(p) for p in tqdm(paths, desc="Telegram JSON")]
    df  = pd.concat(dfs, ignore_index=True)
    df  = df[df["text"].astype(str).str.strip().astype(bool)]
    df  = df.reset_index(drop=True)
    df.insert(0, "row_id", range(1, len(df) + 1))

    ensure_dir(PROC_DIR)
    df.to_parquet(ALL_OUT, index=False)
    print(f"Total rows: {len(df):,}  →  {ALL_OUT}")
    print(df["platform"].value_counts())

    run_yield_report()
    print("\nDone. Next: python 02_clean.py")


if __name__ == "__main__":
    main()
