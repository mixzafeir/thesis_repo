# 02_clean.py — Deduplication + language filter
# Run: python 02_clean.py

import os
import pandas as pd
from tqdm import tqdm
from langdetect import detect, LangDetectException

PROC_DIR  = "data_processed"
ALL_OUT   = os.path.join(PROC_DIR, "messages_clean.parquet")
DEDUP_OUT = os.path.join(PROC_DIR, "dedup_stats.csv")
LANG_OUT  = os.path.join(PROC_DIR, "lang_filter_stats.csv")


def run_deduplication():
    if not os.path.exists(ALL_OUT):
        raise SystemExit(f"Missing: {ALL_OUT} — run 01_collect.py first.")
    df = pd.read_parquet(ALL_OUT)
    total_before = len(df)

    df["_text_norm"] = (
        df["text"].astype(str)
        .str.lower()
        .str.replace(r"http\S+", " ", regex=True)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["is_duplicate"] = df.duplicated(subset="_text_norm", keep="first")
    total_dupes  = df["is_duplicate"].sum()
    unique_count = total_before - total_dupes

    print(f"\n── Deduplication report ──")
    print(f"  Total rows        : {total_before:,}")
    print(f"  Duplicate rows    : {total_dupes:,}  ({total_dupes / total_before * 100:.1f}%)")
    print(f"  Unique rows       : {unique_count:,}")

    per_platform = (
        df.groupby("platform")["is_duplicate"]
        .agg(total="count", duplicates="sum")
        .assign(pct=lambda x: (x["duplicates"] / x["total"] * 100).round(1))
    )
    print("\n  By platform:")
    print(per_platform.to_string())

    rows = [{"group": "TOTAL", "total": total_before,
             "duplicates": total_dupes,
             "pct_duplicate": round(total_dupes / total_before * 100, 1)}]
    for plat, row in per_platform.iterrows():
        rows.append({"group": plat, "total": int(row["total"]),
                     "duplicates": int(row["duplicates"]),
                     "pct_duplicate": row["pct"]})
    pd.DataFrame(rows).to_csv(DEDUP_OUT, index=False)

    df_clean = df[~df["is_duplicate"]].drop(columns=["_text_norm", "is_duplicate"]).copy()
    df_clean = df_clean.reset_index(drop=True)
    df_clean["row_id"] = range(1, len(df_clean) + 1)
    df_clean.to_parquet(ALL_OUT, index=False)
    print(f"\n  Saved ({len(df_clean):,} rows)  →  {ALL_OUT}")
    return df_clean


def _is_english(text):
    try:
        return detect(str(text)) == "en"
    except LangDetectException:
        return False


def run_language_filter():
    if not os.path.exists(ALL_OUT):
        raise SystemExit(f"Missing: {ALL_OUT} — run deduplication first.")
    df = pd.read_parquet(ALL_OUT)
    total_before = len(df)

    print("\n── Language filter ──")
    print(f"  Rows before : {total_before:,}")

    tqdm.pandas(desc="Detecting language")
    df["_is_english"] = df["text"].astype(str).progress_apply(_is_english)

    dropped = (~df["_is_english"]).sum()
    per_platform = (
        df.groupby("platform")["_is_english"]
        .agg(total="count", kept="sum")
        .assign(dropped=lambda x: x["total"] - x["kept"],
                pct_kept=lambda x: (x["kept"] / x["total"] * 100).round(1))
    )
    print(per_platform.to_string())
    print(f"\n  Dropped (non-English) : {dropped:,}  ({dropped / total_before * 100:.1f}%)")

    pd.DataFrame([{
        "total_before": total_before,
        "dropped": int(dropped),
        "kept": int(total_before - dropped),
        "pct_kept": round((total_before - dropped) / total_before * 100, 1),
    }]).to_csv(LANG_OUT, index=False)

    df_clean = df[df["_is_english"]].drop(columns=["_is_english"]).copy()
    df_clean = df_clean.reset_index(drop=True)
    df_clean["row_id"] = range(1, len(df_clean) + 1)
    df_clean.to_parquet(ALL_OUT, index=False)
    print(f"  Rows after  : {len(df_clean):,}  →  {ALL_OUT}")
    return df_clean


def main():
    run_deduplication()
    run_language_filter()
    print("\nDone. Next: python 03_sentiment.py")


if __name__ == "__main__":
    main()
