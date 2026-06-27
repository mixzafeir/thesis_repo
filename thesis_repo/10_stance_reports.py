# 10_stance_reports.py — Stance reports for all 3 methods + comparisons
# Run: python 10_stance_reports.py
# Depends on: 04_keywords.py, 06_zeroshot.py, 07_finetune.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROC_DIR      = "data_processed"
PLOTS_DIR     = os.path.join(PROC_DIR, "plots", "stance")
PLOTS_METHODS = os.path.join(PLOTS_DIR, "methods")
PLOTS_FRAMING = os.path.join(PLOTS_DIR, "framing")
DEDUP_OUT  = os.path.join(PROC_DIR, "dedup_stats.csv")
EXCEL_OUT  = os.path.join(PROC_DIR, "results_tables.xlsx")
CSV_OUT    = os.path.join(PROC_DIR, "messages_all_three_methods.csv")

KW_PARQUET = os.path.join(PROC_DIR, "messages_plus_sentiment_keywords.parquet")
ZS_PARQUET = os.path.join(PROC_DIR, "messages_clean_plus_zeroshot.parquet")
FT_PARQUET = os.path.join(PROC_DIR, "messages_plus_all_plus_finetuned.parquet")

METHODS = [
    ("stance_label",    "Keyword"),
    ("stance_zeroshot", "Zero-shot"),
    ("stance_finetuned","Fine-tuned"),
]

STANCES       = ["pro_palestine", "pro_israel", "neutral"]
STANCE_LABELS = ["Pro-Palestine", "Pro-Israel", "Neutral"]
STANCE_COLORS = ["#2ca02c", "#1f77b4", "#7f7f7f"]

DEATH_CONTEXT = {
    "killed", "dead", "casualties", "martyr", "martyred", "eliminated",
    "airstrike", "bombing", "massacre", "civilian deaths", "bodies",
    "wounded", "died", "death toll", "fatalities", "slain", "murdered",
    "executions", "corpses", "genocide victims", "body bags", "dead bodies",
}
VICTIM_FRAMING = {
    "civilians", "children", "child", "kids", "babies", "infant",
    "hospitals", "hospital", "refugees", "refugee camp", "displaced",
    "innocent", "families", "women", "elderly", "humanitarian",
    "aid workers", "medics", "doctors", "nurses", "patients",
}
MILITARY_FRAMING = {
    "operation", "strike", "targeted", "eliminated", "neutralized",
    "airstrike", "precision strike", "surgical strike", "ground offensive",
    "troops", "soldiers", "idf", "forces", "destroyed", "dismantled",
    "launched", "fired", "offensive", "combat", "mission",
}
LEGAL_FRAMING = {
    "war crimes", "icc", "icj", "genocide", "international law",
    "humanitarian law", "geneva convention", "war criminal",
    "accountability", "arrest warrant", "violations", "unlawful",
    "illegal", "international court", "human rights", "amnesty",
    "un resolution", "security council", "ceasefire resolution",
}

FRAMING_FLAGS = [
    ("flag_death_context", DEATH_CONTEXT,    "Death context"),
    ("flag_victim",        VICTIM_FRAMING,   "Victim framing"),
    ("flag_military",      MILITARY_FRAMING, "Military framing"),
    ("flag_legal",         LEGAL_FRAMING,    "Legal/intl law framing"),
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _clean(s):
    import re
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def has_any(text, terms):
    t = _clean(text)
    return any(term in t for term in terms)


def load_data():
    missing = []
    for path in [KW_PARQUET, ZS_PARQUET, FT_PARQUET]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise SystemExit("Missing parquets:\n" + "\n".join(missing))

    df_kw = pd.read_parquet(KW_PARQUET)
    df_zs = pd.read_parquet(ZS_PARQUET)[["row_id", "stance_zeroshot"]]
    df_ft = pd.read_parquet(FT_PARQUET)[["row_id", "stance_finetuned"]]

    df = df_kw.merge(df_zs, on="row_id", how="left")
    df = df.merge(df_ft, on="row_id", how="left")
    print(f"Loaded {len(df):,} rows")
    return df


# ── Framing flags ─────────────────────────────────────────────────────────────
def ensure_framing_flags(df):
    from tqdm import tqdm
    for flag, terms, label in FRAMING_FLAGS:
        if flag not in df.columns:
            tqdm.pandas(desc=f"Flagging {label}")
            df[flag] = df["text"].progress_apply(lambda x, t=terms: has_any(x, t))
    return df


# ── 1. Stance distribution — all 3 methods ───────────────────────────────────
def plot_distribution(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle("Stance distribution — all methods", fontsize=14, fontweight="bold")
    for ax, (col, label) in zip(axes, METHODS):
        counts = df[col].value_counts().reindex(STANCES, fill_value=0)
        pcts   = counts / counts.sum() * 100
        bars   = ax.bar(STANCE_LABELS, pcts.values, color=STANCE_COLORS, edgecolor="white")
        ax.set_title(label, fontsize=12)
        ax.set_ylabel("% of posts")
        ax.set_ylim(0, 75)
        ax.tick_params(axis="x", rotation=15)
        for bar, pct in zip(bars, pcts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_METHODS, "stance_distribution_all_methods.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: stance_distribution_all_methods.png")


# ── 2. Stance distribution grouped bar ───────────────────────────────────────
def plot_distribution_grouped(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    x     = np.arange(len(STANCES))
    width = 0.25
    for i, (col, label) in enumerate(METHODS):
        counts = df[col].value_counts().reindex(STANCES, fill_value=0)
        pcts   = counts / counts.sum() * 100
        bars   = ax.bar(x + i * width, pcts.values, width, label=label)
        for bar, pct in zip(bars, pcts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(STANCE_LABELS)
    ax.set_ylabel("% of posts")
    ax.set_ylim(0, 75)
    ax.set_title("Stance distribution by method", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_METHODS, "stance_distribution_grouped.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: stance_distribution_grouped.png")


# ── 3. Stance over time — 3 subplots ─────────────────────────────────────────
def plot_over_time(df):
    if "created_at" not in df.columns:
        print("Skipped over-time plot: no created_at column")
        return
    df_time = df.copy()
    df_time["created_at"] = pd.to_datetime(df_time["created_at"], errors="coerce", utc=True)
    df_time = df_time.dropna(subset=["created_at"])
    if len(df_time) == 0:
        print("Skipped over-time plot: no parseable dates")
        return
    df_time["month"] = df_time["created_at"].dt.to_period("M")

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    fig.suptitle("Stance over time — all methods (monthly %)", fontsize=14, fontweight="bold")
    for ax, (col, label) in zip(axes, METHODS):
        ct  = pd.crosstab(df_time["month"], df_time[col])
        pct = ct.reindex(columns=STANCES, fill_value=0).div(ct.sum(axis=1), axis=0).mul(100)
        for stance, color, slabel in zip(STANCES, STANCE_COLORS, STANCE_LABELS):
            if stance in pct.columns:
                ax.plot(pct.index.astype(str), pct[stance], marker="o",
                        markersize=3, label=slabel, color=color)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("% of posts")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_METHODS, "stance_over_time_all_methods.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: stance_over_time_all_methods.png")


# ── 4. Stance by top Telegram channels — 3 subplots ──────────────────────────
def plot_by_channel(df):
    tg = df[df["platform"] == "telegram"].copy()
    if len(tg) == 0:
        print("Skipped channel plot: no telegram rows")
        return
    top_channels = tg["source"].value_counts().head(15).index
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    fig.suptitle("Stance by channel — top 15 Telegram channels (%)", fontsize=14, fontweight="bold")
    for ax, (col, label) in zip(axes, METHODS):
        tg_top = tg[tg["source"].isin(top_channels)].copy()
        ct     = pd.crosstab(tg_top["source"], tg_top[col])
        ct_pct = ct.reindex(columns=STANCES, fill_value=0).div(ct.sum(axis=1), axis=0).mul(100)
        ct_pct = ct_pct.sort_values("pro_palestine", ascending=True)
        ct_pct.plot(kind="barh", stacked=True, ax=ax,
                    color=STANCE_COLORS, legend=(ax == axes[0]))
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("% of messages")
        if ax == axes[0]:
            ax.legend(STANCE_LABELS, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_METHODS, "stance_by_channel_all_methods.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: stance_by_channel_all_methods.png")


# ── 5. Sentiment by stance — 3 subplots ──────────────────────────────────────
def plot_sentiment_by_stance(df):
    sentiments  = ["negative", "neutral", "positive"]
    sent_labels = ["Negative", "Neutral", "Positive"]
    sent_colors = ["tomato", "lightgray", "steelblue"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Sentiment distribution within each stance — by method",
                 fontsize=13, fontweight="bold")
    for ax, (col, label) in zip(axes, METHODS):
        ct  = pd.crosstab(df[col], df["sentiment_label"])
        pct = ct.reindex(index=STANCES, columns=sentiments, fill_value=0)
        pct = pct.div(pct.sum(axis=1), axis=0).mul(100)
        x     = np.arange(len(STANCES))
        width = 0.25
        for i, (sent, color, slabel) in enumerate(zip(sentiments, sent_colors, sent_labels)):
            ax.bar(x + i * width, pct[sent].values, width, label=slabel, color=color)
        ax.set_xticks(x + width)
        ax.set_xticklabels(STANCE_LABELS, rotation=10)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("% of posts")
        ax.set_ylim(0, 100)
        if ax == axes[0]:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_METHODS, "sentiment_by_stance_all_methods.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: sentiment_by_stance_all_methods.png")


# ── 6. Method agreement matrix ────────────────────────────────────────────────
def plot_method_agreement(df):
    cols  = [col for col, _ in METHODS]
    labs  = [lab for _, lab in METHODS]
    pairs = [
        (cols[0], cols[1], f"{labs[0]} vs {labs[1]}"),
        (cols[0], cols[2], f"{labs[0]} vs {labs[2]}"),
        (cols[1], cols[2], f"{labs[1]} vs {labs[2]}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Method agreement — where do they agree/disagree?",
                 fontsize=13, fontweight="bold")
    for ax, (c1, c2, title) in zip(axes, pairs):
        ct  = pd.crosstab(df[c1], df[c2])
        ct  = ct.reindex(index=STANCES, columns=STANCES, fill_value=0)
        pct = ct.div(ct.sum(axis=1), axis=0).mul(100)
        ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(STANCES)))
        ax.set_yticks(range(len(STANCES)))
        ax.set_xticklabels(STANCE_LABELS, rotation=20, ha="right", fontsize=9)
        ax.set_yticklabels(STANCE_LABELS, fontsize=9)
        ax.set_xlabel("Predicted →", fontsize=9)
        ax.set_ylabel("← Reference", fontsize=9)
        for i in range(len(STANCES)):
            for j in range(len(STANCES)):
                ax.text(j, i, f"{pct.values[i, j]:.1f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if pct.values[i, j] > 55 else "black")
        agree = (df[c1] == df[c2]).mean() * 100
        ax.set_title(f"{title}\n(overall agreement: {agree:.1f}%)", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_METHODS, "method_agreement.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: method_agreement.png")


# ── 7. Framing plots (framing flags are independent; broken down by fine-tuned stance)
def plot_framing(df):
    stances     = ["pro_palestine", "pro_israel", "neutral"]
    sentiments  = ["negative", "neutral", "positive"]
    sent_colors = ["tomato", "lightgray", "steelblue"]
    stance_colors = ["green", "royalblue", "gray"]
    stance_col  = "stance_finetuned"

    # Sentiment within each framing type, by stance (fine-tuned stance)
    for flag, _, flag_label in FRAMING_FLAGS:
        flagged = df[df[flag]].copy()
        if len(flagged) < 10:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(stances))
        width = 0.25
        for i, (sent, color) in enumerate(zip(sentiments, sent_colors)):
            vals = []
            for stance in stances:
                sub = flagged[flagged[stance_col] == stance]
                vals.append((sub["sentiment_label"] == sent).mean() * 100 if len(sub) else 0)
            ax.bar(x + i * width, vals, width, label=sent, color=color)
        ax.set_xticks(x + width)
        ax.set_xticklabels(stances)
        ax.set_ylabel("% of posts")
        ax.set_ylim(0, 100)
        ax.set_title(f"Sentiment within '{flag_label}' posts — by stance")
        ax.legend()
        plt.tight_layout()
        fname = f"framing_{flag.replace('flag_', '')}_sentiment.png"
        plt.savefig(os.path.join(PLOTS_FRAMING, fname), dpi=200)
        plt.close()
        print(f"Saved: {fname}")

    # Positive sentiment rate in death-context posts by stance
    flagged_death = df[df["flag_death_context"]].copy()
    pos_rates = []
    for stance in stances:
        sub = flagged_death[flagged_death[stance_col] == stance]
        pos_rates.append((sub["sentiment_label"] == "positive").mean() * 100 if len(sub) else 0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(stances, pos_rates, color=stance_colors)
    ax.set_ylabel("% positive sentiment")
    ax.set_title("Positive sentiment in death-context posts — by stance")
    ax.set_ylim(0, max(pos_rates) * 1.4 + 1)
    for i, v in enumerate(pos_rates):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FRAMING, "death_context_positive_rate.png"), dpi=200)
    plt.close()
    print("Saved: death_context_positive_rate.png")

    # Framing type usage rate by stance
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(FRAMING_FLAGS))
    width = 0.25
    for i, stance in enumerate(stances):
        sub   = df[df[stance_col] == stance]
        rates = [sub[flag].mean() * 100 for flag, _, _ in FRAMING_FLAGS]
        ax.bar(x + i * width, rates, width, label=stance, color=stance_colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels([lbl for _, _, lbl in FRAMING_FLAGS], rotation=15, ha="right")
    ax.set_ylabel("% of posts using framing")
    ax.set_title("Framing type usage rate by stance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FRAMING, "framing_usage_by_stance.png"), dpi=200)
    plt.close()
    print("Saved: framing_usage_by_stance.png")

    # Emotional intensity by stance
    fig, ax = plt.subplots(figsize=(7, 5))
    intensity = df.groupby(stance_col)["sentiment_score"].mean().reindex(stances)
    ax.bar(stances, intensity.values, color=stance_colors)
    ax.set_ylabel("Avg sentiment confidence score")
    ax.set_title("Emotional intensity by stance (avg sentiment score)")
    ax.set_ylim(0, 1)
    for i, v in enumerate(intensity.values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FRAMING, "emotional_intensity_by_stance.png"), dpi=200)
    plt.close()
    print("Saved: emotional_intensity_by_stance.png")


# ── 8. Deduplication chart ────────────────────────────────────────────────────
def plot_deduplication():
    if not os.path.exists(DEDUP_OUT):
        print("Skipped deduplication chart: dedup_stats.csv not found")
        return
    dedup = pd.read_csv(DEDUP_OUT)
    dedup = dedup[dedup["group"] != "TOTAL"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    dedup["unique"] = dedup["total"] - dedup["duplicates"]
    dedup.set_index("group")[["unique", "duplicates"]].plot(
        kind="bar", stacked=True, color=["steelblue", "tomato"], ax=axes[0], rot=0,
    )
    axes[0].set_title("Unique vs duplicate messages by platform")
    axes[0].set_ylabel("Message count")
    axes[0].legend(["Unique", "Duplicate"])
    axes[1].bar(dedup["group"], dedup["pct_duplicate"], color="tomato")
    axes[1].set_title("Duplicate rate by platform (%)")
    axes[1].set_ylabel("% duplicate")
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(dedup["pct_duplicate"]):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    plt.suptitle("Coordinated message repetition analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "deduplication_by_platform.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: deduplication_by_platform.png")


# ── 9. Excel export ───────────────────────────────────────────────────────────
def export_excel(df):
    def pct(ct):
        return (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)

    total = len(df)
    pc    = df["platform"].value_counts().to_frame("count")
    pc["pct"] = (pc["count"] / total * 100).round(1)

    tg  = df[df["platform"] == "telegram"]
    top = tg["source"].value_counts().head(15).to_frame("count")
    if len(tg):
        top["pct_of_telegram"] = (top["count"] / len(tg) * 100).round(1)

    with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as w:
        pc.to_excel(w, sheet_name="Platform_counts")
        for col, label in METHODS:
            sc  = df[col].value_counts().to_frame("count")
            sc["pct"] = (sc["count"] / total * 100).round(1)
            sbp = pd.crosstab(df["platform"], df[col])
            sbs = pd.crosstab(df[col], df["sentiment_label"])
            sc.to_excel(w,       sheet_name=f"{label}_stance_counts")
            sbp.to_excel(w,      sheet_name=f"{label}_by_platform_counts")
            pct(sbp).to_excel(w, sheet_name=f"{label}_by_platform_pct")
            sbs.to_excel(w,      sheet_name=f"{label}_sentiment_counts")
            pct(sbs).to_excel(w, sheet_name=f"{label}_sentiment_pct")
        top.to_excel(w, sheet_name="Top_Telegram_Channels")
        if os.path.exists(DEDUP_OUT):
            pd.read_csv(DEDUP_OUT).to_excel(w, sheet_name="Deduplication_stats", index=False)
    print("Saved: results_tables.xlsx")


# ── 10. CSV export ────────────────────────────────────────────────────────────
def export_csv(df):
    keep = ["platform", "source", "post_id", "created_at", "text",
            "sentiment_label", "sentiment_score",
            "stance_label", "stance_score",
            "stance_zeroshot", "stance_finetuned"]
    df[[c for c in keep if c in df.columns]].to_csv(CSV_OUT, index=False, encoding="utf-8")
    print("Saved: messages_all_three_methods.csv")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(df):
    print("\n── Stance distribution summary ──")
    rows = []
    for col, label in METHODS:
        counts = df[col].value_counts().reindex(STANCES, fill_value=0)
        pcts   = counts / counts.sum() * 100
        rows.append({
            "Method":        label,
            "Pro-Palestine": f"{counts['pro_palestine']:,} ({pcts['pro_palestine']:.1f}%)",
            "Pro-Israel":    f"{counts['pro_israel']:,} ({pcts['pro_israel']:.1f}%)",
            "Neutral":       f"{counts['neutral']:,} ({pcts['neutral']:.1f}%)",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n── Pairwise agreement ──")
    cols = [col for col, _ in METHODS]
    labs = [lab for _, lab in METHODS]
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        agree = (df[cols[i]] == df[cols[j]]).mean() * 100
        print(f"  {labs[i]} vs {labs[j]}: {agree:.1f}% agreement")


def main():
    ensure_dir(PLOTS_DIR)
    ensure_dir(PLOTS_METHODS)
    ensure_dir(PLOTS_FRAMING)
    df = load_data()
    df = ensure_framing_flags(df)
    print_summary(df)
    plot_distribution(df)
    plot_distribution_grouped(df)
    plot_over_time(df)
    plot_by_channel(df)
    plot_sentiment_by_stance(df)
    plot_method_agreement(df)
    plot_framing(df)
    plot_deduplication()
    export_excel(df)
    export_csv(df)
    print("\nDone. All plots saved to", PLOTS_DIR)


if __name__ == "__main__":
    main()
