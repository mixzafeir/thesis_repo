# A Telegram Analysis of Discourse on the Israel-Palestine Conflict

Multi-method computational analysis of pro-Israel and pro-Palestine Telegram
channels, combining sentiment analysis, three stance-detection methods (keyword,
zero-shot DeBERTa, fine-tuned BERTweet), and framing analysis.

This repository contains the analysis pipeline accompanying the MSc thesis of the
same title. The pipeline collects messages from Telegram channels, cleans them,
and runs sentiment, stance, and framing analysis, evaluating the three stance
methods against a manually annotated ground-truth set of 736 messages.

---

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Telegram credentials**

   Copy the example config and fill in your own values:

   ```bash
   cp config.example.py config.py
   ```

   Then edit `config.py` with your Telegram API credentials (obtain these from
   https://my.telegram.org). `config.py` is gitignored and must never be
   committed.

---

## Pipeline

Run the scripts in numerical order. Each step reads the output of a previous
step and writes a new file into `data_processed/`.

| Step | Script | Reads | Writes |
|------|--------|-------|--------|
| 00 | `00_crawler.py` | Telegram API | `data_raw/*.json` |
| 01 | `01_collect.py` | `data_raw/*.json` | `messages_clean.parquet` |
| 02 | `02_clean.py` | `messages_clean.parquet` | `messages_clean.parquet` (dedup + language filter) |
| 03 | `03_sentiment.py` | `messages_clean.parquet` | `messages_clean_plus_sentiment.parquet` |
| 04 | `04_keywords.py` | `messages_clean_plus_sentiment.parquet` | `messages_plus_sentiment_keywords.parquet` |
| 05 | `05_make_validation_sample.py` | `messages_plus_sentiment_keywords.parquet` | `validation_sample_1000.csv` (blank labels) |
| —  | *manual labeling* | — | you fill the `manual_label` column |
| 06 | `06_zeroshot.py` | `messages_clean.parquet` | `messages_clean_plus_zeroshot.parquet` |
| 07 | `07_finetune.py` | `validation_sample_1000.csv` + `messages_plus_sentiment_keywords.parquet` | `messages_plus_all_plus_finetuned.parquet`, `finetuned_oof_predictions.csv` |
| 08 | `08_compare.py` | all 3 stance parquets + manual labels | `comparison_report.xlsx` + plots |
| 09 | `09_sentiment_reports.py` | `messages_clean_plus_sentiment.parquet` | sentiment plots |
| 10 | `10_stance_reports.py` | all 3 stance parquets | `results_tables.xlsx`, `messages_all_three_methods.csv` + plots |

### Pipeline structure

The three stance methods do not run in a single linear chain. The **keyword** and
**fine-tuned** methods build on the sentiment-annotated parquet, while the
**zero-shot** method reads the clean base parquet independently. This keeps the
expensive zero-shot step isolated, so it does not need to re-run when an
unrelated upstream step changes:

```
01 collect -> 02 clean -> messages_clean.parquet
                              |
                              +-> 03 sentiment -> 04 keyword -> 07 finetune
                              |     (messages_clean_plus_sentiment ->
                              |      messages_plus_sentiment_keywords ->
                              |      messages_plus_all_plus_finetuned)
                              |
                              +-> 06 zeroshot
                                    (messages_clean_plus_zeroshot)
                                          |
              10 reports: merge all three on row_id -> messages_all_three_methods.csv
```

All three methods classify the **same messages** (joined on `row_id`); they are
assembled into separate intermediate parquets and merged at the reporting stage.

### Intermediate files

| File | Contents |
|------|----------|
| `messages_clean.parquet` | all messages after collection, deduplication, and language filtering (the clean base everything builds on) |
| `messages_clean_plus_sentiment.parquet` | clean base + `sentiment_label`, `sentiment_score` |
| `messages_plus_sentiment_keywords.parquet` | + `stance_label`, `stance_score`, framing flags (keyword method) |
| `messages_clean_plus_zeroshot.parquet` | clean base + `stance_zeroshot` |
| `messages_plus_all_plus_finetuned.parquet` | keyword parquet + `stance_finetuned` |
| `messages_all_three_methods.csv` | every message with all three stance labels side by side (final merged output) |

### Where the comparison happens

- **`08_compare.py`** produces the evaluation numbers (accuracy, macro F1,
  per-class metrics, confusion matrices) by comparing all three methods against
  the 736 manually labeled messages. This is the source of the headline results.
- **`10_stance_reports.py`** merges all three stance parquets on `row_id` into a
  single file (`messages_all_three_methods.csv`) containing every message with
  all three stance labels side by side, and produces the full-dataset
  distribution and framing plots.

---

## Method summary

- **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest`, applied to
  the full corpus.
- **Keyword stance:** two manually constructed weighted lexicons (pro-Palestine
  and pro-Israel), plus four keyword-based framing categories (death context,
  victim, military, legal).
- **Zero-shot stance:** `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`, applied via
  natural-language inference with the three stance labels as hypotheses.
- **Fine-tuned stance:** `vinai/bertweet-base`, fine-tuned on the 736 manually
  labeled messages with 5-fold stratified cross-validation. Evaluation uses
  leakage-free out-of-fold predictions.

---

## Utilities (not part of the numbered pipeline)

- `stats_significance_tests.py` — reproduces the chi-square significance tests
  reported in §6.6 (framing-usage × stance, and sentiment × stance within each
  framing category). Run after `07_finetune.py`.

## Notes on outputs

Generated files (parquets, CSVs, plots, `.xlsx` reports) are written to
`data_processed/` and are **not** committed to the repository — they are
regenerated by running the pipeline. Raw scraped messages in `data_raw/` are
likewise not committed.

The event-analysis step (`08_compare.py`) processes all events in its `EVENTS`
list and automatically skips any event with too few messages in its time window;
with a larger corpus, additional events will produce output.

---

## Requirements

See `requirements.txt`. Core dependencies: `transformers`, `torch`, `pandas`,
`numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `telethon`, `langdetect`,
`openpyxl`, `tqdm`, `pyarrow`.
