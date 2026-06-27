"""
Significance tests for the framing analysis (reproduces the chi-square
results reported in thesis §6.6).

Runs two families of tests, pro_israel vs pro_palestine (fine-tuned stance):
  1. framing-usage × stance   — is each framing category used at different
     rates by the two stance groups?
  2. sentiment × stance within each framing — within posts carrying a given
     framing, do the two groups differ in sentiment distribution?

All reported effects are p < 0.001, which clears a Bonferroni-corrected
threshold for the eight tests run, so no conclusion depends on the correction.

Not part of the numbered 00–10 pipeline. Run after 07_finetune.py has produced
messages_plus_all_plus_finetuned.parquet.
Run: python stats_significance_tests.py
"""

import pandas as pd
from scipy.stats import chi2_contingency

# load the framing parquet, merge in fine-tuned stance (same as 10_stance_reports.py)
df = pd.read_parquet("data_processed/messages_plus_sentiment_keywords.parquet")
ft = pd.read_parquet("data_processed/messages_plus_all_plus_finetuned.parquet")[["row_id","stance_finetuned"]]
df = df.merge(ft, on="row_id", how="left")

FLAGS = ["flag_death_context","flag_victim","flag_military","flag_legal"]
SC = "stance_finetuned"

# ── framing usage rate WITHIN each fine-tuned stance (matches your plot) ──
print("=== framing usage by fine-tuned stance (denominator = each stance class) ===")
for stance in ["pro_israel","pro_palestine","neutral"]:
    sub = df[df[SC]==stance]
    rates = {f: f"{sub[f].mean()*100:.1f}%" for f in FLAGS}
    print(f"  {stance} (n={len(sub)}): {rates}")

# ── chi-square: framing presence × fine-tuned stance, PP vs PI only ──
print("\n=== significance: framing × stance (pro_israel vs pro_palestine) ===")
two = df[df[SC].isin(["pro_israel","pro_palestine"])]
for f in FLAGS:
    table = pd.crosstab(two[SC], two[f])
    chi2,p,dof,_ = chi2_contingency(table)
    sig = "p<0.001" if p<0.001 else ("p<0.01" if p<0.01 else ("p<0.05" if p<0.05 else f"p={p:.3f} NS"))
    print(f"  {f}: chi2={chi2:.1f}  {sig}")

# ── chi-square: sentiment × stance WITHIN each framing (PP vs PI) ──
print("\n=== significance: sentiment × stance within each framing (PP vs PI) ===")
for f in FLAGS:
    sub = two[two[f]]
    table = pd.crosstab(sub[SC], sub["sentiment_label"])
    chi2,p,dof,_ = chi2_contingency(table)
    sig = "p<0.001" if p<0.001 else ("p<0.01" if p<0.01 else ("p<0.05" if p<0.05 else f"p={p:.3f} NS"))
    print(f"  {f}: chi2={chi2:.1f}  {sig}")