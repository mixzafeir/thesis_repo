"""
Fine-tune a stance classifier on manual labels — with 5-fold stratified CV.

Run after:
  05_make_validation_sample.py
Run: python finetune.py

What it does:
  • 5-fold stratified cross-validation on the labeled rows
    -> reports F1_macro and accuracy as  mean ± std  (this is your headline number)
  • collects out-of-fold (OOF) predictions: every labeled row gets a prediction
    from a model that did NOT see it in training -> leakage-free, for compare.py
  • trains a FINAL model on ALL labeled rows and classifies the full dataset
  • saves a parquet with column 'stance_finetuned'
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
from torch.utils.data import Dataset

PROC_DIR      = "data_processed"
VALID_CSV     = os.path.join(PROC_DIR, "validation_sample_1000.csv")
STANCE_OUT    = os.path.join(PROC_DIR, "messages_plus_sentiment_keywords.parquet")
FINETUNE_OUT  = os.path.join(PROC_DIR, "messages_plus_all_plus_finetuned.parquet")
OOF_OUT       = os.path.join(PROC_DIR, "finetuned_oof_predictions.csv")
MODEL_SAVE    = os.path.join(PROC_DIR, "finetuned_stance_model")

BASE_MODEL    = "vinai/bertweet-base"
MAX_LEN       = 128
EPOCHS        = 3
BATCH_SIZE    = 16
INFER_BATCH   = 512
N_FOLDS       = 5          # set to 3 if you want it to run faster
SEED          = 32         # seed for the fold split (fixed = reproducible)
TRAIN_SEED    = 42         # fixed across all folds: the only difference left is the split

LABEL2ID      = {"pro_palestine": 0, "pro_israel": 1, "neutral": 2}
ID2LABEL      = {0: "pro_palestine", 1: "pro_israel", 2: "neutral"}


class StanceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def encode(tokenizer, texts):
    return tokenizer(
        list(texts),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )


def fresh_model():
    return AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )


def train_args(output_dir):
    # eval/save off: within folds we train a fixed 3 epochs and evaluate manually
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=20,
        seed=TRAIN_SEED,
        report_to="none",
    )


def cleanup(*objs):
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    # ── 1. Load labeled data ────────────────────────────────────────────────
    if not os.path.exists(VALID_CSV):
        raise SystemExit(f"Missing: {VALID_CSV}\nRun 05_make_validation_sample.py first")

    df_lab = pd.read_csv(VALID_CSV)
    df_lab["manual_label"] = df_lab["manual_label"].astype(str).str.strip()
    df_lab = df_lab[df_lab["manual_label"].isin(LABEL2ID)].copy().reset_index(drop=True)

    if len(df_lab) < 50:
        raise SystemExit(
            f"Only {len(df_lab)} labeled rows found in manual_label column.\n"
            "Fill in more labels before fine-tuning (aim for 800+)."
        )

    print(f"Labeled rows available: {len(df_lab)}")
    print(df_lab["manual_label"].value_counts().to_string())

    texts  = df_lab["text"].fillna("").tolist()
    labels = [LABEL2ID[l] for l in df_lab["manual_label"]]
    y      = np.array(labels)

    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/bertweet-base", use_fast=False, normalization=True
    )

    # ── 2. 5-fold stratified cross-validation ───────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_f1, fold_acc = [], []
    oof_pred = np.full(len(df_lab), -1, dtype=int)   # out-of-fold predictions

    for fold, (tr_idx, te_idx) in enumerate(skf.split(texts, y), start=1):
        print(f"\n──────── Fold {fold}/{N_FOLDS} ────────")
        tr_enc = encode(tokenizer, [texts[i] for i in tr_idx])
        te_enc = encode(tokenizer, [texts[i] for i in te_idx])
        tr_ds  = StanceDataset(tr_enc, [labels[i] for i in tr_idx])
        te_ds  = StanceDataset(te_enc, [labels[i] for i in te_idx])

        model   = fresh_model()
        trainer = Trainer(
            model=model,
            args=train_args(os.path.join(MODEL_SAVE, f"_cv_fold{fold}")),
            train_dataset=tr_ds,
        )
        trainer.train()

        out   = trainer.predict(te_ds)
        preds = np.argmax(out.predictions, axis=-1)

        rep = classification_report(
            [labels[i] for i in te_idx], preds,
            target_names=list(LABEL2ID.keys()),
            output_dict=True, zero_division=0,
        )
        fold_f1.append(rep["macro avg"]["f1-score"])
        fold_acc.append(rep["accuracy"])
        oof_pred[te_idx] = preds
        print(f"Fold {fold}:  F1_macro={fold_f1[-1]:.3f}   accuracy={fold_acc[-1]:.3f}")

        cleanup(model, trainer)

    # ── 3. Report CV result (the number for the thesis) ─────────────────────
    print("\n" + "═" * 48)
    print("CROSS-VALIDATION RESULT  ({}-fold stratified)".format(N_FOLDS))
    print("═" * 48)
    print(f"F1 macro : {np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}")
    print(f"Accuracy : {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}")
    print(f"per-fold F1 : {[round(f, 3) for f in fold_f1]}")

    print("\nAggregated out-of-fold report (leakage-free, all labeled rows):")
    print(classification_report(
        y, oof_pred, target_names=list(LABEL2ID.keys()), zero_division=0
    ))

    # OOF predictions per row_id -> feed these to compare.py for a fair comparison
    df_lab["stance_finetuned_oof"] = [ID2LABEL[p] for p in oof_pred]
    df_lab[["row_id", "stance_finetuned_oof"]].to_csv(OOF_OUT, index=False)
    print(f"OOF predictions saved -> {OOF_OUT}")

    # ── 4. Final model on ALL labeled rows ──────────────────────────────────
    print("\nTraining FINAL model on all labeled rows ...")
    all_enc = encode(tokenizer, texts)
    all_ds  = StanceDataset(all_enc, labels)
    model   = fresh_model()
    trainer = Trainer(model=model, args=train_args(MODEL_SAVE), train_dataset=all_ds)
    trainer.train()

    # ── 5. Classify full dataset ────────────────────────────────────────────
    if not os.path.exists(STANCE_OUT):
        raise SystemExit(f"Missing: {STANCE_OUT}\nRun 04_keywords.py first.")

    df_full = pd.read_parquet(STANCE_OUT)
    print(f"\nClassifying full dataset ({len(df_full):,} rows) ...")

    all_texts = df_full["text"].fillna("").tolist()
    all_preds = []

    model.eval()
    infer_device = next(model.parameters()).device
    for i in range(0, len(all_texts), INFER_BATCH):
        batch = all_texts[i : i + INFER_BATCH]
        enc   = encode(tokenizer, batch)
        enc_t = {k: torch.tensor(v).to(infer_device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc_t).logits
        batch_preds = torch.argmax(logits, dim=-1).tolist()
        all_preds.extend([ID2LABEL[p] for p in batch_preds])
        if (i // INFER_BATCH) % 10 == 0:
            print(f"  {i + len(batch):,} / {len(all_texts):,}")

    df_full["stance_finetuned"] = all_preds

    # For the labeled rows: replace with the OOF predictions (no in-sample optimism)
    if "row_id" in df_full.columns:
        oof_map = dict(zip(df_lab["row_id"], df_lab["stance_finetuned_oof"]))
        mask = df_full["row_id"].isin(oof_map)
        df_full.loc[mask, "stance_finetuned"] = df_full.loc[mask, "row_id"].map(oof_map)
        print(f"Overwrote {int(mask.sum()):,} labeled rows with OOF predictions.")

    dist = df_full["stance_finetuned"].value_counts()
    print("\nFine-tuned stance distribution:")
    print(dist.to_string())

    df_full.to_parquet(FINETUNE_OUT, index=False)
    print(f"\nSaved -> {FINETUNE_OUT}")
    trainer.save_model(MODEL_SAVE)
    print(f"Model saved -> {MODEL_SAVE}")


if __name__ == "__main__":
    main()