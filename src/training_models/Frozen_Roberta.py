# ==============================================================================
# SCRIPT FOR TRAINING A "BAD" CLASSIFIER: FROZEN-BASE ROBERTA
# ------------------------------------------------------------------------------
# This script trains a RoBERTa model for sequence classification but with a
# crucial handicap: the entire pre-trained RoBERTa base is "frozen". This means
# only the newly added, randomly initialized classification layer on top is
# trained.
#
# WHY THIS IS A "BAD" CLASSIFIER FOR XAI COMPARISON:
# 1. No Fine-Tuning: The model cannot adapt its deep, contextual understanding
#    of language to the specific nuances of your emotion dataset.
# 2. Static Features: It uses the generic, pre-trained RoBERTa model as a
#    fixed feature extractor. This is much less powerful than end-to-end
#    fine-tuning where the entire model learns from the data.
# 3. Simpler Learning: The model only learns a simple mapping from RoBERTa's
#    general-purpose sentence embeddings to your emotion labels.
# ==============================================================================

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)
from evaluate import load

# ============================================================
# 1️⃣ Load and clean data
# ============================================================

Train = '../data/train_split.csv'
print("🔹 Loading dataset...")

try:
    df = pd.read_csv(Train)
except FileNotFoundError:
    print("\n❌ ERROR: 'merged_emotions_dataset.csv' not found.")
    print("Please make sure the dataset file is in the same directory as this script.")
    exit()

print(f"✅ Loaded {len(df)} samples with columns: {df.columns.tolist()}")
original_len = len(df)
df = df[df['label'] != 'label'].copy()
print(f"✅ Removed {original_len - len(df)} bad header row(s).")

# Clean and unify labels
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[["text", "label"]].dropna() # Ensure no missing values

# Define unified label set dynamically
unique_labels = sorted(df["label"].unique().tolist())
print(f"\n✅ Found {len(unique_labels)} unique labels:")
print(unique_labels)

# Map labels to IDs
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# ============================================================
# 2️⃣ Split train / val / test (80 / 10 / 10)
# ============================================================
print("\n🔹 Splitting dataset (80% train, 10% val, 10% test)...")
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42)
print(f"✅ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# ============================================================
# 3️⃣ Tokenizer setup
# ============================================================
model_name = "roberta-base"
print(f"\n🔹 Loading tokenizer: {model_name}")
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

print("🔹 Tokenizing full datasets...")
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
print("✅ Tokenization complete.")

print("\n🔹 Renaming 'label_id' column to 'labels' for the Trainer...")
train_dataset = train_dataset.rename_column("label_id", "labels")
val_dataset = val_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")
print("✅ Column renamed.")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ============================================================
# 4️⃣ Model setup (The crucial difference is here!)
# ============================================================
print(f"\n🔹 Loading model for freezing: {model_name}")
model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# -------------------------------------------------------------
# ❗️ CRUCIAL STEP: FREEZE THE BASE MODEL'S PARAMETERS ❗️
# We iterate through all the parameters of the `roberta` base
# and set `requires_grad` to `False`. This prevents them from
# being updated during training. Only the classifier head will train.
# -------------------------------------------------------------
print("🔹 Freezing all layers in the RoBERTa base model...")
for param in model.roberta.parameters():
    param.requires_grad = False
print("✅ Base model frozen. Only the classifier head is trainable.")


# ============================================================
# 5️⃣ Metrics setup
# ============================================================
accuracy = load("accuracy")
f1 = load("f1")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_macro = f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1_macro": f1_macro["f1"]}

# ============================================================
# 6️⃣ Training arguments (Adjusted for frozen model)
# ============================================================
print("\n🔹 Preparing training arguments for frozen model...")
# Note: Hyperparameters are adjusted because we are only training a small
# classification head, not fine-tuning the entire large model.
training_args = TrainingArguments(
    output_dir="models/roberta-frozen-emotions",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,  # Higher LR is okay for a small head
    per_device_train_batch_size=32, # Can use a larger batch size
    per_device_eval_batch_size=32,
    num_train_epochs=5,  # Train for a few more epochs
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=torch.cuda.is_available(),
    report_to="none", # Disables wandb/etc. reporting
)
print("✅ Training arguments ready.")

# ============================================================
# 7️⃣ Trainer initialization
# ============================================================
print("\n🔹 Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print("✅ Trainer initialized.")

# ============================================================
# 8️⃣ Train and evaluate
# ============================================================
print("\n🚀 Starting training for the frozen model...")
trainer.train()
print("\n✅ Frozen model training complete.")

print("\n🎯 Evaluating the best frozen model on the unseen test set...")
test_results = trainer.evaluate(test_dataset)
print("\n--- Test Set Results (Frozen Model) ---")
for key, value in test_results.items():
    print(f"{key}: {value:.4f}")
print("---------------------------------------")

print("\n💾 Saving final frozen model...")
outputdir= '../models/roberta-frozen-emotions/final'
trainer.save_model(outputdir)
print("\n🎯 Model saved to: models/roberta-frozen-emotions/final/")

print("\nDone ✅")