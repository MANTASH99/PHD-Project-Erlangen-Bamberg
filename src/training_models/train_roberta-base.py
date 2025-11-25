import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
import numpy as np
from evaluate import load

# ============================================================
# 1️⃣ Load and clean data
# ============================================================
Train = '../data/train_split.csv'
print("🔹 Loading dataset...")
df = pd.read_csv(Train)
print(f"✅ Loaded {len(df)} samples with columns: {df.columns.tolist()}")
original_len = len(df)
df = df[df['label'] != 'label'].copy()
print(f"✅ Removed {original_len - len(df)} bad header row(s).")
# Clean and unify labels
df["label"] = df["label"].astype(str).str.strip().str.lower()

# Show label distribution
print("\n📊 Label distribution:")
print(df["label"].value_counts())

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
# 3️⃣ Tokenizer setup and example inspection
# ============================================================
model_name = "roberta-base"
print(f"\n🔹 Loading tokenizer: {model_name}")
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

print("\n🔹 Tokenizing sample texts for inspection:")
for i in range(3):
    sample_text = train_df.iloc[i]["text"]
    tokens = tokenizer.tokenize(sample_text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"\n--- Sample {i+1} ---")
    print(f"Text: {sample_text[:150]}...")
    print(f"Tokens ({len(tokens)}): {tokens[:20]}")
    print(f"IDs: {ids[:20]}")

print("\n🔹 Tokenizing full datasets...")
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
# 4️⃣ Model setup
# ============================================================
print(f"\n🔹 Loading model: {model_name}")
model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)
print("✅ Model loaded with classification head for", len(unique_labels), "emotions")

# ============================================================
# 5️⃣ Metrics
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
# 6️⃣ Training arguments
# ============================================================
print("\n🔹 Preparing training arguments...")
training_args = TrainingArguments(
    output_dir="models/roberta-base-emotions",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50
)
print("✅ Training arguments ready.")

# ============================================================
# 7️⃣ Trainer
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
# 8️⃣ Train and save
# ============================================================
print("\n🚀 Starting training...")
trainer.train()
print("\n✅ Training complete. Saving model...")
trainer.save_model("models/roberta-base-emotions/final")

print("\n🎯 Model saved to: models/roberta-base-emotions/final/")
print("Done ✅")
