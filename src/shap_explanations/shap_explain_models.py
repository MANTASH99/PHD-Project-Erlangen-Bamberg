#!/usr/bin/env python3
"""
generate_shap_annotations.py

For each model under ../models, this script:
 - computes predictions on ../data/test_split.csv (expects columns: id, text, label)
 - selects up to N_CORRECT correct predictions
 - computes SHAP token attributions (word-level masking)
 - saves a bar-plot PNG (top-20 tokens) per (model, sample)
 - writes per-model JSON annotations with tokens + shap values + image path

Usage:
  cd emotion_project/scripts
  python generate_shap_annotations.py
"""

import os
import re
import json
import math
import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

TEST_PATH = os.path.join(PROJECT_ROOT, "data", "test_split.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SHAP_RESULTS_DIR = os.path.join(PROJECT_ROOT, "shap_results")

# How many correct samples per model to produce
N_CORRECT_PER_MODEL = 1000


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map model folder -> tokenizer base model (if tokenizer missing in folder)
# Extend this map with the actual folder names you have
MODEL_BASE_MAP = {
    "bert-base-uncased-emotions": "bert-base-uncased",
    "distilbert-base-emotions": "distilbert-base-uncased",
    "roberta-base-emotions": "roberta-base",
    "roberta-frozen-emotions": "roberta-base",
    # add more mappings if needed
}

# ensure output directory exists
os.makedirs(SHAP_RESULTS_DIR, exist_ok=True)

# -----------------------------
# HELPERS
# -----------------------------
def parse_to_words(text):
    """Split text into words (keeps contractions)."""
    if not isinstance(text, str):
        return []
    # \w includes letters digits underscore, keep apostrophes in contractions by allowing '
    return re.findall(r"\w[\w']*", text)

def predict_proba_from_words(word_lists, tokenizer, model):
    """
    Accepts list of list-of-words or list-of-strings.
    Returns numpy array shape (batch, n_classes) of probabilities.
    """
    if isinstance(word_lists, (pd.Series, np.ndarray)):
        word_lists = word_lists.tolist()
    texts = []
    for wl in word_lists:
        if isinstance(wl, (list, tuple)):
            texts.append(" ".join(wl))
        else:
            texts.append(str(wl))
    # tokenization -> to device
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
    return probs

def save_bar_png(tokens, shap_vals, pred_label, model_name, global_id, seq, save_dir):
    """
    Save a clean horizontal bar plot with values in a separate column (no overlap possible).
    Top-20 tokens by absolute SHAP value.
    """
    png_name = f"sample_{global_id}_{seq}_bar.png"
    png_path = os.path.join(save_dir, png_name)

    tokens = list(tokens)
    shap_vals = np.asarray(shap_vals, dtype=float)

    k = min(20, len(tokens))
    if len(tokens) == 0:
        fig = plt.figure(figsize=(6, 1.2))
        plt.text(0.5, 0.5, "NO TOKENS", ha="center", va="center")
        plt.axis("off")
        fig.savefig(png_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return png_path

    idx_top = np.argsort(np.abs(shap_vals))[::-1][:k]
    top_tokens = [tokens[i] for i in idx_top]
    top_vals = shap_vals[idx_top]

    # Create figure with gridspec for better control
    fig = plt.figure(figsize=(13, max(3, 0.4 * len(top_tokens))))
    gs = fig.add_gridspec(1, 20, hspace=0, wspace=0)
    ax_main = fig.add_subplot(gs[0, :17])
    ax_vals = fig.add_subplot(gs[0, 17:], sharey=ax_main)
    
    y_pos = np.arange(len(top_tokens))[::-1]
    colors = ['#66bb6a' if v > 0 else '#ef5350' for v in top_vals]
    
    # Main plot - bars only
    ax_main.barh(y_pos, top_vals[::-1], color=colors[::-1], edgecolor='black', height=0.65)
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(top_tokens[::-1], fontsize=10, fontweight='500')
    ax_main.set_xlabel(f"SHAP contribution to '{pred_label}'", fontsize=11, fontweight='600')
    ax_main.axvline(0, color='black', linewidth=0.8)
    ax_main.grid(axis='x', linestyle='--', alpha=0.25)
    ax_main.set_title(f"Sample {global_id}  |  Model: {model_name}", 
                      fontsize=13, fontweight='bold', pad=15, loc='left')
    
    # Value column - text only
    ax_vals.set_xlim(0, 1)
    ax_vals.axis('off')
    for i, v in enumerate(top_vals[::-1]):
        color = '#2e7d32' if v > 0 else '#c62828'
        ax_vals.text(0.5, y_pos[i], f"{v:.3f}",
                    va='center', ha='center',
                    fontsize=10, fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', 
                             edgecolor=color, 
                             linewidth=1.5))
    
    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return png_path

# -----------------------------
# LOAD TEST SET
# -----------------------------
print("Loading test CSV:", TEST_PATH)
df_test = pd.read_csv(TEST_PATH)

# expected: columns include 'id', 'text', 'label'
if 'id' not in df_test.columns:
    raise SystemExit("ERROR: test_split.csv must contain an 'id' column (global sample id).")

# remove potential header rows saved as data
if 'label' in df_test.columns:
    df_test = df_test[df_test['label'] != 'label'].copy()
df_test["label"] = df_test["label"].astype(str).str.strip().str.lower()
df_test = df_test.reset_index(drop=True)

print(f"Total test samples: {len(df_test)}")

# -----------------------------
# DISCOVER models to process
# -----------------------------
model_candidates = []
for md in sorted(os.listdir(MODELS_DIR)):
    md_path = os.path.join(MODELS_DIR, md)
    if not os.path.isdir(md_path):
        continue
    # prefer 'final' subfolder
    cand_final = os.path.join(md_path, "final")
    model_path = cand_final if os.path.isdir(cand_final) else md_path
    base = MODEL_BASE_MAP.get(md, None)
    if base is None:
        print(f"Skipping model '{md}' — no base mapping found. Add to MODEL_BASE_MAP to include.")
        continue
    model_candidates.append({"name": md, "path": model_path, "base": base})

if not model_candidates:
    raise SystemExit("No models found to process (check MODELS_DIR and MODEL_BASE_MAP).")

print("Models to process:", [m["name"] for m in model_candidates])

# -----------------------------
# MAIN LOOP over models
# -----------------------------
for cfg in model_candidates:
    model_name = cfg["name"]
    model_path = cfg["path"]
    base_tokenizer_name = cfg["base"]
    print("\n" + "="*60)
    print("Processing model:", model_name)
    print("="*60)

    # load tokenizer (prefer from model folder if available, else from base)
    try:
        # try loading tokenizer from model folder
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        print("Loaded tokenizer from model folder:", model_path)
    except Exception:
        print("Tokenizer not in model folder. Loading base tokenizer:", base_tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, use_fast=True)

    # load model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        continue

    # obtain id2label mapping (model.config.id2label)
    id2label = getattr(model.config, "id2label", None)
    label2id = getattr(model.config, "label2id", None)
    if id2label is None or label2id is None:
        # attempt to construct mapping from numeric labels if available
        num_labels = getattr(model.config, "num_labels", None)
        if num_labels:
            id2label = {i: str(i) for i in range(num_labels)}
            label2id = {v: k for k, v in id2label.items()}
        else:
            print("Warning: model has no label mapping. Skipping.")
            continue

    # Predict across test set
    preds = []
    confidences = []
    pred_label_names = []
    print("Running predictions on test set...")
    for text in tqdm(df_test["text"].tolist(), desc=f"Predict {model_name}", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_id = int(torch.argmax(logits, dim=-1).cpu().item())
            pred_conf = float(probs[0, pred_id].cpu().item())
        preds.append(pred_id)
        confidences.append(pred_conf)
        pred_label_names.append(str(id2label.get(pred_id, str(pred_id))).strip().lower())

    # attach predictions to DataFrame (use temp columns)
    pred_col = f"pred_id__{model_name}"
    conf_col = f"conf__{model_name}"
    pred_label_col = f"pred_label__{model_name}"
    df_test[pred_col] = preds
    df_test[conf_col] = confidences
    df_test[pred_label_col] = pred_label_names

    # decide correct predictions by string equality (case-insensitive)
    df_test["true_label_str"] = df_test["label"].astype(str).str.strip().str.lower()
    correct_mask = df_test["true_label_str"] == df_test[pred_label_col].astype(str).str.strip().str.lower()
    num_correct = int(correct_mask.sum())
    total = len(df_test)
    print(f"Model {model_name} -> correct {num_correct}/{total} ({num_correct/total:.4f})")

    if num_correct == 0:
        print("No correct predictions for this model; skipping SHAP.")
        continue

    # select up to N_CORRECT_PER_MODEL correct rows
    correct_rows = df_test[correct_mask].copy().head(N_CORRECT_PER_MODEL)
    print(f"Will compute SHAP for {len(correct_rows)} correct samples (limit {N_CORRECT_PER_MODEL}).")

    # prepare save folders
    model_save_dir = os.path.join(SHAP_RESULTS_DIR, model_name)
    png_dir = os.path.join(model_save_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    tasks_for_json = []
    seq = 0

    # iterate correct rows
    for idx, row in tqdm(correct_rows.iterrows(), total=len(correct_rows), desc=f"SHAP {model_name}", leave=False):
        global_id = int(row["id"])  # expects 'id' column in test CSV
        text = str(row["text"])
        true_label = str(row["true_label_str"])
        pred_label = str(row[pred_label_col])
        confidence = float(row[conf_col])

        words = parse_to_words(text)
        if not words:
            print(f"Skipping sample {global_id} — no tokens")
            continue

        # predictor for SHAP: expects mask arrays (batch, n_words)
        def predict_for_sample(mask_matrix):
            # mask_matrix rows correspond to which words to keep (>0.5)
            out_probs = []
            for mask in mask_matrix:
                kept = [w for w, m in zip(words, mask) if float(m) > 0.5]
                if len(kept) == 0:
                    kept = [""]  # keep empty so tokenizer returns something
                probs = predict_proba_from_words([kept], tokenizer, model)[0]  # returns np array
                out_probs.append(probs)
            return np.array(out_probs)

        # SHAP masker and explainer
        try:
            baseline = np.zeros((1, len(words)))
            masker = shap.maskers.Independent(baseline)
            explainer = shap.Explainer(predict_for_sample, masker, algorithm="auto")
        except Exception as e:
            print(f"Failed to construct SHAP explainer for sample {global_id}: {e}")
            continue

        # explain the full input (all ones)
        try:
            word_input = np.ones((1, len(words)))
            shap_expl = explainer(word_input)
        except Exception as e:
            print(f"SHAP failed for sample {global_id}: {e}")
            continue

        # get tokens and shap values for predicted class
        # shap_expl.data may contain tokens
        tokens = None
        try:
            tokens = shap_expl.data[0] if hasattr(shap_expl, "data") and shap_expl.data else words
        except Exception:
            tokens = words

        shap_vals_arr = shap_expl.values
        # shap_expl.values shape logic
        try:
            if isinstance(shap_vals_arr, np.ndarray):
                if shap_vals_arr.ndim == 3:
                    # (1, n_words, n_classes)
                    pred_id = int(row[pred_col])
                    shap_vals_for_class = shap_vals_arr[0, :, pred_id]
                elif shap_vals_arr.ndim == 2:
                    # (1, n_words)
                    shap_vals_for_class = shap_vals_arr[0, :]
                else:
                    shap_vals_for_class = np.array(shap_vals_arr).reshape(-1)
            else:
                shap_vals_for_class = np.array(shap_vals_arr).astype(float).reshape(-1)
        except Exception:
            shap_vals_for_class = np.zeros(len(tokens))

        # ensure lengths match
        if len(shap_vals_for_class) != len(tokens):
            # try to align by truncation/padding
            m = min(len(shap_vals_for_class), len(tokens))
            shap_vals_for_class = shap_vals_for_class[:m]
            tokens = tokens[:m]

        # Save PNG bar plot
        seq += 1
        try:
            png_path = save_bar_png(tokens, shap_vals_for_class, pred_label, model_name, global_id, seq, png_dir)
        except Exception as e:
            print(f"PNG save failed for sample {global_id}: {e}")
            png_path = ""

        # build json entry
        rel_png_path = os.path.relpath(png_path, PROJECT_ROOT) if png_path else ""
        entry = {
            "global_id": int(global_id),
            "model": model_name,
            "text": text,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "image": rel_png_path.replace("\\", "/"),
            "tokens": [str(t) for t in tokens],
            "shap_values": [float(x) for x in shap_vals_for_class.tolist()]
        }
        tasks_for_json.append(entry)

    # write per-model JSON under shap_results/<model_name>/annotations.json
    out_json_path = os.path.join(model_save_dir, "annotations.json")
    with open(out_json_path, "w", encoding="utf-8") as fj:
        json.dump(tasks_for_json, fj, indent=2, ensure_ascii=False)

    print(f"Wrote {len(tasks_for_json)} entries to {out_json_path}")
    print(f"Saved PNGs to {png_dir}")

print("\nAll done. Per-model JSON files are in shap_results/<model_name>/annotations.json")