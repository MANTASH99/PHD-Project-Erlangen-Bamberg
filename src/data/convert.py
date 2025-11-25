import pandas as pd
import numpy as np

print("🔹 Starting data merging process...")

# ===================================================================
# 1️⃣ Process Hidden_emotions_cleaned.csv 
# ===================================================================
try:
    df_hidden = pd.read_csv("Hidden_emotions_cleaned.csv", sep='\t', engine='python')
    print("✅ Loaded Hidden Emotions.")
    
    df_hidden = df_hidden.rename(columns=lambda x: x.strip())
    df_hidden = df_hidden[["label", "text"]]
    df_hidden["label"] = df_hidden["label"].astype(str).str.strip().str.lower()
    df_hidden["text"] = df_hidden["text"].astype(str).str.strip()
    print(f"   - Cleaned Hidden Emotions dataframe. Shape: {df_hidden.shape}")

except Exception as e:
    print(f"❌ Error loading Hidden_emotions_cleaned.csv: {e}")
    exit()

# ===================================================================
# 2️⃣ Process data-super-emotion.csv
# ===================================================================
print("\n🔹 Loading and processing the 2-column Super Emotion dataset...")
try:
    df_super_raw = pd.read_csv("data-super-emotion.csv", header=None, sep=',')
    print(f"   - Raw Super Emotion loaded with {len(df_super_raw.columns)} columns.")

    df_super_clean = pd.DataFrame()
    df_super_clean['text'] = df_super_raw[1] 
    
    def extract_first_emotion(label_string):
        if not isinstance(label_string, str):
            return np.nan
        cleaned_str = label_string.strip("[]' ")
        emotions = cleaned_str.split()
        return emotions[0].lower().strip("'") if emotions else np.nan

    df_super_clean['label'] = df_super_raw[0].apply(extract_first_emotion)
    print(f"   - Processed Super Emotion data, converting multi-word labels to single labels.")
    
except Exception as e:
    print(f"❌ Error loading or processing data-super-emotion.csv: {e}")
    exit()

# ===================================================================
# 3️⃣ Finalize, Merge, and Save (WITH THE FIX)
# ===================================================================
print("\n🔹 Finalizing dataframes before merge...")

def finalize_df(df, name):
    df.dropna(subset=['label', 'text'], inplace=True)
    
    # --- THIS IS THE KEY FIX ---
    # Use the .str accessor to apply string operations to the whole column
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df[df["label"].astype(str).str.strip() != ""]
    # --- END OF FIX ---

    print(f"    - Finalized {name} dataframe. Shape: {df.shape}")
    return df

df_hidden = finalize_df(df_hidden, "Hidden Emotions")
df_super = finalize_df(df_super_clean, "Super Emotion")

print("\n🔹 Merging the two clean, single-label datasets...")
df_merged = pd.concat([df_hidden, df_super], ignore_index=True)
print(f"    - Total rows after merge: {len(df_merged)}")

df_merged = df_merged.drop_duplicates(subset=["text"], keep="first")
print(f"    - Total rows after removing text duplicates: {len(df_merged)}")

df_merged.to_csv("merged_emotions_dataset.csv", index=False)
print("\n✅ Merged dataset saved as 'merged_emotions_dataset.csv'")

# ===================================================================
# 4️⃣ Final Sanity Check
# ===================================================================
print(f"\nTotal rows in final file: {len(df_merged)}")
print(f"Unique labels ({df_merged['label'].nunique()}):")
print(df_merged['label'].value_counts().head(20))

bad_labels = [label for label in df_merged['label'].unique() if ' ' in str(label).strip()]
if bad_labels:
    print("\n❗️ SANITY CHECK FAILED: Found labels with spaces.")
    print(bad_labels[:10])
else:
    print("\n✅ SANITY CHECK PASSED: The merged file contains only single-word labels.")