import pandas as pd
from sklearn.model_selection import train_test_split

# 1️⃣ Load your dataset
data_path = "../data/filtered_full_data_set.csv"
df = pd.read_csv(data_path)

print(f"✅ Loaded dataset with {len(df)} samples.")
print(f"Columns: {list(df.columns)}")

# 2️⃣ Split into train (75%) and temp (25%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.25,          # 25% left for test + validation
    stratify=df["label"],    # keeps label distribution balanced
    random_state=42,
)

# 3️⃣ Split temp into test (15%) and validation (10%)
# test = 15/25 = 0.6 of temp → 0.15 total
test_df, val_df = train_test_split(
    temp_df,
    test_size=0.4,           # 40% of 25% → 10% of total
    stratify=temp_df["label"],
    random_state=42,
)

# 4️⃣ Print out how many samples per split
print("\n📊 Dataset split summary:")
print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.2f}%)")
print(f"Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.2f}%)")
print(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.2f}%)")

# 5️⃣ Save to CSV
train_df.to_csv("../data/train_split.csv", index=False)
val_df.to_csv("../data/val_split.csv", index=False)
test_df.to_csv("../data/test_split.csv", index=False)

print("\n✅ Splits saved to:")
print(" - data/train_split.csv")
print(" - data/val_split.csv")
print(" - data/test_split.csv")
