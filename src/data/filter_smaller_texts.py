import pandas as pd

# Load dataset
df = pd.read_csv("merged_emotions_dataset.csv")

# Count words
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# Filter: keep only texts with >= 6 words
df = df[df['word_count'] >= 15]

print(f"Remaining samples: {len(df)}")
df.to_csv('filtered_full_data_set')