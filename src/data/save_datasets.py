import os
import pandas as pd
from datasets import load_dataset

def save_datasets(train_df, test_df, val_df, data_dir="data", file_format="csv"):
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, f"train.{file_format}")
    test_path  = os.path.join(data_dir, f"test.{file_format}")
    val_path   = os.path.join(data_dir, f"validate.{file_format}")

    if file_format == "csv":
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        val_df.to_csv(val_path, index=False)
    else:
        raise ValueError("Unsupported format. Use ‘csv’ for now.")

    print("✅ Super Emotion data saved to:", os.path.abspath(data_dir))
    print(f"  - train → {train_path}")
    print(f"  - test  → {test_path}")
    print(f"  - val   → {val_path}")

if __name__ == "__main__":
    # Use the correct dataset ID
    dataset = load_dataset("cirimus/super-emotion")
    print(dataset)

    train_df = dataset["train"].to_pandas()
    val_df   = dataset["validation"].to_pandas()
    test_df  = dataset["test"].to_pandas()

    save_datasets(train_df, test_df, val_df, data_dir="data", file_format="csv")
