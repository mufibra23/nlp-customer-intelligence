import re, os
import pandas as pd
import numpy as np

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"#\d{4,}", "[ORDER_ID]", text)
    text = re.sub(r"tkt-\d+", "[TICKET_ID]", text)
    text = re.sub(r"cust-\d+", "[CUSTOMER_ID]", text)
    text = re.sub(r"\$\d+(?:\.\d{2})?", "[AMOUNT]", text)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_synthetic_tickets(path="data/synthetic/support_tickets.csv"):
    df = pd.read_csv(path)
    return pd.DataFrame({
        "source": "support_ticket", "text": df["text"],
        "clean_text": df["text"].apply(clean_text), "label": df["sentiment_label"],
        "category": df["category"], "customer_id": df["customer_id"],
        "timestamp": pd.to_datetime(df["created_date"]),
        "resolution_status": df["resolution_status"],
        "days_to_resolve": df["days_to_resolve"],
        "is_repeat_contact": df["is_repeat_contact"], "priority": df["priority"],
    })


def build_unified_dataset(include_synthetic=True):
    frames = []
    if include_synthetic:
        print("Loading synthetic support tickets...")
        tickets = load_synthetic_tickets()
        print(f"  Loaded {len(tickets):,} tickets")
        frames.append(tickets)
    if not frames:
        raise ValueError("No data sources loaded.")
    unified = pd.concat(frames, ignore_index=True)
    unified = unified[unified["clean_text"].str.len() > 10].reset_index(drop=True)
    print(f"Unified dataset: {len(unified):,} records")
    print(f"  Labels: {unified['label'].value_counts().to_dict()}")
    return unified


def create_splits(df, random_state=42):
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=random_state, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, stratify=temp_df["label"])
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df


def save_processed_data(train_df, val_df, test_df, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    print(f"  Saved to {output_dir}/")


if __name__ == "__main__":
    unified = build_unified_dataset()
    train_df, val_df, test_df = create_splits(unified)
    save_processed_data(train_df, val_df, test_df)
    print(f"Sample: {train_df['clean_text'].iloc[0][:100]}...")
    print(f"Avg text length: {train_df['clean_text'].str.len().mean():.0f} chars")
    print("Done!")
