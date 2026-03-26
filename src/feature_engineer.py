"""
Feature Engineering - Per-customer NLP feature extraction + churn labels.
  Run: python src/feature_engineer.py
"""
import os
import re
import numpy as np
import pandas as pd

DATA_PATH = "data/processed/tickets_with_topics.csv"
OUTPUT_PATH = "data/processed/customer_features.csv"

SENTIMENT_SCORE_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
ESCALATION_WORDS = {
    "cancel", "refund", "lawyer", "bbb", "never again", "lawsuit", "attorney",
    "report", "scam", "fraud", "unacceptable", "disgusting", "horrible",
    "worst", "furious", "immediately", "right now",
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["sentiment_score"] = df["sentiment_label"].map(SENTIMENT_SCORE_MAP)
    print(f"Loaded {len(df):,} tickets for {df['customer_id'].nunique()} customers")
    return df


def compute_sentiment_trend(group):
    """Slope of sentiment scores over time (OLS). Negative slope = declining sentiment."""
    if len(group) < 2:
        return 0.0
    x = (group["created_date"] - group["created_date"].min()).dt.days.values.astype(float)
    y = group["sentiment_score"].values
    if x.std() == 0:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def has_escalation_language(text):
    """Check if text contains escalation keywords."""
    text_lower = str(text).lower()
    return int(any(word in text_lower for word in ESCALATION_WORDS))


def extract_features(df):
    """Extract 10 NLP features per customer."""
    ref_date = df["created_date"].max()

    # Pre-compute per-ticket features
    df["escalation_flag"] = df["text"].apply(has_escalation_language)
    df["text_length"] = df["text"].str.len()

    features = []
    for cust_id, grp in df.groupby("customer_id"):
        grp_sorted = grp.sort_values("created_date")

        # 1. avg_sentiment_score
        avg_sentiment = grp["sentiment_score"].mean()

        # 2. sentiment_trend (slope)
        sentiment_trend = compute_sentiment_trend(grp_sorted)

        # 3. negative_ratio
        n_total = len(grp)
        negative_ratio = (grp["sentiment_label"] == "negative").sum() / n_total

        # 4. complaint_frequency (tickets in last 90 days)
        recent_mask = grp["created_date"] >= (ref_date - pd.Timedelta(days=90))
        complaint_freq = recent_mask.sum()

        # 5. topic_diversity (unique non-outlier topics)
        topic_ids = grp["topic_id"][grp["topic_id"] != -1]
        topic_diversity = topic_ids.nunique()

        # 6. escalation_language (any ticket has escalation words)
        escalation_language = int(grp["escalation_flag"].any())

        # 7. response_time_avg
        response_time_avg = grp["days_to_resolve"].mean()

        # 8. repeat_contact_rate
        repeat_contact_rate = grp["is_repeat_contact"].mean()

        # 9. text_length_avg
        text_length_avg = grp["text_length"].mean()

        # 10. recency_days
        recency_days = (ref_date - grp["created_date"].max()).days

        features.append({
            "customer_id": cust_id,
            "avg_sentiment_score": round(avg_sentiment, 4),
            "sentiment_trend": round(sentiment_trend, 6),
            "negative_ratio": round(negative_ratio, 4),
            "complaint_frequency": complaint_freq,
            "topic_diversity": topic_diversity,
            "escalation_language": escalation_language,
            "response_time_avg": round(response_time_avg, 2),
            "repeat_contact_rate": round(repeat_contact_rate, 4),
            "text_length_avg": round(text_length_avg, 2),
            "recency_days": recency_days,
            "total_tickets": n_total,
        })

    return pd.DataFrame(features)


def construct_churn_labels(features_df, df):
    """
    Churn label construction (target ~15-20% churn rate):
    - Customers with 2+ cancellation tickets AND negative sentiment → churned
    - Customers with strongly declining sentiment + inactive 60+ days → churned
    - Customers with very high negative ratio + escalation + inactive → churned
    """
    # Count cancellation tickets per customer
    cancel_counts = (
        df[df["category"] == "cancellation"]
        .groupby("customer_id")
        .size()
        .to_dict()
    )

    churn_labels = []
    for _, row in features_df.iterrows():
        cid = row["customer_id"]
        n_cancel = cancel_counts.get(cid, 0)

        is_churned = False
        # Strong cancellation signal: 3+ cancel tickets AND overall negative sentiment
        if n_cancel >= 3 and row["avg_sentiment_score"] < -0.4:
            is_churned = True
        # Declining sentiment + long inactivity
        elif row["sentiment_trend"] < -0.002 and row["recency_days"] > 60:
            is_churned = True
        # Very frustrated: high negative ratio + escalation + inactive
        elif (row["negative_ratio"] > 0.75
              and row["escalation_language"] == 1
              and row["recency_days"] > 60):
            is_churned = True

        churn_labels.append(int(is_churned))

    features_df["churn_label"] = churn_labels
    churn_rate = features_df["churn_label"].mean()
    print(f"Churn rate: {churn_rate:.1%} ({features_df['churn_label'].sum()}/{len(features_df)})")
    return features_df


def main():
    df = load_data()

    print("\nExtracting per-customer features...")
    features_df = extract_features(df)

    print("\nConstructing churn labels...")
    features_df = construct_churn_labels(features_df, df)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(features_df)} customer feature rows to {OUTPUT_PATH}")

    # Summary stats
    print("\nFeature summary:")
    feature_cols = [c for c in features_df.columns if c not in ("customer_id", "churn_label")]
    print(features_df[feature_cols].describe().round(3).to_string())

    print("\nChurn label distribution:")
    print(features_df["churn_label"].value_counts().to_string())
    print("Done!")


if __name__ == "__main__":
    main()
