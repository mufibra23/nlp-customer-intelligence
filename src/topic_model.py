"""
Topic Model - BERTopic with separate negative/positive models.
Uses ground-truth sentiment_label from CSV (swap to model predictions later).
  Run: python src/topic_model.py
"""
import os
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from umap import UMAP

DATA_PATH = "data/synthetic/support_tickets.csv"
MODEL_DIR = "models/topic"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["created_date"] = pd.to_datetime(df["created_date"])
    print(f"Loaded {len(df):,} tickets")
    print(f"  Sentiment dist: {df['sentiment_label'].value_counts().to_dict()}")
    return df


def build_topic_model(docs, label, min_cluster_size=50):
    """Train a BERTopic model on a list of documents."""
    print(f"\n{'='*60}")
    print(f"Training {label} topic model on {len(docs):,} documents...")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10)
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)

    topic_info = topic_model.get_topic_info()
    print(f"\n  Found {len(topic_info) - 1} topics (excluding outliers)")
    print(f"  Outlier docs: {(pd.Series(topics) == -1).sum():,}")
    for _, row in topic_info.head(11).iterrows():
        print(f"    Topic {row['Topic']}: {row['Name']} ({row['Count']} docs)")

    return topic_model, topics, probs


def save_model_and_visuals(topic_model, docs, label):
    """Save model and generate HTML visualizations."""
    save_dir = os.path.join(MODEL_DIR, f"{label}_topics")
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    topic_model.save(save_dir, serialization="safetensors", save_ctfidf=True,
                     save_embedding_model="all-MiniLM-L6-v2")
    print(f"  Model saved to {save_dir}/")

    # Generate HTML visualizations
    viz_dir = os.path.join(MODEL_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    try:
        fig = topic_model.visualize_topics()
        fig.write_html(os.path.join(viz_dir, f"{label}_topic_map.html"))
        print(f"  Saved {label}_topic_map.html")
    except Exception as e:
        print(f"  Skip topic map: {e}")

    try:
        fig = topic_model.visualize_hierarchy()
        fig.write_html(os.path.join(viz_dir, f"{label}_hierarchy.html"))
        print(f"  Saved {label}_hierarchy.html")
    except Exception as e:
        print(f"  Skip hierarchy: {e}")

    try:
        fig = topic_model.visualize_barchart(top_n_topics=10)
        fig.write_html(os.path.join(viz_dir, f"{label}_barchart.html"))
        print(f"  Saved {label}_barchart.html")
    except Exception as e:
        print(f"  Skip barchart: {e}")

    try:
        fig = topic_model.visualize_heatmap()
        fig.write_html(os.path.join(viz_dir, f"{label}_heatmap.html"))
        print(f"  Saved {label}_heatmap.html")
    except Exception as e:
        print(f"  Skip heatmap: {e}")

    # Save topic info as CSV for downstream use
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(save_dir, "topic_info.csv"), index=False)


def main():
    df = load_data()

    # Split by ground-truth sentiment
    neg_docs = df[df["sentiment_label"] == "negative"]["text"].tolist()
    pos_docs = df[df["sentiment_label"] == "positive"]["text"].tolist()

    # Train negative topic model
    neg_model, neg_topics, neg_probs = build_topic_model(neg_docs, "negative", min_cluster_size=40)
    save_model_and_visuals(neg_model, neg_docs, "negative")

    # Train positive topic model
    pos_model, pos_topics, pos_probs = build_topic_model(pos_docs, "positive", min_cluster_size=25)
    save_model_and_visuals(pos_model, pos_docs, "positive")

    # Save per-ticket topic assignments back to CSV for feature engineering
    df["topic_id"] = -1
    neg_mask = df["sentiment_label"] == "negative"
    pos_mask = df["sentiment_label"] == "positive"
    df.loc[neg_mask, "topic_id"] = neg_topics
    df.loc[pos_mask, "topic_id"] = pos_topics

    # Get topic labels
    df["topic_label"] = ""
    neg_info = neg_model.get_topic_info().set_index("Topic")["Name"]
    pos_info = pos_model.get_topic_info().set_index("Topic")["Name"]
    for idx in df[neg_mask].index:
        tid = df.loc[idx, "topic_id"]
        df.loc[idx, "topic_label"] = neg_info.get(tid, "outlier")
    for idx in df[pos_mask].index:
        tid = df.loc[idx, "topic_id"]
        df.loc[idx, "topic_label"] = pos_info.get(tid, "outlier")

    output_path = "data/processed/tickets_with_topics.csv"
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved enriched data to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
