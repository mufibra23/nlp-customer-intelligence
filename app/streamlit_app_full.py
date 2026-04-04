"""
NLP Customer Intelligence Dashboard - 5-tab Streamlit app.
  Run: streamlit run app/streamlit_app.py
"""
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Customer Intelligence Engine", layout="wide")

DATA_DIR = "data"
TICKETS_PATH = os.path.join(DATA_DIR, "synthetic", "support_tickets.csv")
TOPICS_PATH = os.path.join(DATA_DIR, "processed", "tickets_with_topics.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "processed", "customer_features.csv")
CHURN_PATH = os.path.join(DATA_DIR, "processed", "customer_churn_scores.csv")

SENTIMENT_COLORS = {"negative": "#EF553B", "neutral": "#636EFA", "positive": "#00CC96"}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_tickets():
    df = pd.read_csv(TICKETS_PATH)
    df["created_date"] = pd.to_datetime(df["created_date"])
    return df


@st.cache_data
def load_topics():
    df = pd.read_csv(TOPICS_PATH)
    df["created_date"] = pd.to_datetime(df["created_date"])
    return df


@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_PATH)


@st.cache_data
def load_churn():
    return pd.read_csv(CHURN_PATH)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Customer Intelligence")
st.sidebar.markdown("**NLP-powered insights from support tickets**")

# Load data
tickets = load_tickets()
topics_df = load_topics()
features = load_features()
churn_df = load_churn()

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=(tickets["created_date"].min().date(), tickets["created_date"].max().date()),
    min_value=tickets["created_date"].min().date(),
    max_value=tickets["created_date"].max().date(),
)
if len(date_range) == 2:
    mask = (tickets["created_date"].dt.date >= date_range[0]) & (
        tickets["created_date"].dt.date <= date_range[1]
    )
    tickets_filtered = tickets[mask]
    topics_filtered = topics_df[
        (topics_df["created_date"].dt.date >= date_range[0])
        & (topics_df["created_date"].dt.date <= date_range[1])
    ]
else:
    tickets_filtered = tickets
    topics_filtered = topics_df

categories = st.sidebar.multiselect(
    "Categories",
    options=sorted(tickets["category"].unique()),
    default=sorted(tickets["category"].unique()),
)
tickets_filtered = tickets_filtered[tickets_filtered["category"].isin(categories)]
topics_filtered = topics_filtered[topics_filtered["category"].isin(categories)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Total tickets: {len(tickets_filtered):,}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sentiment Overview",
    "Topic Explorer",
    "Trend Analysis",
    "Churn Risk",
    "Product Improvement Signals",
])

# ===================== TAB 1: SENTIMENT OVERVIEW =====================
with tab1:
    st.header("Sentiment Overview")

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    total = len(tickets_filtered)
    neg_pct = (tickets_filtered["sentiment_label"] == "negative").mean() * 100
    pos_pct = (tickets_filtered["sentiment_label"] == "positive").mean() * 100
    neu_pct = (tickets_filtered["sentiment_label"] == "neutral").mean() * 100
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    avg_sent = tickets_filtered["sentiment_label"].map(sentiment_map).mean()

    col1.metric("Total Tickets", f"{total:,}")
    col2.metric("Avg Sentiment", f"{avg_sent:.2f}")
    col3.metric("% Negative", f"{neg_pct:.1f}%")
    col4.metric("% Neutral", f"{neu_pct:.1f}%")
    col5.metric("% Positive", f"{pos_pct:.1f}%")

    st.markdown("---")

    # Row 1: donut + histogram
    c1, c2 = st.columns(2)
    with c1:
        sent_counts = tickets_filtered["sentiment_label"].value_counts()
        fig = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            color=sent_counts.index,
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.45,
            title="Sentiment Distribution",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        scores = tickets_filtered["sentiment_label"].map(sentiment_map)
        fig = px.histogram(
            x=scores, nbins=20,
            title="Sentiment Score Distribution",
            labels={"x": "Sentiment Score", "y": "Count"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: time series + category breakdown
    c3, c4 = st.columns(2)
    with c3:
        daily = tickets_filtered.copy()
        daily["date"] = daily["created_date"].dt.date
        daily["score"] = daily["sentiment_label"].map(sentiment_map)
        daily_avg = daily.groupby("date")["score"].mean().reset_index()
        daily_avg.columns = ["date", "avg_sentiment"]
        fig = px.line(
            daily_avg, x="date", y="avg_sentiment",
            title="Daily Average Sentiment",
            labels={"avg_sentiment": "Avg Sentiment", "date": "Date"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        cat_sent = tickets_filtered.groupby(["category", "sentiment_label"]).size().reset_index(name="count")
        fig = px.bar(
            cat_sent, x="category", y="count", color="sentiment_label",
            color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment by Category",
            barmode="group",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ===================== TAB 2: TOPIC EXPLORER =====================
with tab2:
    st.header("Topic Explorer")

    sentiment_filter = st.radio(
        "View topics for:", ["Negative", "Positive"], horizontal=True
    )
    sent_key = sentiment_filter.lower()
    topic_subset = topics_filtered[topics_filtered["sentiment_label"] == sent_key]

    if topic_subset["topic_id"].nunique() > 1:
        # Topic distribution bar chart
        topic_counts = (
            topic_subset[topic_subset["topic_id"] != -1]
            .groupby(["topic_id", "topic_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(15)
        )
        fig = px.bar(
            topic_counts, x="count", y="topic_label",
            orientation="h",
            title=f"Top {sentiment_filter} Topics by Volume",
            labels={"count": "Documents", "topic_label": "Topic"},
            color="count",
            color_continuous_scale="Reds" if sent_key == "negative" else "Greens",
        )
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        # Topic details
        st.subheader("Topic Details")
        selected_topic = st.selectbox(
            "Select a topic to view sample documents:",
            options=topic_counts["topic_label"].tolist(),
        )
        if selected_topic:
            samples = topic_subset[topic_subset["topic_label"] == selected_topic]["text"].head(5)
            for i, text in enumerate(samples, 1):
                st.markdown(f"**{i}.** {text}")
    else:
        st.info("Not enough topics to display. Try adjusting filters.")

    # Embedded HTML visualization
    st.subheader("Interactive Topic Map")
    viz_path = f"models/topic/visualizations/{sent_key}_topic_map.html"
    if os.path.exists(viz_path):
        with open(viz_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)
    else:
        st.warning(f"Visualization not found: {viz_path}")


# ===================== TAB 3: TREND ANALYSIS =====================
with tab3:
    st.header("Trend Analysis")

    # Sentiment trend over time by category
    trend_data = tickets_filtered.copy()
    trend_data["month"] = trend_data["created_date"].dt.to_period("M").astype(str)
    trend_data["score"] = trend_data["sentiment_label"].map(sentiment_map)

    c1, c2 = st.columns(2)
    with c1:
        monthly_cat = (
            trend_data.groupby(["month", "category"])["score"]
            .mean()
            .reset_index()
        )
        fig = px.line(
            monthly_cat, x="month", y="score", color="category",
            title="Monthly Avg Sentiment by Category",
            labels={"score": "Avg Sentiment", "month": "Month"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        monthly_vol = (
            trend_data.groupby(["month", "category"])
            .size()
            .reset_index(name="count")
        )
        fig = px.area(
            monthly_vol, x="month", y="count", color="category",
            title="Monthly Ticket Volume by Category",
            labels={"count": "Tickets", "month": "Month"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Topic trends (negative topics over time)
    st.subheader("Negative Topic Trends")
    neg_topics_trend = topics_filtered[
        (topics_filtered["sentiment_label"] == "negative")
        & (topics_filtered["topic_id"] != -1)
    ].copy()
    if not neg_topics_trend.empty:
        neg_topics_trend["month"] = neg_topics_trend["created_date"].dt.to_period("M").astype(str)
        top_topics = neg_topics_trend["topic_label"].value_counts().head(8).index.tolist()
        neg_topics_trend = neg_topics_trend[neg_topics_trend["topic_label"].isin(top_topics)]
        topic_monthly = (
            neg_topics_trend.groupby(["month", "topic_label"])
            .size()
            .reset_index(name="count")
        )
        fig = px.line(
            topic_monthly, x="month", y="count", color="topic_label",
            title="Top Negative Topics Over Time",
            labels={"count": "Tickets", "month": "Month"},
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly detection: z-score spikes in negative sentiment
    st.subheader("Negative Sentiment Anomalies")
    daily_neg = tickets_filtered.copy()
    daily_neg["date"] = daily_neg["created_date"].dt.date
    daily_neg_count = daily_neg[daily_neg["sentiment_label"] == "negative"].groupby("date").size().reset_index(name="neg_count")
    if len(daily_neg_count) > 7:
        daily_neg_count["rolling_mean"] = daily_neg_count["neg_count"].rolling(7, min_periods=1).mean()
        daily_neg_count["rolling_std"] = daily_neg_count["neg_count"].rolling(7, min_periods=1).std().fillna(1)
        daily_neg_count["z_score"] = (daily_neg_count["neg_count"] - daily_neg_count["rolling_mean"]) / daily_neg_count["rolling_std"]
        daily_neg_count["anomaly"] = daily_neg_count["z_score"] > 2.0

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_neg_count["date"], y=daily_neg_count["neg_count"],
            mode="lines", name="Negative Tickets", line=dict(color="#636EFA"),
        ))
        anomalies = daily_neg_count[daily_neg_count["anomaly"]]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["date"], y=anomalies["neg_count"],
                mode="markers", name="Anomaly (z>2)",
                marker=dict(color="red", size=10, symbol="x"),
            ))
        fig.update_layout(title="Daily Negative Tickets with Anomaly Detection", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ===================== TAB 4: CHURN RISK =====================
with tab4:
    st.header("Churn Risk Dashboard")

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    high_risk = (churn_df["churn_probability"] > 0.7).sum()
    med_risk = ((churn_df["churn_probability"] > 0.3) & (churn_df["churn_probability"] <= 0.7)).sum()
    low_risk = (churn_df["churn_probability"] <= 0.3).sum()
    avg_prob = churn_df["churn_probability"].mean()

    c1.metric("High Risk (>70%)", high_risk)
    c2.metric("Medium Risk (30-70%)", med_risk)
    c3.metric("Low Risk (<30%)", low_risk)
    c4.metric("Avg Churn Probability", f"{avg_prob:.1%}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            churn_df, x="churn_probability", nbins=30,
            title="Churn Probability Distribution",
            labels={"churn_probability": "Churn Probability", "count": "Customers"},
            color_discrete_sequence=["#EF553B"],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Risk bucket pie
        churn_df["risk_bucket"] = pd.cut(
            churn_df["churn_probability"],
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Low", "Medium", "High"],
        )
        risk_counts = churn_df["risk_bucket"].value_counts()
        fig = px.pie(
            values=risk_counts.values, names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={"Low": "#00CC96", "Medium": "#FFA15A", "High": "#EF553B"},
            hole=0.4,
            title="Customer Risk Segments",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # High-risk customer table
    st.subheader("High-Risk Customers")
    risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)
    high_risk_df = churn_df[churn_df["churn_probability"] >= risk_threshold].sort_values(
        "churn_probability", ascending=False
    )
    display_cols = [
        "customer_id", "churn_probability", "avg_sentiment_score",
        "negative_ratio", "complaint_frequency", "topic_diversity",
        "escalation_language", "recency_days",
    ]
    display_cols = [c for c in display_cols if c in high_risk_df.columns]
    st.dataframe(
        high_risk_df[display_cols].head(50).style.format({"churn_probability": "{:.3f}"}),
        use_container_width=True,
    )

    # Export button
    csv_data = high_risk_df[display_cols].to_csv(index=False)
    st.download_button(
        "Download High-Risk List (CSV)",
        csv_data, "high_risk_customers.csv", "text/csv",
    )

    # SHAP plots
    st.subheader("Feature Importance (SHAP)")
    shap_summary_path = "models/churn_plots/shap_summary.png"
    shap_bar_path = "models/churn_plots/shap_bar.png"
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(shap_bar_path):
            st.image(shap_bar_path, caption="SHAP Feature Importance")
        else:
            st.warning("SHAP bar plot not found")
    with c2:
        if os.path.exists(shap_summary_path):
            st.image(shap_summary_path, caption="SHAP Summary (Beeswarm)")
        else:
            st.warning("SHAP summary plot not found")


# ===================== TAB 5: PRODUCT IMPROVEMENT SIGNALS =====================
with tab5:
    st.header("Product Improvement Signals")
    st.markdown("*Actionable insights from negative feedback, grouped by topic*")

    neg_data = topics_filtered[
        (topics_filtered["sentiment_label"] == "negative")
        & (topics_filtered["topic_id"] != -1)
    ].copy()

    if not neg_data.empty:
        # Top negative topics ranked by volume + severity
        topic_stats = (
            neg_data.groupby("topic_label")
            .agg(
                volume=("ticket_id", "count"),
                escalated=("resolution_status", lambda x: (x == "escalated").sum()),
                avg_resolve_days=("days_to_resolve", "mean"),
                categories=("category", lambda x: ", ".join(x.value_counts().head(2).index)),
            )
            .reset_index()
            .sort_values("volume", ascending=False)
        )
        topic_stats["severity_score"] = (
            topic_stats["volume"] / topic_stats["volume"].max() * 0.5
            + topic_stats["escalated"] / max(topic_stats["escalated"].max(), 1) * 0.3
            + topic_stats["avg_resolve_days"] / max(topic_stats["avg_resolve_days"].max(), 1) * 0.2
        )
        topic_stats = topic_stats.sort_values("severity_score", ascending=False)

        # Severity chart
        fig = px.bar(
            topic_stats.head(12),
            x="severity_score", y="topic_label",
            orientation="h",
            color="severity_score",
            color_continuous_scale="Reds",
            title="Top Issues by Severity Score",
            labels={"severity_score": "Severity", "topic_label": "Issue Topic"},
        )
        fig.update_layout(height=450, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        # Detailed breakdown
        st.subheader("Issue Details")
        for _, row in topic_stats.head(8).iterrows():
            with st.expander(f"**{row['topic_label']}** — {row['volume']} complaints, {row['escalated']} escalated"):
                st.markdown(f"- **Categories:** {row['categories']}")
                st.markdown(f"- **Avg Resolution Time:** {row['avg_resolve_days']:.1f} days")
                st.markdown(f"- **Severity Score:** {row['severity_score']:.3f}")

                # Sample complaints
                st.markdown("**Sample complaints:**")
                samples = neg_data[neg_data["topic_label"] == row["topic_label"]]["text"].head(3)
                for i, text in enumerate(samples, 1):
                    st.markdown(f"> {i}. {text}")

        # Action items summary
        st.subheader("Action Items Summary")
        st.markdown("---")
        top_issues = topic_stats.head(5)
        for i, (_, row) in enumerate(top_issues.iterrows(), 1):
            st.markdown(
                f"**{i}. {row['topic_label']}** — "
                f"{row['volume']} complaints, {row['escalated']} escalated, "
                f"avg {row['avg_resolve_days']:.0f}-day resolution. "
                f"Primary categories: {row['categories']}"
            )
    else:
        st.info("No negative topic data available with current filters.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with PyTorch + BERTopic + XGBoost + Streamlit")
