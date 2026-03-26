"""
Churn Propensity Model - XGBoost with SHAP explainability, 5-fold CV.
  Run: python src/churn_model.py
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
)
import shap

DATA_PATH = "data/processed/customer_features.csv"
MODEL_PATH = "models/churn_xgboost.pkl"
PLOTS_DIR = "models/churn_plots"
FEATURE_COLS = [
    "avg_sentiment_score", "sentiment_trend", "negative_ratio",
    "complaint_frequency", "topic_diversity", "escalation_language",
    "response_time_avg", "repeat_contact_rate", "text_length_avg",
    "recency_days",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df["churn_label"].values
    print(f"Loaded {len(df)} customers | {y.sum()} churned ({y.mean():.1%})")
    print(f"Features: {FEATURE_COLS}")
    return df, X, y


def train_and_evaluate(X, y):
    """Train XGBoost with 5-fold cross-validation."""
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    cv_preds = (cv_probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y, cv_probs)
    avg_prec = average_precision_score(y, cv_probs)
    print(f"\n5-Fold CV Results:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Avg Precision: {avg_prec:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y, cv_preds, target_names=["retained", "churned"]))

    cm = confusion_matrix(y, cv_preds)
    print(f"Confusion Matrix:\n{cm}")

    # Train final model on all data
    model.fit(X, y)
    print(f"\nFinal model trained on all {len(y)} samples")

    metrics = {
        "roc_auc": float(roc_auc),
        "avg_precision": float(avg_prec),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y)),
        "n_churned": int(y.sum()),
        "churn_rate": float(y.mean()),
    }

    return model, cv_probs, cv_preds, metrics


def generate_shap_plots(model, X, feature_names):
    """Generate SHAP explainability plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP summary plot (beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_summary.png")

    # SHAP bar plot (mean absolute)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_bar.png")

    # SHAP waterfall for a high-risk customer
    high_risk_idx = np.argmax(model.predict_proba(X)[:, 1])
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[high_risk_idx],
            base_values=explainer.expected_value,
            data=X[high_risk_idx],
            feature_names=feature_names,
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_waterfall_highrisk.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_waterfall_highrisk.png")

    return shap_values


def generate_performance_plots(y, cv_probs, cv_preds):
    """Generate ROC and Precision-Recall curve plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, cv_probs)
    roc_auc = roc_auc_score(y, cv_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Churn Propensity Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print("  Saved roc_curve.png")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, cv_probs)
    avg_prec = average_precision_score(y, cv_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {avg_prec:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Churn Propensity Model")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pr_curve.png"), dpi=150)
    plt.close()
    print("  Saved pr_curve.png")

    # Confusion matrix heatmap
    cm = confusion_matrix(y, cv_preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    classes = ["Retained", "Churned"]
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  Saved confusion_matrix.png")


def save_predictions(df, model, X):
    """Add churn probability to customer dataframe and save."""
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df = df.sort_values("churn_probability", ascending=False)
    output_path = "data/processed/customer_churn_scores.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved churn scores to {output_path}")
    print(f"High-risk (>0.7): {(df['churn_probability'] > 0.7).sum()} customers")
    print(f"Top 5 highest risk:")
    for _, row in df.head(5).iterrows():
        print(f"  {row['customer_id']}: {row['churn_probability']:.3f}")


def main():
    df, X, y = load_data()

    model, cv_probs, cv_preds, metrics = train_and_evaluate(X, y)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Save metrics
    os.makedirs(PLOTS_DIR, exist_ok=True)
    with open(os.path.join(PLOTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate plots
    print("\nGenerating performance plots...")
    generate_performance_plots(y, cv_probs, cv_preds)

    print("\nGenerating SHAP plots...")
    generate_shap_plots(model, X, FEATURE_COLS)

    # Save predictions
    save_predictions(df, model, X)

    print("\nDone!")


if __name__ == "__main__":
    main()
