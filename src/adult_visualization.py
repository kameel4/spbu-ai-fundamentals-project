import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def configure_plots():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 5)


def target_distribution(df):
    return pd.DataFrame({
        "count": df["income"].value_counts(),
        "percent": df["income"].value_counts(normalize=True) * 100,
    })


def plot_target_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="income", order=["<=50K", ">50K"])
    plt.title("Income class distribution")
    plt.xlabel("Income")
    plt.ylabel("Count")
    plt.show()


def plot_numeric_distributions(df, numeric_columns):
    df[numeric_columns].hist(figsize=(14, 8), bins=30)
    plt.suptitle("Numeric feature distributions", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_numeric_by_income(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.boxplot(data=df, x="income", y="age", order=["<=50K", ">50K"], ax=axes[0])
    axes[0].set_title("Age by income")

    sns.boxplot(data=df, x="income", y="education_num", order=["<=50K", ">50K"], ax=axes[1])
    axes[1].set_title("Education years by income")

    sns.boxplot(data=df, x="income", y="hours_per_week", order=["<=50K", ">50K"], ax=axes[2])
    axes[2].set_title("Hours per week by income")

    plt.tight_layout()
    plt.show()


def plot_capital_by_income(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    sns.histplot(data=df, x="capital_gain", hue="income", bins=40, ax=axes[0])
    axes[0].set_title("Capital gain by income")

    sns.histplot(data=df, x="capital_loss", hue="income", bins=40, ax=axes[1])
    axes[1].set_title("Capital loss by income")

    plt.tight_layout()
    plt.show()


def high_income_rate(df, column, top_n=None):
    data = df.copy()

    if top_n is not None:
        top_categories = data[column].value_counts().head(top_n).index
        data = data[data[column].isin(top_categories)]

    income_rate = pd.crosstab(data[column], data["income"], normalize="index")
    return income_rate[">50K"].sort_values(ascending=True).to_frame(name=">50K_share")


def plot_high_income_rate(df, column, top_n=None, figsize=(9, 5)):
    rate = high_income_rate(df, column, top_n=top_n)

    plt.figure(figsize=figsize)
    sns.barplot(x=rate[">50K_share"].values, y=rate.index)
    plt.title(f">50K share by {column}")
    plt.xlabel("Share of >50K")
    plt.ylabel(column)
    plt.xlim(0, 1)
    plt.show()

    return rate


def plot_correlation_heatmap(df, numeric_columns):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Numeric feature correlations")
    plt.show()


def plot_model_comparison(results_summary, top_model_names=None):
    top_model_names = top_model_names or []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = ["Accuracy", "F1", "Balanced Accuracy", "ROC-AUC"]
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sorted_data = results_summary[metric].sort_values(ascending=False)
        colors = [
            "#2ecc71" if model_name in top_model_names else "#3498db"
            for model_name in sorted_data.index
        ]
        bars = ax.bar(range(len(sorted_data)), sorted_data.values, color=colors)
        ax.set_xticks(range(len(sorted_data)))
        ax.set_xticklabels(sorted_data.index, rotation=45, ha="right")
        ax.set_ylabel(metric, fontsize=11, fontweight="bold")
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle("Model comparison: 5-fold cross-validation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_rankings_heatmap(rankings):
    rank_data = rankings.copy()

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        rank_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        cbar_kws={"label": "Rank / mean rank (1=best)"},
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title("Model rankings by metric (1=best)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Model", fontsize=11, fontweight="bold")
    plt.xlabel("Metric", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_final_evaluation(y_true, y_pred, y_score):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["<=50K", ">50K"],
        ax=axes[0],
        values_format="d",
    )
    axes[0].set_title("Confusion matrix")

    RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[1])
    axes[1].set_title("ROC curve")

    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axes[2])
    axes[2].set_title("Precision-Recall curve")

    plt.tight_layout()
    plt.show()


def plot_top_features(feature_table, value_column):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_table, x=value_column, y="feature")
    plt.title("Top model features")
    plt.xlabel(value_column)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
