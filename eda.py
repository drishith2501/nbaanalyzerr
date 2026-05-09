"""
eda.py
──────
Exploratory Data Analysis for the processed playoff dataset.
Generates seaborn plots that help you understand:
  • Home-win rate by season
  • Distribution of key differentials
  • Correlation heatmap of model features
  • Pair-plot of top discriminating features

All plots are saved to plots/eda_*.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config

sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({"figure.dpi": 130})

os.makedirs(config.PLOTS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(config.PROCESSED_DATA_FILE)
    available = [f for f in config.MODEL_FEATURES if f in df.columns]
    return df, available


# ── 1. Home-win rate by season ────────────────────────────────────────────────

def plot_home_win_rate(df):
    rate = df.groupby("SEASON")["HOME_WIN"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=rate, x="SEASON", y="HOME_WIN", palette="Blues_d", ax=ax)
    ax.axhline(rate["HOME_WIN"].mean(), color="red", linestyle="--", label="Overall mean")
    ax.set_title("Playoff Home-Win Rate by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Home Win %")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.xticks(rotation=35)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "eda_home_win_rate.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 2. Distribution of key differentials ──────────────────────────────────────

def plot_diff_distributions(df, features):
    diff_cols = [f for f in features if f.endswith("_DIFF") and f in df.columns][:6]
    n = len(diff_cols)
    if n == 0:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, diff_cols):
        sns.histplot(
            data=df, x=col, hue="HOME_WIN", kde=True,
            palette={0: "#e74c3c", 1: "#2ecc71"},
            alpha=0.6, ax=ax,
        )
        ax.set_title(col.replace("_DIFF", "").replace("_", " "))
        ax.set_xlabel("Home − Away")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Key Metric Differentials: Home Win vs Away Win", fontsize=13)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "eda_diff_distributions.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 3. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(df, features):
    cols = [f for f in features if f in df.columns] + ["HOME_WIN"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 0.7), max(8, len(cols) * 0.6)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "eda_correlation_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 4. Box-plots: key metrics by outcome ─────────────────────────────────────

def plot_boxplots(df, features):
    key = [f for f in ["NET_RATING_DIFF", "OFF_RATING_DIFF", "DEF_RATING_DIFF", "PACE_DIFF", "TS_PCT_DIFF"] if f in df.columns]
    if not key:
        return
    melted = df[key + ["HOME_WIN"]].melt(id_vars="HOME_WIN", var_name="Metric", value_name="Differential")
    melted["Outcome"] = melted["HOME_WIN"].map({1: "Home Win", 0: "Away Win"})
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=melted, x="Metric", y="Differential", hue="Outcome",
                palette={"Home Win": "#2ecc71", "Away Win": "#e74c3c"}, ax=ax)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Key Metric Differentials by Game Outcome")
    ax.set_xlabel("")
    plt.xticks(rotation=25)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "eda_boxplots.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 5. Rest days impact ───────────────────────────────────────────────────────

def plot_rest_days(df):
    if "REST_DIFF" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=df, x="REST_DIFF", hue="HOME_WIN",
                 palette={0: "#e74c3c", 1: "#2ecc71"}, kde=True, ax=axes[0])
    axes[0].set_title("Rest Differential Distribution")
    axes[0].set_xlabel("Home Rest Days − Away Rest Days")

    # Win rate by rest bucket
    df2 = df.copy()
    df2["REST_BUCKET"] = pd.cut(df2["REST_DIFF"], bins=[-10, -2, -1, 0, 1, 2, 10],
                                labels=["≤-3", "-2", "-1", "0", "+1", "≥+2"])
    rb = df2.groupby("REST_BUCKET")["HOME_WIN"].mean().reset_index()
    sns.barplot(data=rb, x="REST_BUCKET", y="HOME_WIN", palette="Blues_d", ax=axes[1])
    axes[1].set_ylim(0, 1)
    axes[1].axhline(df["HOME_WIN"].mean(), color="red", linestyle="--")
    axes[1].set_title("Home Win Rate by Rest Differential Bucket")
    axes[1].set_xlabel("Rest Differential")
    axes[1].set_ylabel("Home Win %")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "eda_rest_days.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("NBA PLAYOFF PREDICTOR — EDA")
    print("=" * 60)
    df, features = load_data()
    print(f"Dataset: {len(df)} games | Seasons: {sorted(df['SEASON'].unique())}")
    print(f"Home win rate: {df['HOME_WIN'].mean():.1%}\n")

    plot_home_win_rate(df)
    plot_diff_distributions(df, features)
    plot_correlation_heatmap(df, features)
    plot_boxplots(df, features)
    plot_rest_days(df)

    print("\nAll EDA plots saved to:", config.PLOTS_DIR)
