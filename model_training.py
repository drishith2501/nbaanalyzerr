"""
model_training.py
─────────────────
Trains a Logistic Regression classifier on the processed dataset.

Steps
  1. Load processed_dataset.csv
  2. Train/test split (time-based — last season = test)
  3. StandardScaler → LogisticRegression with cross-validation
  4. Evaluate with accuracy, ROC-AUC, classification report, confusion matrix
  5. Persist model + scaler with joblib

Outputs
  models/logistic_regression_model.joblib
  models/scaler.joblib
  plots/  (several diagnostic plots)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "figure.figsize": (8, 5)})


# ── utility ───────────────────────────────────────────────────────────────────

def _ensure_dirs():
    os.makedirs(config.DATA_DIR,  exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)


# ── load & split ──────────────────────────────────────────────────────────────

def load_dataset(path: str = config.PROCESSED_DATA_FILE):
    df = pd.read_csv(path)
    # Available feature columns only (some optional context cols may be missing)
    available_features = [f for f in config.MODEL_FEATURES if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    return df, available_features


def time_based_split(df: pd.DataFrame, features: list[str]):
    """Hold out the most recent season as test set."""
    seasons = sorted(df["SEASON"].unique())
    test_season = seasons[-1]
    train_season = seasons[:-1]

    train = df[df["SEASON"].isin(train_season)]
    test  = df[df["SEASON"] == test_season]

    X_train, y_train = train[features], train["HOME_WIN"]
    X_test,  y_test  = test[features],  test["HOME_WIN"]

    print(f"Train seasons: {train_season}")
    print(f"Test season:   {test_season}")
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")
    return X_train, X_test, y_train, y_test, test_season


# ── training ──────────────────────────────────────────────────────────────────

def train(X_train, y_train):
    # Build a pipeline: impute NaNs → scale → classify
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
            C=config.LR_C,
            max_iter=config.LR_MAX_ITER,
            class_weight=config.LR_CLASS_WEIGHT,
            random_state=config.RANDOM_STATE,
            solver="lbfgs",
        )),
    ])

    # 5-fold stratified CV on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\nCross-val ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)
    return pipeline


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(pipeline, X_test, y_test, features: list[str], test_season: str):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"Test season: {test_season}")
    print(f"Accuracy :  {acc:.4f}")
    print(f"ROC-AUC  :  {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Away Win','Home Win'])}")

    model = pipeline.named_steps["model"]
    _plot_confusion_matrix(y_test, y_pred, test_season)
    _plot_roc_curve(y_test, y_prob, auc, test_season)
    _plot_feature_importance(model, features)
    _plot_prob_distribution(pipeline, X_test, y_test)

    return acc, auc


# ── plots ─────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, season):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Away Win", "Home Win"],
                yticklabels=["Away Win", "Home Win"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {season} Playoffs")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_roc_curve(y_true, y_prob, auc, season):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {season} Playoffs")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "roc_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_feature_importance(model, features):
    coefs = pd.Series(model.coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(5, len(features) * 0.4)))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in coefs]
    coefs.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Logistic Regression Coefficients\n(green → favors home win)")
    ax.set_xlabel("Coefficient")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "feature_importance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_prob_distribution(pipeline, X_test, y_test):
    """Predicted probability distribution for home wins vs losses."""
    probs = pipeline.predict_proba(X_test)[:, 1]
    y_arr = np.array(y_test)
    fig, ax = plt.subplots()
    sns.histplot(probs[y_arr == 1], color="#2ecc71", label="Home Win", kde=True, ax=ax, bins=20)
    sns.histplot(probs[y_arr == 0], color="#e74c3c", label="Away Win", kde=True, ax=ax, bins=20)
    ax.set_xlabel("Predicted P(Home Win)")
    ax.set_title("Predicted Probability Distribution")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "prob_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── persist ───────────────────────────────────────────────────────────────────

def save_artifacts(pipeline, features: list[str]):
    # Save pipeline (includes imputer + scaler + model) and feature list
    joblib.dump({"pipeline": pipeline, "features": features}, config.MODEL_FILE)
    # Keep scaler file as a no-op shim so old code doesn't break
    joblib.dump(pipeline.named_steps["scaler"], config.SCALER_FILE)
    print(f"\n✅  Pipeline saved → {config.MODEL_FILE}")
    print(f"✅  Scaler saved   → {config.SCALER_FILE}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _ensure_dirs()
    print("=" * 60)
    print("NBA PLAYOFF PREDICTOR — Model Training")
    print("=" * 60)

    df, features = load_dataset()
    X_train, X_test, y_train, y_test, test_season = time_based_split(df, features)

    print("\nTraining Logistic Regression…")
    pipeline = train(X_train, y_train)

    print("\nEvaluating on hold-out test season…")
    evaluate(pipeline, X_test, y_test, features, test_season)

    save_artifacts(pipeline, features)
    print("\nDone.")
