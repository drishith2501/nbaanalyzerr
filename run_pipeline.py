"""
run_pipeline.py
───────────────
One-shot pipeline runner.  Run this to execute all steps in order:

  Step 1 – Data Collection     (skips if cached files already exist)
  Step 2 – Data Preprocessing
  Step 3 – EDA
  Step 4 – Model Training

Usage
  python run_pipeline.py
  python run_pipeline.py --force   # re-download even if cache exists
"""

import argparse
import os
import sys

import config

def _step(n, title):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'='*60}")


def main(force: bool = False):
    # ── step 1: data collection ───────────────────────────────────────────────
    _step(1, "Data Collection")
    if (
        not force
        and os.path.exists(config.RAW_TEAM_STATS_FILE)
        and os.path.exists(config.RAW_GAME_LOG_FILE)
    ):
        print("  Cached files found — skipping API calls (use --force to re-download).")
    else:
        from data_collection import fetch_team_stats, fetch_game_logs
        fetch_team_stats(config.SEASONS)
        fetch_game_logs(config.SEASONS)

    # ── step 2: preprocessing ─────────────────────────────────────────────────
    _step(2, "Data Preprocessing")
    from data_preprocessing import preprocess
    df = preprocess()

    # ── step 3: EDA ───────────────────────────────────────────────────────────
    _step(3, "Exploratory Data Analysis")
    from eda import (
        plot_home_win_rate, plot_diff_distributions,
        plot_correlation_heatmap, plot_boxplots, plot_rest_days,
        load_data,
    )
    df_eda, features = load_data()
    plot_home_win_rate(df_eda)
    plot_diff_distributions(df_eda, features)
    plot_correlation_heatmap(df_eda, features)
    plot_boxplots(df_eda, features)
    plot_rest_days(df_eda)

    # ── step 4: model training ────────────────────────────────────────────────
    _step(4, "Model Training & Evaluation")
    import model_training as mt
    mt._ensure_dirs()
    df_m, feats = mt.load_dataset()
    X_train, X_test, y_train, y_test, test_season = mt.time_based_split(df_m, feats)
    pipeline = mt.train(X_train, y_train)
    mt.evaluate(pipeline, X_test, y_test, feats, test_season)
    mt.save_artifacts(pipeline, feats)

    print(f"\n{'='*60}")
    print("  ✅  Pipeline complete!")
    print(f"  Plots  → {config.PLOTS_DIR}")
    print(f"  Model  → {config.MODEL_FILE}")
    print(f"{'='*60}")
    print("\n  To make a prediction, run:")
    print("    python predict.py --home BOS --away MIA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()
    main(force=args.force)
