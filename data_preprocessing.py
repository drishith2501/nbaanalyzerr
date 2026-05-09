"""
data_preprocessing.py
─────────────────────
Joins raw game logs with per-season team advanced stats to produce
one row per playoff GAME containing:
  • Differential features (home − away) for every advanced metric
  • Rest-days context features
  • Playoff seeding features
  • Binary label: HOME_WIN (1 = home team won)

Output → data/processed_dataset.csv
"""

import os
import pandas as pd
import numpy as np

import config


# ── helpers ───────────────────────────────────────────────────────────────────

def _calc_rest_days(game_log: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game add a REST_DAYS column = days since previous game.
    First game of the playoffs gets rest = 7 (bye assumption).
    """
    game_log = game_log.copy()
    game_log["GAME_DATE"] = pd.to_datetime(game_log["GAME_DATE"])
    game_log = game_log.sort_values(["SEASON", "TEAM_ID", "GAME_DATE"])
    game_log["REST_DAYS"] = (
        game_log.groupby(["SEASON", "TEAM_ID"])["GAME_DATE"]
        .diff()
        .dt.days
        .fillna(7)          # first game of series → 7 days rest
    )
    return game_log


def _get_playoff_seeds(game_log: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate playoff seeding from the first character of MATCHUP.
    nba_api game logs include the series matchup code (e.g., "BOS vs. MIA").
    We derive seed ordering from win/loss records during the regular season
    via a simple rank within conference.  Here we use TEAM_ID order within
    a series as a proxy when seed data isn't directly available.

    Returns a mapping: {(SEASON, TEAM_ID): SEED} using W/L record rank.
    """
    # Build seed as rank by total wins in that playoff year (higher wins = lower seed number)
    wins = (
        game_log[game_log["WL"] == "W"]
        .groupby(["SEASON", "TEAM_ID"])
        .size()
        .reset_index(name="PLAYOFF_WINS")
    )
    wins["SEED"] = wins.groupby("SEASON")["PLAYOFF_WINS"].rank(ascending=False, method="first").astype(int)
    return wins[["SEASON", "TEAM_ID", "SEED"]]


def _build_game_pairs(game_log: pd.DataFrame) -> pd.DataFrame:
    """
    Each GAME_ID appears twice in the log (one row per team).
    Split into home / away and join on GAME_ID + SEASON.
    """
    home_mask = game_log["MATCHUP"].str.contains(r" vs\. ")
    home = game_log[home_mask].copy().rename(columns=lambda c: "HOME_" + c if c not in ["GAME_ID", "SEASON", "GAME_DATE"] else c)
    away = game_log[~home_mask].copy().rename(columns=lambda c: "AWAY_" + c if c not in ["GAME_ID", "SEASON", "GAME_DATE"] else c)

    merged = pd.merge(home, away, on=["GAME_ID", "SEASON", "GAME_DATE"], how="inner")
    return merged


# ── main preprocessing routine ────────────────────────────────────────────────

def preprocess(
    raw_game_log_path: str = config.RAW_GAME_LOG_FILE,
    raw_team_stats_path: str = config.RAW_TEAM_STATS_FILE,
    output_path: str = config.PROCESSED_DATA_FILE,
) -> pd.DataFrame:

    os.makedirs(config.DATA_DIR, exist_ok=True)

    print("Loading raw data…")
    game_log   = pd.read_csv(raw_game_log_path)
    team_stats = pd.read_csv(raw_team_stats_path)

    # ── rest days ──────────────────────────────────────────────────────────────
    print("Computing rest days…")
    game_log = _calc_rest_days(game_log)

    # ── seed proxy ────────────────────────────────────────────────────────────
    print("Estimating playoff seeds…")
    seed_df = _get_playoff_seeds(game_log)
    game_log = pd.merge(game_log, seed_df, on=["SEASON", "TEAM_ID"], how="left")

    # ── pair home / away rows ─────────────────────────────────────────────────
    print("Pairing home & away rows…")
    pairs = _build_game_pairs(game_log)

    # ── merge advanced team stats (season-level) ───────────────────────────────
    print("Merging advanced team stats…")

    # Select only the columns we need from team_stats
    adv_cols = ["SEASON", "TEAM_ID"] + [c for c in config.TEAM_FEATURES if c in team_stats.columns]
    adv = team_stats[adv_cols].copy()

    # Home side
    pairs = pd.merge(
        pairs,
        adv.rename(columns={c: f"HOME_{c}" for c in config.TEAM_FEATURES}),
        left_on=["SEASON", "HOME_TEAM_ID"],
        right_on=["SEASON", "TEAM_ID"],
        how="left",
    ).drop(columns=["TEAM_ID"], errors="ignore")

    # Away side
    pairs = pd.merge(
        pairs,
        adv.rename(columns={c: f"AWAY_{c}" for c in config.TEAM_FEATURES}),
        left_on=["SEASON", "AWAY_TEAM_ID"],
        right_on=["SEASON", "TEAM_ID"],
        how="left",
    ).drop(columns=["TEAM_ID"], errors="ignore")

    # ── differential features ─────────────────────────────────────────────────
    print("Building differential features…")
    for feat in config.TEAM_FEATURES:
        h_col = f"HOME_{feat}"
        a_col = f"AWAY_{feat}"
        if h_col in pairs.columns and a_col in pairs.columns:
            pairs[f"{feat}_DIFF"] = pairs[h_col] - pairs[a_col]

    # ── context features ──────────────────────────────────────────────────────
    if "HOME_REST_DAYS" not in pairs.columns:
        pairs["HOME_REST_DAYS"] = 4
    if "AWAY_REST_DAYS" not in pairs.columns:
        pairs["AWAY_REST_DAYS"] = 4
    pairs["REST_DIFF"] = pairs["HOME_REST_DAYS"] - pairs["AWAY_REST_DAYS"]

    if "HOME_SEED" not in pairs.columns:
        pairs["HOME_SEED"] = np.nan
    if "AWAY_SEED" not in pairs.columns:
        pairs["AWAY_SEED"] = np.nan
    pairs["SEED_DIFF"] = pairs["HOME_SEED"] - pairs["AWAY_SEED"]
    pairs["IS_HOME_HIGHER_SEED"] = (pairs["HOME_SEED"] < pairs["AWAY_SEED"]).astype(int)

    # ── label ─────────────────────────────────────────────────────────────────
    pairs["HOME_WIN"] = (pairs["HOME_WL"] == "W").astype(int)

    # ── keep only model-relevant columns + identifiers ────────────────────────
    keep_cols = (
        ["GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"]
        + config.MODEL_FEATURES
        + ["HOME_WIN"]
    )
    keep_cols = [c for c in keep_cols if c in pairs.columns]
    dataset = pairs[keep_cols].dropna(subset=config.DIFF_FEATURES)

    dataset.to_csv(output_path, index=False)
    print(f"\n✅  Processed dataset saved → {output_path} ({len(dataset)} rows, {dataset['HOME_WIN'].mean():.1%} home wins)")
    return dataset


if __name__ == "__main__":
    print("=" * 60)
    print("NBA PLAYOFF PREDICTOR — Data Preprocessing")
    print("=" * 60)
    df = preprocess()
    print(df.head())
