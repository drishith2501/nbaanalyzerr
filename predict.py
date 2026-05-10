"""
predict.py
──────────
Interactive CLI to predict the winner of a single playoff matchup.

Usage
  python predict.py
  python predict.py --home BOS --away MIA --home-rest 3 --away-rest 2

The script:
  1. Loads the trained model and scaler from models/
  2. Fetches current-season advanced stats for both teams (live API call)
  3. Prompts for any missing info (rest days, seeds)
  4. Outputs win probability for each team
"""

import argparse
import time
import sys
import os
import joblib
import numpy as np
import pandas as pd

from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams as nba_teams_static

import config
import nba_utils

# ── helpers ───────────────────────────────────────────────────────────────────

TEAM_ABB_TO_ID = {t["abbreviation"]: t["id"] for t in nba_teams_static.get_teams()}
TEAM_ABB_TO_NAME = {t["abbreviation"]: t["full_name"] for t in nba_teams_static.get_teams()}


def _load_model():
    artifact = joblib.load(config.MODEL_FILE)
    # Support both 'pipeline' and 'model' keys
    model = artifact.get("pipeline") or artifact.get("model")
    features = artifact["features"]
    scaler   = joblib.load(config.SCALER_FILE)
    return model, scaler, features


def _fetch_current_season_stats(season: str = "2024-25") -> pd.DataFrame:
    """Pull latest season advanced stats (regular season for baseline)."""
    print(f"  Fetching current team stats ({season})…")
    try:
        endpoint = nba_utils.fetch_with_retry(
            leaguedashteamstats.LeagueDashTeamStats,
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Per100Possessions",
        )
        df = endpoint.get_data_frames()[0]
        # nba_api 1.11+ dropped TEAM_ABBREVIATION — add from static lookup
        id_to_abbr = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
        df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(id_to_abbr)
        return df
    except Exception as e:
        print(f"  [WARNING] API fetch failed: {e}. Trying fallback...")
        df_fallback = nba_utils.get_team_stats_fallback(season)
        if df_fallback is not None:
            return df_fallback
        raise e


def _get_team_stats(abbr: str, stats_df: pd.DataFrame) -> dict:
    """Return a dict of advanced stats for a team abbreviation."""
    row = stats_df[stats_df["TEAM_ABBREVIATION"] == abbr]
    if row.empty:
        raise ValueError(f"Team '{abbr}' not found in current season stats.")
    return row.iloc[0].to_dict()


def _build_feature_vector(home_stats, away_stats,
                           home_rest, away_rest,
                           home_seed, away_seed,
                           features) -> pd.DataFrame:
    row = {}
    for feat in config.TEAM_FEATURES:
        h_val = home_stats.get(feat, np.nan)
        a_val = away_stats.get(feat, np.nan)
        row[f"{feat}_DIFF"] = h_val - a_val

    row["HOME_REST_DAYS"]       = home_rest
    row["AWAY_REST_DAYS"]       = away_rest
    row["REST_DIFF"]            = home_rest - away_rest
    row["HOME_SEED"]            = home_seed
    row["AWAY_SEED"]            = away_seed
    row["SEED_DIFF"]            = home_seed - away_seed
    row["IS_HOME_HIGHER_SEED"]  = int(home_seed < away_seed)

    # Keep only features the model was trained on
    return pd.DataFrame([{f: row.get(f, np.nan) for f in features}])


# ── prediction ────────────────────────────────────────────────────────────────

def predict_matchup(
    home_abb: str,
    away_abb: str,
    home_rest: int  = 2,
    away_rest: int  = 2,
    home_seed: int  = 1,
    away_seed: int  = 8,
    season: str     = "2024-25",
):
    home_abb = home_abb.upper()
    away_abb = away_abb.upper()

    model, scaler, features = _load_model()
    stats_df = _fetch_current_season_stats(season)

    home_stats = _get_team_stats(home_abb, stats_df)
    away_stats = _get_team_stats(away_abb, stats_df)

    X = _build_feature_vector(
        home_stats, away_stats,
        home_rest, away_rest,
        home_seed, away_seed,
        features,
    )

    X_scaled = scaler.transform(X)
    prob_home = model.predict_proba(X_scaled)[0, 1]
    prob_away = 1.0 - prob_home

    home_name = TEAM_ABB_TO_NAME.get(home_abb, home_abb)
    away_name = TEAM_ABB_TO_NAME.get(away_abb, away_abb)

    print("\n" + "=" * 55)
    print(f"  Playoff Matchup Prediction")
    print("=" * 55)
    print(f"  🏠 Home: {home_name} (#{home_seed} seed, {home_rest}d rest)")
    print(f"  ✈️  Away: {away_name} (#{away_seed} seed, {away_rest}d rest)")
    print("-" * 55)

    bar_len = 40
    home_bar = int(prob_home * bar_len)
    away_bar = bar_len - home_bar
    print(f"  {'█' * home_bar}{'░' * away_bar}")
    print(f"  {home_abb:<6} {prob_home:>6.1%}   vs   {away_abb:<6} {prob_away:>6.1%}")
    print("=" * 55)

    winner = home_name if prob_home > 0.5 else away_name
    win_prob = max(prob_home, prob_away)
    print(f"\n  ➡  Predicted winner: {winner}  ({win_prob:.1%} confidence)\n")

    return {
        "home_team": home_name,
        "away_team": away_name,
        "prob_home_win": round(prob_home, 4),
        "prob_away_win": round(prob_away, 4),
        "predicted_winner": winner,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _interactive():
    print("\n🏀  NBA Playoff Game Predictor")
    print("─" * 35)
    home = input("  Home team abbreviation (e.g. BOS): ").strip().upper()
    away = input("  Away team abbreviation (e.g. MIA): ").strip().upper()
    try:
        hr = int(input("  Home team rest days [default 2]: ").strip() or "2")
        ar = int(input("  Away team rest days [default 2]: ").strip() or "2")
        hs = int(input("  Home seed [default 1]: ").strip() or "1")
        as_ = int(input("  Away seed [default 8]: ").strip() or "8")
    except ValueError:
        print("Invalid input. Using defaults.")
        hr, ar, hs, as_ = 2, 2, 1, 8
    predict_matchup(home, away, hr, ar, hs, as_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict NBA playoff game winner")
    parser.add_argument("--home",       type=str, help="Home team abbreviation (e.g. BOS)")
    parser.add_argument("--away",       type=str, help="Away team abbreviation (e.g. MIA)")
    parser.add_argument("--home-rest",  type=int, default=2)
    parser.add_argument("--away-rest",  type=int, default=2)
    parser.add_argument("--home-seed",  type=int, default=1)
    parser.add_argument("--away-seed",  type=int, default=8)
    parser.add_argument("--season",     type=str, default="2024-25")
    args = parser.parse_args()

    if args.home and args.away:
        predict_matchup(
            args.home, args.away,
            args.home_rest, args.away_rest,
            args.home_seed, args.away_seed,
            args.season,
        )
    else:
        _interactive()
