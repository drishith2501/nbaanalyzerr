"""
data_collection.py
──────────────────
Pulls 10 seasons of NBA PLAYOFF data from the nba_api:
  1. Team advanced stats per season (OFF/DEF rating, pace, etc.)
  2. Game-by-game logs for every playoff series

Outputs
  data/raw_team_stats.csv   – one row per team per season
  data/raw_game_log.csv     – one row per playoff game
"""

import os
import time
import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import (
    leaguedashteamstats,
    leaguegamelog,
    teamgamelog,
)
from nba_api.stats.static import teams as nba_teams_static

import config
import nba_utils

# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_get(endpoint_fn, *args, **kwargs):
    """Call an nba_api endpoint and retry once on transient failure."""
    return nba_utils.fetch_with_retry(endpoint_fn, **kwargs)


# ── 1. Advanced team stats per season ─────────────────────────────────────────

def fetch_team_stats(seasons: list[str]) -> pd.DataFrame:
    """
    Pull LeagueDashTeamStats (advanced) for each season, playoffs only.
    Returns a combined DataFrame.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    all_stats = []
    for season in tqdm(seasons, desc="Fetching team stats"):
        print(f"\n  Season {season}…")
        endpoint = _safe_get(
            leaguedashteamstats.LeagueDashTeamStats,
            season=season,
            season_type_all_star="Playoffs",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Per100Possessions",
        )
        df = endpoint.get_data_frames()[0]
        df["SEASON"] = season
        all_stats.append(df)

    combined = pd.concat(all_stats, ignore_index=True)

    # nba_api 1.11+ does not include TEAM_ABBREVIATION — add it from static lookup
    id_to_abbr = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
    combined["TEAM_ABBREVIATION"] = combined["TEAM_ID"].map(id_to_abbr)

    combined.to_csv(config.RAW_TEAM_STATS_FILE, index=False)
    print(f"\n✅  Team stats saved → {config.RAW_TEAM_STATS_FILE} ({len(combined)} rows)")
    return combined


# ── 2. Playoff game logs ───────────────────────────────────────────────────────

def fetch_game_logs(seasons: list[str]) -> pd.DataFrame:
    """
    Pull LeagueGameLog for each playoff season.
    Each row = one team's line in one game; we later pair home/away.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    all_games = []
    for season in tqdm(seasons, desc="Fetching game logs"):
        print(f"\n  Season {season}…")
        endpoint = _safe_get(
            leaguegamelog.LeagueGameLog,
            season=season,
            season_type_all_star="Playoffs",
            direction="ASC",
        )
        df = endpoint.get_data_frames()[0]
        df["SEASON"] = season
        all_games.append(df)

    combined = pd.concat(all_games, ignore_index=True)
    combined.to_csv(config.RAW_GAME_LOG_FILE, index=False)
    print(f"\n✅  Game logs saved → {config.RAW_GAME_LOG_FILE} ({len(combined)} rows)")
    return combined


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("NBA PLAYOFF PREDICTOR — Data Collection")
    print("=" * 60)

    team_stats = fetch_team_stats(config.SEASONS)
    game_logs  = fetch_game_logs(config.SEASONS)

    print("\nDone. Files written:")
    print(f"  {config.RAW_TEAM_STATS_FILE}")
    print(f"  {config.RAW_GAME_LOG_FILE}")
