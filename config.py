"""
Central configuration for NBA Playoff Predictor
"""
import os

# ── Seasons to pull (10 seasons of playoff data) ──────────────────────────────
SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
]

# ── Feature columns fed into the model ────────────────────────────────────────
TEAM_FEATURES = [
    "OFF_RATING",   # points scored per 100 possessions
    "DEF_RATING",   # points allowed per 100 possessions
    "NET_RATING",   # OFF_RATING - DEF_RATING
    "PACE",         # possessions per 48 minutes
    "AST_PCT",      # assist percentage
    "REB_PCT",      # rebound percentage
    "TS_PCT",       # true-shooting percentage
    "EFG_PCT",      # effective field-goal %
    "TM_TOV_PCT",   # team turnover %
]

# Differential features (home - away) created at model-build time
DIFF_FEATURES = [f + "_DIFF" for f in TEAM_FEATURES]

# Extra context features
CONTEXT_FEATURES = [
    "HOME_REST_DAYS",   # days since last game for home team
    "AWAY_REST_DAYS",   # days since last game for away team
    "REST_DIFF",        # home rest - away rest
    "HOME_SEED",        # playoff seeding
    "AWAY_SEED",
    "SEED_DIFF",        # home seed - away seed (lower is better)
    "IS_HOME_HIGHER_SEED",  # binary: 1 if home team is better seeded
]

MODEL_FEATURES = DIFF_FEATURES + CONTEXT_FEATURES

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")

RAW_TEAM_STATS_FILE  = os.path.join(DATA_DIR, "raw_team_stats.csv")
RAW_GAME_LOG_FILE    = os.path.join(DATA_DIR, "raw_game_log.csv")
PROCESSED_DATA_FILE  = os.path.join(DATA_DIR, "processed_dataset.csv")
MODEL_FILE           = os.path.join(MODEL_DIR, "logistic_regression_model.joblib")
SCALER_FILE          = os.path.join(MODEL_DIR, "scaler.joblib")

# ── nba_api rate-limit safety delay (seconds between requests) ─────────────────
REQUEST_DELAY = 0.5

# ── Logistic Regression hyper-params ──────────────────────────────────────────
LR_C          = 1.0      # inverse regularisation strength
LR_MAX_ITER   = 1000
LR_CLASS_WEIGHT = "balanced"

# ── Random seed ───────────────────────────────────────────────────────────────
RANDOM_STATE = 42
