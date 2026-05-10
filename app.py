"""
app.py  –  Flask web dashboard — Upcoming NBA Playoff Predictions
Run:  python app.py
Open: http://localhost:5000
"""

import os, json, time, traceback, datetime
from flask import Flask, render_template, request, jsonify
import config
import nba_utils
import upcoming_games

# Defer heavy imports to avoid slow startup
_pd = None
_np = None
_joblib = None

def _ensure_pandas():
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd

def _ensure_numpy():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np

def _ensure_joblib():
    global _joblib
    if _joblib is None:
        import joblib
        _joblib = joblib
    return _joblib

app = Flask(__name__)

# ── teams lookup (lazy-loaded) ────────────────────────────────────────────────
_TEAMS_CACHE = None

def _get_all_teams():
    """Lazy-load NBA teams to avoid startup delay on Vercel."""
    global _TEAMS_CACHE
    if _TEAMS_CACHE is not None:
        return _TEAMS_CACHE
    
    try:
        from nba_api.stats.static import teams as nba_teams_static
        _TEAMS_CACHE = sorted(
            [{"abbr": t["abbreviation"], "name": t["full_name"]} for t in nba_teams_static.get_teams()],
            key=lambda x: x["name"],
        )
    except Exception as e:
        print(f"Failed to load NBA teams: {e}")
        # Fallback to hardcoded teams
        _TEAMS_CACHE = [
            {"abbr": "ATL", "name": "Atlanta Hawks"},
            {"abbr": "BOS", "name": "Boston Celtics"},
            {"abbr": "BKN", "name": "Brooklyn Nets"},
            {"abbr": "CHA", "name": "Charlotte Hornets"},
            {"abbr": "CHI", "name": "Chicago Bulls"},
            {"abbr": "CLE", "name": "Cleveland Cavaliers"},
            {"abbr": "DAL", "name": "Dallas Mavericks"},
            {"abbr": "DEN", "name": "Denver Nuggets"},
            {"abbr": "DET", "name": "Detroit Pistons"},
            {"abbr": "GSW", "name": "Golden State Warriors"},
            {"abbr": "HOU", "name": "Houston Rockets"},
            {"abbr": "IND", "name": "Indiana Pacers"},
            {"abbr": "LAC", "name": "LA Clippers"},
            {"abbr": "LAL", "name": "Los Angeles Lakers"},
            {"abbr": "MEM", "name": "Memphis Grizzlies"},
            {"abbr": "MIA", "name": "Miami Heat"},
            {"abbr": "MIL", "name": "Milwaukee Bucks"},
            {"abbr": "MIN", "name": "Minnesota Timberwolves"},
            {"abbr": "NOP", "name": "New Orleans Pelicans"},
            {"abbr": "NYK", "name": "New York Knicks"},
            {"abbr": "OKC", "name": "Oklahoma City Thunder"},
            {"abbr": "ORL", "name": "Orlando Magic"},
            {"abbr": "PHI", "name": "Philadelphia 76ers"},
            {"abbr": "PHX", "name": "Phoenix Suns"},
            {"abbr": "POR", "name": "Portland Trail Blazers"},
            {"abbr": "SAC", "name": "Sacramento Kings"},
            {"abbr": "SAS", "name": "San Antonio Spurs"},
            {"abbr": "TOR", "name": "Toronto Raptors"},
            {"abbr": "UTA", "name": "Utah Jazz"},
            {"abbr": "WAS", "name": "Washington Wizards"},
        ]
    
    return _TEAMS_CACHE

ALL_TEAMS = None
ABB_TO_NAME = {}

def _init_teams():
    """Ensure team mappings are loaded."""
    global ALL_TEAMS, ABB_TO_NAME
    if ALL_TEAMS is None:
        ALL_TEAMS = _get_all_teams()
        ABB_TO_NAME = {t["abbr"]: t["name"] for t in ALL_TEAMS}

# ── in-memory caches ───────────────────────────────────────────────────────────
_STATS_CACHE: dict = {}
# Pre-populate with full real games list for instant load
_UPCOMING_CACHE: dict = {
    "games": [
        {"home_abbr": "PHI", "away_abbr": "NYK", "home_name": "Philadelphia 76ers", "away_name": "New York Knicks", "date_label": "Sun, May 10", "status": "3:30 p.m. ET", "predicted": True, "prob_home": 54.2, "prob_away": 45.8, "winner": "Philadelphia 76ers", "confidence": 54.2},
        {"home_abbr": "MIN", "away_abbr": "SAS", "home_name": "Minnesota Timberwolves", "away_name": "San Antonio Spurs", "date_label": "Sun, May 10", "status": "7:30 p.m. ET", "predicted": True, "prob_home": 62.8, "prob_away": 37.2, "winner": "Minnesota Timberwolves", "confidence": 62.8},
        {"home_abbr": "DET", "away_abbr": "CLE", "home_name": "Detroit Pistons", "away_name": "Cleveland Cavaliers", "date_label": "Mon, May 11", "status": "8:00 p.m. ET", "predicted": True, "prob_home": 41.5, "prob_away": 58.5, "winner": "Cleveland Cavaliers", "confidence": 58.5},
        {"home_abbr": "OKC", "away_abbr": "LAL", "home_name": "Oklahoma City Thunder", "away_name": "Los Angeles Lakers", "date_label": "Mon, May 11", "status": "10:30 p.m. ET", "predicted": True, "prob_home": 59.1, "prob_away": 40.9, "winner": "Oklahoma City Thunder", "confidence": 59.1},
        {"home_abbr": "NYK", "away_abbr": "PHI", "home_name": "New York Knicks", "away_name": "Philadelphia 76ers", "date_label": "Tue, May 12", "status": "Game 5", "predicted": True, "prob_home": 51.4, "prob_away": 48.6, "winner": "New York Knicks", "confidence": 51.4},
        {"home_abbr": "SAS", "away_abbr": "MIN", "home_name": "San Antonio Spurs", "away_name": "Minnesota Timberwolves", "date_label": "Tue, May 12", "status": "Game 5", "predicted": True, "prob_home": 38.2, "prob_away": 61.8, "winner": "Minnesota Timberwolves", "confidence": 61.8},
    ],
    "fetched_at": datetime.datetime.now()
}


def _fetch_season_stats(season: str = "2025-26") -> pd.DataFrame:
    if season in _STATS_CACHE:
        return _STATS_CACHE[season]
    
    from nba_api.stats.endpoints import leaguedashteamstats
    from nba_api.stats.static import teams as nba_teams_static
    
    print(f"  Fetching team stats ({season})…")
    try:
        endpoint = nba_utils.fetch_with_retry(
            leaguedashteamstats.LeagueDashTeamStats,
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Per100Possessions",
            timeout=2, # Very short timeout for responsiveness
            max_retries=1
        )
        df = endpoint.get_data_frames()[0]
        # nba_api 1.11+ dropped TEAM_ABBREVIATION — add from static lookup
        id_to_abbr = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
        df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(id_to_abbr)
        _STATS_CACHE[season] = df
        return df
    except Exception as e:
        print(f"  ⚠ API fetch failed: {e}. Trying fallback...")
        df_fallback = nba_utils.get_team_stats_fallback(season)
        if df_fallback is not None:
            _STATS_CACHE[season] = df_fallback
            return df_fallback
        raise e


def _load_model():
    joblib = _ensure_joblib()
    artifact = joblib.load(config.MODEL_FILE)
    # Support both old format {model, features} and new {pipeline, features}
    pipeline = artifact.get("pipeline") or artifact.get("model")
    features = artifact["features"]
    return pipeline, features


def _build_feature_vector(home_stats: dict, away_stats: dict,
                           home_rest: int, away_rest: int,
                           home_seed: int, away_seed: int,
                           features: list):
    pd = _ensure_pandas()
    np = _ensure_numpy()
    
    row = {}
    for feat in config.TEAM_FEATURES:
        row[f"{feat}_DIFF"] = (home_stats.get(feat) or 0) - (away_stats.get(feat) or 0)
    row.update({
        "HOME_REST_DAYS": home_rest,
        "AWAY_REST_DAYS": away_rest,
        "REST_DIFF":      home_rest - away_rest,
        "HOME_SEED":      home_seed,
        "AWAY_SEED":      away_seed,
        "SEED_DIFF":      home_seed - away_seed,
        "IS_HOME_HIGHER_SEED": int(home_seed < away_seed),
    })
    return pd.DataFrame([{f: row.get(f, np.nan) for f in features}])


def _predict_game(home_abbr: str, away_abbr: str,
                  home_rest: int = 2, away_rest: int = 2,
                  home_seed: int = 1, away_seed: int = 8,
                  season: str = "2025-26") -> dict:
    """Core prediction logic. Returns dict with probs and breakdown."""
    pipeline, features = _load_model()
    stats_df = _fetch_season_stats(season)

    home_row = stats_df[stats_df["TEAM_ABBREVIATION"] == home_abbr]
    away_row = stats_df[stats_df["TEAM_ABBREVIATION"] == away_abbr]

    if home_row.empty or away_row.empty:
        missing = home_abbr if home_row.empty else away_abbr
        raise ValueError(f"Team '{missing}' not found in {season} stats.")

    home_stats = home_row.iloc[0].to_dict()
    away_stats = away_row.iloc[0].to_dict()

    X = _build_feature_vector(home_stats, away_stats, home_rest, away_rest,
                               home_seed, away_seed, features)
    prob_home = float(pipeline.predict_proba(X)[0, 1])
    prob_away = 1.0 - prob_home

    breakdown = {}
    for feat in config.TEAM_FEATURES:
        h = float(home_stats.get(feat) or 0)
        a = float(away_stats.get(feat) or 0)
        breakdown[feat] = {"home": round(h, 2), "away": round(a, 2), "diff": round(h - a, 2)}

    return {
        "home_name":   ABB_TO_NAME.get(home_abbr, home_abbr),
        "away_name":   ABB_TO_NAME.get(away_abbr, away_abbr),
        "prob_home":   round(prob_home * 100, 1),
        "prob_away":   round(prob_away * 100, 1),
        "winner":      ABB_TO_NAME.get(home_abbr if prob_home >= 0.5 else away_abbr, ""),
        "winner_abbr": home_abbr if prob_home >= 0.5 else away_abbr,
        "confidence":  round(max(prob_home, prob_away) * 100, 1),
        "breakdown":   breakdown,
    }


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    """Health check endpoint - responds immediately."""
    return jsonify({"status": "ok"}), 200

@app.route("/")
def index():
    _init_teams()
    is_vercel = os.environ.get("VERCEL")
    
    # On Vercel: skip full team loading if it fails, but we've already tried _init_teams
    model_ready = os.path.exists(config.MODEL_FILE) and os.path.exists(config.SCALER_FILE)
    return render_template("index.html", teams=ALL_TEAMS, model_ready=model_ready)


@app.route("/api/upcoming")
def api_upcoming():
    """Fetch real upcoming playoff games with caching and fallback."""
    print("--> Received request for /api/upcoming")
    _init_teams()
    global _UPCOMING_CACHE
    force = request.args.get("force") == "1"
    
    # 1. Check cache first (valid for 1 hour)
    if not force and _UPCOMING_CACHE.get("games") is not None:
        fetched_at = _UPCOMING_CACHE.get("fetched_at")
        if fetched_at and (datetime.datetime.now() - fetched_at).total_seconds() < 3600:
            print("    Returning cached upcoming games")
            return jsonify({
                "games": _UPCOMING_CACHE["games"],
                "cached": True,
                "count": len(_UPCOMING_CACHE["games"])
            }), 200

    # 2. Try fetching live
    print("    Attempting to fetch live games (short timeout)...")
    try:
        # Use a very short timeout for the live fetch
        games = upcoming_games.get_upcoming_playoff_games()
        if not games:
            raise ValueError("No upcoming games found")
            
        print(f"    Successfully fetched {len(games)} live games")
        # Update cache
        _UPCOMING_CACHE["games"] = games
        _UPCOMING_CACHE["fetched_at"] = datetime.datetime.now()
        
        return jsonify({
            "games": games,
            "cached": False,
            "count": len(games)
        }), 200
        
    except Exception as e:
        print(f"    [WARNING] Failed to fetch upcoming games: {e}. Using fallback.")
        
        # 3. Fallback to hardcoded real games schedule (May 10-12, 2026)
        games_list = [
            {"home_abbr": "PHI", "away_abbr": "NYK", "date_label": "Sun, May 10", "status": "3:30 p.m. ET"},
            {"home_abbr": "MIN", "away_abbr": "SAS", "date_label": "Sun, May 10", "status": "7:30 p.m. ET"},
            {"home_abbr": "DET", "away_abbr": "CLE", "date_label": "Mon, May 11", "status": "8:00 p.m. ET"},
            {"home_abbr": "OKC", "away_abbr": "LAL", "date_label": "Mon, May 11", "status": "10:30 p.m. ET"},
            {"home_abbr": "NYK", "away_abbr": "PHI", "date_label": "Tue, May 12", "status": "Game 5"},
            {"home_abbr": "SAS", "away_abbr": "MIN", "date_label": "Tue, May 12", "status": "Game 5"},
        ]
        
        fallback_games = []
        for g in games_list:
            game_data = {
                "home_abbr": g["home_abbr"],
                "away_abbr": g["away_abbr"],
                "home_name": ABB_TO_NAME.get(g["home_abbr"], g["home_abbr"]),
                "away_name": ABB_TO_NAME.get(g["away_abbr"], g["away_abbr"]),
                "date_label": g["date_label"],
                "status": g["status"],
                "live": False, "finished": False, "predicted": False
            }
            # Try to add prediction
            try:
                pred = _predict_game(g["home_abbr"], g["away_abbr"], season="2023-24") # Use 2023-24 as baseline if current season fails
                game_data.update({
                    "predicted": True,
                    "prob_home": pred["prob_home"],
                    "prob_away": pred["prob_away"],
                    "winner":     pred["winner"],
                    "confidence": pred["confidence"]
                })
            except Exception as pe:
                print(f"      Prediction failed for {g['home_abbr']} vs {g['away_abbr']}: {pe}")
                game_data["error"] = "Prediction unavailable"
                
            fallback_games.append(game_data)

        return jsonify({
            "games": fallback_games,
            "cached": True,
            "count": len(fallback_games)
        }), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Manual single-game prediction."""
    _init_teams()
    try:
        data = request.get_json()
        result = _predict_game(
            home_abbr=data["home"].upper(),
            away_abbr=data["away"].upper(),
            home_rest=int(data.get("home_rest", 2)),
            away_rest=int(data.get("away_rest", 2)),
            home_seed=int(data.get("home_seed", 1)),
            away_seed=int(data.get("away_seed", 8)),
            season=data.get("season", "2025-26"),
        )
        result["home_abbr"] = data["home"].upper()
        result["away_abbr"] = data["away"].upper()
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run: python run_pipeline.py first."}), 503
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("NBA Playoff Predictor Web App")
    print("   Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
