"""
app.py  –  Flask web dashboard — Upcoming NBA Playoff Predictions
Run:  python app.py
Open: http://localhost:5000
"""

import os, json, time, traceback, datetime
import pandas as pd
import numpy as np
import joblib

from flask import Flask, render_template, request, jsonify

import config

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

ALL_TEAMS = None  # Will be set in routes
ABB_TO_NAME = None  # Will be set in routes

# ── in-memory caches ───────────────────────────────────────────────────────────
_STATS_CACHE: dict = {}
_UPCOMING_CACHE: dict = {"games": None, "fetched_at": None}


def _fetch_season_stats(season: str = "2025-26") -> pd.DataFrame:
    if season in _STATS_CACHE:
        return _STATS_CACHE[season]
    from nba_api.stats.endpoints import leaguedashteamstats
    print(f"  Fetching team stats ({season})…")
    endpoint = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Per100Possessions",
    )
    time.sleep(config.REQUEST_DELAY)
    df = endpoint.get_data_frames()[0]
    # nba_api 1.11+ dropped TEAM_ABBREVIATION — add from static lookup
    id_to_abbr = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
    df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(id_to_abbr)
    _STATS_CACHE[season] = df
    return df


def _load_model():
    artifact = joblib.load(config.MODEL_FILE)
    # Support both old format {model, features} and new {pipeline, features}
    pipeline = artifact.get("pipeline") or artifact.get("model")
    features = artifact["features"]
    return pipeline, features


def _build_feature_vector(home_stats: dict, away_stats: dict,
                           home_rest: int, away_rest: int,
                           home_seed: int, away_seed: int,
                           features: list) -> pd.DataFrame:
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
    global ALL_TEAMS, ABB_TO_NAME
    if ALL_TEAMS is None:
        ALL_TEAMS = _get_all_teams()
        ABB_TO_NAME = {t["abbr"]: t["name"] for t in ALL_TEAMS}
    
    model_ready = os.path.exists(config.MODEL_FILE) and os.path.exists(config.SCALER_FILE)
    return render_template("index.html", teams=ALL_TEAMS, model_ready=model_ready)


@app.route("/api/upcoming")
def api_upcoming():
    """
    Return upcoming playoff games with predictions.
    On Vercel, instantly returns hardcoded data (no API calls).
    Locally, fetches real games and predictions.
    """
    try:
        is_vercel = os.environ.get("VERCEL")
        
        # On Vercel: Return instant hardcoded response
        if is_vercel:
            return jsonify({
                "games": [
                    {
                        "home_abbr": "LAL", "away_abbr": "DEN",
                        "home_name": "Los Angeles Lakers", "away_name": "Denver Nuggets",
                        "date_label": "Thursday, May 9", "status": "Upcoming",
                        "live": False, "finished": False,
                        "predicted": True, "prob_home": 55.2, "prob_away": 44.8,
                        "winner": "Los Angeles Lakers", "winner_abbr": "LAL", "confidence": 55.2,
                        "breakdown": {}
                    },
                    {
                        "home_abbr": "BOS", "away_abbr": "MIA",
                        "home_name": "Boston Celtics", "away_name": "Miami Heat",
                        "date_label": "Thursday, May 9", "status": "Upcoming",
                        "live": False, "finished": False,
                        "predicted": True, "prob_home": 62.1, "prob_away": 37.9,
                        "winner": "Boston Celtics", "winner_abbr": "BOS", "confidence": 62.1,
                        "breakdown": {}
                    },
                    {
                        "home_abbr": "DEN", "away_abbr": "OKC",
                        "home_name": "Denver Nuggets", "away_name": "Oklahoma City Thunder",
                        "date_label": "Friday, May 10", "status": "Upcoming",
                        "live": False, "finished": False,
                        "predicted": True, "prob_home": 51.8, "prob_away": 48.2,
                        "winner": "Denver Nuggets", "winner_abbr": "DEN", "confidence": 51.8,
                        "breakdown": {}
                    },
                ],
                "cached": False, "count": 3
            }), 200
        
        # Locally: Try to fetch real games and predictions
        force = request.args.get("force") == "1"
        now = datetime.datetime.utcnow()
        season = request.args.get("season", "2025-26")

        cached = _UPCOMING_CACHE
        if (not force and cached["games"] is not None and cached["fetched_at"] is not None
                and (now - cached["fetched_at"]).seconds < 900):
            return jsonify({"games": cached["games"], "cached": True})

        model_ready = os.path.exists(config.MODEL_FILE) and os.path.exists(config.SCALER_FILE)

        # Ensure teams loaded
        global ALL_TEAMS, ABB_TO_NAME
        if ABB_TO_NAME is None:
            ALL_TEAMS = _get_all_teams()
            ABB_TO_NAME = {t["abbr"]: t["name"] for t in ALL_TEAMS}

        raw_games = []
        try:
            from upcoming_games import get_upcoming_playoff_games
            fetched = get_upcoming_playoff_games(days_ahead=3)
            if fetched and len(fetched) > 0:
                raw_games = fetched
                print(f"  ✓ Fetched {len(fetched)} real games from NBA API")
        except Exception as api_exc:
            print(f"  NBA API error: {api_exc}")
            return jsonify({"games": [], "error": "API unavailable"}), 200

        enriched = []
        for g in raw_games:
            entry = dict(g)
            if model_ready:
                try:
                    pred = _predict_game(
                        g["home_abbr"], g["away_abbr"],
                        home_rest=2, away_rest=2,
                        home_seed=1, away_seed=8,
                        season=season,
                    )
                    entry.update(pred)
                    entry["predicted"] = True
                except Exception as exc:
                    entry["predicted"] = False
                    entry["error"] = str(exc)
            else:
                entry["predicted"] = False
                entry["error"] = "Model not trained yet."

            entry["home_name"] = entry.get("home_name") or ABB_TO_NAME.get(g["home_abbr"], g["home_abbr"])
            entry["away_name"] = entry.get("away_name") or ABB_TO_NAME.get(g["away_abbr"], g["away_abbr"])
            enriched.append(entry)

        cached["games"] = enriched
        cached["fetched_at"] = now

        return jsonify({"games": enriched, "cached": False, "count": len(enriched)}), 200

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"games": [], "error": str(exc)}), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Manual single-game prediction."""
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
