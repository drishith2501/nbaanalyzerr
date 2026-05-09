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

# ── teams lookup ───────────────────────────────────────────────────────────────
from nba_api.stats.static import teams as nba_teams_static
ALL_TEAMS = sorted(
    [{"abbr": t["abbreviation"], "name": t["full_name"]} for t in nba_teams_static.get_teams()],
    key=lambda x: x["name"],
)
ABB_TO_NAME = {t["abbr"]: t["name"] for t in ALL_TEAMS}

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

@app.route("/")
def index():
    model_ready = os.path.exists(config.MODEL_FILE) and os.path.exists(config.SCALER_FILE)
    return render_template("index.html", teams=ALL_TEAMS, model_ready=model_ready)


@app.route("/api/upcoming")
def api_upcoming():
    """
    Fetch upcoming playoff games (today + 10 days) and run predictions on each.
    Results cached for 15 minutes to avoid hammering the API.
    """
    try:
        force = request.args.get("force") == "1"
        now   = datetime.datetime.utcnow()
        season = request.args.get("season", "2025-26")

        # Use cache if less than 15 minutes old and not forced
        cached = _UPCOMING_CACHE
        if (not force
                and cached["games"] is not None
                and cached["fetched_at"] is not None
                and (now - cached["fetched_at"]).seconds < 900):
            return jsonify({"games": cached["games"], "cached": True})

        model_ready = os.path.exists(config.MODEL_FILE) and os.path.exists(config.SCALER_FILE)

        from upcoming_games import get_upcoming_playoff_games
        raw_games = get_upcoming_playoff_games(days_ahead=3)

        enriched = []
        for g in raw_games:
            entry = dict(g)
            if model_ready:
                try:
                    pred = _predict_game(
                        g["home_abbr"], g["away_abbr"],
                        home_rest=2, away_rest=2,   # default; no injury/rest API available
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

            entry["home_name"] = ABB_TO_NAME.get(g["home_abbr"], g["home_abbr"])
            entry["away_name"] = ABB_TO_NAME.get(g["away_abbr"], g["away_abbr"])
            enriched.append(entry)

        cached["games"] = enriched
        cached["fetched_at"] = now

        return jsonify({"games": enriched, "cached": False, "count": len(enriched)})

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


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
