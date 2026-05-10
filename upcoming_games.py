"""
upcoming_games.py
─────────────────
Fetches NBA playoff games from today onward using:
  - nba_api.live  → today's live/current games
  - ScoreboardV3  → future scheduled games (next N days)

Returns structured game dicts ready for prediction.
"""

import time
import datetime
from zoneinfo import ZoneInfo
import signal
from contextlib import contextmanager

import config
import nba_utils

ET = ZoneInfo("America/New_York")

# Playoff game IDs start with "004"
PLAYOFF_PREFIX = "004"

# API timeout in seconds (Vercel has 10s soft limit)
API_TIMEOUT = 8


def _timeout_handler(signum, frame):
    raise TimeoutError("NBA API request timed out")


def _today_et() -> datetime.date:
    return datetime.datetime.now(ET).date()


def _parse_live_games(include_finished: bool = False) -> list[dict]:
    """Pull today's games from the NBA live API."""
    from nba_api.live.nba.endpoints import scoreboard as live_sb

    try:
        # Set timeout alarm (Unix only; ignored on Windows/Vercel)
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(API_TIMEOUT)
        except (ValueError, AttributeError):
            pass  # signal.alarm not available on Windows/some systems
        
        data = nba_utils.fetch_with_retry(live_sb.ScoreBoard, timeout=3, max_retries=1).get_dict()
        games = data["scoreboard"]["games"]
        
        try:
            signal.alarm(0)  # Cancel alarm
        except (ValueError, AttributeError):
            pass
    except Exception as exc:
        print(f"  Live API error: {exc}")
        return []

    today = _today_et()
    date_label = today.strftime("%A, %b %d").replace(" 0", " ")
    results = []

    for g in games:
        gid = g.get("gameId", "")
        if not gid.startswith(PLAYOFF_PREFIX):
            continue

        status_id = g.get("gameStatus", 1)   # 1=scheduled, 2=live, 3=final
        if status_id == 3 and not include_finished:
            continue

        home = g["homeTeam"]
        away = g["awayTeam"]
        status_text = g.get("gameStatusText", "").strip()

        results.append({
            "game_id":    gid,
            "date":       today.isoformat(),
            "date_label": date_label,
            "home_abbr":  home["teamTricode"],
            "away_abbr":  away["teamTricode"],
            "home_name":  home.get("teamName", home["teamTricode"]),
            "away_name":  away.get("teamName", away["teamTricode"]),
            "status":     status_text,
            "live":       status_id == 2,
            "finished":   status_id == 3,
            "home_score": home.get("score"),
            "away_score": away.get("score"),
        })

    return results


def _parse_future_games(days_ahead: int = 10) -> list[dict]:
    """Pull scheduled games for the next N days using ScoreboardV3."""
    from nba_api.stats.endpoints import scoreboardv3

    today = _today_et()
    results = []

    for offset in range(-2, days_ahead + 1):
        check_date = today + datetime.timedelta(days=offset)
        date_str = check_date.strftime("%Y-%m-%d")
        date_label = check_date.strftime("%A, %b %d").replace(" 0", " ")

        try:
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(API_TIMEOUT)
            except (ValueError, AttributeError):
                pass
            
            sb3 = nba_utils.fetch_with_retry(
                scoreboardv3.ScoreboardV3,
                game_date=date_str,
                league_id="00",
                timeout=3,
                max_retries=1
            )
            games = sb3.get_dict()["scoreboard"]["games"]
            
            try:
                signal.alarm(0)
            except (ValueError, AttributeError):
                pass
                
        except Exception as exc:
            print(f"  ScoreboardV3 error for {date_str}: {exc}")
            continue

        for g in games:
            gid = g.get("gameId", "")
            if not gid.startswith(PLAYOFF_PREFIX):
                continue

            home = g["homeTeam"]
            away = g["awayTeam"]
            status_text = g.get("gameStatusText", "Scheduled").strip()

            results.append({
                "game_id":    gid,
                "date":       check_date.isoformat(),
                "date_label": date_label,
                "home_abbr":  home["teamTricode"],
                "away_abbr":  away["teamTricode"],
                "home_name":  home.get("teamName", home["teamTricode"]),
                "away_name":  away.get("teamName", away["teamTricode"]),
                "status":     status_text,
                "live":       False,
                "finished":   False,
            })

        if games:
            print(f"  {date_str}: {len([g for g in games if g.get('gameId','').startswith(PLAYOFF_PREFIX)])} playoff game(s)")

    return results


def get_upcoming_playoff_games(days_ahead: int = 10) -> list[dict]:
    """
    Returns all upcoming + live (non-finished) playoff games
    from today through the next `days_ahead` days.
    """
    today_games  = _parse_live_games(include_finished=True)
    future_games = _parse_future_games(days_ahead=days_ahead)

    all_games = today_games + future_games

    # Deduplicate by game_id
    seen = set()
    unique = []
    for g in all_games:
        if g["game_id"] not in seen:
            seen.add(g["game_id"])
            unique.append(g)

    return sorted(unique, key=lambda x: x["date"])

if __name__ == "__main__":
    print("Fetching upcoming playoff games...")
    games = get_upcoming_playoff_games()
    if not games:
        print("No upcoming playoff games found.")
    else:
        for g in games:
            print(f"{g['date_label']}: {g['away_abbr']} @ {g['home_abbr']} - {g['status']}")
