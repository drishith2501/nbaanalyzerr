
import time
import pandas as pd
import os
import config

# Robust headers to avoid bot detection
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Connection': 'keep-alive',
    'Referer': 'https://stats.nba.com/',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}

def fetch_with_retry(endpoint_class, max_retries=3, delay=1.0, timeout=10, **kwargs):
    """
    Calls an nba_api endpoint with custom headers, retries, and timeout.
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            # Inject headers into kwargs if supported by the endpoint
            instance = endpoint_class(headers=HEADERS, timeout=timeout, **kwargs)
            time.sleep(delay)
            return instance
        except Exception as e:
            last_exception = e
            wait_time = delay * (attempt + 1) * 2
            print(f"  [ATTEMPT {attempt+1} FAILED]: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    print(f"  [ERROR] All {max_retries} attempts failed for {endpoint_class.__name__}")
    raise last_exception

def get_team_stats_fallback(season):
    """
    Tries to load team stats from local raw_team_stats.csv if API fails.
    """
    if os.path.exists(config.RAW_TEAM_STATS_FILE):
        try:
            df = pd.read_csv(config.RAW_TEAM_STATS_FILE)
            if "SEASON" in df.columns:
                season_df = df[df["SEASON"] == season]
                if not season_df.empty:
                    print(f"  [INFO] Using cached stats for {season} from {config.RAW_TEAM_STATS_FILE}")
                    return season_df
            print(f"  [WARNING] No cached stats found for season {season} in {config.RAW_TEAM_STATS_FILE}")
        except Exception as e:
            print(f"  [ERROR] Error reading cache file: {e}")
    return None
