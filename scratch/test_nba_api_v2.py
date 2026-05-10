
from nba_api.stats.endpoints import leaguedashteamstats
import nba_utils
import time

try:
    print("Testing nba_api connection with custom headers...")
    endpoint = nba_utils.fetch_with_retry(
        leaguedashteamstats.LeagueDashTeamStats,
        season="2023-24",
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Per100Possessions",
    )
    df = endpoint.get_data_frames()[0]
    print(f"Success! Fetched {len(df)} teams.")
except Exception as e:
    print(f"Failed: {e}")
