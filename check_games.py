from upcoming_games import get_upcoming_playoff_games
import datetime

print(f"Checking for games from {datetime.date.today()} to {datetime.date.today() + datetime.timedelta(days=20)}")
games = get_upcoming_playoff_games(days_ahead=20)
print(f"Total games found: {len(games)}")
for g in games:
    if 'LAL' in (g['home_abbr'], g['away_abbr']) and 'OKC' in (g['home_abbr'], g['away_abbr']):
        print(f"{g['date_label']} - {g['away_abbr']} @ {g['home_abbr']} ({g['status']}) - ID: {g['game_id']}")
