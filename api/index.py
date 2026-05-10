"""
Vercel serverless WSGI application (v2)
Returns hardcoded data immediately to avoid NBA API timeouts
"""
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class VercelWSGI:
    """Lightweight WSGI app for Vercel that short-circuits slow API calls"""
    
    def __init__(self):
        self.flask_app = None
    
    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '/')
        
        # CRITICAL: Return API data immediately without calling NBA API
        if path == '/api/upcoming':
            response_data = {
                "games": [
                    {
                        "home_abbr": "LAL", "away_abbr": "DEN",
                        "home_name": "Los Angeles Lakers", "away_name": "Denver Nuggets",
                        "date_label": "Thu, May 9", "status": "Upcoming",
                        "live": False, "finished": False,
                        "predicted": False
                    },
                    {
                        "home_abbr": "BOS", "away_abbr": "MIA",
                        "home_name": "Boston Celtics", "away_name": "Miami Heat",
                        "date_label": "Thu, May 9", "status": "Upcoming",
                        "live": False, "finished": False,
                        "predicted": False
                    },
                    {
                        "home_abbr": "DEN", "away_abbr": "OKC",
                        "home_name": "Denver Nuggets", "away_name": "Oklahoma City Thunder",
                        "date_label": "Fri, May 10", "status": "Upcoming",
                        "live": False, "finished": False,
                        "predicted": False
                    },
                ],
                "cached": False,
                "count": 3
            }
            body = json.dumps(response_data).encode('utf-8')
            start_response('200 OK', [
                ('Content-Type', 'application/json'),
                ('Content-Length', str(len(body))),
                ('Cache-Control', 'max-age=300')
            ])
            return [body]
        
        # For all other routes, use Flask
        try:
            if self.flask_app is None:
                from app import app
                self.flask_app = app
            return self.flask_app(environ, start_response)
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode('utf-8')
            start_response('500 Internal Server Error', [
                ('Content-Type', 'application/json'),
                ('Content-Length', str(len(body)))
            ])
            return [body]


# Export the WSGI app
app = VercelWSGI()

