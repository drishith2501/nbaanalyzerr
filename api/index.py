"""
Vercel serverless function handler
This creates a lightweight WSGI app for Vercel deployment
"""
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Minimal WSGI app that doesn't import heavy dependencies upfront
def application(environ, start_response):
    """WSGI application for Vercel"""
    path = environ.get('PATH_INFO', '/')
    method = environ.get('REQUEST_METHOD', 'GET')
    
    # Handle API endpoints with minimal dependencies
    if path == '/api/upcoming' and method == 'GET':
        # Return hardcoded fallback immediately
        fallback_data = {
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
        response_body = json.dumps(fallback_data).encode('utf-8')
        start_response('200 OK', [
            ('Content-Type', 'application/json'),
            ('Content-Length', str(len(response_body)))
        ])
        return [response_body]
    
    # For all other routes, use the Flask app
    try:
        from app import app as flask_app
        return flask_app(environ, start_response)
    except Exception as e:
        error_response = json.dumps({"error": str(e)}).encode('utf-8')
        start_response('500 Internal Server Error', [
            ('Content-Type', 'application/json'),
            ('Content-Length', str(len(error_response)))
        ])
        return [error_response]

# Export as 'app' for Vercel's Python builder
app = application

