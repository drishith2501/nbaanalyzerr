"""
Minimal Flask app for Vercel - returns hardcoded data without heavy imports
"""
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api/upcoming")
def api_upcoming():
    """Return hardcoded games immediately"""
    return jsonify({
        "games": [
            {
                "home_abbr": "LAL", "away_abbr": "DEN",
                "home_name": "Los Angeles Lakers", "away_name": "Denver Nuggets",
                "date_label": "Thu, May 9", "status": "Upcoming",
                "live": False, "finished": False, "predicted": False
            },
            {
                "home_abbr": "BOS", "away_abbr": "MIA",
                "home_name": "Boston Celtics", "away_name": "Miami Heat",
                "date_label": "Thu, May 9", "status": "Upcoming",
                "live": False, "finished": False, "predicted": False
            },
            {
                "home_abbr": "DEN", "away_abbr": "OKC",
                "home_name": "Denver Nuggets", "away_name": "Oklahoma City Thunder",
                "date_label": "Fri, May 10", "status": "Upcoming",
                "live": False, "finished": False, "predicted": False
            },
        ],
        "cached": False,
        "count": 3
    }), 200

@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>NBA Playoff Predictor</title></head>
    <body>
        <h1>NBA Playoff Game Predictor</h1>
        <p>Loading...</p>
        <script>
            fetch('/api/upcoming')
                .then(r => r.json())
                .then(data => {
                    console.log('Games:', data);
                    document.body.innerHTML = '<h1>Games Loaded!</h1>' + 
                        JSON.stringify(data.games, null, 2);
                })
                .catch(e => {
                    console.error('Error:', e);
                    document.body.innerHTML = '<h1>Error: ' + e.message + '</h1>';
                });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True, port=5000)
