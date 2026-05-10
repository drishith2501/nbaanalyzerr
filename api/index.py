"""
Vercel serverless function handler for Flask app
This wraps the Flask app to work with Vercel's Python runtime
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app as flask_app

# Vercel expects ASGI app or handler function
# For Flask, we need to wrap it in ASGI
try:
    from asgiref.wsgi import WsgiToAsgi
    app = WsgiToAsgi(flask_app)
except ImportError:
    # Fallback: expose the Flask app directly
    app = flask_app

