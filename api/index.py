"""
Vercel serverless function handler for Flask app
This wraps the Flask app to work with Vercel's Python runtime
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app

# Export the app for Vercel
__all__ = ['app']
