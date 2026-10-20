"""Test setup: gives the app a complete fake .env so it can be imported without real keys.

The values come from .env.example, and every empty key gets a harmless fake value.
This file runs before any test, so app.config never reads your real .env during the tests.
"""
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for name, value in dotenv_values(ROOT / ".env.example").items():
    os.environ[name] = value or "test-value"
os.environ["GOOGLE_CALENDAR_API_TOKEN"] = "ya29.fake-test-token"
os.environ["GOOGLE_CALENDAR_ID"] = "primary"
