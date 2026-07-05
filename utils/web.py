import json
import urllib.request
import re # Not directly used by _get_json or _fmt_number, but kept in mind for other utils

def _get_json(url: str, timeout: int = 8):
    """Fetches a JSON response from the given URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "ask.py/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _fmt_number(n: float) -> str:
    """Formats a large number into a human-readable string (e.g., '1.23 trillion')."""
    if n >= 1e12:
        return f"{n / 1e12:.2f} trillion"
    if n >= 1e9:
        return f"{n / 1e9:.2f} billion"
    if n >= 1e6:
        return f"{n / 1e6:.2f} million"
    return f"{n:,.0f}"
