"""Answer factual questions by looking things up online (retrieval, not memorization).

This is deliberately NOT the transformer model in model.py — a small trained
network can't hold broad, current world knowledge. Instead this looks the
answer up live, the same idea behind "RAG" (retrieval-augmented generation):
fetch a real source, then show the relevant piece of it.

Sources tried in order:
  1. World Bank API   - precise numeric stats (GDP, population, etc.) by country
  2. Wikipedia REST API - short summary for general "what is X" questions
  3. DuckDuckGo Instant Answer API - fallback abstract for anything else

Usage:
    python ask.py "what is the gdp of china"
    python ask.py "what is the population of japan"
    python ask.py "who is the president of france"
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request

# Windows consoles default to cp1252, which can't print many Wikipedia
# characters (em dashes, the middle-dot in "x . y", etc). Force UTF-8 so
# real article text never crashes the script.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Common countries -> ISO3 code, for the World Bank lookup. Sorted longest-name
# first at lookup time so "south korea" matches before a shorter substring could.
COUNTRY_CODES = {
    "united states": "USA", "usa": "USA", "america": "USA",
    "united kingdom": "GBR", "uk": "GBR", "britain": "GBR",
    "south korea": "KOR", "north korea": "PRK", "south africa": "ZAF",
    "saudi arabia": "SAU", "new zealand": "NZL",
    "china": "CHN", "japan": "JPN", "germany": "DEU", "india": "IND",
    "france": "FRA", "italy": "ITA", "brazil": "BRA", "canada": "CAN",
    "russia": "RUS", "mexico": "MEX", "australia": "AUS", "spain": "ESP",
    "netherlands": "NLD", "sweden": "SWE", "norway": "NOR", "switzerland": "CHE",
    "turkey": "TUR", "egypt": "EGY", "nigeria": "NGA", "argentina": "ARG",
    "indonesia": "IDN", "pakistan": "PAK", "vietnam": "VNM", "thailand": "THA",
    "poland": "POL", "ukraine": "UKR", "greece": "GRC", "portugal": "PRT",
    "israel": "ISR", "ireland": "IRL", "singapore": "SGP", "philippines": "PHL",
    "colombia": "COL", "chile": "CHL", "peru": "PER",
}

# indicator keyword -> (World Bank indicator code, human label, is currency).
# Order matters: checked longest-key-first so "gdp per capita" beats plain "gdp".
INDICATORS = {
    "gdp per capita": ("NY.GDP.PCAP.CD", "GDP per capita (current US$)", True),
    "gdp": ("NY.GDP.MKTP.CD", "GDP (current US$)", True),
    "population": ("SP.POP.TOTL", "Population", False),
    "unemployment": ("SL.UEM.TOTL.ZS", "Unemployment rate (%)", False),
    "inflation": ("FP.CPI.TOTL.ZG", "Inflation rate (%)", False),
    "life expectancy": ("SP.DYN.LE00.IN", "Life expectancy (years)", False),
    "exports": ("NE.EXP.GNFS.CD", "Exports of goods and services (current US$)", True),
    "imports": ("NE.IMP.GNFS.CD", "Imports of goods and services (current US$)", True),
    "military spending": ("MS.MIL.XPND.CD", "Military expenditure (current US$)", True),
    "literacy rate": ("SE.ADT.LITR.ZS", "Adult literacy rate (%)", False),
    "internet users": ("IT.NET.USER.ZS", "Internet users (% of population)", False),
    "co2 emissions": ("EN.GHG.CO2.PC.CE.AR5", "CO2 emissions per capita (tonnes)", False),
    "electricity access": ("EG.ELC.ACCS.ZS", "Access to electricity (% of population)", False),
}


import urllib.error

def _get_json(url: str, timeout: int = 8):
    req = urllib.request.Request(url, headers={"User-Agent": "ask.py/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Error fetching URL {url}: {e}", file=sys.stderr)
        raise  # Re-raise the exception to be caught by the calling function's generic except
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from URL {url}: {e}", file=sys.stderr)
        raise  # Re-raise the exception


def _fmt_number(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.2f} trillion"
    if n >= 1e9:
        return f"{n / 1e9:.2f} billion"
    if n >= 1e6:
        return f"{n / 1e6:.2f} million"
    return f"{n:,.0f}"


def try_worldbank(question: str):
    """If the question names a known country + indicator, fetch a real number."""
    q = question.lower()
    country_names = sorted(COUNTRY_CODES, key=len, reverse=True)
    country_name = next((name for name in country_names if name in q), None)
    country_code = COUNTRY_CODES.get(country_name)

    indicator_keys = sorted(INDICATORS, key=len, reverse=True)
    indicator_key = next((k for k in indicator_keys if k in q), None)
    indicator = INDICATORS.get(indicator_key)
    if not country_code or not indicator:
        return None

    code, label, is_currency = indicator
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/{code}"
        f"?format=json&per_page=5"
    )
    data = _get_json(url)
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return None

    for entry in data[1]:
        if entry.get("value") is not None:
            value = entry["value"]
            year = entry["date"]
            shown = f"${_fmt_number(value)}" if is_currency else _fmt_number(value)
            country_name = entry["country"]["value"]
            return f"{label} of {country_name} ({year}): {shown}\n  Source: World Bank"
    return None


# Leading question scaffolding to strip, longest/most-specific patterns first.
_LEAD_PATTERNS = [
    r"^how (tall|big|long|old|far|much|wide|deep|high) (is|are|was|were) (the )?",
    r"^who (invented|discovered|created|founded|wrote|built|painted|designed) ",
    r"^(what|who|when|where|why) (is|are|was|were) (the )?",
    r"^when did (the )?",
]
# Trailing attribute/action words that are part of the question, not the entity.
_TRAILING_WORDS = {
    "built", "invented", "made", "founded", "discovered", "created", "born",
    "die", "died", "tall", "high", "long", "big", "old", "wide", "deep",
}


def _clean_topic(question: str) -> str:
    q = re.sub(r"[?.!]", "", question.lower()).strip()
    for pattern in _LEAD_PATTERNS:
        new_q = re.sub(pattern, "", q)
        if new_q != q:
            q = new_q.strip()
            break
    words = q.split()
    while words and words[-1] in _TRAILING_WORDS:
        words.pop()
    return " ".join(words)


def _wiki_search_title(query: str):
    """Use Wikipedia's real search index to resolve a fuzzy query to a title."""
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
        "&srlimit=1&srsearch=" + urllib.parse.quote(query)
    )
    data = _get_json(url)
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    return hits[0]["title"]


def try_wikipedia(question: str):
    """Resolve the likely topic via search, then pull its summary."""
    topic = _clean_topic(question)
    if not topic:
        return None

    try:
        title = _wiki_search_title(topic)
    except Exception as e:
        print(f"Error in _wiki_search_title: {e}", file=sys.stderr)
        title = None
    if not title:
        return None

    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    try:
        data = _get_json(url)
    except Exception as e:
        print(f"Error getting Wikipedia summary: {e}", file=sys.stderr)
        return None
    extract = data.get("extract")
    if not extract:
        return None
    page_title = data.get("title", title)
    return f"{extract}\n  Source: Wikipedia ({page_title})"


def try_duckduckgo(question: str):
    """Last resort: DuckDuckGo's instant-answer abstract."""
    url = (
        "https://api.duckduckgo.com/?q=" + urllib.parse.quote(question)
        + "&format=json&no_html=1&skip_disambig=1"
    )
    data = _get_json(url)
    abstract = data.get("AbstractText")
    if not abstract:
        return None
    source = data.get("AbstractSource", "DuckDuckGo")
    return f"{abstract}\n  Source: {source}"


def answer(question: str) -> str:
    for lookup in (try_worldbank, try_wikipedia, try_duckduckgo):
        try:
            result = lookup(question)
        except Exception:
            result = None
        if result:
            return result
    return "Couldn't find a confident answer to that online."


def main():
    parser = argparse.ArgumentParser(description="Answer a factual question by looking it up.")
    parser.add_argument("question", nargs="*", help="the question to ask")
    args = parser.parse_args()

    question = " ".join(args.question) if args.question else input("Ask something: ")
    print(answer(question))


if __name__ == "__main__":
    sys.exit(main() or 0)
