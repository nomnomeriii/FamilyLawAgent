from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

URL = "https://www.courtlistener.com/api/rest/v4/search/"

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick CourtListener connectivity/token check")
    parser.add_argument("--token", default=os.getenv("COURTLISTENER_TOKEN", ""), help="CourtListener token")
    parser.add_argument("--query", default="new york custody relocation best interests", help="Search query")
    args = parser.parse_args()

    token = (args.token or "").strip()
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"

    params = {"q": args.query, "type": "o", "order_by": "score desc", "page_size": 5}

    response = requests.get(URL, headers=headers, params=params, timeout=20)
    out = {
        "status_code": response.status_code,
        "url": response.url,
    }

    try:
        payload = response.json()
        out["result_count"] = len(payload.get("results", [])) if isinstance(payload, dict) else None
        if isinstance(payload, dict):
            out["top_cases"] = [
                (r.get("caseName") or r.get("case_name") or "Unknown") for r in payload.get("results", [])[:3]
            ]
        out["response_excerpt"] = payload if response.status_code < 400 else payload
    except Exception:
        out["response_excerpt"] = response.text[:1000]

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
