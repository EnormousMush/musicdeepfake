"""Debug: try submitting with minimal body to isolate what's causing 403."""
import os, sys, json
import requests

api_key = os.environ.get("TTAPI_KEY")
if not api_key:
    sys.exit("TTAPI_KEY not set")

headers = {"TT-API-KEY": api_key, "Content-Type": "application/json"}

tests = [
    ("chirp-v4-5+ no extras", {
        "custom": False, "instrumental": True, "mv": "chirp-v4-5+",
        "gpt_description_prompt": "smooth jazz piano",
        "tags": "jazz, piano",
    }),
    ("chirp-v5 no extras", {
        "custom": False, "instrumental": True, "mv": "chirp-v5",
        "gpt_description_prompt": "smooth jazz piano",
        "tags": "jazz, piano",
    }),
    ("chirp-v5 with title", {
        "custom": False, "instrumental": True, "mv": "chirp-v5",
        "gpt_description_prompt": "smooth jazz piano",
        "tags": "jazz, piano",
        "title": "jazz test",
    }),
    ("chirp-v5 with negativeTags", {
        "custom": False, "instrumental": True, "mv": "chirp-v5",
        "gpt_description_prompt": "smooth jazz piano",
        "tags": "jazz, piano",
        "title": "jazz test",
        "negativeTags": "vocals, voice, singing",
    }),
]

for name, body in tests:
    resp = requests.post(
        "https://api.ttapi.io/suno/v1/music",
        headers=headers, json=body, timeout=30,
    )
    print(f"[{name}] HTTP {resp.status_code}: {resp.text[:200]}")
    if resp.status_code == 200:
        print("  -> SUCCESS, stopping here (job submitted, no need to continue)")
        break
    print()
