"""Quick debug: manually fetch one job and print the raw response."""
import csv, json, os, sys
import requests

api_key = os.environ.get("TTAPI_KEY")
if not api_key:
    sys.exit("TTAPI_KEY not set")

with open("test_manifest.csv") as f:
    rows = list(csv.DictReader(f))

submitted = [r for r in rows if r["status"] == "submitted"]
if not submitted:
    sys.exit("No submitted jobs in test_manifest.csv")

row = submitted[0]
job_id = row["ttapi_job_id"]
print(f"Fetching job: {job_id}  (genre: {row['genre']})\n")

resp = requests.post(
    "https://api.ttapi.io/suno/v1/fetch",
    headers={"TT-API-KEY": api_key, "Content-Type": "application/json"},
    json={"jobId": job_id},
    timeout=30,
)
print(f"HTTP {resp.status_code}")
print(json.dumps(resp.json(), indent=2))
