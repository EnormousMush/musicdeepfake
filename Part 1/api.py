"""
TTAPI client. Thin wrapper around the two endpoints used.
Callers are responsible for retry / backoff logic.
"""
import requests

MUSIC_URL = "https://api.ttapi.io/suno/v1/music"
FETCH_URL = "https://api.ttapi.io/suno/v1/fetch"


class TTAPIClient:
    def __init__(self, api_key: str):
        self._headers = {
            "TT-API-KEY": api_key,
            "Content-Type": "application/json",
        }

    def submit(self, prompt: str, tags: str, mv: str, instrumental: bool) -> str:
        """POST /music. Returns ttapi jobId string."""
        body = {
            "custom": False,
            "instrumental": instrumental,
            "mv": mv,
            "gpt_description_prompt": prompt,
            "tags": tags,
        }
        resp = requests.post(MUSIC_URL, headers=self._headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("data", {}).get("jobId") or data.get("data", {}).get("job_id")
        if not job_id:
            raise ValueError(f"No jobId in response: {data}")
        return job_id

    def fetch(self, ttapi_job_id: str) -> dict:
        """POST /fetch. Returns full response dict; caller checks ['status']."""
        resp = requests.post(
            FETCH_URL,
            headers=self._headers,
            json={"jobId": ttapi_job_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
