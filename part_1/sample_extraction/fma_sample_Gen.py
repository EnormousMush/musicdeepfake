import requests
from bs4 import BeautifulSoup
import os
import time
import re

GENRE_URL = "https://freemusicarchive.org/genre/Jazz_Vocal"
DOWNLOAD_DIR = "fma_jazz_vocal"
TOTAL_TRACKS = 100
TRACKS_PER_PAGE = 20
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Paste your browser cookies here after logging into freemusicarchive.org
# DevTools → Application → Cookies → freemusicarchive.org
COOKIES = {
    "free_music_archive_session": "eyJpdiI6Im1QRE41MVVSdzBUbVFrKzBxTFEyc0E9PSIsInZhbHVlIjoia2lqZ0Y0SkM1dkxiNlhoZi9NcGRqRURCOWJOR3RkWXNHZnlRTitaM0NrVHlSZ0hOeTAzMkpRUXNod0hSZmd3c0ZpTG9LYjNxbk56bzJBZTJETHdXY2N3b1A0QVUzZHhMbTkvVHo0OXJhVlVsSUpTaEN5TmNDMlZXQWgvUXZVQWkiLCJtYWMiOiI1OGJiN2VhNjY1MjZhMmFlYmIzYTM3ZDIxMzYzOTdjY2M5YmE2NmM1MzI3MjlhMzdlNmJjNTM2N2MzZDU0NDVkIn0%3D",
    "XSRF-TOKEN": "eyJpdiI6IlRHR2dBdGhTS0xKUTEwSXZ4RXNaZmc9PSIsInZhbHVlIjoiUkdkb0pzam0wbi92akM0NUxzZkFBNWNIZ3ZDZm83d3BHQ2QvTG5BSW5rOVdzRzhDYTZNR3J3bkU3ZXl2RlNkdmFvWUVCeUhwUGM4R1lLeWVGNHFlaHVJMnp4MzlZV0NUM3hxajBGTEJoL1g0R0t0N1IvVlY1ektwcTN6OWFWZGkiLCJtYWMiOiI0ZjExN2UyMDBhZDAyYmYwMWJjZWMwMGZkNTZlZmIyZDBkYWEzZTY1ZDAxN2JmN2VmMmQ4OGZiNzA5NWIxYzIzIn0%3D",
}

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def get_track_links_from_page(page_num):
    """Scrape track page URLs from a genre listing page."""
    url = f"{GENRE_URL}?pageSize=20&page={page_num}&search-genre=Jazz%3A%20Vocal&sort=_score&d=0"
    resp = requests.get(url, headers=HEADERS, cookies=COOKIES)
    soup = BeautifulSoup(resp.text, "html.parser")

    track_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Track pages have 4 path segments: /music/artist/album/track/
        parts = href.replace("https://freemusicarchive.org", "").strip("/").split("/")
        if len(parts) == 4 and parts[0] == "music":
            if href not in track_links:
                track_links.append(href)
    return track_links

def get_download_url(track_page_url):
    """Given a track page URL, return the direct download URL."""
    # Extract the track slug (last path segment)
    slug = track_page_url.strip("/").split("/")[-1]
    return f"https://freemusicarchive.org/track/{slug}/download/"

def download_track(download_url, filename):
    """Download the MP3 file."""
    resp = requests.get(download_url, headers=HEADERS, cookies=COOKIES, stream=True, allow_redirects=True)
    if resp.status_code == 200 and "audio" in resp.headers.get("Content-Type", ""):
        with open(filename, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

# --- Main ---
all_track_links = []
page = 1

print("Collecting track links...")
while len(all_track_links) < TOTAL_TRACKS:
    links = get_track_links_from_page(page)
    if not links:
        print(f"No more tracks found at page {page}.")
        break
    all_track_links.extend(links)
    print(f"  Page {page}: found {len(links)} tracks (total so far: {len(all_track_links)})")
    page += 1
    time.sleep(1)  # Be polite to the server

all_track_links = all_track_links[:TOTAL_TRACKS]
print(f"\nDownloading {len(all_track_links)} tracks to '{DOWNLOAD_DIR}/'...\n")

success = 0
for i, track_url in enumerate(all_track_links):
    slug = track_url.strip("/").split("/")[-1]
    filename = os.path.join(DOWNLOAD_DIR, f"{i+1:03d}_{slug}.mp3")

    if os.path.exists(filename):
        print(f"[{i+1:3d}/100] Already exists, skipping: {slug}")
        success += 1
        continue

    download_url = get_download_url(track_url)
    print(f"[{i+1:3d}/100] Downloading: {slug}")

    ok = download_track(download_url, filename)
    if ok:
        success += 1
        print(f"         ✓ Saved")
    else:
        print(f"         ✗ Failed (may require login or track unavailable)")

    time.sleep(0.5)  # Be polite

print(f"\nDone! {success}/{len(all_track_links)} tracks downloaded to '{DOWNLOAD_DIR}/'")