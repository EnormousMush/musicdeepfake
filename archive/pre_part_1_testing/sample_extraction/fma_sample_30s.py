import os
import random
from pydub import AudioSegment

INPUT_DIR = "fma_jazz_vocal"
OUTPUT_DIR = "fma_jazz_vocal_30s"
CLIP_DURATION_MS = 30_000  # 30 seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mp3")]
print(f"Found {len(files)} files in '{INPUT_DIR}/'\n")

for i, filename in enumerate(files):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path):
        print(f"[{i+1:3d}/{len(files)}] Skipping (already exists): {filename}")
        continue

    audio = AudioSegment.from_mp3(input_path)
    duration_ms = len(audio)

    if duration_ms <= CLIP_DURATION_MS:
        # File is shorter than 30s, just copy it as-is
        clip = audio
        print(f"[{i+1:3d}/{len(files)}] Short file, using full track: {filename}")
    else:
        max_start = duration_ms - CLIP_DURATION_MS
        start_ms = random.randint(0, max_start)
        clip = audio[start_ms : start_ms + CLIP_DURATION_MS]
        print(f"[{i+1:3d}/{len(files)}] Cut {start_ms//1000}s–{(start_ms + CLIP_DURATION_MS)//1000}s: {filename}")

    clip.export(output_path, format="mp3")

print(f"\nDone! Clips saved to '{OUTPUT_DIR}/'")
