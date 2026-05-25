import pandas as pd

# Load with multi-level header
tracks = pd.read_csv('tracks.csv', index_col=0, header=[0, 1])

# Filter to "small" subset
small = tracks[tracks[('set', 'subset')] == 'large']

# List all genres in small
print("Genres in 'small':")
print(small[('track', 'genre_top')].value_counts())

print(f"\nTotal tracks in 'small': {len(small)}")