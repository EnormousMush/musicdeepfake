import pandas as pd

df = pd.read_csv('annotations_final.csv', sep='\t')

# Genre-like tags in the dataset
genre_tags = ['rock', 'pop', 'classical', 'jazz', 'electronic', 'country',
              'metal', 'blues', 'folk', 'ambient', 'punk', 'techno',
              'reggae', 'disco', 'house', 'trance', 'hip hop', 'funk',
              'indie', 'hard rock', 'soft rock', 'heavy metal', 'electronica',
              'new age', 'baroque', 'celtic', 'industrial', 'jungle']

# Keep only genre tags that actually exist in the columns
genre_tags = [g for g in genre_tags if g in df.columns]

# For each track, count how many genre tags it has
df['genre_count'] = df[genre_tags].sum(axis=1)

# Filter to tracks with exactly one genre tag
single = df[df['genre_count'] == 1]

# Find which genre each single-genre track has
for g in genre_tags:
    count = single[g].sum()
    if count > 0:
        print(f'{g}: {int(count)}')

print(f'\nTotal single-genre tracks: {len(single)}')
print(f'Total tracks in dataset: {len(df)}')