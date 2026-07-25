genres = {}
total = 0
with open('raw_30s_cleantags_50artists.tsv', 'r') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        tags = parts[5:]
        has_mood = any(t.startswith('mood/theme---') for t in tags)
        genre_tags = [t.replace('genre---', '') for t in tags if t.startswith('genre---')]
        if has_mood and len(genre_tags) == 1:
            total += 1
            g = genre_tags[0]
            genres[g] = genres.get(g, 0) + 1

print('Single-genre tracks that also have mood/theme tags:')
for g, c in sorted(genres.items(), key=lambda x: -x[1]):
    print(f'  {g}: {c}')
print(f'\nUnique genres: {len(genres)}')
print(f'Total single-genre tracks with mood tags: {total}')