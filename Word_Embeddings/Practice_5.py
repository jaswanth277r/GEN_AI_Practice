# Word2Vec with more dimensions, a wider window and many epochs
from gensim.models import Word2Vec

corpus = [['rice', 'is', 'a', 'staple', 'grain'], ['wheat', 'is', 'a', 'staple', 'grain'], ['lentil', 'is', 'a', 'protein', 'rich', 'legume'], ['chickpea', 'is', 'a', 'protein', 'rich', 'legume'], ['rice', 'needs', 'water', 'and', 'heat'], ['wheat', 'needs', 'water', 'and', 'heat'], ['lentil', 'needs', 'soaking', 'and', 'heat'], ['chickpea', 'needs', 'soaking', 'and', 'heat'], ['yogurt', 'is', 'a', 'fermented', 'food'], ['idli', 'is', 'a', 'fermented', 'food']]

model = Word2Vec(corpus, vector_size=20, window=3, min_count=1, workers=1, epochs=200, seed=42)

print("Vocabulary size:", len(model.wv))
print("Vector shape for 'rice':", model.wv["rice"].shape)
print("Top 4 neighbours of 'rice':")
for word, score in model.wv.most_similar("rice", topn=4):
    print(f"   {word:<12} {score:.3f}")
