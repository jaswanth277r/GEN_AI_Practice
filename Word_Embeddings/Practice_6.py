# Pairwise similarity matrix for a handful of words
import pandas as pd
from gensim.models import Word2Vec

sentences = [['rice', 'is', 'a', 'staple', 'grain'], ['wheat', 'is', 'a', 'staple', 'grain'], ['lentil', 'is', 'a', 'protein', 'rich', 'legume'], ['chickpea', 'is', 'a', 'protein', 'rich', 'legume'], ['rice', 'needs', 'water', 'and', 'heat'], ['wheat', 'needs', 'water', 'and', 'heat'], ['lentil', 'needs', 'soaking', 'and', 'heat'], ['chickpea', 'needs', 'soaking', 'and', 'heat'], ['yogurt', 'is', 'a', 'fermented', 'food'], ['idli', 'is', 'a', 'fermented', 'food']]
model = Word2Vec(sentences, vector_size=12, window=2, min_count=1, workers=1, seed=5, epochs=120)

chosen = sorted({w for pair in [('rice', 'wheat'), ('lentil', 'chickpea'), ('rice', 'lentil')] for w in pair})
table = pd.DataFrame(
    [[model.wv.similarity(a, b) for b in chosen] for a in chosen],
    index=chosen, columns=chosen,
).round(2)
print(table)
