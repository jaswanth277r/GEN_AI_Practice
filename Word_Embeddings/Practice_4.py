# One-hot encoding shown as a labelled DataFrame
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

raw = ['rice', 'wheat', 'millet', 'oats', 'rice', 'lentil', 'millet', 'corn']
enc = OneHotEncoder(sparse_output=False)
arr = enc.fit_transform([[w] for w in raw])

df = pd.DataFrame(arr.astype(int), columns=enc.get_feature_names_out(["word"]), index=raw)
print(df)
print("\nVector size:", df.shape[1])

# Repeated words must produce identical rows
dups = [w for w in set(raw) if raw.count(w) > 1]
for w in dups:
    rows = df.loc[w]
    same = (rows.values == rows.values[0]).all()
    print(f"'{w}' appears {raw.count(w)} times and all rows identical: {same}")
