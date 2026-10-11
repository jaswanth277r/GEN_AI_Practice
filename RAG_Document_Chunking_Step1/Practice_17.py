# Sliding window over sentences (window=3, stride=1)
import re

article_text = """A balanced meal contains carbohydrates, proteins, and fats in sensible proportions. Carbohydrates supply quick
energy, proteins repair muscles and tissues, and fats help the body absorb certain vitamins.

Vegetables and fruits provide fibre, vitamins, and minerals. Eating a variety of colours across the week is a
simple way to cover a wide range of nutrients.

Whole grains such as brown rice, oats, and millet keep you full for longer than refined flour because the fibre
slows digestion. Pairing grains with lentils or beans gives a complete protein profile.

Drinking enough water supports digestion and concentration. Sugary drinks add calories without nutrients, so
water, buttermilk, or unsweetened tea are better daily choices.

Reading food labels helps you compare products. Look at serving size first, then at sugar, sodium, and
saturated fat, since those are the values people most often underestimate."""

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def sliding_window(text, window=3, stride=1):
    s = split_sentences(text)
    return [" ".join(s[i:i + window]) for i in range(0, max(1, len(s) - window + 1), stride)]

windows = sliding_window(article_text)
print("Total sentences:", len(split_sentences(article_text)))
print("Windows produced:", len(windows))
for i, w in enumerate(windows[:3]):
    print(f"\nWindow {i}: {w}")
