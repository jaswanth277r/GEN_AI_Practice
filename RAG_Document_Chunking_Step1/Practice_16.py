# Three chunk sizes side by side in a DataFrame
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

article_text = """Nutrition Basics for Everyday Cooking

A balanced meal contains carbohydrates, proteins, and fats in sensible proportions. Carbohydrates supply quick
energy, proteins repair muscles and tissues, and fats help the body absorb certain vitamins.

Vegetables and fruits provide fibre, vitamins, and minerals. Eating a variety of colours across the week is a
simple way to cover a wide range of nutrients.

Whole grains such as brown rice, oats, and millet keep you full for longer than refined flour because the fibre
slows digestion. Pairing grains with lentils or beans gives a complete protein profile.

Drinking enough water supports digestion and concentration. Sugary drinks add calories without nutrients, so
water, buttermilk, or unsweetened tea are better daily choices.

Reading food labels helps you compare products. Look at serving size first, then at sugar, sodium, and
saturated fat, since those are the values people most often underestimate.
"""

rows = []
for size in (500, 250, 100):
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size // 10, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = sp.split_text(article_text)
    rows.append({"chunk_size": size, "chunks": len(chunks),
                 "avg_chars": round(sum(map(len, chunks)) / len(chunks), 1),
                 "words_in_first_chunk": len(chunks[0].split())})
print(pd.DataFrame(rows).to_string(index=False))
