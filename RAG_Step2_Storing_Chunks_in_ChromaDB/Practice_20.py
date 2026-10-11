import chromadb
from chromadb.utils import embedding_functions
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

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
def build(name, size, overlap):
    col = client.create_collection(name=name, embedding_function=embedder)
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
    pieces = sp.split_text(article_text)
    col.add(documents=pieces, ids=[f"c{i}" for i in range(len(pieces))])
    return col

collections = {"large(500)": build("food_large", 500, 50), "small(120)": build("food_small", 120, 10)}
for label, col in collections.items():
    print(label, "->", col.count(), "chunks")

print()
for q in ['Why are whole grains filling?', 'What does fibre do in vegetables?', 'How does braising work?']:
    row = []
    for label, col in collections.items():
        res = col.query(query_texts=[q], n_results=1, include=["distances"])
        row.append(f"{label}: {res['distances'][0][0]:.3f}")
    print(f"{q[:42]:<44}", " | ".join(row))
