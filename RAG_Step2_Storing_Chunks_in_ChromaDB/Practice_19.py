import os
import hashlib
import tempfile
from collections import Counter
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

first_text = """Nutrition Basics for Everyday Cooking

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
second_text = """Kitchen Techniques

Sauteing cooks food quickly in a small amount of hot oil, while braising cooks tougher cuts slowly in liquid
until they turn tender. Choosing the right technique changes both texture and flavour.

Letting meat rest after cooking allows the juices to redistribute, which makes each slice more moist.
"""
third_text = """Fermentation at Home

Fermentation uses bacteria or yeast to transform food. Yogurt, idli batter, and sourdough all depend on
microbes that produce acids and gases.

Fermented foods can add tangy flavour and may support gut health. Clean jars and correct salt levels are
essential to keep unwanted microbes away.
"""
sources = {"nutrition_basics.txt": first_text, "kitchen_techniques.txt": second_text, "fermentation.txt": third_text}

os.makedirs("docs_food", exist_ok=True)
for name, text in sources.items():
    with open(os.path.join("docs_food", name), "w", encoding="utf-8") as f:
        f.write(text)

splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.get_or_create_collection(name="kitchen_chunks", embedding_function=embedder)

def store(text, source):
    """Idempotent: ids come from a hash of the chunk text, so re-running never duplicates."""
    chunks = splitter.split_text(text)
    ids = [hashlib.md5(c.encode("utf-8")).hexdigest()[:12] for c in chunks]
    collection.upsert(
        documents=chunks, ids=ids,
        metadatas=[{"source": source, "chunk_index": i, "chars": len(c)} for i, c in enumerate(chunks)],
    )
    return len(chunks)

for name, text in sources.items():
    print(f"{name:<28} -> {store(text, name)} chunks")
print("Total:", collection.count())

# Storing the same document again must not change the count
store(first_text, "nutrition_basics.txt")
print("Total after re-storing 'nutrition_basics.txt':", collection.count())

per_source = {}
for m in collection.get(include=["metadatas"])["metadatas"]:
    per_source[m["source"]] = per_source.get(m["source"], 0) + 1
print("Per source:", per_source)

hit = collection.query(query_texts=["Why does resting meat make it moist?"], n_results=1, include=["documents", "metadatas"])
print("\nBest match for 'Why does resting meat make it moist?':", hit["metadatas"][0][0]["source"])
print(hit["documents"][0][0])
