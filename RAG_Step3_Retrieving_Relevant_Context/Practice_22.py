import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = {
    "nutrition_basics.txt": """Nutrition Basics for Everyday Cooking

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
""",
    "kitchen_techniques.txt": """Kitchen Techniques

Sauteing cooks food quickly in a small amount of hot oil, while braising cooks tougher cuts slowly in liquid
until they turn tender. Choosing the right technique changes both texture and flavour.

Letting meat rest after cooking allows the juices to redistribute, which makes each slice more moist.
"""
}

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chromadb.Client().create_collection(
    name="kitchen_chunks", embedding_function=embedder, metadata={"hnsw:space": "cosine"}
)

def index_document(source, text):
    chunks = splitter.split_text(text)
    collection.add(
        documents=chunks,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        ids=[f"{source}_chunk_{i}" for i in range(len(chunks))],
    )

for _source, _text in documents.items():
    index_document(_source, _text)
print("Total chunks stored:", collection.count())

class Retriever:
    def __init__(self, collection, threshold=0.25):
        self.col = collection
        self.threshold = threshold

    @staticmethod
    def similarity(distance):
        return 1 - distance

    def search(self, query, n=5, threshold=None, where=None, max_words=None):
        cutoff = self.threshold if threshold is None else threshold
        kwargs = dict(query_texts=[query], n_results=n, include=["documents", "metadatas", "distances"])
        if where:
            kwargs["where"] = where
        res = self.col.query(**kwargs)

        hits, words = [], 0
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            sim = self.similarity(dist)
            if sim < cutoff:
                continue
            if max_words is not None and words + len(doc.split()) > max_words:
                break
            words += len(doc.split())
            hits.append({"text": doc, "source": meta["source"], "chunk_index": meta["chunk_index"], "similarity": round(sim, 4)})
        return hits

retriever = Retriever(collection)

for label, q in (("in-scope", "Why are whole grains filling?"), ("out-of-scope", "How do I configure a firewall rule?")):
    print(f"\n[{label}] {q}")
    for cutoff in (0.5, 0.3, 0.1):
        found = retriever.search(q, n=5, threshold=cutoff)
        best = max((h["similarity"] for h in found), default=None)
        print(f"   cutoff={cutoff}: kept {len(found)} chunk(s), best similarity = {best}")
