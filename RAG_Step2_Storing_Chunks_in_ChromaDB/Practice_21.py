import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentStore:
    def __init__(self, name, chunk_size=240, overlap=30):
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = chromadb.Client().create_collection(name=name, embedding_function=embedder)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])

    def add(self, source, text, **meta):
        chunks = self.splitter.split_text(text)
        self.collection.add(
            documents=chunks,
            metadatas=[{"source": source, "chunk_index": i, **meta} for i in range(len(chunks))],
            ids=[f"{source}:{i}" for i in range(len(chunks))],
        )
        return len(chunks)

    def remove(self, source):
        self.collection.delete(where={"source": source})

    def stats(self):
        counts = {}
        for m in self.collection.get(include=["metadatas"])["metadatas"]:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return counts

    def search(self, query, n=2):
        r = self.collection.query(query_texts=[query], n_results=n, include=["documents", "metadatas"])
        return list(zip(r["documents"][0], r["metadatas"][0]))

doc_a = """Kitchen Techniques

Sauteing cooks food quickly in a small amount of hot oil, while braising cooks tougher cuts slowly in liquid
until they turn tender. Choosing the right technique changes both texture and flavour.

Letting meat rest after cooking allows the juices to redistribute, which makes each slice more moist.
"""

doc_b = """Fermentation at Home

Fermentation uses bacteria or yeast to transform food. Yogurt, idli batter, and sourdough all depend on
microbes that produce acids and gases.

Fermented foods can add tangy flavour and may support gut health. Clean jars and correct salt levels are
essential to keep unwanted microbes away.
"""

doc_c = """Pantry Inventory Notes

The organic basmati rice is stored under item code ITM-4521 and is currently the most used staple in the pantry.

The cold-pressed sesame oil is stored under item code ITM-7788 and is restocked within three business days.

Any stock check must quote the exact item code so the kitchen team can locate the right shelf quickly.
"""

store = DocumentStore("kitchen_chunks")
store.add("kitchen_techniques.txt", doc_a, team="alpha")
store.add("fermentation.txt", doc_b, team="beta")
store.add("pantry_inventory.txt", doc_c, team="alpha")
print("Stats:", store.stats())

store.remove("pantry_inventory.txt")
print("After removing pantry_inventory.txt:", store.stats())

for text, meta in store.search("What is fermentation?"):
    print(meta["source"], "|", text[:70].replace("\n", " "))
