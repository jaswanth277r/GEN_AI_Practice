import re
from rank_bm25 import BM25Okapi
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
""",
    "pantry_inventory.txt": """Pantry Inventory Notes

The organic basmati rice is stored under item code ITM-4521 and is currently the most used staple in the pantry.

The cold-pressed sesame oil is stored under item code ITM-7788 and is restocked within three business days.

Any stock check must quote the exact item code so the kitchen team can locate the right shelf quickly.
""",
}

STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "do", "does", "did", "what", "whats", "which",
             "who", "how", "why", "when", "where", "can", "could", "would", "should", "you", "your", "me", "my", "i",
             "we", "our", "it", "its", "this", "that", "of", "for", "to", "in", "on", "at", "by", "with", "about",
             "and", "or", "please", "tell", "s"}

class HybridIndex:
    def __init__(self, name, chunk_size=230, overlap=20):
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.col = chromadb.Client().create_collection(name=name, embedding_function=embedder,
                                                       metadata={"hnsw:space": "cosine"})
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
        self.texts, self.meta, self.bm25 = [], [], None

    @staticmethod
    def tokens(text, drop=False):
        toks = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in toks if not (drop and t in STOPWORDS)]

    def add(self, source, text):
        chunks = self.splitter.split_text(text)
        meta = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        self.col.add(documents=chunks, metadatas=meta, ids=[f"{source}|{i}" for i in range(len(chunks))])
        self.texts += chunks
        self.meta += meta
        self.bm25 = BM25Okapi([self.tokens(t) for t in self.texts])
        return len(chunks)

    def semantic(self, query, n=5):
        r = self.col.query(query_texts=[query], n_results=min(n, self.col.count()),
                           include=["documents", "metadatas", "distances"])
        return [{"text": t, "source": m["source"], "similarity": round(1 - x, 4)}
                for t, m, x in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def keyword(self, query, n=5, drop=False):
        q = self.tokens(query, drop)
        if not q:
            return []
        sc = self.bm25.get_scores(q)
        idx = sorted(range(len(sc)), key=lambda i: -sc[i])[:n]
        return [{"text": self.texts[i], "source": self.meta[i]["source"], "bm25": round(float(sc[i]), 4)}
                for i in idx if sc[i] > 0]

    def hybrid(self, query, n=3, k=60, floor=-1.0, drop=False):
        scores, info = {}, {}
        sem = [r for r in self.semantic(query, 10) if r["similarity"] >= floor]
        for ranking in (sem, self.keyword(query, 10, drop)):
            for pos, r in enumerate(ranking):
                scores[r["text"]] = scores.get(r["text"], 0) + 1 / (k + pos + 1)
                info.setdefault(r["text"], r)
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [{"text": t, "source": info[t]["source"], "rrf": round(s, 5)} for t, s in top]

    def smart(self, query, n=4, threshold=0.5):
        hits = [r for r in self.semantic(query, n) if r["similarity"] >= threshold]
        if hits:
            return {"status": "ok", "method": "semantic", "chunks": hits}
        hits = self.hybrid(query, n)
        if hits:
            return {"status": "ok", "method": "hybrid_fallback", "chunks": hits}
        return {"status": "empty", "method": None, "chunks": [],
                "message": "I don't know based on the provided documents. Covered files: "
                           + ", ".join(sorted({m["source"] for m in self.meta}))}

index = HybridIndex("kitchen_chunks_hybrid")
for _src, _txt in documents.items():
    index.add(_src, _txt)
print("Total chunks stored:", index.col.count())

index.add("pantry_extra.txt", """Pantry Addendum

The stone-ground turmeric powder is stored under item code ITM-9034 and comes in a one kilogram sealed pack.

The black pepper whole is stored under item code ITM-2210 and is available in two bag sizes.
""")
print("Total chunks now:", index.col.count())

def rank(results, needle):
    return next((i for i, r in enumerate(results, 1) if needle in r["text"]), None)

query = "ITM-9034"
n_all = index.col.count()
report = {
    "semantic": rank(index.semantic(query, n_all), "ITM-9034"),
    "bm25": rank(index.keyword(query, n_all), "ITM-9034"),
    "hybrid": rank(index.hybrid(query, n_all), "ITM-9034"),
}
print(f"Position of the ITM-9034 chunk -> {report}")

forced = index.smart(query, threshold=0.99)
print("Forced fallback:", forced["status"], forced["method"])
print("Fallback surfaced ITM-9034:", any("ITM-9034" in c["text"] for c in forced["chunks"]))
