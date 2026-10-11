import os
import getpass
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from google import genai
from google.genai import types

load_dotenv()

def get_key(name):
    key = os.environ.get(name, "")
    if not key:
        key = getpass.getpass(f"Enter {name} (press Enter to skip): ").strip()
    return key

GROQ_API_KEY = get_key("GROQ_API_KEY")
GEMINI_API_KEY = get_key("GEMINI_API_KEY")
if not (GROQ_API_KEY or GEMINI_API_KEY):
    raise SystemExit("Add GROQ_API_KEY or GEMINI_API_KEY to your .env file and run again.")

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else ("gemini" if GEMINI_API_KEY else "groq"))
print("Using provider:", PROVIDER)

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
}

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chromadb.Client().create_collection(
    name="kitchen_chunks_llm", embedding_function=embedder, metadata={"hnsw:space": "cosine"}
)

def store_document(source, text):
    chunks = splitter.split_text(text)
    collection.add(
        documents=chunks,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        ids=[f"{source}__{i}" for i in range(len(chunks))],
    )

for _s, _t in documents.items():
    store_document(_s, _t)
print("Total chunks stored:", collection.count())

class RagBot:
    GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    SYSTEM_PROMPT = (
        "You are a grounded Q&A bot. Base every statement on the context block. If the context is silent, answer exactly: \"I don't know based on the provided documents.\" Mention the files you used at the end."
    )

    def __init__(self, collection, provider=PROVIDER, min_similarity=0.3):
        self.collection = collection
        self.provider = provider
        self.min_similarity = min_similarity

    def retrieve(self, query, n_results=4):
        res = self.collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas", "distances"])
        rows = zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        return [{"text": t, "source": m["source"], "chunk_index": m["chunk_index"], "similarity": round(1 - x, 4)}
                for t, m, x in rows if 1 - x >= self.min_similarity]

    @staticmethod
    def build_prompt(query, chunks):
        ctx = "\n---\n".join(f"({c['source']}, #{c['chunk_index']}) {c['text']}" for c in chunks)
        return f"CONTEXT\n{ctx}\n\nQUESTION\n{query}"

    def _groq(self, prompt, system_prompt, temperature):
        if not GROQ_API_KEY:
            return "[Groq skipped: no API key]"
        client, error = Groq(api_key=GROQ_API_KEY), None
        for model in self.GROQ_MODELS:
            try:
                r = client.chat.completions.create(model=model, temperature=temperature, max_tokens=500,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
                if r.choices[0].message.content:
                    return r.choices[0].message.content.strip()
            except Exception as e:
                error = e
        return f"[Groq error: {error}]"

    def _gemini(self, prompt, system_prompt, temperature):
        if not GEMINI_API_KEY:
            return "[Gemini skipped: no API key]"
        client, error = genai.Client(api_key=GEMINI_API_KEY), None
        for model in self.GEMINI_MODELS:
            try:
                r = client.models.generate_content(model=model, contents=prompt,
                    config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature, max_output_tokens=1200))
                if r.text:
                    return r.text.strip()
            except Exception as e:
                error = e
        return f"[Gemini error: {error}]"

    def ask(self, prompt, system_prompt=None, temperature=0.1):
        fn = self._gemini if self.provider == "gemini" else self._groq
        return fn(prompt, system_prompt or self.SYSTEM_PROMPT, temperature)

    def answer(self, query, n_results=4, system_prompt=None, temperature=0.1, show_sources=True):
        chunks = self.retrieve(query, n_results)
        if not chunks:
            return "I don't know based on the provided documents. (nothing relevant was retrieved)"
        text = self.ask(self.build_prompt(query, chunks), system_prompt, temperature)
        if show_sources:
            text += "\n\nSources: " + ", ".join(sorted({f"{c['source']}#{c['chunk_index']}" for c in chunks}))
        return text

bot = RagBot(collection)

# thin function wrappers so the later practice files can use the same names in either style
def retrieve_context(query, n_results=4):
    return bot.retrieve(query, n_results)

def build_user_prompt(query, chunks):
    return bot.build_prompt(query, chunks)

def ask_llm(prompt, system_prompt=None, temperature=0.1):
    return bot.ask(prompt, system_prompt, temperature)

def rag_answer(query, n_results=4, system_prompt=None, temperature=0.1, show_sources=True):
    return bot.answer(query, n_results, system_prompt, temperature, show_sources)

SYSTEM_PROMPT = RagBot.SYSTEM_PROMPT
