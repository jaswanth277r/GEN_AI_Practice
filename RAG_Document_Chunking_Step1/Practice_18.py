import os
import re
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

file_path = "my_food_notes.txt"
if not os.path.exists(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("""Kitchen Techniques

Sauteing cooks food quickly in a small amount of hot oil, while braising cooks tougher cuts slowly in liquid
until they turn tender. Choosing the right technique changes both texture and flavour.

Letting meat rest after cooking allows the juices to redistribute, which makes each slice more moist.

Fermentation at Home

Fermentation uses bacteria or yeast to transform food. Yogurt, idli batter, and sourdough all depend on
microbes that produce acids and gases.

Fermented foods can add tangy flavour and may support gut health. Clean jars and correct salt levels are
essential to keep unwanted microbes away.""")

with open(file_path, "r", encoding="utf-8") as f:
    my_text = f.read()
print("Loaded", file_path, "->", len(my_text), "characters")

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    encode, decode = _enc.encode, _enc.decode
except Exception:                      # offline fallback: treat words as tokens
    encode = lambda t: t.split(" ")
    decode = lambda toks: " ".join(toks)

def fixed_chunks(text, size=250, overlap=40):
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out

def sentence_chunks(text, n=2):
    s = [x for x in re.split(r'(?<=[.!?])\s+', text.strip()) if x]
    return [" ".join(s[i:i + n]) for i in range(0, len(s), n)]

def token_chunks(text, max_tokens=50, overlap=10):
    toks = encode(text)
    out, start = [], 0
    while start < len(toks):
        out.append(decode(toks[start:start + max_tokens]))
        start += max_tokens - overlap
    return out

strategies = {
    "fixed-size": fixed_chunks(my_text),
    "character": CharacterTextSplitter(separator="\n\n", chunk_size=250, chunk_overlap=40).split_text(my_text),
    "recursive": RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40, separators=["\n\n", "\n", ". ", " ", ""]).split_text(my_text),
    "sentence": sentence_chunks(my_text),
    "token": token_chunks(my_text),
}

import pandas as pd
rows = [{"strategy": n, "chunks": len(c), "avg_chars": round(sum(map(len, c)) / len(c), 1), "max_chars": max(map(len, c))}
        for n, c in strategies.items()]
print(pd.DataFrame(rows).to_string(index=False))
