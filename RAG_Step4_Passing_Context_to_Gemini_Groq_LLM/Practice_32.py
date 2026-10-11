from Practice_28 import retrieve_context, rag_answer

question = "How do nutrition and cooking technique affect a healthy meal?"
for n in (6, 3, 1):
    chunks = retrieve_context(question, n_results=n)
    sources = sorted({c["source"] for c in chunks})
    reply = rag_answer(question, n_results=n, show_sources=False)
    print(f"n_results={n}: {len(chunks)} chunks from {sources} -> answer has {len(reply.split())} words")
    print("   ", reply[:200].replace("\n", " "))
