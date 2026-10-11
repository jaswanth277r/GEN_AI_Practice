from Practice_28 import collection, store_document, rag_answer

NEW_FILE = "fermentation.txt"
store_document(NEW_FILE, """Fermentation at Home

Fermentation uses bacteria or yeast to transform food. Yogurt, idli batter, and sourdough all depend on
microbes that produce acids and gases.

Fermented foods can add tangy flavour and may support gut health. Clean jars and correct salt levels are
essential to keep unwanted microbes away.
""")
print(f"Collection grew to {collection.count()} chunks after adding {NEW_FILE}")

questions = ['What is fermentation?', 'Why are clean jars important for fermenting?', 'How do I configure a firewall rule?']
for number, q in enumerate(questions, start=1):
    print(f"\nQ{number}: {q}")
    print("A :", rag_answer(q, n_results=3))
