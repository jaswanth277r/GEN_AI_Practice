import json
from Practice_28 import rag_answer

JSON_PROMPT = (
    "Answer using ONLY the provided context. Respond with a single JSON object and nothing else, shaped like "
    '{"answer": "<one or two sentences>", "sources": ["<file name>", ...], "confident": true or false}. '
    'If the context does not contain the answer use {"answer": "I don\'t know based on the provided documents.", "sources": [], "confident": false}.'
)

for q in ['Why are whole grains filling?', 'How do nutrition and cooking technique affect a healthy meal?']:
    raw = rag_answer(q, system_prompt=JSON_PROMPT, show_sources=False)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    print("\nQuestion:", q)
    if raw.startswith("I don't know"):
        print("Refused (nothing relevant retrieved):", raw)
        continue
    try:
        data = json.loads(cleaned)
        print("Valid JSON ->", data["answer"], "| sources:", data["sources"], "| confident:", data["confident"])
    except (json.JSONDecodeError, KeyError) as err:
        print("Model did not return valid JSON:", err, "\nRaw reply:", raw)
