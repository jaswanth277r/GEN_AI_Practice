from Practice_28 import retrieve_context, build_user_prompt, ask_llm

question = "Why does resting meat make it moist?"
prompt = build_user_prompt(question, retrieve_context(question))

def sample(temp, runs=3):
    return [ask_llm(prompt, temperature=temp) for _ in range(runs)]

for temp in (0.2, 1.3):
    answers = sample(temp)
    unique = len(set(answers))
    print(f"temperature={temp}: {unique} distinct answer(s) out of {len(answers)}")
    print("   first answer:", answers[0][:160].replace("\n", " "))
