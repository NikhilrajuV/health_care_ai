
from src.rag import retrieve_medical_context

questions = [
    "What are common causes of headaches?",
    "Does my uploaded medical PDF explain diabetes treatment?",
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    try:
        results = retrieve_medical_context(question, k=4)

        if not results:
            print("No passages returned.")
            continue

        for index, item in enumerate(results, start=1):
            print(f"\nPassage {index}")
            print("Source:", repr(item.get("source")))
            print("Page:", repr(item.get("page")))
            print("Score:", repr(item.get("relevance_score")))
            print("Text:", item.get("text", "")[:350].replace("\n", " "))

    except Exception as exc:
        print(type(exc).__name__, str(exc))
