from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_PATH = "chroma"
RELEVANCE_THRESHOLD = 0.3  # Minimum score to accept a result

PROMPT_TEMPLATE = """
You are an experienced F1 race engineer and strategy expert. You have deep knowledge of tyre 
compounds, pit stop windows, safety car scenarios, undercuts, overcuts, and circuit-specific 
strategy patterns.

Using ONLY the context provided below from the F1 circuit knowledge base, answer the user's 
strategy question. Be specific, confident, and direct — like a race engineer on the pit wall. 
Reference specific lap windows, tyre compounds, and circuit characteristics where relevant.

If the context does not contain enough information to answer the question, say so clearly 
rather than guessing.

Context from knowledge base:
{context}

---

Strategy question: {question}

Respond as a race engineer giving a clear strategic recommendation.
"""

def main():
    # Load the vector DB
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = ChatOpenAI(model="gpt-4.1-nano")

    print("=" * 60)
    print("       F1 RACE STRATEGY AI — PIT WALL ADVISOR")
    print("=" * 60)
    print("Ask a strategy question (e.g. 'I'm P4 at Monza on lap 30")
    print("on Medium tyres showing heavy deg. Box now or stay out?')")
    print("Type 'exit' to quit.\n")

    while True:
        query_text = input("Strategy Query: ").strip()

        if query_text.lower() == "exit":
            print("Session ended. Good race.")
            break

        if not query_text:
            print("Please enter a query.\n")
            continue

        # Retrieve relevant chunks from ChromaDB
        results = db.similarity_search_with_relevance_scores(query_text, k=4)

        # Filter by relevance threshold
        filtered_results = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]

        if not filtered_results:
            print("\nNo relevant data found in the knowledge base for that query.")
            print("Try mentioning a specific circuit (Monza, Baku, Abu Dhabi), tyre compound, or scenario.\n")
            continue

        # Show which chunks were retrieved (helpful for debugging)
        print(f"\n[Retrieved {len(filtered_results)} relevant chunks]")
        for doc, score in filtered_results:
            source = doc.metadata.get("source", "unknown")
            print(f"  {score:.4f} — {source} — {doc.page_content[:60]}...")

        # Build context from retrieved chunks
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered_results])

        # Build and send prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        print("\n[Thinking...]\n")
        response = model.invoke(prompt)
        response_text = response.content

        sources = list(set([doc.metadata.get("source", "unknown") for doc, _ in filtered_results]))

        print("=" * 60)
        print("RACE ENGINEER RESPONSE:")
        print("=" * 60)
        print(response_text)
        print(f"\nSources: {sources}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
