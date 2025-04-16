# cli_chat.py
from agents.retriever import RetrieverAgent
from agents.generator import GeneratorAgent
from agents.reference_tracker import ReferenceTrackerAgent

def main():
    print("Initializing agents...")
    retriever = RetrieverAgent()
    generator = GeneratorAgent()
    reference_tracker = ReferenceTrackerAgent()
    print("--- Chatbot CLI Initialized --- Type 'exit' or 'quit' to end ---")

    while True:
        try:
            query = input("\n🗣️ You: ")
            if query.lower() in ["exit", "quit"]:
                print("👋 Exiting chatbot.")
                break

            # 1. Retrieve
            context_chunks = retriever.run(query=query)

            if not context_chunks:
                print("\n🤖 Bot: I couldn't find relevant information for that query.")
                continue

            # 2. Generate
            answer = generator.run(query=query, context_chunks=context_chunks)

            # 3. Track References (and display)
            references = reference_tracker.run(context_chunks=context_chunks)
            page_refs = references.get("pages", [])

            print("\n🤖 Bot:")
            print(answer)
            if page_refs:
                print(f"\n*References: [p. {', '.join(map(str, page_refs))}]*" ) # Ensure refs are shown

        except EOFError:
            print("\n👋 Exiting chatbot.")
            break
        except KeyboardInterrupt:
            print("\n👋 Exiting chatbot.")
            break
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            # Optionally add more robust error handling or logging here

if __name__ == "__main__":
    main()
