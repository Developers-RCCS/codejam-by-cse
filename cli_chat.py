# cli_chat.py
from agents.orchestrator import OrchestratorAgent
import pprint # For potentially pretty-printing debug info

def main():
    print("Initializing chatbot...")
    orchestrator = OrchestratorAgent()
    print("--- Chatbot CLI Initialized --- Type 'exit' or 'quit' to end ---")

    chat_history = [] # Simple list to store conversation turns

    while True:
        try:
            query = input("\n🗣️ You: ")
            if query.lower() in ["exit", "quit"]:
                print("👋 Exiting chatbot.")
                break

            # Run the orchestrator
            result = orchestrator.run(query=query, chat_history=chat_history)

            # Display the result
            print("\n🤖 Bot:")
            print(result["answer"])

            # Optional: Display references if not included in the answer string already
            # page_refs = result["references"].get("pages", [])
            # if page_refs and "*References:" not in result["answer"]:
            #     print(f"\n*References: [p. {', '.join(map(str, page_refs))}]*")

            # Optional: Display debug info
            # print("\n--- Debug Info ---")
            # print("Query Analysis:", result["query_analysis"])
            # print("Retrieved Chunks:", len(result["retrieved_chunks"]))
            # print("References:", result["references"])
            # print("------------------")


            # Update chat history (simple version)
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": result["answer"]})
            # Keep history manageable (e.g., last 10 turns)
            chat_history = chat_history[-10:]


        except EOFError:
            print("\n👋 Exiting chatbot.")
            break
        except KeyboardInterrupt:
            print("\n👋 Exiting chatbot.")
            break
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            # Consider adding more specific error handling or logging

if __name__ == "__main__":
    main()
