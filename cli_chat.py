# cli_chat.py
import logging
from agents.orchestrator import OrchestratorAgent
import pprint # For potentially pretty-printing debug info

# --- Logging Setup ---
# Configure logging for the CLI application
logging.basicConfig(level=logging.WARNING, # Set to INFO or DEBUG for more verbosity
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger for this module
# --- End Logging Setup ---


def main():
    logger.info("Initializing chatbot...")
    try:
        orchestrator = OrchestratorAgent()
        logger.info("Orchestrator Agent initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize OrchestratorAgent: {e}", exc_info=True)
        print("❌ Critical Error: Could not initialize the chatbot components. Exiting.")
        return

    print("--- Chatbot CLI Initialized --- Type 'exit' or 'quit' to end ---")

    chat_history = [] # Simple list to store conversation turns

    while True:
        try:
            query = input("\n🗣️ You: ")
            if query.lower() in ["exit", "quit"]:
                print("👋 Exiting chatbot.")
                break

            logger.info(f"CLI received query: '{query}'")
            # Run the orchestrator
            result = orchestrator.run(query=query, chat_history=chat_history)
            logger.debug(f"Orchestrator result: {pprint.pformat(result)}") # Debug log for full result

            # Display the result (User-facing output remains print)
            print("\n🤖 Bot:")
            print(result["answer"])

            # Optional: Display references if not included in the answer string already
            page_refs = result.get("references", {}).get("pages", [])
            section_refs = result.get("references", {}).get("sections", [])
            ref_string_parts = []
            if page_refs:
                 ref_string_parts.append(f"pages {', '.join(map(str, sorted(list(set(page_refs)))))}")
            if section_refs:
                 ref_string_parts.append(f"sections like '{', '.join(map(str, sorted(list(set(section_refs)))))}'")

            if ref_string_parts and "*References:" not in result["answer"]:
                 print(f"\n*References: [{'; '.join(ref_string_parts)}]*")


            # Optional: Display debug info via logging
            logger.debug("\n--- Debug Info ---")
            logger.debug(f"Query Analysis: {result.get('query_analysis', 'N/A')}")
            logger.debug(f"Retrieved Chunks: {len(result.get('retrieved_chunks', []))}")
            logger.debug(f"References: {result.get('references', 'N/A')}")
            logger.debug("------------------")


            # Update chat history (simple version)
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": result["answer"]})
            # Keep history manageable (e.g., last 10 turns = 20 messages)
            chat_history = chat_history[-20:]


        except EOFError:
            print("\n👋 Exiting chatbot.")
            break
        except KeyboardInterrupt:
            print("\n👋 Exiting chatbot.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            print(f"\n❌ An unexpected error occurred: {e}")
            # Consider adding more specific error handling or logging

if __name__ == "__main__":
    main()
