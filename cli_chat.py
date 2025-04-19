# cli_chat.py
import logging
from agents.retriever import RetrieverAgent
from agents.generator import GeneratorAgent
import pprint

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---


def main():
    logger.info("Initializing chatbot components (Retriever, Generator)...")
    try:
        retriever = RetrieverAgent()
        generator = GeneratorAgent()
        logger.info("Retriever and Generator Agents initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize Agents: {e}", exc_info=True)
        print("❌ Critical Error: Could not initialize the chatbot components. Exiting.")
        return

    print("--- Chatbot CLI Initialized (Simple RAG) --- Type 'exit' or 'quit' to end ---")

    chat_history = []

    while True:
        try:
            query = input("\n🗣️ You: ")
            if query.lower() in ["exit", "quit"]:
                print("👋 Exiting chatbot.")
                break

            logger.info(f"CLI received query: '{query}'")

            # --- Simplified RAG Pipeline ---
            logger.info("Step 1: Retrieving context...")
            retrieved_chunks = retriever.run(query=query)
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks.")
            if retrieved_chunks:
                 log_chunks = []
                 for i, chunk in enumerate(retrieved_chunks):
                     meta = chunk.get('metadata', {})
                     log_chunks.append(f"  Chunk {i+1}: p.{meta.get('page', '?')}, sec: {meta.get('section', '?')}, score: {chunk.get('score', -1):.4f}")
                 logger.info("Retrieved Chunk Metadata:\n" + "\n".join(log_chunks))
            else:
                 logger.info("No chunks retrieved.")

            logger.info("Step 2: Generating answer...")
            final_answer = generator.run(query=query, context_chunks=retrieved_chunks)
            logger.info(f"Generated answer: {final_answer[:100]}...")
            # --- End Simplified RAG Pipeline ---

            print("\n🤖 Bot:")
            print(final_answer)

            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": final_answer})
            chat_history = chat_history[-20:]

        except EOFError:
            print("\n👋 Exiting chatbot.")
            break
        except KeyboardInterrupt:
            print("\n👋 Exiting chatbot.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred during the chat loop: {e}", exc_info=True)
            print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
