# agents/orchestrator.py
"""
This file previously contained the OrchestratorAgent, which managed the complex RAG pipeline.
As part of the refactor towards a simpler, direct pipeline (Commit 2), this agent's
functionality has been moved directly into the main application logic (e.g., web.py).

The OrchestratorAgent class below is commented out and no longer used.
"""
# nadula
# import logging
# import time
# from .retriever import RetrieverAgent
# from .generator import GeneratorAgent
# from .base import BaseAgent
# import traceback
#
# logger = logging.getLogger(__name__)
#
# class OrchestratorAgent(BaseAgent):
#     def __init__(self):
#         init_start_time = time.time()
#         logger.info("Initializing Orchestrator and sub-agents...")
#         try:
#             self.retriever = RetrieverAgent()
#             self.generator = GeneratorAgent()
#             logger.info(f"Orchestrator ready. Duration: {time.time() - init_start_time:.2f}s")
#         except Exception as e:
#             logger.error(f"Error initializing orchestrator sub-agents: {e}", exc_info=True)
#             raise
#
#     def run(self, query: str, chat_history: list = None):
#         orchestration_start_time = time.time()
#         logger.info(f"\nOrchestrating response for query: '{query}'")
#         if chat_history is None:
#             chat_history = []
#
#         try:
#             retrieval_start_time = time.time()
#             logger.info("Step 1: Retrieving context...")
#             retrieved_chunks = self.retriever.run(query=query)
#             logger.info(f"Step 1: Retrieval complete ({len(retrieved_chunks)} chunks). Duration: {time.time() - retrieval_start_time:.4f}s")
#
#             if retrieved_chunks:
#                 log_chunks = []
#                 for i, chunk in enumerate(retrieved_chunks):
#                     meta = chunk.get('metadata', {})
#                     log_chunks.append(f"  Chunk {i+1}: p.{meta.get('page', '?')}, sec: {meta.get('section', '?')}, score: {chunk.get('score', -1):.4f}")
#                 logger.info("Retrieved Chunk Metadata:\n" + "\n".join(log_chunks))
#             else:
#                 logger.info("No chunks retrieved.")
#
#             generation_start_time = time.time()
#             logger.info("Step 2: Generating answer...")
#             final_answer = self.generator.run(query=query, context_chunks=retrieved_chunks)
#             logger.info(f"Step 2: Generation complete. Duration: {time.time() - generation_start_time:.4f}s")
#             logger.info(f"Final Answer generated: {final_answer[:150]}...")
#
#             total_orchestration_time = time.time() - orchestration_start_time
#             logger.info(f"Orchestration successful. Total time: {total_orchestration_time:.4f}s")
#
#             return {
#                 "answer": final_answer,
#                 "retrieved_chunks": retrieved_chunks,
#             }
#
#         except Exception as e:
#             logger.error(f"Error during orchestration: {e}", exc_info=True)
#             logger.error(f"Traceback: {traceback.format_exc()}")
#             return {
#                 "answer": "Sorry, I encountered an error while processing your request.",
#                 "error": str(e)
#             }
