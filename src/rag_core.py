import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")

def load_retriever(embedding_model_name, index_path):
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.load_local(index_path, embedding_function, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={'k': 4})
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag_chain(retriever, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0)

    template = """
    You are a helpful history tutor. Answer the following question based only on the provided context.
    Provide a concise answer and list the page numbers the information was found on.
    If the context doesn't contain the answer, state that clearly.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def get_rag_response(query, rag_chain, retriever):
    response = rag_chain.invoke(query)
    relevant_docs = retriever.invoke(query)
    pages = sorted(list(set(doc.metadata.get('page_number', 'N/A') for doc in relevant_docs)))
    return {'answer': response, 'context': relevant_docs, 'pages': pages}
