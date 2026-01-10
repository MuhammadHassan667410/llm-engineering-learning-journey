from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from dotenv import load_dotenv
from openai import AzureOpenAI
import os
from langchain_openai import AzureChatOpenAI

load_dotenv(override=True)
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
# CONFIGURATION
# Using "gpt-4o-mini" because it exists and is smarter than GPT-3.5
#MODEL = "gpt-4.1-mini"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
RETRIEVAL_K = 15 # Retrieve 7 chunks for maximum context

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    api_version="2024-02-15-preview",
    azure_endpoint="https://llm-engineering-azure-learn.openai.azure.com/",
    api_key=subscription_key
)

# Connect to the Database
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

# Configure the Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# Initialize the Brain
llm = AzureChatOpenAI(
    azure_deployment="gpt-oss-120b",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://llm-engineering-azure-learn.openai.azure.com/"),
    api_key=subscription_key,
    temperature=0.7
)

# --- THE IMPROVED SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are an expert AI assistant for Insurellm. Your mission is to provide accurate, well-structured answers based EXCLUSIVELY on the provided context.

### GUIDELINES:
1. **Scope:** Answer ONLY using the information in the 'Context' section below. Do not use outside knowledge.
2. **Synthesis:** If multiple documents contain relevant details (e.g., for "Who" or "List" questions), combine them into a single comprehensive answer.
3. **Citations:** You must cite the source file for every key fact provided (e.g., "Maxine Thompson was the recipient of the IIOTY (Insurellm Innovator of the Year) 2023 award. [Source: : Maxine Thompson.md]").
4. **Unknowns:** If the exact answer is not in the context, say: "I'm sorry, I don't have that information in my knowledge base."
5. **Format:** Use Markdown (bullet points, bolding) for clarity.

### CONTEXT:
{context}
"""

def fetch_context(query: str) -> list[Document]:
    """
    Searches the Vector DB for the most relevant chunks.
    """
    return retriever.invoke(query)

# ... (imports remain the same)

def contextualize_question(question: str, history: list[dict]) -> str:
    """
    Uses the LLM to rewrite the question to include context from history.
    Example: 
    History: "Who is John?"
    Current: "Where did he go?"
    Result: "Where did John go?"
    """
    if not history:
        return question
        
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:]) # Keep last 4 turns
    
    prompt = f"""
    Given a chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    
    Chat History:
    {history_text}
    
    Latest Question: {question}
    
    Standalone Question:
    """
    
    msg = [HumanMessage(content=prompt)]
    response = llm.invoke(msg)
    return response.content

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Main RAG Loop:
    1. Contextualize Question (Rewrite "he" to "John")
    2. Retrieve Context
    3. Format Prompt
    4. Ask LLM
    """
    # 1. Contextualize the question
    standalone_question = contextualize_question(question, history)
    print(f"DEBUG: Rewrote '{question}' to '{standalone_question}'") # Helpful for debugging
    
    # 2. Retrieve relevant documents using the standalone question
    docs = fetch_context(standalone_question) 
    
    # 3. Build the Context String
    context_text = "\n\n".join(doc.page_content for doc in docs)
    
    # 4. Fill the System Prompt
    final_system_prompt = SYSTEM_PROMPT.format(context=context_text)
    
    # 5. Prepare Messages
    messages = [SystemMessage(content=final_system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question)) # We pass original question to LLM, context is already in system prompt
    
    # 6. Generate Answer
    response = llm.invoke(messages)
    
    return response.content, docs
# Print this after "if" if you want to test the function directly
#ans, context = answer_question("Who went to Manchester University?")
#print(f"Answer: {ans}")
#if __name__ == "__main__":
