from typing import List, Dict

# 1. The System Prompt (The AI's Identity)
# This is the "Base Instruction" that is always active. 
# It sets the rules for how the AI must behave.
SYSTEM_PROMPT = """You are an Intelligent Research Assistant. Your goal is to provide accurate, 
detailed, and evidence-based answers using the provided document snippets.

RULES:
1. Only use the information provided in the 'Context' section below.
2. If the answer is not in the context, say: "I'm sorry, but I couldn't find information about that in the uploaded documents." Do NOT make up an answer.
3. You MUST cite your sources. After every claim or paragraph, add a citation like [Source: filename, Page: X] or [Source: YouTube Link].
4. Be professional, objective, and concise.
5. If the user asks a follow-up question, use the 'Conversation History' to understand the context.
"""

# 2. The RAG Prompt Template
# This is a template with placeholders (the words inside curly brackets {}).
# Our code will replace these placeholders with real data before sending it to the AI.
RAG_PROMPT_TEMPLATE = """
---
CONVERSATION HISTORY:
{chat_history}

---
CONTEXT FROM DOCUMENTS:
{context}

---
USER QUESTION: 
{question}

---
Final Answer Instructions: 
Provide a detailed response based ONLY on the context above. 
Remember to include citations for every piece of information used.
"""

def format_context(retrieved_docs: List[Dict]) -> str:
    """
    Takes the raw list of documents from the Vector Store and 
    turns them into a single, clean string for the AI to read.
    """
    if not retrieved_docs:
        return "No relevant document snippets were found."
    
    formatted_chunks = []
    for i, doc in enumerate(retrieved_docs):
        text = doc['text']
        metadata = doc['metadata']
        
        # We label each chunk so the AI knows which is which
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'N/A')
        
        chunk_header = f"--- Document {i+1} (Source: {source}, Page: {page}) ---"
        formatted_chunks.append(f"{chunk_header}\n{text}")
    
    return "\n\n".join(formatted_chunks)

def format_chat_history(history: List[Dict[str, str]]) -> str:
    """
    Converts the list of previous messages into a string.
    Expected format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    if not history:
        return "No previous conversation."
    
    formatted_history = []
    for msg in history:
        role = "User" if msg['role'] == 'user' else "Assistant"
        content = msg['content']
        formatted_history.append(f"{role}: {content}")
    
    return "\n".join(formatted_history)

def create_final_prompt(question: str, context_docs: List[Dict], chat_history: List[Dict]) -> str:
    """
    The main function that assembles everything into one giant message.
    """
    # 1. Prepare the context string
    context_str = format_context(context_docs)
    
    # 2. Prepare the history string
    history_str = format_chat_history(chat_history)
    
    # 3. Fill in the template
    final_prompt = RAG_PROMPT_TEMPLATE.format(
        chat_history=history_str,
        context=context_str,
        question=question
    )
    
    return final_prompt
