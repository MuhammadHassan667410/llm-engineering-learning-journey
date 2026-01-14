from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any, List
import config
from src.retrieval.vector_store import VectorStore

from src.retrieval.conversation import ConversationManager
from src.retrieval.query_enhancement import QueryEnhancer
from src.retrieval import prompts


class QAChain:
    def __init__(self, vector_store: VectorStore, conversation_manager: ConversationManager):
        """
        The Master Orchestrator (RAG Chain).
        It connects the Database, Memory, and AI Model together.
        """
        print("Initializing QA Chain (Azure)...")
        
        self.vector_store = vector_store
        self.conversation = conversation_manager
        self.query_enhancer = QueryEnhancer() # New Smart Search Tool
        
        # Initialize the Chat Model (The "Brain")
        if not config.AZURE_OPENAI_API_KEY:
             raise ValueError("Azure API Key missing.")

        self.llm = AzureChatOpenAI(
            azure_deployment=config.AZURE_OPENAI_CHAT_DEPLOYMENT,
            openai_api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_AI_FOUNDRY_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            temperature=0.3, 
            max_tokens=1000  
        )

    def _format_sources(self, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        # ... (stays the same)
        unique_sources = {}
        formatted_list = []

        for doc in docs:
            meta = doc['metadata']
            source_name = meta.get('source', 'Unknown')
            
            if meta.get('type') == 'youtube':
                title = meta.get('title', 'Video')
                timestamp = int(meta.get('start_time', 0))
                minutes = timestamp // 60
                seconds = timestamp % 60
                time_str = f"{minutes}:{seconds:02d}"
                label = f"📺 {title} ({time_str})"
                link = meta.get('url_with_timestamp', source_name)
                unique_key = f"{title}_{timestamp}"
            else:
                page = meta.get('page', 'N/A')
                label = f"📄 {source_name} (Page {page})"
                link = None
                unique_key = f"{source_name}_{page}"

            if unique_key not in unique_sources:
                unique_sources[unique_key] = True
                formatted_list.append({
                    "label": label,
                    "link": link,
                    "text": doc['text'][:150] + "..."
                })
        return formatted_list

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Enhanced ask method using Multi-Query Retrieval.
        """
        if not question.strip():
            return {"answer": "Please ask a question.", "sources": []}

        print(f"User asked: {question}")
        
        # 1. Smart Retrieval (The "Enhanced Research" Phase)
        # Brainstorm 3 versions of the question
        print("Enhancing query...")
        query_variants = self.query_enhancer.generate_variations(question)
        
        # Perform search for each variant
        all_search_results = []
        for variant in query_variants:
            print(f"Searching for variant: {variant}")
            # We search for more initially (k=10) to give RRF more data to work with
            results = self.vector_store.search(variant, k=10)
            all_search_results.append(results)
        
        # Merge and Re-Rank results using Reciprocal Rank Fusion
        # This keeps the "best of the best" across all 3 searches
        relevant_docs = self.query_enhancer.reciprocal_rank_fusion(all_search_results)[:5]
        
        # 2. Prompt Construction
        history = self.conversation.get_history()
        final_prompt_text = prompts.create_final_prompt(
            question=question,
            context_docs=relevant_docs,
            chat_history=history
        )
        
        # 3. Generation
        messages = [
            SystemMessage(content=prompts.SYSTEM_PROMPT),
            HumanMessage(content=final_prompt_text)
        ]
        
        print("Sending request to Azure OpenAI...")
        try:
            response = self.llm.invoke(messages)
            answer_text = response.content
            
            # 4. Memory Update
            self.conversation.add_user_message(question)
            self.conversation.add_assistant_message(answer_text)
            
            # 5. Format Output
            clean_sources = self._format_sources(relevant_docs)
            
            return {
                "answer": answer_text,
                "sources": clean_sources
            }
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return {
                "answer": f"I encountered an error. Technical details: {str(e)}",
                "sources": []
            }

    def clear_history(self):
        """Reset the conversation memory."""
        self.conversation.clear()
