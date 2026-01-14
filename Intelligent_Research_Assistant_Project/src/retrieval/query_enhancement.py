from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import config

class QueryEnhancer:
    def __init__(self):
        """
        A tool that improves search results by rewriting the user's question.
        """
        # We reuse the same Azure Chat settings
        self.llm = AzureChatOpenAI(
            azure_deployment=config.AZURE_OPENAI_CHAT_DEPLOYMENT,
            openai_api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_AI_FOUNDRY_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            temperature=0.5 # A bit creative to generate variations
        )

    def generate_variations(self, original_query: str) -> List[str]:
        """
        Ask the AI to come up with 2 alternative versions of the question.
        """
        system_msg = """You are a search optimization assistant. 
        Generate 2 alternative versions of the user's query to improve retrieval from a vector database. 
        Focus on keywords and semantic meaning. 
        Output ONLY the 2 variations, separated by a newline."""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=original_query)
            ])
            
            # Split the answer by lines to get a list
            # e.g., "Variation 1\nVariation 2" -> ["Variation 1", "Variation 2"]
            variations = response.content.strip().split('\n')
            
            # Clean up potential numbering (e.g. "1. Question")
            cleaned = []
            for v in variations:
                clean_v = v.split('. ', 1)[-1].strip()
                if clean_v:
                    cleaned.append(clean_v)
            
            # Always include the original!
            return [original_query] + cleaned[:2]
            
        except Exception as e:
            print(f"Query enhancement failed: {e}")
            return [original_query]

    def reciprocal_rank_fusion(self, results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        Combine multiple lists of search results into one ranked list.
        
        Algorithm:
        1. Assign a score to each doc based on its rank in each list.
        2. Score = 1 / (rank + k)
        3. Add up scores from all lists.
        4. Sort by highest score.
        """
        fused_scores = {}
        doc_map = {}
        
        # Iterate through each list of results (from different queries)
        for results in results_list:
            for rank, doc in enumerate(results):
                # Use a unique ID (filename + chunk index)
                # If we don't have a unique ID, we make one from the text hash or metadata
                # Assuming our 'metadata' has source and page
                doc_id = f"{doc['metadata'].get('source')}_{doc['metadata'].get('chunk_index')}"
                
                # Save the full document object so we can return it later
                doc_map[doc_id] = doc
                
                # Calculate RRF score
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (rank + k)
        
        # Sort documents by their final accumulated score (highest first)
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        # Return the actual document objects
        return [doc_map[did] for did in sorted_ids]
