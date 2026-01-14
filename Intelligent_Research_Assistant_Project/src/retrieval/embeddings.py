import time
import tiktoken
from typing import List, Any
from langchain_openai import AzureOpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import config

class EmbeddingManager:
    def __init__(self):
        """
        Manages the conversion of text into mathematical vectors (embeddings)
        using Azure OpenAI.
        """
        print("Initializing Embedding Manager (Azure)...")
        
        if not config.AZURE_OPENAI_API_KEY or not config.AZURE_OPENAI_ENDPOINT:
             raise ValueError("Azure configuration missing. Please check .env")

        # Initialize the Azure client
        self.client = AzureOpenAIEmbeddings(
            azure_deployment=config.AZURE_EMBEDDING_DEPLOYMENT,
            openai_api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
        )
        
        # Setup token counter for cost estimation
        # We use 'cl100k_base' encoding which is used by newer OpenAI models
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Pricing for text-embedding-3-large (approximate)
        # $0.00013 per 1k tokens (check current Azure pricing)
        self.price_per_1k_tokens = 0.00013 

    def estimate_cost(self, texts: List[str]) -> float:
        """
        Calculate how much it will cost to embed these texts.
        """
        total_tokens = 0
        for text in texts:
            total_tokens += len(self.tokenizer.encode(text))
        
        cost = (total_tokens / 1000) * self.price_per_1k_tokens
        return cost, total_tokens

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts with automatic retry logic.
        If the API fails, it waits and tries again up to 3 times.
        """
        return self.client.embed_documents(texts)

    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Main function to handle large lists of text.
        1. Estimates cost.
        2. Splits data into batches.
        3. Sends batches to Azure.
        4. Collects results.
        """
        if not texts:
            return []

        # 1. Estimate and Show Cost
        cost, tokens = self.estimate_cost(texts)
        print(f"--- Embedding Job ---")
        print(f"Documents: {len(texts)}")
        print(f"Total Tokens: {tokens}")
        print(f"Estimated Cost: ${cost:.5f}")

        all_embeddings = []
        
        # 2. Process in Batches
        # Loop from 0 to end, stepping by batch_size (e.g., 0, 100, 200...)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(f"Processing batch {i // batch_size + 1} of {(len(texts) // batch_size) + 1}...")
            
            try:
                # 3. Call API (with retry protection)
                batch_embeddings = self.embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Small pause to be polite to the API rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error embedding batch starting at index {i}: {e}")
                # In a production app, you might save progress here to resume later
                raise e

        print("Embedding generation complete.")
        return all_embeddings
