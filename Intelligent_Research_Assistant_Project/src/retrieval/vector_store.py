import os
from sys import api_version

import chromadb
from chromadb.config import Settings
from langchain_openai import AzureOpenAIEmbeddings
from typing import List, Dict, Any
from dotenv import load_dotenv
import config

# Load environment variables (API Keys)
load_dotenv()

class VectorStore:
    def __init__(self, collection_name: str = "research_documents"):
        """
        Initialize the Vector Store.
        
        1. Sets up the OpenAI embedding model.
        2. Connects to the local ChromaDB database.
        3. Gets or creates the specific collection for our documents.
        """
        
        print("Initializing Vector Store...")
        
        # 1. Setup Embedding Function
        # This is the "Translator" that converts English text into list of numbers (vectors).
        # We use 'text-embedding-3-large' for high accuracy.
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in .env file")
            
        print(f"DEBUG: Endpoint={config.AZURE_OPENAI_ENDPOINT}")
        print(f"DEBUG: Deployment={config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
        print(f"DEBUG: Version={config.AZURE_OPENAI_API_VERSION}")

        self.embedding_fn = AzureOpenAIEmbeddings(
            azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY
        )
        # 2. Initialize ChromaDB Client
        # 'PersistentClient' ensures data is saved to the disk folder 'chroma_db'.
        # If we used 'Client()', data would be lost when the program stops.
        self.client = chromadb.PersistentClient(path="chroma_db")
        
        # 3. Create/Get Collection
        # A 'Collection' is like a table in SQL or a folder for specific data.
        # We set 'hnsw:space': 'cosine' to tell the DB to use Cosine Similarity 
        # for finding matches (measuring the angle between vectors).
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Connected to collection: '{collection_name}'")

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add processed text chunks to the database.
        
        Args:
            chunks: List of dictionaries containing 'text' and 'metadata'.
        """
        if not chunks:
            print("No documents to add.")
            return

        print(f"Adding {len(chunks)} documents to vector store...")
        
        # Prepare lists for ChromaDB
        texts = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # 1. Text: The actual content
            texts.append(chunk['text'])
            
            # 2. Metadata: Source, page number, etc.
            metadatas.append(chunk['metadata'])
            
            # 3. ID: A unique name for this chunk. 
            # We create it from the source filename and chunk index.
            # e.g., "report.pdf_chunk_5"
            # This prevents adding the same chunk twice (idempotency).
            safe_source = chunk['metadata']['source'].replace(" ", "_")
            uid = f"{safe_source}_{chunk['metadata'].get('chunk_index', '0')}"
            ids.append(uid)

        # 4. Generate Embeddings
        # We manually generate embeddings here to have full control, 
        # though Chroma can also do this automatically if configured.
        print("Generating embeddings (this may take a moment)...")
        embeddings = self.embedding_fn.embed_documents(texts)
        
        # 5. Add to Collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("Successfully added documents to ChromaDB.")

    def search(self, query: str, k: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for documents relevant to the query.
        
        Args:
            query (str): The user's question.
            k (int): Number of results to return.
            filter_dict (Dict): Optional filters (e.g., {"type": "pdf"}).
            
        Returns:
            List[Dict]: Top matches with text, metadata, and similarity score.
        """
        print(f"Searching for: '{query}' (Top {k})")
        
        # 1. Embed the Query
        # We convert the question into numbers using the SAME model as the documents.
        query_embedding = self.embedding_fn.embed_query(query)
        
        # 2. Query ChromaDB
        # We pass the 'filter_dict' to the 'where' parameter.
        # This lets us say: "Find matches, BUT only inside 'report.pdf'"
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_dict  # This handles the filtering logic
        )
        
        formatted_results = []
        
        # 3. Process Results
        if results['documents']:
            # results['documents'] is a list of lists (one list per query).
            # We only asked one question, so we look at index 0.
            for i in range(len(results['documents'][0])):
                
                # Get the raw distance score (Cosine Distance)
                # Lower is better (0 = identical, 1 = opposite)
                distance = results['distances'][0][i] if results['distances'] else 0
                
                # Convert to "Similarity Score" (0 to 1) for easier understanding
                # 1.0 means perfect match.
                similarity = 1 - distance
                
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": round(similarity, 4)
                })
        
        print(f"Found {len(formatted_results)} relevant documents.")
        return formatted_results

    def clear(self):
        """Helper to wipe the database if needed (for testing)."""
        print("Clearing all data...")
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
