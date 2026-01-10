import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(override=True)
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
# CONFIGURATION
# This points to the vector_db and knowledge-base in the same parent folder (week-5)
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# Use the best embedding model available
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    api_version="2024-02-15-preview",
    azure_endpoint="https://llm-engineering-azure-learn.openai.azure.com/",
    api_key=subscription_key
)

def fetch_documents():
    """
    Loads text files and INJECTS the filename into the content.
    This fixes the 'Lost Context' problem.
    """
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    
    if not folders:
        print(f"Warning: No folders found in {KNOWLEDGE_BASE}")
    
    for folder in folders:
        if not os.path.isdir(folder):
            continue
            
        doc_type = os.path.basename(folder)
        # Load all Markdown files
        loader = DirectoryLoader(
            folder, 
            glob="**/*.md", 
            loader_cls=TextLoader, 
            loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        
        for doc in folder_docs:
            # CLEANUP: Get just the filename (e.g., "John_Doe_Bio.md")
            filename = os.path.basename(doc.metadata["source"])
            
            # THE TRICK: Prepend source info to the text content
            # Now every chunk will know who it is talking about!
            doc.page_content = f"[Source File: {filename}] content:\n{doc.page_content}"
            
            # Save metadata
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
            
    print(f"Loaded {len(documents)} documents.")
    return documents

#def create_chunks(documents):
    """
    Splits documents into smart chunks.
    Size 1000: Captures full context/paragraphs.
    Overlap 200: Prevents cutting sentences in half at the edges.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", " ", ""] # Try to split by paragraph first
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
             chunk_size=1000,
             chunk_overlap=200,
             separators=["\n\n", "\n", ".", " ", ""]
         )
    chunks = text_splitter.split_documents(documents)

         # --- ADD THIS LOOP ---
    for chunk in chunks:
            # Get filename from metadata
        filename = os.path.basename(chunk.metadata["source"])
            # Force it into the text of EVERY chunk
        chunk.page_content = f"[{filename}] {chunk.page_content}"
        # ---------------------

        print(f"Created {len(chunks)} chunks.")
    return chunks
def create_embeddings(chunks):
    """
    Stores chunks in the Vector Database (Chroma).
    """
    # Clear old database to avoid duplicates/mess
    if os.path.exists(DB_NAME):
        try:
            # Connect and delete collection if it exists
            vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
            vectorstore.delete_collection()
            print("Cleared old database collection.")
        except Exception as e:
            print(f"Note: Could not clear database (it might be empty): {e}")

    print("Generating embeddings... (this costs money, please wait)")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_NAME
    )

    count = vectorstore._collection.count()
    print(f"Success! Database contains {count:,} vector chunks.")
    return vectorstore

if __name__ == "__main__":
    print("--- Starting Ingestion (Improved) ---")
    docs = fetch_documents()
    if docs:
        chunks = create_chunks(docs)
        create_embeddings(chunks)
        print("--- Ingestion Complete ---")
    else:
        print("No documents found to process.")
