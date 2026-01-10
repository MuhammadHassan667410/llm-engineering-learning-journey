import os
import chromadb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px

# Configuration
# Path to the vector DB created by improved_ingest.py
DB_PATH = str(Path(__file__).parent / "vector_db")

def load_vectors():
    """
    Connects to ChromaDB and extracts embeddings + metadata.
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}. Run improved_ingest.py first.")
        return None, None, None

    print(f"Connecting to database at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # LangChain usually creates a collection named "langchain" by default
    # We check available collections to be safe
    collections = client.list_collections()
    if not collections:
        print("Error: No collections found in the database.")
        return None, None, None
    
    collection_name = collections[0].name
    print(f"Using collection: '{collection_name}'")
    
    collection = client.get_collection(collection_name)
    
    # Fetch all data (embeddings, metadata, documents)
    data = collection.get(include=['embeddings', 'metadatas', 'documents'])
    
    embeddings = np.array(data['embeddings'])
    metadatas = data['metadatas']
    documents = data['documents']
    
    print(f"Successfully loaded {len(embeddings)} vectors.")
    return embeddings, metadatas, documents

def create_visualizations():
    embeddings, metadatas, documents = load_vectors()
    
    if embeddings is None or len(embeddings) == 0:
        print("No data to visualize.")
        return

    print("Processing metadata...")
    # Extract useful labels for coloring the plots
    # We try to use 'doc_type' (folder name) or fallback to 'source' (filename)
    sources = []
    filenames = []
    
    for m in metadatas:
        # Get category/folder
        src = m.get('doc_type', 'Unknown')
        sources.append(src)
        
        # Get actual filename
        f_path = m.get('source', 'Unknown')
        filenames.append(os.path.basename(f_path))

    # Truncate text for cleaner hover tooltips
    short_texts = [doc[:300] + "..." if len(doc) > 300 else doc for doc in documents]

    print("Running dimensionality reduction...")
    
    # 1. t-SNE for 2D (Great for seeing clusters of related chunks)
    # Perplexity must be less than number of samples
    perp = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    projections_2d = tsne.fit_transform(embeddings)
    
    # 2. PCA for 3D (Better for preserving global structure in 3D)
    pca = PCA(n_components=3)
    projections_3d = pca.fit_transform(embeddings)

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'x_2d': projections_2d[:, 0],
        'y_2d': projections_2d[:, 1],
        'x_3d': projections_3d[:, 0],
        'y_3d': projections_3d[:, 1],
        'z_3d': projections_3d[:, 2],
        'Category': sources,
        'Filename': filenames,
        'Content': short_texts
    })

    print("Generating interactive plots...")

    # --- 2D Plot (t-SNE) ---
    fig_2d = px.scatter(
        df, 
        x='x_2d', 
        y='y_2d',
        color='Category', 
        symbol='Category',
        hover_data=['Filename', 'Content'],
        title="RAG Knowledge Base - 2D Projection (t-SNE)",
        template="plotly_dark",
        labels={'x_2d': 'Dimension 1', 'y_2d': 'Dimension 2'}
    )
    fig_2d.update_traces(marker=dict(size=8, opacity=0.8))
    
    print("Opening 2D Plot...")
    fig_2d.show()

    # --- 3D Plot (PCA) ---
    fig_3d = px.scatter_3d(
        df, 
        x='x_3d', 
        y='y_3d', 
        z='z_3d',
        color='Category',
        hover_data=['Filename', 'Content'],
        title="RAG Knowledge Base - 3D Projection (PCA)",
        template="plotly_dark",
        labels={'x_3d': 'PCA 1', 'y_3d': 'PCA 2', 'z_3d': 'PCA 3'}
    )
    fig_3d.update_traces(marker=dict(size=5, opacity=0.7))
    
    print("Opening 3D Plot...")
    fig_3d.show()

if __name__ == "__main__":
    create_visualizations()
