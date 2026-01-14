import os
import PyPDF2
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Process a PDF file: extract text, split it into chunks, and preserve metadata.
    
    Args:
        file_path (str): The path to the PDF file.
        chunk_size (int): The target number of tokens per chunk.
        chunk_overlap (int): The number of tokens to overlap between chunks.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains 
                              the text chunk and its associated metadata.
    """
    
    # 1. Initialize the Output List
    # We will store all our processed chunks here to return them at the end.
    processed_chunks = []
    
    # Check if the file actually exists before trying to open it.
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    try:
        # 2. Open the PDF File
        # We open the file in 'rb' (read binary) mode because PDFs are binary files.
        with open(file_path, 'rb') as file:
            
            # Create a PDF reader object using PyPDF2
            reader = PyPDF2.PdfReader(file)
            
            # Get the total number of pages to loop through
            num_pages = len(reader.pages)
            print(f"Processing '{os.path.basename(file_path)}' with {num_pages} pages...")
            
            # 3. Initialize the Text Splitter
            # We use LangChain's RecursiveCharacterTextSplitter.
            # 'from_tiktoken_encoder' ensures we count 'tokens' (how LLMs see text) 
            # instead of just raw characters. 
            # - chunk_size=1000: Each chunk will be roughly 1000 tokens.
            # - chunk_overlap=100: The last 100 tokens of chunk 1 will repeat at the start 
            #   of chunk 2. This ensures context isn't lost at the cut points.
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            # Global counter to ensure unique IDs across all pages
            global_chunk_index = 0

            # 4. Iterate Through Each Page
            for page_num, page in enumerate(reader.pages):
                try:
                    # Extract text from the current page
                    page_text = page.extract_text()
                    
                    # Skip empty pages or pages where text extraction failed
                    if not page_text or not page_text.strip():
                        continue
                        
                    # 5. Split Text into Chunks
                    chunks = text_splitter.split_text(page_text)
                    
                    # 6. Add Metadata and Store
                    for chunk_text in chunks:
                        chunk_data = {
                            "text": chunk_text,
                            "metadata": {
                                "source": os.path.basename(file_path),  # Filename
                                "page": page_num + 1,                   # Page number (1-based)
                                "chunk_index": global_chunk_index       # Unique ID across file
                            }
                        }
                        processed_chunks.append(chunk_data)
                        global_chunk_index += 1
                        
                except Exception as e:
                    # If a specific page fails, we log it but continue processing other pages.
                    print(f"Warning: Could not process page {page_num + 1} in {file_path}. Error: {e}")

    except Exception as e:
        # If the file itself is corrupted or cannot be read, we catch that here.
        print(f"Error processing PDF {file_path}: {e}")
        return []

    print(f"Successfully created {len(processed_chunks)} chunks from {os.path.basename(file_path)}.")
    return processed_chunks
