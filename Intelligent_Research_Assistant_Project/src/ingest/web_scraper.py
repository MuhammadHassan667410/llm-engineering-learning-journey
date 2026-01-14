import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_url(url: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch a web page, extract its main content, clean it, and split it into chunks.
    
    Args:
        url (str): The web address to scrape.
        chunk_size (int): Target size of each text chunk in tokens.
        chunk_overlap (int): Overlap between chunks to preserve context.
        
    Returns:
        List[Dict[str, Any]]: A list of processed text chunks with metadata.
    """
    processed_chunks = []
    
    try:
        # 1. Fetch the Web Page
        # We use a 'User-Agent' header so the website thinks we are a standard browser 
        # (like Chrome) and not a bot. This helps prevent being blocked.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        # If the page didn't load correctly (e.g., 404 error), raise an error.
        response.raise_for_status()
        
        # 2. Parse the HTML
        # BeautifulSoup takes the raw HTML text and creates a tree of objects 
        # that we can search through (like "find all links" or "find the title").
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 3. Clean up "Noise"
        # We define a list of tags that usually contain content we don't want.
        # script/style: Code, not text.
        # nav: Menu buttons.
        # footer: Copyright info, sitemaps.
        # aside: Sidebars with ads or related links.
        noise_tags = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'noscript']
        for tag in soup(noise_tags):
            tag.decompose()  # This completely removes the tag and its content from the tree.

        # 4. Extract Main Content
        # We look for the most specific container first.
        # <article> is usually the best bet for news/blogs.
        # <main> is the standard HTML5 tag for the primary content.
        # If those fail, we fall back to the <body>.
        content_element = soup.find('article') or \
                          soup.find('main') or \
                          soup.find('div', class_='content') or \
                          soup.find('div', class_='main') or \
                          soup.body
        
        if not content_element:
            print(f"Warning: Could not identify content on {url}")
            return []

        # Get the clean text, separating blocks with newlines.
        # 'strip=True' removes leading/trailing whitespace.
        text = content_element.get_text(separator='\n', strip=True)
        
        # Get the page title for metadata
        page_title = soup.title.string.strip() if soup.title else "No Title"

        # 5. Split Text into Chunks
        # We use the same splitter as the PDF processor for consistency.
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = text_splitter.split_text(text)
        
        # 6. Add Metadata
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        for i, chunk_text in enumerate(chunks):
            chunk_data = {
                "text": chunk_text,
                "metadata": {
                    "source": url,
                    "title": page_title,
                    "date_scraped": current_date,
                    "chunk_index": i
                }
            }
            processed_chunks.append(chunk_data)
            
    except requests.RequestException as e:
        print(f"Network error fetching {url}: {e}")
    except Exception as e:
        print(f"Error processing {url}: {e}")
        
    print(f"Successfully scraped {len(processed_chunks)} chunks from {url}")
    return processed_chunks
