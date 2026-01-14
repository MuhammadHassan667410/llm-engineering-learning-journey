import re
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_video_id(url: str) -> str:
    """
    Extract the 11-character video ID from a YouTube URL.
    Examples:
    - https://www.youtube.com/watch?v=dQw4w9WgXcQ -> dQw4w9WgXcQ
    - https://youtu.be/dQw4w9WgXcQ -> dQw4w9WgXcQ
    """
    # Regex pattern to find the ID. 
    # It looks for 'v=' followed by 11 chars, OR the last 11 chars after a slash.
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_video_title(url: str) -> str:
    """
    Fetch the video title by downloading the page HTML.
    We do this because the Transcript API doesn't provide the title.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # YouTube titles are usually in the <title> tag, ending with " - YouTube"
        title = soup.title.string.replace(" - YouTube", "")
        return title
    except Exception:
        return "Unknown Video Title"

def process_youtube_video(url: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Download the transcript of a YouTube video, chunk it, and add timestamped links.
    
    Args:
        url (str): The full URL of the YouTube video.
        chunk_size (int): Target size of text chunks.
        chunk_overlap (int): Overlap size.
        
    Returns:
        List[Dict[str, Any]]: Processed chunks with metadata and timestamp links.
    """
    processed_chunks = []
    video_id = get_video_id(url)
    
    if not video_id:
        print(f"Error: Could not find a valid video ID in {url}")
        return []

    try:
        # 1. Fetch Metadata (Title)
        print(f"Fetching title for {video_id}...")
        video_title = get_video_title(url)

        # 2. Fetch Transcript
        # We use list_transcripts first to find available languages
        print(f"Fetching transcript list for {video_title}...")
        
        # Instantiate the API client
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.list(video_id)
        
        # Try to fetch English, or translate, or fallback to any generated one
        try:
             # Try fetching manually created English transcript
             transcript_obj = transcript_list.find_transcript(['en'])
        except:
             try:
                 # If no manual english, try auto-generated english
                 transcript_obj = transcript_list.find_transcript(['en-US', 'en-GB'])
             except:
                 # If no english at all, take the first available and translate to english
                 # or just take the first one found
                 transcript_obj = transcript_list.find_generated_transcript(['en'])
        
        # If still nothing, just grab the first one we can iterate over
        if not 'transcript_obj' in locals():
            # This iterates to find the first available one
            for t in transcript_list:
                transcript_obj = t
                break

        print(f"Downloading transcript ({transcript_obj.language})...")
        transcript = transcript_obj.fetch()
        
        # 3. Combine Text for Chunking
        # The splitter expects one long string. We join all lines with spaces.
        # However, we need to map the result back to timestamps later. 
        # For simplicity in this version, we will group the raw transcript items 
        # until they reach the chunk size, instead of using the TextSplitter on a plain string.
        # This ensures our timestamps are perfectly accurate.
        
        current_chunk_text = []
        current_chunk_start = 0.0
        current_token_count = 0
        chunk_idx = 0  # Initialize chunk counter
        
        # Simple approximation: 1 word ≈ 1.3 tokens (safe overestimate)
        # We iterate through every small caption line.
        for item in transcript:
            text = item.text
            start_time = item.start
            
            # Estimate token count (splitting by space)
            item_tokens = len(text.split())
            
            # If this is the start of a new chunk, record the time
            if not current_chunk_text:
                current_chunk_start = start_time
            
            current_chunk_text.append(text)
            current_token_count += item_tokens
            
            # If chunk is full, save it and reset
            if current_token_count >= chunk_size:
                # Join the list of sentences into one paragraph
                full_text = " ".join(current_chunk_text)
                
                # Create the timestamped link
                # Cast to int to remove decimals (e.g., 120.5 -> 120)
                timestamp_link = f"https://www.youtube.com/watch?v={video_id}&t={int(current_chunk_start)}s"
                
                chunk_data = {
                    "text": full_text,
                    "metadata": {
                        "source": url,
                        "title": video_title,
                        "start_time": current_chunk_start,
                        "url_with_timestamp": timestamp_link,
                        "type": "youtube",
                        "chunk_index": chunk_idx  # Add unique index
                    }
                }
                processed_chunks.append(chunk_data)
                
                # Reset for next chunk
                current_chunk_text = []
                current_token_count = 0
                chunk_idx += 1  # Increment counter

        # Add any remaining text as the final chunk
        if current_chunk_text:
            full_text = " ".join(current_chunk_text)
            timestamp_link = f"https://www.youtube.com/watch?v={video_id}&t={int(current_chunk_start)}s"
            
            processed_chunks.append({
                "text": full_text,
                "metadata": {
                    "source": url,
                    "title": video_title,
                    "start_time": current_chunk_start,
                    "url_with_timestamp": timestamp_link,
                    "type": "youtube",
                    "chunk_index": chunk_idx  # Add unique index
                }
            })

    except Exception as e:
        # Common errors: Video has no captions, Video is private, Bad ID
        print(f"Error processing YouTube video {url}: {e}")
        return []

    print(f"Successfully created {len(processed_chunks)} chunks from video.")
    return processed_chunks
