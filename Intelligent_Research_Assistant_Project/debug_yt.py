import youtube_transcript_api
import inspect
print(f"Location: {youtube_transcript_api.__file__}")
print(f"Version: {youtube_transcript_api.__version__ if hasattr(youtube_transcript_api, '__version__') else 'Unknown'}")
from youtube_transcript_api import YouTubeTranscriptApi
print(f"Class: {YouTubeTranscriptApi}")
print(f"Dir: {dir(YouTubeTranscriptApi)}")