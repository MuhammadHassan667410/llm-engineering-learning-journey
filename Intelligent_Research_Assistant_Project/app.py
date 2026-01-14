import gradio as gr
import os
from src.ingest.pdf_processor import process_pdf
from src.ingest.web_scraper import process_url
from src.ingest.youtube_processor import process_youtube_video
from src.retrieval.vector_store import VectorStore
from src.retrieval.conversation import ConversationManager
from src.retrieval.qa_chain import QAChain

# 1. Setup Backend Components
# These are the "Engine" parts that stay running while the app is open.
vector_db = VectorStore()
memory = ConversationManager()
rag_chain = QAChain(vector_db, memory)

# 2. Ingestion Logic
def ingest_files(files, urls, current_doc_list, progress=gr.Progress()):
    """
    Processes both uploaded files and URLs, adding them to the database.
    Updates the list of tracked documents.
    """
    if not files and not urls:
        return current_doc_list, "⚠️ No content provided."

    new_docs = []
    log_msg = ""

    # 1. Handle PDFs
    if files:
        progress(0, desc="Starting PDF processing...")
        for i, file in enumerate(files):
            try:
                progress((i / len(files)) * 0.5, desc=f"Processing {os.path.basename(file.name)}")
                chunks = process_pdf(file.name)
                vector_db.add_documents(chunks)
                
                # Add to our tracking list [Name, Type, Chunks, Status]
                new_docs.append([os.path.basename(file.name), "PDF", len(chunks), "✅ Ingested"])
                
            except Exception as e:
                new_docs.append([os.path.basename(file.name), "PDF", 0, "❌ Error"])
                log_msg += f"Error processing {file.name}: {e}\n"

    # 2. Handle URL
    if urls:
        url_list = urls.split("\n") # Support multiple URLs
        for i, url in enumerate(url_list):
            url = url.strip()
            if not url: continue
            
            try:
                progress(0.5 + (i / len(url_list)) * 0.5, desc=f"Scraping {url}...")
                
                if "youtube.com" in url or "youtu.be" in url:
                    chunks = process_youtube_video(url)
                    doc_type = "YouTube"
                else:
                    chunks = process_url(url)
                    doc_type = "Web"

                if chunks:
                    vector_db.add_documents(chunks)
                    new_docs.append([url, doc_type, len(chunks), "✅ Ingested"])
                else:
                    new_docs.append([url, doc_type, 0, "⚠️ No Text Found"])
                    
            except Exception as e:
                new_docs.append([url, doc_type, 0, "❌ Error"])
                log_msg += f"Error processing {url}: {e}\n"

    # Update the master list
    updated_list = current_doc_list + new_docs
    
    final_msg = f"Processed {len(new_docs)} new items."
    if log_msg:
        final_msg += f"\nErrors:\n{log_msg}"
        
    return updated_list, final_msg

# 3. Chat Logic
def chat_response(message, history):
    """
    The bridge between the Chatbot UI and our RAG chain.
    - message: What the user typed.
    - history: List of message dictionaries.
    """
    history = history or []
    
    # 1. Add User Message
    history.append({"role": "user", "content": message})
    
    # 2. Ask the Brain (QAChain)
    result = rag_chain.ask(message)
    answer = result['answer']
    sources = result['sources']
    
    # 3. Format Citations
    source_html = "<br><br><details><summary><strong>📚 Sources Used</strong></summary><ul>"
    if sources:
        for s in sources:
            label = s['label']
            link = s['link']
            if link:
                source_html += f"<li><a href='{link}' target='_blank'>{label}</a></li>"
            else:
                source_html += f"<li>{label}</li>"
    else:
        source_html += "<li>No specific context found.</li>"
    source_html += "</ul></details>"
    
    # 4. Construct AI Response
    full_response = answer + source_html
    history.append({"role": "assistant", "content": full_response})
    
    return history

def clear_all():
    """Wipes the chat memory and the database."""
    rag_chain.clear_history()
    return []

# 4. UI Layout (The "Blueprint")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Intelligent Research Assistant")
    gr.Markdown("Upload documents or paste links to start a conversation about your data.")
    
    doc_state = gr.State([])

    with gr.Tabs():
        # --- TAB 1: DATA INGESTION ---
        with gr.TabItem("📁 Manage Documents"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Add Content")
                    file_input = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
                    url_input = gr.Textbox(label="Enter URLs (one per line)", placeholder="https://example.com\nhttps://youtube.com/...", lines=3)
                    process_btn = gr.Button("🚀 Process Documents", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 2. Knowledge Base Status")
                    doc_table = gr.Dataframe(
                        headers=["Source Name", "Type", "Chunks", "Status"],
                        datatype=["str", "str", "number", "str"],
                        interactive=False
                    )
                    status_output = gr.Markdown("Ready to ingest.")

        # --- TAB 2: CHAT ASSISTANT ---
        with gr.TabItem("💬 Ask Questions"):
            # REMOVED type="messages", but logic expects dictionaries now
            chatbot = gr.Chatbot(height=500, placeholder="Ask me anything...")
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your question here...",
                    show_label=False,
                    scale=9
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("Clear Conversation")

    # 5. Connecting Everything (The "Wiring")
    
    process_btn.click(
        ingest_files, 
        inputs=[file_input, url_input, doc_state], 
        outputs=[doc_state, status_output]
    ).then(
        lambda x: x, inputs=[doc_state], outputs=[doc_table]
    )
    
    # Chat Wiring
    msg_input.submit(chat_response, inputs=[msg_input, chatbot], outputs=[chatbot])
    submit_btn.click(chat_response, inputs=[msg_input, chatbot], outputs=[chatbot])
    
    # Clear history logic
    clear_btn.click(clear_all, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
