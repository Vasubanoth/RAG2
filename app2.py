__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
import time
from pypdf import PdfReader
from fastembed import TextEmbedding
import chromadb
from chromadb.config import Settings
from groq import Groq

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="📚 Smart RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR BETTER UI ---
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1E3A8A;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #4338CA;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    /* Chat input */
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 70%;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #4F46E5;
    }
    
    /* Status indicators */
    .success-status {
        background-color: #10B981;
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
    }
    
    .info-status {
        background-color: #3B82F6;
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("📚 Smart RAG Chatbot")
st.markdown("**Upload documents, ask questions, get intelligent answers**")
st.markdown("---")

# --- SIDEBAR: DOCUMENT UPLOAD & SETTINGS ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # API Key Input (with better UI)
    st.markdown("#### 🔑 API Configuration")
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
        except:
            pass
    
    if not api_key:
        api_key = st.text_input("Enter Groq API Key", type="password", help="Get your API key from console.groq.com")
        st.markdown("[Get API Key](https://console.groq.com)")
        if not api_key:
            st.warning("⚠️ API key required to continue")
            st.stop()
    
    st.markdown("---")
    
    # Document Upload Section
    st.markdown("### 📄 Document Upload")
    st.markdown("Upload PDF or text files to build your knowledge base")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload multiple PDF or text files"
    )
    
    if uploaded_files:
        st.markdown("#### 📋 Selected Files")
        for i, file in enumerate(uploaded_files):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{i+1}. {file.name}**")
            with col2:
                st.markdown(f"`{file.size // 1024} KB`")
    
    # Processing Options
    st.markdown("---")
    st.markdown("### ⚡ Processing Options")
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", min_value=200, max_value=2000, value=800, step=100, 
                                    help="Size of text chunks in characters")
    with col2:
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=100, step=50,
                                       help="Overlap between chunks for context preservation")
    
    # Process Button
    st.markdown("---")
    process_col1, process_col2 = st.columns([3, 1])
    with process_col1:
        process_btn = st.button("🚀 Process & Train", use_container_width=True)
    with process_col2:
        if st.button("🔄 Clear", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()

# --- MAIN CONTENT AREA ---
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "collection_count" not in st.session_state:
    st.session_state.collection_count = 0

# --- PROCESS FILES WITH VISUAL FEEDBACK ---
if process_btn and uploaded_files:
    with st.spinner("🔄 Initializing..."):
        # Initialize resources
        @st.cache_resource
        def load_resources():
            embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            DB_DIR = os.path.join(tempfile.gettempdir(), "chroma_db_persistent")
            chroma_client = chromadb.PersistentClient(
                path=DB_DIR, 
                settings=Settings(anonymized_telemetry=False)
            )
            return embedder, chroma_client
        
        embedder, chroma_client = load_resources()
        client = Groq(api_key=api_key)
    
    # Progress container
    progress_container = st.container()
    with progress_container:
        st.markdown("### 📊 Processing Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Clear existing collection
        status_text.markdown("<div class='info-status'>Step 1: Preparing database...</div>", unsafe_allow_html=True)
        try:
            chroma_client.delete_collection("rag_collection")
        except:
            pass
        progress_bar.progress(10)
        time.sleep(0.5)
        
        # Step 2: Extract text
        status_text.markdown("<div class='info-status'>Step 2: Extracting text from documents...</div>", unsafe_allow_html=True)
        all_chunks = []
        for idx, file in enumerate(uploaded_files):
            text = ""
            try:
                if file.name.endswith(".pdf"):
                    reader = PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                elif file.name.endswith(".txt"):
                    text = file.read().decode("utf-8")
                
                # Chunking
                for i in range(0, len(text), chunk_size - chunk_overlap):
                    chunk = text[i : i + chunk_size]
                    if len(chunk) > 50:
                        all_chunks.append(chunk)
                        
                progress_bar.progress(20 + (idx * 30 // len(uploaded_files)))
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                continue
        
        progress_bar.progress(50)
        
        if all_chunks:
            # Step 3: Create collection and embeddings
            status_text.markdown("<div class='info-status'>Step 3: Creating embeddings...</div>", unsafe_allow_html=True)
            
            def get_collection():
                return chroma_client.get_or_create_collection(
                    name="rag_collection",
                    metadata={"hnsw:space": "cosine"}
                )
            
            collection = get_collection()
            
            # Batch processing with progress
            batch_size = 100
            total_batches = (len(all_chunks) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(all_chunks), batch_size):
                batch = all_chunks[batch_idx : batch_idx + batch_size]
                embeddings = [e.tolist() for e in list(embedder.embed(batch))]
                ids = [f"id_{batch_idx + j}" for j in range(len(batch))]
                collection.add(documents=batch, embeddings=embeddings, ids=ids)
                
                current_progress = 50 + (batch_idx * 40 // len(all_chunks))
                progress_bar.progress(current_progress)
                
                status_text.markdown(f"""
                <div class='info-status'>
                    Processing batch {batch_idx//batch_size + 1}/{total_batches}: {len(batch)} chunks
                </div>
                """, unsafe_allow_html=True)
            
            progress_bar.progress(100)
            
            # Success message
            status_text.markdown(f"""
            <div class='success-status'>
                ✅ Processing Complete!
            </div>
            """, unsafe_allow_html=True)
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Documents", len(uploaded_files))
            with col2:
                st.metric("🔗 Chunks", len(all_chunks))
            with col3:
                st.metric("💾 Database", "Ready")
            
            st.session_state.processed = True
            st.session_state.collection_count = len(all_chunks)
            st.balloons()
            
        else:
            st.error("❌ No text could be extracted from the uploaded files")

# --- CHAT INTERFACE ---
st.markdown("### 💬 Chat with Your Documents")

# Show status indicator
if st.session_state.processed:
    st.success(f"✅ Knowledge base ready with {st.session_state.collection_count} chunks")
else:
    st.info("ℹ️ Upload documents and click 'Process & Train' to start chatting")

# Display chat messages with avatars
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Chat input with better placeholder
if prompt := st.chat_input(f"💭 Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # Check if processed
    if not st.session_state.processed:
        with st.chat_message("assistant", avatar="🤖"):
            st.warning("Please upload and process documents first!")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Please upload and process documents first!"
        })
        st.rerun()
    
    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching documents..."):
            try:
                # Get collection
                embedder, chroma_client = load_resources()
                collection = chroma_client.get_collection("rag_collection")
                client = Groq(api_key=api_key)
                
                # Get query embedding
                q_embed = list(embedder.embed([prompt]))[0].tolist()
                
                # Retrieve relevant chunks
                results = collection.query(query_embeddings=[q_embed], n_results=5)
                
                if results['documents'] and results['documents'][0]:
                    context = "\n".join(results['documents'][0])
                    
                    # Limit context length
                    if len(context) > 8000:
                        context = context[:8000]
                    
                    # Create enhanced prompt
                    sys_prompt = f"""You are a helpful assistant answering questions based on provided context.

                    CONTEXT:
                    {context}

                    QUESTION: {prompt}

                    INSTRUCTIONS:
                    1. Answer based ONLY on the provided context
                    2. If the context doesn't contain relevant information, say "I couldn't find relevant information in the documents"
                    3. Be concise and accurate
                    4. Cite specific information when possible
                    
                    ANSWER:"""
                    
                    # Get response
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": sys_prompt}],
                        max_tokens=800,
                        temperature=0.3
                    )
                    answer = response.choices[0].message.content
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show source information (collapsible)
                    with st.expander("📚 View Source Context"):
                        st.markdown("**Relevant passages found:**")
                        for i, doc in enumerate(results['documents'][0][:3]):
                            st.markdown(f"**Passage {i+1}:**")
                            st.info(doc[:300] + "..." if len(doc) > 300 else doc)
                            st.divider()
                
                else:
                    answer = "I couldn't find relevant information in the uploaded documents."
                    st.warning(answer)
                    
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    answer = "⏳ **Rate Limit Hit:** Please wait 60 seconds before asking another question."
                    st.error(answer)
                else:
                    answer = f"❌ Error: {str(e)}"
                    st.error(answer)
    
    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- FOOTER / INFO SECTION ---
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🔧 Built with:**")
    st.markdown("- Streamlit")
    st.markdown("- Groq API")
    st.markdown("- ChromaDB")
with col2:
    st.markdown("**⚡ Features:**")
    st.markdown("- PDF/TXT Support")
    st.markdown("- Smart Chunking")
    st.markdown("- Vector Search")
with col3:
    st.markdown("**📊 Stats:**")
    if st.session_state.processed:
        st.markdown(f"- Documents: {len(uploaded_files) if uploaded_files else 0}")
        st.markdown(f"- Chunks: {st.session_state.collection_count}")
    else:
        st.markdown("- No documents processed")
