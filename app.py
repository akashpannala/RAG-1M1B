import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Page config
st.set_page_config(
    page_title="Sustainability Assistant",
    page_icon="🌱",
    layout="wide"
)

# Paths
INDEX_PATH = "index"
MODEL_PATH = "models/GRANITE-1B"

# Load models (with caching)
@st.cache_resource
def load_models():
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load FAISS
    db = FAISS.load_local(
        INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Load Granite
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    
    return db, pipe

# Query function
def query_rag(db, pipe, question, k=3):
    # Retrieve chunks
    docs = db.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Build prompt
    prompt = f"""Use the following context to answer the question. If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    
    # Generate
    output = pipe(prompt)
    response = output[0]["generated_text"]
    answer = response.split("Answer:")[-1].strip()
    
    return answer, docs

# Main UI
def main():
    st.title("🌱 Sustainability Policy & Awareness Assistant")
    st.markdown("*Ask questions about India's sustainability policies, SDGs, waste management, and energy conservation*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        k = st.slider("Number of chunks to retrieve", 1, 5, 3)
        
        st.markdown("---")
        st.header("📚 About")
        st.info("""
        This RAG system helps you understand:
        - 🗑️ Waste Management Rules
        - ⚡ Energy Conservation
        - 🌍 SDG 11, 12, 13
        - ♻️ Sustainability Policies
        """)
        
        st.markdown("---")
        st.header("💡 Example Questions")
        st.markdown("""
        - What are India's plastic waste rules?
        - How to save energy at home?
        - What is SDG 12?
        - How to segregate waste properly?
        """)
    
    # Load models
    if not os.path.exists(INDEX_PATH):
        st.error("❌ Index not found! Run `python build_index.py` first.")
        return
    
    with st.spinner("Loading models... (this takes a moment on first run)"):
        db, pipe = load_models()
    
    st.success("✅ Models loaded successfully!")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📄 View Sources"):
                    for i, doc in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:300] + "...")
                        st.markdown("---")
    
    # Chat input
    if question := st.chat_input("Ask a question about sustainability..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, docs = query_rag(db, pipe, question, k)
            
            st.markdown(answer)
            
            # Show sources
            with st.expander("📄 View Sources"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(doc.page_content[:300] + "...")
                    st.markdown("---")
            
            # Add to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": docs
            })
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()