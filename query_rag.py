from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

INDEX_PATH = "index"
MODEL_PATH = "models/GRANITE-1B"

print("="*60)
print("SUSTAINABILITY POLICY & AWARENESS ASSISTANT")
print("="*60)

# Load FAISS index
print("\nLoading FAISS index...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Load Granite model
print("Loading Granite model...")
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

print("="*60)
print("\n🤖 Granite RAG System Ready!")
print("Type your questions about sustainability policies\n")

def query_rag(question, k=3):
    # Retrieve relevant chunks
    docs = db.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Build prompt
    prompt = f"""Use the following context to answer the question. If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    
    # Generate response
    output = pipe(prompt)
    response = output[0]["generated_text"]
    answer = response.split("Answer:")[-1].strip()
    
    # Display results
    print("\n" + "="*60)
    print(f"QUESTION: {question}")
    print("="*60)
    
    print(f"\nRETRIEVED CHUNKS ({k}):")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:200].replace('\n', ' ')
        print(f"\n[{i}] {preview}...")
    
    print("\n" + "="*60)
    print(f"ANSWER:\n{answer}")
    print("="*60 + "\n")

# Interactive loop
while True:
    try:
        question = input("Ask a question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!\n")
            break
        
        if not question:
            continue
        
        query_rag(question)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}\n")