import os
import sys
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOCS_PATH = "docs"
INDEX_PATH = "index"

print("="*60)
print("BUILDING RAG INDEX")
print("="*60)

# Check if docs folder exists
if not os.path.exists(DOCS_PATH):
    print(f"ERROR: {DOCS_PATH} folder doesn't exist!")
    sys.exit(1)

# Load documents
docs = []
files = os.listdir(DOCS_PATH)

if not files:
    print(f"ERROR: No files in {DOCS_PATH}/")
    print("Add .pdf or .txt files first!")
    sys.exit(1)

print(f"Found {len(files)} files in {DOCS_PATH}/\n")

for file in files:
    path = os.path.join(DOCS_PATH, file)
    try:
        if file.lower().endswith(".pdf"):
            loaded = PyPDFLoader(path).load()
            docs.extend(loaded)
            print(f"✓ Loaded {len(loaded)} pages from {file}")
        elif file.lower().endswith(".txt"):
            loaded = TextLoader(path).load()
            docs.extend(loaded)
            print(f"✓ Loaded {file}")
    except Exception as e:
        print(f"✗ Failed to load {file}: {e}")

if not docs:
    print("\nERROR: No documents loaded!")
    sys.exit(1)

print(f"\nTotal documents: {len(docs)}")

# Split into chunks
print("\nSplitting documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# Load embeddings
print("\nLoading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build FAISS index
print("\nBuilding FAISS index...")
db = FAISS.from_documents(chunks, embeddings)

# Save
os.makedirs(INDEX_PATH, exist_ok=True)
db.save_local(INDEX_PATH)

print("="*60)
print("✅ INDEX BUILD COMPLETE!")
print("="*60)
print("\nRun 'python query_rag.py' to start querying\n")