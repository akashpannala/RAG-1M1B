# 🌱 Sustainability RAG Chatbot

Mini project for querying India's sustainability policies using RAG + Granite 1B.

## Setup

```bash
pip install -r requirements.txt
install from https://huggingface.co/ibm-granite/granite-4.0-1b/tree/main   and place it in as(models/GRANITE-1B/model.safetensors) due to push constrints i was not able to push it 
python build_index.py
streamlit run app.py
```

## How it Works

1. Loads sustainability docs from `docs/` folder(feel free to add your own documents into it and build index agian)
2. User asks a question
3. Finds relevant chunks using FAISS
4. Granite 1B generates answer from those chunks

## Try These

- "What are plastic waste rules in India?"
- "Energy saving tips?"
- "What is SDG 12?"

## Tech Stack

- Granite 1B (IBM LLM)
- LangChain (RAG pipeline)
- FAISS (vector search)
- Streamlit (UI)

## Project Goal

Help people understand sustainability policies aligned with SDG 11, 12, 13.

---

Built with ☕ and ADHD focus sessions
