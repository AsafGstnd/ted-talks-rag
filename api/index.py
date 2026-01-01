from fastapi import FastAPI, Request
from openai import OpenAI
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.llmod.ai/v1")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# --- CONFIGURATION (Must match your ingest.py) ---
CONFIG = {
    "chunk_size": 768,
    "overlap_ratio": 0.2,
    "top_k": 7  # You can increase this up to 30
}


@app.get("/")
def read_root():
    """This prevents the 404 error on your main Vercel link."""
    return {"message": "TED Talks RAG API is live. Use /api/stats or /api/prompt."}

@app.get("/api/stats")
async def get_stats():
    """Mandatory endpoint to report your project settings."""
    return CONFIG


@app.post("/api/prompt")
async def ask_question(request: Request):
    data = await request.json()
    user_query = data.get("question")

    # 1. Embed Question
    xq = client.embeddings.create(
        input=user_query,
        model="RPRTHPB-text-embedding-3-small"
    ).data[0].embedding

    # 2. Retrieve Context
    res = index.query(vector=xq, top_k=CONFIG["top_k"], include_metadata=True)

    contexts = []
    context_text = ""
    for match in res['matches']:
        meta = match['metadata']
        contexts.append({
            "talk_id": meta.get("talk_id"),
            "title": meta.get("title"),
            "chunk": meta.get("chunk"),
            "score": match['score']
        })
        context_text += f"\n---\nTALK: {meta.get('title')}\n{meta.get('chunk')}\n"

    # 3. Formulate Prompt (MANDATORY SECTION)
    system_prompt = (
        "You are a TED Talk assistant that answers questions strictly and only based on the "
        "TED dataset context provided to you (metadata and transcript passages). "
        "You must not use any external knowledge, the open internet, or information that is "
        "not explicitly contained in the retrieved context. If the answer cannot be determined "
        "from the provided context, respond: 'I don't know based on the provided TED data.' "
        "Always explain your answer using the given context, quoting or paraphrasing the "
        "relevant transcript or metadata when helpful."
    )

    user_prompt = f"Context Material:\n{context_text}\n\nUser Question: {user_query}"

    # 4. Generate Answer
    response = client.chat.completions.create(
        model="RPRTHPB-gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 5. Return JSON [cite: 70-85]
    return {
        "response": response.choices[0].message.content,
        "context": contexts,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt
        }
    }