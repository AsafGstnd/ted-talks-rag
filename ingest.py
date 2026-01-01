import os
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 1. SETUP & CONFIGURATION
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("LLMOD_API_KEY")  # Use your LLMod key
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
BASE_URL = "https://api.llmod.ai/v1"  # Mandatory gateway

# 2. HYPERPARAMETERS
CHUNK_SIZE = 768  
OVERLAP_RATIO = 0.2  
OVERLAP_SIZE = int(CHUNK_SIZE * OVERLAP_RATIO)
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small" # Required model
DIMENSIONS = 1536 # Must match the model

# 3. INITIALIZE CLIENTS
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
tokenizer = tiktoken.get_encoding("cl100k_base") # Accurate tokenization

def get_token_chunks(text, chunk_size, overlap):
    """Accurately chunks text based on token counts."""
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i: i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

def process_and_upload_talk(row):
    """Handles the full pipeline for a single TED talk."""
    try:
        # Build the searchable text body
        combined_text = f"Title: {row['title']}\nDescription: {row['description']}\nTranscript: {row['transcript']}"
        chunks = get_token_chunks(combined_text, CHUNK_SIZE, OVERLAP_SIZE)
        
        vectors = []
        for i, chunk_text in enumerate(chunks):
            # Generate embedding
            res = client.embeddings.create(input=chunk_text, model=EMBEDDING_MODEL)
            embedding = res.data[0].embedding
            
            # Metadata must contain these 3 keys for the API
            vectors.append({
                "id": f"{row['talk_id']}_{i}",
                "values": embedding,
                "metadata": {
                    "talk_id": str(row['talk_id']),
                    "title": str(row['title']),
                    "chunk": chunk_text
                }
            })
        
        # Batch upsert for speed
        if vectors:
            index.upsert(vectors=vectors)
        return len(vectors)
    except Exception as e:
        return f"Error in talk {row['talk_id']}: {str(e)}"

# 4. EXECUTION ENGINE
if __name__ == "__main__":
    df = pd.read_csv("ted_talks_en.csv")
    print(f"🚀 Starting concurrent ingestion of {len(df)} talks...")

    # We use 10 workers to optimize speed vs rate limits
    total_chunks = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all talks to the thread pool
        future_to_talk = {executor.submit(process_and_upload_talk, row): row for _, row in df.iterrows()}
        
        # Use tqdm for a real-time progress bar
        for future in tqdm(as_completed(future_to_talk), total=len(df), desc="Ingesting"):
            result = future.result()
            if isinstance(result, int):
                total_chunks += result
            else:
                print(f"\n⚠️ {result}")

    print(f"\n✅ SUCCESS: Ingested {len(df)} talks ({total_chunks} total chunks).")
