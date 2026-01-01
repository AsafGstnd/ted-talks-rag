import pandas as pd
import tiktoken
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# 1. Load your API keys from the .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# 2. Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
# Using OpenAI client pointed to your LLMod.ai endpoint if needed
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.llmod.ai/v1"  # This is the crucial line for llmod.ai
)
index = pc.Index(INDEX_NAME)

# 3. Setup Hyperparameters (Based on Assignment Requirements)
CHUNK_SIZE = 768  # Max allowed is 2048 [cite: 42, 94]
OVERLAP_RATIO = 0.2  # Max allowed is 0.3 [cite: 43, 95]
OVERLAP_SIZE = int(CHUNK_SIZE * OVERLAP_RATIO)

# Use tiktoken to count tokens accurately for OpenAI models [cite: 98]
tokenizer = tiktoken.get_encoding("cl100k_base")


def get_token_chunks(text, chunk_size, overlap):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i: i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks


# 4. Load and Process the Data
# Ensure 'ted_talks_en.csv' is in your project root folder
df = pd.read_csv("ted_talks_en.csv")  # Start with 50 talks to stay under budget [cite: 39]

print(f"Starting ingestion for {len(df)} talks...")

for _, row in df.iterrows():
    # Combine Metadata and Transcript for richer retrieval [cite: 12]
    # We include title and description so the vector captures the topic well
    combined_text = f"Title: {row['title']}\nDescription: {row['description']}\nTranscript: {row['transcript']}"

    chunks = get_token_chunks(combined_text, CHUNK_SIZE, OVERLAP_SIZE)

    vectors = []
    for i, chunk_text in enumerate(chunks):
        # Generate embedding using the required model [cite: 33]
        embedding_res = client.embeddings.create(
            input=chunk_text,
            model="RPRTHPB-text-embedding-3-small"  # Updated to match your assignment
        )
        embedding = embedding_res.data[0].embedding

        # Prepare metadata exactly as required for the API output [cite: 74, 75, 76]
        metadata = {
            "talk_id": str(row['talk_id']),
            "title": str(row['title']),
            "chunk": chunk_text
        }

        vectors.append({
            "id": f"{row['talk_id']}_{i}",
            "values": embedding,
            "metadata": metadata
        })

    # Upsert to Pinecone [cite: 54]
    index.upsert(vectors=vectors)
    print(f"Uploaded {len(chunks)} chunks for talk: {row['title']}")

print("Ingestion complete!")
