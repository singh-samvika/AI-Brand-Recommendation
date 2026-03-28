import os
import json
import re
import openai
from tqdm import tqdm
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
openai.api_key = os.getenv("OPENAI_API_KEY")

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# --- Pinecone Index ---
INDEX_NAME = "brand-embed"
index = pc.Index(INDEX_NAME)

# --- Helpers ---
def sanitize_id(name: str) -> str:
    """
    Sanitize brand name into Pinecone-friendly ID
    (only lowercase, ascii, digits, underscores, hyphens).
    """
    clean = name.lower()
    clean = re.sub(r"[^a-z0-9_-]", "_", clean)
    return clean

def upsert_brand_embeddings(metadata_list, batch_size=50, max_len=2000):
    """
    Generate embeddings for brand metadata and upsert into Pinecone.
    - metadata_list: list of dicts {brand, text}
    - batch_size: how many vectors to send per upsert
    - max_len: truncate metadata text to avoid size errors
    """
    vectors = []

    for i, brand in enumerate(tqdm(metadata_list, desc="Embedding brands"), 1):
        try:
            # sanitize ID
            brand_id = sanitize_id(brand["brand"])

            # truncate long metadata
            text = brand["text"][:max_len]

            # generate embedding
            embedding = openai.embeddings.create(
                model="text-embedding-3-large",
                input=text
            ).data[0].embedding

            # prepare vector
            vectors.append({
                "id": brand_id,
                "values": embedding,
                "metadata": {
                    "brand": brand["brand"],
                    "text": text
                }
            })

            # batch upsert
            if i % batch_size == 0 or i == len(metadata_list):
                index.upsert(vectors)
                print(f"✅ Upserted {i}/{len(metadata_list)} brands")
                vectors = []

        except Exception as e:
            print(f"❌ Error embedding {brand['brand']}: {e}")

# --- Main ---
if __name__ == "__main__":
    # Load brand metadata file
    with open("brand_metadata.json", "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    print(f"Loaded {len(metadata_list)} brands from brand_metadata.json")

    # Run upsert
    upsert_brand_embeddings(metadata_list)
