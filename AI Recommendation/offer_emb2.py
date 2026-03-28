import os
import json
import openai
from pinecone import Pinecone
from tqdm import tqdm
import re
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
openai.api_key = os.getenv("OPENAI_API_KEY")

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# --- Pinecone Index ---
INDEX_NAME = "offers-emb"
index = pc.Index(INDEX_NAME)

# --- Clean up IDs (ASCII only) ---
def safe_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(text))[:100]

# --- Sanitize metadata ---
def clean_metadata(meta: dict) -> dict:
    cleaned = {}
    for k, v in meta.items():
        if v is None:
            cleaned[k] = ""  # replace null with empty string
        elif isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, list):
            # Only keep strings inside list
            cleaned[k] = [str(x) for x in v if x is not None]
        else:
            cleaned[k] = str(v)  # fallback: cast to string
    return cleaned

# --- Load offers.json ---
with open("offers.json", "r", encoding="utf-8") as f:
    offers_data = json.load(f)

print(f"Loaded {len(offers_data)} offers")

# --- Step 2: Create embeddings and upsert ---
def upsert_offer_embeddings():
    batch = []
    total = len(offers_data)

    pbar = tqdm(total=total, desc="Embedding offers")
    count = 0

    for entry in offers_data:
        brand = entry.get("brand", {}).get("name", "Unknown")
        title = entry.get("title", "")
        description = entry.get("description", "")
        category = entry.get("brand", {}).get("brand_category", {}).get("name", "")
        region = entry.get("region", {}).get("name", "")
        expiry = entry.get("expiry", "")

        # Build embedding text
        text = f"""
        Brand: {brand}
        Title: {title}
        Description: {description}
        Category: {category}
        Region: {region}
        Expiry: {expiry}
        """

        try:
            emb = openai.embeddings.create(
                model="text-embedding-3-large",
                input=text
            ).data[0].embedding
        except Exception as e:
            print(f"❌ Error embedding {brand}: {e}")
            continue

        metadata = {
            "brand": brand,
            "title": title,
            "description": description,
            "category": category,
            "region": region,
            "expiry": expiry
        }

        vector = {
            "id": safe_id(f"{brand}_{entry['id']}"),
            "values": emb,
            "metadata": clean_metadata(metadata)
        }

        batch.append(vector)
        count += 1
        pbar.update(1)

        if len(batch) == 50:  # Upsert in chunks
            index.upsert(batch)
            print(f"✅ Upserted {count}/{total} offers")
            batch = []

    if batch:
        index.upsert(batch)
        print(f"✅ Upserted final {len(batch)} offers")

    pbar.close()

# --- Run ---
if __name__ == "__main__":
    upsert_offer_embeddings()
    print("🎉 All offers embedded and upserted into Pinecone!")
