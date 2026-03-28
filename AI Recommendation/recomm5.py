import os
import json
from datetime import datetime
from pinecone import Pinecone
from openai import OpenAI
import re
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

# --- Pinecone Indexes ---
BRAND_INDEX = "brand-embed"
OFFER_INDEX = "offers-emb"

brand_index = pc.Index(BRAND_INDEX)
offer_index = pc.Index(OFFER_INDEX)


# --- Query Pinecone ---
def query_index(index, query, top_k=623):
    """Query Pinecone index for vector matches"""
    query_embedding = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding

    res = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return res.get("matches", [])


# --- Build GPT prompt ---
def build_prompt(user_query, brand_matches, offer_matches, today):
    brands_str = "\n".join([
        f"- {b['metadata'].get('brand_name', b['id'])}" for b in brand_matches
    ])
    offers_str = "\n".join([
        f"- Brand: {o['metadata'].get('brand', 'Unknown')} | Offer: {o['metadata'].get('title', '')} | Description: {o['metadata'].get('description', '')}"
        for o in offer_matches
    ])

    return f"""
You are simulating a **LightFM hybrid recommendation model** (collaborative filtering + content-based filtering).

Today's date: {today}
User query: "{user_query}"

Candidate Brands:
{brands_str}

Candidate Offers:
{offers_str}

Your task:
1. Pretend you have implicit user-item interactions (clicks, views, purchases) AND content features (offer text, brand, seasonality).
2. Score each offer using a LightFM-like reasoning process:
   - Collaborative: Does this query align with offers other users chose in the past?
   - Content: Does the offer text/brand/features semantically align with the query?
3. Output **all offers that are relevant**, ranked by LightFM-style hybrid score.
4. Return strictly JSON with fields: brand, offer, rank, score, reason.
5. ⚠️ You must output all offers. Do not truncate, summarize, or limit to top 10/20. Every relevant offer must appear in the ranked JSON.


Output JSON only:
{{
  "data": [
    {{
      "brand": "Brand Name",
      "offer": "Offer text",
      "rank": 1,
      "score": 0.92,
      "reason": "Matched due to strong content similarity + seasonal trend"
    }}
  ]
}}


"""


# --- Get ranked offers ---
def get_ranked_offers(user_query, today):
    brand_matches = query_index(brand_index, user_query, top_k=50)
    offer_matches = query_index(offer_index, user_query, top_k=100)

    if not brand_matches and not offer_matches:
        return {"data": []}

    prompt = build_prompt(user_query, brand_matches, offer_matches, today)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()

    # Try extracting JSON using regex
    try:
        match = re.search(r"\{[\s\S]*\}", raw_output)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            return {"data": [], "error": "No JSON found in response"}
    except Exception as e:
        return {"data": [], "error": f"Invalid JSON: {str(e)}"}

# --- Main ---
if __name__ == "__main__":
    user_query = input("What are you looking for? ").strip()
    today = datetime.now().strftime("%B %d, %Y")

    result = get_ranked_offers(user_query, today)

    print("\n✅ Ranked Offers:\n")
    print(json.dumps(result, indent=2))
