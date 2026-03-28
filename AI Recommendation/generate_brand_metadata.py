import os
import json
import openai
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- API Key ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load offers.json ---
with open("offers.json", "r", encoding="utf-8") as f:
    offers = json.load(f)

# --- Extract unique brand names ---
brands = set()
for offer in offers:
    brand_info = offer.get("brand")
    if isinstance(brand_info, dict):
        brand_name = brand_info.get("name")
        if brand_name:
            brands.add(brand_name.strip())

brands = sorted(brands)
print(f"Found {len(brands)} unique brands")

# --- Function: Generate metadata using GPT ---
def generate_metadata(brand_name):
    prompt = f"""
Generate structured metadata for the brand below in the same format as this example:

Brand Name: Puma
Category: Sportswear and accessories
Products: athletic shoes, sports apparel, fashion collaborations, sports equipment
Target Audience: youth-oriented, active lifestyle customers, athletes, sports enthusiasts
Operating Regions: global presence, strong markets in India, Europe, USA, and Asia-Pacific
Cultural or Seasonal Association: popular before sports seasons, during back-to-school sales, and festive shopping events like Diwali
Offer Types: discounts on footwear, apparel bundles, limited-edition releases, free shipping
Discount Range: 10% to 30%
Relevant Product Categories in Offers: sports shoes, running gear, jerseys, gym bags
Position in Market: premium sportswear brand with lifestyle appeal
USP: high-performance sportswear combining athletic technology with fashion-forward design

---
Now generate metadata for brand: {brand_name}
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# --- Generate metadata for all brands ---
brand_metadata = []
for brand in tqdm(brands, desc="Generating metadata"):
    try:
        metadata_text = generate_metadata(brand)
        brand_metadata.append({
            "id": brand.lower().replace(" ", "_"),
            "brand": brand,
            "text": metadata_text
        })
    except Exception as e:
        print(f"Error with {brand}: {e}")

# --- Save to brand_metadata.json ---
with open("brand_metadata.json", "w", encoding="utf-8") as f:
    json.dump(brand_metadata, f, indent=2, ensure_ascii=False)

print("✅ brand_metadata.json created successfully!")
