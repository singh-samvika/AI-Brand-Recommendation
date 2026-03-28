# 🚀 AI-Powered Hybrid Brand Recommendation System

## A production-style recommendation engine that combines **LLM reasoning + vector search** to simulate a **hybrid recommendation system inspired by LightFM** (collaborative + content-based filtering).

## This project demonstrates how modern AI systems can replace traditional ML pipelines using **semantic retrieval and LLM-based reasoning**.
---

## 🧠 System Overview

This system is designed to mimic real-world recommendation pipelines used in:

* E-commerce personalization
* Credit card reward systems
* Offer targeting platforms

Instead of training a traditional model, it leverages:

* **Embeddings (semantic similarity)**
* **LLM-based ranking (reasoning layer)**

---

## ⚙️ Architecture


User Query
    ↓
Embedding Generation (OpenAI)
    ↓
Vector Search (Pinecone)
    ↓
Candidate Retrieval
    ↓
LLM Hybrid Scoring (LightFM-inspired reasoning)
    ↓
Ranked Recommendations (JSON)


---

## 🧩 Core Components

### 1️⃣ Brand Metadata Generation

* Uses LLM to generate structured brand intelligence:

  * Category, audience, positioning
  * Seasonal relevance
  * Offer compatibility

---

### 2️⃣ Offer Embeddings

* Converts offers into dense vectors using:

  * `text-embedding-3-large`
* Stores vectors in Pinecone for:

  * Fast similarity search
  * Scalable retrieval

---

### 3️⃣ Hybrid Recommendation Layer (Key Innovation)

Instead of a trained LightFM model, this system **simulates hybrid scoring using LLM reasoning**:

#### 🔹 Collaborative Filtering (Simulated)

* Assumes implicit signals:

  * clicks, purchases, preferences
* Infers:

  > “Would similar users engage with this offer?”

#### 🔹 Content-Based Filtering

* Measures semantic alignment between:

  * user query
  * brand metadata
  * offer text

#### 🔹 Final Ranking

* LLM assigns:

  * score
  * rank
  * reasoning
* Outputs structured JSON

---

## 🔥 Why This Approach Matters

Traditional systems:

* Require large datasets
* Need training + tuning
* Struggle with cold start

This system:

* Leverages **LLM-based reasoning for contextual ranking**
* Handles **new users & new offers instantly**
* Uses **semantic understanding instead of keywords**

---

## 🛠️ Tech Stack

* **Python**
* **OpenAI API**

  * GPT (reasoning)
  * Embeddings
* **Pinecone**

  * Vector database
* JSON-based pipeline

---

## 📁 Project Structure

```
AI-Recommendation/
│
├── generate_brand_metadata.py   # LLM-based brand profiling
├── offer_emb2.py                # Embedding + Pinecone upload
├── recomm5.py                   # Hybrid recommendation engine
├── upsert_emb.py                # Vector storage utilities
├── offers.json                  # Input dataset
├── brand_metadata.json          # Generated metadata
├── .env                         # Environment variables (ignored)
```

---

## ⚙️ Setup

### 1. Clone repository

```
git clone https://github.com/YOUR_USERNAME/ai-brand-recommendation.git
cd ai-brand-recommendation
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
```

---

## ▶️ Running the Pipeline

### Generate metadata

```
python generate_brand_metadata.py
```

### Create embeddings

```
python offer_emb2.py
```

### Run recommendations

```
python recomm5.py
```

---

## 🧪 Example

**Query:**

> “Looking for sports shoe discounts for college students”

**System Behavior:**

* Understands:

  * “sports shoes” → category
  * “college students” → audience
* Retrieves semantically relevant offers
* Ranks them using hybrid reasoning

---

## 📊 Output Format

```json
[
  {
    "brand": "Nike",
    "offer": "20% off on running shoes",
    "rank": 1,
    "score": 0.92,
    "reason": "Strong semantic match with sports footwear and youth audience"
  }
]
```

---


## 🧠 Key Takeaways

* Demonstrates **modern recommendation system design**
* Combines:

  * Vector search
  * LLM reasoning
  * Hybrid ranking logic
* Avoids limitations of traditional ML pipelines

---

## 👩‍💻 Author

**Samvika Singh**
AI Enthusiast | Backend Developer
Focused on building intelligent, scalable systems

---

## ⭐ If you find this useful

Consider giving it a star — it helps visibility and supports further development.
