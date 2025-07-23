from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import re
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
model = SentenceTransformer("all-mpnet-base-v2")

# Load embeddings
BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(BASE_DIR, "tag_embeddings.pkl"), "rb") as f:
    tag_data = pickle.load(f)
with open(os.path.join(BASE_DIR, "facility_embeddings.pkl"), "rb") as f:
    facility_data = pickle.load(f)
with open(os.path.join(BASE_DIR, "milestone_embeddings.pkl"), "rb") as f:
    milestone_data = pickle.load(f)

def get_all_states_and_districts():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT state FROM areas")
            states = [row[0] for row in cur.fetchall() if row[0]]
            cur.execute("SELECT DISTINCT district FROM areas")
            districts = [row[0] for row in cur.fetchall() if row[0]]
            return states, districts
    except Exception as e:
        print("Error loading states/districts:", e)
        return [], []
    finally:
        conn.close()

def match_filters_from_query(query, top_k=3, threshold=0.4):
    query_embedding = model.encode([query])[0].reshape(1, -1)

    def find_matches(data):
        names = [x[0] for x in data]
        embeddings = np.array([x[1] for x in data])
        sims = cosine_similarity(query_embedding, embeddings)[0]
        # Optional debug log:
        print(f"\n🔬 Similarity scores for :")
        for i, score in enumerate(sims):
            print(f"{names[i]} → {score:.3f}")
        matches = [names[i] for i, score in enumerate(sims) if score >= threshold]
        return matches[:top_k]

    matched_tags = find_matches(tag_data)
    matched_facilities = find_matches(facility_data)
    matched_milestones = find_matches(milestone_data)

    # Match state/district using keyword presence
    states, districts = get_all_states_and_districts()
    lower_query = query.lower()
    matched_state = next((s for s in states if s.lower() in lower_query), None)
    matched_district = next((d for d in districts if d.lower() in lower_query), None)

    # Optional: Extract max cost from query
    cost_match = re.search(r'(under|below)\s+(\d{3,5})', lower_query)
    max_cost = int(cost_match.group(2)) if cost_match else None

    return {
        "state": matched_state,
        "district": matched_district,
        "tags": matched_tags,
        "facilities": matched_facilities,
        "milestones": matched_milestones,
        "max_cost": max_cost
    }
