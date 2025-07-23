import psycopg2
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv


load_dotenv()
# ✅ Load DB URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL environment variable is not set.")


# ✅ Load sentence embedding model
#model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer('all-mpnet-base-v2')  


def get_all_homestay_data():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, area_id, cost, altitude, website, youtube, instagram, facebook 
                FROM homestays
            """)
            homestays = cur.fetchall()

            homestay_texts = []
            id_mapping = []

            for hid, name, area_id, cost, altitude, website, youtube, instagram, facebook in homestays:
                # Area info
                cur.execute("SELECT country, state, district FROM areas WHERE id = %s", (area_id,))
                area_row = cur.fetchone()
                if area_row:
                    country, state, district = area_row
                else:
                    country = state = district = ""

                # Tags
                cur.execute("""
                    SELECT t.name FROM tags t
                    JOIN homestay_tags ht ON ht.tag_id = t.id
                    WHERE ht.homestay_id = %s
                """, (hid,))
                tags = [r[0] for r in cur.fetchall()]

                # Facilities
                cur.execute("""
                    SELECT f.name FROM facilities f
                    JOIN homestay_facilities hf ON hf.facility_id = f.id
                    WHERE hf.homestay_id = %s
                """, (hid,))
                facilities = [r[0] for r in cur.fetchall()]

                # Milestone Tags
                cur.execute("""
                    SELECT mt.tag FROM milestone_tags mt
                    JOIN homestay_milestones hm ON hm.milestone_tag_id = mt.id
                    WHERE hm.homestay_id = %s
                """, (hid,))
                milestones = [r[0] for r in cur.fetchall()]

                # ✅ Construct full text description for embedding
                description = f"{name}, {district}, {state}, {country}. Cost {cost}."
                description += f" Tags: {', '.join(tags)}. Facilities: {', '.join(facilities)}."
                description += f" Nearby: {', '.join(milestones)}."

                if altitude:
                    description += f" Altitude: {altitude}."
                if website:
                    description += f" Website: {website}."
                if youtube:
                    description += f" YouTube: {youtube}."
                if instagram:
                    description += f" Instagram: {instagram}."
                if facebook:
                    description += f" Facebook: {facebook}."
                if tags:
                    description += f" Tags: {', '.join(tags)}."
                if facilities:
                    description += f" Facilities: {', '.join(facilities)}."
                if milestones:
                    description += f" Nearby: {', '.join(milestones)}."

                homestay_texts.append(description)
                id_mapping.append(hid)

            return homestay_texts, id_mapping
    finally:
        conn.close()

def build_and_save_index():
    try:
        texts, ids = get_all_homestay_data()
        embeddings = model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, "vector.index")
        with open("id_mapping.pkl", "wb") as f:
            pickle.dump(ids, f)

        print(f"✅ Vector index built and saved for {len(ids)} homestays.")
    except Exception as e:
        print("❌ Error building vector index:", e)

if __name__ == "__main__":
    build_and_save_index()
