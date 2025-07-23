import psycopg2
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL environment variable is not set.")

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
                cur.execute("SELECT country, state, district FROM areas WHERE id = %s", (area_id,))
                area_row = cur.fetchone()
                country, state, district = area_row if area_row else ("", "", "")

                cur.execute("""
                    SELECT t.name FROM tags t
                    JOIN homestay_tags ht ON ht.tag_id = t.id
                    WHERE ht.homestay_id = %s
                """, (hid,))
                tags = [r[0] for r in cur.fetchall()]

                cur.execute("""
                    SELECT f.name FROM facilities f
                    JOIN homestay_facilities hf ON hf.facility_id = f.id
                    WHERE hf.homestay_id = %s
                """, (hid,))
                facilities = [r[0] for r in cur.fetchall()]

                cur.execute("""
                    SELECT mt.tag FROM milestone_tags mt
                    JOIN homestay_milestones hm ON hm.milestone_tag_id = mt.id
                    WHERE hm.homestay_id = %s
                """, (hid,))
                milestones = [r[0] for r in cur.fetchall()]

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

                homestay_texts.append(description)
                id_mapping.append(hid)

            return homestay_texts, id_mapping
    finally:
        conn.close()


def build_tag_facility_milestone_embeddings():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            # # Tags
            # cur.execute("SELECT name FROM tags")
            # tags = [row[0] for row in cur.fetchall()]
            # tag_embeddings = model.encode(tags)
            # tag_data = list(zip(tags, tag_embeddings))
            # with open("tag_embeddings.pkl", "wb") as f:
            #     pickle.dump(tag_data, f)

            # # Facilities
            # cur.execute("SELECT name FROM facilities")
            # facilities = [row[0] for row in cur.fetchall()]
            # facility_embeddings = model.encode(facilities)
            # facility_data = list(zip(facilities, facility_embeddings))
            # with open("facility_embeddings.pkl", "wb") as f:
            #     pickle.dump(facility_data, f)

            # Tags
            cur.execute("SELECT name, description FROM tags")
            tags = cur.fetchall()
            # Combine name and description for better embeddings
            tag_texts = [f"{name}: {desc}" for name, desc in tags]
            tag_names = [name for name, _ in tags]
            tag_embeddings = model.encode(tag_texts)
            tag_data = list(zip(tag_names, tag_embeddings))
            with open("tag_embeddings.pkl", "wb") as f:
                pickle.dump(tag_data, f)

            # ✅ Facilities
            cur.execute("SELECT name, description FROM facilities")
            facilities = cur.fetchall()
            facility_texts = [f"{name}: {desc}" for name, desc in facilities]
            facility_names = [name for name, _ in facilities]
            facility_embeddings = model.encode(facility_texts)
            facility_data = list(zip(facility_names, facility_embeddings))
            with open("facility_embeddings.pkl", "wb") as f:
                pickle.dump(facility_data, f)

            # Milestones
            cur.execute("SELECT DISTINCT tag FROM milestone_tags")
            milestones = [row[0] for row in cur.fetchall()]
            milestone_embeddings = model.encode(milestones)
            milestone_data = list(zip(milestones, milestone_embeddings))
            with open("milestone_embeddings.pkl", "wb") as f:
                pickle.dump(milestone_data, f)
            print("✅ Tag, facility, and milestone embeddings saved.")
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

        print(f"✅ Homestay vector index built and saved for {len(ids)} homestays.")

        build_tag_facility_milestone_embeddings()

    except Exception as e:
        print("❌ Error building vector index or other embeddings:", e)


if __name__ == "__main__":
    print ("here")
    build_and_save_index()
