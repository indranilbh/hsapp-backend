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

                cur.execute("""
                    SELECT rating, road_condition, max_capacity, number_of_rooms, avg_days_spend,
                        local_itenaries, duration_to_reach, local_festivals, special_note
                    FROM homestay_insight WHERE homestay_id = %s
                """, (hid,))
                insight = cur.fetchone()
                insight_text = ""
                if insight:
                    rating, road, cap, rooms, avg_days, itin, dur, fest, note = insight
                    if rating: insight_text += f" Rated {rating}/5."
                    if road: insight_text += f" Road: {road}."
                    if cap: insight_text += f" Capacity: {cap}."
                    if rooms: insight_text += f" Rooms: {rooms}."
                    if avg_days: insight_text += f" Avg stay: {avg_days} days."
                    if itin: insight_text += f" Itineraries: {itin}."
                    if dur: insight_text += f" Time to reach: {dur}."
                    if fest: insight_text += f" Festivals: {fest}."
                    if note: insight_text += f" Note: {note}."

                description = f"{name}, {district}, {state}, {country}. Cost {cost}."
                description += f" Tags: {', '.join(tags)}. Facilities: {', '.join(facilities)}. Milestones: {', '.join(milestones)}."
                if altitude: description += f" Altitude: {altitude}."
                if website: description += f" Website: {website}."
                if youtube: description += f" YouTube: {youtube}."
                if instagram: description += f" Instagram: {instagram}."
                if facebook: description += f" Facebook: {facebook}."
                description += f" {insight_text}"

                homestay_texts.append(description)
                id_mapping.append(hid)

            return homestay_texts, id_mapping
    finally:
        conn.close()


def build_tag_facility_milestone_embeddings():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name, description FROM tags")
            tags = cur.fetchall()
            tag_texts = [f"{name}: {desc}" for name, desc in tags]
            tag_embeddings = model.encode(tag_texts)
            with open("tag_embeddings.pkl", "wb") as f:
                pickle.dump(list(zip([t[0] for t in tags], tag_embeddings)), f)

            cur.execute("SELECT name, description FROM facilities")
            facilities = cur.fetchall()
            facility_texts = [f"{name}: {desc}" for name, desc in facilities]
            facility_embeddings = model.encode(facility_texts)
            with open("facility_embeddings.pkl", "wb") as f:
                pickle.dump(list(zip([f[0] for f in facilities], facility_embeddings)), f)

            cur.execute("SELECT DISTINCT tag FROM milestone_tags")
            milestones = [r[0] for r in cur.fetchall()]
            milestone_embeddings = model.encode(milestones)
            with open("milestone_embeddings.pkl", "wb") as f:
                pickle.dump(list(zip(milestones, milestone_embeddings)), f)

            print("✅ Tags, Facilities, Milestones embeddings saved.")
    finally:
        conn.close()


def build_community_driver_event_embeddings():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            # Community Guides
            cur.execute("""
            SELECT g.name, g.bio, g.region, a.country, a.state, a.district
            FROM community_guides g
            LEFT JOIN areas a ON g.area_id = a.id
            """)
            rows = cur.fetchall()
            texts = [f"{name}, Guide from {district}, {state}, {country}. Region: {region}. {bio}" for name, bio, region, country, state, district in rows]
            embeddings = model.encode(texts)
            with open("community_guides_embeddings.pkl", "wb") as f:
                pickle.dump(list(zip(texts, embeddings)), f)

            # Driver Detail
            cur.execute("""
            SELECT d.name, d.region, d.car_type, d.car_model, d.bio, a.country, a.state, a.district
            FROM driver_detail d
            LEFT JOIN areas a ON d.area_id = a.id
            """)
            rows = cur.fetchall()
            texts = [f"{name} is a {car_type} driver based in {district}, {state}, {country}. Region: {region}. Drives a {car_model}. {bio}"for name, region, car_type, car_model, bio, country, state, district in rows]
            embeddings = model.encode(texts)
            with open("driver_embeddings.pkl", "wb") as f:
                pickle.dump(list(zip(texts, embeddings)), f)

            # Events
            cur.execute("""
            SELECT e.location, e.description, e.special_note, a.country, a.state, a.district
            FROM events e
            LEFT JOIN areas a ON e.area_id = a.id
            """)
            rows = cur.fetchall()
            texts = [f"Event at {location} in {district}, {state}, {country}: {desc}. Note: {note}" for location, desc, note, country, state, district in rows]
            embeddings = model.encode(texts)
            with open("event_embeddings.pkl", "wb") as f:
                pickle.dump(list(zip(texts, embeddings)), f)

            print("✅ Community guides, drivers, and event embeddings saved.")
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

        build_tag_facility_milestone_embeddings()
        build_community_driver_event_embeddings()

    except Exception as e:
        print("❌ Error building vector index or embeddings:", e)


if __name__ == "__main__":
    build_and_save_index()
