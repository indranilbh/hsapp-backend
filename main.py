from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import sqlite3
import psycopg2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi import Body
from datetime import datetime
from difflib import get_close_matches
import faiss
import pickle
import re,json
from fastapi import BackgroundTasks
from openai import OpenAIError
from filter_matcher import match_filters_from_query
from lexical_routes import router as lexical_router
from homestay_routes import router as homestay_router
from user_routes import router as user_router
from event_routes import router as events_router
from guide_routes import router as guide_router
from driver_routes import router as driver_router




load_dotenv()
client = OpenAI()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lexical_router)
app.include_router(homestay_router)
app.include_router(user_router)
app.include_router(events_router)
app.include_router(guide_router)
app.include_router(driver_router)
# Load model, index, and SQLite DB on startup
#index = faiss.read_index("../homestays.index")
#conn = sqlite3.connect("../homestays.db", check_same_thread=False)
DATABASE_URL = os.getenv("DATABASE_URL")
UNANSWERED_LOG = "unanswered_queries.json"
#conn = psycopg2.connect(DATABASE_URL)
#cursor = conn.cursor()

# Load ID mapping
# id_mapping = {}
# with open("../id_mapping.txt", "r") as f:
#     for line in f:
#         idx, hid = line.strip().split(",")
#         id_mapping[int(idx)] = int(hid)

#model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("all-mpnet-base-v2")

index = faiss.read_index("vector.index")
with open("id_mapping.pkl", "rb") as f:
    id_mapping = pickle.load(f)

def get_matching_region_condition(column, query_region):
    return f"LOWER({column}) LIKE '%" + query_region.lower() + "%'"

@app.get("/featured-minimal")
def get_featured_homestays_minimal():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT f.homestay_id, f.description, img.url
                FROM featured_homestay f
                JOIN images img ON img.homestay_id = f.homestay_id
                WHERE img.rank = 1
                ORDER BY f.id DESC
            """)
            rows = cur.fetchall()
            return [{"id": row[0], "description": row[1], "thumbnail": row[2]} for row in rows]
    except Exception as e:
        print("🔥 INTERNAL SERVER ERROR:", e)
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)
    finally:
        conn.close()


@app.get("/homestay-insight")
def get_homestay_insight(region: str):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT hi.id, hi.homestay_id, hi.rating, hi.road_condition, hi.max_capacity,
                       hi.number_of_rooms, hi.avg_days_spend, hi.local_itenaries,
                       hi.duration_to_reach, hi.local_festivals, hi.special_note,
                       a.country, a.state, a.district
                FROM homestay_insight hi
                JOIN homestays h ON hi.homestay_id = h.id
                JOIN areas a ON h.area_id = a.id
                WHERE LOWER(a.district) LIKE %s OR LOWER(a.state) LIKE %s OR LOWER(a.country) LIKE %s
            """, (f"%{region.lower()}%", f"%{region.lower()}%", f"%{region.lower()}%"))
            
            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "homestay_id": row[1],
                    "rating": row[2],
                    "road_condition": row[3],
                    "max_capacity": row[4],
                    "number_of_rooms": row[5],
                    "avg_days_spend": row[6],
                    "local_itenaries": row[7],
                    "duration_to_reach": row[8],
                    "local_festivals": row[9],
                    "special_note": row[10],
                    "country": row[11],
                    "state": row[12],
                    "district": row[13]
                }
                for row in rows
            ]
    except Exception as e:
        print("Error fetching homestay insights:", e)
        return JSONResponse(content={"error": "Unable to fetch homestay insights"}, status_code=500)
    finally:
        conn.close()

@app.get("/drivers/by-region")
def get_drivers(region: str):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, name, contact1, car_type, car_model, region
                FROM driver_detail
                WHERE {get_matching_region_condition('region', region)}
            """)
            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "contact": row[2],
                    "car_type": row[3],
                    "car_model": row[4],
                    "region": row[5]
                }
                for row in rows
            ]
    except Exception as e:
        print("Error fetching drivers:", e)
        return JSONResponse(content={"error": "Unable to fetch drivers"}, status_code=500)
    finally:
        conn.close()

@app.get("/community-guides/by-region")
def get_guides(region: str):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, name, contact1, area_id, region, social_media1
                FROM community_guides
                WHERE {get_matching_region_condition('region', region)}
            """)
            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "phone": row[2],
                    "area_id": row[3],
                    "region": row[4],
                    "social_media1": row[5]
                }
                for row in rows
            ]
    except Exception as e:
        print("Error fetching guides:", e)
        return JSONResponse(content={"error": "Unable to fetch community guides"}, status_code=500)
    finally:
        conn.close()

@app.get("/events/by-region")
def get_events(region: str):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT e.id, e.description, e.event_date, e.location,
                       a.country, a.state, a.district
                FROM events e
                JOIN areas a ON e.area_id = a.id
                WHERE LOWER(a.district) LIKE %s OR LOWER(a.state) LIKE %s OR LOWER(a.country) LIKE %s
            """, (f"%{region.lower()}%", f"%{region.lower()}%", f"%{region.lower()}%"))
            
            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "description": row[1],
                    "event_date": row[2],
                    "location": row[3],
                    "country": row[4],
                    "state": row[5],
                    "district": row[6]
                }
                for row in rows
            ]
    except Exception as e:
        print("Error fetching events:", e)
        return JSONResponse(content={"error": "Unable to fetch events"}, status_code=500)
    finally:
        conn.close()


def log_unanswered_query_async(query: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(log_unanswered_query, query)

def log_unanswered_query(query: str):
    data = {
        "query": query,
        "timestamp": datetime.now().isoformat()
    }
    if os.path.exists(UNANSWERED_LOG):
        with open(UNANSWERED_LOG, "r+", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
            existing.append(data)
            f.seek(0)
            json.dump(existing, f, indent=2)
    else:
        with open(UNANSWERED_LOG, "w", encoding="utf-8") as f:
            json.dump([data], f, indent=2)

def extract_query_filters_with_gpt(query: str):
    prompt = f"""
    You are an AI travel assistant helping extract structured filters from user queries about offbeat homestays.

    Return a valid JSON object with the following fields:
    - "state": string or null
    - "district": string or null
    - "tags": list of strings
    - "facilities": list of strings
    - "milestones": list of strings
    - "max_cost": number or null

    Example user query: "homestays in Himachal under 2000 with snow view"
    Expected JSON:
    {{
    "state": "Himachal Pradesh",
    "district": null,
    "tags": ["snow view"],
    "facilities": [],
    "milestones": [],
    "max_cost": 2000
    }}

    Now extract filters from this user query: "{query}"
    Only return a valid JSON object. No explanation or formatting.
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract structured filters from user travel queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )

        raw_content = completion.choices[0].message.content.strip()
        print("🔎 GPT raw content:", raw_content)

        # Attempt to parse JSON
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            print("⚠️ Failed to parse GPT response as JSON.")
            return fallback_filters()

    except OpenAIError as e:
        print("❌ Error from OpenAI:", e)
        return fallback_filters()
    except Exception as e:
        print("❌ Unexpected error:", e)
        return fallback_filters()


def fallback_filters():
    return {
        "state": None,
        "district": None,
        "tags": [],
        "facilities": [],
        "milestones": [],
        "max_cost": None
    }


@app.get("/")
def root():
    return {"message": "FastAPI backend is working!"}

@app.get("/dbtest")
def test_db():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM homestays")
            count = cur.fetchone()[0]
        return {"homestay_count": count}
    except Exception as e:
        print("DB error:", e)
        return {"error": "DB test failed"}
    finally:
        conn.close()  # ✅ Always closes

def contains_milestone_hint(query: str):
    hints = ["near", "close to", "around", "beside", "nearby"]
    return any(hint in query.lower() for hint in hints)

@app.get("/query")
def query_homestays(q: str = Query(..., description="Search query")):
    filters = extract_query_filters_with_gpt(q)
    semantic_matches  = match_filters_from_query(q)
    filter_state = filters.get("state")
    filter_district = filters.get("district")
    filter_max_cost = filters.get("max_cost")
    if not filters["tags"]:
        filters["tags"] = semantic_matches["tags"]
    if not filters["facilities"]:
        filters["facilities"] = semantic_matches["facilities"]
    if not filters["milestones"] and contains_milestone_hint(q):
        filters["milestones"] = semantic_matches["milestones"]
    
    # filter_tags = [t.lower() for t in filters.get("tags", [])]
    # filter_facilities = [f.lower() for f in filters.get("facilities", [])]
    # filter_milestones = [m.lower() for m in filters.get("milestones", [])]
    filter_tags = [str(t).lower() for t in filters.get("tags", [])]
    filter_facilities = [str(f).lower() for f in filters.get("facilities", [])]
    filter_milestones = [str(m).lower() for m in filters.get("milestones", [])]
    query_vector = model.encode(q)
    query_vector = np.array([query_vector]).astype("float32")
    # Get the total number of indexed vectors
    total_vectors = index.ntotal
    # Choose top_k safely based on available data
    top_k = min(200, total_vectors)
    # Perform the search
    distances, indices = index.search(query_vector, top_k)
    print("🔎 User query:", q)
    print("🔎 FAISS Distances:", distances)
    print("🧭 FAISS Indices:", indices)
    print("📌 State:", filter_state)
    print("📌 District:", filter_district)
    print("📌 Max Cost:", filter_max_cost)
    print("🔎 Final Tags:", filter_tags)
    print("🔎 Final Facilities:", filter_facilities)
    print("🔎 Final Milestones:", filter_milestones)

    result_ids = [id_mapping[i] for i in indices[0] if i != -1 and i < len(id_mapping)]
    print("📍 Resulting DB Homestay IDs:", result_ids)

    if not result_ids:
        return {"query": q, "matches": []}

    results = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            for hid in result_ids:
                cursor.execute("""
                    SELECT h.id, h.name, h.cost, a.state, a.district, a.id, h.altitude, h.website,
                           h.youtube, h.facebook, h.instagram,
                           c.name, c.phone, c.email,
                           d.name, d.phone, d.email
                    FROM homestays h
                    LEFT JOIN areas a ON h.area_id = a.id
                    LEFT JOIN contacts c ON h.contact_id = c.id
                    LEFT JOIN contacts d ON h.driver_contact_id = d.id
                    WHERE h.id = %s
                """, (hid,))
                row = cursor.fetchone()
                if not row:
                    continue

                (
                    _id, name, cost, state, district, area_id, altitude, website,
                    youtube, facebook, instagram,
                    owner_name, owner_phone, owner_email,
                    driver_name, driver_phone, driver_email
                ) = row

                # State and district filtering
                if filter_state and state and filter_state.lower() != state.lower():
                    continue
                if filter_district and district and filter_district.lower() != district.lower():
                    continue
                if filter_max_cost and cost and cost > filter_max_cost:
                    continue

                # Tags
                cursor.execute("""
                    SELECT t.name FROM tags t
                    JOIN homestay_tags ht ON ht.tag_id = t.id
                    WHERE ht.homestay_id = %s
                """, (hid,))
                tags = [r[0] for r in cursor.fetchall()]
                tag_set = set(t.lower() for t in tags)
                if filter_tags and not any(tag in tag_set for tag in filter_tags):
                    continue

                # Facilities
                cursor.execute("""
                    SELECT f.name FROM facilities f
                    JOIN homestay_facilities hf ON hf.facility_id = f.id
                    WHERE hf.homestay_id = %s
                """, (hid,))
                facilities = [r[0] for r in cursor.fetchall()]
                facility_set = set(f.lower() for f in facilities)
                if filter_facilities and not all(f in facility_set for f in filter_facilities):
                    continue

                # Milestone Tags
                cursor.execute("""
                    SELECT mt.tag FROM milestone_tags mt
                    JOIN homestay_milestones hm ON hm.milestone_tag_id = mt.id
                    WHERE hm.homestay_id = %s
                """, (hid,))
                milestone_tags = [r[0] for r in cursor.fetchall()]
                milestone_set = set(m.lower() for m in milestone_tags)
                if filter_milestones and not all(m in milestone_set for m in filter_milestones):
                    continue

                # Images
                cursor.execute("SELECT url FROM images WHERE homestay_id = %s", (hid,))
                images = [r[0] for r in cursor.fetchall()]

                results.append({
                    "id": hid,
                    "name": name,
                    "cost": cost,
                    "state": state,
                    "district": district,
                    "altitude": altitude,
                    "website": website,
                    "youtube": youtube,
                    "facebook": facebook,
                    "instagram": instagram,
                    "owner_contact": {"name": owner_name, "phone": owner_phone, "email": owner_email},
                    "driver_contact": {"name": driver_name, "phone": driver_phone, "email": driver_email},
                    "tags": tags,
                    "facilities": facilities,
                    "milestone_tags": milestone_tags,
                    "images": images
                })

        return {"query": q, "matches": results,"filters": filters}

    except Exception as e:
        print("❌ Error in query_homestays:", e)
        return {"error": "query failed"}

    finally:
        conn.close()

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def extract_location_from_query(query):
    query = query.lower()
    area_set = set()
    state_set = set()
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT district FROM areas")
            area_set.update([row[0].lower() for row in cur.fetchall() if row[0]])

            cur.execute("SELECT DISTINCT state FROM areas")
            state_set.update([row[0].lower() for row in cur.fetchall() if row[0]])
    except Exception as e:
        print("Error in location extraction:", e)
        return None
    finally:
        conn.close()

    all_locations = list(area_set.union(state_set))
    for loc in all_locations:
        if loc in query or query in loc:
            print(f"📍 Matched location: {loc}")
            return loc
    return None


def extract_facilities_from_query(query):
    query = query.lower()
    query_clean = re.sub(r"[^\w\s]", "", query)

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT name FROM facilities")
            all_facilities = [row[0].lower() for row in cur.fetchall()]
    except Exception as e:
        print("Error in facility extraction:", e)
        return []
    finally:
        conn.close()

    matched = []
    for f in sorted(all_facilities, key=len, reverse=True):  # Longest match first
        if f in query_clean:
            matched.append(f)
    return matched


def extract_milestones_from_query(query):
    query = query.lower()
    query_clean = re.sub(r"[^\w\s]", "", query)  # Remove punctuation

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT tag FROM milestone_tags")
            all_milestones = [row[0].lower() for row in cur.fetchall()]
    except Exception as e:
        print("Error in milestone extraction:", e)
        return []
    finally:
        conn.close()

    matched = []
    for m in sorted(all_milestones, key=len, reverse=True):  # Longest match first
        if m in query_clean:
            matched.append(m)
    return matched


def extract_tags_from_query(query):
    query = query.lower()
    query_clean = re.sub(r"[^\w\s]", "", query)  # Remove punctuation

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT name FROM tags")
            all_tags = [row[0].lower() for row in cur.fetchall()]
    except Exception as e:
        print("Error in tag extraction:", e)
        return []
    finally:
        conn.close()

    matched = []
    for tag in sorted(all_tags, key=len, reverse=True):  # Match longer tags first
        if tag in query_clean:
            matched.append(tag)

    return matched


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
def chat_with_ai(req: ChatRequest):
    user_query = req.query
    response = query_homestays(q=user_query)

    results = response.get("matches", [])
    filters = response.get("filters", {})
    # Debug print for tuning
    print(f"🧠 Interpreted Filters from GPT: {filters}")

    if not results:
        # Save unanswered query (optional file logging)
        with open("unanswered_queries.log", "a", encoding="utf-8") as f:
            f.write(user_query.strip() + "\n")

        # Return a user-friendly fallback response
        return {
            "message": f"🙏 Sorry, I couldn't find any homestays for: '{user_query}'.\n"
                       "Would you like to try a different location, view, or price range?",
            "matches": []
        }

    # Build context for GPT recommendation
    context = "\n---\n".join(
        f"{r['name']} in {r['district']}, {r['state']} — ₹{r['cost']}/night\n"
        f"Tags: {', '.join(r['tags'])}\n"
        f"Facilities: {', '.join(r['facilities'])}\n"
        f"Milestones: {', '.join(r['milestone_tags'])}"
        for r in results
    )

    prompt = f"""
        You are a friendly multilingual travel assistant. Detect the user's language (English, Hindi, or Bengali) and reply in the same language.

        Here are some offbeat homestays:\n{context}

        Respond to the user's query: "{user_query}".
        Give a short, friendly recommendation that fits the user's need.
        Only recommend homestays returned in the list.
        """

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or any other model you're using
            messages=[
                {"role": "system", "content": "You are a helpful travel planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        gpt_reply = completion.choices[0].message.content.strip()

        return {
            "message": gpt_reply,
            "matches": results
        }

    except Exception as e:
        print("❌ Error calling OpenAI:", e)

        # Basic fallback response
        fallback = f"Here are {len(results)} offbeat homestays based on your query: \"{user_query}\"\n"
        for i, r in enumerate(results, 1):
            fallback += f"\n{i}. {r['name']} in {r['district']}, {r['state']} — ₹{r['cost']}/night\n"
            fallback += f"   Tags: {', '.join(r['tags'])}\n"
            fallback += f"   Facilities: {', '.join(r['facilities'])}\n"
            fallback += f"   Milestones: {', '.join(r['milestone_tags'])}\n"

        return {
            "message": fallback.strip(),
            "matches": results
        }



class RegistrationRequest(BaseModel):
    name: str
    phone: str
    email: Optional[str] = None
    category: str
    whatsapp_opt_in: bool

@app.post("/register")
def register_user(req: RegistrationRequest):
    try:
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO registrations (name, phone, email, category, whatsapp_opt_in) VALUES (%s, %s, %s, %s, %s)",
                    (req.name, req.phone, req.email, req.category, req.whatsapp_opt_in)
                )
                conn.commit()
        except Exception as e:
            print("Error during registration:", e)
            return {"success": False, "message": "Registration failed."}
        finally:
            conn.close()  # ✅ Always closes
        # Send email to admin
        import smtplib
        from email.mime.text import MIMEText
        subject = "New User Registration"
        body = f"""
        Name: {req.name}
        Phone: {req.phone}
        Email: {req.email or 'N/A'}
        Category: {req.category}
        WhatsApp Opt-In: {req.whatsapp_opt_in}
        """
        message = MIMEText(body)
        message["Subject"] = subject
        message["From"] = os.getenv("EMAIL_USER")
        message["To"] = os.getenv("EMAIL_TO")
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        server.sendmail(os.getenv("EMAIL_USER"), os.getenv("EMAIL_TO"), message.as_string())
        server.quit()
        return {"success": True, "message": "Registration successful!"}
    except Exception as e:
        print("Error during registration:", e)
        return {"success": False, "message": "Registration failed."}

class Area(BaseModel):
    id: int
    state: Optional[str]
    district: Optional[str]
    country: Optional[str]

# GET all areas
@app.get("/areas")
def get_areas():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, state, district, country FROM areas ORDER BY state NULLS LAST")
            rows = cur.fetchall()
            #print("Returning area data:", rows)
            return [
                {
                    "id": row[0],
                    "state": row[1],
                    "district": row[2],
                    "country": row[3],
                }
                for row in rows
            ]
    except Exception as e:
        print("Error fetching areas:", e)
        return JSONResponse(content={"error": "Unable to fetch areas"}, status_code=500)
    finally:
            conn.close()

@app.get("/area-coordinates")
def get_area_coordinates(country: str, state: str, district: str):
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Step 1: Get area ID
        cur.execute("""
            SELECT id FROM areas
            WHERE country = %s AND state = %s AND district = %s
            LIMIT 1
        """, (country, state, district))
        area = cur.fetchone()
        if not area:
            return {"error": "Area not found"}, 404
        area_id = area[0]

        # Step 2: Get lat/lon from area_coordinates
        cur.execute("""
            SELECT latitude, longitude FROM area_coordinates
            WHERE area_id = %s
        """, (area_id,))
        row = cur.fetchone()
        if row:
            return {"latitude": row[0], "longitude": row[1]}
        return {"error": "Coordinates not found"}, 404


@app.get("/milestone-tags")
def get_milestone_tags(country: str, state: str, district: str):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Get area IDs matching the given country, state, district
            cur.execute("""
                SELECT id FROM areas
                WHERE country = %s AND state = %s AND district = %s
            """, (country, state, district))
            area_ids = [row[0] for row in cur.fetchall()]

            if not area_ids:
                return []

            # Get milestone tags linked to those areas
            cur.execute("""
                SELECT DISTINCT tag FROM milestone_tags
                WHERE area_id = ANY(%s)
            """, (area_ids,))
            tags = [row[0] for row in cur.fetchall()]

            # Format for react-select
            return [{"label": tag, "value": tag} for tag in tags]

    except Exception as e:
        print("Error fetching milestone tags:", e)
        return JSONResponse(content={"error": "Unable to fetch milestone tags"}, status_code=500)
    finally:
        conn.close()



class Tag(BaseModel):
    id: int
    name: str
    slug: Optional[str]
    category: Optional[str]
    description: Optional[str]
    priority: Optional[int]

# GET all tags
@app.get("/tags")
def get_tags():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, slug, category, description, priority FROM tags ORDER BY priority, name")
            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "slug": row[2],
                    "category": row[3],
                    "description": row[4],
                    "priority": row[5],
                }
                for row in rows
            ]
    except Exception as e:
        print("Error fetching tags:", e)
        return JSONResponse(content={"error": "Unable to fetch tags"}, status_code=500)
    finally:
            conn.close()

class Facility(BaseModel):
    id: int
    name: str
    slug: Optional[str]
    category: Optional[str]
    description: Optional[str]
    priority: Optional[int]
# GET all facilities
@app.get("/facilities")
def get_facilities():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, slug, category, description, priority FROM facilities ORDER BY priority, name")
            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "slug": row[2],
                    "category": row[3],
                    "description": row[4],
                    "priority": row[5],
                }
                for row in rows
            ]
    except Exception as e:
        print("Error fetching facilities:", e)
        return JSONResponse(content={"error": "Unable to fetch facilities"}, status_code=500)
    finally:
            conn.close()


class HomestayCreate(BaseModel):
    name: str
    cost: str
    website: str
    altitude: str
    area_id: int
    contact: dict
    driver: dict
    youtube: str
    facebook: str
    instagram: str
    facilities: List[int]
    tags: List[int]
    milestone_tags: List[str]
    images: List[str]

@app.post("/homestays")
def create_homestay(data: HomestayCreate = Body(...)):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # Insert contact
        cur.execute(
            "INSERT INTO contacts (name, phone, email, role) VALUES (%s, %s, %s, %s) RETURNING id",
            (data.contact["name"], data.contact["phone"], data.contact["email"], 'owner'),
        )
        contact_id = cur.fetchone()[0]
        # Insert driver
        cur.execute(
            "INSERT INTO contacts (name, phone, email, role) VALUES (%s, %s, %s, %s) RETURNING id",
            (data.driver["name"], data.driver["phone"], data.driver["email"], 'driver'),
        )
        driver_id = cur.fetchone()[0]
        # Insert homestay
        cur.execute(
            """INSERT INTO homestays
               (name, cost, area_id, contact_id, driver_contact_id, youtube, facebook,
                instagram, created_at, altitude, website)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (
                data.name, data.cost, data.area_id, contact_id, driver_id,
                data.youtube, data.facebook, data.instagram,
                datetime.utcnow(), data.altitude, data.website,
            )
        )
        homestay_id = cur.fetchone()[0]
        # Facilities
        for fid in data.facilities:
            cur.execute("INSERT INTO homestay_facilities (homestay_id, facility_id) VALUES (%s, %s)", (homestay_id, fid))
        # Tags
        for tid in data.tags:
            cur.execute("INSERT INTO homestay_tags (homestay_id, tag_id) VALUES (%s, %s)", (homestay_id, tid))
        # Milestone Tags: insert if not exist, then link to homestay
        for tag in data.milestone_tags:
            # Get existing tag ID or insert
            cur.execute("SELECT id FROM milestone_tags WHERE area_id = %s AND tag = %s", (data.area_id, tag))
            row = cur.fetchone()
            if row:
                tag_id = row[0]
            else:
                cur.execute("INSERT INTO milestone_tags (area_id, tag) VALUES (%s, %s) RETURNING id", (data.area_id, tag))
                tag_id = cur.fetchone()[0]
            # Insert into homestay_milestones
            cur.execute("INSERT INTO homestay_milestones (homestay_id, milestone_tag_id) VALUES (%s, %s)", (homestay_id, tag_id))

        # Images
        for url in data.images:
            if url.strip():
                cur.execute("INSERT INTO images (homestay_id, url) VALUES (%s, %s)", (homestay_id, url))

        conn.commit()
        return {"success": True, "message": "Homestay inserted successfully"}

    except Exception as e:
        conn.rollback()
        print("Error inserting homestay:", e)
        return JSONResponse(status_code=500, content={"success": False, "message": "Failed to insert homestay", "error": str(e)})

    finally:
        conn.close()


# GET endpoint for fetching homestays with optional country/state/district filters and lazy loading
@app.get("/homestays")
def get_homestays(
    country: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    skip: int = 0,
    limit: int = 20
):
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Build dynamic query
        query = """
            SELECT h.id, h.name, h.cost, a.country, a.state, a.district, img.url
            FROM homestays h
            LEFT JOIN areas a ON h.area_id = a.id
            LEFT JOIN (
                SELECT DISTINCT ON (homestay_id) homestay_id, url
                FROM images ORDER BY homestay_id, rank
            ) img ON img.homestay_id = h.id
            WHERE 1=1
        """
        params = []

        if country:
            query += " AND a.country = %s"
            params.append(country)
        if state:
            query += " AND a.state = %s"
            params.append(state)
        if district:
            query += " AND a.district = %s"
            params.append(district)

        query += " ORDER BY h.id DESC OFFSET %s LIMIT %s"
        params.extend([skip, limit])
        
        cur.execute(query, params)
        results = cur.fetchall()
        homestays = [
            {
                "id": row[0],
                "name": row[1],
                "cost": row[2],
                "country": row[3],
                "state": row[4],
                "district": row[5],
                "image": row[6],
            } for row in results
        ]
        return {"homestays": homestays}

    except Exception as e:
        print("Error fetching homestays:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        conn.close()