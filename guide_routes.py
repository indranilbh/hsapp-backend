from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psycopg2
import os
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor


load_dotenv()
router = APIRouter()
DATABASE_URL = os.getenv("DATABASE_URL")


class Guide(BaseModel):
    name: str
    contact1: str
    contact2: str = None
    social_media1: str = None
    social_media2: str = None
    bio: str = None
    notes: str = None
    photo_url: str = None
    area_id: int = None
    region: str = None


@router.post("/guides")
def create_guide(guide: Guide):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO community_guides (
                    name, contact1, contact2, social_media1, social_media2,
                    bio, notes, photo_url, area_id,region
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
            """, (
                guide.name, guide.contact1, guide.contact2, guide.social_media1, guide.social_media2,
                guide.bio, guide.notes, guide.photo_url, guide.area_id, guide.region
            ))
            conn.commit()
            return {"message": "✅ Guide created successfully"}
    except Exception as e:
        print("Error inserting guide:", e)
        return JSONResponse(content={"error": "Failed to create guide"}, status_code=500)
    finally:
        conn.close()


@router.get("/guides")
def get_guides():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT g.id, g.name, g.contact1, g.contact2, g.social_media1, g.social_media2,
                       g.bio, g.notes, g.photo_url, g.area_id,g.region,
                       a.country, a.state, a.district
                FROM community_guides g
                LEFT JOIN areas a ON g.area_id = a.id
                ORDER BY g.id DESC
            """)
            rows = cur.fetchall()
            guides = []
            for row in rows:
                print(row)
                guides.append({
                    "id": row[0],
                    "name": row[1],
                    "contact1": row[2],
                    "contact2": row[3],
                    "social_media1": row[4],
                    "social_media2": row[5],
                    "bio": row[6],
                    "notes": row[7],
                    "photo_url": row[8],
                    "area_id": row[9],
                    "region": row[10],
                    "country": row[11],
                    "state": row[12],
                    "district": row[13],
                    
                })
           
            return guides
    except Exception as e:
        print("Error fetching guides:", e)
        return JSONResponse(content={"error": "Failed to fetch guides"}, status_code=500)
    finally:
        conn.close()


@router.put("/guides/{guide_id}")
def update_guide(guide_id: int, guide: Guide):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE community_guides SET
                    name=%s, contact1=%s, contact2=%s, social_media1=%s,
                    social_media2=%s, bio=%s, notes=%s, photo_url=%s, area_id=%s, region=%s
                WHERE id=%s
            """, (
                guide.name, guide.contact1, guide.contact2, guide.social_media1, guide.social_media2,
                guide.bio, guide.notes, guide.photo_url, guide.area_id,guide.region, guide_id
            ))
            conn.commit()
            return {"message": "✏️ Guide updated successfully"}
    except Exception as e:
        print("Error updating guide:", e)
        return JSONResponse(content={"error": "Failed to update guide"}, status_code=500)
    finally:
        conn.close()


@router.delete("/guides/{guide_id}")
def delete_guide(guide_id: int):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM community_guides WHERE id = %s", (guide_id,))
            conn.commit()
            return {"message": "🗑️ Guide deleted successfully"}
    except Exception as e:
        print("Error deleting guide:", e)
        return JSONResponse(content={"error": "Failed to delete guide"}, status_code=500)
    finally:
        conn.close()



@router.get("/homestay_guides/{homestay_id}")
def get_guides_for_homestay(homestay_id: int):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Step 1: Get area_id for the homestay
            cur.execute("SELECT area_id FROM homestays WHERE id = %s", (homestay_id,))
            area_result = cur.fetchone()
            if not area_result:
                return JSONResponse(content={"error": "Homestay not found"}, status_code=404)

            area_id = area_result["area_id"]
            print(f"[DEBUG] Homestay {homestay_id} has area_id = {area_id}")

            # Step 2: Get milestone_tag_ids for this homestay
            cur.execute("SELECT milestone_tag_id FROM homestay_milestones WHERE homestay_id = %s", (homestay_id,))
            milestone_ids = [row["milestone_tag_id"] for row in cur.fetchall()]
            print(f"[DEBUG] Milestone tag IDs: {milestone_ids}")

            if not milestone_ids:
                return []

            # Step 3: Get regions from milestone_tags
            cur.execute(
                "SELECT tag FROM milestone_tags WHERE id = ANY(%s)",
                (milestone_ids,)
            )
            regions = [row["tag"] for row in cur.fetchall()]
            print(f"[DEBUG] Milestone regions: {regions}")

            # Step 4: Get all guides for the same area_id
            cur.execute("""
                SELECT id, name, contact1, contact2, social_media1, social_media2,
                       bio, notes, photo_url, area_id, region
                FROM community_guides
                WHERE area_id = %s
            """, (area_id,))
            guide_rows = cur.fetchall()
            print(f"[DEBUG] Fetched {len(guide_rows)} guides for area_id {area_id}")

            # Step 5: Score guides by region match
            scored_guides = []
            for g in guide_rows:
                guide_regions = [r.strip().lower() for r in g["region"].split(",") if r.strip()] if g["region"] else []
                match_score = len(set(r.lower() for r in regions) & set(guide_regions))
                g["match_score"] = match_score
                scored_guides.append(g)

            scored_guides.sort(key=lambda x: x["match_score"], reverse=True)
            print(f"[DEBUG] Scored and sorted guides by match score")

            return scored_guides

    except Exception as e:
        print("Error in /homestay_guides:", e)
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)
    finally:
        conn.close()
