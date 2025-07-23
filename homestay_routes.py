from fastapi import APIRouter, HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv


load_dotenv()
router = APIRouter()
DATABASE_URL = os.getenv("DATABASE_URL")


@router.get("/homestay/{homestay_id}")
def get_homestay_by_id(homestay_id: int):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    h.id, h.name, h.cost,
                    a.country, a.state, a.district,
                    c1.name AS owner_name, c1.phone AS owner_contact,
                    c2.name AS driver_name, c2.phone AS driver_contact,
                    h.youtube, h.facebook, h.instagram,
                    h.altitude, h.website
                FROM homestays h
                LEFT JOIN areas a ON h.area_id = a.id
                LEFT JOIN contacts c1 ON h.contact_id = c1.id
                LEFT JOIN contacts c2 ON h.driver_contact_id = c2.id
                WHERE h.id = %s
            """, (homestay_id,))
            homestay = cur.fetchone()

            if not homestay:
                raise HTTPException(status_code=404, detail="Homestay not found")

            # Tags
            cur.execute("""
                SELECT t.name FROM tags t
                JOIN homestay_tags ht ON ht.tag_id = t.id
                WHERE ht.homestay_id = %s
            """, (homestay_id,))
            homestay["tags"] = [r["name"] for r in cur.fetchall()]

            # Facilities
            cur.execute("""
                SELECT f.name FROM facilities f
                JOIN homestay_facilities hf ON hf.facility_id = f.id
                WHERE hf.homestay_id = %s
            """, (homestay_id,))
            homestay["facilities"] = [r["name"] for r in cur.fetchall()]

            # Images
            cur.execute("""
                SELECT url FROM images
                WHERE homestay_id = %s
            """, (homestay_id,))
            homestay["images"] = [r["url"] for r in cur.fetchall()]

            return homestay
    finally:
        conn.close()

@router.get("/homestay_insight/{homestay_id}")
def get_homestay_insight(homestay_id: int):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    rating,
                    road_condition,
                    max_capacity,
                    number_of_rooms,
                    avg_days_spend,
                    local_itenaries,
                    duration_to_reach,
                    local_festivals,
                    special_note,
                    region
                FROM homestay_insight
                WHERE homestay_id = %s
            """, (homestay_id,))
            
            insight = cur.fetchone()

            if not insight:
                raise HTTPException(status_code=404, detail="Homestay insight not found")
            
            return insight
    finally:
        conn.close()